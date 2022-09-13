"""
This code is a joint work with Tim Dettmers, based on his library https://github.com/facebookresearch/bitsandbytes ;
Unlike the rest of bnb optimizers, CPULAMB8Bit is tuned to further reduce memory footprint at the cost of performance.
The intended use-case of CPULamb8Bit is to run in background on CPU while training with large batches.
"""
import math
from typing import Any, Dict, Optional

import torch
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise
from bitsandbytes.optim.optimizer import Optimizer2State
from torch_optimizer.types import Betas2, Params

__all__ = ("CPULAMB8Bit",)

from hivemind.utils.logging import get_logger, use_hivemind_log_handler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

class CPULAMB8Bit(Optimizer2State):
    r"""
    Implements Lamb with quantized 8-bit statistics. The statistics are stored in host memory in the quantized form.
    The LAMB optimizer and block-wise quantization are described in the following papers:
    - LAMB: "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" https://arxiv.org/abs/1904.00962
    - Quantization: "8-bit Optimizers via Block-wise Quantization" https://arxiv.org/abs/2110.02861
    This specific implementation of LAMB is based on https://github.com/cybertronai/pytorch-lamb
    - bias correction defaults to False because paper v3 does not use debiasing
    - it has baked in clipping by global max_grad_norm
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        clamp_value: clamp weight_norm in (0,clamp_value) (default: 10)
            set to a high value to avoid it (e.g 10e3)
        bias_correction: debias statistics by (1 - beta**step) (default: True)
        min_8bit_size: statistics for parameters with fewer than this many elements will not be quantized
        reuse_grad_buffers: if True, optimizer will modify gradients in-place to save memory.
            If enabled, one must ensure that .zero_grad() is called after each optimizer step.
        update_chunk_size: quantized statistics will be de-quantized in chunks of up to this many elements.
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
        bias_correction: bool = True,
        min_8bit_size: int = 65536,
        reuse_grad_buffers: bool = False,
        update_chunk_size: int = 2 ** 24,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if clamp_value < 0.0:
            raise ValueError("Invalid clamp value: {}".format(clamp_value))

        self.clamp_value = clamp_value
        self.bias_correction = bias_correction
        self.reuse_grad_buffers = reuse_grad_buffers
        self.update_chunk_size = update_chunk_size
        self.max_grad_norm = max_grad_norm

        super(CPULAMB8Bit, self).__init__(
            "cpu-lamb",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            optim_bits=8,
            min_8bit_size=min_8bit_size,
            args=None,
            percentile_clipping=100,
            block_wise=4096,
            max_unorm=0,
        )

    @torch.no_grad()
    def step(self, closure=None):
        if self.max_grad_norm is not None:
            iter_params = (param for group in self.param_groups for param in group["params"])
            torch.nn.utils.clip_grad_norm_(iter_params, self.max_grad_norm)
        return super().step(closure=closure)

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)
        assert config["percentile_clipping"] == 100, "percentile clipping is not implemented on CPU"
        assert config["max_unorm"] == 0

        if config["optim_bits"] == 32:
            dtype = torch.float32
        elif config["optim_bits"] == 8:
            dtype = torch.uint8
        else:
            raise NotImplementedError(f'Amount of optimizer bits not supported: {config["optim_bits"]}')

        if p.numel() < config["min_8bit_size"]:
            dtype = torch.float32

        state = self.state[p]
        state["step"] = 0

        if dtype == torch.float32 or (dtype == torch.uint8 and p.numel() < 4096):
            state["state1"] = torch.zeros_like(
                p,
                memory_format=torch.preserve_format,
                dtype=torch.float32,
                device=p.device,
            )
            state["state2"] = torch.zeros_like(
                p,
                memory_format=torch.preserve_format,
                dtype=torch.float32,
                device=p.device,
            )
        elif dtype == torch.uint8:
            if state["step"] == 0:
                if "dynamic" not in self.name2qmap:
                    self.fill_qmap()
                self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(p.device)
                self.name2qmap["udynamic"] = self.name2qmap["udynamic"].to(p.device)

            n = p.numel()
            blocks = (n - 1) // config["block_wise"] + 1

            state["state1"] = torch.zeros_like(
                p,
                memory_format=torch.preserve_format,
                dtype=torch.uint8,
                device=p.device,
            )
            state["qmap1"] = self.name2qmap["dynamic"]

            state["state2"] = torch.zeros_like(
                p,
                memory_format=torch.preserve_format,
                dtype=torch.uint8,
                device=p.device,
            )
            state["qmap2"] = self.name2qmap["udynamic"]

            state["absmax1"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)
            state["absmax2"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)

    @torch.no_grad()
    def update_step(self, group: Dict[str, Any], p: torch.Tensor, gindex: int, pindex: int):
        state = self.state[p]
        config = self.get_config(gindex, pindex, group)

        p_cpu, grad_cpu = p.cpu(), p.grad.cpu()
        # this is a no-op if parameters are already on CPU

        step = state["step"] = state["step"] + 1
        beta1, beta2 = group["betas"]

        param_delta = self._update_moments_and_compute_delta(
            state, config, p_cpu, grad_cpu, beta1, beta2, group["eps"], group["weight_decay"]
        )
        del grad_cpu  # grad_cpu is no longer needed and may be modified if self.reuse_grad_buffers

        step_norm = torch.norm(param_delta)
        weight_norm = p_cpu.norm().clamp(0, self.clamp_value)

        trust_ratio = weight_norm / step_norm if weight_norm != 0 and step_norm != 0 else 1.0
        state["weight_norm"], state["step_norm"], state["trust_ratio"] = (weight_norm, step_norm, trust_ratio)

        # Apply bias to lr to avoid broadcast.
        bias_correction = math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step) if self.bias_correction else 1
        step_size = group["lr"] * bias_correction
        p.data.add_(param_delta.to(p.device), alpha=-step_size * trust_ratio)

    def _update_moments_and_compute_delta(
        self,
        state: Dict,
        config: Dict,
        p_cpu: torch.Tensor,
        grad_cpu: torch.Tensor,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> torch.Tensor:
        step, block_size, chunk_size = (state["step"], config["block_wise"], self.update_chunk_size)

        if state["state1"].dtype != torch.uint8:
            # not quantized: update normally
            exp_avg, exp_avg_sq = state["state1"], state["state2"]
            exp_avg.mul_(beta1).add_(grad_cpu, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)

            sqrt_out = grad_cpu if self.reuse_grad_buffers else None
            _denominator = torch.sqrt(exp_avg_sq, out=sqrt_out).add_(eps)
            param_delta = torch.div(exp_avg, _denominator, out=_denominator)
            if weight_decay != 0:
                param_delta.add_(p_cpu, alpha=weight_decay)
            return param_delta
        elif p_cpu.numel() <= chunk_size:
            # quantized tensor within chunk size
            exp_avg = dequantize_blockwise(state["state1"], (state["absmax1"], state["qmap1"]), blocksize=block_size)
            exp_avg_sq = dequantize_blockwise(state["state2"], (state["absmax2"], state["qmap2"]), blocksize=block_size)

            exp_avg.mul_(beta1).add_(grad_cpu, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)

            quantize_blockwise(exp_avg, state["qmap1"], state["absmax1"], out=state["state1"])
            quantize_blockwise(exp_avg_sq, state["qmap2"], state["absmax2"], out=state["state2"])
            # note: quantize_blockwise also modifies qmap and absmax in-place

            param_delta = exp_avg.div_(exp_avg_sq.sqrt_().add_(eps))
            # note: this changes statistics in-place, but it's okay b/c we saved quantized version

            if weight_decay != 0:
                param_delta.add_(p_cpu, alpha=weight_decay)
            return param_delta

        else:
            # very large quantized tensor, compute updates in chunks to save RAM
            flat_p, flat_grad, flat_state1, flat_state2 = (
                tensor.view(-1) for tensor in (p_cpu, grad_cpu, state["state1"], state["state2"])
            )
            output_buffer = flat_grad if self.reuse_grad_buffers else torch.empty_like(flat_grad)

            for chunk_index, chunk_start in enumerate(range(0, len(flat_p), chunk_size)):
                chunk = slice(chunk_start, chunk_start + chunk_size)
                chunk_blocks = slice(chunk_start // block_size, (chunk_start + chunk_size) // block_size)

                chunk_p, chunk_grad = flat_p[chunk], flat_grad[chunk]
                chunk_state1, chunk_state2 = flat_state1[chunk], flat_state2[chunk]
                chunk_absmax1, chunk_absmax2 = (
                    state["absmax1"][chunk_blocks],
                    state["absmax2"][chunk_blocks],
                )
                if chunk_state1.storage_offset() != 0:
                    # clone chunks to ensure that tensors do not have offsets (bnb hack, possibly no longer needed)
                    chunk_state1, chunk_state2, chunk_absmax1, chunk_absmax2 = map(
                        torch.clone, (chunk_state1, chunk_state2, chunk_absmax1, chunk_absmax2),
                    )

                exp_avg_chunk = dequantize_blockwise(
                    chunk_state1, (chunk_absmax1, state["qmap1"]), blocksize=block_size
                )
                exp_avg_sq_chunk = dequantize_blockwise(
                    chunk_state2, (chunk_absmax2, state["qmap2"]), blocksize=block_size
                )

                exp_avg_chunk.mul_(beta1).add_(chunk_grad, alpha=1 - beta1)
                exp_avg_sq_chunk.mul_(beta2).addcmul_(chunk_grad, chunk_grad, value=1 - beta2)

                # note: output_buffer cannot be modified until this line because it shares memory with grad_cpu
                del chunk_grad

                flat_state1[chunk], (
                    state["absmax1"][chunk_blocks],
                    state["qmap1"],
                ) = quantize_blockwise(exp_avg_chunk, state["qmap1"], chunk_absmax1, out=chunk_state1)
                flat_state2[chunk], (
                    state["absmax2"][chunk_blocks],
                    state["qmap2"],
                ) = quantize_blockwise(exp_avg_sq_chunk, state["qmap2"], chunk_absmax2, out=chunk_state2)
                # note: we need to explicitly assign new quantized tensors because of cloning earlier

                torch.div(
                    exp_avg_chunk,
                    exp_avg_sq_chunk.sqrt_().add_(eps),
                    out=output_buffer[chunk],
                )
                # note: this changes statistics in-place, but it's okay b/c we saved quantized version

                if weight_decay != 0:
                    output_buffer[chunk].add_(flat_p[chunk], alpha=weight_decay)

            param_delta = output_buffer.view_as(grad_cpu)

            return param_delta
