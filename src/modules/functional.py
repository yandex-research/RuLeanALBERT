import functools
import os

import torch
from transformers.activations import ACT2FN as HF_ACT2FN


@functools.lru_cache()
def maybe_script(fn: callable) -> callable:
    """Apply torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script."""
    using_tpu = bool(os.environ.get("TPU_NAME"))
    # this is a reserved variable that must be set to TPU address (e.g. grpc://11.22.33.44:1337) for TPU to function
    should_script = int(os.environ.get("LEAN_USE_JIT", not using_tpu))
    return torch.jit.script(fn) if should_script else fn


@maybe_script
def gelu_fused(x):
    """
    Approximate GELU activation, same as in Google BERT and OpenAI GPT (as of Dec 2021)
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))  # note: 0.7988.. = sqrt(2/pi)


@maybe_script
def gelu_fused_grad(grad_output, x):
    """Gradients of gelu_fwd w.r.t. input"""
    tanh = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    jac = 0.5 * x * ((1 - tanh * tanh) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh)
    return jac * grad_output


class GELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        ctx.save_for_backward(input)
        return gelu_fused(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, = ctx.saved_tensors
        return gelu_fused_grad(grad_output, input)


ACT2FN = dict(HF_ACT2FN, gelu_fused=GELU.apply)
