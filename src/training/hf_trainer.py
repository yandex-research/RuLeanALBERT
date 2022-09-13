"""A catch-all module for the dirty hacks required to make HF Trainer work with collaborative training"""

import hivemind
import torch
import torch.distributed
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from torch import nn
from torch.utils.data import DataLoader
from transformers.trainer import Trainer

from arguments import HFTrainerArguments
from src.modules.rotary import RotaryEmbeddings
from src.training.sync import is_main_process

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)
LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)


class CollaborativeHFTrainer(Trainer):
    """
    A version of HuggingFace trainer that shuffles the dataset using a separate random seed.
    Used to ensure that peers don't process batches in the same order.
    """

    def __init__(self, *, data_seed: int, optimizer: hivemind.Optimizer, reuse_grad_buffers: bool, **kwargs):
        self.data_seed = data_seed
        self.optimizer = optimizer
        self.reuse_grad_buffers = reuse_grad_buffers
        assert not torch.distributed.is_initialized() or not reuse_grad_buffers, "DDP with reuse_grad_buffers is not implemented (yet)"
        super().__init__(optimizers=(optimizer, NoOpScheduler(optimizer)), **kwargs)

        if self.args.fp16 and self.reuse_grad_buffers:
            self.scaler = hivemind.GradScaler()

    def get_train_dataloader(self) -> DataLoader:
        """Shuffle data independently for each peer to avoid duplicating batches [important for quality]"""
        seed = self.data_seed
        if torch.distributed.is_initialized():
            seed += torch.distributed.get_rank()
        torch.manual_seed(seed)
        return super().get_train_dataloader()

    def _wrap_model(self, model: nn.Module, training=True):
        assert training, "Evaluation (training=False) should be run on a separate dedicated worker."
        model = super()._wrap_model(model, training)
        if torch.distributed.is_initialized():
            assert isinstance(model, nn.parallel.DistributedDataParallel)
            assert model.require_forward_param_sync
            logger.info("Pre-populating rotary embedding cache up to maximum length to enforce static graph")
            assert isinstance(self.args, HFTrainerArguments)
            device = f"{model.device_type}:{model.output_device}"
            for module in model.modules():
                if isinstance(module, RotaryEmbeddings):
                    module._set_auxiliary_buffers(max_len=self.args.max_sequence_length, device=device)

            logger.warning("DistributedDataParallel: triggering _set_static_graph() to allow checkpointing")
            model._set_static_graph()

        # if reuse_grad_buffers is True, we should accumulate gradients in .grad without zeroing them after each step
        should_override_zero_grad = self.reuse_grad_buffers if is_main_process() else False  # replicas can reset grad
        return IgnoreGradManipulations(model, override_zero_grad=should_override_zero_grad)


class NOPtimizer(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0)

    def step(self, *args, **kwargs):
        pass


class NoOpScheduler(LRSchedulerBase):
    """Dummy scheduler for transformers.Trainer. The real scheduler is defined in CollaborativeOptimizer.scheduler"""

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, *args, **kwargs):
        if self.optimizer.scheduler:
            return self.optimizer.scheduler.print_lr(*args, **kwargs)

    def step(self):
        logger.debug("Called NoOpScheduler.step")
        self._last_lr = self.get_lr()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        logger.debug("Called NoOpScheduler.load_state_dict")


class IgnoreGradManipulations(nn.Module):
    """Wrapper for model that blocks gradient manipulations in huggingface Trainer (e.g. zero_grad, clip_grad)"""

    def __init__(self, module, override_clipping: bool = True, override_zero_grad: bool = True):
        super().__init__()
        self.module = module
        self.override_clipping = override_clipping
        self.override_zero_grad = override_zero_grad

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.override_zero_grad:
            grad_is_nan = all(param.grad.isfinite().all() for param in self.parameters())
            if grad_is_nan:
                logger.debug("Successfully bypassed zero_grad")
            else:
                logger.debug("Encountered non-finite value in gradients!")
                self.module.zero_grad(set_to_none=set_to_none)
        else:

            self.module.zero_grad(set_to_none=set_to_none)

    def clip_grad_norm_(self, max_norm: float, norm_type: int = 2):
        """ignore clip_grad_norm on each step, clip in optimizer instead"""
        if self.override_clipping:
            logger.debug("Successfully bypassed clip_grad_norm_")
        else:
            return torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm, norm_type=norm_type)
