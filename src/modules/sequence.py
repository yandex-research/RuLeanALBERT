"""
A module that implements sequential model type with optional keyword arguments.
When using gradient checkpoints, keyword arguments should NOT require grad.
"""
from typing import Callable, Sequence

import torch
from logging import getLogger
from torch import nn as nn
from torch.utils.checkpoint import checkpoint

logger = getLogger(__name__)


class ActiveKwargs(nn.Module):
    """
    A module with selective kwargs, compatible with sequential, gradient checkpoints and
    Usage: ony use this as a part of SequentialWithKwargs
    """

    def __init__(self, module: nn.Module, active_keys: Sequence[str], use_first_output: bool = False):
        super().__init__()
        self.module, self.active_keys, self.use_first_output = module, set(active_keys), use_first_output

    def forward(self, input: torch.Tensor, *args, **kwargs):
        kwargs = {key: value for key, value in kwargs.items() if key in self.active_keys}
        output = self.module(input, *args, **kwargs)
        if self.use_first_output and not isinstance(output, torch.Tensor):
            output = output[0]
        return output


class SequentialWithKwargs(nn.Sequential):
    def __init__(self, *modules: ActiveKwargs):
        for module in modules:
            assert isinstance(module, ActiveKwargs)
        super().__init__(*modules)
        self.gradient_checkpointing = False

    def forward(self, input: torch.Tensor, *args, **kwargs):
        kwarg_keys, kwarg_values = zip(*kwargs.items()) if (self.gradient_checkpointing and kwargs) else ([], [])
        for module in self:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                # pack kwargs with args since gradient checkpoint does not support kwargs
                input = checkpoint(self._checkpoint_forward, module, input, kwarg_keys, *kwarg_values, *args)
            else:
                input = module(input, *args, **kwargs)
        return input

    def _checkpoint_forward(self, module: Callable, input: torch.Tensor, kwarg_keys: Sequence[str], *etc):
        kwargs = {key: etc[i] for i, key in enumerate(kwarg_keys)}
        args = etc[len(kwarg_keys) :]
        return module(input, *args, **kwargs)
