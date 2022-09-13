"""
Auxiliary modules for implementing Rotary Position Embeddings
Original paper: https://arxiv.org/abs/2104.09864
Based on reference implementation from https://blog.eleuther.ai/rotary-embeddings
"""

import torch
import torch.distributed
import torch.nn as nn

from logging import getLogger
from src.modules.functional import maybe_script

logger = getLogger(__file__)


class RotaryEmbeddings(nn.Module):
    """Applies rotary position embeddings to a tensor, uses caching to improve performance"""

    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        self.dim, self.base = dim, base
        self._rotate = maybe_script(rotate)
        self._get_auxiliary_tensors = maybe_script(get_auxiliary_tensors)
        self.register_buffer("cos", torch.empty(1, dim), persistent=False)
        self.register_buffer("sin", torch.empty(1, dim), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0):
        """
        :param x: tensor of shape [batch_size, seq_len, nhead, hid_size]
        :param offset: add this value to all position indices
        """
        seq_len = x.shape[1]
        if seq_len + offset > self.cos.shape[0] or x.device != self.cos.device:
            if torch.distributed.is_initialized():
                logger.warning("Rebuilding auxiliary tensors for rotary embeddings, this may cause DDP to freeze. To "
                               "avoid this, call _set_auxiliary_tensors for maximum length before the training starts")
            self._set_auxiliary_buffers(max_len=seq_len + offset, device=x.device)
        cosines_for_position = self.cos[None, offset : seq_len + offset, None, :]
        sines_for_position = self.sin[None, offset : seq_len + offset, None, :]
        return self._rotate(x, cosines_for_position, sines_for_position)

    def _set_auxiliary_buffers(self, max_len: int, device: torch.device):
        _cos, _sin = self._get_auxiliary_tensors(max_len, self.dim, torch.float32, device, self.base)
        self.register_buffer("cos", _cos, persistent=False)
        self.register_buffer("sin", _sin, persistent=False)


@torch.no_grad()
def get_auxiliary_tensors(seq_len: int, dim: int, dtype: torch.dtype, device: torch.device, base: int):
    """
    Compute auxiliary sine and cosine tensors for rotary position embedding
    :returns: a tuple of (cos, sin) tensors of shape [seq_len, hid_size]
    """
    _buf = torch.linspace(0, -1 + 2 / dim, dim // 2, dtype=torch.float32, device=device)
    inv_freq = torch.pow(base, _buf, out=_buf).repeat(2)
    time_ix = torch.arange(seq_len, dtype=inv_freq.dtype, device=device)

    freqs = time_ix[:, None] * inv_freq[None, :]
    cos = torch.cos(freqs)
    sin = freqs.sin_()
    return cos.to(dtype), sin.to(dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """rotate pairwise coordinate using precomputed cos & sin tensors"""
    dim = x.shape[-1]
    x_left, x_right = x.split(split_size=dim // 2, dim=x.ndim - 1)
    x_rotated = torch.cat([x_right.neg(), x_left], dim=x.ndim - 1)
    return x * cos + x_rotated * sin


