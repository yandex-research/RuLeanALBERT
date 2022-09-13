import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.modules.rotary import RotaryEmbeddings


class LeanSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0,
        layer_norm_eps: float = 1e-12,
        sandwich_norm: bool = False,
        dense_qkv: Optional[nn.Linear] = None,
        dense_out: Optional[nn.Linear] = None,
        residual: bool = True,
        attention_core: Optional[nn.Module] = None,
        checkpoint_attention_core: bool = True,
        **kwargs,
    ):
        """
        Attention layer that does not hog GPU memory. Re-computes pairwise attention weights instead of storing them.

        :note: this code is relatively less optimized than FFN because attention is usually not a memory bottleneck
          for typical sequence lengths (e.g. 2048 in language or 1024 in vision). If training with longer sequences,
          one can use query chunking: running one chunk of queries at a time, without storing the full QxK matrix.
          This technique runs in O(length) memory instead of O(length^2), making it not-a-bottleneck compared to FFN

        :param hidden_size: base hidden size of the transformer, before q/k/v projections
        :param num_attention_heads: number of heads, as defined in the original transformer
        :param dropout: hidden dropout probability, applied to the output projection (before adding residual)
        :param layer_norm_eps: see torch.nn.functional.layer_norm
        :param sandwich_norm: if set, applies an additional layer norm to projected attention outputs before residuals,
           as proposed in the CogView paper ( arXiv:2105.13290 ). This is meant to make fp16 training
           more stable for deep transformers. This technique is also a part of NormFormer ( arXiv:2110.09456 )
        :param residual: if True, adds the original layer input to the final layer output
        :param attention_core: optionally provide custom attention function. See SimpleAttentionCore for inspiration.
        :param checkpoint_attention_core: re-compute attention weights during backward pass instead of storing them
        :param dense_qkv: custom QKV projection layer (hidden_size -> 3 * hidden_size)
        :param dense_out: custom output projection layer (hidden_size -> hidden_size)
        :param kwargs: additional kwargs are passed to the chosen attention core
        """
        super().__init__()
        if attention_core is None:
            attention_core = SimpleAttentionCore(hidden_size, num_attention_heads, **kwargs)
        else:
            assert len(kwargs) == 0, f"Unexpected parameters: {kwargs}"

        self.hidden_size = hidden_size
        self.attention_core = attention_core
        self.dense_qkv = nn.Linear(hidden_size, hidden_size * 3) if dense_qkv is None else dense_qkv
        self.dense_out = nn.Linear(hidden_size, hidden_size) if dense_out is None else dense_out
        assert self.dense_qkv.in_features == self.dense_out.in_features == self.dense_out.out_features == hidden_size
        assert self.dense_qkv.out_features == hidden_size * 3

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.sandwich_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if sandwich_norm else None
        self.output_dropout = nn.Dropout(dropout, inplace=False)
        self.residual, self.checkpoint_attention_core = residual, checkpoint_attention_core

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        hidden_states_ln = self.layer_norm(hidden_states)
        qkv_output = self.dense_qkv(hidden_states_ln)
        query, key, value = qkv_output.split(self.hidden_size, dim=qkv_output.ndim - 1)
        attention_output, attention_probs = self._maybe_checkpoint(
            self.attention_core, query, key, value, attention_mask
        )
        outputs = self.dense_out(attention_output)
        if self.sandwich_norm:
            outputs = self.sandwich_norm(outputs)
        outputs = self.output_dropout(outputs)
        if self.residual:
            outputs = outputs + hidden_states.to(torch.float32, copy=False)
        return (outputs, attention_probs) if output_attentions else (outputs,)

    def _maybe_checkpoint(self, func, *args):
        return checkpoint(func, *args) if torch.is_grad_enabled() and self.checkpoint_attention_core else func(*args)


class SimpleAttentionCore(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.attention_dropout = nn.Dropout(attention_probs_dropout, inplace=False)
        self.hidden_size, self.num_attention_heads = hidden_size, num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

    def forward(self, query, key, value, attention_mask):
        """
        :param query: [batch_size, query_seq_len, hidden_size]
        :param key: [batch_size, kv_seq_len, hidden_size]
        :param value: [batch_size, kv_seq_len, hidden_size]
        :param attention_mask: float [(optional heads), batch, query_seq_len, kv_seq_length]
        :note: attention_mask should be equal to zero for non-masked tokens and a large negative value for masked ones
        :return: (outputs, probs)
          - outputs shape: [batch_size, query_seq_len, hidden_size]
          - probs shape: [batch_size, num_heads, query_seq_len, kv_seq_len]
        """
        if attention_mask is not None:
            assert torch.is_floating_point(attention_mask), "expected float mask with negative values for masked items"
        return self._attention_core_forward(
            query, key, value, attention_mask, self.num_attention_heads, self.attention_dropout.p, self.training
        )

    @staticmethod
    def _attention_core_forward(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            num_attention_heads: int, attention_dropout: float, training: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # transpose from [batch, seq_length, full_hid_size] to [batch, num_heads, seq_length, head_size]
        new_query_shape = query.shape[:-1] + (num_attention_heads, -1)
        new_kv_shape = key.shape[:-1] + (num_attention_heads, -1)

        query = query.view(new_query_shape).permute(0, 2, 1, 3)
        key_transposed = key.view(new_kv_shape).permute(0, 2, 3, 1)  # swap to [..., head_size, seq_length]
        value = value.view(new_kv_shape).permute(0, 2, 1, 3)
        del key  # not to confuse with key_transposed

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key_transposed / math.sqrt(query.shape[-1]))

        if attention_mask is not None:
            attention_scores += attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = torch.dropout(attention_probs, attention_dropout, training)
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(2, 1).flatten(2)

        return attention_output, attention_probs


class RotaryAttentionCore(SimpleAttentionCore):
    """Attention core that applies rotary embeddings to queries and keys before computing dot products"""

    def __init__(
        self, hidden_size: int, num_attention_heads: int, rotary_emb: Optional[RotaryEmbeddings] = None, **kwargs
    ):
        super().__init__(hidden_size, num_attention_heads, **kwargs)
        if rotary_emb is None:
            rotary_emb = RotaryEmbeddings(self.attention_head_size)
        self.rotary_emb = rotary_emb

    def rotate(self, tensor: torch.Tensor):
        """:param tensor: query or key, shape: [batch_size, query_seq_len, hidden_size]"""
        tensor_split_heads = tensor.view(*(tensor.shape[:-1] + (self.num_attention_heads, self.attention_head_size)))
        return self.rotary_emb(tensor_split_heads).view(*tensor.shape)

    def forward(self, query, key, value, attention_mask):
        return super().forward(self.rotate(query), self.rotate(key), value, attention_mask)
