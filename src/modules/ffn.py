from itertools import zip_longest
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from src.modules.functional import ACT2FN
from src.modules.linear import GeneralizedLinear, _GeneralizedLinear


class LeanFFN(nn.Module):
    """
    A transformer FFN module that doesn't hog your GPU memory. Uses a manually optimized differentiation algorithm.

    :param hidden_size: base hidden size of the transformer
    :param intermediate_size: a (typically larger) hidden dimension where activation is applied
    :param activation: a pytorch nonlinearity to use in the intermediate layer
    :param gated: use gated activations based on https://arxiv.org/abs/2002.05202 and https://arxiv.org/abs/2102.11972
      note: gated activations require 1.5x more parameters compared to their non-gated variants.
    :param layer_norm_eps: see torch.nn.functional.layer_norm
    :param sandwich_norm: if set, applies an additional layer norm to projected attention outputs before residuals,
       as proposed in the CogView paper ( arXiv:2105.13290 ). This is meant to make fp16 training
       more stable for deep transformers. This technique is also a part of NormFormer ( arXiv:2110.09456 )
    :param dropout: hidden dropout probability, applied to the output projection (before adding residual)
    :param residual: if True, adds the original layer input to the final layer output

    :param dense_i2h: custom *first* linear layer (hidden_size -> intermediate_size or 2x indermediate_size)
    :param dense_h2o: custom *second* linear layer (intermediate_size -> hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation=ACT2FN["gelu_fused"],
        gated: bool = False,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.0,
        sandwich_norm: bool = False,
        dense_i2h: Optional[nn.Linear] = None,
        dense_h2o: Optional[nn.Linear] = None,
        residual: bool = True,
    ):
        super().__init__()
        i2h_out_features = intermediate_size * 2 if gated else intermediate_size
        self.dense_i2h = nn.Linear(hidden_size, i2h_out_features) if dense_i2h is None else dense_i2h
        self.dense_h2o = nn.Linear(intermediate_size, hidden_size) if dense_h2o is None else dense_h2o
        assert type(self.dense_i2h) in (
        nn.Linear, GeneralizedLinear), "only Linear and GeneralizedLinear are supported"
        assert type(self.dense_h2o) in (
        nn.Linear, GeneralizedLinear), "only Linear and GeneralizedLinear are supported"
        assert self.dense_i2h.in_features == self.dense_h2o.out_features == hidden_size
        assert self.dense_i2h.out_features == i2h_out_features and self.dense_h2o.in_features == intermediate_size
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.sandwich_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if sandwich_norm else None
        self.activation = activation
        self.gated = gated
        self.dropout = dropout
        self.residual = residual

    def forward(self, input):
        sandwich_ln_weight = sandwich_ln_bias = None
        if self.sandwich_norm is not None:
            sandwich_ln_weight, sandwich_ln_bias = self.sandwich_norm.weight, self.sandwich_norm.bias
        i2h_lowrank_first = i2h_lowrank_second = h2o_lowrank_first = h2o_lowrank_second = None
        i2h_forward_indices = i2h_backward_indices = h2o_forward_indices = h2o_backward_indices = None
        if isinstance(self.dense_i2h, GeneralizedLinear):
            i2h_lowrank_first, i2h_lowrank_second = self.dense_i2h.get_combined_lowrank_components()
            i2h_forward_indices = self.dense_i2h.shared_matrix.forward_indices
            i2h_backward_indices = self.dense_i2h.shared_matrix.backward_indices
        if isinstance(self.dense_h2o, GeneralizedLinear):
            h2o_lowrank_first, h2o_lowrank_second = self.dense_h2o.get_combined_lowrank_components()
            h2o_forward_indices = self.dense_h2o.shared_matrix.forward_indices
            h2o_backward_indices = self.dense_h2o.shared_matrix.backward_indices

        output = _LeanFFN.apply(
            input,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.dense_i2h.weight,
            self.dense_i2h.bias,
            i2h_lowrank_first,
            i2h_lowrank_second,
            i2h_forward_indices,
            i2h_backward_indices,
            self.dense_h2o.weight,
            self.dense_h2o.bias,
            h2o_lowrank_first,
            h2o_lowrank_second,
            h2o_forward_indices,
            h2o_backward_indices,
            sandwich_ln_weight,
            sandwich_ln_bias,
            self.activation,
            self.gated,
            self.dropout,
            self.training,
            self.layer_norm.eps,
            self.residual,
        )
        return output


class _LeanFFN(torch.autograd.Function):
    """Autograd function for transformer FFN, manually optimized to reduce memory without affecting performance"""

    @staticmethod
    def _apply_activation(pre_activation: torch.Tensor, activation: callable, gated: bool):
        if not gated:
            return activation(pre_activation)
        else:
            pre_gate, lin = pre_activation.split(pre_activation.shape[-1] // 2, dim=-1)
            return activation(pre_gate).mul_(lin)

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        i2h_weight: torch.Tensor,
        i2h_bias: Optional[torch.Tensor],
        i2h_lowrank_first: Optional[torch.Tensor],
        i2h_lowrank_second: Optional[torch.Tensor],
        i2h_forward_indices: Optional[torch.IntTensor],
        i2h_backward_indices: Optional[torch.IntTensor],
        h2o_weight: torch.Tensor,
        h2o_bias: Optional[torch.Tensor],
        h2o_lowrank_first: Optional[torch.Tensor],
        h2o_lowrank_second: Optional[torch.Tensor],
        h2o_forward_indices: Optional[torch.IntTensor],
        h2o_backward_indices: Optional[torch.IntTensor],
        sandwich_ln_weight: Optional[torch.Tensor],
        sandwich_ln_bias: Optional[torch.Tensor],
        activation: callable,
        gated: bool,
        dropout: float,
        training: bool,
        ln_eps: float,
        residual: bool,
    ):
        ctx._dropout, ctx._training, ctx._ln_eps = dropout, training, ln_eps
        ctx._activation, ctx._gated, ctx._residual = activation, gated, residual
        ctx._use_sandwich = sandwich_ln_weight is not None

        dropout_mask, pre_sandwich = None, None  # optional tensors to save
        input_2d = input.view(-1, input.shape[-1])

        input_ln = F.layer_norm(input_2d, input.shape[-1:], ln_weight, ln_bias, ln_eps)

        pre_activation, *i2h_tensors = _GeneralizedLinear._forward_impl(
            input_ln, i2h_weight, i2h_bias, i2h_lowrank_first, i2h_lowrank_second, i2h_forward_indices,
            i2h_backward_indices
        )

        hid_act = _LeanFFN._apply_activation(pre_activation, ctx._activation, ctx._gated)

        out, *h2o_tensors = _GeneralizedLinear._forward_impl(
            hid_act, h2o_weight, h2o_bias, h2o_lowrank_first, h2o_lowrank_second, h2o_forward_indices,
            h2o_backward_indices
        )

        if ctx._use_sandwich:
            pre_sandwich = out
            out = F.layer_norm(pre_sandwich, pre_sandwich.shape[-1:], sandwich_ln_weight, sandwich_ln_bias, eps=ln_eps)

        out = F.dropout(out, dropout, training, inplace=True)
        if training and dropout:
            dropout_mask = (out == 0.0).to(torch.int8)

        if residual:
            out = torch.add(out, input_2d, out=out if 'xla' not in out.device.type else None)

        assert i2h_tensors[0] is input_ln and h2o_tensors[0] is hid_act  # we can rematerialize these tensors
        tensors_to_save = [
            input, pre_activation, ln_weight, ln_bias, pre_sandwich, sandwich_ln_weight, sandwich_ln_bias, dropout_mask
        ]
        tensors_to_save.extend((*i2h_tensors[1:], *h2o_tensors[1:]))
        ctx.save_for_backward(*tensors_to_save)
        ctx._num_i2h_tensors = len(i2h_tensors)
        ctx._num_h2o_tensors = len(h2o_tensors)
        return out.view(*input.shape)

    @staticmethod
    def _h2o_backward(ctx, grad_output: torch.Tensor, hid_act: torch.Tensor):
        h2o_tensors = ctx.saved_tensors[-ctx._num_h2o_tensors + 1:]
        needs_input_grad = [hid_act.requires_grad, *ctx.needs_input_grad[9:15]]
        grads = _GeneralizedLinear._backward_impl(grad_output, hid_act, *h2o_tensors,
                                                  needs_input_grad=needs_input_grad)
        return tuple(grad if needed else None for grad, needed in zip_longest(grads, needs_input_grad))

    @staticmethod
    def _i2h_backward(ctx, grad_output: torch.Tensor, input_ln: torch.Tensor):
        i2h_tensors = ctx.saved_tensors[-ctx._num_i2h_tensors - ctx._num_h2o_tensors + 2: -ctx._num_h2o_tensors + 1]
        needs_input_grad = [input_ln.requires_grad, *ctx.needs_input_grad[3:9]]
        grads = _GeneralizedLinear._backward_impl(grad_output, input_ln, *i2h_tensors,
                                                  needs_input_grad=needs_input_grad)
        return tuple(grad if needed else None for grad, needed in zip_longest(grads, needs_input_grad))

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_input = grad_ln_weight = grad_ln_bias = grad_sandwich_ln_weight = grad_sandwich_ln_bias = None
        input, pre_activation, ln_weight, ln_bias, = ctx.saved_tensors[:4]
        pre_sandwich, sandwich_ln_weight, sandwich_ln_bias, dropout_mask = ctx.saved_tensors[4: 8]
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])

        # backward(... -> sandwich_norm -> dropout -> residual)
        grad_residual_2d = grad_output_2d if ctx._residual else None
        if dropout_mask is not None:
            grad_output_2d = grad_output_2d.mul(dropout_mask.to(grad_output_2d.dtype))
        if ctx._use_sandwich:
            assert pre_sandwich is not None
            with torch.enable_grad():
                required_grad = pre_sandwich.requires_grad
                pre_sandwich.requires_grad_(True)
                sandwich_out = F.layer_norm(
                    pre_sandwich, pre_sandwich.shape[-1:], sandwich_ln_weight, sandwich_ln_bias, eps=ctx._ln_eps
                )
                grad_output, grad_sandwich_ln_weight, grad_sandwich_ln_bias = torch.autograd.grad(
                    sandwich_out, [pre_sandwich, sandwich_ln_weight, sandwich_ln_bias], grad_outputs=grad_output_2d
                )
                pre_sandwich.requires_grad_(required_grad)
                del pre_sandwich, sandwich_out

        # backward(... -> nonlinearity -> intermediate_layernorm -> linear_h2o -> ...)
        input_2d = input.view(-1, input.shape[-1])
        grad_h2o_output_2d = grad_output.view(-1, grad_output.shape[-1])

        with torch.enable_grad():
            # rematerialize activation
            pre_activation.requires_grad_(True)
            hid_act = _LeanFFN._apply_activation(pre_activation, ctx._activation, ctx._gated)

            with torch.no_grad():
                (grad_hid_act, grad_h2o_weight, grad_h2o_bias, grad_h2o_lowrank_first, grad_h2o_lowrank_second,
                 unused_grad_forward_indices, unused_grad_backward_indices) = \
                    _LeanFFN._h2o_backward(ctx, grad_h2o_output_2d, hid_act)

            (grad_hid,) = torch.autograd.grad(hid_act, pre_activation, grad_outputs=grad_hid_act)
            pre_activation.requires_grad_(False)
            del hid_act

        # backward(... -> input_layernorm -> linear_i2h -> ...)
        with torch.enable_grad():
            # rematerialize input_ln
            input_2d.requires_grad_(True)
            input_ln_2d = F.layer_norm(input_2d, input.shape[-1:], ln_weight, ln_bias, ctx._ln_eps)

            with torch.no_grad():
                (grad_input_ln_2d, grad_i2h_weight, grad_i2h_bias, grad_i2h_lowrank_first, grad_i2h_lowrank_second,
                 unused_grad_forward_indices, unused_grad_backward_indices) = \
                    _LeanFFN._i2h_backward(ctx, grad_hid, input_ln_2d)

            if any(ctx.needs_input_grad[0:3]):
                partial_grad_input_2d, grad_ln_weight, grad_ln_bias = torch.autograd.grad(
                    outputs=input_ln_2d, inputs=[input_2d, ln_weight, ln_bias], grad_outputs=grad_input_ln_2d
                )
            del input_2d, input_ln_2d, grad_input_ln_2d

        # add up residual grads
        if ctx.needs_input_grad[0]:
            grad_input = partial_grad_input_2d
            if ctx._residual:
                grad_input = grad_input.add_(grad_residual_2d)
            grad_input = grad_input.view(*input.shape)

        return (grad_input, grad_ln_weight, grad_ln_bias,
                grad_i2h_weight, grad_i2h_bias, grad_i2h_lowrank_first, grad_i2h_lowrank_second, None, None,
                grad_h2o_weight, grad_h2o_bias, grad_h2o_lowrank_first, grad_h2o_lowrank_second, None, None,
                grad_sandwich_ln_weight, grad_sandwich_ln_bias, None, None, None, None, None, None)
