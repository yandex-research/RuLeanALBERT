import functools
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.modules.functional import maybe_script


@functools.lru_cache
def get_butterfly_indices(
    out_features: int,
    in_features: int,
    block_size: int = 256,
    butterfly_size: Optional[int] = None,
    n_factors: Optional[int] = None,
    stretch: bool = False,
) -> Tuple[torch.IntTensor, torch.IntTensor]:
    """
    Get a matrix [num_output_blocks, num_active_input_blocks] with int32 indices for additive butterfly.
    The values in the matrix represent
    Based on the original implementation from https://arxiv.org/abs/2112.00029 .

    :param stretch: by default, non-square matrices will have stretched butterfly patterns,
      otherwise the square pattern will be repeated a given number of times

    :returns: tuple (forward_indices, backward_indices), where
     - (forward) indices of non-zero blocks that contribute to each output -- assuming all input blocks are flattened
     - (backward) indices of output blocks to which a given input block contributes
    """
    if butterfly_size is None:
        butterfly_size = 2 ** int(math.ceil(math.log2(min(in_features, out_features) / block_size)))
    assert out_features % in_features == 0 or in_features % out_features == 0, \
        "if matrix is not square, the longer dimension must be a multiple of the shorter dimension"
    assert out_features % block_size == 0 and in_features % block_size == 0
    log_n = int(math.log2(butterfly_size))
    n_factors = log_n if n_factors is None else n_factors
    if butterfly_size != 2 ** log_n or butterfly_size < 2:
        raise NotImplementedError("butterfly_size must be a power of 2")
    if not (1 <= n_factors <= log_n):
        raise NotImplementedError("n_factors must be a between 1 and log_2(butterfly_size)")

    twiddle = torch.ones(butterfly_size // 2, 2, 2)
    layout = sum(butterfly_factor_to_matrix(twiddle, index) for index in range(n_factors)).bool().int()
    # Convert from (butterfly_size, butterfly_size) mask to (out_features, in_features) mask
    layout = einops.repeat(
        layout,
        "b b1 -> (b f) (b1 f1)",
        f=out_features // butterfly_size,
        f1=in_features // butterfly_size,
    )
    # Convert from (out_features, in_features) mask to
    # (out_features // block_size, in_features // block_size) mask
    layout = einops.rearrange(
        layout,
        "(p blksz) (r blksz1) -> p r (blksz blksz1)",
        blksz=block_size,
        blksz1=block_size,
    )

    layout = (layout > 0).any(dim=-1)  # [out_features // block_size, in_features // block_size]
    if not stretch:
        out_blocks, in_blocks = layout.shape
        if out_blocks > in_blocks:
            ratio = out_blocks // in_blocks
            layout = layout.view(out_blocks // ratio, ratio, in_blocks).permute(1, 0, 2).reshape_as(layout)
        elif out_blocks < in_blocks:
            ratio = in_blocks // out_blocks
            layout = layout.view(out_blocks, in_blocks // ratio, ratio).permute(0, 2, 1).reshape_as(layout)

    # convert boolean layout to indices for F.embedding_bag
    num_output_blocks = out_features // block_size
    num_input_blocks = in_features // block_size
    active_blocks_per_output = layout.sum(1).unique()
    assert len(active_blocks_per_output) == 1, "butterfly layout must have the same number of blocks per row"
    active_blocks_per_output = active_blocks_per_output.item()

    active_blocks_per_input = layout.sum(0).unique()
    assert len(active_blocks_per_input) == 1, "butterfly layout must have the same number of blocks per row"
    active_blocks_per_input = active_blocks_per_input.item()

    # which input blocks should be added for i-th output
    input_block_index = layout.nonzero()[:, 1].view(num_output_blocks, active_blocks_per_output)
    # which output blocks does j-th input contribute to
    output_block_index = layout.t().nonzero()[:, 1].view(num_input_blocks, active_blocks_per_input)

    # which of the active blocks from the corresponding input_block should be used for i-th output
    active_block_index = torch.where(
        torch.eq(
            output_block_index[input_block_index],
            torch.arange(len(input_block_index))[:, None, None],
        )
    )[-1].view(input_block_index.shape)

    forward_indices = input_block_index * active_blocks_per_input + active_block_index
    backward_indices = output_block_index
    return forward_indices.to(torch.int32), backward_indices.to(torch.int64)  # dtypes tuned for max throughput


def butterfly_factor_to_matrix(twiddle: torch.Tensor, factor_index: int) -> torch.Tensor:
    """
    Let b be the base (most commonly 2).
    Parameters:
        twiddle: (n // b, b, b)
        factor_index: an int from 0 to log_b(n) - 1
    """
    n_div_b, b, _ = twiddle.shape
    n = b * n_div_b
    log_b_n = int(math.log(n) / math.log(b))
    assert n == b ** log_b_n, f"n must be a power of {b}"
    assert twiddle.shape == (n // b, b, b)
    assert 0 <= factor_index <= log_b_n
    stride = b ** factor_index
    x = einops.rearrange(torch.eye(n), "bs (diagblk j stride) -> bs diagblk j stride", stride=stride, j=b)
    t = einops.rearrange(twiddle, "(diagblk stride) i j -> diagblk stride i j", stride=stride)
    out = torch.einsum("d s i j, b d j s -> b d i s", t, x)
    out = einops.rearrange(out, "b diagblk i stride -> b (diagblk i stride)")
    return out.t()  # Transpose because we assume the 1st dimension of x is the batch dimension


@maybe_script
def butterfly_matmul(input: torch.Tensor, weight: torch.Tensor, forward_indices: torch.Tensor) -> torch.Tensor:
    """
    :param input: tensor [*batch_dims, in_features]
    :param weight: tensor [in_features, active_blocks_per_input, block_size]
    :param forward_indices: the first output of get_butterfly_indices(...)
    :returns: tensor [*batch_dims, out_features]
    """
    assert input.shape[-1] == weight.shape[0]
    in_features, active_blocks_per_input, block_size = weight.shape
    num_input_blocks = in_features // block_size
    batch_dims = input.shape[:-1]
    input = input.flatten(0, -2)

    input_permuted = input.t().view(input.shape[1] // block_size, block_size, input.shape[0])
    output_blocks = torch.matmul(weight.view(num_input_blocks, -1, block_size), input_permuted)
    # ^-- shape: [num_input_blocks, (active_blocks_per_input * block_size), flat_batch_dims]

    blocks_for_indexing = output_blocks.view(num_input_blocks * active_blocks_per_input, block_size * input.shape[0])
    # ^-- shape: [(num_input_blocks * active_blocks_per_input),  (block_size, flat_batch_dims)]

    aggregated_blocks = F.embedding_bag(forward_indices, blocks_for_indexing, mode="sum")
    # ^-- shape: [num_ouput_blocks, (block_size, flat_batch_dims)]

    outputs = aggregated_blocks.view(-1, input.shape[0]).t()
    # ^-- shape: [flat_batch_dims, (num_output_blocks * block_size)] aka [flat_batch_dims, out_features]
    return outputs.view(batch_dims + outputs.shape[-1:])


@maybe_script
def butterfly_matmul_backward(
        grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, backward_indices: torch.Tensor,
        input_requires_grad: bool = True, weight_requires_grad: bool = True):
    """Compute gradients of butterfly_matmul w.r.t. input and/or weight without relying on pytorch autograd"""
    assert input_requires_grad or weight_requires_grad, "computing backward but none of the inputs requires grad"
    grad_input = grad_weight = torch.empty(0)
    out_features = grad_output.shape[-1]
    in_features, active_blocks_per_input, block_size = weight.shape
    num_input_blocks = input.shape[-1] // block_size
    num_output_blocks = out_features // block_size
    grad_output_flat = grad_output.flatten(0, -2)
    input_flat = input.flatten(0, -2)

    flat_batch_dims = grad_output_flat.shape[0]

    grad_aggregated_blocks = grad_output_flat.t().reshape(num_output_blocks, (block_size * flat_batch_dims))
    # [num_output_blocks, (block_size, flat_batch_dims)]

    grad_blocks_for_indexing = F.embedding(backward_indices, grad_aggregated_blocks).flatten(0, -2)
    # ^-- shape: [(num_input_blocks * active_blocks_per_input),  (block_size, flat_batch_dims)]

    grad_output_blocks = grad_blocks_for_indexing.view(
        num_input_blocks, active_blocks_per_input * block_size, flat_batch_dims
    )
    # ^-- shape: [num_input_blocks, (active_blocks_per_input * block_size), flat_batch_dims]

    if input_requires_grad:
        grad_input_permuted = torch.matmul(
            weight.view(num_input_blocks, -1, block_size).permute(0, 2, 1), grad_output_blocks
        )
        grad_input = grad_input_permuted.flatten(0, -2).t().view(grad_output.shape[:-1] + input.shape[-1:])

    if weight_requires_grad:
        grad_weight = torch.matmul(
            grad_output_blocks, input_flat.t().view(num_input_blocks, block_size, flat_batch_dims).permute(0, 2, 1)
        ).view_as(weight)

    return grad_input, grad_weight
