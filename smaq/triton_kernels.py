"""
SMAQ attention score kernels.

Provides a unified entry point for computing attention scores against
SMAQ-quantized keys.  This module now includes a narrow Triton fast path for
the first deployment target: a single decode-step query against SMAQ-packed
keys, returning attention scores without materializing the full dequantized
keys.  Any unsupported shape or missing Triton/CUDA runtime falls back to the
PyTorch reference implementation.
"""

from __future__ import annotations

import logging

import torch

from smaq.quantizer import SMAQQuantized

logger = logging.getLogger("smaq.triton")

_HAS_TRITON = False
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    pass


if _HAS_TRITON:

    @triton.jit
    def _smaq_scores_kernel(
        query_ptr,
        packed_ptr,
        centroids_ptr,
        norms_ptr,
        out_ptr,
        n_qh,
        n_kv,
        packed_dim,
        d_model,
        query_stride_h,
        query_stride_d,
        packed_stride_t,
        packed_stride_p,
        norm_stride_t,
        out_stride_h,
        out_stride_t,
        bits: tl.constexpr,
        vals_per_byte: tl.constexpr,
        block_q: tl.constexpr,
        block_t: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_t = tl.program_id(1)

        q_offsets = pid_q * block_q + tl.arange(0, block_q)
        t_offsets = pid_t * block_t + tl.arange(0, block_t)

        q_mask = q_offsets < n_qh
        t_mask = t_offsets < n_kv

        acc = tl.zeros((block_q, block_t), dtype=tl.float32)
        index_mask = (1 << bits) - 1

        for p in range(0, packed_dim):
            packed_vals = tl.load(
                packed_ptr + (t_offsets * packed_stride_t) + (p * packed_stride_p),
                mask=t_mask,
                other=0,
            ).to(tl.int32)

            for v in range(0, vals_per_byte):
                d_idx = p * vals_per_byte + v
                if d_idx < d_model:
                    indices = (packed_vals >> (v * bits)) & index_mask
                    y_hat = tl.load(centroids_ptr + indices, mask=t_mask, other=0.0)
                    q_vec = tl.load(
                        query_ptr + (q_offsets * query_stride_h) + (d_idx * query_stride_d),
                        mask=q_mask,
                        other=0.0,
                    ).to(tl.float32)
                    acc += q_vec[:, None] * y_hat[None, :]

        norms = tl.load(norms_ptr + t_offsets * norm_stride_t, mask=t_mask, other=0.0).to(tl.float32)
        acc *= norms[None, :]

        tl.store(
            out_ptr + (q_offsets[:, None] * out_stride_h) + (t_offsets[None, :] * out_stride_t),
            acc,
            mask=q_mask[:, None] & t_mask[None, :],
        )


# ------------------------------------------------------------------
# Reference implementation (always used today)
# ------------------------------------------------------------------

def _torch_attention_scores(
    quantizer,
    query: torch.Tensor,
    quantized_key: SMAQQuantized,
) -> torch.Tensor:
    """PyTorch reference: unpack indices, gather centroids, matmul."""
    query_rot = quantizer.rotate_query(query)
    from smaq.quantizer import _unpack_indices

    unpacked = _unpack_indices(quantized_key.indices, quantized_key.bits, quantizer.dim)
    y_hat = quantizer.centroids[unpacked]
    scores = torch.matmul(query_rot.float(), y_hat.float().transpose(-2, -1))
    return scores * quantized_key.norms.unsqueeze(-2)


def _supports_triton_fast_path(
    query: torch.Tensor,
    quantized_key: SMAQQuantized,
) -> bool:
    """
    Narrow first kernel target:
    - query: [num_query_heads, dim]
    - packed keys: [num_tokens, packed_dim]
    - norms: [num_tokens]
    """
    return (
        _HAS_TRITON
        and query.is_cuda
        and quantized_key.indices.is_cuda
        and quantized_key.norms.is_cuda
        and query.ndim == 2
        and quantized_key.indices.ndim == 2
        and quantized_key.norms.ndim == 1
        and quantized_key.indices.shape[0] == quantized_key.norms.shape[0]
    )


def _triton_attention_scores(
    quantizer,
    query: torch.Tensor,
    quantized_key: SMAQQuantized,
) -> torch.Tensor:
    """Single-step Triton fast path matching the Torch reference exactly."""
    query_rot = quantizer.rotate_query(query).contiguous().to(torch.float32)
    packed = quantized_key.indices.contiguous()
    norms = quantized_key.norms.contiguous().to(torch.float32)
    centroids = quantizer.centroids.contiguous().to(torch.float32)

    n_qh, d_model = query_rot.shape
    n_kv, packed_dim = packed.shape
    vals_per_byte = max(1, 8 // int(quantized_key.bits))

    output = torch.empty((n_qh, n_kv), device=query.device, dtype=torch.float32)

    block_q = 8
    block_t = 64
    grid = (triton.cdiv(n_qh, block_q), triton.cdiv(n_kv, block_t))

    _smaq_scores_kernel[grid](
        query_rot,
        packed,
        centroids,
        norms,
        output,
        n_qh,
        n_kv,
        packed_dim,
        d_model,
        query_rot.stride(0),
        query_rot.stride(1),
        packed.stride(0),
        packed.stride(1),
        norms.stride(0),
        output.stride(0),
        output.stride(1),
        bits=int(quantized_key.bits),
        vals_per_byte=vals_per_byte,
        block_q=block_q,
        block_t=block_t,
        num_warps=4,
    )
    return output


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def smaq_attention_scores(
    quantizer,
    query: torch.Tensor,
    quantized_key: SMAQQuantized,
) -> torch.Tensor:
    """
    Compute ``<query, k_hat>`` against SMAQ scalar-quantized keys.

    Triton is used for the narrow validated kernel target:
    a single decode-step query tile against a 2D SMAQ-packed key matrix.
    All other cases fall back to the PyTorch reference path.
    """
    if _supports_triton_fast_path(query, quantized_key):
        try:
            return _triton_attention_scores(quantizer, query, quantized_key)
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            logger.warning("Falling back to Torch attention path after Triton failure: %s", exc)

    return _torch_attention_scores(quantizer, query, quantized_key)
