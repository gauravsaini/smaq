"""
SMAQ attention scoring over compressed history plus exact recent tokens.

The compressed path uses SMAQ's asymmetric score computation for keys and
standard value dequantization for the weighted sum.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from smaq.core import AttentionBackend
from smaq.kv_cache import dequantize_values
from smaq.store import CompressedKVStore, FlatCache

MIN_HISTORY_FOR_SMAQ = 16


def compute_hybrid_attention(
    query: torch.Tensor,
    store: CompressedKVStore,
    recent_k: Optional[torch.Tensor],
    recent_v: Optional[torch.Tensor],
    num_query_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention output combining compressed history and exact recent KV."""
    head_dim = store.head_dim
    num_kv_heads = store.num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_history = flat is not None and flat.num_tokens >= MIN_HISTORY_FOR_SMAQ
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return torch.zeros(
            query.shape[0],
            num_query_heads,
            head_dim,
            device=query.device,
            dtype=query.dtype,
        )

    gqa_ratio = num_query_heads // num_kv_heads

    if has_history and not has_recent:
        hist_scores = _quantized_scores(query, flat, store.quantizer, gqa_ratio, num_kv_heads, scale)
        hist_values = dequantize_values(flat.value_q, store.value_group_size)
        weights = F.softmax(hist_scores, dim=-1)
        return _apply_weights(weights, hist_values, gqa_ratio, num_kv_heads, query.dtype)

    if not has_history and has_recent:
        return _attend_exact_only(query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale)

    hist_scores = _quantized_scores(query, flat, store.quantizer, gqa_ratio, num_kv_heads, scale)
    recent_scores = _exact_scores(query, recent_k, gqa_ratio, num_kv_heads, scale)
    logits = torch.cat([hist_scores, recent_scores], dim=-1)
    weights = F.softmax(logits, dim=-1)

    hist_len = hist_scores.shape[-1]
    hist_values = dequantize_values(flat.value_q, store.value_group_size)
    recent_values = recent_v.transpose(0, 1)

    out_hist = _apply_weights(weights[..., :hist_len], hist_values, gqa_ratio, num_kv_heads, query.dtype)
    out_recent = _apply_weights(weights[..., hist_len:], recent_values, gqa_ratio, num_kv_heads, query.dtype)
    return out_hist + out_recent


def _quantized_scores(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Compute logits against SMAQ-compressed historical keys."""
    q = query.float().view(query.shape[0], num_kv_heads, gqa_ratio, query.shape[-1]).permute(1, 2, 0, 3)
    head_scores = []

    for head_idx in range(num_kv_heads):
        key_q = quantizer.select_head(flat.key_q, head_idx)
        if getattr(quantizer, "supports_kernel", False):
            scores = quantizer.attention_score(q[head_idx], key_q, scale=scale, use_kernel=True)
        else:
            scores = quantizer.attention_score(q[head_idx], key_q, scale=scale)
        head_scores.append(scores)

    stacked = torch.stack(head_scores, dim=0)
    return stacked.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], flat.num_tokens)


def _exact_scores(
    query: torch.Tensor,
    recent_k: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Compute exact logits against the recent ring buffer."""
    q = query.float().view(query.shape[0], num_kv_heads, gqa_ratio, query.shape[-1]).permute(1, 2, 0, 3)
    k = recent_k.transpose(0, 1).float().unsqueeze(1)
    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    return scores.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], recent_k.shape[0])


def _apply_weights(
    weights: torch.Tensor,
    values: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply attention weights to either compressed-history or exact values."""
    v = values.float().unsqueeze(1)
    w = weights.float().view(weights.shape[0], num_kv_heads, gqa_ratio, weights.shape[-1]).permute(1, 2, 0, 3)
    out = torch.einsum("hgtn,hgnd->hgtd", w, v)
    return out.permute(2, 0, 1, 3).reshape(weights.shape[0], num_kv_heads * gqa_ratio, values.shape[-1]).to(out_dtype)


def _attend_exact_only(
    query: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Exact attention over the recent ring buffer only."""
    scores = _exact_scores(query, recent_k, gqa_ratio, num_kv_heads, scale)
    weights = F.softmax(scores, dim=-1)
    return _apply_weights(weights, recent_v.transpose(0, 1), gqa_ratio, num_kv_heads, query.dtype)


class HybridAttentionBackend(AttentionBackend):
    """Concrete torch attention backend using compressed history and exact tail."""

    def compute(
        self,
        query,
        store,
        recent_k,
        recent_v,
        num_query_heads,
        scale: Optional[float] = None,
    ):
        return compute_hybrid_attention(
            query=query,
            store=store,
            recent_k=recent_k,
            recent_v=recent_v,
            num_query_heads=num_query_heads,
            scale=scale,
        )
