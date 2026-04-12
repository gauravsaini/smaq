"""Concrete backends built on top of generic SMAQ contracts."""

from __future__ import annotations

from smaq.core import AttentionBackend
from smaq.score import compute_hybrid_attention


class TorchHybridAttentionBackend(AttentionBackend):
    """Torch backend for compressed history plus exact recent tail."""

    def compute(self, query, store, recent_k, recent_v, num_query_heads, scale=None):
        return compute_hybrid_attention(
            query=query,
            store=store,
            recent_k=recent_k,
            recent_v=recent_v,
            num_query_heads=num_query_heads,
            scale=scale,
        )
