"""
SMAQ compressed KV store.

Historical tokens are quantized in chunks and flattened lazily on first read,
mirroring the TurboQuant storage pattern.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from smaq.kv_cache import ValueQuantized, quantize_values
from smaq.quantizer import SMAQQuantized, SMAQQuantizer


class FlatCache(NamedTuple):
    """Flattened view for fast read access."""

    key_q: SMAQQuantized
    value_q: ValueQuantized
    num_tokens: int


class CompressedKVStore:
    """Chunked SMAQ KV store with lazy flattening."""

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        Sigma_q: torch.Tensor | None = None,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        device: torch.device | None = None,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = min(value_group_size, head_dim)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_idx = layer_idx

        self.quantizer = SMAQQuantizer(
            dim=head_dim,
            Sigma_q=Sigma_q,
            bits=key_bits,
            device=self.device,
            dtype=torch.float32,
        )

        self._key_chunks: list[SMAQQuantized] = []
        self._value_chunks: list[ValueQuantized] = []
        self._chunk_lengths: list[int] = []
        self._flat: Optional[FlatCache] = None

    @property
    def num_tokens(self) -> int:
        return sum(self._chunk_lengths)

    @property
    def num_chunks(self) -> int:
        return len(self._chunk_lengths)

    def append_chunk(self, key: torch.Tensor, value: torch.Tensor):
        """Quantize and append a contiguous KV chunk."""
        chunk_len = key.shape[0]
        k = key.transpose(0, 1).unsqueeze(0)
        v = value.transpose(0, 1).unsqueeze(0)

        key_q = self.quantizer.quantize(k)
        val_q = quantize_values(v, bits=self.value_bits, group_size=self.value_group_size)

        self._key_chunks.append(key_q)
        self._value_chunks.append(val_q)
        self._chunk_lengths.append(chunk_len)
        self._flat = None

    def get_flat_cache(self) -> Optional[FlatCache]:
        """Return a cached flattened view of the compressed history."""
        if not self._key_chunks:
            return None
        if self._flat is not None:
            return self._flat

        if len(self._key_chunks) == 1:
            flat_kq = _flatten_key_q(self._key_chunks[0])
            flat_vq = _flatten_value_q(self._value_chunks[0])
        else:
            flat_kq = _concat_key_q([_flatten_key_q(chunk) for chunk in self._key_chunks])
            flat_vq = _concat_value_q([_flatten_value_q(chunk) for chunk in self._value_chunks])

        self._flat = FlatCache(key_q=flat_kq, value_q=flat_vq, num_tokens=self.num_tokens)
        return self._flat

    def memory_bytes(self) -> int:
        total = 0
        for key_q in self._key_chunks:
            total += key_q.indices.nelement()
            total += key_q.norms.nelement() * 2
        for value_q in self._value_chunks:
            total += value_q.data.nelement()
            total += value_q.scales.nelement() * 2
            total += value_q.zeros.nelement() * 2
        return total

    def reset(self):
        self._key_chunks.clear()
        self._value_chunks.clear()
        self._chunk_lengths.clear()
        self._flat = None


def _flatten_key_q(key_q: SMAQQuantized) -> SMAQQuantized:
    """Collapse batch dim `(1, H, T, ...) -> (H, T, ...)`."""
    return SMAQQuantized(
        indices=key_q.indices.reshape(-1, key_q.indices.shape[-2], key_q.indices.shape[-1]).contiguous(),
        norms=key_q.norms.reshape(-1, key_q.norms.shape[-1]).contiguous(),
        bits=key_q.bits,
    )


def _flatten_value_q(value_q: ValueQuantized) -> ValueQuantized:
    """Collapse batch dim `(1, H, T, ...) -> (H, T, ...)`."""
    value_bits = value_q.bits if len(value_q) > 3 else 2
    return ValueQuantized(
        data=value_q.data.reshape(-1, value_q.data.shape[-2], value_q.data.shape[-1]).contiguous(),
        scales=value_q.scales.reshape(-1, value_q.scales.shape[-2], value_q.scales.shape[-1]).contiguous(),
        zeros=value_q.zeros.reshape(-1, value_q.zeros.shape[-2], value_q.zeros.shape[-1]).contiguous(),
        bits=value_bits,
    )


def _concat_key_q(chunks: list[SMAQQuantized]) -> SMAQQuantized:
    """Concatenate flattened key chunks along token dimension."""
    return SMAQQuantized(
        indices=torch.cat([chunk.indices for chunk in chunks], dim=-2),
        norms=torch.cat([chunk.norms for chunk in chunks], dim=-1),
        bits=chunks[0].bits,
    )


def _concat_value_q(chunks: list[ValueQuantized]) -> ValueQuantized:
    """Concatenate flattened value chunks along token dimension."""
    value_bits = chunks[0].bits if len(chunks[0]) > 3 else 2
    return ValueQuantized(
        data=torch.cat([chunk.data for chunk in chunks], dim=-2),
        scales=torch.cat([chunk.scales for chunk in chunks], dim=-2),
        zeros=torch.cat([chunk.zeros for chunk in chunks], dim=-2),
        bits=value_bits,
    )
