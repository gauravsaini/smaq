"""
SMAQ capture module.

This keeps decode cheap by buffering recent exact KV tokens and only flushing
older chunks into the compressed SMAQ store.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from smaq.store import CompressedKVStore


class RingBuffer:
    """Fixed-size ring buffer for recent exact KV tokens."""

    __slots__ = (
        "capacity",
        "num_kv_heads",
        "head_dim",
        "device",
        "dtype",
        "_k",
        "_v",
        "_pos",
        "_total_written",
    )

    def __init__(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self._k = torch.zeros(capacity, num_kv_heads, head_dim, device=device, dtype=dtype)
        self._v = torch.zeros(capacity, num_kv_heads, head_dim, device=device, dtype=dtype)
        self._pos = 0
        self._total_written = 0

    @property
    def size(self) -> int:
        return self._pos

    @property
    def total_written(self) -> int:
        return self._total_written

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        num_tokens: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Append tokens and return overflow when the ring wraps."""
        overflow_k_parts = []
        overflow_v_parts = []
        offset = 0
        remaining = num_tokens

        while remaining > 0:
            space = self.capacity - self._pos
            if space <= 0:
                overflow_k_parts.append(self._k[: self._pos].clone())
                overflow_v_parts.append(self._v[: self._pos].clone())
                self._pos = 0
                space = self.capacity

            n_write = min(remaining, space)
            self._k[self._pos : self._pos + n_write] = key[offset : offset + n_write]
            self._v[self._pos : self._pos + n_write] = value[offset : offset + n_write]
            self._pos += n_write
            offset += n_write
            remaining -= n_write

        self._total_written += num_tokens

        if overflow_k_parts:
            return torch.cat(overflow_k_parts, dim=0), torch.cat(overflow_v_parts, dim=0)
        return None

    def drain(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return buffered tokens and reset."""
        if self._pos == 0:
            return None
        k = self._k[: self._pos].clone()
        v = self._v[: self._pos].clone()
        self._pos = 0
        return k, v

    def peek(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Read current buffered tokens without draining."""
        if self._pos == 0:
            return None
        return self._k[: self._pos], self._v[: self._pos]

    def reset(self):
        self._pos = 0
        self._total_written = 0


class KVCaptureEngine:
    """Bulk capture and decode ingestion for the SMAQ compressed store."""

    def __init__(
        self,
        store: "CompressedKVStore",
        ring_capacity: int = 128,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.store = store
        self.ring = RingBuffer(
            capacity=ring_capacity,
            num_kv_heads=store.num_kv_heads,
            head_dim=store.head_dim,
            device=device or store.device,
            dtype=dtype,
        )
        self._prefill_done = False

    @property
    def total_compressed_tokens(self) -> int:
        return self.store.num_tokens

    @property
    def total_buffered_tokens(self) -> int:
        return self.ring.size

    @property
    def total_tokens(self) -> int:
        return self.total_compressed_tokens + self.total_buffered_tokens

    def ingest_prefill(self, key: torch.Tensor, value: torch.Tensor, num_tokens: int):
        """Ingest a prefill block, leaving only the recent tail exact."""
        if num_tokens <= self.ring.capacity:
            self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        else:
            n_compress = num_tokens - self.ring.capacity
            self.store.append_chunk(key[:n_compress], value[:n_compress])
            self.ring.write(key[n_compress:num_tokens], value[n_compress:num_tokens], self.ring.capacity)
        self._prefill_done = True

    def ingest_prefill_from_paged_cache(
        self,
        kv_cache_tensor: torch.Tensor,
        num_tokens: int,
        block_table: torch.Tensor,
        block_size: int,
    ):
        """Read a prefill region directly from a vLLM paged KV tensor."""
        num_blocks_needed = (num_tokens + block_size - 1) // block_size
        physical_blocks = block_table[:num_blocks_needed]

        keys_list = []
        vals_list = []
        collected = 0
        for phys_idx in physical_blocks:
            end = min(block_size, num_tokens - collected)
            keys_list.append(kv_cache_tensor[0, phys_idx, :end])
            vals_list.append(kv_cache_tensor[1, phys_idx, :end])
            collected += end

        all_k = torch.cat(keys_list, dim=0)
        all_v = torch.cat(vals_list, dim=0)
        self.ingest_prefill(all_k, all_v, num_tokens)

    def ingest_decode(self, key: torch.Tensor, value: torch.Tensor, num_tokens: int):
        """Append cheap decode tokens and flush ring overflow to the store."""
        overflow = self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        if overflow is not None:
            self.store.append_chunk(*overflow)

    def flush(self):
        """Force-flush the ring buffer into compressed storage."""
        data = self.ring.drain()
        if data is not None:
            self.store.append_chunk(*data)

    def reset(self):
        self.ring.reset()
        self.store.reset()
        self._prefill_done = False


class SlidingWindowBuffer:
    """KV buffer for sliding-window attention layers.

    Unlike ``KVCaptureEngine``, this buffer simply **discards** overflow when
    the ring wraps instead of forwarding it to a compressed store.  This is
    correct for sliding-window layers where keys older than the window are
    never attended to — compressing them would be wasted work and memory.

    The ring capacity should be set equal to the attention window size so
    that the buffer holds exactly the set of keys the layer can attend to.

    Typical usage::

        buf = SlidingWindowBuffer(
            window_size=4096,
            num_kv_heads=8,
            head_dim=64,
            device=torch.device("cuda"),
        )
        buf.ingest(key_tokens, value_tokens, num_tokens)
        recent_k, recent_v = buf.peek()
    """

    __slots__ = ("ring", "_total_evicted")

    def __init__(
        self,
        window_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.ring = RingBuffer(
            capacity=window_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        self._total_evicted = 0

    @property
    def window_size(self) -> int:
        return self.ring.capacity

    @property
    def size(self) -> int:
        return self.ring.size

    @property
    def total_evicted(self) -> int:
        return self._total_evicted

    def ingest(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        num_tokens: int,
    ):
        """Write tokens into the window buffer, silently discarding overflow.

        This is the key difference from ``KVCaptureEngine``: overflow tokens
        that exit the window are dropped, not compressed.
        """
        overflow = self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        if overflow is not None:
            self._total_evicted += overflow[0].shape[0]
            # Deliberately discard — these keys are beyond the window

    def peek(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return the current window contents without draining."""
        return self.ring.peek()

    def reset(self):
        self.ring.reset()
        self._total_evicted = 0

    def memory_bytes(self) -> int:
        """Constant memory: just the ring buffer, no compressed store."""
        return (
            self.ring._k.nelement() * self.ring._k.element_size()
            + self.ring._v.nelement() * self.ring._v.element_size()
        )

