"""
SMAQ KV cache.

Keys use the SMAQ quantizer; values use standard group quantization so the
runtime can slot into the same decode pattern as TurboQuant.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch

from smaq.quantizer import SMAQQuantized, SMAQQuantizer


class ValueQuantized(NamedTuple):
    """Bit-packed quantized values."""

    data: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: int = 2


def unpack_values(vq: ValueQuantized) -> torch.Tensor:
    """Unpack bit-packed value tensors into per-element uint8 values."""
    bits = vq.bits if len(vq) > 3 else 2
    packed = vq.data
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(
            *packed.shape[:-1], packed.shape[-1] * 4
        )
    if bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        return torch.stack([v0, v1], dim=-1).reshape(
            *packed.shape[:-1], packed.shape[-1] * 2
        )
    return packed


def quantize_values(
    v: torch.Tensor,
    bits: int = 2,
    group_size: int = 32,
) -> ValueQuantized:
    """Groupwise asymmetric quantization for value vectors."""
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    if d % group_size != 0:
        raise ValueError(f"head_dim {d} must be divisible by group_size {group_size}")

    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)
    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values

    n_levels = (2**bits) - 1
    scale = ((v_max - v_min) / n_levels).clamp(min=1e-10)
    zero = v_min

    v_q = ((v_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    if bits == 2:
        if d % 4 != 0:
            raise ValueError(f"head_dim {d} must be divisible by 4 for 2-bit packing")
        v_4 = v_q_flat.reshape(*orig_shape[:-1], d // 4, 4)
        packed = v_4[..., 0] | (v_4[..., 1] << 2) | (v_4[..., 2] << 4) | (v_4[..., 3] << 6)
        v_q_flat = packed
    elif bits == 4:
        if d % 2 != 0:
            raise ValueError(f"head_dim {d} must be divisible by 2 for 4-bit packing")
        v_2 = v_q_flat.reshape(*orig_shape[:-1], d // 2, 2)
        packed = v_2[..., 0] | (v_2[..., 1] << 4)
        v_q_flat = packed

    return ValueQuantized(
        data=v_q_flat,
        scales=scale.squeeze(-1),
        zeros=zero.squeeze(-1),
        bits=bits,
    )


def dequantize_values(vq: ValueQuantized, group_size: int = 32) -> torch.Tensor:
    """Reconstruct quantized values."""
    data = unpack_values(vq).float()
    d = data.shape[-1]
    n_groups = d // group_size

    data = data.reshape(*data.shape[:-1], n_groups, group_size)
    scales = vq.scales.unsqueeze(-1)
    zeros = vq.zeros.unsqueeze(-1)
    return (data * scales + zeros).reshape(*data.shape[:-2], d)


class SMAQKVCache:
    """Drop-in KV cache with SMAQ-compressed history plus an exact recent buffer."""

    def __init__(
        self,
        head_dim: int,
        Sigma_q: torch.Tensor | None = None,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.layer_idx = layer_idx

        self.key_quantizer = SMAQQuantizer(
            dim=head_dim,
            Sigma_q=Sigma_q,
            bits=key_bits,
            device=self.device,
            dtype=torch.float32,
        )

        self.seq_len = 0
        self.key_quantized: Optional[SMAQQuantized] = None
        self.value_quantized: Optional[ValueQuantized] = None
        self.key_buffer: Optional[torch.Tensor] = None
        self.value_buffer: Optional[torch.Tensor] = None

    def prefill(self, keys: torch.Tensor, values: torch.Tensor):
        """Capture a full prefill segment."""
        seq_len = keys.shape[-2]
        self.seq_len = seq_len

        if seq_len <= self.buffer_size:
            self.key_buffer = keys
            self.value_buffer = values
            return

        n_quant = seq_len - self.buffer_size
        keys_to_quant = keys[..., :n_quant, :]
        values_to_quant = values[..., :n_quant, :]
        self.key_buffer = keys[..., n_quant:, :]
        self.value_buffer = values[..., n_quant:, :]
        self.key_quantized = self.key_quantizer.quantize(keys_to_quant)
        self.value_quantized = quantize_values(
            values_to_quant,
            bits=self.value_bits,
            group_size=self.value_group_size,
        )

    def append(self, key: torch.Tensor, value: torch.Tensor):
        """Append a decode token into the exact ring and flush as needed."""
        self.seq_len += key.shape[-2]

        if self.key_buffer is not None:
            self.key_buffer = torch.cat([self.key_buffer, key], dim=-2)
            self.value_buffer = torch.cat([self.value_buffer, value], dim=-2)
        else:
            self.key_buffer = key
            self.value_buffer = value

        if self.key_buffer.shape[-2] > self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        n_flush = self.key_buffer.shape[-2] - self.buffer_size
        keys_flush = self.key_buffer[..., :n_flush, :]
        values_flush = self.value_buffer[..., :n_flush, :]

        self.key_buffer = self.key_buffer[..., n_flush:, :]
        self.value_buffer = self.value_buffer[..., n_flush:, :]

        new_key_q = self.key_quantizer.quantize(keys_flush)
        new_val_q = quantize_values(
            values_flush,
            bits=self.value_bits,
            group_size=self.value_group_size,
        )

        if self.key_quantized is None:
            self.key_quantized = new_key_q
            self.value_quantized = new_val_q
            return

        self.key_quantized = SMAQQuantized(
            indices=torch.cat([self.key_quantized.indices, new_key_q.indices], dim=-2),
            norms=torch.cat([self.key_quantized.norms, new_key_q.norms], dim=-1),
            bits=new_key_q.bits,
        )
        self.value_quantized = ValueQuantized(
            data=torch.cat([self.value_quantized.data, new_val_q.data], dim=-2),
            scales=torch.cat([self.value_quantized.scales, new_val_q.scales], dim=-2),
            zeros=torch.cat([self.value_quantized.zeros, new_val_q.zeros], dim=-2),
            bits=self.value_bits,
        )

    def attention_scores(self, query: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        """Compute attention scores against compressed history and exact buffer."""
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        scores_parts = []
        if self.key_quantized is not None:
            scores_parts.append(
                self.key_quantizer.attention_score(
                    query, self.key_quantized, scale=scale, use_kernel=True
                )
            )

        if self.key_buffer is not None:
            scores_parts.append(torch.matmul(query, self.key_buffer.transpose(-2, -1)) * scale)

        if not scores_parts:
            shape = (*query.shape[:-2], query.shape[-2], 0)
            return torch.empty(shape, device=query.device, dtype=query.dtype)

        return torch.cat(scores_parts, dim=-1)

    def attend(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Apply attention weights to compressed values and exact buffer values."""
        output_parts = []
        col_offset = 0

        if self.value_quantized is not None:
            n_quant = self.value_quantized.data.shape[-2]
            w_quant = attn_weights[..., col_offset : col_offset + n_quant]
            v_dequant = dequantize_values(self.value_quantized, self.value_group_size)
            output_parts.append(torch.matmul(w_quant, v_dequant))
            col_offset += n_quant

        if self.value_buffer is not None:
            n_buf = self.value_buffer.shape[-2]
            w_buf = attn_weights[..., col_offset : col_offset + n_buf]
            output_parts.append(torch.matmul(w_buf, self.value_buffer))

        if not output_parts:
            return torch.zeros(
                *attn_weights.shape[:-1],
                self.head_dim,
                device=attn_weights.device,
                dtype=attn_weights.dtype,
            )
        return sum(output_parts)

    def memory_bytes(self) -> dict[str, int]:
        """Estimate memory usage of the cache."""
        info = {"quantized_keys": 0, "quantized_values": 0, "buffer": 0, "total": 0}

        if self.key_quantized is not None:
            info["quantized_keys"] += self.key_quantized.indices.nelement()
            info["quantized_keys"] += self.key_quantized.norms.nelement() * 2

        if self.value_quantized is not None:
            info["quantized_values"] += self.value_quantized.data.nelement()
            info["quantized_values"] += self.value_quantized.scales.nelement() * 2
            info["quantized_values"] += self.value_quantized.zeros.nelement() * 2

        if self.key_buffer is not None:
            info["buffer"] += self.key_buffer.nelement() * 2
        if self.value_buffer is not None:
            info["buffer"] += self.value_buffer.nelement() * 2

        info["total"] = info["quantized_keys"] + info["quantized_values"] + info["buffer"]
        return info

    def get_seq_length(self) -> int:
        return self.seq_len
