"""Concrete quantization strategies for the generic SMAQ core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from smaq.block_vq import BlockVQQuantized, SMAQBlockVQ
from smaq.core import CacheCapabilities, MetricStrategy, QuantizationStrategy
from smaq.quantizer import SMAQQuantized, SMAQQuantizer
from smaq.ssf import build_smaq_metric
from smaq.weighted_scalar import RotationAdapter


class IdentityMetricStrategy(MetricStrategy):
    name = "identity"

    def __init__(self, dim: int, device: torch.device | None = None):
        self.dim = dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma_q = torch.eye(dim, dtype=torch.float32, device=self.device)

    def fit(self, calibration_queries: Any, calibration_keys: Any | None = None) -> "IdentityMetricStrategy":
        return self

    def export_state(self) -> dict[str, Any]:
        return {"Sigma_q": self.sigma_q}


class SMAQFullMetricStrategy(MetricStrategy):
    name = "smaq_full_metric"

    def __init__(self, dim: int, Sigma_q: torch.Tensor | None = None, device: torch.device | None = None):
        self.dim = dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma_q = Sigma_q if Sigma_q is not None else torch.eye(dim, dtype=torch.float32, device=self.device)
        self.E, self.E_inv = build_smaq_metric(self.sigma_q.to(torch.float32), c=5.0)

    def fit(self, calibration_queries: Any, calibration_keys: Any | None = None) -> "SMAQFullMetricStrategy":
        q = calibration_queries.float()
        centered = q - q.mean(dim=0, keepdim=True)
        self.sigma_q = (centered.T @ centered) / max(1, centered.shape[0])
        self.E, self.E_inv = build_smaq_metric(self.sigma_q.to(torch.float32), c=5.0)
        return self

    def export_state(self) -> dict[str, Any]:
        return {"Sigma_q": self.sigma_q, "E": self.E, "E_inv": self.E_inv}


class SMAQDiagonalMetricState(MetricStrategy):
    name = "smaq_diagonal_metric"

    def __init__(self, dim: int, Sigma_q: torch.Tensor | None = None, device: torch.device | None = None):
        self.dim = dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma_q = Sigma_q if Sigma_q is not None else torch.eye(dim, dtype=torch.float32, device=self.device)

    def fit(self, calibration_queries: Any, calibration_keys: Any | None = None) -> "SMAQDiagonalMetricState":
        q = calibration_queries.float()
        centered = q - q.mean(dim=0, keepdim=True)
        self.sigma_q = (centered.T @ centered) / max(1, centered.shape[0])
        return self

    def export_state(self) -> dict[str, Any]:
        return {"Sigma_q": self.sigma_q}


class ExactShadowQuantized(NamedTuple):
    values: torch.Tensor


def _flatten_shadow(q: ExactShadowQuantized) -> ExactShadowQuantized:
    return ExactShadowQuantized(values=q.values.reshape(-1, q.values.shape[-2], q.values.shape[-1]).contiguous())


def _concat_shadow(chunks: list[ExactShadowQuantized]) -> ExactShadowQuantized:
    return ExactShadowQuantized(values=torch.cat([chunk.values for chunk in chunks], dim=-2))


def _select_shadow_head(q: ExactShadowQuantized, head_idx: int) -> ExactShadowQuantized:
    return ExactShadowQuantized(values=q.values[head_idx])


@dataclass
class _StrategyMixin(QuantizationStrategy):
    """Small helper for strategies backed by an existing quantizer object."""

    def flatten_quantized(self, quantized: Any) -> Any:
        raise NotImplementedError

    def concat_quantized(self, chunks: list[Any]) -> Any:
        raise NotImplementedError

    def select_head(self, quantized: Any, head_idx: int) -> Any:
        raise NotImplementedError


class IdentityScalarStrategy(_StrategyMixin):
    name = "identity_scalar"
    metric_name = "identity"
    quantization_name = "scalar"

    def __init__(self, dim: int, bits: int = 3, device: torch.device | None = None):
        self.quantizer = RotationAdapter(dim=dim, bits=bits, coord_scales=torch.ones(dim), device=device)
        self.bits = bits

    def quantize(self, keys: Any) -> Any:
        return self.quantizer.quantize(keys)

    def dequantize(self, quantized: Any) -> Any:
        return self.quantizer.dequantize(quantized)

    def attention_score(self, query: Any, quantized: Any, scale: float | None = None, **kwargs: Any) -> Any:
        return self.quantizer.attention_score(query, quantized, scale=scale)

    def flatten_quantized(self, quantized: SMAQQuantized) -> SMAQQuantized:
        return SMAQQuantized(
            indices=quantized.indices.reshape(-1, quantized.indices.shape[-2], quantized.indices.shape[-1]).contiguous(),
            norms=quantized.norms.reshape(-1, quantized.norms.shape[-1]).contiguous(),
            bits=quantized.bits,
        )

    def concat_quantized(self, chunks: list[SMAQQuantized]) -> SMAQQuantized:
        return SMAQQuantized(
            indices=torch.cat([chunk.indices for chunk in chunks], dim=-2),
            norms=torch.cat([chunk.norms for chunk in chunks], dim=-1),
            bits=chunks[0].bits,
        )

    def select_head(self, quantized: SMAQQuantized, head_idx: int) -> SMAQQuantized:
        return SMAQQuantized(
            indices=quantized.indices[head_idx],
            norms=quantized.norms[head_idx],
            bits=quantized.bits,
        )

    def memory_bytes(self, quantized: SMAQQuantized) -> int:
        return quantized.indices.nelement() + (quantized.norms.nelement() * 2)


class SMAQFullMetricScalarStrategy(IdentityScalarStrategy):
    name = "smaq_full_metric_scalar"
    metric_name = "smaq_full_metric"

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        Sigma_q: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.quantizer = SMAQQuantizer(dim=dim, bits=bits, Sigma_q=Sigma_q, device=device, dtype=dtype)
        self.bits = bits

    @property
    def supports_kernel(self) -> bool:
        return True

    def fit(self, calibration_queries: Any, calibration_keys: Any | None = None) -> "SMAQFullMetricScalarStrategy":
        self.quantizer.fit(calibration_queries)
        return self

    def attention_score(self, query: Any, quantized: Any, scale: float | None = None, **kwargs: Any) -> Any:
        use_kernel = kwargs.get("use_kernel", False)
        return self.quantizer.attention_score(query, quantized, scale=scale, use_kernel=use_kernel)


class SMAQDiagonalMetricStrategy(IdentityScalarStrategy):
    name = "smaq_diagonal_metric"
    metric_name = "smaq_diagonal_metric"

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        Sigma_q: torch.Tensor | None = None,
        rotation: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.quantizer = RotationAdapter(
            dim=dim,
            bits=bits,
            Sigma_q=Sigma_q,
            rotation=rotation,
            device=device,
            dtype=dtype,
        )
        self.bits = bits


class SMAQBlockVQStrategy(_StrategyMixin):
    name = "smaq_block_vq"
    metric_name = "smaq_full_metric"
    quantization_name = "block_vq"

    def __init__(
        self,
        head_dim: int,
        block_dim: int = 8,
        n_centroids: int = 256,
        c: float = 5.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.quantizer = SMAQBlockVQ(
            head_dim=head_dim,
            block_dim=block_dim,
            n_centroids=n_centroids,
            c=c,
            device=device,
            dtype=dtype,
        )

    def fit(self, calibration_queries: Any, calibration_keys: Any | None = None) -> "SMAQBlockVQStrategy":
        if calibration_keys is None:
            calibration_keys = calibration_queries
        self.quantizer.fit(calibration_keys, calibration_queries)
        return self

    def quantize(self, keys: Any) -> Any:
        return self.quantizer.quantize(keys)

    def dequantize(self, quantized: Any) -> Any:
        return self.quantizer.dequantize(quantized)

    def attention_score(self, query: Any, quantized: Any, scale: float | None = None, **kwargs: Any) -> Any:
        return self.quantizer.attention_score(query, quantized, scale=scale)

    def flatten_quantized(self, quantized: BlockVQQuantized) -> BlockVQQuantized:
        return BlockVQQuantized(
            indices=quantized.indices.reshape(-1, quantized.indices.shape[-2], quantized.indices.shape[-1]).contiguous(),
            n_blocks=quantized.n_blocks,
            block_dim=quantized.block_dim,
        )

    def concat_quantized(self, chunks: list[BlockVQQuantized]) -> BlockVQQuantized:
        return BlockVQQuantized(
            indices=torch.cat([chunk.indices for chunk in chunks], dim=-2),
            n_blocks=chunks[0].n_blocks,
            block_dim=chunks[0].block_dim,
        )

    def select_head(self, quantized: BlockVQQuantized, head_idx: int) -> BlockVQQuantized:
        return BlockVQQuantized(
            indices=quantized.indices[head_idx],
            n_blocks=quantized.n_blocks,
            block_dim=quantized.block_dim,
        )

    def memory_bytes(self, quantized: BlockVQQuantized) -> int:
        return quantized.indices.nelement() * quantized.indices.element_size()


class ExactShadowStrategy(_StrategyMixin):
    name = "exact_shadow"
    metric_name = "identity"
    quantization_name = "shadow_exact"

    @property
    def capabilities(self) -> CacheCapabilities:
        return CacheCapabilities(
            strategy_name=self.name,
            metric_name=self.metric_name,
            quantization_name=self.quantization_name,
            compressed_history=False,
            compressed_history_shadow_only=True,
            values_compressed=False,
            decode_uses_compressed_keys=False,
            decode_uses_compressed_values=False,
        )

    def quantize(self, keys: Any) -> Any:
        return ExactShadowQuantized(values=keys.detach().clone())

    def dequantize(self, quantized: Any) -> Any:
        return quantized.values

    def attention_score(self, query: Any, quantized: Any, scale: float | None = None, **kwargs: Any) -> Any:
        scores = torch.matmul(query.float(), quantized.values.float().transpose(-2, -1))
        if scale is not None:
            scores = scores * scale
        return scores.to(query.dtype)

    def flatten_quantized(self, quantized: ExactShadowQuantized) -> ExactShadowQuantized:
        return _flatten_shadow(quantized)

    def concat_quantized(self, chunks: list[ExactShadowQuantized]) -> ExactShadowQuantized:
        return _concat_shadow(chunks)

    def select_head(self, quantized: ExactShadowQuantized, head_idx: int) -> ExactShadowQuantized:
        return _select_shadow_head(quantized, head_idx)

    def memory_bytes(self, quantized: ExactShadowQuantized) -> int:
        return quantized.values.nelement() * quantized.values.element_size()
