"""
Backend-agnostic SMAQ core contracts.

These contracts separate:
  - calibration state
  - task metric shaping
  - key quantization layout
  - cache storage capabilities
  - model-specific KV layout quirks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CacheCapabilities:
    """Describes what a cache/backend is actually doing at runtime."""

    strategy_name: str
    metric_name: str
    quantization_name: str
    compressed_history: bool
    compressed_history_shadow_only: bool
    values_compressed: bool
    decode_uses_compressed_keys: bool
    decode_uses_compressed_values: bool


@dataclass(frozen=True)
class LayoutInfo:
    """Normalized view of a backend/model KV layout."""

    adapter_name: str
    effective_head_dim: int
    observed_key_dim: int
    observed_value_dim: int
    unified_kv: bool = False


class CalibrationProvider(ABC):
    """Provides calibration state such as per-layer query covariance."""

    @abstractmethod
    def get_sigma_q(
        self,
        layer_idx: int,
        head_dim: int,
        device: Any | None = None,
    ) -> Any | None:
        raise NotImplementedError


class MetricStrategy(ABC):
    """Owns query-aware transforms and calibration state."""

    name: str = "metric"

    @abstractmethod
    def fit(self, calibration_queries: Any, calibration_keys: Any | None = None) -> "MetricStrategy":
        raise NotImplementedError

    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        raise NotImplementedError


class QuantizationStrategy(ABC):
    """Owns encode/decode layout and score-time reconstruction."""

    name: str = "quantization"
    metric_name: str = "metric"
    quantization_name: str = "quantization"

    @property
    def capabilities(self) -> CacheCapabilities:
        return CacheCapabilities(
            strategy_name=self.name,
            metric_name=self.metric_name,
            quantization_name=self.quantization_name,
            compressed_history=True,
            compressed_history_shadow_only=False,
            values_compressed=True,
            decode_uses_compressed_keys=True,
            decode_uses_compressed_values=False,
        )

    def fit(self, calibration_queries: Any, calibration_keys: Any | None = None) -> "QuantizationStrategy":
        return self

    @property
    def supports_kernel(self) -> bool:
        return False

    @abstractmethod
    def quantize(self, keys: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def dequantize(self, quantized: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def attention_score(self, query: Any, quantized: Any, scale: float | None = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def flatten_quantized(self, quantized: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def concat_quantized(self, chunks: list[Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def select_head(self, quantized: Any, head_idx: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def memory_bytes(self, quantized: Any) -> int:
        raise NotImplementedError


class AttentionBackend(ABC):
    """Computes scores/outputs over compressed history and optional exact tail."""

    @abstractmethod
    def compute(
        self,
        query: Any,
        store: Any,
        recent_k: Any | None,
        recent_v: Any | None,
        num_query_heads: int,
        scale: Optional[float] = None,
    ) -> Any:
        raise NotImplementedError


class CacheBackend(ABC):
    """Owns prefill/decode append, flush, and historical storage."""

    @property
    @abstractmethod
    def capabilities(self) -> CacheCapabilities:
        raise NotImplementedError


class ModelLayoutAdapter(ABC):
    """Normalizes KV layout differences across model families."""

    name: str = "generic"

    def resolve_head_dim(self, model: Any, layer_idx: int) -> int:
        layer = model.layers[layer_idx]
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "head_dim"):
            return int(attn.head_dim)
        if attn is not None and hasattr(attn, "hidden_size") and hasattr(attn, "num_heads"):
            return int(attn.hidden_size // attn.num_heads)
        return 128

    @abstractmethod
    def normalize_kv(
        self,
        keys: Any,
        values: Any,
        expected_head_dim: int | None = None,
    ) -> LayoutInfo:
        raise NotImplementedError
