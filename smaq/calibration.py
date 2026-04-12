"""Calibration providers for the generic SMAQ core."""

from __future__ import annotations

from typing import Any, Mapping

from smaq.core import CalibrationProvider


class IdentityCalibrationProvider(CalibrationProvider):
    """Return no explicit calibration and let the caller use identity state."""

    def get_sigma_q(self, layer_idx: int, head_dim: int, device: Any | None = None) -> Any | None:
        return None


class StaticCalibrationProvider(CalibrationProvider):
    """Serve precomputed calibration state keyed by layer index or name."""

    def __init__(self, per_layer: Mapping[Any, Any]):
        self.per_layer = dict(per_layer)

    def get_sigma_q(self, layer_idx: int, head_dim: int, device: Any | None = None) -> Any | None:
        if layer_idx in self.per_layer:
            return self.per_layer[layer_idx]
        layer_name = f"layer_{layer_idx}"
        return self.per_layer.get(layer_name)
