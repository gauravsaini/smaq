"""Model-layout adapters for backend-agnostic KV cache handling."""

from __future__ import annotations

from typing import Any

from smaq.core import LayoutInfo, ModelLayoutAdapter


class GenericModelLayoutAdapter(ModelLayoutAdapter):
    """Default adapter: treat the observed KV width as authoritative."""

    name = "generic"

    def normalize_kv(self, keys: Any, values: Any, expected_head_dim: int | None = None) -> LayoutInfo:
        key_dim = int(keys.shape[-1])
        value_dim = int(values.shape[-1])
        effective = key_dim
        unified = expected_head_dim is not None and key_dim != expected_head_dim
        return LayoutInfo(
            adapter_name=self.name,
            effective_head_dim=effective,
            observed_key_dim=key_dim,
            observed_value_dim=value_dim,
            unified_kv=unified,
        )


class QwenModelLayoutAdapter(GenericModelLayoutAdapter):
    """Qwen path: standard KV shapes, but explicit naming improves reporting."""

    name = "qwen"


class GemmaModelLayoutAdapter(GenericModelLayoutAdapter):
    """Gemma path: tolerate widened or unified KV layouts."""

    name = "gemma"

    def normalize_kv(self, keys: Any, values: Any, expected_head_dim: int | None = None) -> LayoutInfo:
        info = super().normalize_kv(keys, values, expected_head_dim=expected_head_dim)
        return LayoutInfo(
            adapter_name=self.name,
            effective_head_dim=info.observed_key_dim,
            observed_key_dim=info.observed_key_dim,
            observed_value_dim=info.observed_value_dim,
            unified_kv=expected_head_dim is not None and info.observed_key_dim != expected_head_dim,
        )


def infer_model_layout_adapter(model: Any) -> ModelLayoutAdapter:
    """Infer a layout adapter from model class/module names."""
    model_name = type(model).__name__.lower()
    module_name = type(model).__module__.lower()
    joined = f"{module_name} {model_name}"
    if "gemma" in joined:
        return GemmaModelLayoutAdapter()
    if "qwen" in joined:
        return QwenModelLayoutAdapter()
    return GenericModelLayoutAdapter()
