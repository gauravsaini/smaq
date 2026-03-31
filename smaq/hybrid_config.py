"""Hybrid attention layer configuration for SMAQ.

Modern LLMs interleave multiple layer types — standard softmax attention,
sliding-window (local) attention, SSM/Mamba recurrences, and multi-head latent
attention (MLA).  Each type requires a different quantization policy.  This
module provides the type detection and configuration plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class LayerType(str, Enum):
    """Attention mechanism class for a single transformer layer."""

    FULL_ATTENTION = "full_attention"
    SLIDING_WINDOW = "sliding_window"
    SSM = "ssm"
    MLA = "mla"
    UNKNOWN = "unknown"

    def needs_kv_cache(self) -> bool:
        """Whether this layer type produces a KV cache that could be compressed."""
        return self in (LayerType.FULL_ATTENTION, LayerType.SLIDING_WINDOW, LayerType.MLA)

    def needs_state_compression(self) -> bool:
        """Whether this layer carries a recurrent hidden state worth compressing."""
        return self == LayerType.SSM


@dataclass
class HybridLayerConfig:
    """Per-layer configuration for hybrid-architecture SMAQ.

    Extends the base LayerConfig concept from ``integration.vllm`` with
    hybrid-specific fields.  Each layer type uses a different subset:

    - **FULL_ATTENTION**: standard SMAQ (Σ_q metric, compressed store)
    - **SLIDING_WINDOW**: ring-only buffer of size ``window_size``, no
      compressed store (keys outside the window are evicted, not compressed)
    - **SSM**: hidden-state quantization using Σ_C + error persistence metric
    - **MLA**: latent-space quantization using combined key+value metric
    """

    layer_type: LayerType = LayerType.FULL_ATTENTION

    # Core geometry
    head_dim: int = 64
    num_kv_heads: int = 1
    num_query_heads: int = 1
    layer_idx: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Full attention / shared fields
    Sigma_q: Optional[torch.Tensor] = None
    key_bits: int = 3
    value_bits: int = 2
    value_group_size: int = 32
    ring_capacity: int = 128

    # Sliding window
    window_size: Optional[int] = None

    # SSM / Mamba
    ssm_state_dim: Optional[int] = None         # state size (n in d×n)
    ssm_A_eigenvalues: Optional[torch.Tensor] = None  # discretized Ā diagonal
    Sigma_C: Optional[torch.Tensor] = None       # output projection covariance
    ssm_bits: int = 8                             # bits for state quantization

    # MLA
    mla_latent_dim: Optional[int] = None
    mla_W_UK: Optional[torch.Tensor] = None      # latent → key projection
    mla_W_UV: Optional[torch.Tensor] = None      # latent → value projection
    Sigma_attn: Optional[torch.Tensor] = None     # attention weight covariance

    @property
    def effective_ring_capacity(self) -> int:
        """Ring capacity adjusted for layer type.

        Sliding-window layers use the window size as the ring capacity to
        avoid compressing keys that will be evicted before they are read.
        """
        if self.layer_type == LayerType.SLIDING_WINDOW and self.window_size is not None:
            return self.window_size
        return self.ring_capacity


# ---------------------------------------------------------------------------
# Automatic detection helpers
# ---------------------------------------------------------------------------


def _has_ssm_signature(module) -> bool:
    """Heuristic: Mamba blocks expose A_log, D, dt_proj, x_proj, etc."""
    ssm_attrs = {"A_log", "D", "dt_proj", "x_proj"}
    return len(ssm_attrs & set(dir(module))) >= 3


def _has_mla_signature(module) -> bool:
    """Heuristic: MLA modules have kv_lora_rank, q_lora_rank, etc."""
    mla_attrs = {"kv_lora_rank", "q_lora_rank", "kv_b_proj"}
    return len(mla_attrs & set(dir(module))) >= 2


def detect_layer_type(
    module,
    config=None,
    layer_idx: int = 0,
) -> LayerType:
    """Detect the attention mechanism type for a given layer module.

    Works with HuggingFace model modules and vLLM attention impls.

    Args:
        module: The layer or attention module to inspect.
        config: Model config object (e.g. ``model.config``).
        layer_idx: Index of the layer (used for models that specify which
                   layers use which attention type, like Qwen3.5).

    Returns:
        The detected ``LayerType``.
    """
    # 1. SSM / Mamba check
    if _has_ssm_signature(module):
        return LayerType.SSM

    # 2. MLA check (DeepSeek-V2/V3 style)
    attn = getattr(module, "self_attn", module)
    impl = getattr(attn, "impl", attn)
    if _has_mla_signature(impl) or _has_mla_signature(attn):
        return LayerType.MLA

    # 3. Sliding window check — multiple detection strategies
    # 3a. Explicit per-layer attention type map (e.g., Qwen3.5)
    if config is not None:
        # Qwen3.5 style: config.attention_type_list or config.layer_types
        attn_types = getattr(config, "attention_type_list", None) or getattr(config, "layer_types", None)
        if attn_types is not None and layer_idx < len(attn_types):
            layer_type_str = str(attn_types[layer_idx]).lower()
            if "sliding" in layer_type_str or "local" in layer_type_str:
                return LayerType.SLIDING_WINDOW
            if "full" in layer_type_str or "global" in layer_type_str:
                return LayerType.FULL_ATTENTION

        # Gemma-2 style: alternating local/global
        if hasattr(config, "attention_pattern"):
            pattern = config.attention_pattern
            if isinstance(pattern, (list, tuple)) and layer_idx < len(pattern):
                if pattern[layer_idx] == "local":
                    return LayerType.SLIDING_WINDOW

        # General sliding window config attribute
        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is not None and isinstance(sliding_window, int):
            # Some models apply sliding window to all layers (e.g., Mistral)
            # unless the layer is explicitly marked as full-attention
            full_attn_layers = getattr(config, "full_attention_layers", None)
            if full_attn_layers is not None:
                if layer_idx not in full_attn_layers:
                    return LayerType.SLIDING_WINDOW
            else:
                # If no per-layer override, assume all layers are sliding window
                return LayerType.SLIDING_WINDOW

    # 4. Check module-level attributes
    if hasattr(attn, "sliding_window") and attn.sliding_window is not None:
        return LayerType.SLIDING_WINDOW

    # Default: standard full attention
    return LayerType.FULL_ATTENTION


def detect_window_size(
    module,
    config=None,
) -> int | None:
    """Extract the sliding window size for a layer.

    Returns ``None`` for non-sliding-window layers.
    """
    # Module-level attribute
    attn = getattr(module, "self_attn", module)
    if hasattr(attn, "sliding_window") and attn.sliding_window is not None:
        return int(attn.sliding_window)

    # Config-level attribute
    if config is not None:
        sw = getattr(config, "sliding_window", None)
        if sw is not None:
            return int(sw)

    return None
