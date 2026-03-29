"""
SMAQ attention backend shim for vLLM.

This keeps the public entry point aligned with TurboQuant's layout while
delegating the real work to `smaq.integration.vllm`.
"""

from __future__ import annotations

import logging

import smaq.integration.vllm as _backend

logger = logging.getLogger("smaq.attn")

MODE_SHADOW = "shadow"
MODE_ACCUMULATE = "accumulate"
MODE_ACTIVE = "active"
_VALID_MODES = (MODE_SHADOW, MODE_ACCUMULATE, MODE_ACTIVE)

_LEGACY_TO_NEW = {
    MODE_SHADOW: _backend.MODE_CAPTURE_ONLY,
    MODE_ACCUMULATE: _backend.MODE_CAPTURE_ONLY,
    MODE_ACTIVE: _backend.MODE_HYBRID,
}

_GLOBAL_MODE = MODE_ACCUMULATE
_SMAQ_NO_ALLOC_CONFIG = None


def set_mode(mode: str):
    global _GLOBAL_MODE
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid mode: {mode}. Valid: {_VALID_MODES}")
    _GLOBAL_MODE = mode
    _backend.set_mode(_LEGACY_TO_NEW[mode])


def get_mode() -> str:
    return _GLOBAL_MODE


def install_smaq_hooks(
    model_runner,
    calibration_covariances=None,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: int | None = None,
    mode: str = MODE_ACCUMULATE,
    no_alloc: bool = False,
):
    """Legacy-compatible entry point for installing SMAQ vLLM hooks."""
    global _GLOBAL_MODE
    new_mode = _LEGACY_TO_NEW.get(mode, _backend.MODE_CAPTURE_ONLY)

    layer_states = _backend.install_hooks(
        model_runner,
        calibration_covariances=calibration_covariances,
        key_bits=key_bits,
        value_bits=value_bits,
        value_group_size=value_group_size,
        ring_capacity=buffer_size,
        initial_layers_count=initial_layers_count,
        initial_layers_key_bits=initial_layers_key_bits,
        mode=new_mode,
        no_alloc=no_alloc,
    )

    _GLOBAL_MODE = mode
    model_runner._smaq_states = layer_states
    model_runner._smaq_no_alloc = no_alloc
    return layer_states


def enable_no_alloc(
    calibration_covariances=None,
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
    initial_layers_count: int = 4,
):
    """
    Patch vLLM so SMAQ hooks are installed automatically during engine setup.

    This mirrors TurboQuant's pattern but keeps the state in-process.
    """
    global _SMAQ_NO_ALLOC_CONFIG
    _SMAQ_NO_ALLOC_CONFIG = dict(
        calibration_covariances=calibration_covariances,
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
        initial_layers_count=initial_layers_count,
    )

    try:
        from vllm.v1.executor.abstract import Executor
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.warning("[SMAQ] vLLM is not installed; enable_no_alloc is inert.")
        return

    if hasattr(Executor, "_smaq_patched"):
        return

    if not hasattr(GPUModelRunner, "_smaq_layout_patch"):
        orig_layout_update = GPUModelRunner._update_hybrid_attention_mamba_layout

        def patched_layout_update(self, kv_caches):
            for layer_name, target_layer_name in getattr(self, "shared_kv_cache_layers", {}).items():
                if layer_name not in kv_caches and target_layer_name in kv_caches:
                    kv_caches[layer_name] = kv_caches[target_layer_name]
            return orig_layout_update(self, kv_caches)

        GPUModelRunner._update_hybrid_attention_mamba_layout = patched_layout_update
        GPUModelRunner._smaq_layout_patch = True

    orig_get_specs = Executor.get_kv_cache_specs

    def patched_get_kv_cache_specs(self):
        cfg = _SMAQ_NO_ALLOC_CONFIG
        if cfg is None:
            return orig_get_specs(self)

        def _worker_install_smaq(worker):
            states = install_smaq_hooks(
                worker.model_runner,
                calibration_covariances=cfg["calibration_covariances"],
                key_bits=cfg["key_bits"],
                value_bits=cfg["value_bits"],
                buffer_size=cfg["buffer_size"],
                initial_layers_count=cfg["initial_layers_count"],
                mode=MODE_ACTIVE,
                no_alloc=True,
            )
            return {"hooks": len(states)}

        try:
            self.collective_rpc(_worker_install_smaq)
        except Exception as exc:
            logger.exception("[SMAQ] collective_rpc failed during no-alloc install: %s", exc)
        return orig_get_specs(self)

    Executor.get_kv_cache_specs = patched_get_kv_cache_specs
    Executor._smaq_patched = True
    logger.info("[SMAQ] Patched Executor for automatic no-alloc hook installation")


def free_kv_cache(model_runner):
    """Free paged KV cache for SMAQ-hooked layers."""
    return _backend.free_kv_cache(model_runner)
