"""Low-dependency tests for generic SMAQ core contracts."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PKG_ROOT = ROOT / "smaq"


def _load_module(module_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(module_name, PKG_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_minimal_package():
    pkg = sys.modules.get("smaq")
    if pkg is None:
        pkg = types.ModuleType("smaq")
        pkg.__path__ = [str(PKG_ROOT)]
        sys.modules["smaq"] = pkg
    core = _load_module("smaq.core", "core.py")
    calibration = _load_module("smaq.calibration", "calibration.py")
    layout = _load_module("smaq.layout", "layout.py")
    return core, calibration, layout


class TestGenericCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core, cls.calibration, cls.layout = _bootstrap_minimal_package()

    def test_identity_calibration_provider_returns_none(self):
        provider = self.calibration.IdentityCalibrationProvider()
        self.assertIsNone(provider.get_sigma_q(layer_idx=0, head_dim=128))

    def test_static_calibration_provider_resolves_layer_index(self):
        provider = self.calibration.StaticCalibrationProvider({0: "sigma0", "layer_1": "sigma1"})
        self.assertEqual(provider.get_sigma_q(layer_idx=0, head_dim=64), "sigma0")
        self.assertEqual(provider.get_sigma_q(layer_idx=1, head_dim=64), "sigma1")

    def test_gemma_adapter_marks_unified_kv_when_dim_changes(self):
        adapter = self.layout.GemmaModelLayoutAdapter()
        keys = types.SimpleNamespace(shape=(1, 8, 16, 256))
        values = types.SimpleNamespace(shape=(1, 8, 16, 256))
        info = adapter.normalize_kv(keys, values, expected_head_dim=128)
        self.assertEqual(info.adapter_name, "gemma")
        self.assertTrue(info.unified_kv)
        self.assertEqual(info.effective_head_dim, 256)

    def test_qwen_adapter_keeps_standard_shape(self):
        adapter = self.layout.QwenModelLayoutAdapter()
        keys = types.SimpleNamespace(shape=(1, 8, 16, 128))
        values = types.SimpleNamespace(shape=(1, 8, 16, 128))
        info = adapter.normalize_kv(keys, values, expected_head_dim=128)
        self.assertEqual(info.adapter_name, "qwen")
        self.assertFalse(info.unified_kv)
        self.assertEqual(info.effective_head_dim, 128)

    def test_cache_capabilities_shape(self):
        caps = self.core.CacheCapabilities(
            strategy_name="diag",
            metric_name="smaq_diagonal_metric",
            quantization_name="scalar",
            compressed_history=True,
            compressed_history_shadow_only=False,
            values_compressed=True,
            decode_uses_compressed_keys=True,
            decode_uses_compressed_values=False,
        )
        self.assertEqual(caps.strategy_name, "diag")
        self.assertTrue(caps.compressed_history)
        self.assertFalse(caps.compressed_history_shadow_only)


if __name__ == "__main__":
    unittest.main()
