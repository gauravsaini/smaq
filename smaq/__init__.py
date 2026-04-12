from smaq.backends import TorchHybridAttentionBackend
from smaq.calibration import IdentityCalibrationProvider, StaticCalibrationProvider
from smaq.block_vq import BlockVQQuantized, SMAQBlockVQ
from smaq.core import (
    AttentionBackend,
    CacheBackend,
    CacheCapabilities,
    CalibrationProvider,
    LayoutInfo,
    MetricStrategy,
    ModelLayoutAdapter,
    QuantizationStrategy,
)
from smaq.capture import KVCaptureEngine, RingBuffer
from smaq.kv_cache import SMAQKVCache
from smaq.layout import (
    GemmaModelLayoutAdapter,
    GenericModelLayoutAdapter,
    QwenModelLayoutAdapter,
    infer_model_layout_adapter,
)
from smaq.quantizer import SMAQQuantized, SMAQQuantizer
from smaq.score import compute_hybrid_attention
from smaq.strategies import (
    ExactShadowStrategy,
    IdentityMetricStrategy,
    IdentityScalarStrategy,
    SMAQBlockVQStrategy,
    SMAQDiagonalMetricState,
    SMAQDiagonalMetricStrategy,
    SMAQFullMetricScalarStrategy,
    SMAQFullMetricStrategy,
)
from smaq.store import CompressedKVStore
from smaq.weighted_scalar import (
    RotationAdapter,
    SMAQWeightedRotationQuantizer,
    build_rotated_diagonal_metric_scales,
    build_block_diagonal_metric_scales,
    build_codebook,
)

__version__ = "0.1.0"
