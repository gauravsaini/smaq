# SMAQ: Workload-Aware Vector Quantization for KV Cache Compression

SMAQ is a KV-cache compression method for LLM inference.

Technically, SMAQ is a query-aware compression framework that replaces
conventional orthogonal preprocessing (rotations, Hadamard) with
log-compressed spectral metric shaping derived from downstream logit MSE.

## What This Means for LLMs

Most KV-cache compression methods treat all directions in the key space more or
less the same. SMAQ does not. It uses calibration queries to identify which
directions matter more to attention, then preserves those directions more
carefully during quantization.

That makes SMAQ a particularly good fit for:

- coding assistants
- domain-specific copilots
- vertical inference workloads with stable prompt patterns

## Key Idea

Standard block vector quantizers are **rotation-invariant**: if you rotate the
data, the codebook simply adapts.  This means orthogonal preprocessing
(TurboQuant's random rotation, block Hadamard, etc.) provides **zero gain** for
adaptive VQ.  SMAQ breaks this wall by changing the *metric*, not the
coordinates — reshaping quantization noise to align with query-sensitive
directions.

### Results (TinyLlama-1.1B, 8D blocks, 256 centroids)

| Method               | L4    | L8    | L12    | L16    | Mean   |
|:---------------------|------:|------:|-------:|-------:|-------:|
| Standard VQ          | 0.0%  | 0.0%  | 0.0%   | 0.0%   | 0.0%   |
| TurboQuant (rotation)| 0.0%  | 0.0%  | 0.0%   | 0.0%   | 0.0%   |
| **SMAQ (Log c=5.0)** |+5.2%  |+0.1%  |+14.1%  |+13.9%  |**+8.3%**|

### Additional Coder-Model Signal

On `Qwen/Qwen2.5-Coder-7B-Instruct`, a lightweight coder-focused trace run also
showed strong gains:

| Layer | Std VQ | SMAQ | Gain |
|:------|------:|-----:|-----:|
| L0    | 1.2157 | 0.3992 | +67.2% |
| L27   | 0.0981 | 0.0390 | +60.2% |

This is early evidence, not a full serving benchmark, but it supports the idea
that SMAQ is especially promising for coding-oriented workloads.

## Architecture

```text
smaq/
├── ssf.py               # Log-compressed spectral shaping (core math)
├── block_vq.py          # Block VQ quantizer matching the paper's experiments
├── quantizer.py         # Scalar quantizer (faster deployment alternative)
├── kv_cache.py          # SMAQ KV cache with prefill/append/attend
├── capture.py           # Ring buffer + streaming capture engine
├── store.py             # Chunked compressed KV store with lazy flatten
├── score.py             # Hybrid attention: compressed history + exact recent
├── triton_kernels.py    # Packed dequant + score Triton microkernel
├── vllm_attn_backend.py # vLLM backend shim
└── integration/
    └── vllm.py          # Full vLLM hook system: capture/hybrid/full modes
```

### Two quantizer paths

| Path | Module | What it does | Paper match? |
|:-----|:-------|:-------------|:------------|
| **Block VQ** | `block_vq.py` | K-means in SMAQ-shaped space (256 centroids, 8D blocks) | ✅ Yes — Table 1 |
| **Scalar** | `quantizer.py` | Per-dimension scalar quantization with SMAQ metric | Faster alternative |

## Usage

```python
from smaq import SMAQBlockVQ

# Calibrate from data
vq = SMAQBlockVQ(head_dim=64, block_dim=8, n_centroids=256, c=5.0)
vq.fit(calibration_keys, calibration_queries)

# Encode (applies E @ k + nearest centroid search)
quantized = vq.quantize(keys)

# Decode (pure table lookup — zero extra FLOPs)
k_hat = vq.dequantize(quantized)

# Attention score
scores = vq.attention_score(query, quantized, scale=1.0/8.0)
```

## Validation

```bash
# Verify paper claims (rotation invariance + SMAQ gain)
python validate_paper.py

# Module smoke tests
python test_modular.py

# Qwen2.5-Coder benchmark (GPU + model download)
python benchmark.py --model-id Qwen/Qwen2.5-Coder-7B-Instruct --load-in-4bit --num-samples 4 --seq-len 96 --num-layers 2

# Minimal vLLM example
python example_vllm_smaq.py --model Qwen/Qwen2.5-Coder-7B-Instruct

# Qwen 3.5 hybrid long-context stress test
python qwen35_hybrid_long_context_eval.py --task stress --model Qwen/Qwen3.5-27B
python qwen35_hybrid_long_context_eval.py --task stress --model Qwen/Qwen3.5-27B --use-smaq
```

### Triton Kernel Validation

The repo now includes a self-contained Colab-ready notebook:

- `/Users/ektasaini/Desktop/turboquant/smaq/Triton_Kernel_Validation.ipynb`

This notebook hardcodes the SMAQ metric helpers, quantizer path, and the first
real Triton microkernel for the narrow target:

- single decode-step query
- packed SMAQ keys
- fused `dequant + score`

Initial Colab validation on a Tesla T4 showed:

| Metric | Result |
|:-------|------:|
| Max abs diff vs Torch | `5.72e-05` |
| Mean abs diff vs Torch | `5.07e-06` |
| `allclose(atol=1e-4, rtol=1e-4)` | `True` |
| Torch reference | `0.3439 ms` |
| Triton kernel | `0.1515 ms` |
| Speedup | `2.27x` |

Across `256` to `4096` tokens, the observed speedup range was approximately
`1.55x` to `2.41x`. This is a kernel-level result, not yet an end-to-end vLLM
serving benchmark.

## How It Works

1. **Query covariance**: Compute per-block Σ_q from calibration queries
2. **Spectral shaping**: Apply f(λ) = log(1 + 5λ) to eigenvalues — preserves
   anisotropy while preventing finite-rate collapse
3. **Metric construction**: Build E = V · diag(f(λ)^½) · V^T (volume-normalised)
4. **K-means in shaped space**: Standard k-means on E·k directly minimises the
   logit MSE trace objective
5. **Pre-decode centroids**: Store E⁻¹ · centroid in the codebook — decode
   becomes a pure table lookup with zero extra FLOPs

## Status

- ✅ Block VQ quantizer (paper experiments)
- ✅ Scalar quantizer (deployment path)
- ✅ vLLM integration hooks
- ✅ Hybrid attention (compressed + exact)
- ✅ Triton packed `dequant + score` microkernel validated against Torch
- ✅ Coder-model benchmark path for `Qwen/Qwen2.5-Coder-7B-Instruct`
- ✅ Hybrid long-context experiment scaffold for `Qwen/Qwen3.5-27B`
- 🔧 Full fused serving path and end-to-end vLLM benchmarking still pending
