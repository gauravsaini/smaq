# SMAQ

Paper : https://doi.org/10.5281/zenodo.19342144

SMAQ is the parent research/integration repo for **Spectral Metric-Aware Quantization**.

This repo now contains the **generic SMAQ core** and supporting benchmark scripts:

- `smaq/` — importable Python package
- `scripts/` — benchmark, quality-compare, and long-context test runners
- `smaq-mlx/` — separate repo for the MLX integration package (ignored by this repo)

## What Lives Here

The top-level `smaq` package is the reusable core layer:

- cache/attention contracts
- calibration providers
- layout adapters
- scalar and block-VQ strategy variants
- weighted scalar / rotation bridge utilities

The MLX-specific serving integration is maintained separately in
[`gauravsaini/smaq-mlx`](https://github.com/gauravsaini/smaq-mlx).

That repo is the right place for:

- real `mlx_lm` runtime integration
- backend selection across `smaq`, `turboquant`, `polarquant`, and `rotorquant`
- MLX-specific quality, PPL, and long-context benchmark work
- the public application-facing API for MLX usage

## Install

```bash
uv venv
uv pip install torch
uv pip install -e .
```

## Quick Import Check

```python
import smaq

print(smaq.__version__)
```

## Package Layout

```text
smaq/
  __init__.py
  backends.py
  block_vq.py
  calibration.py
  capture.py
  core.py
  integration/
  kv_cache.py
  layout.py
  quantizer.py
  score.py
  ssf.py
  store.py
  strategies.py
  triton_kernels.py
  vllm_attn_backend.py
  weighted_scalar.py
```

## Notes

- This parent repo is the **generic SMAQ codebase**, not the polished MLX package.
- Qwen-family MLX serving validation and the public MLX runtime API now live in [`smaq-mlx`](https://github.com/gauravsaini/smaq-mlx).
- Local experiment notes such as Gemma run logs are intentionally not tracked.
