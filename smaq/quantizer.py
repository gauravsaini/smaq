"""
SMAQ scalar quantizer — fast deployment path.

This module provides a per-dimension scalar quantizer that applies the SMAQ
spectral metric before quantising each coordinate independently.  It mirrors
TurboQuant's quantizer interface but replaces the random rotation with the
calibrated query-aware metric.

**Note:** The paper's main experiments (Table 1) use *block vector quantization*
(k-means with 256 centroids in 8D blocks) — see ``block_vq.py``.  This scalar
path is a faster deployment alternative that still benefits from metric shaping
but trades some quality for speed and simplicity.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

from smaq.ssf import build_smaq_metric


class SMAQQuantized(NamedTuple):
    """Bit-packed SMAQ key representation."""

    indices: torch.Tensor
    norms: torch.Tensor
    bits: int


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack scalar centroid indices into uint8 bytes."""
    vals_per_byte = max(1, 8 // bits)
    batch_shape = indices.shape[:-1]
    d = indices.shape[-1]

    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)

    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, vals_per_byte)
    shifts = torch.arange(vals_per_byte, device=indices.device, dtype=torch.uint8) * bits
    return (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)


def _unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Unpack scalar centroid indices from uint8 bytes."""
    vals_per_byte = max(1, 8 // bits)
    mask = (1 << bits) - 1
    shifts = torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8) * bits
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :d].long()


def _normal_ppf(probs: torch.Tensor) -> torch.Tensor:
    """Inverse standard normal CDF implemented with Torch only."""
    return math.sqrt(2.0) * torch.erfinv((2.0 * probs) - 1.0)


class SMAQQuantizer(torch.nn.Module):
    """
    Spectral Metric-Aware Quantizer.

    This is the SMAQ analogue of TurboQuantMSE: keys are normalized, mapped by
    the shaped Mahalanobis metric `E`, quantized with a scalar codebook, and
    reconstructed with `E_inv`.
    """

    def __init__(
        self,
        dim: int,
        Sigma_q: torch.Tensor | None = None,
        bits: int = 3,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        E: torch.Tensor | None = None,
        E_inv: torch.Tensor | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metric_dtype = dtype

        if E is None or E_inv is None:
            if Sigma_q is None:
                Sigma_q = torch.eye(dim, dtype=torch.float32, device=self.device)
            E, E_inv = build_smaq_metric(Sigma_q.to(torch.float32), c=5.0)

        self.register_buffer("E", E.to(self.device, dtype=metric_dtype))
        self.register_buffer("E_inv", E_inv.to(self.device, dtype=metric_dtype))

        probs = torch.linspace(
            0.0,
            1.0,
            (2**bits) + 2,
            device=self.device,
            dtype=torch.float32,
        )[1:-1]
        centroids = _normal_ppf(probs).to(metric_dtype)

        boundaries_p = torch.linspace(
            0.0,
            1.0,
            (2**bits) + 1,
            device=self.device,
            dtype=torch.float32,
        )
        clipped = boundaries_p.clamp(1e-6, 1.0 - 1e-6)
        boundaries = _normal_ppf(clipped).to(metric_dtype)
        boundaries[0] = -float("inf")
        boundaries[-1] = float("inf")

        self.register_buffer("centroids", centroids)
        self.register_buffer("boundaries", boundaries)
        self.register_buffer("decision_boundaries", boundaries[1:-1].contiguous())

    def rotate_query(self, query: torch.Tensor) -> torch.Tensor:
        """
        Project queries into the inverse metric space so dot products can be
        computed directly from SMAQ-packed keys.
        """
        return torch.matmul(query.float(), self.E_inv.T)

    def quantize(self, k: torch.Tensor) -> SMAQQuantized:
        """Compress key vectors of shape `(..., dim)`."""
        norms = k.norm(dim=-1, keepdim=False)
        k_unit = k / (norms.unsqueeze(-1) + 1e-10)
        y = torch.matmul(k_unit.float(), self.E.T)
        indices = torch.searchsorted(self.decision_boundaries, y.contiguous())
        packed = _pack_indices(indices, self.bits)
        return SMAQQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q: SMAQQuantized) -> torch.Tensor:
        """Reconstruct approximate keys from the packed representation."""
        indices = _unpack_indices(q.indices, q.bits, self.dim)
        y_hat = self.centroids[indices]
        k_hat = torch.matmul(y_hat.float(), self.E_inv.T)
        return (k_hat * q.norms.unsqueeze(-1)).to(self.E.dtype)

    def attention_score(
        self,
        query: torch.Tensor,
        quantized_key: SMAQQuantized,
        scale: float | None = None,
        use_kernel: bool = False,
    ) -> torch.Tensor:
        """
        Compute `<query, key_hat>` without materializing the full dequantized key
        vectors when possible.

        Args:
            query: `(..., n_q, dim)`
            quantized_key: SMAQ-packed keys with shape `(..., n_k, packed_dim)`
            scale: optional attention scaling factor
            use_kernel: route through the Triton wrapper when available
        """
        if use_kernel:
            from smaq.triton_kernels import smaq_attention_scores

            scores = smaq_attention_scores(self, query, quantized_key)
        else:
            query_rot = self.rotate_query(query)
            indices = _unpack_indices(quantized_key.indices, quantized_key.bits, self.dim)
            y_hat = self.centroids[indices]
            scores = torch.matmul(query_rot.float(), y_hat.float().transpose(-2, -1))
            scores = scores * quantized_key.norms.unsqueeze(-2)

        if scale is not None:
            scores = scores * scale
        return scores.to(query.dtype)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize for smoke tests."""
        return self.dequantize(self.quantize(k))
