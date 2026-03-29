"""
SMAQ Block Vector Quantizer.

This implements the block VQ pipeline described in the paper: k-means in the
SMAQ-shaped metric space with pre-decoded centroids for zero-FLOP decode.

The paper's main results (Table 1) use this quantizer:
  - 8D blocks, 256 centroids = 1 bit/dim
  - Log-compressed spectral shaping with c=5.0

For a faster scalar deployment path, see ``quantizer.py``.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from smaq.ssf import build_smaq_metric


class BlockVQQuantized(NamedTuple):
    """Quantized representation from SMAQ block VQ."""

    indices: torch.Tensor  # (..., n_blocks) long — one centroid index per block
    n_blocks: int
    block_dim: int


# ---------------------------------------------------------------------------
# K-means (matches appendix_experiments.py)
# ---------------------------------------------------------------------------

def _kmeans(
    data: torch.Tensor,
    n_centroids: int,
    n_iters: int = 20,
    seed: int = 42,
) -> torch.Tensor:
    """K-means++ initialisation followed by Lloyd iterations."""
    n, d = data.shape
    rng = torch.Generator(device=data.device).manual_seed(seed)

    if n <= n_centroids:
        out = torch.zeros((n_centroids, d), device=data.device, dtype=data.dtype)
        out[:n] = data
        return out

    # K-means++ seeding
    indices = [torch.randint(n, (1,), generator=rng, device=data.device).item()]
    for _ in range(n_centroids - 1):
        dists = torch.cdist(data, data[indices]).min(dim=1).values
        probs = dists.square()
        denom = probs.sum()
        probs = probs / denom if denom > 0 else torch.full(
            (n,), 1.0 / n, device=data.device, dtype=data.dtype
        )
        idx = torch.multinomial(probs, 1, generator=rng).item()
        indices.append(idx)

    centroids = data[indices].clone()

    # Lloyd iterations
    for _ in range(n_iters):
        dists = torch.cdist(data, centroids)
        assignments = dists.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_centroids, device=data.device)
        new_centroids.index_add_(0, assignments, data)
        counts.index_add_(0, assignments, torch.ones(n, device=data.device))
        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(1)
        new_centroids[~mask] = centroids[~mask]
        centroids = new_centroids

    return centroids


# ---------------------------------------------------------------------------
# Block Vector Quantizer
# ---------------------------------------------------------------------------

class SMAQBlockVQ(nn.Module):
    """
    Block vector quantizer matching the SMAQ paper.

    Keys of dimension ``head_dim`` are partitioned into ``n_blocks`` blocks of
    ``block_dim`` dimensions.  Each block is transformed by the per-block SMAQ
    metric **E**, quantized via k-means lookup, and reconstructed using
    *pre-decoded* centroids (**E_inv @ centroid**).

    The pre-decoded centroid design means that ``dequantize`` is a **pure table
    lookup** with zero extra FLOPs beyond standard VQ.  The encode path
    (``quantize``) does require the forward metric transform ``E @ k``.

    Typical usage::

        vq = SMAQBlockVQ(head_dim=64)
        vq.fit(cal_keys, cal_queries)          # one-time calibration
        q = vq.quantize(keys)                  # encode (runs E @ k + nearest search)
        k_hat = vq.dequantize(q)               # decode (pure lookup — zero extra FLOPs)
        scores = vq.attention_score(query, q)   # attention against quantised keys
    """

    def __init__(
        self,
        head_dim: int,
        block_dim: int = 8,
        n_centroids: int = 256,
        c: float = 5.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if head_dim % block_dim != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by block_dim={block_dim}"
            )

        self.head_dim = head_dim
        self.block_dim = block_dim
        self.n_blocks = head_dim // block_dim
        self.n_centroids = n_centroids
        self.c = c
        self._device = device or torch.device("cpu")
        self._dtype = dtype

        # Per-block metric transforms: (n_blocks, block_dim, block_dim)
        eye = torch.eye(block_dim, device=self._device, dtype=dtype)
        self.register_buffer(
            "E_blocks", eye.unsqueeze(0).expand(self.n_blocks, -1, -1).clone()
        )
        self.register_buffer(
            "E_inv_blocks", eye.unsqueeze(0).expand(self.n_blocks, -1, -1).clone()
        )
        # Centroids in E-space (for encoding)
        self.register_buffer(
            "centroids",
            torch.zeros(self.n_blocks, n_centroids, block_dim, device=self._device, dtype=dtype),
        )
        # Pre-decoded centroids (for decoding — zero-FLOP path)
        self.register_buffer(
            "decoded_centroids",
            torch.zeros(self.n_blocks, n_centroids, block_dim, device=self._device, dtype=dtype),
        )
        self._fitted = False

    @property
    def bits_per_dim(self) -> float:
        """Effective bits per dimension: log2(n_centroids) / block_dim."""
        return math.log2(self.n_centroids) / self.block_dim

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(
        self,
        calibration_keys: torch.Tensor,
        calibration_queries: torch.Tensor,
        kmeans_iters: int = 20,
        seed: int = 42,
    ) -> "SMAQBlockVQ":
        """
        Calibrate the quantizer from data.

        This computes per-block query covariance Σ_q, builds the SMAQ metric
        ``E`` via log-compressed spectral shaping, runs k-means in the shaped
        space, and stores pre-decoded centroids ``E_inv @ centroid``.

        Args:
            calibration_keys: ``(N, head_dim)`` key vectors.
            calibration_queries: ``(N, head_dim)`` query vectors.
            kmeans_iters: Lloyd iterations for k-means.
            seed: Random seed for k-means++ initialisation.
        """
        N = calibration_keys.shape[0]
        E_list, E_inv_list, cent_list, dec_list = [], [], [], []

        for bi in range(self.n_blocks):
            bj = slice(bi * self.block_dim, (bi + 1) * self.block_dim)

            # Per-block query covariance
            q_block = calibration_queries[:, bj].float()
            Sigma_q = (q_block.T @ q_block) / N

            # Build SMAQ metric
            E, E_inv = build_smaq_metric(Sigma_q, c=self.c)
            E_list.append(E)
            E_inv_list.append(E_inv)

            # Transform calibration keys into E-space
            k_shaped = calibration_keys[:, bj].float() @ E.T

            # K-means in shaped space
            cents = _kmeans(k_shaped, self.n_centroids, kmeans_iters, seed + bi)
            cent_list.append(cents)

            # Pre-decode centroids: zero-FLOP decode path
            dec_list.append(cents @ E_inv.T)

        self.E_blocks = torch.stack(E_list).to(self._device, self._dtype)
        self.E_inv_blocks = torch.stack(E_inv_list).to(self._device, self._dtype)
        self.centroids = torch.stack(cent_list).to(self._device, self._dtype)
        self.decoded_centroids = torch.stack(dec_list).to(self._device, self._dtype)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def quantize(self, k: torch.Tensor) -> BlockVQQuantized:
        """
        Encode key vectors into block VQ indices.

        Each ``block_dim``-dimensional block is transformed by ``E`` and mapped
        to the nearest k-means centroid.

        Args:
            k: ``(..., head_dim)`` key tensor.

        Returns:
            ``BlockVQQuantized`` with indices ``(..., n_blocks)``.
        """
        batch_shape = k.shape[:-1]
        k_flat = k.reshape(-1, self.head_dim).float()

        block_indices = []
        for bi in range(self.n_blocks):
            bj = slice(bi * self.block_dim, (bi + 1) * self.block_dim)
            k_shaped = k_flat[:, bj] @ self.E_blocks[bi].T
            dists = torch.cdist(k_shaped, self.centroids[bi])
            block_indices.append(dists.argmin(dim=-1))

        indices = torch.stack(block_indices, dim=-1)
        return BlockVQQuantized(
            indices=indices.reshape(*batch_shape, self.n_blocks),
            n_blocks=self.n_blocks,
            block_dim=self.block_dim,
        )

    def dequantize(self, q: BlockVQQuantized) -> torch.Tensor:
        """
        Reconstruct keys from block VQ indices.

        Uses pre-decoded centroids — this is a **pure table lookup** with
        zero extra FLOPs beyond standard VQ.

        Args:
            q: ``BlockVQQuantized`` with indices ``(..., n_blocks)``.

        Returns:
            Reconstructed keys ``(..., head_dim)``.
        """
        batch_shape = q.indices.shape[:-1]
        idx = q.indices.reshape(-1, self.n_blocks).long()

        blocks = []
        for bi in range(self.n_blocks):
            blocks.append(self.decoded_centroids[bi][idx[:, bi]])

        k_hat = torch.cat(blocks, dim=-1)
        return k_hat.reshape(*batch_shape, self.head_dim)

    # ------------------------------------------------------------------
    # Attention scoring
    # ------------------------------------------------------------------

    def attention_score(
        self,
        query: torch.Tensor,
        quantized_key: BlockVQQuantized,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute attention scores ``<query, k_hat>``.

        Uses pre-decoded centroids so cost is identical to standard VQ decode
        followed by a dot product.

        Args:
            query: ``(..., n_q, head_dim)``.
            quantized_key: indices ``(..., n_k, n_blocks)``.
            scale: optional attention scale factor.

        Returns:
            Scores ``(..., n_q, n_k)``.
        """
        k_hat = self.dequantize(quantized_key)
        scores = torch.matmul(query.float(), k_hat.float().transpose(-2, -1))
        if scale is not None:
            scores = scores * scale
        return scores.to(query.dtype)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def logit_mse(self, queries: torch.Tensor, keys: torch.Tensor) -> float:
        """
        Compute held-out logit MSE: ``E[(q^T (k - k_hat))^2]``.

        This is the paper's primary evaluation metric.
        """
        k_hat = self.dequantize(self.quantize(keys))
        delta = keys.float() - k_hat.float()
        return (queries.float() * delta).sum(dim=-1).square().mean().item()

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize (for smoke tests)."""
        return self.dequantize(self.quantize(k))
