"""
Generic adapter for rotation-based scalar quantizers with per-dimension scaling.

This module bridges different rotation-based scalar quantization schemes:

  1. Accept external rotation matrix (TurboQuant random, QJL, identity, learned)
  2. Accept external codebook (Lloyd-Max centroids/boundaries from any source)
  3. Apply per-dimension scales (SMAQ diagonal metric, uniform, or custom)

Designed to be agnostic to the source of rotation and codebook — the caller
provides these, this adapter applies the scaling transformation.
"""

from __future__ import annotations

import torch

from smaq.quantizer import SMAQQuantized, _normal_ppf, _pack_indices, _unpack_indices
from smaq.ssf import ssf_log


def build_codebook(
    bits: int,
    device: torch.device,
    dtype: torch.dtype,
    centroids: torch.Tensor | None = None,
    boundaries: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build scalar codebook — accepts external centroids/boundaries or generates uniform."""
    if centroids is None:
        probs = torch.linspace(
            0.0,
            1.0,
            (2**bits) + 2,
            device=device,
            dtype=torch.float32,
        )[1:-1]
        centroids = _normal_ppf(probs).to(device=device, dtype=dtype)
    else:
        centroids = centroids.to(device=device, dtype=dtype)

    if boundaries is None:
        mids = 0.5 * (centroids[:-1] + centroids[1:])
        boundaries = torch.empty(centroids.shape[0] + 1, device=device, dtype=dtype)
        boundaries[0] = -float("inf")
        boundaries[-1] = float("inf")
        boundaries[1:-1] = mids
    else:
        boundaries = boundaries.to(device=device, dtype=dtype)

    return centroids, boundaries


def build_rotated_diagonal_metric_scales(
    Sigma_q: torch.Tensor,
    rotation: torch.Tensor | None = None,
    c: float = 5.0,
    min_scale: float = 1e-6,
) -> torch.Tensor:
    """
    Approximate the SMAQ metric by a diagonal metric in a chosen rotation basis.

    Let ``M`` be the full SMAQ metric tensor after spectral shaping. For a
    scalar quantizer with forward transform ``x -> (x R^T) * s``, we keep only
    the diagonal of ``R M R^T`` and use its square root as the per-dimension
    scale vector ``s``.

    For block-diagonal rotations (e.g., PlanarQuant 2D, IsoQuant 4D), pass a
    callable that applies the block transform, or use the block-wise variant
    ``build_block_diagonal_metric_scales``.
    """
    Sigma_q = Sigma_q.to(torch.float32)
    evals, evecs = torch.linalg.eigh(Sigma_q)
    shaped = ssf_log(evals, c=c)
    metric = evecs @ torch.diag(shaped) @ evecs.T

    if rotation is not None:
        if callable(rotation):
            rotation = rotation(torch.eye(Sigma_q.shape[0], device=Sigma_q.device, dtype=torch.float32))
        rotation = rotation.to(dtype=torch.float32, device=Sigma_q.device)
        metric = rotation @ metric @ rotation.T

    diag = metric.diagonal(dim1=-2, dim2=-1).clamp(min=min_scale)
    scales = diag.sqrt()

    log_scales = torch.log(scales)
    return torch.exp(log_scales - log_scales.mean())


def build_block_diagonal_metric_scales(
    Sigma_q: torch.Tensor,
    block_size: int = 2,
    c: float = 5.0,
    min_scale: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-coordinate scales for block-diagonal rotation (PlanarQuant, IsoQuant).

    For block-diagonal rotations, we compute the metric per block and expand to
    full per-coordinate scales.

    Args:
        Sigma_q: Query covariance matrix of shape (dim, dim)
        block_size: Size of each rotation block (2 for PlanarQuant, 4 for IsoQuant)
        c: Spectral shaping parameter
        min_scale: Minimum scale value

    Returns:
        Per-dimension scales of shape (dim,)
    """
    dim = Sigma_q.shape[0]
    n_blocks = dim // block_size
    scales_list = []

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        Sigma_block = Sigma_q[start:end, start:end].to(torch.float32)

        evals, evecs = torch.linalg.eigh(Sigma_block)
        shaped = ssf_log(evals, c=c)
        metric_block = evecs @ torch.diag(shaped) @ evecs.T

        diag_block = metric_block.diagonal().clamp(min=min_scale)
        scales_block = diag_block.sqrt()

        log_scales = torch.log(scales_block)
        scales_block = torch.exp(log_scales - log_scales.mean())

        scales_list.append(scales_block)

    if dim % block_size != 0:
        remaining = dim - n_blocks * block_size
        scales_list.append(torch.ones(remaining, device=Sigma_q.device, dtype=torch.float32))

    return torch.cat(scales_list)


class RotationAdapter(torch.nn.Module):
    """
    Generic adapter for rotation-based scalar quantizers.

    This adapter is agnostic to the source of:
      - rotation matrix (random orthogonal, QJL, identity, learned, external)
      - codebook (Lloyd-Max from any quantizer, uniform, external)

    The caller provides these components; this adapter applies per-dimension
    scaling while preserving the fixed scalar kernel layout.

    Quantization path:
        x_unit -> x_rot = x_unit R^T -> z = x_rot * s -> scalar quantize(z)

    Reconstruction path:
        z_hat -> x_rot_hat = z_hat / s -> x_hat = x_rot_hat R

    Args:
        dim: Embedding dimension
        bits: Bits per coordinate (affects packing)
        rotation: Rotation matrix R of shape (dim, dim). If None, uses identity.
        coord_scales: Per-dimension scales s of shape (dim,). If None and
            Sigma_q provided, computes SMAQ-derived diagonal scales. If both
            None, uses uniform scales (1.0).
        Sigma_q: Query covariance for SMAQ-derived scaling. Ignored if
            coord_scales provided.
        c: Spectral shaping parameter for SMAQ scaling.
        codebook_mode: How to obtain codebook.
            - "auto": Generate uniform Lloyd-Max codebook (default)
            - "external": Use provided centroids/boundaries
        centroids: External centroids of shape (2^bits,) — requires
            codebook_mode="external"
        boundaries: External boundaries of shape (2^bits + 1,) — requires
            codebook_mode="external"
        device: Computation device
        dtype: Computation dtype
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        Sigma_q: torch.Tensor | None = None,
        rotation: torch.Tensor | None = None,
        coord_scales: torch.Tensor | None = None,
        c: float = 5.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        centroids: torch.Tensor | None = None,
        boundaries: torch.Tensor | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if rotation is None:
            rotation = torch.eye(dim, dtype=torch.float32)
        rotation = rotation.to(device=self.device, dtype=dtype)

        if coord_scales is None:
            if Sigma_q is None:
                Sigma_q = torch.eye(dim, dtype=torch.float32, device=self.device)
            coord_scales = build_rotated_diagonal_metric_scales(
                Sigma_q.to(device=self.device, dtype=torch.float32),
                rotation=rotation.to(torch.float32),
                c=c,
            )

        coord_scales = coord_scales.to(device=self.device, dtype=dtype)

        codebook, full_boundaries = build_codebook(
            bits=bits,
            device=self.device,
            dtype=dtype,
            centroids=centroids,
            boundaries=boundaries,
        )

        self.register_buffer("rotation", rotation)
        self.register_buffer("coord_scales", coord_scales)
        self.register_buffer("inv_coord_scales", 1.0 / coord_scales.clamp(min=1e-6))
        self.register_buffer("centroids", codebook)
        self.register_buffer("boundaries", full_boundaries)
        self.register_buffer("decision_boundaries", full_boundaries[1:-1].contiguous())

        self._c = c
        self._Sigma_q = Sigma_q

    @classmethod
    def fit(
        cls,
        dim: int,
        bits: int,
        calibration_queries: torch.Tensor,
        calibration_keys: torch.Tensor,
        rotation: torch.Tensor | None = None,
        c_values: list[float] | None = None,
        use_rotation_values: list[bool] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        centroids: torch.Tensor | None = None,
        boundaries: torch.Tensor | None = None,
    ) -> tuple["RotationAdapter", dict]:
        """
        Fit and auto-tune the adapter on calibration data.

        Finds the optimal spectral shaping parameter c and whether to use rotation
        by evaluating logit MSE on held-out calibration data.

        Args:
            dim: Embedding dimension
            bits: Bits per coordinate
            calibration_queries: Query vectors for calibration (N, dim)
            calibration_keys: Key vectors for calibration (N, dim)
            rotation: Optional external rotation matrix
            c_values: List of c values to try (default: [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
            use_rotation_values: List of rotation options to try (default: [True, False])
            device: Computation device
            dtype: Computation dtype
            centroids: Optional external Lloyd-Max centroids
            boundaries: Optional external Lloyd-Max boundaries

        Returns:
            Tuple of (best_adapter, tuning_results)
            - best_adapter: RotationAdapter with optimal configuration
            - tuning_results: dict with 'best_c', 'best_use_rotation', 'all_results'
        """
        c_values = c_values or [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        use_rotation_values = use_rotation_values or [True, False]
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        N = calibration_queries.shape[0]
        mid = N // 2
        q_cal, k_cal = calibration_queries[:mid], calibration_keys[:mid]
        q_val, k_val = calibration_queries[mid:], calibration_keys[mid:]

        Sigma_q = (q_cal.T @ q_cal) / mid

        if rotation is None:
            rotation = torch.eye(dim, dtype=torch.float32, device=device)
        rotation = rotation.to(device=device, dtype=dtype)

        def logit_mse(q, k, k_hat):
            delta = k - k_hat
            return ((q * delta).sum(dim=1) ** 2).mean().item()

        best_c = 5.0
        best_use_rotation = True
        best_lmse = float('inf')
        all_results = []

        for use_rot in use_rotation_values:
            rot_matrix = rotation if use_rot else torch.eye(dim, dtype=torch.float32, device=device)
            for c_val in c_values:
                adapter = cls(
                    dim=dim,
                    bits=bits,
                    Sigma_q=Sigma_q,
                    rotation=rot_matrix,
                    c=c_val,
                    device=device,
                    dtype=dtype,
                    centroids=centroids,
                    boundaries=boundaries,
                )
                quantized = adapter.quantize(k_val)
                k_hat = adapter.dequantize(quantized)
                lmse = logit_mse(q_val.float(), k_val.float(), k_hat.float())

                all_results.append({
                    'c': c_val,
                    'use_rotation': use_rot,
                    'lmse': lmse,
                })

                if lmse < best_lmse:
                    best_lmse = lmse
                    best_c = c_val
                    best_use_rotation = use_rot

        best_rot = rotation if best_use_rotation else torch.eye(dim, dtype=torch.float32, device=device)
        best_adapter = cls(
            dim=dim,
            bits=bits,
            Sigma_q=Sigma_q,
            rotation=best_rot,
            c=best_c,
            device=device,
            dtype=dtype,
            centroids=centroids,
            boundaries=boundaries,
        )

        tuning_results = {
            'best_c': best_c,
            'best_use_rotation': best_use_rotation,
            'best_lmse': best_lmse,
            'all_results': all_results,
        }

        return best_adapter, tuning_results

    def rotate_query(self, query: torch.Tensor) -> torch.Tensor:
        """Project queries into the rotated, inverse-scaled basis."""
        rotated = torch.matmul(query.float(), self.rotation.T.float())
        return rotated * self.inv_coord_scales.float()

    def quantize(self, k: torch.Tensor) -> SMAQQuantized:
        """Compress key vectors of shape ``(..., dim)``."""
        norms = k.norm(dim=-1, keepdim=False)
        k_unit = k / (norms.unsqueeze(-1) + 1e-10)
        rotated = torch.matmul(k_unit.float(), self.rotation.T.float())
        shaped = rotated * self.coord_scales.float()
        indices = torch.searchsorted(self.decision_boundaries, shaped.contiguous())
        packed = _pack_indices(indices, self.bits)
        return SMAQQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q: SMAQQuantized) -> torch.Tensor:
        """Reconstruct approximate keys from packed indices."""
        indices = _unpack_indices(q.indices, q.bits, self.dim)
        shaped_hat = self.centroids[indices]
        rotated_hat = shaped_hat.float() * self.inv_coord_scales.float()
        k_hat = torch.matmul(rotated_hat, self.rotation.float())
        return (k_hat * q.norms.unsqueeze(-1)).to(self.rotation.dtype)

    def attention_score(
        self,
        query: torch.Tensor,
        quantized_key: SMAQQuantized,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Compute ``<query, key_hat>`` in the rotated diagonal metric basis."""
        query_rot = self.rotate_query(query)
        indices = _unpack_indices(quantized_key.indices, quantized_key.bits, self.dim)
        shaped_hat = self.centroids[indices]
        scores = torch.matmul(query_rot.float(), shaped_hat.float().transpose(-2, -1))
        scores = scores * quantized_key.norms.unsqueeze(-2)
        if scale is not None:
            scores = scores * scale
        return scores.to(query.dtype)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize for smoke tests."""
        return self.dequantize(self.quantize(k))


# Backward compatibility alias
SMAQWeightedRotationQuantizer = RotationAdapter
