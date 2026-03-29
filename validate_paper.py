"""
SMAQ paper validation suite.

Runs two concrete tests that verify the paper's core claims:
  1. Rotation invariance: random orthogonal rotation does NOT improve
     block VQ distortion for adaptive k-means codebooks.
  2. SMAQ gain: log-compressed spectral shaping reduces logit MSE
     relative to standard VQ on synthetic traces.
"""

from __future__ import annotations

import sys


def _run_validations():
    try:
        import torch
    except ImportError:
        print("PyTorch is required.  Install with:  pip install torch")
        sys.exit(1)

    from smaq.block_vq import SMAQBlockVQ, _kmeans

    SEED = 42
    DIM = 8
    N = 2000
    N_CENTROIDS = 256
    KMEANS_ITERS = 20

    rng = torch.Generator().manual_seed(SEED)

    # ------------------------------------------------------------------
    # Claim 1: Rotation invariance of adaptive block VQ
    #
    # For fully adaptive k-means, orthogonal preprocessing cannot
    # change the optimal distortion.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Claim 1: Rotation invariance of adaptive block VQ")
    print("=" * 60)

    k = torch.randn(N, DIM, generator=rng)
    q = torch.randn(N, DIM, generator=rng)

    # Random orthogonal rotation
    M = torch.randn(DIM, DIM, generator=rng)
    R, _ = torch.linalg.qr(M)

    mid = N // 2
    k_cal, k_test = k[:mid], k[mid:]
    q_test = q[mid:]

    # Standard VQ
    cents_std = _kmeans(k_cal, N_CENTROIDS, KMEANS_ITERS, SEED)
    dists_std = torch.cdist(k_test, cents_std)
    k_hat_std = cents_std[dists_std.argmin(dim=1)]
    lmse_std = ((q_test * (k_test - k_hat_std)).sum(dim=1) ** 2).mean().item()

    # Rotated VQ
    k_cal_r = k_cal @ R.T
    k_test_r = k_test @ R.T
    cents_r = _kmeans(k_cal_r, N_CENTROIDS, KMEANS_ITERS, SEED)
    dists_r = torch.cdist(k_test_r, cents_r)
    k_hat_r = cents_r[dists_r.argmin(dim=1)] @ R  # rotate back
    lmse_rot = ((q_test * (k_test - k_hat_r)).sum(dim=1) ** 2).mean().item()

    ratio = lmse_rot / lmse_std if lmse_std > 0 else 1.0
    passed_1 = abs(ratio - 1.0) < 0.05  # within 5% tolerance

    print(f"  Standard VQ logit MSE : {lmse_std:.6f}")
    print(f"  Rotated  VQ logit MSE : {lmse_rot:.6f}")
    print(f"  Ratio (should be ~1.0): {ratio:.4f}")
    print(f"  RESULT: {'PASS' if passed_1 else 'FAIL'}")
    print()

    # ------------------------------------------------------------------
    # Claim 2: SMAQ (log c=5.0) reduces logit MSE
    #
    # Metric-shaped VQ should beat standard VQ on query-anisotropic data.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Claim 2: SMAQ reduces logit MSE vs standard VQ")
    print("=" * 60)

    # Create anisotropic query distribution (condition ~20x)
    rng2 = torch.Generator().manual_seed(SEED + 1)
    evals = torch.ones(DIM)
    evals[0] = 20.0
    basis, _ = torch.linalg.qr(torch.randn(DIM, DIM, generator=rng2))
    q_aniso = torch.randn(N, DIM, generator=rng2) * torch.sqrt(evals)
    q_aniso = q_aniso @ basis.T
    k_aniso = torch.randn(N, DIM, generator=rng2)

    q_cal_a, q_test_a = q_aniso[:mid], q_aniso[mid:]
    k_cal_a, k_test_a = k_aniso[:mid], k_aniso[mid:]

    # Standard VQ baseline
    cents_s = _kmeans(k_cal_a, N_CENTROIDS, KMEANS_ITERS, SEED)
    k_hat_s = cents_s[torch.cdist(k_test_a, cents_s).argmin(dim=1)]
    lmse_base = ((q_test_a * (k_test_a - k_hat_s)).sum(dim=1) ** 2).mean().item()

    # SMAQ block VQ
    vq = SMAQBlockVQ(head_dim=DIM, block_dim=DIM, n_centroids=N_CENTROIDS, c=5.0)
    vq.fit(k_cal_a, q_cal_a, kmeans_iters=KMEANS_ITERS, seed=SEED)
    lmse_smaq = vq.logit_mse(q_test_a, k_test_a)

    gain = (1.0 - lmse_smaq / lmse_base) * 100.0 if lmse_base > 0 else 0.0
    passed_2 = gain > 0.0

    print(f"  Standard VQ logit MSE: {lmse_base:.6f}")
    print(f"  SMAQ (c=5) logit MSE : {lmse_smaq:.6f}")
    print(f"  Gain                 : {gain:+.2f}%")
    print(f"  RESULT: {'PASS' if passed_2 else 'FAIL'}")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    all_pass = passed_1 and passed_2
    print(f"Paper claims validated: {'2/2 PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    _run_validations()
