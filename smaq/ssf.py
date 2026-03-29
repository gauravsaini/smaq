"""Spectral shaping utilities for SMAQ."""

from __future__ import annotations

import torch


def ssf_log(eigvals: torch.Tensor, c: float = 5.0) -> torch.Tensor:
    """
    Apply the log-compressed spectral shaping function from the SMAQ paper.

    The output is volume-normalized so the metric changes shape without
    introducing a global scale term.
    """
    shaped = torch.log1p(c * eigvals.clamp(min=0))
    log_shaped = torch.log(shaped.clamp(min=1e-8))
    log_shaped = log_shaped - log_shaped.mean()
    return torch.exp(log_shaped)


def build_smaq_metric(Sigma_q: torch.Tensor, c: float = 5.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the shaped metric matrix `E` and its inverse."""
    evals, evecs = torch.linalg.eigh(Sigma_q)
    shaped_evals = ssf_log(evals, c)
    sqrt_diag = torch.diag(shaped_evals.sqrt())
    inv_sqrt_diag = torch.diag(1.0 / shaped_evals.sqrt())
    E = evecs @ sqrt_diag @ evecs.T
    E_inv = evecs @ inv_sqrt_diag @ evecs.T
    return E, E_inv
