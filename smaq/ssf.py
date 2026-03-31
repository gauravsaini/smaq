"""Spectral shaping utilities for SMAQ.

Includes the original query-aware metric for attention KV cache compression
and extensions for hybrid attention architectures (SSM hidden states, MLA
latent spaces).
"""

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
    """Construct the shaped metric matrix ``E`` and its inverse for attention keys."""
    evals, evecs = torch.linalg.eigh(Sigma_q)
    shaped_evals = ssf_log(evals, c)
    sqrt_diag = torch.diag(shaped_evals.sqrt())
    inv_sqrt_diag = torch.diag(1.0 / shaped_evals.sqrt())
    E = evecs @ sqrt_diag @ evecs.T
    E_inv = evecs @ inv_sqrt_diag @ evecs.T
    return E, E_inv


# ---------------------------------------------------------------------------
# SSM hidden-state metric (Hybrid Attention Extension)
# ---------------------------------------------------------------------------


def _error_persistence_weights(
    A_eigenvalues: torch.Tensor,
    max_amplification: float = 1000.0,
) -> torch.Tensor:
    """Compute per-channel error amplification from SSM state-transition eigenvalues.

    For a linear recurrence ``h_{t+1} = A h_t + B x_{t+1}``, quantization error
    ``ε`` on ``h_t`` propagates as ``|λ|^k ε`` after ``k`` steps.  The
    steady-state squared error amplification is ``1 / (1 - |λ|²)``.  We clamp
    the amplification to ``max_amplification`` to avoid numerical blow-up for
    channels with |λ| ≈ 1 (long-memory channels).
    """
    abs_sq = A_eigenvalues.abs().square()
    # Clamp to prevent division by zero for |λ| ≥ 1
    persistence = 1.0 / (1.0 - abs_sq.clamp(max=1.0 - 1.0 / max_amplification))
    return persistence


def build_ssm_smaq_metric(
    Sigma_C: torch.Tensor,
    A_eigenvalues: torch.Tensor | None = None,
    c: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a SMAQ-style metric for SSM hidden-state quantization.

    This is the SSM analogue of ``build_smaq_metric``.  The calibration
    covariance ``Σ_C = E[C_t^T C_t]`` captures *which* hidden-state directions
    the output projection reads (the SSM counterpart of query covariance).
    Optionally, the eigenvalues of the discretized state-transition matrix
    ``Ā`` are used to up-weight channels whose quantization errors persist
    across many time steps.

    Args:
        Sigma_C: ``(d, d)`` output-projection covariance, analogous to Σ_q.
        A_eigenvalues: ``(d,)`` diagonal of the discretized ``Ā`` matrix.
            If ``None``, persistence weighting is omitted and the metric
            degenerates to the standard SMAQ attention metric applied to Σ_C.
        c: Log-compression constant (same semantics as attention SMAQ).

    Returns:
        ``(E, E_inv)`` shaped metric and its inverse, suitable for quantizing
        SSM hidden states in the spectrally-shaped space.
    """
    evals_C, evecs_C = torch.linalg.eigh(Sigma_C)

    if A_eigenvalues is not None:
        persistence = _error_persistence_weights(A_eigenvalues)
        # Combined sensitivity: output importance × error persistence.
        # We work in the eigenbasis of Σ_C, which is diagonal.  The
        # persistence vector lives in the original basis, so we rotate it
        # into the Σ_C eigenbasis before multiplication.
        #
        # For diagonal A (the Mamba/S4 case), persistence is already
        # per-channel-aligned.  If the eigenbasis of Σ_C differs from the
        # SSM hidden channels, we project:
        persistence_in_eigenbasis = (evecs_C.T @ torch.diag(persistence) @ evecs_C).diag()
        combined_evals = evals_C * persistence_in_eigenbasis.clamp(min=1e-8)
    else:
        combined_evals = evals_C

    shaped = ssf_log(combined_evals, c)
    sqrt_diag = torch.diag(shaped.sqrt())
    inv_sqrt_diag = torch.diag(1.0 / shaped.sqrt())
    E = evecs_C @ sqrt_diag @ evecs_C.T
    E_inv = evecs_C @ inv_sqrt_diag @ evecs_C.T
    return E, E_inv


def build_mla_smaq_metric(
    Sigma_q: torch.Tensor,
    W_UK: torch.Tensor,
    W_UV: torch.Tensor,
    Sigma_attn: torch.Tensor | None = None,
    beta: float = 0.5,
    c: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a SMAQ metric for MLA latent-space quantization.

    In Multi-head Latent Attention, keys and values are reconstructed from a
    shared latent ``c_t``.  Quantization of ``c_t`` simultaneously affects
    logit accuracy (through ``W_UK``) and value reconstruction (through
    ``W_UV``).  This metric optimises the combined objective.

    Args:
        Sigma_q: Query covariance in key-space, ``(d_head, d_head)``.
        W_UK: Uprojection from latent to keys, ``(d_head, d_latent)``.
        W_UV: Uprojection from latent to values, ``(d_head, d_latent)``.
        Sigma_attn: Attention-weight covariance for value error weighting.
            If ``None``, uses identity (uniform weight across tokens).
        beta: Relative weight of value error vs. logit error.
        c: Log-compression constant.

    Returns:
        ``(E, E_inv)`` shaped metric in the latent space.
    """
    d_latent = W_UK.shape[1]

    # Logit-error contribution: W_UK^T Σ_q W_UK
    logit_term = W_UK.T @ Sigma_q @ W_UK

    # Value-error contribution: W_UV^T Σ_attn W_UV
    if Sigma_attn is None:
        Sigma_attn = torch.eye(W_UV.shape[0], device=W_UV.device, dtype=W_UV.dtype)
    value_term = W_UV.T @ Sigma_attn @ W_UV

    # Combined sensitivity in latent space
    combined = logit_term + beta * value_term

    evals, evecs = torch.linalg.eigh(combined)
    shaped = ssf_log(evals, c)
    sqrt_diag = torch.diag(shaped.sqrt())
    inv_sqrt_diag = torch.diag(1.0 / shaped.sqrt())
    E = evecs @ sqrt_diag @ evecs.T
    E_inv = evecs @ inv_sqrt_diag @ evecs.T
    return E, E_inv
