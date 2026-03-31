"""SMAQ SSM hidden-state quantizer.

Applies spectral metric-aware quantization to SSM (Mamba/S4) hidden states.
This is the recurrent-state analogue of the attention KV quantizer: instead
of compressing key vectors using the query covariance Σ_q, we compress hidden
states using the output-projection covariance Σ_C weighted by error persistence
from the state-transition eigenvalues.

The key difference from attention KV quantization is that SSM states are
*fixed-size* (not growing), so the motivation is not memory growth but rather
per-step memory footprint and bandwidth.  The quantized state participates in
the recurrence, so error *propagation* through the recurrence is the dominant
quality concern.

Typical usage::

    quantizer = SMAQSSMQuantizer(state_dim=16)
    quantizer.calibrate(calibration_C, calibration_A_eigs)
    h_q = quantizer.quantize_state(h_t)
    h_hat = quantizer.dequantize_state(h_q)
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from smaq.ssf import build_ssm_smaq_metric, ssf_log


class SSMStateQuantized(NamedTuple):
    """Quantized SSM hidden state representation."""

    data: torch.Tensor       # quantized state values (int8 or similar)
    scales: torch.Tensor     # per-channel or per-group scale factors
    zeros: torch.Tensor      # per-channel or per-group zero points
    bits: int = 8


class SMAQSSMQuantizer(nn.Module):
    """Spectral Metric-Aware Quantizer for SSM hidden states.

    Like ``SMAQBlockVQ`` but designed for fixed-size recurrent states rather
    than growing key sequences.  The metric is derived from the output
    projection covariance Σ_C and the state-transition eigenvalues, combining
    "what matters for output quality" with "what persists across time steps".

    The quantizer works channel-wise (or group-wise) with asymmetric uniform
    quantization in the spectrally-shaped space.  This is a simpler scheme than
    block VQ because SSM states are typically smaller and updated every step.
    """

    def __init__(
        self,
        state_dim: int,
        bits: int = 8,
        group_size: int = 0,  # 0 = per-channel
        c: float = 5.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.bits = bits
        self.group_size = group_size if group_size > 0 else state_dim
        self.c = c
        self._device = device or torch.device("cpu")
        self._dtype = dtype

        # Metric matrices — initialised to identity until calibrated
        eye = torch.eye(state_dim, device=self._device, dtype=dtype)
        self.register_buffer("E", eye.clone())
        self.register_buffer("E_inv", eye.clone())
        self._calibrated = False

    def calibrate(
        self,
        Sigma_C: torch.Tensor,
        A_eigenvalues: Optional[torch.Tensor] = None,
    ) -> "SMAQSSMQuantizer":
        """Calibrate the metric from output projection covariance and SSM dynamics.

        Args:
            Sigma_C: ``(state_dim, state_dim)`` output projection covariance.
            A_eigenvalues: ``(state_dim,)`` diagonal of the discretized state
                transition matrix.  If ``None``, persistence is not used.
        """
        E, E_inv = build_ssm_smaq_metric(
            Sigma_C.to(torch.float32).to(self._device),
            A_eigenvalues=A_eigenvalues.to(torch.float32).to(self._device) if A_eigenvalues is not None else None,
            c=self.c,
        )
        self.E = E.to(self._device, self._dtype)
        self.E_inv = E_inv.to(self._device, self._dtype)
        self._calibrated = True
        return self

    def quantize_state(self, h: torch.Tensor) -> SSMStateQuantized:
        """Quantize SSM hidden state in the spectrally-shaped space.

        Args:
            h: ``(..., state_dim)`` hidden state tensor.

        Returns:
            ``SSMStateQuantized`` with data, scales, zeros, and bits.
        """
        # Transform into shaped space
        h_shaped = torch.matmul(h.float(), self.E.T)

        # Group-wise asymmetric quantization
        orig_shape = h_shaped.shape
        d = orig_shape[-1]
        n_groups = d // self.group_size

        h_grouped = h_shaped.reshape(*orig_shape[:-1], n_groups, self.group_size)
        h_min = h_grouped.min(dim=-1, keepdim=True).values
        h_max = h_grouped.max(dim=-1, keepdim=True).values

        n_levels = (2 ** self.bits) - 1
        scale = ((h_max - h_min) / n_levels).clamp(min=1e-10)
        zero = h_min

        h_q = ((h_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.int8)
        h_q_flat = h_q.reshape(*orig_shape)

        return SSMStateQuantized(
            data=h_q_flat,
            scales=scale.squeeze(-1),
            zeros=zero.squeeze(-1),
            bits=self.bits,
        )

    def dequantize_state(self, sq: SSMStateQuantized) -> torch.Tensor:
        """Reconstruct hidden state from quantized representation.

        Args:
            sq: Quantized state from ``quantize_state``.

        Returns:
            Reconstructed hidden state, ``(..., state_dim)``.
        """
        d = sq.data.shape[-1]
        n_groups = d // self.group_size

        data = sq.data.float().reshape(*sq.data.shape[:-1], n_groups, self.group_size)
        scales = sq.scales.unsqueeze(-1)
        zeros = sq.zeros.unsqueeze(-1)

        h_shaped = (data * scales + zeros).reshape(*sq.data.shape)
        # Transform back from shaped space
        return torch.matmul(h_shaped, self.E_inv.T)

    def output_mse(
        self,
        h: torch.Tensor,
        C: torch.Tensor,
    ) -> float:
        """Compute held-out output MSE: ``E[(C(h - h_hat))^2]``.

        This is the SSM analogue of SMAQ's logit MSE for attention keys.

        Args:
            h: ``(N, state_dim)`` ground-truth hidden states.
            C: ``(N, state_dim)`` output projection vectors.

        Returns:
            Mean squared output error.
        """
        h_hat = self.dequantize_state(self.quantize_state(h))
        delta = h.float() - h_hat.float()
        output_errors = (C.float() * delta).sum(dim=-1)
        return output_errors.square().mean().item()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize (for smoke tests)."""
        return self.dequantize_state(self.quantize_state(h))
