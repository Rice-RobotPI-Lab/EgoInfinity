"""
Vector Neuron (VN) building blocks for SO(3)/SE(3)-equivariant networks.

Features are tensors of shape (..., C, 3):
  C : number of channels (analogous to feature width)
  3 : spatial dimension

Under SO(3) rotation R:
    x[..., c, :] → R @ x[..., c, :]  for every channel c

References
----------
Deng et al., "Vector Neurons: A General Framework for SO(3)-Equivariant
Networks," ICCV 2021.  https://arxiv.org/abs/2104.12229
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VNLinear(nn.Module):
    """
    Equivariant linear layer: mixes channels, preserves 3D vector structure.

    y[..., o, :] = Σ_c  W[o, c] * x[..., c, :]

    Input : (..., C_in,  3)
    Output: (..., C_out, 3)

    No bias: a constant bias vector would require a fixed direction in 3D space,
    which breaks rotation equivariance.  Use VNLayerNorm instead.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.map = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose so nn.Linear acts on the channel dim, not the spatial dim.
        # (..., C_in, 3) → (..., 3, C_in) → linear → (..., 3, C_out) → (..., C_out, 3)
        return self.map(x.transpose(-1, -2)).transpose(-1, -2)


class VNLayerNorm(nn.Module):
    """
    Equivariant layer normalization.

    Normalises by the RMS channel-norm, then rescales per-channel.

        rms  = sqrt( mean_c ||x_c||² )
        out  = (x / rms) * scale

    This is invariant to rotation (norms are rotation-invariant) and learns
    a per-channel gain.

    Input/Output: (..., C, 3)
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sq_norms = (x * x).sum(dim=-1)                        # (..., C)
        rms = sq_norms.mean(dim=-1, keepdim=True)             # (..., 1)
        rms = rms.clamp(min=self.eps).sqrt()                  # (..., 1)
        x = x / rms.unsqueeze(-1)                             # (..., C, 3)
        return x * self.scale.view(*([1] * (x.dim() - 2)), -1, 1)


class VNLeakyReLU(nn.Module):
    """
    Equivariant nonlinearity: scales each channel vector by LeakyReLU of its norm.

        output_c = leaky_relu(||x_c||) / ||x_c||  *  x_c

    Equivariance: ||Rx_c|| = ||x_c||, so the scalar multiplier is invariant.

    Input/Output: (..., C, 3)
    """

    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # (..., C, 1)
        scale = F.leaky_relu(norms, self.negative_slope) / norms
        return x * scale


class VNMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention on VN equivariant features.

    Q, K, V are VN features.  Attention weights are derived from *invariant*
    inner products of VN channels (dot products are SO(3)-invariant):

        score(q, k) = Σ_c <q_c, k_c>  /  sqrt(C_head * 3)

    The weighted sum of V inherits equivariance from V.

    An optional invariant scalar bias (e.g. positional) can be added to scores.

    Input : x      (B, T, C, 3)
            bias   (1, H, T, T)  or None  — invariant attention bias
    Output:        (B, T, C, 3)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self._scale = (self.d_head * 3) ** -0.5

        self.q_proj   = VNLinear(d_model, d_model)
        self.k_proj   = VNLinear(d_model, d_model)
        self.v_proj   = VNLinear(d_model, d_model)
        self.out_proj = VNLinear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,  # (B, T) bool, True=valid
    ) -> torch.Tensor:
        B, T, C, _ = x.shape
        H, d_h = self.num_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, d_h, 3)   # (B, T, H, d_h, 3)
        K = self.k_proj(x).view(B, T, H, d_h, 3)
        V = self.v_proj(x).view(B, T, H, d_h, 3)

        # Invariant scores: sum of channel-wise dot products → (B, H, T, T)
        scores = torch.einsum("bthci, bshci -> bhts", Q, K) * self._scale
        if attn_bias is not None:
            scores = scores + attn_bias              # (B, H, T, T)
        if key_padding_mask is not None:
            # mask out fully-occluded key positions: (B,T) → (B,1,1,T)
            scores = scores.masked_fill(
                ~key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhts, bshci -> bthci", attn, V)   # (B, T, H, d_h, 3)
        out = out.reshape(B, T, C, 3)
        return self.out_proj(out)


class VNTransformerLayer(nn.Module):
    """
    Single VN-Transformer encoder layer (Pre-LN):

        x ← x + Dropout( VNAttention( VNLayerNorm(x) ) )
        x ← x + Dropout( VN-FFN(     VNLayerNorm(x) ) )

    Input/Output: (B, T, C, 3)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = VNLayerNorm(d_model)
        self.norm2 = VNLayerNorm(d_model)
        self.attn  = VNMultiHeadAttention(d_model, num_heads, dropout)
        self.ff    = nn.Sequential(
            VNLinear(d_model, dim_feedforward),
            VNLeakyReLU(),
            VNLinear(dim_feedforward, d_model),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,  # (B, T) bool, True=valid
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), attn_bias, key_padding_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
