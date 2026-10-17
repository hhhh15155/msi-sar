"""Dual CNN/Transformer classifier with ACFFN-inspired fuzzy feature fusion.

Fusion borrows Gaussian memberships and fuzzy AND/OR aggregation from ACFFN:
https://github.com/Zzp-12/ACFFN (DOI: 10.1109/TGRS.2026.3684169).
This is an independent MSI/SAR adaptation, not a reproduction of ACFFN.
No LiDAR height priors, pretrained weights, or sensor calibration are assumed.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GaussianMembership(nn.Module):
    """Channel-wise feature memberships; these are not calibrated probabilities."""

    def __init__(self, dim: int, count: int = 7, min_width: float = 0.15):
        super().__init__()
        if dim < 1 or count < 2 or not 0 < min_width < 1:
            raise ValueError("Require dim > 0, count >= 2 and 0 < min_width < 1")
        self.norm = nn.LayerNorm(dim)
        self.centers = nn.Parameter(torch.linspace(-2, 2, count)[:, None].repeat(1, dim))
        self.raw_widths = nn.Parameter(torch.full((count, dim), math.log(math.expm1(1 - min_width))))
        self.min_width = min_width

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, N, M, D]. FP32 and log aggregation prevent
        # underflow from multiplying memberships, including under autocast.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            features = F.layer_norm(
                tokens.float(), self.norm.normalized_shape,
                self.norm.weight.float(), self.norm.bias.float(), self.norm.eps,
            )
            widths = F.softplus(self.raw_widths.float()) + self.min_width
            distances = (features.unsqueeze(-2) - self.centers.float()) / widths
            log_membership = -distances.square()
            fuzzy_and = log_membership.mean(dim=-2).clamp(min=-60).exp()
            fuzzy_or = log_membership.amax(dim=-2).clamp(min=-60).exp()
            return torch.cat((fuzzy_and, fuzzy_or), dim=-1).to(tokens.dtype)


class FuzzyFeatureFusion(nn.Module):
    """Normalize membership-derived weights across MSI/SAR at each token/channel.

    Unweighted modality features remain available in the output projection, so
    uncertain memberships cannot remove the original observations entirely.
    mode='concat' removes the fuzzy path for a controlled fusion ablation.
    """

    def __init__(self, dim: int, count: int, min_width: float, temperature: float,
                 dropout: float, mode: str = "fuzzy"):
        super().__init__()
        if mode not in {"fuzzy", "concat"} or temperature <= 0:
            raise ValueError("fusion must be fuzzy/concat and temperature must be positive")
        self.mode = mode
        self.temperature = temperature
        if mode == "fuzzy":
            self.memberships = nn.ModuleList([GaussianMembership(dim, count, min_width) for _ in range(2)])
            self.scorers = nn.ModuleList([
                nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
                for _ in range(2)
            ])
        self.project = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, dim),
                                     nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, ms: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if ms.shape != sar.shape:
            raise ValueError("Fusion requires spatially aligned features with equal shapes")
        weights = None
        if self.mode == "fuzzy":
            scores = torch.stack([
                scorer(membership(tokens))
                for scorer, membership, tokens in zip(self.scorers, self.memberships, (ms, sar))
            ], dim=-2)  # [B, N, 2, D], modality axis is -2
            weights = (scores.float() / self.temperature).softmax(dim=-2).to(ms.dtype)
            mixed = weights[..., 0, :] * ms + weights[..., 1, :] * sar
        else:
            mixed = (ms + sar) * 0.5
        fused = self.norm(mixed + self.project(torch.cat((mixed, ms, sar), dim=-1)))
        return fused, weights


class ConvStem(nn.Module):
    def __init__(self, channels: int, dim: int):
        super().__init__()
        # MSI mixes bands first; independent spatial convolutions learn SAR
        # backscatter texture without treating it as a height measurement.
        self.channel_mix = nn.Conv2d(channels, dim, kernel_size=1)
        self.spatial = nn.Sequential(
            nn.GroupNorm(1, dim), nn.GELU(), nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(1, dim), nn.GELU(), nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.channel_mix(image)
        features = features + self.spatial(features)
        return self.norm(features.flatten(2).transpose(1, 2))


def transformer(dim: int, heads: int, depth: int, mlp_ratio: float, dropout: float) -> nn.Module:
    if depth < 1:
        raise ValueError("Transformer depth must be positive")
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    encoder = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(dim),
                                    enable_nested_tensor=False)
    # TransformerEncoder clones one layer; initialize each clone independently.
    for block in encoder.layers:
        nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
        nn.init.zeros_(block.self_attn.in_proj_bias)
        for projection in (block.self_attn.out_proj, block.linear1, block.linear2):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
    return encoder


class FuzzyTransformer(nn.Module):
    """Aligned patch -> two modality streams -> two fuzzy stages -> center class.

    All attention operates on complete token grids. The center is used only in
    the final classifier readout; it never queries the other modality.
    """

    def __init__(self, ms_channels: int, sar_channels: int, num_classes: int,
                 embed_dim: int = 192, num_heads: int = 6, depth_per_stage: int = 2,
                 fusion_depth: int = 2, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 num_memberships: int = 7, membership_min_width: float = 0.15,
                 fusion_temperature: float = 1.0, fusion: str = "fuzzy",
                 auxiliary_weight: float = 0.2, label_smoothing: float = 0.0):
        super().__init__()
        if min(ms_channels, sar_channels, num_classes, embed_dim, num_heads) < 1:
            raise ValueError("Channel counts, class count, embedding dimension and heads must be positive")
        if embed_dim % num_heads or mlp_ratio <= 0 or not 0 <= dropout < 1:
            raise ValueError("Require embed_dim divisible by heads, positive mlp_ratio and dropout in [0, 1)")
        if auxiliary_weight < 0 or not 0 <= label_smoothing < 1:
            raise ValueError("Invalid auxiliary weight or label smoothing")
        self.ms_channels, self.sar_channels = ms_channels, sar_channels
        self.auxiliary_weight, self.label_smoothing = auxiliary_weight, label_smoothing
        self.ms_stem = ConvStem(ms_channels, embed_dim)
        self.sar_stem = ConvStem(sar_channels, embed_dim)
        self.position = nn.Sequential(nn.Linear(2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.stages = nn.ModuleList([
            nn.ModuleDict({
                "ms": transformer(embed_dim, num_heads, depth_per_stage, mlp_ratio, dropout),
                "sar": transformer(embed_dim, num_heads, depth_per_stage, mlp_ratio, dropout),
                "fusion": FuzzyFeatureFusion(embed_dim, num_memberships, membership_min_width,
                                             fusion_temperature, dropout, fusion),
            }) for _ in range(2)
        ])
        self.ms_exchange = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.sar_exchange = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.fusion_transformer = transformer(embed_dim, num_heads, fusion_depth, mlp_ratio, dropout)
        self.head = nn.Sequential(nn.LayerNorm(2 * embed_dim), nn.Linear(2 * embed_dim, embed_dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        self.ms_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.sar_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def _position(self, height: int, width: int, image: torch.Tensor) -> torch.Tensor:
        # Relative coordinates support variable odd patch sizes, including 1x1.
        y = (torch.arange(height, device=image.device, dtype=image.dtype) - height // 2) / max(height // 2, 1)
        x = (torch.arange(width, device=image.device, dtype=image.dtype) - width // 2) / max(width // 2, 1)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return self.position(torch.stack((yy, xx), dim=-1).reshape(1, height * width, 2))

    def forward(self, ms: torch.Tensor, sar: torch.Tensor, labels: torch.Tensor | None = None,
                return_fusion_weights: bool = False) -> dict:
        if ms.ndim != 4 or sar.ndim != 4:
            raise ValueError("Expected MSI and SAR tensors [B, C, H, W]")
        if ms.shape[0] != sar.shape[0] or ms.shape[-2:] != sar.shape[-2:]:
            raise ValueError("MSI and SAR patches must have matching batch/spatial shapes")
        if ms.shape[1] != self.ms_channels or sar.shape[1] != self.sar_channels:
            raise ValueError("Input channels do not match model configuration")
        height, width = ms.shape[-2:]
        if height < 1 or width < 1 or height % 2 == 0 or width % 2 == 0:
            raise ValueError("Center classification requires positive odd patch dimensions")
        position = self._position(height, width, ms)
        ms_tokens, sar_tokens = self.ms_stem(ms) + position, self.sar_stem(sar) + position
        diagnostics = []
        for index, stage in enumerate(self.stages):
            ms_tokens, sar_tokens = stage["ms"](ms_tokens), stage["sar"](sar_tokens)
            fused, weights = stage["fusion"](ms_tokens, sar_tokens)
            if return_fusion_weights:
                diagnostics.append(weights)
            if index == 0:
                ms_tokens = ms_tokens + self.ms_exchange(fused)
                sar_tokens = sar_tokens + self.sar_exchange(fused)
        fused = self.fusion_transformer(fused)
        center = (height // 2) * width + width // 2
        logits = self.head(torch.cat((fused[:, center], fused.mean(dim=1)), dim=-1))
        output = {"logits": logits}
        if labels is not None:
            ce = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            if self.auxiliary_weight:
                auxiliary = 0.5 * (
                    F.cross_entropy(self.ms_head(ms_tokens[:, center]), labels, label_smoothing=self.label_smoothing)
                    + F.cross_entropy(self.sar_head(sar_tokens[:, center]), labels, label_smoothing=self.label_smoothing)
                )
            else:
                auxiliary = ce.new_zeros(())
            output["losses"] = {"ce": ce, "auxiliary": auxiliary, "total": ce + self.auxiliary_weight * auxiliary}
        if return_fusion_weights:
            output["fusion_weights"] = diagnostics
        return output
