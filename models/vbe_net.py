"""VBE-Net: class-conditional variational Bures energy network."""

from __future__ import annotations

import math
from typing import NamedTuple, Union

import torch
from torch import Tensor, nn

from .vbe_geometry import (
    estimate_grouped_gaussian,
    spd_from_raw_tril,
    variational_bures_energy,
)


class VBEModelOutput(NamedTuple):
    """Detailed VBE-Net output for analysis and diagnostics."""

    logits: Tensor
    energy: Tensor
    responsibility: Tensor
    ms_shrinkage: Tensor
    sar_shrinkage: Tensor


class LayerNorm2d(nn.Module):
    """LayerNorm over channels while preserving an image tensor layout."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        channels_last = inputs.permute(0, 2, 3, 1)
        return self.norm(channels_last).permute(0, 3, 1, 2).contiguous()


class ModalityStem(nn.Module):
    """Modality-private spectral/spatial projection without downsampling."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm = LayerNorm2d(width)
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.norm(self.projection(inputs)))


class SemiSharedResidualBlock(nn.Module):
    """Private spatial operators with a shared channel-mixing transform."""

    def __init__(self, width: int, expansion: int = 4) -> None:
        super().__init__()
        hidden_width = width * expansion

        self.ms_spatial_norm = LayerNorm2d(width)
        self.sar_spatial_norm = LayerNorm2d(width)
        self.ms_depthwise = nn.Conv2d(
            width, width, kernel_size=3, padding=1, groups=width
        )
        self.sar_depthwise = nn.Conv2d(
            width, width, kernel_size=3, padding=1, groups=width
        )

        self.ms_mixer_norm = LayerNorm2d(width)
        self.sar_mixer_norm = LayerNorm2d(width)
        self.shared_pw1 = nn.Conv2d(width, hidden_width, kernel_size=1)
        self.shared_pw2 = nn.Conv2d(hidden_width, width, kernel_size=1)
        self.activation = nn.GELU()

    def _mix(self, inputs: Tensor, norm: nn.Module) -> Tensor:
        return self.shared_pw2(self.activation(self.shared_pw1(norm(inputs))))

    def forward(self, ms: Tensor, sar: Tensor) -> tuple[Tensor, Tensor]:
        ms_spatial = ms + self.ms_depthwise(self.ms_spatial_norm(ms))
        sar_spatial = sar + self.sar_depthwise(self.sar_spatial_norm(sar))
        ms_output = ms_spatial + self._mix(ms_spatial, self.ms_mixer_norm)
        sar_output = sar_spatial + self._mix(sar_spatial, self.sar_mixer_norm)
        return ms_output, sar_output


class SemiSharedEncoder(nn.Module):
    """Two modality lanes sharing channel mixers but not spatial filters."""

    def __init__(
        self,
        ms_channels: int,
        sar_channels: int,
        width: int = 64,
        depth: int = 5,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.ms_stem = ModalityStem(ms_channels, width)
        self.sar_stem = ModalityStem(sar_channels, width)
        self.blocks = nn.ModuleList(
            SemiSharedResidualBlock(width, expansion) for _ in range(depth)
        )

    def forward(self, ms: Tensor, sar: Tensor) -> tuple[Tensor, Tensor]:
        ms_features = self.ms_stem(ms)
        sar_features = self.sar_stem(sar)
        for block in self.blocks:
            ms_features, sar_features = block(ms_features, sar_features)
        return ms_features, sar_features


class VBENet(nn.Module):
    """End-to-end MSI-SAR classifier driven by variational Bures energy."""

    def __init__(
        self,
        ms_channels: int = 10,
        sar_channels: int = 4,
        num_classes: int = 8,
        patch_size: int = 11,
        width: int = 64,
        depth: int = 5,
        groups: int = 8,
        expansion: int = 4,
        lambda_proto: float = 1.0,
        tau_r: float = 0.3,
        tau_c: float = 0.1,
        inner_iters: int = 3,
        outer_updates: int = 1,
        modality_dropout: float = 0.1,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if width <= 0 or width % groups:
            raise ValueError("width must be positive and divisible by groups")
        if depth <= 0 or expansion <= 0:
            raise ValueError("depth and expansion must be positive")
        if num_classes <= 1 or patch_size <= 0:
            raise ValueError("num_classes must exceed one and patch_size must be positive")
        if tau_r <= 0 or tau_c <= 0 or lambda_proto <= 0:
            raise ValueError("lambda_proto, tau_r, and tau_c must be positive")
        if inner_iters < 0 or outer_updates < 0:
            raise ValueError("solver iteration counts must be non-negative")
        if not 0.0 <= modality_dropout <= 1.0:
            raise ValueError("modality_dropout must be in [0, 1]")

        self.ms_channels = ms_channels
        self.sar_channels = sar_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.width = width
        self.groups = groups
        self.group_dim = width // groups
        self.lambda_proto = lambda_proto
        self.tau_r = tau_r
        self.tau_c = tau_c
        self.inner_iters = inner_iters
        self.outer_updates = outer_updates
        self.modality_dropout = modality_dropout
        self.eps = eps

        self.encoder = SemiSharedEncoder(
            ms_channels=ms_channels,
            sar_channels=sar_channels,
            width=width,
            depth=depth,
            expansion=expansion,
        )
        self.prototype_mean = nn.Parameter(
            torch.empty(num_classes, groups, self.group_dim)
        )
        raw_tril = torch.zeros(
            num_classes,
            groups,
            self.group_dim,
            self.group_dim,
        )
        unit_diagonal_raw = math.log(math.expm1(1.0 - 1e-3))
        raw_tril.diagonal(dim1=-2, dim2=-1).fill_(unit_diagonal_raw)
        self.prototype_raw_tril = nn.Parameter(raw_tril)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.prototype_mean, mean=0.0, std=0.02)

    def _validate_inputs(self, ms: Tensor, sar: Tensor) -> None:
        if ms.ndim != 4 or sar.ndim != 4:
            raise ValueError("MS and SAR inputs must have shape [B,C,H,W]")
        if ms.shape[0] != sar.shape[0]:
            raise ValueError("MS and SAR batch sizes must match")
        if ms.shape[1] != self.ms_channels:
            raise ValueError(
                f"Expected {self.ms_channels} MS channels, got {ms.shape[1]}"
            )
        if sar.shape[1] != self.sar_channels:
            raise ValueError(
                f"Expected {self.sar_channels} SAR channels, got {sar.shape[1]}"
            )
        expected_spatial = (self.patch_size, self.patch_size)
        if ms.shape[-2:] != expected_spatial or sar.shape[-2:] != expected_spatial:
            raise ValueError(
                f"Expected spatial size {expected_spatial}, got "
                f"MS={tuple(ms.shape[-2:])}, SAR={tuple(sar.shape[-2:])}"
            )

    def encode_tokens(self, ms: Tensor, sar: Tensor) -> tuple[Tensor, Tensor]:
        """Encode both modalities as full-resolution spatial token sequences."""

        self._validate_inputs(ms, sar)
        ms_features, sar_features = self.encoder(ms, sar)
        ms_tokens = ms_features.flatten(2).transpose(1, 2)
        sar_tokens = sar_features.flatten(2).transpose(1, 2)
        return ms_tokens, sar_tokens

    def _select_modalities(
        self,
        modality_mean: Tensor,
        modality_covariance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            not self.training
            or self.modality_dropout == 0.0
            or torch.rand((), device=modality_mean.device) >= self.modality_dropout
        ):
            return modality_mean, modality_covariance
        keep_index = int(torch.randint(2, (), device=modality_mean.device).item())
        return (
            modality_mean[:, keep_index : keep_index + 1],
            modality_covariance[:, keep_index : keep_index + 1],
        )

    def forward(
        self,
        ms: Tensor,
        sar: Tensor,
        return_details: bool = False,
    ) -> Union[Tensor, VBEModelOutput]:
        ms_tokens, sar_tokens = self.encode_tokens(ms, sar)

        # EVD-based geometry is intentionally excluded from mixed precision.
        with torch.autocast(device_type=ms.device.type, enabled=False):
            ms_gaussian = estimate_grouped_gaussian(
                ms_tokens.float(), groups=self.groups, eps=self.eps
            )
            sar_gaussian = estimate_grouped_gaussian(
                sar_tokens.float(), groups=self.groups, eps=self.eps
            )
            modality_mean = torch.stack(
                (ms_gaussian.mean, sar_gaussian.mean), dim=1
            )
            modality_covariance = torch.stack(
                (ms_gaussian.covariance, sar_gaussian.covariance), dim=1
            )
            modality_mean, modality_covariance = self._select_modalities(
                modality_mean, modality_covariance
            )
            prototype_covariance = spd_from_raw_tril(
                self.prototype_raw_tril.float(), eps=self.eps
            )
            result = variational_bures_energy(
                self.prototype_mean.float(),
                prototype_covariance,
                modality_mean,
                modality_covariance,
                lambda_proto=self.lambda_proto,
                tau_r=self.tau_r,
                inner_iters=self.inner_iters,
                outer_updates=self.outer_updates,
                eps=self.eps,
            )
            logits = -result.energy / self.tau_c

        if not return_details:
            return logits
        return VBEModelOutput(
            logits=logits,
            energy=result.energy,
            responsibility=result.responsibility,
            ms_shrinkage=ms_gaussian.shrinkage,
            sar_shrinkage=sar_gaussian.shrinkage,
        )


__all__ = [
    "LayerNorm2d",
    "ModalityStem",
    "SemiSharedResidualBlock",
    "SemiSharedEncoder",
    "VBEModelOutput",
    "VBENet",
]
