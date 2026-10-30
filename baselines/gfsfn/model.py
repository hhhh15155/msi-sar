from __future__ import annotations

import torch
from torch import nn

from baselines._hsi_adapters.gfsfn_patch import GFSFNPatch


class GFSFN(nn.Module):
    """GFSFN: [B,1,MSI+SAR,P,P] -> center-pixel class logits."""

    def __init__(self, num_classes: int = 8, patch_size: int = 11,
                 spectral_channels: int = 10, sar_channels: int = 4):
        super().__init__()
        self.patch_size = int(patch_size)
        self.spectral_channels = int(spectral_channels)
        self.sar_channels = int(sar_channels)
        self.num_channels = self.spectral_channels + self.sar_channels
        # Stem reduces P by 2, then the CMSF path reduces it by another 4.
        if self.patch_size < 7 or self.patch_size % 2 != 1:
            raise ValueError("GFSFN requires an odd patch_size >= 7")
        if min(self.spectral_channels, self.sar_channels, int(num_classes)) < 1:
            raise ValueError("Modality channel counts and num_classes must be positive")
        self.net = GFSFNPatch(self.spectral_channels, self.sar_channels, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (1, self.num_channels, self.patch_size, self.patch_size)
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{','.join(map(str, expected))}], got {tuple(x.shape)}")
        x = x[:, 0]
        return self.net(x[:, :self.spectral_channels], x[:, self.spectral_channels:])
