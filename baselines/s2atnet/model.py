from __future__ import annotations

import torch
from torch import nn

from baselines._hsi_adapters.s2atnet_patch import S2ATNetPatch


class S2ATNet(nn.Module):
    """S²ATNet: paired neighborhood patches -> center-pixel class logits."""

    def __init__(self, num_classes: int = 8, patch_size: int = 11,
                 spectral_channels: int = 10, sar_channels: int = 4):
        super().__init__()
        self.patch_size = int(patch_size)
        self.spectral_channels = int(spectral_channels)
        self.sar_channels = int(sar_channels)
        self.num_channels = self.spectral_channels + self.sar_channels
        if self.patch_size < 1 or self.patch_size % 2 != 1:
            raise ValueError("S2ATNet requires a positive odd patch_size")
        if min(self.spectral_channels, self.sar_channels, int(num_classes)) < 1:
            raise ValueError("Modality channel counts and num_classes must be positive")
        self.net = S2ATNetPatch(self.spectral_channels, self.sar_channels,
                               int(num_classes), self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (1, self.num_channels, self.patch_size, self.patch_size)
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{','.join(map(str, expected))}], got {tuple(x.shape)}")
        # Upstream HSI path expects [B,1,C,P,P]; auxiliary path is [B,C,P,P].
        return self.net(x[:, :, :self.spectral_channels], x[:, 0, self.spectral_channels:])
