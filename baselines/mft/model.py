from __future__ import annotations

import torch
from torch import nn

from baselines.hsi_lidar.mft_patch import MFTPatch


class MFT(nn.Module):
    """Project-interface wrapper around the adapted MFT model.

    The released MFT code exists only as a Jupyter notebook; the model classes
    are re-implemented in ``baselines/hsi_lidar/mft_patch.py`` with a
    band-adaptive spectral stem so 6/10-band MSI inputs are valid.
    Input layout: ``[B, 1, C, H, W]`` with MSI then SAR channels.
    """

    def __init__(self, num_classes: int = 8, patch_size: int = 11,
                 spectral_channels: int = 10, sar_channels: int = 4):
        super().__init__()
        self.patch_size = int(patch_size)
        self.spectral_channels = int(spectral_channels)
        self.sar_channels = int(sar_channels)
        self.num_channels = self.spectral_channels + self.sar_channels
        self.net = MFTPatch(ms_channels=self.spectral_channels,
                            sar_channels=self.sar_channels,
                            num_classes=int(num_classes),
                            patch_size=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(dim=1)
        if x.ndim != 4 or x.shape[1] != self.num_channels:
            raise ValueError(f"Expected [B,{self.num_channels},H,W], got {tuple(x.shape)}")
        msi = x[:, : self.spectral_channels]
        sar = x[:, self.spectral_channels:]
        return self.net(msi, sar)['logits']
