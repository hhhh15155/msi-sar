"""DFFNet (GRSL 2025) patch-classification adaptation for the MSI-SAR project.

Vendored source: ``vendor/hsi_lidar/DFFNet-original`` (oucailab/DFFNet,
commit a9beb2b2b43b452cfca364160822f0dee9b34462).  Only the dynamic filter
block and the spectral-spatial adaptive fusion block are imported unchanged;
the outer network is rebuilt for patch classification with explicit channel
counts, because the original ``Net`` reads a dataset YAML and a file-based
loader.

Deviations from the original repository (recorded for fairness):
- inputs: MSI ``[B,C,H,W]`` + SAR ``[B,S,H,W]``; HSI branch keeps the original
  3D-conv spectral reduction but derives its spectral kernel from ``C``;
- no PCA stage inside the model (the project's prepared data is already
  per-channel normalised); ``C`` plays the role of ``pca_num``;
- two DFFM stages (paper reports two as best; the released YAML default is 3);
- ``filter_bases_F`` is initialised in ``__init__`` before the optimiser is
  created (the original creates it lazily on the first forward, so the
  parameters would be missing from the optimiser);
- patch-level classifier (flatten + two linear layers), no pretraining.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_VENDOR = Path(__file__).resolve().parents[2] / 'vendor' / 'hsi_lidar' / 'DFFNet-original'


def _load_module(alias: str, relative: str):
    spec = importlib.util.spec_from_file_location(alias, _VENDOR / relative)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load vendored module {relative}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_dynamic = _load_module('_dffnet_dynamic_block', 'modules/DynamicBlock.py')
_fusion = _load_module('_dffnet_fusion', 'modules/fusion.py')
DynamicFilterBlock = _dynamic.DynamicFilterBlock
SpectralSpatialAdaptiveFusionBlock = _fusion.SpectralSpatialAdaptiveFusionBlock


class MultiStageDFFMBlock(nn.Module):
    """One DFFM stage: per-stream dynamic filter + cross-modal SSAFB residual."""

    def __init__(self, channels, ssafb_out_channels, num_filter_bases=4,
                 mlp_ratio=2, shuffle_groups=2, reduction_ratio_ca=16, kernel_size_sa=5):
        super().__init__()
        dfb_params = {'num_filter_bases': num_filter_bases, 'mlp_ratio': mlp_ratio, 'bias': False}
        self.dfb_msi = DynamicFilterBlock(dim=channels, **dfb_params)
        self.dfb_sar = DynamicFilterBlock(dim=channels, **dfb_params)
        self.ssafb = SpectralSpatialAdaptiveFusionBlock(
            hsi_channels=channels, sar_channels=channels, out_channels=ssafb_out_channels,
            shuffle_groups=shuffle_groups, reduction_ratio_ca=reduction_ratio_ca,
            kernel_size_sa=kernel_size_sa)
        self.adjust = nn.Conv2d(ssafb_out_channels, channels, kernel_size=1)

    def forward(self, msi_features, sar_features):
        msi_out = self.dfb_msi(msi_features)
        sar_out = self.dfb_sar(sar_features)
        fused = self.adjust(self.ssafb(msi_features, sar_features))
        return msi_out + fused, sar_out + fused


class DFFNetPatch(nn.Module):
    """Patch classifier with the DFFNet frequency fusion mechanism."""

    def __init__(self, ms_channels, sar_channels, num_classes, patch_size=11,
                 dim=64, num_stages=2):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError('dim must be divisible by 4 to mirror the original 3D stem')
        self.patch_size = int(patch_size)
        target_spectral = 4 if ms_channels >= 4 else 1
        spectral_kernel = ms_channels - target_spectral + 1
        if spectral_kernel <= 0:
            spectral_kernel = 1
            target_spectral = ms_channels
        maps_3d = dim // 4
        self.msi_stem = nn.Sequential(
            nn.Conv3d(1, maps_3d, kernel_size=(spectral_kernel, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(maps_3d), nn.ReLU(inplace=True))
        self.msi_adjust = nn.Conv2d(maps_3d * target_spectral, dim, kernel_size=1)
        self.sar_stem = nn.Sequential(
            nn.Conv2d(sar_channels, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

        self.stages = nn.ModuleList([
            MultiStageDFFMBlock(channels=dim, ssafb_out_channels=dim) for _ in range(int(num_stages))])

        # Initialise the lazily-created dynamic filter bases before the optimiser
        # is built, otherwise those parameters are invisible to optimisation.
        for module in self.stages.modules():
            init = getattr(module, '_initialize_filter_bases', None)
            if callable(init) and not getattr(module, 'initialized_bases', True):
                init(torch.device('cpu'))

        self.msi_final = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.sar_final = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.sar_adjust = nn.Conv2d(dim, dim, kernel_size=1)
        hidden = 256
        self.classifier = nn.Sequential(
            nn.Linear(dim * self.patch_size ** 2, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(hidden, num_classes))

    def forward(self, ms, sar, labels=None):
        spectral = self.msi_stem(ms.unsqueeze(1))  # [B,1,C,H,W] -> [B,maps,t,H,W]
        batch, maps, spectral_dim, height, width = spectral.shape
        msi_features = self.msi_adjust(spectral.reshape(batch, maps * spectral_dim, height, width))
        sar_features = self.sar_stem(sar)
        for stage in self.stages:
            msi_features, sar_features = stage(msi_features, sar_features)
        msi_out = self.msi_final(msi_features)
        sar_out = self.sar_adjust(self.sar_final(sar_features))
        fused = msi_out + sar_out
        logits = self.classifier(fused.flatten(1))
        result = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result['losses'] = {'total': loss, 'ce': loss}
        return result
