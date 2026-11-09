"""M2Heat branch combinations and a raw-patch linear reference."""
from __future__ import annotations

import torch
from torch import nn

from baselines._hsi_adapters.m2heat_patch import _m2heat

VARIANTS = ('wo_specific_hco', 'wo_shared_hco', 'wo_cff')
MODULE_VARIANTS = ('linear_base', 'private', 'private_shared', 'private_fusion')
PRIVATE_PART_VARIANTS = ('private_neither', 'private_stem_only', 'private_hclayer_only')
SUITE_VARIANTS = {'legacy': VARIANTS, 'modules': MODULE_VARIANTS, 'private_parts': PRIVATE_PART_VARIANTS}
CONFIGURED_VARIANTS = MODULE_VARIANTS + PRIVATE_PART_VARIANTS

_NO_PRIVATE_HCLAYERS = ('wo_specific_hco', 'private_neither', 'private_stem_only')
_NO_STEMS = ('private_neither', 'private_hclayer_only')
_AVERAGE_PRIVATE_ONLY = ('private',) + PRIVATE_PART_VARIANTS


class _AblatedCore(_m2heat.M2Heat):
    def __init__(self, variant: str, **kwargs):
        if variant not in VARIANTS + CONFIGURED_VARIANTS or variant == 'linear_base':
            raise ValueError(f'Unsupported M2Heat core variant: {variant!r}')
        # Construct in the original order before removing bypassed parameters:
        # all retained weights and the RNG state match the original initialization.
        super().__init__(**kwargs)
        self.variant = variant
        self.feature_width = kwargs['dim']
        if variant in _NO_STEMS and sum(kwargs['num_patches']) > self.feature_width:
            raise ValueError('Raw modalities must fit in disjoint slots of the feature width')
        removed = {
            'wo_shared_hco': ('hclayer_fuse', 'fves_fuse'),
            'wo_cff': ('cff',),
            'private_shared': ('cff', 'cof1', 'cof2'),
            'private_fusion': ('hclayer_fuse', 'fves_fuse', 'proj', 'cof2'),
        }.get(variant, ())
        if variant in _AVERAGE_PRIVATE_ONLY:
            removed += ('hclayer_fuse', 'fves_fuse', 'proj', 'cff', 'cof1', 'cof2')
        if variant in _NO_PRIVATE_HCLAYERS:
            removed += ('hclayer_hsi', 'hclayer_aux', 'fves_hsi', 'fves_aux')
        if variant in _NO_STEMS:
            removed += ('stem_hsi', 'stem_aux')
        for name in removed:
            delattr(self, name)

    def forward(self, hsi, auxiliary):
        if self.variant in _NO_STEMS:
            msi_channels, sar_channels = hsi.shape[1], auxiliary.shape[1]
            # Disjoint slots keep the original MSI/SAR bands separate before the mean.
            hsi = torch.nn.functional.pad(hsi, (0, 0, 0, 0, 0, self.feature_width - msi_channels))
            auxiliary = torch.nn.functional.pad(auxiliary, (0, 0, 0, 0, msi_channels,
                                                             self.feature_width - msi_channels - sar_channels))
        else:
            hsi, hsi_gate = self.stem_hsi(hsi).chunk(2, dim=1)
            auxiliary, aux_gate = self.stem_aux(auxiliary).chunk(2, dim=1)
        if self.variant not in _NO_PRIVATE_HCLAYERS:
            hsi = self.hclayer_hsi(hsi, self.fves_hsi)
            auxiliary = self.hclayer_aux(auxiliary, self.fves_aux)
        if self.variant in _AVERAGE_PRIVATE_ONLY:
            return self.cls((hsi + auxiliary) / 2)
        if self.variant == 'private_fusion':
            return self.cls(self.cff(hsi, auxiliary) * self.cof1)

        fused = torch.cat((hsi_gate, aux_gate), dim=1)
        if self.variant != 'wo_shared_hco':
            fused = self.hclayer_fuse(fused, self.fves_fuse)
        if self.variant == 'private_shared':
            return self.cls((hsi + auxiliary) / 2 + self.proj(fused))
        specific = (hsi + auxiliary) / 2 if self.variant == 'wo_cff' else self.cff(hsi, auxiliary)
        return self.cls(specific * self.cof1 + self.proj(fused) * self.cof2)


class M2HeatAblation(nn.Module):
    def __init__(self, variant: str, num_classes=8, patch_size=11,
                 spectral_channels=10, sar_channels=4):
        super().__init__()
        if int(patch_size) <= 0 or int(patch_size) % 2 != 1:
            raise ValueError('patch_size must be a positive odd integer')
        self.patch_size = int(patch_size)
        self.spectral_channels = int(spectral_channels)
        self.num_channels = self.spectral_channels + int(sar_channels)
        self.variant = variant
        if variant == 'linear_base':
            self.classifier = nn.Linear(self.num_channels * self.patch_size ** 2, int(num_classes))
            return
        # Preserve original checkpoint key prefixes and original log-prob interface.
        self.net = nn.Module()
        self.net.model = _AblatedCore(
            variant=variant, patch_size=self.patch_size, num_classes=int(num_classes),
            num_patches=(self.spectral_channels, int(sar_channels)), dim=64,
            hidden_dim=64, depth=2, fves_init='trunc_normal', fves_std=0.02,
            fves_trainable=True)

    def forward(self, x):
        x = x.squeeze(dim=1)
        if x.ndim != 4 or x.shape[1] != self.num_channels or x.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError(f'Expected [B,1,{self.num_channels},{self.patch_size},{self.patch_size}], got {tuple(x.shape)}')
        if self.variant == 'linear_base':
            return self.classifier(x.flatten(1))
        probabilities = self.net.model(x[:, :self.spectral_channels], x[:, self.spectral_channels:])
        return torch.log(probabilities.clamp_min(1e-7))
