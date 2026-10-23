"""HSLiNets (2024-2025) patch-classification adaptation for the MSI-SAR project.

The released repository ships only notebooks; the two class cells needed for the
HSI backbone (``HSIVimBlock``, ``SpatialFeatureProcessing``, ``Classifier``,
``HSIClassificationMambaModel``) were extracted verbatim into
``baselines/_hsi_adapters/_hslinets_vendor.py`` (see that file's header).

Blocker and chosen variant (recorded in ``baselines/hslinets/README.md``):
- the notebook references ``HSILidClassificationMambaModel`` but never defines
  it (only commented instantiations with ``hsi_num_bands`` / ``lidar_num_bands``
  arguments), so the paper's middle-fusion dual-branch model cannot be restored
  from the released code without inventing a network;
- the notebook *does* document a "LiDAR as additional band" early-fusion cell
  ("At The MEarly Fusion Stage, Lidar as additional band"), i.e. one of the
  band-ordering strategies evaluated by HSLiNets.  This adapter implements that
  early-fusion strategy: MSI and SAR channels are concatenated as extra bands
  (MSI first, then SAR) and fed to the released HSI model, which keeps the
  released Vim/SSM block untouched.  It is not a reproduction of the missing
  middle-fusion network.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._hsi_adapters._hslinets_vendor import HSIClassificationMambaModel


class HSLiNetsPatch(nn.Module):
    def __init__(self, ms_channels, sar_channels, num_classes, patch_size=11,
                 hidden_dim=512, output_dim=256, delta_param_init=0.01):
        super().__init__()
        if patch_size % 2 != 1:
            raise ValueError('HSLiNets adaptation requires an odd patch_size')
        self.ms_channels = int(ms_channels)
        self.sar_channels = int(sar_channels)
        self.num_bands = self.ms_channels + self.sar_channels
        self.model = HSIClassificationMambaModel(
            spatial_dim=int(patch_size), num_bands=self.num_bands,
            hidden_dim=int(hidden_dim), output_dim=int(output_dim),
            delta_param_init=float(delta_param_init), num_classes=int(num_classes))

    def forward(self, ms, sar, labels=None):
        bands = torch.cat((ms, sar), dim=1).permute(0, 2, 3, 1)  # [B,H,W,C]
        logits = self.model(bands)
        result = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result['losses'] = {'total': loss, 'ce': loss}
        return result
