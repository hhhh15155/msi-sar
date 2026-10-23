"""M2Heat (TCSVT 2026) patch-classification adaptation for the MSI-SAR project.

Vendored source: ``vendor/hsi_lidar/M2Heat-original`` (Weikan0425/M2Heat_HSI_LiDAR,
commit 8fcdce85e92a5d8b92c0a86ca78f835113dd3e3f).  The released ``m2heat.py`` is
used unchanged; this adapter only maps the project interface.

Deviations / notes:
- the released classifier ends with ``softmax``; we expose ``log`` probabilities
  as logits so the project's cross-entropy protocol stays comparable;
- the auxiliary stream accepts any channel count (1/4 SAR here, LiDAR 1 in the
  original demos), the FVE parameters depend on ``patch_size`` and are built in
  ``__init__``;
- no pretraining, patch 11 (the original demos use 13, configurable).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_VENDOR = Path(__file__).resolve().parents[2] / 'vendor' / 'hsi_lidar' / 'M2Heat-original'


def _load_module(alias: str, relative: str):
    spec = importlib.util.spec_from_file_location(alias, _VENDOR / relative)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load vendored module {relative}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_m2heat = _load_module('_m2heat_vendor', 'm2heat.py')


class M2HeatPatch(nn.Module):
    def __init__(self, ms_channels, sar_channels, num_classes, patch_size=11, dim=64, depth=2):
        super().__init__()
        self.patch_size = int(patch_size)
        self.model = _m2heat.M2Heat(
            patch_size=self.patch_size, num_classes=num_classes,
            num_patches=(int(ms_channels), int(sar_channels)),
            dim=int(dim), hidden_dim=int(dim), depth=int(depth),
            fves_init='trunc_normal', fves_std=0.02, fves_trainable=True)

    def forward(self, ms, sar, labels=None):
        probabilities = self.model(ms, sar)
        logits = torch.log(probabilities.clamp_min(1e-7))
        result = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result['losses'] = {'total': loss, 'ce': loss}
        return result
