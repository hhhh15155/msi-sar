"""GFSFN core with native modality channels and a batch-safe output shape.

Source: Zzp-12/GFSFN, commit 065cb808870515b70c4980642895fa45da63a1af.
The upstream repository has no explicit code license; see the model README.
All active convolution, gating, texture and frequency modules are unchanged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_SOURCE = Path(__file__).resolve().parents[2] / "vendor/GFSFN-original/newnet.py"
_SPEC = importlib.util.spec_from_file_location("_gfsfn_original", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load GFSFN source: {_SOURCE}")
_VENDOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VENDOR)


class GFSFNPatch(_VENDOR.GFSFN):
    def __init__(self, ms_channels: int, sar_channels: int, num_classes: int):
        # These are native upstream constructor arguments, not invented bands.
        super().__init__(num_classes=num_classes, dim=224,
                         band1=ms_channels, band2=sar_channels)
        # Registered upstream but never used by MixTrans.forward.
        del self.Mt.p

    def forward(self, ms: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        # Upstream squeeze() removes the batch axis when B=1; restore it.
        return super().forward(ms, sar).reshape(ms.shape[0], -1)
