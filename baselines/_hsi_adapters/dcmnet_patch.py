"""DCMNet (TGRS 2025) patch-classification adaptation for the MSI-SAR project.

Vendored source: ``vendor/DCMNet-original`` (oucailab/DCMNet, commit
b88ead6db8bc8a5669290e2e6c3af7d062158171).  The released routing module
(``net/Cells.py``, ``net/Feature_Interaction.py``, ``net/routing_modules.py``),
the router (``net/Router.py``) and the classifier (``net/classify.py``) are
imported unchanged; only the two modality encoders and the argument stub are
rebuilt so that 6/10-band MSI and 1/4-band SAR patches of the project protocol
can be fed.

Adaptation decisions (recorded in ``baselines/dcmnet/README.md``):
- the released encoders are hard-coded per dataset (PCA-30 / 20 / 63 spectral
  bands, fixed patch sizes); here the 3D spectral kernels are depth-1 and the
  spectral mixing happens in the released 2D projections, which reproduces the
  released Trento branch shape (C=10) and extends it to C=6;
- the SAR branch is the released 2D LiDAR stem with its first layer taking the
  actual SAR channel count (no LiDAR-height semantics are assumed and the band
  count is not faked);
- ``patch_size`` must be odd and >= 9; the routing cells require a square
  feature map of side ``patch_size - 6`` (5 for patch 11), which matches the
  released Houston-style re-arrangements.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_VENDOR = Path(__file__).resolve().parents[2] / 'vendor' / 'DCMNet-original'

# The released modules read ``from parameter import args`` at import time.  A
# stub module keeps the vendored code untouched and avoids its data-loading
# ``parameter.py`` (which reads .mat files at import).
_ARGS = types.SimpleNamespace()
if 'parameter' not in sys.modules:
    _stub = types.ModuleType('parameter')
    _stub.args = _ARGS
    sys.modules['parameter'] = _stub
elif not hasattr(sys.modules['parameter'], 'args'):
    sys.modules['parameter'].args = _ARGS
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

RoutingModule = importlib.import_module('net.routing_modules').RoutingModule
_classify = importlib.import_module('net.classify').classify


def _prune_released_dead_modules(routing_module) -> None:
    """Remove released modules that the released forward pass never uses.

    The vendor ``Cells``/``Feature_Interaction`` code instantiates both stage
    variants of every interaction layer (``c11..c13`` and ``c21..c23``) while a
    given layer only runs one stage, defines ``Cross_Self_attention.dense`` but
    never applies it, and keeps ``RoutingModule.bn``/``path_mapping`` unused.
    Those parameters would be registered in the optimizer yet never receive a
    gradient.  Pruning keeps the active parameter set explicit; the vendor
    source files themselves stay untouched.
    """
    for module in list(routing_module.modules()):
        if hasattr(module, 'sa') and hasattr(module.sa, 'dense'):
            del module.sa.dense
    for layer in (routing_module.dynamic_itr_l0, routing_module.dynamic_itr_l1,
                  routing_module.dynamic_itr_l2):
        if hasattr(layer, 'conv2'):
            del layer.conv2
    for layer in (routing_module.dynamic_itr_l1, routing_module.dynamic_itr_l2):
        stage = int(getattr(layer, 'stage', 1))
        for name in (('c21', 'c22', 'c23') if stage == 1 else ('c11', 'c12', 'c13')):
            if hasattr(layer, name):
                delattr(layer, name)
    for name in ('bn', 'path_mapping'):
        if hasattr(routing_module, name):
            delattr(routing_module, name)


def _require_patch(patch_size: int) -> int:
    if patch_size % 2 != 1 or patch_size < 9:
        raise ValueError('DCMNet adaptation requires an odd patch_size >= 9 (routing cells need patch-6)')
    return patch_size - 6


class _MSIEncoder(nn.Module):
    """Released Trento-style HSI stem with depth-1 spectral kernels."""

    def __init__(self, channels: int, uni_dimension: int):
        super().__init__()
        self.step1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 3, 3), padding=0),
            nn.ReLU(inplace=True), nn.BatchNorm3d(8))
        self.step2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(1, 3, 3), padding=0),
            nn.ReLU(inplace=True), nn.BatchNorm3d(16))
        self.step3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1, 3, 3), padding=0),
            nn.ReLU(inplace=True), nn.BatchNorm3d(32))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=8 * channels, out_channels=uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(), nn.BatchNorm2d(uni_dimension))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16 * channels, out_channels=uni_dimension, kernel_size=3, padding=0),
            nn.ReLU(), nn.BatchNorm2d(uni_dimension))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32 * channels, out_channels=uni_dimension, kernel_size=1, padding=0),
            nn.ReLU(), nn.BatchNorm2d(uni_dimension))

    def forward(self, x):
        maps = []
        x1 = self.step1(x)
        maps.append(self.conv1(x1.reshape(-1, 8 * x.shape[2], x1.shape[3], x1.shape[4])))
        x2 = self.step2(x1)
        maps.append(self.conv2(x2.reshape(-1, 16 * x.shape[2], x2.shape[3], x2.shape[4])))
        x3 = self.step3(x2)
        maps.append(self.conv3(x3.reshape(-1, 32 * x.shape[2], x3.shape[3], x3.shape[4])))
        return torch.stack(maps, dim=0)


class _SAREncoder(nn.Module):
    """Released 2D LiDAR stem, first layer takes the actual SAR channel count."""

    def __init__(self, channels: int, uni_dimension: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True), nn.BatchNorm2d(64))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True), nn.BatchNorm2d(128))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=uni_dimension, kernel_size=3, padding=0),
            nn.ReLU(inplace=True), nn.BatchNorm2d(uni_dimension))
        self.out1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(), nn.BatchNorm2d(uni_dimension))
        self.out2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=uni_dimension, kernel_size=3, padding=0),
            nn.ReLU(), nn.BatchNorm2d(uni_dimension))

    def forward(self, x):
        maps = []
        x1 = self.conv1(x)
        maps.append(self.out1(x1))
        x2 = self.conv2(x1)
        maps.append(self.out2(x2))
        maps.append(self.conv3(x2))
        return torch.stack(maps, dim=0)


class DCMNetPatch(nn.Module):
    def __init__(self, ms_channels, sar_channels, num_classes, patch_size=11,
                 uni_dimension=64, embed_size=64, hid_router=256):
        super().__init__()
        self.patch_size = int(patch_size)
        side = _require_patch(self.patch_size)
        _ARGS.uni_dimension = int(uni_dimension)
        _ARGS.embed_size = int(embed_size)
        _ARGS.hid_router = int(hid_router)
        _ARGS.lidar_channel = int(uni_dimension)
        _ARGS.hsi_channel = int(uni_dimension)
        _ARGS.hsi_first_windowSize = self.patch_size
        _ARGS.size_SA = side * side
        _ARGS.hsi_windowSize = side
        _ARGS.num_classes = int(num_classes)
        _ARGS.learning_rate = 0.001
        self.hsi_encoder = _MSIEncoder(int(ms_channels), int(uni_dimension))
        self.lidar_encoder = _SAREncoder(int(sar_channels), int(uni_dimension))
        self.routing_module = RoutingModule(_ARGS)
        _prune_released_dead_modules(self.routing_module)
        self.classifier = _classify()

    def forward(self, ms, sar, labels=None):
        e_hsi = self.hsi_encoder(ms.unsqueeze(1))
        e_sar = self.lidar_encoder(sar)
        features, _paths = self.routing_module(e_hsi, e_sar)
        logits = self.classifier(features[0])
        result = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result['losses'] = {'total': loss, 'ce': loss}
        return result
