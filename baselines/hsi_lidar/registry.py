"""Registry for the HSI–LiDAR-derived patch baselines adapted to MSI–SAR.

Only models with a verified adapter live here. DCMNet and HSLiNets are kept in
``vendor/hsi_lidar/`` for the next adaptation batch; they are deliberately not
listed, so a queue cannot silently claim to have run them.
"""
from __future__ import annotations

from baselines.hsi_lidar.dffnet_patch import DFFNetPatch
from baselines.hsi_lidar.m2heat_patch import M2HeatPatch
from baselines.hsi_lidar.mft_patch import MFTPatch

MODELS = {
    'dffnet': DFFNetPatch,
    'm2heat': M2HeatPatch,
    'mft': MFTPatch,
}

PENDING = {
    'dcmnet': '按数据集硬编码的光谱卷积核/Cell 尺寸需要重写，尚未适配',
    'hslinets': '自定义 Vim/SSM 块与 notebook 结构，尚未适配',
}

DISPLAY_NAMES = {
    'dffnet': 'DFFNet (GRSL 2025, DFFM/SSAFB)',
    'm2heat': 'M2Heat (TCSVT 2026, heat conduction + CFF)',
    'mft': 'MFT (TGRS 2023, mCrossPA Transformer)',
}


def build_model(name, ms_channels, sar_channels, num_classes, patch_size=11):
    if name not in MODELS:
        reason = PENDING.get(name, 'unknown model')
        raise KeyError(f'{name}: {reason}')
    builder = MODELS[name]
    return builder(ms_channels=ms_channels, sar_channels=sar_channels,
                   num_classes=num_classes, patch_size=patch_size)
