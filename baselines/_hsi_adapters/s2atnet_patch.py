"""Load the fixed S2ATNet source without its unused timm runtime dependency.

Only the timm import is removed from the in-memory AST. Its sole executed
symbol, trunc_normal_, is supplied by torch.nn.init (same distribution and
defaults). DropPath is never constructed: the released S2ATNet uses rate 0.
The vendored source files remain byte-for-byte unchanged.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

from torch import nn

_DIRECTORY = Path(__file__).resolve().parents[2] / "vendor/S2ATNet-original/models/s2at"
_ALIAS = "_msi_s2atnet_original"
_PACKAGE = types.ModuleType(_ALIAS)
_PACKAGE.__path__ = [str(_DIRECTORY)]
sys.modules[_ALIAS] = _PACKAGE
_SPEC = importlib.util.spec_from_file_location(_ALIAS + ".cfsm", _DIRECTORY / "cfsm.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load S2ATNet CFSM: {_DIRECTORY}")
_CFSM = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CFSM
_SPEC.loader.exec_module(_CFSM)

_SOURCE = _DIRECTORY / "s2at.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"), filename=str(_SOURCE))
_TREE.body = [node for node in _TREE.body if not (
    isinstance(node, ast.ImportFrom) and node.module == "timm.models.layers"
)]
_VENDOR = types.ModuleType(_ALIAS + ".s2at")
_VENDOR.__file__ = str(_SOURCE)
_VENDOR.__package__ = _ALIAS
_VENDOR.trunc_normal_ = nn.init.trunc_normal_
sys.modules[_VENDOR.__name__] = _VENDOR
exec(compile(_TREE, str(_SOURCE), "exec"), _VENDOR.__dict__)


class S2ATNetPatch(_VENDOR.S2ATNet):
    def __init__(self, ms_channels: int, sar_channels: int, num_classes: int, patch_size: int):
        super().__init__(num_classes=num_classes, hsi_in_channels=ms_channels,
                         lidar_in_channels=sar_channels, num_tokens=patch_size ** 2,
                         dim=64, heads=8, depth=1, dropout=0.1)
