"""Adapter around the official MSFMamba network components.

The upstream source is preserved verbatim in ``vendor/MSFMamba-original``.
Only the YAML-bound ``Net`` constructor is replaced here so channel counts and
class count come from this repository's experiment configuration.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .channel_policy import required_spectral_channels


def _selective_scan_torch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """Pure-PyTorch equivalent of Mamba's selective_scan_ref.

    MSFMamba only uses this public Mamba operator.  The fallback keeps the
    upstream tensor equations and autograd semantics when no binary extension
    exists for a recent GPU / PyTorch combination.  It is slower than the CUDA
    extension, but practical for this baseline's 11x11 input patches.
    """
    input_dtype = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, length = u.shape
    state_size = A.shape[1]
    variable_b = B.dim() >= 3
    variable_c = C.dim() >= 3
    if variable_b:
        B = B.float()
        if B.dim() == 3:
            B = B.unsqueeze(1)
        if B.shape[1] != dim:
            B = B.repeat_interleave(dim // B.shape[1], dim=1)
    if variable_c:
        C = C.float()
        if C.dim() == 3:
            C = C.unsqueeze(1)
        if C.shape[1] != dim:
            C = C.repeat_interleave(dim // C.shape[1], dim=1)

    state = A.new_zeros((batch, dim, state_size))
    outputs = []
    for index in range(length):
        delta_a = torch.exp(delta[:, :, index].unsqueeze(-1) * A)
        if variable_b:
            delta_b_u = delta[:, :, index].unsqueeze(-1) * B[:, :, :, index] * u[:, :, index].unsqueeze(-1)
        else:
            delta_b_u = delta[:, :, index].unsqueeze(-1) * B.unsqueeze(0) * u[:, :, index].unsqueeze(-1)
        state = delta_a * state + delta_b_u
        if variable_c:
            output = (state * C[:, :, :, index]).sum(dim=-1)
        else:
            output = (state * C.unsqueeze(0)).sum(dim=-1)
        outputs.append(output)

    output = torch.stack(outputs, dim=-1)
    if D is not None:
        output = output + u * D.float().view(1, -1, 1)
    if z is not None:
        output = output * F.silu(z.float())
    output = output.to(dtype=input_dtype)
    return (output, state) if return_last_state else output


def _install_mamba_fallback() -> None:
    """Expose the two upstream import names without modifying vendor source."""
    for name in tuple(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            sys.modules.pop(name, None)
    package = types.ModuleType("mamba_ssm")
    package.__path__ = []
    ops = types.ModuleType("mamba_ssm.ops")
    ops.__path__ = []
    interface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    interface.selective_scan_fn = _selective_scan_torch
    interface.selective_scan_ref = _selective_scan_torch
    package.ops = ops
    ops.selective_scan_interface = interface
    sys.modules[package.__name__] = package
    sys.modules[ops.__name__] = ops
    sys.modules[interface.__name__] = interface


def _official_syn_layer():
    vendor_root = Path(__file__).resolve().parents[2] / "vendor" / "MSFMamba-original"
    if not vendor_root.exists():
        raise FileNotFoundError(
            f"Official MSFMamba snapshot is missing: {vendor_root}. "
            "Restore vendor/MSFMamba-original from https://github.com/oucailab/MSFMamba."
        )
    vendor_text = str(vendor_root)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    try:
        importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    except ImportError:
        _install_mamba_fallback()
    try:
        return importlib.import_module("model.MSFMamba").Syn_layer
    except ImportError as exc:
        raise RuntimeError(
            "MSFMamba needs timm and einops. Install the project requirements first."
        ) from exc


class MSFMamba(nn.Module):
    """Official MSFMamba architecture with configurable modality dimensions.

    The paper's first stage uses a 9-band spectral kernel and a 3x3 spatial
    kernel. Inputs with fewer than nine optical bands are zero-padded along
    the spectral axis so the released layer can be reused without inventing
    additional observations.
    """

    def __init__(
        self,
        ms_channels: int,
        sar_channels: int,
        num_classes: int,
        patch_size: int = 11,
        d_state: int = 16,
        expand: float = 0.75,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if patch_size != 11:
            raise ValueError("MSFMamba's released architecture is configured for patch_size=11")
        if ms_channels < 1:
            raise ValueError("MSFMamba requires at least one MSI/HSI channel")
        if sar_channels < 1:
            raise ValueError("MSFMamba requires at least one SAR channel")
        if num_layers != 1:
            raise ValueError(
                "The 10-band MSI adaptation uses one official Syn_layer; set num_layers: 1."
            )

        SynLayer = _official_syn_layer()
        hsi_out, sar_out = 8, 64
        layer_ms_channels = required_spectral_channels(ms_channels)
        self.ms_channels = ms_channels
        self.spectral_padding = layer_ms_channels - ms_channels
        spectral_after_conv = layer_ms_channels - 8  # Conv3d spectral kernel=9.
        spatial_after_conv = patch_size - 2
        self.layer = SynLayer(
            1, hsi_out, sar_channels, sar_out, layer_ms_channels, patch_size, d_state, 0, expand
        )
        self.classifier = nn.Linear(
            (sar_out + hsi_out * spectral_after_conv) * spatial_after_conv**2,
            num_classes,
        )

    def forward(self, ms: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        if ms.ndim != 4 or sar.ndim != 4:
            raise ValueError("MSFMamba expects ms and sar tensors shaped [B, C, H, W]")
        if ms.shape[1] != self.ms_channels:
            raise ValueError(f"Expected {self.ms_channels} MSI/HSI channels, got {ms.shape[1]}")
        if self.spectral_padding:
            ms = F.pad(ms, (0, 0, 0, 0, 0, self.spectral_padding))
        hsi, sar = self.layer(ms.unsqueeze(1), sar)
        batch, channels, spectral, height, width = hsi.shape
        hsi = hsi.reshape(batch, channels * spectral, height, width)
        fused = torch.cat((hsi, sar), dim=1)
        return self.classifier(fused.reshape(batch, -1))
