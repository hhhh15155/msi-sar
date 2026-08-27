"""Spectral input policy shared by the MSFMamba adapter and its tests."""

from __future__ import annotations


OFFICIAL_SPECTRAL_KERNEL = 9


def required_spectral_channels(ms_channels: int) -> int:
    """Return the input depth needed by the released nine-band kernel."""
    return max(int(ms_channels), OFFICIAL_SPECTRAL_KERNEL)
