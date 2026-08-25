"""Overwrite YRD2509NEW data.mat with the YRD-compatible 14-channel layout.

YRD channels 11--14 (one-based) follow exactly:
VH linear, VV linear, VV linear minus VH linear, VV linear / VH linear.
YRD2509NEW stores VV and VH in dB in channels 11--12 (one-based), so they
are converted to linear power before the last two derived channels are made.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import build_yrd2509new as base


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "yrd2509new" / "data.mat"


def main() -> None:
    key, data = base.read_mat_v5(TARGET)
    if key != "data" or data.ndim != 3 or data.shape[-1] != 12:
        raise ValueError(f"Expected unchanged 1024x1024x12 data.mat, got key={key}, shape={data.shape}")
    vv_db, vh_db = data[:, :, 10], data[:, :, 11]
    if float(np.nanmedian(vv_db)) >= 0 or float(np.nanmedian(vh_db)) >= 0:
        raise ValueError("Expected SAR inputs in dB; refusing a second conversion")
    vv_linear = np.power(np.float32(10.0), vv_db / np.float32(10.0))
    vh_linear = np.power(np.float32(10.0), vh_db / np.float32(10.0))
    sar = np.stack((vh_linear, vv_linear, vv_linear - vh_linear, vv_linear / np.maximum(vh_linear, np.float32(1e-12))), axis=-1)
    upgraded = np.concatenate((data[:, :, :10], sar), axis=-1).astype(np.float32, copy=False)
    temporary = TARGET.with_name("data_14ch.tmp.mat")
    base.write_mat_v5(temporary, "data", upgraded)
    temporary.replace(TARGET)
    description = TARGET.with_name("sar_channels.txt")
    description.write_text(
        "YRD2509NEW SAR channels (zero-based 10--13; one-based 11--14)\n"
        "11: VH linear power = 10^(VH_dB/10)\n"
        "12: VV linear power = 10^(VV_dB/10)\n"
        "13: VV linear - VH linear\n"
        "14: VV linear / VH linear\n"
        "This matches the numerical relationship verified in the original YRD data.mat.\n",
        encoding="utf-8",
    )
    print("Overwrote", TARGET)
    print("shape", upgraded.shape)
    print("SAR percentiles", np.percentile(upgraded[:, :, 10:], [1, 50, 99], axis=(0, 1)))


if __name__ == "__main__":
    main()
