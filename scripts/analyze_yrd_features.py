"""Full-scene 14-channel feature separability audit for original YRD.

Run this script in a Python environment with compatible h5py.  It uses the
same median/IQR global normalisation and +/-8 clipping as the YRD2509NEW
14-channel audit, making the reported distances directly comparable.
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yrd" / "data.mat"
LABEL = ROOT / "data" / "yrd" / "label.mat"
OUTPUT = ROOT / "data" / "yrd"
NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "VH_linear", "VV_linear", "VV_minus_VH", "VV_div_VH", "NDVI", "NDRE", "NDMI"]
SPACES = {"optical_10": slice(0, 10), "sar_4": slice(10, 14), "joint_14": slice(0, 14)}
PAIRS = ((4, 5), (4, 3), (5, 3), (3, 7), (4, 7), (5, 7))


def metrics(z: np.ndarray, label: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for left, right in PAIRS:
        a, b = z[:, label == left].T, z[:, label == right].T
        for name, columns in SPACES.items():
            aa, bb = a[:, columns], b[:, columns]
            mean_a, mean_b = aa.mean(0), bb.mean(0)
            distance = float(np.linalg.norm(mean_a - mean_b))
            ra = float(np.linalg.norm(aa - mean_a, axis=1).mean())
            rb = float(np.linalg.norm(bb - mean_b, axis=1).mean())
            radius = (ra + rb) / 2
            rows.append({"dataset": "yrd_original_14ch", "pair": f"{left}-{right}", "space": name,
                         "n_left": len(aa), "n_right": len(bb), "centroid_distance": distance,
                         "within_radius": radius, "snr": distance / max(radius, 1e-8)})
    return rows


def main() -> None:
    with h5py.File(DATA, "r") as file:
        data = file["I"][:].astype(np.float32, copy=False)
    with h5py.File(LABEL, "r") as file:
        label = file["T"][:].astype(np.uint8, copy=False)
    if data.shape != (14, 2048, 2048) or label.shape != (2048, 2048):
        raise ValueError(f"Unexpected shapes: {data.shape}, {label.shape}")
    median = np.median(data, axis=(1, 2))[:, None, None]
    iqr = (np.percentile(data, 75, axis=(1, 2)) - np.percentile(data, 25, axis=(1, 2)))[:, None, None]
    z = np.clip((data - median) / np.maximum(iqr, 1e-6), -8.0, 8.0)
    rows = metrics(z, label)
    with (OUTPUT / "feature_pair_separability_14ch.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    optical = data[:10]
    eps = np.float32(1e-6)
    derived = np.stack(((optical[6] - optical[2]) / (optical[6] + optical[2] + eps),
                        (optical[7] - optical[3]) / (optical[7] + optical[3] + eps),
                        (optical[6] - optical[8]) / (optical[6] + optical[8] + eps)))
    full = np.concatenate((data, derived), axis=0)
    a, b = full[:, label == 4].T, full[:, label == 5].T
    effect = (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + b.var(0)) / 2 + 1e-12)
    with (OUTPUT / "tamarix_suaeda_feature_effects_14ch.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out); writer.writerow(["feature", "tamarix_mean", "suaeda_mean", "cohen_d"])
        writer.writerows((name, float(x), float(y), float(d)) for name, x, y, d in zip(NAMES, a.mean(0), b.mean(0), effect))
    for row in rows:
        if row["pair"] in {"4-5", "3-7"}:
            print(row)


if __name__ == "__main__":
    main()
