"""Feature-space separability audit for the 14-channel YRD2509NEW candidates.

Distances use one global, robust (median/IQR) standardisation per feature,
then clip each standardised feature to +/-8 (the same robust treatment used
during component ranking) so the heavy tail of VV/VH cannot dominate.
SNR is centroid distance divided by the mean Euclidean radius of the two
classes in that same space.  It is a separability diagnostic, not accuracy.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio

import build_yrd2509new as base


ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data" / "yrd2509new"
SOURCE = ROOT / "data" / "yrd2509_landuse_9c"
NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "VH_linear", "VV_linear", "VV_minus_VH", "VV_div_VH", "NDVI", "NDRE", "NDMI"]
SPACES = {"optical_10": slice(0, 10), "sar_4_yrd_matched": slice(10, 14), "joint_14": slice(0, 14)}
VEGETATION_PAIRS = ((2, 4), (2, 6), (2, 8), (4, 6), (4, 8), (6, 8))


def global_features(data: np.ndarray) -> np.ndarray:
    if data.shape[-1] != 14:
        raise ValueError(f"Expected 14-channel YRD-compatible input, got {data.shape}")
    optical = data[:, :, :10]
    eps = np.float32(1e-6)
    derived = np.stack(
        ((optical[:, :, 6] - optical[:, :, 2]) / (optical[:, :, 6] + optical[:, :, 2] + eps),
         (optical[:, :, 7] - optical[:, :, 3]) / (optical[:, :, 7] + optical[:, :, 3] + eps),
         (optical[:, :, 6] - optical[:, :, 8]) / (optical[:, :, 6] + optical[:, :, 8] + eps)), axis=-1)
    return np.concatenate((data, derived), axis=-1).astype(np.float32, copy=False)


def robust_z(features: np.ndarray) -> np.ndarray:
    median = np.median(features, axis=(0, 1))
    iqr = np.percentile(features, 75, axis=(0, 1)) - np.percentile(features, 25, axis=(0, 1))
    return np.clip((features - median) / np.maximum(iqr, 1e-6), -8.0, 8.0)


def metric_rows(z: np.ndarray, label: np.ndarray, dataset: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left, right in VEGETATION_PAIRS:
        a, b = z[label == left], z[label == right]
        for name, columns in SPACES.items():
            aa, bb = a[:, columns], b[:, columns]
            mean_a, mean_b = aa.mean(0), bb.mean(0)
            distance = float(np.linalg.norm(mean_a - mean_b))
            radius_a = float(np.linalg.norm(aa - mean_a, axis=1).mean())
            radius_b = float(np.linalg.norm(bb - mean_b, axis=1).mean())
            rows.append({"dataset": dataset, "pair": f"{left}-{right}", "space": name,
                         "n_left": len(aa), "n_right": len(bb), "centroid_distance": distance,
                         "within_radius": (radius_a + radius_b) / 2, "snr": distance / max((radius_a + radius_b) / 2, 1e-8)})
    return rows


def main() -> None:
    _, data = base.read_mat_v5(NEW / "data.mat")
    _, new_label = base.read_mat_v5(NEW / "label.mat")
    with rasterio.open(SOURCE / "label_landuse_9c_1024.tif") as ds:
        old_label = ds.read(1).astype(np.uint8)
    raw = global_features(data.astype(np.float32, copy=False))
    z = robust_z(raw)
    rows = metric_rows(z, old_label, "yrd2509_original_all_labels_14ch") + metric_rows(z, new_label, "yrd2509new_continuous_candidates_14ch")
    with (NEW / "feature_pair_separability_14ch.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    means: list[dict[str, object]] = []
    for class_id, class_name, _ in base.CLASSES:
        values = raw[new_label == class_id]
        if not len(values):
            continue
        row: dict[str, object] = {"class_id": class_id, "class_name": class_name, "pixels": len(values)}
        row.update({name: float(value) for name, value in zip(NAMES, values.mean(0))})
        means.append(row)
    with (NEW / "feature_class_means_14ch.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(means[0])); writer.writeheader(); writer.writerows(means)
    # Band-level standardised effect sizes for the key Tamarix/Suaeda pair.
    a, b = raw[new_label == 2], raw[new_label == 4]
    effect = (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + b.var(0)) / 2 + 1e-12)
    with (NEW / "tamarix_suaeda_feature_effects_14ch.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out); writer.writerow(["feature", "tamarix_mean", "suaeda_mean", "cohen_d"])
        writer.writerows((name, float(x), float(y), float(d)) for name, x, y, d in zip(NAMES, a.mean(0), b.mean(0), effect))
    for row in rows:
        if row["pair"] in {"2-4", "6-8"}:
            print(row)


if __name__ == "__main__":
    main()
