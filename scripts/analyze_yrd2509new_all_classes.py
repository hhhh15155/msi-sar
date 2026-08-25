"""All-pair 14-channel separability table for current YRD2509NEW labels."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

import analyze_yrd2509new_features as analysis
import build_yrd2509new as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "yrd2509new"
SPACES = {"optical_10": slice(0, 10), "sar_4_yrd_matched": slice(10, 14), "joint_14": slice(0, 14)}


def main() -> None:
    _, data = base.read_mat_v5(OUTPUT / "data.mat")
    _, label = base.read_mat_v5(OUTPUT / "label.mat")
    z = analysis.robust_z(analysis.global_features(data.astype(np.float32, copy=False)))
    names = {class_id: name for class_id, name, _ in base.CLASSES}
    rows = []
    for left, right in combinations(names, 2):
        a, b = z[label == left], z[label == right]
        for space, columns in SPACES.items():
            aa, bb = a[:, columns], b[:, columns]
            mean_a, mean_b = aa.mean(0), bb.mean(0)
            distance = float(np.linalg.norm(mean_a - mean_b))
            radius = float((np.linalg.norm(aa - mean_a, axis=1).mean() + np.linalg.norm(bb - mean_b, axis=1).mean()) / 2)
            rows.append({"left_id": left, "left_class": names[left], "right_id": right, "right_class": names[right], "space": space,
                         "n_left": len(aa), "n_right": len(bb), "centroid_distance": distance, "within_radius": radius, "snr": distance / max(radius, 1e-8)})
    with (OUTPUT / "all_class_pair_separability_14ch.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    joint = sorted((row for row in rows if row["space"] == "joint_14"), key=lambda row: row["snr"])
    with (OUTPUT / "all_class_pair_separability_summary.txt").open("w", encoding="utf-8") as out:
        out.write("Current YRD2509NEW all-pair separability, robust median/IQR normalisation with +/-8 clipping.\n")
        out.write("SNR = centroid distance / mean within-class Euclidean radius.\n\n")
        out.write("Joint-14 pairs ordered from least to most separable:\n")
        for row in joint:
            out.write(f"{row['left_id']} {row['left_class']} <-> {row['right_id']} {row['right_class']}: distance={row['centroid_distance']:.3f}, SNR={row['snr']:.3f}\n")
    for row in joint[:12]:
        print(f"{row['left_class']} <-> {row['right_class']}: d={row['centroid_distance']:.3f}, SNR={row['snr']:.3f}")


if __name__ == "__main__":
    main()
