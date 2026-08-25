"""Expand strict YRD2509NEW vegetation labels without losing separability.

Adds whole vector/raster-agreeing components only if the optical-10 and
joint-14 SNR of Tamarix/Suaeda and Reed/Willow stay at least 90% of the
feature-separated strict label.  The strict version is retained as
*_feature_strict.* for rollback and comparison.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio

import analyze_yrd2509new_features as analysis
import build_yrd2509new as base
import build_yrd2509new_continuous as continuous


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "yrd2509_landuse_9c"
OUTPUT = ROOT / "data" / "yrd2509new"
EXPECTED_STRICT = {2: 6921, 4: 15700, 6: 6815, 8: 3075}
CAPS = {2: 1, 4: 6, 6: 6, 8: 5}
PAIR_FOR_CLASS = {2: (2, 4), 4: (2, 4), 6: (6, 8), 8: (6, 8)}
SPACES = {"optical_10": slice(0, 10), "joint_14": slice(0, 14)}
FLOOR = 0.90


def component_key(class_id: int, points: np.ndarray) -> tuple[int, int, int, int, int, int]:
    row0, col0 = points.min(axis=0)
    row1, col1 = points.max(axis=0)
    return class_id, len(points), int(row0), int(col0), int(row1), int(col1)


def snr(z: np.ndarray, label: np.ndarray, left: int, right: int, columns: slice) -> float:
    a, b = z[label == left][:, columns], z[label == right][:, columns]
    mean_a, mean_b = a.mean(0), b.mean(0)
    distance = np.linalg.norm(mean_a - mean_b)
    radius = (np.linalg.norm(a - mean_a, axis=1).mean() + np.linalg.norm(b - mean_b, axis=1).mean()) / 2
    return float(distance / max(radius, 1e-8))


def preserve_strict_files() -> None:
    copies = {"label.mat": "label_feature_strict.mat", "label.tif": "label_feature_strict.tif", "label_envi.dat": "label_feature_strict_envi.dat", "label_envi.hdr": "label_feature_strict_envi.hdr", "roi_components.csv": "roi_components_feature_strict.csv", "label_rois.geojson": "label_rois_feature_strict.geojson"}
    for source, destination in copies.items():
        target = OUTPUT / destination
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite strict backup: {target}")
        shutil.copy2(OUTPUT / source, target)
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        target = OUTPUT / f"label_rois_feature_strict{suffix}"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite strict backup: {target}")
        shutil.copy2(OUTPUT / f"label_rois{suffix}", target)


def main() -> None:
    _, label = base.read_mat_v5(OUTPUT / "label.mat")
    counts = {class_id: int((label == class_id).sum()) for class_id in EXPECTED_STRICT}
    if counts != EXPECTED_STRICT:
        raise ValueError(f"Expected strict-label counts {EXPECTED_STRICT}, got {counts}")
    preserve_strict_files()
    _, source_data = base.read_mat_v5(SOURCE / "data.mat")
    _, data = base.read_mat_v5(OUTPUT / "data.mat")
    with rasterio.open(SOURCE / "label_landuse_9c_1024.tif") as source:
        original_label = source.read(1).astype(np.uint8)
    with rasterio.open(OUTPUT / "label.tif") as source:
        profile, transform, crs = source.profile.copy(), source.transform, source.crs
    prior, _ = continuous.vector_agreed_prior(original_label, transform, crs)
    z = analysis.robust_z(analysis.global_features(data.astype(np.float32, copy=False)))
    baseline = {pair: {name: snr(z, label, *pair, columns) for name, columns in SPACES.items()} for pair in {(2, 4), (6, 8)}}
    feature12 = base.robust_features(source_data.astype(np.float32, copy=False))
    local_variance = base.local_feature_variance(feature12, continuous.LOCAL_WINDOW)
    candidates: dict[int, list[tuple[float, np.ndarray]]] = {}
    for class_id in CAPS:
        core = base.erode_8_connected(prior == class_id, continuous.BUFFER)
        prototype = np.median(feature12[core], axis=0)
        distance = np.mean((feature12 - prototype) ** 2, axis=-1)
        dscale = max(float(np.median(distance[core])), 1e-6)
        vscale = max(float(np.median(local_variance[core])), 1e-6)
        ranked = []
        for points in continuous.connected_components(core):
            if len(points) < 100 or (label[points[:, 0], points[:, 1]] == class_id).all():
                continue
            score = float(distance[points[:, 0], points[:, 1]].mean() / dscale + local_variance[points[:, 0], points[:, 1]].mean() / vscale)
            ranked.append((score, points))
        candidates[class_id] = sorted(ranked, key=lambda item: item[0])
    expanded = label.copy()
    additions: list[tuple[int, float, np.ndarray]] = []
    for class_id in (2, 4, 6, 8):
        accepted = 0
        for score, points in candidates[class_id]:
            if accepted >= CAPS[class_id]:
                break
            proposal = expanded.copy()
            proposal[points[:, 0], points[:, 1]] = class_id
            pair = PAIR_FOR_CLASS[class_id]
            if all(snr(z, proposal, *pair, columns) >= FLOOR * baseline[pair][name] for name, columns in SPACES.items()):
                expanded = proposal
                additions.append((class_id, score, points))
                accepted += 1
    with (OUTPUT / "roi_components.csv").open(encoding="utf-8") as file:
        old_records = list(csv.DictReader(file))
    record_by_key = {(int(row["class_id"]), int(row["pixels"]), int(row["row_min"]), int(row["col_min"]), int(row["row_max"]), int(row["col_max"])): row for row in old_records}
    records: list[dict[str, object]] = []
    points_list: list[np.ndarray] = []
    for class_id, _, _ in base.CLASSES:
        for points in continuous.connected_components(label == class_id):
            row = record_by_key.get(component_key(class_id, points))
            if row is None:
                raise ValueError("Could not reconstruct an existing ROI record")
            records.append({"roi_id": int(row["roi_id"]), "class_id": class_id, "class_name": row["class_name"], "pixels": int(row["pixels"]), "row_min": int(row["row_min"]), "col_min": int(row["col_min"]), "row_max": int(row["row_max"]), "col_max": int(row["col_max"]), "component_score": float(row["component_score"]), "confidence": row["confidence"]})
            points_list.append(points)
    next_id = max(record["roi_id"] for record in records) + 1
    for class_id, score, points in additions:
        row0, col0 = points.min(axis=0); row1, col1 = points.max(axis=0)
        name = next(name for cid, name, _ in base.CLASSES if cid == class_id)
        records.append({"roi_id": next_id, "class_id": class_id, "class_name": name, "pixels": int(len(points)), "row_min": int(row0), "col_min": int(col0), "row_max": int(row1), "col_max": int(col1), "component_score": score, "confidence": "feature_safe_expansion"})
        points_list.append(points)
        next_id += 1
    base.write_mat_v5(OUTPUT / "label.mat", "label", expanded.astype(np.uint8))
    with rasterio.open(OUTPUT / "label.tif", "w", **profile) as target:
        target.write(expanded.astype(np.uint8), 1)
    base.OUTPUT = OUTPUT
    base.write_envi_label(expanded.astype(np.uint8), transform, crs)
    geometries = [continuous.component_geometry(points, transform) for points in points_list]
    continuous.write_shapefile(OUTPUT / "label_rois", records, geometries, crs)
    with (OUTPUT / "roi_components.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": record, "geometry": geometry} for record, geometry in zip(records, geometries)]}
    (OUTPUT / "label_rois.geojson").write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    continuous.write_previews(expanded.astype(np.uint8), records)
    final = {pair: {name: snr(z, expanded, *pair, columns) for name, columns in SPACES.items()} for pair in baseline}
    summary = [(class_id, int(len(points)), float(score), int(points[:, 0].min()), int(points[:, 1].min())) for class_id, score, points in additions]
    (OUTPUT / "feature_safe_expansion_report.txt").write_text(
        "Feature-safe vegetation expansion\n\n"
        f"snr_floor_fraction={FLOOR}\nbaseline_snr={baseline}\nfinal_snr={final}\n"
        f"added_components=(class_id, pixels, homogeneity_score, row_min, col_min)\n{summary}\n"
        f"pixels_by_class={{cid: int((expanded == cid).sum()) for cid, _, _ in base.CLASSES}}\n", encoding="utf-8")
    print("additions", summary)
    print("baseline", baseline)
    print("final", final)
    print("pixels_by_class", {cid: int((expanded == cid).sum()) for cid, _, _ in base.CLASSES})


if __name__ == "__main__":
    main()
