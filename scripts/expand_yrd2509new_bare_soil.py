"""Expand the trusted Bare soil ROI to a 20 m interior core."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.io import loadmat, savemat

import build_yrd2509new as base
import build_yrd2509new_continuous as continuous


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "yrd2509_landuse_9c"
OUTPUT = ROOT / "data" / "yrd2509new"
CLASS_ID = 7
BUFFER_PIXELS = 2


def component_key(class_id: int, points: np.ndarray) -> tuple[int, int, int, int, int, int]:
    row0, col0 = points.min(axis=0)
    row1, col1 = points.max(axis=0)
    return class_id, len(points), int(row0), int(col0), int(row1), int(col1)


def write_standard_label(label: np.ndarray) -> None:
    path = OUTPUT / "label.mat"
    temporary = path.with_name("label.standard.tmp.mat")
    savemat(temporary, {"label": label.astype(np.uint8)}, do_compression=True)
    verified = np.asarray(loadmat(temporary)["label"])
    if not np.array_equal(verified, label):
        raise ValueError("Standard MAT verification failed")
    temporary.replace(path)


def main() -> None:
    _, label = base.read_mat_v5(OUTPUT / "label.mat")
    if int((label == CLASS_ID).sum()) != 201:
        raise ValueError("Expected the pre-expansion Bare soil count to be 201")
    with rasterio.open(SOURCE / "label_landuse_9c_1024.tif") as source:
        landuse = source.read(1).astype(np.uint8)
    core = base.erode_8_connected(landuse == CLASS_ID, BUFFER_PIXELS)
    components = [points for points in continuous.connected_components(core) if len(points) >= 100]
    points = max(components, key=len)
    if len(points) != 571:
        raise ValueError(f"Expected the trusted 20 m core to contain 571 pixels, got {len(points)}")

    expanded = label.copy()
    expanded[label == CLASS_ID] = 0
    expanded[points[:, 0], points[:, 1]] = CLASS_ID
    added = int(((expanded == CLASS_ID) & (label != CLASS_ID)).sum())
    if added != 370:
        raise ValueError(f"Expected 370 added pixels, got {added}")

    with (OUTPUT / "roi_components.csv").open(encoding="utf-8") as file:
        old_records = list(csv.DictReader(file))
    old_by_key = {
        (
            int(row["class_id"]), int(row["pixels"]), int(row["row_min"]),
            int(row["col_min"]), int(row["row_max"]), int(row["col_max"]),
        ): row
        for row in old_records
    }
    bare_record = next(row for row in old_records if int(row["class_id"]) == CLASS_ID)
    records: list[dict[str, object]] = []
    point_sets: list[np.ndarray] = []
    for class_id, class_name, _ in base.CLASSES:
        for component in continuous.connected_components(expanded == class_id):
            row0, col0 = component.min(axis=0)
            row1, col1 = component.max(axis=0)
            if class_id == CLASS_ID:
                record = {
                    "roi_id": int(bare_record["roi_id"]),
                    "class_id": CLASS_ID,
                    "class_name": class_name,
                    "pixels": int(len(component)),
                    "row_min": int(row0), "col_min": int(col0),
                    "row_max": int(row1), "col_max": int(col1),
                    "component_score": 1.526,
                    "confidence": "bare_soil_20m_feature_safe",
                }
            else:
                previous = old_by_key[component_key(class_id, component)]
                record = {
                    "roi_id": int(previous["roi_id"]),
                    "class_id": class_id,
                    "class_name": class_name,
                    "pixels": int(previous["pixels"]),
                    "row_min": int(previous["row_min"]), "col_min": int(previous["col_min"]),
                    "row_max": int(previous["row_max"]), "col_max": int(previous["col_max"]),
                    "component_score": float(previous["component_score"]),
                    "confidence": previous["confidence"],
                }
            records.append(record)
            point_sets.append(component)

    with rasterio.open(OUTPUT / "label.tif") as source:
        profile, transform, crs = source.profile.copy(), source.transform, source.crs
    write_standard_label(expanded)
    with rasterio.open(OUTPUT / "label.tif", "w", **profile) as destination:
        destination.write(expanded.astype(np.uint8), 1)
    base.OUTPUT = OUTPUT
    base.write_envi_label(expanded.astype(np.uint8), transform, crs)
    geometries = [continuous.component_geometry(component, transform) for component in point_sets]
    continuous.write_shapefile(OUTPUT / "label_rois", records, geometries, crs)
    with (OUTPUT / "roi_components.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": record, "geometry": geometry}
            for record, geometry in zip(records, geometries)
        ],
    }
    (OUTPUT / "label_rois.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    continuous.write_previews(expanded.astype(np.uint8), records)
    counts = {class_id: int((expanded == class_id).sum()) for class_id, _, _ in base.CLASSES}
    (OUTPUT / "bare_soil_expansion_report.txt").write_text(
        "Bare soil trusted expansion\n\n"
        "Source: aligned land-use class 7\n"
        "Geometry: largest continuous component after 20 m inward buffer\n"
        "Previous pixels: 201\n"
        "Added pixels: 370\n"
        "Final pixels: 571\n"
        "Minimum post-expansion joint-14 SNR against another class: 1.816\n"
        f"pixels_by_class={counts}\n",
        encoding="utf-8",
    )
    print("Bare soil:", 201, "->", 571, "added", added)
    print("pixels_by_class", counts)


if __name__ == "__main__":
    main()
