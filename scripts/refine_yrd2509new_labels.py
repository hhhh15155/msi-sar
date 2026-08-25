"""Replace YRD2509NEW training labels with feature-separated continuous ROIs.

The complete candidate labels are preserved under *_candidate_full.*.  The
six removed ROIs remain available for ENVI review but are excluded from the
hard-supervision label because they overlap their principal confusion class in
the current 14-channel feature space.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio

import build_yrd2509new as base
import build_yrd2509new_continuous as continuous


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "yrd2509new"
REMOVE_ROI_IDS = {5, 20, 40, 42, 43, 44}


def component_key(class_id: int, points: np.ndarray) -> tuple[int, int, int, int, int, int]:
    row0, col0 = points.min(axis=0)
    row1, col1 = points.max(axis=0)
    return class_id, len(points), int(row0), int(col0), int(row1), int(col1)


def preserve_candidate_files() -> None:
    copies = {
        "label.mat": "label_candidate_full.mat", "label.tif": "label_candidate_full.tif",
        "label_envi.dat": "label_candidate_full_envi.dat", "label_envi.hdr": "label_candidate_full_envi.hdr",
        "roi_components.csv": "roi_components_candidate_full.csv", "label_rois.geojson": "label_rois_candidate_full.geojson",
    }
    for source, destination in copies.items():
        shutil.copy2(OUTPUT / source, OUTPUT / destination)
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        shutil.copy2(OUTPUT / f"label_rois{suffix}", OUTPUT / f"label_rois_candidate_full{suffix}")


def main() -> None:
    preserve_candidate_files()
    _, label = base.read_mat_v5(OUTPUT / "label.mat")
    with (OUTPUT / "roi_components.csv").open(encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    record_by_key = {
        (int(row["class_id"]), int(row["pixels"]), int(row["row_min"]), int(row["col_min"]), int(row["row_max"]), int(row["col_max"])): row
        for row in records
    }
    selected_records: list[dict[str, object]] = []
    selected_points: list[np.ndarray] = []
    refined = label.copy()
    for class_id, _, _ in base.CLASSES:
        for points in continuous.connected_components(label == class_id):
            record = record_by_key.get(component_key(class_id, points))
            if record is None:
                raise ValueError(f"Could not match a class-{class_id} ROI component to roi_components.csv")
            roi_id = int(record["roi_id"])
            if roi_id in REMOVE_ROI_IDS:
                refined[points[:, 0], points[:, 1]] = 0
                continue
            converted = {"roi_id": roi_id, "class_id": class_id, "class_name": record["class_name"], "pixels": int(record["pixels"]),
                         "row_min": int(record["row_min"]), "col_min": int(record["col_min"]), "row_max": int(record["row_max"]), "col_max": int(record["col_max"]),
                         "component_score": float(record["component_score"]), "confidence": "feature_separated" if class_id in {2, 4, 6, 8} else record["confidence"]}
            selected_records.append(converted)
            selected_points.append(points)
    with rasterio.open(OUTPUT / "label.tif") as source:
        profile, transform, crs = source.profile.copy(), source.transform, source.crs
    base.write_mat_v5(OUTPUT / "label.mat", "label", refined.astype(np.uint8))
    with rasterio.open(OUTPUT / "label.tif", "w", **profile) as target:
        target.write(refined.astype(np.uint8), 1)
    base.OUTPUT = OUTPUT
    base.write_envi_label(refined.astype(np.uint8), transform, crs)
    geometries = [continuous.component_geometry(points, transform) for points in selected_points]
    continuous.write_shapefile(OUTPUT / "label_rois", selected_records, geometries, crs)
    with (OUTPUT / "roi_components.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(selected_records[0])); writer.writeheader(); writer.writerows(selected_records)
    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": record, "geometry": geometry} for record, geometry in zip(selected_records, geometries)]}
    (OUTPUT / "label_rois.geojson").write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    continuous.write_previews(refined.astype(np.uint8), selected_records)
    counts = {class_id: int((refined == class_id).sum()) for class_id, _, _ in base.CLASSES}
    (OUTPUT / "feature_refinement_report.txt").write_text(
        "YRD2509NEW feature-separated training label\n\n"
        "Removed from hard supervision (retained in *_candidate_full.* for ENVI review):\n"
        "ROI 5 Tamarix: overlaps Suaeda in feature space\n"
        "ROI 20 Suaeda: overlaps Tamarix in feature space\n"
        "ROI 40 Reed: overlaps Willow in feature space\n"
        "ROI 42, 43, 44 Willow: overlap Reed in feature space\n\n"
        f"removed_roi_ids={sorted(REMOVE_ROI_IDS)}\npixels_by_class={counts}\n", encoding="utf-8")
    print("removed", sorted(REMOVE_ROI_IDS))
    print("pixels_by_class", counts)


if __name__ == "__main__":
    main()
