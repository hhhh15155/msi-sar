"""Synchronise YRD2509NEW metadata/ROI names with the original YRD taxonomy.

Pixel IDs are intentionally unchanged: YRD2509NEW has nine classes and
splits water into rivers and ponds, whereas YRD has eight classes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import rasterio

import build_yrd2509new as base
import build_yrd2509new_continuous as continuous


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "yrd2509new"
ALIGNMENT = {
    1: ("Artificial facilities", "No YRD equivalent"),
    2: ("Tamarix chinensis", "YRD class 4"),
    3: ("River water", "YRD class 1: Waterbody"),
    4: ("Suaeda salsa", "YRD class 5"),
    5: ("Pond water", "YRD class 1: Waterbody"),
    6: ("Phragmites communis", "YRD class 3"),
    7: ("Bare soil", "No YRD equivalent"),
    8: ("Natural willow forest", "YRD class 7"),
    9: ("Tidal flat", "YRD class 6"),
}


def key(class_id, points):
    row0, col0 = points.min(axis=0); row1, col1 = points.max(axis=0)
    return class_id, len(points), int(row0), int(col0), int(row1), int(col1)


def main() -> None:
    _, label = base.read_mat_v5(OUTPUT / "label.mat")
    with (OUTPUT / "roi_components.csv").open(encoding="utf-8") as file:
        old = list(csv.DictReader(file))
    records_by_key = {(int(row["class_id"]), int(row["pixels"]), int(row["row_min"]), int(row["col_min"]), int(row["row_max"]), int(row["col_max"])): row for row in old}
    records, points_list = [], []
    for class_id, _, _ in base.CLASSES:
        for points in continuous.connected_components(label == class_id):
            row = records_by_key[key(class_id, points)]
            records.append({"roi_id": int(row["roi_id"]), "class_id": class_id, "class_name": ALIGNMENT[class_id][0], "pixels": int(row["pixels"]), "row_min": int(row["row_min"]), "col_min": int(row["col_min"]), "row_max": int(row["row_max"]), "col_max": int(row["col_max"]), "component_score": float(row["component_score"]), "confidence": row["confidence"]})
            points_list.append(points)
    with rasterio.open(OUTPUT / "label.tif") as source:
        transform, crs = source.transform, source.crs
    base.OUTPUT = OUTPUT
    base.write_envi_label(label, transform, crs)
    geometries = [continuous.component_geometry(points, transform) for points in points_list]
    continuous.write_shapefile(OUTPUT / "label_rois", records, geometries, crs)
    with (OUTPUT / "roi_components.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": record, "geometry": geometry} for record, geometry in zip(records, geometries)]}
    (OUTPUT / "label_rois.geojson").write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT / "class_mapping.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["label", "class_name_en", "rgb"])
        writer.writerows((class_id, name, ",".join(map(str, rgb))) for class_id, name, rgb in base.CLASSES)
    with (OUTPUT / "yrd_taxonomy_alignment.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["yrd2509new_label", "yrd2509new_name", "yrd_alignment"])
        writer.writerows((class_id, *ALIGNMENT[class_id]) for class_id, _, _ in base.CLASSES)
    print("Updated class names for", len(records), "ROIs")


if __name__ == "__main__":
    main()
