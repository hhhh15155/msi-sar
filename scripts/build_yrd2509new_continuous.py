"""Rebuild YRD2509NEW using complete, continuous interior components.

Unlike the first candidate builder, this script never samples isolated squares.
It preserves a selected component in its entirety after a four-pixel inward
buffer, which makes the output suitable for visual polygon editing in ENVI.
"""

from __future__ import annotations

import csv
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from rasterio.features import rasterize, shapes
from rasterio.warp import transform_geom

import build_yrd2509new as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "yrd2509_landuse_9c"
LANDUSE_VECTOR = ROOT / "data" / "土地利用" / "landuse_2025new.shp"
OUTPUT = ROOT / "data" / "yrd2509new"
BUFFER = 4  # 40 m, followed by retention of the entire connected core component
LOCAL_WINDOW = 5

# Minimum useful area and maximum number of whole components.  The limits are
# deliberately class-specific because facilities/bare soil are genuinely small
# and fragmented in this scene; they remain review candidates, not gold truth.
TARGETS = {
    1: (500, 2), 2: (20000, 3), 3: (6000, 1), 4: (15000, 18), 5: (6000, 5),
    6: (15000, 25), 7: (200, 1), 8: (8000, 6), 9: (6000, 2),
}

# Only these four vector attributes are accepted as vegetation truth.  They are
# intersected with the aligned 9-class raster before component selection, so a
# polygon alone cannot introduce a geometrically shifted label.
VECTOR_VEGETATION = {"柽柳林": 2, "碱蓬": 4, "芦苇": 6, "天然柳林": 8}


def read_dbf_classes(path: Path) -> list[str]:
    """Read the first character field from the supplied DBF without GDAL."""
    raw = path.with_suffix(".dbf").read_bytes()
    count, header_length, record_length = struct.unpack_from("<IHH", raw, 4)
    fields: list[tuple[str, int]] = []
    offset = 32
    while raw[offset] != 13:
        name = raw[offset:offset + 11].split(b"\0")[0].decode("gbk")
        fields.append((name, raw[offset + 16]))
        offset += 32
    field_offset = 1
    class_offset = None
    for name, width in fields:
        if name == "地物类":
            class_offset = (field_offset, width)
            break
        field_offset += width
    if class_offset is None:
        raise ValueError("The land-use DBF does not contain a 地物类 field")
    start, width = class_offset
    return [raw[header_length + i * record_length + start:header_length + i * record_length + start + width].decode("gbk").strip() for i in range(count)]


def read_shapefile_geometries(path: Path, destination_crs) -> list[dict[str, object]]:
    """Read polygon rings and reproject them with rasterio/PROJ."""
    source_crs = CRS.from_wkt(path.with_suffix(".prj").read_text(encoding="utf-8"))
    raw = path.read_bytes()
    geometries: list[dict[str, object]] = []
    offset = 100
    while offset < len(raw):
        _, content_words = struct.unpack_from(">2i", raw, offset)
        content = offset + 8
        shape_type = struct.unpack_from("<i", raw, content)[0]
        if shape_type != 5:
            raise ValueError(f"Expected polygon shape type 5, found {shape_type}")
        part_count, point_count = struct.unpack_from("<2i", raw, content + 36)
        starts = struct.unpack_from("<" + "i" * part_count, raw, content + 44)
        point_offset = content + 44 + 4 * part_count
        points = [struct.unpack_from("<2d", raw, point_offset + 16 * i) for i in range(point_count)]
        ends = list(starts[1:]) + [point_count]
        # The source contains multipart land-use regions.  Treat every part as
        # a polygon; the subsequent raster-agreement test removes overlap risk.
        geometry = {"type": "MultiPolygon", "coordinates": [[points[a:b]] for a, b in zip(starts, ends)]}
        geometries.append(transform_geom(source_crs, destination_crs, geometry, precision=6))
        offset += 8 + content_words * 2
    return geometries


def vector_agreed_prior(source_label: np.ndarray, transform, crs) -> tuple[np.ndarray, dict[int, int]]:
    """Restrict vegetation candidates to vector/raster agreement pixels."""
    class_names = read_dbf_classes(LANDUSE_VECTOR)
    geometries = read_shapefile_geometries(LANDUSE_VECTOR, crs)
    if len(class_names) != len(geometries):
        raise ValueError("DBF and SHP record counts differ")
    vector_shapes = [(geometry, VECTOR_VEGETATION[name]) for name, geometry in zip(class_names, geometries) if name in VECTOR_VEGETATION]
    vector_label = rasterize(vector_shapes, out_shape=source_label.shape, transform=transform, fill=0, dtype="uint8")
    prior = source_label.copy()
    agreement_counts: dict[int, int] = {}
    for class_id in VECTOR_VEGETATION.values():
        agreed = (source_label == class_id) & (vector_label == class_id)
        prior[source_label == class_id] = 0
        prior[agreed] = class_id
        agreement_counts[class_id] = int(agreed.sum())
    return prior, agreement_counts


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for row, col in zip(*np.where(mask)):
        if visited[row, col]:
            continue
        visited[row, col] = True
        stack = [(int(row), int(col))]
        pixels: list[tuple[int, int]] = []
        while stack:
            current_row, current_col = stack.pop()
            pixels.append((current_row, current_col))
            for delta_row in (-1, 0, 1):
                for delta_col in (-1, 0, 1):
                    if delta_row == 0 and delta_col == 0:
                        continue
                    next_row, next_col = current_row + delta_row, current_col + delta_col
                    if (
                        0 <= next_row < height and 0 <= next_col < width
                        and mask[next_row, next_col] and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
        components.append(np.asarray(pixels, dtype=np.int32))
    return components


def component_geometry(points: np.ndarray, transform) -> dict[str, object]:
    row0, col0 = points.min(axis=0)
    row1, col1 = points.max(axis=0) + 1
    mask = np.zeros((row1 - row0, col1 - col0), dtype=np.uint8)
    mask[points[:, 0] - row0, points[:, 1] - col0] = 1
    local_transform = transform * Affine.translation(int(col0), int(row0))
    geometry, _ = next(shapes(mask, mask=mask.astype(bool), transform=local_transform, connectivity=8))
    return geometry


def choose_components(data: np.ndarray, source_label: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    features = base.robust_features(data)
    local_variance = base.local_feature_variance(features, LOCAL_WINDOW)
    output = np.zeros(source_label.shape, dtype=np.uint8)
    records: list[dict[str, object]] = []
    component_points: list[np.ndarray] = []
    roi_id = 1
    for class_id, class_name, _ in base.CLASSES:
        core = base.erode_8_connected(source_label == class_id, BUFFER)
        if not core.any():
            continue
        prototype = np.median(features[core], axis=0)
        distance = np.mean((features - prototype) ** 2, axis=-1)
        distance_scale = max(float(np.median(distance[core])), 1e-6)
        variance_scale = max(float(np.median(local_variance[core])), 1e-6)
        ranked: list[tuple[float, np.ndarray]] = []
        for pixels in connected_components(core):
            if len(pixels) < 100:
                continue
            score = float(
                distance[pixels[:, 0], pixels[:, 1]].mean() / distance_scale
                + local_variance[pixels[:, 0], pixels[:, 1]].mean() / variance_scale
            )
            ranked.append((score, pixels))
        ranked.sort(key=lambda item: item[0])
        target_pixels, maximum_components = TARGETS[class_id]
        retained = 0
        for score, pixels in ranked[:maximum_components]:
            output[pixels[:, 0], pixels[:, 1]] = class_id
            row0, col0 = pixels.min(axis=0)
            row1, col1 = pixels.max(axis=0)
            records.append(
                {
                    "roi_id": roi_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "pixels": int(len(pixels)),
                    "row_min": int(row0), "col_min": int(col0),
                    "row_max": int(row1), "col_max": int(col1),
                    "component_score": score,
                    "confidence": "vector_raster_agree" if class_id in VECTOR_VEGETATION.values() else "raster_continuous",
                }
            )
            component_points.append(pixels)
            roi_id += 1
            retained += len(pixels)
            if retained >= target_pixels:
                break
    return output, records, component_points


def geometry_rings(geometry: dict[str, object]) -> list[list[tuple[float, float]]]:
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        raise ValueError(f"Unexpected geometry type: {geometry['type']}")
    return [[(float(x), float(y)) for x, y in ring] for polygon in polygons for ring in polygon]


def write_shapefile(path: Path, records: list[dict[str, object]], geometries: list[dict[str, object]], crs) -> None:
    contents: list[bytes] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for geometry in geometries:
        rings = geometry_rings(geometry)
        offsets, points = [], []
        for ring in rings:
            offsets.append(len(points)); points.extend(ring)
        xs, ys = zip(*points); all_x.extend(xs); all_y.extend(ys)
        body = struct.pack("<i4d2i", 5, min(xs), min(ys), max(xs), max(ys), len(rings), len(points))
        body += struct.pack("<" + "i" * len(offsets), *offsets)
        body += b"".join(struct.pack("<2d", x, y) for x, y in points)
        contents.append(body)
    bounds = (min(all_x), min(all_y), max(all_x), max(all_y))
    file_words = (100 + sum(8 + len(body) for body in contents)) // 2
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, file_words) + struct.pack("<2i4d4d", 1000, 5, *bounds, 0.0, 0.0, 0.0, 0.0)
    with path.with_suffix(".shp").open("wb") as file:
        file.write(header)
        for number, body in enumerate(contents, start=1):
            file.write(struct.pack(">2i", number, len(body) // 2)); file.write(body)
    offset = 50
    with path.with_suffix(".shx").open("wb") as file:
        file.write(struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + 8 * len(contents)) // 2) + struct.pack("<2i4d4d", 1000, 5, *bounds, 0.0, 0.0, 0.0, 0.0))
        for body in contents:
            file.write(struct.pack(">2i", offset, len(body) // 2)); offset += 4 + len(body) // 2
    fields = [("ROI_ID", "N", 8, 0), ("CLASS_ID", "N", 4, 0), ("CLASS_NAME", "C", 32, 0), ("CONF", "C", 24, 0), ("SCORE", "N", 12, 4), ("PIXELS", "N", 10, 0)]
    header_length, record_length = 32 + 32 * len(fields) + 1, 1 + sum(field[2] for field in fields)
    with path.with_suffix(".dbf").open("wb") as file:
        file.write(struct.pack("<BBBBIHH20x", 3, 126, 7, 28, len(records), header_length, record_length))
        for name, kind, width, decimals in fields:
            file.write(name.encode("ascii")[:11].ljust(11, b"\0") + kind.encode("ascii") + b"\0" * 4 + bytes((width, decimals)) + b"\0" * 14)
        file.write(b"\r")
        for record in records:
            values = [f"{record['roi_id']:>8}", f"{record['class_id']:>4}", str(record['class_name'])[:32].ljust(32), str(record['confidence'])[:24].ljust(24), f"{record['component_score']:12.4f}", f"{record['pixels']:>10}"]
            file.write(b" " + "".join(values).encode("ascii"))
        file.write(b"\x1a")
    path.with_suffix(".prj").write_text(crs.to_wkt(), encoding="utf-8")


def write_previews(label: np.ndarray, records: list[dict[str, object]]) -> None:
    palette = np.zeros((10, 3), dtype=np.uint8)
    for class_id, _, rgb in base.CLASSES:
        palette[class_id] = rgb
    color = palette[label]
    truecolor = np.asarray(Image.open(SOURCE / "YRD_12ch_1024_truecolor_B4_B3_B2.png").convert("RGB"))
    overlay = (truecolor.astype(np.float32) * 0.55 + color.astype(np.float32) * 0.45).astype(np.uint8)
    overlay[label == 0] = truecolor[label == 0]
    Image.fromarray(color).save(OUTPUT / "label_color.png")
    Image.fromarray(overlay).save(OUTPUT / "truecolor_label_overlay.png")
    review = Image.fromarray(truecolor).convert("RGB")
    draw = ImageDraw.Draw(review)
    for record in records:
        draw.text((record["col_min"], record["row_min"]), str(record["roi_id"]), fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    review.save(OUTPUT / "roi_review.png")


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    _, data = base.read_mat_v5(SOURCE / "data.mat")
    with rasterio.open(SOURCE / "label_landuse_9c_1024.tif") as source:
        source_label = source.read(1).astype(np.uint8)
        profile, transform, crs = source.profile.copy(), source.transform, source.crs
    source_label, agreement_counts = vector_agreed_prior(source_label, transform, crs)
    label, records, points = choose_components(data.astype(np.float32, copy=False), source_label)
    geometries = [component_geometry(component, transform) for component in points]
    shutil.copy2(SOURCE / "data.mat", OUTPUT / "data.mat")
    base.write_mat_v5(OUTPUT / "label.mat", "label", label)
    profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    with rasterio.open(OUTPUT / "label.tif", "w", **profile) as destination:
        destination.write(label, 1)
    base.OUTPUT = OUTPUT
    base.write_envi_label(label, transform, crs)
    write_shapefile(OUTPUT / "label_rois", records, geometries, crs)
    with (OUTPUT / "roi_components.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (OUTPUT / "class_mapping.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["label", "class_name_en", "rgb"])
        writer.writerows((class_id, name, ",".join(map(str, rgb))) for class_id, name, rgb in base.CLASSES)
    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": record, "geometry": geometry} for record, geometry in zip(records, geometries)]}
    (OUTPUT / "label_rois.geojson").write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    output_idl = str(OUTPUT).replace("\\", "/")
    (OUTPUT / "create_label_xml_in_envi.pro").write_text(
        "PRO create_yrd2509new_label_xml\n  COMPILE_OPT idl2\n  e = ENVI()\n"
        f"  root = '{output_idl}'\n  vector = e.OpenVector(FILEPATH('label_rois.shp', ROOT_DIR=root))\n"
        "  task = ENVITask('VectorAttributeToROIs')\n  task.INPUT_VECTOR = vector\n"
        "  task.ATTRIBUTE_NAME = 'CLASS_NAME'\n  task.OUTPUT_ROI_URI = FILEPATH('label.xml', ROOT_DIR=root)\n"
        "  task.Execute\n  PRINT, 'Created: ' + task.OUTPUT_ROI_URI\nEND\n", encoding="utf-8")
    write_previews(label, records)
    counts = {class_id: int((label == class_id).sum()) for class_id, _, _ in base.CLASSES}
    component_counts = {class_id: sum(record["class_id"] == class_id for record in records) for class_id, _, _ in base.CLASSES}
    (OUTPUT / "README.txt").write_text(
        "YRD2509NEW: CONTINUOUS CANDIDATE COMPONENTS\n\n"
        "Labels are complete 8-connected components of the 40 m-inset land-use prior, not sampled tiles. Components are ranked by class-consensus optical/SAR/index distance and 5x5 local homogeneity, then retained whole.\n"
        "The supplied complete landuse_2025new.shp/.dbf/.shx/.prj is reprojected from CGCS2000 GK (120E) to the raster's UTM 50N grid. Vegetation candidates are retained only where the vector attribute and the aligned 9-class raster agree: 柽柳林->Tamarix, 碱蓬->Suaeda, 芦苇->Reed, 天然柳林->Willow forest.\n"
        "All regions are candidate_continuous for ENVI review, not independently field-verified gold labels.\n\n"
        "Open label_rois.shp in ENVI, revise polygons, then run create_label_xml_in_envi.pro to create native label.xml grouped by CLASS_NAME. label_envi.dat/.hdr is directly displayable in ENVI; 0 is ignore.\n\n"
        f"buffer_pixels={BUFFER}; local_window={LOCAL_WINDOW}\nvector_raster_agreement_pixels={agreement_counts}\ncomponents_by_class={component_counts}\npixels_by_class={counts}\n", encoding="utf-8")
    print(f"Created {OUTPUT}")
    print("components_by_class", component_counts)
    print("pixels_by_class", counts)


if __name__ == "__main__":
    main()
