"""Build a conservative, reviewable YRD2509 label subset.

The source land-use map was rasterised from landuse_2025new.shp before it was
placed in this repository.  The original Shapefile is not included here, so
this script treats its raster derivative as the land-use prior.  It does *not*
claim independent botanical ground truth: it selects compact, interior tiles
whose optical and SAR signatures are locally homogeneous and close to the
class-consensus signature.  Every selected tile is exported as an editable
Shapefile for a final ENVI/manual review.
"""

from __future__ import annotations

import csv
import json
import shutil
import struct
import zlib
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "yrd2509_landuse_9c"
OUTPUT = ROOT / "data" / "yrd2509new"
TILE_SIZE = 11  # 110 m x 110 m at the 10 m output grid
INTERIOR_BUFFER = 6  # retain only pixels >= 60 m inside a source land-use class
LOCAL_WINDOW = 5

CLASSES = [
    (1, "Artificial facilities", (214, 39, 40)),
    (2, "Tamarix chinensis", (27, 158, 119)),
    (3, "River water", (44, 127, 184)),
    (4, "Suaeda salsa", (230, 171, 2)),
    (5, "Pond water", (65, 182, 196)),
    (6, "Phragmites communis", (102, 166, 30)),
    (7, "Bare soil", (166, 118, 29)),
    (8, "Natural willow forest", (0, 104, 55)),
    (9, "Tidal flat", (194, 165, 97)),
]


def _tag(blob: bytes, offset: int) -> tuple[int, int, int, int]:
    first, size = struct.unpack_from("<II", blob, offset)
    if first >> 16:
        return first & 0xFFFF, first >> 16, offset + 4, offset + 8
    return first, size, offset + 8, offset + 8 + ((size + 7) // 8) * 8


def read_mat_v5(path: Path) -> tuple[str, np.ndarray]:
    blob = path.read_bytes()
    element_type, size, data_start, _ = _tag(blob, 128)
    if element_type != 15:
        raise ValueError(f"Expected a compressed MAT v5 array: {path}")
    raw = zlib.decompress(blob[data_start : data_start + size])
    matrix_type, _, pos, matrix_end = _tag(raw, 0)
    if matrix_type != 14:
        raise ValueError(f"Expected a MAT matrix: {path}")
    elements = []
    while pos < matrix_end:
        element = _tag(raw, pos)
        elements.append(element)
        pos = element[3]
    dims = np.frombuffer(raw[elements[1][2] : elements[1][2] + elements[1][1]], dtype="<i4")
    name = raw[elements[2][2] : elements[2][2] + elements[2][1]].decode("ascii")
    data_type, data_size, data_offset, _ = elements[3]
    dtypes = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 9: "f8"}
    array = np.frombuffer(raw[data_offset : data_offset + data_size], dtype="<" + dtypes[data_type])
    return name, array.reshape(tuple(dims), order="F").copy()


def _element(element_type: int, payload: bytes) -> bytes:
    return struct.pack("<II", element_type, len(payload)) + payload + (b"\0" * ((-len(payload)) % 8))


def write_mat_v5(path: Path, name: str, array: np.ndarray) -> None:
    array = np.asfortranarray(array)
    numeric = {np.dtype("uint8"): (2, 9), np.dtype("float32"): (7, 7)}
    data_type, mx_class = numeric[array.dtype]
    matrix = _element(
        14,
        b"".join(
            [
                _element(6, struct.pack("<II", mx_class, 0)),
                _element(5, np.asarray(array.shape, dtype="<i4").tobytes()),
                _element(1, name.encode("ascii")),
                _element(data_type, array.tobytes(order="F")),
            ]
        ),
    )
    header_text = b"MATLAB 5.0 MAT-file, Platform: Codex, Created by build_yrd2509new.py"
    header = header_text.ljust(116, b" ") + (b"\0" * 8) + b"\0\1IM"
    path.write_bytes(header + _element(15, zlib.compress(matrix, level=6)))


def erode_8_connected(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1)
        result = (
            padded[:-2, :-2] & padded[:-2, 1:-1] & padded[:-2, 2:]
            & padded[1:-1, :-2] & padded[1:-1, 1:-1] & padded[1:-1, 2:]
            & padded[2:, :-2] & padded[2:, 1:-1] & padded[2:, 2:]
        )
    return result


def box_sum(image: np.ndarray, size: int) -> np.ndarray:
    """Sum each size x size window; output is indexed by its top-left pixel."""
    integral = np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]


def local_feature_variance(features: np.ndarray, size: int) -> np.ndarray:
    total = np.zeros(features.shape[:2], dtype=np.float32)
    for channel in range(features.shape[-1]):
        value = features[:, :, channel]
        mean = box_sum(value, size) / (size * size)
        mean2 = box_sum(value * value, size) / (size * size)
        variance = np.maximum(mean2 - mean * mean, 0.0)
        pad_before = size // 2
        pad_after = size - 1 - pad_before
        total += np.pad(variance, ((pad_before, pad_after), (pad_before, pad_after)), mode="edge")
    return total / features.shape[-1]


def robust_features(data: np.ndarray) -> np.ndarray:
    """Optical, SAR, and physically meaningful derived features on one robust scale."""
    optical = data[:, :, :10]
    vv, vh = data[:, :, 10], data[:, :, 11]
    eps = np.float32(1e-6)
    ndvi = (optical[:, :, 6] - optical[:, :, 2]) / (optical[:, :, 6] + optical[:, :, 2] + eps)
    ndre = (optical[:, :, 7] - optical[:, :, 3]) / (optical[:, :, 7] + optical[:, :, 3] + eps)
    ndmi = (optical[:, :, 6] - optical[:, :, 8]) / (optical[:, :, 6] + optical[:, :, 8] + eps)
    derived = np.stack((ndvi, ndre, ndmi, vv - vh), axis=-1)
    features = np.concatenate((data, derived), axis=-1).astype(np.float32, copy=False)
    median = np.median(features, axis=(0, 1))
    q25 = np.percentile(features, 25, axis=(0, 1))
    q75 = np.percentile(features, 75, axis=(0, 1))
    scale = np.maximum(q75 - q25, 1e-5)
    return np.clip((features - median) / scale, -8.0, 8.0)


def choose_tiles(data: np.ndarray, source_label: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    features = robust_features(data)
    local_var = local_feature_variance(features, LOCAL_WINDOW)
    selected_label = np.zeros(source_label.shape, dtype=np.uint8)
    records: list[dict[str, object]] = []
    roi_id = 1

    for class_id, class_name, _ in CLASSES:
        source_mask = source_label == class_id
        # Facilities and bare-soil polygons are fragmented at this resolution.
        # They remain review candidates, but their relaxed geometry is explicitly
        # recorded instead of pretending that they satisfy the strict core rule.
        buffers = (INTERIOR_BUFFER,) if class_id not in (1, 7) else (6, 5, 4, 3)
        # The two fragmented classes use 3x3 review tiles; retain enough
        # independent candidates for a 100-shot train/validation experiment.
        target = 40 if class_id in (2, 4) else (45 if class_id in (1, 7) else 25)
        selected: list[tuple[int, int, float]] = []
        final_buffer = INTERIOR_BUFFER
        final_tile_size = TILE_SIZE
        for buffer in buffers:
            interior = erode_8_connected(source_mask, buffer)
            if not interior.any():
                continue
            prototype = np.median(features[interior], axis=0)
            distance = np.mean((features - prototype) ** 2, axis=-1)
            # Scale the two quality terms inside each class, so no class is favoured
            # only because it has naturally higher SAR texture or reflectance variance.
            d_scale = max(float(np.median(distance[interior])), 1e-6)
            v_scale = max(float(np.median(local_var[interior])), 1e-6)
            quality = distance / d_scale + local_var / v_scale
            # Small/linear classes sometimes cannot contain a 110 m square.  Fall
            # back only when necessary and record the actual size for review.
            for tile_size in (TILE_SIZE, 9, 7, 5, 3):
                valid = box_sum(interior.astype(np.int16), tile_size) == tile_size * tile_size
                if not valid.any():
                    continue
                scores = box_sum(quality, tile_size) / (tile_size * tile_size)
                positions = np.argwhere(valid)
                order = np.argsort(scores[valid])
                blocked = np.zeros(source_label.shape, dtype=bool)
                selected = []
                for index in order:
                    row, col = map(int, positions[index])
                    if blocked[row : row + tile_size, col : col + tile_size].any():
                        continue
                    selected.append((row, col, float(scores[row, col])))
                    # A one-tile gap makes samples spatially independent enough for
                    # polygon-level train/test splits and easy manual inspection.
                    r0, r1 = max(0, row - tile_size), min(blocked.shape[0], row + 2 * tile_size)
                    c0, c1 = max(0, col - tile_size), min(blocked.shape[1], col + 2 * tile_size)
                    blocked[r0:r1, c0:c1] = True
                    if len(selected) >= target:
                        break
                final_buffer, final_tile_size = buffer, tile_size
                if len(selected) >= min(12, target):
                    break
            if len(selected) >= min(12, target):
                break

        for row, col, score in selected:
            selected_label[row : row + final_tile_size, col : col + final_tile_size] = class_id
            records.append(
                {
                    "roi_id": roi_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "row0": row,
                    "col0": col,
                    "tile_pixels": final_tile_size,
                    "pixel_count": final_tile_size * final_tile_size,
                    "interior_buffer_pixels": final_buffer,
                    "quality_score": score,
                    "confidence": "candidate_high" if final_buffer >= 6 and final_tile_size >= 7 else "candidate_review",
                }
            )
            roi_id += 1
    return selected_label, records


def pixel_box_to_map(transform, row: int, col: int, size: int) -> list[tuple[float, float]]:
    corners = [
        transform * (col, row), transform * (col + size, row),
        transform * (col + size, row + size), transform * (col, row + size),
    ]
    return [(float(x), float(y)) for x, y in corners]


def write_shapefile(path: Path, records: list[dict[str, object]], transform, crs) -> None:
    """Write simple rectangular PolygonZ-free ESRI Shapefile records without a GIS dependency."""
    shapes: list[tuple[dict[str, object], list[tuple[float, float]]]] = []
    for record in records:
        ring = pixel_box_to_map(transform, int(record["row0"]), int(record["col0"]), int(record["tile_pixels"]))
        shapes.append((record, ring + [ring[0]]))
    all_x = [x for _, ring in shapes for x, _ in ring]
    all_y = [y for _, ring in shapes for _, y in ring]
    bounds = (min(all_x), min(all_y), max(all_x), max(all_y)) if shapes else (0.0, 0.0, 0.0, 0.0)

    contents = []
    for _, ring in shapes:
        xs, ys = zip(*ring)
        body = struct.pack("<i4d2i", 5, min(xs), min(ys), max(xs), max(ys), 1, len(ring))
        body += struct.pack("<i", 0) + b"".join(struct.pack("<2d", x, y) for x, y in ring)
        contents.append(body)
    file_words = (100 + sum(8 + len(body) for body in contents)) // 2
    header = struct.pack(
        ">7i", 9994, 0, 0, 0, 0, 0, file_words
    ) + struct.pack("<2i4d4d", 1000, 5, *bounds, 0.0, 0.0, 0.0, 0.0)
    with path.with_suffix(".shp").open("wb") as file:
        file.write(header)
        for number, body in enumerate(contents, start=1):
            file.write(struct.pack(">2i", number, len(body) // 2))
            file.write(body)
    offset = 50
    shx_words = (100 + 8 * len(contents)) // 2
    with path.with_suffix(".shx").open("wb") as file:
        file.write(struct.pack(">7i", 9994, 0, 0, 0, 0, 0, shx_words) + struct.pack("<2i4d4d", 1000, 5, *bounds, 0.0, 0.0, 0.0, 0.0))
        for body in contents:
            file.write(struct.pack(">2i", offset, len(body) // 2))
            offset += 4 + len(body) // 2
    fields = [("ROI_ID", "N", 8, 0), ("CLASS_ID", "N", 4, 0), ("CLASS_NAME", "C", 32, 0), ("CONF", "C", 16, 0), ("SCORE", "N", 12, 4)]
    header_length = 32 + 32 * len(fields) + 1
    record_length = 1 + sum(field[2] for field in fields)
    with path.with_suffix(".dbf").open("wb") as file:
        file.write(struct.pack("<BBBBIHH20x", 3, 126, 7, 28, len(records), header_length, record_length))
        for name, kind, width, decimals in fields:
            encoded = name.encode("ascii")[:11].ljust(11, b"\0")
            file.write(encoded + kind.encode("ascii") + b"\0" * 4 + bytes((width, decimals)) + b"\0" * 14)
        file.write(b"\r")
        for record in records:
            values = [
                f"{record['roi_id']:>8}", f"{record['class_id']:>4}", str(record["class_name"])[:32].ljust(32),
                str(record["confidence"])[:16].ljust(16), f"{record['quality_score']:12.4f}",
            ]
            file.write(b" " + "".join(values).encode("ascii"))
        file.write(b"\x1a")
    if crs:
        path.with_suffix(".prj").write_text(crs.to_wkt(), encoding="utf-8")


def write_envi_label(label: np.ndarray, transform, crs) -> None:
    (OUTPUT / "label_envi.dat").write_bytes(label.astype(np.uint8).tobytes(order="C"))
    x0, y0 = transform * (0, 0)
    pixel_x, pixel_y = abs(transform.a), abs(transform.e)
    names = ["Unclassified"] + [name for _, name, _ in CLASSES]
    lookup = [0, 0, 0] + [value for _, _, rgb in CLASSES for value in rgb]
    header = "\n".join(
        [
            "ENVI",
            "description = {YRD2509NEW conservative candidate labels; 0 is ignore/unreviewed}",
            f"samples = {label.shape[1]}", f"lines = {label.shape[0]}", "bands = 1",
            "header offset = 0", "file type = ENVI Classification", "data type = 1", "interleave = bsq", "byte order = 0",
            "data ignore value = 0",
            f"map info = {{UTM, 1.000, 1.000, {x0:.6f}, {y0:.6f}, {pixel_x:.6f}, {pixel_y:.6f}, 50, North, WGS-84, units=Meters}}",
            f"coordinate system string = {{{crs.to_wkt() if crs else ''}}}",
            "class names = {" + ", ".join(names) + "}",
            "class lookup = {" + ", ".join(map(str, lookup)) + "}",
        ]
    )
    (OUTPUT / "label_envi.hdr").write_text(header + "\n", encoding="utf-8")


def write_previews(label: np.ndarray, records: list[dict[str, object]]) -> None:
    palette = np.zeros((10, 3), dtype=np.uint8)
    for class_id, _, rgb in CLASSES:
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
        row, col, size = int(record["row0"]), int(record["col0"]), int(record["tile_pixels"])
        rgb = tuple(CLASSES[int(record["class_id"]) - 1][2])
        draw.rectangle((col, row, col + size - 1, row + size - 1), outline=rgb, width=2)
        draw.text((col + 1, row + 1), str(record["roi_id"]), fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    review.save(OUTPUT / "roi_review.png")


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    _, data = read_mat_v5(SOURCE / "data.mat")
    with rasterio.open(SOURCE / "label_landuse_9c_1024.tif") as source:
        source_label = source.read(1).astype(np.uint8)
        profile, transform, crs = source.profile.copy(), source.transform, source.crs
    label, records = choose_tiles(data.astype(np.float32, copy=False), source_label)
    if not records:
        raise ValueError("No review tiles were selected")
    shutil.copy2(SOURCE / "data.mat", OUTPUT / "data.mat")
    write_mat_v5(OUTPUT / "label.mat", "label", label)
    profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    with rasterio.open(OUTPUT / "label.tif", "w", **profile) as destination:
        destination.write(label, 1)
    write_envi_label(label, transform, crs)
    write_shapefile(OUTPUT / "label_rois", records, transform, crs)
    with (OUTPUT / "roi_tiles.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    with (OUTPUT / "class_mapping.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["label", "class_name_en", "rgb"])
        writer.writerows((class_id, name, ",".join(map(str, rgb))) for class_id, name, rgb in CLASSES)
    geojson = {"type": "FeatureCollection", "features": []}
    for record in records:
        ring = pixel_box_to_map(transform, int(record["row0"]), int(record["col0"]), int(record["tile_pixels"]))
        geojson["features"].append({"type": "Feature", "properties": record, "geometry": {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}})
    (OUTPUT / "label_rois.geojson").write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    # The native ENVI XML ROI serialization is written by ENVI itself.  This
    # small, documented IDL task groups our editable Shapefile records by
    # CLASS_NAME and emits the requested label.xml without guessing its
    # proprietary on-disk XML schema.
    output_idl = str(OUTPUT).replace("\\", "/")
    (OUTPUT / "create_label_xml_in_envi.pro").write_text(
        "PRO create_yrd2509new_label_xml\n"
        "  COMPILE_OPT idl2\n"
        "  e = ENVI()\n"
        f"  root = '{output_idl}'\n"
        "  vector = e.OpenVector(FILEPATH('label_rois.shp', ROOT_DIR=root))\n"
        "  task = ENVITask('VectorAttributeToROIs')\n"
        "  task.INPUT_VECTOR = vector\n"
        "  task.ATTRIBUTE_NAME = 'CLASS_NAME'\n"
        "  task.OUTPUT_ROI_URI = FILEPATH('label.xml', ROOT_DIR=root)\n"
        "  task.Execute\n"
        "  PRINT, 'Created: ' + task.OUTPUT_ROI_URI\n"
        "END\n",
        encoding="utf-8",
    )
    write_previews(label, records)
    counts = {class_id: int((label == class_id).sum()) for class_id, _, _ in CLASSES}
    tile_counts = {class_id: sum(record["class_id"] == class_id for record in records) for class_id, _, _ in CLASSES}
    (OUTPUT / "README.txt").write_text(
        "YRD2509NEW\n\n"
        "A conservative, reviewable subset derived from the rasterised landuse_2025new.shp prior.\n"
        "Each retained ROI is a 70-110 m interior tile selected using class-consensus optical/SAR/derived-index distance and 5x5 local feature homogeneity.\n"
        "This is candidate_high, not independently field-verified botanical truth. Review every tile in label_rois.shp (or label_rois.geojson) against contemporaneous high-resolution imagery before calling it gold.\n\n"
        "For ENVI editing: open the reference raster plus label_rois.shp, edit ROIs, then run create_label_xml_in_envi.pro to create native label.xml grouped by CLASS_NAME.\n"
        "The repository does not contain ENVI, so the native XML is deliberately emitted by ENVI's own VectorAttributeToROIs task instead of fabricating an unvalidated proprietary XML schema.\n"
        "label_envi.dat/.hdr is an ENVI Classification raster; label=0 is ignore/unreviewed.\n\n"
        f"tile_size_nominal={TILE_SIZE}; interior_buffer={INTERIOR_BUFFER}; local_window={LOCAL_WINDOW}\n"
        f"tiles_by_class={tile_counts}\n"
        f"pixels_by_class={counts}\n",
        encoding="utf-8",
    )
    print(f"Created {OUTPUT}")
    print("tiles_by_class", tile_counts)
    print("pixels_by_class", counts)


if __name__ == "__main__":
    main()
