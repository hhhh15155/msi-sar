from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.dfinet.dataset import load_mat_array
from baselines.dfinet.io import load_yaml, resolve_path


def load_dataset_config(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = Path("configs") / "datasets" / f"{name_or_path}.yaml"
    return load_yaml(resolve_path(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a MAT-format MSI-SAR dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name or dataset YAML path.")
    args = parser.parse_args()

    config = load_dataset_config(args.dataset)
    dataset_dir = resolve_path(config["path"])
    data_path = dataset_dir / config.get("data_file", "data.mat")
    label_path = dataset_dir / config.get("label_file", "label.mat")
    expected_channels = int(config["num_channels"])

    image = load_mat_array(data_path, config.get("data_key"), expected_channels=expected_channels)
    label = load_mat_array(label_path, config.get("label_key"))
    label = np.squeeze(label)

    print(f"dataset: {config['name']}")
    print(f"data: {data_path}")
    print(f"label: {label_path}")
    print(f"image shape: {image.shape}, dtype: {image.dtype}")
    print(f"label shape: {label.shape}, dtype: {label.dtype}")
    print(f"image min/max: {np.nanmin(image):.6f} / {np.nanmax(image):.6f}")
    print(f"image nan count: {int(np.isnan(image).sum()) if np.issubdtype(image.dtype, np.floating) else 0}")

    values, counts = np.unique(label, return_counts=True)
    class_names = config.get("class_names", [])
    print("label counts:")
    for value, count in zip(values.tolist(), counts.tolist()):
        value_int = int(value)
        name = "Undefined" if value_int == 0 else class_names[value_int - 1]
        print(f"  {value_int}: {name}: {count}")

    if image.ndim != 3:
        raise SystemExit(f"Expected 3D image cube, got {image.shape}")
    if image.shape[-1] != expected_channels:
        raise SystemExit(f"Expected {expected_channels} channels, got {image.shape[-1]}")
    if label.ndim != 2:
        raise SystemExit(f"Expected 2D label map, got {label.shape}")
    if image.shape[:2] != label.shape:
        raise SystemExit(f"Image and label shapes do not match: {image.shape[:2]} vs {label.shape}")

    expected_max = int(config["num_classes"])
    if int(np.nanmin(label)) < 0 or int(np.nanmax(label)) > expected_max:
        raise SystemExit(f"Expected labels in 0-{expected_max}, got {np.nanmin(label)}-{np.nanmax(label)}")

    print("dataset check: OK")


if __name__ == "__main__":
    main()
