"""Convert the NCGLF2 GRSS-DFC-2007 subset to the project MAT layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


OPTICAL_KEY = "HSI_data"
SAR_KEY = "SAR_data"
LABEL_KEY = "ground"


def prepare_dataset(source: Path, output_dir: Path) -> tuple[Path, Path]:
    source = Path(source)
    output_dir = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)

    source_data = loadmat(source)
    missing = [key for key in (OPTICAL_KEY, SAR_KEY, LABEL_KEY) if key not in source_data]
    if missing:
        raise KeyError(f"Missing required MATLAB variables: {', '.join(missing)}")

    optical = np.asarray(source_data[OPTICAL_KEY])
    sar = np.squeeze(np.asarray(source_data[SAR_KEY]))
    label = np.squeeze(np.asarray(source_data[LABEL_KEY]))

    if optical.ndim != 3 or optical.shape[-1] != 6:
        raise ValueError(f"Expected six-channel optical data, got {optical.shape}")
    if sar.ndim != 2 or label.ndim != 2:
        raise ValueError(f"Expected 2D SAR and label maps, got {sar.shape} and {label.shape}")
    if optical.shape[:2] != sar.shape or optical.shape[:2] != label.shape:
        raise ValueError(
            "Optical, SAR, and label spatial shapes must match: "
            f"{optical.shape[:2]}, {sar.shape}, {label.shape}"
        )
    if not np.issubdtype(label.dtype, np.integer):
        raise ValueError(f"Expected integer labels, got {label.dtype}")
    if int(label.min()) < 0 or int(label.max()) > 5:
        raise ValueError(f"Expected labels in 0-5, got {int(label.min())}-{int(label.max())}")

    data = np.concatenate(
        (optical.astype(np.float32, copy=False), sar.astype(np.float32, copy=False)[..., None]),
        axis=-1,
    )
    label = label.astype(np.uint8, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.mat"
    label_path = output_dir / "label.mat"
    savemat(data_path, {"data": data}, do_compression=True)
    savemat(label_path, {"label": label}, do_compression=True)
    return data_path, label_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare GRSS-DFC-2007 as a 6-MSI + 1-SAR project dataset.")
    parser.add_argument("--source", type=Path, required=True, help="Path to NCGLF2 GRSS07_SAR_MS.mat.")
    parser.add_argument("--output", type=Path, default=root / "data" / "grss07")
    args = parser.parse_args()

    data_path, label_path = prepare_dataset(args.source, args.output)
    print(f"data: {data_path}")
    print(f"label: {label_path}")


if __name__ == "__main__":
    main()
