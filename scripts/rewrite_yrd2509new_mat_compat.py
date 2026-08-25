"""Rewrite YRD2509NEW MAT files in SciPy/MATLAB-compatible v5 format."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_yrd2509new as base  # noqa: E402


def rewrite(path: Path, expected_key: str) -> None:
    key, original = base.read_mat_v5(path)
    if key != expected_key:
        raise ValueError(f"Expected key {expected_key!r} in {path}, got {key!r}")
    temporary = path.with_name(path.stem + ".scipy_compatible.tmp.mat")
    savemat(temporary, {expected_key: original}, do_compression=True)
    verified = np.asarray(loadmat(temporary)[expected_key])
    if verified.dtype != original.dtype or verified.shape != original.shape:
        raise ValueError(f"Rewritten array metadata changed for {path}")
    if not np.array_equal(verified, original):
        raise ValueError(f"Rewritten array values changed for {path}")
    temporary.replace(path)
    print(f"rewrote {path}: shape={original.shape}, dtype={original.dtype}")


def main() -> None:
    dataset = ROOT / "data" / "yrd2509new"
    rewrite(dataset / "data.mat", "data")
    rewrite(dataset / "label.mat", "label")


if __name__ == "__main__":
    main()
