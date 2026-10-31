from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .io import ensure_dir
from baselines.prediction import infer_full_image as _infer_full_image


DEFAULT_PALETTE = {
    "background": [0, 0, 0],
    "colors": [
        [62, 69, 191],
        [224, 190, 82],
        [158, 219, 74],
        [77, 214, 83],
        [63, 197, 153],
        [174, 205, 45],
        [181, 77, 218],
        [255, 250, 0],
    ],
}


def infer_full_image(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 64,
    pad_mode: str = "constant",
) -> np.ndarray:
    if pad_mode not in ('constant', 'reflect'):
        raise ValueError('Direct inference supports constant or reflect padding')
    config = {'model': 'dffnet', 'patch_size': patch_size, 'pad_mode': pad_mode}
    # This direct helper follows the requested padding; run evaluation uses its saved protocol.
    if pad_mode == 'constant':
        config['training_protocol'] = 'mghofnet-v1'
    return _infer_full_image(model, image, patch_size, num_classes, device,
                             batch_size=batch_size, config=config)


def colorize_label_map(label_map: np.ndarray, palette: dict | None = None) -> np.ndarray:
    palette = palette or DEFAULT_PALETTE
    colors = [palette.get("background", [0, 0, 0])] + palette.get("colors", DEFAULT_PALETTE["colors"])
    output = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    for value, color in enumerate(colors):
        output[label_map == value] = np.asarray(color, dtype=np.uint8)
    return output


def save_label_tif(label_map: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    imageio.imwrite(output_path, label_map.astype(np.uint8))


def save_label_png(label_map: np.ndarray, output_path: str | Path, palette: dict | None = None) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    imageio.imwrite(output_path, colorize_label_map(label_map, palette))


def save_label_outputs(label_map: np.ndarray, output_prefix: str | Path, palette: dict | None = None) -> list[Path]:
    output_prefix = Path(output_prefix)
    tif_path = output_prefix.with_suffix(".tif")
    png_path = output_prefix.with_suffix(".png")
    save_label_tif(label_map, tif_path)
    save_label_png(label_map, png_path, palette)
    return [tif_path, png_path]
