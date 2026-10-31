from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .io import ensure_dir
from baselines.prediction import infer_full_image as _infer_full_image


DEFAULT_PALETTE = {
    "background": [0, 0, 0],
    "colors": [[62, 69, 191], [224, 190, 82], [158, 219, 74], [77, 214, 83], [63, 197, 153], [174, 205, 45], [181, 77, 218], [255, 250, 0]],
}


def infer_full_image(model: torch.nn.Module, image: np.ndarray, config: dict, num_classes: int, device: torch.device) -> np.ndarray:
    return _infer_full_image(model, image, int(config['patch_size']), num_classes, device,
                             batch_size=int(config.get('infer', {}).get('batch_size', 64)), config=config)


def colorize_label_map(label_map: np.ndarray, palette: dict | None = None) -> np.ndarray:
    palette = palette or DEFAULT_PALETTE
    colors = [palette.get("background", [0, 0, 0])] + palette.get("colors", DEFAULT_PALETTE["colors"])
    output = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    for value, color in enumerate(colors):
        output[label_map == value] = np.asarray(color, dtype=np.uint8)
    return output


def save_label_outputs(label_map: np.ndarray, output_prefix: str | Path, palette: dict | None = None) -> list[Path]:
    prefix = Path(output_prefix)
    ensure_dir(prefix.parent)
    tif_path, png_path = prefix.with_suffix(".tif"), prefix.with_suffix(".png")
    imageio.imwrite(tif_path, label_map.astype(np.uint8))
    imageio.imwrite(png_path, colorize_label_map(label_map, palette))
    return [tif_path, png_path]
