from __future__ import annotations

import itertools
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from .io import ensure_dir


DEFAULT_PALETTE = {
    "background": [0, 0, 0],
    "colors": [[62, 69, 191], [224, 190, 82], [158, 219, 74], [77, 214, 83], [63, 197, 153], [174, 205, 45], [181, 77, 218], [255, 250, 0]],
}


def _windows(image: np.ndarray, patch_size: int):
    for x in range(image.shape[0] - patch_size + 1):
        for y in range(image.shape[1] - patch_size + 1):
            yield image[x : x + patch_size, y : y + patch_size], x, y


def _batches(size: int, iterable):
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, size)):
        yield batch


def infer_full_image(model: torch.nn.Module, image: np.ndarray, config: dict, num_classes: int, device: torch.device) -> np.ndarray:
    model.eval()
    patch_size = int(config["patch_size"])
    ms_channels = int(config.get("ms_channels", 10))
    pad_size = patch_size // 2
    image_h, image_w = image.shape[:2]
    padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="reflect")
    probabilities = np.zeros(padded.shape[:2] + (num_classes,), dtype=np.float32)
    total = (padded.shape[0] - patch_size + 1) * (padded.shape[1] - patch_size + 1)
    batch_size = int(config.get("infer", {}).get("batch_size", config.get("batch_size", 64)))
    for batch in tqdm(_batches(batch_size, _windows(padded, patch_size)), total=(total + batch_size - 1) // batch_size, desc="inference"):
        data = np.asarray([item[0] for item in batch]).transpose((0, 3, 1, 2)).copy()
        with torch.no_grad():
            tensor = torch.from_numpy(data).to(device)
            outputs = model(tensor[:, :ms_channels], tensor[:, ms_channels:]).cpu().numpy()
        for (_, x, y), output in zip(batch, outputs):
            probabilities[x + pad_size, y + pad_size] += output
    return probabilities[pad_size : image_h + pad_size, pad_size : image_w + pad_size]


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
