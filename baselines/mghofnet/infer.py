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


def _windows(image: np.ndarray, patch_size: int):
    for x in range(0, image.shape[0] - patch_size + 1):
        for y in range(0, image.shape[1] - patch_size + 1):
            yield image[x : x + patch_size, y : y + patch_size], x, y, patch_size, patch_size


def _batches(size: int, iterable):
    iterator = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


def infer_full_image(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    num_classes: int,
    hsi_channels: int,
    aux_channels: int,
    device: torch.device,
    batch_size: int = 64,
    pad_mode: str = "constant",
) -> np.ndarray:
    model.eval()
    pad_size = patch_size // 2
    image_h, image_w = image.shape[:2]
    padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode=pad_mode)
    probabilities = np.zeros(padded.shape[:2] + (num_classes,), dtype=np.float32)
    total = (padded.shape[0] - patch_size + 1) * (padded.shape[1] - patch_size + 1)

    for batch in tqdm(_batches(batch_size, _windows(padded, patch_size)), total=(total + batch_size - 1) // batch_size, desc="inference"):
        with torch.no_grad():
            patches = np.copy([item[0] for item in batch])
            hsi = patches[:, :, :, :hsi_channels].transpose((0, 3, 1, 2))
            aux = patches[:, :, :, hsi_channels : hsi_channels + aux_channels].transpose((0, 3, 1, 2))
            hsi_tensor = torch.from_numpy(hsi).to(device)
            aux_tensor = torch.from_numpy(aux).to(device)
            output = model(hsi_tensor, aux_tensor).detach().cpu().numpy()
            for (x, y, width, height), out in zip([item[1:] for item in batch], output):
                probabilities[x + width // 2, y + height // 2] += out

    return probabilities[pad_size : image_h + pad_size, pad_size : image_w + pad_size, :]


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
