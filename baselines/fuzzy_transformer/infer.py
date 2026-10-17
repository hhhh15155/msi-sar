from __future__ import annotations

import itertools

import numpy as np
import torch
from tqdm import tqdm

from baselines.frekfuse.infer import save_label_outputs

__all__ = ["infer_full_image", "save_label_outputs"]


def infer_full_image(model: torch.nn.Module, image: np.ndarray, patch_size: int,
                     num_classes: int, device: torch.device, ms_channels: int,
                     sar_channels: int, batch_size: int = 128) -> np.ndarray:
    if patch_size <= 0 or patch_size % 2 == 0 or batch_size < 1:
        raise ValueError("Require a positive odd patch_size and a positive batch_size")
    if image.ndim != 3 or image.shape[-1] != ms_channels + sar_channels:
        raise ValueError("Expected [H, W, MSI_channels + SAR_channels] image")
    model.eval()
    pad = patch_size // 2
    height, width = image.shape[:2]
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    probabilities = np.empty((height, width, num_classes), dtype=np.float32)
    coordinates = iter(itertools.product(range(height), range(width)))
    with torch.no_grad():
        for _ in tqdm(range((height * width + batch_size - 1) // batch_size), desc="inference"):
            batch = list(itertools.islice(coordinates, batch_size))
            patches = np.stack([padded[x:x + patch_size, y:y + patch_size] for x, y in batch])
            images = torch.from_numpy(patches.transpose(0, 3, 1, 2).copy()).float().to(device)
            logits = model(ms=images[:, :ms_channels], sar=images[:, ms_channels:])["logits"]
            values = logits.softmax(dim=-1).cpu().numpy()
            for (x, y), value in zip(batch, values):
                probabilities[x, y] = value
    return probabilities
