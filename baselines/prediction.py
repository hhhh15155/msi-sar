"""One patch/padding path for test positions, labeled maps and explicit full maps."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PatchDataset
from .protocol import inference_pad_mode
from .training import forward_batch


def _predictions(model, image, gt, config, device, batch_size):
    dataset = PatchDataset(image, gt, int(config['patch_size']), data_aug=False,
                           pad_mode=inference_pad_mode(config))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            logits, _ = forward_batch(model, batch, device, config)
            scores = logits.detach().cpu().numpy()
            coords = dataset.indices[offset:offset + len(scores)] - dataset.pad_size
            yield coords, scores
            offset += len(scores)


def predict_labeled_samples(model, image, gt, config: dict, device) -> np.ndarray:
    batch_size = int(config.get('eval', {}).get('batch_size', config.get('test_batch_size', 1024)))
    result = np.full_like(gt, -1)
    for coords, scores in _predictions(model, image, gt, config, device, batch_size):
        result[coords[:, 0], coords[:, 1]] = scores.argmax(axis=1)
    return result


def infer_full_image(model, image, patch_size: int, num_classes: int, device,
                     batch_size: int = 64, *, config: dict) -> np.ndarray:
    config = {**config, 'patch_size': patch_size}
    gt = np.zeros(image.shape[:2], dtype=np.int64)
    result = np.zeros((*image.shape[:2], num_classes), dtype=np.float32)
    for coords, scores in _predictions(model, image, gt, config, device, batch_size):
        result[coords[:, 0], coords[:, 1]] = scores
    return result
