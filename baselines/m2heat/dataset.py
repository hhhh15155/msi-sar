from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy import io as scipy_io

from baselines.split import split_from_config
from .io import resolve_path


def _choose_key(data: dict[str, Any], preferred_key: str | None) -> str:
    if preferred_key and preferred_key in data:
        return preferred_key
    keys = [key for key in data.keys() if not key.startswith("__")]
    if not keys:
        raise ValueError("No MATLAB array keys found")
    return keys[0]


def load_mat_array(path: str | Path, key: str | None = None, expected_channels: int | None = None) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import hdf5storage

        data = hdf5storage.loadmat(str(path))
        return np.asarray(data[_choose_key(data, key)])
    except ImportError:
        pass
    except Exception:
        pass

    try:
        data = scipy_io.loadmat(path)
        return np.asarray(data[_choose_key(data, key)])
    except NotImplementedError:
        pass

    with h5py.File(path, "r") as f:
        selected_key = key if key is not None and key in f else next(iter(f.keys()))
        array = np.asarray(f[selected_key])
    if array.ndim == 3 and expected_channels is not None and array.shape[0] == expected_channels:
        array = np.moveaxis(array, 0, -1)
    return array


def load_dataset(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset_dir = resolve_path(config["path"])
    image = load_mat_array(
        dataset_dir / config.get("data_file", "data.mat"),
        key=config.get("data_key"),
        expected_channels=int(config["num_channels"]),
    )
    label = load_mat_array(dataset_dir / config.get("label_file", "label.mat"), key=config.get("label_key"))
    label = np.squeeze(label)

    if image.ndim != 3:
        raise ValueError(f"Expected image cube, got {image.shape}")
    if label.ndim != 2:
        raise ValueError(f"Expected 2D label map, got {label.shape}")
    if image.shape[:2] != label.shape:
        raise ValueError(f"Image/label shape mismatch: {image.shape[:2]} vs {label.shape}")
    if image.shape[-1] != int(config["num_channels"]):
        raise ValueError(f"Expected {config['num_channels']} channels, got {image.shape[-1]}")

    nan_mask = np.isnan(image.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        image[nan_mask] = 0
        label[nan_mask] = int(config.get("undefined_label", 0))

    image = np.asarray(image, dtype=np.float32)
    normalization = config.get("normalization", "global_minmax_center")
    if normalization == "per_channel_minmax":
        for channel in range(image.shape[-1]):
            channel_data = image[:, :, channel]
            minimum = np.min(channel_data)
            denominator = np.max(channel_data) - minimum
            if denominator > 0:
                image[:, :, channel] = (channel_data - minimum) / denominator
    elif normalization == "global_minmax_center":
        denominator = np.max(image) - np.min(image)
        if denominator > 0:
            image = (image - np.min(image)) / denominator
        mean_by_channel = np.mean(image, axis=(0, 1))
        for channel in range(image.shape[-1]):
            image[:, :, channel] = image[:, :, channel] - mean_by_channel[channel]
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    gt = label.astype("int64") - int(config.get("undefined_label", 0)) - 1
    return image, gt, list(config["class_names"])


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, image: np.ndarray, gt: np.ndarray, patch_size: int, data_aug: bool = True):
        super().__init__()
        self.data_aug = data_aug
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.data = np.pad(image, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)), mode="reflect")
        self.label = np.pad(gt, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size)), mode="reflect")
        mask = np.ones_like(self.label)
        mask[self.label < 0] = 0
        x_pos, y_pos = np.nonzero(mask)
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if self.pad_size <= x < image.shape[0] + self.pad_size
                and self.pad_size <= y < image.shape[1] + self.pad_size
            ]
        )
        np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, data: np.ndarray) -> np.ndarray:
        if np.random.random() <= 0.5:
            return data
        prob = np.random.random()
        if prob <= 0.2:
            return np.fliplr(data)
        if prob <= 0.4:
            return np.flipud(data)
        if prob <= 0.6:
            return np.rot90(data, k=1)
        if prob <= 0.8:
            return np.rot90(data, k=2)
        return np.rot90(data, k=3)

    def __getitem__(self, index: int):
        x, y = self.indices[index]
        x1, y1 = x - self.pad_size, y - self.pad_size
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]
        if self.data_aug:
            data = self._augment(data)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")
        return torch.from_numpy(data).unsqueeze(0), torch.from_numpy(label)
