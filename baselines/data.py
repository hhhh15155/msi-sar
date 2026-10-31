from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import io as scipy_io

from baselines.split import split_from_config
from baselines.io import resolve_path


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
    import h5py
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
    label = np.squeeze(load_mat_array(dataset_dir / config.get("label_file", "label.mat"), key=config.get("label_key")))
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
    normalization = config.get("normalization", "per_channel_minmax")
    if normalization == "per_channel_minmax":
        for channel in range(image.shape[-1]):
            channel_data = image[:, :, channel]
            minimum = np.min(channel_data)
            denominator = np.max(channel_data) - minimum
            if denominator > 0:
                image[:, :, channel] = (channel_data - minimum) / denominator
    elif normalization == "global_minmax_center":
        # Compatibility for explicit historical dataset snapshots only.
        denominator = np.max(image) - np.min(image)
        if denominator > 0:
            image = (image - np.min(image)) / denominator
        image -= np.mean(image, axis=(0, 1), keepdims=True)
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    gt = label.astype("int64") - int(config.get("undefined_label", 0)) - 1
    return image, gt, list(config["class_names"])


class PatchDataset(torch.utils.data.Dataset):
    """MGHOFNet zero-padded patches, with a packed input for model adapters."""

    def __init__(self, image: np.ndarray, gt: np.ndarray, patch_size: int,
                 data_aug: bool = True, pad_mode: str = "constant"):
        if patch_size <= 0 or patch_size % 2 != 1:
            raise ValueError("patch_size must be a positive odd integer")
        self.data_aug = data_aug
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.data = np.pad(image, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)), mode=pad_mode)
        # Coordinates in the original extent only, in the same row-major order as MGHOFNet.
        self.indices = np.argwhere(gt >= 0) + self.pad_size
        np.random.shuffle(self.indices)
        self.label = gt

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        if np.random.random() <= 0.5:
            return patch
        prob = np.random.random()
        if prob <= 0.25:
            return np.fliplr(patch)
        if prob <= 0.5:
            return np.flipud(patch)
        return np.rot90(patch, k=np.random.randint(1, 4))

    def __getitem__(self, index: int):
        x, y = self.indices[index] - self.pad_size
        patch = self.data[x:x + self.patch_size, y:y + self.patch_size]
        if self.data_aug:
            patch = self._augment(patch)
        tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32))
        return tensor.unsqueeze(0), torch.tensor(self.label[x, y], dtype=torch.int64)


class FusionPatchDataset(PatchDataset):
    """Same patches/random draws; only the model input interface differs."""

    def __init__(self, image: np.ndarray, gt: np.ndarray, patch_size: int,
                 hsi_channels: int, aux_channels: int, data_aug: bool = True,
                 pad_mode: str = "constant"):
        if image.shape[-1] != hsi_channels + aux_channels:
            raise ValueError("MSI/SAR channel counts do not match the image")
        super().__init__(image, gt, patch_size, data_aug, pad_mode)
        self.hsi_channels = hsi_channels
        self.aux_channels = aux_channels

    def __getitem__(self, index: int):
        packed, label = super().__getitem__(index)
        return packed[0, :self.hsi_channels], packed[0, self.hsi_channels:], label
