from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy import io as scipy_io
from sklearn.model_selection import train_test_split

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
    label = np.squeeze(load_mat_array(dataset_dir / config.get("label_file", "label.mat"), key=config.get("label_key")))
    if image.ndim != 3:
        raise ValueError(f"Expected image cube, got {image.shape}")
    if label.ndim != 2:
        raise ValueError(f"Expected 2D label map, got {label.shape}")
    if image.shape[:2] != label.shape:
        raise ValueError(f"Image/label shape mismatch: {image.shape[:2]} vs {label.shape}")

    nan_mask = np.isnan(image.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        image[nan_mask] = 0
        label[nan_mask] = int(config.get("undefined_label", 0))

    image = np.asarray(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        channel_data = image[:, :, channel]
        denom = np.max(channel_data) - np.min(channel_data)
        if denom > 0:
            image[:, :, channel] = (channel_data - np.min(channel_data)) / denom

    gt = label.astype("int64") - int(config.get("undefined_label", 0)) - 1
    return image, gt, list(config["class_names"])


def sample_gt(gt: np.ndarray, percentage: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.where(gt >= 0)
    x = list(zip(*indices))
    y = gt[indices].ravel()
    selected_gt = np.full_like(gt, fill_value=-1)
    rest_gt = np.full_like(gt, fill_value=-1)
    selected_indices, rest_indices = train_test_split(x, train_size=percentage, random_state=seed, stratify=y)
    selected_indices = [list(t) for t in zip(*selected_indices)]
    rest_indices = [list(t) for t in zip(*rest_indices)]
    selected_gt[tuple(selected_indices)] = gt[tuple(selected_indices)]
    rest_gt[tuple(rest_indices)] = gt[tuple(rest_indices)]
    return selected_gt, rest_gt


def split_train_val_test(gt: np.ndarray, train_ratio: float, val_ratio: float, seed: int):
    train_val_gt, test_gt = sample_gt(gt, train_ratio + val_ratio, seed)
    train_gt, val_gt = sample_gt(train_val_gt, train_ratio / (train_ratio + val_ratio), seed)
    return train_gt, val_gt, test_gt


def _expand_class_counts(counts: list[int] | int, n_classes: int, name: str) -> list[int]:
    if isinstance(counts, int):
        return [counts] * n_classes
    if len(counts) != n_classes:
        raise ValueError(f"Expected {n_classes} {name} counts, got {len(counts)}")
    return counts


def split_fixed_counts(gt: np.ndarray, train_counts: list[int] | int, val_counts: list[int] | int, seed: int):
    n_classes = int(np.max(gt)) + 1
    train_counts = _expand_class_counts(train_counts, n_classes, "train")
    val_counts = _expand_class_counts(val_counts, n_classes, "val")
    rng = np.random.default_rng(seed)
    train_gt = np.full_like(gt, fill_value=-1)
    val_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)
    for class_index in range(n_classes):
        coords = np.column_stack(np.where(gt == class_index))
        rng.shuffle(coords)
        train_count = int(train_counts[class_index])
        val_count = int(val_counts[class_index])
        if train_count + val_count > len(coords):
            raise ValueError(
                f"Class {class_index} has {len(coords)} samples, "
                f"but train+val requires {train_count + val_count}"
            )
        train_coords = coords[:train_count]
        val_coords = coords[train_count : train_count + val_count]
        test_coords = coords[train_count + val_count :]
        train_gt[train_coords[:, 0], train_coords[:, 1]] = class_index
        val_gt[val_coords[:, 0], val_coords[:, 1]] = class_index
        test_gt[test_coords[:, 0], test_coords[:, 1]] = class_index
    return train_gt, val_gt, test_gt


def split_fixed_train_counts(gt: np.ndarray, train_counts: list[int] | int, seed: int):
    n_classes = int(np.max(gt)) + 1
    train_counts = _expand_class_counts(train_counts, n_classes, "train")
    rng = np.random.default_rng(seed)
    train_gt = np.full_like(gt, fill_value=-1)
    val_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)
    for class_index in range(n_classes):
        coords = np.column_stack(np.where(gt == class_index))
        rng.shuffle(coords)
        train_count = int(train_counts[class_index])
        if train_count > len(coords):
            raise ValueError(f"Class {class_index} has {len(coords)} samples, but train requires {train_count}")
        train_coords = coords[:train_count]
        test_coords = coords[train_count:]
        train_gt[train_coords[:, 0], train_coords[:, 1]] = class_index
        test_gt[test_coords[:, 0], test_coords[:, 1]] = class_index
    return train_gt, val_gt, test_gt


def split_from_config(gt: np.ndarray, config: dict[str, Any], seed: int):
    split_config = config.get("split", {})
    if split_config.get("method") == "fixed_counts":
        train_counts = split_config.get("train_counts", split_config.get("train_count_per_class"))
        val_counts = split_config.get("val_counts", split_config.get("val_count_per_class"))
        if train_counts is None or val_counts is None:
            raise ValueError("fixed_counts requires train/val counts or train/val count_per_class")
        return split_fixed_counts(gt, train_counts, val_counts, seed)
    if split_config.get("method") == "fixed_train_counts":
        train_counts = split_config.get("train_counts", split_config.get("train_count_per_class"))
        if train_counts is None:
            raise ValueError("fixed_train_counts requires train_counts or train_count_per_class")
        return split_fixed_train_counts(gt, train_counts, seed)
    return split_train_val_test(gt, float(config["train_ratio"]), float(config["val_ratio"]), seed)


class FusionPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image: np.ndarray,
        gt: np.ndarray,
        patch_size: int,
        hsi_channels: int,
        aux_channels: int,
        data_aug: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.data_aug = data_aug
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.hsi_channels = hsi_channels
        self.aux_channels = aux_channels
        self.data = np.pad(image, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)), mode=pad_mode)
        self.label = np.pad(gt, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size)), mode=pad_mode)
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

    def _augment(self, hsi: np.ndarray, aux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() <= 0.5:
            return hsi, aux
        prob = np.random.random()
        if prob <= 0.25:
            return np.fliplr(hsi), np.fliplr(aux)
        if prob <= 0.5:
            return np.flipud(hsi), np.flipud(aux)
        k = np.random.randint(1, 4)
        return np.rot90(hsi, k=k), np.rot90(aux, k=k)

    def __getitem__(self, index: int):
        x, y = self.indices[index]
        x1, y1 = x - self.pad_size, y - self.pad_size
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        patch = self.data[x1:x2, y1:y2]
        hsi = patch[:, :, : self.hsi_channels]
        aux = patch[:, :, self.hsi_channels : self.hsi_channels + self.aux_channels]
        label = self.label[x, y]
        if self.data_aug:
            hsi, aux = self._augment(hsi, aux)
        hsi = np.asarray(np.copy(hsi).transpose((2, 0, 1)), dtype="float32")
        aux = np.asarray(np.copy(aux).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")
        return torch.from_numpy(hsi), torch.from_numpy(aux), torch.from_numpy(label)
