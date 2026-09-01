"""Deterministic train/test split utilities for few-shot experiments."""

from __future__ import annotations

from typing import Any

import numpy as np


def _expand_class_counts(counts: list[int] | int, n_classes: int) -> list[int]:
    if isinstance(counts, int):
        return [counts] * n_classes
    if len(counts) != n_classes:
        raise ValueError(f"Expected {n_classes} train counts, got {len(counts)}")
    return counts


def split_fixed_train_counts(
    gt: np.ndarray, train_counts: list[int] | int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    n_classes = int(np.max(gt)) + 1
    train_counts = _expand_class_counts(train_counts, n_classes)

    rng = np.random.default_rng(seed)
    train_gt = np.full_like(gt, fill_value=-1)
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

    return train_gt, test_gt


def split_from_config(gt: np.ndarray, config: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    split_config = config.get("split", {})
    if split_config.get("method") != "fixed_train_counts":
        raise ValueError("Only fixed_train_counts split configurations are supported")

    train_counts = split_config.get("train_counts", split_config.get("train_count_per_class"))
    if train_counts is None:
        raise ValueError("fixed_train_counts requires train_counts or train_count_per_class")
    return split_fixed_train_counts(gt, train_counts, seed)
