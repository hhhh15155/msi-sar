"""Shared MGHOFNet data pipeline; legacy evaluation supplies explicit padding."""
from baselines.data import PatchDataset, FusionPatchDataset, load_dataset, load_mat_array  # noqa: F401
from baselines.split import split_fixed_train_counts, split_from_config  # noqa: F401
