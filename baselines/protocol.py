"""Versioned, dependency-free defaults for new MGHOFNet-protocol experiments."""
from __future__ import annotations

from copy import deepcopy

PROTOCOL = 'mghofnet-v1'
TRAINING_DEFAULTS = {
    'training_protocol': PROTOCOL,
    'optimizer': 'adamw', 'learning_rate': .001, 'weight_decay': .01,
    'scheduler': 'cosine', 'eta_min': .000001,
    'pad_mode': 'constant', 'data_aug': True, 'augmentation': 'mghofnet',
    'gradient_clip': None, 'cudnn_deterministic': True, 'cudnn_benchmark': False,
}


def prepare_training_config(config: dict) -> dict:
    """Fill defaults; reject stale explicit settings rather than silently rewriting them.

    Budgets, patch dimensions and sample counts remain configurable for smoke tests
    and explicitly authorized experiments. Existing run snapshots are never passed
    through this function during evaluation.
    """
    result = deepcopy(config)
    for key, value in TRAINING_DEFAULTS.items():
        if key in result and result[key] != value:
            raise ValueError(f'{PROTOCOL} requires {key}={value!r}; got {result[key]!r}')
        result[key] = value
    result.setdefault('eval', {}).setdefault('run_index', 'best')
    result.setdefault('infer', {})['save_full_map'] = False
    return result


def inference_pad_mode(config: dict) -> str:
    """Old runs keep their original padding even after project defaults change."""
    version = config.get('training_protocol')
    if version is not None and version != PROTOCOL:
        raise ValueError(f'Unsupported training protocol: {version!r}')
    if version == PROTOCOL:
        if config.get('pad_mode', 'constant') != 'constant':
            raise ValueError(f'{PROTOCOL} requires constant padding')
        return 'constant'
    if config.get('model') in ('mghofnet', 'softformer'):
        return config.get('pad_mode', 'constant')
    return 'reflect'


def effective_dataset_config(dataset_config: dict, run_config: dict) -> dict:
    """Respect the normalization actually implemented by old model readers."""
    result = deepcopy(dataset_config)
    if run_config.get('training_protocol') == PROTOCOL:
        if result.get('normalization') != 'per_channel_minmax':
            raise ValueError(f'{PROTOCOL} requires per_channel_minmax')
    elif run_config.get('model') in ('mghofnet', 'softformer'):
        result['normalization'] = 'per_channel_minmax'
    else:
        result.setdefault('normalization', 'global_minmax_center')
    return result
