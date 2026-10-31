"""Shared training operations; model construction stays in each baseline package."""
from __future__ import annotations

import json
import hashlib
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from .protocol import TRAINING_DEFAULTS, prepare_training_config
from .io import repo_root


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer_scheduler(model: torch.nn.Module, config: dict):
    config = prepare_training_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config['epochs']), eta_min=config['eta_min'])
    return optimizer, scheduler


def forward_batch(model, batch, device, config: dict):
    """Normalize packed/dual inputs and tensor/dictionary outputs without changing models."""
    *inputs, targets = batch
    inputs = [x.to(device) for x in inputs]
    if len(inputs) == 1 and config.get('model') in ('frekfuse', 'msfmamba', 'mghofnet', 'softformer'):
        x = inputs[0].squeeze(1)
        ms = int(config.get('ms_channels', config.get('hsi_channels', 10)))
        sar = int(config.get('sar_channels', config.get('aux_channels', 4)))
        if x.shape[1] != ms + sar:
            raise ValueError(f'Expected {ms + sar} channels, got {x.shape[1]}')
        inputs = [x[:, :ms], x[:, ms:]]
    output = model(*inputs)
    logits = output['logits'] if isinstance(output, dict) else output
    return logits, targets.to(device)


def train_epoch(model, loader, optimizer, device, config: dict) -> float:
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        logits, targets = forward_batch(model, batch, device, config)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def record_protocol(config: dict, dataset_config: dict, run_dir: Path) -> None:
    if dataset_config.get('normalization') != 'per_channel_minmax':
        raise ValueError('New runs require per_channel_minmax on the full unlabeled image')
    (run_dir / 'dataset_config.yaml').write_text(
        yaml.safe_dump(dataset_config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    try:
        code = {key: subprocess.check_output(['git', *args], cwd=repo_root(), text=True).strip()
                for key, args in (('commit', ['rev-parse', 'HEAD']), ('status', ['status', '--short']))}
    except (OSError, subprocess.CalledProcessError):
        code = {'commit': None, 'status': 'git unavailable'}
    record = {**TRAINING_DEFAULTS, 'epochs': config['epochs'], 'model': config['model'],
              'code': code, 'seeds': config['seeds'], 'split': config['split'],
              'config_sha256': hashlib.sha256((run_dir / 'config.yaml').read_bytes()).hexdigest(),
              'dataset_config_sha256': hashlib.sha256((run_dir / 'dataset_config.yaml').read_bytes()).hexdigest(),
              'normalization': 'per_channel_minmax', 'normalization_scope': 'full unlabeled image',
              'checkpoint': 'final epoch', 'precision': 'FP32',
              'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
              'torch': torch.__version__, 'numpy': np.__version__}
    (run_dir / 'training_protocol.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
