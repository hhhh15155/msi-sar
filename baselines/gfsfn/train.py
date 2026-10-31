from __future__ import annotations

import random
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.protocol import prepare_training_config
from baselines.training import set_seed, build_optimizer_scheduler, train_epoch, record_protocol
from .dataset import PatchDataset, load_dataset, split_from_config
from .io import ensure_dir, load_yaml, resolve_path, repo_root, save_json, save_yaml
from .metrics import aggregate_metrics, classification_metrics
from .model import GFSFN




def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {device_name}, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def next_run_dir(output_root: str | Path, dataset_name: str) -> Path:
    root = Path(output_root) / "gfsfn" / dataset_name
    ensure_dir(root)
    indices = [int(path.name[4:]) for path in root.glob('run_*')
               if path.is_dir() and path.name[4:].isdigit()]
    return root / f"run_{max(indices, default=0) + 1:03d}"


def record_identity(config_path: Path, dataset_path: Path, run_dir: Path, config: dict) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(['git', *args], cwd=repo_root(), text=True).strip()
    try:
        version = {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--short')}
    except (OSError, subprocess.CalledProcessError):
        version = {'commit': None, 'status': 'git unavailable'}
    save_json({
        'model': 'GFSFN', 'code': version,
        'config_path': str(config_path),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'dataset_config_path': str(dataset_path),
        'dataset_config_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        'seeds': config['seeds'], 'pretrained': False,
        'checkpoint': 'final epoch; model_best.pth is a compatibility filename',
        'normalization_scope': 'full unlabeled image, per channel',
        'source_commit': '065cb808870515b70c4980642895fa45da63a1af',
        'input_adaptation': 'native MSI/SAR channels; no PCA; upstream fusion and frequency modules',
        'python': sys.version, 'python_executable': sys.executable,
        'torch': torch.__version__, 'torch_path': torch.__file__,
        'numpy': np.__version__, 'numpy_path': np.__file__, 'platform': platform.platform(),
        'pythonpath': __import__('os').environ.get('PYTHONPATH'),
        'ld_library_path': __import__('os').environ.get('LD_LIBRARY_PATH'),
        'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, run_dir / 'run_identity.json')


def log(message: str, log_file: Path) -> None:
    print(message)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def test_labeled_samples(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int) -> dict:
    model.eval()
    predictions = []
    targets_all = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="test labeled samples"):
            images = images.to(device)
            outputs = model(images)
            predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())
            targets_all.append(targets.numpy())

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets_all, axis=0)
    return classification_metrics(prediction, target, n_classes)


def train_one_run(config: dict, image: np.ndarray, gt: np.ndarray, labels: list[str], run_dir: Path, run_index: int, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    train_gt, test_gt = split_from_config(gt, config, seed)
    ensure_dir(run_dir / "splits")
    np.savez_compressed(run_dir / "splits" / f"run_{run_index}.npz", train_gt=train_gt, test_gt=test_gt)

    patch_size = int(config["patch_size"])
    train_loader = DataLoader(PatchDataset(image, train_gt, patch_size, data_aug=True), batch_size=int(config["batch_size"]), shuffle=True)
    test_batch_size = int(config.get("test_batch_size", 1024))
    test_loader = DataLoader(PatchDataset(image, test_gt, patch_size, data_aug=False), batch_size=test_batch_size, shuffle=False)

    model = GFSFN(
        num_classes=len(labels),
        patch_size=patch_size,
        spectral_channels=int(config.get("spectral_channels", 10)),
        sar_channels=int(config.get("sar_channels", 4)),
    ).to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, config)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints" / f"run_{run_index}")
    log_file = run_dir / "log.txt"

    for epoch in tqdm(range(1, int(config["epochs"]) + 1), desc=f"run {run_index}"):
        loss_text = train_epoch(model, train_loader, optimizer, device, config)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0:
            log(f"run={run_index} epoch={epoch} loss={loss_text:.6f}", log_file)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoint_dir / "model.pth")

    torch.save(model.state_dict(), checkpoint_dir / "model_best.pth")
    model.load_state_dict(torch.load(checkpoint_dir / "model_best.pth", map_location=device))
    metrics = test_labeled_samples(model, test_loader, device, len(labels))
    metrics.update({
        "run": run_index,
        "seed": seed,
        "class_names": labels,
    })
    save_json(metrics, run_dir / f"metrics_run_{run_index}.json")
    return metrics


def run_training(config_path: str | Path) -> Path:
    config_path = resolve_path(config_path)
    config = prepare_training_config(load_yaml(config_path))
    dataset_path = resolve_path(config["dataset_config"])
    dataset_config = load_yaml(dataset_path)
    seeds = list(config["seeds"])
    if len(seeds) != int(config['num_runs']) or len(set(seeds)) != len(seeds):
        raise ValueError('num_runs must match a list of distinct seeds')
    expected_channels = int(config['spectral_channels']) + int(config['sar_channels'])
    if expected_channels != int(dataset_config['num_channels']):
        raise ValueError('MSI/SAR channel counts disagree with dataset configuration')
    if len(dataset_config['class_names']) != int(dataset_config['num_classes']):
        raise ValueError('Class names disagree with dataset num_classes')
    run_dir = next_run_dir(resolve_path(config.get("output_root", "runs")), config["dataset"])
    ensure_dir(run_dir)
    save_yaml(config, run_dir / "config.yaml")
    record_protocol(config, dataset_config, run_dir)
    save_yaml(dataset_config, run_dir / "dataset_config.yaml")
    record_identity(config_path, dataset_path, run_dir, config)

    image, gt, labels = load_dataset(dataset_config)
    device = resolve_device(config.get("device", "cuda:0"))
    log(f"dataset={config['dataset']} image_shape={image.shape} labels={len(labels)}", run_dir / "log.txt")
    log(f"device={device} run_dir={run_dir}", run_dir / "log.txt")

    seeds = list(config["seeds"])[: int(config["num_runs"])]
    results = []
    for run_index, seed in enumerate(seeds):
        result = train_one_run(config, image, gt, labels, run_dir, run_index, int(seed), device)
        results.append(result)
        log(f"run={run_index} oa={result['oa']:.2f} aa={result['aa']:.2f} kappa={result['kappa']:.2f}", run_dir / "log.txt")
    save_json({"runs": results, "aggregate": aggregate_metrics(results) if len(results) > 1 else None}, run_dir / "metrics.json")
    return run_dir
