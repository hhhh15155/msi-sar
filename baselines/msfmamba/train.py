from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PatchDataset, load_dataset, split_from_config
from .io import ensure_dir, load_yaml, save_json, save_yaml
from .metrics import aggregate_metrics, classification_metrics
from .model import MSFMamba


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {device_name}, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def next_run_dir(output_root: str | Path, dataset_name: str) -> Path:
    root = Path(output_root) / "msfmamba" / dataset_name
    ensure_dir(root)
    return root / f"run_{len([path for path in root.glob('run_*') if path.is_dir()]) + 1:03d}"


def log(message: str, log_file: Path) -> None:
    print(message)
    with log_file.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def split_modalities(images: torch.Tensor, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    images = images.squeeze(1)
    ms_channels = int(config.get("ms_channels", 10))
    sar_channels = int(config["sar_channels"])
    if images.shape[1] != ms_channels + sar_channels:
        raise ValueError(
            f"Expected {ms_channels + sar_channels} concatenated channels, got {images.shape[1]}"
        )
    return images[:, :ms_channels], images[:, ms_channels : ms_channels + sar_channels]


def build_model(config: dict, num_classes: int) -> MSFMamba:
    return MSFMamba(
        ms_channels=int(config.get("ms_channels", 10)),
        sar_channels=int(config["sar_channels"]),
        num_classes=num_classes,
        patch_size=int(config["patch_size"]),
        d_state=int(config.get("d_state", 16)),
        expand=float(config.get("expand", 0.75)),
        num_layers=int(config.get("num_layers", 1)),
    )


def _accuracy(model: nn.Module, loader: DataLoader, config: dict, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            ms, sar = split_modalities(images.to(device), config)
            predicted = model(ms, sar).argmax(dim=1)
            correct += int((predicted == targets.to(device)).sum().item())
            total += int(targets.numel())
    return correct / total if total else 0.0


def test_labeled_samples(
    model: nn.Module, loader: DataLoader, config: dict, device: torch.device, n_classes: int
) -> dict:
    model.eval()
    predictions, targets_all = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="test labeled samples"):
            ms, sar = split_modalities(images.to(device), config)
            predictions.append(model(ms, sar).argmax(dim=1).cpu().numpy())
            targets_all.append(targets.numpy())
    return classification_metrics(np.concatenate(predictions), np.concatenate(targets_all), n_classes)


def train_one_run(
    config: dict, image: np.ndarray, gt: np.ndarray, labels: list[str], run_dir: Path,
    run_index: int, seed: int, device: torch.device,
) -> dict:
    set_seed(seed)
    train_gt, val_gt, test_gt = split_from_config(gt, config, seed)
    ensure_dir(run_dir / "splits")
    np.savez_compressed(run_dir / "splits" / f"run_{run_index}.npz", train_gt=train_gt, val_gt=val_gt, test_gt=test_gt)

    patch_size, batch_size = int(config["patch_size"]), int(config["batch_size"])
    data_aug = bool(config.get("data_aug", True))
    train_loader = DataLoader(PatchDataset(image, train_gt, patch_size, data_aug=data_aug), batch_size=batch_size, shuffle=True)
    use_validation = bool(config.get("use_validation", True)) and bool(np.any(val_gt >= 0))
    val_loader = DataLoader(PatchDataset(image, val_gt, patch_size, data_aug=False), batch_size=batch_size) if use_validation else None
    test_loader = DataLoader(PatchDataset(image, test_gt, patch_size, data_aug=False), batch_size=batch_size)

    model = build_model(config, len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)), weight_decay=float(config.get("weight_decay", 0.0)))
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = ensure_dir(run_dir / "checkpoints" / f"run_{run_index}")
    log_file = run_dir / "log.txt"
    best_acc = -1.0

    for epoch in tqdm(range(1, int(config["epochs"]) + 1), desc=f"run {run_index}"):
        model.train()
        losses = []
        for images, targets in train_loader:
            ms, sar = split_modalities(images.to(device), config)
            optimizer.zero_grad()
            loss = criterion(model(ms, sar), targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_acc = _accuracy(model, val_loader, config, device) if use_validation else 0.0
        if epoch == 1 or epoch % 10 == 0:
            log(f"run={run_index} epoch={epoch} loss={np.mean(losses):.6f} val_oa={val_acc:.4f}", log_file)
        if use_validation and val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "model_best.pth")

    if not use_validation:
        torch.save(model.state_dict(), checkpoint_dir / "model_best.pth")
    model.load_state_dict(torch.load(checkpoint_dir / "model_best.pth", map_location=device))
    metrics = test_labeled_samples(model, test_loader, config, device, len(labels))
    metrics.update({"run": run_index, "seed": seed, "best_val_oa": best_acc * 100.0 if use_validation else None, "class_names": labels})
    save_json(metrics, run_dir / f"metrics_run_{run_index}.json")
    return metrics


def run_training(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    if config.get("model") != "msfmamba":
        raise ValueError(f"Expected model: msfmamba, got {config.get('model')!r}")
    dataset_config = load_yaml(config["dataset_config"])
    run_dir = next_run_dir(config.get("output_root", "runs"), config["dataset"])
    ensure_dir(run_dir)
    save_yaml(config, run_dir / "config.yaml")
    image, gt, labels = load_dataset(dataset_config)
    device = resolve_device(config.get("device", "cuda:0"))
    log(f"dataset={config['dataset']} image_shape={image.shape} labels={len(labels)}", run_dir / "log.txt")
    log(f"device={device} run_dir={run_dir}", run_dir / "log.txt")
    results = []
    for run_index, seed in enumerate(list(config["seeds"])[: int(config["num_runs"])]):
        result = train_one_run(config, image, gt, labels, run_dir, run_index, int(seed), device)
        results.append(result)
        log(f"run={run_index} oa={result['oa']:.2f} aa={result['aa']:.2f} kappa={result['kappa']:.2f}", run_dir / "log.txt")
    save_json({"runs": results, "aggregate": aggregate_metrics(results) if len(results) > 1 else None}, run_dir / "metrics.json")
    return run_dir
