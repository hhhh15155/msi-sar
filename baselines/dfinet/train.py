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
from .model import DFINet


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
    root = Path(output_root) / "dfinet" / dataset_name
    ensure_dir(root)
    return root / f"run_{len([path for path in root.glob('run_*') if path.is_dir()]) + 1:03d}"


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

    model = DFINet(
        num_classes=len(labels),
        patch_size=patch_size,
        spectral_channels=int(config.get("spectral_channels", 10)),
        sar_channels=int(config.get("sar_channels", 4)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = ensure_dir(run_dir / "checkpoints" / f"run_{run_index}")
    log_file = run_dir / "log.txt"

    losses: list[float] = []
    epochs = int(config["epochs"])
    for epoch in tqdm(range(1, epochs + 1), desc=f"run {run_index}"):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        loss_text = np.mean(losses) if losses else 0.0
        if epoch == 1 or epoch % 10 == 0:
            log(f"run={run_index} epoch={epoch} loss={loss_text:.6f}", log_file)
        losses = []

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
    config = load_yaml(config_path)
    dataset_config = load_yaml(config["dataset_config"])
    run_dir = next_run_dir(config.get("output_root", "runs"), config["dataset"])
    ensure_dir(run_dir)
    save_yaml(config, run_dir / "config.yaml")

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
