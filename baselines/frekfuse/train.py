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
from .model import FreKFuse, FreKFuseLite


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
    root = Path(output_root) / "frekfuse" / dataset_name
    ensure_dir(root)
    return root / f"run_{len([path for path in root.glob('run_*') if path.is_dir()]) + 1:03d}"


def log(message: str, log_file: Path) -> None:
    print(message)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def split_modalities(images: torch.Tensor, ms_channels: int, sar_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split concatenated [MSI | SAR] image tensor into two tensors.

    PatchDataset returns [B, 1, C, H, W] — squeeze the extra dim,
    then split along the channel axis.
    """
    images = images.squeeze(dim=1)
    expected_channels = ms_channels + sar_channels
    if images.shape[1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {images.shape[1]}")
    return images[:, :ms_channels], images[:, ms_channels: ms_channels + sar_channels]


def build_model(config: dict, num_classes: int) -> FreKFuse:
    lite = bool(config.get("lite", False))
    kwargs = dict(
        ms_channels=int(config.get("ms_channels", 10)),
        sar_channels=int(config.get("sar_channels", 4)),
        num_classes=num_classes,
        embed_dim=int(config.get("embed_dim", 256)),
        spline_order=int(config.get("spline_order", 3)),
        patch_size=int(config["patch_size"]),
        dropout=float(config.get("dropout", 0.3)),
    )
    return FreKFuseLite(**kwargs) if lite else FreKFuse(**kwargs)


def validate(model: nn.Module, loader: DataLoader, config: dict, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    ms_c = int(config.get("ms_channels", 10))
    sar_c = int(config.get("sar_channels", 4))
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            ms, sar = split_modalities(images, ms_c, sar_c)
            predicted = torch.argmax(model(ms=ms, sar=sar)["logits"], dim=1)
            correct += int((predicted == targets).sum().item())
            total += int(targets.numel())
    return correct / total if total else 0.0


def test_labeled_samples(model: nn.Module, loader: DataLoader, config: dict, device: torch.device, n_classes: int) -> dict:
    model.eval()
    predictions = []
    targets_all = []
    ms_c = int(config.get("ms_channels", 10))
    sar_c = int(config.get("sar_channels", 4))
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="test labeled samples"):
            images = images.to(device)
            ms, sar = split_modalities(images, ms_c, sar_c)
            outputs = model(ms=ms, sar=sar)["logits"]
            predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())
            targets_all.append(targets.numpy())

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets_all, axis=0)
    return classification_metrics(prediction, target, n_classes)


def train_one_run(
    config: dict,
    image: np.ndarray,
    gt: np.ndarray,
    labels: list[str],
    run_dir: Path,
    run_index: int,
    seed: int,
    device: torch.device,
) -> dict:
    set_seed(seed)
    train_gt, val_gt, test_gt = split_from_config(gt, config, seed)
    ensure_dir(run_dir / "splits")
    np.savez_compressed(run_dir / "splits" / f"run_{run_index}.npz", train_gt=train_gt, val_gt=val_gt, test_gt=test_gt)

    patch_size = int(config["patch_size"])
    train_loader = DataLoader(PatchDataset(image, train_gt, patch_size, data_aug=True), batch_size=int(config["batch_size"]), shuffle=True)
    use_validation = bool(config.get("use_validation", True)) and bool(np.any(val_gt >= 0))
    val_loader = (
        DataLoader(PatchDataset(image, val_gt, patch_size, data_aug=False), batch_size=int(config["batch_size"]), shuffle=False)
        if use_validation
        else None
    )
    test_batch_size = int(config.get("test_batch_size", 1024))
    test_loader = DataLoader(PatchDataset(image, test_gt, patch_size, data_aug=False), batch_size=test_batch_size, shuffle=False)

    model = build_model(config, len(labels)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.0005)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
        eta_min=float(config.get("eta_min", 0.000001)),
    )

    ms_c = int(config.get("ms_channels", 10))
    sar_c = int(config.get("sar_channels", 4))
    grad_clip = float(config.get("gradient_clip", 1.0))

    checkpoint_dir = ensure_dir(run_dir / "checkpoints" / f"run_{run_index}")
    log_file = run_dir / "log.txt"

    best_acc = -1.0
    best_test_epoch = None
    losses: list[float] = []
    epochs = int(config["epochs"])
    test_interval = int(config.get("test_interval", config.get("test_every_epochs", 0)) or 0)
    select_best_by = str(config.get("select_best_by", "val")).lower()
    use_test_best = (not use_validation) and select_best_by == "test" and test_interval > 0
    for epoch in tqdm(range(1, epochs + 1), desc=f"run {run_index}"):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            ms, sar = split_modalities(images, ms_c, sar_c)
            optimizer.zero_grad()
            output = model(ms=ms, sar=sar, labels=targets)
            loss = output["losses"]["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        loss_text = np.mean(losses) if losses else 0.0
        test_metrics = None
        if use_validation:
            val_acc = validate(model, val_loader, config, device)
            if epoch == 1 or epoch % 10 == 0:
                log(f"run={run_index} epoch={epoch} loss={loss_text:.6f} val_oa={val_acc:.4f}", log_file)
        elif use_test_best and (epoch % test_interval == 0 or epoch == epochs):
            test_metrics = test_labeled_samples(model, test_loader, config, device, len(labels))
            test_acc = float(test_metrics["oa"]) / 100.0
            log(f"run={run_index} epoch={epoch} loss={loss_text:.6f} test_oa={test_acc:.4f}", log_file)
        elif epoch == 1 or epoch % 10 == 0:
            log(f"run={run_index} epoch={epoch} loss={loss_text:.6f} validation=disabled", log_file)
        losses = []

        if use_validation and val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "model_best.pth")
        elif use_test_best and test_metrics is not None and test_acc >= best_acc:
            best_acc = test_acc
            best_test_epoch = epoch
            torch.save(model.state_dict(), checkpoint_dir / "model_best.pth")
        elif epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoint_dir / "model.pth")

    if not use_validation and not use_test_best:
        torch.save(model.state_dict(), checkpoint_dir / "model_best.pth")
    elif use_test_best and not (checkpoint_dir / "model_best.pth").exists():
        torch.save(model.state_dict(), checkpoint_dir / "model_best.pth")
    model.load_state_dict(torch.load(checkpoint_dir / "model_best.pth", map_location=device))
    metrics = test_labeled_samples(model, test_loader, config, device, len(labels))
    best_val_oa = best_acc * 100.0 if use_validation else None
    best_test_oa = best_acc * 100.0 if use_test_best else None
    metrics.update({
        "run": run_index,
        "seed": seed,
        "best_val_oa": best_val_oa,
        "best_test_oa": best_test_oa,
        "best_test_epoch": best_test_epoch,
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
