from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.mghofnet.io import ensure_dir, load_yaml, save_json, save_yaml
from models import VBENet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {device_name}, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def validate_config(config: dict) -> None:
    required = (
        "patch_size",
        "ms_channels",
        "sar_channels",
        "width",
        "groups",
        "batch_size",
        "test_batch_size",
        "epochs",
        "num_runs",
        "seeds",
        "split",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing VBE-Net config keys: {missing}")
    if int(config["patch_size"]) != 11:
        raise ValueError("VBE-Net experiments require patch_size=11")
    if int(config["width"]) % int(config["groups"]):
        raise ValueError("width must be divisible by groups")
    if config["split"].get("method") != "fixed_train_counts":
        raise ValueError("VBE-Net experiments require fixed_train_counts")
    if int(config["batch_size"]) != 128 or int(config["test_batch_size"]) != 1024:
        raise ValueError("Aligned experiments require batch_size=128 and test_batch_size=1024")
    if int(config["epochs"]) != 200:
        raise ValueError("Aligned experiments require epochs=200")
    if int(config["num_runs"]) != 5 or len(config["seeds"]) < 5:
        raise ValueError("Aligned experiments require five runs and five seeds")
    for key in ("use_validation", "select_best_by", "test_interval"):
        if key in config:
            raise ValueError(f"Legacy validation key is not allowed: {key}")


def build_model(config: dict, num_classes: int) -> VBENet:
    return VBENet(
        ms_channels=int(config.get("ms_channels", 10)),
        sar_channels=int(config.get("sar_channels", 4)),
        num_classes=num_classes,
        patch_size=int(config.get("patch_size", 11)),
        width=int(config.get("width", 64)),
        depth=int(config.get("encoder_depth", 5)),
        groups=int(config.get("groups", 8)),
        expansion=int(config.get("expansion", 4)),
        lambda_proto=float(config.get("lambda_proto", 1.0)),
        tau_r=float(config.get("tau_r", 0.3)),
        tau_c=float(config.get("tau_c", 0.1)),
        inner_iters=int(config.get("inner_iters", 3)),
        outer_updates=int(config.get("outer_updates", 1)),
        modality_dropout=float(config.get("modality_dropout", 0.1)),
        eps=float(config.get("geometry_eps", 1e-4)),
    )


def next_run_dir(output_root: str | Path, dataset_name: str) -> Path:
    root = Path(output_root) / "vbenet" / dataset_name
    ensure_dir(root)
    run_count = len([path for path in root.glob("run_*") if path.is_dir()])
    return root / f"run_{run_count + 1:03d}"


def log(message: str, log_file: Path) -> None:
    print(message)
    with log_file.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def make_loader(
    config: dict,
    image: np.ndarray,
    gt: np.ndarray,
    data_aug: bool,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    from baselines.mghofnet.dataset import FusionPatchDataset

    dataset = FusionPatchDataset(
        image,
        gt,
        int(config["patch_size"]),
        hsi_channels=int(config["ms_channels"]),
        aux_channels=int(config["sar_channels"]),
        data_aug=data_aug,
        pad_mode=config.get("pad_mode", "constant"),
    )
    actual_batch_size = int(config["batch_size"] if batch_size is None else batch_size)
    return DataLoader(dataset, batch_size=actual_batch_size, shuffle=shuffle)


def test_labeled_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    from baselines.mghofnet.metrics import classification_metrics

    from baselines.mghofnet.dataset import split_from_config

    model.eval()
    predictions: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    with torch.no_grad():
        for ms, sar, targets in tqdm(loader, desc="test labeled samples"):
            logits = model(ms.to(device), sar.to(device))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets_all.append(targets.numpy())
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets_all, axis=0)
    return classification_metrics(prediction, target, num_classes)


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
    train_gt, test_gt = split_from_config(gt, config, seed)
    ensure_dir(run_dir / "splits")
    np.savez_compressed(
        run_dir / "splits" / f"run_{run_index}.npz",
        train_gt=train_gt,
        test_gt=test_gt,
    )
    train_loader = make_loader(
        config,
        image,
        train_gt,
        data_aug=bool(config.get("data_aug", True)),
        shuffle=True,
    )
    test_loader = make_loader(
        config,
        image,
        test_gt,
        data_aug=False,
        shuffle=False,
        batch_size=int(config.get("test_batch_size", 1024)),
    )

    model = build_model(config, len(labels)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-2)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
        eta_min=float(config.get("eta_min", 1e-6)),
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = ensure_dir(run_dir / "checkpoints" / f"run_{run_index}")
    log_file = run_dir / "log.txt"
    epochs = int(config["epochs"])

    for epoch in tqdm(range(1, epochs + 1), desc=f"run {run_index}"):
        model.train()
        losses: list[float] = []
        for ms, sar, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ms.to(device), sar.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0:
            log(
                f"run={run_index} epoch={epoch} loss={np.mean(losses):.6f} "
                f"lr={scheduler.get_last_lr()[0]:.8f}",
                log_file,
            )
        if epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoint_dir / "model.pth")

    best_checkpoint = checkpoint_dir / "model_best.pth"
    torch.save(model.state_dict(), best_checkpoint)
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    metrics = test_labeled_samples(model, test_loader, device, len(labels))
    metrics.update({"run": run_index, "seed": seed, "class_names": labels})
    save_json(metrics, run_dir / f"metrics_run_{run_index}.json")
    return metrics


def run_training(config_path: str | Path) -> Path:
    from baselines.mghofnet.dataset import load_dataset
    from baselines.mghofnet.metrics import aggregate_metrics

    config = load_yaml(config_path)
    validate_config(config)
    dataset_config = load_yaml(config["dataset_config"])
    run_dir = next_run_dir(config.get("output_root", "runs"), config["dataset"])
    ensure_dir(run_dir)
    save_yaml(config, run_dir / "config.yaml")

    image, gt, labels = load_dataset(dataset_config)
    expected_channels = int(config["ms_channels"]) + int(config["sar_channels"])
    if image.shape[-1] < expected_channels:
        raise ValueError(
            f"Expected at least {expected_channels} channels, got {image.shape[-1]}"
        )
    device = resolve_device(config.get("device", "cuda:0"))
    log(
        f"dataset={config['dataset']} image_shape={image.shape} labels={len(labels)}",
        run_dir / "log.txt",
    )
    log(f"device={device} run_dir={run_dir}", run_dir / "log.txt")

    seeds = list(config["seeds"])[: int(config["num_runs"])]
    results = []
    for run_index, seed in enumerate(seeds):
        result = train_one_run(
            config,
            image,
            gt,
            labels,
            run_dir,
            run_index,
            int(seed),
            device,
        )
        results.append(result)
        log(
            f"run={run_index} oa={result['oa']:.2f} aa={result['aa']:.2f} "
            f"kappa={result['kappa']:.2f}",
            run_dir / "log.txt",
        )
    payload = {
        "runs": results,
        "aggregate": aggregate_metrics(results) if len(results) > 1 else None,
    }
    save_json(payload, run_dir / "metrics.json")
    return run_dir


__all__ = [
    "build_model",
    "make_loader",
    "next_run_dir",
    "resolve_device",
    "run_training",
    "test_labeled_samples",
    "train_one_run",
    "validate_config",
]
