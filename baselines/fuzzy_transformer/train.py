from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.dfinet.dataset import PatchDataset, load_dataset
from baselines.frekfuse.io import ensure_dir, load_yaml, repo_root, resolve_path, save_json, save_yaml
from baselines.frekfuse.metrics import aggregate_metrics
from baselines.frekfuse.train import log, resolve_device, set_seed, split_modalities, test_labeled_samples
from baselines.split import split_from_config
from .model import FuzzyTransformer


def build_model(config: dict, num_classes: int) -> FuzzyTransformer:
    if config.get("model") != "fuzzy_transformer":
        raise ValueError("Expected model: fuzzy_transformer in experiment configuration")
    return FuzzyTransformer(
        ms_channels=int(config["ms_channels"]), sar_channels=int(config["sar_channels"]),
        num_classes=num_classes, embed_dim=int(config.get("embed_dim", 192)),
        num_heads=int(config.get("num_heads", 6)), depth_per_stage=int(config.get("depth_per_stage", 2)),
        fusion_depth=int(config.get("fusion_depth", 2)), mlp_ratio=float(config.get("mlp_ratio", 4)),
        dropout=float(config.get("dropout", 0.1)), num_memberships=int(config.get("num_memberships", 7)),
        membership_min_width=float(config.get("membership_min_width", 0.15)),
        fusion_temperature=float(config.get("fusion_temperature", 1)),
        fusion=config.get("fusion", "fuzzy"), auxiliary_weight=float(config.get("auxiliary_weight", 0.2)),
        label_smoothing=float(config.get("label_smoothing", 0)),
    )


def next_run_dir(output_root: str | Path, dataset_name: str) -> Path:
    root = ensure_dir(resolve_path(output_root) / "fuzzy_transformer" / dataset_name)
    existing = [int(path.name[4:]) for path in root.glob("run_*")
                if path.is_dir() and path.name[4:].isdigit()]
    run_dir = root / f"run_{max(existing, default=0) + 1:03d}"
    run_dir.mkdir()  # Refuse to overwrite a concurrently created run.
    return run_dir


def train_one_run(config: dict, image: np.ndarray, gt: np.ndarray, labels: list[str],
                  run_dir: Path, run_index: int, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    train_gt, test_gt = split_from_config(gt, config, seed)
    np.savez_compressed(ensure_dir(run_dir / "splits") / f"run_{run_index}.npz",
                        train_gt=train_gt, test_gt=test_gt)
    patch_size = int(config["patch_size"])
    train_data = PatchDataset(image, train_gt, patch_size, data_aug=True)
    test_data = PatchDataset(image, test_gt, patch_size, data_aug=False)
    if not len(train_data) or not len(test_data):
        raise ValueError("Both training and held-out test sets must contain labeled samples")
    train_loader = DataLoader(train_data, batch_size=int(config["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_data, batch_size=int(config.get("test_batch_size", 128)), shuffle=False)
    model = build_model(config, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 0.0005)),
                                  weight_decay=float(config.get("weight_decay", 0.0001)))
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(config.get("eta_min", 0.000001)))
    checkpoint_dir = ensure_dir(run_dir / "checkpoints" / f"run_{run_index}")
    log_file = run_dir / "log.txt"
    log(f"run={run_index} seed={seed} parameters={sum(p.numel() for p in model.parameters())} "
        f"train={len(train_data)} test={len(test_data)}", log_file)
    for epoch in tqdm(range(1, epochs + 1), desc=f"run {run_index}"):
        model.train()
        loss_sum, examples = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            ms, sar = split_modalities(images, int(config["ms_channels"]), int(config["sar_channels"]))
            optimizer.zero_grad(set_to_none=True)
            loss = model(ms=ms, sar=sar, labels=targets)["losses"]["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1)))
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            examples += len(targets)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            log(f"run={run_index} epoch={epoch} loss={loss_sum / examples:.6f}", log_file)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoint_dir / "model.pth")

    # Match the existing final-epoch protocol: no epoch or seed selection on test OA.
    torch.save(model.state_dict(), checkpoint_dir / "model_last.pth")
    metrics = test_labeled_samples(model, test_loader, config, device, len(labels))
    metrics.update({"run": run_index, "seed": seed, "class_names": labels,
                    "checkpoint": "model_last.pth", "evaluation_scope": "held_out_test",
                    "train_counts": [int(np.sum(train_gt == c)) for c in range(len(labels))],
                    "test_counts": [int(np.sum(test_gt == c)) for c in range(len(labels))]})
    save_json(metrics, run_dir / f"metrics_run_{run_index}.json")
    return metrics


def run_training(config_path: str | Path) -> Path:
    config = load_yaml(resolve_path(config_path))
    dataset_config_path = resolve_path(config["dataset_config"])
    dataset_config = load_yaml(dataset_config_path)
    try:
        config["dataset_config"] = dataset_config_path.relative_to(repo_root()).as_posix()
    except ValueError:
        config["dataset_config"] = str(dataset_config_path)
    patch_size = int(config["patch_size"])
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    if int(config["epochs"]) < 1 or int(config["num_runs"]) < 1:
        raise ValueError("epochs and num_runs must be positive")
    if int(config["batch_size"]) < 1 or int(config.get("test_batch_size", 128)) < 1:
        raise ValueError("Batch sizes must be positive")
    seeds = list(config["seeds"])
    if len(seeds) < int(config["num_runs"]):
        raise ValueError("Provide one seed for each requested run")
    seeds = seeds[:int(config["num_runs"])]
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("Each run requires a distinct seed")
    eval_config = config.get("eval", {})
    if eval_config.get("enabled", True):
        try:
            eval_run = int(eval_config.get("run_index", 0))
        except (TypeError, ValueError) as error:
            raise ValueError("eval.run_index must be a fixed integer, not a test-selected seed") from error
        if not 0 <= eval_run < int(config["num_runs"]):
            raise ValueError("eval.run_index must refer to a configured run")
        if int(eval_config.get("batch_size", config.get("test_batch_size", 128))) < 1:
            raise ValueError("eval.batch_size must be positive")
    image, gt, labels = load_dataset(dataset_config)
    if int(config["ms_channels"]) + int(config["sar_channels"]) != image.shape[-1]:
        raise ValueError("Configured MSI/SAR channels do not match the dataset")
    # Validate architecture options before creating output directories.
    build_model(config, len(labels))
    run_dir = next_run_dir(config.get("output_root", "runs"), config["dataset"])
    save_yaml(config, run_dir / "config.yaml")
    device = resolve_device(config.get("device", "cuda:0"))
    log(f"dataset={config['dataset']} image_shape={image.shape} classes={len(labels)} "
        f"device={device} run_dir={run_dir}", run_dir / "log.txt")
    results = []
    for run_index, seed in enumerate(seeds):
        result = train_one_run(config, image, gt, labels, run_dir, run_index, int(seed), device)
        results.append(result)
        log(f"run={run_index} OA={result['oa']:.2f} AA={result['aa']:.2f} Kappa={result['kappa']:.2f}",
            run_dir / "log.txt")
        save_json({"runs": results, "aggregate": aggregate_metrics(results) if len(results) > 1 else None,
                   "completed": len(results) == len(seeds), "expected_runs": len(seeds)},
                  run_dir / "metrics.json")
    return run_dir
