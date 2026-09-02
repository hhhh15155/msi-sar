from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.mghofnet.io import load_yaml, save_json
from models import VBENet

from .train import build_model, resolve_device


def load_model(
    config: dict,
    labels: list[str],
    checkpoint: Path,
    device: torch.device,
) -> VBENet:
    model = build_model(config, len(labels)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def predict_labeled_samples(
    model: torch.nn.Module,
    image: np.ndarray,
    gt: np.ndarray,
    config: dict,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    from baselines.mghofnet.dataset import FusionPatchDataset

    dataset = FusionPatchDataset(
        image,
        gt,
        int(config["patch_size"]),
        hsi_channels=int(config["ms_channels"]),
        aux_channels=int(config["sar_channels"]),
        data_aug=False,
        pad_mode=config.get("pad_mode", "constant"),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    prediction = np.full_like(gt, fill_value=-1)
    offset = 0
    model.eval()
    with torch.no_grad():
        for ms, sar, _ in tqdm(loader, desc="predict labeled samples"):
            batch_prediction = torch.argmax(
                model(ms.to(device), sar.to(device)), dim=1
            ).cpu().numpy()
            batch_indices = dataset.indices[offset : offset + len(batch_prediction)]
            for (x, y), label in zip(batch_indices, batch_prediction):
                prediction[x - dataset.pad_size, y - dataset.pad_size] = int(label)
            offset += len(batch_prediction)
    return prediction


def save_labeled_maps(run_dir: str | Path, run_index: int = 0) -> dict:
    from baselines.mghofnet.dataset import load_dataset
    from baselines.mghofnet.infer import save_label_outputs
    from baselines.mghofnet.metrics import classification_metrics

    run_dir = Path(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    dataset_config = load_yaml(config["dataset_config"])
    image, gt, labels = load_dataset(dataset_config)
    device = resolve_device(config.get("device", "cuda:0"))
    checkpoint = run_dir / "checkpoints" / f"run_{run_index}" / "model_best.pth"
    model = load_model(config, labels, checkpoint, device)
    eval_config = config.get("eval", {})
    prediction = predict_labeled_samples(
        model,
        image,
        gt,
        config,
        int(eval_config.get("batch_size", config.get("test_batch_size", 1024))),
        device,
    )
    mask = gt >= 0
    labeled_gt = np.zeros_like(gt, dtype=np.uint8)
    labeled_prediction = np.zeros_like(gt, dtype=np.uint8)
    labeled_gt[mask] = gt[mask].astype(np.uint8) + 1
    labeled_prediction[mask] = prediction[mask].astype(np.uint8) + 1
    metrics = classification_metrics(prediction, gt, len(labels))
    metrics.update({"run": run_index, "class_names": labels})
    output = {"runs": [metrics], "outputs": {}}
    if eval_config.get("save_labeled_maps", True):
        output["outputs"] = {
            "labeled_gt": [
                str(path)
                for path in save_label_outputs(labeled_gt, run_dir / "labeled_gt", config.get("palette"))
            ],
            "labeled_prediction": [
                str(path)
                for path in save_label_outputs(
                    labeled_prediction, run_dir / "labeled_prediction", config.get("palette")
                )
            ],
        }
    save_json(output, run_dir / "eval_metrics.json")
    return output


def save_full_map(
    run_dir: str | Path,
    run_index: int = 0,
    output_prefix: str | Path | None = None,
) -> list[Path]:
    from baselines.mghofnet.dataset import load_dataset
    from baselines.mghofnet.infer import infer_full_image, save_label_outputs

    run_dir = Path(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    dataset_config = load_yaml(config["dataset_config"])
    image, _, labels = load_dataset(dataset_config)
    device = resolve_device(config.get("device", "cuda:0"))
    checkpoint = run_dir / "checkpoints" / f"run_{run_index}" / "model_best.pth"
    model = load_model(config, labels, checkpoint, device)
    logits = infer_full_image(
        model,
        image,
        int(config["patch_size"]),
        len(labels),
        int(config["ms_channels"]),
        int(config["sar_channels"]),
        device,
        batch_size=int(config.get("infer", {}).get("batch_size", 512)),
        pad_mode=config.get("pad_mode", "constant"),
    )
    prediction = np.argmax(logits, axis=-1).astype(np.uint8) + 1
    prefix = Path(output_prefix) if output_prefix else run_dir / "full_prediction"
    return save_label_outputs(prediction, prefix, config.get("palette"))


__all__ = ["load_model", "predict_labeled_samples", "save_full_map", "save_labeled_maps"]
