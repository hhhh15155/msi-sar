from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PatchDataset, load_dataset
from .infer import infer_full_image, save_label_outputs
from .io import load_yaml, save_json
from .metrics import classification_metrics
from .train import build_model, resolve_device, split_modalities


def load_model(config: dict, labels: list[str], checkpoint: Path, device: torch.device):
    model = build_model(config, len(labels)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def predict_labeled_samples(model: torch.nn.Module, image: np.ndarray, gt: np.ndarray, config: dict, device: torch.device) -> np.ndarray:
    dataset = PatchDataset(image, gt, int(config["patch_size"]), data_aug=False)
    loader = DataLoader(dataset, batch_size=int(config.get("eval", {}).get("batch_size", config.get("test_batch_size", 1024))))
    prediction = np.full_like(gt, -1)
    offset = 0
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="predict labeled samples"):
            ms, sar = split_modalities(images.to(device), config)
            batch_prediction = model(ms, sar).argmax(dim=1).cpu().numpy()
            for (x, y), label in zip(dataset.indices[offset : offset + len(batch_prediction)], batch_prediction):
                prediction[x - dataset.pad_size, y - dataset.pad_size] = int(label)
            offset += len(batch_prediction)
    return prediction


def save_labeled_maps(run_dir: str | Path, run_index: int = 0) -> dict:
    run_dir = Path(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    image, gt, labels = load_dataset(load_yaml(config["dataset_config"]))
    device = resolve_device(config.get("device", "cuda:0"))
    model = load_model(config, labels, run_dir / "checkpoints" / f"run_{run_index}" / "model_best.pth", device)
    prediction = predict_labeled_samples(model, image, gt, config, device)
    mask = gt >= 0
    labeled_gt = np.zeros_like(gt, dtype=np.uint8)
    labeled_prediction = np.zeros_like(gt, dtype=np.uint8)
    labeled_gt[mask] = gt[mask].astype(np.uint8) + 1
    labeled_prediction[mask] = prediction[mask].astype(np.uint8) + 1
    metrics = classification_metrics(prediction, gt, len(labels))
    metrics.update({"run": run_index, "class_names": labels})
    output = {"runs": [metrics], "outputs": {}}
    if config.get("eval", {}).get("save_labeled_maps", True):
        output["outputs"] = {
            "labeled_gt": [str(path) for path in save_label_outputs(labeled_gt, run_dir / "labeled_gt", config.get("palette"))],
            "labeled_prediction": [str(path) for path in save_label_outputs(labeled_prediction, run_dir / "labeled_prediction", config.get("palette"))],
        }
    save_json(output, run_dir / "eval_metrics.json")
    return output


def save_full_map(run_dir: str | Path, run_index: int = 0, output_prefix: str | Path | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    image, _, labels = load_dataset(load_yaml(config["dataset_config"]))
    device = resolve_device(config.get("device", "cuda:0"))
    model = load_model(config, labels, run_dir / "checkpoints" / f"run_{run_index}" / "model_best.pth", device)
    probabilities = infer_full_image(model, image, config, len(labels), device)
    prefix = Path(output_prefix) if output_prefix else run_dir / "full_prediction"
    return save_label_outputs(np.argmax(probabilities, axis=-1).astype(np.uint8) + 1, prefix, config.get("palette"))
