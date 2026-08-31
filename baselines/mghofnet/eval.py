from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FusionPatchDataset, load_dataset
from .infer import infer_full_image, save_label_outputs
from .io import load_yaml, save_json
from .metrics import classification_metrics
from .model import MGHOFNet
from .train import resolve_device


def load_model(config: dict, labels: list[str], checkpoint: Path, device: torch.device) -> MGHOFNet:
    model = MGHOFNet(
        in_ch_hsi=int(config.get("hsi_channels", 10)),
        in_ch_lidar=int(config.get("aux_channels", 4)),
        num_classes=len(labels),
        emb_dim=int(config.get("emb_dim", 128)),
        depth=int(config.get("depth", 2)),
        drop_path_rate=float(config.get("drop_path_rate", 0.2)),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def predict_labeled_samples(
    model: torch.nn.Module,
    image: np.ndarray,
    gt: np.ndarray,
    patch_size: int,
    hsi_channels: int,
    aux_channels: int,
    batch_size: int,
    device: torch.device,
    pad_mode: str = "constant",
) -> np.ndarray:
    dataset = FusionPatchDataset(
        image,
        gt,
        patch_size,
        hsi_channels=hsi_channels,
        aux_channels=aux_channels,
        data_aug=False,
        pad_mode=pad_mode,
    )
    labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    prediction = np.full_like(gt, fill_value=-1)
    offset = 0
    model.eval()
    with torch.no_grad():
        for hsi, aux, _ in tqdm(labeled_loader, desc="predict labeled samples"):
            hsi = hsi.to(device)
            aux = aux.to(device)
            outputs = model(hsi, aux)
            batch_prediction = torch.argmax(outputs, dim=1).cpu().numpy()
            batch_indices = dataset.indices[offset : offset + len(batch_prediction)]
            for (x, y), label in zip(batch_indices, batch_prediction):
                prediction[x - dataset.pad_size, y - dataset.pad_size] = int(label)
            offset += len(batch_prediction)
    return prediction


def save_labeled_maps(run_dir: str | Path, run_index: int = 0) -> dict:
    run_dir = Path(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    dataset_config = load_yaml(config["dataset_config"])
    image, gt, labels = load_dataset(dataset_config)
    device = resolve_device(config.get("device", "cuda:0"))
    checkpoint = run_dir / "checkpoints" / f"run_{run_index}" / "model_best.pth"
    model = load_model(config, labels, checkpoint, device)
    eval_config = config.get("eval", {})
    palette = config.get("palette")
    batch_size = int(eval_config.get("batch_size", config.get("test_batch_size", 1024)))
    hsi_channels = int(config.get("hsi_channels", 10))
    aux_channels = int(config.get("aux_channels", 4))
    pad_mode = config.get("pad_mode", "constant")

    prediction = predict_labeled_samples(
        model,
        image,
        gt,
        int(config["patch_size"]),
        hsi_channels,
        aux_channels,
        batch_size,
        device,
        pad_mode=pad_mode,
    )
    mask = gt >= 0
    labeled_gt = np.zeros_like(gt, dtype=np.uint8)
    labeled_prediction = np.zeros_like(gt, dtype=np.uint8)
    labeled_gt[mask] = gt[mask].astype(np.uint8) + 1
    labeled_prediction[mask] = prediction[mask].astype(np.uint8) + 1

    metrics = classification_metrics(prediction, gt, len(labels))
    metrics["run"] = run_index
    metrics["class_names"] = labels
    output = {"runs": [metrics]}
    save_json(output, run_dir / "eval_metrics.json")

    paths = {}
    if eval_config.get("save_labeled_maps", True):
        gt_paths = save_label_outputs(labeled_gt, run_dir / "labeled_gt", palette)
        prediction_paths = save_label_outputs(labeled_prediction, run_dir / "labeled_prediction", palette)
        paths = {
            "labeled_gt": [str(path) for path in gt_paths],
            "labeled_prediction": [str(path) for path in prediction_paths],
        }
    output["outputs"] = paths
    save_json(output, run_dir / "eval_metrics.json")
    return output


def save_full_map(run_dir: str | Path, run_index: int = 0, output_prefix: str | Path | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    dataset_config = load_yaml(config["dataset_config"])
    image, _, labels = load_dataset(dataset_config)
    device = resolve_device(config.get("device", "cuda:0"))
    checkpoint = run_dir / "checkpoints" / f"run_{run_index}" / "model_best.pth"
    model = load_model(config, labels, checkpoint, device)
    probabilities = infer_full_image(
        model,
        image,
        int(config["patch_size"]),
        len(labels),
        int(config.get("hsi_channels", 10)),
        int(config.get("aux_channels", 4)),
        device,
        batch_size=int(config.get("infer", {}).get("batch_size", 64)),
        pad_mode=config.get("pad_mode", "constant"),
    )
    prediction = np.argmax(probabilities, axis=-1).astype(np.uint8) + 1
    prefix = Path(output_prefix) if output_prefix else run_dir / "full_prediction"
    return save_label_outputs(prediction, prefix, config.get("palette"))
