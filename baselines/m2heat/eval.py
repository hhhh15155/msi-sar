from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PatchDataset, load_dataset
from .infer import save_label_outputs
from baselines.prediction import infer_full_image, predict_labeled_samples
from .io import load_yaml, save_json
from .metrics import classification_metrics
from .model import M2Heat
from .train import resolve_device
from baselines.map_selection import select_map_run
from baselines.protocol import effective_dataset_config


def load_model(config: dict, labels: list[str], checkpoint: Path, device: torch.device) -> M2Heat:
    model = M2Heat(
        num_classes=len(labels),
        patch_size=int(config["patch_size"]),
        spectral_channels=int(config.get("spectral_channels", 10)),
        sar_channels=int(config.get("sar_channels", 4)),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model



def evaluate_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    dataset_config = load_yaml(run_dir / "dataset_config.yaml" if (run_dir / "dataset_config.yaml").exists() else config["dataset_config"])
    image, gt, labels = load_dataset(effective_dataset_config(dataset_config, config))
    device = resolve_device(config.get("device", "cuda:0"))
    results = []
    for split_file in sorted((run_dir / "splits").glob("run_*.npz")):
        run_index = int(split_file.stem.split("_")[-1])
        model = load_model(config, labels, run_dir / "checkpoints" / f"run_{run_index}" / "model_best.pth", device)
        with np.load(split_file) as split:
            test_gt = split["test_gt"].copy()
        prediction = predict_labeled_samples(
            model, image, test_gt,
            {**config, "eval": {"batch_size": int(config.get("test_batch_size", 1024))}}, device,
        )
        metrics = classification_metrics(prediction, test_gt, len(labels))
        metrics["run"] = run_index
        metrics["class_names"] = labels
        results.append(metrics)
    output = {"runs": results}
    save_json(output, run_dir / "eval_metrics.json")
    return output



def save_labeled_maps(run_dir: str | Path, run_index: int | str = "best") -> dict:
    run_dir = Path(run_dir)
    selection = select_map_run(run_dir, run_index)
    run_index = selection["run"]
    config = load_yaml(run_dir / "config.yaml")
    dataset_config = load_yaml(run_dir / "dataset_config.yaml" if (run_dir / "dataset_config.yaml").exists() else config["dataset_config"])
    image, gt, labels = load_dataset(effective_dataset_config(dataset_config, config))
    device = resolve_device(config.get("device", "cuda:0"))
    model = load_model(config, labels, run_dir / "checkpoints" / f"run_{run_index}" / "model_best.pth", device)
    eval_config = config.get("eval", {})
    palette = config.get("palette")
    batch_size = int(eval_config.get("batch_size", config.get("test_batch_size", 1024)))

    prediction = predict_labeled_samples(model, image, gt, config, device)
    mask = gt >= 0
    labeled_gt = np.zeros_like(gt, dtype=np.uint8)
    labeled_prediction = np.zeros_like(gt, dtype=np.uint8)
    labeled_gt[mask] = gt[mask].astype(np.uint8) + 1
    labeled_prediction[mask] = prediction[mask].astype(np.uint8) + 1

    output = {"mode": "labeled", **selection, "labeled_count": int(mask.sum())}

    paths = {}
    if eval_config.get("save_labeled_maps", True):
        gt_paths = save_label_outputs(labeled_gt, run_dir / "labeled_gt", palette)
        prediction_paths = save_label_outputs(labeled_prediction, run_dir / "labeled_prediction", palette)
        paths = {
            "labeled_gt": [str(path) for path in gt_paths],
            "labeled_prediction": [str(path) for path in prediction_paths],
        }
    output["outputs"] = paths
    save_json(output, run_dir / "labeled_map_info.json")
    return output



def save_full_map(run_dir: str | Path, run_index: int | str = "best", output_prefix: str | Path | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    selection = select_map_run(run_dir, run_index)
    run_index = selection["run"]
    config = load_yaml(run_dir / "config.yaml")
    dataset_config = load_yaml(run_dir / "dataset_config.yaml" if (run_dir / "dataset_config.yaml").exists() else config["dataset_config"])
    image, _, labels = load_dataset(effective_dataset_config(dataset_config, config))
    device = resolve_device(config.get("device", "cuda:0"))
    model = load_model(config, labels, run_dir / "checkpoints" / f"run_{run_index}" / "model_best.pth", device)
    probabilities = infer_full_image(
        model,
        image,
        int(config["patch_size"]),
        len(labels),
        device,
        batch_size=int(config.get("infer", {}).get("batch_size", 64)),
        config=config,
    )
    prediction = np.argmax(probabilities, axis=-1).astype(np.uint8) + 1
    prefix = Path(output_prefix) if output_prefix else run_dir / "full_prediction"
    paths = save_label_outputs(prediction, prefix, config.get("palette"))
    save_json({"mode": "full", **selection, "outputs": [str(path) for path in paths]},
              prefix.parent / (prefix.name + "_info.json"))
    return paths
