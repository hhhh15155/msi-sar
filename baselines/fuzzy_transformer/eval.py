from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from baselines.dfinet.dataset import load_dataset
from baselines.frekfuse.eval import predict_labeled_samples
from baselines.frekfuse.io import load_yaml, resolve_path, save_json
from baselines.frekfuse.metrics import classification_metrics
from baselines.frekfuse.train import resolve_device
from .infer import infer_full_image, save_label_outputs
from .train import build_model


def load_run(run_dir: str | Path, run_index: int):
    run_dir = resolve_path(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    image, gt, labels = load_dataset(load_yaml(resolve_path(config["dataset_config"])))
    device = resolve_device(config.get("device", "cuda:0"))
    model = build_model(config, len(labels)).to(device)
    checkpoint = run_dir / "checkpoints" / f"run_{run_index}" / "model_last.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return run_dir, config, image, gt, labels, device, model


def save_labeled_maps(run_dir: str | Path, run_index: int = 0) -> dict:
    run_dir, config, image, gt, labels, device, model = load_run(run_dir, run_index)
    with np.load(run_dir / "splits" / f"run_{run_index}.npz") as split:
        test_gt = split["test_gt"]
    # Map all annotated pixels for inspection, but score only saved test pixels.
    prediction = predict_labeled_samples(model, image, gt, config, device)
    metrics = classification_metrics(prediction, test_gt, len(labels))
    metrics.update({"run": run_index, "class_names": labels, "evaluation_scope": "held_out_test"})
    paths = {}
    if config.get("eval", {}).get("save_labeled_maps", True):
        mask = gt >= 0
        labeled_gt = np.where(mask, gt + 1, 0).astype(np.uint8)
        labeled_prediction = np.where(mask, prediction + 1, 0).astype(np.uint8)
        prefix = run_dir / f"eval_run_{run_index}"
        for name, values in (("labeled_gt", labeled_gt), ("labeled_prediction", labeled_prediction)):
            paths[name] = [str(path) for path in save_label_outputs(values, prefix.with_name(prefix.name + "_" + name),
                                                                   config.get("palette"))]
    output = {"runs": [metrics], "outputs": paths}
    save_json(output, run_dir / f"eval_metrics_run_{run_index}.json")
    return output


def save_full_map(run_dir: str | Path, run_index: int = 0,
                  output_prefix: str | Path | None = None) -> list[Path]:
    run_dir, config, image, _, labels, device, model = load_run(run_dir, run_index)
    probabilities = infer_full_image(model, image, int(config["patch_size"]), len(labels), device,
                                     int(config["ms_channels"]), int(config["sar_channels"]),
                                     int(config.get("infer", {}).get("batch_size", 128)))
    prediction = probabilities.argmax(axis=-1).astype(np.uint8) + 1
    prefix = resolve_path(output_prefix) if output_prefix else run_dir / f"full_prediction_run_{run_index}"
    return save_label_outputs(prediction, prefix, config.get("palette"))
