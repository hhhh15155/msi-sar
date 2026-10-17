"""Generate fuzzy Transformer configs using the project's existing split protocols."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("yrd", "yrd2509new", "grss07")
SHOTS = (5, 10, 20, 50, 100, 150, 200)
# Keep the checked-in Transformer configuration as the architecture template;
# borrow dataset/split/training settings, not FreKFuse architecture settings.
PROTOCOL_KEYS = (
    "dataset", "dataset_config", "output_root", "device", "patch_size", "split",
    "epochs", "num_runs", "seeds", "ms_channels", "sar_channels", "batch_size",
    "learning_rate", "weight_decay", "eta_min", "gradient_clip", "palette",
)


def generate_configs(output_dir: Path, datasets: tuple[str, ...] = DATASETS,
                     shots: tuple[int, ...] = SHOTS, overwrite: bool = False) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load((ROOT / "configs/fuzzy_transformer_yrd2509new_fs50.yaml").read_text(encoding="utf-8"))
    references = []
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        for shot in shots:
            if shot not in SHOTS:
                raise ValueError(f"Unsupported per-class count: {shot}")
            references.append(ROOT / "configs" / f"frekfuse_{dataset}_fs{shot}.yaml")
    if "grss07" in datasets:
        references.append(ROOT / "configs/frekfuse_grss07_custom.yaml")
    generated = []
    for reference_path in references:
        target = output_dir / reference_path.name.replace("frekfuse_", "fuzzy_transformer_", 1)
        if target.exists() and not overwrite:
            continue
        reference = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
        config = deepcopy(template)
        for key in PROTOCOL_KEYS:
            if key in reference:
                config[key] = deepcopy(reference[key])
        config["model"] = "fuzzy_transformer"
        metadata = yaml.safe_load((ROOT / config["dataset_config"]).read_text(encoding="utf-8"))
        num_classes = int(metadata["num_classes"])
        if len(config["palette"]["colors"]) < num_classes:
            raise ValueError(f"Not enough class colors in {reference_path}")
        config["palette"]["colors"] = config["palette"]["colors"][:num_classes]
        target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        generated.append(target)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing Transformer configs.")
    args = parser.parse_args()
    paths = generate_configs(args.output_dir, tuple(args.datasets), tuple(args.shots), args.overwrite)
    for path in paths:
        print(path)
    print(f"Wrote {len(paths)} configs; existing files are preserved unless --overwrite is supplied.")


if __name__ == "__main__":
    main()
