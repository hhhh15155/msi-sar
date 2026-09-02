"""Generate VBE-Net configs by cloning the repository's current protocol."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("yrd", "yrd2509new", "grss07")
SHOTS = (5, 10, 20, 50, 100, 150, 200)


def convert_reference(reference: dict) -> dict:
    config = copy.deepcopy(reference)
    config["model"] = "vbenet"
    config["ms_channels"] = int(config.pop("hsi_channels"))
    config["sar_channels"] = int(config.pop("aux_channels"))
    config.pop("emb_dim", None)
    config.pop("depth", None)
    config.pop("drop_path_rate", None)
    config.update(
        {
            "width": 64,
            "encoder_depth": 5,
            "groups": 8,
            "expansion": 4,
            "lambda_proto": 1.0,
            "tau_r": 0.3,
            "tau_c": 0.1,
            "inner_iters": 3,
            "outer_updates": 1,
            "modality_dropout": 0.1,
            "geometry_eps": 0.0001,
        }
    )
    return config


def generate_configs(
    output_dir: Path = ROOT / "configs",
    reference_dir: Path = ROOT / "configs",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    reference_names = [
        f"mghofnet_{dataset}_fs{shot}.yaml"
        for dataset in DATASETS
        for shot in SHOTS
    ] + ["mghofnet_grss07_custom.yaml"]
    for reference_name in reference_names:
        reference_path = reference_dir / reference_name
        reference = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
        target_name = reference_name.replace("mghofnet_", "vbenet_", 1)
        target = output_dir / target_name
        target.write_text(
            yaml.safe_dump(
                convert_reference(reference),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        generated.append(target)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate aligned VBE-Net experiment configs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs")
    args = parser.parse_args()
    for path in generate_configs(args.output_dir):
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
