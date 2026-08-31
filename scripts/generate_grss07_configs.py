"""Generate the five-model few-shot experiment grid for GRSS-DFC-2007."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SHOTS = (5, 10, 20, 50, 100, 150, 200)
MODELS = ("dfinet", "frekfuse", "mghofnet", "msfmamba", "softformer")
PALETTE = [
    [230, 25, 75],
    [255, 225, 25],
    [245, 130, 48],
    [0, 130, 200],
    [60, 180, 75],
]


def _common(model: str, shot: int) -> dict:
    return {
        "model": model,
        "dataset": "grss07",
        "dataset_config": "configs/datasets/grss07.yaml",
        "output_root": f"runs_fewshot/fs{shot}",
        "device": "cuda:0",
        "patch_size": 11,
        "split": {
            "method": "fixed_train_counts",
            "train_count_per_class": shot,
        },
        "use_validation": False,
        "select_best_by": "test",
        "test_interval": 20,
        "epochs": 200,
        "num_runs": 5,
        "test_batch_size": 1024,
        "seeds": [202201, 202202, 202203, 202204, 202205],
        "eval": {"enabled": True, "run_index": "best", "batch_size": 1024, "save_labeled_maps": True},
        "infer": {"batch_size": 512, "save_full_map": True},
        "palette": {"background": [0, 0, 0], "colors": PALETTE},
    }


def _model_config(model: str, shot: int) -> dict:
    config = _common(model, shot)
    if model == "dfinet":
        config.update(
            spectral_channels=6,
            sar_channels=1,
            batch_size=128,
            learning_rate=0.001,
            weight_decay=0.0001,
        )
    elif model == "frekfuse":
        config.update(
            ms_channels=6,
            sar_channels=1,
            embed_dim=128,
            spline_order=2,
            dropout=0.5,
            lite=True,
            batch_size=128,
            learning_rate=0.0005,
            weight_decay=0.0001,
            eta_min=0.000001,
            gradient_clip=1.0,
        )
    elif model == "mghofnet":
        config.update(
            hsi_channels=6,
            aux_channels=1,
            pad_mode="constant",
            emb_dim=128,
            depth=2,
            drop_path_rate=0.2,
            batch_size=128,
            data_aug=True,
            learning_rate=0.001,
            weight_decay=0.01,
            eta_min=0.000001,
        )
    elif model == "msfmamba":
        config.update(
            ms_channels=6,
            sar_channels=1,
            num_layers=1,
            d_state=16,
            expand=0.75,
            batch_size=128,
            data_aug=True,
            learning_rate=0.0001,
            weight_decay=0.0,
        )
    elif model == "softformer":
        config.update(
            model_img_size=8,
            hsi_channels=6,
            aux_channels=1,
            pad_mode="constant",
            stem_chans=16,
            embed_dim=[24, 48, 96],
            num_heads=[4, 8, 16],
            mlp_ratio=4.0,
            depths=[2, 8, 2],
            drop=0.0,
            attn_drop=0.1,
            drop_path_rate=0.1,
            use_isa=True,
            batch_size=128,
            data_aug=True,
            learning_rate=0.001,
            weight_decay=0.0001,
            eta_min=0.0,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")
    return config


def generate_configs(output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for model in MODELS:
        for shot in SHOTS:
            target = output_dir / f"{model}_grss07_fs{shot}.yaml"
            target.write_text(
                yaml.safe_dump(_model_config(model, shot), sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            generated.append(target)
    return generated


def main() -> None:
    for path in generate_configs(ROOT / "configs"):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
