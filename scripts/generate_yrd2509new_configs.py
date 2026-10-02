"""Verify the checked-in YRD2509NEW few-shot config grid."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
MODELS = ("dfinet", "frekfuse", "mghofnet", "softformer")
SHOTS = (5, 10, 20, 50, 100, 150, 200)


def main() -> None:
    for model in MODELS:
        for shot in SHOTS:
            target = CONFIGS / f"{model}_yrd2509new_fs{shot}.yaml"
            if not target.exists():
                raise FileNotFoundError(f"Missing checked-in config: {target}")
            print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
