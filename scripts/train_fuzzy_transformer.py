from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MSI/SAR fuzzy Transformer.")
    parser.add_argument("--config", default="configs/fuzzy_transformer_grss07_custom.yaml")
    args = parser.parse_args()
    from baselines.fuzzy_transformer.train import run_training

    print(f"finished: {run_training(args.config)}")


if __name__ == "__main__":
    main()
