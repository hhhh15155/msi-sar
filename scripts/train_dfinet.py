from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DFI-Net baseline.")
    parser.add_argument("--config", default="configs/dfinet_yrd_fs5.yaml")
    args = parser.parse_args()

    from baselines.dfinet.train import run_training

    run_dir = run_training(args.config)
    print(f"finished: {run_dir}")


if __name__ == "__main__":
    main()
