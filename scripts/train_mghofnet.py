from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MGHOFNet baseline.")
    parser.add_argument("--config", default="configs/mghofnet_yrd_fs5.yaml")
    args = parser.parse_args()

    from baselines.mghofnet.train import run_training

    run_dir = run_training(args.config)
    print(f"finished: {run_dir}")


if __name__ == "__main__":
    main()
