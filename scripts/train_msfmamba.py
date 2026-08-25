from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MSFMamba baseline.")
    parser.add_argument("--config", default="configs/msfmamba_yrd_fs5.yaml")
    args = parser.parse_args()
    from baselines.msfmamba.train import run_training
    print(f"finished: {run_training(args.config)}")


if __name__ == "__main__":
    main()
