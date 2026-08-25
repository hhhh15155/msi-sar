from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a FreKFuse prediction map.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from baselines.frekfuse.eval import save_full_map

    for path in save_full_map(args.run, args.run_index, args.output):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
