from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FreKFuse baseline.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args()

    from baselines.frekfuse.eval import save_labeled_maps

    print(json.dumps(save_labeled_maps(args.run, args.run_index), indent=2))


if __name__ == "__main__":
    main()
