from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw hslinets predictions; labeled samples by default.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-index", default="best", help="best test OA, or a zero-based run index")
    parser.add_argument("--mode", choices=("labeled", "full"), default="labeled")
    parser.add_argument("--output", default=None, help="Output prefix for explicit full-image inference")
    args = parser.parse_args()
    if args.output is not None and args.mode != "full":
        parser.error("--output is supported only with --mode full")
    from baselines.hslinets.eval import save_full_map, save_labeled_maps
    if args.mode == "full":
        for path in save_full_map(args.run, args.run_index, args.output):
            print(f"saved: {path}")
    else:
        print(json.dumps(save_labeled_maps(args.run, args.run_index), indent=2))


if __name__ == "__main__":
    main()
