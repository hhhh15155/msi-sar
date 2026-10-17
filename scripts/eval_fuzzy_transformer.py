from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved fuzzy Transformer on its held-out split.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args()
    from baselines.fuzzy_transformer.eval import save_labeled_maps

    print(json.dumps(save_labeled_maps(args.run, args.run_index), indent=2))


if __name__ == "__main__":
    main()
