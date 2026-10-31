from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate s2atnet on saved test splits only.")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    from baselines.s2atnet.eval import evaluate_run
    print(json.dumps(evaluate_run(args.run), indent=2))


if __name__ == "__main__":
    main()
