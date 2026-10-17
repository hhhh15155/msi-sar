from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fuzzy Transformer and export labeled maps.")
    parser.add_argument("--config", default="configs/fuzzy_transformer_grss07_custom.yaml")
    args = parser.parse_args()
    from baselines.frekfuse.io import load_yaml
    from baselines.fuzzy_transformer.eval import save_labeled_maps
    from baselines.fuzzy_transformer.train import run_training

    run_dir = run_training(args.config)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"finished: {run_dir}")
    for result in metrics["runs"]:
        print("run={run} seed={seed} OA={oa:.2f} AA={aa:.2f} Kappa={kappa:.2f}".format(**result))
        for name, accuracy in zip(result["class_names"], result["class_acc"]):
            print(f"  {name}: {accuracy:.2f}")
    if metrics.get("aggregate"):
        print("mean OA={oa_mean:.2f}+-{oa_std:.2f} AA={aa_mean:.2f}+-{aa_std:.2f} "
              "Kappa={kappa_mean:.2f}+-{kappa_std:.2f}".format(**metrics["aggregate"]))
    eval_config = load_yaml(run_dir / "config.yaml").get("eval", {})
    if eval_config.get("enabled", True):
        # Fixed seed index; do not select a run using held-out test accuracy.
        run_index = int(eval_config.get("run_index", 0))
        output = save_labeled_maps(run_dir, run_index)
        for paths in output["outputs"].values():
            for path in paths:
                print(f"saved: {path}")


if __name__ == "__main__":
    main()
