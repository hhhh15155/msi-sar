from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DFI-Net and print test metrics.")
    parser.add_argument("--config", default="configs/dfinet_yrd_fs5.yaml")
    args = parser.parse_args()

    from baselines.dfinet.eval import save_labeled_maps
    from baselines.dfinet.train import run_training

    run_dir = run_training(args.config)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    print(f"finished: {run_dir}")
    for result in metrics["runs"]:
        print("run={run} seed={seed} OA={oa:.2f} AA={aa:.2f} Kappa={kappa:.2f}".format(**result))
        for class_name, class_acc in zip(result["class_names"], result["class_acc"]):
            print(f"  {class_name}: {class_acc:.2f}")

    aggregate = metrics.get("aggregate")
    if aggregate:
        print(
            "mean OA={oa_mean:.2f}+-{oa_std:.2f} AA={aa_mean:.2f}+-{aa_std:.2f} "
            "Kappa={kappa_mean:.2f}+-{kappa_std:.2f}".format(**aggregate)
        )

    eval_config = {}
    run_config_path = run_dir / "config.yaml"
    if run_config_path.exists():
        from baselines.dfinet.io import load_yaml

        eval_config = load_yaml(run_config_path).get("eval", {})

    if eval_config.get("enabled", True):
        requested_run = eval_config.get("run_index", "best")
        if requested_run == "best":
            selected = max(metrics["runs"], key=lambda item: item["oa"])
            run_index = int(selected["run"])
        else:
            run_index = int(requested_run)
        print(f"selected run for labeled maps: {run_index}")
        eval_output = save_labeled_maps(run_dir, run_index=run_index)
        for group, paths in eval_output.get("outputs", {}).items():
            for path in paths:
                print(f"{group}: {path}")


if __name__ == "__main__":
    main()
