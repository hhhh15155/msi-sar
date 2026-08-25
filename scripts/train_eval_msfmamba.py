from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MSFMamba and export labeled evaluation maps.")
    parser.add_argument("--config", default="configs/msfmamba_yrd_fs5.yaml")
    args = parser.parse_args()
    from baselines.msfmamba.eval import save_labeled_maps
    from baselines.msfmamba.io import load_yaml
    from baselines.msfmamba.train import run_training

    run_dir = run_training(args.config)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"finished: {run_dir}")
    for result in metrics["runs"]:
        print("run={run} seed={seed} OA={oa:.2f} AA={aa:.2f} Kappa={kappa:.2f}".format(**result))
    if metrics.get("aggregate"):
        print("mean OA={oa_mean:.2f}+-{oa_std:.2f} AA={aa_mean:.2f}+-{aa_std:.2f} Kappa={kappa_mean:.2f}+-{kappa_std:.2f}".format(**metrics["aggregate"]))
    eval_config = load_yaml(run_dir / "config.yaml").get("eval", {})
    if eval_config.get("enabled", True):
        selected = max(metrics["runs"], key=lambda item: item["oa"])["run"] if eval_config.get("run_index", "best") == "best" else int(eval_config["run_index"])
        for group, paths in save_labeled_maps(run_dir, int(selected)).get("outputs", {}).items():
            for path in paths:
                print(f"{group}: {path}")


if __name__ == "__main__":
    main()
