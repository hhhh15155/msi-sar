from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_SCRIPTS = {
    "dfinet": ROOT / "scripts" / "train_eval_dfinet.py",
    "frekfuse": ROOT / "scripts" / "train_eval_frekfuse.py",
    "mghofnet": ROOT / "scripts" / "train_eval_mghofnet.py",
    "msfmamba": ROOT / "scripts" / "train_eval_msfmamba.py",
    "softformer": ROOT / "scripts" / "train_eval_softformer.py",
}
DEFAULT_SHOTS = [5, 10, 20, 50, 100, 150, 200]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YRD2509NEW few-shot train/test experiments (no validation, final test only)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SCRIPTS),
        choices=sorted(MODEL_SCRIPTS),
        help="Models to run. Default: all four models.",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=DEFAULT_SHOTS,
        choices=DEFAULT_SHOTS,
        help="Per-class training counts. Default: 5 10 20 50 100 150 200.",
    )
    args = parser.parse_args()

    for model in args.models:
        script = MODEL_SCRIPTS[model]
        for shot in args.shots:
            config = ROOT / "configs" / f"{model}_yrd2509new_fs{shot}.yaml"
            if not config.exists():
                raise FileNotFoundError(config)
            command = [sys.executable, str(script), "--config", str(config)]
            print("running:", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
