from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


MODELS = ("dfinet", "frekfuse", "mghofnet", "msfmamba", "softformer")
AVAILABLE_MODELS = MODELS + ("fuzzy_transformer",)
DATASETS = ("yrd", "yrd2509new", "grss07")
AVAILABLE_DATASETS = DATASETS
SHOTS = (5, 10, 20, 50, 100, 150, 200)


@dataclass(frozen=True)
class Experiment:
    model: str
    dataset: str
    shot: int

    @property
    def script(self) -> str:
        return f"scripts/train_eval_{self.model}.py"

    @property
    def config(self) -> str:
        return f"configs/{self.model}_{self.dataset}_fs{self.shot}.yaml"

    @property
    def name(self) -> str:
        return f"{self.model}_{self.dataset}_fs{self.shot}"


def iter_experiments(datasets: tuple[str, ...] = DATASETS,
                     models: tuple[str, ...] = MODELS) -> Iterator[Experiment]:
    for model in models:
        if model not in AVAILABLE_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        for shot in SHOTS:
            for dataset in datasets:
                if dataset not in AVAILABLE_DATASETS:
                    raise ValueError(f"Unsupported dataset: {dataset}")
                yield Experiment(model=model, dataset=dataset, shot=shot)


def is_complete(root: Path, experiment: Experiment) -> bool:
    result_dir = root / f"runs_fewshot/fs{experiment.shot}/{experiment.model}/{experiment.dataset}"
    metrics_paths = result_dir.glob("run_*/metrics.json")
    if experiment.model != "fuzzy_transformer":
        return any(metrics_paths)
    for path in metrics_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("completed") is True:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the selected few-shot experiment grid as pipe-separated rows.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", choices=AVAILABLE_DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=AVAILABLE_MODELS, default=list(MODELS))
    args = parser.parse_args()
    root = args.root.resolve()

    for experiment in iter_experiments(tuple(args.datasets), tuple(args.models)):
        for relative_path in (experiment.script, experiment.config):
            if not (root / relative_path).is_file():
                raise FileNotFoundError(f"Missing required experiment file: {relative_path}")
        done = 1 if is_complete(root, experiment) else 0
        print(f"{experiment.script}|{experiment.config}|{experiment.name}|{done}")


if __name__ == "__main__":
    main()
