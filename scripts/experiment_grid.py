from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


MODELS = ("dfinet", "frekfuse", "mghofnet", "msfmamba", "softformer")
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


def iter_experiments(datasets: tuple[str, ...] = DATASETS) -> Iterator[Experiment]:
    for model in MODELS:
        for shot in SHOTS:
            for dataset in datasets:
                if dataset not in AVAILABLE_DATASETS:
                    raise ValueError(f"Unsupported dataset: {dataset}")
                yield Experiment(model=model, dataset=dataset, shot=shot)


def is_complete(root: Path, experiment: Experiment) -> bool:
    result_dir = root / f"runs_fewshot/fs{experiment.shot}/{experiment.model}/{experiment.dataset}"
    return any(result_dir.glob("run_*/metrics.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the selected few-shot experiment grid as pipe-separated rows.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", choices=AVAILABLE_DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    root = args.root.resolve()

    for experiment in iter_experiments(tuple(args.datasets)):
        for relative_path in (experiment.script, experiment.config):
            if not (root / relative_path).is_file():
                raise FileNotFoundError(f"Missing required experiment file: {relative_path}")
        done = 1 if is_complete(root, experiment) else 0
        print(f"{experiment.script}|{experiment.config}|{experiment.name}|{done}")


if __name__ == "__main__":
    main()
