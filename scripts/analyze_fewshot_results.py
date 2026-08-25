"""Summarise migrated few-shot runs without changing experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_fewshot"
OUT = ROOT / "experiment_logs"
MODELS = ("dfinet", "frekfuse", "mghofnet", "softformer")
DATASETS = ("yrd", "yrd2509", "yrd2509new")
SHOTS = (5, 10, 20, 50, 100, 150, 200)


def main() -> None:
    rows: list[dict[str, object]] = []
    for shot in SHOTS:
        for model in MODELS:
            for dataset in DATASETS:
                path = RUNS / f"fs{shot}" / model / dataset / "run_001" / "metrics.json"
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                aggregate = payload["aggregate"]
                rows.append({"shot": shot, "model": model, "dataset": dataset, "runs": len(payload["runs"]),
                             "oa_mean": aggregate["oa_mean"], "oa_std": aggregate["oa_std"],
                             "aa_mean": aggregate["aa_mean"], "aa_std": aggregate["aa_std"],
                             "kappa_mean": aggregate["kappa_mean"], "kappa_std": aggregate["kappa_std"],
                             "path": str(path.relative_to(ROOT))})
    OUT.mkdir(exist_ok=True)
    fields = list(rows[0])
    with (OUT / "fewshot_results_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["Few-shot results summary (mean +/- std, %, five runs per cell)\n"]
    for dataset in DATASETS:
        lines.append(f"{dataset.upper()}\n")
        lines.append("shot | " + " | ".join(MODELS) + "\n")
        lines.append("--- | " + " | ".join("---" for _ in MODELS) + "\n")
        for shot in SHOTS:
            values = []
            for model in MODELS:
                row = next((row for row in rows if row["dataset"] == dataset and row["shot"] == shot and row["model"] == model), None)
                values.append("--" if row is None else f"{row['oa_mean']:.2f} +/- {row['oa_std']:.2f}")
            lines.append(f"{shot} | " + " | ".join(values) + "\n")
        lines.append("\n")
    (OUT / "fewshot_results_summary.md").write_text("".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} records")


if __name__ == "__main__":
    main()
