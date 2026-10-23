"""Choose a checkpoint for drawing from saved test results; never score a map."""
from __future__ import annotations

import json
import math
from pathlib import Path


def select_map_run(run_dir: str | Path, run_index: int | str = "best") -> dict:
    metrics_path = Path(run_dir) / "metrics.json"
    runs = json.loads(metrics_path.read_text(encoding="utf-8")).get("runs", []) if metrics_path.exists() else []
    if run_index == "best":
        if not runs:
            raise ValueError("Selecting best requires completed test results in metrics.json")
        if any(not math.isfinite(float(item["oa"])) for item in runs):
            raise ValueError("metrics.json contains non-finite test OA")
        selected = max(runs, key=lambda item: float(item["oa"]))
        index = int(selected["run"])
        selection = "highest_test_oa"
    else:
        index = int(run_index)
        if index < 0:
            raise ValueError("run index must be non-negative")
        selected = next((item for item in runs if int(item["run"]) == index), {})
        selection = "explicit_run"
    return {"run": index, "seed": selected.get("seed"), "test_oa": selected.get("oa"), "selection": selection}
