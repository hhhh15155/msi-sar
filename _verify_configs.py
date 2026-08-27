"""Verify all few-shot configs follow the no-validation + test-every-20-epochs strategy."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
ERRORS: list[str] = []

for path in sorted(ROOT.glob("configs/*_fs*.yaml")):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = path.name
    split = config.get("split", {})
    if split.get("method") != "fixed_train_counts":
        ERRORS.append(f"{name}: split.method = {split.get('method')!r}")
    if "val_counts" in split or "val_count_per_class" in split:
        ERRORS.append(f"{name}: stale val split keys present")
    if config.get("use_validation") is not False:
        ERRORS.append(f"{name}: use_validation = {config.get('use_validation')!r}")
    if config.get("select_best_by") != "test":
        ERRORS.append(f"{name}: select_best_by = {config.get('select_best_by')!r}")
    if config.get("test_interval") != 20:
        ERRORS.append(f"{name}: test_interval = {config.get('test_interval')!r}")
    if "train_ratio" in config or "val_ratio" in config or "test_ratio" in config:
        ERRORS.append(f"{name}: stale ratio keys present")
    train_counts = split.get("train_counts", split.get("train_count_per_class"))
    if train_counts is None:
        ERRORS.append(f"{name}: missing train counts")

if ERRORS:
    print(f"FAILED ({len(ERRORS)} issues):")
    for error in ERRORS:
        print(" -", error)
    raise SystemExit(1)
print(f"OK: {len(list(ROOT.glob('configs/*_fs*.yaml')))} configs verified")
