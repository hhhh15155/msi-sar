"""One-off migration: switch few-shot configs to no-validation + test-every-20-epochs.

- split.method: fixed_counts -> fixed_train_counts (val samples join the test set)
- drop val_counts / val_count_per_class / train_ratio / val_ratio / test_ratio / use_validation
- add use_validation: false, select_best_by: test, test_interval: 20 after the split block
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DROP_LINE_PREFIXES = (
    "val_counts:",
    "val_count_per_class:",
    "train_ratio:",
    "val_ratio:",
    "test_ratio:",
    "use_validation:",
)


def migrate(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    changed = False
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in DROP_LINE_PREFIXES):
            changed = True
            continue
        if stripped.startswith("method: fixed_counts"):
            out.append(line.replace("method: fixed_counts", "method: fixed_train_counts"))
            changed = True
            continue
        out.append(line)
        # right after the split block's train-count line, append the new strategy keys
        if stripped.startswith("train_counts:") or stripped.startswith("train_count_per_class:"):
            out.append("use_validation: false")
            out.append("select_best_by: test")
            out.append("test_interval: 20")
            changed = True
    if changed:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    targets = sorted(ROOT.glob("configs/*_fs*.yaml"))
    for path in targets:
        if migrate(path):
            print(f"updated: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
