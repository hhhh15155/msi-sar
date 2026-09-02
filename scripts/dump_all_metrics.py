"""完整指标输出"""
import json, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

models = ["dfinet", "frekfuse", "mghofnet", "msfmamba", "softformer", "vbenet"]
shots = [5, 10, 20, 50, 100, 150, 200]

for model in models:
    print("")
    print("━" * 100)
    print(f"  {model}")
    print("━" * 100)

    # YRD
    print(f"\n  YRD (8-class) — fs100+ per-class")
    # header
    ref = ROOT / f"runs_fewshot/fs5/{model}/yrd/run_001/metrics.json"
    if ref.exists():
        m = json.loads(ref.read_text())
        classes = m["runs"][0]["class_names"]
        print(f"  {'Shot':<6} {'OA':>7} {'AA':>7} {'κ':>7}  " +
              "  ".join(f"{c:>20}" for c in classes))
        print("  " + "-" * 96)

    for shot in shots:
        f = ROOT / f"runs_fewshot/fs{shot}/{model}/yrd/run_001/metrics.json"
        if not f.exists():
            print(f"  fs{shot:<4} {'-':>7} {'-':>7} {'-':>7}")
            continue
        m = json.loads(f.read_text())
        agg = m["aggregate"]
        all_acc = np.array([r["class_acc"] for r in m["runs"]])
        mean_acc = np.mean(all_acc, axis=0)
        per_class = "  ".join(f"{x*100:5.1f}" for x in mean_acc)
        print(f"  fs{shot:<4} {agg['oa_mean']*100:6.2f} {agg['aa_mean']*100:6.2f} "
              f"{agg['kappa_mean']*100:6.2f}  {per_class}")

    # YRD2509
    print(f"\n  YRD2509 (9-class) — fs100+ per-class")
    ref9 = ROOT / f"runs_fewshot/fs5/{model}/yrd2509/run_001/metrics.json"
    if ref9.exists():
        m = json.loads(ref9.read_text())
        classes9 = m["runs"][0]["class_names"]
        print(f"  {'Shot':<6} {'OA':>7} {'AA':>7} {'κ':>7}  " +
              "  ".join(f"{c:>20}" for c in classes9))
        print("  " + "-" * 106)

    for shot in shots:
        f = ROOT / f"runs_fewshot/fs{shot}/{model}/yrd2509/run_001/metrics.json"
        if not f.exists():
            print(f"  fs{shot:<4} {'-':>7} {'-':>7} {'-':>7}")
            continue
        m = json.loads(f.read_text())
        agg = m["aggregate"]
        all_acc = np.array([r["class_acc"] for r in m["runs"]])
        mean_acc = np.mean(all_acc, axis=0)
        per_class = "  ".join(f"{x*100:5.1f}" for x in mean_acc)
        print(f"  fs{shot:<4} {agg['oa_mean']*100:6.2f} {agg['aa_mean']*100:6.2f} "
              f"{agg['kappa_mean']*100:6.2f}  {per_class}")
