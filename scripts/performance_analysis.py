"""完整性能分析"""
import json, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
models = ["dfinet", "frekfuse", "mghofnet", "softformer"]
shots = [5, 10, 20, 50, 100, 150, 200]

print("=" * 90)
print("1. YRD — OA 趋势 (fs5 → fs200)")
print("=" * 90)
print(f"{'Model':<12}", end="")
for s in shots:
    print(f"fs{s:>7}", end="")
print(f"  {'Δ(fs100→200)':>12}  {'饱和?':>6}")
print("-" * 90)

for model in models:
    print(f"{model:<12}", end="")
    vals = []
    for s in shots:
        f = ROOT / f"runs_fewshot/fs{s}/{model}/yrd/run_001/metrics.json"
        if f.exists():
            v = json.loads(f.read_text())["aggregate"]["oa_mean"] * 100
            vals.append(v)
            print(f"{v:7.2f}", end="")
        else:
            vals.append(None)
            print(f"   -   ", end="")
    if vals[4] and vals[6]:
        delta = vals[6] - vals[4]
        sat = "是" if delta < 2.0 else "否"
        print(f"  {delta:+12.2f}  {sat:>6}", end="")
    print()

print("\n" + "=" * 90)
print("2. YRD2509 — OA 趋势")
print("=" * 90)
print(f"{'Model':<12}", end="")
for s in shots:
    print(f"fs{s:>7}", end="")
print(f"  {'Δ(fs100→200)':>12}  {'饱和?':>6}")
print("-" * 90)

for model in models:
    print(f"{model:<12}", end="")
    vals = []
    for s in shots:
        f = ROOT / f"runs_fewshot/fs{s}/{model}/yrd2509/run_001/metrics.json"
        if f.exists():
            v = json.loads(f.read_text())["aggregate"]["oa_mean"] * 100
            vals.append(v)
            print(f"{v:7.2f}", end="")
        else:
            vals.append(None)
            print(f"   -   ", end="")
    if vals[4] and vals[6]:
        delta = vals[6] - vals[4]
        sat = "是" if delta < 2.0 else "否"
        print(f"  {delta:+12.2f}  {sat:>6}", end="")
    print()

# 2. 每类难度排名
print("\n" + "=" * 90)
print("3. 类别难度分析 (fs200 平均 per-class accuracy)")
print("=" * 90)

print("\nYRD (8-class):")
accs = {}
for model in models:
    f = ROOT / f"runs_fewshot/fs200/{model}/yrd/run_001/metrics.json"
    if f.exists():
        m = json.loads(f.read_text())
        accs[model] = np.mean([r["class_acc"] for r in m["runs"]], axis=0)
    else:
        accs[model] = None

classes_yrd = json.loads((ROOT / "runs_fewshot/fs5/dfinet/yrd/run_001/metrics.json").read_text())["runs"][0]["class_names"]

print(f"{'Class':<25}", end="")
for model in models:
    print(f"{model:>12}", end="")
print(f"  {'Avg':>8}  {'难度'}")
print("-" * 95)

for ci, cn in enumerate(classes_yrd):
    print(f"{cn:<25}", end="")
    vals = []
    for model in models:
        if accs[model] is not None:
            v = accs[model][ci] * 100
            vals.append(v)
            print(f"{v:12.1f}", end="")
        else:
            print(f"     -      ", end="")
    avg = np.mean(vals)
    diff = "★☆☆ 易" if avg > 95 else "★★☆ 中" if avg > 85 else "★★★ 难"
    print(f"  {avg:8.1f}  {diff}")

print("\nYRD2509 (9-class):")
accs9 = {}
for model in models:
    f = ROOT / f"runs_fewshot/fs200/{model}/yrd2509/run_001/metrics.json"
    if f.exists():
        m = json.loads(f.read_text())
        accs9[model] = np.mean([r["class_acc"] for r in m["runs"]], axis=0)
    else:
        accs9[model] = None

classes_yrd9 = json.loads((ROOT / "runs_fewshot/fs5/dfinet/yrd2509/run_001/metrics.json").read_text())["runs"][0]["class_names"]

print(f"{'Class':<25}", end="")
for model in models:
    print(f"{model:>12}", end="")
print(f"  {'Avg':>8}  {'难度'}")
print("-" * 95)

for ci, cn in enumerate(classes_yrd9):
    print(f"{cn:<25}", end="")
    vals = []
    for model in models:
        if accs9[model] is not None:
            v = accs9[model][ci] * 100
            vals.append(v)
            print(f"{v:12.1f}", end="")
        else:
            print(f"     -      ", end="")
    avg = np.mean(vals)
    diff = "★☆☆ 易" if avg > 95 else "★★☆ 中" if avg > 85 else "★★★ 难"
    print(f"  {avg:8.1f}  {diff}")

# 3. 模型稳定性
print("\n" + "=" * 90)
print("4. 模型稳定性 (fs100↔fs200 波动)")
print("=" * 90)

for model in models:
    for ds, dsname in [("yrd","YRD"), ("yrd2509","YRD2509")]:
        f100 = ROOT / f"runs_fewshot/fs100/{model}/{ds}/run_001/metrics.json"
        f150 = ROOT / f"runs_fewshot/fs150/{model}/{ds}/run_001/metrics.json"
        f200 = ROOT / f"runs_fewshot/fs200/{model}/{ds}/run_001/metrics.json"
        if f100.exists() and f150.exists() and f200.exists():
            oa100 = json.loads(f100.read_text())["aggregate"]["oa_mean"] * 100
            oa150 = json.loads(f150.read_text())["aggregate"]["oa_mean"] * 100
            oa200 = json.loads(f200.read_text())["aggregate"]["oa_mean"] * 100
            trend = "↗稳定增长" if oa200 > oa150 > oa100 else \
                    "↘下降" if oa200 < oa100 else \
                    "→波动"
            print(f"  {model} {dsname}: fs100={oa100:.1f} fs150={oa150:.1f} fs200={oa200:.1f} {trend}")

# 4. 边际收益
print("\n" + "=" * 90)
print("5. 样本边际收益 (每增加50样本的OA提升)")
print("=" * 90)

for ds, dsname in [("yrd","YRD"), ("yrd2509","YRD2509")]:
    print(f"\n{dsname}:")
    print(f"  {'步长':<15}", end="")
    for model in models:
        print(f"{model:>12}", end="")
    print()
    for step, (lo, hi) in enumerate([(5,10),(10,20),(20,50),(50,100),(100,150),(150,200)]):
        print(f"  fs{lo}→fs{hi:<5}", end="")
        for model in models:
            flo = ROOT / f"runs_fewshot/fs{lo}/{model}/{ds}/run_001/metrics.json"
            fhi = ROOT / f"runs_fewshot/fs{hi}/{model}/{ds}/run_001/metrics.json"
            if flo.exists() and fhi.exists():
                lo_v = json.loads(flo.read_text())["aggregate"]["oa_mean"] * 100
                hi_v = json.loads(fhi.read_text())["aggregate"]["oa_mean"] * 100
                print(f"{hi_v-lo_v:+12.2f}", end="")
            else:
                print(f"     -      ", end="")
        print()
