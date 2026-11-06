"""Export all seed metrics and complete-five-seed summaries without re-evaluation."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarize(batch_dir):
    from baselines.m2heat.ablation_experiment import completed_metrics, validate_config
    from baselines.m2heat.metrics import aggregate_metrics
    from baselines.io import load_yaml
    from scripts.run_m2heat_ablations import make_tasks
    batch_dir = Path(batch_dir)
    rows, seeds, missing = [], [], []
    for task in make_tasks():
        target = batch_dir / task['variant'] / task['dataset']
        try:
            config = validate_config(load_yaml(target / 'config.yaml'))
            results = completed_metrics(target, config)
        except (OSError, ValueError, KeyError) as exc:
            missing.append(f"{task['key']}: {exc}")
            continue
        aggregate = aggregate_metrics(results)
        row = dict(variant=task['variant'], dataset=task['dataset'], n=5, **aggregate)
        for index, name in enumerate(results[0]['class_names']):
            row[f'class_{index}_{name}'] = sum(r['class_acc'][index] for r in results) / 5
        rows.append(row)
        for result in results:
            seed_row = dict(variant=task['variant'], dataset=task['dataset'],
                            seed=result['seed'], oa=result['oa'], aa=result['aa'], kappa=result['kappa'])
            seed_row['class_acc_json'] = json.dumps(dict(zip(result['class_names'], result['class_acc'])))
            seeds.append(seed_row)
    batch_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (('summary.csv', rows), ('seed_metrics.csv', seeds)):
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (batch_dir / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
    lines = ['# M2Heat 消融结果', '', 'OA/AA/Kappa 单位均为百分数；标准差 ddof=0；只汇总完整五个 seeds。', '',
             '| 变体 | 数据集 | OA 均值±标准差 | AA 均值±标准差 | Kappa 均值±标准差 |',
             '| --- | --- | --- | --- | --- |']
    for row in rows:
        values = [f"{row[key + '_mean']:.2f} ± {row[key + '_std']:.2f}" for key in ('oa', 'aa', 'kappa')]
        lines.append(f"| {row['variant']} | {row['dataset']} | " + ' | '.join(values) + ' |')
    if missing:
        lines += ['', '未完成／无效项：', '', *['- ' + item for item in missing]]
    (batch_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'complete settings={len(rows)}/9; output={batch_dir / "summary.md"}')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', required=True)
    summarize(parser.parse_args().batch_dir)
