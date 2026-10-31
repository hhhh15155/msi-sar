"""Read-only audit: no torch import, datasets, training or server commands."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.protocol import TRAINING_DEFAULTS


def audit() -> tuple[list[str], int, int]:
    problems = []
    models = sorted(p.parent.name for p in (ROOT / 'baselines').glob('*/train.py'))
    count = 0
    for path in (ROOT / 'configs').glob('*.yaml'):
        config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        if config.get('model') not in models:
            continue
        count += 1
        for key, expected in TRAINING_DEFAULTS.items():
            if config.get(key) != expected or key not in config:
                problems.append(f'{path.name}: {key} must be {expected!r}')
        if config.get('infer', {}).get('save_full_map') is not False:
            problems.append(f'{path.name}: full inference must default to false')
    for model in models:
        tree = ast.parse((ROOT / 'baselines' / model / 'train.py').read_text(encoding='utf-8-sig'))
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        for required in ('prepare_training_config', 'build_optimizer_scheduler', 'train_epoch', 'scheduler.step', 'record_protocol', 'set_seed'):
            if required not in calls:
                problems.append(f'{model}/train.py: missing {required}')
        if any(c.startswith('torch.optim.') or c.endswith('clip_grad_norm_') for c in calls):
            problems.append(f'{model}/train.py: independent optimizer/scheduler/clipping remains')
        for filename, module in (('dataset.py', 'baselines.data'), ('eval.py', 'baselines.prediction')):
            tree = ast.parse((ROOT / 'baselines' / model / filename).read_text(encoding='utf-8-sig'))
            if not any(isinstance(n, ast.ImportFrom) and n.module == module for n in tree.body):
                problems.append(f'{model}/{filename}: must reuse {module}')
    return problems, len(models), count


def main() -> None:
    problems, models, configs = audit()
    if problems:
        print('\n'.join(problems))
        raise SystemExit(1)
    print(f'PASS: {models} models, {configs} configs use mghofnet-v1; no experiments started.')


if __name__ == '__main__':
    main()
