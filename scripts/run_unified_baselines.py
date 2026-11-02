"""Run only the 36 authorized main experiments, with explicit replacement and GPU admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('mghofnet', 'dfinet', 'frekfuse', 'msfmamba', 'softformer',
          'dffnet', 'm2heat', 'mft', 'dcmnet', 'thsgr', 'gfsfn', 's2atnet')
SETTINGS = (('grss07', 'grss07_custom', 'grss07_custom'),
            ('yrd2509new', 'yrd2509new_fs50', 'fs50'), ('yrd', 'yrd_fs50', 'fs50'))
SEEDS = list(range(202201, 202206))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def make_tasks() -> list[dict]:
    return [dict(key=f'{model}_{setting}', model=model, dataset=dataset,
                 config=f'configs/{model}_{setting}.yaml',
                 target=f'runs_fewshot/{folder}/{model}/{dataset}/run_001', status='pending')
            for model in MODELS for dataset, setting, folder in SETTINGS]


def fingerprint(target: Path) -> str | None:
    if target.is_symlink():
        raise ValueError(f'Result symlink is not permitted: {target}')
    if not target.exists():
        return None
    records = []
    for path in sorted(target.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Result contains a symlink: {path}')
        st = path.stat()
        records.append((str(path.relative_to(target)), st.st_size, st.st_mtime_ns, path.is_dir()))
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def checked_target(root: Path, relative: str) -> Path:
    if relative not in {task['target'] for task in make_tasks()}:
        raise ValueError(f'Target is outside the explicit 39-result replacement list: {relative}')
    root = root.resolve()
    target = root / relative
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError(f'Symlink in output path: {current}')
        current = current.parent
    if target.resolve() != target or not target.resolve().is_relative_to(root / 'runs_fewshot'):
        raise ValueError(f'Unsafe result target: {target}')
    return target


def replace_target(root: Path, relative: str, expected: str | None, authorized: bool) -> None:
    target = checked_target(root, relative)
    if fingerprint(target) != expected:
        raise RuntimeError(f'Results changed after queue preparation: {target}')
    if any(path != target for path in target.parent.glob('run_*')):
        raise RuntimeError(f'Unexpected additional run directories: {target.parent}')
    if target.exists():
        if not authorized:
            raise RuntimeError('Replacing an existing result requires --replace-existing')
        shutil.rmtree(target)


def can_start(required: int, active_reservations: list[int], total: int,
              foreign_used: int, free: int) -> bool:
    return (sum(active_reservations) + required <= total - foreign_used - 1024
            and free >= required + 512)


def save_state(directory: Path, state: dict) -> None:
    state['updated'] = now()
    temp = directory / 'queue_status.tmp'
    temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temp.replace(directory / 'queue_status.json')
    text = '\n'.join(f"{t['status']:12} {t['key']} pid={t.get('pid', '-')} {t.get('error', '')}"
                     for t in state['tasks'])
    (directory / 'queue_status.txt').write_text(state['status'] + '\n' + text + '\n', encoding='utf-8')


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def verify_source(expected: str) -> None:
    if git('rev-parse', 'HEAD') != expected or git('diff', '--name-only', 'HEAD'):
        raise RuntimeError('Repository HEAD or tracked source changed while queue was running')


def gpu_usage(gpu: int, active_pids) -> tuple[int, int, int]:
    total, used, free = map(int, subprocess.check_output(
        ['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.total,memory.used,memory.free',
         '--format=csv,noheader,nounits'], text=True).strip().split(','))
    rows = subprocess.check_output(
        ['nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid,used_gpu_memory',
         '--format=csv,noheader,nounits'], text=True)
    own = 0
    for row in rows.splitlines():
        fields = row.split(',')
        if len(fields) == 2 and fields[0].strip().isdigit() and int(fields[0]) in active_pids:
            own += int(fields[1])
    return total, max(0, used - own), free


def validate_result(task: dict, expected_head: str) -> None:
    target = ROOT / task['target']
    result = json.loads((target / 'metrics.json').read_text())
    runs = result['runs']
    if [r['seed'] for r in runs] != SEEDS:
        raise RuntimeError('Output does not contain all five configured seeds')
    for record in runs:
        if not all(math.isfinite(float(record[key])) for key in ('oa', 'aa', 'kappa')):
            raise RuntimeError('Non-finite test metric')
        if not (target / 'checkpoints' / f"run_{record['run']}" / 'model_best.pth').is_file():
            raise RuntimeError('Missing final checkpoint')
    protocol = json.loads((target / 'training_protocol.json').read_text())
    if (protocol['training_protocol'] != 'mghofnet-v1' or protocol['epochs'] != 200
            or protocol['code']['commit'] != expected_head):
        raise RuntimeError('Output protocol/version mismatch')
    info = json.loads((target / 'labeled_map_info.json').read_text())
    best = max(runs, key=lambda r: r['oa'])
    if info['seed'] != best['seed'] or info['test_oa'] != best['oa']:
        raise RuntimeError('Labeled map does not use best test OA run')
    if not (target / 'labeled_prediction.tif').is_file():
        raise RuntimeError('Missing labeled prediction map')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-dir', required=True)
    parser.add_argument('--memory-profile', required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--replace-existing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error('--workers must be between 1 and 3')
    verify_source(args.expected_head)
    tasks = make_tasks()
    if args.dry_run:
        print(json.dumps(tasks, indent=2))
        return
    import fcntl
    directory = (ROOT / args.queue_dir).resolve()
    if not directory.is_relative_to(ROOT / 'runs_fewshot'):
        parser.error('--queue-dir must be inside runs_fewshot')
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'queue.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory / 'queue_status.json').exists():
        raise RuntimeError('Existing queue state: do not restart or retry completed/partial runs automatically')
    profile_path = Path(args.memory_profile)
    profile = json.loads(profile_path.read_text())
    reservations = {}
    for model in MODELS:
        rows = [r for r in profile if r['model'] == model and r['status'] == 'passed']
        if {tuple(r['channels']) for r in rows} != {(6, 1), (10, 4)}:
            raise RuntimeError(f'Missing successful two-channel-combination preflight: {model}')
        reservations[model] = math.ceil(max(r['peak_reserved_mib'] for r in rows) * 1.25 + 768)
    import yaml
    for task in tasks:
        config = yaml.safe_load((ROOT / task['config']).read_text())
        if (config['training_protocol'] != 'mghofnet-v1' or config['epochs'] != 200
                or config['seeds'] != SEEDS or config['num_runs'] != 5
                or config['model'] != task['model'] or config['dataset'] != task['dataset']):
            raise RuntimeError(f'Unexpected main experiment configuration: {task["config"]}')
        configured_target = ROOT / config['output_root'] / task['model'] / task['dataset'] / 'run_001'
        if configured_target.resolve() != (ROOT / task['target']).resolve():
            raise RuntimeError('Configuration output differs from the replacement manifest')
        target = checked_target(ROOT, task['target'])
        task['original_fingerprint'] = fingerprint(target)
        task['reserved_mib'] = reservations[task['model']]
        task['log'] = str(directory / (task['key'] + '.log'))
        if target.exists() and not args.replace_existing:
            raise RuntimeError('Existing results require explicit --replace-existing authorization')
    (directory / 'replacement_manifest.json').write_text(json.dumps(tasks, indent=2))
    shutil.copyfile(profile_path, directory / 'preflight_profile.json')
    state = dict(status='running', started=now(), launcher_pid=os.getpid(), gpu=args.gpu,
                 workers=args.workers, commit=args.expected_head, tasks=tasks)
    active = {}
    save_state(directory, state)
    while active or any(t['status'] == 'pending' for t in tasks):
        for key, (process, stream, task) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            task.update(returncode=code, finished=now())
            try:
                if code != 0:
                    raise RuntimeError(f'Training process exited with code {code}; see task log')
                validate_result(task, args.expected_head)
                task['status'] = 'completed'
            except Exception as error:
                task.update(status='failed', error=str(error))
            del active[key]
            print(now(), task['status'], key, flush=True)
            save_state(directory, state)
        if state['status'] == 'running':
            try:
                verify_source(args.expected_head)
                total, foreign, free = gpu_usage(args.gpu, {v[0].pid for v in active.values()})
                for task in tasks:
                    if len(active) >= args.workers:
                        break
                    if task['status'] != 'pending':
                        continue
                    held = [v[2]['reserved_mib'] for v in active.values()]
                    if not can_start(task['reserved_mib'], held, total, foreign, free):
                        continue
                    task.update(status='replacing', started=now())
                    save_state(directory, state)
                    try:
                        replace_target(ROOT, task['target'], task['original_fingerprint'], args.replace_existing)
                        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu),
                               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
                        command = [sys.executable, f"scripts/train_eval_{task['model']}.py", '--config', task['config']]
                        stream = open(task['log'], 'w')
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                        task.update(status='running', pid=process.pid, command=command)
                        active[task['key']] = (process, stream, task)
                        free -= task['reserved_mib']
                        print(now(), 'started', task['key'], process.pid, flush=True)
                    except Exception as error:
                        task.update(status='failed', error=str(error), finished=now())
                    save_state(directory, state)
            except Exception as error:
                state.update(status='draining', error=str(error))
                for task in tasks:
                    if task['status'] == 'pending':
                        task.update(status='blocked', error='Queue stopped launching: ' + str(error))
                save_state(directory, state)
        if active or any(t['status'] == 'pending' for t in tasks):
            time.sleep(10)
    state.update(status='completed' if all(t['status'] == 'completed' for t in tasks) else 'finished_with_failures', finished=now())
    save_state(directory, state)
    print(now(), state['status'], flush=True)


if __name__ == '__main__':
    main()
