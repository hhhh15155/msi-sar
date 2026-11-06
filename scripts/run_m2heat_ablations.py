"""Preview or run the nine ablation settings (45 seeds); never include Full."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('wo_specific_hco', 'wo_shared_hco', 'wo_cff')
SETTINGS = ('yrd_fs50', 'yrd2509new_fs50', 'grss07_custom')


def make_tasks():
    return [dict(key=f'{variant}_{setting}', variant=variant, dataset=setting.split('_')[0],
                 config=f'configs/m2heat_ablation/{variant}_{setting}.yaml', status='pending')
            for variant in VARIANTS for setting in SETTINGS]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def save_state(directory, state):
    state['updated'] = datetime.now().astimezone().isoformat()
    temp = directory / 'queue_status.tmp'
    temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temp.replace(directory / 'queue_status.json')


def can_start(reservation, active_count, total, foreign_used, free):
    return (free >= reservation + 1024
            and (active_count + 1) * reservation + foreign_used + 1024 <= total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-parallel', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--reserve-mib', type=int, default=10000,
                        help='Per-task reservation determined by synthetic GPU profile')
    parser.add_argument('--expected-commit')
    parser.add_argument('--execute', action='store_true', help='Without this flag only print the task list')
    args = parser.parse_args()
    tasks = make_tasks()
    if not args.execute:
        print(json.dumps(tasks, indent=2)); return
    if not args.expected_commit or git('rev-parse', 'HEAD') != args.expected_commit:
        raise RuntimeError('Execution requires the verified --expected-commit')
    if args.reserve_mib < 1024:
        raise ValueError('Unreasonable memory reservation')
    directory = Path(args.batch_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)  # never reuse or overwrite a batch
    (directory / 'logs').mkdir()
    state = dict(status='running', commit=args.expected_commit, gpu=args.gpu,
                 pid=os.getpid(), max_parallel=args.max_parallel, reserve_mib=args.reserve_mib,
                 tasks=tasks)
    active = []
    failed = False
    try:
        while any(t['status'] == 'pending' for t in tasks) or active:
            if git('rev-parse', 'HEAD') != args.expected_commit or git('diff', '--name-only', 'HEAD'):
                raise RuntimeError('Tracked code changed while this queue was active')
            for item in active[:]:
                task, process, stream = item
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                target = directory / task['variant'] / task['dataset']
                complete = (target / 'metrics.json').is_file() and (target / 'labeled_map_info.json').is_file()
                task.update(status='completed' if code == 0 and complete else 'failed', exit_code=code)
                failed |= task['status'] == 'failed'
                active.remove(item)
            pending = next((t for t in tasks if t['status'] == 'pending'), None)
            if pending and len(active) < args.max_parallel:
                total, used, free = map(int, subprocess.check_output([
                    'nvidia-smi', f'--id={args.gpu}', '--query-gpu=memory.total,memory.used,memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip().split(','))
                processes = subprocess.check_output([
                    'nvidia-smi', f'--id={args.gpu}', '--query-compute-apps=pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'], text=True)
                own_pids = {item[1].pid for item in active}
                own_used = 0
                for row in processes.splitlines():
                    fields = row.split(',')
                    if len(fields) == 2 and fields[0].strip().isdigit() and int(fields[0]) in own_pids:
                        own_used += int(fields[1])
                foreign_used = max(0, used - own_used)
                # Reserve worst-case memory even while an active task is between kernels.
                if can_start(args.reserve_mib, len(active), total, foreign_used, free):
                    log_path = directory / 'logs' / f"{pending['key']}.log"
                    stream = log_path.open('x', encoding='utf-8')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                    command = [sys.executable, '-u', str(ROOT / 'scripts/train_eval_m2heat_ablation.py'),
                               '--config', pending['config'], '--batch-dir', str(directory)]
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    pending.update(status='running', pid=process.pid, log=str(log_path), command=command)
                    active.append((pending, process, stream))
            save_state(directory, state)
            time.sleep(5)
        state['status'] = 'failed' if failed else 'completed'
        save_state(directory, state)
        subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_m2heat_ablations.py'),
                        '--batch-dir', str(directory)], cwd=ROOT, check=True)
    except BaseException as exc:
        # Do not kill any running task on a scheduler error; record its PID for recovery.
        state.update(status='scheduler_failed', error=repr(exc))
        save_state(directory, state)
        raise
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
