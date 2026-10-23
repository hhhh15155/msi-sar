"""Sequential queue for the adapted HSI–LiDAR baselines (run inside screen on 110).

Reads ``configs/hsi_lidar/manifest.json``, skips completed runs, and records a
queue status file. Failures are recorded and the queue stops that model/dataset
pair without retrying silently.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='configs/hsi_lidar/manifest.json')
    parser.add_argument('--models')
    parser.add_argument('--datasets')
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    manifest = read_json(ROOT / args.manifest)
    models = args.models.split(',') if args.models else manifest['models']
    datasets = args.datasets.split(',') if args.datasets else list(manifest['datasets'])
    run_root = ROOT / manifest['output_root']
    log_root = run_root / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / 'queue_status.json'
    status = read_json(status_path) if status_path.exists() else {'jobs': {}, 'started': time.time()}
    status.update({'pid': os.getpid(), 'phase': 'running', 'host': os.uname().nodename,
                   'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                   'models': models, 'datasets': datasets, 'updated': time.time()})
    write_json(status_path, status)
    failed = 0
    for model in models:
        for dataset in datasets:
            job_id = f'{model}_{dataset}'
            metrics = run_root / job_id / 'metrics.json'
            if metrics.exists() and read_json(metrics).get('completed') is True:
                status['jobs'][job_id] = {'status': 'completed', 'skipped': True}
                write_json(status_path, status)
                print(f'SKIP completed {job_id}', flush=True)
                continue
            command = [sys.executable, '-u', str(ROOT / 'scripts' / 'hsi_lidar_bench.py'),
                       '--manifest', args.manifest, '--model', model, '--dataset', dataset]
            if args.epochs:
                command += ['--epochs', str(args.epochs)]
            log_path = log_root / f'{job_id}.console.log'
            status['jobs'][job_id] = {'status': 'running', 'started': time.time(),
                                      'log': str(log_path.relative_to(ROOT))}
            write_json(status_path, status)
            print(f'LAUNCH {job_id}', flush=True)
            with log_path.open('a', encoding='utf-8') as log:
                code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            complete = code == 0 and (run_root / job_id / 'metrics.json').exists()
            status['jobs'][job_id].update(status='completed' if complete else 'failed',
                                          returncode=code, finished=time.time(),
                                          updated=time.time())
            write_json(status_path, status)
            print(f'EXIT {job_id} code={code} completed={complete}', flush=True)
            if not complete:
                failed += 1
    status.update({'phase': 'finished' if failed == 0 else 'finished_with_errors', 'updated': time.time()})
    write_json(status_path, status)
    print('QUEUE FINISHED', flush=True)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
