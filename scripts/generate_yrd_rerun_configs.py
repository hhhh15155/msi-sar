"""Prepare the seven YRD jobs; never start training or replace results."""
import json
from pathlib import Path

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'configs/main_first_pass/manifest.json'
    original = json.loads(source.read_text())
    folder = root / 'configs/yrd_per_channel'
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for item in original['experiments']:
        if item['dataset'] != 'yrd':
            continue
        target = folder / (item['id'] + '.yaml')
        target.write_bytes((root / item['config']).read_bytes())
        jobs.append({**item, 'config': target.relative_to(root).as_posix()})
    manifest = {'name': 'yrd_per_channel_20261017', 'seed': 202201, 'workers': 1,
                'screen_session': 'msi_yrd_pc_gpu1', 'wait_for_gpu_idle': False,
                'min_free_memory_mib': 20480, 'cpu_threads_per_worker': 2,
                'output_root': 'runs/yrd_per_channel_20261017',
                'log_root': 'experiment_logs/yrd_per_channel_20261017',
                'datasets': {'yrd': {**original['datasets']['yrd'],
                            'dataset_config': 'configs/datasets/yrd.yaml',
                            'reuse_split': {'path': 'data/yrd/splits/main_seed202201_train50.npz',
                                           'sha256': '842aac597d09cb62e169c217bdafd51534fae36ff729e8a117085852a9c3a994'}}},
                'experiments': jobs,
                'promotion': {'target_root': 'runs/main_first_pass_20261017',
                              'archive_root': 'runs/archive/main_first_pass_20261017_before_yrd_pc',
                              'base_manifest': 'configs/main_first_pass/manifest.json'}}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Generated seven YRD configs; training parameters unchanged.')


if __name__ == '__main__':
    main()
