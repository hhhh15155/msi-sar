"""Generate only six unfinished ablations; never rerun completed full models."""
import json
from pathlib import Path

import yaml

from scripts.generate_paper_candidate_configs import SPLITS


def generate(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    original = json.loads((root/'configs/main_first_pass/manifest.json').read_text(encoding='utf-8'))
    folder = root/'configs/v6_ablation'
    folder.mkdir(parents=True, exist_ok=True)
    variants = ['a01_late_equal', 'a02_no_early_path']
    datasets = {name: {**original['datasets'][name],
                      'dataset_config': f'configs/datasets/{name}.yaml',
                      'reuse_split': {'path': f'runs/main_first_pass_20261017/prepared/{name}/split.npz',
                                      'sha256': SPLITS[name]}}
                for name in ('yrd', 'yrd2509new', 'grss07')}
    jobs = []
    for variant in variants:
        for name in datasets:
            config = yaml.safe_load((root/f'configs/main_first_pass/v6_current_{name}.yaml').read_text(encoding='utf-8'))
            config['variant'] = variant
            identifier = f'{variant}_{name}'
            path = folder/'ablations'/(identifier+'.yaml')
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            jobs.append({'id': identifier, 'variant': variant, 'dataset': name,
                         'config': path.relative_to(root).as_posix()})
    namespace = 'v6_ablation_20261021_110'
    manifest = {'name': namespace, 'variants': variants, 'seed': 202201, 'workers': 1,
                'screen_session': 'msi_v6_ablate_gpu0', 'wait_for_gpu_idle': True,
                'min_free_memory_mib': 20480, 'cpu_threads_per_worker': 2,
                'output_root': 'runs/'+namespace, 'log_root': 'experiment_logs/'+namespace,
                'datasets': datasets, 'experiments': jobs}
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    print(f'Generated {len(generate()["experiments"])} unfinished ablations; no full-model reruns or training launched.')
