"""Generate the approved six-job candidate queue; never launch training."""
import json
from pathlib import Path

import yaml


SPLITS = {
    'yrd': '842aac597d09cb62e169c217bdafd51534fae36ff729e8a117085852a9c3a994',
    'yrd2509new': 'd7844c61dc5875b7c32b058a1b2f909bce3ca64e5a4f7d3e7460f41e1e409e2d',
    'grss07': 'e73f306b348f02d7e656745a9238be3b23faabedd5acd95636eeb7d9206e24ac',
}


def generate(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    original = json.loads((root/'configs/main_first_pass/manifest.json').read_text(encoding='utf-8'))
    folder = root/'configs/paper_candidates'
    folder.mkdir(parents=True, exist_ok=True)
    variants = ['v7_fdmf_fixed', 'v8_msaf_fixed']
    jobs = []
    datasets = {}
    for dataset, data in original['datasets'].items():
        datasets[dataset] = {**data, 'dataset_config': f'configs/datasets/{dataset}.yaml',
                             'reuse_split': {'path': f'runs/main_first_pass_20261017/prepared/{dataset}/split.npz',
                                             'sha256': SPLITS[dataset]}}
    for variant in variants:
        for dataset in datasets:
            config = yaml.safe_load((root/f'configs/main_first_pass/v2_concat_{dataset}.yaml').read_text(encoding='utf-8'))
            config['variant'] = variant
            if variant == 'v8_msaf_fixed':
                # This architecture uses the author's 16/24-dim stages rather
                # than the common encoder. Avoid misleading shared-encoder keys.
                for key in ('embed_dim', 'num_heads', 'depth'):
                    config.pop(key, None)
            name = f'{variant}_{dataset}'
            path = folder/(name+'.yaml')
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            jobs.append({'id': name, 'variant': variant, 'dataset': dataset,
                         'config': path.relative_to(root).as_posix()})
    manifest = {'name': 'paper_candidates_20261018', 'variants': variants,
                'seed': 202201, 'workers': 1, 'screen_session': 'msi_papers_gpu1',
                'wait_for_gpu_idle': False, 'min_free_memory_mib': 20480,
                'cpu_threads_per_worker': 2, 'output_root': 'runs/paper_candidates_20261018',
                'log_root': 'experiment_logs/paper_candidates_20261018',
                'datasets': datasets, 'experiments': jobs}
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    print('Generated',len(generate()['experiments']),'candidate jobs; no training launched.')
