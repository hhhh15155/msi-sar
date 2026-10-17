"""Generate the approved 21-job manifest without starting any experiment."""
import json
from pathlib import Path

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'configs/main_first_pass'
    folder.mkdir(parents=True, exist_ok=True)
    variants = ('v0_msi', 'v1_sar', 'v2_concat', 'v3_fcm_ffm', 'v4_mafi', 'v5_fuzzy', 'v6_current')
    datasets = {'yrd': {'ms_channels': 10, 'sar_channels': 4, 'train_counts': [50] * 8},
                'yrd2509new': {'ms_channels': 10, 'sar_channels': 4, 'train_counts': [50] * 9},
                'grss07': {'ms_channels': 6, 'sar_channels': 1, 'train_counts': [674, 1481, 752, 45, 30]}}
    jobs = []
    for variant in variants:
        for dataset, data in datasets.items():
            name = f'{variant}_{dataset}'
            config = {'variant': variant, 'dataset': dataset, 'dataset_config': f'configs/datasets/{dataset}.yaml',
                      'ms_channels': data['ms_channels'], 'sar_channels': data['sar_channels'],
                      'seed': 202201, 'num_runs': 1, 'patch_size': 11, 'epochs': 200, 'batch_size': 128,
                      'test_batch_size': 1024, 'precision': 'fp32', 'learning_rate': 0.0005, 'weight_decay': 0.0001,
                      'embed_dim': 192 if variant == 'v6_current' else 128,
                      'num_heads': 6 if variant == 'v6_current' else 4,
                      'dropout': 0.1, 'pretrained': False,
                      'split': {'method': 'fixed_train_counts', 'train_counts': data['train_counts']}}
            if variant == 'v6_current':
                config.update(depth_per_stage=2, fusion_depth=2, auxiliary_weight=0.2)
            else:
                config.update(depth=2, auxiliary_weight=0.0)
            path = folder / f'{name}.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
            jobs.append({'id': name, 'variant': variant, 'dataset': dataset,
                         'config': path.relative_to(root).as_posix()})
    manifest = {'name': 'main_first_pass_20261017', 'seed': 202201, 'workers': 2,
                'screen_session': 'msi_cls_main_20261017', 'wait_for_gpu_idle': True,
                'cpu_threads_per_worker': 2, 'output_root': 'runs/main_first_pass_20261017',
                'log_root': 'experiment_logs/main_first_pass_20261017', 'datasets': datasets,
                'experiments': jobs}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Generated {len(jobs)} configs and manifest; no training launched.')


if __name__ == '__main__':
    main()
