"""Independent ablation training orchestration using the existing shared protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.io import ensure_dir, load_yaml, save_yaml, save_json
from baselines.protocol import prepare_training_config
from baselines.training import set_seed, build_optimizer_scheduler, train_epoch, record_protocol
from baselines.prediction import predict_labeled_samples
from baselines.map_selection import select_map_run
from .ablation import M2HeatAblation, VARIANTS
from .dataset import load_dataset, PatchDataset, split_from_config
from .metrics import aggregate_metrics
from .train import test_labeled_samples
from .infer import save_label_outputs

SEEDS = list(range(202201, 202206))


def validate_config(config):
    config = prepare_training_config(config)
    if config.get('ablation') not in VARIANTS:
        raise ValueError('Only the three requested ablations are supported; Full uses existing results')
    if config.get('seeds') != SEEDS or config.get('num_runs') != 5:
        raise ValueError('Expected five seeds 202201--202205')
    if config.get('model') != 'm2heat' or config.get('dataset') not in ('yrd', 'yrd2509new', 'grss07'):
        raise ValueError('Unexpected model/dataset')
    return config


def build_model(config, n_classes, device):
    return M2HeatAblation(variant=config['ablation'], num_classes=n_classes,
                         patch_size=config['patch_size'], spectral_channels=config['spectral_channels'],
                         sar_channels=config['sar_channels']).to(device)


def completed_metrics(directory, config):
    """Require exactly five matching, finite results before declaring completion."""
    results = []
    for index, seed in enumerate(SEEDS):
        directory_seed = directory / f'seed_{seed}'
        result = json.loads((directory_seed / 'metrics.json').read_text(encoding='utf-8'))
        if result['seed'] != seed or result['run'] != index or result['ablation'] != config['ablation']:
            raise ValueError(f'Result identity mismatch: {directory_seed}')
        saved = load_yaml(directory_seed / 'config.yaml')
        if saved != config:
            raise ValueError(f'Config mismatch: {directory_seed}')
        values = [result['oa'], result['aa'], result['kappa'], *result['class_acc']]
        if not np.isfinite(values).all() or len(result['class_acc']) != len(result['class_names']):
            raise ValueError(f'Invalid metrics: {directory_seed}')
        if not (directory_seed / 'checkpoints' / 'model_best.pth').is_file():
            raise ValueError(f'Missing final checkpoint: {directory_seed}')
        results.append(result)
    return results


def train(config_path, batch_dir, seed=None):
    config = validate_config(load_yaml(config_path))
    dataset_config = load_yaml(config['dataset_config'])
    device = torch.device(config.get('device', 'cuda:0'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing silent CPU training')
    directory = Path(batch_dir) / config['ablation'] / config['dataset']
    ensure_dir(directory)
    config_path_saved = directory / 'config.yaml'
    if config_path_saved.exists():
        if load_yaml(config_path_saved) != config:
            raise ValueError(f'Output already belongs to another config: {directory}')
        if load_yaml(directory / 'dataset_config.yaml') != dataset_config:
            raise ValueError(f'Dataset config changed since batch creation: {directory}')
    else:
        save_yaml(config, config_path_saved)
        record_protocol(config, dataset_config, directory)
    image, gt, labels = load_dataset(dataset_config)
    if image.shape[-1] != config['spectral_channels'] + config['sar_channels']:
        raise ValueError('Dataset channels do not match model configuration')
    seeds = SEEDS if seed is None else [seed]
    if any(s not in SEEDS for s in seeds):
        raise ValueError('Unexpected seed')
    for seed in seeds:
        index = SEEDS.index(seed)
        target = directory / f'seed_{seed}'
        if (target / 'metrics.json').exists():
            saved = load_yaml(target / 'config.yaml')
            if saved != config:
                raise ValueError(f'Existing seed config mismatch: {target}')
            print(f'skip completed seed: {target}', flush=True)
            continue
        # An incomplete seed is never silently overwritten or retried.
        target.mkdir(exist_ok=False)
        save_yaml(config, target / 'config.yaml')
        record_protocol(config, dataset_config, target)
        set_seed(seed)
        train_gt, test_gt = split_from_config(gt, config, seed)
        np.savez_compressed(target / 'split.npz', train_gt=train_gt, test_gt=test_gt)
        train_loader = DataLoader(PatchDataset(image, train_gt, config['patch_size'], data_aug=True),
                                  batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(PatchDataset(image, test_gt, config['patch_size'], data_aug=False),
                                 batch_size=config['test_batch_size'], shuffle=False)
        model = build_model(config, len(labels), device)
        optimizer, scheduler = build_optimizer_scheduler(model, config)
        checkpoint_dir = ensure_dir(target / 'checkpoints')
        with (target / 'log.txt').open('x', encoding='utf-8') as stream:
            def log(message):
                print(message, flush=True)
                stream.write(message + '\n'); stream.flush()
            log(f"ablation={config['ablation']} dataset={config['dataset']} seed={seed} device={device}")
            for epoch in range(1, config['epochs'] + 1):
                loss = train_epoch(model, train_loader, optimizer, device, config)
                scheduler.step()
                if epoch == 1 or epoch % 10 == 0:
                    log(f'seed={seed} epoch={epoch} loss={loss:.6f}')
                if epoch % 10 == 0:
                    torch.save(model.state_dict(), checkpoint_dir / 'model.pth')
            checkpoint = checkpoint_dir / 'model_best.pth'
            torch.save(model.state_dict(), checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            metrics = test_labeled_samples(model, test_loader, device, len(labels))
            metrics.update(run=index, seed=seed, ablation=config['ablation'], class_names=labels)
            log(json.dumps(metrics))
            save_json(metrics, target / 'metrics.json')
        del model, optimizer, scheduler
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if all((directory / f'seed_{s}' / 'metrics.json').exists() for s in SEEDS):
        results = completed_metrics(directory, config)
        save_json({'runs': results, 'aggregate': aggregate_metrics(results)}, directory / 'metrics.json')
        if config.get('eval', {}).get('enabled', True):
            selection = select_map_run(directory, 'best')
            best_seed = selection['seed']
            model = build_model(config, len(labels), device)
            checkpoint = directory / f'seed_{best_seed}' / 'checkpoints' / 'model_best.pth'
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            prediction = predict_labeled_samples(model, image, gt, config, device)
            mask = gt >= 0
            labeled_gt = np.zeros_like(gt, dtype=np.uint8)
            labeled_prediction = np.zeros_like(gt, dtype=np.uint8)
            labeled_gt[mask] = gt[mask].astype(np.uint8) + 1
            labeled_prediction[mask] = prediction[mask].astype(np.uint8) + 1
            paths = {}
            if config['eval'].get('save_labeled_maps', True):
                for name, array in (('labeled_gt', labeled_gt), ('labeled_prediction', labeled_prediction)):
                    paths[name] = [str(p) for p in save_label_outputs(array, directory / name, config.get('palette'))]
            save_json({'mode': 'labeled', **selection, 'labeled_count': int(mask.sum()), 'outputs': paths},
                      directory / 'labeled_map_info.json')
    return directory
