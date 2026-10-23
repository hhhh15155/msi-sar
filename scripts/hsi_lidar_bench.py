"""Train/evaluate one HSI–LiDAR-derived baseline on one prepared MSI–SAR dataset.

Protocol is deliberately identical to ``scripts/main_experiment.py`` (same
prepared caches, saved split, seed, patch size, optimiser, augmentation,
200 epochs and final-epoch held-out test) so results are comparable with the
existing 27-item table; the only difference is that these baselines get their
own run root and are not merged into that table automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from baselines.hsi_lidar.registry import build_model, DISPLAY_NAMES  # noqa: E402
from scripts.main_experiment import (  # noqa: E402
    PatchBank, augment, atomic_json, digest, metrics_from_confusion, read_json, validate_split)


def source_hashes():
    files = sorted((ROOT / 'baselines' / 'hsi_lidar').glob('*.py'))
    files += [ROOT / 'vendor' / 'hsi_lidar' / 'DFFNet-original' / 'modules' / 'DynamicBlock.py',
              ROOT / 'vendor' / 'hsi_lidar' / 'DFFNet-original' / 'modules' / 'fusion.py',
              ROOT / 'vendor' / 'hsi_lidar' / 'M2Heat-original' / 'm2heat.py']
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def select_device():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no silent CPU fallback')
    return torch.device('cuda:0')


def load_benchmark(manifest, dataset):
    prepared = ROOT / manifest['prepared_root'] / dataset
    metadata = read_json(prepared / 'metadata.json')
    image = np.load(prepared / 'image.npy', mmap_mode='r')
    gt = np.load(prepared / 'gt.npy', mmap_mode='r')
    with np.load(prepared / 'split.npz') as split:
        train_indices, test_indices = split['train_indices'], split['test_indices']
    validate_split(gt, train_indices, test_indices, metadata['train_counts'])
    return prepared, metadata, image, gt, train_indices, test_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='configs/hsi_lidar/manifest.json')
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--smoke', action='store_true', help='one forward/backward batch, no run files')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    manifest = read_json(ROOT / args.manifest)
    specification = manifest['datasets'][args.dataset]
    ms_channels, sar_channels = specification['ms_channels'], specification['sar_channels']
    patch_size = int(manifest['patch_size'])
    seed = int(manifest['seed'])
    epochs = int(args.epochs or manifest['epochs'])
    batch_size = int(manifest['batch_size'])
    device = select_device()

    prepared, metadata, image, gt, train_indices, test_indices = load_benchmark(manifest, args.dataset)
    num_classes = len(metadata['class_names'])
    run_root = ROOT / manifest['output_root']
    run_dir = run_root / f'{args.model}_{args.dataset}'
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / 'metrics.json'
    if metrics_path.exists() and not args.force:
        old = read_json(metrics_path)
        if old.get('completed') is True:
            print(f'SKIP completed {args.model}_{args.dataset}', flush=True)
            return

    torch.set_num_threads(int(manifest.get('cpu_threads_per_worker', 2)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    generator = torch.Generator().manual_seed(seed)

    bank = PatchBank(image, patch_size, device)
    labels = torch.from_numpy(np.array(gt.reshape(-1), copy=True)).to(device)
    model = build_model(args.model, ms_channels, sar_channels, num_classes, patch_size).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f'MODEL {args.model} dataset={args.dataset} parameters={parameters} '
          f'classes={num_classes} train={len(train_indices)} test={len(test_indices)} '
          f'ms={ms_channels} sar={sar_channels}', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=manifest['learning_rate'],
                                  weight_decay=manifest['weight_decay'])
    owned = {id(p) for group in optimizer.param_groups for p in group['params']}
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and id(parameter) not in owned]
    if missing:
        raise ValueError(f'Optimizer does not cover trainable parameters: {missing[:5]}')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    if args.smoke:
        model.train()
        selected = torch.from_numpy(train_indices[:min(batch_size, len(train_indices))]).to(device)
        inputs = augment(bank.take(selected), generator)
        optimizer.zero_grad(set_to_none=True)
        out = model(inputs[:, :ms_channels], inputs[:, ms_channels:], labels=labels[selected])
        loss = out['losses']['total']
        loss.backward()
        grads = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
        model.eval()
        with torch.inference_mode():
            logits = model(inputs[:, :ms_channels], inputs[:, ms_channels:])['logits']
        print(f'SMOKE OK {args.model}_{args.dataset} loss={float(loss):.4f} '
              f'grad_tensors={grads} logits={tuple(logits.shape)}', flush=True)
        return

    cfg = {'model': args.model, 'display_name': DISPLAY_NAMES.get(args.model),
           'dataset': args.dataset, 'ms_channels': ms_channels, 'sar_channels': sar_channels,
           'num_classes': num_classes, 'patch_size': patch_size, 'seed': seed, 'epochs': epochs,
           'batch_size': batch_size, 'test_batch_size': int(manifest['test_batch_size']),
           'learning_rate': manifest['learning_rate'], 'weight_decay': manifest['weight_decay'],
           'precision': 'fp32', 'pretrained': False, 'protocol': 'same-as-main-experiment-v1'}
    atomic_json(run_dir / 'config.json', cfg)
    atomic_json(run_dir / 'data_reference.json', {
        'prepared': str(prepared.relative_to(ROOT)),
        'metadata_sha256': digest(prepared / 'metadata.json'),
        'image_sha256': digest(prepared / 'image.npy'),
        'split_sha256': metadata['prepared_sha256']['split.npz']})
    atomic_json(run_dir / 'environment.json', {
        'hostname': os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', ''),
        'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES', '0'), 'seed': seed,
        'source_sha256': source_hashes(), 'pid': os.getpid()})

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    history = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_indices), generator=generator).numpy()
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            selected = torch.from_numpy(train_indices[order[start:start + batch_size]]).to(device)
            inputs = augment(bank.take(selected), generator)
            optimizer.zero_grad(set_to_none=True)
            loss = model(inputs[:, :ms_channels], inputs[:, ms_channels:], labels=labels[selected])['losses']['total']
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
        scheduler.step()
        history.append({'epoch': epoch, 'loss': loss_sum / len(train_indices),
                        'lr': scheduler.get_last_lr()[0], 'seconds': time.monotonic() - started})
        atomic_json(run_dir / 'progress.json', {'phase': 'training', 'epoch': epoch, 'epochs': epochs,
                                                'loss': history[-1]['loss'], 'pid': os.getpid(),
                                                'updated': time.time()})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'{args.model}_{args.dataset} epoch={epoch}/{epochs} '
                  f'loss={history[-1]["loss"]:.6f} seconds={time.monotonic() - started:.1f}', flush=True)
            atomic_json(run_dir / 'history.json', history)
    train_seconds = time.monotonic() - started

    torch.save(model.state_dict(), run_dir / 'model_last.pth')
    model.eval()
    test_batch = int(manifest['test_batch_size'])
    confusion = torch.zeros(num_classes * num_classes, dtype=torch.int64, device=device)
    predictions = np.lib.format.open_memmap(run_dir / 'test_predictions.npy', mode='w+',
                                            dtype=np.int16, shape=(len(test_indices),))
    eval_started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(test_indices), test_batch):
            selected = torch.from_numpy(test_indices[start:start + test_batch]).to(device)
            inputs = bank.take(selected)
            logits = model(inputs[:, :ms_channels], inputs[:, ms_channels:])['logits']
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite final test logits')
            predicted = logits.argmax(1)
            confusion += torch.bincount(labels[selected] * num_classes + predicted,
                                        minlength=num_classes * num_classes)
            predictions[start:start + len(selected)] = predicted.cpu().numpy()
    predictions.flush()
    result = metrics_from_confusion(confusion.view(num_classes, num_classes).cpu().numpy())
    result.update({'completed': True, 'model': args.model, 'dataset': args.dataset, 'seed': seed,
                   'epochs': epochs, 'train_counts': metadata['train_counts'],
                   'test_counts': metadata['test_counts'], 'class_names': metadata['class_names'],
                   'parameters': parameters, 'train_seconds': train_seconds,
                   'test_seconds': time.monotonic() - eval_started,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
                   'evaluation_scope': 'held_out_test_final_epoch', 'precision': 'fp32',
                   'checkpoint': 'model_last.pth'})
    atomic_json(metrics_path, result)
    atomic_json(run_dir / 'progress.json', {'phase': 'completed', 'updated': time.time()})
    print(f'COMPLETE {args.model}_{args.dataset} OA={result["oa"]:.2f} AA={result["aa"]:.2f} '
          f'F1={result["macro_f1"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
