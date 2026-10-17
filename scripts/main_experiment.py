"""Single-seed main experiments: prepare, train, queue, detach, and inspect.

No third-party training entry points are executed. All seven versions use the
same prepared images and saved split. A final held-out test pass is mandatory;
test scores never select epochs. Failed experiments are not retried silently.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

# GPU indices are physical PCI-order indices for every entry, including direct
# queue/train calls. Set this before importing Torch or initializing CUDA.
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from baselines.dfinet.dataset import load_dataset
from baselines.split import split_from_config
from models.main_classification import VARIANTS, build_main_model

DEFAULT_MANIFEST = ROOT / 'configs/main_first_pass/manifest.json'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_manifest(path):
    manifest = read_json(path)
    if len(manifest['experiments']) != 21 or len({x['id'] for x in manifest['experiments']}) != 21:
        raise ValueError('Expected exactly 21 distinct experiments')
    return manifest


def run_root(manifest):
    return ROOT / manifest['output_root']


def validate_split(gt, train, test, counts):
    valid = np.flatnonzero(gt.reshape(-1) >= 0)
    joined = np.concatenate((train, test))
    if len(joined) != len(valid) or not np.array_equal(np.sort(joined), valid):
        raise ValueError('Split must partition every labeled pixel exactly once')
    actual = np.bincount(gt.reshape(-1)[train], minlength=len(counts))
    if actual.tolist() != list(counts):
        raise ValueError(f'Train counts differ: {actual.tolist()} vs {counts}')


def prepare(manifest):
    output = run_root(manifest)
    for dataset in manifest['datasets']:
        target = output / 'prepared' / dataset
        metadata_path = target / 'metadata.json'
        data_config = yaml.safe_load((ROOT / f'configs/datasets/{dataset}.yaml').read_text(encoding='utf-8'))
        files = [ROOT / data_config['path'] / data_config.get(key, default)
                 for key, default in (('data_file', 'data.mat'), ('label_file', 'label.mat'))]
        hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
        counts = manifest['datasets'][dataset]['train_counts']
        fingerprint = {'data_sha256': hashes, 'dataset_config_sha256': digest(ROOT / f'configs/datasets/{dataset}.yaml'),
                       'reader_sha256': digest(ROOT / 'baselines/dfinet/dataset.py'),
                       'splitter_sha256': digest(ROOT / 'baselines/split.py'),
                       'seed': manifest['seed'], 'train_counts': counts}
        if metadata_path.exists():
            old = read_json(metadata_path)
            if old['fingerprint'] != fingerprint:
                raise ValueError(f'{dataset}: prepared data no longer matches input data/config')
            for name, expected in old['prepared_sha256'].items():
                if digest(target / name) != expected:
                    raise ValueError(f'{dataset}: prepared artifact changed: {name}')
            print(f'PREPARED verified {dataset}', flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f'PREPARING {dataset}', flush=True)
        image, gt, names = load_dataset(data_config)
        if not np.isfinite(image).all() or len(names) != len(counts):
            raise ValueError(f'{dataset}: nonfinite image or mismatched classes')
        train_gt, test_gt = split_from_config(gt, {'split': {'method': 'fixed_train_counts', 'train_counts': counts}},
                                             manifest['seed'])
        train, test = np.flatnonzero(train_gt.reshape(-1) >= 0), np.flatnonzero(test_gt.reshape(-1) >= 0)
        validate_split(gt, train, test, counts)
        np.save(target / 'image.npy', image)
        np.save(target / 'gt.npy', gt)
        np.savez_compressed(target / 'split.npz', train_indices=train, test_indices=test)
        atomic_json(metadata_path, {'fingerprint': fingerprint, 'shape': list(image.shape), 'class_names': names,
                                   'normalization': data_config.get('normalization', 'global_minmax_center'),
                                   'normalization_scope': 'whole_image_unlabeled', 'padding': 'reflect',
                                   'train_counts': counts,
                                   'test_counts': np.bincount(gt.reshape(-1)[test], minlength=len(names)).tolist(),
                                   'prepared_sha256': {name: digest(target / name)
                                                       for name in ('image.npy', 'gt.npy', 'split.npz')}})
        print(f'PREPARED {dataset} shape={image.shape} train={len(train)} test={len(test)}', flush=True)


class PatchBank:
    """Keep one padded image on device and gather batches, avoiding Python pixel loops."""
    def __init__(self, image, patch_size, device):
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError('patch_size must be positive and odd')
        self.width, self.patch_size = image.shape[1], patch_size
        pad = patch_size // 2
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        self.image = torch.from_numpy(padded).to(device)
        self.dy, self.dx = torch.meshgrid(torch.arange(patch_size, device=device),
                                        torch.arange(patch_size, device=device), indexing='ij')

    def take(self, indices):
        indices = indices.to(self.image.device)
        rows = indices.div(self.width, rounding_mode='floor')[:, None, None] + self.dy
        cols = indices.remainder(self.width)[:, None, None] + self.dx
        return self.image[rows, cols].permute(0, 3, 1, 2).contiguous()


def augment(images, generator):
    # Match the old augmentation probabilities: identity .5, each of five transforms .1.
    draws = torch.rand(len(images), generator=generator)
    choices = torch.where(draws <= 0.5, 0, ((draws - 0.5) * 10).long().clamp(0, 4) + 1).to(images.device)
    transforms = (images.flip(-1), images.flip(-2), images.rot90(1, (-2, -1)),
                  images.rot90(2, (-2, -1)), images.rot90(3, (-2, -1)))
    for i, transformed in enumerate(transforms, 1):
        images = torch.where((choices == i)[:, None, None, None], transformed, images)
    return images


def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64)
    rows, cols = cm.sum(1), cm.sum(0)
    total, correct = int(cm.sum()), cm.diagonal()
    recall = np.divide(correct, rows, out=np.zeros(len(cm), dtype=float), where=rows > 0)
    precision = np.divide(correct, cols, out=np.zeros(len(cm), dtype=float), where=cols > 0)
    f1 = np.divide(2 * correct, rows + cols, out=np.zeros(len(cm), dtype=float), where=rows + cols > 0)
    oa = float(correct.sum() / total) if total else 0
    chance = float(np.dot(rows.astype(float), cols.astype(float)) / total ** 2) if total else 0
    return {'oa': 100 * oa, 'aa': 100 * float(recall.mean()),
            'kappa': 100 * (oa - chance) / (1 - chance) if chance < 1 else 0,
            'macro_f1': 100 * float(f1.mean()), 'class_acc': (100 * recall).tolist(),
            'class_precision': (100 * precision).tolist(), 'class_f1': (100 * f1).tolist(),
            'confusion_matrix': cm.tolist()}


def source_fingerprint():
    names = ['models/main_classification.py', 'models/fuzzy_transformer.py', 'scripts/main_experiment.py',
             'baselines/dfinet/dataset.py', 'baselines/split.py']
    return {name: digest(ROOT / name) for name in names}


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def experiment_identity(config_path, prepared):
    return {'config_sha256': digest(config_path), 'source_sha256': source_fingerprint(),
            'prepared_metadata_sha256': digest(prepared / 'metadata.json')}


def verify_completed(output, identity):
    metrics_path = output / 'metrics.json'
    if not metrics_path.exists():
        return False
    result = read_json(metrics_path)
    if result.get('completed') is not True or result.get('identity') != identity:
        raise ValueError(f'Completed result has different code/config/data: {output}')
    if digest(output / 'model_last.pth') != result['checkpoint_sha256']:
        raise ValueError(f'Final checkpoint changed: {output}')
    return True


def worker_alive(pid, experiment_id):
    try:
        arguments = Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    return (any(arg.endswith(b'main_experiment.py') for arg in arguments) and
            b'train' in arguments and experiment_id.encode() in arguments)


def train(manifest, experiment_id):
    import fcntl
    output = run_root(manifest) / experiment_id
    output.mkdir(parents=True, exist_ok=True)
    # The worker owns this lock from before reading configuration until final
    # metrics are written, even if its parent queue dies before recording its PID.
    with (output / 'experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return train_impl(manifest, experiment_id)


def train_impl(manifest, experiment_id):
    experiment = next(item for item in manifest['experiments'] if item['id'] == experiment_id)
    config_path = ROOT / experiment['config']
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    output = run_root(manifest) / experiment_id
    output.mkdir(parents=True, exist_ok=True)
    prepared = run_root(manifest) / 'prepared' / experiment['dataset']
    metadata = read_json(prepared / 'metadata.json')
    identity = experiment_identity(config_path, prepared)
    if verify_completed(output, identity):
        print(f'SKIP complete {experiment_id}', flush=True)
        return
    if (output / 'identity.json').exists() and read_json(output / 'identity.json') != identity:
        raise ValueError('Refuse to resume an experiment with changed code/config/data')
    atomic_json(output / 'identity.json', identity)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    atomic_json(output / 'data_reference.json', metadata)
    seed = int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no silent CPU fallback')
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(manifest.get('cpu_threads_per_worker', 2)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0')
    amp = config.get('precision', 'fp32') == 'bf16'
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Configured BF16 is not supported by this GPU')
    generator = torch.Generator().manual_seed(seed)
    image = np.load(prepared / 'image.npy', mmap_mode='r')
    gt = np.load(prepared / 'gt.npy', mmap_mode='r')
    with np.load(prepared / 'split.npz') as split:
        train_indices, test_indices = split['train_indices'], split['test_indices']
    validate_split(gt, train_indices, test_indices, metadata['train_counts'])
    # Each run references the exact common split; no per-model random split is made.
    atomic_json(output / 'split_reference.json', {'path': str((prepared / 'split.npz').relative_to(ROOT)),
                                                'sha256': metadata['prepared_sha256']['split.npz']})
    bank = PatchBank(image, int(config['patch_size']), device)
    labels = torch.from_numpy(np.array(gt.reshape(-1), copy=True)).to(device)
    model = build_main_model(config, len(metadata['class_names'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    owned = {id(p) for group in optimizer.param_groups for p in group['params']}
    if owned != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Optimizer does not cover trainable model parameters')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    start_epoch, history, previous_seconds = 0, [], 0.0
    checkpoint = output / 'resume.pth'
    if checkpoint.exists():
        # This is our own resume.pth, paired with the checked identity.json.
        # It contains Python/NumPy RNG states, not only tensors. Never use this
        # loading path for downloaded or otherwise untrusted checkpoints.
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, history, previous_seconds = state['epoch'], state['history'], state['train_seconds']
        generator.set_state(state['data_rng'].cpu())
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda_rng']])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        print(f'RESUME {experiment_id} epoch={start_epoch}', flush=True)
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    atomic_json(output / 'environment.json', {'commit': git_commit, 'source_sha256': identity['source_sha256'],
                                            'python': sys.version, 'executable': sys.executable,
                                            'torch': torch.__version__, 'cuda': torch.version.cuda,
                                            'hostname': socket.gethostname(), 'torch_module': torch.__file__,
                                            'numpy_module': np.__file__,
                                            'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                                            'gpu': torch.cuda.get_device_name(0), 'precision': config['precision'],
                                            'matmul_tf32': False, 'cudnn_tf32': True,
                                            'seed': seed, 'pid': os.getpid()})
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    parameters = sum(p.numel() for p in model.parameters())
    print(f'START {experiment_id} parameters={parameters} train={len(train_indices)} test={len(test_indices)}', flush=True)
    for epoch in range(start_epoch + 1, config['epochs'] + 1):
        epoch_start = time.monotonic()
        model.train()
        order = torch.randperm(len(train_indices), generator=generator).numpy()
        loss_sum = 0.0
        for start in range(0, len(order), config['batch_size']):
            selected = torch.from_numpy(train_indices[order[start:start + config['batch_size']]]).to(device)
            inputs = augment(bank.take(selected), generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                losses = model(inputs[:, :config['ms_channels']], inputs[:, config['ms_channels']:],
                               labels=labels[selected])['losses']
                loss = losses['total']
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
        scheduler.step()
        elapsed = previous_seconds + time.monotonic() - started
        history.append({'epoch': epoch, 'loss': loss_sum / len(train_indices),
                        'lr': scheduler.get_last_lr()[0], 'seconds': time.monotonic() - epoch_start})
        atomic_json(output / 'progress.json', {'phase': 'training', 'epoch': epoch, 'epochs': config['epochs'],
                                              'loss': history[-1]['loss'], 'pid': os.getpid(), 'updated': time.time()})
        if epoch == 1 or epoch % 10 == 0 or epoch == config['epochs']:
            print(f'{experiment_id} epoch={epoch}/{config["epochs"]} loss={history[-1]["loss"]:.6f} seconds={elapsed:.1f}', flush=True)
            atomic_json(output / 'history.json', history)
            save_checkpoint(checkpoint, {'identity': identity, 'model': model.state_dict(),
                                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                                        'epoch': epoch, 'history': history, 'train_seconds': elapsed,
                                        'data_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                                        'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
                                        'numpy_rng': np.random.get_state()})
    train_seconds = previous_seconds + time.monotonic() - started
    save_checkpoint(output / 'model_last.pth', model.state_dict())
    model.eval()
    classes = len(metadata['class_names'])
    cm = torch.zeros(classes * classes, dtype=torch.int64, device=device)
    predictions = np.lib.format.open_memmap(output / 'test_predictions.npy', mode='w+', dtype=np.int16,
                                           shape=(len(test_indices),))
    eval_started = time.monotonic()
    last_report = eval_started
    eval_batch = config['test_batch_size']
    with torch.inference_mode():
        for start in range(0, len(test_indices), eval_batch):
            selected = torch.from_numpy(test_indices[start:start + eval_batch]).to(device)
            inputs = bank.take(selected)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                logits = model(inputs[:, :config['ms_channels']], inputs[:, config['ms_channels']:])['logits']
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite final test logits')
            predicted = logits.argmax(1)
            cm += torch.bincount(labels[selected] * classes + predicted, minlength=classes * classes)
            predictions[start:start + len(selected)] = predicted.cpu().numpy()
            if time.monotonic() - last_report > 20 or start + eval_batch >= len(test_indices):
                atomic_json(output / 'progress.json', {'phase': 'testing', 'tested': min(start + eval_batch, len(test_indices)),
                                                      'total': len(test_indices), 'pid': os.getpid(), 'updated': time.time()})
                print(f'{experiment_id} test={min(start + eval_batch, len(test_indices))}/{len(test_indices)}', flush=True)
                last_report = time.monotonic()
    predictions.flush()
    result = metrics_from_confusion(cm.view(classes, classes).cpu().numpy())
    result.update({'completed': True, 'experiment_id': experiment_id, 'variant': config['variant'],
                   'dataset': experiment['dataset'], 'seed': seed, 'epochs': config['epochs'],
                   'train_counts': metadata['train_counts'], 'test_counts': metadata['test_counts'],
                   'class_names': metadata['class_names'], 'parameters': parameters,
                   'train_seconds': train_seconds, 'test_seconds': time.monotonic() - eval_started,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
                   'checkpoint': 'model_last.pth', 'evaluation_scope': 'held_out_test_final_epoch',
                   'checkpoint_sha256': digest(output / 'model_last.pth'),
                   'precision': config['precision'], 'git_commit': git_commit, 'identity': identity})
    atomic_json(output / 'metrics.json', result)
    atomic_json(output / 'progress.json', {'phase': 'completed', 'pid': os.getpid(), 'updated': time.time()})
    print(f'COMPLETE {experiment_id} OA={result["oa"]:.4f} AA={result["aa"]:.4f} F1={result["macro_f1"]:.4f}', flush=True)


def summarize(manifest):
    root = run_root(manifest)
    results = [read_json(root / item['id'] / 'metrics.json') for item in manifest['experiments']
               if (root / item['id'] / 'metrics.json').exists()]
    atomic_json(root / 'summary.json', results)
    if results:
        fields = ('experiment_id', 'dataset', 'seed', 'oa', 'aa', 'kappa', 'macro_f1', 'parameters',
                  'train_seconds', 'test_seconds', 'peak_allocated_mib')
        with (root / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
    return results


def selected_gpu():
    selector = os.environ.get('CUDA_VISIBLE_DEVICES', '0').strip()
    if not selector or ',' in selector or not (selector.isdigit() or selector.startswith('GPU-')):
        raise ValueError('Select exactly one physical GPU index or GPU UUID in CUDA_VISIBLE_DEVICES')
    return selector


def effective_workers(manifest):
    limit = int(manifest['workers'])
    workers = int(os.environ.get('MAIN_EXPERIMENT_WORKERS', str(limit)))
    if not 1 <= workers <= limit:
        raise ValueError(f'MAIN_EXPERIMENT_WORKERS must be between 1 and {limit}')
    return workers


def external_gpu_processes(allowed_pids):
    text = subprocess.check_output(['nvidia-smi', '--id=' + selected_gpu(), '--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    external = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(',')]
        if parts and parts[0].isdigit() and int(parts[0]) not in allowed_pids:
            external.append({'pid': int(parts[0]), 'memory_mib': parts[1] if len(parts) > 1 else None})
    return external


def queue(manifest, manifest_path):
    import fcntl
    root = run_root(manifest)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    workers, gpu = effective_workers(manifest), selected_gpu()
    atomic_json(root / 'queue_pid.json', {'pid': os.getpid(), 'started': time.time(), 'sid': os.getsid(0),
                                        'hostname': socket.gethostname(), 'physical_gpu': gpu, 'workers': workers})
    state_path = root / 'queue_status.json'
    state = read_json(state_path) if state_path.exists() else {'experiments': {}, 'started': time.time()}
    for name, item in state.get('experiments', {}).items():
        if item.get('status') == 'running' and worker_alive(item.get('pid', -1), name):
            raise RuntimeError(f'Previous worker still alive: {name} pid={item["pid"]}; refusing duplicate queue')
    for item in manifest['experiments']:
        lock_path = root / item['id'] / 'experiment.lock'
        if lock_path.exists():
            with lock_path.open('a') as worker_lock:
                fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(worker_lock, fcntl.LOCK_UN)
    prepare(manifest)
    for variant in VARIANTS:
        pending = []
        for item in manifest['experiments']:
            if item['variant'] != variant:
                continue
            identity = experiment_identity(ROOT / item['config'], root / 'prepared' / item['dataset'])
            if verify_completed(root / item['id'], identity):
                state['experiments'][item['id']] = {'status': 'completed'}
            elif state['experiments'].get(item['id'], {}).get('status') == 'failed':
                print(f'SKIP failed (manual investigation required): {item["id"]}', flush=True)
            else:
                pending.append(item)
        active = []
        last_blockers = None
        while pending or active:
            blockers = external_gpu_processes({process.pid for _, process, _ in active}) \
                if manifest.get('wait_for_gpu_idle', True) else []
            blocker_pids = tuple(item['pid'] for item in blockers)
            if blocker_pids != last_blockers:
                if blockers:
                    print(f'WAIT GPU: other compute processes {blockers}', flush=True)
                last_blockers = blocker_pids
            while pending and len(active) < workers:
                if blockers:
                    break
                item = pending.pop(0)
                folder = root / item['id']
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / 'console.log').open('a', encoding='utf-8')
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                            '--manifest', str(manifest_path), 'train', '--experiment', item['id']],
                                           cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                active.append((item, process, log))
                state['experiments'][item['id']] = {'status': 'running', 'pid': process.pid, 'started': time.time()}
                atomic_json(state_path, state)
                print(f'LAUNCH {item["id"]} pid={process.pid}', flush=True)
            for item, process, log in list(active):
                code = process.poll()
                if code is not None:
                    complete = code == 0 and (root / item['id'] / 'metrics.json').exists()
                    state['experiments'][item['id']].update(status='completed' if complete else 'failed',
                                                          returncode=code, finished=time.time())
                    print(f'EXIT {item["id"]} code={code} completed={complete}', flush=True)
                    log.close()
                    active.remove((item, process, log))
                    summarize(manifest)
            state.update({'pid': os.getpid(), 'updated': time.time(),
                          'phase': 'waiting_gpu' if blockers and not active else 'running',
                          'physical_gpu': gpu, 'workers': workers,
                          'gpu_blockers': blockers, 'variant': variant})
            atomic_json(state_path, state)
            if active or pending:
                time.sleep(5)
    summarize(manifest)
    state.update({'phase': 'finished', 'updated': time.time()})
    atomic_json(state_path, state)
    print('QUEUE FINISHED', flush=True)


def launch(manifest, manifest_path):
    root = run_root(manifest)
    root.mkdir(parents=True, exist_ok=True)
    pid_path = root / 'queue_pid.json'
    if pid_path.exists():
        old = read_json(pid_path)['pid']
        try:
            command = Path(f'/proc/{old}/cmdline').read_bytes()
        except FileNotFoundError:
            command = b''
        if b'main_experiment.py' in command:
            raise RuntimeError(f'Queue already running: {old}')
    log_dir = ROOT / manifest['log_root']
    log_dir.mkdir(parents=True, exist_ok=True)
    session = manifest['screen_session']
    existing = subprocess.run(['screen', '-ls', session], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if existing.returncode == 0:
        raise RuntimeError(f'screen session already exists: {session}')
    # Positional arguments avoid interpolation of paths into shell source.
    subprocess.run(['screen', '-dmS', session, 'bash', '-c', 'exec "$@" >>"$MAIN_QUEUE_LOG" 2>&1',
                    'main-queue', sys.executable, '-u', str(Path(__file__).resolve()), '--manifest',
                    str(manifest_path), 'queue'], cwd=ROOT, check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': selected_gpu(), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                        'MAIN_EXPERIMENT_WORKERS': str(effective_workers(manifest)), 'OMP_NUM_THREADS': '2',
                        'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1', 'MAIN_QUEUE_LOG': str(log_dir / 'queue.log')})
    for _ in range(20):
        time.sleep(0.5)
        if pid_path.exists():
            info = read_json(pid_path)
            command = Path(f'/proc/{info["pid"]}/cmdline')
            if command.exists() and b'main_experiment.py' in command.read_bytes():
                print(json.dumps({'screen_session': session, 'pid': info['pid'],
                                  'log': str(log_dir / 'queue.log'), 'output': str(root)}, indent=2))
                return
    raise RuntimeError(f'Queue not confirmed; inspect screen -ls and {log_dir / "queue.log"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('action', choices=('prepare', 'train', 'queue', 'launch', 'status'))
    parser.add_argument('--experiment')
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    if args.action == 'prepare':
        prepare(manifest)
    elif args.action == 'train':
        train(manifest, args.experiment)
    elif args.action == 'queue':
        queue(manifest, manifest_path)
    elif args.action == 'launch':
        launch(manifest, manifest_path)
    else:
        state_path = run_root(manifest) / 'queue_status.json'
        state = read_json(state_path) if state_path.exists() else {'phase': 'not_started'}
        for name, item in state.get('experiments', {}).items():
            progress = run_root(manifest) / name / 'progress.json'
            if progress.exists():
                item['progress'] = read_json(progress)
        print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
