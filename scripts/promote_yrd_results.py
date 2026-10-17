"""Publish all seven verified YRD replacements together, keeping an archive."""
from pathlib import Path
import json
import os
import shutil
import time

import numpy as np

from scripts.main_experiment import (ROOT, atomic_json, digest, experiment_identity, metrics_from_confusion,
                                     read_json, summarize, verify_completed)


def _copy(source, destination):
    def link_file(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return dst
    shutil.copytree(source, destination, copy_function=link_file)


def promote_results(manifest, project_root=None):
    import fcntl
    root = Path(project_root or ROOT).resolve()
    source = (root / manifest['output_root']).resolve()
    specification = manifest['promotion']
    current = (root / specification['target_root']).resolve()
    archive = (root / specification['archive_root']).resolve()
    runs = (root / 'runs').resolve()
    if any(runs not in path.parents for path in (source, current, archive)) or len({source, current, archive}) != 3:
        raise ValueError('Publication paths must be distinct directories inside this project runs/')
    stage = current.with_name('.' + current.name + '_staging_yrd_pc')
    with (runs / '.yrd_publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _publish(manifest, root, source, current, archive, stage)


def _publish(manifest, root, source, current, archive, stage):
    # Validate every candidate before renaming or modifying the current result set.
    metadata = read_json(source / 'prepared/yrd/metadata.json')
    reference = manifest['datasets']['yrd']['reuse_split']['sha256']
    if metadata['normalization'] != 'per_channel_minmax' or digest(source / 'prepared/yrd/split.npz') != reference:
        raise ValueError('Expected per-channel YRD preprocessing with the exact original split')
    for name, expected in metadata['prepared_sha256'].items():
        if digest(source / 'prepared/yrd' / name) != expected:
            raise ValueError(f'Prepared artifact changed: {name}')
    with np.load(source / 'prepared/yrd/split.npz') as split:
        indices = split['test_indices']
    gt = np.load(source / 'prepared/yrd/gt.npy', mmap_mode='r').reshape(-1)
    verified = {}
    for job in manifest['experiments']:
        if job['dataset'] != 'yrd':
            raise ValueError('This publication only replaces YRD')
        folder = source / job['id']
        identity = experiment_identity(root / job['config'], source / 'prepared/yrd')
        if not verify_completed(folder, identity):
            raise ValueError(f'Cannot publish incomplete experiment: {job["id"]}')
        result = read_json(folder / 'metrics.json')
        predictions = np.load(folder / 'test_predictions.npy', mmap_mode='r')
        classes = len(result['class_names'])
        if predictions.shape != indices.shape or np.any(predictions < 0) or np.any(predictions >= classes):
            raise ValueError('Invalid saved predictions')
        cm = np.bincount(gt[indices] * classes + predictions, minlength=classes ** 2).reshape(classes, classes)
        recomputed = metrics_from_confusion(cm)
        if not np.array_equal(cm, np.asarray(result['confusion_matrix'])):
            raise ValueError('Saved predictions do not reproduce the confusion matrix')
        for key in ('oa', 'aa', 'kappa', 'macro_f1'):
            if abs(result[key] - recomputed[key]) > 1e-7:
                raise ValueError(f'Invalid metric: {job["id"]} {key}')
        if result['train_counts'] != metadata['train_counts'] or result['test_counts'] != metadata['test_counts']:
            raise ValueError('Class counts do not match the shared split')
        verified[job['id']] = digest(folder / 'metrics.json')
    if len(verified) != 7:
        raise ValueError('Publish exactly seven YRD versions together')

    marker_name = 'current_results.json'
    marker = {'revision': manifest['name'], 'host': 'server116', 'normalization_yrd': 'per_channel_minmax',
              'yrd_origin': manifest['output_root'], 'unchanged_datasets_origin': 'server110 original 21-run snapshot',
              'archive': str(archive.relative_to(root)), 'verified_yrd_metrics_sha256': verified,
              'published_at': time.time()}
    if (current / marker_name).exists() and read_json(current / marker_name).get('revision') == manifest['name']:
        if any(digest(current / name / 'metrics.json') != sha for name, sha in verified.items()):
            raise ValueError('Published metrics changed after publication')
        return {'status': 'already_published', 'current': str(current), 'archive': str(archive)}

    # Recovery after the old directory was archived but before the final rename.
    if not current.exists() and archive.exists() and (stage / marker_name).exists():
        if read_json(stage / marker_name)['verified_yrd_metrics_sha256'] != verified:
            raise ValueError('Interrupted publication has a different candidate set')
        stage.rename(current)
        atomic_json(source / 'publication.json', {'status': 'published', **marker})
        return {'status': 'published_recovered', 'current': str(current), 'archive': str(archive)}
    if not current.is_dir() or archive.exists():
        raise ValueError('Missing current snapshot or archive path already occupied; preserve all files')
    baseline = read_json(root / manifest['promotion']['base_manifest'])
    if len(baseline['experiments']) != 21:
        raise ValueError('Expected the original 21-experiment result set')
    unchanged = {}
    for job in baseline['experiments']:
        metrics = current / job['id'] / 'metrics.json'
        if not metrics.exists() or read_json(metrics).get('completed') is not True:
            raise ValueError('The imported baseline snapshot is incomplete')
        if job['dataset'] != 'yrd':
            unchanged[job['id']] = digest(metrics)
    if stage.exists():
        # Retain partial staging artifacts rather than recursively deleting them.
        stage.rename(stage.with_name(stage.name + '_interrupted_' + str(time.time_ns())))
    stage.mkdir()
    for job in baseline['experiments']:
        origin = source if job['dataset'] == 'yrd' else current
        _copy(origin / job['id'], stage / job['id'])
    (stage / 'prepared').mkdir()
    for dataset in baseline['datasets']:
        origin = source if dataset == 'yrd' else current
        _copy(origin / 'prepared' / dataset, stage / 'prepared' / dataset)
    for name, expected in unchanged.items():
        if digest(stage / name / 'metrics.json') != expected:
            raise ValueError('Unchanged dataset metrics were modified')
    summarize({**baseline, 'output_root': str(stage)})
    marker['unchanged_metrics_sha256'] = unchanged
    atomic_json(stage / marker_name, marker)
    atomic_json(stage / 'queue_status.json', {'phase': 'finished', 'revision': manifest['name'],
                'updated': time.time(), 'experiments': {job['id']: {'status': 'completed',
                'origin_host': 'server116' if job['dataset'] == 'yrd' else 'server110'}
                for job in baseline['experiments']}})
    atomic_json(source / 'publication.json', {'status': 'ready_to_swap', **marker})
    archive.parent.mkdir(parents=True, exist_ok=True)
    current.rename(archive)
    stage.rename(current)
    atomic_json(source / 'publication.json', {'status': 'published', **marker})
    print(f'PUBLISHED 7 YRD replacements: {current}; previous snapshot archived: {archive}', flush=True)
    return {'status': 'published', 'current': str(current), 'archive': str(archive)}
