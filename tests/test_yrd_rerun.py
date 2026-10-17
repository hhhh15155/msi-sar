import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.main_experiment import read_manifest


class YRDRerunTests(unittest.TestCase):
    def test_seven_version_single_dataset_manifest_is_accepted(self):
        from models.main_classification import VARIANTS
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.json'
            jobs = [{'id': f'{v}_yrd', 'variant': v, 'dataset': 'yrd', 'config': f'{v}.yaml'} for v in VARIANTS]
            path.write_text(json.dumps({'datasets': {'yrd': {}}, 'experiments': jobs}))
            self.assertEqual(len(read_manifest(path)['experiments']), 7)
            jobs[-1] = jobs[0]
            path.write_text(json.dumps({'datasets': {'yrd': {}}, 'experiments': jobs}))
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_rerun_configs_preserve_all_training_parameters(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        manifest = read_manifest(root / 'configs/yrd_per_channel/manifest.json')
        self.assertEqual(manifest['datasets']['yrd']['train_counts'], [50] * 8)
        self.assertEqual(manifest['seed'], 202201)
        for job in manifest['experiments']:
            current = yaml.safe_load((root / job['config']).read_text())
            original = yaml.safe_load((root / 'configs/main_first_pass' / (job['id'] + '.yaml')).read_text())
            self.assertEqual(current, original)

    def test_prepare_reuses_split_bytes_and_checks_reference_hash(self):
        from scripts import main_experiment as engine
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('data/yrd/data.mat', 'data/yrd/label.mat', 'baselines/dfinet/dataset.py', 'baselines/split.py'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            data_config = root / 'dataset.json'
            data_config.write_text(json.dumps({'path': 'data/yrd', 'normalization': 'per_channel_minmax'}))
            split = root / 'original.npz'
            np.savez_compressed(split, train_indices=[0, 1], test_indices=[2, 3])
            reference = {'path': 'original.npz', 'sha256': engine.digest(split)}
            manifest = {'output_root': 'runs/new', 'seed': 202201,
                        'datasets': {'yrd': {'dataset_config': 'dataset.json', 'train_counts': [1, 1],
                                             'reuse_split': reference}}}
            image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            gt = np.array([[0, 1], [0, 1]])
            with patch.object(engine, 'ROOT', root), patch.object(engine, 'load_dataset', return_value=(image, gt, ['a', 'b'])):
                engine.prepare(manifest)
                self.assertEqual(engine.digest(root / 'runs/new/prepared/yrd/split.npz'), engine.digest(split))
                reference['sha256'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'reference split'):
                    engine.prepare(manifest)


class PublicationTests(unittest.TestCase):
    def fixture(self, temporary):
        from models.main_classification import VARIANTS
        from scripts import main_experiment as engine
        root = Path(temporary)
        current, source = root / 'runs/current', root / 'runs/new'
        jobs = [{'id': f'{v}_{d}', 'variant': v, 'dataset': d}
                for v in VARIANTS for d in ('yrd', 'yrd2509new', 'grss07')]
        config_dir = root / 'configs'
        config_dir.mkdir()
        baseline = {'output_root': 'runs/current', 'datasets': {d: {} for d in ('yrd', 'yrd2509new', 'grss07')},
                    'experiments': jobs}
        (root / 'configs/base.json').write_text(json.dumps(baseline))
        new_jobs = []
        for job in jobs:
            folder = current / job['id']
            folder.mkdir(parents=True)
            result = {**engine.metrics_from_confusion(np.eye(2, dtype=np.int64)), 'completed': True,
                      'experiment_id': job['id'], 'dataset': job['dataset'], 'seed': 202201,
                      'parameters': 1, 'train_seconds': 1, 'test_seconds': 1, 'peak_allocated_mib': 1}
            engine.atomic_json(folder / 'metrics.json', result)
            if job['dataset'] == 'yrd':
                config = 'configs/' + job['id'] + '.json'
                (root / config).write_text(json.dumps({'variant': job['variant']}))
                new_jobs.append({**job, 'config': config})
        for base, datasets in ((current, ('yrd', 'yrd2509new', 'grss07')), (source, ('yrd',))):
            for dataset in datasets:
                prepared = base / 'prepared' / dataset
                prepared.mkdir(parents=True)
                np.save(prepared / 'image.npy', np.zeros((2, 2, 2), dtype=np.float32))
                np.save(prepared / 'gt.npy', np.array([[0, 1], [0, 1]]))
                np.savez_compressed(prepared / 'split.npz', train_indices=[0, 1], test_indices=[2, 3])
                engine.atomic_json(prepared / 'metadata.json', {'normalization': 'per_channel_minmax',
                    'train_counts': [1, 1], 'test_counts': [1, 1], 'class_names': ['a', 'b'],
                    'prepared_sha256': {n: engine.digest(prepared / n) for n in ('image.npy', 'gt.npy', 'split.npz')}})
        prepared = source / 'prepared/yrd'
        for job in new_jobs:
            folder = source / job['id']
            folder.mkdir()
            (folder / 'model_last.pth').write_bytes(b'own checkpoint fixture')
            np.save(folder / 'test_predictions.npy', np.array([0, 1], dtype=np.int16))
            result = {**engine.metrics_from_confusion(np.eye(2, dtype=np.int64)), 'completed': True,
                      'experiment_id': job['id'], 'dataset': 'yrd', 'seed': 202201, 'class_names': ['a', 'b'],
                      'train_counts': [1, 1], 'test_counts': [1, 1],
                      'identity': engine.experiment_identity(root / job['config'], prepared),
                      'checkpoint_sha256': engine.digest(folder / 'model_last.pth'),
                      'parameters': 1, 'train_seconds': 1, 'test_seconds': 1, 'peak_allocated_mib': 1}
            engine.atomic_json(folder / 'metrics.json', result)
        manifest = {'name': 'unit_yrd_revision', 'output_root': 'runs/new',
                    'datasets': {'yrd': {'reuse_split': {'sha256': engine.digest(prepared / 'split.npz')}}},
                    'experiments': new_jobs,
                    'promotion': {'target_root': 'runs/current', 'archive_root': 'runs/archive/old',
                                  'base_manifest': 'configs/base.json'}}
        return root, manifest

    def test_publish_all_seven_and_preserve_other_fourteen_with_archive(self):
        from scripts.promote_yrd_results import promote_results
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest = self.fixture(temporary)
            before = {p.parent.name: p.read_bytes() for p in (root / 'runs/current').glob('v*/metrics.json')
                      if not p.parent.name.endswith('_yrd')}
            result = promote_results(manifest, root)
            self.assertEqual(result['status'], 'published')
            self.assertTrue((root / 'runs/archive/old/v0_msi_yrd/metrics.json').exists())
            self.assertEqual(len(list((root / 'runs/current').glob('v*/metrics.json'))), 21)
            for name, data in before.items():
                self.assertEqual((root / 'runs/current' / name / 'metrics.json').read_bytes(), data)
            self.assertEqual(promote_results(manifest, root)['status'], 'already_published')

    def test_incomplete_candidate_does_not_touch_current_results(self):
        from scripts.promote_yrd_results import promote_results
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest = self.fixture(temporary)
            (root / 'runs/new/v6_current_yrd/metrics.json').unlink()
            with self.assertRaises(ValueError):
                promote_results(manifest, root)
            self.assertTrue((root / 'runs/current/v0_msi_yrd/metrics.json').exists())
            self.assertFalse((root / 'runs/archive/old').exists())

    def test_recovery_after_old_results_archived_before_final_rename(self):
        from scripts.promote_yrd_results import promote_results
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest = self.fixture(temporary)
            original_rename = Path.rename
            stage = root / 'runs/.current_staging_yrd_pc'
            def interrupted(path, target):
                if path == stage:
                    raise RuntimeError('simulated interruption during final rename')
                return original_rename(path, target)
            with patch.object(Path, 'rename', interrupted):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    promote_results(manifest, root)
            self.assertFalse((root / 'runs/current').exists())
            self.assertEqual(promote_results(manifest, root)['status'], 'published_recovered')
            self.assertEqual(len(list((root / 'runs/current').glob('v*/metrics.json'))), 21)


if __name__ == '__main__':
    unittest.main()
