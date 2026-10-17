"""Contracts for the seven-version single-seed classification experiment."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.main_classification import MainClassifier, build_main_model
from scripts.main_experiment import PatchBank, metrics_from_confusion, validate_split


class MainClassificationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_all_variants_odd_patches_and_registered_gradients(self):
        for mode in ('msi', 'sar', 'concat', 'fcm_ffm', 'mafi', 'fuzzy'):
            for patch in (1, 9, 11):
                with self.subTest(mode=mode, patch=patch):
                    model = MainClassifier(6, 1, 5, mode, dim=16, heads=4, depth=1, dropout=0)
                    out = model(torch.randn(1, 6, patch, patch), torch.randn(1, 1, patch, patch),
                                labels=torch.tensor([2]))
                    self.assertEqual(out['logits'].shape, (1, 5))
                    self.assertTrue(torch.isfinite(out['losses']['total']))
                    out['losses']['total'].backward()
                    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
                    self.assertEqual(missing, [])
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_single_source_ignores_other_input_and_eval_is_stable(self):
        ms, sar = torch.randn(2, 10, 5, 5), torch.randn(2, 4, 5, 5)
        for mode in ('msi', 'sar'):
            model = MainClassifier(10, 4, 8, mode, dim=16, heads=4, depth=1).eval()
            with torch.no_grad():
                first = model(ms, sar)['logits']
                changed = model(ms if mode == 'msi' else ms + 100,
                                sar if mode == 'sar' else sar - 100)['logits']
            torch.testing.assert_close(first, changed, rtol=0, atol=0)

    def test_reject_even_patch_and_support_two_channel_configs(self):
        for ms_c, sar_c in ((6, 1), (10, 4)):
            model = MainClassifier(ms_c, sar_c, 5, 'concat', dim=16, heads=4, depth=1)
            with self.assertRaises(ValueError):
                model(torch.randn(2, ms_c, 4, 4), torch.randn(2, sar_c, 4, 4))

    def test_full_version_keeps_existing_architecture(self):
        from models.fuzzy_transformer import FuzzyTransformer
        model = build_main_model({'variant': 'v6_current', 'ms_channels': 6, 'sar_channels': 1}, 5)
        self.assertIsInstance(model, FuzzyTransformer)
        self.assertEqual(model.ms_stem.channel_mix.out_channels, 192)
        self.assertEqual(model.auxiliary_weight, 0.2)

    def test_patch_bank_matches_reflect_edges(self):
        image = np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2)
        bank = PatchBank(image, 3, torch.device('cpu'))
        indices = torch.tensor([0, 4, 7, 19])
        patches = bank.take(indices)
        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='reflect')
        expected = np.stack([padded[i // 5:i // 5 + 3, i % 5:i % 5 + 3].transpose(2, 0, 1)
                             for i in indices.tolist()])
        np.testing.assert_array_equal(patches.numpy(), expected)

    def test_split_partition_and_counts_are_checked(self):
        gt = np.array([[0, 0, 1], [1, 0, -1]])
        train = np.array([0, 2])
        test = np.array([1, 3, 4])
        validate_split(gt, train, test, [1, 1])
        with self.assertRaises(ValueError):
            validate_split(gt, train, np.array([0, 3, 4]), [1, 1])
        with self.assertRaises(ValueError):
            validate_split(gt, train, test, [2, 1])

    def test_metrics_include_macro_f1(self):
        result = metrics_from_confusion(np.array([[2, 1], [0, 1]]))
        self.assertAlmostEqual(result['oa'], 75)
        self.assertAlmostEqual(result['aa'], 250 / 3)
        self.assertAlmostEqual(result['macro_f1'], 100 * ((4 / 5 + 2 / 3) / 2))

    @unittest.skipUnless(__import__('os').name == 'posix', 'POSIX deployment lock')
    def test_worker_lock_prevents_duplicate_training_before_reading_config(self):
        import fcntl
        from scripts import main_experiment as engine
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'already_running'
            folder.mkdir()
            manifest = {'output_root': temporary, 'experiments': [
                {'id': 'already_running', 'dataset': 'synthetic', 'config': '/missing.yaml'}]}
            with (folder / 'experiment.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    engine.train(manifest, 'already_running')

    def test_gpu_admission_distinguishes_external_and_own_workers(self):
        from unittest.mock import patch
        from scripts import main_experiment as engine
        with patch.object(engine.subprocess, 'check_output', return_value='123, 12000\n456, 3000\n'):
            self.assertEqual(engine.external_gpu_processes({456}), [{'pid': 123, 'memory_mib': '12000'}])
            self.assertEqual(engine.external_gpu_processes({123, 456}), [])

    def test_shared_dual_encoders_and_head_have_identical_initial_weights(self):
        reference = None
        for mode in ('concat', 'fcm_ffm', 'mafi', 'fuzzy'):
            torch.manual_seed(202201)
            model = MainClassifier(10, 4, 8, mode, dim=16, heads=4, depth=1)
            common = {n: p for n, p in model.state_dict().items() if not n.startswith('fusion.')}
            if reference is None:
                reference = common
            else:
                for name, tensor in common.items():
                    torch.testing.assert_close(tensor, reference[name], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA integration check')
    def test_every_variant_runs_bf16_training_and_evaluation(self):
        from models.main_classification import VARIANTS
        for variant in VARIANTS:
            for ms_c, sar_c in ((6, 1), (10, 4)):
                with self.subTest(variant=variant, channels=(ms_c, sar_c)):
                    model = build_main_model({'variant': variant, 'ms_channels': ms_c, 'sar_channels': sar_c,
                                              'embed_dim': 16, 'num_heads': 4, 'depth': 1}, 5).cuda()
                    ms, sar = torch.randn(2, ms_c, 11, 11, device='cuda'), torch.randn(2, sar_c, 11, 11, device='cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = model(ms, sar, labels=torch.tensor([1, 2], device='cuda'))['losses']['total']
                    loss.backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                    model.eval()
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        first, second = model(ms, sar)['logits'], model(ms, sar)['logits']
                    self.assertTrue(torch.isfinite(first).all())
                    torch.testing.assert_close(first, second, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA integration check')
    def test_gpu_training_resume_matches_uninterrupted_and_completed_identity_is_checked(self):
        import json
        from unittest.mock import patch
        import yaml
        from scripts import main_experiment as engine
        (engine.ROOT / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=engine.ROOT / 'tmp') as temporary:
            root = Path(temporary)
            prepared = root / 'prepared' / 'synthetic'
            prepared.mkdir(parents=True)
            rng = np.random.default_rng(2)
            image = rng.normal(size=(6, 7, 7)).astype(np.float32)
            gt = (np.arange(42).reshape(6, 7) % 3).astype(np.int64)
            train = np.array([0, 1, 2, 3, 4, 5])
            test = np.arange(6, 42)
            np.save(prepared / 'image.npy', image)
            np.save(prepared / 'gt.npy', gt)
            np.savez(prepared / 'split.npz', train_indices=train, test_indices=test)
            engine.atomic_json(prepared / 'metadata.json', {'class_names': ['a', 'b', 'c'],
                               'train_counts': [2, 2, 2], 'test_counts': [12, 12, 12],
                               'prepared_sha256': {'split.npz': engine.digest(prepared / 'split.npz')}})
            config = {'variant': 'v2_concat', 'ms_channels': 6, 'sar_channels': 1, 'embed_dim': 16,
                      'num_heads': 4, 'depth': 1, 'seed': 202201, 'patch_size': 3, 'epochs': 2,
                      'batch_size': 3, 'test_batch_size': 8, 'learning_rate': 0.0005,
                      'weight_decay': 0.0001, 'precision': 'bf16'}
            config_path = root / 'test_config.yaml'
            config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
            manifest = {'output_root': str(root), 'experiments': [
                {'id': name, 'dataset': 'synthetic', 'config': str(config_path)}
                for name in ('uninterrupted', 'resume')]}
            engine.train(manifest, 'uninterrupted')
            original_save = engine.save_checkpoint

            def interrupt_after_checkpoint(path, payload):
                original_save(path, payload)
                if path.name == 'resume.pth' and payload['epoch'] == 1:
                    raise RuntimeError('simulated interruption')

            with patch.object(engine, 'save_checkpoint', side_effect=interrupt_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    engine.train(manifest, 'resume')
            engine.train(manifest, 'resume')
            expected = torch.load(root / 'uninterrupted' / 'model_last.pth', map_location='cpu')
            actual = torch.load(root / 'resume' / 'model_last.pth', map_location='cpu')
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
            result = json.loads((root / 'resume' / 'metrics.json').read_text())
            self.assertEqual(sum(map(sum, result['confusion_matrix'])), 36)
            config_path.write_text(yaml.safe_dump({**config, 'seed': 99}))
            with self.assertRaisesRegex(ValueError, 'different code/config/data'):
                engine.train(manifest, 'resume')


if __name__ == '__main__':
    unittest.main()
