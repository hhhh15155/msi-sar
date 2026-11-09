"""Synthetic ablation checks: no real data or formal training."""
import io
import json
import unittest
from pathlib import Path
import importlib.util
import tempfile
from unittest.mock import patch as mock_patch

import torch

from baselines.m2heat.model import M2Heat


class M2HeatAblationTests(unittest.TestCase):
    def test_private_parts_share_the_head_and_use_disjoint_raw_channel_slots(self):
        from baselines.m2heat.ablation import M2HeatAblation
        torch.set_num_threads(1)
        torch.manual_seed(42)
        reference = M2HeatAblation('private').net.model
        for variant, stem, hclayer in (('private_neither', False, False),
                                       ('private_stem_only', True, False),
                                       ('private_hclayer_only', False, True)):
            with self.subTest(variant=variant):
                torch.manual_seed(42)
                core = M2HeatAblation(variant).net.model.eval()
                self.assertEqual(hasattr(core, 'stem_hsi'), stem)
                self.assertEqual(hasattr(core, 'hclayer_hsi'), hclayer)
                for name, weight in core.state_dict().items():
                    torch.testing.assert_close(weight, reference.state_dict()[name], rtol=0, atol=0)
                if not stem and not hclayer:
                    msi, sar = torch.randn(2, 10, 11, 11), torch.randn(2, 4, 11, 11)
                    expected = torch.cat((msi, sar, torch.zeros(2, 50, 11, 11)), dim=1) / 2
                    with torch.no_grad():
                        torch.testing.assert_close(core(msi, sar), core.cls(expected), rtol=0, atol=0)

    def test_three_variants_match_original_with_requested_bypasses(self):
        # Import here so a missing implementation produces an explicit test failure.
        self.assertIsNotNone(importlib.util.find_spec('baselines.m2heat.ablation'))
        from baselines.m2heat.ablation import M2HeatAblation, VARIANTS
        self.assertEqual(set(VARIANTS), {'wo_specific_hco', 'wo_shared_hco', 'wo_cff'})
        torch.set_num_threads(1)
        for ms, sar, classes in ((10, 4, 8), (10, 4, 9), (6, 1, 5)):
            for patch in (11, 13):
                kwargs = dict(num_classes=classes, spectral_channels=ms,
                              sar_channels=sar, patch_size=patch)
                for variant in VARIANTS:
                    with self.subTest(channels=(ms, sar), classes=classes, patch=patch, variant=variant):
                        torch.manual_seed(42)
                        original = M2Heat(**kwargs)
                        torch.manual_seed(42)
                        model = M2HeatAblation(variant=variant, **kwargs)
                        # Retained modules must have exactly the original initialization.
                        old = original.state_dict()
                        for name, value in model.state_dict().items():
                            torch.testing.assert_close(value, old[name], rtol=0, atol=0)
                        core = original.net.model
                        handles = []
                        if variant == 'wo_specific_hco':
                            handles.extend(layer.register_forward_hook(lambda m, args, out: args[0])
                                           for layer in (core.hclayer_hsi, core.hclayer_aux))
                        elif variant == 'wo_shared_hco':
                            handles.append(core.hclayer_fuse.register_forward_hook(lambda m, args, out: args[0]))
                        else:
                            handles.append(core.cff.register_forward_hook(lambda m, args, out: (args[0] + args[1]) / 2))
                        x = torch.randn(2, 1, ms + sar, patch, patch)
                        original.eval(); model.eval()
                        with torch.no_grad():
                            torch.testing.assert_close(model(x), original(x), rtol=0, atol=0)
                        for handle in handles:
                            handle.remove()
                        model.train()
                        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                        output = model(x)
                        self.assertEqual(tuple(output.shape), (2, classes))
                        self.assertTrue(torch.isfinite(output).all())
                        torch.nn.functional.cross_entropy(output, torch.tensor([0, classes - 1])).backward()
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                            for p in model.parameters() if p.requires_grad))
                        optimizer.step()
                        memory = io.BytesIO()
                        torch.save(model.state_dict(), memory); memory.seek(0)
                        restored = M2HeatAblation(variant=variant, **kwargs)
                        restored.load_state_dict(torch.load(memory, weights_only=True))
                        model.eval(); restored.eval()
                        with torch.no_grad():
                            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_full_remains_original_only_and_invalid_variants_fail(self):
        self.assertIsNotNone(importlib.util.find_spec('baselines.m2heat.ablation'))
        from baselines.m2heat.ablation import M2HeatAblation
        for variant in ('full', 'typo', ''):
            with self.assertRaises(ValueError):
                M2HeatAblation(variant=variant)

    def test_configs_preserve_original_protocol_and_queue_has_no_full(self):
        import yaml
        from scripts.run_m2heat_ablations import make_tasks
        from baselines.m2heat.ablation_experiment import validate_config
        tasks = make_tasks()
        self.assertEqual(len(tasks), 9)
        self.assertEqual(len({t['key'] for t in tasks}), 9)
        for task in tasks:
            config = yaml.safe_load(Path(task['config']).read_text())
            validate_config(config)
            suffix = task['key'].removeprefix(task['variant'] + '_')
            original = yaml.safe_load(Path(f'configs/m2heat_{suffix}.yaml').read_text())
            expected = dict(original, ablation=task['variant'], output_root='runs/m2heat_ablation')
            self.assertEqual(config, expected)

    def test_queue_reserves_memory_for_foreign_and_not_yet_allocated_active_tasks(self):
        from scripts.run_m2heat_ablations import can_start
        self.assertFalse(can_start(10000, 1, 24576, 6682, 17574))
        self.assertTrue(can_start(10000, 0, 24576, 6682, 17574))
        self.assertFalse(can_start(10000, 0, 24576, 15000, 9576))

    def test_synthetic_training_all_seeds_metrics_splits_and_no_overwrite(self):
        import numpy as np
        import yaml
        from baselines.io import save_yaml, load_yaml
        from baselines.m2heat.ablation_experiment import train, completed_metrics
        from baselines.split import split_from_config
        torch.set_num_threads(1)
        config = load_yaml('configs/m2heat_ablation/wo_cff_grss07_custom.yaml')
        config.update(epochs=1, batch_size=2, test_batch_size=8, device='cpu',
                      split=dict(method='fixed_train_counts', train_count_per_class=1),
                      eval=dict(enabled=True, batch_size=8, save_labeled_maps=True))
        image = np.random.default_rng(0).random((5, 5, 7), dtype=np.float32)
        gt = np.tile(np.arange(5), (5, 1))
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'config.yaml'
            save_yaml(config, config_path)
            with mock_patch('baselines.m2heat.ablation_experiment.load_dataset', return_value=(image, gt, list('abcde'))):
                directory = train(config_path, Path(tmp) / 'batch')
                results = completed_metrics(directory, load_yaml(directory / 'config.yaml'))
                self.assertEqual([r['seed'] for r in results], list(range(202201, 202206)))
                info = json.loads((directory / 'labeled_map_info.json').read_text())
                self.assertEqual(info['seed'], max(results, key=lambda r: r['oa'])['seed'])
                self.assertEqual(info['labeled_count'], 25)
                self.assertTrue((directory / 'labeled_prediction.tif').is_file())
                from scripts.summarize_m2heat_ablations import summarize
                rows = summarize(Path(tmp) / 'batch')
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['n'], 5)
                self.assertAlmostEqual(rows[0]['oa_std'], np.std([r['oa'] for r in results]))
                for seed in range(202201, 202206):
                    expected_train, expected_test = split_from_config(gt, config, seed)
                    with np.load(directory / f'seed_{seed}' / 'split.npz') as split:
                        np.testing.assert_array_equal(split['train_gt'], expected_train)
                        np.testing.assert_array_equal(split['test_gt'], expected_test)
                target = directory / 'seed_202201' / 'checkpoints' / 'model_best.pth'
                original_bytes = target.read_bytes()
                identity = {name: (directory / name).read_bytes()
                            for name in ('config.yaml', 'dataset_config.yaml', 'training_protocol.json')}
                train(config_path, Path(tmp) / 'batch')
                self.assertEqual(original_bytes, target.read_bytes())
                for name, contents in identity.items():
                    self.assertEqual(contents, (directory / name).read_bytes())
                config['ablation'] = 'full'
                save_yaml(config, config_path)
                with self.assertRaises(ValueError):
                    train(config_path, Path(tmp) / 'batch')


if __name__ == '__main__':
    unittest.main()
