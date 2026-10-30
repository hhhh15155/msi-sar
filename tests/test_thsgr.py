"""THSGR adapter contracts; synthetic tensors only, no project datasets."""
from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class THSGRIntegrationTests(unittest.TestCase):
    def test_project_entrypoints_exist(self):
        for name in ("model", "train", "eval", "infer", "dataset", "io", "metrics"):
            self.assertTrue((ROOT / "baselines" / "thsgr" / f"{name}.py").is_file(), name)
        for name in ("train_eval", "eval", "infer"):
            self.assertTrue((ROOT / "scripts" / f"{name}_thsgr.py").is_file(), name)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class THSGRModelTests(unittest.TestCase):
    def test_training_test_reevaluation_and_labeled_maps(self):
        import numpy as np
        import torch
        from scipy.io import savemat
        import imageio.v2 as imageio
        from baselines.thsgr.train import run_training
        from baselines.thsgr.eval import evaluate_run, save_labeled_maps
        from baselines.thsgr.io import save_yaml
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image = np.random.default_rng(1).normal(size=(13, 15, 7)).astype('float32')
            gt = np.tile(np.array([1, 2, 3] * 5), (13, 1))
            gt[-1] = 0
            savemat(directory / 'data.mat', {'data': image})
            savemat(directory / 'label.mat', {'label': gt})
            dataset = dict(path=str(directory), data_file='data.mat', label_file='label.mat',
                           data_key='data', label_key='label', num_channels=7, num_classes=3,
                           class_names=['a', 'b', 'c'], normalization='per_channel_minmax')
            save_yaml(dataset, directory / 'dataset.yaml')
            config = dict(model='thsgr', dataset='synthetic', dataset_config=str(directory / 'dataset.yaml'),
                          output_root=str(directory / 'runs'), device='cpu', patch_size=11,
                          spectral_channels=6, sar_channels=1, epochs=1, batch_size=4,
                          test_batch_size=17, seeds=[202201], num_runs=1,
                          split=dict(method='fixed_train_counts', train_counts=[2, 2, 2]),
                          eval=dict(batch_size=19, save_labeled_maps=True))
            save_yaml(config, directory / 'config.yaml')
            run = run_training(directory / 'config.yaml')
            metrics_before = (run / 'metrics.json').read_bytes()
            initial = json.loads(metrics_before)['runs'][0]
            evaluation = evaluate_run(run)
            self.assertEqual(evaluation['runs'][0]['confusion_matrix'], initial['confusion_matrix'])
            eval_before = (run / 'eval_metrics.json').read_bytes()
            info = save_labeled_maps(run)
            self.assertEqual(info['seed'], 202201)
            self.assertEqual(info['labeled_count'], 180)
            self.assertEqual((run / 'metrics.json').read_bytes(), metrics_before)
            self.assertEqual((run / 'eval_metrics.json').read_bytes(), eval_before)
            predicted = imageio.imread(run / 'labeled_prediction.tif')
            self.assertTrue((predicted[-1] == 0).all())
            self.assertTrue((predicted[:-1] > 0).all())
            self.assertTrue((run / 'dataset_config.yaml').is_file())
            self.assertTrue((run / 'run_identity.json').is_file())
            self.assertFalse((run / 'full_prediction.tif').exists())

    def test_eval_predictions_do_not_depend_on_batch_composition(self):
        import torch
        from baselines.thsgr.model import THSGR
        torch.set_num_threads(1)
        torch.manual_seed(12)
        model = THSGR(num_classes=5, spectral_channels=6, sar_channels=1).eval()
        x = torch.randn(3, 1, 7, 11, 11)
        with torch.no_grad():
            together = model(x)
            alone = torch.cat([model(sample.unsqueeze(0)) for sample in x])
            reversed_batch = model(x.flip(0)).flip(0)
        torch.testing.assert_close(together, alone, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(together, reversed_batch, atol=1e-5, rtol=1e-4)

    def test_mean_forward_is_training_only(self):
        import torch
        from baselines._hsi_adapters.thsgr_patch import MeanForward
        layer = MeanForward()
        x = torch.tensor([[[2., 4.]], [[6., 8.]]])
        torch.testing.assert_close(layer.train()(x), torch.tensor([[[3., 5.]], [[5., 7.]]]))
        torch.testing.assert_close(layer.eval()(x), x)

    def test_channels_patch_backward_and_checkpoint(self):
        import torch
        from baselines.thsgr.model import THSGR
        torch.set_num_threads(1)
        for ms, sar, classes in ((6, 1, 5), (10, 4, 8), (10, 4, 9)):
            for patch in (11, 13):
                with self.subTest(ms=ms, sar=sar, classes=classes, patch=patch):
                    kwargs = dict(num_classes=classes, patch_size=patch,
                                  spectral_channels=ms, sar_channels=sar)
                    torch.manual_seed(41)
                    model = THSGR(**kwargs)
                    optimizer = torch.optim.Adam(model.parameters())
                    registered = {id(p) for group in optimizer.param_groups for p in group["params"]}
                    x = torch.randn(2, 1, ms + sar, patch, patch)
                    logits = model(x)
                    self.assertEqual(tuple(logits.shape), (2, classes))
                    self.assertTrue(torch.isfinite(logits).all())
                    torch.nn.functional.cross_entropy(logits, torch.tensor([0, classes - 1])).backward()
                    self.assertEqual(registered, {id(p) for p in model.parameters() if p.requires_grad})
                    for name, p in model.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    optimizer.step()
                    buffer = io.BytesIO()
                    torch.save(model.state_dict(), buffer)
                    buffer.seek(0)
                    restored = THSGR(**kwargs)
                    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
                    model.eval()
                    restored.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(model(x), restored(x))

    def test_invalid_shapes_fail_at_interface(self):
        import torch
        from baselines.thsgr.model import THSGR
        for patch in (6, 10):
            with self.assertRaises(ValueError):
                THSGR(patch_size=patch)
        model = THSGR(num_classes=5, spectral_channels=6, sar_channels=1)
        for shape in ((2, 7, 11, 11), (2, 2, 7, 11, 11), (2, 1, 8, 11, 11),
                      (2, 1, 7, 13, 13), (2, 1, 7, 11, 13)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))


if __name__ == "__main__":
    unittest.main()
