"""Contracts for the two explicitly approved paper adaptations."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from models.main_classification import build_main_model
from scripts import main_experiment as engine


class PaperCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(202201)

    def model(self, variant, ms=6, sar=1):
        return build_main_model({'variant': variant, 'ms_channels': ms, 'sar_channels': sar,
                                 'embed_dim': 16, 'num_heads': 4, 'depth': 1, 'dropout': 0.1}, 5)

    def test_both_models_support_channels_patches_and_registered_finite_gradients(self):
        for variant in ('v7_fdmf_fixed', 'v8_msaf_fixed'):
            for ms, sar, patch in ((6, 1, 5), (10, 4, 11), (6, 1, 15)):
                with self.subTest(variant=variant, channels=(ms, sar), patch=patch):
                    model = self.model(variant, ms, sar)
                    registered = list(dict(model.named_parameters()))
                    output = model(torch.randn(2, ms, patch, patch), torch.randn(2, sar, patch, patch),
                                   labels=torch.tensor([1, 3]))
                    self.assertEqual(output['logits'].shape, (2, 5))
                    output['losses']['total'].backward()
                    self.assertEqual(registered, list(dict(model.named_parameters())))
                    missing = [n for n, p in model.named_parameters() if p.grad is None]
                    self.assertEqual(missing, [])
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_eval_is_exactly_repeatable_without_rng_consumption_and_uses_both_inputs(self):
        ms, sar = torch.randn(2, 6, 11, 11), torch.randn(2, 1, 11, 11)
        for variant in ('v7_fdmf_fixed', 'v8_msaf_fixed'):
            model = self.model(variant).eval()
            state = torch.get_rng_state().clone()
            with torch.no_grad():
                first = model(ms, sar)['logits']
                second = model(ms, sar)['logits']
                no_sar = model(ms, torch.zeros_like(sar))['logits']
                no_ms = model(torch.zeros_like(ms), sar)['logits']
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertGreater(float((first - no_sar).abs().max()), 1e-8)
            self.assertGreater(float((first - no_ms).abs().max()), 1e-8)

    def test_fdmf_has_same_initial_common_encoder_and_head_as_concat(self):
        torch.manual_seed(9)
        reference = self.model('v2_concat')
        torch.manual_seed(9)
        candidate = self.model('v7_fdmf_fixed')
        for name, tensor in reference.state_dict().items():
            if not name.startswith('fusion.'):
                torch.testing.assert_close(tensor, candidate.state_dict()[name], rtol=0, atol=0)

    def test_fixed_frequency_split_matches_explicit_author_geometry(self):
        from models.fdmf_patch import FixedFrequencySplit
        layer = FixedFrequencySplit(4)
        x = torch.randn(2, 4, 11, 9, requires_grad=True)
        high, low = layer(x)
        projected = layer.conv(x)
        spectrum = torch.fft.fftshift(torch.fft.fft2(projected, norm='forward'), dim=(-2, -1))
        mask = torch.zeros(11, 9)
        mask[3:7, 2:6] = 1
        def inverse(s):
            return torch.fft.ifft2(torch.fft.ifftshift(s, dim=(-2, -1)), norm='forward').abs()
        torch.testing.assert_close(low, inverse(spectrum * mask))
        torch.testing.assert_close(high, inverse(spectrum * (1-mask)))
        (high.mean()+low.mean()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(set(dict(layer.named_parameters())), {'conv.weight'})

    def test_fdmf_does_not_train_redundant_bias_immediately_before_batchnorm(self):
        self.assertIsNone(self.model('v7_fdmf_fixed').fusion.output_conv.bias)

    def test_msaf_keeps_two_active_stages_and_training_only_mask(self):
        from models.msaf_patch import MaskedSelfAttention
        model = self.model('v8_msaf_fixed')
        self.assertEqual(model.fusion1.embedding, 16)
        self.assertEqual(model.fusion2.embedding, 24)
        attention = MaskedSelfAttention(16, heads=4, head_dim=16, dropout=0, mask_probability=1)
        x = torch.randn(2, 5, 16, requires_grad=True)
        y = attention.train()(x)
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        attention.eval()
        torch.testing.assert_close(attention(x), attention(x), rtol=0, atol=0)
        self.assertEqual(len(model.ms_features), 3)
        self.assertFalse(any('classifier' in n for n, _ in model.named_parameters()))

    def test_msaf_removes_only_stage1_final_norm_bias_cancelled_by_next_batchnorm(self):
        model = self.model('v8_msaf_fixed')
        self.assertIsNone(model.fusion1.norm.bias)
        self.assertIsNotNone(model.fusion2.norm.bias)

    def test_manifest_selects_only_requested_variants_and_rejects_unknown_or_duplicate(self):
        manifest = {'variants': ['v7_fdmf_fixed', 'v8_msaf_fixed'], 'datasets': {'yrd': {}},
                    'experiments': [{'id': v+'_yrd', 'variant': v, 'dataset': 'yrd'}
                                    for v in ('v7_fdmf_fixed', 'v8_msaf_fixed')]}
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'manifest.json'
            p.write_text(json.dumps(manifest))
            self.assertEqual(len(engine.read_manifest(p)['experiments']), 2)
            for invalid in (['v7_fdmf_fixed', 'unknown'], ['v7_fdmf_fixed']*2):
                p.write_text(json.dumps({**manifest, 'variants': invalid}))
                with self.assertRaises(ValueError):
                    engine.read_manifest(p)

    def test_fingerprints_cover_both_new_model_sources(self):
        for name in ('models/fdmf_patch.py', 'models/msaf_patch.py'):
            self.assertIn(name, engine.source_fingerprint())

    def test_six_job_manifest_preserves_training_protocol_and_original_splits(self):
        import yaml
        from scripts.generate_paper_candidate_configs import generate
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            import shutil
            shutil.copytree(engine.ROOT/'configs/main_first_pass', root/'configs/main_first_pass')
            manifest = generate(root)
            self.assertEqual(len(manifest['experiments']), 6)
            self.assertEqual(len(engine.read_manifest(root/'configs/paper_candidates/manifest.json')['experiments']), 6)
            self.assertNotIn('promotion', manifest)
            for job in manifest['experiments']:
                candidate = yaml.safe_load((root/job['config']).read_text())
                old = yaml.safe_load((root/'configs/main_first_pass'/('v2_concat_'+job['dataset']+'.yaml')).read_text())
                for key in ('seed','patch_size','epochs','batch_size','test_batch_size','precision',
                            'learning_rate','weight_decay','pretrained','split','num_runs'):
                    self.assertEqual(candidate[key], old[key], (job['id'],key))
                self.assertEqual(len(manifest['datasets'][job['dataset']]['reuse_split']['sha256']), 64)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA deployment check')
    def test_fp32_gpu_training_and_real_test_batch_1024(self):
        for variant in ('v7_fdmf_fixed', 'v8_msaf_fixed'):
            for ms_c, sar_c in ((6, 1), (10, 4)):
                config = {'variant': variant, 'ms_channels': ms_c, 'sar_channels': sar_c,
                          'embed_dim': 128, 'num_heads': 4, 'depth': 2, 'dropout': 0.1}
                model = build_main_model(config, 8).cuda()
                ms = torch.randn(4, ms_c, 11, 11, device='cuda')
                sar = torch.randn(4, sar_c, 11, 11, device='cuda')
                model(ms, sar, labels=torch.tensor([0, 1, 2, 3], device='cuda'))['losses']['total'].backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                model.zero_grad(set_to_none=True)
                model.eval()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    out = model(torch.randn(1024, ms_c, 11, 11, device='cuda'),
                                torch.randn(1024, sar_c, 11, 11, device='cuda'))['logits']
                self.assertEqual(out.shape, (1024, 8))
                self.assertTrue(torch.isfinite(out).all())
                print('GPU CHECK', variant, ms_c, sar_c, 'parameters=',sum(p.numel() for p in model.parameters()),
                      'batch1024_peak_MiB=',round(torch.cuda.max_memory_allocated()/1024**2, 1), flush=True)
                del out, model, ms, sar
                torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA resume integration check')
    def test_candidate_resume_matches_uninterrupted_predictions_and_logits(self):
        import numpy as np
        import yaml
        from unittest.mock import patch
        (engine.ROOT/'tmp').mkdir(exist_ok=True)
        for variant in ('v7_fdmf_fixed', 'v8_msaf_fixed'):
            with tempfile.TemporaryDirectory(dir=engine.ROOT/'tmp') as temporary:
                root = Path(temporary)
                prepared = root/'prepared/synthetic'
                prepared.mkdir(parents=True)
                image = np.random.default_rng(2).normal(size=(7, 8, 7)).astype(np.float32)
                gt = (np.arange(56).reshape(7, 8)%2).astype(np.int64)
                train, test = np.arange(8), np.arange(8, 56)
                np.save(prepared/'image.npy', image)
                np.save(prepared/'gt.npy', gt)
                np.savez(prepared/'split.npz', train_indices=train, test_indices=test)
                engine.atomic_json(prepared/'metadata.json', {'class_names':['a','b'],
                    'train_counts':[4,4], 'test_counts':[24,24],
                    'prepared_sha256':{'split.npz':engine.digest(prepared/'split.npz')}})
                config = {'variant':variant, 'ms_channels':6, 'sar_channels':1, 'embed_dim':16,
                    'num_heads':4, 'depth':1, 'seed':202201, 'patch_size':5, 'epochs':2,
                    'batch_size':4, 'test_batch_size':16, 'learning_rate':0.0005,
                    'weight_decay':0.0001, 'precision':'fp32', 'dropout':0.1}
                path = root/'config.yaml'
                path.write_text(yaml.safe_dump(config))
                manifest = {'output_root':str(root), 'experiments':[
                    {'id':n,'dataset':'synthetic','config':str(path)} for n in ('full','resume')]}
                engine.train(manifest,'full')
                original = engine.save_checkpoint
                def interrupt(path, payload):
                    original(path,payload)
                    if path.name=='resume.pth' and payload['epoch']==1:
                        raise RuntimeError('candidate interruption')
                with patch.object(engine,'save_checkpoint',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'candidate interruption'):
                        engine.train(manifest,'resume')
                engine.train(manifest,'resume')
                np.testing.assert_array_equal(np.load(root/'full/test_predictions.npy'),
                                               np.load(root/'resume/test_predictions.npy'))
                left, right = build_main_model(config,2).eval(),build_main_model(config,2).eval()
                left.load_state_dict(torch.load(root/'full/model_last.pth',map_location='cpu',weights_only=True))
                right.load_state_dict(torch.load(root/'resume/model_last.pth',map_location='cpu',weights_only=True))
                for name, expected in left.state_dict().items():
                    actual = right.state_dict()[name]
                    if name.endswith('self_attn.in_proj_bias'):
                        # Same mathematically redundant K-bias exception as the
                        # original runner's FP32 resume integration check.
                        d = expected.numel()//3
                        torch.testing.assert_close(actual[:d],expected[:d],rtol=1e-5,atol=1e-6)
                        torch.testing.assert_close(actual[2*d:],expected[2*d:],rtol=1e-5,atol=1e-6)
                        self.assertLess(float((actual[d:2*d]-expected[d:2*d]).abs().max()),2e-4)
                    else:
                        # The unchanged runner permits nondeterministic FP32
                        # cuDNN algorithms. Measured active-parameter drift is
                        # ~2e-6; test numerical agreement, not bitwise recovery.
                        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5,
                                                   msg=lambda msg: name+': '+msg)
                inputs = engine.PatchBank(image,5,torch.device('cpu')).take(torch.tensor(test))
                with torch.inference_mode():
                    torch.testing.assert_close(left(inputs[:,:6],inputs[:,6:])['logits'],
                                               right(inputs[:,:6],inputs[:,6:])['logits'], rtol=1e-5,atol=1e-5)


if __name__ == '__main__':
    unittest.main()
