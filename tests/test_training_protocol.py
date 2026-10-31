"""Shared MGHOFNet protocol: no real data, servers or experiment scheduling."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('dfinet', 'frekfuse', 'mghofnet', 'msfmamba', 'softformer',
          'dffnet', 'm2heat', 'mft', 'dcmnet', 'hslinets', 'thsgr', 'gfsfn', 's2atnet')


def test_all_checked_in_configs_use_mghofnet_protocol():
    from baselines.protocol import TRAINING_DEFAULTS
    found = set()
    for path in (ROOT / 'configs').glob('*.yaml'):
        config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        if config.get('model') not in MODELS:
            continue
        found.add(config['model'])
        for key, value in TRAINING_DEFAULTS.items():
            assert config[key] == value, (path.name, key)
        assert config['infer']['save_full_map'] is False, path.name
    assert found == set(MODELS)


def test_every_training_entrypoint_uses_common_runtime():
    for model in MODELS:
        tree = ast.parse((ROOT / 'baselines' / model / 'train.py').read_text(encoding='utf-8-sig'))
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        assert 'build_optimizer_scheduler' in calls, model
        assert 'train_epoch' in calls, model
        assert 'prepare_training_config' in calls, model
        assert 'scheduler.step' in calls, model
        assert not any(c.startswith('torch.optim.') or c.endswith('clip_grad_norm_') for c in calls), model


def test_legacy_padding_is_selected_from_run_not_current_defaults():
    from baselines.protocol import inference_pad_mode, TRAINING_DEFAULTS
    for model in MODELS:
        legacy = dict(model=model)
        assert inference_pad_mode(legacy) == ('constant' if model in ('mghofnet', 'softformer') else 'reflect')
        assert inference_pad_mode(dict(legacy, **TRAINING_DEFAULTS)) == 'constant'
    assert inference_pad_mode(dict(model='mghofnet', pad_mode='reflect')) == 'reflect'
    with pytest.raises(ValueError):
        inference_pad_mode(dict(model='dfinet', training_protocol='unknown'))


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch unavailable')
def test_optimizer_scheduler_seed_and_real_step():
    import torch
    from baselines.training import set_seed, build_optimizer_scheduler, train_epoch
    from baselines.protocol import prepare_training_config
    torch.set_num_threads(1)
    config = prepare_training_config(dict(epochs=2, model='dfinet'))
    set_seed(202201)
    first = torch.rand(4)
    set_seed(202201)
    assert torch.equal(first, torch.rand(4))
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(14 * 3 * 3, 3))
    optimizer, scheduler = build_optimizer_scheduler(model, config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]['weight_decay'] == .01
    before = model[1].weight.detach().clone()
    batch = (torch.randn(2, 1, 14, 3, 3), torch.tensor([0, 2]))
    loss = train_epoch(model, [batch], optimizer, torch.device('cpu'), config)
    assert np.isfinite(loss)
    assert not torch.equal(before, model[1].weight)
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(.0005005)
    train_epoch(model, [batch], optimizer, torch.device('cpu'), config)
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(.000001)


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch unavailable')
def test_packed_and_dual_patches_match_mghofnet_reference_at_edges_and_after_augmentation():
    import torch
    from baselines.data import PatchDataset, FusionPatchDataset
    for ms, sar in ((6, 1), (10, 4)):
        image = np.arange(5 * 7 * (ms + sar), dtype=np.float32).reshape(5, 7, ms + sar)
        gt = np.zeros((5, 7), dtype=np.int64)
        np.random.seed(17)
        packed = PatchDataset(image, gt, 3, data_aug=True)
        np.random.seed(17)
        dual = FusionPatchDataset(image, gt, 3, ms, sar, data_aug=True)
        assert np.array_equal(packed.indices, dual.indices)
        for index in range(len(packed)):
            np.random.seed(index)
            x, label = packed[index]
            np.random.seed(index)
            hsi, aux, other = dual[index]
            torch.testing.assert_close(x[0], torch.cat([hsi, aux]))
            assert label.item() == other.item()
        packed.data_aug = False
        index = next(i for i, xy in enumerate(packed.indices) if tuple(xy) == (1, 1))
        assert (packed[index][0][0, :, 0, :] == 0).all()
        assert (packed[index][0][0, :, :, 0] == 0).all()


def test_config_generators_do_not_restore_old_settings(tmp_path):
    from scripts.generate_grss07_configs import generate_configs
    from scripts.generate_msfmamba_configs import _yaml_config
    from baselines.protocol import TRAINING_DEFAULTS
    configs = [yaml.safe_load(p.read_text(encoding='utf-8')) for p in generate_configs(tmp_path)]
    configs.append(yaml.safe_load(_yaml_config('yrd', 4, 50, [[0, 0, 0]])))
    for config in configs:
        for key, value in TRAINING_DEFAULTS.items():
            assert config[key] == value, (config['model'], key)


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch unavailable')
def test_test_labeled_and_full_predictions_agree_at_zero_padded_edges():
    import torch
    from baselines.prediction import predict_labeled_samples, infer_full_image
    from baselines.protocol import prepare_training_config

    class EdgeClassifier(torch.nn.Module):
        def forward(self, x):
            value = x[:, 0, 0, 0, 0]
            return torch.stack([value, 1 - value], dim=1)

    image = np.ones((3, 4, 7), dtype=np.float32)
    gt = np.zeros((3, 4), dtype=np.int64)
    gt[1, 2] = -1
    model = EdgeClassifier()
    config = prepare_training_config(dict(model='dfinet', patch_size=3, test_batch_size=2))
    pred = predict_labeled_samples(model, image, gt, config, torch.device('cpu'))
    full = infer_full_image(model, image, 3, 2, torch.device('cpu'), batch_size=3, config=config).argmax(-1)
    assert pred[0, 0] == full[0, 0] == 1
    assert np.array_equal(pred[gt >= 0], full[gt >= 0])
    assert pred[1, 2] == -1
    # The unversioned historical model used reflect padding.
    legacy = dict(model='dfinet', patch_size=3, test_batch_size=2)
    old = predict_labeled_samples(model, image, gt, legacy, torch.device('cpu'))
    assert old[0, 0] == 0


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch unavailable')
def test_mghofnet_actual_model_dual_training_and_packed_prediction(tmp_path):
    import torch
    from baselines.mghofnet.model import MGHOFNet
    from baselines.training import set_seed, build_optimizer_scheduler, train_epoch, forward_batch
    from baselines.protocol import prepare_training_config
    torch.set_num_threads(1)
    for ms, sar in ((6, 1), (10, 4)):
        set_seed(202201)
        config = prepare_training_config(dict(model='mghofnet', hsi_channels=ms, aux_channels=sar, epochs=2))
        model = MGHOFNet(in_ch_hsi=ms, in_ch_lidar=sar, num_classes=3)
        optimizer, scheduler = build_optimizer_scheduler(model, config)
        assert {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in model.parameters()}
        hsi, aux, labels = torch.randn(2, ms, 11, 11), torch.randn(2, sar, 11, 11), torch.tensor([0, 2])
        assert np.isfinite(train_epoch(model, [(hsi, aux, labels)], optimizer, torch.device('cpu'), config))
        scheduler.step()
        model.eval()
        packed = torch.cat([hsi, aux], dim=1).unsqueeze(1)
        with torch.no_grad():
            dual, _ = forward_batch(model, (hsi, aux, labels), torch.device('cpu'), config)
            joined, _ = forward_batch(model, (packed, labels), torch.device('cpu'), config)
        torch.testing.assert_close(dual, joined)
        torch.save(model.state_dict(), tmp_path / 'weights.pth')
        restored = MGHOFNet(in_ch_hsi=ms, in_ch_lidar=sar, num_classes=3).eval()
        restored.load_state_dict(torch.load(tmp_path / 'weights.pth', weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(restored(hsi, aux), dual)


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch unavailable')
def test_mat_normalization_and_saved_protocol_identity(tmp_path):
    from scipy.io import savemat
    from baselines.data import load_dataset
    from baselines.io import save_yaml
    from baselines.training import record_protocol
    from baselines.protocol import prepare_training_config, effective_dataset_config
    image = np.arange(3 * 4 * 7, dtype=np.float32).reshape(3, 4, 7)
    image[..., 0] = 5  # MGHOFNet leaves constant channels unchanged.
    labels = np.array([[0, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 2]])
    savemat(tmp_path / 'data.mat', {'data': image})
    savemat(tmp_path / 'label.mat', {'label': labels})
    dataset = dict(path=str(tmp_path), num_channels=7, num_classes=2,
                   class_names=['a', 'b'], normalization='per_channel_minmax')
    loaded, gt, names = load_dataset(dataset)
    assert (loaded[..., 0] == 5).all()
    for channel in range(1, 7):
        expected = (image[..., channel] - image[..., channel].min()) / np.ptp(image[..., channel])
        np.testing.assert_array_equal(loaded[..., channel], expected)
    np.testing.assert_array_equal(gt, labels - 1)
    config = prepare_training_config(dict(model='dfinet', epochs=2, seeds=[202201],
                                           split=dict(method='fixed_train_counts', train_counts=[1, 1])))
    save_yaml(config, tmp_path / 'config.yaml')
    record_protocol(config, dataset, tmp_path)
    saved = json.loads((tmp_path / 'training_protocol.json').read_text())
    assert saved['training_protocol'] == 'mghofnet-v1'
    assert saved['config_sha256'] == hashlib.sha256((tmp_path / 'config.yaml').read_bytes()).hexdigest()
    assert saved['dataset_config_sha256'] == hashlib.sha256((tmp_path / 'dataset_config.yaml').read_bytes()).hexdigest()
    assert saved['seeds'] == [202201]
    assert effective_dataset_config({}, dict(model='dfinet'))['normalization'] == 'global_minmax_center'
    assert effective_dataset_config({}, dict(model='mghofnet'))['normalization'] == 'per_channel_minmax'


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch unavailable')
def test_frekfuse_dictionary_output_uses_same_cross_entropy_step():
    import torch
    from models.frekfuse import FreKFuseLite
    from baselines.training import build_optimizer_scheduler, train_epoch, forward_batch
    from baselines.protocol import prepare_training_config
    torch.set_num_threads(1)
    config = prepare_training_config(dict(model='frekfuse', ms_channels=6, sar_channels=1, epochs=2))
    model = FreKFuseLite(ms_channels=6, sar_channels=1, num_classes=3, patch_size=11)
    optimizer, _ = build_optimizer_scheduler(model, config)
    x, targets = torch.randn(2, 1, 7, 11, 11), torch.tensor([0, 2])
    assert np.isfinite(train_epoch(model, [(x, targets)], optimizer, torch.device('cpu'), config))
    model.eval()
    with torch.no_grad():
        logits, _ = forward_batch(model, (x, targets), torch.device('cpu'), config)
        original = model(x[:, 0, :6], x[:, 0, 6:], labels=targets)
    torch.testing.assert_close(logits, original['logits'])
    torch.testing.assert_close(torch.nn.functional.cross_entropy(logits, targets), original['losses']['total'])
