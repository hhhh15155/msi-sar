from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from baselines.vbenet.train import build_model, next_run_dir, validate_config
from models import VBENet
from scripts.generate_vbenet_configs import generate_configs


ROOT = Path(__file__).resolve().parents[1]


class VBENetExperimentTests(unittest.TestCase):
    def test_build_model_uses_configured_architecture_and_trains_one_step(self) -> None:
        config = {
            "patch_size": 11,
            "ms_channels": 10,
            "sar_channels": 4,
            "width": 16,
            "encoder_depth": 1,
            "groups": 4,
            "expansion": 2,
            "lambda_proto": 1.0,
            "tau_r": 0.3,
            "tau_c": 0.1,
            "inner_iters": 1,
            "outer_updates": 1,
            "modality_dropout": 0.0,
        }
        model = build_model(config, num_classes=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        logits = model(torch.randn(2, 10, 11, 11), torch.randn(2, 4, 11, 11))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 6]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertIsInstance(model, VBENet)
        self.assertEqual(logits.shape, (2, 8))
        self.assertTrue(torch.isfinite(loss))

    def test_next_run_dir_matches_existing_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "runs_fewshot" / "fs20"
            first = next_run_dir(output_root, "yrd")
            first.mkdir(parents=True)
            second = next_run_dir(output_root, "yrd")

            self.assertEqual(first, output_root / "vbenet" / "yrd" / "run_001")
            self.assertEqual(second, output_root / "vbenet" / "yrd" / "run_002")

    def test_validate_config_rejects_misaligned_patch_or_channel_grouping(self) -> None:
        config = {
            "patch_size": 9,
            "ms_channels": 10,
            "sar_channels": 4,
            "width": 64,
            "groups": 8,
            "batch_size": 128,
            "test_batch_size": 1024,
            "epochs": 200,
            "num_runs": 5,
            "seeds": [202201, 202202, 202203, 202204, 202205],
            "split": {"method": "fixed_train_counts"},
        }
        with self.assertRaisesRegex(ValueError, "patch_size"):
            validate_config(config)

        config["patch_size"] = 11
        config["groups"] = 7
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_config(config)

    def test_generator_clones_current_experiment_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = generate_configs(Path(directory), ROOT / "configs")
            target = Path(directory) / "vbenet_yrd_fs20.yaml"
            actual = yaml.safe_load(target.read_text(encoding="utf-8"))
            reference = yaml.safe_load(
                (ROOT / "configs/mghofnet_yrd_fs20.yaml").read_text(encoding="utf-8")
            )

            self.assertEqual(len(generated), 22)
            self.assertEqual(actual["model"], "vbenet")
            for key in (
                "dataset", "dataset_config", "output_root", "device", "patch_size",
                "pad_mode", "split", "batch_size", "test_batch_size", "epochs",
                "num_runs", "seeds", "data_aug", "learning_rate", "weight_decay",
                "eta_min", "eval", "infer", "palette",
            ):
                self.assertEqual(actual[key], reference[key], key)
            self.assertEqual(actual["ms_channels"], reference["hsi_channels"])
            self.assertEqual(actual["sar_channels"], reference["aux_channels"])
            self.assertEqual(actual["width"], 64)
            self.assertEqual(actual["encoder_depth"], 5)
            self.assertTrue((Path(directory) / "vbenet_grss07_custom.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
