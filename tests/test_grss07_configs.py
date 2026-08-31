from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.generate_grss07_configs import generate_configs


ROOT = Path(__file__).resolve().parents[1]
POLICY_SPEC = importlib.util.spec_from_file_location(
    "msfmamba_channel_policy",
    ROOT / "baselines" / "msfmamba" / "channel_policy.py",
)
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
assert POLICY_SPEC.loader is not None
POLICY_SPEC.loader.exec_module(POLICY_MODULE)
required_spectral_channels = POLICY_MODULE.required_spectral_channels


class Grss07ConfigTests(unittest.TestCase):
    def test_msfmamba_zero_padding_policy_reaches_the_nine_band_kernel(self) -> None:
        self.assertEqual(required_spectral_channels(6), 9)
        self.assertEqual(required_spectral_channels(9), 9)
        self.assertEqual(required_spectral_channels(10), 10)

    def test_generator_writes_all_five_models_and_seven_shot_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)

            generated = generate_configs(output)

            self.assertEqual(len(generated), 35)
            self.assertEqual(
                {path.name for path in generated},
                {
                    f"{model}_grss07_fs{shot}.yaml"
                    for model in ("dfinet", "frekfuse", "mghofnet", "msfmamba", "softformer")
                    for shot in (5, 10, 20, 50, 100, 150, 200)
                },
            )

    def test_generated_configs_use_six_optical_and_one_sar_channel(self) -> None:
        channel_keys = {
            "dfinet": ("spectral_channels", "sar_channels"),
            "frekfuse": ("ms_channels", "sar_channels"),
            "mghofnet": ("hsi_channels", "aux_channels"),
            "msfmamba": ("ms_channels", "sar_channels"),
            "softformer": ("hsi_channels", "aux_channels"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            generated = generate_configs(Path(temp_dir))

            for path in generated:
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                primary_key, sar_key = channel_keys[config["model"]]
                shot = int(path.stem.rsplit("fs", 1)[1])
                self.assertEqual(config["dataset"], "grss07")
                self.assertEqual(config["dataset_config"], "configs/datasets/grss07.yaml")
                self.assertEqual(config[primary_key], 6)
                self.assertEqual(config[sar_key], 1)
                self.assertEqual(config["split"]["train_count_per_class"], shot)
                self.assertEqual(config["split"]["method"], "fixed_train_counts")
                self.assertNotIn("val_count_per_class", config["split"])
                self.assertFalse(config["use_validation"])
                self.assertEqual(config["select_best_by"], "test")
                self.assertEqual(config["test_interval"], 20)
                self.assertEqual(config["batch_size"], 128)
                self.assertEqual(config["test_batch_size"], 1024)
                self.assertEqual(config["eval"]["batch_size"], 1024)
                self.assertEqual(config["output_root"], f"runs_fewshot/fs{shot}")


if __name__ == "__main__":
    unittest.main()
