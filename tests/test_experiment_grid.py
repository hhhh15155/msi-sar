from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.experiment_grid import DATASETS, MODELS, SHOTS, is_complete, iter_experiments


ROOT = Path(__file__).resolve().parents[1]


class ExperimentGridTests(unittest.TestCase):
    def test_grid_contains_only_the_two_selected_datasets(self) -> None:
        experiments = list(iter_experiments())

        self.assertEqual(MODELS, ("dfinet", "frekfuse", "mghofnet", "msfmamba", "softformer"))
        self.assertEqual(DATASETS, ("yrd", "yrd2509new"))
        self.assertEqual(SHOTS, (5, 10, 20, 50, 100, 150, 200))
        self.assertEqual(len(experiments), 70)
        self.assertNotIn("yrd2509", {experiment.dataset for experiment in experiments})
        self.assertNotIn("yrd2509_landuse_9c", {experiment.dataset for experiment in experiments})

    def test_completion_accepts_any_numbered_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics = root / "runs_fewshot/fs20/msfmamba/yrd/run_002/metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text("{}", encoding="utf-8")

            experiment = next(
                experiment
                for experiment in iter_experiments()
                if experiment.model == "msfmamba" and experiment.dataset == "yrd" and experiment.shot == 20
            )
            self.assertTrue(is_complete(root, experiment))

    def test_grid_can_select_grss07_without_changing_the_default_datasets(self) -> None:
        experiments = list(iter_experiments(("grss07",)))

        self.assertEqual(len(experiments), 35)
        self.assertEqual({experiment.dataset for experiment in experiments}, {"grss07"})
        self.assertEqual(DATASETS, ("yrd", "yrd2509new"))

    def test_all_yrd2509new_configs_match_dataset_channels_and_classes(self) -> None:
        dataset_config = yaml.safe_load((ROOT / "configs/datasets/yrd2509new.yaml").read_text(encoding="utf-8"))
        self.assertEqual(dataset_config["path"], "data/yrd2509new")
        self.assertEqual(dataset_config["num_channels"], 14)
        self.assertEqual(dataset_config["num_classes"], 9)
        self.assertEqual(len(dataset_config["class_names"]), 9)

        channel_keys = {
            "dfinet": ("spectral_channels", "sar_channels"),
            "frekfuse": ("ms_channels", "sar_channels"),
            "mghofnet": ("hsi_channels", "aux_channels"),
            "msfmamba": ("ms_channels", "sar_channels"),
            "softformer": ("hsi_channels", "aux_channels"),
        }
        for experiment in iter_experiments():
            if experiment.dataset != "yrd2509new":
                continue
            config_path = ROOT / experiment.config
            self.assertTrue(config_path.exists(), config_path)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            primary_key, auxiliary_key = channel_keys[experiment.model]
            self.assertEqual(config["dataset"], "yrd2509new")
            self.assertEqual(config["dataset_config"], "configs/datasets/yrd2509new.yaml")
            self.assertEqual(config[primary_key], 10)
            self.assertEqual(config[auxiliary_key], 4)
            self.assertEqual(config["split"]["train_count_per_class"], experiment.shot)
            self.assertEqual(config["split"]["val_count_per_class"], experiment.shot)


if __name__ == "__main__":
    unittest.main()
