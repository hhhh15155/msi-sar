"""Exercise map/evaluation orchestration without importing optional model runtimes."""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("dffnet", "m2heat", "mft", "thsgr")


def load_functions(path: Path, names: tuple[str, ...], dependencies: dict) -> dict:
    """Execute real orchestration functions with only external I/O boundaries supplied.

    Model/data imports require optional torch/h5py/sklearn installations. Keeping
    those boundaries out lets the same mask/file regressions run on the host.
    """
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    namespace = {"Path": Path, "np": np, **dependencies}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class BaselineMapProtocolTests(unittest.TestCase):
    def make_context(self, model: str, run_dir: Path) -> tuple[dict, dict]:
        gt = np.array([[0, -1, 1], [1, 0, -1]], dtype=np.int64)
        image = np.zeros((2, 3, 14), dtype=np.float32)
        config = {"dataset_config": "dataset.yaml", "patch_size": 11, "test_batch_size": 1024,
                  "seeds": [202201, 202202], "eval": {"save_labeled_maps": True}, "infer": {"save_full_map": False}}
        records = {"runs": [{"run": 0, "seed": 202201, "oa": 83.0}, {"run": 1, "seed": 202202, "oa": 94.0}]}
        (run_dir / "metrics.json").write_text(json.dumps(records), encoding="utf-8")
        (run_dir / "eval_metrics.json").write_text('test-metrics-sentinel', encoding="utf-8")
        saved_maps = {}

        def save_outputs(array, prefix, palette):
            saved_maps[Path(prefix).name] = array.copy()
            return [Path(prefix).with_suffix(".png")]

        def save_json(value, path):
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        boundaries = {
            "load_yaml": lambda path: config if Path(path).name == "config.yaml" else {},
            "load_dataset": lambda _: (image, gt, ["first", "second"]),
            "resolve_device": lambda _: "cpu", "load_model": Mock(return_value=object()),
            "predict_labeled_samples": Mock(return_value=gt.copy()),
            "infer_full_image": Mock(side_effect=AssertionError("unexpected full-image inference")),
            "classification_metrics": Mock(side_effect=AssertionError("drawing must not compute test metrics")),
            "save_label_outputs": save_outputs, "save_json": save_json,
        }
        # The selector has no model dependency and is exercised through the real maps.
        selector = ROOT / "baselines" / "map_selection.py"
        if selector.exists():
            from baselines.map_selection import select_map_run
            boundaries["select_map_run"] = select_map_run
        context = load_functions(ROOT / "baselines" / model / "eval.py",
                                 ("evaluate_run", "save_labeled_maps", "save_full_map"), boundaries)
        return context, {"gt": gt, "saved_maps": saved_maps, "records": records, **boundaries}

    def test_labeled_maps_include_all_labels_without_scoring_or_overwriting_test_metrics(self):
        for model in MODELS:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temp:
                run_dir = Path(temp)
                context, state = self.make_context(model, run_dir)
                result = context["save_labeled_maps"](run_dir, run_index=1)
                np.testing.assert_array_equal(state["predict_labeled_samples"].call_args.args[2], state["gt"])
                np.testing.assert_array_equal(state["saved_maps"]["labeled_prediction"], [[1, 0, 2], [2, 1, 0]])
                self.assertEqual((run_dir / "eval_metrics.json").read_text(encoding="utf-8"), "test-metrics-sentinel")
                self.assertEqual(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")), state["records"])
                self.assertNotIn("runs", result)
                self.assertEqual(result["mode"], "labeled")
                self.assertEqual(result["run"], 1)
                self.assertEqual(result["seed"], 202202)
                self.assertEqual(result["test_oa"], 94.0)
                self.assertTrue((run_dir / "labeled_map_info.json").exists())

    def test_default_labeled_map_uses_highest_test_oa_checkpoint(self):
        for model in MODELS:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temp:
                run_dir = Path(temp)
                context, state = self.make_context(model, run_dir)
                result = context["save_labeled_maps"](run_dir)
                self.assertEqual(result["run"], 1)
                self.assertEqual(result["selection"], "highest_test_oa")
                checkpoint = state["load_model"].call_args.args[2]
                self.assertEqual(checkpoint, run_dir / "checkpoints/run_1/model_best.pth")

    def test_reassessment_only_predicts_saved_test_positions(self):
        for model in MODELS:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temp:
                run_dir = Path(temp)
                (run_dir / "splits").mkdir()
                test_gt = np.array([[-1, -1, 1], [1, -1, -1]], dtype=np.int64)
                np.savez(run_dir / "splits/run_0.npz", test_gt=test_gt)
                context, state = self.make_context(model, run_dir)
                context["predict_labeled_samples"] = Mock(return_value=test_gt.copy())
                scorer = Mock(return_value={"oa": 100.0})
                context["classification_metrics"] = scorer
                result = context["evaluate_run"](run_dir)
                np.testing.assert_array_equal(context["predict_labeled_samples"].call_args.args[2], test_gt)
                np.testing.assert_array_equal(scorer.call_args.args[1], test_gt)
                self.assertEqual(result["runs"][0]["oa"], 100.0)
                self.assertFalse((run_dir / "labeled_map_info.json").exists())

    def test_full_map_only_runs_when_explicitly_called(self):
        for model in MODELS:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temp:
                run_dir = Path(temp)
                context, state = self.make_context(model, run_dir)
                full = np.zeros((2, 3, 2)); full[..., 1] = 1
                infer = Mock(return_value=full)
                context["infer_full_image"] = infer
                paths = context["save_full_map"](run_dir, run_index=1)
                infer.assert_called_once()
                np.testing.assert_array_equal(state["saved_maps"]["full_prediction"], np.full((2, 3), 2))
                self.assertTrue(paths)
                state["predict_labeled_samples"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
