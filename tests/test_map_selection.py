from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class MapSelectionTests(unittest.TestCase):
    def test_best_uses_saved_test_metrics_and_first_tie(self):
        from baselines.map_selection import select_map_run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metrics = {"runs": [
                {"run": 0, "seed": 202201, "oa": 82.0},
                {"run": 1, "seed": 202202, "oa": 94.0},
                {"run": 2, "seed": 202203, "oa": 94.0},
            ]}
            path = root / "metrics.json"
            path.write_text(json.dumps(metrics), encoding="utf-8")
            self.assertEqual(select_map_run(root), {"run": 1, "seed": 202202, "test_oa": 94.0, "selection": "highest_test_oa"})
            self.assertEqual(select_map_run(root, 0)["run"], 0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), metrics)

    def test_explicit_run_can_draw_without_metrics_but_best_cannot(self):
        from baselines.map_selection import select_map_run
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(select_map_run(temp, "2")["run"], 2)
            with self.assertRaisesRegex(ValueError, "metrics.json"):
                select_map_run(temp, "best")
            with self.assertRaises(ValueError):
                select_map_run(temp, -1)


if __name__ == "__main__":
    unittest.main()
