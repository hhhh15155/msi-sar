from __future__ import annotations

import unittest

import numpy as np

from baselines.split import split_from_config


class DatasetSplitTests(unittest.TestCase):
    def test_fixed_train_counts_returns_only_train_and_test_masks(self) -> None:
        gt = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype="int64")

        train_gt, test_gt = split_from_config(
            gt,
            {"split": {"method": "fixed_train_counts", "train_counts": [2, 1]}},
            seed=202201,
        )

        self.assertEqual(int(np.count_nonzero(train_gt >= 0)), 3)
        self.assertEqual(int(np.count_nonzero(test_gt >= 0)), 5)
        self.assertTrue(np.all((train_gt >= 0) == (test_gt < 0)))

    def test_legacy_validation_split_method_is_rejected(self) -> None:
        gt = np.array([[0, 0], [1, 1]], dtype="int64")

        with self.assertRaisesRegex(ValueError, "fixed_train_counts"):
            split_from_config(
                gt,
                {
                    "split": {
                        "method": "fixed_counts",
                        "train_counts": [1, 1],
                        "val_counts": [1, 1],
                    }
                },
                seed=202201,
            )


if __name__ == "__main__":
    unittest.main()
