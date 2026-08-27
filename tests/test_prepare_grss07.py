from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from scripts.prepare_grss07 import prepare_dataset


class PrepareGrss07Tests(unittest.TestCase):
    def test_prepare_dataset_writes_optical_then_sar_in_project_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            optical = np.arange(3 * 4 * 6, dtype=np.int16).reshape(3, 4, 6)
            sar = (np.arange(3 * 4, dtype=np.uint16).reshape(3, 4) + 100)
            label = np.array(
                [[0, 1, 2, 3], [4, 5, 0, 1], [2, 3, 4, 5]],
                dtype=np.uint8,
            )
            source = root / "GRSS07_SAR_MS.mat"
            savemat(source, {"HSI_data": optical, "SAR_data": sar, "ground": label})

            prepare_dataset(source, root / "grss07")

            data = loadmat(root / "grss07" / "data.mat")["data"]
            saved_label = loadmat(root / "grss07" / "label.mat")["label"]
            self.assertEqual(data.shape, (3, 4, 7))
            self.assertEqual(data.dtype, np.float32)
            np.testing.assert_array_equal(data[:, :, :6], optical.astype(np.float32))
            np.testing.assert_array_equal(data[:, :, 6], sar.astype(np.float32))
            self.assertEqual(saved_label.dtype, np.uint8)
            np.testing.assert_array_equal(saved_label, label)

    def test_prepare_dataset_rejects_misaligned_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "bad.mat"
            savemat(
                source,
                {
                    "HSI_data": np.zeros((3, 4, 6), dtype=np.int16),
                    "SAR_data": np.zeros((2, 4), dtype=np.uint16),
                    "ground": np.zeros((3, 4), dtype=np.uint8),
                },
            )

            with self.assertRaisesRegex(ValueError, "spatial shapes"):
                prepare_dataset(source, root / "grss07")

    def test_prepare_dataset_rejects_labels_outside_zero_to_five(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "bad_label.mat"
            savemat(
                source,
                {
                    "HSI_data": np.zeros((3, 4, 6), dtype=np.int16),
                    "SAR_data": np.zeros((3, 4), dtype=np.uint16),
                    "ground": np.full((3, 4), 6, dtype=np.uint8),
                },
            )

            with self.assertRaisesRegex(ValueError, "labels in 0-5"):
                prepare_dataset(source, root / "grss07")


if __name__ == "__main__":
    unittest.main()
