import unittest

import torch

from models.vbe_geometry import (
    estimate_grouped_gaussian,
    gaussian_bures_distance_sq,
    matrix_invsqrt_spd,
    matrix_sqrt_spd,
)


class SPDAndDistanceTests(unittest.TestCase):
    def test_sqrt_and_invsqrt_reconstruct(self):
        torch.manual_seed(7)
        raw = torch.randn(4, 3, 3, dtype=torch.float64)
        spd = raw @ raw.transpose(-1, -2) + 0.5 * torch.eye(3, dtype=torch.float64)
        root = matrix_sqrt_spd(spd, eps=1e-8)
        invroot = matrix_invsqrt_spd(spd, eps=1e-8)
        torch.testing.assert_close(root @ root, spd, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            invroot @ spd @ invroot,
            torch.eye(3).double().expand_as(spd),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_grouped_estimator_shapes_and_spd(self):
        estimate = estimate_grouped_gaussian(torch.randn(3, 121, 64), groups=8)
        self.assertEqual(estimate.mean.shape, (3, 8, 8))
        self.assertEqual(estimate.covariance.shape, (3, 8, 8, 8))
        self.assertTrue(torch.all((estimate.shrinkage >= 0) & (estimate.shrinkage <= 1)))
        self.assertGreaterEqual(
            torch.linalg.eigvalsh(estimate.covariance).min().item(), 1e-4 - 1e-6
        )

    def test_bures_identity_symmetry_and_scalar_formula(self):
        mean_a = torch.tensor([[[1.0]]], dtype=torch.float64)
        mean_b = torch.tensor([[[3.0]]], dtype=torch.float64)
        cov_a = torch.tensor([[[[4.0]]]], dtype=torch.float64)
        cov_b = torch.tensor([[[[9.0]]]], dtype=torch.float64)
        ab = gaussian_bures_distance_sq(mean_a, cov_a, mean_b, cov_b, eps=1e-8)
        ba = gaussian_bures_distance_sq(mean_b, cov_b, mean_a, cov_a, eps=1e-8)
        aa = gaussian_bures_distance_sq(mean_a, cov_a, mean_a, cov_a, eps=1e-8)
        torch.testing.assert_close(ab, torch.tensor([[5.0]], dtype=torch.float64))
        torch.testing.assert_close(ab, ba)
        self.assertLess(aa.abs().max().item(), 1e-8)


if __name__ == "__main__":
    unittest.main()
