import unittest

import torch

from models.vbe_geometry import (
    bures_barycenter,
    estimate_grouped_gaussian,
    gaussian_bures_distance_sq,
    matrix_invsqrt_spd,
    matrix_sqrt_spd,
    product_bures_distance_sq,
    project_spd,
    responsibility_from_distances,
    variational_bures_energy,
)


class SpectralGradientRegressionTests(unittest.TestCase):
    def test_isotropic_spd_spectral_maps_have_finite_gradient(self) -> None:
        covariance = (2.0 * torch.eye(4)).requires_grad_()

        mapped = (
            project_spd(covariance)
            + matrix_sqrt_spd(covariance)
            + matrix_invsqrt_spd(covariance)
        ).square().sum()
        gradient, = torch.autograd.grad(mapped, covariance)

        self.assertTrue(torch.isfinite(gradient).all())


def _make_spd(shape, dtype, *, offset=0.5):
    """Create a deterministic, well-conditioned SPD batch for solver tests."""

    raw = torch.randn(*shape, dtype=dtype)
    dim = shape[-1]
    eye = torch.eye(dim, dtype=dtype).expand(*shape[:-2], dim, dim)
    return raw @ raw.transpose(-1, -2) + offset * eye


def make_test_case(batch, classes, groups, dim, dtype, requires_grad):
    """Return differentiable prototype and modality Gaussian parameters."""

    torch.manual_seed(19)
    prototype_mean = torch.randn(classes, groups, dim, dtype=dtype)
    prototype_covariance = _make_spd((classes, groups, dim, dim), dtype)
    modality_mean = torch.randn(batch, 2, groups, dim, dtype=dtype)
    modality_covariance = _make_spd((batch, 2, groups, dim, dim), dtype)
    if requires_grad:
        return tuple(
            tensor.detach().requires_grad_()
            for tensor in (
                prototype_mean,
                prototype_covariance,
                modality_mean,
                modality_covariance,
            )
        )
    return prototype_mean, prototype_covariance, modality_mean, modality_covariance


def differentiable_leaves(case):
    return [tensor for tensor in case if tensor.requires_grad]


def run_solver_gradcheck(seed, atol, rtol):
    torch.manual_seed(seed)
    case = make_test_case(
        batch=1,
        classes=2,
        groups=1,
        dim=2,
        dtype=torch.float64,
        requires_grad=True,
    )

    def energy_sum(*inputs):
        return variational_bures_energy(
            *inputs, inner_iters=2, outer_updates=1, eps=1e-6
        ).energy.sum()

    return torch.autograd.gradcheck(energy_sum, case, atol=atol, rtol=rtol)


def _variational_objective(
    fused, prototype_mean, prototype_covariance, modality_mean, modality_covariance,
    alpha, lambda_proto, tau_r, prior, eps,
):
    prototype_distance = product_bures_distance_sq(
        fused.mean, fused.covariance, prototype_mean, prototype_covariance, eps
    )
    modality_distance = product_bures_distance_sq(
        fused.mean.unsqueeze(-3), fused.covariance.unsqueeze(-3),
        modality_mean, modality_covariance, eps,
    )
    safe_alpha = alpha.clamp_min(torch.finfo(alpha.dtype).tiny)
    safe_prior = prior.clamp_min(torch.finfo(prior.dtype).tiny)
    return (
        lambda_proto * prototype_distance
        + (alpha * modality_distance).sum(-1)
        + tau_r * (alpha * (safe_alpha.log() - safe_prior.log())).sum(-1)
    )


def _reference_coordinate_energies():
    """Evaluate the two exact coordinate updates in high precision."""

    torch.manual_seed(31)
    prototype_mean = torch.tensor([[[0.2, -0.3, 0.6]]], dtype=torch.float64)
    prototype_covariance = _make_spd((1, 1, 3, 3), torch.float64, offset=1.0)
    modality_mean = torch.tensor(
        [[[[0.0, -0.1, 0.4]], [[0.9, 0.3, -0.4]]]], dtype=torch.float64
    )
    modality_covariance = _make_spd((1, 2, 1, 3, 3), torch.float64, offset=1.0)
    lambda_proto, tau_r, eps = 1.0, 0.3, 1e-8
    prior = torch.full((2,), 0.5, dtype=torch.float64)
    alpha0 = prior.unsqueeze(0)

    def fuse(alpha):
        return bures_barycenter(
            torch.cat((prototype_mean.unsqueeze(-3), modality_mean), dim=-3),
            torch.cat((prototype_covariance.unsqueeze(-4), modality_covariance), dim=-4),
            torch.cat((torch.full_like(alpha[..., :1], lambda_proto), alpha), dim=-1),
            inner_iters=30,
            eps=eps,
        )

    fused0 = fuse(alpha0)
    energy0 = _variational_objective(
        fused0, prototype_mean, prototype_covariance, modality_mean,
        modality_covariance, alpha0, lambda_proto, tau_r, prior, eps,
    )
    distances0 = product_bures_distance_sq(
        fused0.mean.unsqueeze(-3), fused0.covariance.unsqueeze(-3),
        modality_mean, modality_covariance, eps,
    )
    alpha1 = responsibility_from_distances(distances0, tau_r=tau_r, prior=prior)
    energy_alpha = _variational_objective(
        fused0, prototype_mean, prototype_covariance, modality_mean,
        modality_covariance, alpha1, lambda_proto, tau_r, prior, eps,
    )
    fused1 = fuse(alpha1)
    energy_fused = _variational_objective(
        fused1, prototype_mean, prototype_covariance, modality_mean,
        modality_covariance, alpha1, lambda_proto, tau_r, prior, eps,
    )
    return energy0, energy_alpha, energy_fused


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


class VariationalSolverTests(unittest.TestCase):
    def test_identical_barycenter_returns_input(self):
        mean = torch.tensor([[[1.0, -1.0]]], dtype=torch.float64)
        covariance = torch.tensor(
            [[[[2.0, 0.3], [0.3, 1.0]]]], dtype=torch.float64
        )
        result = bures_barycenter(
            mean.unsqueeze(-3).expand(1, 3, 1, 2),
            covariance.unsqueeze(-4).expand(1, 3, 1, 2, 2),
            torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64),
            inner_iters=10,
            eps=1e-8,
        )
        torch.testing.assert_close(result.mean, mean, rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(result.covariance, covariance, rtol=1e-5, atol=1e-6)

    def test_responsibility_is_simplex_and_favors_near_modality(self):
        alpha = responsibility_from_distances(
            torch.tensor([[[0.1, 0.9], [1.2, 0.2]]]), tau_r=0.3
        )
        torch.testing.assert_close(alpha.sum(-1), torch.ones(1, 2))
        self.assertGreater(alpha[0, 0, 0], alpha[0, 0, 1])
        self.assertGreater(alpha[0, 1, 1], alpha[0, 1, 0])

    def test_solver_backward_is_finite(self):
        case = make_test_case(
            batch=2,
            classes=3,
            groups=2,
            dim=3,
            dtype=torch.float32,
            requires_grad=True,
        )
        result = variational_bures_energy(*case, inner_iters=3, outer_updates=1)
        self.assertEqual(result.energy.shape, (2, 3))
        result.energy.sum().backward()
        for leaf in differentiable_leaves(case):
            self.assertIsNotNone(leaf.grad)
            self.assertTrue(torch.isfinite(leaf.grad).all())

    def test_small_float64_gradcheck(self):
        self.assertTrue(run_solver_gradcheck(seed=23, atol=3e-4, rtol=2e-3))

    def test_exact_coordinate_updates_do_not_raise_reference_energy(self):
        energy0, energy_alpha, energy_fused = _reference_coordinate_energies()
        self.assertLessEqual(energy_alpha.item(), energy0.item() + 1e-7)
        self.assertLessEqual(energy_fused.item(), energy_alpha.item() + 1e-7)

    def test_near_repeated_spectrum_backward_is_finite(self):
        prototype_mean = torch.zeros(1, 1, 3, requires_grad=True)
        prototype_covariance = torch.diag_embed(
            torch.tensor([[[1.0, 1.0 + 1e-5, 1.0 + 2e-5]]])
        ).requires_grad_()
        modality_mean = torch.tensor(
            [[[[0.1, -0.2, 0.3]], [[-0.4, 0.5, -0.6]]]], requires_grad=True
        )
        modality_covariance = torch.diag_embed(
            torch.tensor(
                [[[[1.0, 1.0 + 1e-5, 1.0 + 2e-5]],
                  [[1.0 + 2e-5, 1.0, 1.0 + 1e-5]]]]
            )
        ).requires_grad_()
        result = variational_bures_energy(
            prototype_mean,
            prototype_covariance,
            modality_mean,
            modality_covariance,
            inner_iters=3,
            outer_updates=1,
        )
        result.energy.sum().backward()
        for leaf in (
            prototype_mean,
            prototype_covariance,
            modality_mean,
            modality_covariance,
        ):
            self.assertIsNotNone(leaf.grad)
            self.assertTrue(torch.isfinite(leaf.grad).all())


if __name__ == "__main__":
    unittest.main()
