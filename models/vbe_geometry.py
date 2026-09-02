"""Small SPD and grouped-Gaussian primitives for the VBE numerical prototype."""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class GroupedGaussian:
    """Grouped Gaussian parameters and the shrinkage used to estimate them."""

    mean: Tensor
    covariance: Tensor
    shrinkage: Tensor


class Barycenter(NamedTuple):
    """Mean and covariance of a Bures barycenter."""

    mean: Tensor
    covariance: Tensor


class VBEResult(NamedTuple):
    """Class energies, modal responsibilities, and fused Gaussian parameters."""

    energy: Tensor
    responsibility: Tensor
    fused_mean: Tensor
    fused_covariance: Tensor


def symmetrize(matrix: Tensor) -> Tensor:
    """Return the symmetric part of a square matrix (or batch of matrices)."""

    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _spectral_map(matrix: Tensor, transform, eps: float) -> Tensor:
    values, vectors = torch.linalg.eigh(symmetrize(matrix))
    mapped = transform(values.clamp_min(eps))
    return symmetrize((vectors * mapped.unsqueeze(-2)) @ vectors.transpose(-1, -2))


def project_spd(matrix: Tensor, eps: float = 1e-4) -> Tensor:
    """Project a symmetric matrix onto the positive-definite cone."""

    return _spectral_map(matrix, lambda value: value, eps)


def matrix_sqrt_spd(matrix: Tensor, eps: float = 1e-4) -> Tensor:
    """Compute a symmetric positive-definite matrix square root."""

    return _spectral_map(matrix, torch.sqrt, eps)


def matrix_invsqrt_spd(matrix: Tensor, eps: float = 1e-4) -> Tensor:
    """Compute a symmetric positive-definite matrix inverse square root."""

    return _spectral_map(matrix, torch.rsqrt, eps)


def spd_from_raw_tril(
    raw: Tensor, eps: float = 1e-4, diagonal_floor: float = 1e-3
) -> Tensor:
    """Map unconstrained lower-triangular parameters to an SPD covariance."""

    lower = torch.tril(raw, diagonal=-1)
    diagonal = F.softplus(torch.diagonal(raw, dim1=-2, dim2=-1)) + diagonal_floor
    lower = lower + torch.diag_embed(diagonal)
    eye = torch.eye(raw.shape[-1], dtype=raw.dtype, device=raw.device)
    return symmetrize(lower @ lower.transpose(-1, -2) + eps * eye)


def estimate_grouped_gaussian(
    tokens: Tensor, groups: int = 8, eps: float = 1e-4
) -> GroupedGaussian:
    """Estimate per-group means and OAS-shrunk covariances from token features."""

    batch, samples, channels = tokens.shape
    if channels % groups:
        raise ValueError("channels must be divisible by groups")
    dim = channels // groups
    grouped = tokens.reshape(batch, samples, groups, dim).transpose(1, 2)
    mean = grouped.mean(-2)
    centered = grouped - mean.unsqueeze(-2)
    empirical = centered.transpose(-1, -2) @ centered / samples
    trace = torch.diagonal(empirical, dim1=-2, dim2=-1).sum(-1)
    trace2 = empirical.square().sum((-2, -1))
    numerator = (1 - 2 / dim) * trace2 + trace.square()
    denominator = (samples + 1 - 2 / dim) * (
        trace2 - trace.square() / dim
    )
    rho = (numerator / denominator.clamp_min(1e-12)).clamp(0, 1)
    eye = torch.eye(dim, dtype=tokens.dtype, device=tokens.device)
    target = (trace / dim)[..., None, None] * eye
    covariance = (
        (1 - rho)[..., None, None] * empirical
        + rho[..., None, None] * target
        + eps * eye
    )
    return GroupedGaussian(mean, symmetrize(covariance), rho)


def gaussian_bures_distance_sq(
    mean_a: Tensor,
    covariance_a: Tensor,
    mean_b: Tensor,
    covariance_b: Tensor,
    eps: float = 1e-4,
) -> Tensor:
    """Squared 2-Wasserstein/Bures distance between Gaussian batches."""

    covariance_a = project_spd(covariance_a, eps)
    covariance_b = project_spd(covariance_b, eps)
    root_a = matrix_sqrt_spd(covariance_a, eps)
    cross_root = matrix_sqrt_spd(root_a @ covariance_b @ root_a, eps)
    mean_term = (mean_a - mean_b).square().sum(-1)
    covariance_term = torch.diagonal(
        covariance_a + covariance_b - 2 * cross_root,
        dim1=-2,
        dim2=-1,
    ).sum(-1)
    return (mean_term + covariance_term).clamp_min(0)


def product_bures_distance_sq(
    mean_a: Tensor,
    covariance_a: Tensor,
    mean_b: Tensor,
    covariance_b: Tensor,
    eps: float = 1e-4,
    normalize: bool = True,
) -> Tensor:
    """Squared Bures distance summed over Gaussian groups."""

    per_group = gaussian_bures_distance_sq(
        mean_a, covariance_a, mean_b, covariance_b, eps
    )
    total = per_group.sum(-1)
    return total / (mean_a.shape[-2] * mean_a.shape[-1]) if normalize else total


def bures_barycenter(
    means: Tensor,
    covariances: Tensor,
    weights: Tensor,
    inner_iters: int = 3,
    eps: float = 1e-4,
) -> Barycenter:
    """Compute a weighted grouped-Gaussian Bures barycenter by fixed points."""

    weights = weights / weights.sum(-1, keepdim=True)
    mean = (weights[..., :, None, None] * means).sum(-3)
    covariance_weights = weights[..., :, None, None, None]
    covariance = project_spd((covariance_weights * covariances).sum(-4), eps)
    for _ in range(inner_iters):
        root = matrix_sqrt_spd(covariance, eps)
        invroot = matrix_invsqrt_spd(covariance, eps)
        transported = matrix_sqrt_spd(
            root.unsqueeze(-4) @ covariances @ root.unsqueeze(-4), eps
        )
        average = (covariance_weights * transported).sum(-4)
        covariance = project_spd(invroot @ average @ average @ invroot, eps)
    return Barycenter(mean, covariance)


def responsibility_from_distances(
    distances: Tensor, tau_r: float = 0.3, prior: Optional[Tensor] = None
) -> Tensor:
    """Return entropy-regularized modal responsibilities on the simplex."""

    if prior is None:
        prior = distances.new_full(
            (distances.shape[-1],), 1 / distances.shape[-1]
        )
    return torch.softmax(prior.log() - distances / tau_r, dim=-1)


def variational_bures_energy(
    prototype_mean: Tensor,
    prototype_covariance: Tensor,
    modality_mean: Tensor,
    modality_covariance: Tensor,
    lambda_proto: float = 1.0,
    tau_r: float = 0.3,
    inner_iters: int = 3,
    outer_updates: int = 1,
    eps: float = 1e-4,
    prior: Optional[Tensor] = None,
) -> VBEResult:
    """Minimize the class-conditional variational Bures energy approximately."""

    batch, modality_count, groups, group_dim = modality_mean.shape
    classes = prototype_mean.shape[0]
    prototype_means = prototype_mean[None].expand(batch, classes, groups, group_dim)
    prototype_covariances = prototype_covariance[None].expand(
        batch, classes, groups, group_dim, group_dim
    )
    modality_means = modality_mean[:, None].expand(
        batch, classes, modality_count, groups, group_dim
    )
    modality_covariances = modality_covariance[:, None].expand(
        batch, classes, modality_count, groups, group_dim, group_dim
    )
    if prior is None:
        prior = prototype_mean.new_full((modality_count,), 1 / modality_count)
    responsibility = prior.expand(batch, classes, modality_count)

    def fuse(current_responsibility: Tensor) -> Barycenter:
        means = torch.cat((prototype_means.unsqueeze(2), modality_means), dim=2)
        covariances = torch.cat(
            (prototype_covariances.unsqueeze(2), modality_covariances), dim=2
        )
        weights = torch.cat(
            (
                torch.full_like(
                    current_responsibility[..., :1], lambda_proto
                ),
                current_responsibility,
            ),
            dim=-1,
        )
        return bures_barycenter(means, covariances, weights, inner_iters, eps)

    fused = fuse(responsibility)
    for _ in range(outer_updates):
        modality_distance = product_bures_distance_sq(
            fused.mean.unsqueeze(2),
            fused.covariance.unsqueeze(2),
            modality_means,
            modality_covariances,
            eps,
        )
        responsibility = responsibility_from_distances(
            modality_distance, tau_r, prior
        )
        fused = fuse(responsibility)

    prototype_distance = product_bures_distance_sq(
        fused.mean,
        fused.covariance,
        prototype_means,
        prototype_covariances,
        eps,
    )
    modality_distance = product_bures_distance_sq(
        fused.mean.unsqueeze(2),
        fused.covariance.unsqueeze(2),
        modality_means,
        modality_covariances,
        eps,
    )
    safe_responsibility = responsibility.clamp_min(
        torch.finfo(responsibility.dtype).tiny
    )
    safe_prior = prior.clamp_min(torch.finfo(prior.dtype).tiny)
    kl = (
        responsibility
        * (safe_responsibility.log() - safe_prior.log())
    ).sum(-1)
    energy = (
        lambda_proto * prototype_distance
        + (responsibility * modality_distance).sum(-1)
        + tau_r * kl
    )
    return VBEResult(energy, responsibility, fused.mean, fused.covariance)


__all__ = [
    "GroupedGaussian",
    "Barycenter",
    "VBEResult",
    "bures_barycenter",
    "estimate_grouped_gaussian",
    "gaussian_bures_distance_sq",
    "matrix_invsqrt_spd",
    "matrix_sqrt_spd",
    "product_bures_distance_sq",
    "project_spd",
    "responsibility_from_distances",
    "spd_from_raw_tril",
    "symmetrize",
    "variational_bures_energy",
]
