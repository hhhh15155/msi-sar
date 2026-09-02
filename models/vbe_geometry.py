"""Small SPD and grouped-Gaussian primitives for the VBE numerical prototype."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class GroupedGaussian:
    """Grouped Gaussian parameters and the shrinkage used to estimate them."""

    mean: Tensor
    covariance: Tensor
    shrinkage: Tensor


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


__all__ = [
    "GroupedGaussian",
    "estimate_grouped_gaussian",
    "gaussian_bures_distance_sq",
    "matrix_invsqrt_spd",
    "matrix_sqrt_spd",
    "product_bures_distance_sq",
    "project_spd",
    "spd_from_raw_tril",
    "symmetrize",
]
