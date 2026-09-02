"""VBE-Net: class-conditional variational Bures energy network."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class GroupedGaussian:
    """Grouped Gaussian parameters and their OAS shrinkage values."""

    mean: Tensor
    covariance: Tensor
    shrinkage: Tensor


class Barycenter(NamedTuple):
    """Mean and covariance of a grouped-Gaussian Bures barycenter."""

    mean: Tensor
    covariance: Tensor


class VBEResult(NamedTuple):
    """Class energies, responsibilities, and fused Gaussian parameters."""

    energy: Tensor
    responsibility: Tensor
    fused_mean: Tensor
    fused_covariance: Tensor


def symmetrize(matrix: Tensor) -> Tensor:
    """Return the symmetric part of a matrix or matrix batch."""

    return 0.5 * (matrix + matrix.transpose(-1, -2))


class _StableMatrixSquareRoot(torch.autograd.Function):
    """SPD square root with a repeated-eigenvalue-safe first derivative."""

    @staticmethod
    def forward(ctx, matrix: Tensor, eps: float) -> Tensor:
        values, vectors = torch.linalg.eigh(symmetrize(matrix))
        root_values = values.clamp_min(eps).sqrt()
        ctx.save_for_backward(vectors, root_values)
        root = (vectors * root_values.unsqueeze(-2)) @ vectors.transpose(-1, -2)
        return symmetrize(root)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        vectors, root_values = ctx.saved_tensors
        local_gradient = (
            vectors.transpose(-1, -2) @ symmetrize(gradient) @ vectors
        )
        denominator = root_values.unsqueeze(-1) + root_values.unsqueeze(-2)
        local_gradient = local_gradient / denominator
        input_gradient = vectors @ local_gradient @ vectors.transpose(-1, -2)
        return symmetrize(input_gradient), None


def project_spd(matrix: Tensor, eps: float = 1e-4) -> Tensor:
    """Shift a symmetric matrix batch onto the positive-definite cone."""

    matrix = symmetrize(matrix)
    with torch.no_grad():
        minimum = torch.linalg.eigvalsh(matrix).amin(dim=-1)
        shift = (eps - minimum).clamp_min(0)
    identity = torch.eye(
        matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    )
    return symmetrize(matrix + shift[..., None, None] * identity)


def matrix_sqrt_spd(matrix: Tensor, eps: float = 1e-4) -> Tensor:
    """Compute a stable symmetric positive-definite matrix square root."""

    return _StableMatrixSquareRoot.apply(project_spd(matrix, eps), eps)


def matrix_invsqrt_spd(matrix: Tensor, eps: float = 1e-4) -> Tensor:
    """Compute a stable symmetric positive-definite inverse square root."""

    root = matrix_sqrt_spd(matrix, eps)
    identity = torch.eye(
        root.shape[-1], dtype=root.dtype, device=root.device
    ).expand_as(root)
    return symmetrize(torch.linalg.solve(root, identity))


def spd_from_raw_tril(
    raw: Tensor,
    eps: float = 1e-4,
    diagonal_floor: float = 1e-3,
) -> Tensor:
    """Map unconstrained lower-triangular parameters to SPD covariance."""

    lower = torch.tril(raw, diagonal=-1)
    diagonal = F.softplus(torch.diagonal(raw, dim1=-2, dim2=-1))
    diagonal = diagonal + diagonal_floor
    lower = lower + torch.diag_embed(diagonal)
    identity = torch.eye(raw.shape[-1], dtype=raw.dtype, device=raw.device)
    return symmetrize(lower @ lower.transpose(-1, -2) + eps * identity)


def estimate_grouped_gaussian(
    tokens: Tensor,
    groups: int = 8,
    eps: float = 1e-4,
) -> GroupedGaussian:
    """Estimate grouped means and OAS-shrunk covariance matrices."""

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B,N,D]")
    batch, samples, channels = tokens.shape
    if channels % groups:
        raise ValueError("channels must be divisible by groups")
    group_dim = channels // groups
    grouped = tokens.reshape(
        batch, samples, groups, group_dim
    ).transpose(1, 2)
    mean = grouped.mean(dim=-2)
    centered = grouped - mean.unsqueeze(-2)
    empirical = centered.transpose(-1, -2) @ centered / samples
    trace = torch.diagonal(empirical, dim1=-2, dim2=-1).sum(dim=-1)
    trace_square = empirical.square().sum(dim=(-2, -1))
    numerator = (1.0 - 2.0 / group_dim) * trace_square + trace.square()
    denominator = (samples + 1.0 - 2.0 / group_dim) * (
        trace_square - trace.square() / group_dim
    )
    shrinkage = (numerator / denominator.clamp_min(1e-12)).clamp(0.0, 1.0)
    identity = torch.eye(
        group_dim, dtype=tokens.dtype, device=tokens.device
    )
    target = (trace / group_dim)[..., None, None] * identity
    covariance = (
        (1.0 - shrinkage)[..., None, None] * empirical
        + shrinkage[..., None, None] * target
        + eps * identity
    )
    return GroupedGaussian(mean, symmetrize(covariance), shrinkage)


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
    mean_term = (mean_a - mean_b).square().sum(dim=-1)
    covariance_term = torch.diagonal(
        covariance_a + covariance_b - 2.0 * cross_root,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)
    return (mean_term + covariance_term).clamp_min(0.0)


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
    total = per_group.sum(dim=-1)
    if normalize:
        total = total / (mean_a.shape[-2] * mean_a.shape[-1])
    return total


def bures_barycenter(
    means: Tensor,
    covariances: Tensor,
    weights: Tensor,
    inner_iters: int = 3,
    eps: float = 1e-4,
) -> Barycenter:
    """Compute a weighted grouped-Gaussian Bures barycenter."""

    weights = weights / weights.sum(dim=-1, keepdim=True)
    mean = (weights[..., :, None, None] * means).sum(dim=-3)
    covariance_weights = weights[..., :, None, None, None]
    covariance = project_spd(
        (covariance_weights * covariances).sum(dim=-4), eps
    )
    for _ in range(inner_iters):
        root = matrix_sqrt_spd(covariance, eps)
        inverse_root = matrix_invsqrt_spd(covariance, eps)
        transported = matrix_sqrt_spd(
            root.unsqueeze(-4) @ covariances @ root.unsqueeze(-4), eps
        )
        average = (covariance_weights * transported).sum(dim=-4)
        covariance = project_spd(
            inverse_root @ average @ average @ inverse_root, eps
        )
    return Barycenter(mean, covariance)


def responsibility_from_distances(
    distances: Tensor,
    tau_r: float = 0.3,
    prior: Optional[Tensor] = None,
) -> Tensor:
    """Return entropy-regularized modal responsibilities on the simplex."""

    if prior is None:
        prior = distances.new_full(
            (distances.shape[-1],), 1.0 / distances.shape[-1]
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
    """Approximately minimize the class-conditional VBE objective."""

    batch, modality_count, groups, group_dim = modality_mean.shape
    classes = prototype_mean.shape[0]
    prototype_means = prototype_mean[None].expand(
        batch, classes, groups, group_dim
    )
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
        prior = prototype_mean.new_full(
            (modality_count,), 1.0 / modality_count
        )
    responsibility = prior.expand(batch, classes, modality_count)

    def fuse(current_responsibility: Tensor) -> Barycenter:
        means = torch.cat(
            (prototype_means.unsqueeze(2), modality_means), dim=2
        )
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
        return bures_barycenter(
            means, covariances, weights, inner_iters, eps
        )

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
        responsibility * (safe_responsibility.log() - safe_prior.log())
    ).sum(dim=-1)
    energy = (
        lambda_proto * prototype_distance
        + (responsibility * modality_distance).sum(dim=-1)
        + tau_r * kl
    )
    return VBEResult(
        energy, responsibility, fused.mean, fused.covariance
    )


class VBEModelOutput(NamedTuple):
    """Detailed VBE-Net output for analysis and diagnostics."""

    logits: Tensor
    energy: Tensor
    responsibility: Tensor
    ms_shrinkage: Tensor
    sar_shrinkage: Tensor


class LayerNorm2d(nn.Module):
    """LayerNorm over channels while preserving an image tensor layout."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        channels_last = inputs.permute(0, 2, 3, 1)
        return self.norm(channels_last).permute(0, 3, 1, 2).contiguous()


class ModalityStem(nn.Module):
    """Modality-private spectral/spatial projection without downsampling."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm = LayerNorm2d(width)
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.norm(self.projection(inputs)))


class SemiSharedResidualBlock(nn.Module):
    """Private spatial operators with a shared channel-mixing transform."""

    def __init__(self, width: int, expansion: int = 4) -> None:
        super().__init__()
        hidden_width = width * expansion

        self.ms_spatial_norm = LayerNorm2d(width)
        self.sar_spatial_norm = LayerNorm2d(width)
        self.ms_depthwise = nn.Conv2d(
            width, width, kernel_size=3, padding=1, groups=width
        )
        self.sar_depthwise = nn.Conv2d(
            width, width, kernel_size=3, padding=1, groups=width
        )

        self.ms_mixer_norm = LayerNorm2d(width)
        self.sar_mixer_norm = LayerNorm2d(width)
        self.shared_pw1 = nn.Conv2d(width, hidden_width, kernel_size=1)
        self.shared_pw2 = nn.Conv2d(hidden_width, width, kernel_size=1)
        self.activation = nn.GELU()

    def _mix(self, inputs: Tensor, norm: nn.Module) -> Tensor:
        return self.shared_pw2(self.activation(self.shared_pw1(norm(inputs))))

    def forward(self, ms: Tensor, sar: Tensor) -> tuple[Tensor, Tensor]:
        ms_spatial = ms + self.ms_depthwise(self.ms_spatial_norm(ms))
        sar_spatial = sar + self.sar_depthwise(self.sar_spatial_norm(sar))
        ms_output = ms_spatial + self._mix(ms_spatial, self.ms_mixer_norm)
        sar_output = sar_spatial + self._mix(sar_spatial, self.sar_mixer_norm)
        return ms_output, sar_output


class SemiSharedEncoder(nn.Module):
    """Two modality lanes sharing channel mixers but not spatial filters."""

    def __init__(
        self,
        ms_channels: int,
        sar_channels: int,
        width: int = 64,
        depth: int = 5,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.ms_stem = ModalityStem(ms_channels, width)
        self.sar_stem = ModalityStem(sar_channels, width)
        self.blocks = nn.ModuleList(
            SemiSharedResidualBlock(width, expansion) for _ in range(depth)
        )

    def forward(self, ms: Tensor, sar: Tensor) -> tuple[Tensor, Tensor]:
        ms_features = self.ms_stem(ms)
        sar_features = self.sar_stem(sar)
        for block in self.blocks:
            ms_features, sar_features = block(ms_features, sar_features)
        return ms_features, sar_features


class VBENet(nn.Module):
    """End-to-end MSI-SAR classifier driven by variational Bures energy."""

    def __init__(
        self,
        ms_channels: int = 10,
        sar_channels: int = 4,
        num_classes: int = 8,
        patch_size: int = 11,
        width: int = 64,
        depth: int = 5,
        groups: int = 8,
        expansion: int = 4,
        lambda_proto: float = 1.0,
        tau_r: float = 0.3,
        tau_c: float = 0.1,
        inner_iters: int = 3,
        outer_updates: int = 1,
        modality_dropout: float = 0.1,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if width <= 0 or width % groups:
            raise ValueError("width must be positive and divisible by groups")
        if depth <= 0 or expansion <= 0:
            raise ValueError("depth and expansion must be positive")
        if num_classes <= 1 or patch_size <= 0:
            raise ValueError("num_classes must exceed one and patch_size must be positive")
        if tau_r <= 0 or tau_c <= 0 or lambda_proto <= 0:
            raise ValueError("lambda_proto, tau_r, and tau_c must be positive")
        if inner_iters < 0 or outer_updates < 0:
            raise ValueError("solver iteration counts must be non-negative")
        if not 0.0 <= modality_dropout <= 1.0:
            raise ValueError("modality_dropout must be in [0, 1]")

        self.ms_channels = ms_channels
        self.sar_channels = sar_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.width = width
        self.groups = groups
        self.group_dim = width // groups
        self.lambda_proto = lambda_proto
        self.tau_r = tau_r
        self.tau_c = tau_c
        self.inner_iters = inner_iters
        self.outer_updates = outer_updates
        self.modality_dropout = modality_dropout
        self.eps = eps

        self.encoder = SemiSharedEncoder(
            ms_channels=ms_channels,
            sar_channels=sar_channels,
            width=width,
            depth=depth,
            expansion=expansion,
        )
        self.prototype_mean = nn.Parameter(
            torch.empty(num_classes, groups, self.group_dim)
        )
        raw_tril = torch.zeros(
            num_classes,
            groups,
            self.group_dim,
            self.group_dim,
        )
        unit_diagonal_raw = math.log(math.expm1(1.0 - 1e-3))
        raw_tril.diagonal(dim1=-2, dim2=-1).fill_(unit_diagonal_raw)
        self.prototype_raw_tril = nn.Parameter(raw_tril)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.prototype_mean, mean=0.0, std=0.02)

    def _validate_inputs(self, ms: Tensor, sar: Tensor) -> None:
        if ms.ndim != 4 or sar.ndim != 4:
            raise ValueError("MS and SAR inputs must have shape [B,C,H,W]")
        if ms.shape[0] != sar.shape[0]:
            raise ValueError("MS and SAR batch sizes must match")
        if ms.shape[1] != self.ms_channels:
            raise ValueError(
                f"Expected {self.ms_channels} MS channels, got {ms.shape[1]}"
            )
        if sar.shape[1] != self.sar_channels:
            raise ValueError(
                f"Expected {self.sar_channels} SAR channels, got {sar.shape[1]}"
            )
        expected_spatial = (self.patch_size, self.patch_size)
        if ms.shape[-2:] != expected_spatial or sar.shape[-2:] != expected_spatial:
            raise ValueError(
                f"Expected spatial size {expected_spatial}, got "
                f"MS={tuple(ms.shape[-2:])}, SAR={tuple(sar.shape[-2:])}"
            )

    def encode_tokens(self, ms: Tensor, sar: Tensor) -> tuple[Tensor, Tensor]:
        """Encode both modalities as full-resolution spatial token sequences."""

        self._validate_inputs(ms, sar)
        ms_features, sar_features = self.encoder(ms, sar)
        ms_tokens = ms_features.flatten(2).transpose(1, 2)
        sar_tokens = sar_features.flatten(2).transpose(1, 2)
        return ms_tokens, sar_tokens

    def _select_modalities(
        self,
        modality_mean: Tensor,
        modality_covariance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            not self.training
            or self.modality_dropout == 0.0
            or torch.rand((), device=modality_mean.device) >= self.modality_dropout
        ):
            return modality_mean, modality_covariance
        keep_index = int(torch.randint(2, (), device=modality_mean.device).item())
        return (
            modality_mean[:, keep_index : keep_index + 1],
            modality_covariance[:, keep_index : keep_index + 1],
        )

    def forward(
        self,
        ms: Tensor,
        sar: Tensor,
        return_details: bool = False,
    ) -> Union[Tensor, VBEModelOutput]:
        ms_tokens, sar_tokens = self.encode_tokens(ms, sar)

        # EVD-based geometry is intentionally excluded from mixed precision.
        with torch.autocast(device_type=ms.device.type, enabled=False):
            ms_gaussian = estimate_grouped_gaussian(
                ms_tokens.float(), groups=self.groups, eps=self.eps
            )
            sar_gaussian = estimate_grouped_gaussian(
                sar_tokens.float(), groups=self.groups, eps=self.eps
            )
            modality_mean = torch.stack(
                (ms_gaussian.mean, sar_gaussian.mean), dim=1
            )
            modality_covariance = torch.stack(
                (ms_gaussian.covariance, sar_gaussian.covariance), dim=1
            )
            modality_mean, modality_covariance = self._select_modalities(
                modality_mean, modality_covariance
            )
            prototype_covariance = spd_from_raw_tril(
                self.prototype_raw_tril.float(), eps=self.eps
            )
            result = variational_bures_energy(
                self.prototype_mean.float(),
                prototype_covariance,
                modality_mean,
                modality_covariance,
                lambda_proto=self.lambda_proto,
                tau_r=self.tau_r,
                inner_iters=self.inner_iters,
                outer_updates=self.outer_updates,
                eps=self.eps,
            )
            logits = -result.energy / self.tau_c

        if not return_details:
            return logits
        return VBEModelOutput(
            logits=logits,
            energy=result.energy,
            responsibility=result.responsibility,
            ms_shrinkage=ms_gaussian.shrinkage,
            sar_shrinkage=sar_gaussian.shrinkage,
        )


__all__ = [
    "GroupedGaussian",
    "Barycenter",
    "VBEResult",
    "symmetrize",
    "project_spd",
    "matrix_sqrt_spd",
    "matrix_invsqrt_spd",
    "spd_from_raw_tril",
    "estimate_grouped_gaussian",
    "gaussian_bures_distance_sq",
    "product_bures_distance_sq",
    "bures_barycenter",
    "responsibility_from_distances",
    "variational_bures_energy",
    "LayerNorm2d",
    "ModalityStem",
    "SemiSharedResidualBlock",
    "SemiSharedEncoder",
    "VBEModelOutput",
    "VBENet",
]
