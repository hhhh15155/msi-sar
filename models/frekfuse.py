"""
FreKFuse: Frequency Kolmogorov-Arnold Fusion Network
for MSI-SAR pixel-level land cover classification.

Designed for:
  - MSI: 10 spectral channels (limited spectral info → spatial freq compensation)
  - SAR: 4 channels (VV, VH, VVbiVH, AreaSigma → rich texture via freq bands)
  - 8-class Yellow River Delta wetland classification
  - Supports both full-data and few-shot (FS20/FS50) scenarios

Key design decisions (and why):
  1. 2D spatial FFT for MSI (not 1D spectral FFT like HSI methods)
     → MSI has only 10 bands; spectral FFT is uninformative.
       Instead, use CNN stem to expand channels, then 2D FFT to capture
       spatial texture patterns in the frequency domain.

  2. Multi-band frequency decomposition for SAR
     → SAR texture spans multiple spatial scales; separating low/mid/high
       frequency bands before KAN processing lets each band specialize.

  3. Cross-Frequency Gating (CFG) for fusion
     → Instead of cross-attention (expensive) or simple concat (weak),
       each modality learns to gate the other's frequency components.
       This is a lightweight, frequency-native fusion mechanism.

  4. Pure CE loss — no auxiliary losses
     → KAN's B-spline smoothness provides implicit regularization.
       Multiple loss terms can cause training instability (as observed in
       prior multi-loss fusion methods).

  5. No spatial downsampling
     → Frequency transforms don't need downsampling to capture multi-scale
       info. Keeping full resolution avoids the 11×11→2×2 collapse.

Differences from related methods:
  - vs Two-Timer-KAN (TIP 2026): FreKFuse uses 2D spatial FFT (not 1D
    spectral FFT), multi-band SAR processing (not single FFT), cross-freq
    gating (not Gaussian fusion), and NO visual-textual alignment.
  - vs MSFMamba (TGRS 2025): FreKFuse operates in frequency domain with
    KAN activations, not state-space models.
  - vs MGHOFNet: FreKFuse replaces LSK attention with frequency-domain
    KAN blocks; fundamentally different feature extraction paradigm.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# 核心模块 1: FourierKAN — 频域 Kolmogorov-Arnold 层
# ============================================================================

class FourierKAN(nn.Module):
    """
    Frequency-domain KAN layer.

    Standard MLP:  y = σ(W·x + b)          ← fixed activation σ
    KAN:           y = Σᵢ φᵢ(xᵢ)           ← each input dim has its own
                                               learnable B-spline curve φᵢ

    In the frequency domain, each input dimension corresponds to a spatial
    frequency component (after FFT). KAN learns which frequencies matter
    for classification and applies nonlinear transformations to them.

    This replaces the MLP inside a standard Conv-BN-ReLU block when
    operating on frequency-domain features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        spline_order: int = 3,
        grid_size: int = 8,
        base_activation: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # ----- B-spline basis weights -----
        # Each (in, out) pair has (grid_size + spline_order) basis coefficients
        self.spline_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order)
            * 0.02 / math.sqrt(in_features)
        )

        # ----- Residual (linear) path -----
        # base_weight provides a linear "skip connection" through the KAN
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
        self.base_activation = nn.SiLU() if base_activation else nn.Identity()

        # ----- Fixed B-spline grid -----
        # Uniform grid in [-1, 1]; spline_order extra points extend beyond range
        self.register_buffer(
            'grid',
            torch.linspace(-1.0, 1.0, grid_size + spline_order)
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, in_features]  or  [B, in_features]
        Returns:
            [B, N, out_features]  or  [B, out_features]
        """
        squeeze = False
        if x.ndim == 2:
            squeeze = True
            x = x.unsqueeze(1)  # [B, 1, in_features]

        # Normalize input to [-1, 1] range for B-spline stability
        x_norm = torch.tanh(x)  # [B, N, in_features]

        # Compute B-spline basis values via Gaussian RBF
        # (using RBF instead of Cox-de Boor for numerical stability)
        bases = self._rbf_basis(x_norm)  # [B, N, in_features, G+S]

        # Spline path: y_spline_j = Σ_i Σ_k w_{j,i,k} * B_k(x_i)
        spline_out = torch.einsum('bnig,oig->bno', bases, self.spline_coeffs)

        # Base (linear residual) path: y_base_j = Σ_i w_{j,i} * x_i
        base_out = torch.einsum('bni,oi->bno', x, self.base_weight)
        base_out = self.base_activation(base_out)

        out = self.norm(spline_out + base_out)

        if squeeze:
            out = out.squeeze(1)
        return out

    def _rbf_basis(self, x: Tensor) -> Tensor:
        """Gaussian RBF: B_k(x_i) = exp(-((x_i - t_k) / σ)²)"""
        grid = self.grid.view(1, 1, 1, -1)     # [1, 1, 1, G+S]
        x_expanded = x.unsqueeze(-1)            # [B, N, in_features, 1]
        sigma = 2.0 / (self.grid_size + self.spline_order - 1)
        return torch.exp(-((x_expanded - grid) / sigma) ** 2)


# ============================================================================
# 核心模块 2: MSI Spatial-Freq Encoder
# ============================================================================

class MSISpatialFreqEncoder(nn.Module):
    """
    MSI encoder: CNN stem → 2D spatial FFT → FourierKAN → IFFT → pool.

    Why this design (not 1D spectral FFT):
      - MSI has only 10 bands. A 1D FFT along 10 points gives very little
        frequency resolution (5 unique components).
      - Instead: use a shallow CNN to expand 10→64 channels (spectral mixing),
        then apply 2D FFT on each channel's spatial dimension. Each of the 64
        channels captures a different spectral-spatial pattern, and the 2D FFT
        reveals its spatial frequency structure.
      - The FourierKAN then learns which spatial frequencies in which spectral
        patterns are discriminative for each land cover class.
    """

    def __init__(
        self,
        in_channels: int = 10,
        stem_channels: int = 64,
        embed_dim: int = 256,
        spline_order: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels

        # CNN stem: spectral-spatial mixing, 10→64 channels
        # Keeps full spatial resolution (no stride > 1)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
        )

        # FourierKAN blocks
        # After FFT: each spatial position has 'stem_channels' frequency values
        # We apply KAN to learn per-frequency nonlinearities
        self.kan1 = FourierKAN(stem_channels, stem_channels * 2, spline_order)
        self.kan2 = FourierKAN(stem_channels * 2, embed_dim, spline_order)

        # Frequency mixing conv (operates on frequency-domain features)
        self.freq_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 10, H, W]
        B, _, H, W = x.shape

        # Step 1: CNN stem — expand channels, mix spectral info spatially
        feat = self.stem(x)  # [B, stem_channels, H, W]
        stem_c = feat.shape[1]

        # Step 2: 2D spatial FFT
        feat_fft = torch.fft.fft2(feat.float())  # [B, stem_c, H, W] complex

        # Step 3: FourierKAN applied to frequency-domain features
        # Reshape: treat each spatial frequency position as a sample
        # [B, stem_c, H, W] → [B*H*W, stem_c]
        fft_flat = feat_fft.real.permute(0, 2, 3, 1).reshape(B * H * W, stem_c)
        fft_flat = self.kan1(fft_flat)       # [B*H*W, stem_c*2]
        fft_flat = self.kan2(fft_flat)       # [B*H*W, embed_dim]

        # Step 4: Reshape back and apply frequency-domain conv
        fft_feat = fft_flat.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, D, H, W]
        fft_feat = self.freq_conv(fft_feat)   # [B, D, H, W]

        # Step 5: Global pooling
        out = self.pool(fft_feat).flatten(1)  # [B, D]
        return F.normalize(out, dim=-1)


# ============================================================================
# 核心模块 3: SAR Multi-Band Freq Encoder
# ============================================================================

class SARFreqBandEncoder(nn.Module):
    """
    SAR encoder: stem → 2D FFT → frequency band split → per-band KAN → fuse.

    Why multi-band:
      SAR images have characteristic speckle noise (high-freq) and structural
      information at multiple scales (low→mid→high frequencies). Processing
      all frequencies together confuses noise with signal.

      Splitting into bands lets each KAN specialize:
        - Low band  (< 0.15×Nyquist): Large-scale structures (water boundaries,
          large crop fields, tidal flat extent)
        - Mid band   (0.1–0.3): Medium texture (vegetation canopy, tidal creeks,
          patch edges)
        - High band  (0.25–0.5): Fine texture + speckle discrimination

      Bands overlap slightly to avoid hard boundaries.
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        spline_order: int = 3,
    ):
        super().__init__()

        # Light stem (SAR has fewer channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Three frequency bands, each with its own KAN
        self.num_bands = 3
        # Distribute embed_dim across bands (e.g. 256 → [85, 85, 86])
        base_dim = embed_dim // self.num_bands
        band_dims = [base_dim] * (self.num_bands - 1) + [embed_dim - base_dim * (self.num_bands - 1)]

        self.band_kans = nn.ModuleList([
            FourierKAN(64, bd, spline_order)
            for bd in band_dims
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # Frequency band thresholds (in normalized frequency units [0, 0.5])
        self.register_buffer('band_lo', torch.tensor([0.00, 0.10, 0.25]))
        self.register_buffer('band_hi', torch.tensor([0.15, 0.30, 0.50]))

    def _band_mask(self, H: int, W: int, band_idx: int) -> Tensor:
        """Create a soft mask for the specified frequency band."""
        y = torch.fft.fftfreq(H, device=self.band_lo.device)
        x = torch.fft.fftfreq(W, device=self.band_lo.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt(yy ** 2 + xx ** 2)

        lo, hi = self.band_lo[band_idx], self.band_hi[band_idx]
        # Soft mask using sigmoid ramps (differentiable)
        mask_lo = torch.sigmoid((radius - lo) * 50.0)   # 0 below lo, 1 above
        mask_hi = torch.sigmoid((hi - radius) * 50.0)   # 1 below hi, 0 above
        return (mask_lo * mask_hi).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 4, H, W]
        B, _, H, W = x.shape

        # Step 1: CNN stem
        feat = self.stem(x)  # [B, 64, H, W]

        # Step 2: 2D FFT
        feat_fft = torch.fft.fft2(feat.float())  # [B, 64, H, W] complex
        feat_mag = torch.sqrt(feat_fft.real ** 2 + feat_fft.imag ** 2 + 1e-8)

        # Step 3: Per-band processing
        band_outputs = []
        for i, kan in enumerate(self.band_kans):
            mask = self._band_mask(H, W, i)           # [1, 1, H, W]
            band_feat = feat_mag * mask                # [B, 64, H, W]
            band_pooled = self.pool(band_feat).flatten(1)  # [B, 64]
            band_out = kan(band_pooled)                # [B, band_dim]
            band_outputs.append(band_out)

        # Step 4: Concatenate bands and project
        fused = torch.cat(band_outputs, dim=-1)  # [B, embed_dim]

        return F.normalize(self.proj(fused), dim=-1)


# ============================================================================
# 核心模块 4: Cross-Frequency Gating (CFG) Fusion
# ============================================================================

class CrossFreqGate(nn.Module):
    """
    Cross-Frequency Gating: modality-aware fusion in the frequency domain.

    Instead of cross-attention (computationally heavy) or Gaussian fusion
    (distribution assumptions), CFG lets each modality learn to gate the
    other's frequency components:

        MSI_gate = σ(W_msi · [msi_feat, sar_feat])   ← MSI decides what
        SAR_gate = σ(W_sar · [sar_feat, msi_feat])   ← SAR decides what

        Fused = msi_gate ⊙ msi_feat + sar_gate ⊙ sar_feat

    This is:
      - Lightweight: two small MLPs, not full attention
      - Frequency-native: gating in frequency space is like spectral filtering
      - Modality-aware: each modality's gate sees BOTH modalities' features
    """

    def __init__(self, feat_dim: int, reduction: int = 4):
        super().__init__()
        hidden = feat_dim // reduction

        # MSI gate: MSI "asks" what SAR features are useful
        self.msi_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim),
        )

        # SAR gate: SAR "asks" what MSI features are useful
        self.sar_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim),
        )

        # Post-fusion projection
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
        )

    def forward(self, msi_feat: Tensor, sar_feat: Tensor) -> Tensor:
        # msi_feat: [B, D], sar_feat: [B, D]

        # MSI-gated SAR features
        msi_input = torch.cat([msi_feat, sar_feat], dim=-1)  # [B, 2D]
        msi_gate_vals = torch.sigmoid(self.msi_gate(msi_input))  # [B, D]
        gated_sar = msi_gate_vals * sar_feat

        # SAR-gated MSI features
        sar_input = torch.cat([sar_feat, msi_feat], dim=-1)
        sar_gate_vals = torch.sigmoid(self.sar_gate(sar_input))
        gated_msi = sar_gate_vals * msi_feat

        # Fuse gated features
        fused = self.proj(torch.cat([gated_msi, gated_sar], dim=-1))  # [B, D]
        return fused


# ============================================================================
# Full Model: FreKFuse
# ============================================================================

class FreKFuse(nn.Module):
    """
    FreKFuse: Frequency Kolmogorov-Arnold Fusion Network
    for MSI-SAR image classification.

    Args:
        ms_channels:   Number of MSI spectral bands (default: 10)
        sar_channels:  Number of SAR channels (default: 4)
        num_classes:   Number of land cover classes (default: 8)
        embed_dim:     Feature embedding dimension (default: 256)
        spline_order:  B-spline order for KAN layers (3 for full data,
                       lower (2) for few-shot to reduce overfitting)
        patch_size:    Input patch size in pixels (default: 32)

    Usage:
        >>> model = FreKFuse(ms_channels=10, sar_channels=4, num_classes=8)
        >>> ms = torch.randn(4, 10, 32, 32)   # batch of MSI patches
        >>> sar = torch.randn(4, 4, 32, 32)   # batch of SAR patches
        >>> output = model(ms, sar, labels=torch.randint(0, 8, (4,)))
        >>> output['logits'].shape  # [4, 8]
        >>> output['losses']['total']  # scalar loss
    """

    def __init__(
        self,
        ms_channels: int = 10,
        sar_channels: int = 4,
        num_classes: int = 8,
        embed_dim: int = 256,
        spline_order: int = 3,
        patch_size: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.ms_channels = ms_channels
        self.sar_channels = sar_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # ----- Encoders -----
        self.msi_encoder = MSISpatialFreqEncoder(
            in_channels=ms_channels,
            stem_channels=64,
            embed_dim=embed_dim,
            spline_order=spline_order,
        )

        self.sar_encoder = SARFreqBandEncoder(
            in_channels=sar_channels,
            embed_dim=embed_dim,
            spline_order=spline_order,
        )

        # ----- Fusion -----
        self.fusion = CrossFreqGate(feat_dim=embed_dim)

        # ----- Classifier -----
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        ms: Tensor,
        sar: Tensor,
        labels: Optional[Tensor] = None,
    ) -> dict:
        """
        Args:
            ms:     MSI patch  [B, ms_channels, H, W]
            sar:    SAR patch  [B, sar_channels, H, W]
            labels: Class labels [B] (optional, only for training)

        Returns:
            dict with keys:
                'logits':    Classification logits [B, num_classes]
                'losses':    Dict with 'total' and 'ce' (only if labels given)
                'features':  Dict with 'msi', 'sar', 'fused' embeddings
        """
        # Encode
        msi_feat = self.msi_encoder(ms)   # [B, D]
        sar_feat = self.sar_encoder(sar)  # [B, D]

        # Fuse
        fused = self.fusion(msi_feat, sar_feat)  # [B, D]

        # Classify
        logits = self.classifier(fused)  # [B, num_classes]

        output = {
            'logits': logits,
            'features': {
                'msi': msi_feat,
                'sar': sar_feat,
                'fused': fused,
            },
        }

        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            output['losses'] = {
                'total': ce_loss,
                'ce': ce_loss,
            }

        return output


# ============================================================================
# Few-shot variant: FreKFuse-Lite
# ============================================================================

class FreKFuseLite(FreKFuse):
    """
    Lightweight FreKFuse for few-shot scenarios (FS20, FS50).

    Differences from full FreKFuse:
      - Smaller embed_dim (128 vs 256)
      - Lower spline_order (2 vs 3) → smoother, less overfitting
      - Higher dropout
      - Shallower MSI stem
    """

    def __init__(
        self,
        ms_channels: int = 10,
        sar_channels: int = 4,
        num_classes: int = 8,
        embed_dim: int = 128,
        spline_order: int = 2,
        patch_size: int = 32,
        dropout: float = 0.5,
    ):
        # Bypass FreKFuse.__init__ to change defaults
        nn.Module.__init__(self)
        self.ms_channels = ms_channels
        self.sar_channels = sar_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Lighter MSI stem (32 channels instead of 64)
        self.msi_encoder = MSISpatialFreqEncoder(
            in_channels=ms_channels,
            stem_channels=32,
            embed_dim=embed_dim,
            spline_order=spline_order,
        )

        # Lighter SAR encoder (fewer stem channels)
        self.sar_encoder = SARFreqBandEncoder(
            in_channels=sar_channels,
            embed_dim=embed_dim,
            spline_order=spline_order,
        )

        self.fusion = CrossFreqGate(feat_dim=embed_dim, reduction=2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()
