"""Controlled MSI/SAR patch classifiers; these are adaptations, not full reproductions.

FCM/FFM reference: masurq/CFFormer, eee03787, models/new_model/modules.py.
MAFI reference: masurq/EG-CMFNet, eed0ad86, models/new_model/modules.py.
Implementations below retain the selected interaction paths, omit dense decoders,
and use finite spatial standard deviations for a single-token input. No pretrained
weights, timm, thop, Mamba or center-query cross attention are required.
"""
import torch
from torch import nn
from torch.nn import functional as F

from models.fuzzy_transformer import ConvStem, FuzzyFeatureFusion, FuzzyTransformer, transformer


def spatial_std(x):
    return x.std(dim=(-2, -1), keepdim=True, unbiased=x.shape[-2] * x.shape[-1] > 1)


class ConcatFusion(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, left, right, height, width):
        return self.project(torch.cat((left, right), dim=-1))


class FeatureCorrection(nn.Module):
    """FCFormer spatial-then-channel correction, with all gates registered."""
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2))
        self.spatial = nn.Sequential(nn.Conv2d(2 * dim, dim, 1), nn.ReLU(),
                                     nn.Conv2d(dim, 2, 1), nn.Sigmoid())
        self.channel = nn.Sequential(nn.Linear(6 * dim, 6 * dim), nn.ReLU(),
                                     nn.Linear(6 * dim, 2 * dim), nn.Sigmoid())

    def forward(self, left, right):
        gates = self.weights.relu()
        gates = gates / (gates.sum() + 1e-8)
        spatial = self.spatial(torch.cat((left, right), dim=1))
        x = left + gates[0] * spatial[:, 1:2] * right
        y = right + gates[0] * spatial[:, 0:1] * left
        both = torch.cat((x, y), dim=1)
        statistics = torch.cat((both.mean((-2, -1)), spatial_std(both).flatten(1),
                                both.amax((-2, -1))), dim=1)
        channel = self.channel(statistics).view(len(left), 2, left.shape[1], 1, 1)
        return x + gates[1] * channel[:, 1] * y, y + gates[1] * channel[:, 0] * x


class CrossInteraction(nn.Module):
    """FCFormer FFM interaction; sr_ratio=1 on the full local token grid."""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.projections = nn.ModuleList([nn.Linear(dim, 2 * dim) for _ in range(2)])
        self.queries = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(2)])
        self.key_values = nn.ModuleList([nn.Linear(dim, 2 * dim, bias=False) for _ in range(2)])
        self.ends = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])

    def forward(self, left, right):
        inputs = (left, right)
        parts = [projection(x).relu().chunk(2, dim=-1)
                 for projection, x in zip(self.projections, inputs)]
        queries, keys, values = [], [], []
        for i in range(2):
            z = parts[i][1]
            b, n, d = z.shape
            queries.append(self.queries[i](z).view(b, n, self.heads, d // self.heads).transpose(1, 2))
            kv = self.key_values[i](z).view(b, n, 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
            keys.append(kv[0])
            values.append(kv[1])
        outputs = []
        for i in range(2):
            cross = F.scaled_dot_product_attention(queries[i], keys[1 - i], values[1 - i], dropout_p=0)
            cross = cross.transpose(1, 2).reshape_as(inputs[i])
            outputs.append(self.norms[i](inputs[i] + self.ends[i](torch.cat((parts[i][0], cross), -1))))
        return outputs


class FCMFFMFusion(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.correction = FeatureCorrection(dim)
        self.interaction = CrossInteraction(dim, heads)
        self.residual = nn.Conv2d(2 * dim, dim, 1, bias=False)
        self.channel_embed = nn.Sequential(nn.Conv2d(2 * dim, dim, 1),
                                          nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.ReLU(),
                                          nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim))
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, left, right, height, width):
        b, _, d = left.shape
        images = [x.transpose(1, 2).reshape(b, d, height, width) for x in (left, right)]
        images = self.correction(*images)
        tokens = [x.flatten(2).transpose(1, 2) for x in images]
        tokens = self.interaction(*tokens)
        combined = torch.cat(tokens, dim=-1).transpose(1, 2).reshape(b, 2 * d, height, width)
        # BN cannot estimate variance from B*H*W=1. Use running statistics only
        # for that degenerate smoke-test case; the 11x11 experiment uses BN normally.
        if self.training and b * height * width == 1:
            modules = [m for m in self.modules() if isinstance(m, nn.BatchNorm2d)]
            for m in modules:
                m.eval()
            out = self.norm(self.residual(combined) + self.channel_embed(combined))
            for m in modules:
                m.train()
        else:
            out = self.norm(self.residual(combined) + self.channel_embed(combined))
        return out.flatten(2).transpose(1, 2)


class CrossChannelGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Conv2d(dim, max(1, dim // 16), 1, bias=False), nn.ReLU(),
                                nn.Conv2d(max(1, dim // 16), dim, 1, bias=False))

    def forward(self, image):
        return (self.fc(image.mean((-2, -1), keepdim=True)) +
                self.fc(image.amax((-2, -1), keepdim=True)) + self.fc(spatial_std(image))).sigmoid()


class CrossSpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 7, padding=3, bias=False)

    def forward(self, image):
        statistics = torch.cat((image.mean(1, keepdim=True), image.amax(1, keepdim=True),
                                image.std(1, keepdim=True, unbiased=image.shape[1] > 1)), dim=1)
        return self.conv(statistics).sigmoid()


class MAFIFusion(nn.Module):
    """Preserve the author's sequential, zero-initialized cross modulation."""
    def __init__(self, dim, dropout):
        super().__init__()
        self.channel = nn.ModuleList([CrossChannelGate(dim) for _ in range(2)])
        self.spatial = nn.ModuleList([CrossSpatialGate() for _ in range(2)])
        self.gamma = nn.Parameter(torch.zeros(4))
        self.readout = ConcatFusion(dim, dropout)

    def forward(self, left, right, height, width):
        b, _, d = left.shape
        x, y = [t.transpose(1, 2).reshape(b, d, height, width) for t in (left, right)]
        x = x + self.gamma[0] * self.channel[0](y) * x
        y = y + self.gamma[1] * self.channel[1](x) * y
        x = x + self.gamma[2] * self.spatial[0](y) * x
        y = y + self.gamma[3] * self.spatial[1](x) * y
        return self.readout(x.flatten(2).transpose(1, 2), y.flatten(2).transpose(1, 2), height, width)


class MainClassifier(nn.Module):
    def __init__(self, ms_channels, sar_channels, num_classes, mode='concat',
                 dim=128, heads=4, depth=2, dropout=0.1):
        super().__init__()
        if mode not in ('msi', 'sar', 'concat', 'fcm_ffm', 'mafi', 'fuzzy', 'fdmf'):
            raise ValueError('Unknown fusion mode')
        if min(ms_channels, sar_channels, num_classes, dim, heads, depth) < 1 or dim % heads:
            raise ValueError('Invalid model dimensions')
        self.mode, self.ms_channels, self.sar_channels = mode, ms_channels, sar_channels
        # Instantiate common components before the fusion, so V2--V5 have the
        # same initial encoders and head when built with the same random seed.
        if mode != 'sar':
            self.ms_stem = ConvStem(ms_channels, dim)
            self.ms_encoder = transformer(dim, heads, depth, 4, dropout)
        if mode != 'msi':
            self.sar_stem = ConvStem(sar_channels, dim)
            self.sar_encoder = transformer(dim, heads, depth, 4, dropout)
        self.position = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(dim, num_classes))
        if mode == 'concat':
            self.fusion = ConcatFusion(dim, dropout)
        elif mode == 'fcm_ffm':
            self.fusion = FCMFFMFusion(dim, heads)
        elif mode == 'mafi':
            self.fusion = MAFIFusion(dim, dropout)
        elif mode == 'fuzzy':
            self.fusion = FuzzyFeatureFusion(dim, 7, 0.15, 1, dropout)
        elif mode == 'fdmf':
            from models.fdmf_patch import FrequencyFusion
            self.fusion = FrequencyFusion(dim, heads, dropout)

    def forward(self, ms, sar, labels=None):
        if ms.ndim != 4 or sar.ndim != 4 or ms.shape[0] != sar.shape[0] or ms.shape[-2:] != sar.shape[-2:]:
            raise ValueError('Expected aligned [B,C,H,W] inputs')
        if ms.shape[1] != self.ms_channels or sar.shape[1] != self.sar_channels:
            raise ValueError('Input channel mismatch')
        h, w = ms.shape[-2:]
        if min(h, w) < 1 or h % 2 == 0 or w % 2 == 0:
            raise ValueError('Positive odd patch dimensions required')
        ys = (torch.arange(h, device=ms.device, dtype=ms.dtype) - h // 2) / max(h // 2, 1)
        xs = (torch.arange(w, device=ms.device, dtype=ms.dtype) - w // 2) / max(w // 2, 1)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        position = self.position(torch.stack((yy, xx), -1).reshape(1, h * w, 2))
        left = self.ms_encoder(self.ms_stem(ms) + position) if self.mode != 'sar' else None
        right = self.sar_encoder(self.sar_stem(sar) + position) if self.mode != 'msi' else None
        if self.mode == 'msi':
            fused = left
        elif self.mode == 'sar':
            fused = right
        elif self.mode == 'fuzzy':
            fused, _ = self.fusion(left, right)
        else:
            fused = self.fusion(left, right, h, w)
        logits = self.head(torch.cat((fused[:, (h // 2) * w + w // 2], fused.mean(1)), dim=-1))
        out = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            out['losses'] = {'total': loss, 'ce': loss}
        return out


VARIANTS = ('v0_msi', 'v1_sar', 'v2_concat', 'v3_fcm_ffm', 'v4_mafi', 'v5_fuzzy', 'v6_current')
CANDIDATE_VARIANTS = ('v7_fdmf_fixed', 'v8_msaf_fixed')
ABLATION_VARIANTS = ('a01_late_equal', 'a02_no_early_path')


def build_main_model(config, num_classes):
    variant = config['variant']
    if variant not in VARIANTS + CANDIDATE_VARIANTS + ABLATION_VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    common = dict(ms_channels=int(config['ms_channels']), sar_channels=int(config['sar_channels']),
                  num_classes=num_classes)
    if variant in ABLATION_VARIANTS:
        from models.v6_ablation import V6Ablation
        return V6Ablation(**common, ablation=variant)
    if variant == 'v8_msaf_fixed':
        from models.msaf_patch import MSAFPatchClassifier
        return MSAFPatchClassifier(**common)
    if variant == 'v6_current':
        return FuzzyTransformer(**common)  # Preserve current 192/6, two stages, feedback and auxiliary loss.
    mode = 'fdmf' if variant == 'v7_fdmf_fixed' else ('msi', 'sar', 'concat', 'fcm_ffm', 'mafi', 'fuzzy')[VARIANTS.index(variant)]
    return MainClassifier(**common, mode=mode,
                          dim=int(config.get('embed_dim', 128)), heads=int(config.get('num_heads', 4)),
                          depth=int(config.get('depth', 2)), dropout=float(config.get('dropout', 0.1)))
