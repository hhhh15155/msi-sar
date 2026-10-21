"""Two controlled V6 ablations with identical common initialization."""
import torch
from torch.nn import functional as F

from models.fuzzy_transformer import FuzzyTransformer


class V6Ablation(FuzzyTransformer):
    def __init__(self, ms_channels, sar_channels, num_classes, ablation):
        if ablation not in ('a01_late_equal', 'a02_no_early_path'):
            raise ValueError(f'Unknown V6 ablation: {ablation}')
        # Construct the full, untrained V6 first. Deleting selected modules then
        # preserves every common initial tensor and the constructor RNG state.
        super().__init__(ms_channels, sar_channels, num_classes)
        self.ablation = ablation
        if ablation == 'a01_late_equal':
            fusion = self.stages[1]['fusion']
            fusion.mode = 'concat'  # Existing equal-mix + projection branch.
            del fusion.memberships
            del fusion.scorers
        else:
            del self.stages[0]['fusion']
            del self.ms_exchange
            del self.sar_exchange

    def forward(self, ms, sar, labels=None, return_fusion_weights=False):
        if self.ablation == 'a01_late_equal':
            return super().forward(ms, sar, labels, return_fusion_weights)
        if ms.ndim != 4 or sar.ndim != 4:
            raise ValueError('Expected MSI and SAR tensors [B, C, H, W]')
        if ms.shape[0] != sar.shape[0] or ms.shape[-2:] != sar.shape[-2:]:
            raise ValueError('MSI and SAR patches must have matching batch/spatial shapes')
        if ms.shape[1] != self.ms_channels or sar.shape[1] != self.sar_channels:
            raise ValueError('Input channels do not match model configuration')
        height, width = ms.shape[-2:]
        if min(height, width) < 1 or height % 2 == 0 or width % 2 == 0:
            raise ValueError('Center classification requires positive odd patch dimensions')
        position = self._position(height, width, ms)
        ms_tokens, sar_tokens = self.ms_stem(ms) + position, self.sar_stem(sar) + position
        for stage in self.stages:
            ms_tokens, sar_tokens = stage['ms'](ms_tokens), stage['sar'](sar_tokens)
        fused, weights = self.stages[1]['fusion'](ms_tokens, sar_tokens)
        fused = self.fusion_transformer(fused)
        center = (height // 2) * width + width // 2
        logits = self.head(torch.cat((fused[:, center], fused.mean(dim=1)), dim=-1))
        output = {'logits': logits}
        if labels is not None:
            ce = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            auxiliary = 0.5 * (
                F.cross_entropy(self.ms_head(ms_tokens[:, center]), labels,
                                label_smoothing=self.label_smoothing)
                + F.cross_entropy(self.sar_head(sar_tokens[:, center]), labels,
                                  label_smoothing=self.label_smoothing))
            output['losses'] = {'ce': ce, 'auxiliary': auxiliary,
                                'total': ce + self.auxiliary_weight * auxiliary}
        if return_fusion_weights:
            output['fusion_weights'] = [None, weights]
        return output
