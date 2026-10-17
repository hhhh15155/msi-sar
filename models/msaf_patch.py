"""MSAF patch adaptation, based on lulushh/MSAF @ 253641e / MCF_demo_1.py.

Keep the active shallow MobileNet stages, 16/24-dim two-level token fusion and
GAP/sum classifier. Both branches use weights=None. Remove the debugger and
inactive MobileNet tails; Bernoulli attention masks apply during training only.
"""
import torch
from torch import nn
import torch.nn.functional as F


def shallow_mobilenet(channels):
    from torchvision.models import mobilenet_v3_small
    features = mobilenet_v3_small(weights=None).features[:3]
    first = features[0][0]
    features[0][0] = nn.Conv2d(channels, first.out_channels, 1, stride=1, padding=0,
                              bias=first.bias is not None)
    depthwise = features[1].block[0][0]
    features[1].block[0][0] = nn.Conv2d(depthwise.in_channels, depthwise.out_channels,
                                      depthwise.kernel_size, stride=1, padding=depthwise.padding,
                                      groups=depthwise.groups, bias=depthwise.bias is not None)
    return features


class MaskedSelfAttention(nn.Module):
    def __init__(self, embedding, heads=4, head_dim=16, dropout=0.1, mask_probability=0.1):
        super().__init__()
        if min(embedding, heads, head_dim) < 1 or not 0 <= mask_probability <= 1:
            raise ValueError('Invalid attention configuration')
        self.heads, self.head_dim, self.mask_probability = heads, head_dim, mask_probability
        inner = heads*head_dim
        self.query, self.key, self.value = (nn.Linear(embedding, inner, bias=False) for _ in range(3))
        self.projection = nn.Sequential(nn.Linear(inner, embedding), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = [layer(x).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
                   for layer in (self.query, self.key, self.value)]
        attention = (q @ k.transpose(-2, -1))*(self.head_dim**-0.5)
        if self.training and self.mask_probability:
            mask = torch.rand_like(attention) < self.mask_probability
            # Preserve the normal .1-mask distribution; only repair fully masked rows.
            mask[..., 0] &= ~mask.all(-1)
            attention = attention.masked_fill(mask, float('-inf'))
        y = attention.softmax(-1) @ v
        return self.projection(y.transpose(1, 2).reshape(b, n, self.heads*self.head_dim))


class MaskedBlock(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(embedding), nn.LayerNorm(embedding)
        self.attention = MaskedSelfAttention(embedding, head_dim=embedding)
        self.mlp = nn.Sequential(nn.Linear(embedding, 4*embedding), nn.ReLU(inplace=True),
                                 nn.Linear(4*embedding, embedding), nn.Dropout(0.1))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MaskedTokenFusion(nn.Module):
    def __init__(self, embedding, grid=5, norm_bias=True):
        super().__init__()
        self.embedding, self.grid = embedding, grid
        self.inner_projection = nn.Linear(grid*grid, embedding)
        self.position = nn.Parameter(torch.zeros(1, 2*grid*grid+embedding, embedding))
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(MaskedBlock(embedding), MaskedBlock(embedding))
        self.norm = nn.LayerNorm(embedding)
        if not norm_bias:
            # Stage 1's spatially constant bias is cancelled by the next
            # MobileNet pointwise-conv/BN in training, leaving Adam to fit noise.
            self.norm.register_parameter('bias', None)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ms, sar):
        b, c, h, w = ms.shape
        if (c, h, w) != (self.embedding, self.grid, self.grid) or sar.shape != ms.shape:
            raise ValueError('MSAF fusion requires matching latent grids/channels')
        spatial = ms.flatten(2).transpose(1, 2)
        channel_view = self.inner_projection(ms.flatten(2))
        auxiliary = sar.flatten(2).transpose(1, 2)
        tokens = torch.cat((spatial, channel_view, auxiliary), 1)
        x = self.norm(self.blocks(self.drop(tokens+self.position))).transpose(1, 2)
        return x[:, :, :h*w].reshape(b, c, h, w), x[:, :, h*w+c:].reshape(b, c, h, w)


class MSAFPatchClassifier(nn.Module):
    def __init__(self, ms_channels, sar_channels, num_classes):
        super().__init__()
        if min(ms_channels, sar_channels, num_classes) < 1:
            raise ValueError('Invalid input channels/class count')
        self.ms_channels, self.sar_channels = ms_channels, sar_channels
        self.ms_features, self.sar_features = shallow_mobilenet(ms_channels), shallow_mobilenet(sar_channels)
        self.fusion1, self.fusion2 = MaskedTokenFusion(16, norm_bias=False), MaskedTokenFusion(24)
        self.head = nn.Linear(24, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, std=1e-6)

    def forward(self, ms, sar, labels=None):
        if (ms.ndim != 4 or sar.ndim != 4 or ms.shape[0] != sar.shape[0] or
                ms.shape[1] != self.ms_channels or sar.shape[1] != self.sar_channels or
                ms.shape[-2:] != sar.shape[-2:]):
            raise ValueError('Expected aligned MSI/SAR tensors with configured channels')
        h, w = ms.shape[-2:]
        if min(h, w) < 5 or h % 2 == 0 or w % 2 == 0:
            raise ValueError('MSAF adaptation requires odd patch dimensions >= 5')
        ms = self.ms_features[1](self.ms_features[0](ms))
        sar = self.sar_features[1](self.sar_features[0](sar))
        left, right = self.fusion1(F.adaptive_avg_pool2d(ms, 5), F.adaptive_avg_pool2d(sar, 5))
        ms = ms + F.interpolate(left, (h, w), mode='bilinear', align_corners=False)
        sar = sar + F.interpolate(right, (h, w), mode='bilinear', align_corners=False)
        ms = F.adaptive_avg_pool2d(self.ms_features[2](ms), 5)
        sar = F.adaptive_avg_pool2d(self.sar_features[2](sar), 5)
        left, right = self.fusion2(ms, sar)
        fused = (ms+left).mean((-2, -1)) + (sar+right).mean((-2, -1))
        logits = self.head(fused)
        result = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result['losses'] = {'total': loss, 'ce': loss}
        return result
