"""Single-scale FDMF-inspired fusion for the shared patch classifier.

Reference: fy-sun/FDMF-Net @ 15c5aa2898d6b8636aedd8dc997f446c47030f12,
models/frequency_modules.py, encoder_agg.py::Fusion and attention.py.
This is a classification adaptation, not the four-stage segmentation network.
The non-differentiable threshold generator is replaced by an explicit fixed .5
cutoff. The original abs(IFFT) convention and rectangular mask are retained.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class FixedFrequencySplit(nn.Module):
    def __init__(self, dim, fraction=0.5):
        super().__init__()
        if dim < 1 or not 0 < fraction < 1:
            raise ValueError('Invalid frequency split dimensions/cutoff')
        self.fraction = fraction
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        h, w = x.shape[-2:]
        rh, rw = max(1, round((h//2)*self.fraction)), max(1, round((w//2)*self.fraction))
        yy = torch.arange(h, device=x.device)[:, None]
        xx = torch.arange(w, device=x.device)[None, :]
        mask = ((yy >= h//2-rh) & (yy < h//2+rh) &
                (xx >= w//2-rw) & (xx < w//2+rw))
        spectrum = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'), dim=(-2, -1))
        def inverse(s):
            return torch.fft.ifft2(torch.fft.ifftshift(s, dim=(-2, -1)), norm='forward').abs()
        return inverse(spectrum * (~mask)), inverse(spectrum * mask)


class SpatialWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(4, 16, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, x, frequency):
        statistics = (x.mean(1, keepdim=True), x.amax(1, keepdim=True),
                      frequency.mean(1, keepdim=True), frequency.amax(1, keepdim=True))
        return self.mlp(torch.cat(statistics, 1))


class FrequencySpatialEnhancement(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_weight, self.low_weight = SpatialWeight(), SpatialWeight()

    def forward(self, x, high, low):
        # In the author network, the additional spatial_gate only feeds the next
        # encoder stage. There is no next stage here, so do not register dead weights.
        return x + self.high_weight(x, high)*high, x + self.low_weight(x, low)*low


class SpatialFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4*dim)
        self.depthwise = nn.Conv2d(4*dim, 4*dim, 3, padding=1, groups=4*dim)
        self.fc2 = nn.Linear(4*dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, h, w):
        x = self.fc1(x)
        b, _, c = x.shape
        x = self.depthwise(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
        return self.drop(self.fc2(self.drop(F.gelu(x))))


class FrequencyAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout, cross=False):
        super().__init__()
        self.heads, self.dropout, self.cross = heads, dropout, cross
        self.norm1 = nn.LayerNorm(dim)
        if cross:
            self.context_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2*dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SpatialFeedForward(dim, dropout)

    def forward(self, x, h, w, context=None):
        query = self.norm1(x)
        context = self.context_norm(context) if self.cross else query
        b, n, c = query.shape
        q = self.q(query).reshape(b, n, self.heads, c//self.heads).transpose(1, 2)
        k, v = self.kv(context).reshape(b, -1, 2, self.heads, c//self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0)
        y = y.transpose(1, 2).reshape(b, n, c)
        x = x + self.drop(self.proj(y))
        return x + self.ffn(self.norm2(x), h, w)


class FrequencyFusion(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.ms_split, self.sar_split = FixedFrequencySplit(dim), FixedFrequencySplit(dim)
        self.ms_enhance, self.sar_enhance = FrequencySpatialEnhancement(), FrequencySpatialEnhancement()
        self.low_projection = nn.Conv2d(2*dim, dim, 1)
        self.low_self = FrequencyAttentionBlock(dim, heads, dropout)
        self.ms_cross = FrequencyAttentionBlock(dim, heads, dropout, cross=True)
        self.sar_cross = FrequencyAttentionBlock(dim, heads, dropout, cross=True)
        # BN subtracts a spatially constant bias in training. Do not optimize
        # that redundant near-zero-gradient parameter (Adam amplifies FP32 noise).
        self.output_conv = nn.Conv2d(2*dim, dim, 3, padding=1, bias=False)
        self.output_bn = nn.BatchNorm2d(dim)
        self.output_projection = nn.Conv2d(dim, dim, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0]*module.kernel_size[1]*module.out_channels/module.groups
            nn.init.normal_(module.weight, std=math.sqrt(2/fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ms, sar, h, w):
        b, _, c = ms.shape
        ms = ms.transpose(1, 2).reshape(b, c, h, w)
        sar = sar.transpose(1, 2).reshape(b, c, h, w)
        ms_high, ms_low = self.ms_enhance(ms, *self.ms_split(ms))
        sar_high, sar_low = self.sar_enhance(sar, *self.sar_split(sar))
        low = self.low_self(self.low_projection(torch.cat((ms_low, sar_low), 1)).flatten(2).transpose(1, 2), h, w)
        ms_high = self.ms_cross(low, h, w, ms_high.flatten(2).transpose(1, 2))
        sar_high = self.sar_cross(low, h, w, sar_high.flatten(2).transpose(1, 2))
        fused = torch.cat((ms_high, sar_high), -1).transpose(1, 2).reshape(b, 2*c, h, w)
        fused = self.output_conv(fused)
        if self.training and b*h*w == 1:
            fused = F.batch_norm(fused, self.output_bn.running_mean, self.output_bn.running_var,
                                 self.output_bn.weight, self.output_bn.bias, training=False, eps=self.output_bn.eps)
        else:
            fused = self.output_bn(fused)
        return self.output_projection(F.relu(fused)).flatten(2).transpose(1, 2)
