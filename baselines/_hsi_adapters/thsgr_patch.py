"""THSGR core, based on jqyang22/THSGR dd2a0ef (Apache-2.0).

Modified for low-channel MSI, multichannel SAR and configurable small patches.
Unused upstream branches are omitted. Mean-forward is training-only so that
test and map predictions are independent of batch composition. See README.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MeanForward(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return (x + x.mean(dim=0, keepdim=True)) * 0.5
        return x


class DynamicGraphConvolution(nn.Module):
    """The upstream dynamic path; both node and correlation outputs are used."""
    def __init__(self, features: int = 64):
        super().__init__()
        self.dynamic_weight = nn.Conv1d(features, features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = torch.sigmoid(torch.bmm(x.transpose(1, 2), x))
        channels = torch.softmax(torch.bmm(x, x.transpose(1, 2)), dim=-1)
        correlation = torch.bmm(channels, x)
        graph = F.leaky_relu(torch.matmul(x * self.dynamic_weight(x), nodes), 0.2)
        return graph, correlation


class ConvMod(nn.Module):
    """Cross-modal convolution modulation of the original 64x64 correlations."""
    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(65, 65, 1), nn.GELU(),
                               nn.Conv2d(65, 65, 3, padding=1, groups=65))
        self.v = nn.Conv2d(65, 65, 1)
        self.proj = nn.Conv2d(65, 65, 1)

    def forward(self, x: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.reshape(b, 65, 8, 8)
        auxiliary = auxiliary.reshape(b, 65, 8, 8)
        return self.proj(self.a(x) * self.v(auxiliary)).flatten(2)


class ModulationBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.modulation = ConvMod()
        self.ff_norm = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 1024), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(1024, 64), nn.Dropout(0.1), MeanForward())

    def forward(self, x: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        x = self.modulation(self.norm(x), self.norm(auxiliary)) + x + auxiliary
        return x + self.ff(self.ff_norm(x))


class CorrelationTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(64, 64)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
        self.pos_embedding = nn.Parameter(torch.randn(1, 65, 64))
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([ModulationBlock() for _ in range(5)])

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        token = self.cls_token.expand(x.shape[0], -1, -1)
        return self.dropout(torch.cat((token, x), dim=1) + self.pos_embedding)

    def forward(self, x: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        x, auxiliary = self.embed(x), self.embed(auxiliary)
        for block in self.blocks:
            x = block(x, auxiliary)
        return x[:, 0]


class THSGRPatch(nn.Module):
    def __init__(self, ms_channels: int, sar_channels: int, num_classes: int):
        super().__init__()
        # Keep the original 7/5/3 depth kernels where possible. Never pad,
        # duplicate or interpolate spectral bands to manufacture PCA32 inputs.
        remaining = ms_channels
        kernels = []
        for preferred in (7, 5, 3):
            depth = min(preferred, remaining)
            if depth % 2 == 0:
                depth -= 1
            kernels.append(depth)
            remaining -= depth - 1
        self.spectral_kernels = tuple(kernels)
        self.spectral = nn.ModuleList([
            nn.Conv3d(1, 8, (kernels[0], 3, 3)),
            nn.Conv3d(8, 16, (kernels[1], 3, 3)),
            nn.Conv3d(16, 32, (kernels[2], 3, 3)),
        ])
        self.spectral_norm = nn.ModuleList([nn.BatchNorm3d(n) for n in (8, 16, 32)])
        self.ms_projection = nn.Conv2d(32 * remaining, 64, 3)
        self.sar_conv1 = nn.Conv2d(sar_channels, 32, 3)
        self.sar_norm1 = nn.BatchNorm2d(32)
        self.sar_conv2 = nn.Conv2d(32, 64, 3)
        self.sar_norm2 = nn.BatchNorm2d(64)
        # Original code shares the saliency encoder and GCN across modalities.
        self.saliency = nn.Conv2d(64, num_classes, 1, bias=False)
        self.transform = nn.Conv2d(64, 64, 1)
        self.graph = DynamicGraphConvolution()
        self.graph_projection = nn.Linear(64 * num_classes, 64)
        self.ms_correlation = nn.Linear(64 * num_classes, 64 * 64)
        self.sar_correlation = nn.Linear(num_classes, 64)
        self.transformer = CorrelationTransformer()
        self.classifier = nn.Sequential(nn.Linear(64, 256), nn.LeakyReLU(0.01),
                                        nn.BatchNorm1d(256), nn.Dropout(0.5),
                                        nn.Linear(256, num_classes))

    def encode_graph(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.sigmoid(self.saliency(x).flatten(2)).transpose(1, 2)
        nodes = torch.matmul(self.transform(x).flatten(2), mask)
        return self.graph(nodes)

    def forward(self, ms: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        ms = ms.unsqueeze(1)
        for conv, norm in zip(self.spectral, self.spectral_norm):
            ms = norm(F.leaky_relu(conv(ms)))
        ms = F.leaky_relu(self.ms_projection(ms.flatten(1, 2)))
        graph, ms_corr = self.encode_graph(ms)
        graph = self.graph_projection(graph.flatten(1))
        sar = self.sar_norm1(F.leaky_relu(self.sar_conv1(sar)))
        sar = self.sar_norm2(F.leaky_relu(self.sar_conv2(sar)))
        _, sar_corr = self.encode_graph(sar)
        sar_corr = self.sar_correlation(sar_corr)
        ms_corr = self.ms_correlation(ms_corr.flatten(1)).reshape(-1, 64, 64)
        fused = self.transformer(ms_corr, sar_corr)
        # The author's executable entry uses w=0.9, not a tuned parameter here.
        return self.classifier(0.1 * fused + 0.9 * graph)
