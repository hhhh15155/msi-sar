from __future__ import annotations

import torch
from einops import rearrange
from torch import einsum, nn
import torch.nn.init as init


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(module.weight)


class PAMModule(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.BatchNorm2d(128), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, width * height)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value_conv(x).view(batch, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, channels, height, width)
        out = self.gamma * out + x
        return self.conv(out) + out


class CAMModule(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.BatchNorm2d(128), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        query = x.view(batch, channels, -1)
        key = x.view(batch, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        attention = self.softmax(torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy)
        value = x.view(batch, channels, -1)
        out = torch.bmm(attention, value).view(batch, channels, height, width)
        out = self.gamma * out + x
        return self.conv(out) + out


class AGA(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 8, dropout: float = 0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(inner_dim, dim, bias=False)
        self.to_v = nn.Linear(inner_dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.Dropout(dropout))
        self.pool = nn.AdaptiveAvgPool2d(output_size=(49, dim))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        heads = self.heads
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x2)
        agent_tokens = self.pool(q)
        agent_tokens = rearrange(agent_tokens, "b n (h d) -> b h n d", h=heads)
        q1 = rearrange(q, "b n (h d) -> b h n d", h=heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=heads)
        v1 = rearrange(v, "b n (h d) -> b h n d", h=heads)
        attn0 = self.dropout(self.attend(einsum("b h i d, b h j d -> b h i j", agent_tokens, k) * self.scale))
        out0 = einsum("b h i j, b h j d -> b h i d", attn0, v1)
        attn1 = self.dropout(self.attend(einsum("b h i d, b h j d -> b h i j", q1, agent_tokens) * self.scale))
        out1 = einsum("b h i j, b h j d -> b h i d", attn1, out0)
        out1 = rearrange(out1, "b h n d -> b n (h d)")
        return self.to_out(torch.cat((out1, x1), dim=-1))


class Mlp(nn.Module):
    def __init__(self, features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(features, features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(features, features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DFINet(nn.Module):
    def __init__(self, num_classes: int = 8, patch_size: int = 7, spectral_channels: int = 10, sar_channels: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.spectral_channels = spectral_channels
        self.sar_channels = sar_channels
        self.num_channels = spectral_channels + sar_channels
        self.spectral_branch = nn.Sequential(nn.Conv3d(spectral_channels, 64, kernel_size=1), nn.BatchNorm3d(64), nn.ReLU())
        self.sar_branch = nn.Sequential(nn.Conv2d(sar_channels, 64, kernel_size=5, padding=2), nn.BatchNorm2d(64), nn.ReLU())
        self.cross1 = AGA(64)
        self.cross2 = AGA(64)
        self.spe = CAMModule(128)
        self.spa = PAMModule(128)
        self.mlp = Mlp(128)
        self.fc = nn.Linear(128, num_classes)
        self.norm = nn.LayerNorm(num_classes)
        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(dim=1)
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {x.shape[1]}")
        height, width = x.shape[-2:]
        spectral = x[:, : self.spectral_channels].unsqueeze(dim=-1)
        sar = x[:, self.spectral_channels : self.spectral_channels + self.sar_channels]
        x1 = rearrange(self.spectral_branch(spectral), "b c h w y -> b (h w) (c y)")
        x2 = self.sar_branch(sar).flatten(2).transpose(-1, -2)
        x = self.cross1(x1, x2) + self.cross2(x2, x1)
        batch, _, channels = x.shape
        x = x.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        x = self.spe(x) + self.spa(x)
        x = self.mlp(x.flatten(2).transpose(-1, -2)).mean(dim=1)
        return self.norm(self.fc(x))
