"""MFT (TGRS 2023) patch-classification adaptation for the MSI-SAR project.

Vendored source: ``vendor/hsi_lidar/MFT-original`` (AnkurDeria/MFT,
commit de214e512ff1a8f6aefdf27fc2cb60c75e0f559a).  The released code exists only
as ``Model/MFT.ipynb``; this module re-implements the model classes (HetConv,
MCrossAttention, Mlp, Block, TransformerEncoder, MFT) faithfully from that
notebook, with the minimal changes below.

Deviations:
- the spectral 3D-conv kernel in the notebook is fixed at depth 9 with
  ``8*(NC-8)`` expected input channels; here the depth is
  ``min(9, NC-1)`` so 6/10-band MSI inputs remain valid (10 bands reproduces
  the original depth-9 behaviour, 6 bands uses depth 5);
- ``patchsize`` is an attribute instead of a notebook global;
- no pretraining, no LiDAR-only mode, patch-level classifier.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Dropout, LayerNorm, Linear


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,
                 bias=None, p=64, g=64):
        super().__init__()
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=g,
                             padding=kernel_size // 3, stride=stride)
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, stride=stride)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, num_tokens, channels = x.shape
        q = self.wq(x[:, 0:1, ...].reshape(batch, 1, self.num_heads, channels // self.num_heads)).permute(0, 2, 1, 3)
        k = self.wk(x.reshape(batch, num_tokens, self.num_heads, channels // self.num_heads)).permute(0, 2, 1, 3)
        v = self.wv(x.reshape(batch, num_tokens, self.num_heads, channels // self.num_heads)).permute(0, 2, 1, 3)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
        out = out.reshape(batch, 1, channels * self.num_heads)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.dropout(self.act_fn(self.fc1(x)))
        return self.dropout(self.fc2(x))


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim)
        self.attn = MCrossAttention(dim=dim)

    def forward(self, x):
        x = self.attn(self.attention_norm(x)) + x
        return self.ffn(self.ffn_norm(x)) + x


class TransformerEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.ModuleList([copy.deepcopy(Block(dim)) for _ in range(2)])
        self.encoder_norm = LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        for block in self.layer:
            x = block(x)
        return self.encoder_norm(x)[:, 0]


class MFTPatch(nn.Module):
    """MFT with a batch-agnostic patch size and band-adaptive spectral stem."""

    def __init__(self, ms_channels, sar_channels, num_classes, patch_size=11, dim=64):
        super().__init__()
        self.patch_size = int(patch_size)
        self.ms_channels = int(ms_channels)
        self.sar_channels = int(sar_channels)
        feature_dim = int(dim)
        spectral_kernel = min(9, max(1, self.ms_channels - 1))
        self.spectral_kernel = spectral_kernel
        self.spectral_out = self.ms_channels - spectral_kernel + 1
        if feature_dim % 8 != 0:
            raise ValueError('dim must be divisible by 8 for HetConv groups')
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (spectral_kernel, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(8), nn.ReLU())
        het_in = 8 * self.spectral_out
        out_channels = feature_dim * 4
        # The released code assumes a 144-band HSI where 8*(NC-8) is divisible by
        # the feature dim. With 6/10 MSI bands the reduced spectral size is 16, so
        # pick the largest valid group count instead of the hard-coded branch.
        groups = math.gcd(het_in, out_channels)
        if groups < 1:
            groups = 1
        self.conv6 = nn.Sequential(
            HetConv(het_in, out_channels, p=1, g=groups),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.lidar_conv = nn.Sequential(
            nn.Conv2d(self.sar_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU())
        self.ca = TransformerEncoder(feature_dim * 4)
        self.out = nn.Linear(feature_dim * 4, num_classes)
        self.position_embeddings = nn.Parameter(torch.randn(1, 4 + 1, feature_dim * 4))
        self.dropout = nn.Dropout(0.1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.normal_(self.out.bias, std=1e-6)
        self.token_wA = nn.Parameter(torch.empty(1, 4, 64))
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64))
        self.token_wA_L = nn.Parameter(torch.empty(1, 1, 64))
        self.token_wV_L = nn.Parameter(torch.empty(1, 64, 64))
        for parameter in (self.token_wA, self.token_wV, self.token_wA_L, self.token_wV_L):
            nn.init.xavier_normal_(parameter)

    def forward(self, ms, sar, labels=None):
        batch = ms.shape[0]
        spectral = ms.reshape(batch, -1, self.patch_size, self.patch_size).unsqueeze(1)
        spectral = self.conv5(spectral)
        spectral = self.conv6(spectral.reshape(batch, -1, self.patch_size, self.patch_size))
        auxiliary = self.lidar_conv(sar)
        auxiliary = auxiliary.reshape(batch, -1, self.patch_size ** 2).transpose(-1, -2)

        wa_l = rearrange(self.token_wA_L.expand(batch, -1, -1), 'b h w -> b w h')
        attn_l = rearrange(torch.einsum('bij,bjk->bik', auxiliary, wa_l), 'b h w -> b w h').softmax(dim=-1)
        vv_l = torch.einsum('bij,bjk->bik', auxiliary, self.token_wV_L.expand(batch, -1, -1))
        auxiliary = torch.einsum('bij,bjk->bik', attn_l, vv_l)

        tokens = spectral.flatten(2).transpose(-1, -2)
        wa = rearrange(self.token_wA.expand(batch, -1, -1), 'b h w -> b w h')
        attn = rearrange(torch.einsum('bij,bjk->bik', tokens, wa), 'b h w -> b w h').softmax(dim=-1)
        vv = torch.einsum('bij,bjk->bik', tokens, self.token_wV.expand(batch, -1, -1))
        tokens = torch.einsum('bij,bjk->bik', attn, vv)

        embeddings = torch.cat((auxiliary, tokens), dim=1) + self.position_embeddings
        encoded = self.ca(self.dropout(embeddings))
        logits = self.out(encoded.reshape(encoded.shape[0], -1))
        result = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result['losses'] = {'total': loss, 'ce': loss}
        return result
