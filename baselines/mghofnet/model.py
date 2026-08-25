from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LSKBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(dim // 2, dim, 1)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.local_path = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1_red = self.conv1(attn1)
        attn2_red = self.conv2(attn2)
        spatial_gate = torch.sigmoid(self.conv_squeeze(attn1_red + attn2_red))
        lsk_feat = attn1 * spatial_gate + attn2 * (1 - spatial_gate)
        lsk_out = self.conv3(lsk_feat)
        local_out = self.local_path(x)
        return lsk_out + self.alpha * local_out + residual


class ModernFeedForward(nn.Module):
    def __init__(self, dim: int, expansion_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion_ratio, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class GatedBiCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_hsi = nn.Linear(dim, dim)
        self.kv_hsi = nn.Linear(dim, dim * 2)
        self.q_lidar = nn.Linear(dim, dim)
        self.kv_lidar = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.gate_hsi = nn.Parameter(torch.zeros(1))
        self.gate_lidar = nn.Parameter(torch.zeros(1))

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = hsi.shape
        q_h = self.q_hsi(hsi).reshape(batch, tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        kv_l = self.kv_lidar(lidar).reshape(batch, tokens, 2, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        k_l, v_l = kv_l[0], kv_l[1]
        out_h = ((q_h @ k_l.transpose(-2, -1)) * self.scale).softmax(dim=-1) @ v_l
        out_h = out_h.transpose(1, 2).reshape(batch, tokens, channels)

        q_l = self.q_lidar(lidar).reshape(batch, tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        kv_h = self.kv_hsi(hsi).reshape(batch, tokens, 2, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        k_h, v_h = kv_h[0], kv_h[1]
        out_l = ((q_l @ k_h.transpose(-2, -1)) * self.scale).softmax(dim=-1) @ v_h
        out_l = out_l.transpose(1, 2).reshape(batch, tokens, channels)

        fused = self.proj(out_h + out_l)
        return hsi + torch.tanh(self.gate_hsi) * fused + lidar * torch.tanh(self.gate_lidar)


class SecondOrderPoolingHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.proj1 = nn.Conv2d(in_dim, hidden_dim, 1)
        self.proj2 = nn.Conv2d(in_dim, hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bi = self.proj1(x).mul(self.proj2(x))
        x_pool = x_bi.mean(dim=[2, 3])
        x_power = torch.sign(x_pool) * torch.sqrt(torch.abs(x_pool) + 1e-8)
        x_norm = F.normalize(x_power, p=2, dim=1)
        return self.classifier(self.dropout(x_norm))


class MGHOFNet(nn.Module):
    def __init__(
        self,
        in_ch_hsi: int = 10,
        in_ch_lidar: int = 4,
        num_classes: int = 8,
        emb_dim: int = 128,
        depth: int = 2,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.hsi_stem = nn.Sequential(nn.Conv2d(in_ch_hsi, emb_dim, 3, padding=1), nn.BatchNorm2d(emb_dim), nn.GELU())
        self.lidar_stem = nn.Sequential(nn.Conv2d(in_ch_lidar, emb_dim, 3, padding=1), nn.BatchNorm2d(emb_dim), nn.GELU())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.hsi_layers = nn.ModuleList(
            [nn.ModuleList([LSKBlock(emb_dim), ModernFeedForward(emb_dim, drop_path=dpr[i])]) for i in range(depth)]
        )
        self.lidar_layers = nn.ModuleList(
            [nn.ModuleList([LSKBlock(emb_dim), ModernFeedForward(emb_dim, drop_path=dpr[i])]) for i in range(depth)]
        )
        self.fusion_block = GatedBiCrossAttention(emb_dim)
        self.classifier_head = SecondOrderPoolingHead(in_dim=emb_dim, num_classes=num_classes, hidden_dim=256)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        x_h = self.hsi_stem(hsi)
        x_l = self.lidar_stem(lidar)
        for lsk, ffn in self.hsi_layers:
            x_h = ffn(lsk(x_h))
        for lsk, ffn in self.lidar_layers:
            x_l = ffn(lsk(x_l))
        batch, channels, height, width = x_h.shape
        x_h_flat = x_h.flatten(2).transpose(1, 2)
        x_l_flat = x_l.flatten(2).transpose(1, 2)
        x_fused = self.fusion_block(x_h_flat, x_l_flat).transpose(1, 2).reshape(batch, channels, height, width)
        return self.classifier_head(x_fused)
