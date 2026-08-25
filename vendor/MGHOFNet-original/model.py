import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# 0. 辅助模块: DropPath (随机深度)
# ============================================================================
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


# ============================================================================
# 1. 核心改进: Detail-Preserving LSK Block (细节保留型 LSK)
# ============================================================================
class LSKBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # --- 分支 A: Large Spatial Kernel (看大局) ---
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(dim // 2, dim, 1)
        self.conv3 = nn.Conv2d(dim, dim, 1)

        # --- 分支 B: Local Detail Branch (看细节) ---
        self.local_path = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

        # 学习一个权重来平衡 "大局观" 和 "细节控"
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        clone = x

        # --- Path A: LSK Forward ---
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1_red = self.conv1(attn1)
        attn2_red = self.conv2(attn2)

        attn_cat = torch.cat([attn1_red, attn2_red], dim=1)
        spatial_gate = torch.sigmoid(self.conv_squeeze(attn1_red + attn2_red))

        lsk_feat = attn1 * spatial_gate + attn2 * (1 - spatial_gate)
        lsk_out = self.conv3(lsk_feat)

        # --- Path B: Local Detail Forward ---
        local_out = self.local_path(x)

        # --- Fusion ---
        out = lsk_out + self.alpha * local_out

        return out + clone


# ============================================================================
# 2. 基础组件: Modern FFN
# ============================================================================
class ModernFeedForward(nn.Module):
    def __init__(self, dim, expansion_ratio=4, drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion_ratio, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        x = input + self.drop_path(x)
        return x


# ============================================================================
# 3. 核心组件: Gated Bi-CrossAttention
# ============================================================================
class GatedBiCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
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

    def forward(self, hsi, lidar):
        B, N, C = hsi.shape

        # Path A: HSI queries LiDAR
        q_h = self.q_hsi(hsi).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_l = self.kv_lidar(lidar).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_l, v_l = kv_l[0], kv_l[1]
        attn_h = (q_h @ k_l.transpose(-2, -1)) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        out_h = (attn_h @ v_l).transpose(1, 2).reshape(B, N, C)

        # Path B: LiDAR queries HSI
        q_l = self.q_lidar(lidar).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_h = self.kv_hsi(hsi).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_h, v_h = kv_h[0], kv_h[1]
        attn_l = (q_l @ k_h.transpose(-2, -1)) * self.scale
        attn_l = attn_l.softmax(dim=-1)
        out_l = (attn_l @ v_h).transpose(1, 2).reshape(B, N, C)

        fused = self.proj(out_h + out_l)

        gate_f = torch.tanh(self.gate_hsi)
        gate_l = torch.tanh(self.gate_lidar)

        return hsi + gate_f * fused + lidar * gate_l


# ============================================================================
# 4. 新增组件: Second-Order Pooling Head (类似 FHFL)
#    作用: 替代 GAP，捕捉更丰富的二阶纹理统计信息，提高小样本分类性能
# ============================================================================
class SecondOrderPoolingHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=256):
        super().__init__()
        # 1. 两个独立的投影层 (对应 FHFL 中的 Linear_dataproj_k 和 Linear_dataproj2_k)
        # 使用 1x1 卷积代替 Linear，方便处理 [B, C, H, W]
        self.proj1 = nn.Conv2d(in_dim, hidden_dim, 1)
        self.proj2 = nn.Conv2d(in_dim, hidden_dim, 1)

        # 2. 最终分类层
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]

        # --- Step 1: Bilinear Feature Extraction ---
        x1 = self.proj1(x)  # [B, hidden_dim, H, W]
        x2 = self.proj2(x)  # [B, hidden_dim, H, W]

        # 逐元素相乘 (Hadamard Product) 模拟双线性交互
        x_bi = x1.mul(x2)  # [B, hidden_dim, H, W]

        # --- Step 2: Global Spatial Pooling (二阶统计量) ---
        # 对空间维度 (H, W) 求均值
        x_pool = x_bi.mean(dim=[2, 3])  # [B, hidden_dim]

        # --- Step 3: Power Normalization (FHFL 关键步骤) ---
        # Signed Square Root: sign(x) * sqrt(|x|)
        # 这能使特征分布更接近高斯分布，极大地帮助分类器训练
        x_power = torch.sign(x_pool) * torch.sqrt(torch.abs(x_pool) + 1e-8)

        # --- Step 4: L2 Normalization ---
        x_norm = F.normalize(x_power, p=2, dim=1)

        # --- Step 5: Classification ---
        logits = self.classifier(self.dropout(x_norm))
        return logits


# ============================================================================
# 5. 主模型: SS-LSK-Net (Updated with Second-Order Head)
# ============================================================================
class TTTFusionNet(nn.Module):
    def __init__(self,
                 in_ch_hsi=30,
                 in_ch_lidar=1,
                 num_classes=11,
                 emb_dim=128,
                 depth=2,
                 drop_path_rate=0.2):
        super().__init__()

        print(f"Initializing SS-LSK-Net with Second-Order Pooling Head...")

        # --- Stage 1: Stem ---
        self.hsi_stem = nn.Sequential(
            nn.Conv2d(in_ch_hsi, emb_dim, 3, padding=1),
            nn.BatchNorm2d(emb_dim),
            nn.GELU()
        )
        self.lidar_stem = nn.Sequential(
            nn.Conv2d(in_ch_lidar, emb_dim, 3, padding=1),
            nn.BatchNorm2d(emb_dim),
            nn.GELU()
        )

        # --- Stage 2: Backbone ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.hsi_layers = nn.ModuleList([
            nn.ModuleList([
                LSKBlock(emb_dim),
                ModernFeedForward(emb_dim, drop_path=dpr[i])
            ]) for i in range(depth)
        ])

        self.lidar_layers = nn.ModuleList([
            nn.ModuleList([
                LSKBlock(emb_dim),
                ModernFeedForward(emb_dim, drop_path=dpr[i])
            ]) for i in range(depth)
        ])

        # --- Stage 3: Fusion ---
        self.fusion_block = GatedBiCrossAttention(emb_dim)

        # --- Stage 4: Second-Order Classifier (New!) ---
        # 移除原来的 GAP 和 Sequential Classifier
        # 使用 embedding 维度扩展到 256 进行双线性交互
        self.classifier_head = SecondOrderPoolingHead(in_dim=emb_dim, num_classes=num_classes, hidden_dim=256)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, hsi, lidar):
        x_h = self.hsi_stem(hsi)
        x_l = self.lidar_stem(lidar)

        for lsk, ffn in self.hsi_layers:
            x_h = lsk(x_h)
            x_h = ffn(x_h)

        for lsk, ffn in self.lidar_layers:
            x_l = lsk(x_l)
            x_l = ffn(x_l)

        # Fusion: Flatten -> Attend -> Reshape
        B, C, H, W = x_h.shape
        x_h_flat = x_h.flatten(2).transpose(1, 2)
        x_l_flat = x_l.flatten(2).transpose(1, 2)

        x_fused_flat = self.fusion_block(x_h_flat, x_l_flat)
        x_fused = x_fused_flat.transpose(1, 2).reshape(B, C, H, W)

        # Classification using Second-Order Head
        logits = self.classifier_head(x_fused)

        return logits