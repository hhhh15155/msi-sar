# DCMNet（TGRS 2025，动态跨模态特征交互）

本目录是 **第三方方法在本项目 MSI–SAR 数据上的基线适配**，不是原论文整网复现。

- 来源：[oucailab/DCMNet](https://github.com/oucailab/DCMNet)，commit `b88ead6db8bc8a5669290e2e6c3af7d062158171`；只读快照见 [`vendor/DCMNet-original/`](../../vendor/DCMNet-original)。
- 论文：*Dynamic Cross-Modal Feature Interaction Network for Hyperspectral and LiDAR Data Classification*, IEEE TGRS 2025。
- 适配实现：`model.py` 提供项目统一接口；共享实现见 [`baselines/_hsi_adapters/dcmnet_patch.py`](../_hsi_adapters/dcmnet_patch.py)。

## 接口

```python
DCMNet(num_classes=8, patch_size=11, spectral_channels=10, sar_channels=4)
# forward(x): x = [B, 1, C, H, W]，C = spectral_channels + sar_channels（MSI 在前，SAR 在后）
# 返回 [B, num_classes] logits
```

## 保留与适配

**原样 import 的 vendor 模块**：`net/Cells.py`、`net/Feature_Interaction.py`、`net/Router.py`、`net/routing_modules.py`、`net/classify.py`（动态路由、跨模态交互 Cell、分类头均未修改）。

**重写的部分**（原实现按数据集硬编码，无法直接改通道）：

1. **光谱编码器**：原 `hsi_feature_extractor_*.py` 固定对应 PCA-30／20／63 波段的 3D 核（如 9/7/5）。这里采用深度为 1 的谱核，把谱混合交给原设计的 2D 投影层：对 10 通道 MSI，形状与公开的 Trento 分支（C=10：80/160/320 输入）完全一致；6 通道 GRSS07 按同一规则缩减（48/96/192）。
2. **SAR 编码器**：沿用原 2D LiDAR stem 的层数与核，仅把首层输入通道改为实际 SAR 通道数（1／4），不把 LiDAR 高度语义套到 SAR，也不插值复制波段。
3. **参数注入**：原 `parameter.py` 在 import 时加载 `.mat` 数据并写死 `hsi_windowSize`、`size_SA` 等常量；适配层用一个参数 stub 模块替代，按 `patch_size` 动态设置 `hsi_windowSize = patch_size - 6`、`size_SA = (patch_size-6)²`。
4. **patch 限制**：路由 Cell 要求方形特征图边长 `patch_size - 6`，因此入口校验 **patch 必须为奇数且 ≥9**；主实验 patch 11（5×5）、13（7×7）均在合成测试范围内。
5. 无预训练；分类头输出保持原 `1×1 conv → 512 → num_classes`。

## 已知限制

- **合成数据验证通过、未完成正式实验**：目前只有静态检查与合成数据前向／反向验证；真实数据精度未运行。
- 原仓库**未见明确 LICENSE**；仅作研究对照。
- 原论文 Houston2013/Houston2018/Trento 精度不能直接迁移到本项目数据。
- 谱核改为深度 1 是与原文的已知差异；未做 PCA 30 波段，避免对 6/10 通道 MSI 做无依据的波段复制。

## 运行

```bash
python scripts/train_eval_dcmnet.py --config configs/dcmnet_yrd_fs50.yaml
python scripts/eval_dcmnet.py --run runs_fewshot/fs50/dcmnet/yrd/run_001
python scripts/infer_dcmnet.py --run runs_fewshot/fs50/dcmnet/yrd/run_001 --mode labeled
```

配置：`configs/dcmnet_{yrd_fs50,yrd2509new_fs50,grss07_custom}.yaml`（5 seeds、patch 11、200 epochs、Adam lr 1e-3／wd 1e-4、FP32、无预训练；`eval.run_index: best` 选测试 OA 最高者出图，全图默认关闭）。
