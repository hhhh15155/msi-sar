# DFFNet（GRSL 2025，HSI–LiDAR/HSI–SAR 频域融合）

本目录是 **第三方方法在本项目 MSI–SAR 数据上的基线适配**，不是原论文整网复现，也不属于项目自研方法。

- 来源：[oucailab/DFFNet](https://github.com/oucailab/DFFNet)，commit `a9beb2b2b43b452cfca364160822f0dee9b34462`；只读快照见 [`vendor/DFFNet-original/`](../../vendor/DFFNet-original)。
- 论文：*Dynamic Frequency Feature Fusion Network for Multi-Source Remote Sensing Data Classification*, IEEE GRSL 2025。
- 代码位置：`model.py` 提供项目统一接口；真正的适配实现在 [`baselines/_hsi_adapters/dffnet_patch.py`](../_hsi_adapters/dffnet_patch.py)，它直接导入 vendor 中未改动的 `DynamicFilterBlock` 与 `SpectralSpatialAdaptiveFusionBlock`。

## 接口

```python
DFFNet(num_classes=8, patch_size=11, spectral_channels=10, sar_channels=4)
# forward(x): x = [B, 1, C, H, W]，C = spectral_channels + sar_channels（MSI 在前，SAR 在后）
# 返回 [B, num_classes] logits（不含 softmax）
```

## 与原仓库的差异（公平比较时必须记录）

1. 原仓库的 `Net` 从 `dataset_info.yaml` 读取数据集尺寸，这里改为显式通道/类别/patch 参数，用于项目统一数据管线。
2. 输入不做仓库内的 PCA 30 维步骤；项目数据已逐通道归一化，MSI 的 6/10 通道直接进入原 3D 卷积谱降维（谱核由通道数推导：10 通道时核 7、6 通道时核 3）。
3. 固定 **2 个 DFFM stage**（论文报告 2 个为佳；仓库 YAML 默认是 3）。
4. 修复了原实现的参数注册缺陷：`filter_bases_F` 原为首次 forward 惰性创建，晚于优化器构建；这里在 `__init__` 中创建，使所有可训练参数进入优化器。
5. 分类头保留原 `flatten → 256 → num_classes` 结构；无预训练；patch 11。

## 已知限制

- **未运行验证**：本地无 GPU/PyTorch 环境，本次只做了语法与结构检查；尚未在服务器上做前向/训练 smoke test，也未产生任何精度数字。
- 原仓库**未见明确 LICENSE**；公开可读不等于可复用，仅作研究对照。
- 原论文的 Berlin/Houston 等数据集精度不能直接迁移到本项目数据。

## 配置与运行

```powershell
python scripts/train_eval_dffnet.py --config configs/dffnet_yrd_fs50.yaml
```

配置模板：`configs/dffnet_yrd_fs50.yaml`、`configs/dffnet_yrd2509new_fs50.yaml`、`configs/dffnet_grss07_custom.yaml`（与仓库既有 baseline 配置同格式：每类固定训练样本数、5 seeds、200 epochs、Adam）。
