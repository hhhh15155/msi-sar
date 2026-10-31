# DFFNet（GRSL 2025，HSI–LiDAR/HSI–SAR 频域融合）

> **2026-10-31 当前流程**：所有后续运行统一使用 MGHOFNet 训练协议 `mghofnet-v1`，公共实现为 `baselines/{protocol,data,training,prediction}.py`；优化、补零、增强和随机性设置以[当前实验协议](../../docs/experiments/README.md)为准。已完成实验仍属于各自保存的旧配置，本文此前的验证/运行记录不能视为新协议精度。此次只改代码，不启动正式实验；历史重评估按原填充处理。

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

- **CPU 合成数据验证通过，未完成正式实验**：Windows 本机已有 PyTorch 2.5.1 环境验证了 MSI/SAR `6/1`、`10/4` 和 patch `11/13` 的前向、反向、有限梯度、优化器参数覆盖与 checkpoint 重载一致性；未使用真实数据、未做 GPU 或精度验证。
- 原仓库**未见明确 LICENSE**；公开可读不等于可复用，仅作研究对照。
- 原论文的 Berlin/Houston 等数据集精度不能直接迁移到本项目数据。

## 配置与运行

```powershell
python scripts/train_eval_dffnet.py --config configs/dffnet_yrd_fs50.yaml
```

主实验配置：`configs/dffnet_yrd_fs50.yaml`、`configs/dffnet_yrd2509new_fs50.yaml`、`configs/dffnet_grss07_custom.yaml`。已对齐当前主实验的训练数量、5 seeds `202201–202205`、patch 11、200 epochs、train128／test1024、FP32 与统一 AdamW/余弦调度；按最高测试 OA 的运行出图，保留全部运行及汇总指标。详细参数见 [配置索引](../../configs/README.md#新适配基线的主实验配置)。配置／出图流程回归检查通过；正式实验尚未运行。

## 测试与两种出图

`eval_dffnet.py --run <目录>` 只对保存的测试划分重评估。`infer_dffnet.py --run <目录>` 默认选测试 OA 最高的最终 checkpoint，推理全部有标注位置（包括训练位置）出图，不重新计分；图及 `labeled_map_info.json` 与测试指标分开保存。只有用户需要时，加 `--mode full` 才运行全图推理。上述脚本均位于 `scripts/`，默认训练入口仅生成标注图，`infer.save_full_map: false`。
