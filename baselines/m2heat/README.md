# M2Heat（TCSVT 2026，热传导 + 跨频域融合）

本目录是 **第三方方法在本项目 MSI–SAR 数据上的基线适配**，不是原论文整网复现。

- 来源：[Weikan0425/M2Heat_HSI_LiDAR](https://github.com/Weikan0425/M2Heat_HSI_LiDAR)，commit `8fcdce85e92a5d8b92c0a86ca78f835113dd3e3f`；只读快照见 [`vendor/M2Heat-original/`](../../vendor/M2Heat-original)（未复制 LFS 数据集）。
- 论文：*Toward Interpretable Multimodal Fusion: Heat Conduction Modeling for Hyperspectral and LiDAR Joint Classification*, IEEE TCSVT 2026。
- 适配实现：[`baselines/_hsi_adapters/m2heat_patch.py`](../_hsi_adapters/m2heat_patch.py)，直接实例化 vendor 中未改动的 `M2Heat`。

## 接口

```python
M2Heat(num_classes=8, patch_size=11, spectral_channels=10, sar_channels=4)
# forward(x): x = [B, 1, C, H, W]（MSI 在前，SAR 在后）→ [B, num_classes] logits
```

## 与原仓库的差异

1. 原分类器末尾是 `softmax`；这里取 `log(prob)` 作为 logits 交给统一交叉熵训练协议（等价于 NLL，排序与评价不变，但概率数值不是原样输出）。
2. 辅助模态改为本项目的 SAR（1/4 通道）；原 demo 使用 1 通道 LiDAR/DSM，模型本身对辅助通道数不敏感。
3. patch 11（原 demo 默认 13，可配置）；FVE（频率值嵌入）参数随 patch 尺寸在 `__init__` 中创建。
4. 无预训练。

## 已知限制

- **未运行验证**：本次只做语法检查，未在 GPU 上做前向/训练 smoke test，无精度数字。
- 原仓库**未见明确 LICENSE**；仅作研究对照。
- 原论文 Trento/Houston2013/Augsburg 精度不能直接迁移。

## 配置与运行

```powershell
python scripts/train_eval_m2heat.py --config configs/m2heat_yrd_fs50.yaml
```

配置：`configs/m2heat_yrd_fs50.yaml`、`configs/m2heat_yrd2509new_fs50.yaml`、`configs/m2heat_grss07_custom.yaml`。
