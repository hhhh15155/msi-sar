# MFT（TGRS 2023，多模态融合 Transformer / mCrossPA）

本目录是 **第三方方法在本项目 MSI–SAR 数据上的基线适配**，不是原论文整网复现。

- 来源：[AnkurDeria/MFT](https://github.com/AnkurDeria/MFT)，commit `de214e512ff1a8f6aefdf27fc2cb60c75e0f559a`；只读快照见 [`vendor/MFT-original/`](../../vendor/MFT-original)。
- 论文：*Multimodal Fusion Transformer for Remote Sensing Image Classification*, IEEE TGRS 2023。
- 原仓库只发布 **Jupyter notebook**（`Model/MFT.ipynb`），没有独立 Python 模型文件；本适配把其中的 `HetConv / MCrossAttention / Mlp / Block / TransformerEncoder / MFT` 重新实现为 [`baselines/hsi_lidar/mft_patch.py`](../hsi_lidar/mft_patch.py)。

## 接口

```python
MFT(num_classes=8, patch_size=11, spectral_channels=10, sar_channels=4)
# forward(x): x = [B, 1, C, H, W]（MSI 在前，SAR 在后）→ [B, num_classes] logits
```

## 与原 notebook 的差异

1. **谱核自适应**：notebook 的 `Conv3d(1, 8, (9,3,3))` 假定 144 波段 HSI；这里核深度取 `min(9, C-1)`（10 通道→9，6 通道→5），并相应调整 `HetConv` 的输入通道与分组数。
2. `patchsize` 从 notebook 的全局变量改为实例属性。
3. 只保留分类路径，去掉 notebook 中的训练/绘图/数据加载 cell；未做任何预训练。
4. 去掉了 `HSIOnly` 分支相关的实验逻辑（本项目恒为双模态）。

## 已知限制

- **未运行验证**：本次只做语法检查，未在 GPU 上做前向/训练 smoke test，无精度数字。
- 原仓库**未见明确 LICENSE**；仅作研究对照。
- 原 notebook 的分组卷积/Token 数与新输入通道的对应关系已重算，但尚未用真实数据验证梯度与精度。
- 原论文 Houston2013/Trento/MUUFL 精度不能直接迁移。

## 配置与运行

```powershell
python scripts/train_eval_mft.py --config configs/mft_yrd_fs50.yaml
```

配置：`configs/mft_yrd_fs50.yaml`、`configs/mft_yrd2509new_fs50.yaml`、`configs/mft_grss07_custom.yaml`。
