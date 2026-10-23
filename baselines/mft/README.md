# MFT（TGRS 2023，多模态融合 Transformer / mCrossPA）

本目录是 **第三方方法在本项目 MSI–SAR 数据上的基线适配**，不是原论文整网复现。

- 来源：[AnkurDeria/MFT](https://github.com/AnkurDeria/MFT)，commit `de214e512ff1a8f6aefdf27fc2cb60c75e0f559a`；只读快照见 [`vendor/MFT-original/`](../../vendor/MFT-original)。
- 论文：*Multimodal Fusion Transformer for Remote Sensing Image Classification*, IEEE TGRS 2023。
- 原仓库只发布 **Jupyter notebook**（`Model/MFT.ipynb`），没有独立 Python 模型文件；本适配把其中的 `HetConv / MCrossAttention / Mlp / Block / TransformerEncoder / MFT` 重新实现为 [`baselines/_hsi_adapters/mft_patch.py`](../_hsi_adapters/mft_patch.py)。

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
5. 使用原模型 FM=16，保持两路 token、位置编码、Transformer 为64维；原先误用 FM=64 的维度错误已修复。仅用于转置的 einops 调用改为等价 Tensor.transpose，无需该额外依赖。

## 已知限制

- **CPU 合成数据验证通过，未完成正式实验**：Windows 本机已有 PyTorch 2.5.1 环境验证了 MSI/SAR `6/1`、`10/4` 和 patch `11/13` 的前向、反向、有限梯度、优化器参数覆盖与 checkpoint 重载一致性；未使用真实数据、未做 GPU 或精度验证。
- 原仓库**未见明确 LICENSE**；仅作研究对照。
- 保持原 notebook 的 FM=16（64维 token）；不支持随意改变 FM。6/10 MSI 通道下的谱核和分组已适配，未做真实数据精度验证。
- 原论文 Houston2013/Trento/MUUFL 精度不能直接迁移。

## 配置与运行

```powershell
python scripts/train_eval_mft.py --config configs/mft_yrd_fs50.yaml
```

配置：`configs/mft_yrd_fs50.yaml`、`configs/mft_yrd2509new_fs50.yaml`、`configs/mft_grss07_custom.yaml`。

已对齐当前主实验的训练数量、5 seeds `202201–202205`、patch 11、200 epochs、train128／test1024、FP32 与 Adam；按最高测试 OA 的运行出图，保留全部运行及汇总指标。详细参数见 [配置索引](../../configs/README.md#新适配基线的主实验配置)。配置／出图流程回归检查通过；正式实验尚未运行。

## 测试与两种出图

`eval_mft.py --run <目录>` 只对保存的测试划分重评估。`infer_mft.py --run <目录>` 默认选测试 OA 最高的最终 checkpoint，推理全部有标注位置（包括训练位置）出图，不重新计分；图及 `labeled_map_info.json` 与测试指标分开保存。只有用户需要时，加 `--mode full` 才运行全图推理。上述脚本均位于 `scripts/`，默认训练入口仅生成标注图，`infer.save_full_map: false`。
