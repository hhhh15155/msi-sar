# HSLiNets（2024–2025，波段排序/融合策略）— 早融合变体适配

> **2026-10-31 当前流程**：所有后续运行统一使用 MGHOFNet 训练协议 `mghofnet-v1`，公共实现为 `baselines/{protocol,data,training,prediction}.py`；优化、补零、增强和随机性设置以[当前实验协议](../../docs/experiments/README.md)为准。已完成实验仍属于各自保存的旧配置，本文此前的验证/运行记录不能视为新协议精度。此次只改代码，不启动正式实验；历史重评估按原填充处理。

本目录是 **第三方方法在本项目 MSI–SAR 数据上的基线适配**，不是原论文整网复现。

- 来源：[Judyxyang/HSLiNets](https://github.com/Judyxyang/HSLiNets)，commit `ee315e991bd6a48ed5941a093cd9f73a62020645`；只读快照见 [`vendor/HSLiNets-original/`](../../vendor/HSLiNets-original)。
- 论文：*HSLiNets: Evaluating Band Ordering Strategies in Hyperspectral and LiDAR Fusion*（2024–2025）。

## 原仓库的阻塞（已核对）

原仓库只发布 Jupyter notebook，且**双模态模型类缺失**：

1. notebook 实际定义的类只有 `HSIVimBlock`（自实现 Vim/SSM 块）、`SpatialFeatureProcessing`、`Classifier`、`HSIClassificationMambaModel`（单模态 HSI）。
2. 出现 `HSILidClassificationMambaModel(...)` 的调用，但**没有任何类定义**（见 `HSLiNets_Trento_notebook_extracted.py` 中注释的中融合实例化），直接运行会 `NameError`。
3. 因此论文的中融合双分支网络无法从发布代码还原；按项目规则不能凭空编造缺失网络。

## 采用的适配方案：早融合（LiDAR as additional band）

notebook 中有一处明确注释的早融合策略：

> `# At The MEarly Fusion Stage, Lidar as additional band`

这属于 HSLiNets 论文评估的波段排序／早融合策略之一。本适配：

- 使用 **notebook 中已定义的类**（通过提取脚本原样抽取到 [`baselines/_hsi_adapters/_hslinets_vendor.py`](../_hsi_adapters/_hslinets_vendor.py)，未改动语义）；
- 把 SAR 通道作为额外波段与 MSI 通道拼接（**MSI 在前、SAR 在后**），`num_bands = MSI + SAR` 送入原 HSI 模型；
- `HSIVimBlock`、`SpatialFeatureProcessing`、`Classifier` 均保持 notebook 原实现，无预训练；
- 输入 `[B,1,C,H,W]` → `[B,H,W,C]` → `[B,num_classes]` logits；patch 只要求为奇数。

**这不是论文缺失的中融合双分支网络**；若需要中融合版本，需作者补充代码或明确允许按论文描述重建，再另行实现与对照。

## 已知限制

- **合成数据验证通过、未完成正式实验**：目前只有静态检查与合成数据前向／反向验证；真实数据精度未运行。
- 该实现是"早融合变体"，不能声称是 HSLiNets 论文整网复现；论文精度不能直接迁移。
- `HSIVimBlock` 是自实现的 Vim/SSM 式状态空间块；作为对照基线保留（项目仅"优先非 Mamba"，不删除已有 Mamba 基线）。
- 原仓库**未见明确 LICENSE**；仅作研究对照。

## 运行

```bash
python scripts/train_eval_hslinets.py --config configs/hslinets_yrd_fs50.yaml
python scripts/eval_hslinets.py --run runs_fewshot/fs50/hslinets/yrd/run_001
python scripts/infer_hslinets.py --run runs_fewshot/fs50/hslinets/yrd/run_001 --mode labeled
```

配置：`configs/hslinets_{yrd_fs50,yrd2509new_fs50,grss07_custom}.yaml`（5 seeds、patch 11、200 epochs、AdamW lr 1e-3／wd 0.01＋余弦衰减至 1e-6、FP32、无预训练；`eval.run_index: best` 选测试 OA 最高者出图，全图默认关闭）。
