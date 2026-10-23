# HSLiNets（2024–2025）— 已收录源码，尚未完成项目适配

- 来源：[Judyxyang/HSLiNets](https://github.com/Judyxyang/HSLiNets)，commit `ee315e991bd6a48ed5941a093cd9f73a62020645`；只读快照见 [`vendor/HSLiNets-original/`](../../vendor/HSLiNets-original)。
- 论文：*HSLiNets: Evaluating Band Ordering Strategies in Hyperspectral and LiDAR Fusion*（2024–2025）。

## 当前状态：不可运行（未适配）

原仓库只发布 Jupyter notebook，且双模态模型定义缺失：

1. 两个 notebook 中实际定义的类只有 `Config`、`HSIVimBlock`（自定义 Vim/SSM 块）、`SpatialFeatureProcessing`、`Classifier`、`HSIClassificationMambaModel`（**单模态 HSI**）。
2. notebook 里出现 `HSILidClassificationMambaModel(` 的调用，但**没有任何 `class HSILidClassificationMambaModel` 定义**（其中一处还被注释）；直接运行状态行会 `NameError`。
3. `HSIVimBlock` 是自实现的 Mamba/Vim 式状态空间块（非 `mamba_ssm` 依赖），与项目"优先非 Mamba"的约定也不一致；作为对照基线可以接受，但需要先补齐双模态分支。
4. 未见明确 LICENSE。

## 下一步（未执行）

- 从论文/作者处确认双模态网络结构（LiDAR/辅助分支如何与 Vim 块连接），再决定复刻或仅保留 HSI 分支作单模态对照。
- 补齐后同样需要 110 上的 smoke test，再登记为可用基线。

在完成上述工作前，本目录不会出现在可用基线索引中。
