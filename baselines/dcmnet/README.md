# DCMNet（TGRS 2025）— 已收录源码，尚未完成项目适配

- 来源：[oucailab/DCMNet](https://github.com/oucailab/DCMNet)，commit `b88ead6db8bc8a5669290e2e6c3af7d062158171`；只读快照见 [`vendor/DCMNet-original/`](../../vendor/DCMNet-original)。
- 论文：*Dynamic Cross-Modal Feature Interaction Network for Hyperspectral and LiDAR Data Classification*, IEEE TGRS 2025。

## 当前状态：不可运行（未适配）

原实现不能只改通道参数就迁移到本项目，原因已核对到源码：

1. **光谱卷积核按数据集硬编码**：`net/hsi_feature_extractor_*.py` 分别对应 Houston2013（PCA 后 30 波段，3D 核 9/7/5→谱维 22/16/12）、Houston2018（20 波段）、Trento（63 波段）；feed 6/10 通道的 MSI 需要重写整套光谱降维，而不是改一个 in_channels。
2. **Cell 的 reshape 与窗口尺寸强耦合**：`net/Cells.py` 通过全局 `args.hsi_windowSize` 做 `rearrange(..., d=args.hsi_windowSize)`，`parameter.py` 还存在 `hsi_windowSize=5`、`size_SA=25` 等与 patch 尺寸绑定的常量；11×11 patch 下三层 3×3 卷积后空间为 5，与原数据集的尺寸假设需要逐项重新核对。
3. **代码组织方式**：`DCMNet` 是训练包装类，`parameter.py` 在 import 时即加载 `.mat` 数据；不能直接作为 `baselines/` 的 `model.py` 使用。
4. **未见明确 LICENSE**。

## 下一步（未执行）

- 先只读推导 11×11、6/10 通道下的 Cell 张量形状，再决定是"输入 1×1 卷积投影到 30 波段"还是"重写自适应光谱核"。
- 在 110 上做单 batch 前向/反向 smoke test，通过后再补 `model.py / train.py / eval.py / infer.py` 与配置，届时才登记为可用基线。

在完成上述工作前，本目录不会出现在可用基线索引中，也不应被调度器引用。
