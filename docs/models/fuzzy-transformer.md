# Fuzzy Transformer：当前项目实现

本页描述当前工程实现，不是 ACFFN 的复现认证，也不代表该组合已经构成充分的论文创新。核心定义为 [models/fuzzy_transformer.py](../../models/fuzzy_transformer.py) 的 `FuzzyTransformer`；[基线 model.py](../../baselines/fuzzy_transformer/model.py) 仅导出该类，实际训练构造来自 [train.py](../../baselines/fuzzy_transformer/train.py) 的 `build_model`。

## 网络路径

```text
MSI patch → MSI CNN stem → 全网格 tokens ＋ 坐标位置编码 → MSI Transformer ┐
                                                                       ├→ 第一级模糊融合
SAR patch → SAR CNN stem → 全网格 tokens ＋ 坐标位置编码 → SAR Transformer ┘
                  ↑ 第一级融合结果分别投影后残差反馈到两路
两路第二级 Transformer → 第二级模糊融合 → 融合 Transformer
                     → 中心 token 与平均 token 拼接 → 分类头
```

交互采用完整空间 token 网格；中心像素只用于最终读出与辅助分类，不作为 Query 查询另一模态邻域。模型类接收 `[B,C,H,W]` 的两路输入，检查空间对齐和通道数；高宽均须为正奇数。训练入口使用单一 `patch_size` 构造正方形 patch，默认配置为 11，并未硬编码只能使用 11。

## 融合与训练设置

- `GaussianMembership`／`FuzzyFeatureFusion` 借鉴 ACFFN 的高斯隶属度与模糊 AND／OR 聚合思想，来源见源码说明及 [完整调研](../research/2025-2026-multisource-classification-review.md)。项目采用正值宽度、稳定对数聚合、模态归一化和保留两路原始特征的残差投影。
- 当前示例配置：`embed_dim: 192`、`num_heads: 6`、每路每级深度 2、融合 Transformer 深度 2、`num_memberships: 7`、`auxiliary_weight: 0.2`。以实际 YAML 为准。
- 不加载预训练权重，不引入 LiDAR 高度假设，不额外对 SAR 做一次未经确认的对数变换。数据读取复用 DFI-Net 路径，实际归一化由数据配置决定。
- 辅助损失来自两路中心 token 分类；主分类头同时读融合中心 token 和平均上下文。
- `return_fusion_weights=True` 时可读取每级权重，形状为 `[B, patch_pixels, 2, embed_dim]`，在模态轴归一化。权重只是模型学到的特征偏好，不是已经校准的传感器不确定性。

## 入口与配置

| 用途 | 路径 |
| --- | --- |
| 训练并导出标注图 | [train_eval_fuzzy_transformer.py](../../scripts/train_eval_fuzzy_transformer.py) |
| 评估 | [eval_fuzzy_transformer.py](../../scripts/eval_fuzzy_transformer.py) |
| 整图推理 | [infer_fuzzy_transformer.py](../../scripts/infer_fuzzy_transformer.py) |
| 配置生成 | [generate_fuzzy_transformer_configs.py](../../scripts/generate_fuzzy_transformer_configs.py) |
| GRSS07 自定义数量 | [fuzzy_transformer_grss07_custom.yaml](../../configs/fuzzy_transformer_grss07_custom.yaml) |
| YRD2509NEW 每类 50 个 | [fuzzy_transformer_yrd2509new_fs50.yaml](../../configs/fuzzy_transformer_yrd2509new_fs50.yaml) |

现有配置覆盖三数据集的 `fs5/10/20/50/100/150/200`，另有 GRSS07 自定义数量，共 22 份。本模型示例的训练／最终测试／评估／推理批大小均为 128，与部分旧基线的测试批大小 1024 不同。

训练结束保存 `model_last.pth` 并在保留测试集上评估；标注图固定选择整数 run（当前为 0），不接受 `eval.run_index: best`。图中可展示全部有标注像元，但 `eval_metrics_run_*.json` 评分只针对保存的测试划分。汇总未完成的种子组时，`completed: false` 不算完整实验。

## 优先保留的对照

当前已完成九版本的候选和融合机制比较，见 [结果汇总](../experiments/main-results-20261018.md)；V6逐路径消融尚未实现／运行，具体定义见 [后续实验路线](../experiments/next-experiments.md)。V5不是保持V6其余配置不变的单级消融。

源码的 `fusion='concat'` 实际将模糊权重替换为两路等权混合，同时保留投影／残差／归一化。当前构造参数会作用于两个融合级；若只研究第二级，需要独立控制该级，不能直接改全局字段。第一级融合只通过反馈生效，因此去反馈应称“移除早期融合—反馈路径”。另可单独研究 `auxiliary_weight=0`，但不与前两项同时改变。

既有包装训练器会传结构参数，而主实验的 `build_main_model` 对V6只传通道／类别、其余取类默认；新消融必须接入实际入口并校验。用户目前只要求方案与文档更新，不据此启动上述实验。主实验测试batch是1024；前文128指旧包装配置，不是当前主实验协议。
