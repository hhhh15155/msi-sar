# DCMNet 与 HSLiNets 适配记录（2026-10-23）

用户要求把检索到的两个 HSI–LiDAR 方法适配为项目 baseline。本记录按
[模型适配规则](../models/baseline-adaptation.md) 与
[项目 skill](../../.agents/skills/msi-sar-baseline-adaptation/SKILL.md) 执行。

## 交付

| 模型 | 状态 | 实现 |
| --- | --- | --- |
| DCMNet（TGRS 2025） | 适配完成、合成数据验证通过、未完成正式实验 | `baselines/dcmnet/` + `baselines/_hsi_adapters/dcmnet_patch.py` |
| HSLiNets（2024–2025） | 早融合变体适配完成、合成数据验证通过、未完成正式实验 | `baselines/hslinets/` + `baselines/_hsi_adapters/hslinets_patch.py` + `_hslinets_vendor.py` |

配套：三数据集主实验配置各 3 份、`scripts/{train_eval,eval,infer}_<model>.py`、回归测试
`tests/test_hsi_adapter_models.py`（现覆盖 5 个适配模型 × 2 种通道组合 × patch 11/13）、
各索引与 README 更新。

## DCMNet 的适配决策

1. **保留源码模块**：`net/Cells.py`、`net/Feature_Interaction.py`、`net/Router.py`、
   `net/routing_modules.py`、`net/classify.py` 原样 import。
2. **重写编码器**：原 `hsi_feature_extractor_*.py` 把 PCA-30/20/63 波段与固定 patch 硬编码
   （3D 核 9/7/5 等），不能只改通道。适配采用深度为 1 的谱核，把谱混合交给原 2D 投影层：
   10 通道时形状与公开的 Trento 分支（80/160/320 输入）一致，6 通道按同规则缩减。
3. **SAR 编码器**：保留原 2D LiDAR stem 的层与核，仅首层输入通道数改为实际 SAR 通道（1/4）；
   不把 LiDAR 高度语义套到 SAR，不复制/插值波段。
4. **参数注入**：原 `parameter.py` 在 import 时加载 `.mat` 并写死窗口常量；改用参数 stub，
   按 `patch_size` 设置 `hsi_windowSize = patch_size - 6`、`size_SA = (patch_size-6)²`。
5. **patch 限制**：路由 Cell 要求方形特征图，入口校验 patch 为奇数且 ≥9；11/13 已覆盖。
   主实验 patch 11 与协议一致。
6. **死模块清理**：vendor 路由层同时构造两个 stage 的 Cell、`Cross_Self_attention.dense`、
   `RoutingModule.bn`／`path_mapping` 在 forward 中从未使用，会形成永远无梯度的已注册参数。
   适配层在构建后删除这些模块（vendor 源文件不改），使"优化器覆盖参数"检查有明确含义。
7. 无预训练；分类头保持原 `1×1 conv → 512 → num_classes`。

## HSLiNets 的适配决策与阻塞

1. **原仓库阻塞（已核对源码）**：只有 notebook，且只定义了单模态
   `HSIClassificationMambaModel`；`HSILidClassificationMambaModel` 只有调用与注释实例化，
   没有类定义，直接运行会 `NameError`。因此论文的中融合双分支网络无法还原，按规则不能编造。
2. **采用策略**：notebook 中明确注释了早融合策略
   `# At The MEarly Fusion Stage, Lidar as additional band`，这是该论文评估的波段排序/
   早融合策略之一。适配把 SAR 通道作为额外波段与 MSI 拼接（MSI 在前），送 notebook 已定义的
   `HSIClassificationMambaModel`；`HSIVimBlock`、`SpatialFeatureProcessing`、`Classifier`
   保持 notebook 原实现（类代码原样抽取到 `_hslinets_vendor.py`）。
3. **必须明示**：该实现是"早融合变体"，**不是论文缺失的中融合双分支网络**，不能声称整网复现；
   若后续需要中融合版本，需作者补充代码或明确允许按论文描述重建。
4. 无预训练；patch 仅要求为奇数。

## 验证

- 静态：`py_compile`、`git diff --check`、配置/入口路径检查。
- 合成数据（不加载项目数据、不训练实验）：
  - `tests/test_hsi_adapter_models.py` 覆盖 5 模型 ×（6+1 / 10+4 通道）×（patch 11/13）：
    前向输出形状与有限性、反向梯度、优化器参数覆盖、checkpoint 保存/重载一致性；
  - DCMNet 依赖 vendor 路由模块的 `einops`，在 server110 `gjc` 环境执行；HSLiNets 另在本地
    已有 PyTorch 环境执行通过。
- 未完成：真实数据训练、精度、出图。真实数据 smoke test 与正式队列须按用户授权另行执行。

## 未做 / 边界

- 未安装、升级任何依赖；未修改 vendor 源码；未提交/运行真实实验。
- 两个仓库均未见明确 LICENSE，仅作研究对照。
- 结果不能与论文精度直接比较；DCMNet 的深度 1 谱核、HSLiNets 的早融合选择均为与原文的
  已知差异，已写入各自 README。
