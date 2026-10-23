# HSI–LiDAR 第三方代码快照（只读来源，不是项目当前方法）

本目录保存 2026-10-23 检索到的 5 个高光谱–LiDAR 分类代码库的**代码快照**，用于审查与适配参照。所有仓库均**未见明确 LICENSE**，公开可读不等于可任意复用；本目录只作为研究来源记录，不构成对再分发或商用的授权。

| 目录 | 论文 | 仓库与 commit | 快照内容 | 适配状态 |
| --- | --- | --- | --- | --- |
| `DFFNet-original/` | IEEE GRSL 2025, *Dynamic Frequency Feature Fusion Network for Multi-Source Remote Sensing Data Classification* | https://github.com/oucailab/DFFNet, `a9beb2b2b43b452cfca364160822f0dee9b34462` | 训练/测试/数据/模型/模块代码；`dataset_info.yaml`（Berlin 4 通道 SAR、Augsburg 4 通道 SAR、Houston/Trento 1 通道 LiDAR） | 已适配为 `baselines/hsi_lidar/dffnet_patch.py`（`v9_dffnet`），见 `docs/experiments/hsi-lidar-baselines.md` |
| `M2Heat-original/` | IEEE TCSVT 2026, *Toward Interpretable Multimodal Fusion: Heat Conduction Modeling for Hyperspectral and LiDAR Joint Classification* | https://github.com/Weikan0425/M2Heat_HSI_LiDAR, `8fcdce85e92a5d8b92c0a86ca78f835113dd3e3f` | 单文件模型 `m2heat.py` + `demo.py`；**未复制** `datasets/`（Git LFS 数据，均未下载） | 已适配为 `baselines/hsi_lidar/m2heat_patch.py`（`v10_m2heat`） |
| `DCMNet-original/` | IEEE TGRS 2025, *Dynamic Cross-Modal Feature Interaction Network for Hyperspectral and LiDAR Data Classification* | https://github.com/oucailab/DCMNet, `b88ead6db8bc8a5669290e2e6c3af7d062158171` | 主脚本、数据加载、routing/特征交互/分类模块（按数据集硬编码 HSI 波段核） | 未适配（波段核与 Cell 尺寸假设需重写，列为下一批） |
| `MFT-original/` | IEEE TGRS 2023, *Multimodal Fusion Transformer for Remote Sensing Image Classification* | https://github.com/AnkurDeria/MFT, `de214e512ff1a8f6aefdf27fc2cb60c75e0f559a` | README、`record.py`、`Model/MFT.ipynb`；另存 notebook 提取版 `MFT_notebook_extracted.py` 供审查 | 已适配为 `baselines/hsi_lidar/mft_patch.py`（`v11_mft`） |
| `HSLiNets-original/` | 2024–2025, *HSLiNets: Evaluating Band Ordering Strategies in Hyperspectral and LiDAR Fusion* | https://github.com/Judyxyang/HSLiNets, `ee315e991bd6a48ed5941a093cd9f73a62020645` | README、两个 Jupyter notebook 与提取版 `.py`（含自定义 Vim/SSM 块） | 未适配（自定义 SSM 块 + 依赖 notebook 结构；列为下一批） |

## 快照规则

- 只复制代码与说明，**未复制数据集、权重、图片、PDF**；`tmp/hsi-lidar/` 中的克隆仅用于只读审查且已被 `.gitignore` 忽略。
- 原始仓库代码未做语义修改；适配代码在 `baselines/hsi_lidar/` 单独实现，并在文件头说明偏差（分类读出、通道投影、去除预训练等）。
- notebook 提取脚本未纳入仓库；提取规则为"只保留 code cell，按顺序拼接"，供人工核对，不直接执行。
- `DCMNet-original/net/*_feature_extractor_*.py` 中的光谱卷积核大小按原数据集波段数硬编码（如 Houston2013 依赖 PCA 后 30 波段），迁移到 6/10 通道 MSI 需重写自适应光谱核，不能只改通道参数。
