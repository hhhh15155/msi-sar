# HSI–LiDAR 第三方代码快照（只读来源，不是项目当前方法）

本项目以 `vendor/<Name>-original/` 保存 2026-10-23 检索到的 5 个高光谱–LiDAR 分类代码库的**代码快照**，用于审查与适配参照。所有仓库均**未见明确 LICENSE**，公开可读不等于可任意复用；这些快照只作为研究来源记录，不构成对再分发或商用的授权。

| 目录 | 论文 | 仓库与 commit | 快照内容 | 项目适配状态 |
| --- | --- | --- | --- | --- |
| `DFFNet-original/` | IEEE GRSL 2025, *Dynamic Frequency Feature Fusion Network for Multi-Source Remote Sensing Data Classification* | https://github.com/oucailab/DFFNet, `a9beb2b2b43b452cfca364160822f0dee9b34462` | 训练/测试/数据/模型/模块代码与 `dataset_info.yaml`（含 Berlin/Augsburg 4 通道 SAR 配置） | ✅ 已适配为 `baselines/dffnet/`（`DFFNet`），共享实现在 `baselines/_hsi_adapters/dffnet_patch.py` |
| `M2Heat-original/` | IEEE TCSVT 2026, *Toward Interpretable Multimodal Fusion: Heat Conduction Modeling for Hyperspectral and LiDAR Joint Classification* | https://github.com/Weikan0425/M2Heat_HSI_LiDAR, `8fcdce85e92a5d8b92c0a86ca78f835113dd3e3f` | 单文件模型 `m2heat.py` + `demo.py`；**未复制** `datasets/`（Git LFS 数据未下载） | ✅ 已适配为 `baselines/m2heat/`（`M2Heat`），共享实现在 `baselines/_hsi_adapters/m2heat_patch.py` |
| `MFT-original/` | IEEE TGRS 2023, *Multimodal Fusion Transformer for Remote Sensing Image Classification* | https://github.com/AnkurDeria/MFT, `de214e512ff1a8f6aefdf27fc2cb60c75e0f559a` | README、`record.py`、`Model/MFT.ipynb` 与 notebook 提取版 `MFT_notebook_extracted.py` | ✅ 已适配为 `baselines/mft/`（`MFT`），共享实现在 `baselines/_hsi_adapters/mft_patch.py` |
| `DCMNet-original/` | IEEE TGRS 2025, *Dynamic Cross-Modal Feature Interaction Network for Hyperspectral and LiDAR Data Classification* | https://github.com/oucailab/DCMNet, `b88ead6db8bc8a5669290e2e6c3af7d062158171` | 主脚本、数据加载、routing/特征交互/分类模块（按数据集硬编码 HSI 波段核） | ⛔ 未适配（见 `baselines/dcmnet/README.md`：光谱核与 Cell reshape 强耦合） |
| `HSLiNets-original/` | 2024–2025, *HSLiNets: Evaluating Band Ordering Strategies in Hyperspectral and LiDAR Fusion* | https://github.com/Judyxyang/HSLiNets, `ee315e991bd6a48ed5941a093cd9f73a62020645` | README、两个 Jupyter notebook 与提取版 `.py`（含自定义 Vim/SSM 块） | ⛔ 未适配（见 `baselines/hslinets/README.md`：notebook 未定义双模态模型类，直接运行会 `NameError`） |

## 快照规则

- 只复制代码与说明，**未复制数据集、权重、图片、PDF**；`tmp/hsi-lidar/` 中的克隆仅用于只读审查且已被 `.gitignore` 忽略。
- 原始仓库代码未做语义修改；适配代码在 `baselines/_hsi_adapters/`（共享实现）与 `baselines/<model>/`（项目统一接口）中单独实现，并在 README 说明偏差（分类读出、通道自适应、去除预训练等）。
- notebook 提取脚本未纳入仓库；提取规则为"只保留 code cell，按顺序拼接"，供人工核对，不直接执行。
- 已适配的三个基线**尚未在 GPU 上做前向/训练验证**（本地无 PyTorch 环境）；在 smoke test 通过前，不应把其配置加入正式运行队列。
