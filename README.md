# MSI–SAR 地物分类项目

空间配准的多光谱与 SAR 邻域 patch 输入，预测中心像素地物类别。当前维护五个对比基线及项目自己的 Fuzzy Transformer；数据、代码和实验结果分别管理。

**先读 [文档总目录](docs/README.md)**。当前说明以对应配置和实际入口为依据，历史分析与日志统一归档。

| 想了解什么 | 入口 |
| --- | --- |
| 数据集、通道、类别与预处理 | [数据集说明](docs/datasets/README.md) |
| 模型、代码入口与输入限制 | [模型索引](docs/models/README.md) |
| 当前 CNN–Transformer 与模糊融合实现 | [Fuzzy Transformer](docs/models/fuzzy-transformer.md) |
| 训练、评估、推理、调度和结果解释 | [实验说明](docs/experiments/README.md) |
| YRD2509NEW 专项运行 | [YRD2509NEW 实验](docs/experiments/yrd2509new.md) |
| 2025—2026 论文与源码调研全文 | [完整调研报告](docs/research/2025-2026-multisource-classification-review.md) |
| 本地、GitHub 与服务器的版本约定 | [AGENTS.md](AGENTS.md)、[统一记录](docs/operations/2026-10-17-version-unification.md) |

## 目录约定

```text
AGENTS.md        协作约束、服务器路径和同步流程
baselines/       各基线包装及相邻的专用说明
models/          项目模型实现
configs/         可执行配置；datasets/ 是数据集描述 YAML
docs/            人阅读的当前说明、研究材料和历史归档
scripts/         训练、评估、推理、数据准备与调度入口
tests/           已有检查
vendor/          第三方源码快照或固定提交子模块
data/            数据文件；不纳入 Git
runs_fewshot/    实验结果；不纳入 Git
experiment_logs/ 运行日志；不纳入 Git
```

默认流程：**Windows 本地开发并提交 → GitHub `main` → server110 快进更新**。远端项目目录为 `/ssd/gaojianchun/msi-sar`。两端代码一致不代表数据、环境或结果一致。

## 常用命令

在项目根目录执行，先确认数据和环境。以下训练／推理命令会实际启动任务；本次文档整理没有执行它们。

```bash
# 检查数据配置与实际文件
python scripts/check_dataset.py --dataset grss07

# 当前模型：GRSS07 自定义训练数量，或 YRD2509NEW 每类 50 个训练像元
python scripts/train_eval_fuzzy_transformer.py --config configs/fuzzy_transformer_grss07_custom.yaml
python scripts/train_eval_fuzzy_transformer.py --config configs/fuzzy_transformer_yrd2509new_fs50.yaml

# 只打印所选实验网格，不启动训练
python scripts/experiment_grid.py --models fuzzy_transformer --datasets grss07
```

运行时以具体 YAML 为准：不能把所有模型的 patch、批大小、预处理或环境需求视为一致。现有配置采用固定训练样本数；“few-shot”不代表 episodic 元学习框架。已知比较条件差异和历史指标边界见 [实验说明](docs/experiments/README.md)。

旧 README、旧分析提示词和调度输出保存在 [历史归档](docs/archive/README.md)，不会作为当前实验结论。完整调研报告原文保留，截止日期仍为报告自身记录的日期。
