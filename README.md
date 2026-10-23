# MSI–SAR 地物分类项目

空间配准的多光谱与 SAR 邻域 patch 输入，预测中心像素地物类别。

**2026-10-23 状态变更**：按用户要求，旧自研 V0–V8 版本线及其全部实验结果（本地与 server110／server116）已删除，不保留归档。当前模型库只保留两批对比基线：旧集成基线（DFI-Net、FreKFuse、MGHOFNet、MSFMamba、SoftFormer）与新的高光谱–LiDAR 适配基线（DFFNet、M2Heat、MFT；已适配、未运行验证）。DCMNet、HSLiNets 仅收录源码，未适配。详见 [对比方法索引](baselines/README.md) 与 [模型索引](docs/models/README.md)。

**先读 [文档总目录](docs/README.md)**。当前说明以对应配置和实际入口为依据。

| 想了解什么 | 入口 |
| --- | --- |
| 数据集、通道、类别与预处理 | [数据集说明](docs/datasets/README.md) |
| 模型、代码入口与输入限制 | [模型索引](docs/models/README.md)、[对比方法索引](baselines/README.md) |
| 训练、评估、推理与结果解释 | [实验说明](docs/experiments/README.md) |
| 配置哪些是当前有效、哪些是历史网格 | [配置索引](configs/README.md) |
| 脚本入口怎么选 | [脚本索引](scripts/README.md) |
| 第三方源码快照与来源 | [vendor 说明](vendor/README.md) |
| 2025—2026 论文与源码调研全文 | [调研索引](docs/research/README.md)、[完整报告](docs/research/2025-2026-multisource-classification-review.md) |
| 本地、GitHub 与服务器的版本约定 | [AGENTS.md](AGENTS.md)、[统一记录](docs/operations/2026-10-17-version-unification.md) |

## 目录约定

```text
AGENTS.md        协作约束、服务器路径和同步流程
baselines/       各对比方法包装、训练/评估/推理入口与专用说明
models/          旧基线共享模型实现（当前无自研模型）
configs/         可执行配置；datasets/ 是数据集描述 YAML
docs/            人阅读的说明、研究材料和历史归档
scripts/         训练、评估、推理、数据准备与分析入口
tests/           已有检查
vendor/          第三方源码只读快照
data/            数据文件；不纳入 Git
runs_fewshot/    旧基线实验结果；不纳入 Git
```

默认流程：**Windows 本地开发并提交 → GitHub `main` → server110／server116 快进更新**。110 项目目录为 `/ssd/gaojianchun/msi-sar`，116 为 `/data/gjc/msi-sar`。环境、权重和结果分别管理，见 [双服务器运行说明](docs/operations/2026-10-17-dual-server-setup.md)。

## 常用命令

以下命令会实际启动训练或推理，只有在用户明确授权后执行。

```bash
# 检查数据配置与实际文件（只读）
python scripts/check_dataset.py --dataset grss07

# 新适配基线（未运行验证；先做 smoke test 再训练）
python scripts/train_eval_dffnet.py --config configs/dffnet_grss07_custom.yaml

# 只打印实验网格，不启动训练（旧基线）
python scripts/experiment_grid.py --models dfinet --datasets grss07
```

运行时以具体 YAML 为准；新适配基线的输入假设、与原文差异和已知限制见 `baselines/<model>/README.md`。旧 V0–V8 结果与说明已删除，历史记录只存在于 Git 提交。
