# 项目文档总目录

此目录是本项目说明材料的统一入口。2026-10-17 整理时以提交 `29bb4f3` 对应代码、现有配置和只读源码检查为依据，没有重新运行实验。

## 按问题找文档

| 问题 | 当前入口 |
| --- | --- |
| 数据放哪里、多少通道、多少类别、如何预处理 | [数据集说明](datasets/README.md) |
| 模型实现在哪、输入有何限制 | [模型索引](models/README.md) |
| 当前双路 CNN–Transformer 如何融合 | [Fuzzy Transformer](models/fuzzy-transformer.md) |
| 如何运行、如何解释指标与结果目录 | [实验说明](experiments/README.md) |
| 本轮 7 版本、单种子、21 项后台分类实验 | [第一轮分类实验与部署](experiments/main-first-pass.md) |
| 三个数据集的主实验样本配置 | [YRD：每类 50](experiments/yrd.md)、[YRD2509NEW：每类 50](experiments/yrd2509new.md)、[GRSS07：custom](experiments/grss07.md) |
| YRD2509NEW 专项配置与命令 | [YRD2509NEW](experiments/yrd2509new.md) |
| 哪些论文、仓库与机制值得继续研究 | [调研索引](research/README.md)、[完整合并报告](research/2025-2026-multisource-classification-review.md) |
| 如何连接服务器、以哪端为准 | [根目录 AGENTS.md](../AGENTS.md) |
| 110／116 双服务器的环境、数据同步和 GPU 选择 | [双服务器运行说明](operations/2026-10-17-dual-server-setup.md) |
| 两端版本如何统一、旧版本备份在哪 | [版本统一记录](operations/2026-10-17-version-unification.md) |
| 这次文档移到哪里、原文是否完整 | [文档整理记录](operations/2026-10-17-document-organization.md) |
| 旧分析提示词、旧 README 和日志 | [历史归档](archive/README.md) |

## 目录职责

```text
docs/
  README.md       总导航，不重复其他文档的全文
  datasets/       当前数据集、标签、通道与预处理说明
  models/         模型索引与本项目方法说明
  experiments/    当前实验协议、命令与已知比较限制
  research/       完整文献报告、原始源码审查 JSON、研究来源归档
  operations/     版本同步、文档迁移等可追溯记录
  archive/        已过时的项目说明、分析提示词和历史日志
```

模型特有的详细 README 留在 `baselines/<model>/`，由 [模型索引](models/README.md) 统一链接。`configs/` 保留运行配置，不继续散放独立实验说明。原始第三方材料留在 `vendor/`，不作为项目当前说明改写。

## 维护规则

- 当前事实按“配置与代码 → 当前说明 → 历史记录”的顺序核对。遇到冲突先报告，不用文档替代实际实现。
- 当前使用说明只保留一个主要维护位置；其他位置用链接导航。完整调研报告虽长，但仍保持单文件供后续任务读取。
- 历史材料保留原文，读取其文件夹说明后再使用；旧提示词与任务模板不是当前执行授权。
- 新说明使用稳定的英文文件名、中文正文和相对链接。阶段性研究报告加日期或时间范围，避免 `最终版2`、`新版final` 等名称。
- 新运行日志进入 `experiment_logs/`，结果进入 `runs_fewshot/` 等输出目录；本次归档的旧调度输出只保留为历史来源。
- 本轮只迁移已明确识别的相关 Markdown 和项目旧说明，没有搬运下载目录里的 PDF，也没有扫描其他私人目录。
