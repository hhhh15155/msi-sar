# 历史项目材料归档

此目录保留被替代的项目说明、旧分析提示词和旧日志。它们不是当前使用说明，不是当前任务指令，也不是重新核实过的实验结果。

## 2026-10-17 整理前原文

| 文件 | 来源与状态 |
| --- | --- |
| [project-readme.original.md](2026-10-17/project-readme.original.md) | 整理前根 README，完整保留其命令及描述；当前入口为 [项目 README](../../README.md) |
| [frekfuse-readme.original.md](2026-10-17/frekfuse-readme.original.md) | 含旧验证集数量和 patch 描述；已由 [现说明](../../baselines/frekfuse/README.md) 纠正 |
| [yrd2509new-experiments.original.md](2026-10-17/yrd2509new-experiments.original.md) | 原 `configs/YRD2509NEW_EXPERIMENTS.md`，模型列表与完成判断过时；当前说明在 [experiments/](../experiments/yrd2509new.md) |
| [yrd-vs-yrd2509-analysis-prompt.md](2026-10-17/yrd-vs-yrd2509-analysis-prompt.md) | 原根 `prompt_for_codex.md`，包含历史 OA、特征距离和待求解问题，本轮未重算 |
| [scheduler-output.txt](2026-10-17/scheduler-output.txt) | 原根 `scheduler_output.txt`，日志中的 `/home/wwl/gjc/msi-sar` 不是当前 server110 项目路径 |

另有 [网页端文献交接原文](../research/archive/2026-10-17-web-handoff.md) 留在研究资料的来源归档下。

## 已知误读风险

- 旧分析提示词中关于额外 SAR 通道“独立物理来源”的说法未经核实，不能作为事实；当前约束明确部分通道是极化派生特征。
- 旧 OA 可能来自不同标签版本、划分、配置或运行状态，不能在没有对应 run 记录时用于模型排名。
- 历史日志和文档出现的服务器路径、硬件环境、默认模型、验证集数量及 batch 值，不代表当前项目事实。
- 原文保留并不表示其中每项建议都被采纳。需要继续历史问题时，应重新核实数据和代码。

原文件均按字节校验保存，迁移来源与 SHA256 见 [迁移清单](../operations/2026-10-17-document-migrations.json)。不要直接改写这些原文；应在当前说明或新的带日期记录中更正。
