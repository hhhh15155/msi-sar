# 项目文档总目录

2026-10-23 按用户要求删除旧自研 V0–V8 版本线及其全部实验结果后，本目录已同步清理：旧结果表、V6 说明与旧批次实验页已删除（历史见 Git 提交）。

## 按问题找文档

| 问题 | 当前入口 |
| --- | --- |
| 数据放哪里、多少通道、多少类别、如何预处理 | [数据集说明](datasets/README.md) |
| 模型实现在哪、输入有何限制 | [模型索引](models/README.md)、[对比方法索引](../baselines/README.md) |
| 训练、评估、推理与结果解释 | [实验说明](experiments/README.md) |
| 配置哪些是当前有效、哪些是历史网格 | [配置索引](../configs/README.md) |
| 脚本入口怎么选、哪些是一次性脚本 | [脚本索引](../scripts/README.md) |
| 第三方源码快照、来源与许可证状态 | [vendor 说明](../vendor/README.md) |
| 从哪篇文献／仓库得到这些候选 | [调研索引](research/README.md)、[完整合并报告](research/2025-2026-multisource-classification-review.md)、[2026-10-23 顶会补检](research/2026-10-23-topconference-followup.md)、[HSI–LiDAR 源码快照](research/2026-10-23-hsi-lidar-source-snapshot.md) |
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
  models/         当前模型与入口索引
  experiments/    当前实验入口、协议与比较限制
  research/       文献报告、源码审查与快照来源
  operations/     版本同步、文档迁移等可追溯记录
  archive/        已过时的项目说明和分析提示词
```

模型专用 README 留在 `baselines/<model>/`，由 [模型索引](models/README.md) 与 [对比方法索引](../baselines/README.md) 链接。运行配置在 `configs/`，第三方材料在 `vendor/`。

## 维护规则

- 当前事实按“配置与代码 → 当前说明 → 历史记录”的顺序核对；冲突时先报告。
- 当前使用说明只保留一个主要维护位置，其他位置用链接导航。
- 删除或替换说明时同步更新本页链接；历史材料保留原文时须标明其历史属性。
- 新说明使用稳定的英文文件名、中文正文和相对链接；阶段性记录加日期。
