# 脚本目录索引

`scripts/` 里既有**当前入口**，也有历史一次性脚本。按用途读取，不要默认每个脚本都是当前主流程。

## 当前入口

| 脚本 | 用途 |
| --- | --- |
| `main_experiment.py` | 主实验调度器：`prepare / train / queue / launch / status`，所有正式训练都通过它和 manifest 运行 |
| `train_eval_<model>.py` | 单个基线的训练 + 测试指标 + 标签图（当前推荐入口，10 个：dfinet、frekfuse、fuzzy_transformer、mghofnet、msfmamba、softformer、dffnet、m2heat、mft、yrd2509new_fewshot） |
| `eval_<model>.py` | 加载已完成 run 做整图/标签图评估 |
| `infer_<model>.py` | 全图推理 |
| `generate_*_configs.py` | 生成配置网格（历史网格与候选配置） |
| `promote_yrd_results.py` | 结果发布与归档校验 |

## 数据准备（按需运行，不属于训练入口）

`prepare_grss07.py`、`build_yrd2509new*.py`、`expand_yrd2509new*.py`、`refine_yrd2509new_labels.py`、`rewrite_yrd2509new_mat_compat.py`、`upgrade_yrd2509new_sar14.py`、`align_yrd2509new_class_names.py`、`check_dataset.py`。

## 分析／一次性脚本

`analyze_*.py`、`feature_analysis*.py`、`performance_analysis.py`、`dump_all_metrics.py`、`experiment_grid.py`、`prepare_grss07.py` 等。它们用于当时的分析或数据修补，**不是可复现实验入口**；后续整理时优先归档到 `scripts/archive/`（本次未移动，避免破坏历史文档引用）。

## 历史入口（保留但不再推荐）

`train_<model>.py`（6 个）是早期单基线入口，功能已由 `train_eval_<model>.py` 覆盖；部分旧 README 仍引用它们。新任务请使用 `train_eval_*`。

## 约定

- 训练必须带配置快照、种子、划分与输出目录记录（`main_experiment.py` 已内置身份校验）；
- 不新增与 `train_eval_*` 职责重复的入口；
- 一次性脚本如果仍被文档引用，移动前先更新链接。
