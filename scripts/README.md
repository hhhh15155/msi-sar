# 脚本目录索引

2026-10-23 删除旧 V0–V8 后，本目录中与 V 线相关的入口已删除：`main_experiment.py`、`generate_main_configs.py`、`generate_paper_candidate_configs.py`、`generate_v6_ablation_configs.py`、`generate_yrd_rerun_configs.py`、`promote_yrd_results.py`、`generate_fuzzy_transformer_configs.py`、`train/eval/infer_fuzzy_transformer.py` 等。

## 当前入口

| 脚本 | 用途 |
| --- | --- |
| `train_eval_<model>.py` | 单基线训练 + 测试指标 + 标签图（当前推荐入口） |
| `eval_<model>.py` | 加载已完成 run 做整图/标签图评估 |
| `infer_<model>.py` | 全图推理 |
| `experiment_grid.py` | 只打印旧基线实验网格，不启动训练 |
| `generate_grss07_configs.py`、`generate_msfmamba_configs.py` | 生成旧基线配置网格 |
| `analyze_fewshot_results.py` | 汇总 `runs_fewshot/` 下旧基线结果 |
| `performance_analysis.py`、`dump_all_metrics.py` | 结果分析辅助 |

当前基线入口对应关系：

- 旧集成基线：`dfinet`、`frekfuse`、`mghofnet`、`msfmamba`、`softformer`；
- 新适配基线（未运行验证）：`dffnet`、`m2heat`、`mft`；
- 新基线配置：`configs/{dffnet,m2heat,mft}_*.yaml`。

## 数据准备（按需运行，不属于训练入口）

`check_dataset.py`、`prepare_grss07.py`、`build_yrd2509new*.py`、`expand_yrd2509new*.py`、`refine_yrd2509new_labels.py`、`rewrite_yrd2509new_mat_compat.py`、`upgrade_yrd2509new_sar14.py`、`align_yrd2509new_class_names.py`。

## 分析／一次性脚本

`analyze_*.py`、`feature_analysis*.py` 等用于当时分析，不是可复现实验入口。若确认无文档引用，可统一归档到 `scripts/archive/`。

## 约定

- 训练必须带配置快照、种子、划分与输出目录记录；
- 不新增与 `train_eval_*` 职责重复的入口；
- 一次性脚本如果仍被文档引用，移动前先更新链接；
- 没有用户明确授权不启动训练或推理。
