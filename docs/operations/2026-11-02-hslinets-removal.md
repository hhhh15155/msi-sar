# HSLiNets 删除与三端同步（2026-11-02）

## 用户授权

用户 2026-11-02 要求：HSLiNets 直接删除（代码、脚本一并删），更新项目文档，三端（本地 / server110 / server116）同步。

## 删除范围

- **代码**：`baselines/hslinets/`（model/train/eval/infer/dataset/io/metrics/README）、`baselines/_hsi_adapters/hslinets_patch.py`、`baselines/_hsi_adapters/_hslinets_vendor.py`；
- **入口脚本**：`scripts/train_eval_hslinets.py`、`scripts/eval_hslinets.py`、`scripts/infer_hslinets.py`；
- **配置**：`configs/hslinets_grss07_custom.yaml`、`configs/hslinets_yrd2509new_fs50.yaml`、`configs/hslinets_yrd_fs50.yaml`（仓库模型配置 134 → 131 份）；
- **第三方快照**：`vendor/HSLiNets-original/`（README、两个 notebook、两个 notebook 提取 `.py`、`Model/.README`）；
- **测试**：`tests/test_hsi_adapter_models.py` 中的 HSLiNets 用例，以及 `tests/test_baseline_map_protocol.py`、`tests/test_training_protocol.py` 的模型清单；
- **队列代码**：`scripts/run_unified_baselines.py` 的模型清单与项数（39 → 36 项，12 模型 × 3 数据集）；
- **110 结果与队列任务**：`runs_fewshot/{fs50,grss07_custom}/hslinets/` 下的 `run_001` 半成品全部清除；统一队列中 3 个 HSLiNets 任务标记 `removed`、不再运行；
- **116**：核实无 HSLiNets 结果目录（`runs`、`runs_fewshot` 下均无）；代码随 git 同步删除。

## 文档更新

根 `README.md`、`AGENTS.md`、`baselines/README.md`、`baselines/_hsi_adapters/__init__.py`、`configs/README.md`、`scripts/README.md`、`docs/README.md`、`docs/models/README.md`、`docs/experiments/README.md`、`vendor/README.md`、`docs/research/README.md`、`docs/research/2026-10-23-hsi-lidar-source-snapshot.md` 均已移除 HSLiNets 的当前模型/入口/配置，或标明其为历史记录。文献调研主报告等历史材料保留原文，不因删除改写。

## 队列收尾（110）

- 2026-11-02 13:36 前后终止仍在收尾的 HSLiNets 重跑（`hslinets_yrd2509new_fs50`，测试中断）并清除其 `run_001` 半成品；队列控制器随后停止。
- 队列最终状态：**36 项完成**（12 模型 × 3 数据集）、**3 项 `removed`**（HSLiNets）；明细见 `runs_fewshot/unified_queue_20261101/queue_status.json`、`queue_status.txt` 与 `transition_20261102_remove_hslinets_tasks.json`。
- 未动其他用户进程（显卡上他人 MSEONet 任务未在操作范围内）。

## 验证

- 本地：`git grep -i hslinets` 仅命中历史文档/研究记录与队列日志（不参与运行）；pytest 结果见下方同步记录。
- 110／116：`git rev-parse HEAD` 三端一致；`find . -iname '*hslinets*' -not -path './.git/*'` 仅剩历史文档与队列历史日志。

## 三端同步

（同步结果随后补记于本节：本地提交、origin、110 与 116 HEAD，以及删除后的残留检查结果。）
