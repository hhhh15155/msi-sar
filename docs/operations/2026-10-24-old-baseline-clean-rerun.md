# 旧 5 个 baseline 干净对照重跑（2026-10-24）

## 目的

把旧集成基线（DFI-Net、FreKFuse、MGHOFNet、MSFMamba、SoftFormer）在 **server110 当前环境**下按现有 15 份配置重跑一次，消除与之前历史结果的差异来源：

- 历史结果来自本地 `runs_fewshot/`（执行环境、时间与当前 110 不同）；
- YRD 旧共享读取器曾使用 global min-max＋中心化，而当前 `configs/datasets/yrd.yaml` 为逐通道 min-max；
- 之前与新增 HSI–LiDAR 基线的对比因此不是完全同预处理/同环境。

本次重跑后，旧 5 个基线与新增 5 个基线可在**相同数据配置、相同划分、相同 5 seeds、相同 patch/epoch/batch/test 协议、同一台服务器**下对比。

## 部署

| 项目 | 内容 |
| --- | --- |
| 服务器 | server110 `/ssd/gaojianchun/msi-sar`，物理 GPU0（RTX 3090） |
| 代码版本 | `0f2a15e206`（正式启动时本地/110/116 一致；队列脚本在运行目录，不改仓库代码） |
| screen 会话 | `msi_old_baseline_rerun_gpu0`（2026-10-24 16:17:26 启动） |
| 并发 | **3 workers**，启动前检查 `nvidia-smi` 空闲显存，低于 4096 MiB 时等待；GPU0 与 `lc` 常驻任务共用 |
| 调度脚本 | `runs_fewshot/old_baseline_rerun_20261024/launcher.py`（结果目录内，不提交仓库） |
| 日志 | `runs_fewshot/old_baseline_rerun_20261024/<model>_<setting>.log`、`queue_status.txt/.json`、`driver.log` |
| 结果 | `runs_fewshot/fs50/<model>/<dataset>/run_001/`（GRSS07：`runs_fewshot/grss07_custom/<model>/grss07/run_001/`） |

15 个任务（5 模型 × 3 数据集）通过 `scripts/train_eval_<model>.py --config configs/<model>_<setting>.yaml` 执行；
每份配置 5 seeds `[202201…202205]`、patch 11、200 epochs、train128 / test1024、FP32、最终 checkpoint 测试；
`eval.run_index: best` 仅用于选标注图运行，不参与指标。

## 配置说明（保留各基线原设置）

15 份配置沿用仓库现有 YAML，未改超参数。各基线的优化器设置本就不同，如实记录：

| 模型 | lr | weight_decay |
| --- | ---: | ---: |
| dfinet | 1e-3 | 1e-4 |
| frekfuse | 5e-4 | 1e-4 |
| mghofnet | 1e-3 | 0.01 |
| msfmamba | 1e-4 | 0 |
| softformer | 1e-3 | 1e-4 |

新增 HSI–LiDAR 基线统一使用 Adam lr 1e-3 / wd 1e-4。**该超参数差异是本次对比的已知不同点**，不能把差异全部归因于模型结构。

## 启动过程记录（如实）

- 第一次启动的 launcher 有一个并发计数 bug（未把子进程加入 active 列表），导致越过 3 并发限制、短时间内启动了多个任务；
- 发现后立即停止 screen 与全部相关进程（GPU 回到仅 `lc` 占用），并删除该次产生的未完成 run 目录；
- 修复 launcher（加入 active 记录、完成检查、显存门限）后从干净状态重新启动；当前结果为修复后的正式队列。

## 监控

```bash
ssh server110 'cd /ssd/gaojianchun/msi-sar && screen -ls && cat runs_fewshot/old_baseline_rerun_20261024/queue_status.txt'
ssh server110 'cd /ssd/gaojianchun/msi-sar && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader'
```

## 完成后的处理

1. 核对 15 份配置各 5 个 run、指标与选图记录（与新增基线相同的检查）；
2. 生成“旧 5 基线（本次重跑）vs 新 5 基线（2026-10-23 队列）”的 5-seed 均值/标准差对比表，并注明超参数差异与比较限制；
3. 本批结果只保存在 110；是否同步 116/本地由用户决定。
