# 110 统一队列：释放 HSLiNets 半成品、改双并发续跑

> **后续更新**：当日稍后按用户要求，HSLiNets 已从项目整体删除（代码/配置/脚本/快照/结果），本文仅作历史记录；删除记录见 [HSLiNets 删除与三端同步](2026-11-02-hslinets-removal.md)。

## 用户授权（2026-11-02）

用户要求：`hslinets` 的 grss07、yrd 两个数据集的半成品可以释放并清除当前结果，**先跑其它待跑实验**，并且**可以并行两个程序**。

## 操作前状态（12:53）

- 队列 `runs_fewshot/unified_queue_20261101/` 共 39 项：完成 28、运行 3（均为 HSLiNets）、待跑 8。
- 当时的调度是 12:36 建立的串行接管控制器 `msi_priority_gpu0`（`priority_handoff.py`，1 并发；两个 HSLiNets 被 SIGSTOP 冻结）。串行的原因是 GPU 上另有他人 MSEONet 训练长期占用约 15.5 GB/24 GB 显存，三个 HSLiNets 并行时训练被拖到约 120 s/epoch。
- HSLiNets 三个 `run_001` 均为不完整结果：
  - `grss07`：seed 1/5，训练约 epoch 146/200；
  - `yrd`：seed 1/5，测试 346/2383 batch；
  - `yrd2509new`：已完成 4 个 seed，第 5 个 seed 训练已结束、测试到 129/141 batch。

## 实际执行的变更

1. 停掉原串行接管控制器（`priority_handoff.py`，pid 37493）及其 screen，随后终止被冻结的 `grss07`、`yrd` 两个训练进程和已挂起的旧 launcher（`run_unified_baselines.py`，pid 47794）。
   - **副作用（已处理）**：旧 launcher 所在 screen 关闭时，会话挂断信号连带杀掉了当时仍在正常收尾的 `hslinets_yrd2509new`（测试 129/141 batch），该项因此也不完整，只能一并重跑。教训：此后新控制器用 `start_new_session=True` 启动训练进程，重启控制器不会再误杀在跑任务。
2. 清除三个 HSLiNets 的 `run_001` 半成品目录（均为本轮重跑产物，非历史结果），并在队列状态中把这 3 项重置为 `pending`、`original_fingerprint=None`：
   - `runs_fewshot/grss07_custom/hslinets/grss07/run_001`（6 个文件，18.7 MB）
   - `runs_fewshot/fs50/hslinets/yrd2509new/run_001`（23 个文件，187.7 MB）
   - `runs_fewshot/fs50/hslinets/yrd/run_001`（7 个文件，37.8 MB）
3. 新接管控制器 `runs_fewshot/unified_queue_20261101/queue_controller_20261102.py`：
   - `--workers 2`（双并发）；排序为**非 HSLiNets 待跑项优先**，HSLiNets 最后重跑；
   - HSLiNets 之间仍只允许 1 个同时运行（三个并行的病态减速有实测记录）；重跑顺序按预计耗时从短到长：yrd2509new → grss07 → yrd；
   - 每个任务启动前仍执行原来的准入检查（HEAD 锁定 `660de32`、目标路径/旧结果指纹校验、显存预留 `can_start`）、结束校验（5 seeds、checkpoint、`mghofnet-v1` 身份、最高 OA 标注图）；
   - 训练进程以独立 session 启动，避免控制器自身重启影响在跑任务。
4. 首次启动（12:56）发现排序键写反，HSLiNets 被优先启动；已在 13:00 前停掉该版本、把受影响的 `thsgr_yrd2509new`、`hslinets_yrd2509new` 两项重置为 pending 并清除刚产生的半成品，修正排序键后 12:59 重新启动。
5. 运维记录文件（均在队列目录）：`driver_20261102.log`、`parallel_control.json`、`transition_20261102_release_hslinets.json`、`transition_20261102_queue_order_fix.json`；辅助脚本 `requeue_hslinets_20261102.py`、`reset_after_order_fix_20261102.py`。

## 变更后状态（13:00 起）

- 28 项完成不变；`thsgr_yrd2509new_fs50`、`thsgr_yrd_fs50` 已并行启动并正常推进（GPU 利用率约 85%、显存约 18.9/24.5 GB，其中他人 MSEONet 15.5 GB 未动）。
- 待跑 9 项：THSGR 剩余 0（两项在跑，`thsgr_grss07_custom` 早前已完成）、GFSFN 3 项、S²ATNet 3 项，以及最后重跑的 HSLiNets 3 项。
- 未改动：其他用户的 MSEONet 进程、116 上的结果、HEAD `660de32bde363ce27d16c4d0b5bfeff2a2291599`（队列身份锁定期间 110 不更新代码；本文件所在提交暂不同步到 110/116，待队列结束后再统一快进）。

## 复现/核查命令（110）

```bash
cd /ssd/gaojianchun/msi-sar/runs_fewshot/unified_queue_20261101
cat queue_status.txt parallel_control.json driver_20261102.log
screen -ls                      # 应为 msi_queue_20261102
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```
