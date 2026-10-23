# Baseline 队列部署记录（2026-10-23）

## 部署对象

- 服务器：**server110**，项目目录 `/ssd/gaojianchun/msi-sar`，物理 GPU **0**（RTX 3090，24576 MiB）。
- 代码版本：`f65538c348`（`main`，工作区干净；本地 / 110 / 116 三端一致）。
- 与用户 `lc` 的常驻任务共用 GPU0；启动时 GPU0 已用 11944 MiB、利用率 2%。单 worker 串行，不占满。
- Python：`/home/fangyan/.conda/envs/gjc/bin/python`（torch 2.1.1+cu118）。

## 队列内容

**screen 会话**：`msi_baselines_gpu0`（detached，PID 34203，2026-10-23 22:21:48 启动）。

顺序执行 **5 个新适配基线 × 3 个数据集 = 15 份配置**，每份 5 seeds：

- 模型：`dffnet`、`m2heat`、`mft`、`dcmnet`、`hslinets`；
- 数据集：`grss07`（custom `[674,1481,752,45,30]`）、`yrd2509new`（每类 50）、`yrd`（每类 50）；
- 顺序：先 `grss07 × 5 模型`，再 `yrd2509new × 5`，最后 `yrd × 5`（小数据先跑，尽早发现失败）；
- 协议：patch 11、200 epochs、train batch 128、test batch 1024、FP32、Adam lr `0.001` / wd `0.0001`、
  无预训练；seeds `[202201, 202202, 202203, 202204, 202205]`；`eval.run_index: best` 选测试 OA 最高者
  出标注图，`infer.save_full_map: false`（全图默认关闭）。

## 日志与输出

| 内容 | 位置（110） |
| --- | --- |
| 队列状态 | `runs_fewshot/baseline_queue_20261023/queue_status.txt` |
| 单配置日志 | `runs_fewshot/baseline_queue_20261023/<model>_<setting>.log` |
| 结果目录 | `runs_fewshot/fs50/<model>/<dataset>/run_###/` 或 `runs_fewshot/grss07_custom/<model>/grss07/run_###/` |

`queue_status.txt` 按 `START <job> <time>` / `EXIT <job> code=<n> <time>` 记录；失败不会静默重试，
按实际 `code` 记录，后续单独处理。

## 启动前验证（已完成）

- 合成数据回归：`tests/test_hsi_adapter_models.py` 覆盖 5 模型 × 2 通道组合 × patch 11/13，
  server110 `gjc` 环境全部通过（前向/反向/参数覆盖/checkpoint 重载）。
- 真实数据 smoke：GRSS07、1 epoch、1 seed，5 个模型全部 exit 0 并生成 `metrics.json`（仅链路验证，非正式结果）。

## 查看方式

```bash
ssh server110 'cd /ssd/gaojianchun/msi-sar && screen -ls && cat runs_fewshot/baseline_queue_20261023/queue_status.txt'
ssh server110 'cd /ssd/gaojianchun/msi-sar && tail -5 runs_fewshot/baseline_queue_20261023/<job>.log'
```

## 注意

- 队列运行期间不要修改训练源码、配置或 checkpoint；文档修改不影响。
- 已完成的实验不重跑；队列只跑这 15 份配置，不扩成多种子网格。
- 本批是**首次正式运行**这 5 个适配基线，完成后需核对每个配置的 5 个 run、指标均值/标准差与
  `labeled_map_info.json` 的选图记录，再决定后续是否调整。
