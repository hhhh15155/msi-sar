# THSGR 适配与执行记录

## 范围与实现计划

用户于 2026-10-30 授权：按当前仓库架构适配 THSGR，提交并推送 GitHub，server110、server116 更新相同提交，然后运行 THSGR。以 Windows 本地为源码主副本；不改其他模型，不重跑已完成任务。

来源：`https://github.com/jqyang22/THSGR`，commit `dd2a0ef7148c05a1261c38ab94845d9ff7b879a0`，Apache-2.0。

采用现有 `baselines/<model>/`、`baselines/_hsi_adapters/`、`configs/<model>_*.yaml`、`scripts/{train_eval,eval,infer}_<model>.py` 布局。原始源码保存在 `vendor/THSGR-original/`；模型参数和接口适配独立放在包装层。

- [x] 先增加模型接口、前后向、参数覆盖、checkpoint 和测试／出图一致性检查。
- [x] 保留作者异构图编码、双模态卷积调制、图分支与融合分支加权分类；使 MSI 6/10、SAR 1/4、patch 11/13 可用，去除原始未调用分支。
- [x] 补齐模型包、三个 CLI、三份共享数据配置引用与索引；训练身份记录源版本、配置／数据配置快照、环境和划分。
- [x] 执行 CPU 合成检查、仓库回归和真实数据 smoke；不安装／升级依赖。
- [x] 本地提交并推送 `origin/main`；两服务器仅快进拉取并核对 HEAD，保留现有改动。
- [ ] 核对重复结果、GPU 和输出位置后，以单 worker 执行三数据集共 15 次训练；记录会话、PID、日志和任务状态。

## 已发现的适配点

作者实现以 PCA32 输入，连续谱核深度为 7/5/3，固定展平通道 `32*20`；不适用于 6/10 波段。将谱核深度随实际通道收缩，按真实剩余谱深度构造展平卷积，不复制／插值波段。SAR 第一层改为实际 1/4 通道，不在前向混合通道。

图节点数默认与类别数相同（作者配置 `num_nodes: 1` 乘类别数）；保留 64 维图特征、8×8 相关性矩阵调制与固定融合权重 `w=0.9`。原实现注册但不调用的 CNN／静态图／分类头不进入适配模型。

作者 `FeedForward` 在 train/eval 均混入 batch 平均特征，导致测试、重评估与不同出图批次的预测依赖批次组成。本次采用已向用户解释的方案：训练保留 mean-forward，测试／出图禁用跨样本混合，并添加批次独立性检查。用户要求模型名只写 **THSGR**，不加命名后缀；差异保留在模型 README 和本记录中。

## 协议

遵循 [当前实验协议](../experiments/README.md)：YRD/YRD2509NEW 每类 50，GRSS07 custom `[674,1481,752,45,30]`；patch 11，200 epochs，5 seeds 202201–202205，train128/test1024，FP32，Adam lr0.001/wd0.0001，无预训练；最终 checkpoint 测试。全部预定运行完成后选最高测试 OA 的模型推理全部标注位置出图，不重新计分；默认不做全图推理。

## 验证及部署

待执行后补充证据。目前本地无 PyTorch，两服务器均有现成依赖。初检两端 HEAD 与本地相同：`222b7b73bdef2f23e3f4b5cf6bc7da2ab279b922`；110 有既存 vendor 子目录未跟踪内容，必须保留。


### 已完成检查

- 本地默认 Python 的 `pytest -q` 因既存 NumPy2.5/SciPy1.13 二进制不兼容在 `test_prepare_grss07.py` 收集失败；未修改或安装环境。使用已有基础环境 `python -s -m pytest -q`：22 passed / 6 skipped（本地无 PyTorch）。
- 116 临时验证副本 `/tmp/msi-thsgr-check-20261030-a`，`gjc` Python 下 `python -m unittest discover -s tests -p test_thsgr.py -v`：6项全部通过。覆盖 MSI6+SAR1、MSI10+SAR4，5/8/9类，patch11/13，前后向全部参数有有限梯度、checkpoint重载、eval单样本／混批／逆序预测一致；临时MAT数据完整一轮训练、测试／重评估混淆矩阵一致、标注图背景及不覆盖测试指标。
- 116 物理GPU1：合成 train128 / test1024 FP32 前后向通过，输出有限，峰值已分配显存707.15 MiB。
- 独立只读审查：对照作者实际前向及项目协议，未发现可确认的P1/P2；正式精度和运行速度尚待实验。

- GRSS07真实数据 GPU smoke：1 epoch、1 seed、原训练样本数及train128/test1024完成，退出0；路径 `/tmp/msi-thsgr-check-20261030-a/smoke_runs/thsgr/grss07/run_001`，仅验证链路，不计入正式结果。

### 运行资源安排

110 GPU0 可用显存约3 GB且高负载；116 GPU1可用约18 GB。选择116物理GPU1、`CUDA_VISIBLE_DEVICES=1`、单worker顺序执行 GRSS07→YRD2509NEW→YRD，避免在两端重复训练。两端都更新相同Git提交。当前原有10模型结果在110环境，本次THSGR将记录116环境；不为补同环境对照重跑已完成模型。

正式输出预定为116 `runs_fewshot/grss07_custom/thsgr/grss07/run_001`、`runs_fewshot/fs50/thsgr/{yrd2509new,yrd}/run_001`；调度记录在 `runs_fewshot/thsgr_queue_20261030/`，screen名 `msi_thsgr_gpu1`。正式启动前再次检查重复结果。


### Git同步与部署核验

- 实现提交 `21abda086851fba1b210194631925e4e3aca920a` 已推送 GitHub `origin/main`。
- 两服务器 `git pull --ff-only origin main` 未成功；改用已推送提交的 Git bundle，经 `git bundle verify`、`git fetch <bundle> main`、`git merge --ff-only FETCH_HEAD` 快进更新。已核对本地、110、116 的实现提交一致；110 原有 vendor/MSFMamba-original 未跟踪内容保留，116工作区干净。
- 同步后116运行 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/wwl/.conda/envs/gjc/bin/python -m unittest discover -s tests -v`：28 tests / OK，无跳过；110运行THSGR专用6项检查全部通过。
- 116六个数据MAT的SHA256逐项匹配 [数据清单](2026-10-17-dual-server-data-manifest.json)。
- 本记录的补充提交也推送并快进到两端后，才启动screen。队列固定当时HEAD，不允许运行期间源码漂移；启动时复核两端无THSGR正式结果。

### 正式队列定位

调度器为116 `runs_fewshot/thsgr_queue_20261030/launcher.py`（运行专用，不作为第二训练入口提交源码）。它只调用已提交的三个 `scripts/train_eval_thsgr.py --config ...`，单worker、每个任务启动前至少4 GiB空闲显存；任何失败停止，不自动重试或重训。已完成指标的任务跳过；仅缺图时只出图；发现不完整run则报告并停止。

screen会话：`msi_thsgr_gpu1`。实际启动时间、调度PID、训练PID、命令、配置SHA256、返回码、结果目录及实时状态统一保存到同目录 `queue_status.json` / `queue_status.txt`；日志为 `<setting>.log` 与 `driver.log`。可用 `screen -ls` 和读取状态文件核查，不能把已启动写成已完成实验。各run的 `run_identity.json` 另保存实际代码HEAD、未提交状态、环境及配置身份。
