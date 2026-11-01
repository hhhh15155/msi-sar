# 110 统一训练流程覆盖重跑

## 用户授权与范围

2026-11-01 用户要求将更新后的统一脚本放到110运行，直接覆盖原来的结果，删除旧的不一致入口；随后明确要求用 screen 后台并发，由助手按资源安排。

范围固定为13模型（MGHOFNet、DFI-Net、FreKFuse、MSFMamba、SoftFormer、DFFNet、M2Heat、MFT、DCMNet、HSLiNets、THSGR、GFSFN、S²ATNet）×3主实验，共39项，每项5 seeds `202201–202205`、200 epochs、patch11、train128/test1024、FP32，采用 `mghofnet-v1`。三组划分仍是 YRD/YRD2509NEW 每类50、GRSS07 custom。未授权其他shot、116结果删除、其他项目进程操作或全图推理。

## 清理与发布

删除重复 `scripts/train_{dfinet,frekfuse,mghofnet,msfmamba,softformer}.py`，以及默认执行全shot网格的 `scripts/train_eval_yrd2509new_fewshot.py`。统一保留 `train_eval_<model>.py`、测试专用 `eval_<model>.py` 和默认标注图的 `infer_<model>.py`。历史归档文档保留原文，不再代表当前入口。

本地提交/推送后，两端快进至同一提交；只在110启动。来源代码不在服务器独立修改。

## 110 预检证据

- 初检110为 `d18cf9d`；既有30项旧结果（10模型×3）均为 `run_001`、5 seeds、无新协议身份文件；THSGR/GFSFN/S²ATNet在110尚无结果。116上的旧结果本轮不处理。
- 现有环境：`/home/fangyan/.conda/envs/gjc/bin/python`，Python3.9.23、PyTorch2.1.1+cu118、NumPy1.26.4，RTX3090物理GPU0。未安装/升级依赖。
- 六个实际MAT文件SHA256均匹配 `2026-10-17-dual-server-data-manifest.json`。
- 13模型各两种通道组合（6/1+5类、10/4+9类）共26项GPU合成全链路检查通过：临时MAT、单轮训练、train128/test1024、最终checkpoint测试、重评估混淆矩阵一致、最高OA标注图、不覆盖测试指标、不启动全图。产物仅在 `/tmp/msi-unified-preflight-20261101/`，不是正式结果。
- 各模型最高 PyTorch reserved MiB：MGHOFNet2908，DFI-Net1442，FreKFuse1378，MSFMamba652，SoftFormer638，DFFNet818，M2Heat1222，MFT416，DCMNet518，HSLiNets2690，THSGR1336，GFSFN1192，S²ATNet2420。
- 日志 `/tmp/msi-unified-preflight-20261101.log`；逐项数据 `/tmp/msi-unified-preflight-20261101/status.json`。110没有pytest，尝试全仓pytest命令未运行任何测试、未安装依赖；本地全仓回归34 passed、22 skipped（缺Torch），队列4项保护回归通过。26项实际GPU全链路已在110执行。

## 后台调度

- screen：`msi_unified_gpu0_20261101`；仓库 `/ssd/gaojianchun/msi-sar`，GPU0。
- 队列目录：`runs_fewshot/unified_queue_20261101/`；入口 `scripts/run_unified_baselines.py`，最多3个任务，各进程 OMP/MKL 线程数2。
- 每模型预留 `ceil(实测最高reserved MiB×1.25+768)`，同时核查当前空闲显存和外部进程占用，保留额外余量，不停止其他项目进程。
- 任务启动前才覆盖该项旧 `run_001`：必须命中39项白名单、路径无符号链接、旧结果指纹未变化、无额外run目录。保存 `replacement_manifest.json`，不删除其他shot、旧队列日志或116结果。尚未轮到的旧结果保留。
- 新结果使用原位置 `runs_fewshot/fs50/<model>/<dataset>/run_001/`，GRSS07为 `runs_fewshot/grss07_custom/<model>/grss07/run_001/`。新config、数据配置、Git版本和协议身份随运行保存；不能在覆盖期间把剩余旧结果和新结果当作同协议汇总。
- 记录 `queue_status.json/.txt`、driver.log、每项日志、PID、返回码。仅在五种子/权重/协议/最佳OA标注图验证通过后标完成；失败记为failed且不自动重试，其余任务继续。启动时锁定HEAD，源代码改变后停止发新任务并等待已运行任务结束。
- 本文描述部署和执行方式；实际是否启动/完成，以队列状态和日志为准，不将预检或排队写成完成。
