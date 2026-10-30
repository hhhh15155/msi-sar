# GFSFN适配与试跑

用户于2026-10-30在确认双路邻域patch中心像素分类后指定“GFSFN你试试吧”。按现有主实验协议完成模型适配、代码同步与GFSFN三数据集试跑；只处理该模型未完成任务，不重跑既有baseline。

## 实现与计划

- [x] 固定作者commit `065cb808870515b70c4980642895fa45da63a1af`，保存原始源码及SHA256，明确代码许可未声明。
- [x] 使用作者实际模型前向，原生band1/band2参数适配6/1、10/4；包装batch=1输出，删除未用标量MixTrans.p。
- [x] 补齐baseline模型包、三种CLI、三份主实验YAML及各入口索引。
- [x] 本地模型前后向、参数注册与梯度、checkpoint、原始前向数值对照及eval批次独立性检查。
- [x] 116完整合成链路、GPU容量与真实数据smoke；未安装／升级依赖。
- 代码部署：本地提交推送origin/main后，110／116仅快进更新；实际提交与核验以交付报告为准。
- 正式运行：核对数据及重复结果、GPU资源后，116 GPU1单worker执行三数据集共15次；实时状态、PID与身份由下文队列文件记录，启动不等于全部完成。

## 环境与证据

开始时三端HEAD一致 `92af7cc4a8fe14f1126fc248c304baa3de3190c6`。Windows工作区干净；116干净；110既存 `vendor/MSFMamba-original` 未跟踪内容保留。两服务器均无GFSFN正式结果。110 GPU0占用约18.9GB、100%利用率；116两张V100空闲，选择物理GPU1。

Windows默认Python无torch；已有 `C:/Users/goh/.conda/envs/gjc/python.exe` 带torch2.5.1，模型专用检查通过，缺少h5py/sklearn，未安装依赖。默认基础环境回归检查另记录结果。

116默认导入的用户目录torch已变为2.1.2+cu118，与Conda NumPy2.0.1不兼容，并且非激活shell的系统libstdc++过旧。Conda中已有torch2.6.0+cu118；验证进程临时显式设置 `PYTHONPATH=/home/wwl/.conda/envs/gjc/lib/python3.11/site-packages` 和 `LD_LIBRARY_PATH=/home/wwl/.conda/envs/gjc/lib`，不设置PYTHONNOUSERSITE，保留必要的用户依赖，不改变已安装环境。历史THSGR为torch2.6.0+cu124，当前实际版本需单独记录，不假称环境相同。

正式协议、必要适配和调用入口见[GFSFN README](../../baselines/gfsfn/README.md)。

## 已完成验证

- 本地基础环境 `python -s -m pytest -q`：24 passed / 11 skipped（该环境无PyTorch）；已有gjc环境分别运行GFSFN模型的通道／patch／反向／checkpoint、原始前向数值对照及eval批次独立性检查，均通过。
- 116临时副本 `/tmp/msi-gfsfn-check-20261030-a`，显式选择Conda包后 `python -m unittest discover -s tests -p test_gfsfn.py -v`：7项全部通过。日志 `/tmp/gfsfn-check-20261030-tests-conda.log`。
- 116物理GPU1，torch2.6.0+cu118、NumPy2.0.1，FP32 train128/test1024，6/1和10/4两种通道均有限输出；峰值已分配显存约865.7 MiB。参数量分别1155590（5类）和1213834（9类）。
- 6个数据MAT逐项SHA256匹配[数据清单](2026-10-17-dual-server-data-manifest.json)。
- 真实GRSS07一轮／单seed smoke（原训练量、train128/test1024）与标注图成功，临时结果 `/tmp/msi-gfsfn-check-20261030-a/smoke_runs/gfsfn/grss07/run_001`，不纳入正式精度比较。GPU日志 `/tmp/gfsfn-check-20261030-gpu.log`。

## 正式队列安排

116物理GPU1，单worker，顺序GRSS07→YRD2509NEW→YRD。只调用已提交的 `scripts/train_eval_gfsfn.py --config configs/gfsfn_<setting>.yaml`；每组5 seeds／200epochs。输出 `runs_fewshot/grss07_custom/gfsfn/grss07/run_001` 和 `runs_fewshot/fs50/gfsfn/{yrd2509new,yrd}/run_001`，全部种子指标保留，按最高测试OA选全部标注图，不启动全图。

运行专用调度器保存于116 `runs_fewshot/gfsfn_queue_20261030/launcher.py`；screen名 `msi_gfsfn_gpu1`。状态与PID／源码提交／配置SHA256／起止时间／返回码见 `queue_status.json` 和 `queue_status.txt`；日志为 `driver.log` 与各setting日志。调度器启动时锁定已推送Git HEAD、拒绝源码脏状态；已有完整任务跳过，仅缺图则补图；不完整run停止并报告，失败不自动重训。每任务启动前至少4GiB空闲显存。正式启动与Git同步核验在交付时报告。

独立只读代码审查未发现可确认的P1/P2/P3；检查了核心继承、尺寸、参数、三份配置、最终checkpoint、测试／标注图分离、选图与来源哈希。

116相同临时验证副本完整回归：`python -m unittest discover -s tests -v`，35 tests / OK，无跳过（17.13秒）；日志 `/tmp/gfsfn-check-20261030-regression.log`。三个CLI的 `--help` 均通过；源码SHA256、语法、文档相对链接和diff空白检查通过。

vendor目录局部 `.gitattributes` 使用 `-text` 保留作者源码原始换行字节；上游原有尾随空白不改写。diff空白检查针对项目新增实现／配置／说明，排除只读上游快照。
