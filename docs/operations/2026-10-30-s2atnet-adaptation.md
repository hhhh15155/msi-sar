# S²ATNet适配与执行记录

用户明确授权：先运行S²ATNet。沿用现有baseline架构、当前三数据集主实验设置，本地开发提交推送origin/main，110／116快进同一提交。只运行S²ATNet未完成任务，不重跑其他baseline。

## 实现与资源

作者仓库RSIP-NJUPT/S2ATNet，commit `020d738807e67812c5deea4e819d327c08c7993d`，未见明确代码许可证。模型与必要差异见[专用README](../../baselines/s2atnet/README.md)。原始源码字节及SHA256保留在vendor；加载时仅移除未必可用的timm导入，以Torch等价trunc_normal_初始化，保留作者实际前向及64维／8头／单层默认结构。MSI10＋SAR4、MSI6＋SAR1，token数P²，默认patch11。

开始时本地／110／116的HEAD同为 `f83ad28de4bafbcee0758db11bbad6e153ec17b6`。本地与116干净；110既存vendor/MSFMamba-original未跟踪内容保留。未发现S²ATNet已完成结果。110 GPU0另有约6GB占用；116 GPU1空闲，正式队列选择116物理GPU1。

使用现有116 Python `/home/wwl/.conda/envs/gjc/bin/python`，临时设置PYTHONPATH至Conda site-packages、LD_LIBRARY_PATH至Conda lib；已有torch2.6.0+cu118／NumPy2.0.1／einops0.8.1。未安装、升级依赖，不关闭user site。Windows已有gjc缺少einops，模型验证在116完成。

## 验证记录

- 先编写接口／配置及模型合同检查；实现前接口不存在、配置缺失按预期失败。
- 临时副本 `/tmp/msi-s2atnet-check-20261030-a` 首轮6项专用检查通过，覆盖通道、patch7/11/13、梯度、参数覆盖、checkpoint、eval批次独立性及临时MAT完整链路。
- 新增原始前向对照通过；116临时副本全仓回归42 tests / OK（无跳过，19.82秒），日志 `/tmp/s2atnet-check-20261030-regression.log`。
- GPU1 FP32 train128/test1024检查：6/1参数296889、峰值1760.01MiB；10/4参数298365、峰值1763.34MiB。
- 6个数据MAT的SHA256与既有数据清单全部一致；真实GRSS07一轮／单seed smoke及全部标注图完成，结果在临时副本 `smoke_runs/s2atnet/grss07/run_001`，GPU日志 `/tmp/s2atnet-check-20261030-gpu.log`。合成／smoke不纳入正式精度结果。

## 部署与队列安排

正式只调用 `scripts/train_eval_s2atnet.py --config configs/s2atnet_<setting>.yaml`。沿用5 seeds 202201–202205、200epochs、train128/test1024、patch11、FP32、Adam lr0.001/wd0.0001、最终checkpoint。全部5次完成后按最高测试OA选全部标注图，不重计分、不做全图。

116单worker顺序GRSS07→YRD2509NEW→YRD；预定结果为 `runs_fewshot/grss07_custom/s2atnet/grss07/run_001` 与 `runs_fewshot/fs50/s2atnet/{yrd2509new,yrd}/run_001`。运行专用调度器在 `runs_fewshot/s2atnet_queue_20261030/launcher.py`，screen名 `msi_s2atnet_gpu1`；queue_status.json／txt保存提交、配置SHA256、PID、状态和返回码，driver.log及各setting日志记录输出。每任务启动前至少6GiB空闲显存，锁定Git HEAD、拒绝源码脏状态；已有完整任务跳过，仅缺图则补图；发现不完整run或执行失败则停止，不自动重训。

实际同步提交和启动状态以交付核验及队列状态为准；启动不代表全部完成。

本地默认基础环境 `python -s -m pytest -q`：26 passed / 16 skipped（缺少torch的模型项跳过）；CLI的三种 `--help`、源码SHA256、语法及新文档相对链接检查通过。vendor保留作者原始空白，项目新增代码／配置／说明单独执行diff空白检查。

独立只读审查未发现可确认P1/P2/P3或提交阻塞：核对了AST去timm的实际调用路径、12个vendor哈希、核心保真、通道／P²尺寸及训练／测试／标注图协议。
