# 统一为 MGHOFNet 训练流程

> **2026-11-02 更新**：HSLiNets 已按用户要求删除，当前模型入口为 12 个、模型配置为 131 份；本文保留 2026-10-31 时点（13 模型、134 配置）的历史记录。

用户要求：所有现有 baseline 使用 MGHOFNet 的训练与数据处理流程，将流程固定为项目 skill；本次不启动正式实验。

## 实施清单

- [x] 公共协议、数据读取/patch/增强、AdamW/余弦调度、随机性控制和单 epoch 训练；模型结构保留。
- [x] 13 个模型入口及全部现有模型配置/配置生成器复用公共实现；测试和出图使用对应运行的填充方式。
- [x] 新运行保存协议版本和数据配置快照。旧运行按原配置解释，不修改结果、不自动重跑。
- [x] 更新项目适配 skill、协作说明、协议和各模型 README，消除旧默认值冲突。
- [x] 静态、公共流程合成数据回归、独立审查；记录未验证项。

统一值：AdamW，lr=0.001，weight_decay=0.01，CosineAnnealingLR(T_max=epochs, eta_min=0.000001)，普通交叉熵，不裁剪梯度；逐通道整图 min-max，补零，双模态同步翻转/旋转，cuDNN deterministic=True / benchmark=False。主实验仍为 patch11、200 epochs、train128/test1024、FP32、5 seeds；网络内部参数不统一。

不把仅改 YAML 视为完成：优化器、调度、增强和填充必须在实际调用链生效。历史无协议版本的 reflect 模型仍按 reflect 重评估；MGHOFNet/SoftFormer 按保存 pad_mode（默认 constant）。测试只计算测试划分，标注图包含全部标注但不计分，全图只能显式调用。

验证重点：边缘像元训练/重评估/出图一致；两路增强同步且随机分布不变；调度每个 epoch 后执行；配置生成器不能恢复旧默认值；历史结果不得被重命名为新协议。

## 实施与验证

- 新增公共 `baselines/{protocol,data,training,prediction,io}.py`；13 个模型的 train/dataset/eval/infer/io 和 CLI 复用对应流程。没有修改网络实现、vendor、数据、权重或实验结果。
- 全部 134 份模型 YAML（包括 39 份主实验与辅助样本量网格）统一协议字段；预算、种子、划分和网络参数不变。两个实际配置生成器同步修改，YRD2509NEW 的检查入口仍只核对文件。
- 新增项目 skill `.agents/skills/msi-sar-training-protocol/SKILL.md`，适配 skill 和 AGENTS.md 强制链接；数值协议由 `baselines/protocol.py` 与当前实验文档维护。
- 配置/调用链审计：`python -s scripts/check_training_protocol.py`，13 模型、134 配置通过。
- Windows 默认 Python 的全仓回归：`python -s -m pytest -q`：**30 passed、22 skipped**。缺 PyTorch 的模型运行项跳过，不能冒充全部模型运行验证。
- Windows 现有 gjc（PyTorch 2.5.1）执行 `pytest tests/test_training_protocol.py -q`：**10 passed**。覆盖 AdamW 与实际余弦数值、种子/cuDNN、6/1 与10/4的 packed/dual patch及同步增强、边缘补零/历史reflect预测、MAT读取/逐通道归一化/协议SHA256、真实MGHOFNet前后向与checkpoint、真实FreKFuse字典输出及CE等价。
- 只读独立审查未发现有充分证据的 P1/P2；其直接推理非标准 pad_mode 提醒已改为明确拒绝。两份 skill 的 quick_validate 使用 Windows UTF-8 模式通过，CLI --help、Python语法、skill相对链接、git diff --check 通过。
- **未验证**：本轮没有在服务器执行模型或对全部13模型重新做真实训练链路；本地 gjc 缺 einops/h5py/sklearn，未安装依赖。历史结果未修改、未重评估，未启动正式实验/批量队列/全图。
- 新代码不改变旧结果协议身份。后续训练若获单独授权，必须另建结果目录并保留全部旧结果；不能直接把新旧结果写成同协议排名。
