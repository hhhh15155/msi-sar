# 第一轮分类基线与后台部署（2026-10-17）

用户已确认本方案并授权在 server110 后台执行。范围仅为分类：7 个版本在 3 个数据集上各训练一次，共 21 项；不是五种子实验，不进行样本量或 patch 网格搜索，不包含反演。

## 数据与训练协议

| 数据集 | MSI＋SAR 通道 | 每类训练数量 | 总训练数 |
| --- | --- | --- | --- |
| YRD | 10＋4 | 8 类各 50 | 400 |
| YRD2509NEW | 10＋4 | 9 类各 50 | 450 |
| GRSS07 | 6＋1 | `[674,1481,752,45,30]` | 2982 |

固定 seed `202201`、patch `11×11`、200 epochs、训练 batch 128、AdamW（lr=5e-4、weight_decay=1e-4）、余弦调度至 1e-6、梯度裁剪 1。新建对照维度 128、4 头、每路 2 层 Transformer、dropout 0.1。输入通道由配置给出，模型支持其他正奇数 patch，但本轮不扫描尺寸。无预训练；不使用 Mamba 或中心 Query 跨模态检索。

所有版本使用共享读取器：YRD 沿用 `global_minmax_center`，另两者为 `per_channel_minmax`，统计来自全幅无标签影像；边界统一 reflect。按种子生成一次固定训练／测试索引，余下标注像元全部测试，无验证集。逐版本复用同一文件并记录 SHA256。随机像元划分允许邻域重叠，不能声称空间外推验证。

只在最终 epoch 权重上测试，不按测试集选 epoch、种子或超参数。报告 OA、AA、Kappa、Macro-F1、各类召回率／精确率／F1、混淆矩阵、参数量、训练和测试耗时、峰值显存。单种子结果只支持初步筛选，不用于显著性或稳定性结论。

## 版本与执行顺序

| 版本 | 内容 | 解释边界 |
| --- | --- | --- |
| `v0_msi` | MSI CNN–Transformer＋分类头 | 单源基线 |
| `v1_sar` | SAR CNN–Transformer＋分类头 | 单源基线 |
| `v2_concat` | 相同双路编码＋concat 投影 | 简单融合基线 |
| `v3_fcm_ffm` | 相同双路编码＋FCM–FFM | CFFormer 交互路径的分类适配，非整网复现 |
| `v4_mafi` | 相同双路编码＋MAFI＋统一 concat 读出 | EG-CMFNet 局部交互适配，不含边缘分支／LMF 整网 |
| `v5_fuzzy` | 相同双路编码＋单级现有模糊融合 | 使用项目已有 FuzzyFeatureFusion，不是 ACFFN 整网 |
| `v6_current` | 当前 FuzzyTransformer 完整默认架构 | 192 维／6 头、两级融合、反馈、融合 Transformer、辅助损失权重 0.2；容量不同，不能视为严格单模块消融 |

V2—V5 的编码器与分类头在构造融合模块前初始化，同种子初始权重一致。模块本身参数量不同，分别报告。FCM 保留空间→通道残差校正；FFM 保留双向全空间交互与 ChannelEmbed，局部 sr_ratio=1。MAFI 保留原实现的顺序更新、跨通道／空间调制及零初始化残差系数。单 token 标准差使用有限值处理；FCM 的 B×H×W=1 特例使用 BN 运行统计，正式 11×11 不触发。独立适配使用 PyTorch 默认初始化，不宣称与原分割整网初始化完全一致。

每个版本先完成三个数据集，之后才进入下一版本；单张 RTX 3090 最多两个训练／测试进程并发。并发仅影响执行顺序，不增加种子或实验数量。已有其他基线历史指标不混入本次结果。

## 提速与可靠性

- 为避免 YRD 数百万测试像元的 Python 逐 patch 开销，归一化影像一次准备并写为缓存，各任务直接在 GPU 提取 patch；已用边界样例检查与 reflect 取样一致。
- 用户随后明确要求其他参数保持之前设置，正式队列使用原 FP32 训练／测试，不额外引入 BF16 或 TF32 矩阵乘加速；卷积 backend 沿用 PyTorch 原默认。用户最新指定测试 batch 为 **1024**，21 项配置统一更新，训练 batch 仍为 **128**；学习率、epoch、结构和损失均保持上文已确定设置。之前 BF16 的预启动已中止，部分输出单独归档，不混入正式结果。
- 数据顺序和增强使用独立随机生成器，避免融合模块构造消耗随机数后改变样本顺序。训练增强沿用既有概率：不变 0.5，其余横翻、纵翻、旋转 90/180/270 各 0.1。
- 按用户要求通过 **screen** 的 `msi_cls_main_20261017` 会话运行，标准输入关闭，输出落盘，不依赖 SSH 连接持续存在。队列和每个实验 worker 分别持有独占文件锁，避免队列异常退出后重复训练覆盖结果；重启前也检查残留 worker。
- 启动前及后续调度时检测其他 GPU 计算进程；其他任务占用时进入 `waiting_gpu`，不终止对方进程，显卡释放后自动开始，正式任务最多两个并发。
- 第 1、每 10 及最终 epoch 原子保存模型、优化器、调度器和随机状态；重启队列可从最近检查点接续，最多回算未落盘的 9 个 epoch，不是额外实验。
- 正常完成的任务跳过；失败任务标记 failed 并保留日志，队列不自动重跑或挑选最好结果。中断的 running 项在旧进程确认退出后可接续。服务器重启后须手动重启队列，不声称自动开机恢复。
- 代码、配置、数据和缓存指纹用于防止误续跑；最终权重命名 `model_last.pth`。本轮不额外导出全图预测，保存测试预测及对应索引即可复算指标。

## 代码、配置与操作

server116 的 `/data/gjc/msi-sar` 已同步代码及三数据集并通过兼容检查；双服务器启动方式见 [运行说明](../operations/2026-10-17-dual-server-setup.md)。110 的原 21 项已完成，本轮没有在 116 重跑原清单。后续 YRD 预处理修正需独立任务清单、缓存和旧结果归档，不能直接在 116 启动原 21 项。

- [模型定义](../../models/main_classification.py)：`MainClassifier`、`FCMFFMFusion`、`MAFIFusion`、`build_main_model`。
- [训练／队列／后台入口](../../scripts/main_experiment.py)。
- [21 项清单](../../configs/main_first_pass/manifest.json)，[配置生成器](../../scripts/generate_main_configs.py)只生成文件，不运行实验。
- [契约检查](../../tests/test_main_classification.py)。
- 原始机制与源码版本依据见 [调研报告](../research/2025-2026-multisource-classification-review.md) A4、B 的源码分析与消融建议；原报告保持原文。

在 server110 的 `/ssd/gaojianchun/msi-sar` 执行：

```bash
# 首次启动：会实际开始队列，重复运行会检查是否已有活动队列
/home/fangyan/.conda/envs/gjc/bin/python scripts/main_experiment.py launch

# 查看状态，不启动实验
/home/fangyan/.conda/envs/gjc/bin/python scripts/main_experiment.py status

# 查看队列日志
tail -n 40 experiment_logs/main_first_pass_20261017/queue.log

# 查看 screen 会话；通过 Ctrl+A、D 脱离会话，不结束队列
screen -ls
screen -r msi_cls_main_20261017
```

调度器目前继承 `CUDA_VISIBLE_DEVICES`，以该物理 GPU（或 UUID）过滤显卡进程；不再检查所有卡后误等待。110 默认卡为 0；116 必须显式使用 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`。通过 `MAIN_EXPERIMENT_WORKERS=1` 可降低并发，默认上限仍来自清单。不要在旧结果目录下使用已更新源码自动续跑；代码指纹不同会拒绝复用。

输出在 `runs/main_first_pass_20261017/`：`prepared/<dataset>/` 保存共享缓存和划分；`<version>_<dataset>/` 保存配置、来源、环境、训练历史、进度、检查点、测试预测和指标；`summary.csv`／`summary.json` 随完成逐项更新。数据、权重与结果不进 Git。PID／SID 在 `queue_pid.json`，任务 PID 在 `queue_status.json`。
