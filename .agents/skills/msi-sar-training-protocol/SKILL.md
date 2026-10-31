---
name: msi-sar-training-protocol
description: 在 MSI-SAR 项目新增或修改 baseline 的训练、数据处理、评估、出图与实验配置时，执行并核查统一的 MGHOFNet 训练流程；仅查结果或文献调研不触发实现或训练。
---

# MSI–SAR 统一训练流程

先读根目录 [AGENTS.md](../../../AGENTS.md) 和 [当前实验协议](../../../docs/experiments/README.md)。用户指定所有 baseline 使用 MGHOFNet 的训练/数据处理流程；这不要求把其他网络改成 MGHOFNet，也不自动授权运行。

## 实现与核查

1. 追踪 `scripts/train_eval_<model>.py → baselines/<model>/train.py` 实际调用链。配置默认值由 [`baselines/protocol.py`](../../../baselines/protocol.py) 维护，不为新模型单独复制训练器或数据流水线。
2. 复用 [`training.py`](../../../baselines/training.py) 的 `set_seed`、`build_optimizer_scheduler`、`train_epoch`、`record_protocol`：AdamW、lr=0.001、wd=0.01，epoch 后余弦调度到 1e-6，普通交叉熵、FP32、无梯度裁剪，cuDNN deterministic=True / benchmark=False。显式旧参数与当前协议冲突时应报错，不能假装配置已生效。
3. `dataset.py` 只转出 [`data.py`](../../../baselines/data.py) 的公共读取器和 patch 类。逐通道整图无标签 min-max；constant 补零；50% 原样、12.5% 水平翻转、12.5% 垂直翻转，其余三种旋转各约8.33%，两模态同步。保留模型所需 packed 或双输入接口，实际通道按共享数据配置校验。不要将网络内部 Dropout、宽度等训练成相同结构。
4. 主实验采用实验文档的三组训练量、patch11、200 epochs、train128/test1024、5 seeds。其他样本量配置只改变划分数量。核对全部现有模型 YAML 和生成器，不能只修改主实验三份或只改 YAML 而遗漏代码中的 Adam/固定学习率/reflect。
5. 新运行保存 `training_protocol: mghofnet-v1`、完整有效配置、数据配置快照和 `training_protocol.json`。历史 run 不补写新版本、不改结果；重评估按保存的协议选择填充：无版本的 MGHOFNet/SoftFormer 按原 pad_mode，其余历史模型为 reflect。数据快照优先于当前共享 YAML；无历史快照时明确不能完整保证原预处理身份。
6. 测试、重评估、标注图和显式全图共用 [`prediction.py`](../../../baselines/prediction.py)，读取运行配置。最终 checkpoint 测试，禁止测试选 epoch。全部种子完成后按最高测试 OA 选图，并列取首个；图包含全部标注位置但不计分、不覆盖测试指标。全图只能用独立 CLI 的 `--mode full` 显式调用。

## 验证与边界

- 先运行 `python -s scripts/check_training_protocol.py` 做不依赖 PyTorch 的配置/调用链审计。
- 有现成依赖时运行 `pytest tests/test_training_protocol.py tests/test_baseline_map_protocol.py`，并做适配模型相称的合成检查；特别验证边缘 patch、双模态增强同步、调度时机、checkpoint、测试与图一致及旧 run 的填充兼容。仅静态检查不等于真实模型验证。
- 不安装/升级依赖，不启动正式数据训练或全图，不重跑已完成实验；用户另有明确授权时只执行其指定范围。统一代码不代表旧结果采用了统一协议，也不承诺精度提高。
- 变更流程时同步实际代码、模型配置、生成器、实验文档及各模型 README；历史操作记录保留原事实。交付说明版本、检查证据、尚未验证项和是否启动过实验。
