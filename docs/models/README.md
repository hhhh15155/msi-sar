# 模型与代码入口

本页索引项目中实际存在的模型和包装入口，依据本轮只读检查。它不等同于论文正式复现认证，也不把文献候选视为已经集成。

## 当前实现

| 模型标识 | 核心实现／类 | 训练包装 | 模态参数 | patch 与适配要点 | 专用说明 |
| --- | --- | --- | --- | --- | --- |
| `dfinet` | [model.py](../../baselines/dfinet/model.py)：`DFINet` | [train.py](../../baselines/dfinet/train.py) | `spectral_channels`、`sar_channels` | 构造函数接收 `patch_size`；按现有 YAML 使用，不承诺任意大小均已验证 | [README](../../baselines/dfinet/README.md) |
| `frekfuse` | [frekfuse.py](../../models/frekfuse.py)：`FreKFuse`／`FreKFuseLite` | [train.py](../../baselines/frekfuse/train.py) | `ms_channels`、`sar_channels` | few-shot 配置为 Lite、patch=11；构造默认 32 不是当前实验的强制尺寸 | [README](../../baselines/frekfuse/README.md) |
| `mghofnet` | [model.py](../../baselines/mghofnet/model.py)：`MGHOFNet` | [train.py](../../baselines/mghofnet/train.py) | `hsi_channels`、`aux_channels` | 由既有 HSI–LiDAR 工程适配；实际输入为 MSI／SAR，保留通道参数历史命名 | [README](../../baselines/mghofnet/README.md) |
| `softformer` | [model.py](../../baselines/softformer/model.py)：`SoftFormerBaseline` → `SoftFormer` | [train.py](../../baselines/softformer/train.py) | `hsi_channels`、`aux_channels` | patch=11 与内部 `model_img_size=8` 不同；内部图像大小必须可被 4 整除，stem 会中心裁剪 | [README](../../baselines/softformer/README.md) |
| `msfmamba` | [model.py](../../baselines/msfmamba/model.py)：`MSFMamba` → 作者 `Syn_layer` | [train.py](../../baselines/msfmamba/train.py) | `ms_channels`、`sar_channels` | Mamba／SSM；当前包装限定 patch=11、单层；不足 9 个 MSI 通道时谱轴补零 | [README](../../baselines/msfmamba/README.md) |
| `fuzzy_transformer` | [fuzzy_transformer.py](../../models/fuzzy_transformer.py)：`FuzzyTransformer` | [train.py](../../baselines/fuzzy_transformer/train.py) | `ms_channels`、`sar_channels` | 两级模糊融合、动态坐标位置编码；正奇数 patch；不依赖 Mamba 或预训练 | [当前模型说明](fuzzy-transformer.md) |

训练与全图推理统一使用 `scripts/train_eval_<model>.py`、`scripts/eval_<model>.py`、`scripts/infer_<model>.py` 这一命名模式；实际命令和输出位置见 [实验说明](../experiments/README.md)。没有训练授权时，不因为阅读本页就启动这些入口。

## 模型来源与工程边界

- 第三方快照位于 `vendor/`，MSFMamba 为固定提交子模块。模型专用 README 记录相应来源；公开源码与复用许可证仍需分别核对。
- `models/__init__.py` 并不是所有基线的统一训练注册器。核实调用链应从具体 `scripts/train_eval_*.py` 到 `baselines/<model>/train.py`，再到实际模型定义。
- FreKFuse 包含 FFT／频带与门控相关模块；MGHOFNet 核心定义包含大核、门控双向交互和二阶池化相关模块。此处仅索引代码，不在未完成逐层审查时重写其论文创新归属。
- MSFMamba 包装有纯 PyTorch selective-scan fallback，但计算结构仍属于 Mamba／SSM。不能将“不使用 CUDA 扩展”误写为“不依赖 Mamba 机制”。
- 当前首选研究方向是不依赖 Mamba 的 MSI–SAR 融合；已有 Mamba 比较基线保留，不为偏好而删除。
- VBE-Net 已从当前源码移除；历史提交、日志或结果中出现该名称，不代表它仍可由当前实验网格运行。

## 文献候选与当前代码分开

MSAF、CFFormer、FDMF 等候选的审查与推荐见 [完整调研报告](../research/2025-2026-multisource-classification-review.md)。没有对应的项目包装、配置与入口前，不能写成“已接入本项目”。报告中的性能、许可证及实现风险也不能直接替换本项目的实验结论。

新增模型时，应补充本索引、专用说明、实际入口和配置，同时记录输入假设、预训练、特殊算子、patch 限制及与现有协议的差异。
