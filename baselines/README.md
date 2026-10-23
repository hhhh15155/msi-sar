# 对比方法（baselines）索引

本目录存放各对比方法的**项目内包装**。每个子目录遵循同一约定：`model.py` 暴露项目统一接口，`train.py`/`eval.py`/`infer.py` 提供训练与推理入口，`dataset.py`/`io.py`/`metrics.py` 是共享数据与评估工具，`README.md` 记录来源、输入假设与已知偏差。

通用入口约定（见 [模型与代码入口](../docs/models/README.md)）：`scripts/train_eval_<model>.py`、`scripts/eval_<model>.py`、`scripts/infer_<model>.py`。

## 状态总表

| 目录 | 方法 | 来源 | 状态 | 入口 / 配置 |
| --- | --- | --- | --- | --- |
| `dfinet/` | DFI-Net | 项目已集成基线 | 已用于主实验 27 项（有精度结果） | `scripts/train_eval_dfinet.py`；`configs/dfinet_*.yaml` |
| `frekfuse/` | FreKFuse | 项目已集成基线 | 已用于主实验 27 项（有精度结果） | `scripts/train_eval_frekfuse.py`；`configs/frekfuse_*.yaml` |
| `fuzzy_transformer/` | Fuzzy Transformer（项目模型包装） | 项目自身模型 | 已用于主实验 27 项（有精度结果） | `scripts/train_eval_fuzzy_transformer.py`；`configs/fuzzy_transformer_*.yaml` |
| `mghofnet/` | MGHOFNet | 项目已集成基线 | 已用于主实验 27 项（有精度结果） | `scripts/train_eval_mghofnet.py`；`configs/mghofnet_*.yaml` |
| `msfmamba/` | MSFMamba | 项目已集成基线（Mamba，保留） | 已用于主实验 27 项（有精度结果） | `scripts/train_eval_msfmamba.py`；`configs/msfmamba_*.yaml` |
| `softformer/` | SoftFormer | 项目已集成基线 | 已用于主实验 27 项（有精度结果） | `scripts/train_eval_softformer.py`；`configs/softformer_*.yaml` |
| `dffnet/` | DFFNet（GRSL 2025，HSI–SAR/LiDAR） | [vendor/DFFNet-original](../vendor/DFFNet-original) | **适配完成、未运行验证**（无精度结果） | `scripts/train_eval_dffnet.py`；`configs/dffnet_*.yaml` |
| `m2heat/` | M2Heat（TCSVT 2026） | [vendor/M2Heat-original](../vendor/M2Heat-original) | **适配完成、未运行验证**（无精度结果） | `scripts/train_eval_m2heat.py`；`configs/m2heat_*.yaml` |
| `mft/` | MFT（TGRS 2023） | [vendor/MFT-original](../vendor/MFT-original) | **适配完成、未运行验证**（无精度结果） | `scripts/train_eval_mft.py`；`configs/mft_*.yaml` |
| `dcmnet/` | DCMNet（TGRS 2025） | [vendor/DCMNet-original](../vendor/DCMNet-original) | **仅源码 + 未适配说明**，不可运行 | — |
| `hslinets/` | HSLiNets（2024–2025） | [vendor/HSLiNets-original](../vendor/HSLiNets-original) | **仅源码 + 未适配说明**，不可运行 | — |

## 共享实现

- [`_hsi_adapters/`](_hsi_adapters)：三个第三方适配基线（dffnet/m2heat/mft）的共享模型实现（`dffnet_patch.py`、`m2heat_patch.py`、`mft_patch.py`），直接加载 `vendor/` 中未改动的上游核心模块。**它不是可训练基线**，不要按模型名引用。
- [`split.py`](split.py)：固定训练样本数划分逻辑，被各基线 `dataset.py` 复用。

## 新增基线时必须做

1. 补齐 `model.py / train.py / eval.py / infer.py / dataset.py / io.py / metrics.py / README.md`；
2. 在本表和 [docs/models/README.md](../docs/models/README.md) 登记；
3. 在 `configs/` 增加配置并在 README 写明来源、预训练、输入假设与已知偏差；
4. 标注验证状态：**未运行验证**的基线不得写入主结果表。
