`baselines/` 存放各对比方法的**项目内包装**。2026-10-23 按用户要求删除旧自研 V0–V8 版本线，`fuzzy_transformer` 基线包同步移除；本索引只列当前实际存在的基线。

通用入口约定（见 [模型与代码入口](../docs/models/README.md)）：`scripts/train_eval_<model>.py`、`scripts/eval_<model>.py`、`scripts/infer_<model>.py`。

## 状态总表

| 目录 | 方法 | 来源 | 状态 | 入口 / 配置 |
| --- | --- | --- | --- | --- |
| `dfinet/` | DFI-Net | 项目已集成基线 | 旧基线（结果目录 `runs_fewshot/`，历史网格） | `scripts/train_eval_dfinet.py`；`configs/dfinet_*.yaml` |
| `frekfuse/` | FreKFuse | 项目已集成基线 | 旧基线（结果目录 `runs_fewshot/`，历史网格） | `scripts/train_eval_frekfuse.py`；`configs/frekfuse_*.yaml` |
| `mghofnet/` | MGHOFNet | 项目已集成基线 | 旧基线（结果目录 `runs_fewshot/`，历史网格） | `scripts/train_eval_mghofnet.py`；`configs/mghofnet_*.yaml` |
| `msfmamba/` | MSFMamba | 项目已集成基线（Mamba，保留） | 旧基线（结果目录 `runs_fewshot/`，历史网格） | `scripts/train_eval_msfmamba.py`；`configs/msfmamba_*.yaml` |
| `softformer/` | SoftFormer | 项目已集成基线 | 旧基线（结果目录 `runs_fewshot/`，历史网格） | `scripts/train_eval_softformer.py`；`configs/softformer_*.yaml` |
| `dffnet/` | DFFNet（GRSL 2025，HSI–SAR/LiDAR） | [vendor/DFFNet-original](../vendor/DFFNet-original) | **CPU 合成数据验证通过、未完成正式实验**（无精度结果） | `scripts/train_eval_dffnet.py`；`configs/dffnet_*.yaml` |
| `m2heat/` | M2Heat（TCSVT 2026） | [vendor/M2Heat-original](../vendor/M2Heat-original) | **CPU 合成数据验证通过、未完成正式实验**（无精度结果） | `scripts/train_eval_m2heat.py`；`configs/m2heat_*.yaml` |
| `mft/` | MFT（TGRS 2023） | [vendor/MFT-original](../vendor/MFT-original) | **CPU 合成数据验证通过、未完成正式实验**（无精度结果） | `scripts/train_eval_mft.py`；`configs/mft_*.yaml` |
| `dcmnet/` | DCMNet（TGRS 2025） | [vendor/DCMNet-original](../vendor/DCMNet-original) | **仅源码 + 未适配说明**，不可运行 | — |
| `hslinets/` | HSLiNets（2024–2025） | [vendor/HSLiNets-original](../vendor/HSLiNets-original) | **仅源码 + 未适配说明**，不可运行 | — |

## 共享实现

- [`_hsi_adapters/`](_hsi_adapters)：三个第三方适配基线（dffnet/m2heat/mft）的共享模型实现（`dffnet_patch.py`、`m2heat_patch.py`、`mft_patch.py`），直接加载 `vendor/` 中未改动的上游核心模块。**它不是可训练基线**，不要按模型名引用。
- [`split.py`](split.py)：固定训练样本数划分逻辑，被各基线 `dataset.py` 复用。

## 新增基线时必须做

所有 agent 必须先读 [项目适配 skill](../.agents/skills/msi-sar-baseline-adaptation/SKILL.md) 与 [模型适配规则](../docs/models/baseline-adaptation.md)，在引入任务内主动完成输入、模型接口、训练／评估和配置适配。仅 vendor 收录不算集成完成；纯调研任务不触发实现。

1. 补齐 `model.py / train.py / eval.py / infer.py / dataset.py / io.py / metrics.py / README.md`；
2. 在本表和 [docs/models/README.md](../docs/models/README.md) 登记；
3. 在 `configs/` 增加配置并在 README 写明来源、预训练、输入假设与已知偏差；
4. 标注验证状态：**未运行验证**的基线不得写入主结果表。
