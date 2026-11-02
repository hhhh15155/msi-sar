# 模型与代码入口

2026-10-23 状态：旧自研 V0–V8 版本线（`models/main_classification.py`、`FuzzyTransformer`、V6 消融、FDMF／MSAF patch、`scripts/main_experiment.py` 调度器）及其全部实验结果已按用户要求删除，不保留归档；历史记录只存在于 Git 提交与本页的删除说明中。本页只索引当前仓库中实际存在的模型入口。

## 当前模型

引入或修改 baseline 前必读 [模型适配规则](baseline-adaptation.md) 与 [项目 skill](../../.agents/skills/msi-sar-baseline-adaptation/SKILL.md)。

| 模型标识 | 核心实现／类 | 训练入口 | 模态参数 | patch 与适配要点 | 专用说明 |
| --- | --- | --- | --- | --- | --- |
| `dfinet` | [model.py](../../baselines/dfinet/model.py)：`DFINet` | [train.py](../../baselines/dfinet/train.py)；`scripts/train_eval_dfinet.py` | `spectral_channels`、`sar_channels` | 旧集成基线；构造接收 `patch_size`，按现有 YAML 使用 | [README](../../baselines/dfinet/README.md) |
| `frekfuse` | [models/frekfuse.py](../../models/frekfuse.py)：`FreKFuse`／`FreKFuseLite` | [train.py](../../baselines/frekfuse/train.py)；`scripts/train_eval_frekfuse.py` | `ms_channels`、`sar_channels` | 旧集成基线；few-shot 配置为 Lite、patch=11 | [README](../../baselines/frekfuse/README.md) |
| `mghofnet` | [model.py](../../baselines/mghofnet/model.py)：`MGHOFNet` | [train.py](../../baselines/mghofnet/train.py)；`scripts/train_eval_mghofnet.py` | `hsi_channels`、`aux_channels` | 旧集成基线；由既有 HSI–LiDAR 工程适配，保留历史变量名 | [README](../../baselines/mghofnet/README.md) |
| `softformer` | [model.py](../../baselines/softformer/model.py)：`SoftFormer` | [train.py](../../baselines/softformer/train.py)；`scripts/train_eval_softformer.py` | `hsi_channels`、`aux_channels` | 旧集成基线；内部 `model_img_size=8`，stem 会中心裁剪 | [README](../../baselines/softformer/README.md) |
| `msfmamba` | [model.py](../../baselines/msfmamba/model.py)：`MSFMamba` → 作者 `Syn_layer` | [train.py](../../baselines/msfmamba/train.py)；`scripts/train_eval_msfmamba.py` | `ms_channels`、`sar_channels` | 旧集成基线；Mamba／SSM，包装限定 patch=11、单层 | [README](../../baselines/msfmamba/README.md) |
| `dffnet`（第三方适配） | [model.py](../../baselines/dffnet/model.py)：`DFFNet` | [train.py](../../baselines/dffnet/train.py)；`scripts/train_eval_dffnet.py` | `spectral_channels`、`sar_channels` | 新适配基线，**CPU 合成数据验证通过、已按统一协议完成 110 主实验**；项目接口 `[B,1,C,H,W]`，2 个 DFFM stage，无预训练 | [README](../../baselines/dffnet/README.md) |
| `m2heat`（第三方适配） | [model.py](../../baselines/m2heat/model.py)：`M2Heat` | [train.py](../../baselines/m2heat/train.py)；`scripts/train_eval_m2heat.py` | `spectral_channels`、`sar_channels` | 新适配基线，**CPU 合成数据验证通过、已按统一协议完成 110 主实验**；热传导 + CFF，输出 softmax 取 log 作 logits | [README](../../baselines/m2heat/README.md) |
| `mft`（第三方适配） | [model.py](../../baselines/mft/model.py)：`MFT` | [train.py](../../baselines/mft/train.py)；`scripts/train_eval_mft.py` | `spectral_channels`、`sar_channels` | 新适配基线，**CPU 合成数据验证通过、已按统一协议完成 110 主实验**；notebook 重实现，谱核随通道数自适应 | [README](../../baselines/mft/README.md) |
| `dcmnet`（第三方适配） | [model.py](../../baselines/dcmnet/model.py)：`DCMNet` | [train.py](../../baselines/dcmnet/train.py)；`scripts/train_eval_dcmnet.py` | `spectral_channels`、`sar_channels` | 新适配基线，**合成数据验证通过、已按统一协议完成 110 主实验**；路由 Cell 要求 patch 奇数且 ≥9；谱核深度 1；无预训练 | [README](../../baselines/dcmnet/README.md) |

入口命名约定：`scripts/train_eval_<model>.py`（训练 + 测试指标 + 标签图）、`scripts/eval_<model>.py`（已有 run 的评估）、`scripts/infer_<model>.py`（整图推理）。训练与推理脚本只在用户明确授权后执行。

新增 **THSGR**：[`model.py`](../../baselines/thsgr/model.py) → `THSGR`，参数 `spectral_channels/sar_channels`，奇数 patch≥9；训练入口 `scripts/train_eval_thsgr.py`，三份主实验 YAML 为 `configs/thsgr_*.yaml`。来源、差异与验证状态见 [模型说明](../../baselines/thsgr/README.md)。

## 已删除的旧版本

旧 V0–V8（MSI-only、SAR-only、concat、FCM–FFM、MAFI、fuzzy、V6 当前模型、FDMF／MSAF 候选）连同 V6 消融、`scripts/main_experiment.py`、对应配置与 tests 已于 2026-10-23 删除；`baselines/fuzzy_transformer/` 同步移除。Git 历史中仍可检索这些文件，但当前代码库不再包含。

**HSLiNets（第三方适配）** 已于 2026-11-02 按用户要求整体删除：`baselines/hslinets/`、`baselines/_hsi_adapters/hslinets_patch.py`、`_hslinets_vendor.py`、三份主实验配置、`scripts/{train_eval,eval,infer}_hslinets.py`、`vendor/HSLiNets-original/` 快照与 110 结果均不保留归档；统一主实验由 13 模型减为 12 模型。删除记录见 [2026-11-02 操作记录](../operations/2026-11-02-hslinets-removal.md)。

## 来源与工程边界

- 第三方快照位于 `vendor/`（见 [vendor/README.md](../../vendor/README.md)）；公开源码与复用许可证需分别核对，多数未见明确 LICENSE。
- `models/` 目录当前只保留旧基线共享实现 `frekfuse.py`；`models/__init__.py` 不是统一训练注册器。核实调用链应从具体 `scripts/train_eval_*.py` 到 `baselines/<model>/train.py`，再到实际模型定义。
- 新适配基线已通过本地 CPU 合成前向／反向和 checkpoint 检查，并已在 110 按统一协议完成主实验；精度解释须以运行内 `metrics.json` 与协议身份为准。
- 不做未验证的模块堆叠；新增模型须同步更新本页、`baselines/README.md`、配置与专用 README。

新增 **GFSFN**：[`model.py`](../../baselines/gfsfn/model.py) → `GFSFN`，两路通道 `spectral_channels/sar_channels`，奇数 patch≥7（默认11）；训练入口 `scripts/train_eval_gfsfn.py`。模型机制、验证状态和运行命令见 [GFSFN README](../../baselines/gfsfn/README.md)。

新增 **S²ATNet**：[`model.py`](../../baselines/s2atnet/model.py) → `S2ATNet`，两路通道参数 `spectral_channels/sar_channels`，正奇数patch（默认11）、token数P²。来源和验证见 [模型说明](../../baselines/s2atnet/README.md)。

所有模型后续训练使用[统一训练 skill](../../.agents/skills/msi-sar-training-protocol/SKILL.md)，模型结构保留各自设计。
