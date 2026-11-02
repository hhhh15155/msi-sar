# 实验说明

## 2026-10-23 状态变更

按用户要求，旧自研 V0–V8 版本线及其全部实验结果已删除：

- **代码/配置**：`models/main_classification.py`、`models/fuzzy_transformer.py`、`models/v6_ablation.py`、`models/{fdmf,msaf}_patch.py`、`baselines/fuzzy_transformer/`、`scripts/main_experiment.py` 及 V 线配置生成/发布脚本、`configs/{main_first_pass,paper_candidates,yrd_per_channel,v6_ablation}/` 均已删除；
- **结果**：server110 `runs/main_first_pass_20261017`、`runs/main_first_pass_20261017_superseded_bf16_1940`、`runs/v6_ablation_20261021_110`，server116 `runs/main_first_pass_20261017`、`runs/paper_candidates_20261018`、`runs/yrd_per_channel_20261017`、`runs/archive/` 已删除，**不保留归档**；
- **文档**：旧结果表、V6 说明、各批次实验方案页同批删除；历史记录只存在于 Git 提交。

因此本页不再包含旧 21／27 项结果与旧协议描述。

## 2026-11-02 模型删除

按用户要求，HSLiNets（第三方适配）连同代码、三份主实验配置、三个入口脚本、vendor 快照与结果整体删除，不保留归档；统一主实验由 **13 模型 × 3 数据集 = 39 项**减为 **12 模型 × 3 数据集 = 36 项**，HSLiNets 的 3 项取消、不再运行。其余 36 项已在 server110 按统一协议完成主实验（执行记录见 [2026-11-01 统一队列](../operations/2026-11-01-unified-queue.md) 与 [2026-11-02 删除记录](../operations/2026-11-02-hslinets-removal.md)）。

## 当前可用的实验入口

| 入口 | 状态 |
| --- | --- |
| `scripts/train_eval_{dfinet,frekfuse,mghofnet,msfmamba,softformer}.py` | 旧集成基线的训练 + 测试入口（代码保留，结果目录见下） |
| `scripts/train_eval_{dffnet,m2heat,mft,dcmnet}.py` | 新适配的 HSI–LiDAR 基线，CPU 合成数据验证通过、已按统一协议完成 110 主实验；见各 `baselines/<model>/README.md` |
| `scripts/eval_<model>.py`、`scripts/infer_<model>.py` | 已有 run 的评估与整图推理 |
| `scripts/experiment_grid.py` | 只打印旧基线网格，不启动训练 |
| `scripts/main_experiment.py` | **已删除**（原用于 V0–V8 队列） |

运行任何训练前必须先做 smoke test；**没有用户明确授权不启动实验**。

## 数据与协议（仍然有效）

- 三个数据集：`yrd`、`yrd2509new`、`grss07`，配置在 `configs/datasets/`。
- 主实验训练量：YRD 每类 50（8 类共 400）、YRD2509NEW 每类 50（9 类共 450）、GRSS07 custom `[674,1481,752,45,30]`（5 类共 2982）。
- 固定划分：按类别与种子抽固定训练数量，其余标注像元测试，无验证集；patch 11、训练 batch 128、测试 batch 1024、FP32，实际以各 YAML 为准。
- 2026-10-23 用户指定多次运行按测试 OA 最高者出图。四个新适配基线（DFFNet／M2Heat／MFT／DCMNet）当前 12 份配置均为 `num_runs: 5`、`seeds: [202201,202202,202203,202204,202205]`、`eval.run_index: best`；后续新 baseline 默认沿用这一协议。
- 2026-10-31 起所有模型统一为 MGHOFNet 流程 `mghofnet-v1`：每次运行训练 200 epochs，AdamW 学习率 `0.001`、权重衰减 `0.01`，epoch 后 CosineAnnealingLR（T_max=epochs、eta_min=`0.000001`）；普通交叉熵，无梯度裁剪；无预训练，训练结束评估最终 checkpoint（即使文件名是 `model_best.pth`）。
- 公共数据流水线：逐通道整图 min-max、constant 补零；50% 不变换、12.5% 水平翻转、12.5% 垂直翻转，三种旋转各约 8.33%，MSI/SAR 同步。cuDNN `deterministic=True`、`benchmark=False`。模型自身结构/宽度/Dropout 保留各自设计。
- 新运行保存有效配置、`dataset_config.yaml` 和 `training_protocol.json`；历史结果仍按原运行协议解释，不能因为仓库默认改变而改名为新协议结果。历史 reflect 模型重评估保留 reflect；MGHOFNet/SoftFormer 按旧保存 pad_mode；优先读取运行内的数据配置快照，没有历史快照时不能完整保证原预处理身份。
- 已完成实验不重跑；不用测试集选 epoch／超参数。最高测试 OA 仅用于选择展示的分类图，全部种子的测试指标和均值／标准差仍是正式结果；OA 并列时按配置顺序选首个运行，记录实际 run 和 seed，注明“按最高测试 OA 选图”。
- 汇总标准差沿用现有 `metrics.py` 的 `np.std` 默认值（`ddof=0`，总体标准差）；报告中注明口径，不混用样本标准差。
- 当前默认仅标注样本出图：用测试 OA 最高的模型推理全部有标注位置（含训练位置），未标注位置留背景。这是绘图推理，不重新计算准确率；保存 `labeled_map_info.json` 记录选图 run／seed／既有测试 OA，不能覆盖 `metrics.json` 或测试重评估的 `eval_metrics.json`。
- 测试与重评估只推理各次保存的测试划分并计算指标，不调用全图。全图是单独出图模式，`infer.save_full_map: false`；只有用户明确需要时使用 `scripts/infer_<model>.py --mode full`，默认 `--mode labeled`。

## 结果目录

THSGR 已接入相同主实验协议，三份配置为 `configs/thsgr_*.yaml`；状态与执行记录见 [THSGR 操作记录](../operations/2026-10-30-thsgr-adaptation.md)，模型细节见 [THSGR README](../../baselines/thsgr/README.md)。

旧基线：`runs_fewshot/fs{shot}/{model}/{dataset}/run_###/`（GRSS07 custom：`runs_fewshot/grss07_custom/{model}/grss07/run_###/`）。新适配基线的 `train.py` 沿用同一目录结构（`output_root` 默认 `runs_fewshot/fs50`），输出配置快照、划分、checkpoint 与指标。

## 比较限制（仍然适用）

1. **不是空间留区评估**：固定数量随机像元划分可能使训练/测试 patch 邻域重叠，不能与跨区域或空间隔离结果直接比较。
2. **整图统计不等于训练集拟合**：归一化在划分前使用整图无标签统计量，未用测试标签，但不是仅训练集拟合。
3. **无验证集**：训练结束评估最终 checkpoint；不能由文件名推断存在验证集选模。
4. **环境差异**：不同服务器、依赖版本与残留进程会影响可比性，跨端比较需注明执行环境。
5. **模型输入假设不同**：各基线的通道命名、patch 限制、预训练与特殊算子需逐模型核对 README。

GFSFN 的三数据集主实验配置为 `configs/gfsfn_*.yaml`，沿用本页协议；执行记录见 [GFSFN操作记录](../operations/2026-10-30-gfsfn-adaptation.md)。

S²ATNet的三数据集配置为 `configs/s2atnet_*.yaml`；沿用本页完整协议。执行记录见 [S²ATNet](../operations/2026-10-30-s2atnet-adaptation.md)。

## 统一流程入口

[项目训练 skill](../../.agents/skills/msi-sar-training-protocol/SKILL.md)固定执行步骤。公共实现为 `baselines/protocol.py`、`data.py`、`training.py`、`prediction.py`；全部 12 个模型入口复用，不另外维护优化器或数据增强。`python -s scripts/check_training_protocol.py` 只审计配置和调用链，不启动训练。变更记录见[统一训练记录](../operations/2026-10-31-unified-training.md)。
