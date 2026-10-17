# 实验入口、协议与结果解释

以下按当前脚本与配置整理。本轮只改文档，没有安装依赖、加载模型、训练、评估或重新计算 OA。运行条件尚未在 server110 完整验证。

## 已确定的主实验设置

用户于 2026-10-17 确定：**YRD、YRD2509NEW 默认以每类 50 个训练样本为主实验；GRSS07 以现有 custom 配置为主实验。** 这不是每个数据集总共 50 个样本。

| 数据集 | 主实验每类训练数量 | 训练总量 | 配置命名 | 专项说明 |
| --- | --- | --- | --- | --- |
| `yrd` | `[50,50,50,50,50,50,50,50]` | 400 | `configs/<model>_yrd_fs50.yaml` | [YRD](yrd.md) |
| `yrd2509new` | 9 类各 50 个 | 450 | `configs/<model>_yrd2509new_fs50.yaml` | [YRD2509NEW](yrd2509new.md) |
| `grss07` | `[674,1481,752,45,30]` | 2982 | `configs/<model>_grss07_custom.yaml` | [GRSS07](grss07.md) |

GRSS07 计数按 City Center、Residential Areas、Sparse Buildings、Water、Vegetation 排列。用户反馈已核对其他论文采用该计数；具体来源状态见 [GRSS07 文献依据说明](grss07.md)。

现有配置采用 5 个种子 `202201..202205`，正式汇报全部种子的均值与标准差。其他 shot 档位保留作样本量对照，不作为默认主实验。本文档更新没有改变 YAML、训练代码或调度器的默认网格；实际运行须显式选择上述配置，不能直接无筛选地启动全网格。

## 单个实验

在项目根目录执行，将模型名、数据标识和 shot 设为实际配置对应值。例如：

```bash
python scripts/train_eval_dfinet.py --config configs/dfinet_yrd_fs5.yaml
python scripts/train_eval_fuzzy_transformer.py --config configs/fuzzy_transformer_yrd2509new_fs50.yaml

python scripts/eval_fuzzy_transformer.py --run runs_fewshot/fs50/fuzzy_transformer/yrd2509new/run_001
python scripts/infer_fuzzy_transformer.py --run runs_fewshot/fs50/fuzzy_transformer/yrd2509new/run_001
```

`train_eval_*.py` 启动训练并按配置导出标注图；`eval_*.py` 读取已有运行目录；`infer_*.py` 单独执行整图滑窗推理，不应只为查看一次指标就运行整图预测。每个模型都另有 `scripts/train_<model>.py` 入口，具体包装参数应阅读对应脚本。

## 当前划分与训练规则

- 当前默认数据网格：`yrd`、`yrd2509new`、`grss07`。默认五个比较方法为 DFI-Net、FreKFuse、MGHOFNet、MSFMamba、SoftFormer；Fuzzy Transformer 需显式选择。
- 可用 uniform-shot 档位：`5, 10, 20, 50, 100, 150, 200`；其中 YRD、YRD2509NEW 以 50-shot 为主实验，GRSS07 以 custom 为主实验。`fixed_train_counts` 从每类按种子抽样，其余有标注像元为测试集，没有验证集；不是 episodic few-shot。
- 当前示例采用 200 epochs、5 个种子 `202201..202205`。这些是配置值，不是入口对任意配置的统一强制条件。
- 对已查看的基线训练路径，训练循环不逐 epoch 评估测试集，结束后测试最终权重。旧基线仍把该最终状态命名为 `model_best.pth`；Fuzzy Transformer 明确命名为 `model_last.pth`。不能由文件名推断存在验证集最优选模。
- 多数既有基线配置的训练 batch 为 128、最终测试／评估为 1024；Fuzzy Transformer 示例相应值为 128。推理另读 `infer.batch_size`，运行前应看具体 YAML。
- GRSS07 custom 使用 `[674,1481,752,45,30]` 的每类训练数量，不能把它与 uniform-shot 当成同一个划分。

## 网格与调度

[experiment_grid.py](../../scripts/experiment_grid.py) 只打印任务，不启动训练：

```bash
python scripts/experiment_grid.py --models fuzzy_transformer --datasets grss07
```

[run_all_experiments.sh](../../scripts/run_all_experiments.sh) 的默认配置为 `GPU_COUNT=2`、`PROCS_PER_GPU=2`，通过 `conda run -n gjc which python` 定位解释器，也可指定 `PYTHON`。这些是脚本默认值，不代表 server110 当前有两张空闲 GPU。先检查资源，再显式指定并发。

```bash
# Linux Bash：只预览队列，仍需可用的解释器与配置
DRY_RUN=1 MODELS_FILTER=fuzzy_transformer DATASETS_FILTER=grss07 GPU_COUNT=1 PROCS_PER_GPU=1 bash scripts/run_all_experiments.sh

# 明确要求运行实验后使用
MODELS_FILTER=fuzzy_transformer DATASETS_FILTER=grss07 GPU_COUNT=1 PROCS_PER_GPU=1 bash scripts/run_all_experiments.sh
```

[run_grss07_experiments.sh](../../scripts/run_grss07_experiments.sh) 限定 GRSS07，默认一 GPU、每 GPU 两任务；[run_grss07_custom_experiments.sh](../../scripts/run_grss07_custom_experiments.sh) 按模型顺序运行 custom 配置。custom 脚本没有通用调度器同样的 dry-run 分支，不能把 `DRY_RUN=1` 当成它的预览保障。

```bash
# 此命令实际训练，仅在需要时执行
MODELS_FILTER=fuzzy_transformer GPU_ID=0 bash scripts/run_grss07_custom_experiments.sh
```

通用网格对旧基线的完成判断为存在任一 `run_*/metrics.json`；对 Fuzzy Transformer 还检查 `completed: true`。旧版“只检查 run_001”的文档已归档。`scripts/analyze_fewshot_results.py` 对 Fuzzy Transformer 选最近完整 attempt；旧基线仍读取 `run_001`，不能默认其自动汇总了每次重跑。

## 结果目录

一般结果位于 `runs_fewshot/fs{shot}/{model}/{dataset}/run_###/`；GRSS07 custom 位于 `runs_fewshot/grss07_custom/{model}/grss07/run_###/`。外层 `run_###` 是一次实验 attempt，内层 `runs`／`run_index` 常表示其中的种子编号，两者不是同一个概念。

当前包装通常保存：

| 文件／目录 | 含义 |
| --- | --- |
| `config.yaml` | 本次运行配置 |
| `splits/run_<index>.npz` | 每种子的训练／测试标签划分 |
| `checkpoints/run_<index>/` | 对应种子的权重 |
| `metrics_run_<index>.json`、`metrics.json` | 单种子及汇总指标，字段以实际模型为准 |
| 标注预测图 | 展示有标签位置，图的覆盖范围不自动等于指标统计范围 |
| 整图预测图 | 单独推理产生，不能与保留测试集指标混为一谈 |

数据、权重、结果和日志默认不进 Git；复现记录应补充源码提交、配置快照、数据版本、种子与环境。

## 已确认的比较限制

1. **预处理不完全一致。** YRD 的共享读取器默认全局 min-max＋通道中心化，MGHOFNet／SoftFormer 独立读取器逐通道 min-max；边界 `reflect` 与 `constant` 也有差异。详见 [数据集说明](../datasets/README.md)。本轮记录差异，没有改变数据或重新跑分。
2. **不是空间留区评估。** 固定数量随机像元划分可能使训练与测试 patch 的空间邻域重叠；与跨区域、跨时相或空间隔离结果不能直接比较。
3. **整图统计不等于训练集拟合。** 当前归一化在划分前利用全图影像统计，未使用测试标签拟合，但仍不同于严格仅在训练数据上拟合预处理的设置。
4. **最佳种子绘图不等于总体精度。** 例如 FreKFuse 的旧包装在 `eval.run_index: best` 时按测试 OA 选择标注图使用的 run；MGHOFNet 旧说明也描述这种选择。Fuzzy Transformer 使用固定整数 run。报告均值、标准差和全部种子，不把旧包装选出的最佳 OA 当成总体表现。
5. **图的指标范围需逐模型核对。** Fuzzy Transformer 的 `eval_metrics_run_*.json` 已读取保存的测试划分；不能默认所有旧基线的标注图导出指标都只统计测试像元。训练路径最终测试指标与绘图再统计指标应分别标注。
6. **环境与预训练条件需明示。** MSFMamba 有额外 runtime／fallback 路径；现有 README 的 GPU 软件版本例子没有在本次服务器检查中验证。不把静态检查通过写成依赖可用或成功复现。

这些限制应在后续实验改进中分别处理。文档整理本身不改变当前基准的行为，也不产生新的性能结论。
