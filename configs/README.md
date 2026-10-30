# 配置目录索引

`configs/` 保存当前配置与旧基线网格。2026-10-23 删除旧自研 V0–V8 后，V 线配置目录（`main_first_pass/`、`paper_candidates/`、`yrd_per_channel/`、`v6_ablation/`）已不存在。

## 数据集配置

- `configs/datasets/{yrd,yrd2509new,grss07}.yaml`：数据文件、通道数、类别名、归一化方式。所有训练配置都引用这里。
- `configs/datasets/` 是当前唯一的共享配置源；新增模型必须复用，不要复制数据集参数。

## 当前可用的模型配置

| 路径 | 内容 | 状态 |
| --- | --- | --- |
| `configs/{dffnet,m2heat,mft,dcmnet,hslinets}_{yrd_fs50,yrd2509new_fs50,grss07_custom}.yaml` | 新适配的 HSI–LiDAR 基线（5 模型 × 3 数据集 = 15 份） | **合成数据验证通过，未完成正式实验** |
| `configs/dfinet_*.yaml` | DFI-Net 旧基线（few-shot 网格） | 历史网格，非当前主实验清单 |
| `configs/frekfuse_*.yaml` | FreKFuse 旧基线（few-shot 网格） | 同上 |
| `configs/mghofnet_*.yaml` | MGHOFNet 旧基线（few-shot 网格） | 同上 |
| `configs/msfmamba_*.yaml` | MSFMamba 旧基线（few-shot 网格） | 同上 |
| `configs/softformer_*.yaml` | SoftFormer 旧基线（few-shot 网格） | 同上 |

旧基线配置每套 22 份，覆盖 `fs5…fs200` 的 few-shot 样本量网格，属于历史实验与辅助对照。

## 主实验数据量（仍然有效）

YRD 每类 50（8 类共 400）、YRD2509NEW 每类 50（9 类共 450）、GRSS07 custom `[674,1481,752,45,30]`（5 类共 2982）。说明见 [docs/experiments/README.md](../docs/experiments/README.md)。

### 新适配基线的主实验配置

`dffnet`、`m2heat`、`mft`、`dcmnet`、`hslinets` 的上述 15 份配置已按当前主实验协议对齐：

| 参数 | 当前设置 |
| --- | --- |
| 运行次数／随机种子 | `num_runs: 5`，`seeds: [202201,202202,202203,202204,202205]` |
| patch／训练轮数 | `11`／`200` |
| 训练／测试 batch | `128`／`1024` |
| 优化器 | 训练入口使用 Adam；学习率 `0.001`，权重衰减 `0.0001` |
| 精度 | 训练入口使用 FP32，无 AMP；不是由 YAML 中额外的精度字段控制 |
| 预处理／划分 | 复用共享数据集配置；整图逐通道 min-max，固定每类训练数量，其余标注像元测试 |
| MSI／SAR 通道数 | YRD、YRD2509NEW 显式设置为 `10/4`，GRSS07 为 `6/1` |
| checkpoint／出图运行 | 每次训练结束的最终 checkpoint；`eval.run_index: best`，选测试 OA 最高者出图 |
| 出图范围 | 默认全部标注位置（含训练位置），不重新计分；`infer.save_full_map: false`，全图仅按需显式调用 |

用户于 2026-10-23 指定恢复多次运行选最高测试 OA 出图，全部运行的测试指标与均值／标准差仍须保留，具体口径以 [实验说明](../docs/experiments/README.md#数据与协议仍然有效) 为准。旧基线历史网格保持原样，输出目录沿用各配置的 `runs_fewshot/` 路径。合成数据检查不代表已完成真实数据训练或精度验证；DCMNet、HSLiNets 也已适配，其偏差与阻塞见各自 README。后续引入必须执行 [项目 skill](../.agents/skills/msi-sar-baseline-adaptation/SKILL.md) 与 [模型适配规则](../docs/models/baseline-adaptation.md)。

## 维护约定

THSGR 主实验使用 `thsgr_{yrd_fs50,yrd2509new_fs50,grss07_custom}.yaml`，沿用上表 5 seeds、200 epochs 与测试／出图协议；入口与差异见 [THSGR](../baselines/thsgr/README.md)。

- 新增当前实验：新建独立配置并写清用途，不要和旧网格混在同一层；
- 配置生成脚本在 `scripts/generate_*_configs.py`，生成后仍以 YAML 和实际入口为准；
- 已运行过的配置不得修改（结果身份依赖配置 SHA256）。
