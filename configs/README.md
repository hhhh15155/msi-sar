# 配置目录索引

`configs/` 保存当前配置与旧基线网格。2026-10-23 删除旧自研 V0–V8 后，V 线配置目录（`main_first_pass/`、`paper_candidates/`、`yrd_per_channel/`、`v6_ablation/`）已不存在。

## 数据集配置

- `configs/datasets/{yrd,yrd2509new,grss07}.yaml`：数据文件、通道数、类别名、归一化方式。所有训练配置都引用这里。
- `configs/datasets/` 是当前唯一的共享配置源；新增模型必须复用，不要复制数据集参数。

## 当前可用的模型配置

| 路径 | 内容 | 状态 |
| --- | --- | --- |
| `configs/{dffnet,m2heat,mft}_{yrd_fs50,yrd2509new_fs50,grss07_custom}.yaml` | 新适配的 HSI–LiDAR 基线（3 模型 × 3 数据集 = 9 份） | **未运行验证** |
| `configs/dfinet_*.yaml` | DFI-Net 旧基线（few-shot 网格） | 历史网格，非当前主实验清单 |
| `configs/frekfuse_*.yaml` | FreKFuse 旧基线（few-shot 网格） | 同上 |
| `configs/mghofnet_*.yaml` | MGHOFNet 旧基线（few-shot 网格） | 同上 |
| `configs/msfmamba_*.yaml` | MSFMamba 旧基线（few-shot 网格） | 同上 |
| `configs/softformer_*.yaml` | SoftFormer 旧基线（few-shot 网格） | 同上 |

旧基线配置每套 22 份，覆盖 `fs5…fs200` 的 few-shot 样本量网格，属于历史实验与辅助对照。

## 主实验数据量（仍然有效）

YRD 每类 50（8 类共 400）、YRD2509NEW 每类 50（9 类共 450）、GRSS07 custom `[674,1481,752,45,30]`（5 类共 2982）。说明见 [docs/experiments/README.md](../docs/experiments/README.md)。

## 维护约定

- 新增当前实验：新建独立配置并写清用途，不要和旧网格混在同一层；
- 配置生成脚本在 `scripts/generate_*_configs.py`，生成后仍以 YAML 和实际入口为准；
- 已运行过的配置不得修改（结果身份依赖配置 SHA256）。
