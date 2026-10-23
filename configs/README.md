# 配置目录索引

`configs/` 同时保存**当前有效配置**与历史实验网格。按下面的分类读取，避免把历史网格当成当前主实验。

## 数据集配置

- `configs/datasets/{yrd,yrd2509new,grss07}.yaml`：三个数据集的数据文件、通道数、类别名、归一化方式。所有训练配置都引用这里。

## 当前有效清单（按实验批次）

| 路径 | 内容 | 状态 |
| --- | --- | --- |
| `configs/main_first_pass/manifest.json` | 第一轮主实验 7 版本 × 3 数据集（21 项） | 已完成，不重跑 |
| `configs/paper_candidates/manifest.json` | 第二批 6 项（`v7_fdmf_fixed`、`v8_msaf_fixed`） | 已完成，不重跑 |
| `configs/v6_ablation/manifest.json` | 剩余 V6 消融 6 项 | 以 [next-experiments.md](../docs/experiments/next-experiments.md) 为准 |
| `configs/yrd_per_channel/manifest.json` | YRD 逐通道归一化修正 7 项 | 已完成，不重跑 |
| `configs/{dffnet,m2heat,mft}_*.yaml` | 三个第三方适配基线（3 数据集 × 3 模型 = 9 份） | **未运行验证**，见 [baselines/README.md](../baselines/README.md) |

## 历史／辅助网格

- `configs/dfinet_*.yaml`、`configs/frekfuse_*.yaml`、`configs/fuzzy_transformer_*.yaml`、`configs/mghofnet_*.yaml`、`configs/msfmamba_*.yaml`、`configs/softformer_*.yaml`：各 22 份，覆盖 `fs5…fs200` 的 few-shot 样本量网格；属于历史实验与辅助对照，**不是当前主实验清单**。
- 主实验实际采用每类固定训练样本（YRD 50/类、YRD2509NEW 50/类、GRSS07 custom `[674,1481,752,45,30]`），见 [docs/experiments/README.md](../docs/experiments/README.md)。

## 维护约定

- 新增当前实验：新建独立子目录 + `manifest.json`，不要和旧网格混在同一层；
- 配置生成脚本在 `scripts/generate_*_configs.py`，生成后仍以 YAML 和实际入口为准；
- 已完成实验的配置不得修改（身份文件会因配置 SHA256 变化而失效）。
