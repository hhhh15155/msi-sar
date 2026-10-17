# GRSS07 custom 主实验

当前数据为 [grss07.yaml](../../configs/datasets/grss07.yaml)：7 通道（Landsat MSI 6＋ERS SAR 1）、5 类。具体数据子集、标签和预处理见 [数据集说明](../datasets/README.md)。

## 主实验设置（2026-10-17 用户确定）

默认使用现有 **custom** 配置 `[674,1481,752,45,30]`，共 **2982** 个训练像元。

| 标签顺序 | 类别 | 训练数量 |
| --- | --- | --- |
| 1 | City Center | 674 |
| 2 | Residential Areas | 1481 |
| 3 | Sparse Buildings | 752 |
| 4 | Water | 45 |
| 5 | Vegetation | 30 |
| 合计 | — | 2982 |

配置命名为 `configs/<model>_grss07_custom.yaml`，例如 [DFI-Net](../../configs/dfinet_grss07_custom.yaml)、[Fuzzy Transformer](../../configs/fuzzy_transformer_grss07_custom.yaml)。各类按种子随机抽样，其余有标注像元用于测试，没有独立验证集；正式实验沿用现有配置的 5 个种子 `202201..202205`，报告均值与标准差。

现有 uniform-shot 档位保留作样本量对照。custom 不是每类相同数量，不能与 uniform 50-shot 当成相同训练预算或划分。

```bash
# 实际训练命令，仅在获得具体实验执行授权后运行
python scripts/train_eval_fuzzy_transformer.py --config configs/fuzzy_transformer_grss07_custom.yaml
```

本次只明确文档中的主实验选择，没有改变 YAML、调度脚本或已保存划分。现有比较限制见 [实验说明](README.md)。

## 文献依据与核对状态

用户于 **2026-10-17** 明确反馈：“我刚核对过了，其他论文都是用这个”，指上述 `[674,1481,752,45,30]`。项目据此将其记录为**用户已核对的文献训练样本配置**，并确定为 GRSS07 主实验设置。

该反馈更新了此前检索中“未核实其他论文是否采用这组计数”的项目记录；核对来源是用户，本次助手没有新增逐篇独立核实。当前对话未提供具体论文清单、表号或页码，后续论文写作时补充这些引用及各论文的验证集、测试集、重复次数和输入模态设置。此处的计数一致性不自动意味着完整实验协议一致，也不将其称为原比赛官方划分。
