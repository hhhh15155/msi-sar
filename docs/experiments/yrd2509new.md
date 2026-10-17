# YRD2509NEW 专项实验

当前数据为 [yrd2509new.yaml](../../configs/datasets/yrd2509new.yaml)：14 通道（MSI 10＋SAR 4）、9 类。不要与历史 12 通道 `yrd2509`／land-use 版本混用。数据和标签生成过程需要结合来源元数据及数据处理脚本追踪，本轮没有重建。

## 主实验设置（2026-10-17 用户确定）

默认以 **每类 50 个训练样本** 为主实验，9 类共 **450** 个训练像元，配置为 `configs/<model>_yrd2509new_fs50.yaml`。其他 shot 档位保留作样本量对照。

各类按种子随机抽样，其余有标注像元用于测试，没有独立验证集。正式实验沿用现有配置的 5 个种子 `202201..202205`，报告均值与标准差。Fuzzy Transformer 对应 [50-shot 配置](../../configs/fuzzy_transformer_yrd2509new_fs50.yaml)。

这里确定的是主实验选择，没有修改专项脚本的默认 shots。运行时显式传入 `--shots 50` 或选择上述 YAML，避免启动全部档位；完整协议与已有比较限制见 [实验说明](README.md)。

## 当前入口

[train_eval_yrd2509new_fewshot.py](../../scripts/train_eval_yrd2509new_fewshot.py) 默认依次运行 DFI-Net、FreKFuse、MGHOFNet、MSFMamba 和 SoftFormer；Fuzzy Transformer 已支持，但默认不启用。

```bash
# 实际训练：仅选当前模型、每类 50 个训练像元
python scripts/train_eval_yrd2509new_fewshot.py --models fuzzy_transformer --shots 50

# 实际训练：只选两个比较方法及两档样本数
python scripts/train_eval_yrd2509new_fewshot.py --models frekfuse mghofnet --shots 100 150
```

可选 shots 为 `5,10,20,50,100,150,200`，配置文件采用 `configs/<model>_yrd2509new_fs<shot>.yaml`，一般包含五种子。结果位于 `runs_fewshot/fs{shot}/{model}/yrd2509new/run_###/`。

若需要多任务调度，使用 [实验说明](README.md) 中的通用调度器并设置 `DATASETS_FILTER=yrd2509new`；不要直接不加筛选地调用脚本而误启动全部三数据集网格。

## 历史描述与待核查项

旧说明曾记录裸土类 571 个可信像元，因而 200-shot 后余 371 个测试像元。该数值来自历史文档，本轮没有读取当前标签文件重新计数，不能作为当前文件已核实统计。原文完整保存于 [历史实验说明](../archive/2026-10-17/yrd2509new-experiments.original.md)。

旧说明只列四个模型并称调度只检查 `run_001`，已不符合当前入口；当前完成判断规则见 [实验说明](README.md)。数据的逐通道物理来源、预处理、实际类数量与划分应在新的实验记录里明确。
