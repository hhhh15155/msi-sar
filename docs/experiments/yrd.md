# YRD 主实验

当前数据为 [yrd.yaml](../../configs/datasets/yrd.yaml)：14 通道（MSI 10＋SAR 4）、8 类。标签顺序和预处理差异见 [数据集说明](../datasets/README.md)。

当前归一化为 `per_channel_minmax`。2026-10-17 获用户授权，在 server116 物理 GPU 1 上通过 screen 运行 7 个版本的单种子修正实验；复用首轮 400 个训练像元及测试划分，其他训练参数不变。当前任务、结果切换与归档位置见 [YRD 归一化修正实验](yrd-per-channel-rerun.md)。下文五种子是后续汇报约定，不是本次队列。

## 主实验设置（2026-10-17 用户确定）

默认以 **每类 50 个训练样本** 为主实验：`[50,50,50,50,50,50,50,50]`，共 **400** 个训练像元。其他现有 shot 档位保留作样本量对照。

配置命名为 `configs/<model>_yrd_fs50.yaml`，例如 [DFI-Net](../../configs/dfinet_yrd_fs50.yaml)、[Fuzzy Transformer](../../configs/fuzzy_transformer_yrd_fs50.yaml)。各类按种子随机抽样，其余有标注像元用于测试，没有独立验证集；正式实验沿用现有配置的 5 个种子 `202201..202205`，报告均值与标准差。

```bash
# 实际训练命令，仅在获得具体实验执行授权后运行
python scripts/train_eval_fuzzy_transformer.py --config configs/fuzzy_transformer_yrd_fs50.yaml
```

本次仅明确文档中的主实验选择，没有改变 YAML 或调度器的默认全网格。运行时显式选择 50-shot 配置。预处理、空间邻域重叠与选模等现有比较限制见 [实验说明](README.md)；不能仅因所有模型使用 50-shot 就认定其全部比较条件一致。
