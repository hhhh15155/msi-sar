# 数据集说明

本页说明当前配置和读取代码，不代表本次重新核验了实际 `.mat` 内容、影像尺寸、采集日期或每类像元数量。数据文件按 `.gitignore` 留在各机器的 `data/` 中，不随源码同步。

## 当前默认实验数据

| 数据标识 | 配置 | 数据目录 | 总通道／两路分配 | 有效类别 | 归一化配置 |
| --- | --- | --- | --- | --- | --- |
| `yrd` | [yrd.yaml](../../configs/datasets/yrd.yaml) | `data/yrd` | 14；前 10 MSI＋后 4 SAR | 8 | 未显式指定；共享读取器默认 `global_minmax_center` |
| `yrd2509new` | [yrd2509new.yaml](../../configs/datasets/yrd2509new.yaml) | `data/yrd2509new` | 14；前 10 MSI＋后 4 SAR | 9 | `per_channel_minmax` |
| `grss07` | [grss07.yaml](../../configs/datasets/grss07.yaml) | `data/grss07` | 7；前 6 Landsat＋后 1 ERS SAR | 5 | `per_channel_minmax` |

GRSS07 的配置显示名及原项目说明对应 Landsat 1994 与 ERS 1994-10-03 的 IEEE GRSS DFC 2007 子集。本页沿用该来源描述；没有重新查询原比赛数据记录。图像转换入口为 [prepare_grss07.py](../../scripts/prepare_grss07.py)。

这些数据名称不能互换：历史文档中的 `yrd2509`、12 通道 land-use 版本与当前 14 通道 `yrd2509new` 不是同一个配置。

## 当前主实验训练样本配置

用户于 2026-10-17 确定以下配置为今后主实验设置；“50 样本”指每类 50 个训练像元。

| 数据集 | 每类训练样本 | 训练总量 | 专项说明 |
| --- | --- | --- | --- |
| YRD | 8 类各 50 个 | 400 | [YRD 主实验](../experiments/yrd.md) |
| YRD2509NEW | 9 类各 50 个 | 450 | [YRD2509NEW 主实验](../experiments/yrd2509new.md) |
| GRSS07 | `[674,1481,752,45,30]` | 2982 | [GRSS07 custom 主实验](../experiments/grss07.md) |

其他现有 shot 配置保留用于样本量对照。主实验设置与脚本默认全网格不同，运行时必须显式选择对应配置；划分、种子和报告规则见 [实验说明](../experiments/README.md)。

## 标签顺序

原始有效类别按 `1..K`，未定义标签为 `0`；读取后转换成 `0..K-1`，未定义像元成为 `-1`。类别名和数量以相应 YAML 为准，不按不同数据集的同名类别直接对齐标签编号。

| 配置 | 标签 1 至 K 的类别顺序 |
| --- | --- |
| YRD | Waterbody；Spartina alterniflora；Phragmites communis；Tamarix chinensis；Suaeda salsa；Tidal flat；Natural willow forest；Cultivated land |
| YRD2509NEW | Artificial facilities；Tamarix chinensis；River water；Suaeda salsa；Pond water；Phragmites communis；Bare soil；Natural willow forest；Tidal flat |
| GRSS07 | City Center；Residential Areas；Sparse Buildings；Water；Vegetation |

## 已保留的其他数据配置

| 配置 | 通道／类别 | 用途与边界 |
| --- | --- | --- |
| [yrd2509_landuse.yaml](../../configs/datasets/yrd2509_landuse.yaml) | 12／15 | 数据构建／历史 land-use 版本，不在当前默认三数据集网格中 |
| [yrd2509_landuse_9c.yaml](../../configs/datasets/yrd2509_landuse_9c.yaml) | 12／9 | 九类映射版本，不等于 `yrd2509new` |

本次不改变这些配置或重建数据。目录里存在同名数据文件也不代表其来源和版本已经核实。

## 文件格式与输入

- 数据描述 YAML 默认指定 `data.mat` 中的 `data`，以及 `label.mat` 中的 `label`。
- 逻辑输入是 `H×W×C` 影像和 `H×W` 标签。读取器会校验空间对应；共享读取器还校验通道数。
- 模态按配置顺序切分。部分包装采用 `spectral_channels`／`sar_channels`，部分采用 `hsi_channels`／`aux_channels`，部分采用 `ms_channels`／`sar_channels`；详见 [模型索引](../models/README.md)。这些历史变量名不改变 MSI–SAR 任务的实际模态。
- SAR 派生特征不能当成独立传感器观测。当前配置只给出通道数，没有完整的逐通道物理语义表；具体通道含义、单位、是否 dB 和生成公式仍需数据来源元数据确认。

## 预处理与划分的实际差异

共享路径为 [dfinet/dataset.py](../../baselines/dfinet/dataset.py)：DFI-Net、FreKFuse、MSFMamba 及 Fuzzy Transformer 复用其中的读取逻辑。归一化在划分前使用整幅无标签影像的统计量：

- `per_channel_minmax`：每个通道按该图全局最小／最大值缩放。
- `global_minmax_center`：整个影像立方体统一 min-max，再按全图通道均值中心化。
- `PatchDataset` 的空间边界采用 `reflect` 填充。

[MGHOFNet](../../baselines/mghofnet/dataset.py) 和 [SoftFormer](../../baselines/softformer/dataset.py) 有单独读取器，实际逐通道 min-max；其 `FusionPatchDataset` 默认边界为 `constant`，可由 `pad_mode` 指定。因此 YRD 不能仅因数据 YAML 相同就声称所有模型预处理一致，其他数据集也需核对填充方式。

划分由 [baselines/split.py](../../baselines/split.py) 按类别和种子随机选固定训练数量，剩余标注像元用于测试，没有独立验证集，也不是空间隔离划分。相邻 patch 可能覆盖重叠区域；这种设置的泛化含义与空间留区测试不同。详细报告规则见 [实验说明](../experiments/README.md)。

## GRSS07 自定义训练数量

各类训练数量为 `[674, 1481, 752, 45, 30]`，顺序与上述 GRSS07 标签一致。对应文件采用 `configs/<model>_grss07_custom.yaml`；Fuzzy Transformer 示例见 [配置](../../configs/fuzzy_transformer_grss07_custom.yaml)。

该配置已确定为 GRSS07 主实验，共 2982 个训练像元。用户于 2026-10-17 反馈已核对其他论文也使用这组计数；此处记录用户核对结果。本次未取得逐篇论文的配置页码／表号，因此不将其表述为助手独立核实或原比赛官方标准。文献与完整协议的记录位置见 [GRSS07 专项说明](../experiments/grss07.md)。

数据准备／检查命令只在用户要求时执行；本轮没有读取大数组、生成数据或修改标签。
