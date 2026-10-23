# 第三方代码快照（vendor）

本目录保存**上游第三方仓库的只读代码快照**，用于审查与适配参照，不是项目当前方法，也不代表已成功复现。

## 快照列表

| 目录 | 对应基线 | 说明 |
| --- | --- | --- |
| `DFI-Net-original/` | `baselines/dfinet/` | 项目已集成基线 |
| `FrekFuse-original/` | `baselines/frekfuse/` | 项目已集成基线 |
| `MGHOFNet-original/` | `baselines/mghofnet/` | 项目已集成基线 |
| `MSFMamba-original/` | `baselines/msfmamba/` | 项目已集成基线（Mamba，保留） |
| `SoftFormer-original/` | `baselines/softformer/` | 项目已集成基线 |
| `DFFNet-original/` | `baselines/dffnet/` | HSI–LiDAR/SAR 适配基线（未运行验证） |
| `M2Heat-original/` | `baselines/m2heat/` | HSI–LiDAR 适配基线（未运行验证） |
| `MFT-original/` | `baselines/mft/` | notebook 重实现适配基线（未运行验证） |
| `DCMNet-original/` | `baselines/dcmnet/` | 仅收录源码，未适配 |
| `HSLiNets-original/` | `baselines/hslinets/` | 仅收录源码，未适配 |

## 规则

- 只复制代码与说明，**不复制数据集、权重、图片、PDF**；原始代码不做语义修改。
- 公开可读不等于有复用许可；每个仓库的许可证状态以对应说明为准（多数未见明确 LICENSE）。
- HSI–LiDAR 五个仓库的来源、commit、快照规则与适配状态见 [docs/research/2026-10-23-hsi-lidar-source-snapshot.md](../docs/research/2026-10-23-hsi-lidar-source-snapshot.md)。
- 新 vendor 快照必须在 `baselines/README.md` 与对应基线 README 中登记来源。
