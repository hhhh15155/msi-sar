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
| `DFFNet-original/` | `baselines/dffnet/` | HSI–LiDAR/SAR 适配基线（已按统一协议完成 110 主实验） |
| `M2Heat-original/` | `baselines/m2heat/` | HSI–LiDAR 适配基线（已按统一协议完成 110 主实验） |
| `MFT-original/` | `baselines/mft/` | notebook 重实现适配基线（已按统一协议完成 110 主实验） |
| `DCMNet-original/` | `baselines/dcmnet/` | HSI–LiDAR 适配基线（合成数据验证通过，已按统一协议完成 110 主实验） |

`HSLiNets-original/` 快照及其对应基线 `baselines/hslinets/` 已于 2026-11-02 按用户要求删除，不保留归档；历史审查记录见 [docs/research/2026-10-23-hsi-lidar-source-snapshot.md](../docs/research/2026-10-23-hsi-lidar-source-snapshot.md)。

## 规则

`THSGR-original/`：作者 `jqyang22/THSGR`，commit `dd2a0ef7148c05a1261c38ab94845d9ff7b879a0`，Apache-2.0。仅保存原始 Python、YAML、README 和 LICENSE，不包含数据／权重／图片。项目实现见 [THSGR](../baselines/thsgr/README.md)。

- 只复制代码与说明，**不复制数据集、权重、图片、PDF**；原始代码不做语义修改。
- 公开可读不等于有复用许可；每个仓库的许可证状态以对应说明为准（多数未见明确 LICENSE）。
- HSI–LiDAR 四个仓库（HSLiNets 快照已于 2026-11-02 按用户要求删除）的来源、commit、快照规则与适配状态见 [docs/research/2026-10-23-hsi-lidar-source-snapshot.md](../docs/research/2026-10-23-hsi-lidar-source-snapshot.md)。
- 新 vendor 快照必须在 `baselines/README.md` 与对应基线 README 中登记来源。

`GFSFN-original/`：作者 [Zzp-12/GFSFN](https://github.com/Zzp-12/GFSFN)，commit `065cb808870515b70c4980642895fa45da63a1af`；保存原始README及Python源码，SHA256见目录内 `SOURCE.json`。该提交未见明确仓库许可证，公开可读不代表任意复用许可。项目适配见 [GFSFN](../baselines/gfsfn/README.md)。

`S2ATNet-original/`：作者RSIP-NJUPT/S2ATNet，commit `020d738807e67812c5deea4e819d327c08c7993d`；保留原始Python和README及文件SHA256，未见明确代码许可证。适配见 [S²ATNet](../baselines/s2atnet/README.md)。
