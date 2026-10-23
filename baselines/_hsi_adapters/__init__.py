"""HSI–LiDAR 第三方模型适配层（共享实现，不是可训练基线）。

三个适配基线 `baselines/{dffnet,m2heat,mft}/model.py` 都从这里导入具体模型：
- `dffnet_patch.py`：DFFNet（GRSL 2025）
- `m2heat_patch.py`：M2Heat（TCSVT 2026）
- `mft_patch.py`：MFT（TGRS 2023）

上游代码快照在 `vendor/<Name>-original/`，本目录只做通道/接口适配。
基线索引见 `baselines/README.md`。
"""
