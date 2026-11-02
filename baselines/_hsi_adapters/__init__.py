"""HSI–LiDAR 第三方模型适配层（共享实现，不是可训练基线）。

四个适配基线 `baselines/{dffnet,m2heat,mft,dcmnet}/model.py` 都从这里导入具体模型：
- `dffnet_patch.py`：DFFNet（GRSL 2025）
- `m2heat_patch.py`：M2Heat（TCSVT 2026）
- `mft_patch.py`：MFT（TGRS 2023）
- `dcmnet_patch.py`：DCMNet（TGRS 2025）——vendor 路由/Cell/分类头原样使用，重写光谱与 SAR 编码器

HSLiNets（2024–2025）适配曾在此实现（`hslinets_patch.py` + `_hslinets_vendor.py`），已于 2026-11-02 按用户要求连同基线、配置、脚本、vendor 快照与结果一并删除。

上游代码快照在 `vendor/<Name>-original/`，本目录只做通道/接口/形状适配。
基线索引见 `baselines/README.md`。
"""

# gfsfn_patch.py: GFSFN native-channel wrapper; see baselines/gfsfn/README.md.

# s2atnet_patch.py: native S2ATNet with per-modality channels and P² tokens.
