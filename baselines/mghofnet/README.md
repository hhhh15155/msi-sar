# MGHOFNet Baseline

> **2026-10-31 当前流程**：所有后续运行统一使用 MGHOFNet 训练协议 `mghofnet-v1`，公共实现为 `baselines/{protocol,data,training,prediction}.py`；优化、补零、增强和随机性设置以[当前实验协议](../../docs/experiments/README.md)为准。已完成实验仍属于各自保存的旧配置，本文此前的验证/运行记录不能视为新协议精度。此次只改代码，不启动正式实验；历史重评估按原填充处理。

Current project navigation: [models](../../docs/models/README.md), [datasets](../../docs/datasets/README.md), and [experiment protocol](../../docs/experiments/README.md). The `best run by OA` selection below refers to the existing map-export choice, not a valid replacement for reporting all seeds. All new runs reuse the shared MGHOFNet data/training pipeline; historical runs retain their recorded protocol.

This baseline is adapted from the previous HSI-LiDAR classification code in
`D:\gaojianchun\fenlei\ttt`. The original snapshot is kept under
`vendor/MGHOFNet-original/`.

For the YRD MSI-SAR data, the default input split is:

- first 10 channels: MSI/HSI branch
- last 4 channels: SAR/auxiliary branch

Edit `configs/mghofnet_yrd.yaml` if another dataset uses different channel
counts.

## Commands

```bash
python scripts/train_mghofnet.py --config configs/mghofnet_yrd.yaml
python scripts/train_eval_mghofnet.py --config configs/mghofnet_yrd.yaml
python scripts/eval_mghofnet.py --run runs/mghofnet/yrd/run_001
python scripts/infer_mghofnet.py --run runs/mghofnet/yrd/run_001
```

`train_eval_mghofnet.py` trains the runs, tests labeled test samples, chooses the
best run by OA, and writes:

```text
labeled_gt.tif
labeled_gt.png
labeled_prediction.tif
labeled_prediction.png
```

`infer_mghofnet.py` runs full-scene sliding-window inference and writes:

```text
full_prediction.tif
full_prediction.png
```
