# MGHOFNet Baseline

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
