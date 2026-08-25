# FreKFuse Baseline

Frequency Kolmogorov-Arnold Fusion Network for MSI-SAR land cover
classification.

For the YRD MSI-SAR data, the default input split is:

- first 10 channels: MSI branch
- last 4 channels: SAR branch

Edit `configs/frekfuse_yrd.yaml` if another dataset uses different channel
counts.

## Commands

```bash
python scripts/train_frekfuse.py --config configs/frekfuse_yrd.yaml
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd.yaml
python scripts/eval_frekfuse.py --run runs/frekfuse/yrd/run_001
python scripts/infer_frekfuse.py --run runs/frekfuse/yrd/run_001
```

`train_eval_frekfuse.py` trains the runs, tests labeled test samples, chooses the
best run by OA, and writes:

```text
labeled_gt.tif
labeled_gt.png
labeled_prediction.tif
labeled_prediction.png
```

`infer_frekfuse.py` runs full-scene sliding-window inference and writes:

```text
full_prediction.tif
full_prediction.png
```

## Few-shot

Few-shot configs switch to the FreKFuseLite variant (smaller embed_dim, lower
spline_order, higher dropout) to reduce overfitting on limited samples.

```bash
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs5.yaml
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs10.yaml
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs20.yaml
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs50.yaml
```

```text
fs5:  train/class=5,  val/class=5,  train total=40,  val total=40
fs10: train/class=10, val/class=10, train total=80,  val total=80
fs20: train/class=20, val/class=20, train total=160, val total=160
fs50: train/class=50, val/class=50, train total=400, val total=400
```

Few-shot output folders:

```text
runs_fewshot/fs5/frekfuse/yrd/run_001/
runs_fewshot/fs10/frekfuse/yrd/run_001/
runs_fewshot/fs20/frekfuse/yrd/run_001/
runs_fewshot/fs50/frekfuse/yrd/run_001/
```
