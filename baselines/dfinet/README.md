# DFI-Net Baseline

This folder contains the DFI-Net comparison method used by the MSI-SAR project.
The original paper code is preserved separately in `vendor/DFI-Net-original/`.

Run:

```bash
python scripts/train_dfinet.py --config configs/dfinet_yrd.yaml
python scripts/train_eval_dfinet.py --config configs/dfinet_yrd.yaml
python scripts/eval_dfinet.py --run runs/dfinet/yrd/run_001
python scripts/infer_dfinet.py --run runs/dfinet/yrd/run_001
```

The training path reports test metrics on labeled test samples. Full-scene
sliding-window inference is only run by `infer_dfinet.py`.

`train_eval_dfinet.py` writes `labeled_gt.tif/png` and
`labeled_prediction.tif/png`. `infer_dfinet.py` writes
`full_prediction.tif/png`.
