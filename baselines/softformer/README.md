# SoftFormer Baseline

This baseline adapts `rl1024/SoftFormer` for the MSI-SAR baseline workflow.
The original files are preserved in `vendor/SoftFormer-original/`.

For YRD, the default channel split is:

- first 10 channels: optical/MSI branch
- last 4 channels: SAR branch

SoftFormer internally expects an even model image size divisible by 4. The
default config uses `patch_size: 11` and center crops features to
`model_img_size: 8` inside the network.

## Commands

```bash
python scripts/train_softformer.py --config configs/softformer_yrd.yaml
python scripts/train_eval_softformer.py --config configs/softformer_yrd.yaml
python scripts/eval_softformer.py --run runs/softformer/yrd/run_001
python scripts/infer_softformer.py --run runs/softformer/yrd/run_001
```
