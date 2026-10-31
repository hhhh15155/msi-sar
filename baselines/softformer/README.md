# SoftFormer Baseline

> **2026-10-31 当前流程**：所有后续运行统一使用 MGHOFNet 训练协议 `mghofnet-v1`，公共实现为 `baselines/{protocol,data,training,prediction}.py`；优化、补零、增强和随机性设置以[当前实验协议](../../docs/experiments/README.md)为准。已完成实验仍属于各自保存的旧配置，本文此前的验证/运行记录不能视为新协议精度。此次只改代码，不启动正式实验；历史重评估按原填充处理。

Current project navigation: [models](../../docs/models/README.md), [datasets](../../docs/datasets/README.md), and [experiment protocol](../../docs/experiments/README.md). The input patch and internal `model_img_size` are different settings; check both in the chosen YAML.

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
