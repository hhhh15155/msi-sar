# DFI-Net Baseline

> **2026-10-31 当前流程**：所有后续运行统一使用 MGHOFNet 训练协议 `mghofnet-v1`，公共实现为 `baselines/{protocol,data,training,prediction}.py`；优化、补零、增强和随机性设置以[当前实验协议](../../docs/experiments/README.md)为准。已完成实验仍属于各自保存的旧配置，本文此前的验证/运行记录不能视为新协议精度。此次只改代码，不启动正式实验；历史重评估按原填充处理。

Current project navigation: [models](../../docs/models/README.md), [datasets](../../docs/datasets/README.md), and [experiment protocol](../../docs/experiments/README.md). Use the selected YAML for channel counts and patch size. A historical `model_best.pth` filename does not imply validation-based epoch selection.

This folder contains the DFI-Net comparison method used by the MSI-SAR project.
The original paper code is preserved separately in `vendor/DFI-Net-original/`.

Run:

```bash
python scripts/train_eval_dfinet.py --config configs/dfinet_yrd_fs50.yaml
python scripts/eval_dfinet.py --run runs/dfinet/yrd/run_001
python scripts/infer_dfinet.py --run runs/dfinet/yrd/run_001
```

The training path reports test metrics on labeled test samples. Full-scene
sliding-window inference is only run by `infer_dfinet.py`.

`train_eval_dfinet.py` writes `labeled_gt.tif/png` and
`labeled_prediction.tif/png`. `infer_dfinet.py` writes
`full_prediction.tif/png`.
