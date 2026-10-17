# FreKFuse 基线

当前模型与命令索引见 [模型说明](../../docs/models/README.md)，划分、评估和结果边界见 [实验说明](../../docs/experiments/README.md)。原始说明已原文归档至 [旧版 README](../../docs/archive/2026-10-17/frekfuse-readme.original.md)。

实现由 [models/frekfuse.py](../../models/frekfuse.py) 的 `FreKFuse`／`FreKFuseLite` 提供；包装入口为本目录的 [train.py](train.py)。MSI／SAR 通道数由 `ms_channels`／`sar_channels` 指定；YRD 和 YRD2509NEW 使用 10＋4，GRSS07 使用 6＋1。

当前 few-shot 配置使用 `lite: true`、`patch_size: 11`。模型构造函数的默认 patch 为 32，但当前工程并未强制所有实验使用 32；不能从旧文档推断 patch=11 无法执行，也不能据此承诺小 patch 的频域性能。

## 训练、评估和推理

```bash
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs5.yaml
python scripts/eval_frekfuse.py --run runs_fewshot/fs5/frekfuse/yrd/run_001
python scripts/infer_frekfuse.py --run runs_fewshot/fs5/frekfuse/yrd/run_001
```

当前划分只有固定数量的训练像元与剩余测试像元，没有旧说明中列出的同等数量验证集。训练结束后保存的 `model_best.pth` 在当前实现中是最终 epoch 状态，文件名不代表使用验证集选出了最优 epoch。

`train_eval_frekfuse.py` 在 `eval.run_index: best` 时，按测试 OA 选择用于标注图导出的 run。这是已有展示行为，应报告全部种子的汇总结果，不应将最佳种子数值当成总体性能。全图滑窗预测通过 `infer_frekfuse.py` 单独执行。
