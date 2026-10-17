# YRD 归一化修正实验与 server116 部署

2026-10-17 用户确认：在 server116 的物理 GPU 1 上通过 screen 部署 YRD 的 7 个版本，每项只跑一次；全部完成后统一替换旧 YRD 结果。YRD2509NEW 和 GRSS07 共 14 项不重跑。

## 完成状态（2026-10-18 核验）

七项均完成，队列 `finished`，发布记录 `published`。当前目录 7 个新 YRD 与其他 14 项的指标 SHA 均与发布记录一致；旧 21 项完整保留在下述 archive。V6 OA 为 86.8682%，FCM–FFM 为 89.7420%；SAR-only 为 58.0278%（AA 37.8813%、Macro-F1 16.6422%），已非全图单类预测，但仍有两类没有预测输出，不能认定其类别平衡性能已解决。这些是单种子、随机像元划分结果。

下一批文献候选见 [FDMF／MSAF 六项实验](paper-candidates.md)。本页原部署命令作为追溯记录保留，不用更新后的源码重启旧实验。

## 修改依据与范围

旧 YRD SAR-only 的 2,440,123 个测试像元全部预测为第 8 类：OA 0.360924%、AA 12.5%、κ 0。原全局 min-max 后，SAR 训练通道标准差约为 `[1.30e-6,6.44e-6,5.96e-6,0.00134]`，光学数值尺度压缩了 SAR 的输入变化。已确认输入连接、权重更新、划分及指标计算；归一化是当前主要疑点，修正后的效果仍待实验验证。

唯一预处理修改：`configs/datasets/yrd.yaml` 从共享读取器默认的 `global_minmax_center` 明确改为 `per_channel_minmax`。这同时改变缩放和中心化语义，不能声称是单独某个操作的消融。统计量仍来自整幅无标签影像，未改成仅训练集拟合。

V0 MSI-only、V1 SAR-only、V2 concat、V3 FCM/FFM、V4 MAFI、V5 fuzzy、V6 current 共 7 项，网络结构不变。训练 YAML 从 `configs/main_first_pass/` 原样复制到 [configs/yrd_per_channel/](../../configs/yrd_per_channel/manifest.json)：seed `202201`、每类 50（共 400）、200 epochs、训练 batch 128、测试 batch 1024、FP32；patch、优化器、学习率、增强与最终 epoch 测试协议保持原值。没有增加多种子或样本量网格。

复用原划分文件 `data/yrd/splits/main_seed202201_train50.npz`，其 SHA256 必须为：

```text
842aac597d09cb62e169c217bdafd51534fae36ff729e8a117085852a9c3a994
```

## 运行位置与命令

- 主机／目录：server116，`/data/gjc/msi-sar`。
- Python：`/home/wwl/.conda/envs/gjc/bin/python`。有效 PyTorch 2.6.0+cu124 来自用户 site-packages；不禁用 user site，不安装或升级依赖。
- 物理 GPU 1（V100 32GB），`CUDA_DEVICE_ORDER=PCI_BUS_ID`，`CUDA_VISIBLE_DEVICES=1`；单 worker。启动时该卡已有约 2GB 的低负载进程，允许在空闲显存至少 20480 MiB 时开始下一项，不结束别人的进程。阈值只控制入队时机，不会抢占正在运行的任务。
- 110 的 RTX 3090 与 116 的 V100／软件版本不同，均使用 FP32，但不能承诺跨硬件逐位一致。

```bash
cd /data/gjc/msi-sar
CUDA_VISIBLE_DEVICES=1 MAIN_EXPERIMENT_WORKERS=1 /home/wwl/.conda/envs/gjc/bin/python scripts/main_experiment.py --manifest configs/yrd_per_channel/manifest.json prepare
CUDA_VISIBLE_DEVICES=1 MAIN_EXPERIMENT_WORKERS=1 /home/wwl/.conda/envs/gjc/bin/python scripts/main_experiment.py --manifest configs/yrd_per_channel/manifest.json launch
screen -r msi_yrd_pc_gpu1
```

使用 Ctrl+A、D 脱离 screen，不终止任务；SSH 断开不影响 screen 内队列。日志为 `experiment_logs/yrd_per_channel_20261017/queue.log`，状态及新结果位于 `runs/yrd_per_channel_20261017/`。失败不静默重试；任务运行中不改模型、配置、数据处理或训练脚本。

## 统一结果与历史保留

原 21 项约 1.14GB 已从 110 直接复制到 116，未下载到 Windows。110 的原始结果保留作历史副本；本次修正后的统一结果以 **116** 为准：

| 位置（均相对 server116 项目） | 含义 |
| --- | --- |
| `runs/yrd_per_channel_20261017/` | 新 7 项的训练、断点、预测与发布记录 |
| `runs/main_first_pass_20261017/` | 当前统一 21 项入口；发布前仍是旧快照 |
| `runs/archive/main_first_pass_20261017_before_yrd_pc/` | 发布时完整保留的旧 21 项快照 |
| `runs/main_first_pass_20261017/current_results.json` | 发布后各结果来源、修订与校验值 |

队列在新 7 项全部完成后，核对身份、权重校验值、原始划分、逐类数量、预测与混淆矩阵／指标一致性，再调用 [promote_yrd_results.py](../../scripts/promote_yrd_results.py)。先构造完整 21 项目录，保留其他 14 项，再归档旧目录并切换；两次改名之间若中断，可在同一修正队列恢复发布。旧结果不删除，部分新结果不提前冒充当前基准。不要用旧 21 项 manifest 重新启动训练。

发布目录采用文件硬链接（不可用时复制）节省空间；完成产物按只读方式保留，不手工覆盖权重或指标。指标以各实验 `metrics.json` 和重新生成的汇总为准，旧文档 OA 只是历史记录。

部署前在 116 完成 36 项检查，包括 GPU 1 FP32 前后向、断点续训、固定划分与发布中断恢复；检查用小合成数据，不是本次正式 7 项的性能结果。
