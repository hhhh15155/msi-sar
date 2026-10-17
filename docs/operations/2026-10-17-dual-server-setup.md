# server110／server116 双服务器运行说明

2026-10-17 用户授权增加 server116 运行副本、同步当前三数据集并检查 `gjc` 环境。Windows 本地仍是源码权威端；两台服务器均用于执行。本文取代“项目只部署在 110”的旧描述，不改变历史操作记录。

## 位置与环境

| 项目 | server110 | server116 |
| --- | --- | --- |
| SSH 别名 | `server110` | `server116` |
| 登录用户 | `fangyan` | `wwl` |
| 项目目录 | `/ssd/gaojianchun/msi-sar` | `/data/gjc/msi-sar` |
| Python | `/home/fangyan/.conda/envs/gjc/bin/python` | `/home/wwl/.conda/envs/gjc/bin/python` |
| 已知 PyTorch | 2.1.1+cu118 | 2.6.0+cu124 |
| 设备 | RTX 3090 24GB，一张 | V100-PCIE 32GB，两张 |
| 建议物理卡 | 0 | 1，执行前重新检查占用 |
| 后台工具 | screen | screen |

116 的有效 Python 为 3.11.13，NumPy 2.0.1、SciPy 1.16.0、h5py 3.14.0、PyYAML 6.0.3、scikit-learn 1.7.1、tqdm 4.69.0 均可导入。其 PyTorch 实际来自 `/home/wwl/.local/lib/python3.11/site-packages/torch`；部分依赖也使用用户级包。本次保留已有运行方式，没有安装或升级依赖。禁用 user-site 的试验会因缺少 `typing_extensions` 而导入失败，不能直接加 `PYTHONNOUSERSITE=1`。

116 的 CUDA 可用，编译架构包含 V100 对应的 `sm_70`。已在物理 GPU 1 上执行小规模 FP32 模型前后向、合成数据训练与中断恢复，不是仅检查 `torch.cuda.is_available()`。正式训练继续使用 FP32；没有因换服务器改变网络、学习率、batch、epoch 或损失。

GPU 1 曾报告 InfoROM 告警；检查时不可纠正 ECC 的 volatile／aggregate 计数均为 0，短时运算通过。这不等于已经做过长时间硬件稳定性检验。两卡均有其他进程，后续任务须重新检查，不擅自终止其他用户作业。

## 代码与数据同步

源码流程：本地修改并提交 → 推送 GitHub `origin/main` → 两台服务器快进到相同提交。远端无法稳定连接 GitHub 时，可从本地通过 SSH 发送已推送提交的 Git bundle，再做 fast-forward；不使用强制覆盖。

三个数据集位于各自项目根目录的 `data/` 下：

```text
data/
  yrd/          data.mat, label.mat
  yrd2509new/   data.mat, label.mat
  grss07/       data.mat, label.mat
```

6 个文件总计 **314,544,305 bytes（约 315 MB）**。来源为 Windows 本地原始文件，此前已与 110 比对，本次传入 116 后逐文件核对 SHA256 一致。完整文件大小和哈希见 [数据清单](2026-10-17-dual-server-data-manifest.json)。数据保持 Git 忽略；不提交 `.mat`、权重或结果。

116 实际读取结果：YRD `2048×2048×14`／8 类，YRD2509NEW `1024×1024×14`／9 类，GRSS07 `787×787×7`／5 类。读取器检查通过，归一化后数值有限，类别计数与本地／110 的既有审查一致。

本轮未复制 110 的 1.14GB 正式实验输出或旧 BF16 预启动档案，也未将它们下载到 Windows。110 上原 21 项完成结果保留原位；116 的测试输出是临时合成数据，不属于正式实验。

## 两处运行兼容修正

1. **续训 checkpoint。** PyTorch 2.6 的默认 `weights_only` 行为会拒绝包含 NumPy 随机状态的本项目续训文件，已复现。只对本项目产生、对应身份文件已检查的 `resume.pth` 显式使用 `weights_only=False`；其他第三方文件不适用。模型权重格式与架构不变。
2. **指定 GPU 与并发。** `launch` 不再硬编码 GPU 0，而是继承单卡 `CUDA_VISIBLE_DEVICES`；脚本在导入 Torch 前统一设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，直接执行 queue／train 入口也遵守同一物理枚举顺序。显卡等待查询使用同一 `nvidia-smi --id`，GPU 0 忙不会误挡 GPU 1。`MAIN_EXPERIMENT_WORKERS` 可以降低到 1，不能超过清单上限；任务内部 `cuda:0` 表示映射后的可见卡。

运行环境记录补充主机、物理 GPU 选择和实际 PyTorch／NumPy 模块路径。不同服务器／库版本不保证浮点结果逐位一致，不能因此混写成同一运行环境。检查中定位到 FP32 注意力冗余 K-bias 的微小舍入差异；恢复检查对其设限，其余权重、功能 logits 及预测分别检查，不用放宽所有参数的比较来掩盖错误。

## 命令与执行边界

只读检查：

```bash
# server110
cd /ssd/gaojianchun/msi-sar
/home/fangyan/.conda/envs/gjc/bin/python scripts/main_experiment.py status

# server116
cd /data/gjc/msi-sar
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /home/wwl/.conda/envs/gjc/bin/python -m unittest discover -s tests
nvidia-smi
```

后续明确授权训练、准备好新的清单后，在 116 使用如下形式（会实际启动，本文建档没有执行）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MAIN_EXPERIMENT_WORKERS=1 \
  /home/wwl/.conda/envs/gjc/bin/python scripts/main_experiment.py --manifest configs/<approved-manifest>.json launch
```

`<approved-manifest>` 是说明占位，不能原样执行。切勿直接启动原 21 项清单，以免重复已经完成的实验。新的 YRD 七版本应先明确逐通道归一化、共用原划分、新缓存、旧结果备份和最终替换规则，再部署。

screen 会话名来自任务清单；`screen -ls` 查看，通过 `screen -r <session>` 进入，按 Ctrl+A、D 脱离。SSH 断开不会结束 screen；服务器重启后仍需人工恢复，不声称开机自动恢复。

## 后续部署更新

本页环境准备完成之后，用户已确认在 116 物理 GPU 1 通过 screen 部署 YRD 的 7 项逐通道归一化修正实验。新清单、原划分、资源阈值、旧结果归档和统一结果入口以 [YRD 修正实验](../experiments/yrd-per-channel-rerun.md) 为准。下文“未重跑”是本页初次检查时的历史状态。

## 本轮验证范围

- server116：30 项检查全部通过，涵盖 7 版本／两组输入通道的 GPU 检查及合成数据训练、保存、恢复、最终预测核对。
- server110：同套 30 项检查通过，其中 2 项 GPU 检查因该卡有其他训练而跳过，未占用该卡复测。
- 三数据集：6 个文件大小、SHA256、数组读取、类别和有限值检查通过。
- 本轮仅完成双服务器准备；未修改 YRD 归一化配置，未覆盖任何正式结果，未启动 YRD 七版本重跑。
- 旧 MSFMamba 等其他模型的第三方 CUDA 扩展／完整训练依赖不属于本次验证，不能据此声称所有历史基线均已在 116 复现。
