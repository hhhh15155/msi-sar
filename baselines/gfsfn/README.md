# GFSFN

GFSFN：配准 MSI–SAR 邻域 patch 输入，输出中心像素类别。采用项目现有 baseline 目录、共享数据配置和训练／测试／标注样本出图协议。

## 来源

- 论文：Multilevel Feature Gated Fusion Based Spatial and Frequency Domain Attention Network for Joint Classification of Hyperspectral and LiDAR Data，IEEE JSTARS 18:5960–5974，2025，DOI [10.1109/JSTARS.2025.3534286](https://doi.org/10.1109/JSTARS.2025.3534286)。
- 作者仓库：[Zzp-12/GFSFN](https://github.com/Zzp-12/GFSFN)，固定 commit `065cb808870515b70c4980642895fa45da63a1af`。
- 原始快照：[`vendor/GFSFN-original/`](../../vendor/GFSFN-original/)，源码原字节 SHA256 见 `SOURCE.json`；没有复制数据、权重或论文图片。
- 该提交未见明确仓库代码许可证，不能将公开可读表述为任意复用许可；论文的出版许可与代码许可分别处理。

## 核心与适配差异

[`model.py`](model.py) 接收 `[B,1,C_msi+C_sar,P,P]`，MSI在前、SAR在后，返回 `[B,num_classes]` logits。核心 [`gfsfn_patch.py`](../_hsi_adapters/gfsfn_patch.py) 直接继承固定快照 `newnet.py::GFSFN`，保留作者实际前向中的两路3D卷积、多层CMSF、MLGF门控、SFA频域与CTEM纹理分支、224维特征及全局池化分类。无Mamba，无预训练。

必要适配仅包括：

1. 使用作者原有 `band1/band2` 参数设置 MSI6＋SAR1、MSI10＋SAR4。原demo先PCA30，本项目直接使用共享数据配置中的真实通道，不做PCA30、不复制或插值波段。两个末级谱核分别覆盖本路全部真实通道；辅助特征不解释为LiDAR高度。
2. patch可配置；项目中心像素标签要求奇数，校验 `P>=7`。首层空间3×3使P减少2，CMSF连续两次空间3×3再减少4，得到P−6；默认11时为5×5。原demo默认patch12，本项目沿用公平比较patch11。
3. 原全局池化后无维度限定的 `squeeze()` 在B=1丢失batch维；在原始前向返回后恢复 `[B,num_classes]`，不改变分类计算。
4. 仅移除 `MixTrans.p`：作者注册但实际前向完全不使用的标量参数。其他核心模块保持原样。所有实际可训练参数在优化器构建前注册。
5. 不使用作者demo的数据读取／调度：demo导入 `util`，仓库文件为 `utils.py`；本项目复用现有数据与训练协议。作者500epochs/lr0.0001等示例参数不覆盖项目统一预算。

## 配置与入口

三份主实验配置：`configs/gfsfn_yrd_fs50.yaml`、`configs/gfsfn_yrd2509new_fs50.yaml`、`configs/gfsfn_grss07_custom.yaml`。共享数据配置见 `configs/datasets/`；YRD和YRD2509NEW每类50，GRSS07 custom。5 seeds 202201–202205，patch11，200epochs，train128/test1024，FP32，Adam lr0.001/wd0.0001，无预训练。

```bash
python scripts/train_eval_gfsfn.py --config configs/gfsfn_yrd_fs50.yaml
python scripts/eval_gfsfn.py --run <run目录>
python scripts/infer_gfsfn.py --run <run目录>
```

训练使用最终checkpoint（兼容文件名 `model_best.pth`），不以测试集选择epoch。全部5次完成后按最高测试OA选展示运行，默认推理全部有标注位置、未标注位置留背景；不重新计分。全图默认关闭，只有明确需要时另用 `infer_gfsfn.py --mode full`。所有种子的指标及总体标准差（ddof=0）保留。完整协议见[实验说明](../../docs/experiments/README.md)。

输出：YRD/YRD2509NEW为 `runs_fewshot/fs50/gfsfn/<dataset>/run_###/`；GRSS07为 `runs_fewshot/grss07_custom/gfsfn/grss07/run_###/`。记录配置与数据配置快照、种子和划分、Git身份、环境实际包位置、最终权重及指标。

## 验证状态

本地已有gjc环境的模型合成检查通过：两路6/1和10/4、5/8/9类、patch7/11/13、前向／反向、全部参数有限梯度、优化器覆盖、checkpoint重载；与作者原始前向在相同权重下数值一致；eval单样本、混批与逆序结果一致。本地该环境缺少h5py/sklearn，不安装依赖，完整数据链路在116现有环境验证。

116现有环境的7项GFSFN检查全部通过（含临时MAT完整一轮训练、测试／重评估混淆矩阵一致、标注图背景和指标文件不覆盖）。GPU1上FP32 train128/test1024检查通过，峰值已分配显存约866 MiB；真实GRSS07一轮smoke及标注图完成，6个数据MAT的SHA256匹配清单。尚未完成正式实验，不能宣称复现论文精度。执行记录见[GFSFN操作记录](../../docs/operations/2026-10-30-gfsfn-adaptation.md)。

116当前进程须显式选择已有Conda包：`PYTHONPATH=/home/wwl/.conda/envs/gjc/lib/python3.11/site-packages LD_LIBRARY_PATH=/home/wwl/.conda/envs/gjc/lib`，否则用户目录torch2.1.2会遮蔽Conda内torch2.6.0+cu118；未安装或升级依赖，运行身份记录真实包路径。

116完整仓库回归为35项全部通过、无跳过；独立只读审查未发现可确认的P1/P2/P3问题。
