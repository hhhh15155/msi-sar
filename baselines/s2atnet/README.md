# S²ATNet

S²ATNet：两路配准MSI–SAR邻域patch输入，预测中心像素类别。模型标识／目录为 `s2atnet`，采用本项目现有baseline架构。

## 来源与核心

- Selective Spectral-Spatial Aggregation Transformer for Hyperspectral and LiDAR Classification，IEEE GRSL，2025卷期；DOI [10.1109/LGRS.2024.3521976](https://doi.org/10.1109/LGRS.2024.3521976)。
- 作者仓库：[RSIP-NJUPT/S2ATNet](https://github.com/RSIP-NJUPT/S2ATNet)，固定commit `020d738807e67812c5deea4e819d327c08c7993d`。
- 原始源码位于 [`vendor/S2ATNet-original/`](../../vendor/S2ATNet-original/)，`SOURCE.json`保存文件SHA256；目录局部 `-text` 保留原始换行字节。未包含数据、权重、图片或IDE文件。
- 该提交未见明确代码许可证，公开源码不代表任意复用许可；论文出版许可与代码许可分别处理。

核心直接加载作者 `models/s2at/s2at.py::S2ATNet` 与 `cfsm.py::CFSM`：保留两路1×1投影、5×5与空洞7×7选择性卷积（CFSM）、级联空间／光谱学习与双路交互（CSLIF）、最大／平均池化分类头（MACH）。宽度64、8头、depth1、dropout0.1均沿用作者模型默认参数，无Mamba、无预训练、无新增融合模块。

## 接口与必要适配

[`model.py`](model.py) 输入 `[B,1,MSI+SAR,P,P]`，MSI在前、SAR在后；给作者HSI分支 `[B,1,C_msi,P,P]`、辅助分支 `[B,C_sar,P,P]`，输出 `[B,num_classes]` logits。

- YRD、YRD2509NEW：MSI10＋SAR4，总14通道，分别8／9类；GRSS07：MSI6＋SAR1，总7通道，5类。均按共享数据配置读取，不增删、复制或插值波段；辅助通道不解释为LiDAR高度。
- 原作者已提供 `hsi_in_channels/lidar_in_channels`；直接设置实际通道数，不做PCA30。两路卷积保留patch空间尺寸，token数为 `P²`，显式传给谱注意力中的Linear层。作者训练选项默认P11、121token；不能使用模型构造器的默认81token搭配11×11。
- 项目中心像素标签要求正奇数patch；默认11，可配置，所有尺寸相关参数在优化器构建前注册。
- 加载层 [`s2atnet_patch.py`](../_hsi_adapters/s2atnet_patch.py) 仅在内存AST去除 `timm.models.layers` 导入。其唯一实际调用的 `trunc_normal_` 使用PyTorch原生等价初始化；`to_2tuple`从未使用，作者S²ATNet中DualBlock的drop_path固定0，DropPath分支不会构造。作者文件及所有前向运算保持原样，无需引入timm／torchvision环境依赖；仍使用现有einops。
- 不运行作者主程序；本项目使用共享划分和整图无标签逐通道min-max、reflect padding、统一增强，无PCA。作者原程序的AdamW、学习率调度、标准化和选模方式不覆盖项目统一协议，差异在此明确记录。

## 配置与运行

三份主配置：`configs/s2atnet_yrd_fs50.yaml`、`configs/s2atnet_yrd2509new_fs50.yaml`、`configs/s2atnet_grss07_custom.yaml`。YRD两组每类50，GRSS07 custom `[674,1481,752,45,30]`；5 seeds 202201–202205，patch11，200epochs，train128/test1024，FP32，Adam lr0.001/wd0.0001，无预训练。

```bash
python scripts/train_eval_s2atnet.py --config configs/s2atnet_yrd_fs50.yaml
python scripts/eval_s2atnet.py --run <run目录>
python scripts/infer_s2atnet.py --run <run目录>
```

最终checkpoint测试，`model_best.pth`为兼容文件名，不以测试集选epoch。5次全部完成后按最高测试OA选全部标注位置出图，未标注位置背景；不重新计分，不覆盖测试指标。全图默认关闭，只有明确需要时另用 `--mode full`。保存全部种子指标与均值／总体标准差（ddof=0），最高单次不是平均结果。完整协议见[实验说明](../../docs/experiments/README.md)。

结果：YRD／YRD2509NEW为 `runs_fewshot/fs50/s2atnet/<dataset>/run_###/`；GRSS07为 `runs_fewshot/grss07_custom/s2atnet/grss07/run_###/`。运行身份记录Git提交／脏状态、配置和共享数据配置SHA256、环境版本和包路径、种子和划分、最终权重及指标。

## 验证状态

116临时副本首轮6项合成检查全部通过：两种通道、5/8/9类、patch7/11/13、前后向全部参数有限梯度、优化器覆盖、checkpoint重载、eval混批／单样本／逆序一致、临时MAT一轮完整训练及测试／重评估／标注图协议。原始前向数值对照通过；116全仓42项检查通过、无跳过。GPU1 FP32 train128/test1024合成检查通过，峰值已分配显存约1763.4MiB；真实GRSS07一轮smoke、标注图完成。未完成正式实验，不宣称复现论文精度。

本地默认Python无torch；现有gjc带torch但缺einops／h5py／sklearn，未安装依赖。116使用已有torch2.6.0+cu118、NumPy2.0.1、einops0.8.1；进程显式 `PYTHONPATH=/home/wwl/.conda/envs/gjc/lib/python3.11/site-packages LD_LIBRARY_PATH=/home/wwl/.conda/envs/gjc/lib`，防止用户目录旧torch覆盖；未设置PYTHONNOUSERSITE、未安装／升级包。

执行与验证记录见[S²ATNet操作记录](../../docs/operations/2026-10-30-s2atnet-adaptation.md)。
