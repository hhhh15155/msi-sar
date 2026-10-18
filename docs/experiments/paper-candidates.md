# FDMF 与 MSAF 分类适配：第二批六项实验

2026-10-18 用户明确选择并批准：FDMF 使用共同编码器与固定边界高低频融合；MSAF 保留两级结构、修复断点及测试随机性；两路均不预训练。各在三个数据集运行一次，共六项，在 server116 物理 GPU 1 通过 screen 执行。不扩展其他模型、多种子、shot 或 patch 网格。

**完成状态（2026-10-18 核验）：六项均正常结束、各200epochs、队列finished，screen已退出。** 已重算保存预测的混淆矩阵及指标、核对最终权重和原缓存／划分；当前没有待执行项。完整结果及与原七版本的对照见 [九版本结果汇总](main-results-20261018.md)。下方部署命令作为追溯记录，不重复执行。

## 模型身份与实际路径

| 版本 | 实际入口 | 结构／解释边界 |
| --- | --- | --- |
| `v7_fdmf_fixed` | `models/main_classification.py::MainClassifier(mode='fdmf')` → `models/fdmf_patch.py::FrequencyFusion` | 共同双路 CNN–Transformer → 单级频域融合 → 原中心特征＋均值分类头。是 FDMF 机制的 patch 分类适配，非四级 MiT／分割整网复现 |
| `v8_msaf_fixed` | `models/msaf_patch.py::MSAFPatchClassifier` | 两路浅层 MobileNetV3-Small → 16 维融合 → 下一浅层 → 24 维融合 → 两路 GAP 求和 → 分类。保留作者实际活跃链路，是修正后的无预训练 MSI–SAR 适配 |

共同构建入口为 `models/main_classification.py::build_main_model`；训练、测试、恢复和 screen 仍复用 `scripts/main_experiment.py`。新 manifest 显式列出这两个版本，旧七版本默认清单不扩展。

### FDMF：保留与修改

作者代码固定为 [fy-sun/FDMF-Net @ 15c5aa2](https://github.com/fy-sun/FDMF-Net/tree/15c5aa2898d6b8636aedd8dc997f446c47030f12)，读取 `models/frequency_modules.py`、`encoder_agg.py::FrequencySpaticalFusion/Fusion` 和 `attention.py`。

保留：两路 3×3 投影后 FFT、中心矩形高低频划分、abs(IFFT) 特征；基于空间均值／最大值的频空残差增强；两路低频 concat → 低频自注意力 → 低频作为 Query 分别对两路高频作交叉注意力 → concat、卷积／BN、投影。交互使用整个邻域，不是中心像素 Query。

明确修改：

- 以原 V2—V5 的 128 维、4 头、每路 2 层编码器与同一分类头替换多尺度 MiT；同种子下这些共同部分的初始权重与 Concat 一致。
- 矩形截止比例固定为 `0.5`，半径按 `round((size//2)*0.5)`，小网格至少 1；保留作者切片的边界约定，不把 abs 后的高低频之和称为精确重构。移除原硬取整后没有普通边界梯度的阈值网络／混合系数；不是可学习软掩码，也不是新机制实验。
- 单级分类没有后续编码阶段，因此不注册原来仅供后续阶段使用的 `spatial_gate`；不带分割头、CLUB／MINE、多尺度辅助损失。
- 融合 Q/K/V 无偏置（作者 `Fusion` 默认如此，但 `mit_b*` 的实际路径传入 `qkv_bias=True`）；融合 dropout 使用当前共同协议的 0.1。注意力通过 PyTorch SDPA 计算。
- 输出卷积紧接 BN，移除理论上训练时被 BN 抵消的 bias，避免 Adam 对接近零的浮点梯度产生不必要更新。这是工程修正。

这组实验检验的是“在共同编码器下，单级频域融合是否有价值”，不能引用其 OA 作为原 FDMF-Net 的论文复现成绩。FFT 特征的相位也不是 SAR 复回波物理相位。

### MSAF：保留与修改

作者代码固定为 [lulushh/MSAF @ 253641e](https://github.com/lulushh/MSAF/tree/253641e897fb75c2479826e9ed992f10c97d9f16)，实际依据 `mcf/MCF_demo_1.py::MCF/MVT/SelfAttention`，没有混用同目录其他演示版。

保留：MobileNet 活跃的 `features[0:3]`，输入投影 1×1／stride 1，第一深度卷积 stride 1；两级 5×5 潜在池化；光学空间 token＋光学通道视图 token＋SAR 空间 token；每级两层，16／24 维、4 头、head_dim 与阶段维度相同；第一级回投残差，第二级池化残差，GAP 求和分类。总 token 数是 66／74（50 个空间 token＋16／24 个通道视图 token）。

明确修改：

- 两路 torchvision `weights=None`；原光学分支请求预训练，本次不下载权重。替换光学／辅助模态输入通道，不使用 HSI PCA 或 LiDAR 高度假设。
- 移除无条件断点、未参与前向的 MobileNet 尾部／原分类器、未调用的优化器分组及 cache 清理。
- 0.1 Bernoulli 注意力掩码仅在训练使用；全屏蔽行保留一个可见 key，测试关闭掩码及 dropout，重复前向不消耗随机数。
- 第一级输出 LayerNorm 的 bias 是空间常量，随后经过下一层 pointwise conv＋BN，训练时被抵消。移除这 16 个冗余偏置；第二级 LayerNorm bias 保留。诊断中人为加 ±0.1 常量偏置，训练 logits 仅变化约 3.3e-7；移除后消除了该偏置引起的续训数值漂移。保留 LayerNorm weight 和其余所有活跃层。
- 输入支持两组通道和不小于 5 的正奇数 patch；5×5 是内部潜在网格，不要求原始 patch 等于 11。

MSAF 的编码器、维度及参数量与共同编码器不同，是完整结构的候选比较，不能当作只替换融合模块的严格消融。与原论文预训练、模态和数据划分也不同。

两作者仓库快照未见明确顶层开源许可证；公开可读不等于任意复用授权。未将整个第三方仓库或权重提交到项目，调研用只读快照留在忽略的 `tmp/`。

## 数据与协议

| 数据集 | MSI＋SAR | 每类训练数 | 原 split.npz SHA256 |
| --- | --- | --- | --- |
| YRD | 10＋4 | 8 类各 50，共 400 | `842aac597d09cb62e169c217bdafd51534fae36ff729e8a117085852a9c3a994` |
| YRD2509NEW | 10＋4 | 9 类各 50，共 450 | `d7844c61dc5875b7c32b058a1b2f909bce3ca64e5a4f7d3e7460f41e1e409e2d` |
| GRSS07 | 6＋1 | `[674,1481,752,45,30]`，共 2982 | `e73f306b348f02d7e656745a9238be3b23faabedd5acd95636eeb7d9206e24ac` |

直接复制 116 当前 21 项入口的原划分文件并核对上述 SHA，不重新抽样。三个数据集均逐通道 min-max，使用全图无标签统计；reflect 填充；没有验证集，随机像元划分允许 patch 重叠，不代表空间外推性能。

全部单种子 `202201`、patch 11、200 epochs、train batch 128、test batch 1024、FP32、AdamW lr 5e-4／wd 1e-4、余弦至 1e-6、clip 1、相同增强、仅最终 epoch 测试、不按测试指标选模型。无预训练、无 Mamba、无额外数据／互信息辅助损失。V7 共同配置 128／4／depth2；V8 的内部 16／24 结构由版本定义，不能误读 YAML 中普通 dropout 0.1 为可修改所有原生架构超参数的接口。

## 后台部署与输出

主机 `server116`，目录 `/data/gjc/msi-sar`；Python `/home/wwl/.conda/envs/gjc/bin/python`。保持其已检查的 user-site 运行时，不安装依赖、不设置 `PYTHONNOUSERSITE=1`。同卡已有约 2GB 低负载进程，单 worker；新任务开始前空闲显存至少 20480 MiB，不停止别人的进程。先完成 V7 的三个数据集，再完成 V8 的三个数据集。

```bash
cd /data/gjc/msi-sar
CUDA_VISIBLE_DEVICES=1 MAIN_EXPERIMENT_WORKERS=1 /home/wwl/.conda/envs/gjc/bin/python scripts/main_experiment.py --manifest configs/paper_candidates/manifest.json prepare
CUDA_VISIBLE_DEVICES=1 MAIN_EXPERIMENT_WORKERS=1 /home/wwl/.conda/envs/gjc/bin/python scripts/main_experiment.py --manifest configs/paper_candidates/manifest.json launch
screen -r msi_papers_gpu1
```

任务使用 PCI 顺序的物理 GPU 1；Ctrl+A、D 脱离 screen，SSH 断开不结束队列。状态／指标／断点：`runs/paper_candidates_20261018/`；队列日志：`experiment_logs/paper_candidates_20261018/queue.log`。结果单独保存，没有 promotion，不修改已经核验的 21 项基准，也不自动下载到 Windows。查看指标不运行额外全图推理。

核心检查见 `tests/test_paper_candidates.py`：两组输入、多个 patch、全部参数梯度／注册、两模态有效、eval 重复及 RNG 不变、频域显式计算对照、共同初始化、6 项范围及超参数、GPU FP32 真实 batch1024、合成数据中断续训／预测／logits 对照。合成检查不是正式性能实验；正式失败保留日志、不静默重跑。CPU Windows 当前没有 Torch，GPU 检查在 116 的隔离检查目录完成。

部署前全套 48 项检查通过。续训对普通参数和 logits 使用 FP32 数值容差 `rtol=1e-5, atol=1e-5`，测试预测必须逐项一致；共同编码器数学上冗余的 K bias 按既有检查单独界定。运行协议仍允许 cuDNN 非确定算法，不承诺逐位一致或跨硬件一致。
