# 2025—2026 多源遥感地物分类论文与代码调研汇总（Codex 交接版）

> 整理日期：2026-10-17  
> 研究目标：为**空间配准的 MSI–SAR 邻域 patch → 中心像素分类**选取可落地网络底座与融合机制。  
> 资料性质：**前轮聊天检索结果的归档、归纳与待复核任务清单**，不是完整的系统性综述，也不是全部候选已通过源码审计的证明。  
> 约束：仅只读检查外部代码；**不安装依赖、不执行第三方代码、不启动实验、不修改当前用户项目**。

## 0. 先读：结论与证据边界

1. **优先看光学/MSI–SAR**：`EG-CMFNet`、`PAD`、`FDMF-Net`，另外 `AFDINet` 依赖 VMamba，虽然纳入检索但非首选。光学–SAR 语义分割工作与用户的中心像素分类**有任务差异**，必须单独说明转换成本。
2. **中心像素 patch 分类更接近的候选**：`RSCNet`、`DFFNet`、`MSAF`、`ACFFN`、`PICNet` 等；不过它们原始模态通常是 HSI–SAR 或 HSI–LiDAR，**不可直接将 LiDAR 高度信息等同于 SAR 后向散射**。
3. **前轮形成的暂定推荐**：`EG-CMFNet` 优先作为“跨模态融合骨干思想”验证对象，`PAD` 和 `RSCNet` 为备选；这是**研究优先级，不是经过同条件实验验证的性能排名，也不是可以直接原样替换的模型名单**。
4. **已发现值得复核的源码风险**：`MSAF` 中可能有无条件 `breakpoint()` 和推理时随机掩码；`DFFNet` 可能在第一次 forward 才创建可训练参数；`PICNet` 可能在 forward 临时创建卷积层。必须由 Codex 对**当前仓库版本的具体文件/行**二次确认，不能把聊天前轮结论直接当作最终代码审计。
5. 本文将**公开可访问仓库**与**核心代码已深入核实**分开标注。仓库存在≠论文发表已被出版社证实≠主训练入口可复现≠具备明确复用许可证。未核实项一律不补写事实。

### 证据状态术语

- **R（仓库可核实）**：本轮网页可打开仓库且可见源码目录或模型/训练文件。
- **H（前轮源码线索）**：前轮聊天已指出文件、类或疑似实现问题；仍需 Codex 对仓库当前提交只读重查。
- **P（发表/作者关系待补核）**：正式发表记录、作者—仓库绑定、卷期/会议身份尚需独立确认；不能仅依赖 README 自述。
- **X（排除或暂缓）**：没有满足核心代码硬性准入的证据，暂不可作为最终推荐。

## 1. 用户实际任务与不能改变的设置

| 项目 | 当前约束 |
|---|---|
| 任务 | 配准 MSI–SAR 地物分类；两个模态对应邻域 patch，**预测中心像素类别** |
| MSI 通道 | 两套配置：**6 通道**或**10 通道** |
| SAR 通道 | 两套配置：**1 通道**或**4 通道**；部分 4 通道是极化观测派生特征，不可视为独立物理传感器 |
| 空间尺寸 | patch 尺寸可配置，**不能硬编码 11×11** |
| 样本 | 有限标注，不强制采用 episodic few-shot 训练 |
| 当前架构 | 双路 CNN–Transformer、两级受 ACFFN 启发的模糊融合、融合反馈、分类头 |
| 目标排序 | 分类精度/鲁棒性优先；在有收益前提下控制冗余模块与计算量 |
| 偏好 | 优先**非 Mamba/SSM**；允许门控；暂不优先中心像素 Query 查询对侧邻域的交叉注意力 |

**对 Codex 的直接要求**：先对当前项目的代码、配置、前向、训练/评估、patch 数据读取做真实审阅，再讨论模块替换；不要因这些论文而自行改变用户已有数据划分和训练协议。

## 2. 第一优先组：光学 / MSI + SAR

> 特别注意：部分论文使用“optical–SAR”，但其光学实际是 RGB，不能因为数据集原始文件包含 NIR 就记成 4 通道 MSI 实验。密集分割结果不应直接与中心像素 OA 横比。

| 模型/论文 | 前轮记录的发表信息 | 原任务与输入 | 融合/骨干概况 | 代码仓库 | 状态与注意点 |
|---|---|---|---|---|---|
| **PAD** — *Phase-Amplitude Decoupling Fusion for Multi-Modal Land Cover Classification* | **IEEE TGRS 2025**；DOI `10.1109/TGRS.2025.3621902` | RGB–SAR 密集地物分割；WHU-OPT-SAR 与 DDHR-SK | 双路特征，幅度/相位解耦与跨模态频域交互；前轮提及 ConvNeXt-L 和预训练 | [RanFeng2/PAD](https://github.com/RanFeng2/PAD) | **R/H**；仓库有 `src/`、`configs/`、`train.py`、`test.py`；README **明确说明从 RGB+NIR 提取 RGB**，不能标记“已验证 MSI 4 波段” |
| **EG-CMFNet** — *An Edge-Guided Cross-Modal Fusion Network for Remote Sensing Semantic Segmentation* | **IEEE TGRS 2026**；DOI `10.1109/TGRS.2026.3680719` | optical–SAR 密集语义分割 | CNN+Transformer 双流；**MAFI** 自适应特征交互、**LFF** 多尺度融合、边缘辅助 EA | [masurq/EG-CMFNet](https://github.com/masurq/EG-CMFNet) | **R/H**；仓库有 `models/`、`interfaces/`、`datasets/`；须确认编码器具体实现及训练实际实例化版本 |
| **FDMF-Net** — *Learning Frequency-Domain Fusion for Multimodal Remote Sensing Semantic Segmentation* | **IEEE TGRS 2025**；DOI `10.1109/TGRS.2025.3622749` | WHU-OPT-SAR、YESeg-OPT-SAR 与其他多模态分割 | 频域特征融合，借鉴 SegFormer/MRFS | [fy-sun/FDMF-Net](https://github.com/fy-sun/FDMF-Net) | **R**；仓库 `models/`、`main.py`、`train.py`、多个训练 shell、`config.yaml`；仓库指明预训练权重，需检查是否必须预训练 |
| **AFDINet** | 前轮记录为 **JAG 2026**，正式出版记录仍待独立核实 | 光学–SAR 多模态语义分割（需重查实际波段） | 自适应频率分解、VMamba 相关模块 | [XD-MG/AFDINet](https://github.com/XD-MG/AFDINet) | **R/P**；源码目录 `model/`、`configs/`、`tools/` 可见；GitHub 显示 Apache-2.0（仅对该仓库核实）；**依赖 Mamba，不作为无 SSM 首选** |
| **MM-OVSeg** — *Optical–SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing* | 前轮记录为 **CVPR 2026 主会**；应以 CVF 官网论文条目再交叉核实 | RGB/optical–SAR 开放词汇密集分割 | 开放词汇表征与跨模态融合，迁移条件受预训练体系影响 | [Jimmyxichen/MM-OVSeg](https://github.com/Jimmyxichen/MM-OVSeg) | **R/H**；仓库可访问；前轮指出模型源码包含 ZIP，**压缩包内部主网络未完成逐文件审查，暂不评级 A/B/C** |

### 2.1 PAD 具体线索

- 仓库：[github.com/RanFeng2/PAD](https://github.com/RanFeng2/PAD)
- 前轮代码线索：`src/PAD_penalty.py::PADPenalty`，`src/PAD_module.py::PAD_module`；训练入口仓库根目录 `train.py`，配置在 `configs/`。
- 借鉴价值：对光学反射特征和 SAR 散射纹理进行具有物理启发的幅度/相位交互，**更适合讨论频域模块**，不代表能直接保留原整图分割主干。
- 迁移事项：`MSI(6/10)` 输入投影、`SAR(1/4)` 分支、中心像素分类 head、patch 尺寸，以及有无预训练时的对比。小 patch 下 FFT 特征有效性需验证。
- 证据风险：原始多光谱数据≠实际多光谱输入；论文/仓库性能指标必须记录自己的数据划分、预训练及 mIoU/OA 定义。

### 2.2 EG-CMFNet 具体线索

- 仓库：[github.com/masurq/EG-CMFNet](https://github.com/masurq/EG-CMFNet)
- 前轮代码线索：`models/new_model/builder.py::EncoderDecoder`、`modules.py::MAFI`。**相对路径需按当前仓库目录确认**。
- 据仓库介绍：多流 CNN/Transformer 提取 optical/SAR 特征，MAFI 做跨模态自适应交互，LFF 做轻量多尺度融合，CNN 边缘分支做边界辅助。
- 借鉴价值：和用户已有双路 CNN–Transformer 更相近，适合以 `MAFI` 为**单一融合主模块**进行实验；不建议再直接叠完整 ACFFN+PAD。
- 迁移事项：从整图密集分割改成邻域 patch 中心分类；验证下采样之后最小空间尺寸；移除/重设边缘监督需要明确是否丢掉论文核心机制；确认预训练要求。

### 2.3 FDMF-Net 具体线索

- 仓库：[github.com/fy-sun/FDMF-Net](https://github.com/fy-sun/FDMF-Net)
- 可见训练入口：`main.py`、`train.py` 和 `train_WHU.sh`、`train_YESeg.sh` 等。
- 仓库明确列出 WHU-OPT-SAR、YESeg-OPT-SAR、ISPRS Vaihingen 和 Potsdam；实际每组输入具体通道/辅助模态应按数据类及配置检查。
- 可以作为光学–SAR **频域融合对照**，不能因为仓库训练脚本齐全就假定具备适合中心像素分类的直接入口。

## 3. 第二组：HSI–SAR、HSI–LiDAR 与多源 patch 分类

| 模型 | 前轮记录的发表信息 | 原始模态和主要机制 | 仓库 | 源码审查深度、风险和迁移判断 |
|---|---|---|---|---|
| **RSCNet** — *Representative Spectral Correlation Network for Multisource Remote Sensing Image Classification* | **TGRS 2026**；DOI `10.1109/TGRS.2026.3684877` | HSI–SAR / HSI–LiDAR；代表波段选择 KBSM、跨源自适应融合 CAFM | [oucailab/RSCNet](https://github.com/oucailab/RSCNet) | **R/H**；可见 `net_RSCNet.py`、`train.py`、`test.py`、`task.py`、`dataset.py`；光谱 Top-k 与 PCA 附加分支依赖 HSI，不可对 6/10 波段直接照搬 |
| **DFFNet** — *Dynamic Frequency Feature Fusion Network for Multi-Source Remote Sensing Data Classification* | **GRSL 2025** | HSI–SAR / HSI–LiDAR；动态频率滤波和谱空适应融合 | [oucailab/DFFNet](https://github.com/oucailab/DFFNet) | **R/H**；有核心模型、模块、训练、测试；前轮发现 forward 期间延迟初始化参数的潜在问题；需重查优化器建立次序 |
| **MSAF** — *Masked Self-Attention Fusion Network for Joint Classification of Hyperspectral and LiDAR Data* | **TIP：仓库写 2025，前轮卷期核对为 2026（Vol.35, 346–360）**；需要出版社记录确认在线日期 | HSI–LiDAR；MobileNetV3 与 masked self-attention，双级融合 | [lulushh/MSAF](https://github.com/lulushh/MSAF) | **R/H**；`mcf/MCF_demo_1.py` 存在；前轮记录实际网络 `MCF`；疑有 `breakpoint()`、随机掩码 eval 仍随机，暂不当“开箱即用”底座 |
| **ACFFN** | 前轮记录 **TGRS 2026**，发表及源码可访问状态需 Codex 重新核实 | HSI–LiDAR；模糊融合、NSCT、方向/形态交互 | [Zzp-12/ACFFN](https://github.com/Zzp-12/ACFFN) | **H/P**；前轮代码线索 `ACFFN.py::SimpleCNN`、`SpectralSpatialNSCT`、`FLB`、`MorphoQKVAttention`；本轮仓库打开工具受限，不能把链接可读性当已通过 |
| **PICNet** — *Prototype-Based Information Compensation Network for Multi-Source Remote Sensing Data Classification* | **TGRS 2025** | HSI–SAR / HSI–LiDAR；频率交互、模态原型补偿 | [oucailab/PICNet](https://github.com/oucailab/PICNet) | **R/H**；前轮怀疑 forward 临时 `nn.Conv2d` 和 batch size 绑定原型，可能影响训练；适合作为模块灵感，不推荐原样底座 |
| **DCMNet** — *Dynamic Cross-Modal Feature Interaction Network for Hyperspectral and LiDAR Data Classification* | **TGRS 2025** | HSI–LiDAR；动态跨模态特征交互 | [oucailab/DCMNet](https://github.com/oucailab/DCMNet) | **R**；仓库存在但当前没有完成逐文件入口审计；优先作机制比较 |
| **RoSENet** | 前轮记录 **TGRS 2025**，仍需正式出版记录核实 | HSI–LiDAR；旋转、相似性增强 | [mabokun618/RoSENet](https://github.com/mabokun618/RoSENet) | **R/P**；仓库存在；需确认哪些模块在实际训练入口启用 |
| **GAS2-FL** | 前轮记录 **TGRS 2026**，仍需正式出版记录核实 | 多源分类；分组注意力、谱空特征 | [RSIP-NJUPT/GAS2-FL](https://github.com/RSIP-NJUPT/GAS2-FL) | **R/H**；前轮代码线索 `model.py::GAS2FL`；依赖波段分组，可能有固定 patch 配置 |
| **DiffCLIP** | 前轮记录 **AAAI 2025 主会**，会议论文链接仍需核实 | 跨模态少样本/语义迁移，通常涉及 CLIP/预训练 | [icey-zhang/DiffCLIP](https://github.com/icey-zhang/DiffCLIP) | **R/P**；更适合作为预训练/语义对齐单独讨论，不宜混为轻量 MSI–SAR patch 分类底座 |

### 3.1 RSCNet：训练真实入口与使用条件

- 仓库：[github.com/oucailab/RSCNet](https://github.com/oucailab/RSCNet)
- 可见文件：[`net_RSCNet.py`](https://github.com/oucailab/RSCNet/blob/main/net_RSCNet.py)、[`train.py`](https://github.com/oucailab/RSCNet/blob/main/train.py)、[`task.py`](https://github.com/oucailab/RSCNet/blob/main/task.py)、`dataset.py`、`test.py`、`parameter.py`、`requirements.txt`。
- 仓库 README 自述主要入口是 `task.py`，且示例 `myTask(0.0001, 1, 3, dev)` **仅 1 epoch**；**不要把 README 示例设为论文正式实验超参数**。
- 其原始 HSI 光谱波段选择 KBSM 与跨源 CAFM 是两个机制；迁到 MSI 6/10 时，可能应保留 CAFM、重新论证 KBSM 是否还有价值，而非直接缩小 Top-k。
- HSI–LiDAR 使用高度属性，HSI–SAR 使用散射特征；两者物理含义不同。适配 MSI 时不应机械沿用 HSI/PCA 三路假设。

### 3.2 DFFNet：需 Codex 核验的关键风险

- 仓库：[github.com/oucailab/DFFNet](https://github.com/oucailab/DFFNet)
- 前轮代码线索：`model/net.py::Net` 和 `modules/DynamicBlock.py::DynamicFilterBlock`，根目录 `train.py`、`test.py`。
- **待核实疑点 A**：`DynamicFilterBlock` 在首次 `forward()` 内创建 `nn.Parameter`；若训练程序在前向前创建优化器，则新参数可能不包含在优化器中。
- **待核实疑点 B**：频域基础权重形状可能为 `[N,C,1,1,2]`，然后 broadcast 到各频点；须分析其频率选择性具体来自哪里，不能把“动态频域”自动解释为每个频点独立核。
- **待核实疑点 C**：分类层的 `window_size²` 与 HSI 3D 卷积对不同输入通道/patch 有硬约束。

### 3.3 MSAF：需 Codex 核验的关键风险

- 仓库：[github.com/lulushh/MSAF](https://github.com/lulushh/MSAF)
- 前轮代码线索：[`mcf/MCF_demo_1.py`](https://github.com/lulushh/MSAF/blob/main/mcf/MCF_demo_1.py)，实际类 `MCF`；模型包含 MobileNetV3-Small、MVT、自注意力掩码与双级融合。
- **待核实疑点 A**：`MVT.forward()` 内存在无条件 `breakpoint()`。
- **待核实疑点 B**：`SelfAttention.forward()` 用 `torch.bernoulli()` 产生随机掩码，未用 `self.training` 分开训练/推理，可能 eval 每次预测不同。
- **待核实疑点 C**：`mobilenet_v3_small(pretrained=True)` 和 5×5 自适应/固定池化、16/24 通道相关维度是否在不同 patch、MSI 通道下成立。
- 实际适配价值：可以抽取 masked cross-modal attention 思路；**尚不宜认定整网适合直接复现**。

### 3.4 PICNet 与 ACFFN 的风险

- PICNet：[仓库](https://github.com/oucailab/PICNet)、[疑似核心文件](https://github.com/oucailab/PICNet/blob/main/code/net.py)。前轮记录 `FIM.forward()` 临时 `nn.Conv2d`，以及原型与 batch size 耦合；需检查是否参数被优化器注册、评估批次变化是否报错。**不能仅凭聊天指出的行号就断言当前提交仍有同一 bug。**
- ACFFN：[仓库](https://github.com/Zzp-12/ACFFN)。前轮提及 `SimpleCNN`、`SpectralSpatialNSCT`、`FLB`、`MorphoQKVAttention`；需重新找到确切源码及调用链。它服务于 HSI–LiDAR 模态，复合方向/形态模块和低通道维度问题须分别测试理论适配性；用户当前实现只是“受 ACFFN 启发”，不是等于原论文结构。

## 4. 已发现但暂不符合严格主表准入的条目

| 工作 | 前轮发现 | 处理 |
|---|---|---|
| **GFS-Net**（前轮记录 JAG 2026） | 曾找到出版商指向 GitHub 的信息，但访问对应仓库报告 404 | **X：排除**；不能因为“将开源”或数据可用声明就收录 |
| **SACFNet** | 前轮认为仓库仅 README/未来公开代码说明 | **X：排除**，除非当前版本已有实际核心模型 |
| **S²PNet**（前轮记录 EAAI 2026） | 没有建立“作者仓库 → 核心网络”的可靠证据链 | **X：暂不收录** |
| **LABLF**（前轮记录 IJRS 2026） | 仓库/压缩内容未完成逐文件模型核验 | **X：暂不收录** |

**排除数量**：这只是可命名的排除线索，不是整个领域的总排除数量；不要报告虚构的系统筛查计数。

## 5. 当前建议：底座优先级（暂定，等待公平实验）

| 角色 | 模型 | 选择理由 | 主要适配代价 |
|---|---|---|---|
| **首选研究底座** | **EG-CMFNet / MAFI 思路** | 在 optical–SAR 分割上验证；双流 CNN–Transformer 与当前架构接近；有明确跨模态交互模块 | 密集分割转中心 patch 分类，深层下采样、小 patch、边缘分支监督及预训练差异 |
| **备选 1** | **PAD** | 光学–SAR 直接相关；频域幅相交互具有独立设计主线 | 实际 RGB 而非 6/10 波段 MSI；分割头→分类头，重模型与预训练 |
| **备选 2** | **RSCNet** | 输入/输出形式更接近邻域 patch 分类；有 HSI–SAR 数据验证及可见训练代码 | HSI 波段 Top-k、PCA 支路与 MSI 窄波段不直接匹配 |
| **模块参考** | MSAF、ACFFN、DFFNet、PICNet、DCMNet、GAS2-FL | 掩码融合、模糊/形态、动态频域、原型、跨模态交互、分组建模 | 有若干当前待核验实现风险；部分机制依赖 HSI/LiDAR |
| **分开研究，不作首选** | AFDINet、DiffCLIP、MM-OVSeg | 分别涉及 VMamba、CLIP/语义迁移、开放词汇分割 | 模型依赖、预训练需求、任务定义均有明显差异 |

**严禁将“推荐底座”理解为已证明的最高 OA。** 模态物理相似度、中心像素任务一致性、代码质量、低样本训练稳定性、预训练可获得性必须共同考虑。

## 6. 一条结构集中的可验证改造路线

```text
配准 MSI patch [B, 6或10, P, P]            SAR patch [B, 1或4, P, P]
                   │                                     │
             MSI 编码器                            SAR 编码器
                   └──────────────────┬──────────────────┘
                                      ▼
                         单一主要跨模态交互模块
                    （例如 EG-CMFNet 的 MAFI 改造）
                                      ▼
                    共同空间尺度的融合特征聚合
                                      ▼
                            中心像素分类头
                                      ▼
                                 类别 logits
```

区分三种贡献：

1. **论文已有**：MAFI 多模态特征交互、PAD 幅相解耦、RSCNet CAFM/KBSM、MSAF 掩码注意力等；不能重新命名当作新方法。
2. **工程适配**：改第一层 `in_channels`、同尺度融合与 token 构造、去掉整图像素级 decoder、更换中心像素分类头、动态 P、适配 batch 与通道；本身不等于创新。
3. **潜在、尚未验证创新**：针对 MSI 反射波段与 SAR 散射/派生极化特征的**可靠性引导跨模态交互**，需提出可明确写出前向方程、权重生成、训练约束与消融的完整机制，尤其要避免把派生 SAR 通道当相互独立观测。**当前仅为方向，不代表已证明优于现有融合。**

**避免简单堆叠**：不建议一开始把完整 MAFI + PAD + ACFFN + 模糊融合 + 多级反馈无差别串联。先建立简单、可归因的对照实验，再决定保留哪些机制。

## 7. 最先进行的对照和消融（建议，但本次不执行）

| 实验 | 唯一核心变化 | 验证问题 |
|---|---|---|
| 0 当前实现 | 当前双路 CNN–Transformer + 两级模糊融合 | 稳定基线；先复查实现与配置 |
| 1 简单融合 | 相同双分支编码器 + concat/加和 | 复杂融合是否必要 |
| 2 MAFI | 固定编码器替换主融合为 MAFI | 跨模态交互的独立贡献 |
| 3 PAD 融合 | 固定编码器替换主融合为幅相频域方案 | 频域设计是否适合该 patch 尺度 |
| 4 MAFI+可靠性 | 在实验 2 上只增加清晰定义的可靠性约束 | 新机制是否有稳定收益 |
| 5 模态消融 | MSI-only、SAR-only、双模态 | 互补性、负迁移、偏依赖 |
| 6 输入条件 | MSI6+SAR1 / MSI10+SAR4、多个 patch 尺寸 | 硬编码、通道适配、尺度稳定性 |
| 7 多 seed | 固定相同划分，报告均值及方差 | 小样本统计稳健性 |

公平比较应统一：数据划分、预处理、标签数量、patch、编码器容量、优化器/epoch/训练预算、预训练条件、随机种子和中心像素 OA/AA/Kappa 定义。跨数据集/训练策略的原论文 OA 不能直接排列为性能榜。

## 8. Codex 下一步：只读审查任务清单

**第一批（优先 optical–SAR）**

1. 验证 `EG-CMFNet`、`PAD`、`FDMF-Net` 的**实际训练入口 → 实例化类 → 双分支 forward → token/频域/空间融合 → 输出 head**，将代码文件、确切类名、行号、仓库 commit SHA 记录下来。
2. 找出原论文实际使用光学通道（RGB、RGB+NIR、Sentinel-2 全波段等）及 SAR 通道，核对数据集加载器和配置；**不要根据数据集名称猜输入波段**。
3. 核查 6+1、10+4 输入以及 P 可变时：第一层卷积、位置编码、池化、token 数、下采样、FFT、BN、reshape、分类头是否可泛化；判断修改复杂度。
4. 检查预训练权重、额外训练数据、特殊 CUDA/SSM 算子、训练/评估脚本完整度，**不运行代码**。

**第二批（中心 patch 分类与源码风险）**

5. 只读审查 `RSCNet`、`DFFNet`、`MSAF`、`ACFFN`、`PICNet`；对上文风险逐项报 **已确认 / 不成立 / 当前版本无法确认**，必要时贴 5–20 行关键代码和上下文。
6. `RSCNet`：确认 `task.py` 实际调用路径，HSI Top-k 与 PCA 支路如何进入前向；给出在 MSI 上使用 CAFM 的最小必要改动。
7. `DFFNet`：参数是否在优化器创建后才被注册；`MSAF`：eval 随机性与 breakpoint；`PICNet`：forward 中新建层和原型形状；`ACFFN`：形态假设、低通道维度、所有分支是否真实启用。

**第三批（论文与授权核对）**

8. 逐篇列明：正式论文出版社/会议主页、在线发表与卷期年份、作者代码对应依据、仓库最近 commit、许可证文件/SPDX 名称、模型/训练/评估/数据/权重的实际公开状况。
9. 对压缩包代码（如 MM-OVSeg）可只读解压/查看结构，但不能安装、导入和执行；没有核心网络就不通过准入。
10. 代码等级只对**明确可读核心网络**的仓库给出：A=模型+训练+评估+数据说明完整；B=模型公开但部分实验流程缺失；C=核心模型可读但脚本零散。**代码质量或疑似 bug 用独立字段描述，不能仅靠 A/B/C 代替。**

**期望 Codex 输出**：

- 一张核验过的表：`paper / year / publisher link / repo / commit / license / entrypoint / model class / architecture / trained inputs / pretrained / code completeness / source risks / patch+channel compatibility / MSI-SAR usefulness`。
- 约 5–8 篇最值得借鉴论文的精简源码调用图；明确“整网底座”与“模块来源”。
- 针对用户**实际项目代码**的完整端到端适配设计（模块结构、前向、训练损失、推理、文件/函数修改点与消融），**仅输出方案，不实际编辑代码**。
- 对无法访问的源、未读全文、无法确定训练入口，明确标记“未核实”，**严禁编造类名、性能数字、作者录用情况或许可证**。

## 9. 便于 Codex 访问的原始链接索引

### 光学/MSI–SAR

- PAD：<https://github.com/RanFeng2/PAD>；论文 DOI：<https://doi.org/10.1109/TGRS.2025.3621902>
- EG-CMFNet：<https://github.com/masurq/EG-CMFNet>；论文 DOI：<https://doi.org/10.1109/TGRS.2026.3680719>
- FDMF-Net：<https://github.com/fy-sun/FDMF-Net>；论文 DOI：<https://doi.org/10.1109/TGRS.2025.3622749>
- AFDINet（VMamba）：<https://github.com/XD-MG/AFDINet>
- MM-OVSeg：<https://github.com/Jimmyxichen/MM-OVSeg>

### HSI–SAR / HSI–LiDAR / 多源分类

- RSCNet：<https://github.com/oucailab/RSCNet>；论文 DOI：<https://doi.org/10.1109/TGRS.2026.3684877>
- DFFNet：<https://github.com/oucailab/DFFNet>
- MSAF：<https://github.com/lulushh/MSAF>
- ACFFN：<https://github.com/Zzp-12/ACFFN>（本轮访问受限，先核查）
- PICNet：<https://github.com/oucailab/PICNet>
- DCMNet：<https://github.com/oucailab/DCMNet>
- RoSENet：<https://github.com/mabokun618/RoSENet>
- GAS2-FL：<https://github.com/RSIP-NJUPT/GAS2-FL>
- DiffCLIP：<https://github.com/icey-zhang/DiffCLIP>

### 应排除或重查

- GFS-Net：前轮找到的仓库地址不可访问，**不提供未经重新核实的链接**。
- SACFNet：前轮只有 README 或将开源声明，必须重新检查是否真正发布模型后才可重新准入。
- S²PNet、LABLF：前轮未建立“论文—作者仓库—核心模型”完整可核验证据链。

## 10. 后续重新检索时的口径

检索截止 2026-10-17，主表只容纳 **2025/2026 正式发表或正式录用 + 作者公开核心网络代码** 的工作。分组优先 MSI–SAR，再 HSI–LiDAR、其他 optical–SAR、HSI–SAR、HSI/MSI+DSM/DEM 与三源模型；额外区分分割/缺失模态/自监督/预训练以及预印本。检索还应覆盖 TGRS、JSTARS、GRSL、ISPRS JPRS、RSE、JAG、GIScience & RS、IJRS、Remote Sensing、TIP、TMM、Information Fusion、Pattern Recognition、TNNLS、Neurocomputing、KBS、EAAI、IEEE Access、CVPR、ICCV、NeurIPS、ICLR、AAAI、IGARSS 等，非封闭白名单。

*本交接文档的工作重点是确保 Codex 能复核具体论文代码，并据此提出适配方案；它没有宣称对上述所有平台做了完整的数量统计。*
