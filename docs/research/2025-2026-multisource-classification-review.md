# 2025—2026 多源遥感地物分类论文与源码调研：完整合并交接版

> 整理日期／检索截止日期：2026-10-17。  
> 目标：为空间配准的 MSI–SAR 邻域 patch → 中心像素分类筛选网络底座与融合机制。  
> 本文件完整保留本轮 Codex 最终调研答复、用户提供的网页端交接材料、原始检索要求；另补充结论更新对照、未完成线索和后续任务清单。不是只保留摘要的精简版。  
> 来源之间的历史结论可能不同，按各部分标明的证据状态阅读。所有推荐均是待公平实验检验的研究优先级。  
> 本轮仅进行了外部论文／仓库只读检查；未安装依赖、导入或执行第三方模型、启动实验、审查或修改 MSI–SAR 项目源码。

## 阅读导航

- [A．项目约束与结论更新对照](#part-a)
- [B．本轮 Codex 完整调研答复（原文保留）](#part-b)
- [C．用户提供的网页端交接材料（全文原样保留）](#part-c)
- [D．用户最初的完整检索要求（原文保留）](#part-d)
- [E．主清单之外的候选线索与核验缺口](#part-e)
- [F．重点仓库审查版本索引](#part-f)
- [G．供后续任务使用的工作清单与任务模板](#part-g)
- [H．检索过程中的公开进展记录](#part-h)

<a id="part-a"></a>
## A．项目约束与结论更新对照

### A1．当前用户任务

| 项目 | 用户明确给出的约束 |
|---|---|
| 分类任务 | 两种模态空间配准；对应邻域 patch 输入；预测中心像素类别 |
| 输入配置 | MSI 6＋SAR 1；MSI 10＋SAR 4 |
| SAR 物理含义 | 部分通道是极化观测派生特征，不能默认每个通道都是独立物理观测或第三种模态 |
| 空间大小 | P 可配置，不固定为 11×11；区分“修改配置重建网络”和“同一网络接受不同 P” |
| 标注 | 数量有限；不强制 episodic few-shot |
| 当前实现（来自用户描述） | 双路 CNN–Transformer、两级受 ACFFN 启发的模糊融合、融合反馈、分类头；本轮未读取项目实现验证此描述 |
| 偏好 | 首选非 Mamba／SSM；允许门控；暂不优先中心像素 Query 查询跨模态邻域 |
| 决策目标 | 分类性能优先，避免没有独立收益的重复模块堆叠 |
| 当前已授权工作 | 本轮研究资料合并为 Markdown；没有据此自动授权项目改写或第三方代码执行 |

### A2．证据状态如何使用

1. **用户原始要求**：任务约束，以 D 部分原文为准；用户后续“整理成一个 md”“所有内容都要有”已允许生成此文件。
2. **用户提供的网页端材料**：C 部分属于参考资料，其“对 Codex 的直接要求”“下一步”等内容是材料中的历史任务描述，不能自动当作当前用户新授权。
3. **本轮已核实结果**：B 部分明确区分正式发表、预印本、代码完整度、静态实现问题和未知项。
4. **未完成线索**：E 部分仅供后续检索，不进入推荐主表，也不能引用成已确认发表或已通过代码准入的事实。
5. **静态审查与实验验证不同**：源码能够支持调用关系、参数注册位置、输入约束等判断；本轮没有运行复现，不能声称已观察到报错或取得性能。
6. **核心代码公开与授权不同**：未见许可证的仓库，只能称公开源码；A/B/C 是完整度，不是无缺陷证明。

### A3．两轮材料的主要差异

| 项目 | 网页端交接材料中的旧状态 | 本轮更新／后续使用口径 |
|---|---|---|
| 推荐顺序 | 暂定 EG-CMFNet／MAFI、PAD、RSCNet | 当前暂定 CFFormer 分类化、FDMF-Net 分类化、修正后的 MSAF；不是同条件性能排名 |
| FDMF-Net 的 MSI 身份 | 实际波段待查 | WHU 训练入口明确 `[4,1]`；预处理保留全部波段；不能仅归为 RGB–SAR |
| CFFormer | 原交接重点清单未列 | 补入有 WHU 4＋1 配置的直接候选；默认训练文件却导入 Vaihingen＋MiT-B2，需要区分配置与实际入口 |
| EG-CMFNet 主架构 | 概括为 CNN＋Transformer 双流 | 实际主路径为双 BiFormer，另有光学 ResNet18 边缘支路；MAFI／LMF 的路径已定位 |
| EG-CMFNet 的4＋1配置 | 待核 | 默认边缘支路仍按3通道构建；完整网络不能只改配置即视为已适配多光谱 |
| PAD | RGB–SAR；发表待核 | IEEE 记录确认2025-10-15发表；作者说明WHU只提取RGB；FFT相位不等同SAR复数回波物理相位 |
| MSAF 年份 | 仓库2025、卷期2026待确认 | 正式卷期2026，TIP 35:346–360；首次在线日期未核实，不能由DOI年份推断 |
| MSAF 源码风险 | breakpoint／eval随机掩码待核 | 实际Notebook导入 `MCF_demo_1.py`；无条件断点与eval随机掩码均确认；该版本掩码概率为0.1 |
| DFFNet | 延迟参数注册待核 | 已确认首次forward注册滤波参数晚于优化器创建；默认主路径不依赖Mamba |
| PICNet | 临时卷积、batch原型待核 | 已确认forward新建卷积，原型首维绑定配置batch size；不建议原样作为底座 |
| RSCNet | HSI Top-k／PCA适配问题 | 额外发现评分仅用于硬索引、普通梯度路径不通；训练按test_loader选择最佳模型，不能照搬其选模协议 |
| ACFFN | 仓库可访问性及核心模块待核 | 已读核心与demo；确认低通道瓶颈、多通道Sobel初始化、形态操作维度、batch=1等适配风险 |
| MM-OVSeg | ZIP内部核心网络未审查 | 已只读查看ZIP内网络与融合文件；通过核心代码存在性检查；主会状态由CVF核实 |
| CerraData-4MM | 新补入 | 正式JSTARS 2025，2025-07-03发表；正文明确12通道MSI＋2通道SAR；不列预印本 |
| AFDINet | JAG与输入待核 | JAG 151:105406（2026）正式记录；公开API为RGB3＋SAR1，依赖VMamba／CUDA selective scan |
| 排除项 | GFS-Net、SACFNet、S²PNet、LABLF线索 | B中新增至少7个当前代码准入失败候选；SACFNet、S²PNet、LABLF的旧线索另在E保留，不冒充本轮已复查结论 |

### A4．当前建议的简明入口

- **首选设计底座**：CFFormer 的 FCM–FFM 分类化版本；必须先修正参数分组并处理小patch尺度。
- **备选一**：FDMF-Net 分类化；直接 MSI–SAR，但频率硬掩码梯度、编码器尺度和预训练条件须处理。
- **备选二**：修正后的 MSAF；原生patch分类，需去断点、固定推理策略、清理未使用模块并统一预训练条件。
- **先做的局部融合对照**：相同编码器下比较 concat／加和、FCM–FFM、MAFI、当前模糊融合。
- **不直接叠加**：完整 MAFI＋PAD＋ACFFN＋多级模糊融合＋多级反馈。
- **待验证研究问题**：SAR基础观测／派生特征的冗余与可靠性是否造成跨模态负迁移；普通门控和已有模块组合本身不等于创新。

---

<a id="part-b"></a>
## B．本轮 Codex 完整调研答复（原文保留）

> 以下完整保留聊天最终调研答复，包含所有正式论文分组表、专项补充表、预印本表、8项重点源码分析、版本与许可证、推荐及消融。没有将未核信息升级为已核实。

我已把你提供的交接材料作为**补充检索线索**重新核验，没有直接沿用其中的发表信息、源码判断或推荐排序。

目前，对你的项目更合理的选择是：**优先考察 CFFormer 的 FCM–FFM 融合底座，FDMF-Net 作为直接 MSI–SAR 备选；MSAF 作为更贴近 patch 分类任务的备选，但必须先修正公开实现中的问题。** EG-CMFNet 值得借鉴 MAFI，但完整网络的边缘监督和输入适配成本，比交接材料中的概括更大。

以下结论均来自只读检索与源码审查；没有安装依赖、运行第三方模型或修改你的项目。

**1．检索口径与证据边界**

检索截止日期：**2026 年 10 月 17 日**。

本次交叉使用了出版社与会议页面、DOI/Crossref 元数据、作者机构页面、作者仓库目录、核心源码及训练配置。检索涉及 IEEE、Elsevier、MDPI、CVF、AAAI、IJCAI、OpenReview、ACM、IGARSS、ISPRS，以及 Springer、Wiley、Taylor & Francis 等渠道。

需要保留三个边界：

- 下表是**通过当前核验的候选清单**，不是所有指定期刊逐卷逐期的穷尽审计。某个平台没有入选条目，不代表该平台没有相关论文。
- IEEE 部分页面存在访问限制，Crossref 对不少论文只提供卷期年份。因此，能确认正式发表、但不能确认首次在线日期的，明确写“上线日未核实”，没有用 DOI 年份或 arXiv 日期代替。
- 重点分析主要依据**摘要／可访问的部分正文＋核心源码＋训练入口和配置**；没有把这些工作统一声称为“已完成全文核查”。

按你的硬性条件，已明确发现的未通过代码准入候选**至少有 7 项**：

| 候选 | 当前核验结果 | 处理 |
|---|---|---|
| [HOSNet](https://github.com/Whiter-Bai/HOSNet) | 仅介绍性文件，未找到核心网络 | 排除 |
| [FADFNet](https://github.com/Whiter-Bai/FADFNet) | 仅 README，未找到核心网络 | 排除 |
| [HMAT](https://github.com/fzero055/IEEE_TGRS_HMAT) | 当前仓库仅 README | 排除 |
| HCKNet 对应的 [HSI-classification](https://github.com/xsxah/HSI-classification) | 出版商给出链接，但当前仓库仅 README | 排除 |
| [AGCL](https://github.com/Zenro2000/AGCL) | 出版信息能找到，所指仓库访问失败 | 排除 |
| GFS-Net | 出版商链接对应仓库当前不可访问 | 排除 |
| [EDC](https://github.com/mengcx0209/EDC) | 2026 预印本声称提供实现，实际目录仅 README | 排除 |

这只是可命名、已检查的排除数量，**不是整个领域的总排除数**。

代码等级沿用你的定义：A＝模型、训练、评估和数据说明较完整；B＝核心模型公开，部分流程或配置欠缺；C＝核心可读，但脚本零散、适配成本高。**等级反映公开完整度，不代表代码没有缺陷。**

---

**2．正式发表／录用论文主表：先展示 MSI–SAR**

下面“输入通道”优先采用实际代码与配置证据。配置声明某种输入，并不等于我运行验证过该配置。

**直接涉及多光谱–SAR 的工作**

| 论文 | 年份及发表状态 | 期刊或会议 | 模态组合 | 分类任务／输入方式 | 骨干 | 核心融合机制 | 官方论文链接 | 作者代码仓库 | 代码等级 | Mamba依赖 | MSI–SAR适配价值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **CFFormer：交叉融合 Transformer 语义分割框架** | 2025 正式卷期，63:1–17；首次上线日未核实，DOI 含 2024 | TGRS | WHU 配置为光学 **4＋SAR 1**；另有光学–DSM | 密集地物分割 | 双流 Transformer，可切换编码器 | 空间→通道特征校正 FCM；双向跨注意力 FFM | [IEEE](https://ieeexplore.ieee.org/document/10786275/) | [masurq/CFFormer](https://github.com/masurq/CFFormer) | B | 否 | **高：直接模态匹配；需分类化和训练脚本清理** |
| **FDMF-Net：Learning Frequency-Domain Fusion for Multimodal Remote Sensing Semantic Segmentation** | 2025 正式，63:1–16；上线日未核实 | TGRS | WHU 实际入口 **4＋1**；另含 RGB–SAR、MSI–DSM | 密集分割 | 双 MiT／SegFormer，默认 MiT-B4 | 高频／低频分解、频空增强、跨模态融合与互信息约束 | [IEEE](https://ieeexplore.ieee.org/document/11206533/) | [fy-sun/FDMF-Net](https://github.com/fy-sun/FDMF-Net) | A | 否 | **高：直接 MSI–SAR；小 patch 和频率阈值实现需处理** |
| **CerraData-4MM：Cerrado 多模态土地利用／覆盖分类基准** | **2025-07-03 在线发表**；JSTARS 18，2025 | JSTARS，遥感领域期刊／数据基准 | Sentinel-2 **12 通道＋S1 2 通道** | 128×128 密集分类，7／14 类 | U-Net–ResNet50、TransNuSeg–Swin | 输入拼接；分割与边缘任务 | [正式期刊 PDF](https://elib.dlr.de/219728/1/CerraData-4_MM_A_Multimodal_Benchmark_Dataset_on_Cerrado_for_Land_Use_and_Land_Cover_Classification.pdf) | [ai4luc/CerraData-4MM](https://github.com/ai4luc/CerraData-4MM) | B | 否 | **高：适合作为真实 MSI–SAR 简单融合对照及外部验证集** |
| **MCF-Mamba＋CMSS：多模态交叉融合与互补掩码自监督** | 2025 正式；145:104960，12 月卷期；上线日未核实 | JAG，遥感领域期刊 | DFC2020 的 S2–S1；另有光学–DEM | 密集土地覆盖分类；另含建筑提取 | 双 VMamba＋U 形 Mamba 解码器 | Cross-Mamba 融合；互补掩码重建预训练 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1569843225006077) | [Xiao-RS/MCFMamba_CMSS](https://github.com/Xiao-RS/MCFMamba_CMSS) | B | **是** | 模态直接相关，但不作为你的首选 |

其中两个核验结果尤其重要：

- **FDMF-Net 不能简单归为 RGB–SAR。** `main.py` 对 WHU 明确构造 `in_chans=[4,1]`，预处理读取并保存全部输入波段；Potsdam 也使用 `[4,1]`。见其[实际训练入口](https://github.com/fy-sun/FDMF-Net/blob/15c5aa2898d6b8636aedd8dc997f446c47030f12/main.py)。
- **CerraData-4MM 已正式发表，不能继续放在预印本表。** 正式 PDF 给出了发表日期、12＋2 通道配置及分类实验；它更适合作为数据与基线证据，不能视为新型融合网络。[期刊正文](https://elib.dlr.de/219728/1/CerraData-4_MM_A_Multimodal_Benchmark_Dataset_on_Cerrado_for_Land_Use_and_Land_Cover_Classification.pdf)

**HSI–LiDAR 工作**

| 论文 | 年份及发表状态 | 期刊或会议 | 模态组合 | 分类任务／输入方式 | 骨干 | 核心融合机制 | 官方论文链接 | 作者代码仓库 | 代码等级 | Mamba依赖 | MSI–SAR适配价值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MSAF：Masked Self-Attention Fusion Network…** | **2026 正式卷期，35:346–360**；仓库标 TIP2025，上线日未核实 | TIP，图像处理核心期刊 | HSI–LiDAR | 邻域 patch→类别 | MobileNetV3-Small＋Transformer | 空间／通道视图 token；两级随机掩码注意力融合 | [DOI](https://doi.org/10.1109/TIP.2025.3648926) | [lulushh/MSAF](https://github.com/lulushh/MSAF) | C | 否 | **较高：任务形式接近，源码需先修正** |
| **ACFFN：Adaptive Contourlet-Guided Fuzzy Fusion Network…** | 2026 正式，64:1–12；上线日未核实 | TGRS | HSI–LiDAR | patch→类别 | 双 CNN | 自适应方向分解、形态注意力、模糊隶属融合 | [IEEE](https://ieeexplore.ieee.org/document/11482211/) | [Zzp-12/ACFFN](https://github.com/Zzp-12/ACFFN) | B | 否 | 更适合作为模块来源；LiDAR 引导假设需重建 |
| **DCMNet：Dynamic Cross-Modal Feature Interaction Network…** | 2025 正式，63:1–16；上线日未核实 | TGRS | HSI–LiDAR | patch 分类 | 模态专属特征提取＋动态路由 | 动态选择跨模态交互路径 | [IEEE](https://ieeexplore.ieee.org/document/10924653/) | [oucailab/DCMNet](https://github.com/oucailab/DCMNet) | B | 否 | 动态交互可迁移；配置和提取器带数据集针对性 |
| **SaDHT：Scale-aware Deep Hierarchical Transformer** | 2026 正式；342:115928，6 月卷期；上线日未核实 | Knowledge-Based Systems，AI 期刊 | HSI–LiDAR | patch→类别 | 可分离 CNN＋分层 Transformer | 多组混合卷积、尺度聚合、token 融合 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705126006544) | [junhengzhu/SaDHT](https://github.com/junhengzhu/SaDHT) | C | 否 | 通道适配相对直接；默认路径与部分模块启用关系需核对 |
| **RoSENet：Rotation and Similarity Enhancement Network…** | **2025-04-17 在线发表**；TGRS 63 | TGRS | HSI–LiDAR | patch 分类 | CNN＋相似性增强 | 旋转融合、谱自相似卷积、中心与邻域相似性注意力 | [IEEE](https://ieeexplore.ieee.org/document/10967546/) | [mabokun618/RoSENet](https://github.com/mabokun618/RoSENet) | B | 否 | 可作鲁棒性对照；谱结构与中心相似性设计降低优先级 |
| **GAS²-FL：Grouped Attention-Based Spectral–Spatial Feature Learning…** | 2026 正式，64:1–18；上线日未核实 | TGRS | HSI–LiDAR；代码另支持 Augsburg 等 | patch 分类 | 分组谱空网络 | 分组注意力与协同特征学习 | [IEEE](https://ieeexplore.ieee.org/document/11428268/) | [RSIP-NJUPT/GAS2-FL](https://github.com/RSIP-NJUPT/GAS2-FL) | B | 否 | 分组思想可借鉴；6／10 波段的分组依据需重新论证 |
| **HGCN-HL：Hypergraph Convolution Network Classification…** | **2025-05-14 正式发表**；25(10):3092 | Sensors，应用期刊 | HSI–LiDAR | 整图／超像素分类 | CNN＋超图卷积 | 加权融合、动态超边、局部与全局联合 | [期刊正文存档](https://pmc.ncbi.nlm.nih.gov/articles/PMC12115920/) | [giswl/HGCN-HL](https://github.com/giswl/HGCN-HL) | B | 否 | 适合整图结构建模；转独立 patch 的成本较高 |
| **AMSFN：Adaptive multi-stage fusion…via selective state space models** | **2026 正式卷期，125:103488**；DOI 含 2025，上线日未核实 | Information Fusion，信息融合期刊 | HSI–LiDAR | patch 分类 | Local／Cross Mamba | 多阶段选择性状态空间交互与聚合 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1566253525005615) | [zhangyiyan001/AMSFN](https://github.com/zhangyiyan001/AMSFN) | B | **是** | 保留作 SSM 对照，不列首选 |

MSAF 的正式卷期确认为 **2026**，不能仅因仓库名称和 DOI 中包含 2025 就统一标为“TIP 2025 正式卷期”。[出版索引记录](https://pubmed.ncbi.nlm.nih.gov/41525539/)

**其他光学–SAR 工作，以及 MSI 身份仍需谨慎处理的工作**

| 论文 | 年份及发表状态 | 期刊或会议 | 模态组合 | 分类任务／输入方式 | 骨干 | 核心融合机制 | 官方论文链接 | 作者代码仓库 | 代码等级 | Mamba依赖 | MSI–SAR适配价值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **EG-CMFNet：An Edge-Guided Cross-Modal Fusion Network…** | 2026 正式，64:1–20；上线日未核实 | TGRS | optical–SAR；WHU 配置声明4＋1，但边缘支路通道不一致 | 密集分割＋边缘辅助 | **双 BiFormer＋光学 ResNet18 边缘支路** | MAFI、LMF〔论文称 LFF〕、边缘辅助 | [IEEE](https://ieeexplore.ieee.org/document/11481201/) | [masurq/EG-CMFNet](https://github.com/masurq/EG-CMFNet) | B | 否 | **MAFI 很值得借鉴；完整底座需额外清理** |
| **PAD：Phase–Amplitude Decoupling Fusion…** | **2025-10-15 在线发表**；TGRS 63 | TGRS | **RGB–SAR**，WHU 明确提取 RGB | 密集土地覆盖分割 | 默认双 ConvNeXt-V2 Large＋UPerHead | 空间互补、相位校正 PSC、幅度融合 ASF | [IEEE](https://ieeexplore.ieee.org/abstract/document/11204689/) | [RanFeng2/PAD](https://github.com/RanFeng2/PAD) | A | 否 | 频域对照价值高；整网重、依赖预训练 |
| **DCIFNet：Cross-Modal Fusion With Correction and Interaction…** | 2025 正式，63:1–18；上线日未核实 | TGRS | 公开实现主要为 RGB–SAR | 密集分割 | 双 MiT＋MLP 解码 | 差异校正、非对称通道／空间交互、门控 | [IEEE](https://ieeexplore.ieee.org/document/11164558/) | [XD-MG/DCIFNet](https://github.com/XD-MG/DCIFNet) | B | 否 | 模态问题直接相关；Paddle→PyTorch 有额外成本 |
| **WHFNet：Wavelet-Driven Heterogeneous Fusion Network…** | 2026 正式，64:1–17；上线日未核实 | TGRS | optical–SAR；公开配置按 RGB 编码器处理 | 密集分割 | Swin Transformer＋ConvNeXt | 小波跨模态交互、空间频率差异校准 | [IEEE](https://ieeexplore.ieee.org/document/11386975/) | [XD-MG/WHFNet](https://github.com/XD-MG/WHFNet) | B | 否 | 异构编码与频域交互可借鉴；不应记为已验证6／10波段 |
| **AFDINet：Adaptive frequency disentanglement and interaction…** | 2026 正式；151:105406，7 月卷期；上线日未核实 | JAG | 公开 API 明确 **RGB 3＋SAR 1** | 密集分类 | 光学 ConvNeXt＋SAR VMamba | 可学习频率分解 LFD、角色区分的高低频融合 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1569843226003225) | [XD-MG/AFDINet](https://github.com/XD-MG/AFDINet) | B | **是** | 模态相关，但依赖 CUDA selective scan，优先级低 |

DCIFNet 的论文原先指向 Gitee，而 Gitee README 又明确链接到上面的 GitHub 实现。因此，它不是“只看到 Gitee 介绍页就排除”的情况；实际 GitHub 中已经公开主网络和训练脚本。

PAD 的作者仓库明确说明，WHU 的 RGB＋NIR 文件经过处理后**只提取 RGB**，所以不能把它当作四波段 MSI 实验。[作者数据说明](https://github.com/RanFeng2/PAD)

**HSI–SAR，以及同时验证 HSI–SAR／HSI–LiDAR 的工作**

| 论文 | 年份及发表状态 | 期刊或会议 | 模态组合 | 分类任务／输入方式 | 骨干 | 核心融合机制 | 官方论文链接 | 作者代码仓库 | 代码等级 | Mamba依赖 | MSI–SAR适配价值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **DFFNet：Dynamic Frequency Feature Fusion Network…** | 2025 正式；GRSL 22:1–5；上线日未核实 | GRSL，遥感快报 | HSI–SAR／LiDAR | patch→类别 | HSI 3D CNN＋辅助模态2D CNN | 动态频域滤波、谱空自适应融合、融合反馈 | [DOI](https://doi.org/10.1109/LGRS.2025.3586958) | [oucailab/DFFNet](https://github.com/oucailab/DFFNet) | B | **否** | 任务接近；延迟参数注册问题降低底座优先级 |
| **RSCNet：Representative Spectral Correlation Network…** | 2026 正式，64:1–14；上线日未核实 | TGRS | HSI–SAR／LiDAR | patch→类别 | 原始HSI、PCA、辅助模态三支路 | Top-k 波段选择 KBSM＋CAFM | [IEEE](https://ieeexplore.ieee.org/document/11483234/) | [oucailab/RSCNet](https://github.com/oucailab/RSCNet) | B | 否 | **CAFM 有价值；完整波段选择框架不宜直接移到 MSI** |
| **PICNet：Prototype-Based Information Compensation Network…** | 2025 正式，63:1–15；上线日未核实 | TGRS | HSI–SAR／LiDAR | patch→类别 | 3D／2D CNN＋原型交互 | 频率交互 FIM、跨模态原型补偿 | [IEEE](https://ieeexplore.ieee.org/document/11002550/) | [oucailab/PICNet](https://github.com/oucailab/PICNet) | B | 否 | 适合研究原型机制；公开实现不建议原样使用 |
| **GCCQTNet：Global Clue-Guided Cross-Memory Quaternion Transformer…** | 2025 正式卷期，36(4):7357–7371；首次上线日未核实，DOI含2024 | TNNLS，神经网络核心期刊 | HSI–SAR／LiDAR | patch 分类 | 局部／全局三分支＋四元数 Transformer | 全局线索引导、交叉记忆、跨模态对比 | [IEEE](https://ieeexplore.ieee.org/document/10557680/) | [作者仓库](https://github.com/WenshuaiHu/MSRS_Classification_GCCQTNet) | B | 否 | 模态和任务较近；四元数与多分支使归因成本较高 |
| **THSGR：Transformer-based Heterogeneously Salient Graph Representation** | 2026 正式；TGRS 64；上线日未核实 | TGRS | HSI–SAR／LiDAR | patch 分类 | 异构图＋多卷积调制器 | 动态图表示、无自注意力的长程调制、mean forward | [DOI](https://doi.org/10.1109/TGRS.2026.3686762) | [jqyang22/THSGR](https://github.com/jqyang22/THSGR) | B | 否 | 可作图结构对照；存在固定维度，适配不算最小 |
| **MCAMamba：Multi-Level Cross-Modal Attention-Guided State Space Model…** | 2025 正式；TGRS 63；上线日未核实 | TGRS | HSI–SAR／LiDAR | patch 分类 | CNN＋谱空 Mamba | 多层跨模态注意力与多尺度融合 | [IEEE](https://ieeexplore.ieee.org/document/11194194/) | [Dmygithub/MCAMamba](https://github.com/Dmygithub/MCAMamba) | B | **是** | 保留为 Mamba 对照 |

DFFNet 仓库中存在其他实验文件，不代表实际主网络依赖 Mamba。沿 `train.py → model/net.py` 的实际导入链，主模型使用的是 CNN、FFT 与融合模块。

**三源联合分类**

| 论文 | 年份及发表状态 | 期刊或会议 | 模态组合 | 分类任务／输入方式 | 骨干 | 核心融合机制 | 官方论文链接 | 作者代码仓库 | 代码等级 | Mamba依赖 | MSI–SAR适配价值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **T-APT：Text-Guided Modality-Aware Prompt Tuning…** | **AAAI 2026 主会正式论文** | AAAI，AI 主会 | 公开实验含 HSI–LiDAR、HSI–LiDAR–SAR | 联合分类＋文本引导 | 预训练 ViT／文本编码器＋模态编码 | 模态感知提示与语义对齐 | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/42411) | [Jiahuiqu/TATP](https://github.com/Jiahuiqu/TATP) | B | **有 VMamba 依赖** | 多源提示学习参考；不适合作为轻量无预训练首选 |
| **SMM-FNet：Synergistic Multi-Mamba Fusion Network…** | 已公开正式文章；331:133260，**2026-12 卷期**；上线日未核实 | Expert Systems with Applications，AI 应用期刊 | HSI＋MSI＋LiDAR／SAR | 三源 patch 分类 | CNN＋多 Mamba | 双向谱建模、中心感知空间扫描、三模态融合 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S095741742602169X) | [cwlnnu/SMM-FNet](https://github.com/cwlnnu/SMM-FNet) | C | **是** | 可了解三源机制；不应把你的SAR派生通道当成第三源 |

SMM-FNet 的卷期日期晚于检索截止日，但正式文章页面已公开，因此与“尚未发表的预印本”不同；其首次在线日期仍需补核。

---

**3．专项补充：分割、缺失模态、预训练与聚类**

这些论文满足核心代码公开条件，但任务和训练条件需要单独看待。这里也明确区分了主会与专业期刊，没有统一称为“顶刊顶会”。

| 工作 | 已核实发表状态 | 目标模态／验证任务 | 公开核心实现 | 代码与依赖 | 对你项目的用途 |
|---|---|---|---|---|---|
| **MetaRS／EarthMiss** | RSE，2026，333:115132；DOI含2025，上线日未核实 | RGB–SAR 缺失模态土地覆盖分割 | `module/baseline/MetaRS.py::MetaRS` | [作者代码](https://github.com/Yi-Heng/EarthMiss)，B；ResNet／FPN，预训练 | 参考共享／专有表示、缺失模态训练；默认配置为3＋1，不是已验证6／10波段。[正式论文](https://www.sciencedirect.com/science/article/pii/S003442572500536X) |
| **DiffCLIP** | **AAAI 2025 主会正式论文** | HSI–LiDAR 少标注分类、扩散预训练与语言对齐 | `model.py`、`clip_model.py::CLIP`；训练由 `train.py` 调用 | [作者代码](https://github.com/icey-zhang/DiffCLIP)，B；非Mamba，需预训练体系 | 有无标注数据与文本语义实验时再考虑；不宜直接视为轻量融合底座。[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34401) |
| **DPMamba** | **IJCAI 2025 主会正式论文**，2224–2232 | HSI–LiDAR／SAR 缺失模态分类 | `models/network.py::DPMamba` | [作者代码](https://github.com/Jiahuiqu/DPMamba)，B；Mamba、文本预训练 | 参考蒸馏和模态缺失训练；你的完整双模态任务不必引入整套框架。[IJCAI](https://www.ijcai.org/proceedings/2025/248) |
| **FVMGN** | **AAAI 2026 主会正式论文** | 多模态分类及模态泛化 | `GitHub-FVMGN/models.py::FVMGN` | [作者代码](https://github.com/ZJier/FVMGN)，B；视觉语言与频率模块 | 适合研究训练与测试模态不一致；不作为普通双模态分类主底座。[AAAI论文](https://ojs.aaai.org/index.php/AAAI/article/download/38251/42213) |
| **CDD** | TCSVT，2025，35(6):5318–5330；机构记录2025-01-23发表 | optical–SAR **无监督聚类** | `Diffusion/Model.py::UNet`，扩散特征后接K-means | [作者代码](https://github.com/suldier/CDD)，B；非Mamba | 只能作为表征预训练／聚类对照，不能与监督OA直接排名。[作者机构记录](https://research.utwente.nl/en/publications/conditional-dual-diffusion-for-multimodal-clustering-of-optical-a/) |
| **MM-OVSeg** | **CVPR 2026 主会正式论文** | optical／RGB–SAR 开放词汇分割 | ZIP中有 `MMOV.py`、`MMOVL14.py`、`FusionAgg.py`、`P2Tformer_dual.py` | [作者代码](https://github.com/Jimmyxichen/MM-OVSeg)，B；CLIP／DINO／Detectron2 | 可参考跨模态统一与文本对齐；当前仓库包含不同DINO版本，需区分原论文设置和后续实现。[CVF论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Wei_MM-OVSeg_Multimodal_Optical-SAR_Fusion_for_Open-Vocabulary_Segmentation_in_Remote_Sensing_CVPR_2026_paper.pdf) |
| **AnySat** | **CVPR 2025 主会正式论文** | 包含S1、S2等传感器；验证土地覆盖、作物／树种分类等 | `encoder/Any.py::AnyModule`，训练、评估、投影器代码公开 | [作者代码](https://github.com/gastruc/AnySat)，A；非Mamba，MIT | 可作为预训练编码器单独实验；新输入通道需要适配投影器，训练数据和预算与从零训练不同。[CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html) |

MM-OVSeg 的压缩包已只读查看内部目录和主要网络文件，**没有导入或执行其中代码**。它通过了核心代码存在性检查。

IGARSS 方面，已核实 *Learning from Noisy Pseudo-Labels for All-Weather Land Cover Mapping* 属于 IGARSS 2025 正式议程，但它主要是**光学辅助产生监督、SAR单模态推理**。本轮没有完成其仓库核心模型审查，因此没有把它加入推荐表。[IGARSS官方议程](https://www.2025.ieeeigarss.org/view_paper.php?PaperNum=6397&SessionID=1631)

**4．满足核心代码条件的预印本补充**

| 论文 | 状态 | 实际模态与任务 | 核心代码证据 | 等级／限制 | 适配价值 |
|---|---|---|---|---|---|
| **Fusion and Grouping Strategies in Deep Learning for Local Climate Zone Classification of Multimodal Remote Sensing Data** | arXiv:2603.04562，2026-03；未确认正式发表或录用 | So2Sat 的 Sentinel-1＋Sentinel-2，patch级LCZ分类 | [作者仓库](https://github.com/GVCL/LCZC-MultiModalHybridFusion)，公开 `FM1-model.py` 至 `FM4-model.py` | B；TensorFlow／Keras，数据路径和内存需求需适配；未见明确许可证 | **值得加入简单CNN融合对照**；LCZ标签是patch级，且类别合并方案不能与原17类结果混比。[预印本](https://arxiv.org/abs/2603.04562) |

这项工作可以补足“直接 MSI–SAR、简单融合”的对照，但目前不应压过已正式发表的候选。

---

**5．八项重点源码审查**

以下流程图描述的是公开实现，不把工程修改当作原论文机制。

**① CFFormer：首选设计底座，但不是直接运行原脚本即可复现**

```text
MSI／SAR → 两路分层编码 → FCM空间→通道校正
                         ↓
                  双向全空间跨注意力FFM
                         ↓
                  多尺度解码 → 密集类别图
```

核心文件和实际调用：

- [`builder.py::EncoderDecoder`](https://github.com/masurq/CFFormer/blob/eee03787af4138607181e1f56cb98f07a6858b85/models/new_model/builder.py) 根据配置选择编码器。
- WHU 的 BiFormer 路径调用 [`dual_biformer.py`](https://github.com/masurq/CFFormer/blob/eee03787af4138607181e1f56cb98f07a6858b85/models/new_model/encoders/Transformer/BiFormer/dual_biformer.py)。
- 该路径真正导入的是 [`modules.py::FeatureCorrection_s2c、FeatureFusion、CrossAttention`](https://github.com/masurq/CFFormer/blob/eee03787af4138607181e1f56cb98f07a6858b85/models/new_model/modules.py)，不能用其他备选编码器的融合文件代替分析。

FCM 用两模态联合特征生成空间和通道权重，进行残差校正；FFM 把每个空间位置变成 token，执行双向跨注意力。**没有中心像素 Query 的强制假设，也不依赖连续高光谱波段或LiDAR高度。**

需要注意：仓库 `train.py` 当前默认导入 **Vaihingen＋MiT-B2**，WHU 的4＋1、BiFormer配置需要显式切换。其 `group_weight()` 按模块类型收集参数，静态检查发现没有完整覆盖直接注册的融合参数及 LayerNorm；采用该训练入口时需要修正参数分组。[训练入口](https://github.com/masurq/CFFormer/blob/eee03787af4138607181e1f56cb98f07a6858b85/models/new_model/train.py)、[参数分组](https://github.com/masurq/CFFormer/blob/eee03787af4138607181e1f56cb98f07a6858b85/models/new_model/init_func.py)

迁移到你的任务：保留双分支与FCM–FFM，重设小patch的下采样、窗口与空间降采样比例，把密集解码器改为中心分类读出。不能把512×512分割配置直接缩成11×11。

**定位：适合作为分类化底座；原论文的分割验证支持融合方向，尚不能证明其小patch优势。**

**② FDMF-Net：直接 MSI–SAR 的频域备选**

```text
两路MiT → 各自FFT高／低频分解 → 频空增强
                    ↓
             跨模态频率融合
                    ↓
         多尺度分割头＋互信息约束
```

实际入口是 [`main.py → Baseline`](https://github.com/fy-sun/FDMF-Net/blob/15c5aa2898d6b8636aedd8dc997f446c47030f12/main.py)，主干位于 [`encoder_agg.py::RGBXTransformer`](https://github.com/fy-sun/FDMF-Net/blob/15c5aa2898d6b8636aedd8dc997f446c47030f12/models/encoder_agg.py)，频率模块在 [`frequency_modules.py::FrequencyModule`](https://github.com/fy-sun/FDMF-Net/blob/15c5aa2898d6b8636aedd8dc997f446c47030f12/models/frequency_modules.py)。

两模态采用相似的分层Transformer提取；交互对象包括空间token和高低频特征，不需要把SAR解释成高度。公开代码包含频率分解、融合、CLUB／MINE约束，核心机制确实存在。

一个重要限制是：频率掩码边界通过 `round().int()` 后作切片赋值。**按当前前向，阈值生成器无法通过这条硬掩码边界获得普通反向传播梯度**；这与“阈值可端到端学习”的直观理解有差别。默认MiT-B4还使用预训练权重，小patch需要调整各层空间降采样。

**定位：完整对照底座或频域模块来源；优先比较“频域模块是否带来独立收益”，不要同时增加多组互信息损失。**

**③ EG-CMFNet：优先借鉴 MAFI，完整网络适配成本较高**

```text
光学BiFormer ─┐
              ├→ MAFI → LMF → 对齐解码
SAR BiFormer ─┘                ↑
光学ResNet18 → 边缘辅助支路 ────┘
```

[`train.py`](https://github.com/masurq/EG-CMFNet/blob/eed0ad865eea8b0dc0e18db1da11afc062a36b52/models/new_model/train.py) 实际调用 [`EncoderDecoder`](https://github.com/masurq/EG-CMFNet/blob/eed0ad865eea8b0dc0e18db1da11afc062a36b52/models/new_model/builder.py)，后者调用双BiFormer并获得边缘输出。MAFI和代码中名为LMF的融合位于 [`modules.py`](https://github.com/masurq/EG-CMFNet/blob/eed0ad865eea8b0dc0e18db1da11afc062a36b52/models/new_model/modules.py)。

MAFI依次使用另一模态的通道统计和空间统计调制当前特征；残差系数初始化为零，适合分析渐进交互。它没有中心Query设计。

但不能把它简化成“光学CNN＋SAR Transformer”：

- 主分类路径是**双BiFormer**。
- ResNet是额外的光学边缘支路。
- 默认配置声明光学4通道，但边缘ResNet仍按默认3通道构建；输入修改函数只修补双BiFormer的第一层。
- `torch.std` 和SimAM中的空间统计在特征退化为1×1时存在数值风险。
- 边缘支路的类别数和输出处理也需要随任务统一修改。

相较CFFormer，该仓库新的参数分组已使用 `named_parameters()`，不能把CFFormer的参数遗漏问题机械套过来。

**定位：很好的单模块来源。只有中心像素标签时，移除边缘支路会改变原论文完整方法，应该称“基于MAFI的分类适配”。**

**④ PAD：适合频域机制对照，不建议先搬整网**

```text
双ConvNeXt → 空间互补预融合 → FFT
                            ├→ 幅度ASF
                            └→ 相位PSC
                                  ↓
                           逆FFT＋残差 → 分割头
```

训练入口通过MMSeg配置构造 [`PADPenalty`](https://github.com/RanFeng2/PAD/blob/5766e84a2b2bdb554b570d170128190fea9247c1/src/PAD_penalty.py)，其融合模块是 [`PAD_module.py::PAD_module、PSC`](https://github.com/RanFeng2/PAD/blob/5766e84a2b2bdb554b570d170128190fea9247c1/src/PAD_module.py)。默认WHU配置使用双ConvNeXt-V2 Large及外部预训练权重。

源码中的实际次序是**先空间融合，再对融合特征做幅相处理**；不能笼统描述成“两路原始频谱分别完全解耦后交互”。代码实现了相位缩放、幅度增强、可学习半径和幅度约束。

迁移需要修改两路输入、预训练权重适配、分割头和patch尺度。还要区分：这里的傅里叶相位是**实值影像／特征的空间频谱相位**，不是直接利用SAR复数回波的物理相位。原论文RGB–SAR结果不能自动证明6／10波段收益。

**定位：局部频域模块与强对照；整网不适合作为小样本小patch的第一步。**

**⑤ MSAF：形式贴近你的任务，但公开版本不宜视为成熟实现**

```text
两路MobileNet浅层 → 5×5特征池化
                       ↓
      光学空间token＋通道视图token＋辅助模态token
                       ↓
               掩码注意力 → 回投残差
                       ↓
                 第二级融合 → GAP分类
```

实际Notebook导入的是 [`mcf/MCF_demo_1.py::MCF`](https://github.com/lulushh/MSAF/blob/253641e897fb75c2479826e9ed992f10c97d9f16/mcf/MCF_demo_1.py)，不能以同目录另一个演示版本代替。主入口是 [`MSAF_hs.ipynb`](https://github.com/lulushh/MSAF/blob/253641e897fb75c2479826e9ed992f10c97d9f16/MSAF_hs.ipynb)。

已确认：

- `MVT.forward()` 有无条件 `breakpoint()`。
- 实际版本的 `SelfAttention` 使用概率 **0.1** 的Bernoulli掩码，没有按 `self.training` 区分推理。
- 光学MobileNet请求预训练，LiDAR分支不采用同样的预训练设置。
- 完整MobileNet对象被注册，但前向只使用部分浅层模块；统计参数量或比较容量前需要清理。

它的5×5是**潜在特征池化网格**，不等于原始输入必须5×5或11×11。第一层可接不同通道，结构上适配6＋1、10＋4比HSI三维谱卷积更直接，但仍需重新初始化与控制预训练条件。

**定位：修正后的patch分类备选。优势来自任务匹配和低维二维提取，不是当前代码的复现成熟度。**

**⑥ RSCNet：CAFM 比完整波段选择框架更值得迁移**

```text
原始HSI → 保波段深度卷积 → Top-k选择 ──┐
PCA影像 → CNN ─┐                     ├→ CAFM → GAP分类
SAR／LiDAR→CNN ┴→ 初始融合 → 迭代引导 ─┘
```

实际路径为 [`task.py → train.py → net_RSCNet.py::RSCNet`](https://github.com/oucailab/RSCNet/blob/ec3a44bbaba15a2998fac0195eb28a4c0275abef/net_RSCNet.py)。训练实例设置 `topk_ratio=0.2`、两层RSCM，且 `embed_dim` 取原HSI通道数；不是类构造器中所有默认值。

CAFM采用卷积、两模态门控及局部／全局权重融合，没有高成本中心查询。但完整方法保留原HSI与PCA两路光学表示，对6／10波段MSI未必必要。

另有两个实现限制：

- `topk` 的分数只用于产生索引，然后 `gather` 特征；**选择器分数没有可微的下游使用路径**，其评分层不能按常规反向传播学习。
- 训练脚本每轮计算 `test_loader` 精度并据此选最佳模型。迁移时应保留你的既有独立验证协议，不能照搬这种测试集选模方式。

**定位：CAFM模块来源；不建议原样保留“原始MSI＋PCA＋硬Top-k”整套结构。**

**⑦ ACFFN：模糊融合可借鉴，LiDAR引导部分不能直接解释为SAR**

```text
HSI → 谱注意力＋LiDAR引导方向分解 → CNN ─┐
LiDAR → 形态式注意力 → CNN ────────────┤
                         FLB门控＋加和 → GAP分类
```

实际网络是 [`ACFFN.py::SimpleCNN`](https://github.com/Zzp-12/ACFFN/blob/765443918a21baa550f6c93cd9befc614e3fb8c6/ACFFN.py)，[`demo.py`](https://github.com/Zzp-12/ACFFN/blob/765443918a21baa550f6c93cd9befc614e3fb8c6/demo.py) 显式传入 `dim=128`。核心包括 `SpectralSpatialNSCT`、`GDDM`、`MorphoQKVAttention` 和 `FLB`。

需要修正的假设包括：

- MSI6时，`bands//8`产生零维注意力瓶颈，应重新设计宽度。
- GDDM只给第一个辅助输入通道初始化Sobel核，随后冻结全部卷积权重；扩展到SAR4时，其他通道核不能直接沿用。
- 形态分支的max／min沿**通道维**进行，不宜直接解释成经典空间形态操作。
- 分类前连续 `squeeze()` 会影响batch＝1的维度。
- FLB中存在未使用参数，隶属宽度也缺少显式正值约束。

其FLB分别处理两路特征，并不等同于你当前“两级模糊交互＋反馈”的全部结构。

**定位：模糊隶属与方向调制模块来源；完整LiDAR引导网络不是MSI–SAR首选。**

**⑧ DFFNet：研究问题贴近，但参数注册风险已经确认**

```text
HSI 3D CNN／SAR 2D CNN
          ↓
  动态频域滤波＋谱空融合
          ↓
  融合结果反馈两路〔多级〕
          ↓
       展平分类
```

实际模型为 [`model/net.py::Net`](https://github.com/oucailab/DFFNet/blob/a9beb2b2b43b452cfca364160822f0dee9b34462/model/net.py)，动态模块为 [`DynamicFilterBlock`](https://github.com/oucailab/DFFNet/blob/a9beb2b2b43b452cfca364160822f0dee9b34462/modules/DynamicBlock.py)，由根目录 [`train.py`](https://github.com/oucailab/DFFNet/blob/a9beb2b2b43b452cfca364160822f0dee9b34462/train.py) 调用。

`filter_bases_F` 在第一次forward才创建，而优化器在这之前已经构造；沿该入口，新增参数不会自动进入已有优化器。其基础滤波器形状为 `[基数,C,1,1,2]`，再广播到频点，不能把它描述成“每个频点都有独立学习核”。

分类层还依赖配置的 `window_size²`：**可以修改配置重建模型，不代表同一个已建模型可直接接受任意P。**

**定位：动态滤波、谱空交互和反馈的机制对照；不是原样复现首选。**

另外，交接材料对PICNet的疑点也已确认：[`FIM.forward()`](https://github.com/oucailab/PICNet/blob/a73650dc4f74a4a14b2a6d15974889c6ec19a1fe/code/net.py) 临时新建压缩与解压卷积，原型首维又绑定配置batch size。临时层不属于持久注册模型参数，且每次前向重新初始化，足以使它退出我的完整底座推荐。

上述运行后果均是**静态调用链推断**，没有被包装成“已运行复现并观察到报错”。

**6．重点仓库的更新、流程与许可证**

| 仓库 | 审查提交／最近提交日期〔UTC〕 | 训练、评估与数据公开程度 | 权重／依赖情况 | 许可证 |
|---|---|---|---|---|
| CFFormer | `eee0378`；2025-06-25 | 模型、训练／测试、多数据集配置；默认入口需切换 | 预训练路径需配置；PyTorch、timm、fairscale等 | **未见明确仓库许可证** |
| FDMF-Net | `15c5aa2`；2026-03-24 | 模型、训练／测试函数、Hydra配置、预处理 | 有初始化权重说明；环境文件公开 | **未见明确仓库许可证** |
| EG-CMFNet | `eed0ad8`；2026-07-21 | 模型、训练／测试、配置；边缘流程需整理 | BiFormer预训练路径；额外工具依赖 | **未见明确仓库许可证** |
| PAD | `5766e84`；2025-11-30 | MMSeg训练／测试、数据处理、完整配置 | 环境文件；外部ConvNeXt预训练 | **未见明确仓库许可证** |
| MSAF | `253641e`；2025-12-22 | Notebook流程、核心模型、部分工具；版本混杂 | torchvision预训练；未见完整锁定环境 | **未见明确仓库许可证** |
| RSCNet | `ec3a44b`；2026-05-21 | 训练／测试、数据读取、参数文件 | 给出数据与权重链接；默认演示仅1 epoch | **未见明确仓库许可证** |
| ACFFN | `7654439`；2026-05-06 | 模型、demo、utils；配置较零散 | 未见完整依赖锁定及成套权重发布 | **未见明确仓库许可证** |
| DFFNet | `a9beb2b`；2025-07-11 | 训练／测试、数据加载、YAML配置 | 未见完整依赖锁定与成套权重说明 | **未见明确仓库许可证** |

明确核到许可证文件的补充候选包括：

- [WHFNet](https://github.com/XD-MG/WHFNet/blob/main/LICENSE)、[AFDINet](https://github.com/XD-MG/AFDINet/blob/main/LICENSE)、[THSGR](https://github.com/jqyang22/THSGR/blob/main/LICENSE)：**Apache-2.0**。
- [RoSENet](https://github.com/mabokun618/RoSENet/blob/main/LICENSE)、[HGCN-HL](https://github.com/giswl/HGCN-HL/blob/main/LICENSE)、[AnySat](https://github.com/gastruc/AnySat/blob/main/LICENSE)：**MIT**。
- AMSFN README声明Apache-2.0，但这与逐一核实完整许可证文件的证据等级不同。
- 其余未明确列出的仓库，许可证尚未完成独立核实。

**公开可读源码不等于已获得任意复用授权**；本文没有把未标许可证的仓库默认视为MIT或Apache项目。

---

**7．针对你的 MSI–SAR 项目的最终选择**

| 角色 | 推荐 | 选择依据 | 必须接受的适配代价 |
|---|---|---|---|
| **首选底座** | **CFFormer 的 FCM–FFM 分类化版本** | 有4＋1 MSI–SAR配置；融合机制不依赖HSI密集波段或LiDAR高度；非Mamba；不使用中心Query | 修正优化器参数覆盖；明确WHU实际入口；小patch下重新安排编码尺度、窗口与分类读出 |
| **备选一** | **FDMF-Net 分类化版本** | 实际WHU入口4＋1；训练／配置较完整；频域主线明确 | 处理硬频率边界的梯度问题；缩小默认B4结构；统一预训练和损失条件 |
| **备选二** | **修正后的 MSAF** | 原生patch→类别，二维提取对6／10通道更自然；5×5池化并不绑定输入11×11 | 去断点、明确eval掩码策略、清理未使用模块、统一预训练条件；原模态缺少SAR直接验证 |

这个排序体现了你的优先级：**前两个先考虑已经直接涉及 MSI–SAR 的模型；MSAF凭任务形式匹配进入备选，但不能凭TIP平台就认定比当前实现更成熟。**

EG-CMFNet的MAFI应当进入你的首批**融合模块对照**，但我暂不把“完整EG-CMFNet”放在前三个完整底座中：你的标签是中心像素类别，无法直接继承其密集边缘监督；公开默认配置还存在边缘分支通道适配问题。

**一条结构集中的改造路线**

建议围绕一个问题展开：**SAR观测质量和派生特征冗余，会不会使跨模态校正产生不必要的负迁移？**

```text
MSI [B,6/10,P,P] ─→ MSI编码 ─┐
                            ├→ 单组FCM–FFM交互 → 中心特征／邻域聚合 → 类别
SAR [B,1/4,P,P] ─→ SAR编码 ──┘
                               ↑
                   待验证的SAR可靠性／冗余约束
```

第一阶段先完成可解释的工程适配：输入投影、适合P的空间尺度、中心类别读出，以及训练／推理一致性。**先不同时叠加两级模糊融合、多级反馈、PAD和多个辅助损失。**

第二阶段再检验一个明确的新机制：依据真实通道来源，把SAR的**基础观测与派生特征**区分处理，限制冗余派生量重复强化同一证据，再控制跨模态残差注入。不能预设你的4通道是4个独立观测，也不能未经元数据核实就指定其物理含义。

这里必须区分：

| 内容 | 应如何定位 |
|---|---|
| FCM–FFM、MAFI、PAD幅相处理、MSAF掩码融合、CAFM | **已有论文机制** |
| 改输入通道、patch尺度、位置／窗口配置、分类头、修正参数注册与eval行为 | **工程适配／实现修正** |
| 对派生SAR冗余和观测可靠性的专门建模及约束 | **可能的研究方向，尚未验证；需要继续查新和消融** |

一个普通门控层、几个已有模块的组合，本身不足以证明创新。

**最值得先做的对照与消融**

| 顺序 | 实验 | 要回答的问题 |
|---|---|---|
| 1 | 当前实现、MSI-only、SAR-only | 当前双模态是否稳定优于单模态？ |
| 2 | 固定编码器，concat／加和＋相同分类头 | 复杂融合是否确有必要？ |
| 3 | 相同编码器下比较FCM–FFM、MAFI、现有模糊融合 | 收益来自哪一种交互，而非编码器容量？ |
| 4 | 分别取消现有第二级融合、融合反馈 | 是否存在无收益重复模块？ |
| 5 | 空间域对照与单一FDMF／PAD频域模块 | 小patch上频域机制是否有效？ |
| 6 | 普通门控与“可靠性／派生冗余约束”门控 | 所提问题是否带来超出普通门控的收益？ |
| 7 | 6＋1、10＋4；SAR基础观测／派生特征消融；多个P | 改进是否只依赖某一输入配置？ |
| 8 | 同划分、多seed，分别报告有／无预训练 | 小样本收益是否稳定、比较是否公平？ |

保留你的既有数据划分与训练协议；PCA、归一化、模型选择也要统一口径。原论文的密集mIoU、不同训练样本量下的OA、预训练和从零训练结果，都不应放在一起作性能排行榜。

**当前最值得先验证的是：在相同双分支编码器下，用一组FCM–FFM或MAFI替换现有复杂融合，检查它是否已经足够；再决定是否值得增加SAR可靠性机制。**

---

<a id="part-c"></a>
## C．用户提供的网页端交接材料（全文原样保留）

> 来源：用户提供的 `MSI_SAR_2025_2026_%E8%AE%BA%E6%96%87%E4%B8%8E%E6%BA%90%E7%A0%81%E8%B0%83%E7%A0%94%E4%BA%A4%E6%8E%A5_Codex.md`。  
> 下文是历史参考材料原文，不是新的核验报告或当前任务授权。旧推荐排序、待核状态与本轮结论不同的地方，见A3与B；为满足完整保留要求，原文未作删改。

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


---

<a id="part-d"></a>
## D．用户最初的完整检索要求（原文保留）

> 原始任务定义全文保留。原文“暂不生成Markdown文件”之后，用户已另行明确要求生成完整合并Markdown，因此本文件的创建符合后续请求；其余只读审查、模态优先级、代码准入和不运行实验等约束继续保留。

# 检索 Prompt：2025—2026 年多源遥感地物分类论文与开源代码

你是一名研究多模态遥感地物分类的研究员。请全面检索 **2025—2026 年发表或正式录用、且已经公开核心模型代码的多源遥感地物分类论文**，为我的 MSI–SAR 项目筛选值得借鉴的网络底座与创新机制。

**多光谱–SAR 地物分类放在第一优先级：优先检索、优先展示、优先深入分析源码，并在最终模型推荐中优先考虑。其他模态仍按以下范围全面检索。**

**本次只收录有真实、公开、可访问代码仓库，且已经公开核心网络实现的论文。论文声称“将开源”、空仓库、仅有 README 或项目介绍页，均不满足要求。**

---

## 一、研究范围与优先顺序

覆盖以下模态组合：

1. **多光谱＋SAR 地物分类——第一优先级。**
2. 高光谱＋LiDAR。
3. 其他光学＋SAR。
4. 高光谱＋SAR。
5. 高光谱／多光谱＋DSM、DEM。
6. 包含上述模态的三源或更多源联合分类。

对于多光谱–SAR，除直接使用“multispectral–SAR”的论文外，也要检查使用“optical–SAR”“optical and radar”“multisensor”等表述的论文，根据实际传感器、输入波段与实验配置判断其归属，避免遗漏。

核心任务包括：

- 地物分类、土地覆盖分类。
- 配准多模态影像的像素级分类。
- 邻域 patch 输入、中心像素类别输出。
- 整图密集土地覆盖分类。

多模态语义分割、缺失模态分类、自监督预训练、跨模态迁移可以纳入，但单独分类，并说明如何迁移到地物分类。

排除以变化检测、目标检测、配准、超分辨率、图像生成为主要任务，且没有相关地物分类验证的论文。

---

## 二、时间与发表状态

检索 **2025—2026 年**工作，注明检索截止日期。

逐篇核实：

- 期刊在线发表日期与正式卷期年份。
- 会议录用状态和会议年份。
- 会议主会与 Workshop。
- 正式发表、已录用、预印本等状态。

不把 arXiv 上传日期直接当作正式发表年份，不将仓库名称或作者自述直接当作录用证明。

正式发表和已录用论文进入主表；有公开核心代码的预印本可以列入单独补充表，不能混入正式发表论文。

---

## 三、期刊与会议覆盖

**不要局限于 TIP、TGRS，也不要把下面名单当作封闭白名单。** 结合任务关键词、作者线索与引用关系继续扩展，避免遗漏其他平台中的相关开源工作。

### A. 遥感、摄影测量与地球观测期刊

至少覆盖：

- IEEE Transactions on Geoscience and Remote Sensing（TGRS）
- IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing（JSTARS）
- IEEE Geoscience and Remote Sensing Letters（GRSL）
- ISPRS Journal of Photogrammetry and Remote Sensing
- Remote Sensing of Environment（RSE）
- International Journal of Applied Earth Observation and Geoinformation（JAG）
- GIScience & Remote Sensing
- International Journal of Remote Sensing
- Photogrammetric Engineering & Remote Sensing
- Remote Sensing
- Computers & Geosciences
- Drones、Sensors 等刊物中直接相关且代码公开的工作

### B. 图像处理、多媒体、模式识别与信息融合期刊

至少覆盖：

- IEEE Transactions on Image Processing（TIP）
- IEEE Transactions on Multimedia（TMM）
- IEEE Transactions on Circuits and Systems for Video Technology（TCSVT）
- IEEE Journal of Selected Topics in Signal Processing（J-STSP）
- Information Fusion
- Pattern Recognition
- Signal Processing
- Signal Processing: Image Communication
- Computer Vision and Image Understanding

### C. 神经网络、人工智能与交叉应用期刊

至少覆盖：

- IEEE Transactions on Neural Networks and Learning Systems（TNNLS）
- IEEE Transactions on Artificial Intelligence（TAI）
- IEEE Transactions on Emerging Topics in Computational Intelligence（TETCI）
- Neural Networks
- Neurocomputing
- Information Sciences
- Knowledge-Based Systems
- Expert Systems with Applications
- Engineering Applications of Artificial Intelligence
- IEEE Access

Nature Communications、Science Advances 等综合或交叉期刊，如有直接相关且公开核心代码的工作，也可以补充。

### D. 计算机视觉、机器学习与多媒体会议

至少覆盖相关年份的：

- CVPR、ICCV、ECCV
- NeurIPS、ICLR、ICML
- AAAI、IJCAI
- ACM Multimedia
- WACV、BMVC

### E. 遥感专业会议

补充覆盖：

- IEEE IGARSS
- ISPRS 相关会议与论文集

**广泛检索不等于将所有平台视为同一等级。** 输出时区分领域核心期刊、其他期刊、顶级会议主会、其他会议及 Workshop，不能统称为“顶刊顶会”。

---

## 四、代码仓库硬性准入条件

进入任何论文推荐表之前，必须同时满足：

1. 存在公开、可访问的代码仓库，例如 GitHub、GitLab、Gitee 或作者机构代码平台。
2. 能核实仓库与论文的对应关系，优先使用论文、作者主页或官方项目页给出的链接。
3. 仓库中已经公开核心网络实现，能够找到模型定义和主要前向路径。
4. 不是空仓库、仅 README、仅数据下载脚本、仅结果展示或“coming soon”。
5. 不以第三方不明复现代替作者代码。确需讨论第三方复现时，另表标注，不能冒充官方开源。

**未找到满足条件的仓库，就排除该论文。不能因为期刊档次高、标题相关或论文很新而放宽要求。**

许可证单独核实：区分明确开源许可证和仅公开源码但未标明许可证，不能默认公开仓库意味着任意复用授权。

---

## 五、我的项目背景

我的任务是 **MSI–SAR 地物分类**：

- 两种模态空间配准。
- 输入对应的邻域 patch，尺寸可配置，不固定为 11×11。
- 输出中心像素的地物类别。
- 数据配置包括 MSI 6 通道＋SAR 1 通道，以及 MSI 10 通道＋SAR 4 通道。
- 部分 SAR 通道是极化观测的派生特征，不能默认是独立物理观测。
- 标注数量有限，但不要求 episodic few-shot 分类框架。
- 分类性能优先，同时希望避免没有收益的重复模块堆叠。

当前实现为双路 CNN–Transformer、两级 ACFFN 启发的模糊融合、融合反馈和分类头。正在评估 MSAF 等成熟结构是否更适合作为改造底座。

偏好：

- **首选不依赖 Mamba／SSM 的模型。**
- 相关 Mamba 工作可以收录，但明确标记依赖，不作为首选推荐。
- 门控机制可以使用。
- 暂不优先推荐中心像素作为 Query 查询跨模态邻域的融合方式。
- 通用遥感基础模型单独讨论；必须有与目标模态和分类任务直接相关的验证，不能占据主列表。

---

## 六、检索方法与来源核实

组合检索任务词、模态词、发表平台与代码词，包括：

- multispectral SAR land cover classification
- optical SAR land cover mapping
- Sentinel-1 Sentinel-2 classification
- hyperspectral LiDAR classification
- hyperspectral SAR classification
- hyperspectral DSM classification
- multisource remote sensing classification
- multimodal remote sensing classification
- GitHub、code、implementation、official repository

结合以下渠道交叉核查：

- IEEE Xplore
- ScienceDirect／Elsevier
- Springer、Wiley、Taylor & Francis
- MDPI
- CVF Open Access
- OpenReview
- ACM Digital Library
- 会议官方录用名单与论文集
- 作者主页、实验室主页
- GitHub 等代码平台
- arXiv，用于全文与版本追踪补充

对重点论文建立：

**正式发表记录 → 论文或作者页面 → 对应仓库 → 核心模型文件与实际入口**

可以从 MSAF、ACFFN、DiffCLIP 等候选追踪作者其他工作、参考文献和后续论文，但这些候选也必须重新核实，不能将此前讨论内容视为事实依据。

不要编造论文、DOI、发表平台或代码地址。没有核实的信息明确标注“未核实”。

---

## 七、代码阅读要求

对进入重点推荐的论文，实际阅读源码，回答：

1. 模型定义在哪个文件、哪个类？
2. 训练入口实际调用哪个版本？
3. 特征提取、token 构造、融合和分类分别如何实现？
4. 是否使用预训练权重、额外数据或特殊算子？
5. 是否支持不同 patch 大小与输入通道数？
6. 是否存在固定尺寸、硬编码类别数、调试断点、推理随机性或未使用模块？
7. 训练、评估、预处理、配置、权重分别公开到什么程度？
8. 论文描述与公开实现是否一致？不能确认的地方明确标注。
9. 最近代码更新日期、依赖说明及许可证是什么？

代码完整度分级：

- **A：** 核心模型、训练、评估及数据说明较完整。
- **B：** 核心模型已公开，但部分流程或配置缺失。
- **C：** 核心实现可读，但实验脚本零散、适配成本较高。

没有核心模型代码的仓库直接排除，不评为 C。

**只读审查，不安装依赖、不执行第三方代码、不启动实验、不修改我的项目。**

---

## 八、论文分析重点

每篇重点论文说明：

- 解决的具体融合问题。
- 网络骨干及融合机制。
- 两种模态分别采用什么特征提取方式。
- 交互单位是空间像素、局部区域、特征通道、频率成分还是其他表示。
- 核心创新是否已经体现在公开代码中。
- 是否依赖高光谱密集波段或 LiDAR 高度属性。
- 迁移到 MSI–SAR 时，需要修改哪些假设和模块。
- 原论文的性能证据是否支持它作为成熟底座。
- 与我的数据划分、标注量、预训练条件有哪些差异。

针对多光谱–SAR 论文，重点判断其特征提取与融合机制是否适合光学反射信息和 SAR 后向散射信息。

针对其他模态论文，明确说明可迁移机制及迁移限制，不能直接把 LiDAR 高度属性解释为 SAR 物理属性。

参数量、FLOPs、速度和分类精度如需引用，注明来源与测量条件。不同数据集、划分和预训练条件下的 OA，不能直接横向排名。

没有阅读全文时，注明判断来自摘要、部分正文或源码，不能声称已完成全文核查。

---

## 九、输出要求

### 1. 检索概况

报告检索日期、覆盖平台、模态范围与筛选规则。

说明因无公开核心代码而排除了多少候选；无法可靠统计时如实说明。

不预设必须找到多少篇，不为凑数量降低条件。

### 2. 已核实正式发表／录用论文主表

**首先展示多光谱–SAR 地物分类论文，再展示其他模态。表中每篇都必须有可点击的代码仓库链接。**

| 论文 | 年份及发表状态 | 期刊或会议 | 模态组合 | 分类任务／输入方式 | 骨干 | 核心融合机制 | 官方论文链接 | 作者代码仓库 | 代码等级 | Mamba依赖 | MSI–SAR适配价值 |
|---|---|---|---|---|---|---|---|---|---|---|---|

按模态组合分组，覆盖不同期刊与会议。

### 3. 开源预印本补充表

仍须满足公开核心代码条件，明确注明未正式发表或录用状态未确认，不能混入正式发表论文。

### 4. 最值得借鉴的候选

依据实际结果选择约 5—8 篇，**优先深入分析多光谱–SAR 地物分类模型**，逐篇给出：

- 简明网络结构图或流程。
- 关键代码文件、类名和链接。
- 最值得借鉴的模块。
- 迁移到我的 MSI–SAR 项目时需要修改的部分。
- 代码与复现限制。
- 适合作为完整底座，还是局部模块来源。

### 5. 最终建议

给出：

- **一个首选底座。**
- **两个备选底座。**
- **一条结构集中、问题明确的改造路线。**
- **最值得先做的对照与消融。**

最终推荐中，优先考虑已在多光谱–SAR 地物分类中验证且公开核心代码的模型。其他模态的模型仍可推荐，但必须解释其相对于 MSI–SAR 候选的优势及适配代价。

清楚区分：

1. 原论文已经提出的机制。
2. MSI–SAR 适配所需的工程修改。
3. 可能形成研究贡献、但尚需验证的新机制。

不能把已有方法的简单组合直接认定为充分创新，不能承诺未经实验验证的性能提升。

**最终所有推荐必须来自已经核实、且公开核心模型代码的论文。**

全部用中文在聊天中输出，暂不生成 Markdown 文件或修改项目代码。


---

<a id="part-e"></a>
## E．主清单之外的候选线索与核验缺口

> 这是本轮搜索中出现、但没有进入B部分最终推荐清单的线索。下表不作推荐，不能将仓库可访问、README自述、论文状态或源码片段之一自动当作完整证据链。未列出的信息不补猜。

| 线索 | 论文／仓库链接 | 当前掌握的证据与未完成项 | 后续处理 |
|---|---|---|---|
| SACFNet | 用户原交接材料仅给名称 | 旧材料称仅README／将开源；本轮未建立新的核心文件审查记录 | 保留待复核；不混入已核代码主表 |
| S²PNet | 用户原交接材料称EAAI 2026 | 论文—作者仓库—核心网络证据链不完整 | 待核，不作为最终推荐 |
| LABLF | 用户原交接材料称IJRS 2026 | 仓库及压缩内容未完成模型审查 | 待核，不作为最终推荐 |
| WIMNet | [DOI](https://doi.org/10.1109/TGRS.2026.3703805)；[仓库](https://github.com/Dufu9/WIMNet) | 检索到TGRS 2026正式记录（网页显示2026-06-15发表）；读到 `model/WIMNet.py::MambaHL` 与 `demo.py`，明确Mamba依赖；未完成成套配置、许可证和复现条件审查 | 可作为整图HSI–LiDAR的SSM补充，不作无SSM首选 |
| GIHCN | [DOI线索](https://doi.org/10.1109/TCSVT.2025.3614153)；[仓库](https://github.com/ZJier/GIHCN) | 已读 `GIHCN.py::MMHSIC、DKFM、DiffViT_CLS`，存在Mamba依赖；仓库主要为模型与Loss；卷期、实际训练链和数据生成流程尚未完成独立审查 | 继续补齐发表与训练入口证据后再决定收录 |
| MCEF | [出版商](https://www.sciencedirect.com/science/article/pii/S1566253526006123)；[仓库](https://github.com/li-zi-qi/MCEF) | 出现Information Fusion DOI `10.1016/j.inffus.2026.104736`，检索卷期为2027；仓库有MCEF.zip，未完成压缩包核心网络审查及2026上线日期核实 | 不能仅因DOI含2026或卷期2027就自动纳入／排除；先核上线日期与核心实现 |
| PS2A-FSLNet | [DOI线索](https://doi.org/10.1016/j.engappai.2026.114488)；[仓库](https://github.com/TingtingGeng/PS2A-FSLNet) | 检索到EAAI 173:114488（2026）及模型文件线索；跨域／少样本协议和实际核心调用未完成审查 | 作为episodic／跨域专题候选，降低当前项目优先级 |
| MPCF | [DOI线索](https://doi.org/10.1016/j.neucom.2025.130782)；[仓库](https://github.com/AIYAU/MPCF) | 检索到Neurocomputing 648:130782（2025），见模型与训练文件；是否符合所要求的传感器多模态融合范围未完成核实 | 不能仅凭“multimodal prototypical”名称收录 |
| SAHARA | [仓库](https://github.com/giu-guarino/SAHARA) | README使用EuroSAT-MS-SAR，但明确每样本只保留一个模态、形成未配对数据；正式发表与完整核心路径待核 | 属于跨模态迁移／未配对少样本方向，不能当配准双输入融合验证 |
| TerraMind | [ICCV 2025正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.pdf)；[仓库](https://github.com/IBM/terramind) | 正式论文与多源下游验证线索可核；本轮没有完成核心网络文件和实际分类入口审查 | 通用基础模型专题；不占主分类列表 |
| Galileo／TerraFM | 仅保留检索名称；[TerraFM预印本](https://arxiv.org/abs/2506.06281) | 有多源基础模型线索；发表状态、作者代码和具体分类验证未完成核查 | 不根据名称猜会议或代码地址 |
| FlexiSAM | [DOI线索](https://doi.org/10.1016/j.isprsjprs.2025.05.028) | 检索到ISPRS 2025相关工作；核心作者代码及目标模态验证未完成核验 | 仅保留线索 |
| SAMDFuse | [出版商](https://www.sciencedirect.com/science/article/pii/S0924271626002303)；[仓库线索](https://github.com/christie6133/SAMDFuse) | 检索到ISPRS 2026、DOI `10.1016/j.isprsjprs.2026.04.049`；需核实主任务与土地覆盖分类验证、核心模型 | 不因有图像融合输出就纳入分类主表 |
| APS-Net | [DOI线索](https://doi.org/10.1111/phor.70053) | 检索到Photogrammetric Record 2026记录；未建立作者公开核心代码证据 | 不入推荐表 |
| HAFNet | [DOI线索](https://doi.org/10.1111/phor.70028) | 检索到Photogrammetric Record相关工作；没有核到符合准入的核心代码 | 不入推荐表 |
| MMLGNet | [预印本](https://arxiv.org/abs/2601.08420)；[仓库线索](https://github.com/AdityaChaudhary2913/CLIP_HSI) | 2026预印本，HSI／LiDAR与语言对齐；核心文件和实际调用尚未完成审查 | 未通过本轮代码准入核验，不进入预印本推荐表 |
| Heat-conduction multimodal fusion | 检索标题：Toward Interpretable Multimodal Fusion: Heat Conduction Modeling for Hyperspectral and LiDAR Joint Classification | 搜索到2026-09线索；尚未独立核实正式记录、作者仓库与核心实现 | 不补造DOI或仓库 |
| ACM MM多层对比聚类 | 检索标题：Deep Multi-Level Contrastive Clustering for Multi-Modal Remote Sensing Images；[官方录用列表](https://acmmm2025.org/accepted-regular-papers/) | 官方录用列表出现该标题，但作者代码和目标数据模态尚未建立完整证据链 | 聚类专题待核，不列监督分类主表 |
| IGARSS2025 noisy pseudo-labels | [预印本](https://arxiv.org/abs/2504.13458)；[代码线索](https://github.com/StuLiu/DFC2025Track1) | 议程正式状态已核；光学辅助监督、SAR推理；核心仓库尚未完成审查 | 不能按配准双输入分类方法直接推荐 |
| IGARSS2025 InternImage pseudo supervision | [官方议程](https://www.2025.ieeeigarss.org/view_paper.php?PaperNum=6397&SessionID=1631)；[代码线索](https://github.com/zxqyiang/DF2025) | 专业会议正式议程可核，核心网络审查未完成 | 缺失模态／伪标签专题待核 |
| OpenEarthMap-SAR | [作者仓库](https://github.com/cliffbb/OpenEarthMap-SAR)；[预印本](https://arxiv.org/abs/2501.10891) | 有公开数据、基线与后续正式发表线索；本轮未完整核查原论文各融合基线的实际调用 | 可用于数据与评估背景，不替代新网络核心代码审查 |
| CMFNet／MHCMamba | [CMFNet线索](https://github.com/li-zi-qi/CMFNet)；[MHCMamba线索](https://github.com/li-zi-qi/MHCMamba) | 目录查询不足以判断压缩文件等内容；未完成核心源码核验 | 不把空的Python文件筛选结果直接等同于空仓库 |
| HSLiNets | [作者仓库线索](https://github.com/Judyxyang/HSLiNets) | 搜索中出现两个相关标题／预印本版本；年份、正式状态和核心调用未完成对照 | 待区分论文版本，不混用题名 |
| HyLiOSR | [仓库](https://github.com/B-Xi/TGRS_2025_HyLiOSR)；[DOI线索](https://doi.org/10.1109/TGRS.2025.3545926) | 有TGRS2025与GPL-3.0声明线索；核心模型未完成本轮审查 | 开集识别专题待核 |
| CloudLULC | [仓库](https://github.com/RSIIPAC/CloudLULC)；[预印本线索](https://arxiv.org/abs/2606.17713) | 当前目录有README与命名为dataset的RAR；未找到可读核心网络，未检查归档内容 | 不准入；不能进一步断言RAR绝不包含源码 |
| SentinelFuseNet | [仓库线索](https://github.com/ranjithkumaranandan/SentinelFuseNet) | 未建立正式发表证据；配对／配准和分类验证不满足当前可信证据要求 | 排除当前推荐，不采用自述性能作依据 |

### E1．年份／任务排除与防误收记录

- ASANet：搜索记录为2024工作，不以2025—2026新工作收录。
- TAFFN：已见核心文件，但对应工作／仓库为更早年份，不因近期搜索出现而当作2025—2026论文。
- HFSNet：2024工作；可作背景，不混入当前年份主表。
- BCLNet：本轮线索指向2024，年份不符合当前主范围；若后续发现新版本，需独立核实。
- M4-SAR：光学–SAR目标检测，缺少本任务所需的分类验证时排除；不能因ECCV录用而放宽任务条件。
- 3MOS：主要为光学–SAR匹配；不能把用于分层匹配数据的地类标签当成融合地物分类验证。
- OptiSAR-Net、OptiSAR-Net++、GeoMamba：搜索中分别涉及检测、视觉定位／grounding或检索；不能仅凭“optical–SAR”纳入当前分类范围。
- 单模态HSI分类、HSI解混、SAR→光学生成、变化检测、配准、超分辨率：除非论文另外给出符合条件的地物分类验证，否则排除。
- 任何README-only、coming-soon、仅数据／结果的仓库，即使正式论文存在，也不能评为C；C必须已有可读核心网络。

### E2．平台覆盖的真实边界

原始要求的期刊／会议清单完整保存在D。检索涉及这些平台的关键词、论文、作者和引用线索，但没有逐刊逐卷期穷尽核查。尤其是TMM、J-STSP、Pattern Recognition、Signal Processing、Signal Processing: Image Communication、CVIU、TAI、TETCI、Neural Networks、Neurocomputing、Information Sciences、EAAI、IEEE Access、GIScience & Remote Sensing、IJRS、Photogrammetric Engineering & Remote Sensing、Computers & Geosciences、Drones、Remote Sensing，以及ECCV、NeurIPS、ICLR、ICML、ACM MM、WACV、BMVC等，仍需进一步做有记录的定向补检。不得将“B中没有入选条目”解释为这些平台没有相关开源论文。

---

<a id="part-f"></a>
## F．重点仓库审查版本索引

> 日期为核到的最近提交日期UTC；这是检索时的版本快照，不保证后续HEAD不变。下一任务先比较提交，再判断是否需要重审。源码行号以所链接提交为准，不能把不同版本的行号混用。

| 工作 | 审查提交 | 日期 | 已实际审查的关键路径 |
|---|---|---|---|
| CFFormer | [eee03787af4138607181e1f56cb98f07a6858b85](https://github.com/masurq/CFFormer/commit/eee03787af4138607181e1f56cb98f07a6858b85) | 2025-06-25 | `models/new_model/{train.py,only_train.py,builder.py,init_func.py,modules.py}`；WHU/Vaihingen配置；双BiFormer／备选MiT；解码器 |
| EG-CMFNet | [eed0ad865eea8b0dc0e18db1da11afc062a36b52](https://github.com/masurq/EG-CMFNet/commit/eed0ad865eea8b0dc0e18db1da11afc062a36b52) | 2026-07-21 | `models/new_model/{train.py,builder.py,init_func.py,modules.py}`；`encoders/Transformer/BiFormer/dual_biformer.py`；WHU配置；边缘路径 |
| FDMF-Net | [15c5aa2898d6b8636aedd8dc997f446c47030f12](https://github.com/fy-sun/FDMF-Net/commit/15c5aa2898d6b8636aedd8dc997f446c47030f12) | 2026-03-24 | `main.py,train.py,config.yaml,data/prepared.py,models/model.py,models/encoder_agg.py,models/frequency_modules.py` |
| PAD | [5766e84a2b2bdb554b570d170128190fea9247c1](https://github.com/RanFeng2/PAD/commit/5766e84a2b2bdb554b570d170128190fea9247c1) | 2025-11-30 | `train.py,configs/PAD_whuoptsar.py,src/PAD_penalty.py,src/PAD_module.py`；作者RGB预处理说明 |
| MSAF | [253641e897fb75c2479826e9ed992f10c97d9f16](https://github.com/lulushh/MSAF/commit/253641e897fb75c2479826e9ed992f10c97d9f16) | 2025-12-22 | `MSAF_hs.ipynb`；`mcf/MCF_demo_1.py`；同目录其他演示版用于区分版本，不用其掩码概率替代实际导入版本 |
| RSCNet | [ec3a44bbaba15a2998fac0195eb28a4c0275abef](https://github.com/oucailab/RSCNet/commit/ec3a44bbaba15a2998fac0195eb28a4c0275abef) | 2026-05-21 | `task.py,train.py,net_RSCNet.py`；硬Top-k、三路输入、CAFM、测试集选模 |
| ACFFN | [765443918a21baa550f6c93cd9befc614e3fb8c6](https://github.com/Zzp-12/ACFFN/commit/765443918a21baa550f6c93cd9befc614e3fb8c6) | 2026-05-06 | `ACFFN.py,demo.py`；FLB、NSCT、GDDM、MorphoQKVAttention、demo维度128 |
| DFFNet | [a9beb2b2b43b452cfca364160822f0dee9b34462](https://github.com/oucailab/DFFNet/commit/a9beb2b2b43b452cfca364160822f0dee9b34462) | 2025-07-11 | `train.py,model/net.py,modules/DynamicBlock.py,modules/fusion.py`；参数注册与优化器次序、固定P分类器 |
| PICNet | [a73650dc4f74a4a14b2a6d15974889c6ec19a1fe](https://github.com/oucailab/PICNet/commit/a73650dc4f74a4a14b2a6d15974889c6ec19a1fe) | 2025-05-19 | `code/net.py::FIM、PICNet`；forward临时卷积与batch绑定原型 |

补充实现索引（审查深度见B，不代表与上表相同的完整入口审计）：

- DCIFNet：`train_dual_seg_whu.py → paddleseg/models/segformer.py::DCIFNet → backbones/dual_segformerv2.py::DualMixVisionTransformer`；交互位于 `paddleseg/models/utils/fusion_utils.py`。主干按前三通道分出RGB。
- WHFNet：`mmseg/models/backbones/whfnet.py::WHFNet、CrossModalInteractor、SpatialFrequencyFusionModule`；WHU配置为Swin／ConvNeXt。
- AFDINet：`model/afdinet.py::AFDINet、LearnableWaveletDecomposition、HLFusion、HLFusionBlock、PyramidFusionDecoder`；SAR编码器 `swin_umamba.py::VSSMEncoder`。
- DCMNet：`main.py → net/all.py::DCMNet`；`net/routing_modules.py::RoutingModule`、`net/Feature_Interaction.py`。
- SaDHT：`main.py → SaDHT.py::MViT`。存在未使用模块和固定token分割常量等进一步复核项；不能把定义过的CTF自动当作默认入口已启用。
- GAS²-FL：`model.py`、`train.py`及数据工具已公开；不同模态配置需独立检查。
- HGCN-HL：核心并非缺失，而是在 `HGCN_Model_Training.ipynb` 中，含 `HGCN、WMF、SSConv、HGCN_Network`；依赖torch_geometric／torch_scatter。
- AMSFN：实际 `main.py` 导入 `vmamba.py::MultimodalClassifier`，`module.py`主要是工具／损失，不能误当主模型。
- GCCQTNet：`models/Dual_channelNet.py::GCCQTNet`，另有cross-memory／quaternion实现。
- THSGR：`models/THSGR.py::THSGR、DynamicGraphConvolution`；有固定维度，需要针对P重新分析。
- MCAMamba：`model/MCAMamba.py::Net、Syn_layer`；实际依赖 `modules/Mamba.py`。
- MCF-Mamba：`models/mcfmamba.py::MCFMamba → models/utils/_mcfmamba/dual_vmamba.py::DualVMamba`；常规与CMSS入口不同。
- T-APT：`modelAll_any_2_new.py::modelAll` 与 `modelALL_any_3_new.py::modelAll`，二源／三源分别调用；含外部预训练ViT及文本编码器。
- SMM-FNet：`MambaTri_Net.py::MambaTri → CNN_Encoder、Mamba_Fusion、CNN_Classifier`；另见forward内临时位置参数的风险，未作为首选。
- MetaRS：`module/baseline/MetaRS.py::MetaRS`，`base_metars/model.py` 与 `configs/baseline/MetaRS.py`。
- DiffCLIP：`train.py → clip_model.py::build_model／CLIP`；扩散／MAE与分类视觉编码在 `model.py`。
- DPMamba：`models/network.py::DPMamba`，两阶段训练入口 `main_RS.py`。
- FVMGN：`GitHub-FVMGN/models.py::FVMGN` 及图像编码器、频率模块。
- CDD：`run.py → Diffusion/Train.py`，主模型 `Diffusion/Model.py::UNet`，输出表征后聚类。
- MM-OVSeg：ZIP中 `mmov_seg/MMOV.py`、`MMOVL14.py`、`modeling/transformer/FusionAgg.py`、`P2Tformer_dual.py`；B16／L14实现与原论文配置需区分。
- AnySat：`src/models/networks/encoder/Any.py::AnyModule`；按传感器投影并编码后拼接token、执行联合Transformer；新通道需新投影器适配。
- CerraData-4MM：实验目录下concat分组包含U-Net构造和TransNuSeg主实现；使用公开数据不等于原始baseline就是新融合机制。
- LCZC-MultiModalHybridFusion：`FM1-model.py`至`FM4-model.py`；数据级、特征级、注意力、多尺度与决策级方案分别比较。

---

<a id="part-g"></a>
## G．供后续任务使用的工作清单与任务模板

### G1．已完成与未完成

- [x] 完整保存原始检索任务与用户网页端交接文档。
- [x] 完整保存本轮25项主表、7项专项补充、1项预印本及全部分析内容（详见B实际表格）。
- [x] 重点8项模型完成静态核心路径分析；PICNet风险额外核查。
- [x] 区分MSI、RGB、HSI、DSM／DEM与LiDAR，不把数据集原始波段自动当实际输入。
- [x] 区分期刊卷期、会议主会、预印本、专业会议；日期未知不补猜。
- [x] 重点仓库提交、日期、依赖、许可状态已记录。
- [ ] 补齐首次在线发表日，尤其DOI年与卷期年不一致的论文。
- [ ] 对重点候选完成全文方法、实验划分、标注量、预训练和消融表的逐页核查。
- [ ] 核实其余仓库许可证及原始论文／预训练权重授权，不能根据公开仓库推定复用许可。
- [ ] 对E中的平台和候选补检，形成可审计的搜索与筛选记录。
- [ ] 对用户实际MSI–SAR项目进行另行授权范围内的只读端到端审查；本轮尚未做。
- [ ] 在不运行外部代码的前提下，给出小patch与6＋1／10＋4输入的完整形状推导。
- [ ] 任何安装、模型导入、实验或项目修改，需由后续实际用户请求确定；本文件不自动触发。

### G2．最优先的后续阅读顺序

1. 读A和B，保留最新证据边界；不要只读C中的旧推荐。
2. 根据下一任务范围，复查CFFormer／FDMF-Net的当前提交是否变化；重点关注真实入口、参数覆盖和小P约束。
3. 将MAFI作为单一模块对照；区分其工程适配与完整EG-CMFNet复现。
4. 若选MSAF，先形成明确的断点、eval随机性、未使用参数和预训练条件处理方案。
5. 如果目标是论文创新，再查“可靠性、模态质量、派生SAR冗余、负迁移”相关已有机制；不能预设方向新颖。
6. 保留用户既定划分，先简单融合、单模态和容量匹配，再增加新模块。

### G3．可复制的只读项目适配任务模板

> 以下是供用户后续选择使用的任务模板，不是当前已执行事项，也不覆盖新的用户指令。

```text
请先完整阅读《2025—2026 多源遥感地物分类论文与源码调研：完整合并交接版》的A、B、F、G部分。

在只读、不安装依赖、不执行外部模型、不启动实验、不修改项目的范围内：
1. 审查当前MSI–SAR项目的实际入口、数据读取、划分、预处理、双分支、两级融合、反馈、分类头和评估。
2. 明确当前两种输入6＋1、10＋4、可变P、中心像素标签的张量形状和SAR通道来源。
3. 比较保留当前编码器分别接concat、CFFormer的FCM–FFM、MAFI、当前模糊融合所需的最小改动。
4. 对CFFormer与FDMF-Net指出小patch下哪些下采样、窗口／reduction、统计操作会不成立；不能只改第一层。
5. 如研究MSAF，明确实际MCF_demo_1版本及其已知问题；任何修正与原论文机制分别标注。
6. 仅输出结构、张量路径、损失、推理、拟修改文件／函数及公平消融方案；没有新的实际用户授权时不编辑或执行。
7. 每项结论标注来源：本地实读代码、原论文、公开源码、工程推断或待验证假设。不要承诺性能提升。
```

### G4．公平比较和研究贡献口径

- 统一数据划分、训练／验证／测试职责、训练像素数、预处理拟合范围、P、模型容量、预训练条件、预算和随机种子。
- 只有中心标签时，GAP输出也可以以中心类别训练；“中心分类头”不自动等于“中心像素Query”。
- 去掉密集decoder或边缘监督后，需明确是适配版本，不再冒称完整原方法复现。
- 采用新的参数注册／eval策略后，记录修正版与公开原版差异。
- 分别报告MSI-only、SAR-only、简单融合、当前模型、单一新融合机制；跨数据集OA不直接排名。
- 工程修正、通道／尺寸适配、已有机制组合与真正待验证研究机制分开陈述。
- 不把LiDAR高度属性解释为SAR物理属性；不把4个派生特征当4次独立观测；不把空间FFT相位当复数SAR观测相位。

---

<a id="part-h"></a>
## H．检索过程中的公开进展记录

> 以下保留本轮最终答复之前的用户可见检索进展，帮助理解哪些旧判断在后续被补充或收敛。时间为会话记录时间；以B中的最终表述及F中的版本快照为当前证据口径。

### 进展 10/17/2026 08:28:30

我会先核实 MSI–SAR 论文的发表记录与核心源码，再扩展到其他模态，并把正式论文、预印本和不满足代码条件的候选分开记录。本次会使用学术检索技能；只读检查网页与源码，不安装依赖、不运行第三方代码、不修改项目或生成报告文件。


### 进展 10/17/2026 08:29:29

检索中已遇到两个需要严格区分的问题：有些论文只给了数据集链接，并未公开模型；“optical–SAR”也可能实际使用 RGB＋SAR，不能一律归为 MSI–SAR。我会按实际输入波段核对，并把网络源码是否存在作为收录门槛。当前没有可用的专用 CrossRef／Scopus 检索工具，正在通过出版社、会议官方页面和代码平台交叉核实。


### 进展 10/17/2026 08:31:09

已核实 MSAF 对应的是 *Masked Self-Attention Fusion Network*：仓库标注“TIP 2025”，正式卷期则是 2026 年，需要同时记录。另有两项光学–SAR 候选 HOSNet、FADFNet 的仓库目前只有介绍文件，不能收录。接下来重点检查已有核心实现的 CFFormer，以及更贴近小 patch 分类的模型。


### 进展 10/17/2026 08:32:46

源码检查发现 MSAF 有几处会影响复现：训练 notebook 实际导入的版本保留了 `breakpoint()`，注意力随机掩码也没有限制在训练阶段，调用 `eval()` 后仍可能随机变化。这不否定论文机制，但说明它需要清理后才能作为稳定底座。DCIFNet 的 Gitee 页面则指向了另一个已有完整源码的官方 GitHub 仓库，已沿链接找到。


### 进展 10/17/2026 08:36:54

已找到直接采用 4 通道光学＋1 通道 SAR 的 CFFormer，正在核对它实际启用的编码器与融合模块。其他模态候选中也发现了会影响复现的实现细节；最终会把“已公开核心代码”和“适合直接作为稳定底座”分开评价。


### 进展 10/17/2026 08:38:09

又补到了两条重要线索：RSE 的 MetaRS 已公开训练与评估代码，但主要研究缺失模态；DCIFNet 作者还公开了 2026 年的 WHFNet。后者采用光学 Transformer、SAR CNN 和小波交互，值得检查，但不能仅凭数据集名称就认定它使用了全部多光谱波段。


### 进展 10/17/2026 09:29:54

我会先读取这份交接材料，把其中的论文和仓库作为补充线索，再逐一核对发表记录与核心代码。材料中的结论会重新验证，并继续按你原定的 MSI–SAR 优先、只读审查要求检索。


### 进展 10/17/2026 09:31:32

交接材料新增的 PAD、EG-CMFNet、FDMF-Net、RSCNet 等仓库都能看到模型文件，PAD 也已由 IEEE 记录确认于 2025 年 10 月 15 日发表，并明确使用 RGB–SAR。源码下载接口刚出现连接失败，我正在切换到 GitHub 文件接口继续核查；尚未读到的内容不会标成已审查。


### 进展 10/17/2026 09:33:10

输入通道核查有一个关键补充：FDMF-Net 的 WHU 训练入口明确传入 `[4,1]`，预处理也保留全部光学波段，因此它可以归入直接 MSI–SAR 候选。EG-CMFNet 的默认配置同样是 4＋1；它的实际主路径是双 BiFormer，再加光学 ResNet 边缘支路，这比“一个 CNN 分支＋一个 Transformer 分支”的概括更准确。


### 进展 10/17/2026 09:35:03

已核实交接材料中的三处风险：MSAF 实际调用的版本有无条件断点，且注意力掩码在 eval 时仍随机；DFFNet 在优化器创建后才首次注册滤波参数；PICNet 在 forward 中临时创建压缩与解压卷积。这些仓库符合“核心代码公开”，但不适合原样作为复现底座。另外，MM-OVSeg 的 ZIP 内确实包含主网络和融合代码，不能因源码打包就排除。


### 进展 10/17/2026 09:38:47

筛选结果已基本稳定。CerraData-4MM 也需要从“预印本线索”改为正式论文：作者公开的期刊 PDF 确认它于 2025 年 7 月 3 日发表在 JSTARS，并验证了 12 通道 MSI＋2 通道 SAR 分类。另一个需要保留的限制是：EG-CMFNet 虽声明 4＋1 输入，但默认边缘支路仍按 3 通道构建，完整网络的多光谱适配不能只改配置。


### 进展 10/17/2026 09:40:24

正在收敛最终推荐，排序会与网页端交接稿有所调整：CFFormer、FDMF-Net 有更明确的 MSI–SAR 输入证据；EG-CMFNet 更适合作为 MAFI 模块来源；MSAF 的 patch 分类形式贴近你的任务，但需要先处理已确认的实现问题。报告会分别列出准入清单、源码限制和建议的对照实验，不按不同条件下的原论文 OA 排名。

