# 2025—2026 顶会定向补检：多光谱／高光谱多源分类与开源代码

> 检索日期：**2026-10-23**（Asia/Hong_Kong）。
> 定位：本项目已有 [2025—2026 多源遥感地物分类完整合并报告](2025-2026-multisource-classification-review.md)（检索截止 2026-10-17，以期刊为主）。本轮**只做顶会定向补检**，并复核其中少量顶会条目的当前状态，不重写原报告。
> 边界：仅外部论文与仓库**只读**检索；未安装依赖、未导入或运行第三方模型、未下载数据或权重、未修改项目源码、未启动任何实验。**“代码可访问”不等于“可复现”或“可商用”**；“正式录用”“工作区可见核心代码”“许可证”是三件分别核实的事。

## 1. 结论摘要（先读这一节）

1. **直接对应“MSI–SAR patch → 中心像素分类”的顶会论文仍然稀少。** 2025—2026 顶会在遥感方向的主战场是**基础模型、语义分割、开放词汇与多模态大模型**，不是本项目这种“双路配准小 patch 分类”。任务最接近的顶会候选是：
   - **T-APT**（AAAI 2026 主会，HSI–LiDAR→SAR 三源联合分类，代码可访问，前轮已核）；
   - **DPMamba**（IJCAI 2025 主会，HSI–LiDAR／SAR 缺失模态，代码可访问，前轮已核）；
   - **DiffCLIP**（AAAI 2025 主会，HSI–LiDAR 少标注 + 语言对齐，代码可访问，前轮已核）；
   - **UniDiff**（WACV 2026 主会，HSI–SAR 地物分类，扩散预训练 + 约 5% 参数适配）——**本轮未找到公开代码**。
   四者均有预训练／文本／Mamba 依赖，不能直接当作当前“无预训练、从零训练”的基线。
2. **多光谱地物分类的顶会产出以“多传感器基础模型/预训练编码器”为主**：AnySat（CVPR 2025，MIT，A 级）、TerraFM（ICLR 2026 Poster）、Copernicus-FM（ICCV 2025）、CARL（ICLR 2026 Poster）、GeoLink（NeurIPS 2025）、SARMAE（CVPR 2026，SAR 自监督）、OlmoEarth／Helios（CVPR 2026）、RAMEN（CVPR 2026，代码声明“接收后发布”）。它们适合“冻结或微调编码器”路线，与当前 27 项主实验协议不同，**必须单独成批、单独标注**，不能与从零训练结果混排。
3. **高光谱 + LiDAR／SAR 的顶会（含 Workshop）补充**：除上面 T-APT／DPMamba／DiffCLIP 外，本轮新查到 **CABIN**（AAAI 2026，HSI 半监督分类，未见代码）、**SSFT**（CVPR 2026 Workshop PBVS，轻量谱–空 Transformer，未见代码）、**CRF 决策融合**（CVPR 2026 Workshop PBVS，HSI+LiDAR+SAR 决策级融合，未见代码）。三项都可作为“机制参考”，但都**没有可直接复用的公开实现**。
4. **会议等级要分清**：CVPR／ICCV／NeurIPS／ICLR／AAAI／IJCAI 为主会；WACV 2026 是 CVF 正式主会（非 CVPR／ICCV 同级）；CVPR 2026 Workshop（PBVS、AI4Space 等）不是主会。本文全部显式标注，不统称“顶会”。
5. **对当前项目的保守建议**：优先做**不引入预训练**的机制对照（决策级融合、轻量谱–空双支路），把基础模型路线放到**独立预训练批次**；不引入 Mamba 系（与项目约定冲突，只读机制）；不把分割／开放词汇／基础模型工作当作 patch 分类性能对照（详见第 5 节）。

## 2. 检索口径与证据等级

| 项目 | 本轮做法 |
| --- | --- |
| 会议范围 | CVPR 2025/2026、ICCV 2025、WACV 2026、NeurIPS 2025、ICLR 2026、AAAI 2025/2026、IJCAI 2025、ACM MM 2025（ACM MM 2026 未检索到可用接收列表） |
| 论文状态核实 | CVF Open Access 单篇页面；AAAI OJS 文章页；OpenReview API；IJCAI Proceedings；arXiv `comments` 字段（作者自述，作为辅助，不单独作为录用证据） |
| 代码状态核实 | GitHub 仓库页面／GitHub Contents API 根目录文件列表（只读）；论文与 CVF 页面是否给出代码链接 |
| 与旧报告关系 | 旧报告已核条目注明“2026-10-17 已核，本轮未重审”；本轮新核条目给出具体 URL 与日期 |
| 已知检索缺口 | CVF `?day=all` 全量列表抓取被截断，只确认会议与部分标题，**未逐篇穷举**；AAAI OJS 站内搜索返回 403，只能单篇核实；ACM MM 2025／IJCAI 2026 未做系统检索 |

代码状态标签：

- **A**：核心模型、训练、评估与说明较完整（沿用旧报告定义）；
- **B**：核心模型公开，训练／数据流程有欠缺；
- **C**：核心可读但脚本零散、适配成本高；
- **R**：仓库存在，但只核到文件列表／README 级别；
- **未发布**：论文声明将发布但当前无可访问入口；
- **未找到**：论文与仓库页面均未给出代码入口（不等于作者永远不发布）；
- **待核**：有线索但本轮未完成核实。

## 3. 类别一：MSI–SAR／光学–SAR 多源分类（顶会）

| 论文 | 会议与状态 | 任务与输入 | 代码入口 | 代码状态 | 与本项目的关系／注意 |
| --- | --- | --- | --- | --- | --- |
| **T-APT**: Text-Guided Modality-Aware Prompt Tuning for Arbitrary Multimodal Remote Sensing Data Joint Classification | **AAAI 2026 主会**（OJS 42411；旧报告已核） | HSI–LiDAR、HSI–LiDAR–SAR 联合分类；文本引导、模态感知提示 | [Jiahuiqu/TATP](https://github.com/Jiahuiqu/TATP) | 前轮已核 B；**本轮复核**根目录可见 `modelALL_any_2_new.py`、`modelALL_any_3_new.py`、`run_this_for_all_two_modality.py`、`run_this_for_all_three_modality.py`、`mamba/`、`utils/`；未见 LICENSE 文件 | 最贴近“多源联合分类”的顶会主会条目；但依赖预训练 ViT／文本编码器，且仓库含 Mamba 代码。可参考“任意模态缺省”与提示对齐，**不作为无预训练轻量底座** |
| **DPMamba** | **IJCAI 2025 主会**，2224–2232（旧报告已核） | HSI–LiDAR／SAR 缺失模态分类 | [Jiahuiqu/DPMamba](https://github.com/Jiahuiqu/DPMamba) | 前轮已核 B；Mamba、文本预训练 | 参考缺失模态训练与蒸馏；与“优先非 Mamba”约定冲突，只读机制 |
| **DiffCLIP** | **AAAI 2025 主会**（OJS 34401；旧报告已核） | HSI–LiDAR 少标注分类、扩散预训练与语言对齐 | [icey-zhang/DiffCLIP](https://github.com/icey-zhang/DiffCLIP) | 前轮已核 B；需预训练体系 | 预训练成本高，不适合作为轻量融合底座 |
| **FVMGN** | **AAAI 2026 主会**（旧报告已核） | 多模态分类与模态泛化（训练／测试模态不一致） | [ZJier/FVMGN](https://github.com/ZJier/FVMGN) | 前轮已核 B；频率模块 + 视觉语言线索 | 若将来研究“SAR 通道缺失／模态不可靠”可参考；不作普通双模态主底座 |
| **UniDiff**: Parameter-Efficient Adaptation of Diffusion Models for Land Cover Classification with Multi-Modal Remotely Sensed Imagery and Sparse Annotations | **WACV 2026 主会**，4611–4620；[CVF 页面](https://openaccess.thecvf.com/content/WACV2026/html/Hu_UniDiff_Parameter-Efficient_Adaptation_of_Diffusion_Models_for_Land_Cover_Classification_WACV_2026_paper.html)；[arXiv 2512.00261](https://arxiv.org/abs/2512.00261) | HSI + SAR 地物分类；ImageNet 预训练扩散模型 + FiLM 模态条件 + 约 5% 参数适配 + 伪 RGB 锚定 | CVF 页面未给代码链接；arXiv 未给出 | **未找到公开代码** | 本轮检索中**任务最接近 HSI–SAR 分类**的新条目；但强依赖 ImageNet 扩散预训练与“稀疏标注”设定，和项目从零训练协议不同。若要试，需自写实现并单列协议 |
| **MM-OVSeg** | **CVPR 2026 主会**（旧报告已核） | optical／RGB–SAR 开放词汇**分割** | [Jimmyxichen/MM-OVSeg](https://github.com/Jimmyxichen/MM-OVSeg) | 前轮已核 B（实现含 ZIP，需区分原论文与后续实现） | 任务不同（分割）；只参考跨模态对齐与融合，不能当分类性能对照 |

> 说明：CerraData-4MM（JSTARS 2025）等是**期刊数据集**，不列入顶会表；旧报告已覆盖。

## 4. 类别二：高光谱 + LiDAR／SAR 分类（顶会与 Workshop）

| 论文 | 会议与状态 | 任务与输入 | 代码入口 | 代码状态 | 注意 |
| --- | --- | --- | --- | --- | --- |
| T-APT／DPMamba／DiffCLIP | 见第 3 节 | HSI–LiDAR（–SAR） | 见第 3 节 | 前轮已核／本轮复核 | 不在本表重复 |
| **CABIN**: Perceive, Act and Correct — Confidence Is Not Enough for Hyperspectral Classification | **AAAI 2026 主会**（[OJS 38147](https://ojs.aaai.org/index.php/AAAI/article/view/38147)；[arXiv 2511.10068](https://arxiv.org/abs/2511.10068) 注释自述录用） | HSI 半监督分类；不确定性估计 + 双采样伪标签 + 细粒度动态分配 | 论文页未给代码入口 | **未找到公开代码** | 方法（置信度不可靠、伪标签分可靠／模糊／噪声三档）可借鉴到“标注有限”的设定；单模态 HSI，不是多源融合 |
| **SSFT**: A Lightweight Spectral-Spatial Fusion Transformer for Generic Hyperspectral Classification | **CVPR 2026 Workshop (PBVS)**，7122–7131；[CVF 页面](https://www.openaccess.thecvf.com/content/CVPR2026W/PBVS/html/Musiat_SSFT_A_Lightweight_Spectral-Spatial_Fusion_Transformer_for_Generic_Hyperspectral_Classification_CVPRW_2026_paper.html)；[arXiv 2604.15828](https://arxiv.org/abs/2604.15828) | 通用 HSI 分类；谱支路 + 空支路 + 交叉注意力；参数量不足对比方法 2% | CVF／arXiv 页面均未给代码链接 | **未找到公开代码** | 结构简单、轻量，适合作为“轻量双支路”对照的**思想来源**；需自行实现。Workshop 非主会，性能不作直接排名 |
| **An End-to-End Trainable Multi-Scale CRF Framework for Decision Fusion in Multi-Modal Remote Sensing Classification** | **CVPR 2026 Workshop (PBVS)**，7027–7036；[CVF 页面](https://openaccess.thecvf.com/content/CVPR2026W/PBVS/html/Chaudhuri_An_End-to-End_Trainable_Multi-Scale_CRF_Framework_for_Decision_Fusion_in_CVPRW_2026_paper.html) | HSI + LiDAR + SAR + 多光谱**决策级融合**；KAN 编码器 + Fourier 块 + CRF 均值场（RNN 展开） | CVF 页面仅 PDF／supp，无代码链接 | **未找到公开代码** | 本轮唯一同时覆盖 HSI／LiDAR／SAR 的 CVPRW 分类工作；其“推理层决策融合 + 外观核”可与项目两级模糊融合做机制对照。Workshop 且无代码，复现成本高 |

> 旧报告已覆盖的期刊条目（**不是顶会，不进入上表**）：RSCNet（TGRS 2026）、DFFNet（GRSL 2025）、MSFMamba（TGRS 2025）等。本轮另见 M2Heat（IEEE TCSVT，arXiv 2609.11040，代码 `Weikan0425/M2Heat_HSI_LiDAR`）为 HSI–LiDAR 期刊工作；这些可在后续期刊补检中使用，但不能标成“顶会”。

## 5. 类别三：多光谱地物分类／多光谱基础模型（顶会）

| 论文 | 会议与状态 | 任务与输入 | 代码／权重入口 | 代码状态 | 注意 |
| --- | --- | --- | --- | --- | --- |
| **AnySat** | **CVPR 2025 主会**（旧报告已核） | Sentinel-1／Sentinel-2 等多传感器、多分辨率预训练；下游土地覆盖、作物／树种分类 | [gastruc/AnySat](https://github.com/gastruc/AnySat) | 前轮已核 A，MIT | 可作为冻结编码器单独实验；新通道需适配投影器，预训练条件与从零训练不同 |
| **TerraFM** | **ICLR 2026 Poster**（[OpenReview cBxuzdUDNx](https://openreview.net/forum?id=cBxuzdUDNx)） | Sentinel-1 + Sentinel-2 自监督基础模型；分类与分割；GEO-Bench／Copernicus-Bench | 权重线索：[HuggingFace MBZUAI/TerraFM](https://huggingface.co/MBZUAI/TerraFM)；论文称“code and pretrained models will be publicly released” | 权重仓库已见，**官方代码仓库未确认**（待核） | SAR+MSI 双模态最接近本项目传感器组合；但论文是预训练路线，不能与现有 27 项混排 |
| **Copernicus-FM** | **ICCV 2025 主会**，9888–9899；[CVF 页面](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Towards_a_Unified_Copernicus_Foundation_Model_for_Earth_Vision_ICCV_2025_paper.html) | Copernicus 多传感器（含多光谱、SAR）统一基础模型与基准 | [zhu-xlab/Copernicus-FM](https://github.com/zhu-xlab/Copernicus-FM) | **本轮已核 R**：根目录含 `Copernicus-FM/`、`Copernicus-Pretrain/`、`Copernicus-Bench/`、`LICENSE`、`NOTICE` | 预训练编码器候选；许可证需逐条读（仓库含 LICENSE 文件，本轮未读全文） |
| **CARL**: Camera-Agnostic Representation Learning | **ICLR 2026 Poster**（[OpenReview TpbhS1yfz0](https://openreview.net/forum?id=TpbhS1yfz0)） | RGB／多光谱／高光谱跨相机统一表示；分类与分割 | [IMSY-DKFZ/CARL](https://github.com/IMSY-DKFZ/CARL) | **本轮已核 R**：含 `carl/`、`configs/`、`main_cls.py`、`main_seg.py`、`main_ssl.py`、`LICENSE` | 对项目“6 通道／10 通道 MSI 输入不同”的通道数变化有直接借鉴意义；非遥感专用，需重训 |
| **GeoLink** | **NeurIPS 2025 主会**（[arXiv 2509.26016](https://arxiv.org/abs/2509.26016)） | 遥感基础模型 + OpenStreetMap 多模态预训练；下游土地覆盖分类等 | [bailubin/GeoLink_NeurIPS2025](https://github.com/bailubin/GeoLink_NeurIPS2025) | **本轮已核 R**：仓库可访问，标题注明官方实现与 checkpoint | 依赖 OSM，不是 MSI–SAR；只作“多模态预训练”线索 |
| **SARMAE** | **CVPR 2026 主会**（[arXiv 2512.16635](https://arxiv.org/abs/2512.16635) 注释自述录用） | SAR 自监督 MAE；SAR-1M 预训练；分类／检测／分割 | [MiliLab/SARMAE](https://github.com/MiliLab/SARMAE) | **本轮已核 R**：含 `SARMAE_Pretrain/`、`SARMAE_Fintune/`；根目录未见 LICENSE | 可用于 SAR 分支预训练，但与当前“无预训练”协议冲突，只能作为独立实验 |
| **OlmoEarth／Helios** | **CVPR 2026 主会**，34806–34817；[CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Herzog_OlmoEarth_Stable_Latent_Image_Modeling_for_Multimodal_Earth_Observation_CVPR_2026_paper.html) | 多模态时空 EO 基础模型（Helios）；24／29 个任务评测 | [allenai/olmoearth_pretrain](https://github.com/allenai/olmoearth_pretrain) | **本轮已核 R**：含训练与评测代码、`LICENSE`；注意 CVF 论文页代码链接被“REDACTED”，以仓库为准 | 通用大模型，训练／推理成本高；只作远期预训练路线 |
| **RAMEN** | **CVPR 2026 主会**，27838–27848；[CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Houdre_RAMEN_Resolution-Adjustable_Multimodal_Encoder_for_Earth_Observation_CVPR_2026_paper.html) | 分辨率可调、传感器无关的多模态 EO 编码器；PANGAEA | 论文写“code and pretrained model will be released upon acceptance”；CVF 页面无链接 | **未发布／待核** | 若发布，与项目“不同分辨率/通道配置”较契合；当前不可用 |
| **SkySense V2** | **ICCV 2025 主会**，9136–9146；[CVF 页面](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_SkySense_V2_A_Unified_Foundation_Model_for_Multi-modal_Remote_Sensing_ICCV_2025_paper.html) | 单一 Transformer 主干的多模态 RS 基础模型；16 数据集／7 任务 | CVF 页面未给代码链接；上一代 [SkySense CVPR 2024 仓库](https://github.com/Jack-bo1220/SkySense)存在 | **待核**（V2 官方入口未确认） | 只作线索，不进入推荐 |

> 另见但**不列入分类主表**：`WHU-MARS`（CVPR 2026，多光谱航空–地面 person re-ID 基准，不是地物分类）、`LUMEN`（CVPR 2026 Workshop AI4Space，火星矿物识别）、`UniGeoSeg`／`SkySense-VITA`（CVPR 2026，开放世界分割）、`TESSERA`（CVPR 2026，时序光谱表示）。这些与“MSI–SAR patch 地物分类”任务差异大。

## 6. 代码与许可证清单（本轮复核结果）

| 仓库 | 可见内容（只读根目录） | 许可证 | 结论 |
| --- | --- | --- | --- |
| `Jiahuiqu/TATP` | 模型文件、两／三模态运行脚本、`mamba/`、`utils/`、数据目录 | 根目录未见 LICENSE | 代码可访问；许可证未明；前轮定 B |
| `zhu-xlab/Copernicus-FM` | `Copernicus-FM/`、`Copernicus-Pretrain/`、`Copernicus-Bench/`、`LICENSE`、`NOTICE` | 含 LICENSE／NOTICE，本轮未读全文 | 代码可访问；许可证类型待读 |
| `IMSY-DKFZ/CARL` | `carl/`、`configs/`、`main_cls.py`、`main_seg.py`、`main_ssl.py`、`LICENSE` | 含 LICENSE，本轮未读全文 | 代码较完整（R，可进一步审 A/B） |
| `MiliLab/SARMAE` | `SARMAE_Pretrain/`、`SARMAE_Fintune/`、README、图 | 根目录未见 LICENSE | 代码可访问；许可证未明 |
| `allenai/olmoearth_pretrain` | `olmoearth_pretrain/`、`helios/`、`scripts/`、`tests/`、`LICENSE` | 含 LICENSE，本轮未读全文 | 训练／评测代码可访问 |
| `bailubin/GeoLink_NeurIPS2025` | 官方实现与 checkpoint（仓库标题说明） | 本轮未核 | 代码可访问；细节待审 |
| `gastruc/AnySat` | 前轮已核 A；MIT | MIT | 沿用前轮结论 |
| `Jimmyxichen/MM-OVSeg` | 前轮已核 B；实现含 ZIP | 前轮未明确 | 沿用前轮结论 |
| UniDiff／SSFT／CRF-CVPRW／CABIN／RAMEN／SkySense V2 | 论文页未给代码入口或声明“将发布” | — | **未找到／未发布**，不能称为“开源代码” |

## 7. 与当前项目结合：建议与不建议

**当前项目事实**（以项目文档为准，不在本文件重述细节）：三数据集（YRD 8 类每类 50、YRD2509NEW 9 类每类 50、GRSS07 custom 5 类）、seed 202201、patch 11、200 epochs、train 128／test 1024、FP32、无预训练、从零训练；已有 9 个版本、27 项结果。任何引入预训练或新协议的实验都**不能与现有 27 项直接排名比较**。

### 建议（按成本／风险从低到高）

1. **机制对照 R1：决策级／后融合**。受 CVPR 2026 Workshop CRF 决策融合与 T-APT 的模态提示思想启发，在现有分类头上做“两个分支各自出分类分布 + 可学习／一致性约束的决策融合”，只改分类头与损失，不引入新的大模块。**无公开代码，需自行实现**；优点是与现有 V6 结构兼容、便于单模块消融；缺点是没有官方实现可对照。
2. **机制对照 R2：轻量谱–空双支路**。借鉴 SSFT 的“谱支路 + 空支路 + 交叉注意力 + 极少参数”，作为当前 VBE 的轻量对照。**无公开代码，需自行实现**；适合小 patch、小样本，但必须做参数与感受野对齐后的消融，不能直接与 RF 级大模型比精度。
3. **预训练路线 R3（独立批次）**：AnySat／CARL／TerraFM／Copernicus-FM 作为冻结或部分解冻编码器，加通道投影器适配 6+1／10+4 输入。必须**单独成批、单独标注“预训练”**，并在文档中记录预训练数据、权重来源、训练预算与选模规则；先做可行性验证（通道投影、显存、数据量），不急于发正式实验清单。
4. **SAR 分支自监督 R4（远期）**：SARMAE 可作为 SAR 单支路预训练参照；与当前“无预训练”协议冲突，仅作为未来独立课题。

### 不建议

- 把 **MM-OVSeg、UniGeoSeg、SkySense-VITA** 等**分割／开放世界**工作当作 patch 分类的精度对照；任务差异必须先说明。
- 把 **基础模型论文（TerraFM／Copernicus-FM／OlmoEarth／SkySense V2）** 的公开数值当作能在本项目数据上复现的目标；它们的预训练数据、分辨率与评测协议不同。
- 直接接入 **Mamba 系**（DPMamba、T-APT 的 `mamba/` 子目录）：与“优先不依赖 Mamba／SSM”的项目约定冲突，只读机制（缺失模态、提示对齐）。
- 把“有 GitHub 仓库”写成“已开源可复现”，或把“会议录用”写成“性能已核”。本文件所有候选**均未运行**。

## 8. 待核项与下一轮检索建议

1. **代码待核**：TerraFM 官方代码仓库；SkySense V2 代码入口；RAMEN 发布状态；UniDiff、SSFT、CRF-CVPRW、CABIN 的代码（可联系作者或等仓库发布）。
2. **许可证待读**：TATP、SARMAE 未见 LICENSE；Copernicus-FM／CARL／OlmoEarth 含 LICENSE 但本轮未读全文；第三方许可证与论文代码的对应关系需单独确认。
3. **检索缺口**：CVF `?day=all` 全量抓取被截断；AAAI OJS 站内搜索 403（只能逐篇核实）；ACM MM 2025／ACM MM 2026、IJCAI 2026 未系统检索；ICLR 2026 与 CVPR 2026 Workshop 仅部分条目。**“本文没有列出的顶会工作”不等于不存在。**
4. **建议下一轮**：以 CVF Open Access + OpenReview + AAAI OJS + IJCAI Proceedings 为主索引，按关键词（MSI–SAR、Sentinel-1/2 land cover、hyperspectral LiDAR/SAR classification、patch classification）做程序化筛选；对每个候选记录“会议页 URL + 代码 URL + 最后提交日期 + 许可证 + 输入模态与通道”五元组，避免把预印本或 Workshop 写成主会。
5. 本文件不修改旧报告原文；旧报告中的期刊主表与源码审查结论仍然有效，顶会补充以本文为准（检索日期 2026-10-23）。

## 附录 A：本轮实际检索入口

- CVF Open Access：`CVPR2026?day=all`、`ICCV2025?day=all`（截断）、WACV 2026／CVPR 2026／ICCV 2025／CVPR 2026 Workshop 单篇页面。
- OpenReview：`api2.openreview.net/notes/search`，`group=ICLR.cc/2026/Conference`，关键词 `hyperspectral`、`TerraFM`、`CARL`。
- arXiv API：`abs:"hyperspectral" AND abs:"LiDAR"`、`abs:"hyperspectral" AND abs:"SAR"`、`abs:"hyperspectral" AND abs:"classification"`、`co:"CVPR 2026" AND abs:"remote sensing"`、`co:"AAAI 2026" AND abs:"remote sensing"`、`co:"NeurIPS 2025" AND abs:"remote sensing"` 等。
- GitHub 只读：`api.github.com/repos/<owner>/<repo>/contents/` 根目录列表；仓库页面标题。
- 交叉来源：Paper-Notes 索引仓库（`zhaoyang97/Paper-Notes-en`）仅用于发现候选，会议状态均回到官方页面核实。

## 附录 B：本项目已有顶会条目（旧报告已核，本轮未重审）

| 条目 | 会议 | 类型 | 当前用途 |
| --- | --- | --- | --- |
| MM-OVSeg | CVPR 2026 主会 | 光学–SAR 开放词汇分割 | 只参考融合／对齐 |
| AnySat | CVPR 2025 主会 | 多传感器预训练编码器（MIT，A） | 预训练路线候选 |
| TerraMind | ICCV 2025 主会 | 多传感器生成式基础模型 | 通用基础模型专题 |
| T-APT | AAAI 2026 主会 | HSI–LiDAR–SAR 联合分类 | 多源联合分类参考 |
| FVMGN | AAAI 2026 主会 | 多模态分类与模态泛化 | 模态缺失研究参考 |
| DiffCLIP | AAAI 2025 主会 | HSI–LiDAR 少标注 | 预训练／语义对齐参考 |
| DPMamba | IJCAI 2025 主会 | HSI–LiDAR／SAR 缺失模态 | 缺失模态参考（Mamba） |

## 附录 C：同日直接检索补充线索（2026-10-23，未升级为主表）

以下条目来自同日用 `web_search`／arXiv API／CVF 单篇页面直接检索（**未使用 MCP 通道**），但**不满足“顶会主会＋核心代码”准入**，只作线索记录，不进入第 3—5 节主表：

| 条目 | 来源与状态 | 任务／输入 | 代码 | 与项目关系 |
| --- | --- | --- | --- | --- |
| **Infra-Bench CLS** | [arXiv 2609.09482](https://arxiv.org/abs/2609.09482)（2026-09），预印本，自述投 Big Earth Data | Sentinel-1 SAR + Sentinel-2 MSI 设施级分类基准；7 个 EO 基础模型做线性探测／微调对比，最佳 macro-F1 57.9%（ResNet-18 基线 39.2%） | 论文未声明代码 | 可作为“基础模型在 S1+S2 小样本分类”的评估参照，不提供新网络 |
| **Space2Ground 2.0** | [arXiv 2607.28247](https://arxiv.org/abs/2607.28247)（2026-07），自述被 EARSeL 2026 接收 | Sentinel-1 SAR + Sentinel-2 时序 + 街景影像，地块级作物分类；公布数据集 | 数据集 Zenodo 公开，代码未核 | 多源数据组织与评估参考，非新的融合网络 |
| **Earth-OneVision** | [arXiv 2606.10819](https://arxiv.org/abs/2606.10819)（2026-06），预印本 | 2B 遥感多模态大模型，统一 6 类传感器（含 SAR、MSI）；BigEarthNet-MS 多光谱分类 recall 75.74% | 未核 | 多模态大模型路线，不是 patch 地物分类底座 |
| **MAVIC-C 2026** | [CVPR 2026 Workshop PBVS, pp.7152–7160](https://openaccess.thecvf.com/content/CVPR2026W/PBVS/html/Ramos_5th_Multi-modal_Aerial_View_Imagery_Challenge_Classification_-_PBVS_2026_CVPRW_2026_paper.html) | EO+SAR 双流跨模态，SAR 空中目标分类与 OOD 检测；UNICORN V2 数据集，63 队参赛 | 页面无代码链接 | 跨模态迁移／OOD 机制参考；任务是目标分类，不是地物分类 |
| AAAI 2026 遥感论文第三方汇总（7 篇） | 第三方 notes 页面（m0rtzz.github.io/paper-notes），本轮抓取失败，仅见搜索摘要 | 含 CABIN 等条目 | — | 仅用于发现候选；会议状态必须回到 AAAI OJS 核实 |

证据边界：以上只核到摘要／页面级别，未读全部正文、未运行代码；第三方汇总页不是录用证据，本轮也未把它作为新增正式条目。
