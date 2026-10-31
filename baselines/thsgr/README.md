# THSGR

> **2026-10-31 当前流程**：所有后续运行统一使用 MGHOFNet 训练协议 `mghofnet-v1`，公共实现为 `baselines/{protocol,data,training,prediction}.py`；优化、补零、增强和随机性设置以[当前实验协议](../../docs/experiments/README.md)为准。已完成实验仍属于各自保存的旧配置，本文此前的验证/运行记录不能视为新协议精度。此次只改代码，不启动正式实验；历史重评估按原填充处理。

论文：*Boosting Multimodal Remote Sensing Image Classification with Transformer-based Heterogeneously Salient Graph Representation*，IEEE TGRS 2026。

- 作者仓库：https://github.com/jqyang22/THSGR
- 固定来源 commit：`dd2a0ef7148c05a1261c38ab94845d9ff7b879a0`
- 许可证：Apache-2.0；原始代码与许可证保存在 [vendor/THSGR-original](../../vendor/THSGR-original/)。
- 模型名、配置标识和结果目录统一为 **THSGR / thsgr**。

## 项目接口与结构

`model.py::THSGR` 接收 `[B,1,C,P,P]`，先 MSI 后 SAR，输出 `[B,num_classes]` logits；类别数从共享数据配置读取。YRD/YRD2509NEW 为 10+4，GRSS07 为 6+1。patch 可配置，要求奇数且至少 9，已为 11/13 设计检查。

核心实现是 [thsgr_patch.py](../_hsi_adapters/thsgr_patch.py)。保留作者连续三层谱空卷积、模态共享显著节点编码、动态图卷积、相关性投影、五层跨模态卷积调制、mean-forward 与图／融合特征加权分类。64 维特征、65×64 token（含分类 token）、8×8 调制网格均是作者隐空间设置，不等于输入 patch 大小。节点数等于类别数，沿用作者配置每类一个节点；固定图分支权重 0.9，未按测试精度调节。

## 必要差异

1. 作者使用 PCA32、谱核深度 7/5/3，展平后固定为 `32*20` 通道。项目保留实际 MSI 波段，不使用 PCA、不复制或插值波段；依次取不大于当前谱深度的最大允许奇数核。MSI10 对应 7/3/1，MSI6 对应 5/1/1，均保留两层谱深度，展平投影按实际深度构造。空间卷积不补零，三层 3×3 加投影 3×3 后为 `(P-8)×(P-8)`。
2. SAR 第一层按实际 1/4 通道构造，移除原输入分支中轴置换和求均值的特殊处理；不将 SAR 解释为高度。
3. **mean-forward 仅训练时启用**：前馈输出与当前训练 batch 均值各占 1/2；测试／重评估／出图时禁止跨样本混合，使同一像素不因批次大小、顺序或出图范围改变预测。向量化替换作者逐样本原位赋值。该行为与作者原始 eval 有差异。
4. 移除原代码注册但未调用的 CNN 分支、静态图参数、独立 LiDAR 图模块、辅助分类头、未使用的归一化与 skipcat，以及未参与前向的 DropBlock 调度器；不引入新融合模块。保留作者实际调用的共享显著编码／图模块。
5. 训练对齐项目协议：普通交叉熵、AdamW lr0.001/wd0.01＋余弦衰减至1e-6、200 epochs、无预训练，最终 checkpoint 测试；不同于作者的标签平滑、120 epochs、lr0.01、衰减调度及最小训练 loss checkpoint。

这些差异影响实验解释；项目结果不表示复现作者论文数值，不预先承诺精度。

## 训练与出图

```bash
python scripts/train_eval_thsgr.py --config configs/thsgr_grss07_custom.yaml
python scripts/train_eval_thsgr.py --config configs/thsgr_yrd2509new_fs50.yaml
python scripts/train_eval_thsgr.py --config configs/thsgr_yrd_fs50.yaml
python scripts/eval_thsgr.py --run <run目录>
python scripts/infer_thsgr.py --run <run目录>
```

三份 YAML 复用 `configs/datasets/`：整图无标签逐通道 min-max，固定数量随机像元划分，patch11、train128/test1024、FP32，5 seeds `[202201,202202,202203,202204,202205]`。YRD/YRD2509NEW 每类50；GRSS07 custom `[674,1481,752,45,30]`。

结果路径：`runs_fewshot/fs50/thsgr/<dataset>/run_###/`；GRSS07：`runs_fewshot/grss07_custom/thsgr/grss07/run_###/`。保存配置与数据配置快照、`run_identity.json`、每次划分、checkpoint、逐次／汇总指标。`model_best.pth` 仅沿用兼容文件名，内容为最终 epoch。

测试仅使用各次保存测试划分。5 次完成后从 `metrics.json` 按最高测试 OA 选图，并列取首个；图覆盖全部标注位置（含训练样本），未标注为背景，不重新计分，仅另写 `labeled_map_info.json`。默认不做全图；显式 `infer_thsgr.py --mode full` 才遍历所有像元。

## 验证状态

CPU 合成数据、两种通道×patch11/13 的前后向／参数覆盖／checkpoint、eval 分批独立性及临时 MAT 数据训练→测试→重评估→出图链路均通过。116 V100 GPU 合成 train128/test1024 检查通过，峰值已分配显存约707 MiB；证据及远端部署记录见 [操作记录](../../docs/operations/2026-10-30-thsgr-adaptation.md)。GRSS07 真实数据 1 epoch/1 seed GPU 冒烟检查通过。正式实验精度尚未验证。
