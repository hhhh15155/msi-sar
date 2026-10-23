---
name: msi-sar-baseline-adaptation
description: Use when an agent adds, integrates, adapts, or aligns baseline models in the MSI-SAR project, including HSI-LiDAR to MSI-SAR ports, modality channels, patch shapes, training configs, and evaluation or map-selection protocols. 适用于本项目新增baseline、模型适配、配置对齐；仅调研或明确仅收录源码时保持只读范围。
---

# MSI–SAR baseline 适配

目标：用户要求“引入一个 baseline”时，主动交付项目可用的适配实现与明确验证状态；只复制源码不算完成。

## 必须读取

从当前工作目录定位 MSI–SAR 仓库根目录，读取适用的 [AGENTS.md](../../../AGENTS.md)、[模型适配规则](../../../docs/models/baseline-adaptation.md) 和 [当前实验协议](../../../docs/experiments/README.md)。本 skill 的链接相对仓库内本文件位置解析。用户最新明确指令优先；不要把旧文档的单 seed 或禁止测试 OA 选图当作当前规则。

## 工作流程

1. **确定范围与来源**：检查本地分支、HEAD、工作区、现有调用链与作者源码 commit／许可；保留用户改动。仅调研或明确仅收录源码的请求不自动转成实现任务。
2. **主动实现适配**：读取真实数据配置，显式分流 MSI／SAR；按 `10/4`、`6/1` 两种输入核对光谱核、分组、token、位置编码、分类头与可配置 patch。包装 `[B,1,C,H,W] → [B,num_classes]` 稳定 logits，核心机制保留，形状／接口差异记入 README。具体规则见上述模型适配文档。
3. **补齐交付**：模型包、训练／评估／推理可执行入口、三个数据集的主实验 YAML、来源与验证状态、索引。只创建 vendor／README／配置不能标记集成完成。核心源码缺失时列出真实阻塞，不编造网络；其余独立部分继续完成。
4. **对齐协议**：按当前实验文档设置 5 seeds `202201–202205`、最终 checkpoint、`eval.run_index: best`。全部预定运行结束后，以各次测试 OA 最高者出图，并列取配置顺序首个。保留所有种子与均值／标准差；最佳图不能替代平均指标，不用测试集选 epoch／超参数。
5. **验证并诚实标记**：静态检查路径／语法／配置；已有依赖时做合成数据前向／反向、参数注册、checkpoint 保存／重载检查，覆盖两种通道组合。未执行则标记“未运行验证”，不承诺精度；源码完整但本机无依赖不算适配成功已验证。

## 执行边界与交付

**测试、标注样本图、全图是三个独立操作**：测试（含 `evaluate_run`）仅对保存的测试划分计算 OA／AA／Kappa，不先跑全图。默认训练后从测试 `metrics.json` 选最佳运行，调用 `save_labeled_maps` 对全部有标注位置推理绘图；其中包含训练位置是预期行为，不是测试泄漏。未标注位置留背景；出图不重新计分、不生成混合准确率，不覆盖 `metrics.json`／`eval_metrics.json`，只保存图和 `labeled_map_info.json`（run、seed、既有测试 OA、选图依据）。`save_full_map` 才遍历所有像元，默认关闭自动全图；只有用户明确需要全图时才调用独立入口。不能把“标注样本图”误改成“仅测试样本图”。

引入模型并不自动授权数据集训练、全图推理、服务器执行、依赖安装／升级或重跑已完成实验。合成数据检查的目的仅是验证实现。授权具体实验后只执行相应未完成项。

交付报告包含：实际实现与配置、验证证据、阻塞／未验证项；正式运行另记录版本、配置／数据配置快照、种子、划分、环境、checkpoint、逐次与汇总指标及选图 run／seed。项目数值和适配细则只维护在上方链接文档中；发生变更同步检查 YAML、AGENTS.md 和模型 README。
