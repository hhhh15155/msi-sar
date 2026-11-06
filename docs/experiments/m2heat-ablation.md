# M2Heat 核心模块消融

## 范围与实施

2026-11-06 用户确认：不新增、不重跑 Full；保留原模型、训练/评估入口与原配置；另加三个消融。随后授权先同步代码，再在 server110 GPU0 用 screen 部署任务。正式任务为 3 变体 × 3 数据集 × 5 seeds，共 45 次训练。

实际模型见 [vendor forward](../../vendor/M2Heat-original/m2heat.py)。Stem 各输出128通道，拆成两个64通道分支：前半进入各自专属 HCLayer，后半拼接为128通道进入共享 HCLayer。专属融合输出64通道，共享输出经原1×1投影降至64通道，原 `cof1`、`cof2` 加权相加后进入原分类头。

| 变体 | 操作 |
| --- | --- |
| `wo_specific_hco` | 绕过整个 MSI/SAR 专属 HCLayer（含其中残差FFN），保留拆分后的原Stem特征，CFF与共享分支保持 |
| `wo_shared_hco` | 绕过整个共享 HCLayer，原始拼接特征直接进入原 `proj` |
| `wo_cff` | 整个CFF调用替换为专属HCLayer输出逐元素平均；三个HCLayer全部保留 |

不修改HCO/FVE/分类头内部实现，不增加动态扩散、动态融合、损失或训练策略。初始化按原完整模型顺序完成，再移除被旁路模块及其专属FVE参数，保证保留权重初始化与原模型一致，也不把无梯度废弃参数交给优化器。Full继续使用原模型与已有结果。

实现见 [ablation.py](../../baselines/m2heat/ablation.py)、[独立训练编排](../../baselines/m2heat/ablation_experiment.py)，9份配置在 [消融配置目录](../../configs/m2heat_ablation/)。对照其他参数来自原三份M2Heat主实验配置；数据、增强、种子和优化均复用公共模块，详见[统一协议](README.md)。最终 checkpoint 名称沿用 `model_best.pth`，实际保存最后 epoch；五次测试全部完成后按最高测试OA选标注图，不启动全图。

## 执行顺序与验证

1. 本地检查三变体在10/4通道（8/9类）、6/1通道（5类）及patch11/13的前反向、保留权重初始化、恒等/平均旁路等价和checkpoint重载。
2. 合成小图跑独立训练入口，核对五seeds、划分、指标及重复调用不覆盖checkpoint。
3. 审计统一协议及原入口回归；110做三变体train128/eval1024的合成GPU峰值测量，不安装依赖。
4. 通过检查后提交到本地main、推送origin/main、110快进相同提交。保留现有MSFMamba vendor未跟踪状态。
5. screen启动独立队列，记录提交、PID、GPU、日志、输出和状态；只处理消融，失败不自动重试，不删除旧结果。

## 命令

从仓库根目录执行，`PY` 指110现有 `/home/fangyan/.conda/envs/gjc/bin/python`。

```bash
# 只预览9组配置，不训练
$PY scripts/run_m2heat_ablations.py --batch-dir runs/m2heat_ablation/BATCH

# 单组5 seeds（不与同一batch中的相同任务并行启动）
$PY scripts/train_eval_m2heat_ablation.py --config configs/m2heat_ablation/wo_specific_hco_yrd_fs50.yaml --batch-dir runs/m2heat_ablation/BATCH
# 替换为wo_shared_hco或wo_cff，以及yrd2509new_fs50/grss07_custom即可切换
# 单seed调试可以增加 --seed 202201

# 合成验证，不读取真实数据
$PY -m unittest discover -s tests -p test_m2heat_ablation.py
CUDA_VISIBLE_DEVICES=0 $PY scripts/check_m2heat_ablations.py --device cuda:0 --profile --output experiment_logs/m2heat_ablation_profile.json

# 整批执行，提交/显存保留量使用实际核验值；必须在screen后台启动
$PY scripts/run_m2heat_ablations.py --batch-dir runs/m2heat_ablation/BATCH --gpu 0 --max-parallel 1 --reserve-mib RESERVATION --expected-commit COMMIT --execute

# 读取已保存的测试指标，不重评估、不训练
$PY scripts/summarize_m2heat_ablations.py --batch-dir runs/m2heat_ablation/BATCH
```

每个新批次要求独立目录，不覆盖旧批次。每组输出 `<batch>/<variant>/<dataset>/`，每个seed独立 `seed_202201/` 等目录，含 `log.txt`、`split.npz`、配置/数据/协议快照、`checkpoints/`、`metrics.json`。组级 `metrics.json` 保存五次与OA/AA/Kappa均值和总体标准差（ddof=0）；批次级 `seed_metrics.csv` 保留各次OA/AA/Kappa和各类准确率，`summary.csv`/`summary.md` 汇总完整五次组。缺失/无效组明确列出。

原 Full 主实验结果路径：`runs_fewshot/fs50/m2heat/{yrd,yrd2509new}/run_001/`、`runs_fewshot/grss07_custom/m2heat/grss07/run_001/`，只读保留，不自动混入新消融均值。

具体验证证据、screen名、提交和批次路径见本轮[部署记录](../operations/2026-11-06-m2heat-ablation.md)。
