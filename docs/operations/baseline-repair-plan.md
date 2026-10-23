# Baseline 修复实施记录

目标：按用户明确区分测试与出图的要求，修复 DFFNet、M2Heat、MFT，实现“测试最佳运行 → 仅标注样本图”为默认流程，全图另行调用。

约束：保留已有工作区修改；不运行正式实验、不访问服务器、不安装依赖、不修改 vendor。当前 9 份主实验配置保持 5 seeds、patch11、200 epochs、train128／test1024、FP32。

## 实施步骤

- [x] 新增回归检查并先观察失败：vendor 导入、MFT 前向维度、测试掩码、标注图不计算测试指标／不覆盖测试文件、最佳运行选择与全图隔离。
- [x] 修复两处 vendor 路径；MFT 恢复原 notebook 的 FM16／64维 token，删除仅用于转置的非必要 einops 依赖。
- [x] 三模型的 `evaluate_run` 仅推理保存的 test_gt；`save_labeled_maps` 只保存全标注图与选图信息；`save_full_map` 保留为显式全图入口。共享选图逻辑依据测试 `metrics.json`，并列取首个。
- [x] 补齐三模型已有 run 的测试／出图 CLI，训练入口只生成标注图；配置的全图自动开关关闭。
- [x] 更新项目 skill、AGENTS.md、模型／实验／脚本说明，说明训练位置参与标注图推理是预期行为。
- [x] 运行回归检查与现有测试套件；已有 PyTorch 环境运行 CPU 合成前向／反向、参数覆盖与 checkpoint 重载检查；独立复核改动（报告单独核对）。

验证重点：最高 OA 使用测试指标；出图不重新计分；未标注位置留背景；全图不出现在默认训练／重评估链路；测试与图文件互不覆盖。

记录：默认 Python 缺少 PyTorch；本机已有 `C:\Users\goh\.conda\envs\gjc\python.exe` 可运行 PyTorch 2.5.1，但缺少 einops／h5py／sklearn，不安装依赖。模型检查与不依赖模型的数据／流程测试分别在可用环境执行，未实际验证的部分明确列出。

## 验证结果

- `python -s -m pytest -q`：21 passed、1 skipped。跳过项为默认 Python 无 PyTorch 的合成模型检查，该项已在已有 gjc 环境单独执行通过。
- 本机 gjc Python 执行 `tests/test_hsi_adapter_models.py`：通过；覆盖 3 模型 × 2 通道组合 × 2 patch，共12个组合，验证前向／反向／梯度／参数覆盖／checkpoint重载。
- 6 个新增 CLI 的 `--help`、9 份配置的5 seeds／best选图／关闭自动全图、skill格式和链接、源码语法及 `git diff --check` 均通过。
- 默认 Python 的首次全量 pytest 因用户目录 NumPy 2.5 与旧 SciPy 二进制不兼容，在 `tests/test_prepare_grss07.py` 收集阶段失败；本地使用 `python -s` 排除用户包后全部可用检查通过，没有安装、升级或删除依赖。
- 未启动真实数据训练、全图推理或服务器任务，未同步服务器。CPU合成通过不代表真实数据精度已验证。
