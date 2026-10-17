# 2026-10-17 版本统一记录

## 基准与范围

用户明确指定以 `D:\gaojianchun\msi-sar` 本地工作目录为准，统一 GitHub `hhhh15155/msi-sar` 的 `main` 分支及 `server110:/ssd/gaojianchun/msi-sar`。

统一前核查：

| 副本 | 分支 | HEAD | 工作区 |
| --- | --- | --- | --- |
| Windows 本地 | main | `1402a141ae68d61685e6a9738d99cba3e5333cdc` | 7 个已修改文件，另有 Fuzzy Transformer、配置、审查记录和 AGENTS 等未跟踪文件 |
| GitHub origin | main | `37f9a70ccbee046e6bdbe48027ec79d6475a9765` | 远程分支比本地少 1 个提交，没有独有提交 |
| server110 | main | `37f9a70ccbee046e6bdbe48027ec79d6475a9765` | 只有上一任务添加的未跟踪 `AGENTS.md`，没有已跟踪源码改动 |

本地领先的提交已移除 VBE-Net；统一时保留此决定，不从旧版本重新引入 VBE-Net。现有数据与实验结果中的旧方法目录不属于源码统一范围，不清理。

本次纳入本地原有的 Fuzzy Transformer 模型、训练／评估／推理入口、22 份配置、调度与汇总改动、4 份源码调研 JSON 记录及 `AGENTS.md`。不在此任务中修改模型行为或重新开展复现验证。

新增 `.gitattributes` 为 `*.sh` 指定 LF 换行，避免 Windows 检出后引入不兼容 Linux 的 CRLF；没有对全仓库进行换行重写。

版本统一当时，完整文献合并报告仍位于 Windows 下载目录，未纳入该轮代码同步。随后的文档整理任务已将它原文迁入 [docs/research/2025-2026-multisource-classification-review.md](../research/2025-2026-multisource-classification-review.md)；当前入口以 [研究索引](../research/README.md) 为准。

## 同步前备份

- 本地：`D:\gaojianchun\msi-sar-backups\20261017-183340`
  - `repository.bundle`：同步前本地 Git 引用与历史。
  - `working-source.tar.gz`：原有已跟踪及未忽略的未跟踪工作文件，包含未提交内容。
  - `working-tree.patch`、`status-before.txt`、`working-files.txt`：差异、状态与文件清单。
- 服务器：`/ssd/gaojianchun/msi-sar-backups/20261017-183340`
  - `repository.bundle`：同步前服务器 Git 引用与历史。
  - `tracked-source.tar.gz`：旧 HEAD 的源码快照。
  - `AGENTS.before.md`、`status-before.txt`：原有说明及工作区状态。

两份 bundle 在生成后均通过 `git bundle verify`。服务器已有数据、权重、结果和日志原地保留，没有纳入上述源码备份；这些源码备份不能当成数据备份。

## 同步与核验规则

1. 将已核查的本地工作文件提交到 `main`，包含之前未提交的工作；不安装依赖或运行训练。
2. 使用普通 push 更新 GitHub，不强制推送；远端分支若意外发生独立变动，应停止并重新比较。
3. 服务器先保存原有 `AGENTS.md`，再通过 `git merge --ff-only` 更新。不得使用 `reset --hard` 或 `git clean`。
4. 主仓库的子模块引用同样按本地基准核对。服务器原先未初始化 `vendor/MSFMamba-original`，应补齐到主仓库指定提交，不运行其中代码。
5. 验证本地 HEAD、GitHub `main`、服务器 HEAD 相同，工作区无待提交源码改动，子模块提交一致。
6. 做 Python 源码语法解析和 shell 脚本 `bash -n` 静态检查；不将这些检查表述为训练、依赖兼容性或性能验证。

本记录与统一后的源码一同提交。实际最终提交号以 `git rev-parse HEAD` 为准，最终检查结果由本次任务交付与备份目录中的核验记录保存，避免在本提交内自引用其哈希。

## 后续使用

- 日常在本地开发，检查差异后提交并推送；在服务器先核对工作区，再快进到指定提交。
- 不在服务器随手改一套未回收源码；运行实验前记录提交号与配置快照。
- 数据、checkpoint、日志及结果由各端独立管理，不因代码一致而假定数据一致。
- 后续若要回退，先检查当前改动并保留，再基于备份或新分支恢复指定版本；不要直接覆盖整个项目。
