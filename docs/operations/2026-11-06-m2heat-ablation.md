# 2026-11-06 M2Heat 消融部署

- 用户授权：只新增三组消融，原训练脚本与Full不变；先同步源码，再在110 GPU0以screen部署45次训练。
- 开始状态：本地/110/116 HEAD 均为 `9752e05368d882bca840010bf7b951427e80d8cf`。本地与110已有 `vendor/MSFMamba-original` 未跟踪内容状态，保留不动。
- 运行环境：110 `/home/fangyan/.conda/envs/gjc/bin/python`，PyTorch `2.1.1+cu118`、NumPy `1.26.4`，screen已有；不安装依赖。
- 本地 `C:/Users/goh/.conda/envs/gjc/python.exe` 为PyTorch2.5.1；三变体18组合（3数据集×2patch×3变体）的forward/backward、逐值旁路等价、checkpoint重载已通过。端到端检查缺少scikit-learn，改用110现有环境，不安装。
- 本地原入口/协议回归：`pytest tests/test_training_protocol.py tests/test_baseline_map_protocol.py tests/test_hsi_adapter_models.py -q`，15项通过。
- 110已有环境不含pytest，用unittest执行新消融检查；5项通过（含18组合和微型合成图5 seeds端到端），另原标注图协议4项、旧adapter检查1项通过。临时验证目录 `/tmp/msi-sar-m2heat-check-20261106`；未读取正式数据训练。
- 110合成GPU检查：9组合（3变体×3通道/类别设置）全部通过，train128反传+AdamW更新、test1024推理；最大allocated约755 MiB、reserved1160 MiB。使用公共set_seed及FP32，与正式协议一致。最初profile在CUDA初始化前重置峰值失败，已修正并重新通过。
- 审查修复：队列准入扣除其他进程显存且为活动任务保留完整预算；重复调用不重写已有组的配置/数据/协议身份记录。原模型、train/eval与三份主配置均无差异。
- 部署计划：screen `m2heat_ablation_20261106_110`；batch `runs/m2heat_ablation/20261106_110`；总日志 `experiment_logs/m2heat_ablation_20261106_110/screen.log`；GPU0最多3并发，每任务保留4096MiB并额外留1024MiB。外部进程显存约6682MiB时可容纳3任务（6682+3×4096+1024<24576）。每任务包含同组5seeds依次执行。
- 队列PID/子任务PID、命令、逐项日志和状态写入batch内 `queue_status.json`；每组独立日志位于batch的 `logs/`。结束自动生成汇总，失败不自动重试。实际部署提交由队列 `commit` 与各输出 `training_protocol.json` 记录。

方案与命令见[消融说明](../experiments/m2heat-ablation.md)。
