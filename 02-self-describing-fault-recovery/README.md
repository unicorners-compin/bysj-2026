# 02 基于自描述任务的故障恢复（Python 教学原型）

单进程**事件驱动仿真**：1 个协调端 + 4 个工作节点，作业 = 12 个分片任务 × 3 个确定性阶段（过滤/变换/聚合）。
阶段输出由 numpy **真实计算**，每次故障运行的最终结果摘要都与无故障参考比对；时间为仿真秒（心跳 0.5 s、租约 2 s、
状态传输 50 MB/s + 2 ms）。`src/task.py` 定义自描述任务 `TaskDesc`，`src/fault.py` 定义自描述故障 `FaultDesc`
与 `plan_recovery`（retry / degrade / isolate），`src/des.py` 为协调端/工作节点协议仿真。
> 非多机部署、非 DPDK、非生产遥测。

```bash
python3 -m src.experiment   # λ 调参(调参配置) + 实验A(状态保存策略) + 实验B(故障类型×恢复策略) + 出图
python3 -m pytest -q
```

- 实验 A：4 种规模组合(A/B/C/D) × 3 故障阶段 × 3 故障时机 × 5 次重复 × 3 策略（整体重算/每阶段保存/选择性保存）。
- 实验 B：致命/瞬态/性能退化 × 无恢复/盲目重试/自描述策略化恢复，各 200 次；另含「盲目重试且关闭去重」消融。
- 输出：`lambda_tuning.csv`、`expA_*.csv`、`expB_*.csv`，`fig1`–`fig8` PNG。
