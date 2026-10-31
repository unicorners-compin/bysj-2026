# 03 通信与负载感知任务调度（Python 教学原型）

学生：闫昉炜（2023级通信工程，学号 23009101169）

指导教师：任智源（西安电子科技大学通信工程学院）

`src/des.py`：8 节点（2 区域 × 3 近端 + 2 远端，远端算力 4 倍）离散事件仿真；任务泊松到达、数据量对数正态、计算量指数分布；
每个节点入口链路 FIFO 串行化（带宽竞争），节点 FCFS + 有限队列；负载信息默认每 20 ms 刷新一次快照。
策略：随机 / 轮询 / 最近节点 / 最小负载(任务数) / 仅通信感知(β=0) / 仅负载感知(α=0) / 联合感知 / 联合感知+迁移(收益阈值 θ)。
> 全部为仿真时间（ms），非真实网络、非可编程交换机、非 DPDK。

```bash
python3 -m src.experiment   # θ 调参(独立调参场景) + 主实验(C1/C2/C3 × 8 策略 × 5 种子 × 1e4 任务) + 4 组扫描 + 出图
python3 -m pytest -q
```

输出：`theta_tuning.csv`、`main_results_per_seed.csv`、`main_results_summary.csv`、`load_sweep.csv`、`bandwidth_sweep.csv`、
`theta_sweep.csv`、`staleness_sweep.csv`，`fig1`–`fig8` PNG。旧版 `src/scheduler.py`（静态打分放置）保留作对照示例。
