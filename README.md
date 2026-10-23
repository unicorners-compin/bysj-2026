# bysj-2026

西电本科毕业设计 2026 — 三个课题原型代码（同一仓库、三个子目录）。

| 目录 | 课题 |
|------|------|
| `01-latency-aware-early-exit` | 时延感知早退（推理/服务路径） |
| `02-self-describing-fault-recovery` | 自描述故障恢复 |
| `03-comm-aware-scheduling` | 通信负载感知调度 |

每个子目录均可独立运行：进入目录后见各自 `README.md`。

## 仿真与图

各课题 `figs/` 下的 CSV/PNG 由对应目录内：

```bash
python3 -m src.experiment
```

生成。全部为 **本科友好的 Python 玩具仿真**（参数扫描 / 蒙特卡洛），**不是** DPDK、硬件或生产集群实测。论文引用时请标明「仿真示意」。

依赖：`matplotlib`、`numpy`（见各目录 `requirements.txt`）。
