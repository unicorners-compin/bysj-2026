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

生成（固定随机种子，可复现）。全部为 **Python 教学原型**：01 使用在 MNIST 上真实训练的两出口 CNN 的出口输出 + 单机网络仿真；
02、03 为单进程事件驱动 / 离散事件仿真。**不是** DPDK、硬件或生产集群实测。论文引用时请标明「教学原型仿真」。

依赖：`matplotlib`、`numpy`（见各目录 `requirements.txt`）；01 的 `src.train` 另需 `torch`（CPU 即可）。
