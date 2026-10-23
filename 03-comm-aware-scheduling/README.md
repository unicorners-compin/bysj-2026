# 03 通信负载感知调度

按链路/节点通信负载为任务选放置点，偏向低负载路径。

> 数值与图均为 **Python 玩具仿真**，**不是** 真实集群轨迹。

```bash
python3 -m src.simulate
python3 -m src.experiment   # CPU-only vs 通信感知对比 + 出图 → figs/
python3 -m pytest -q
```

`figs/` 由 `src.experiment` 重新生成：
- `policy_avg_score.png` — 策略平均放置得分（越低越好）
- `policy_comm_balance.png` — 放置后通信负载标准差（越低越均衡）
