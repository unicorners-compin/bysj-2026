# 02 自描述故障恢复

故障携带结构化自描述信息，恢复器按描述选择策略（重试 / 降级 / 隔离）。

> 数值与图均为 **Python 玩具仿真**，**不是** 生产环境遥测。

```bash
python3 -m src.simulate
python3 -m src.experiment   # 批量故障仿真 + 出图 → figs/
python3 -m pytest -q
```

`figs/` 由 `src.experiment` 重新生成：
- `recovery_action_distribution.png` — 各故障类型的恢复动作分布
- `transient_retry_histogram.png` — 瞬时故障重试次数直方图
