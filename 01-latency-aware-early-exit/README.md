# 01 时延感知早退

模拟多级推理/处理流水线：在时延预算内尽早返回结果。

> 数值与图均为 **Python 玩具仿真**（本科友好），**不是** DPDK / 硬件实测。

```bash
# 简短演示
python3 -m src.simulate

# 参数扫描 + 出图（CSV/PNG 写入 figs/）
python3 -m src.experiment

python3 -m pytest -q
```

`figs/` 由 `src.experiment` 重新生成：
- `budget_vs_success_stage.png` — 时延预算 vs 成功率 / 退出级数
- `confidence_vs_latency.png` — 置信度阈值 vs 时延 / 退出级数
