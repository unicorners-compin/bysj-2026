# 02 自描述故障恢复

故障携带结构化自描述信息，恢复器按描述选择策略（重试 / 降级 / 隔离）。

```bash
python -m src.simulate
python -m pytest -q
```
