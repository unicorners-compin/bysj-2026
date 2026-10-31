# 01 时延感知早退（分布式推理，Python 教学原型）

学生：待定

指导教师：任智源（西安电子科技大学通信工程学院）

**真实部分**：在 MNIST 上用 PyTorch 实际训练带 1 个中间出口 + 1 个最终出口的小型 CNN（3 个种子，10 轮），
导出每个验证/测试样本两个出口的置信度与预测（`figs/exit_outputs_seed*.npz`），并实测单样本各阶段前向耗时。
**仿真部分**：2/3 个逻辑节点之间的链路（带宽 + 固定时延 + 截断正态抖动）由 `src/netsim.py` 计算，不发送真实数据包。
> 非 DPDK、非硬件实测；所有时延为「实测计算时延 × 节点速度系数 + 仿真链路时延」。

```bash
python3 -m src.mnist_data      # 下载 MNIST 到 data/（git 忽略）
python3 -m src.train           # 需要 torch；训练 3 个种子并导出出口输出（已提交结果，可跳过）
python3 -m src.experiment      # 只需 numpy/matplotlib：参数选择(验证集) + 测试集全部实验 + 出图
python3 -m pytest -q
```

| 文件 | 内容 |
|---|---|
| `train_log.csv` / `model_accuracy.csv` / `compute_timing.csv` / `model_summary.csv` / `env.txt` | 训练曲线、各出口准确率、实测阶段耗时、参数量、环境 |
| `params_selected.csv` | 验证集上为每个(种子,网络,预算)选出的 τ_fixed 与 τ_min/τ_max/γ/margin，及共用参数 |
| `main_results_per_seed.csv` / `main_results_summary.csv` | 3 网络 × 3 预算(10/30/80 ms) × 6 方案，3 种子均值±标准差 |
| `budget_sweep.csv`, `tradeoff_mid_budget.csv`, `k3_relay_results.csv`, `dynamic_network_results.csv`, `exit1_calibration.csv`, `confidence_measure_comparison.csv` | 预算扫描、权衡曲线、K=3 中继、时变网络、校准、置信度度量对比 |
| `fig1`–`fig7` PNG | 训练曲线、置信度/可靠性图、权衡曲线、超时率-预算、退出位置分布、消融、时变网络 |
