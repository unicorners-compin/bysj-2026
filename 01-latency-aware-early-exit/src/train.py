"""Train the 2-exit CNN on MNIST for several seeds and dump per-sample exit outputs.

Run (needs PyTorch, CPU is enough):
  python3 -m src.train            # seeds 0 1 2, 10 epochs each

Outputs (committed so that src.experiment runs with numpy only):
  figs/exit_outputs_seed{s}.npz   per-sample conf/pred/label of both exits (val+test)
  figs/train_log.csv              per-epoch loss / val accuracy of both exits
  figs/compute_timing.csv         measured single-sample forward time of each stage
  figs/model_summary.csv          parameter counts and feature size
Model checkpoints go to models/ (git-ignored).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .mnist_data import load_splits
from .model import EarlyExitCNN, count_params

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
MODELS = ROOT / "models"
W1, W2 = 0.5, 1.0
LR, BATCH, EPOCHS = 1e-3, 128, 10


def evaluate(model, x, y, bs=1000):
    model.eval()
    c1s, p1s, c2s, p2s, e1s = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i:i + bs])
            l1, l2, _ = model(xb)
            pr1, pr2 = F.softmax(l1, 1), F.softmax(l2, 1)
            c1, p1 = pr1.max(1)
            c2, p2 = pr2.max(1)
            ent1 = -(pr1 * torch.log(pr1.clamp_min(1e-12))).sum(1)
            top2 = pr1.topk(2, 1).values
            c1s.append(c1.numpy()); p1s.append(p1.numpy())
            c2s.append(c2.numpy()); p2s.append(p2.numpy())
            e1s.append(np.stack([ent1.numpy(), (top2[:, 0] - top2[:, 1]).numpy()], 1))
    e = np.concatenate(e1s)
    return {
        "conf1": np.concatenate(c1s), "pred1": np.concatenate(p1s),
        "conf2": np.concatenate(c2s), "pred2": np.concatenate(p2s),
        "entropy1": e[:, 0], "margin1": e[:, 1], "label": y,
    }


def train_seed(seed: int, d: dict, log_rows: list):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EarlyExitCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xtr, ytr = d["x_train"], d["y_train"]
    best, best_state = -1.0, None
    rng = np.random.default_rng(seed)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = rng.permutation(len(xtr))
        tot, n = 0.0, 0
        t0 = time.perf_counter()
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = torch.from_numpy(xtr[idx]), torch.from_numpy(ytr[idx])
            l1, l2, _ = model(xb)
            loss = W1 * F.cross_entropy(l1, yb) + W2 * F.cross_entropy(l2, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); n += len(idx)
        ev = evaluate(model, d["x_val"], d["y_val"])
        a1 = float((ev["pred1"] == ev["label"]).mean())
        a2 = float((ev["pred2"] == ev["label"]).mean())
        log_rows.append({"seed": seed, "epoch": ep, "train_loss": round(tot / n, 5),
                         "val_acc_exit1": round(a1, 5), "val_acc_exit2": round(a2, 5),
                         "epoch_wall_s": round(time.perf_counter() - t0, 2)})
        print(log_rows[-1], flush=True)
        if a2 > best:
            best = a2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    MODELS.mkdir(exist_ok=True)
    torch.save(best_state, MODELS / f"ee_cnn_seed{seed}.pt")
    return model


def measure_timing(model, x, n=500, warmup=50):
    """Single-sample forward timing of each stage (median of n runs)."""
    model.eval()
    torch.set_num_threads(1)
    t1s, t2s, te = [], [], []
    with torch.no_grad():
        for i in range(n + warmup):
            xb = torch.from_numpy(x[i:i + 1])
            a = time.perf_counter(); h = model.stage1(xb); b = time.perf_counter()
            model.exit1(h); c = time.perf_counter()
            model.stage2(h); e = time.perf_counter()
            if i >= warmup:
                t1s.append((b - a) * 1e3); te.append((c - b) * 1e3); t2s.append((e - c) * 1e3)
    torch.set_num_threads(8)
    return np.array(t1s), np.array(te), np.array(t2s)


def main(seeds=(0, 1, 2)):
    FIGS.mkdir(exist_ok=True)
    d = load_splits()
    log_rows, timing_rows = [], []
    for s in seeds:
        model = train_seed(s, d, log_rows)
        val = evaluate(model, d["x_val"], d["y_val"])
        test = evaluate(model, d["x_test"], d["y_test"])
        np.savez_compressed(
            FIGS / f"exit_outputs_seed{s}.npz",
            **{f"val_{k}": v.astype(np.float32 if v.dtype.kind == 'f' else np.int16) for k, v in val.items()},
            **{f"test_{k}": v.astype(np.float32 if v.dtype.kind == 'f' else np.int16) for k, v in test.items()},
        )
        t1, te, t2 = measure_timing(model, d["x_test"])
        for name, arr in (("stage1", t1), ("exit1_head", te), ("stage2_with_final_exit", t2)):
            timing_rows.append({"seed": s, "part": name, "median_ms": round(float(np.median(arr)), 4),
                                "p10_ms": round(float(np.percentile(arr, 10)), 4),
                                "p90_ms": round(float(np.percentile(arr, 90)), 4),
                                "mean_ms": round(float(arr.mean()), 4), "n": len(arr)})
    with open(FIGS / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys())); w.writeheader(); w.writerows(log_rows)
    with open(FIGS / "compute_timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys())); w.writeheader(); w.writerows(timing_rows)
    m = EarlyExitCNN()
    with open(FIGS / "model_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "params"])
        w.writerow(["stage1", count_params(m.stage1)])
        w.writerow(["exit1", count_params(m.exit1)])
        w.writerow(["stage2+final_exit", count_params(m.stage2)])
        w.writerow(["feature_bytes_float32", 32 * 7 * 7 * 4])
    import platform
    with open(FIGS / "env.txt", "w") as f:
        f.write(f"python {platform.python_version()}\ntorch {torch.__version__}\nnumpy {np.__version__}\n"
                f"platform {platform.platform()}\nprocessor {platform.processor() or 'n/a'}\n"
                f"threads_train {torch.get_num_threads()}\n")


if __name__ == "__main__":
    seeds = tuple(int(s) for s in sys.argv[1:]) or (0, 1, 2)
    main(seeds)
