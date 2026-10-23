"""MNIST loading and the fixed train/val/test split used throughout the thesis.

Raw idx files are read from ``data/`` (download with ``python3 -m src.mnist_data``).
Split: the official 60k training set is split with a fixed seed (20260601) into
55k train / 5k validation; the official 10k test set is only used for final
evaluation.  Pixels are scaled to [0,1] and standardised with train mean/std.
"""
from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
SPLIT_SEED = 20260601
N_VAL = 5000


def download() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        p = DATA / name
        if not p.exists():
            print("downloading", name)
            urllib.request.urlretrieve(URL + name, p)


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_splits():
    """Return dict with x_train,y_train,x_val,y_val,x_test,y_test (float32 NCHW)."""
    download()
    xtr = _read_idx(DATA / FILES["train_x"]).astype(np.float32) / 255.0
    ytr = _read_idx(DATA / FILES["train_y"]).astype(np.int64)
    xte = _read_idx(DATA / FILES["test_x"]).astype(np.float32) / 255.0
    yte = _read_idx(DATA / FILES["test_y"]).astype(np.int64)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(xtr))
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
    x_train, y_train = xtr[tr_idx], ytr[tr_idx]
    x_val, y_val = xtr[val_idx], ytr[val_idx]
    mean, std = float(x_train.mean()), float(x_train.std())

    def norm(x):
        return ((x - mean) / std)[:, None, :, :].astype(np.float32)

    return {
        "x_train": norm(x_train), "y_train": y_train,
        "x_val": norm(x_val), "y_val": y_val,
        "x_test": norm(xte), "y_test": yte,
        "mean": mean, "std": std,
    }


if __name__ == "__main__":
    download()
    s = load_splits()
    print({k: getattr(v, "shape", v) for k, v in s.items()})
