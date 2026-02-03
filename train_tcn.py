"""
Train a Temporal Convolutional Network (TCN) on aligned VitalDB sequences.

Sequences span 30 minutes at 10-second resolution. Inputs include vitals,
drug rates, and missingness flags. Case-level splits are used and training
uses weighted BCE. Metrics are written to metrics/tcn_metrics_seed{seed}.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import hashlib
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")

GRID_SEC = 10
WIN_STEPS = 180  # 30 minutes at 10s

CHANNELS = [
    "MAP",
    "Solar8000_HR",
    "Solar8000_PLETH_SPO2",
    "Solar8000_ETCO2",
    "Solar8000_RR_CO2",
    "Solar8000_BT",
    "Orchestra_PPF20_RATE",
    "Remi_RATE_any",
    "Orchestra_PPF20_RATE_missing",
    "Remi_missing_any",
]


def set_seed(seed: int) -> None:
    """Set RNG seeds for Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def compute_norm_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std over all timepoints."""
    mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X.reshape(-1, X.shape[-1]).std(axis=0)
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def cache_signature() -> str:
    """Return a short signature of the channel schema and window config."""
    h = hashlib.sha1()
    h.update("|".join(CHANNELS).encode("utf-8"))
    h.update(str(WIN_STEPS).encode("utf-8"))
    return h.hexdigest()[:12]


def cache_paths(base: Path, seed: int, stride: int) -> Dict[str, Path]:
    """Return paths for cached arrays keyed by split name."""
    sig = cache_signature()
    d = base / f"seed{seed}_stride{stride}_sig{sig}"
    d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "train": d / "train.npz",
        "val": d / "val.npz",
        "test": d / "test.npz",
        "meta": d / "meta.json",
    }


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["X"], d["y"]


def train_tcn(
    seed: int,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    dilations: List[int] = [1, 2, 4, 8],
    stride: int = 1,
    cache_dir: str = "data/seq_cache",
):
    set_seed(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    except Exception as e:
        raise SystemExit("PyTorch and scikit-learn are required to train the TCN.")

    cache_base = Path(cache_dir)
    paths = cache_paths(cache_base, seed=seed, stride=stride)
    if not (paths["train"].exists() and paths["val"].exists() and paths["test"].exists()):
        raise SystemExit(f"Missing cached sequences under {paths['dir']}. Run build_sequences.py first.")
    X_train, y_train = load_npz(paths["train"]) 
    X_val, y_val = load_npz(paths["val"]) 
    if X_train.size == 0:
        raise SystemExit("No training sequences built. Ensure aligned files exist and CHANNELS are present.")

    mu, sd = compute_norm_stats(X_train)
    X_train -= mu
    X_train /= sd
    if X_val.size:
        X_val -= mu
        X_val /= sd

    train_ds = TensorDataset(torch.from_numpy(X_train).transpose(1, 2), torch.from_numpy(y_train))  # [N,C,T]
    val_ds = TensorDataset(torch.from_numpy(X_val).transpose(1, 2), torch.from_numpy(y_val)) if X_val.size else None
    test_ds = None
    test_loader = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False) if val_ds else None

    class TemporalBlock(nn.Module):
        """A causal residual block with dilated 1D convolutions."""

        def __init__(self, in_ch, out_ch, dilation, kernel_size=3, dropout=0.2):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):
            out = self.net(x)
            crop = out.size(-1) - x.size(-1)
            if crop > 0:
                out = out[..., crop:]
            return out + self.downsample(x)

    class TCN(nn.Module):
        """Stack of TemporalBlock layers with global pooling and linear head."""

        def __init__(self, in_ch, channels=[64, 64, 64, 64], dilations=[1, 2, 4, 8], dropout=0.2):
            super().__init__()
            layers = []
            ch_prev = in_ch
            for ch, d in zip(channels, dilations):
                layers.append(TemporalBlock(ch_prev, ch, dilation=d, kernel_size=3, dropout=dropout))
                ch_prev = ch
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(ch_prev, 1),
            )

        def forward(self, x):
            h = self.tcn(x)
            logits = self.head(h)
            return logits.squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCN(in_ch=len(CHANNELS), dilations=dilations).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def evaluate(loader):
        model.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()
                logits = model(xb)
                prob = torch.sigmoid(logits)
                ys.append(yb.cpu().numpy())
                ps.append(prob.cpu().numpy())
        if not ys:
            return {"auroc": None, "auprc": None, "brier": None, "n": 0}
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        out = {
            "n": int(len(y)),
            "brier": float(brier_score_loss(y, p)),
        }
        if len(np.unique(y)) > 1:
            out["auroc"] = float(roc_auc_score(y, p))
            out["auprc"] = float(average_precision_score(y, p))
        else:
            out["auroc"] = None
            out["auprc"] = None
        return out

    best_val = -np.inf
    best_path = MODELS_DIR / f"tcn_seed{seed}.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
        train_loss = total / max(1, len(train_ds))
        val_metrics = evaluate(val_loader) if val_loader else {"auprc": None}
        print(f"epoch {epoch} train_loss {train_loss:.4f} val_auprc {val_metrics['auprc']}")
        score = val_metrics["auprc"] if val_metrics["auprc"] is not None else -train_loss
        if score is not None and score > best_val:
            best_val = score
            torch.save({
                "model_state": model.state_dict(),
                "mu": mu,
                "sd": sd,
                "channels": CHANNELS,
                "seed": seed,
            }, best_path)

    if best_path.exists():
        try:
            state = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])
    train_metrics = evaluate(train_loader)
    val_metrics = evaluate(val_loader) if val_loader else {"n": 0}
    X_test, y_test = load_npz(paths["test"])
    if X_test.size:
        X_test -= mu
        X_test /= sd
        test_ds = TensorDataset(torch.from_numpy(X_test).transpose(1, 2), torch.from_numpy(y_test))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        test_metrics = evaluate(test_loader)
    else:
        test_metrics = {"n": 0}
    metrics = {
        "seed": seed,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "channels": CHANNELS,
        "win_steps": WIN_STEPS,
    }
    with open(METRICS_DIR / f"tcn_metrics_seed{seed}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=1, help="Window stride in timesteps (1=every 10s)")
    ap.add_argument("--cache_dir", type=str, default="data/seq_cache", help="Sequence cache directory")
    args = ap.parse_args()
    train_tcn(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stride=args.stride,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
