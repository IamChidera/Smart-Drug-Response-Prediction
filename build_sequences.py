"""
Build sequence arrays (train/val/test) from aligned per-case parquet files
and cache them as compressed NPZ files for training.

This script encapsulates all sequence-building logic so that train_tcn.py
can remain focused on model training only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import hashlib

INTERIM_DIR = Path("data/interim")

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


def list_cases() -> List[int]:
    """Return sorted case IDs with aligned parquet files present."""
    return sorted([int(p.name.split("__")[0][4:]) for p in INTERIM_DIR.glob("case*__aligned_10s.parquet")])


def load_case_df(caseid: int) -> pd.DataFrame:
    """Load a case DataFrame and add coalesced Remi rate and flags."""
    df = pd.read_parquet(INTERIM_DIR / f"case{caseid}__aligned_10s.parquet")
    r20 = df["Orchestra_RFTN20_RATE"] if "Orchestra_RFTN20_RATE" in df.columns else None
    r50 = df["Orchestra_RFTN50_RATE"] if "Orchestra_RFTN50_RATE" in df.columns else None
    if r20 is not None or r50 is not None:
        base20 = r20 if r20 is not None else pd.Series([np.nan] * len(df), index=df.index)
        base50 = r50 if r50 is not None else pd.Series([np.nan] * len(df), index=df.index)
        df["Remi_RATE_any"] = base20.combine_first(base50).fillna(0)
        df["Remi_missing_any"] = ((base20.isna()) & (base50.isna())).astype("int8")
    else:
        df["Remi_RATE_any"] = 0.0
        df["Remi_missing_any"] = 1
    if "Orchestra_PPF20_RATE_missing" not in df.columns:
        if "Orchestra_PPF20_RATE" in df.columns:
            df["Orchestra_PPF20_RATE_missing"] = df["Orchestra_PPF20_RATE"].isna().astype("int8")
        else:
            df["Orchestra_PPF20_RATE_missing"] = 1
    for c in CHANNELS:
        if c not in df.columns:
            if c.endswith("_missing") or c == "Remi_missing_any":
                df[c] = 1
            else:
                df[c] = 0.0
    return df


def build_sequences_for_case(df: pd.DataFrame, channels: List[str], stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Construct [N, T, C] sequences and labels from a single case DataFrame."""
    avail = channels[:]  # ensure fixed order and length
    if len(avail) == 0:
        return np.empty((0, WIN_STEPS, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    valid_idx = np.arange(WIN_STEPS - 1, len(df), stride)
    labels = df.loc[valid_idx, "label_next5m_hypo"].astype("float32")
    mask_label = labels.notna().values
    valid_idx = valid_idx[mask_label]
    if len(valid_idx) == 0:
        return np.empty((0, WIN_STEPS, len(avail)), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_list = []
    y_list = []
    val_cols = [c for c in avail if not (c.endswith("_missing") or c == "Remi_missing_any")]
    flag_cols = [c for c in avail if (c.endswith("_missing") or c == "Remi_missing_any")]
    for i in valid_idx:
        start = i - (WIN_STEPS - 1)
        window = df.iloc[start:i + 1]
        w = window.reindex(columns=avail).copy()
        if val_cols:
            w[val_cols] = w[val_cols].ffill().bfill().fillna(0.0)
        if flag_cols:
            w[flag_cols] = w[flag_cols].ffill().bfill().fillna(1).astype("int8")
        X = w.values.astype(np.float32)  # [steps, channels]
        if X.shape[0] != WIN_STEPS:
            continue
        y = int(df.iloc[i]["label_next5m_hypo"])  # 0 or 1
        X_list.append(X)
        y_list.append(y)
    if not X_list:
        return np.empty((0, WIN_STEPS, len(avail)), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X_arr = np.stack(X_list, axis=0)
    y_arr = np.array(y_list, dtype=np.int64)
    return X_arr, y_arr


def load_splits_from_json(seed: int) -> Tuple[set[int], set[int], set[int]]:
    """Load exact train/val/test case IDs from LR split file.

    Requires data/splits/case_splits_seed{seed}.json with keys train/val/test
    and disjoint sets. Fails hard if the file is missing or malformed.
    """
    import json
    split_path = Path(f"data/splits/case_splits_seed{seed}.json")
    if not split_path.exists():
        raise SystemExit(f"Split file not found: {split_path}. This builder requires a predefined split.")
    try:
        d = json.loads(split_path.read_text())
        train = set(int(x) for x in d.get("train", []))
        val = set(int(x) for x in d.get("val", []))
        test = set(int(x) for x in d.get("test", []))
    except Exception as e:
        raise SystemExit(f"Failed to parse split file {split_path}: {e}")
    if not train or not val or not test:
        raise SystemExit(f"Split file {split_path} missing one of train/val/test lists or they are empty.")
    if train & val or train & test or val & test:
        raise SystemExit(f"Split file {split_path} has overlapping case IDs across splits.")
    return train, val, test


def cache_signature() -> str:
    """Return a short signature of the channel schema and window config."""
    h = hashlib.sha1()
    h.update("|".join(CHANNELS).encode("utf-8"))
    h.update(str(WIN_STEPS).encode("utf-8"))
    return h.hexdigest()[:12]


def cache_paths(base: Path, seed: int, stride: int) -> Dict[str, Path]:
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


def save_npz(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride", type=int, default=1, help="Window stride in timesteps (1=every 10s)")
    ap.add_argument("--cache_dir", type=str, default="data/seq_cache", help="Sequence cache base directory")
    args = ap.parse_args()

    all_cases = set(list_cases())
    train_cases, val_cases, test_cases = load_splits_from_json(args.seed)
    missing = (train_cases | val_cases | test_cases) - all_cases
    if missing:
        raise SystemExit(
            f"Split references {len(missing)} case(s) without aligned parquet files. "
            f"Examples: {sorted(list(missing))[:10]}"
        )

    def build_for_caseids(case_set):
        Xs = []
        ys = []
        for cid in sorted(case_set):
            df = load_case_df(cid)
            Xc, yc = build_sequences_for_case(df, CHANNELS, stride=args.stride)
            if Xc.size == 0:
                continue
            Xs.append(Xc)
            ys.append(yc)
        if not Xs:
            return np.empty((0, WIN_STEPS, len(CHANNELS)), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    paths = cache_paths(Path(args.cache_dir), seed=args.seed, stride=args.stride)

    X_train, y_train = build_for_caseids(train_cases)
    save_npz(paths["train"], X_train, y_train)

    X_val, y_val = build_for_caseids(val_cases)
    save_npz(paths["val"], X_val, y_val)

    X_test, y_test = build_for_caseids(test_cases)
    save_npz(paths["test"], X_test, y_test)

    meta = {"seed": args.seed, "stride": args.stride, "channels": CHANNELS, "win_steps": WIN_STEPS}
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote train/val/test NPZ to {paths['dir']}")


if __name__ == "__main__":
    main()
