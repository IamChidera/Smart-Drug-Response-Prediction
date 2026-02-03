"""
Train a logistic regression baseline on engineered features.

Features are standardized (excluding missingness flags). Case-level splits are
frozen to disk. Outputs include metrics JSON, optional isotonic-calibrated
metrics, a coefficient table for interpretability, and serialized model/scaler.
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


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")


def load_feature_files() -> pd.DataFrame:
    """Load all per-case feature parquet files and concatenate them."""
    files = sorted(FEATURES_DIR.glob("case*__features.parquet"))
    dfs = []
    for p in files:
        try:
            df = pd.read_parquet(p)
            if {"caseid", "label"}.issubset(df.columns) and len(df) > 0:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        raise RuntimeError("No feature files found in data/features")
    return pd.concat(dfs, ignore_index=True)


def split_by_case(df: pd.DataFrame, seed: int, train_frac=0.7, val_frac=0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a case-level split into train/val/test sets."""
    caseids = df["caseid"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(caseids)
    n = len(caseids)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_cases = set(caseids[:n_train])
    val_cases = set(caseids[n_train:n_train + n_val])
    test_cases = set(caseids[n_train + n_val:])
    return train_cases, val_cases, test_cases


def compute_scaler(X: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Compute mean/std for continuous features, skipping missing flags."""
    stats: Dict[str, Tuple[float, float]] = {}
    for c in X.columns:
        if c.endswith("_missing"):
            continue
        mu = float(X[c].mean())
        sd = float(X[c].std(ddof=0) or 1.0)
        if sd == 0:
            sd = 1.0
        stats[c] = (mu, sd)
    return stats


def apply_scaler(X: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Standardize features using provided mean/std stats."""
    Xs = X.copy()
    for c, (mu, sd) in stats.items():
        if c in Xs.columns:
            Xs[c] = (Xs[c] - mu) / sd
    return Xs


def train_and_eval(seed: int, C: float = 1.0, penalty: str = "l2") -> None:
    """Train logistic regression, evaluate, calibrate, and persist artifacts."""
    set_seed(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_files()
    drop_cols = {"caseid", "label"}
    with_time = os.environ.get("LR_WITH_TIME", "0") == "1"
    if not with_time:
        drop_cols.add("Time")
    drop_cols.update({
        "Orchestra_RFTN20_RATE__curr",
        "Orchestra_RFTN50_RATE__curr",
    })
    no_map = os.environ.get("LR_NO_MAP", "0") == "1"
    base_feats = [c for c in df.columns if c not in drop_cols and not c.endswith("_missing")]
    if no_map:
        base_feats = [c for c in base_feats if not c.startswith("MAP__")]
    feat_cols = base_feats + [
        c for c in df.columns if c.endswith("_missing")
    ]

    SPLIT_DIR = Path("data/splits"); SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_path = SPLIT_DIR / f"case_splits_seed{seed}.json"
    if split_path.exists():
        with open(split_path) as f:
            saved = json.load(f)
        train_cases = set(saved["train"])  # type: ignore
        val_cases = set(saved["val"])      # type: ignore
        test_cases = set(saved["test"])    # type: ignore
    else:
        train_cases, val_cases, test_cases = split_by_case(df, seed)
        with open(split_path, "w") as f:
            json.dump({
                "seed": seed,
                "train": sorted(map(int, train_cases)),
                "val": sorted(map(int, val_cases)),
                "test": sorted(map(int, test_cases)),
            }, f, indent=2)
    X_train = df[df["caseid"].isin(train_cases)][feat_cols]
    y_train = df[df["caseid"].isin(train_cases)]["label"].astype(int)
    X_val = df[df["caseid"].isin(val_cases)][feat_cols]
    y_val = df[df["caseid"].isin(val_cases)]["label"].astype(int)
    X_test = df[df["caseid"].isin(test_cases)][feat_cols]
    y_test = df[df["caseid"].isin(test_cases)]["label"].astype(int)

    scaler = compute_scaler(X_train)
    X_train_s = apply_scaler(X_train, scaler)
    X_val_s = apply_scaler(X_val, scaler)
    X_test_s = apply_scaler(X_test, scaler)
    X_train_s = X_train_s.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val_s = X_val_s.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_s = X_test_s.replace([np.inf, -np.inf], np.nan).fillna(0)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    except Exception as e:
        raise SystemExit("scikit-learn is required. Please install scikit-learn.")

    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        solver="saga",
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)
    try:
        from sklearn.calibration import CalibratedClassifierCV
        try:
            cal_clf = CalibratedClassifierCV(estimator=clf, method="isotonic", cv="prefit")
        except TypeError:
            cal_clf = CalibratedClassifierCV(base_estimator=clf, method="isotonic", cv="prefit")
        cal_clf.fit(X_val_s, y_val)
    except Exception:
        cal_clf = None
    def eval_split(Xs, y, use_calibrated=False):
        if use_calibrated and cal_clf is not None:
            p = cal_clf.predict_proba(Xs)[:, 1]
        else:
            p = clf.predict_proba(Xs)[:, 1]
        return {
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
            "brier": float(brier_score_loss(y, p)),
            "n": int(len(y)),
            "threshold@0.5": threshold_metrics(y, p, 0.5),
        }

    def threshold_metrics(y_true, y_prob, thr: float):
        y_pred = (y_prob >= thr).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else None
        spec = tn / (tn + fp) if (tn + fp) else None
        ppv = tp / (tp + fp) if (tp + fp) else None
        npv = tn / (tn + fn) if (tn + fn) else None
        return {
            "thr": thr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
        }

    metrics = {
        "seed": seed,
        "n_features": len(feat_cols),
        "train": eval_split(X_train_s, y_train),
        "val": eval_split(X_val_s, y_val),
        "test": eval_split(X_test_s, y_test),
        "val_calibrated": eval_split(X_val_s, y_val, use_calibrated=True) if cal_clf is not None else None,
        "test_calibrated": eval_split(X_test_s, y_test, use_calibrated=True) if cal_clf is not None else None,
        "feat_cols": feat_cols,
    }

    try:
        import joblib
        joblib.dump(clf, MODELS_DIR / f"logreg_seed{seed}.joblib")
    except Exception:
        pass
    with open(MODELS_DIR / f"logreg_scaler_seed{seed}.json", "w") as f:
        json.dump({k: [mu, sd] for k, (mu, sd) in scaler.items()}, f)
    with open(METRICS_DIR / f"logreg_metrics_seed{seed}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        coefs = clf.coef_.ravel()  # binary clf
        coef_df = pd.DataFrame({"feature": feat_cols, "coef": coefs})
        coef_df = coef_df.sort_values(by="coef", key=lambda s: s.abs(), ascending=False)
        coef_df.to_csv(METRICS_DIR / f"logreg_coeffs_seed{seed}.csv", index=False)
    except Exception:
        pass

    print(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--with_time", action="store_true", help="Include Time as a feature (off by default)")
    ap.add_argument("--no_map", action="store_true", help="Ablation: drop all MAP-derived features")
    args = ap.parse_args()
    os.environ["LR_WITH_TIME"] = "1" if args.with_time else "0"
    os.environ["LR_NO_MAP"] = "1" if args.no_map else "0"
    train_and_eval(seed=args.seed, C=args.C)


if __name__ == "__main__":
    main()
