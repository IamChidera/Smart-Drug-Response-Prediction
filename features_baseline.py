"""
Build baseline tabular features from aligned per-case data.

Features include for each signal: last value, 30-minute rolling mean/std,
and slope per minute. Infusion rates include current PPF20 and a coalesced
Remi rate, with corresponding missingness flags. Rows require a full 30-minute
history for MAP.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import List

import numpy as np
import pandas as pd

INTERIM_DIR = Path("data/interim")
CASELIST = Path("data/raw/cohort_caseids.txt")
OUT_DIR = Path("data/features")

GRID_SEC = 10
WIN_SEC = 30 * 60
WIN_STEPS = int(WIN_SEC / GRID_SEC)  # 180

SUMMARY_SIGNALS = [
    "MAP",
    "Solar8000_HR",
    "Solar8000_PLETH_SPO2",
    "Solar8000_ETCO2",
    "Solar8000_RR_CO2",
    "Solar8000_BT",
]

RATE_SIGNALS = [
    "Orchestra_PPF20_RATE",
    "Orchestra_RFTN20_RATE",
    "Orchestra_RFTN50_RATE",
]


def available_cols(df: pd.DataFrame, wanted: List[str]) -> List[str]:
    """Return the subset of requested columns present in the DataFrame."""
    return [c for c in wanted if c in df.columns]


def build_features_for_case(caseid: int) -> Path | None:
    """Create feature table for a single case and save it as parquet.

    Returns the output path, or None if the case cannot produce any valid
    rows after filtering.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"case{caseid}__features.parquet"
    if out.exists() and os.environ.get("REGEN_ALL", "0") != "1":
        try:
            df_existing = pd.read_parquet(out)
            if {"Remi_RATE_any", "Remi_missing_any"}.issubset(df_existing.columns):
                return out
        except Exception:
            pass

    src = INTERIM_DIR / f"case{caseid}__aligned_10s.parquet"
    if not src.exists():
        return None

    df = pd.read_parquet(src)
    if df is None or len(df) == 0:
        return None

    df = df.reset_index(drop=True)

    feats = pd.DataFrame({"caseid": caseid, "Time": df["Time"]})
    present_summary = available_cols(df, SUMMARY_SIGNALS)
    for col in present_summary:
        s = df[col]
        feats[f"{col}__last"] = s
        if s.notna().sum() >= WIN_STEPS:
            roll = s.rolling(window=WIN_STEPS, min_periods=WIN_STEPS)
            feats[f"{col}__mean"] = roll.mean()
            feats[f"{col}__std"] = roll.std()
            first = s.shift(WIN_STEPS - 1)
            feats[f"{col}__slope_per_min"] = (s - first) / 30.0

    if "Orchestra_PPF20_RATE" in df.columns:
        feats["Orchestra_PPF20_RATE__curr"] = df["Orchestra_PPF20_RATE"]
    r20 = df["Orchestra_RFTN20_RATE"] if "Orchestra_RFTN20_RATE" in df.columns else pd.Series(pd.NA, index=df.index)
    r50 = df["Orchestra_RFTN50_RATE"] if "Orchestra_RFTN50_RATE" in df.columns else pd.Series(pd.NA, index=df.index)
    remi_any = r20.combine_first(r50)
    feats["Remi_RATE_any"] = remi_any.fillna(0)
    feats["Remi_missing_any"] = ((r20.isna()) & (r50.isna())).astype("int8")

    miss_cols = [c for c in df.columns if c.endswith("_missing")]
    for mc in miss_cols:
        feats[mc] = df[mc]

    if "label_next5m_hypo" not in df.columns:
        return None
    feats["label"] = df["label_next5m_hypo"].astype("Int8")

    valid = feats["label"].notna()
    if "MAP__mean" in feats.columns:
        valid &= feats["MAP__mean"].notna()
    feats = feats[valid].reset_index(drop=True)

    if len(feats) == 0:
        return None

    feats["label"] = feats["label"].astype("int8")
    for c in feats.columns:
        if c.endswith("_missing") or c == "Remi_missing_any":
            feats[c] = feats[c].astype("int8")
        elif c not in ("caseid", "Time", "label"):
            feats[c] = feats[c].astype("float32")

    feats.to_parquet(out, index=False)
    return out


def main():
    """Build features for every case listed in the cohort file."""
    caseids = [int(x.strip()) for x in CASELIST.read_text().splitlines() if x.strip()]
    written = 0
    for i, cid in enumerate(caseids, 1):
        p = build_features_for_case(cid)
        if p:
            written += 1
        if i % 25 == 0:
            print(f"Processed {i}/{len(caseids)} cases … features written: {written}")
    print(f"Done. Feature files: {written} / {len(caseids)} → {OUT_DIR}")


if __name__ == "__main__":
    main()
