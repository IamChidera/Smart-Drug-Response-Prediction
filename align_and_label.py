"""
Align raw VitalDB track CSVs to a fixed time grid and build labels.

Outputs per-case 10-second aligned tables with:
- Numeric signals forward-filled up to a 60s limit
- Missingness indicators per signal
- MAP constructed from invasive then cuff MAP
- Binary label: any MAP < 65 within the next 5 minutes
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

CASELIST = Path("data/raw/cohort_caseids.txt")
CASES = Path("data/raw/cases.csv.gz")
TRACK_DIR = Path("data/raw/tracks")
OUT_DIR = Path("data/interim")

GRID_SEC = 10
FFILL_LIMIT_SEC = 60

TNAMES = [
    "Solar8000/ART_MBP", "Solar8000/NIBP_MBP",
    "Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/ETCO2", "Solar8000/RR_CO2", "Solar8000/BT",
    "Orchestra/PPF20_RATE", "Orchestra/PPF20_CE", "Orchestra/PPF20_CP",
    "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP",
    "Orchestra/RFTN50_RATE", "Orchestra/RFTN50_CE", "Orchestra/RFTN50_CP",
    "BIS/BIS",
]


def tname_to_col(tname: str) -> str:
    """Return a safe column name by replacing '/' with '_' in tname."""
    return tname.replace("/", "_")


def read_case_duration(caseid: int) -> float | None:
    """Return case duration in seconds if available in cases.csv.gz."""
    try:
        cases = pd.read_csv(CASES)
        if {"caseid", "casestart", "caseend"}.issubset(cases.columns):
            row = cases.loc[cases["caseid"] == caseid]
            if not row.empty:
                return float(row.iloc[0]["caseend"] - row.iloc[0]["casestart"])
    except Exception:
        pass
    return None


def build_grid(max_time: float) -> pd.DataFrame:
    """Build a float64 time grid [0, GRID_SEC, ..., <= max_time]."""
    n = int(math.floor(max_time / GRID_SEC)) + 1
    t = list(range(0, n * GRID_SEC, GRID_SEC))
    return pd.DataFrame({"Time": pd.Series(t, dtype="float64")})


def align_signal_to_grid(df_sig: pd.DataFrame, grid: pd.DataFrame, value_col: str) -> pd.Series:
    """Align a raw signal to the grid using backward-asof and 60s tolerance.

    Returns a Series indexed like the grid with NaN when the last observation
    is older than FFILL_LIMIT_SEC.
    """
    df = df_sig.rename(columns={value_col: "val"}).copy()
    df = df.dropna(subset=["Time"]).sort_values("Time")
    df["Time"] = df["Time"].astype("float64")
    g = grid.copy()
    g["Time"] = g["Time"].astype("float64")
    merged = pd.merge_asof(g, df[["Time", "val"]], on="Time", direction="backward")
    df_time = df[["Time"]].copy()
    df_time["obs_time"] = df_time["Time"]
    last_time = pd.merge_asof(g, df_time[["Time", "obs_time"]], on="Time", direction="backward")["obs_time"]
    age = g["Time"] - last_time
    series = merged["val"]
    series[(age.isna()) | (age > FFILL_LIMIT_SEC)] = pd.NA
    return series


def process_case(caseid: int) -> Path | None:
    """Create aligned 10s table with labels for a single case.

    Returns the path to the saved file, or None if the case cannot be
    processed (e.g., zero duration).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"case{caseid}__aligned_10s.parquet"
    if out.exists():
        try:
            df_check = pd.read_parquet(out)
            if len(df_check) > 0 and "label_next5m_hypo" in df_check.columns:
                return out
        except Exception:
            pass

    duration = read_case_duration(caseid)
    if duration is None:
        max_t = 0.0
        for tname in TNAMES:
            p = TRACK_DIR / f"case{caseid}__{tname_to_col(tname)}.csv.gz"
            if p.exists():
                try:
                    df_tmp = pd.read_csv(p, usecols=["Time"])
                    if not df_tmp.empty:
                        max_t = max(max_t, float(df_tmp["Time"].max()))
                except Exception:
                    pass
        duration = max_t
    if duration is None or duration <= 0:
        return None

    grid = build_grid(duration)
    out_df = pd.DataFrame({"Time": grid["Time"]})

    available_cols: List[str] = []
    for tname in TNAMES:
        path = TRACK_DIR / f"case{caseid}__{tname_to_col(tname)}.csv.gz"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            val_col = [c for c in df.columns if c != "Time"]
            if not val_col:
                continue
            val_col = val_col[0]
            sig = align_signal_to_grid(df[["Time", val_col]], grid, val_col)
            colname = tname_to_col(tname)
            out_df[colname] = sig
            out_df[colname + "_missing"] = out_df[colname].isna().astype("int8")
            available_cols.append(colname)
        except Exception:
            continue

    art = "Solar8000_ART_MBP"
    nibp = "Solar8000_NIBP_MBP"
    if art in out_df.columns or nibp in out_df.columns:
        map_series = out_df.get(art)
        if map_series is None:
            map_series = out_df.get(nibp)
        else:
            if nibp in out_df.columns:
                map_series = map_series.fillna(out_df[nibp])
        out_df["MAP"] = map_series

        steps = int(300 / GRID_SEC)
        future_min = out_df["MAP"].shift(-1).rolling(window=steps, min_periods=steps).min()
        label = (future_min < 65).astype("Int8")
        out_df["label_next5m_hypo"] = label
    else:
        out_df["label_next5m_hypo"] = pd.Series([pd.NA] * len(out_df), dtype="Int8")

    out_df = out_df.dropna(subset=["label_next5m_hypo"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_df.to_parquet(out, index=False)
        return out
    except Exception:
        out_csv = OUT_DIR / f"case{caseid}__aligned_10s.csv.gz"
        out_df.to_csv(out_csv, index=False)
        return out_csv


def main():
    """Process all cohort cases and save aligned outputs."""
    caseids = [int(x.strip()) for x in CASELIST.read_text().splitlines() if x.strip()]
    done = 0
    for i, cid in enumerate(caseids, 1):
        res = process_case(cid)
        done += 1 if res else 0
        if i % 25 == 0:
            print(f"Processed {i}/{len(caseids)} cases … written {done}")
    print(f"Completed. Cases processed: {done} / {len(caseids)}")


if __name__ == "__main__":
    main()
