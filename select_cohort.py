"""
Select the study cohort from VitalDB track index files.

Inclusion criteria:
- Must have Solar8000/ART_MBP
- Must have Orchestra/PPF20_RATE
- Must have Orchestra/RFTN20_RATE or Orchestra/RFTN50_RATE
- Optional: duration ≥ 3600s when case timestamps are present
"""

from pathlib import Path
import pandas as pd

TRKS = Path("data/raw/trks.csv.gz")
CASES = Path("data/raw/cases.csv.gz")
OUT = Path("data/raw/cohort_caseids.txt")

REQUIRED = {
    "Solar8000/ART_MBP",
    "Orchestra/PPF20_RATE",
}
ANY_OF = {"Orchestra/RFTN20_RATE", "Orchestra/RFTN50_RATE"}


def main():
    """Emit a list of eligible case IDs based on inclusion rules."""
    print("Loading trks index …")
    trks = pd.read_csv(TRKS)
    if not {"caseid", "tname"}.issubset(trks.columns):
        raise RuntimeError("/trks CSV missing required columns: caseid, tname")

    case_has = trks.groupby("caseid")["tname"].agg(set)

    eligible = []
    for cid, names in case_has.items():
        if REQUIRED.issubset(names) and (ANY_OF & names):
            eligible.append(int(cid))

    try:
        cases = pd.read_csv(CASES)
        if {"casestart", "caseend", "caseid"}.issubset(cases.columns):
            long_enough = cases.loc[(cases["caseend"] - cases["casestart"]) >= 3600, "caseid"]
            long_set = set(int(x) for x in long_enough.tolist())
            eligible = [cid for cid in eligible if cid in long_set]
    except Exception as e:
        print(f"Note: skipping duration filter ({e})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for cid in sorted(set(eligible)):
            f.write(f"{cid}\n")
    print(f"Selected {len(set(eligible))} cases → {OUT}")


if __name__ == "__main__":
    main()
