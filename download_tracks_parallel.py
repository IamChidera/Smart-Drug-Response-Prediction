import asyncio
import gzip
import sys
from pathlib import Path
from typing import List, Tuple

import aiohttp
import pandas as pd

BASE = "https://api.vitaldb.net"
TRKS = Path("data/raw/trks.csv.gz")
CASELIST = Path("data/raw/cohort_caseids.txt")
OUTDIR = Path("data/raw/tracks")

TNAMES = [
    "Solar8000/ART_MBP", "Solar8000/NIBP_MBP",
    "Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/ETCO2", "Solar8000/RR_CO2", "Solar8000/BT",
    "Orchestra/PPF20_RATE", "Orchestra/PPF20_CE", "Orchestra/PPF20_CP",
    "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP",
    "Orchestra/RFTN50_RATE", "Orchestra/RFTN50_CE", "Orchestra/RFTN50_CP",
    "BIS/BIS",
]


def build_targets() -> List[Tuple[int, str, str, Path]]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    caseids = [int(x.strip()) for x in CASELIST.read_text().splitlines() if x.strip()]
    df = pd.read_csv(TRKS, dtype={"tid": str})
    filtered = df[(df["caseid"].isin(caseids)) & (df["tname"].isin(TNAMES))]
    uniq = filtered.sort_values(["caseid", "tname", "tid"]).drop_duplicates(["caseid", "tname"])  # one tid per pair
    tasks = []
    for caseid, tname, tid in uniq[["caseid", "tname", "tid"]].itertuples(index=False):
        safe_tname = str(tname).replace("/", "_")
        out = OUTDIR / f"case{int(caseid)}__{safe_tname}.csv.gz"
        tasks.append((int(caseid), str(tname), str(tid), out))
    return tasks


async def fetch_one(session: aiohttp.ClientSession, sem: asyncio.Semaphore, tid: str) -> bytes:
    url = f"{BASE}/{tid}"
    async with sem:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            resp.raise_for_status()
            raw = await resp.read()
            if not (len(raw) >= 2 and raw[:2] == b"\x1f\x8b"):
                raw = gzip.compress(raw)
            return raw


async def main_async(max_workers: int = 32):
    tasks = build_targets()
    pending = [(cid, tname, tid, out) for (cid, tname, tid, out) in tasks if not out.exists()]
    print(f"Total targets: {len(tasks)} | Downloaded: {len(tasks)-len(pending)} | Missing: {len(pending)}")
    if not pending:
        return
    sem = asyncio.Semaphore(max_workers)
    connector = aiohttp.TCPConnector(limit=max_workers)
    headers = {"Accept-Encoding": "gzip"}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        i = 0
        for cid, tname, tid, out in pending:
            try:
                data = await fetch_one(session, sem, tid)
                out.write_bytes(data)
                i += 1
                if i % 100 == 0:
                    print(f"Downloaded {i}/{len(pending)} … last: case{cid} {tname}")
            except Exception as e:
                print(f"ERROR tid={tid} case={cid} tname={tname}: {e}")


def main():
    max_workers = 32
    if len(sys.argv) > 1:
        try:
            max_workers = int(sys.argv[1])
        except Exception:
            pass
    asyncio.run(main_async(max_workers))


if __name__ == "__main__":
    main()

