"""
Utilities to download VitalDB index endpoints and cache them as gzip files.

Endpoints:
- /cases → data/raw/cases.csv.gz
- /trks  → data/raw/trks.csv.gz
"""

import gzip
from pathlib import Path
import requests

BASE = "https://api.vitaldb.net"
OUT_CASES = Path("data/raw/cases.csv.gz")
OUT_TRKS = Path("data/raw/trks.csv.gz")


def fetch_to_gz(endpoint: str, out_path: Path):
    """Fetch an endpoint and persist the response as gzip.

    If the server returns a gzipped body, it is saved as-is; otherwise the
    response content is gzipped before writing.
    """
    url = f"{BASE}/{endpoint}"
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    raw = r.content
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        data = raw
    else:
        data = gzip.compress(raw)
    with open(out_path, "wb") as f:
        f.write(data)


def main():
    """Download /cases and /trks to the data/raw directory."""
    print("Downloading /cases …")
    fetch_to_gz("cases", OUT_CASES)
    print(f"Wrote {OUT_CASES}")

    print("Downloading /trks …")
    fetch_to_gz("trks", OUT_TRKS)
    print(f"Wrote {OUT_TRKS}")


if __name__ == "__main__":
    main()
