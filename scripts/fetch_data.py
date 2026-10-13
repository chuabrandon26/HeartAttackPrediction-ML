#!/usr/bin/env python3
"""Download the heart attack dataset from its original open deposit.

Source: Mendeley Data, doi:10.17632/wmhctcrt5v.1
        "Heart Attack Dataset", Zheen Hospital, Erbil, Iraq, Jan-May 2019.

The notebook previously read /kaggle/input/heart-attack-dataset/Medicaldataset.csv,
which only resolves inside a Kaggle kernel. This fetches the same file from the
source deposit so the notebook runs anywhere.

    python scripts/fetch_data.py
"""
import sys
import urllib.request
from pathlib import Path

URL = ("https://data.mendeley.com/public-files/datasets/wmhctcrt5v/files/"
       "959a4949-b1c2-4858-b361-2602bb60a07a/file_downloaded")
OUT = Path(__file__).resolve().parent.parent / "data" / "Medicaldataset.csv"
EXPECTED_ROWS = 1319


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"already present: {OUT}")
    else:
        print(f"downloading to {OUT}")
        urllib.request.urlretrieve(URL, OUT)

    rows = OUT.read_text(encoding="utf8").strip().splitlines()
    n = len(rows) - 1
    print(f"{n} records, columns: {rows[0]}")
    if n != EXPECTED_ROWS:
        sys.exit(f"expected {EXPECTED_ROWS} records, found {n}; the deposit may have changed")
    print("row count matches the published deposit")


if __name__ == "__main__":
    main()
