"""Merge the E10 seed groups (run as separate processes to bound memory) into one file each."""
from __future__ import annotations

import pandas as pd

from .. import config as C


def main():
    for stem in ("e10_baselines", "e10_per_type"):
        parts = sorted(C.RESULTS_DIR.glob(f"{stem}_*.csv"))
        if not parts:
            continue
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df = df.drop_duplicates(subset=["detector", "variant", "seed"], keep="last")
        df = df.sort_values(["detector", "seed", "variant"]).reset_index(drop=True)
        df.to_csv(C.RESULTS_DIR / f"{stem}.csv", index=False)
        print(stem, len(df), "rows from", [p.name for p in parts])


if __name__ == "__main__":
    main()
