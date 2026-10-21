"""
Decision D18: draw cohort 3 (pre-registered robustness study at 95% coverage) - 50 Electricity
clients from the eligible clients used by NEITHER cohort 1 NOR cohort 2. Run ONCE, inside the
cohort-3 working copy, before any cohort-3 step:

    python src/make_cohort3.py --cohort2 D:\\icrest2027_c2\\data\\processed\\selected_clients.csv

Writes data/processed/cohort1_clients.csv and cohort2_clients.csv (for the record) and overwrites
data/processed/selected_clients.csv with the 50 cohort-3 clients. Refuses to run twice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast as fc  # noqa: E402

SEED3 = 2029
N = 50


def draw(eligible, used, n=N, seed=SEED3):
    pool = sorted(set(eligible) - set(used))
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} unused eligible clients")
    return sorted(np.random.default_rng(seed).choice(pool, size=n, replace=False).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort2", required=True, help="selected_clients.csv of the cohort-2 working copy")
    a = ap.parse_args()
    d = Path("data/processed")
    if (d / "cohort2_clients.csv").exists():
        raise SystemExit("cohort 3 was already drawn in this copy.")
    eligible = list(fc._read(d / "electricity_hourly").columns)
    c1 = pd.read_csv(d / "selected_clients.csv")["client"].tolist()
    c2 = pd.read_csv(a.cohort2)["client"].tolist()
    if set(c1) & set(c2):
        raise SystemExit("cohort 1 and cohort 2 overlap: wrong files?")
    pd.DataFrame({"client": c1}).to_csv(d / "cohort1_clients.csv", index=False)
    pd.DataFrame({"client": c2}).to_csv(d / "cohort2_clients.csv", index=False)
    c3 = draw(eligible, c1 + c2)
    assert not set(c3) & (set(c1) | set(c2))
    pd.DataFrame({"client": c3}).to_csv(d / "selected_clients.csv", index=False)
    print(f"eligible {len(eligible)}, used by cohorts 1-2 {len(set(c1) | set(c2))}, "
          f"cohort 3 drawn with seed {SEED3}: {len(c3)} clients, overlap 0")
    print("first five:", ", ".join(c3[:5]))


if __name__ == "__main__":
    main()
