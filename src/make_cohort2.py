"""
Draw the FRESH replication cohort (decision D14): 50 Electricity clients from the eligible
clients that cohort 1 did not use. Run ONCE, inside the cohort-2 working copy, before any
cohort-2 step (forecasting, tuning, ...).

    python src/make_cohort2.py

Writes data/processed/cohort1_clients.csv (the 50 cohort-1 clients, kept for the record) and
overwrites data/processed/selected_clients.csv with the 50 cohort-2 clients. Refuses to run twice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast as fc  # noqa: E402

SEED2 = 2028
N = 50


def draw(eligible: list[str], cohort1: list[str], n: int = N, seed: int = SEED2) -> list[str]:
    pool = sorted(set(eligible) - set(cohort1))
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} unused eligible clients")
    return sorted(np.random.default_rng(seed).choice(pool, size=n, replace=False).tolist())


def main(d: Path = Path("data/processed")):
    c1_file = d / "cohort1_clients.csv"
    if c1_file.exists():
        raise SystemExit("cohort1_clients.csv exists: cohort 2 was already drawn in this copy.")
    wide = fc._read(d / "electricity_hourly")
    eligible = list(wide.columns)
    c1 = pd.read_csv(d / "selected_clients.csv")
    c1.to_csv(c1_file, index=False)
    c2 = draw(eligible, c1["client"].tolist())
    assert not set(c2) & set(c1["client"])
    pd.DataFrame({"client": c2}).to_csv(d / "selected_clients.csv", index=False)
    print(f"eligible {len(eligible)}, cohort 1 {len(c1)}, unused {len(eligible) - len(c1)}; "
          f"cohort 2 drawn with seed {SEED2}: {len(c2)} clients, overlap with cohort 1: 0")
    print("first five:", ", ".join(c2[:5]))


if __name__ == "__main__":
    main()
