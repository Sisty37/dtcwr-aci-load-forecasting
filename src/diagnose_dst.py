"""Diagnose how DST days look in the real UCI file. Run from project root:
    python src/diagnose_dst.py > results\\dst_diagnosis.txt
Prints, for each DST day, every 15-min row between 00:00 and 04:00 (interval START):
  share of active clients reading exactly 0, and total load of active clients.
Also lists the 15 rows in the whole file with the highest share of zero readings among active clients.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_electricity as de  # noqa: E402

raw = de.load_raw(Path("data/LD2011_2014.txt"))
print("rows:", len(raw), "clients:", raw.shape[1], "first start:", raw.index[0], "last start:", raw.index[-1])

days = sorted(de.MARCH_DST + de.OCTOBER_DST + [pd.Timestamp(d) for d in ("2011-03-27", "2011-10-30")])
for day in days:
    if day < raw.index[0] + pd.Timedelta(days=1):
        continue
    active = de._active_clients(raw, day)
    w = raw.loc[day: day + pd.Timedelta(hours=3, minutes=45), active]
    print(f"\n=== {day.date()} ({day.day_name()})  active clients: {len(active)}")
    print(" start   zero_share  total_kW")
    for t, row in w.iterrows():
        print(f" {t:%H:%M}   {float((row == 0).mean()):.3f}      {float(row.sum()):,.0f}")

# whole file: active = client had any non-zero reading in the previous 96 rows
nz = (raw > 0).astype(np.int8)
act = nz.rolling(96, min_periods=96).max().shift(1).fillna(0).astype(bool)
zero = (raw == 0) & act
share = zero.sum(axis=1) / act.sum(axis=1).clip(lower=1)
print("\n=== Top 15 rows by zero share among active clients (whole file)")
for t, v in share.sort_values(ascending=False).head(15).items():
    print(f" {t}  ({t.day_name()})  zero_share={v:.3f}  n_active={int(act.loc[t].sum())}")
