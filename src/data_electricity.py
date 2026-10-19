"""
Step 2 of the ICREST 2027 plan (v5, Section 5): clean the UCI
ElectricityLoadDiagrams20112014 data and select clients.

Run from the project root:
    python src/data_electricity.py

Input : data/LD2011_2014.txt   (semicolon separated, decimal comma, kW per 15 min)
Output: data/processed/electricity_hourly.(parquet|csv.gz)   all eligible clients, hourly kWh
        data/processed/selected_clients.csv                  the 50 sampled clients
        data/processed/excluded_clients.csv                  dropped clients + reason
        data/processed/preprocessing_summary.json            counts, DST fixes, file hashes
        results/preprocessing/dst_*.png                      before/after plots for each DST day

Conventions (from the UCI dataset page and the file itself)
- Each timestamp is the END of its 15-minute interval: "2011-01-01 00:15:00" covers 00:00-00:15.
  We convert to interval START before aggregating, so hour H = readings starting H:00, H:15, H:30, H:45.
- Values are kW averaged over 15 min. Hourly value = sum of the four readings / 4 = mean kW over
  the hour = energy in kWh for that hour. Reported as "hourly energy (kWh)".
- March DST day: four 15-min rows (interval starts 00:45-01:30, found in the real file) collapse:
  ~62-68% of active clients read 0 and the rest read tiny values (total ~3% of normal).
  The UCI page says "zero for all points", which the data does not match exactly.
  -> rows set missing for all clients and linearly interpolated per client.
- October DST day: the same four clock rows hold ~2x consumption -> divided by 2 (ratio checked).
  The rows are located from the data (March) and reused for October; see diagnose_dst.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PERIOD_START = pd.Timestamp("2012-01-01 00:00:00")   # first interval start kept
PERIOD_END = pd.Timestamp("2015-01-01 00:00:00")     # exclusive
MARCH_DST = [pd.Timestamp(d) for d in ("2012-03-25", "2013-03-31", "2014-03-30")]
OCTOBER_DST = [pd.Timestamp(d) for d in ("2012-10-28", "2013-10-27", "2014-10-26")]
MAX_ZERO_RUN_H = 24
N_SELECT = 50
SEED = 2027


# ----------------------------------------------------------------------------- loading
def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw file; index = interval START (timestamp - 15 min); float32 values."""
    cols = pd.read_csv(path, sep=";", nrows=0).columns
    df = pd.read_csv(path, sep=";", decimal=",", index_col=0,
                     dtype={c: "float32" for c in cols[1:]}, low_memory=False)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S") - pd.Timedelta(minutes=15)
    df.index.name = "start"
    if not df.index.is_monotonic_increasing:
        raise ValueError("timestamps are not sorted")
    step = df.index.to_series().diff().dropna().unique()
    if len(step) != 1 or step[0] != pd.Timedelta(minutes=15):
        raise ValueError(f"irregular 15-min grid: {step}")
    if df.isna().any().any():
        raise ValueError("unexpected missing values")
    return df


def restrict_period(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[(df.index >= PERIOD_START) & (df.index < PERIOD_END)]
    if out.index[0] != PERIOD_START or out.index[-1] != PERIOD_END - pd.Timedelta(minutes=15):
        raise ValueError(f"period boundaries wrong: {out.index[0]} .. {out.index[-1]}")
    return out


# ----------------------------------------------------------------------------- DST fixes
def _active_clients(df15: pd.DataFrame, day: pd.Timestamp) -> pd.Index:
    """Clients with non-zero consumption on the day before `day` (used to judge 'all zero')."""
    prev = df15.loc[(df15.index >= day - pd.Timedelta(days=1)) & (df15.index < day)]
    return prev.columns[(prev > 0).any()]


MARCH_DROP_THRESHOLD = 0.10   # block total < 10% of the day's median total load
OCTOBER_RATIO_RANGE = (1.6, 2.4)  # median per-client ratio that confirms a double hour


def find_march_zero_block(df15: pd.DataFrame, day: pd.Timestamp) -> pd.DatetimeIndex:
    """
    The March DST block = 15-min rows on `day` whose TOTAL load of active clients collapses
    below 10% of that day's median total. In the real file ~62-68% of active clients read
    exactly 0 there and the rest read tiny values (total ~3% of normal), so an 'all zero'
    rule does not work. Must be exactly 4 consecutive rows.
    """
    active = _active_clients(df15, day)
    rows = df15.loc[(df15.index >= day) & (df15.index < day + pd.Timedelta(days=1)), active]
    total = rows.sum(axis=1)
    low = total.index[total < MARCH_DROP_THRESHOLD * total.median()]
    if len(low) != 4 or (low[-1] - low[0]) != pd.Timedelta(minutes=45):
        raise ValueError(f"{day.date()}: expected one 4-row collapse block, found {list(low)}")
    return low


def fix_dst(df15: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    March: the collapse block is set missing for ALL clients (the non-zero readings inside it
    are artefacts too) and linearly interpolated per client.
    October: the same clock rows hold two readings' worth of consumption -> divided by 2,
    after confirming the ~2x ratio in the data.
    The block may straddle an hour boundary; fixes are applied at 15-min level before
    hourly aggregation, so that is handled.
    """
    df = df15.copy()
    report = {"march": [], "october": []}
    offsets = set()
    for day in MARCH_DST:
        block = find_march_zero_block(df, day)
        active = _active_clients(df, day)
        zero_share = float((df.loc[block, active] == 0).mean().mean())
        offsets.add(block[0] - day)
        df.loc[block] = np.nan
        report["march"].append({"day": str(day.date()), "rows": [str(t) for t in block],
                                "zero_share_active": round(zero_share, 3)})
    if len(offsets) != 1:
        raise ValueError(f"March block sits at different clock times: {offsets}")
    offset = offsets.pop()
    # linear interpolation in time, per client (only the NaN rows we just made)
    df = df.interpolate(method="time", limit=4, limit_area="inside")
    if df.isna().any().any():
        raise ValueError("interpolation left NaNs")

    for day in OCTOBER_DST:
        block = pd.date_range(day + offset, periods=4, freq="15min")
        before = _double_hour_ratio(df, block)
        lo, hi = OCTOBER_RATIO_RANGE
        if not lo <= before <= hi:
            raise ValueError(f"{day.date()}: rows {block[0]}..{block[-1]} are not ~2x "
                             f"(median ratio {before:.2f}); refusing to halve")
        df.loc[block] = df.loc[block] / 2
        after = _double_hour_ratio(df, block)
        report["october"].append({"day": str(day.date()), "rows": [str(t) for t in block],
                                  "median_ratio_before": round(before, 3),
                                  "median_ratio_after": round(after, 3)})
    report["dst_clock_offset_from_midnight"] = str(offset)
    return df, report


def _double_hour_ratio(df15: pd.DataFrame, block: pd.DatetimeIndex) -> float:
    """Median over active clients of: mean(block) / mean(the hour before and the hour after)."""
    day = block[0].normalize()
    active = _active_clients(df15, day)
    hour = df15.loc[block, active].mean()
    nb = pd.date_range(block[0] - pd.Timedelta(hours=1), periods=4, freq="15min").append(
        pd.date_range(block[-1] + pd.Timedelta(minutes=15), periods=4, freq="15min"))
    ref = df15.loc[nb, active].mean()
    ok = ref > 0
    return float((hour[ok] / ref[ok]).median())


# ----------------------------------------------------------------------------- hourly
def to_hourly_kwh(df15: pd.DataFrame) -> pd.DataFrame:
    counts = df15.groupby(df15.index.floor("h")).size()
    if (counts != 4).any():
        raise ValueError(f"hours without 4 readings: {counts[counts != 4].head()}")
    hourly = df15.groupby(df15.index.floor("h")).sum() / 4.0
    hourly.index.name = "hour_start"
    per_day = hourly.groupby(hourly.index.normalize()).size()
    if (per_day != 24).any():
        raise ValueError(f"days without 24 hours: {per_day[per_day != 24].head()}")
    return hourly.astype("float32")


# ----------------------------------------------------------------------------- clients
def longest_zero_run(x: np.ndarray) -> int:
    z = np.concatenate(([0], (x == 0).astype(np.int8), [0]))
    d = np.diff(z)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((ends - starts).max()) if len(starts) else 0


def select_clients(hourly: pd.DataFrame, n: int = N_SELECT, seed: int = SEED):
    runs = {c: longest_zero_run(hourly[c].to_numpy()) for c in hourly.columns}
    excluded = pd.DataFrame(
        [{"client": c, "longest_zero_run_h": r, "reason": f"zero run > {MAX_ZERO_RUN_H} h"}
         for c, r in runs.items() if r > MAX_ZERO_RUN_H])
    eligible = sorted(c for c, r in runs.items() if r <= MAX_ZERO_RUN_H)
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} eligible clients")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(eligible, size=n, replace=False).tolist())
    selected = pd.DataFrame({"client": chosen,
                             "longest_zero_run_h": [runs[c] for c in chosen]})
    return eligible, selected, excluded


# ----------------------------------------------------------------------------- outputs
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_table(df: pd.DataFrame, base: Path) -> Path:
    try:
        import pyarrow  # noqa: F401
        p = base.with_suffix(".parquet")
        df.to_parquet(p)
    except ImportError:
        p = base.with_suffix(".csv.gz")
        df.to_csv(p)
    return p


def plot_dst(raw15: pd.DataFrame, fixed15: pd.DataFrame, outdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    for day in MARCH_DST + OCTOBER_DST:
        lo, hi = day - pd.Timedelta(days=1), day + pd.Timedelta(days=2)
        active = _active_clients(raw15, day)
        a = raw15.loc[lo:hi, active].sum(axis=1) / 1000
        b = fixed15.loc[lo:hi, active].sum(axis=1) / 1000
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(a.index, a.values, lw=1, color="#9aa0a6", label="raw")
        ax.plot(b.index, b.values, lw=1.2, color="#1a73e8", label="after DST fix")
        ax.axvspan(day, day + pd.Timedelta(days=1), color="#fbbc04", alpha=0.12)
        ax.set_title(f"DST day {day.date()}: total load of active clients", fontsize=10)
        ax.set_ylabel("MW (15-min mean)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        f = outdir / f"dst_{day.date()}.png"
        fig.savefig(f, dpi=130)
        plt.close(fig)
        files.append(str(f))
    return files


# ----------------------------------------------------------------------------- main
def run(raw_path: Path, out_dir: Path, fig_dir: Path, make_plots: bool = True,
        n_select: int = N_SELECT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = restrict_period(load_raw(raw_path))
    fixed, dst_report = fix_dst(raw)
    hourly = to_hourly_kwh(fixed)
    eligible, selected, excluded = select_clients(hourly, n=n_select)

    data_path = save_table(hourly[eligible], out_dir / "electricity_hourly")
    sel_path, exc_path = out_dir / "selected_clients.csv", out_dir / "excluded_clients.csv"
    selected.to_csv(sel_path, index=False)
    excluded.to_csv(exc_path, index=False)
    plots = plot_dst(raw, fixed, fig_dir) if make_plots else []

    summary = {
        "unit": "hourly energy (kWh) = mean kW over the hour",
        "timestamp_convention": "index = hour START; raw labels were interval END",
        "period": [str(hourly.index[0]), str(hourly.index[-1])],
        "n_hours": int(len(hourly)),
        "n_days": int(len(hourly) // 24),
        "n_clients_raw": int(raw.shape[1]),
        "n_excluded": int(len(excluded)),
        "n_eligible": int(len(eligible)),
        "n_selected": int(len(selected)),
        "selection_seed": SEED,
        "max_zero_run_h": MAX_ZERO_RUN_H,
        "dst": dst_report,
        "sha256": {"raw_file": sha256(raw_path),
                   "selected_clients.csv": sha256(sel_path),
                   "hourly_data": sha256(data_path)},
        "outputs": {"hourly": str(data_path), "selected": str(sel_path),
                    "excluded": str(exc_path), "dst_plots": plots},
    }
    (out_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/LD2011_2014.txt")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--figs", default="results/preprocessing")
    args = ap.parse_args()
    s = run(Path(args.raw), Path(args.out), Path(args.figs))
    print(json.dumps({k: s[k] for k in ("period", "n_hours", "n_days", "n_clients_raw",
                                        "n_excluded", "n_eligible", "n_selected")}, indent=2))
    for d in s["dst"]["october"]:
        print(f"October {d['day']}: double-hour ratio {d['median_ratio_before']} -> "
              f"{d['median_ratio_after']} (expect ~2 -> ~1)")
    print("March zero blocks:", [m["rows"][0] for m in s["dst"]["march"]])
    print("Done. Check the DST plots in", args.figs)
