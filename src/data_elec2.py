"""
Elec2 loader for the ICREST 2027 plan (v5, Sections 5 and 6.2). Run from the project root:
    python src/data_elec2.py

Input : data/conformal-time-series/tests/datasets/elec2.csv            (45,312 half-hourly rows)
        data/conformal-time-series/tests/datasets/proc/elec2/<model>.npz  (published-protocol forecasts)
Output: data/processed/elec2_main.(parquet|csv.gz)   P-main: full series, timestamp + y + split label
        data/processed/elec2_pub.(parquet|csv.gz)    P-pub : first 2,000 rows + the repo's forecasts
        data/processed/elec2_summary.json            counts, split boundaries, checks, hashes

What the Conformal PID code does (tests/datasets.py, tests/configs/elec2.yaml), reproduced here:
- timestamp = half-hourly range 1996-05-07 00:00 .. 1998-12-06 23:30 (inclusive)
- target y  = column 'nswdemand' (already min-max normalised to [0, 1] in the file)
- ONLY THE FIRST 2,000 ROWS are used (df[:2000]); T_burnin = 300; alpha = 0.1;
  seasonal_period = 48; score = signed residual.
  => published Elec2 numbers are computed on at most 1,700 post-burn-in points.

Our two protocols (plan v5, Section 6.2):
- P-main: the FULL 45,312-row series, chronological 70/10/20 split, our own forecasters.
- P-pub : exactly the repo's 2,000-row window and its saved forecasts; used for Table C only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("data/conformal-time-series/tests/datasets")
EXPECTED_ROWS = 45_312
TS_START, TS_END = "1996-05-07 00:00:00", "1998-12-06 23:30:00"
PUB_ROWS, PUB_BURNIN = 2_000, 300
SPLIT = (0.70, 0.10, 0.20)
PUB_MODELS = ("ar", "theta", "prophet", "transformer")


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


def load_elec2(csv_path: Path, expected_rows: int = EXPECTED_ROWS,
               ts_start: str = TS_START, ts_end: str = TS_END) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if len(df) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(df)}")
    ts = pd.date_range(start=ts_start, end=ts_end, freq="30min", inclusive="both")
    if len(ts) != len(df):
        raise ValueError(f"timestamp range has {len(ts)} entries for {len(df)} rows")
    out = pd.DataFrame({"y": df["nswdemand"].astype("float64").to_numpy()}, index=ts)
    out.index.name = "timestamp"
    # consistency checks against the file's own calendar columns
    per_day = out.groupby(out.index.normalize()).size()
    if (per_day != 48).any():
        raise ValueError("days without 48 half-hours")
    period = np.round(df["period"].to_numpy() * 47).astype(int)          # 0..47 within the day
    if not np.array_equal(period, out.index.hour.to_numpy() * 2 + out.index.minute.to_numpy() // 30):
        raise ValueError("'period' column does not match the reconstructed timestamps")
    if out["y"].isna().any():
        raise ValueError("missing target values")
    if not ((out["y"] >= 0) & (out["y"] <= 1)).all():
        raise ValueError("target outside [0, 1]; file is expected to be min-max normalised")
    return out


def split_main(n: int, frac=SPLIT) -> np.ndarray:
    n_tr, n_va = int(round(frac[0] * n)), int(round(frac[1] * n))
    lab = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va), dtype=object)
    return lab


def build_pub(series: pd.DataFrame, proc_dir: Path) -> tuple[pd.DataFrame, dict]:
    pub = series.iloc[:PUB_ROWS].copy()
    pub["phase"] = np.where(np.arange(PUB_ROWS) < PUB_BURNIN, "burnin", "eval")
    report = {}
    for m in PUB_MODELS:
        f = proc_dir / f"{m}.npz"
        if not f.exists():
            report[m] = "missing"
            continue
        fc = np.load(f, allow_pickle=True)["forecasts"].astype("float64")
        if len(fc) != PUB_ROWS:
            raise ValueError(f"{m}.npz has {len(fc)} forecasts, expected {PUB_ROWS}")
        pub[f"fc_{m}"] = fc
        ev = pub["phase"] == "eval"
        nz = int((fc[:PUB_BURNIN] == 0).sum())
        report[m] = {"mae_eval": round(float(np.abs(pub.loc[ev, "y"] - fc[ev.to_numpy()]).mean()), 5),
                     "zero_forecasts_in_burnin": nz,
                     "zero_forecasts_in_eval": int((fc[ev.to_numpy()] == 0).sum()),
                     "sha256": sha256(f)}
    return pub, report


def run(repo: Path = REPO, out_dir: Path = Path("data/processed"),
        expected_rows: int = EXPECTED_ROWS, ts_start: str = TS_START, ts_end: str = TS_END) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = repo / "elec2.csv"
    s = load_elec2(csv_path, expected_rows, ts_start, ts_end)

    main = s.copy()
    main["split"] = split_main(len(main))
    pub, pub_report = build_pub(s, repo / "proc" / "elec2")

    p_main = save_table(main, out_dir / "elec2_main")
    p_pub = save_table(pub, out_dir / "elec2_pub")
    bounds = {k: [str(main.index[main["split"] == k][0]), str(main.index[main["split"] == k][-1]),
                  int((main["split"] == k).sum())] for k in ("train", "val", "test")}
    summary = {
        "source": str(csv_path), "sha256_csv": sha256(csv_path),
        "n_rows": int(len(s)), "n_days": int(len(s) // 48),
        "period": [str(s.index[0]), str(s.index[-1])],
        "target": "nswdemand (min-max normalised in source file)",
        "P_main": {"split_fractions": SPLIT, "boundaries_[first,last,count]": bounds,
                   "file": str(p_main)},
        "P_pub": {"rows_used": PUB_ROWS, "burnin": PUB_BURNIN,
                  "eval_points": PUB_ROWS - PUB_BURNIN,
                  "repo_settings": {"alpha": 0.1, "seasonal_period": 48,
                                    "score": "signed-residual"},
                  "forecasts": pub_report, "file": str(p_pub)},
    }
    (out_dir / "elec2_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    sm = run(Path(a.repo), Path(a.out))
    print(json.dumps({k: sm[k] for k in ("n_rows", "n_days", "period")}, indent=2))
    for k, v in sm["P_main"]["boundaries_[first,last,count]"].items():
        print(f"P-main {k:5s}: {v[0]} .. {v[1]}  ({v[2]} points)")
    print(f"P-pub : first {PUB_ROWS} rows, burn-in {PUB_BURNIN}, eval {PUB_ROWS - PUB_BURNIN} points")
    for m, r in sm["P_pub"]["forecasts"].items():
        print(f"  {m:12s} {r if isinstance(r, str) else {k: r[k] for k in ('mae_eval', 'zero_forecasts_in_burnin', 'zero_forecasts_in_eval')}}")
