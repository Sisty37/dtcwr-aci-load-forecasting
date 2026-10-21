"""
First look at the calibration methods on the VALIDATION segment only (plan v5, Step 5).
The TEST segment is NOT touched here: it stays sealed until tuning and the TTR rule are
frozen and the pre-registration commit is made (plan Section 6.2).

Run from the project root:
    python src/run_calib_val.py                 # both datasets, all horizons
    python src/run_calib_val.py --dataset elec2

Output: results/calib_val/<dataset>_h<h>.csv   one row per (series, method), validation metrics
        printed table: median over series per method
Default (untuned) configurations are used; tuning on validation comes in the next step.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib as cb  # noqa: E402

# Miscoverage level. 0.1 for cohorts 1 and 2; cohort 3 (decision D18) runs the unchanged
# pipeline at 0.05 by setting the environment variable ICREST_ALPHA before every script.
ALPHA = float(os.environ.get("ICREST_ALPHA", "0.1"))
LOCAL_W = {"electricity": 168, "elec2": 336}          # one calendar week of targets
HORIZONS = {"electricity": (1, 24), "elec2": (1, 48)}


def default_methods(dataset: str, h: int) -> dict:
    wk = LOCAL_W[dataset]                              # one week in steps
    return {
        "fixed_1w": lambda: cb.FixedWindow(ALPHA, h, wk),
        "fixed_1m": lambda: cb.FixedWindow(ALPHA, h, wk * 30 // 7),
        "fixed_all": lambda: cb.FixedWindow(ALPHA, h, 10**7),
        "aci_0.005": lambda: cb.ACI(ALPHA, h, gamma=0.005),
        "dtcwr": lambda: cb.DTCWR(ALPHA, h, W_S=48, W_L=wk * 30 // 7, H=6, k=0.5, c=24),
        "dtcwr_aci": lambda: cb.DTCWR(ALPHA, h, W_S=48, W_L=wk * 30 // 7, H=6, k=0.5, c=24,
                                      aci_gamma=0.005),
    }


def load_forecasts(dataset: str, h: int, fc_dir=Path("results/forecasts")) -> pd.DataFrame:
    p = fc_dir / f"{dataset}_h{h}.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(p.with_suffix(".csv.gz"), parse_dates=["target"])
    return df.sort_values(["series", "target"])


def miss_autocorr_neff(miss: np.ndarray, K: int = 48) -> float:
    """n_eff = n / (1 + 2 sum rho_k), summing positive autocorrelations (initial positive sequence)."""
    x = miss - miss.mean()
    v = (x * x).mean()
    if v == 0:
        return float(len(x))
    s = 0.0
    for k in range(1, K + 1):
        r = (x[:-k] * x[k:]).mean() / v
        if r <= 0:
            break
        s += r
    return float(len(x) / (1 + 2 * s))


def run(dataset: str, h: int, out_dir=Path("results/calib_val")) -> pd.DataFrame:
    df = load_forecasts(dataset, h)
    val = df[df["split"] == "val"]
    rows = []
    for sid, g in val.groupby("series", sort=True):
        y, yhat = g["y"].to_numpy(), g["yhat"].to_numpy()
        for name, make in default_methods(dataset, h).items():
            m = make()
            r = cb.replay(m, y, yhat)
            met = cb.interval_metrics(y, yhat, r["q"], r["evaluated"], ALPHA, LOCAL_W[dataset])
            miss = (~r["covered"][r["evaluated"]]).astype(float)
            met.update(series=sid, method=name, n_eff=miss_autocorr_neff(miss),
                       n_alarms=len(getattr(m, "alarms", [])))
            rows.append(met)
    res = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / f"{dataset}_h{h}.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "electricity", "elec2"])
    a = ap.parse_args()
    pd.set_option("display.width", 160)
    for ds in (["electricity", "elec2"] if a.dataset == "all" else [a.dataset]):
        for h in HORIZONS[ds]:
            t0 = time.time()
            res = run(ds, h)
            cols = ["coverage", "coverage_error", "winkler", "mean_width",
                    "worst_local_coverage", "longest_under_run", "n_alarms", "n_eff", "n"]
            tab = res.groupby("method")[cols].median().round(4)
            print(f"\n=== {ds}  h={h}  (VALIDATION only, median over {res['series'].nunique()} series)"
                  f"  [{time.time() - t0:.0f} s]")
            print(tab.to_string())
