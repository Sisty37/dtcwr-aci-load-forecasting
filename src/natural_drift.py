"""
EXPLORATORY, POST HOC (decision D16): recovery around NATURALLY occurring changes in the
untouched test segments. The test data were opened before this analysis was written, so
nothing here is confirmatory. All rules below are fixed in this file BEFORE it is run.

1. Natural change points (offline, evaluation only; no method sees them)
   - Signal: the frozen forecaster's residuals r = y - yhat on the TEST segment of each client.
   - Segmentation: exact optimal partitioning with a Gaussian (mean and variance) cost,
     candidate change points at day boundaries (every 24 steps), minimum segment 7 days,
     penalty 3 log(n) per change point (BIC with mean, variance and location).
   - "Material" change: over the 7 days before and after the point, residual SD ratio
     post / pre >= 1.5, or |mean shift| >= 1.0 x pre SD. Only these are analysed.
   - The point must lie >= 7 days after the test start and >= 500 steps before the end.
2. Per method (the frozen cohort configurations, untouched stream, clipped intervals as in
   the test run): TTR from the change point with the frozen TTR cell (cap = min(1000, steps
   left - L)), worst 168-step rolling coverage in the 500 steps after the point, and mean
   half-width over those 500 steps relative to the 2,160 h fixed window.
   For DTCW-R: whether it raised an alarm within 168 steps after the point.
3. Client level: mean over that client's material change points. DTCW-R+ACI vs each
   baseline: median difference, 95% cluster-bootstrap CI, Wilcoxon p (no multiplicity
   correction, no claims).

Run in EACH cohort working copy (cohort 1: D:\\icrest2027, cohort 2: D:\\icrest2027_c2):
    python src/natural_drift.py
Then, in the main copy, after copying the cohort-2 output folder to results/cohort2/natural:
    python src/natural_drift.py --summary
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib as cb  # noqa: E402
import inject as ij  # noqa: E402
import run_test as rt  # noqa: E402
from run_calib_val import ALPHA, load_forecasts  # noqa: E402

OUT = Path("results/natural")
HS = (1, 24)
DAY, WEEK, POST = 24, 168, 500
METHODS = ["fixed_168", "fixed_720", "fixed_2160", "aci", "rocp", "dtaci", "pid", "eci_cutoff",
           "driftocp", "dtcwr", "dtcwr_aci"]
LABEL = {"fixed_168": "Fixed 168", "fixed_720": "Fixed 720", "fixed_2160": "Fixed 2160", "aci": "ACI",
         "rocp": "ROCP", "dtaci": "DtACI", "pid": "Conformal PID", "eci_cutoff": "ECI-cutoff",
         "driftocp": "DriftOCP", "dtcwr": "DTCW-R", "dtcwr_aci": "DTCW-R+ACI"}


# ----------------------------------------------------------------------------- change points
def segment_cost(cs1, cs2, a, b):
    """Gaussian negative log-likelihood (up to constants) of x[a:b] with its own mean/variance."""
    n = b - a
    s1 = cs1[b] - cs1[a]
    s2 = cs2[b] - cs2[a]
    var = max(s2 / n - (s1 / n) ** 2, 1e-12)
    return n * math.log(var)


def change_points(x: np.ndarray, step: int = DAY, min_seg: int = WEEK, pen: float | None = None) -> list[int]:
    """Exact optimal partitioning over candidates at multiples of `step` (O(m^2), m = n / step)."""
    x = np.asarray(x, float)
    n = len(x)
    pen = 3 * math.log(n) if pen is None else pen
    cs1 = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    cand = list(range(0, n - min_seg + 1, step)) + [n]
    cand = sorted(set(c for c in cand if c == 0 or c == n or (c >= min_seg and n - c >= min_seg)))
    F = {0: -pen}
    last = {0: None}
    for j in cand[1:]:
        best, arg = math.inf, None
        for i in cand:
            if i >= j:
                break
            if i not in F or j - i < min_seg:
                continue
            v = F[i] + segment_cost(cs1, cs2, i, j) + pen
            if v < best:
                best, arg = v, i
        if arg is not None:
            F[j], last[j] = best, arg
    cps, j = [], n
    while j is not None and j in last and last[j] is not None:
        if last[j] != 0:
            cps.append(last[j])
        j = last[j]
    return sorted(cps)


def material(r: np.ndarray, cp: int, w: int = WEEK) -> tuple[bool, float, float]:
    pre, post = r[max(0, cp - w):cp], r[cp:cp + w]
    sd0 = float(np.std(pre)) or 1e-12
    ratio = float(np.std(post)) / sd0
    shift = abs(float(np.mean(post) - np.mean(pre))) / sd0
    return (ratio >= 1.5 or shift >= 1.0), ratio, shift


# ----------------------------------------------------------------------------- one client
def analyse_client(y, yhat, n_val, makers: dict, cell: dict) -> tuple[list[dict], list[dict]]:
    r = y - yhat
    n = len(y)
    test_cps = [n_val + c for c in change_points(r[n_val:])]
    keep = []
    for cp in test_cps:
        ok, ratio, shift = material(r, cp)
        if ok and cp >= n_val + WEEK and cp <= n - POST:
            keep.append((cp, ratio, shift))
    cps = [{"cp": cp - n_val, "sd_ratio": ra, "mean_shift_sd": sh} for cp, ra, sh in keep]
    if not keep:
        return cps, []
    runs = {}
    for m, make in makers.items():
        meth = make()
        out = cb.replay(meth, y, yhat, clip_inf=True)
        runs[m] = (out, getattr(meth, "alarms", None))
    ref_q = runs["fixed_2160"][0]["q"]
    rows = []
    for cp, ratio, shift in keep:
        cap = int(min(ij.TTR_CAP, n - cp - cell["L"]))
        for m, (out, alarms) in runs.items():
            cov = out["score"] <= out["q"]
            ttr, ok = ij.time_to_recover(cov, cp, ALPHA, cell["w"], cell["delta"], cell["L"], cap=cap)
            seg = slice(cp, cp + POST)
            roll = pd.Series(cov.astype(float)).rolling(WEEK).mean().to_numpy()
            row = {"method": m, "cp": cp - n_val, "ttr": min(ttr, cap), "cap": cap, "recovered": bool(ok),
                   "worst_cov_post": float(np.nanmin(roll[cp + WEEK - 1:cp + POST])) if POST >= WEEK else np.nan,
                   "rel_width_post": float(np.mean(out["q"][seg]) / np.mean(ref_q[seg])),
                   "sd_ratio": ratio, "mean_shift_sd": shift}
            if m == "dtcwr" and alarms is not None:
                row["alarm_within_week"] = any(cp <= a[0] < cp + WEEK for a in alarms)
            rows.append(row)
    return cps, rows


# ----------------------------------------------------------------------------- run / summary
def run():
    fz = rt.load_frozen()
    cell = rt.frozen_cell("electricity", fz)
    OUT.mkdir(parents=True, exist_ok=True)
    for h in HS:
        df = load_forecasts("electricity", h)
        rows_all = rt.method_rows("electricity", h, fz)
        makers = {m: rows_all[m] for m in METHODS}
        res, cps_all = [], []
        names = sorted(df.series.unique())
        for k, name in enumerate(names, 1):
            g = df[(df.series == name) & df.split.isin(["val", "test"])]
            y, yhat = g.y.to_numpy(float), g.yhat.to_numpy(float)
            n_val = int((g.split == "val").sum())
            cps, rows = analyse_client(y, yhat, n_val, makers, cell)
            for c in cps:
                cps_all.append({"series": name, "h": h, **c})
            for r_ in rows:
                res.append({"series": name, "h": h, **r_})
            print(f"  h={h} {k}/{len(names)} {name}: {len(cps)} material change points", flush=True)
        pd.DataFrame(res).to_csv(OUT / f"natural_h{h}.csv", index=False)
        pd.DataFrame(cps_all).to_csv(OUT / f"change_points_h{h}.csv", index=False)
    print(f"Outputs in {OUT}")


def boot_ci(d: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(d, len(d), replace=True)) for _ in range(B)]
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def summary():
    roots = {"c1": Path("results/natural"), "c2": Path("results/cohort2/natural")}
    out_rows, desc = [], []
    for h in HS:
        frames = []
        for tag, root in roots.items():
            f = root / f"natural_h{h}.csv"
            if f.exists():
                frames.append(pd.read_csv(f).assign(cohort=tag))
        if not frames:
            continue
        df = pd.concat(frames)
        for tag in ["c1", "c2", "pooled"]:
            d = df if tag == "pooled" else df[df.cohort == tag]
            if d.empty:
                continue
            cl = d.groupby(["cohort", "series", "method"]).agg(
                ttr=("ttr", "mean"), worst=("worst_cov_post", "mean"), width=("rel_width_post", "mean"),
                n_cp=("cp", "nunique")).reset_index()
            n_cl = cl[["cohort", "series"]].drop_duplicates().shape[0]
            n_cp = int(d[d.method == "dtcwr_aci"].shape[0])
            det = d[(d.method == "dtcwr")]["alarm_within_week"].astype(str).str.lower().eq("true").mean()
            for m, g in cl.groupby("method"):
                desc.append({"h": h, "cohort": tag, "method": LABEL[m], "clients": n_cl, "change_points": n_cp,
                             "median_ttr": float(g.ttr.median()), "median_worst_cov": float(g.worst.median()),
                             "median_rel_width": float(g.width.median()),
                             "dtcwr_alarm_within_week": float(det)})
            piv = cl.pivot_table(index=["cohort", "series"], columns="method", values="ttr")
            for m in METHODS:
                if m == "dtcwr_aci" or m not in piv:
                    continue
                dd = (piv["dtcwr_aci"] - piv[m]).dropna().to_numpy()
                if len(dd) < 5:
                    continue
                lo, hi = boot_ci(dd)
                p = float(stats.wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
                out_rows.append({"h": h, "cohort": tag, "baseline": LABEL[m], "n_clients": len(dd),
                                 "median_diff_ttr": float(np.median(dd)), "ci_lo": lo, "ci_hi": hi, "p": p,
                                 "share_faster": float((dd < 0).mean())})
    a, b = pd.DataFrame(desc), pd.DataFrame(out_rows)
    a.to_csv(OUT / "summary_methods.csv", index=False)
    b.to_csv(OUT / "summary_vs_baselines.csv", index=False)
    pd.set_option("display.width", 220)
    print("### EXPLORATORY natural change points: per-method medians over clients")
    print(a.round(3).to_string(index=False))
    print("\n### DTCW-R+ACI minus baseline, TTR at natural change points (negative = faster)")
    print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run()
