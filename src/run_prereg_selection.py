"""
Last two choices before the pre-registration commit (plan v5 Sections 8.2 and 9), both on
VALIDATION data only. The test segment stays sealed.

1. TTR sensitivity cell (plan 8.2, decision D11).
   Grid: w in {84, 168, 336} h x delta in {0.025, 0.05} x L in {w/2, w} (12 cells; hours are
   doubled for half-hourly Elec2). For each cell, the false non-recovery rate = share of
   (client, position) pairs on UN-SHIFTED validation streams where the selected ACI or the long
   fixed window (2160 h) does not "recover" within the cap, although nothing was shifted.
   Rule: keep the primary cell (w = 168 h, delta = 0.05, L = w) if its rate is < 5% for both
   methods in every dataset x horizon; otherwise take the strictest cell that meets it in every
   dataset x horizon, strictest = smallest delta, then largest L/w, then w closest to 168 h.
   Coverage indicators include the warm-up intervals (every target from h on has an interval),
   so the rolling window is always full.

2. Matched false-alarm thresholds for the 2x2 ablation (plan Sec. 9, decision D11).
   Cells, all with the selected DTCW-R config (W_S, W_L, k, c, detector variant):
       D  magnitude signal + anchored window  (= DTCW-R)          H from tuning
       B  magnitude signal + reset window                          same H as D (the window rule
                                                                   does not enter the magnitude
                                                                   detector)
       A  miss-indicator signal + anchored window                  H calibrated here
       R  miss-indicator signal + reset window (DriftOCP-like)     H calibrated here
   Calibration as in D6: block bootstrap of validation scores, 0.5 false alarms per 1,000 steps.

Run from the project root (after src/tune.py):
    python src/run_prereg_selection.py
Output: results/prereg/ttr_sensitivity.csv / .json
        results/prereg/ablation_H.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib as cb  # noqa: E402
import detector_calibration as dc  # noqa: E402
import inject as ij  # noqa: E402
import tune as tn  # noqa: E402
from run_calib_val import ALPHA, HORIZONS, LOCAL_W, load_forecasts  # noqa: E402

OUT = Path("results/prereg")
NONREC_MAX = 0.05
# The miss statistic |sum(e - alpha)| / sqrt(n) has null SD sqrt(alpha (1 - alpha)) = 0.3, not 1,
# so its thresholds live on a smaller scale than the magnitude CUSUM's. Fine grid (step 0.05),
# searched by bisection (the alarm rate is non-increasing in H up to Monte Carlo noise).
MISS_GRID = tuple(round(0.5 + 0.05 * i, 2) for i in range(91))          # 0.50 .. 5.00


def calibrate_H_bisect(val_scores, block, make_with_H, grid=MISS_GRID, target=dc.TARGET_FA,
                       length=10_000, reps_per_series=1, seed=2027):
    """Smallest grid H with bootstrap alarm rate <= target (same streams for every H)."""
    rng = np.random.default_rng(seed)
    streams = [dc.block_bootstrap(np.asarray(x, float), length, block, rng)
               for x in val_scores for _ in range(reps_per_series)]
    curve = {}

    def rate(H):
        if H not in curve:
            curve[H] = float(np.mean([dc.alarm_rate(st, lambda: make_with_H(H)) for st in streams]))
        return curve[H]

    lo, hi = 0, len(grid) - 1
    if rate(grid[hi]) > target:
        return {"H": grid[hi], "fa_at_H": rate(grid[hi]), "curve": curve, "warning": "target not reached on grid"}
    while lo < hi:
        mid = (lo + hi) // 2
        if rate(grid[mid]) <= target:
            hi = mid
        else:
            lo = mid + 1
    return {"H": grid[lo], "fa_at_H": rate(grid[lo]), "curve": dict(sorted(curve.items())),
            "n_streams": len(streams)}


def ttr_cells(ds: str) -> list[dict]:
    s = tn.scale(ds)
    return [{"w": w * s, "delta": d, "L": L * s, "w_h": w, "L_over_w": L / w}
            for w in (84, 168, 336) for d in (0.025, 0.05) for L in (w // 2, w)]


def is_primary(c: dict) -> bool:
    return c["w_h"] == 168 and c["delta"] == 0.05 and c["L_over_w"] == 1


def strictness_key(c: dict):
    """Sort key: strictest first."""
    return (c["delta"], -c["L_over_w"], abs(c["w_h"] - 168))


def full_covered(method: cb.Method, y, yhat) -> np.ndarray:
    out = cb.replay(method, y, yhat, clip_inf=True)
    cov = out["score"] <= out["q"]                 # NaN q (first h targets) -> False
    return cov


def nonrecovery_rows(ds, h, sid, y, yhat, sel_aci, cells) -> list[dict]:
    rows = []
    methods = {"aci": tn.make_method("aci", sel_aci, h),
               "fixed_long": tn.make_method("fixed", {"W": 2160 * tn.scale(ds)}, h)}
    for mname, m in methods.items():
        cov = full_covered(m, y, yhat)
        for tau in ij.positions(len(y), h, follow=ij.TTR_CAP):
            for c in cells:
                _, ok = ij.time_to_recover(cov, tau, ALPHA, c["w"], c["delta"], c["L"])
                rows.append({"dataset": ds, "h": h, "series": sid, "method": mname, "tau": tau,
                             "w_h": c["w_h"], "delta": c["delta"], "L_over_w": c["L_over_w"],
                             "nonrecovered": not ok})
    return rows


def choose_cell(df: pd.DataFrame, cells: list[dict]) -> dict:
    rate = (df.groupby(["dataset", "h", "method", "w_h", "delta", "L_over_w"])["nonrecovered"]
              .mean().reset_index())
    worst = rate.groupby(["w_h", "delta", "L_over_w"])["nonrecovered"].max()   # over ds, h, method
    ok = {k for k, v in worst.items() if v < NONREC_MAX}
    key = lambda c: (c["w_h"], c["delta"], c["L_over_w"])   # noqa: E731
    prim = next(c for c in cells if is_primary(c))
    if key(prim) in ok:
        chosen, why = prim, "primary cell meets the rule"
    else:
        cand = sorted([c for c in cells if key(c) in ok], key=strictness_key)
        chosen = cand[0] if cand else None
        why = "strictest cell meeting the rule" if cand else "NO cell meets the rule"
    return {"chosen": None if chosen is None else {k: chosen[k] for k in ("w_h", "delta", "L_over_w")},
            "reason": why, "max_rate_by_cell": {f"w={k[0]}h,delta={k[1]},L/w={k[2]}": round(float(v), 4)
                                               for k, v in worst.items()}}


# ----------------------------------------------------------------------------- workers
_D: dict = {}


def _init(ds, h):
    val = load_forecasts(ds, h)
    val = val[val["split"] == "val"]
    _D["series"] = {sid: (g["y"].to_numpy(float), g["yhat"].to_numpy(float))
                    for sid, g in val.groupby("series", sort=True)}
    _D["ds"], _D["h"] = ds, h


def _ttr_work(args):
    sid, sel_aci = args
    y, yhat = _D["series"][sid]
    return nonrecovery_rows(_D["ds"], _D["h"], sid, y, yhat, sel_aci, ttr_cells(_D["ds"]))


def _cal_work(args):
    cfg, prewhiten, rule, block, reps = args
    h = _D["h"]
    scores = [np.abs(y - yhat) for y, yhat in _D["series"].values()]
    kw = {k: cfg[k] for k in ("W_S", "W_L", "k", "c")}
    return calibrate_H_bisect(scores, block,
                              lambda H: cb.DTCWR(ALPHA, h, H=H, prewhiten=prewhiten, signal="miss",
                                                 window_rule=rule, **kw),
                              reps_per_series=reps)


def main(workers: int = max(1, (os.cpu_count() or 2) - 1)):
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((tn.OUT / "selected.json").read_text())
    choice = json.loads(Path("results/detector_choice.json").read_text())
    ttr_rows, abl = [], {}
    for ds in ("elec2", "electricity"):
        for h in HORIZONS[ds]:
            t0 = time.time()
            key = f"{ds}_h{h}"
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(ds, h)) as pool:
                val = load_forecasts(ds, h)
                sids = sorted(val.loc[val["split"] == "val", "series"].unique())
                for rows in pool.map(_ttr_work, [(sid, sel[key]["aci"]["config"]) for sid in sids]):
                    ttr_rows.extend(rows)
                d = sel[key]["dtcwr"]["config"]
                pw = choice[key]["winner"] == "prewhiten"
                block = max(tn.PERIOD[ds], 2 * h)
                reps = 1 if len(sids) > 1 else 20
                cals = list(pool.map(_cal_work, [(d, pw, rule, block, reps) for rule in ("anchored", "reset")]))
            hcal = json.loads((tn.OUT / key / "dtcwr_H.json").read_text())
            mfa = next(x["fa_at_H"] for x in hcal
                       if all(x["config"][k] == d[k] for k in ("W_S", "W_L", "k", "c")))
            abl[key] = {"config": {k: d[k] for k in ("W_S", "W_L", "k", "c")}, "prewhiten": pw,
                        "D_magnitude_anchored": {"H": d["H"], "fa_at_H": mfa},
                        "B_magnitude_reset": {"H": d["H"], "fa_at_H": mfa},
                        "A_miss_anchored": {"H": cals[0]["H"], "fa_at_H": cals[0]["fa_at_H"],
                                            "warning": cals[0].get("warning", "")},
                        "R_miss_reset": {"H": cals[1]["H"], "fa_at_H": cals[1]["fa_at_H"],
                                         "warning": cals[1].get("warning", "")}}
            print(f"=== {key}: miss-signal H  anchored {cals[0]['H']} (fa {cals[0]['fa_at_H']:.2f})"
                  f"  reset {cals[1]['H']} (fa {cals[1]['fa_at_H']:.2f});  magnitude H {d['H']} (fa {mfa:.2f})"
                  f"  [{time.time() - t0:.0f} s]")
    df = pd.DataFrame(ttr_rows)
    df.to_csv(OUT / "ttr_sensitivity.csv", index=False)
    res = choose_cell(df, ttr_cells("electricity"))
    (OUT / "ttr_sensitivity.json").write_text(json.dumps(res, indent=2))
    (OUT / "ablation_H.json").write_text(json.dumps(abl, indent=2))
    tab = (df.groupby(["w_h", "delta", "L_over_w", "dataset", "h", "method"])["nonrecovered"].mean()
             .unstack(["dataset", "h", "method"]).round(3))
    pd.set_option("display.width", 200)
    print("\nFalse non-recovery rate on un-shifted validation streams (rows: TTR cell)")
    print(tab.to_string())
    print(f"\nTTR cell: {res['chosen']}  ({res['reason']})")


if __name__ == "__main__":
    main()
