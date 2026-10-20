"""
Choose the DTCW-R v2 design on VALIDATION data of the original cohort (decision D13).
The v2 design is then re-tuned and confirmed on a FRESH cohort of 50 clients whose data no step
has touched (cohort 2), so looking at the v1 test results when proposing v2 does not bias the
confirmation.

Candidates (all keep the v1-selected W_S, W_L, k, c, the D8 detector variant and the v1 ACI gamma):
    v1               magnitude CUSUM, H as tuned in v1 (reference = DTCW-R+ACI of the v1 test)
    mag_exact        magnitude CUSUM, H to exactly 0.5 false alarms / 1,000 (fine bisection)
    mag_exact_reset  mag_exact + ACI copies reset to alpha at an alarm
    dual             magnitude OR miss scan, each calibrated to 0.25 / 1,000
    dual_reset       dual + ACI reset at an alarm
Criteria per stream (validation, 22 streams per client as in D10):
    W = D10 score (mean normalised Winkler)   R = mean RMST of TTR on the 21 injected streams
                                                 (frozen TTR cell, D11)
Rule (fixed here, before running): eligible = un-shifted coverage within 0.90 +/- 0.02 in every
stream; winner = lowest mean over the 4 streams of (rank by W + rank by R) / 2; ties go to the
earlier (simpler) candidate in the list above.

Run from the project root:
    python src/v2_design.py
Output: results/v2/design.csv, results/v2/design.json
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
import baselines as bl  # noqa: E402
import calib as cb  # noqa: E402
import calib_v2 as v2  # noqa: E402
import inject as ij  # noqa: E402
import run_prereg_selection as ps  # noqa: E402
import run_test as rt  # noqa: E402
import tune as tn  # noqa: E402
from run_calib_val import ALPHA, HORIZONS, LOCAL_W, load_forecasts  # noqa: E402

OUT = Path("results/v2")
CANDIDATES = ("v1", "mag_exact", "mag_exact_reset", "dual", "dual_reset")


def thresholds(ds, h, fz) -> dict:
    key = f"{ds}_h{h}"
    sel = fz["sel"][key]
    base = {k: sel["dtcwr"]["config"][k] for k in ("W_S", "W_L", "k", "c")}
    pw = fz["choice"][key]["winner"] == "prewhiten"
    val = load_forecasts(ds, h)
    val = val[val.split == "val"]
    scores = [np.abs(g.y.to_numpy() - g.yhat.to_numpy()) for _, g in val.groupby("series", sort=True)]
    block = max(tn.PERIOD[ds], 2 * h)
    reps = 1 if len(scores) > 1 else 20
    mx = ps.calibrate_H_bisect(scores, block, lambda H: cb.DTCWR(ALPHA, h, H=H, prewhiten=pw, **base),
                               grid=v2.MAG_GRID, reps_per_series=reps)
    du = v2.calibrate_dual(scores, block, base, h, prewhiten=pw, reps_per_series=reps)
    return {"base": base, "prewhiten": pw, "gamma": sel["dtcwr_aci"]["config"]["gamma"],
            "H_v1": sel["dtcwr"]["config"]["H"], "H_exact": mx["H"], "fa_exact": mx["fa_at_H"], "dual": du}


def make(cand: str, T: dict, h: int):
    b, pw, g = T["base"], T["prewhiten"], T["gamma"]
    if cand == "v1":
        return lambda: v2.DTCWR2(ALPHA, h, H=T["H_v1"], prewhiten=pw, aci_gamma=g, **b)
    if cand in ("mag_exact", "mag_exact_reset"):
        return lambda: v2.DTCWR2(ALPHA, h, H=T["H_exact"], prewhiten=pw, aci_gamma=g,
                                 aci_reset=cand.endswith("reset"), **b)
    d = T["dual"]
    return lambda: v2.DTCWR2(ALPHA, h, H=d["H"], H_miss=d["H_miss"], prewhiten=pw, aci_gamma=g,
                             aci_reset=cand.endswith("reset"), **b)


def eval_client(y, yhat, h, ds, factories: dict, cell: dict) -> list[dict]:
    week = LOCAL_W[ds]
    sig = bl.robust_scale(np.abs(y - yhat)[:cb.W0])
    rows = []
    for label, ys in tn.streams(y, yhat, h, week):
        tau = int(label.split("@")[1]) if "@" in label else None
        for cand, f in factories.items():
            out = cb.replay(f(), ys, yhat, clip_inf=True)
            ev = out["evaluated"]
            s, q = out["score"][ev], out["q"][ev]
            wk = float(np.mean(2 * q + (2 / ALPHA) * np.maximum(s - q, 0))) / sig
            rec = {"cand": cand, "stream": label, "cov": float((s <= q).mean()), "nwink": wk}
            if tau is not None:
                cov = out["score"] <= out["q"]
                rec["ttr"] = ij.time_to_recover(cov, tau, ALPHA, cell["w"], cell["delta"], cell["L"])[0]
            rows.append(rec)
    return rows


_D: dict = {}


def _init(ds, h, T):
    val = load_forecasts(ds, h)
    val = val[val.split == "val"]
    _D["series"] = {s: (g.y.to_numpy(float), g.yhat.to_numpy(float)) for s, g in val.groupby("series", sort=True)}
    _D["ds"], _D["h"], _D["T"] = ds, h, T
    _D["cell"] = rt.frozen_cell(ds, rt.load_frozen())


def _work(name):
    y, yhat = _D["series"][name]
    fac = {c: make(c, _D["T"], _D["h"]) for c in CANDIDATES}
    rows = eval_client(y, yhat, _D["h"], _D["ds"], fac, _D["cell"])
    return [dict(r, series=name) for r in rows]


def main(workers: int = max(1, (os.cpu_count() or 2) - 1)):
    OUT.mkdir(parents=True, exist_ok=True)
    fz = rt.load_frozen()
    summary, allrows = {}, []
    for ds in ("elec2", "electricity"):
        for h in HORIZONS[ds]:
            key = f"{ds}_h{h}"
            t0 = time.time()
            T = thresholds(ds, h, fz)
            names = json.loads((Path("results/models") / f"{key}.json").read_text())["series"]
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(ds, h, T)) as pool:
                rows = [r for rs in pool.map(_work, names) for r in rs]
            d = pd.DataFrame(rows).assign(stream_key=key)
            allrows.append(d)
            un = d[d.stream == "none"].groupby("cand")["cov"].mean()
            W = d.groupby(["cand", "series"])["nwink"].mean().groupby("cand").mean()
            R = d.dropna(subset=["ttr"]).groupby(["cand", "series"])["ttr"].mean().groupby("cand").mean()
            summary[key] = {"thresholds": T, "cov_unshifted": un.to_dict(), "W": W.to_dict(), "R": R.to_dict()}
            print(f"=== {key}  [{time.time() - t0:.0f} s]  H v1 {T['H_v1']} | exact {T['H_exact']} "
                  f"(fa {T['fa_exact']:.2f}) | dual H {T['dual']['H']} + H_miss {T['dual']['H_miss']} "
                  f"(union fa {T['dual']['fa_union']:.2f})")
            print(pd.DataFrame({"cov_unshifted": un, "W": W, "R_rmst": R}).loc[list(CANDIDATES)].round(3).to_string())
    pd.concat(allrows).to_csv(OUT / "design.csv", index=False)
    ranks = []
    for key, s in summary.items():
        W, R = pd.Series(s["W"]), pd.Series(s["R"])
        ranks.append(((W.rank() + R.rank()) / 2).rename(key))
    rk = pd.concat(ranks, axis=1)
    elig = {c: all(abs(summary[k]["cov_unshifted"][c] - 0.9) <= 0.02 for k in summary) for c in CANDIDATES}
    score = rk.mean(axis=1)
    order = [c for c in CANDIDATES if elig[c]]
    winner = min(order, key=lambda c: (round(score[c], 9), CANDIDATES.index(c))) if order else None
    res = {"rule": "lowest mean over streams of (rank W + rank R)/2 among eligible; ties -> simpler",
           "mean_rank": score.to_dict(), "eligible": elig, "winner": winner, "per_stream": summary}
    (OUT / "design.json").write_text(json.dumps(res, indent=2, default=float))
    print("\nMean rank (lower is better):")
    print(score.loc[list(CANDIDATES)].round(3).to_string())
    print(f"Eligible: {elig}\nv2 design winner: {winner}")


if __name__ == "__main__":
    main()
