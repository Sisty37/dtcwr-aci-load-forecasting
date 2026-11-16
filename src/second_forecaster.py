"""
POST HOC robustness check requested in the ICREST'27 meta-review (point 1) and by both reviewers:
a second, structurally different forecaster.

Forecaster: seasonal naive, yhat(t) = y(t - 24) (one day back), already stored as `yhat_snaive`
in results/forecasts (Step 4 of the pipeline). It is a different model class from LightGBM and has
larger, more autocorrelated residuals.

Protocol (cohort 1, Electricity, h = 1 and 24, test segment only):
  - every method keeps its frozen LightGBM-tuned configuration (no re-tuning);
  - the only change: the CUSUM threshold H of DTCW-R+ACI is re-calibrated on the seasonal-naive
    VALIDATION scores with the same block-bootstrap rule and the same 0.5 / 1,000 target (D6);
  - shifts are injected at the residual level (layer L1_resid of the paper: the same 7 settings
    and 5 positions), because target-level injection needs the LightGBM model;
  - outcomes: test coverage and Winkler score relative to the 2,160 h window on the untouched
    stream, RMST of the time to recover over the 35 shifts (frozen TTR cell).
DTCW-R+ACI is compared with each baseline by median client difference, 95% bootstrap CI and
Wilcoxon p (exploratory: no multiplicity correction, no confirmatory claims).

Run from the cohort-1 project root (D:\\icrest2027):
    python src/second_forecaster.py --workers 3
Outputs in results/posthoc/: snaive_clients_h<h>.csv, snaive_summary.csv, snaive_vs_baselines.csv,
    snaive_H.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib as cb  # noqa: E402
import detector_calibration as dc  # noqa: E402
import inject as ij  # noqa: E402
import run_test as rt  # noqa: E402
import tune as tn  # noqa: E402
from run_calib_val import ALPHA, LOCAL_W, load_forecasts  # noqa: E402

DS = "electricity"
OUT = Path("results/posthoc")
BASELINES = ("fixed_168", "fixed_720", "fixed_2160", "aci", "rocp", "dtaci", "pid", "eci_cutoff", "driftocp")
LABEL = {"fixed_168": "Fixed 168", "fixed_720": "Fixed 720", "fixed_2160": "Fixed 2160", "aci": "ACI",
         "rocp": "ROCP", "dtaci": "DtACI", "pid": "Conformal PID", "eci_cutoff": "ECI-cutoff",
         "driftocp": "DriftOCP", "dtcwr_aci": "DTCW-R+ACI"}


def snaive_streams(h):
    df = load_forecasts(DS, h)
    df = df[df["split"].isin(["val", "test"])]
    return {s: (g["y"].to_numpy(float), g["yhat_snaive"].to_numpy(float), int((g["split"] == "val").sum()))
            for s, g in df.groupby("series", sort=True)}


def calibrate(h, fz, streams):
    key = f"{DS}_h{h}"
    cfg = fz["sel"][key]["dtcwr_aci"]["config"]
    pw = fz["choice"][key]["winner"] == "prewhiten"
    kw = {k: cfg[k] for k in ("W_S", "W_L", "k", "c")}
    val = [np.abs(y - yh)[:nv] for y, yh, nv in streams.values()]
    res = dc.calibrate_H(val, block=max(24, 2 * h), make_with_H=lambda H: cb.DTCWR(ALPHA, h, H=H, prewhiten=pw, **kw))
    return {"H_lightgbm": cfg["H"], "H_snaive": res["H"], "fa_at_H": res["fa_at_H"],
            "warning": res.get("warning")}


_W: dict = {}


def _init(h, H_new):
    fz = rt.load_frozen()
    key = f"{DS}_h{h}"
    sel = fz["sel"][key]
    pw = fz["choice"][key]["winner"] == "prewhiten"
    mk = {}
    for c in sel["fixed"]["configs"]:
        mk[f"fixed_{c['W']}"] = (lambda c=c: tn.make_method("fixed", c, h))
    for fam in ("aci", "rocp", "dtaci", "pid", "eci_cutoff", "driftocp"):
        mk[fam] = (lambda fam=fam, cfg=sel[fam]["config"]: tn.make_method(fam, cfg, h))
    cfg = dict(sel["dtcwr_aci"]["config"])
    cfg["H"] = H_new
    mk["dtcwr_aci"] = lambda: tn.make_method("dtcwr_aci", cfg, h, {"prewhiten": pw})
    _W.update(h=h, fz=fz, makers=mk, streams=snaive_streams(h))


def _work(name):
    h, fz = _W["h"], _W["fz"]
    y, yh, n_val = _W["streams"][name]
    n, week = len(y), LOCAL_W[DS]
    cell = rt.frozen_cell(DS, fz)
    taus = ij.test_positions(n_val, n - n_val, L=cell["L"])
    r = y - yh
    shifted = [(kind, tau, yh + ij.inject_residuals(r, ij.Shift(kind, size, tau, week)))
               for tau in taus for kind, size in ij.SETTINGS]
    test = np.zeros(n, bool)
    test[n_val:] = True
    rows = []
    for mname, make in _W["makers"].items():
        m = make()
        out = cb.replay(m, y, yh, clip_inf=True)
        met = cb.interval_metrics(y, yh, out["q"], test, ALPHA, week)
        al = [a[0] for a in getattr(m, "alarms", [])]
        ttrs = []
        for kind, tau, y_t in shifted:
            o = cb.replay(make(), y_t, yh, clip_inf=True)
            ttr, _ = ij.time_to_recover(o["score"] <= o["q"], tau, ALPHA, cell["w"], cell["delta"], cell["L"])
            ttrs.append(min(ttr, ij.TTR_CAP))
        rows.append({"h": h, "series": name, "method": mname, "coverage_L2": met["coverage"],
                     "winkler_L2": met["winkler"], "rmst_L1_resid": float(np.mean(ttrs)),
                     "fa_test_per_1000": (1000 * sum(t >= n_val for t in al) / (n - n_val))
                     if hasattr(m, "alarms") else np.nan})
    return name, rows


def boot_ci(d, B=2000, seed=0):
    meds = np.median(np.random.default_rng(seed).choice(d, (B, len(d)), replace=True), axis=1)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fz = rt.load_frozen()
    hfile = OUT / "snaive_H.json"
    Hs = json.loads(hfile.read_text()) if hfile.exists() else {}
    frames = []
    for h in (1, 24):
        if str(h) not in Hs:
            print(f"=== h={h}: calibrating H on seasonal-naive validation scores ...", flush=True)
            Hs[str(h)] = calibrate(h, fz, snaive_streams(h))
            hfile.write_text(json.dumps(Hs, indent=2))
        print(f"    H: LightGBM {Hs[str(h)]['H_lightgbm']} -> seasonal naive {Hs[str(h)]['H_snaive']}", flush=True)
        f = OUT / f"snaive_clients_h{h}.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
            continue
        names = sorted(snaive_streams(h).keys())
        t0, rows = time.time(), []
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                                 initargs=(h, Hs[str(h)]["H_snaive"])) as pool:
            futs = [pool.submit(_work, nm) for nm in names]
            for i, fu in enumerate(as_completed(futs), 1):
                nm, r = fu.result()
                rows += r
                print(f"  {i}/{len(names)} {nm}  [{time.time() - t0:.0f} s]", flush=True)
        d = pd.DataFrame(rows)
        d.to_csv(f, index=False)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    ref = df[df.method == "fixed_2160"].set_index(["h", "series"])["winkler_L2"]
    df = df.merge(ref.rename("w_ref").reset_index(), on=["h", "series"])
    df["rel_winkler_L2"] = df.winkler_L2 / df.w_ref
    summ = df.groupby(["h", "method"]).agg(coverage_L2=("coverage_L2", "mean"),
                                           rel_winkler_L2=("rel_winkler_L2", "median"),
                                           rmst_L1_resid=("rmst_L1_resid", "median"),
                                           fa_test_per_1000=("fa_test_per_1000", "median")).reset_index()
    summ["method"] = summ.method.map(LABEL)
    summ.to_csv(OUT / "snaive_summary.csv", index=False)
    cmp = []
    for h in (1, 24):
        piv = {k: df[df.h == h].pivot(index="series", columns="method", values=k)
               for k in ("rmst_L1_resid", "rel_winkler_L2")}
        for b in BASELINES:
            for k, p in piv.items():
                d = (p["dtcwr_aci"] - p[b]).dropna().to_numpy()
                lo, hi = boot_ci(d)
                cmp.append({"h": h, "baseline": LABEL[b], "outcome": k, "median_diff": float(np.median(d)),
                            "ci_lo": lo, "ci_hi": hi,
                            "p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0})
    cmp = pd.DataFrame(cmp)
    cmp.to_csv(OUT / "snaive_vs_baselines.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n### POST HOC: seasonal-naive forecaster, cohort 1 (medians over 50 clients)")
    print(summ.round(4).to_string(index=False))
    print("\n### DTCW-R+ACI minus baseline (negative = lower RMST / lower Winkler); exploratory, no correction")
    print(cmp.round(4).to_string(index=False))
    print(f"\nH: {json.dumps(Hs)}\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
