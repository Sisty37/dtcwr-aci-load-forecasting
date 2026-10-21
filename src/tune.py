"""
Validation tuning (plan v5 Sections 4 and 7.2, step 7): 16 configurations per method family,
the same budget, the same streams and the same pre-specified selection rule for every family.
The TEST segment stays sealed.

Selection rule (decision D10, fixed before any tuning run):
  Streams per client: its VALIDATION stream un-shifted, plus the 7 residual-level injections
  (inject.SETTINGS) at the 3 positions of inject.positions  ->  22 streams.
  Per stream: coverage and Winkler score on the evaluated targets (tau >= W0 + h);
  Winkler is divided by the client's warm-up robust score scale so every client weighs the same.
  An infinite interval (ACI-type methods when alpha_t <= 0) is replaced by the largest score
  known at its origin, for EVERY method (the repo's "ACI (clipped)" rule); n_clipped is reported.
  Config score = mean over clients of the mean over its 22 streams of the normalised Winkler.
  Eligible  = mean coverage on the UN-SHIFTED streams within 0.90 +/- 0.02.
  Selected  = eligible config with the lowest score; if none is eligible, the config with the
              smallest un-shifted coverage error (flagged in the output).

Families and grids (16 configs each; time constants are in hours and doubled for the
half-hourly Elec2 so they keep their calendar meaning):
  fixed      168 / 720 / 2160 h                        (3, a priori, untuned)
  aci        gamma: 16 log-spaced values in [1e-4, 0.1], all-past window
  rocp       window m = c T^(2/3), 16 c in [0.1, 4], T = validation length
  dtaci      gamma grid x {0.25, 0.5, 1, 2}  x  I in {100, 250, 500, 1000}
  pid        lr in {1, 0.5, 0.1, 0.05}  x  KI in {0, 2.5, 10, 40} (Csat 5; KI = 0 is the
             plain quantile tracker), KI rescaled to the score scale (D10)
  eci_cutoff lr in {1, 0.5, 0.1, 0.05}  x  cutoff h in {0.25, 0.5, 1, 2}, c x scale = 0.0185
  driftocp   sigma in {0.5, 1, 2, 4}  x  minimum block in {5, 10, 20, 40}
  dtcwr      16 of the 54 points W_S x W_L x k x c (plan grid without H), drawn with
             default_rng(2027); H of EACH drawn point calibrated to 0.5 false alarms per 1,000
             steps by block bootstrap (D6); detector variant fixed by D8.
  dtcwr_aci  the selected dtcwr config  x  the 16 ACI gammas (second stage)

Run from the project root (resumable: finished families are skipped):
    python src/tune.py                          # everything
    python src/tune.py --dataset elec2 --h 1    # one dataset / horizon
    python src/tune.py --workers 4
Output: results/tuning/<ds>_h<h>/<family>.csv   one row per (config, client)
        results/tuning/<ds>_h<h>/dtcwr_H.json   calibrated H per drawn dtcwr config
        results/tuning/selected.json            selected config per family / dataset / horizon
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cohort  # noqa: E402
import baselines as bl  # noqa: E402
import calib as cb  # noqa: E402
import detector_calibration as dc  # noqa: E402
import inject as ij  # noqa: E402
from run_calib_val import ALPHA, HORIZONS, LOCAL_W, load_forecasts  # noqa: E402

OUT = Path("results/tuning")
COVER_TOL = 0.02
SEED = 2027
FAMILIES = ("fixed", "aci", "rocp", "dtaci", "pid", "eci_cutoff", "driftocp", "dtcwr", "dtcwr_aci")
ACI_GAMMAS = tuple(float(f"{g:.3g}") for g in np.geomspace(1e-4, 0.1, 16))
DTCWR_GRID = dict(W_S=(48, 96, 168), W_L=(336, 720, 1440), k=(0.25, 0.5), c=(0, 24, 72))
PERIOD = {"electricity": 24, "elec2": 48}


def scale(ds: str) -> int:
    """Steps per hour: time constants in hours are multiplied by this."""
    return PERIOD[ds] // 24


def dtcwr_draw() -> list[dict]:
    keys = list(DTCWR_GRID)
    pts = list(itertools.product(*DTCWR_GRID.values()))
    idx = np.random.default_rng(SEED).choice(len(pts), size=16, replace=False)
    return [dict(zip(keys, pts[i])) for i in sorted(idx)]


def grids(ds: str, h: int, T_val: int) -> dict[str, list[dict]]:
    s = scale(ds)
    g = {
        "fixed": [{"W": w * s} for w in (168, 720, 2160)],
        "aci": [{"gamma": x} for x in ACI_GAMMAS],
        "rocp": [{"W": w} for w in bl.rocp_window_grid(T_val)],
        "dtaci": [{"gscale": f, "I": I} for f in (0.25, 0.5, 1, 2) for I in (100, 250, 500, 1000)],
        "pid": [{"lr": lr, "KI": KI} for lr in (1, 0.5, 0.1, 0.05) for KI in (0, 2.5, 10, 40)],
        "eci_cutoff": [{"lr": lr, "h_cut": c} for lr in (1, 0.5, 0.1, 0.05) for c in (0.25, 0.5, 1, 2)],
        "driftocp": [{"sigma": sg, "min_len": m} for sg in (0.5, 1, 2, 4) for m in (5, 10, 20, 40)],
        "dtcwr": [{k: (v * s if k in ("W_S", "W_L", "c") else v) for k, v in p.items()}
                  for p in dtcwr_draw()],
        # D17: added after the tests were opened; tuned with the same budget and rule, not in FAMILIES
        "saocp": [{"lifetime": g_, "scale_mult": m_} for g_ in (4, 8, 16, 32) for m_ in (0.25, 0.5, 1, 2)],
    }
    return g


def make_method(family: str, cfg: dict, h: int, extra: dict | None = None) -> cb.Method:
    extra = extra or {}
    if family == "fixed":
        return cb.FixedWindow(ALPHA, h, cfg["W"])
    if family == "aci":
        return cb.ACI(ALPHA, h, cfg["gamma"])
    if family == "rocp":
        return bl.ROCP(ALPHA, h, W=cfg["W"])
    if family == "dtaci":
        return bl.DtACI(ALPHA, h, gammas=tuple(cfg["gscale"] * g for g in bl.DTACI_GAMMAS), I=cfg["I"])
    if family == "pid":
        return bl.ConformalPID(ALPHA, h, cfg["lr"], Csat=5.0, KI=cfg["KI"], KI_scaled=True)
    if family == "eci_cutoff":
        return bl.ECICutoff(ALPHA, h, cfg["lr"], c_rel=bl.ECI_C_REL, h_cut=cfg["h_cut"])
    if family == "driftocp":
        return bl.DriftOCP(ALPHA, h, sigma=cfg["sigma"], min_len=cfg["min_len"])
    if family == "saocp":
        return bl.SAOCP(ALPHA, h, lifetime=cfg["lifetime"], scale_mult=cfg["scale_mult"])
    if family in ("dtcwr", "dtcwr_aci"):
        kw = {k: cfg[k] for k in ("W_S", "W_L", "k", "c", "H")}
        return cb.DTCWR(ALPHA, h, prewhiten=extra.get("prewhiten", False),
                        aci_gamma=cfg.get("gamma"), **kw)
    raise ValueError(family)


# ----------------------------------------------------------------------------- streams + scoring
def streams(y: np.ndarray, yhat: np.ndarray, h: int, week: int) -> list[tuple[str, np.ndarray]]:
    """Un-shifted stream plus the 21 residual-level injections (y' = yhat + r')."""
    r = y - yhat
    out = [("none", y)]
    for tau in ij.positions(len(r), h, follow=ij.TTR_CAP):
        for kind, size in ij.SETTINGS:
            out.append((f"{kind}_{size}@{tau}", yhat + ij.inject_residuals(r, ij.Shift(kind, size, tau, week))))
    return out


def stream_metrics(method: cb.Method, y, yhat) -> tuple[float, float, int]:
    out = cb.replay(method, y, yhat, clip_inf=True)
    ev = out["evaluated"]
    s, q = out["score"][ev], out["q"][ev]
    cov = float((s <= q).mean())
    with np.errstate(invalid="ignore"):
        wink = np.where(np.isfinite(q), 2 * q + (2 / ALPHA) * np.maximum(s - q, 0), np.inf)
    return cov, float(wink.mean()), out["n_clipped"]


def eval_config_series(family, cfg, h, y, yhat, week, extra=None) -> dict:
    sig = bl.robust_scale(np.abs(y - yhat)[:cb.W0])
    cov0, rows, n_clip = math.nan, [], 0
    for label, ys in streams(y, yhat, h, week):
        cov, w, nc = stream_metrics(make_method(family, cfg, h, extra), ys, yhat)
        rows.append((label, cov, w / sig))
        n_clip += nc
        if label == "none":
            cov0 = cov
    nw = np.array([r[2] for r in rows])
    return {"cov_unshifted": cov0, "nwink_unshifted": rows[0][2],
            "nwink_shifted": float(nw[1:].mean()), "nwink_all": float(nw.mean()),
            "cov_shifted": float(np.mean([r[1] for r in rows[1:]])), "n_streams": len(rows),
            "n_clipped": n_clip}


def select(df: pd.DataFrame) -> dict:
    """D10 rule on one family's (config x client) table."""
    agg = df.groupby("config").agg(cov=("cov_unshifted", "mean"), score=("nwink_all", "mean"),
                                   nwink_unshifted=("nwink_unshifted", "mean"),
                                   cov_shifted=("cov_shifted", "mean")).reset_index()
    agg["eligible"] = (agg["cov"] - (1 - ALPHA)).abs() <= COVER_TOL
    if agg["eligible"].any():
        best = agg[agg["eligible"]].sort_values(["score", "config"]).iloc[0]
        flag = ""
    else:
        best = agg.assign(err=(agg["cov"] - (1 - ALPHA)).abs()).sort_values(["err", "config"]).iloc[0]
        flag = "no config within coverage tolerance"
    return {"config": json.loads(best["config"]), "score": float(best["score"]),
            "cov_unshifted": float(best["cov"]), "flag": flag,
            "n_eligible": int(agg["eligible"].sum())}


# ----------------------------------------------------------------------------- workers
_DATA: dict = {}


def _init(ds: str, h: int):
    val = load_forecasts(ds, h)
    val = val[val["split"] == "val"]
    _DATA["series"] = {sid: (g["y"].to_numpy(float), g["yhat"].to_numpy(float))
                       for sid, g in val.groupby("series", sort=True)}
    _DATA["week"], _DATA["h"] = LOCAL_W[ds], h


def _work(args):
    family, cfg, sid, extra = args
    y, yhat = _DATA["series"][sid]
    res = eval_config_series(family, cfg, _DATA["h"], y, yhat, _DATA["week"], extra)
    return {"family": family, "config": json.dumps(cfg, sort_keys=True), "series": sid, **res}


def _calib_work(args):
    cfg, prewhiten, block, reps = args
    scores = [np.abs(y - yhat) for y, yhat in _DATA["series"].values()]
    h = _DATA["h"]
    kw = {k: cfg[k] for k in ("W_S", "W_L", "k", "c")}
    return dc.calibrate_H(scores, block, lambda H: cb.DTCWR(ALPHA, h, H=H, prewhiten=prewhiten, **kw),
                          reps_per_series=reps)


def run_family(pool, family, configs, sids, out_csv, extra=None):
    if out_csv.exists():
        print(f"  {family}: done earlier -> {out_csv}")
        return pd.read_csv(out_csv)
    t0 = time.time()
    tasks = [(family, cfg, sid, extra) for cfg in configs for sid in sids]
    rows = list(pool.map(_work, tasks, chunksize=max(1, len(tasks) // (8 * (os.cpu_count() or 2)))))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  {family}: {len(configs)} configs x {len(sids)} clients  [{time.time() - t0:.0f} s]")
    return df


def tune(ds: str, h: int, workers: int, sel: dict, choice: dict):
    d = OUT / f"{ds}_h{h}"
    d.mkdir(parents=True, exist_ok=True)
    val = load_forecasts(ds, h)
    val = val[val["split"] == "val"]
    sids = sorted(val["series"].unique())
    T_val = int(val.groupby("series").size().min())
    g = grids(ds, h, T_val)
    prewhiten = choice[f"{ds}_h{h}"]["winner"] == "prewhiten"
    extra = {"prewhiten": prewhiten}
    key = f"{ds}_h{h}"
    sel.setdefault(key, {})
    print(f"\n=== {ds} h={h}: {len(sids)} clients, validation length {T_val}, detector "
          f"{'prewhiten' if prewhiten else 'plain'} (D8)")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(ds, h)) as pool:
        # H for every drawn DTCW-R point (matched false-alarm rate, D6)
        hfile = d / "dtcwr_H.json"
        if hfile.exists():
            Hs = json.loads(hfile.read_text())
        else:
            t0 = time.time()
            block = max(PERIOD[ds], 2 * h)
            reps = 1 if len(sids) > 1 else 20
            cals = list(pool.map(_calib_work, [(c, prewhiten, block, reps) for c in g["dtcwr"]]))
            Hs = [{"config": c, "H": cal["H"], "fa_at_H": cal["fa_at_H"],
                   "warning": cal.get("warning", "")} for c, cal in zip(g["dtcwr"], cals)]
            hfile.write_text(json.dumps(Hs, indent=2))
            print(f"  dtcwr H calibration: {len(Hs)} configs  [{time.time() - t0:.0f} s]")
        g["dtcwr"] = [dict(x["config"], H=x["H"]) for x in Hs]
        for fam in FAMILIES:
            if fam == "dtcwr_aci":
                base = sel[key]["dtcwr"]["config"]
                configs = [dict(base, gamma=x) for x in ACI_GAMMAS]
            else:
                configs = g[fam]
            df = run_family(pool, fam, configs, sids, d / f"{fam}.csv",
                            extra if fam.startswith("dtcwr") else None)
            sel[key][fam] = select(df) if fam != "fixed" else {
                "configs": [json.loads(c) for c in sorted(df["config"].unique())]}
            if fam != "fixed":
                s = sel[key][fam]
                print(f"    selected {s['config']}  score {s['score']:.3f}  cov {s['cov_unshifted']:.3f}"
                      f"  eligible {s['n_eligible']}/16 {s['flag']}")
            (OUT / "selected.json").write_text(json.dumps(sel, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("electricity", "elec2"))
    ap.add_argument("--h", type=int)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "selected.json"
    sel = json.loads(sp.read_text()) if sp.exists() else {}
    choice = json.loads(Path("results/detector_choice.json").read_text())
    for ds in [d for d in ("elec2", "electricity") if d in cohort.DATASETS]:   # small one first
        if a.dataset and ds != a.dataset:
            continue
        for h in HORIZONS[ds]:
            if a.h and h != a.h:
                continue
            tune(ds, h, a.workers, sel, choice)
    print(f"\nSelected configurations -> {sp}")


if __name__ == "__main__":
    main()
