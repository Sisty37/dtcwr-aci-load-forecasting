"""
THE TEST RUN (plan v5 Sections 6.3, 6.4, 8, 9). Runs once, after the pre-registration tag
prereg-v1. It only produces per-run metrics; the statistics live in src/analysis.py.

Refuses to start unless HEAD descends from the tag prereg-v1 and the working tree is clean
(override only with --allow-dirty, which is recorded in the output and must be reported).

For every dataset x horizon x client, and for every method row (11 Table-B rows + the three
other 2x2 cells), it replays the val + test stream (methods warm up on validation; only TEST
targets are evaluated) in these layers:
    L2            untouched stream
    L1_target     target-level injection with the frozen LightGBM (PRIMARY), 7 settings x 5 positions
    L1_resid      residual-level injection (control panel), same settings and positions
All configurations come from the frozen files listed in PREREGISTRATION.md.

Run from the project root:
    python src/run_test.py                       # everything (resumable per client)
    python src/run_test.py --dataset elec2
Output: results/test/<ds>_h<h>/clients/<client>.csv   (one row per method x layer x setting x position)
        results/test/<ds>_h<h>.csv.gz                  (all clients)
        results/test/provenance.json                   (git HEAD, flags, timing)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cohort  # noqa: E402
import calib as cb  # noqa: E402
import inject as ij  # noqa: E402
import inject_target as it  # noqa: E402
import run_prereg_selection as ps  # noqa: E402
import tune as tn  # noqa: E402
from run_calib_val import ALPHA, HORIZONS, LOCAL_W, load_forecasts  # noqa: E402

OUT = Path("results/test")


# ----------------------------------------------------------------------------- frozen inputs
def load_frozen() -> dict:
    return {"sel": json.loads(Path("results/tuning/selected.json").read_text()),
            "ttr": json.loads(Path("results/prereg/ttr_sensitivity.json").read_text()),
            "abl": json.loads(Path("results/prereg/ablation_H.json").read_text()),
            "sig": json.loads(Path("results/prereg/sigma_eps.json").read_text()),
            "choice": json.loads(Path("results/detector_choice.json").read_text())}


def method_rows(ds: str, h: int, fz: dict) -> dict:
    """name -> zero-argument factory. 11 Table-B rows + 2x2 cells B, A, R (D = dtcwr)."""
    key = f"{ds}_h{h}"
    sel = fz["sel"][key]
    pw = fz["choice"][key]["winner"] == "prewhiten"
    rows = {}
    for c in sel["fixed"]["configs"]:
        rows[f"fixed_{c['W']}"] = (lambda c=c: tn.make_method("fixed", c, h))
    for fam in ("aci", "rocp", "dtaci", "pid", "eci_cutoff", "driftocp", "dtcwr", "dtcwr_aci"):
        cfg = sel[fam]["config"]
        rows[fam] = (lambda fam=fam, cfg=cfg: tn.make_method(fam, cfg, h, {"prewhiten": pw}))
    a = fz["abl"][key]
    base = dict(a["config"])
    rows["abl_B_magnitude_reset"] = lambda: cb.DTCWR(ALPHA, h, H=a["B_magnitude_reset"]["H"], prewhiten=pw,
                                                     window_rule="reset", **base)
    rows["abl_A_miss_anchored"] = lambda: cb.DTCWR(ALPHA, h, H=a["A_miss_anchored"]["H"], signal="miss",
                                                   window_rule="anchored", **base)
    rows["abl_R_miss_reset"] = lambda: cb.DTCWR(ALPHA, h, H=a["R_miss_reset"]["H"], signal="miss",
                                                window_rule="reset", **base)
    return rows


def frozen_cell(ds: str, fz: dict) -> dict:
    c = fz["ttr"]["chosen"]
    w = c["w_h"] * tn.scale(ds)
    return {"w": w, "delta": c["delta"], "L": int(round(c["L_over_w"] * w))}


# ----------------------------------------------------------------------------- one stream
def eval_stream(make, y, yhat, n_val: int, ds: str, cells: list[dict], cell: dict,
                tau: int | None = None) -> dict:
    m = make()
    t0 = time.perf_counter()
    out = cb.replay(m, y, yhat, clip_inf=True)
    ms = 1000 * (time.perf_counter() - t0) / len(y)
    n = len(y)
    test = np.zeros(n, bool)
    test[n_val:] = True
    met = cb.interval_metrics(y, yhat, out["q"], test, ALPHA, LOCAL_W[ds])
    met["n_clipped_test"] = int(sum(math.isinf(m.q_issued.get(t, 0.0)) for t in range(n_val, n)))
    met["ms_per_step"] = ms
    alarms = [a for a in getattr(m, "alarms", [])]
    if hasattr(m, "alarms"):
        met["alarms_per_1000_test"] = 1000 * sum(a[0] >= n_val for a in alarms) / (n - n_val)
    cov = out["score"] <= out["q"]                       # every target from h on has an interval
    if tau is not None:
        ttr, ok = ij.time_to_recover(cov, tau, ALPHA, cell["w"], cell["delta"], cell["L"])
        met.update(ttr=ttr, recovered=ok)
        pre = float(cov[tau - cell["w"]:tau].mean())     # own pre-shift local coverage (secondary)
        ttr_r, ok_r = ij.time_to_recover(cov, tau, 1 - pre, cell["w"], cell["delta"], cell["L"])
        met.update(ttr_rel=ttr_r, recovered_rel=ok_r, pre_shift_local_cov=pre)
        seg = slice(tau, min(n, tau + ij.TTR_CAP))
        met["post_coverage"] = float(cov[seg].mean())
        for c in cells:
            t_c, ok_c = ij.time_to_recover(cov, tau, ALPHA, c["w"], c["delta"], c["L"])
            tag = f"w{c['w_h']}_d{c['delta']}_L{c['L_over_w']}"
            met[f"ttr__{tag}"], met[f"rec__{tag}"] = t_c, ok_c
        if hasattr(m, "alarms"):
            d, det = ij.detection_delay([a[0] for a in alarms], tau)
            first = next((a for a in alarms if a[0] >= tau), None)
            met.update(det_delay=d, detected=det,
                       anchor_err=(abs(first[2] - tau) if first is not None else np.nan))
    return met


# ----------------------------------------------------------------------------- one client
def run_client(ds: str, h: int, name: str, y: np.ndarray, yhat: np.ndarray, n_val: int,
               inj, fz: dict) -> list[dict]:
    key = f"{ds}_h{h}"
    week = LOCAL_W[ds]
    n_test = len(y) - n_val
    cell = frozen_cell(ds, fz)
    cells = ps.ttr_cells(ds)
    taus = ij.test_positions(n_val, n_test, L=cell["L"])
    methods = method_rows(ds, h, fz)
    r0 = y - yhat
    sig = float(np.std(r0[:cb.W0]))
    base = (y, yhat)
    # build all injected streams once (shared by every method)
    streams = [("L2", "none", 0.0, None, y, yhat, {})]
    for tau in taus:
        for kind, size in ij.SETTINGS:
            se = fz["sig"][key][name][str(size)]["m"] * sig if kind == "variance" else None
            _, y_t, yh_t, s_h = inj.inject(name, kind, size, tau, week, test=True, sigma_eps=se, base=base)
            assert np.isclose(s_h, sig)
            ach = it.achieved(kind, size, r0, y_t - yh_t, tau, sig, week)
            streams.append(("L1_target", kind, size, tau, y_t, yh_t, ach))
            r1 = ij.inject_residuals(r0, ij.Shift(kind, size, tau, week))
            streams.append(("L1_resid", kind, size, tau, yhat + r1, yhat, {}))
    rows = []
    for layer, kind, size, tau, ys, yh, ach in streams:
        for mname, make in methods.items():
            met = eval_stream(make, ys, yh, n_val, ds, cells, cell, tau)
            rows.append({"dataset": ds, "h": h, "series": name, "method": mname, "layer": layer,
                         "kind": kind, "size": size, "tau": tau,
                         "pos_frac": (None if tau is None else round((tau - n_val) / n_test, 2)),
                         **ach, **met})
    return rows


# ----------------------------------------------------------------------------- workers
_W: dict = {}


def _init(ds, h, fz):
    _W["inj"] = it.TargetInjector(ds, h)
    _W["fz"] = fz
    df = load_forecasts(ds, h)
    df = df[df["split"].isin(["val", "test"])]
    _W["streams"] = {s: (g["y"].to_numpy(float), g["yhat"].to_numpy(float), int((g["split"] == "val").sum()))
                     for s, g in df.groupby("series", sort=True)}
    _W["ds"], _W["h"] = ds, h


def _work(name):
    y, yhat, n_val = _W["streams"][name]
    inj = _W["inj"]
    tg = inj.stream_targets(test=True)
    if len(tg) != len(y):
        raise RuntimeError(f"{name}: stream length {len(y)} != injector stream {len(tg)}")
    return name, run_client(_W["ds"], _W["h"], name, y, yhat, n_val, inj, _W["fz"])


def check_provenance(allow_dirty: bool) -> dict:
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True)
    tag = cohort.PREREG_TAG
    anc = git("merge-base", "--is-ancestor", tag, "HEAD").returncode == 0
    dirty = [l for l in git("status", "--porcelain").stdout.splitlines()
             if l.strip() and "results/test" not in l]
    if not anc:
        raise SystemExit(f"HEAD does not descend from tag {tag}: refusing to open the test segment.")
    if dirty and not allow_dirty:
        raise SystemExit("Working tree not clean (commit first, or --allow-dirty and report it):\n"
                         + "\n".join(dirty))
    return {"head": git("rev-parse", "HEAD").stdout.strip(),
            "prereg_tag": tag, "prereg_commit": git("rev-parse", tag + "^{commit}").stdout.strip(),
            "dirty": dirty, "allow_dirty": allow_dirty}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("electricity", "elec2"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    prov = check_provenance(a.allow_dirty)
    fz = load_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    pfile = OUT / "provenance.json"
    log = json.loads(pfile.read_text()) if pfile.exists() else {"runs": []}
    for ds in [d for d in ("elec2", "electricity") if d in cohort.DATASETS]:
        if a.dataset and ds != a.dataset:
            continue
        for h in HORIZONS[ds]:
            t0 = time.time()
            key = f"{ds}_h{h}"
            cdir = OUT / key / "clients"
            cdir.mkdir(parents=True, exist_ok=True)
            names = json.loads((Path("results/models") / f"{key}.json").read_text())["series"]
            todo = [n for n in names if not (cdir / f"{n}.csv").exists()]
            print(f"=== {key}: {len(names)} clients, {len(todo)} to run", flush=True)
            if todo:
                with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                                         initargs=(ds, h, fz)) as pool:
                    futs = [pool.submit(_work, n) for n in todo]
                    for i, f in enumerate(as_completed(futs), 1):
                        name, rows = f.result()
                        pd.DataFrame(rows).to_csv(cdir / f"{name}.csv", index=False)
                        print(f"  {i}/{len(todo)} {name}  [{time.time() - t0:.0f} s]", flush=True)
            allr = pd.concat([pd.read_csv(cdir / f"{n}.csv") for n in names], ignore_index=True)
            allr.to_csv(OUT / f"{key}.csv.gz", index=False)
            log["runs"].append({"stream": key, **prov, "seconds": round(time.time() - t0),
                                "rows": len(allr)})
            pfile.write_text(json.dumps(log, indent=2))
            print(f"=== {key}: {len(allr)} rows -> {OUT / (key + '.csv.gz')}", flush=True)


if __name__ == "__main__":
    main()
