"""
Calibrate sigma_eps of the target-level VARIANCE injection on VALIDATION data (plan v5
Section 6.3, decision D12), per dataset x horizon x client x kappa, with the frozen LightGBM.
Also checks that the frozen model reproduces the saved validation forecasts (sanity gate).
The test segment stays sealed.

Run from the project root:
    python src/run_sigma_calibration.py
Output: results/prereg/sigma_eps.json   {ds_h: {client: {kappa: {m, sigma_hat, achieved_ratio, ...}}}}
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inject_target as it  # noqa: E402
from run_calib_val import HORIZONS, LOCAL_W, load_forecasts  # noqa: E402

KAPPAS = (1.5, 2.0, 3.0)
OUT = Path("results/prereg/sigma_eps.json")
_I: dict = {}


def _init(ds, h):
    _I["inj"] = it.TargetInjector(ds, h)
    _I["week"] = LOCAL_W[ds]
    val = load_forecasts(ds, h)
    _I["saved"] = {s: g["yhat"].to_numpy(float) for s, g in val[val["split"] == "val"].groupby("series")}


def _work(name):
    inj = _I["inj"]
    tg = inj.stream_targets(False)
    yhat = inj.forecast(name, inj.y(name), tg)
    saved = _I["saved"][name]
    repro = float(np.max(np.abs(yhat - saved)) / (np.mean(np.abs(saved)) or 1.0)) if len(saved) == len(yhat) else np.inf
    res = {str(k): it.calibrate_sigma_eps(inj, name, k, _I["week"]) for k in KAPPAS}
    return name, repro, res


def main(workers: int = max(1, (os.cpu_count() or 2) - 1)):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for ds in ("elec2", "electricity"):
        for h in HORIZONS[ds]:
            key = f"{ds}_h{h}"
            if key in out:
                print(f"{key}: done earlier")
                continue
            t0 = time.time()
            names = json.loads((Path("results/models") / f"{key}.json").read_text())["series"]
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(ds, h)) as pool:
                rows = list(pool.map(_work, names))
            worst = max(r[1] for r in rows)
            if worst > 1e-4:
                raise RuntimeError(f"{key}: frozen model does not reproduce the saved forecasts "
                                   f"(max relative error {worst:.2e})")
            out[key] = {n: res for n, _, res in rows}
            OUT.write_text(json.dumps(out, indent=2))
            for k in KAPPAS:
                ms = [r[2][str(k)]["m"] for r in rows]
                ok = np.mean([r[2][str(k)]["within_tol"] for r in rows])
                ach = [r[2][str(k)]["achieved_ratio"] for r in rows]
                print(f"{key} kappa={k}: sigma_eps/sigma_hat median {np.median(ms):.3f} "
                      f"[IQR {np.quantile(ms, .25):.3f}-{np.quantile(ms, .75):.3f}]  achieved ratio "
                      f"median {np.median(ach):.3f} (target {k * k:.2f})  within 2%: {100 * ok:.0f}%")
            print(f"{key}: forecasts reproduced (max rel. error {worst:.1e})  [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
