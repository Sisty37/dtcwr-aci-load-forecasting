"""
Calibrate H (plain and AR(1)-prewhitened detector) on VALIDATION data, then compare
DTCW-R variants on validation. Test segment stays sealed.

Run from the project root:
    python src/run_detector_calibration.py
Output: results/detector_calibration.json   chosen H, bootstrap false-alarm curve per variant
        printed table per dataset/horizon
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cohort  # noqa: E402
import calib as cb  # noqa: E402
import detector_calibration as dc  # noqa: E402
from run_calib_val import ALPHA, HORIZONS, LOCAL_W, load_forecasts  # noqa: E402

PERIOD = {"electricity": 24, "elec2": 48}


def variants(ds: str, h: int):
    W_L = LOCAL_W[ds] * 30 // 7
    base = dict(W_S=48, W_L=W_L, k=0.5, c=24)
    return {"plain": dict(base, prewhiten=False), "prewhiten": dict(base, prewhiten=True)}


def main(out=Path("results/detector_calibration.json")):
    report = {}
    pd.set_option("display.width", 170)
    for ds in [d for d in ("electricity", "elec2") if d in cohort.DATASETS]:
        for h in HORIZONS[ds]:
            t0 = time.time()
            val = load_forecasts(ds, h)
            val = val[val["split"] == "val"]
            groups = {sid: g for sid, g in val.groupby("series", sort=True)}
            scores = [np.abs(g["y"].to_numpy() - g["yhat"].to_numpy()) for g in groups.values()]
            block = max(PERIOD[ds], 2 * h)
            reps = 1 if len(scores) > 1 else 20
            rows = []
            for vname, kw in variants(ds, h).items():
                cal = dc.calibrate_H(scores, block, lambda H, kw=kw: cb.DTCWR(ALPHA, h, H=H, **kw),
                                     reps_per_series=reps)
                report[f"{ds}_h{h}_{vname}"] = {k: v for k, v in cal.items()}
                for aci in (None, 0.005):
                    mets = []
                    for sid, g in groups.items():
                        y, yhat = g["y"].to_numpy(), g["yhat"].to_numpy()
                        m = cb.DTCWR(ALPHA, h, H=cal["H"], aci_gamma=aci, **kw)
                        r = cb.replay(m, y, yhat)
                        met = cb.interval_metrics(y, yhat, r["q"], r["evaluated"], ALPHA, LOCAL_W[ds])
                        met["alarms_per_1000"] = 1000 * len(m.alarms) / max(1, int(r["evaluated"].sum()))
                        mets.append(met)
                    med = pd.DataFrame(mets).median(numeric_only=True)
                    rows.append({"variant": vname + ("+aci" if aci else ""), "H": cal["H"],
                                 "boot_fa": round(cal["fa_at_H"], 3),
                                 **{k: round(float(med[k]), 4) for k in
                                    ("alarms_per_1000", "coverage", "winkler", "mean_width",
                                     "worst_local_coverage", "longest_under_run")}})
            print(f"\n=== {ds} h={h}  (VALIDATION, median over {len(scores)} series; block={block})"
                  f"  [{time.time() - t0:.0f} s]")
            print(pd.DataFrame(rows).to_string(index=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
