"""
Choose between the plain and the AR(1)-prewhitened DTCW-R detector (plan amendment, v5 Sec. 4.2):
at the SAME bootstrap-calibrated false-alarm rate (results/detector_calibration.json), the variant
with the lower median detection delay on residual-level injections into VALIDATION streams wins.
Ties (difference < 5% of the pooled median) go to the simpler plain detector.
The test segment stays sealed.

Run from the project root (after src/run_detector_calibration.py):
    python src/run_detector_choice.py
Output: results/detector_choice.csv  (one row per series x position x setting x variant)
        results/detector_choice.json (decision per dataset/horizon)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib as cb  # noqa: E402
import inject as ij  # noqa: E402
from run_calib_val import ALPHA, HORIZONS, LOCAL_W, load_forecasts  # noqa: E402
from run_detector_calibration import variants  # noqa: E402


def run_detector(scores: np.ndarray, h: int, H: float, kw: dict) -> list[tuple]:
    m = cb.DTCWR(ALPHA, h, H=H, **kw)
    m.setup(scores[:cb.W0])
    for t, v in enumerate(scores):
        m.observe(t, float(v))
    return m.alarms


def main(cal_path=Path("results/detector_calibration.json"), out_dir=Path("results")):
    cal = json.loads(cal_path.read_text())
    rows, decisions = [], {}
    for ds in ("electricity", "elec2"):
        week = LOCAL_W[ds]
        for h in HORIZONS[ds]:
            t0 = time.time()
            val = load_forecasts(ds, h)
            val = val[val["split"] == "val"]
            for sid, g in val.groupby("series", sort=True):
                r = g["y"].to_numpy() - g["yhat"].to_numpy()
                for tau in ij.positions(len(r), h, follow=ij.TTR_CAP):
                    for kind, size in ij.SETTINGS:
                        sh = ij.Shift(kind, size, tau, week)
                        s = np.abs(ij.inject_residuals(r, sh))
                        for vname, kw in variants(ds, h).items():
                            H = cal[f"{ds}_h{h}_{vname}"]["H"]
                            alarms = run_detector(s, h, H, kw)
                            times = [a[0] for a in alarms]
                            delay, det = ij.detection_delay(times, tau)
                            first = next((a for a in alarms if a[0] >= tau), None)
                            rows.append(dict(dataset=ds, h=h, series=sid, tau=tau, kind=kind,
                                             size=size, variant=vname, H=H, delay=delay,
                                             detected=det,
                                             anchor_error=(abs(first[2] - tau) if first else np.nan),
                                             false_alarms_before=sum(t < tau for t in times)))
            d = pd.DataFrame([x for x in rows if x["dataset"] == ds and x["h"] == h])
            summ = d.groupby(["variant", "kind"]).agg(
                median_delay=("delay", "median"), detect_rate=("detected", "mean"),
                median_anchor_err=("anchor_error", "median")).round(3)
            pooled = d.groupby("variant")["delay"].median()
            plain, pw = float(pooled["plain"]), float(pooled["prewhiten"])
            winner = "prewhiten" if pw < plain - 0.05 * max(plain, pw, 1) else "plain"
            decisions[f"{ds}_h{h}"] = {"pooled_median_delay": {"plain": plain, "prewhiten": pw},
                                       "winner": winner,
                                       "H": {v: cal[f"{ds}_h{h}_{v}"]["H"] for v in ("plain", "prewhiten")}}
            print(f"\n=== {ds} h={h}  (VALIDATION injections; {d['series'].nunique()} series x "
                  f"{d['tau'].nunique()} positions x {len(ij.SETTINGS)} settings)  [{time.time() - t0:.0f} s]")
            print(summ.to_string())
            print(f"pooled median delay: plain={plain:.0f}  prewhiten={pw:.0f}  ->  winner: {winner}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "detector_choice.csv", index=False)
    (out_dir / "detector_choice.json").write_text(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()
