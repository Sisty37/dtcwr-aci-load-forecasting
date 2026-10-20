"""
POST-HOC sensitivity analysis (exploratory; NOT part of the pre-registration; decision D13).

Question: does the 2x2 signal effect (H1) change when the magnitude detector's false-alarm rate
is matched EXACTLY to the miss detector's? In the pre-registered run the magnitude H came from a
coarse grid (achieved 0.25-0.49 per 1,000 steps vs about 0.5 for the miss cells), a limitation
declared before the test.

Steps (original cohort):
  1. On VALIDATION scores only, recalibrate the magnitude H of the selected DTCW-R configuration
     by bisection on a fine grid (calib_v2.MAG_GRID) to the same 0.5 / 1,000 target.
  2. Re-run the two magnitude cells (D anchored, B reset) on the TEST streams with that H.
  3. Recompute the 2x2 effects with the new D, B and the pre-registered A, R.
The paper reports this next to the pre-registered result, labelled post hoc.

Run from the project root:
    python src/posthoc_exact_fa.py
Output: results/posthoc/exact_fa_H.json, results/posthoc/<ds>_h<h>.csv.gz,
        results/posthoc/ablation_exact_fa_<ds>_h<h>.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as an  # noqa: E402
import calib as cb  # noqa: E402
import calib_v2 as v2  # noqa: E402
import run_prereg_selection as ps  # noqa: E402
import run_test as rt  # noqa: E402
import tune as tn  # noqa: E402
from run_calib_val import ALPHA, HORIZONS, load_forecasts  # noqa: E402

OUT = Path("results/posthoc")


def exact_H(ds, h, fz) -> dict:
    key = f"{ds}_h{h}"
    val = load_forecasts(ds, h)
    val = val[val.split == "val"]
    scores = [np.abs(g.y.to_numpy() - g.yhat.to_numpy()) for _, g in val.groupby("series", sort=True)]
    a = fz["abl"][key]
    pw = fz["choice"][key]["winner"] == "prewhiten"
    base = dict(a["config"])
    block = max(tn.PERIOD[ds], 2 * h)
    reps = 1 if len(scores) > 1 else 20
    r = ps.calibrate_H_bisect(scores, block, lambda H: cb.DTCWR(ALPHA, h, H=H, prewhiten=pw, **base),
                              grid=v2.MAG_GRID, reps_per_series=reps)
    return {"H_exact": r["H"], "fa_exact": r["fa_at_H"], "H_prereg": a["D_magnitude_anchored"]["H"],
            "fa_prereg": a["D_magnitude_anchored"].get("fa_at_H"),
            "fa_miss_A": a["A_miss_anchored"]["fa_at_H"], "fa_miss_R": a["R_miss_reset"]["fa_at_H"],
            "config": base, "prewhiten": pw}


def _rows_factory(Hx: dict):
    def method_rows(ds, h, fz):
        e = Hx[f"{ds}_h{h}"]
        base, pw, H = e["config"], e["prewhiten"], e["H_exact"]
        return {"dtcwr": lambda: cb.DTCWR(ALPHA, h, H=H, prewhiten=pw, **base),
                "abl_B_magnitude_reset": lambda: cb.DTCWR(ALPHA, h, H=H, prewhiten=pw,
                                                          window_rule="reset", **base)}
    return method_rows


def _init(ds, h, fz, Hx):
    rt._init(ds, h, fz)
    rt.method_rows = _rows_factory(Hx)          # only the two magnitude cells are re-run


def main(workers: int = max(1, (os.cpu_count() or 2) - 1)):
    OUT.mkdir(parents=True, exist_ok=True)
    fz = rt.load_frozen()
    hfile = OUT / "exact_fa_H.json"
    Hx = json.loads(hfile.read_text()) if hfile.exists() else {}
    for ds in ("elec2", "electricity"):
        for h in HORIZONS[ds]:
            key = f"{ds}_h{h}"
            if key not in Hx:
                t0 = time.time()
                Hx[key] = exact_H(ds, h, fz)
                hfile.write_text(json.dumps(Hx, indent=2))
                e = Hx[key]
                print(f"{key}: magnitude H {e['H_prereg']} (fa {e['fa_prereg']:.2f}) -> {e['H_exact']} "
                      f"(fa {e['fa_exact']:.2f}); miss cells fa {e['fa_miss_A']:.2f} / {e['fa_miss_R']:.2f}"
                      f"  [{time.time() - t0:.0f} s]", flush=True)
    for ds in ("elec2", "electricity"):
        for h in HORIZONS[ds]:
            key = f"{ds}_h{h}"
            out = OUT / f"{key}.csv.gz"
            if not out.exists():
                t0 = time.time()
                names = json.loads((Path("results/models") / f"{key}.json").read_text())["series"]
                rows = []
                with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                         initargs=(ds, h, fz, Hx)) as pool:
                    futs = [pool.submit(rt._work, n) for n in names]
                    for i, f in enumerate(as_completed(futs), 1):
                        rows += f.result()[1]
                        if i % 10 == 0 or i == len(names):
                            print(f"  {key}: {i}/{len(names)} clients  [{time.time() - t0:.0f} s]", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
            if ds != "electricity":
                continue
            new = pd.read_csv(out, low_memory=False)
            old = pd.read_csv(Path("results/test") / f"{key}.csv.gz", low_memory=False)
            comb = pd.concat([old[old.method.isin(["abl_A_miss_anchored", "abl_R_miss_reset"])], new])
            ab = an.ablation(comb)
            ab.to_csv(OUT / f"ablation_exact_fa_{key}.csv", index=False)
            print(f"\n### POST-HOC 2x2 at exactly matched false-alarm rate — {key}")
            print(ab[ab.outcome.isin(["rmst", "det_delay"])][["stratum", "outcome", "effect", "median", "ci_lo",
                                                                "ci_hi", "p", "p_holm_primary"]]
                  .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
