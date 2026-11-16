"""
Numbers requested by the pre-submission review (post hoc, descriptive; no new test runs).
Reads the frozen test outputs of both cohorts and prints:

  A. Clipping: how often an infinite interval was clipped to the largest known score,
     per method, horizon and cohort (L2 untouched test and L1 target-level shift streams).
  B. Detector alarms per 1,000 steps on the UNTOUCHED test streams (L2) for every
     detector-based method, plus the share of clients with no alarm at all.
  C. Detection delay and detection rate on target-level variance shifts (L1).

Run from the project root (main working copy, after combine_cohorts.py):
    python src/review_stats.py
In another cohort's working copy, name its test folder explicitly, e.g. cohort 3:
    python src/review_stats.py --roots c3=results/test
Output: results/review/review_stats.json and printed tables. Alarm medians carry a 95%
bootstrap CI over clients (2,000 resamples, seed 0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOTS = {"c1": Path("results/test"), "c2": Path("results/cohort2/test")}
HS = (1, 24)
OUT = Path("results/review")
DET = ["dtcwr", "dtcwr_aci", "abl_B_magnitude_reset", "abl_A_miss_anchored", "abl_R_miss_reset",
       "driftocp"]


def n_test_targets(h: int) -> int:
    f = pd.read_parquet(Path("results/forecasts") / f"electricity_h{h}.parquet")
    one = f[f.series == f.series.iloc[0]]
    return int((one.split == "test").sum())


def boot_median_ci(v: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, (B, len(v)), replace=True), axis=1)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def main(roots: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    res = {"clipping": [], "alarms_L2": [], "delay_variance": []}
    for h in HS:
        nt = n_test_targets(h)
        for tag, root in roots.items():
            df = pd.read_csv(root / f"electricity_h{h}.csv.gz", low_memory=False)
            for layer in ("L2", "L1_target"):
                d = df[df.layer == layer]
                g = d.groupby("method")["n_clipped_test"]
                for m, v in g:
                    res["clipping"].append({
                        "h": h, "cohort": tag, "layer": layer, "method": m,
                        "pct_targets_clipped": float(100 * v.sum() / (len(v) * nt)),
                        "pct_streams_with_clipping": float(100 * (v > 0).mean()),
                        "max_clipped_in_a_stream": int(v.max())})
            l2 = df[(df.layer == "L2") & df.method.isin(DET)]
            for m, v in l2.groupby("method")["alarms_per_1000_test"]:
                lo, hi = boot_median_ci(v.to_numpy(float))
                res["alarms_L2"].append({
                    "h": h, "cohort": tag, "method": m, "median": float(v.median()),
                    "ci_lo": lo, "ci_hi": hi,
                    "q25": float(v.quantile(.25)), "q75": float(v.quantile(.75)),
                    "mean": float(v.mean()), "pct_clients_no_alarm": float(100 * (v == 0).mean())})
            l1 = df[(df.layer == "L1_target") & (df.kind == "variance") & df.method.isin(DET)]
            for m, v in l1.groupby("method"):
                det = v["detected"].astype(str).str.lower().isin(["true", "1", "1.0"])
                res["delay_variance"].append({
                    "h": h, "cohort": tag, "method": m,
                    "median_delay_detected": float(v.loc[det, "det_delay"].median()) if det.any() else None,
                    "detect_rate": float(det.mean())})
    for k, rows in res.items():
        t = pd.DataFrame(rows)
        t.to_csv(OUT / f"{k}.csv", index=False)
        print(f"\n### {k}")
        if k == "clipping":
            t = t[t.pct_targets_clipped > 0]
            print("(rows with zero clipping omitted)" if len(t) else "no clipping in any method")
        print(t.round(3).to_string(index=False))
    (OUT / "review_stats.json").write_text(json.dumps(res, indent=1))
    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=None,
                    help="tag=folder pairs, e.g. c3=results/test (default: cohorts 1 and 2)")
    a = ap.parse_args()
    roots = ROOTS if a.roots is None else {k: Path(v) for k, v in (x.split("=", 1) for x in a.roots)}
    main(roots)
