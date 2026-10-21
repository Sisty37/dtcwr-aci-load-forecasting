"""
Decision D18: evaluate predictions S1-S4 of PREREG_C3_PREDICTIONS.md (cohort 3, 95% coverage).
S1-S4 use the same rules as R1-R4 (src/check_replication.py). Run AFTER src/analysis.py in the
cohort-3 working copy (with ICREST_ALPHA=0.05 set):
    python src/check_robustness_c3.py
Output: results/analysis/robustness_c3_verdicts.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_replication as cr  # noqa: E402

A = Path("results/analysis")


def main():
    conf = pd.read_csv(A / "confirmatory.csv")
    abl1 = pd.read_csv(A / "ablation_electricity_h1.csv")
    abl24 = pd.read_csv(A / "ablation_electricity_h24.csv")
    v = cr.verdicts(conf, abl1, abl24)
    out = {f"S{i}": v[f"R{i}"] for i in (1, 2, 3, 4)}
    n = sum(out[k]["holds"] for k in ("S1", "S2", "S3"))
    out["overall"] = "holds at 95% coverage" if n == 3 else ("holds partly" if n else "does not hold")
    (A / "robustness_c3_verdicts.json").write_text(json.dumps(out, indent=2, default=str))
    for k in ("S1", "S2", "S3", "S4"):
        extra = {kk: vv for kk, vv in out[k].items() if kk != "holds"}
        print(f"{k}: {'HOLDS' if out[k]['holds'] else 'does not hold'}   {extra}")
    print(f"\nTrade-off at 95% coverage (S1-S3): {out['overall'].upper()}")


if __name__ == "__main__":
    main()
