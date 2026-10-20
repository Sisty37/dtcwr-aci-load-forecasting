"""
Evaluate the cohort-2 replication predictions R1-R6 exactly as written in
PREREG_C2_PREDICTIONS.md (committed before any cohort-2 data step, tag c2-predictions).
Run AFTER src/analysis.py in the cohort-2 working copy:

    python src/check_replication.py
Output: results/analysis/replication_verdicts.json (and printed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as an  # noqa: E402

A = Path("results/analysis")


def verdicts(conf: pd.DataFrame, abl1: pd.DataFrame, abl24: pd.DataFrame) -> dict:
    def fam(h, outcome):
        return conf[(conf.h == h) & (conf.outcome == outcome)].set_index("baseline")["verdict"]

    out = {}
    w24 = fam(24, "winkler_L2")
    out["R1"] = {"holds": int((w24 == "better").sum()) >= 7, "n_better": int((w24 == "better").sum()),
                 "verdicts": w24.to_dict()}
    r1 = fam(1, "rmst")
    out["R2"] = {"holds": int((r1 == "better").sum()) >= 7, "n_better": int((r1 == "better").sum()),
                 "verdicts": r1.to_dict()}
    r24 = fam(24, "rmst")
    out["R3"] = {"holds": bool(r24.get("Conformal PID") == "worse" and r24.get("ECI-cutoff") == "worse"),
                 "verdicts": {k: r24.get(k) for k in ("Conformal PID", "ECI-cutoff")}}
    w1 = fam(1, "winkler_L2")
    out["R4"] = {"holds": bool(w1.get("ACI") != "better" and w1.get("DtACI") != "better"),
                 "verdicts": {k: w1.get(k) for k in ("ACI", "DtACI")}}
    rows = []
    for name, ab, sign in (("R5", abl1, +1), ("R6", abl24, -1)):
        r = ab[(ab.stratum == "variance") & (ab.outcome == "rmst") & (ab.effect == "signal")].iloc[0]
        rows.append((name, r, sign))
    padj = an.holm([r["p"] for _, r, _ in rows])
    for (name, r, sign), pa in zip(rows, padj):
        side = (r["ci_lo"] > 0) if sign > 0 else (r["ci_hi"] < 0)
        out[name] = {"holds": bool(pa < 0.05 and side and sign * r["median"] > 0),
                     "median": float(r["median"]), "ci": [float(r["ci_lo"]), float(r["ci_hi"])],
                     "p": float(r["p"]), "p_holm_2": float(pa)}
    main = sum(out[k]["holds"] for k in ("R1", "R2", "R3"))
    out["overall"] = "replicated" if main == 3 else ("partially replicated" if main else "not replicated")
    return out


def main():
    conf = pd.read_csv(A / "confirmatory.csv")
    abl1 = pd.read_csv(A / "ablation_electricity_h1.csv")
    abl24 = pd.read_csv(A / "ablation_electricity_h24.csv")
    v = verdicts(conf, abl1, abl24)
    (A / "replication_verdicts.json").write_text(json.dumps(v, indent=2, default=str))
    for k in ("R1", "R2", "R3", "R4", "R5", "R6"):
        extra = {kk: vv for kk, vv in v[k].items() if kk != "holds"}
        print(f"{k}: {'HOLDS' if v[k]['holds'] else 'does not hold'}   {extra}")
    print(f"\nMain claim (R1-R3): {v['overall'].upper()}")


if __name__ == "__main__":
    main()
