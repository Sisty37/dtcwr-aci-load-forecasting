"""
Put the two cohorts side by side and run the EXPLORATORY pooled analyses that
PREREG_C2_PREDICTIONS.md allows (decision D15). Nothing here changes a confirmatory verdict.

Expects the cohort-2 outputs copied into this working copy:
    results/cohort2/analysis/*   results/cohort2/test/electricity_h*.csv.gz

Run from the project root (main working copy):
    python src/combine_cohorts.py
Output (results/combined/):
    confirmatory_side_by_side.csv   verdict and median difference per family x baseline, C1 | C2
    table_b_side_by_side.csv        key Table-B columns, C1 | C2
    pooled_2x2.csv                  2x2 effects on the 100 pooled clients (exploratory)
    aci_layer.csv                   DTCW-R+ACI vs DTCW-R alone and vs ACI alone, per cohort and pooled
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as an  # noqa: E402
import inject as ij  # noqa: E402

C1, C2, OUT = Path("results"), Path("results/cohort2"), Path("results/combined")
HS = (1, 24)


def load_test(root: Path) -> dict:
    return {h: pd.read_csv(root / "test" / f"electricity_h{h}.csv.gz", low_memory=False) for h in HS}


def side_by_side_confirmatory() -> pd.DataFrame:
    a = pd.read_csv(C1 / "analysis" / "confirmatory.csv")
    b = pd.read_csv(C2 / "analysis" / "confirmatory.csv")
    a = a[a.dataset == "electricity"]
    k = ["h", "family", "baseline"]
    cols = ["median_diff", "ci_lo", "ci_hi", "p_holm", "verdict"]
    m = a[k + cols].merge(b[k + cols], on=k, suffixes=("_c1", "_c2"))
    m["agree"] = m.verdict_c1 == m.verdict_c2
    return m


def side_by_side_table_b() -> pd.DataFrame:
    rows = []
    for h in HS:
        for tag, root in (("c1", C1), ("c2", C2)):
            t = pd.read_csv(root / "analysis" / f"table_b_electricity_h{h}.csv", index_col=0)
            t = t[~t.index.str.startswith("abl_")]
            t = t[["coverage_L2", "rel_winkler_L2", "worst_local_cov_L2", "rmst_L1", "pct_recovered_L1",
                   "worst_local_cov_L1", "ms_per_step"]].add_suffix(f"_{tag}")
            t["h"] = h
            rows.append(t)
    out = []
    for h in HS:
        a, b = [r for r in rows if r.h.iloc[0] == h]
        out.append(a.drop(columns="h").join(b.drop(columns="h")).assign(h=h))
    return pd.concat(out).rename_axis("method").reset_index()


def client_rmst(df: pd.DataFrame, method: str, kind: str | None = None) -> pd.Series:
    x = df[(df.layer == "L1_target") & (df.method == method)]
    if kind:
        x = x[x.kind == kind]
    return x.groupby("series")["ttr"].apply(lambda v: np.minimum(v, ij.TTR_CAP).mean())


def aci_layer(t1: dict, t2: dict) -> pd.DataFrame:
    rows = []
    for h in HS:
        for tag, dfs in (("c1", [t1[h]]), ("c2", [t2[h]]), ("pooled", [t1[h], t2[h]])):
            df = pd.concat(dfs)
            main = client_rmst(df, "dtcwr_aci")
            for other in ("dtcwr", "aci"):
                d = (main - client_rmst(df, other)).dropna().to_numpy()
                lo, hi = an.cluster_boot_median(d)
                rows.append({"h": h, "cohort": tag, "comparison": f"DTCW-R+ACI - {an.label(other)}",
                             "n_clients": len(d), "median_diff_rmst": float(np.median(d)),
                             "ci_lo": lo, "ci_hi": hi, "p": float(stats.wilcoxon(d).pvalue),
                             "share_clients_faster": float((d < 0).mean())})
    return pd.DataFrame(rows)


def pooled_2x2(t1: dict, t2: dict) -> pd.DataFrame:
    out = []
    for h in HS:
        ab = an.ablation(pd.concat([t1[h], t2[h]]))
        ab["h"] = h
        out.append(ab)
    return pd.concat(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    cs = side_by_side_confirmatory()
    cs.to_csv(OUT / "confirmatory_side_by_side.csv", index=False)
    agree = cs.groupby(["h", "family"])["agree"].mean()
    print("### Confirmatory verdicts, cohort 1 vs cohort 2 (share of the 9 baselines with the same verdict)")
    print(agree.round(2).to_string())
    print("\nDisagreements:")
    print(cs[~cs.agree][["h", "family", "baseline", "verdict_c1", "verdict_c2"]].to_string(index=False))

    tb = side_by_side_table_b()
    tb.to_csv(OUT / "table_b_side_by_side.csv", index=False)

    t1, t2 = load_test(C1), load_test(C2)
    al = aci_layer(t1, t2)
    al.to_csv(OUT / "aci_layer.csv", index=False)
    print("\n### EXPLORATORY: the ACI layer and the detector-driven window (RMST, L1 target; negative = DTCW-R+ACI faster)")
    print(al.round(4).to_string(index=False))

    p2 = pooled_2x2(t1, t2)
    p2.to_csv(OUT / "pooled_2x2.csv", index=False)
    show = p2[(p2.outcome == "rmst") & (p2.stratum == "variance")]
    print("\n### EXPLORATORY: 2x2 on the 100 pooled clients, variance shifts, RMST (signal > 0 = miss sooner)")
    print(show[["h", "effect", "median", "ci_lo", "ci_hi", "p"]].round(4).to_string(index=False))
    (OUT / "README.txt").write_text(
        "Exploratory cross-cohort summaries (D15). Confirmatory verdicts are in results/analysis "
        "(cohort 1) and results/cohort2/analysis (cohort 2, replication_verdicts.json).\n")
    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
