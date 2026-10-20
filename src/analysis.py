"""
Statistics for the test run, exactly as pre-registered (PREREGISTRATION.md + PREREG_AMENDMENT_1.md).
Reads results/test/*.csv.gz (from src/run_test.py); writes results/analysis/.

Confirmatory (Amendment 1): DTCW-R+ACI vs each of 9 baselines on Electricity (50 clients),
6 Holm families = {RMST of TTR (L1 target), Winkler (L2), worst local coverage (L1 target)}
x {h = 1, h = 24}. Wilcoxon signed-rank on client differences (two-sided), Holm within family,
median difference with 95% cluster-bootstrap CI (2,000 client resamples), rank-biserial r.
Claim "better" only if Holm p < 0.05 AND the CI excludes 0 in the favourable direction; a Winkler
claim also needs DTCW-R+ACI's L2 coverage error to be at most 1 pp above the baseline's.

Secondary / exploratory: Table B (all layers), per-shift-type breakdown, DTCW-R alone, 2x2
ablation (PREREGISTRATION Sec. 7), TTR 12-cell sensitivity (Kendall tau), Friedman + Nemenyi,
achieved injection sizes, Elec2 effect sizes (moving-block bootstrap, block 48, 1,000 resamples).

Run from the project root:
    python src/analysis.py                 # everything except the Elec2 block bootstrap
    python src/analysis.py --elec2-mbb     # also the Elec2 moving-block bootstrap (re-replays L2)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inject as ij  # noqa: E402
from run_calib_val import ALPHA, HORIZONS  # noqa: E402

IN, OUT = Path("results/test"), Path("results/analysis")
SEED, BOOT = 2027, 2000
MAIN = "dtcwr_aci"
BASELINE_FAMILIES = ("fixed", "aci", "rocp", "dtaci", "pid", "eci_cutoff", "driftocp")
TABLE_B = None          # filled from the data: fixed_* (3) + 7 families + dtcwr + dtcwr_aci = 11
ABL = {"D": "dtcwr", "B": "abl_B_magnitude_reset", "A": "abl_A_miss_anchored", "R": "abl_R_miss_reset"}
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
               10: 3.164, 11: 3.219, 12: 3.268}            # Demsar (2006), Table 5a
LABEL = {"aci": "ACI", "rocp": "ROCP", "dtaci": "DtACI", "pid": "Conformal PID",
         "eci_cutoff": "ECI-cutoff", "driftocp": "DriftOCP", "dtcwr": "DTCW-R",
         "dtcwr_aci": "DTCW-R+ACI"}


def label(m: str) -> str:
    return f"Fixed {m.split('_')[1]}" if m.startswith("fixed_") else LABEL.get(m, m)


def baselines(methods) -> list[str]:
    return sorted([m for m in methods if m.startswith("fixed_")], key=lambda x: int(x.split("_")[1])) + \
        [m for m in BASELINE_FAMILIES[1:] if m in methods]


# ----------------------------------------------------------------------------- client outcomes
def client_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (series, method): the pre-registered client-level outcomes."""
    l1 = df[df.layer == "L1_target"]
    l2 = df[df.layer == "L2"]
    g1 = l1.groupby(["series", "method"])
    out = pd.DataFrame({
        "rmst": g1["ttr"].apply(lambda x: np.minimum(x, ij.TTR_CAP).mean()),
        "pct_recovered": g1["recovered"].mean(),
        "worst_local_cov_L1": g1["worst_local_coverage"].mean(),
        "coverage_L1": g1["coverage"].mean(),
        "winkler_L1": g1["winkler"].mean(),
    })
    l2i = l2.set_index(["series", "method"])
    out["winkler_L2"] = l2i["winkler"]
    out["coverage_L2"] = l2i["coverage"]
    out["coverage_error_L2"] = (l2i["coverage"] - (1 - ALPHA)).abs()
    out["worst_local_cov_L2"] = l2i["worst_local_coverage"]
    out["mean_width_L2"] = l2i["mean_width"]
    return out.reset_index()


# ----------------------------------------------------------------------------- tests
def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def cluster_boot_median(d: np.ndarray, reps: int = BOOT, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    med = np.median(d[idx], axis=1)
    return float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna().to_numpy(float)
    if np.all(d == 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    lo, hi = cluster_boot_median(d)
    return {"n": len(d), "median_diff": float(np.median(d)), "ci_lo": lo, "ci_hi": hi,
            "p": p, "rank_biserial": rank_biserial(d)}


FAMILIES = (("rmst", "lower", "RMST of TTR (L1 target)"),
            ("winkler_L2", "lower", "Winkler (L2)"),
            ("worst_local_cov_L1", "higher", "Worst local coverage (L1 target)"))


def confirmatory(co: dict, ds: str = "electricity", main: str = MAIN) -> pd.DataFrame:
    rows = []
    for h in HORIZONS[ds]:
        c = co[(ds, h)].set_index(["method", "series"])
        bl = baselines(c.index.get_level_values(0).unique())
        cov_err = c["coverage_error_L2"].groupby(level=0).mean()
        for col, better, name in FAMILIES:
            fam = []
            for b in bl:
                r = paired(c.loc[main, col], c.loc[b, col])
                fav = (r["ci_hi"] < 0) if better == "lower" else (r["ci_lo"] > 0)
                fam.append({"dataset": ds, "h": h, "family": name, "outcome": col, "better": better,
                            "baseline": label(b), **r, "ci_favourable": bool(fav),
                            "cov_err_gap_pp": 100 * float(cov_err[main] - cov_err[b])})
            adj = holm([f["p"] for f in fam])
            for f, pa in zip(fam, adj):
                f["p_holm"] = float(pa)
                ok = pa < 0.05 and f["ci_favourable"]
                if f["outcome"] == "winkler_L2":
                    ok = ok and f["cov_err_gap_pp"] <= 1.0
                worse = pa < 0.05 and ((f["ci_lo"] > 0) if better == "lower" else (f["ci_hi"] < 0))
                f["verdict"] = "better" if ok else ("worse" if worse else "no claim")
            rows += fam
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- tables
def table_b(df: pd.DataFrame, co: pd.DataFrame) -> pd.DataFrame:
    """Per method: medians over clients (Electricity) or the single series (Elec2)."""
    ref = co[co.method.str.startswith("fixed_")].copy()
    ref["W"] = ref.method.str.split("_").str[1].astype(int)
    longest = ref.loc[ref.groupby("series")["W"].idxmax(), ["series", "winkler_L2"]].set_index("series")["winkler_L2"]
    co = co.assign(rel_winkler_L2=co.winkler_L2 / co.series.map(longest),
                   rel_winkler_L1=co.winkler_L1 / co.series.map(longest))
    ms = df[df.layer == "L2"].groupby("method")["ms_per_step"].median()
    t = co.groupby("method").agg(
        coverage_L2=("coverage_L2", "mean"), coverage_error_L2=("coverage_error_L2", "mean"),
        rel_winkler_L2=("rel_winkler_L2", "median"), worst_local_cov_L2=("worst_local_cov_L2", "median"),
        coverage_L1=("coverage_L1", "mean"), rel_winkler_L1=("rel_winkler_L1", "median"),
        rmst_L1=("rmst", "median"), pct_recovered_L1=("pct_recovered", "mean"),
        worst_local_cov_L1=("worst_local_cov_L1", "median"))
    t["ms_per_step"] = ms
    t.index = [label(m) for m in t.index]
    return t


def km_median(ttr: np.ndarray, rec: np.ndarray, cap: int = ij.TTR_CAP) -> float:
    """Kaplan-Meier median time-to-recover (censored at cap); inf if never below 0.5."""
    t = np.minimum(ttr, cap).astype(float)
    ev = rec.astype(bool)
    s = 1.0
    for u in np.unique(t[ev]):
        at_risk = (t >= u).sum()
        d = ((t == u) & ev).sum()
        s *= 1 - d / at_risk
        if s <= 0.5:
            return float(u)
    return float("inf")


def by_shift_type(df: pd.DataFrame, layer: str = "L1_target") -> pd.DataFrame:
    x = df[df.layer == layer]
    rows = []
    for (m, kind), g in x.groupby(["method", "kind"]):
        cl = g.groupby("series")["ttr"].apply(lambda v: np.minimum(v, ij.TTR_CAP).mean())
        rows.append({"method": label(m), "kind": kind, "rmst_median_over_clients": float(cl.median()),
                     "km_median_ttr": km_median(g.ttr.to_numpy(), g.recovered.to_numpy()),
                     "pct_recovered": float(g.recovered.mean()),
                     "worst_local_cov": float(g.worst_local_coverage.median()),
                     "det_delay_median": float(g.det_delay.median()) if g.det_delay.notna().any() else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2x2
def ablation(df: pd.DataFrame) -> pd.DataFrame:
    x = df[(df.layer == "L1_target") & (df.method.isin(ABL.values()))]
    inv = {v: k for k, v in ABL.items()}
    rows = []
    for stratum in ("variance", "level", "gradual", "all"):
        s = x if stratum == "all" else x[x.kind == stratum]
        out = {}
        for col, agg in (("rmst", lambda v: np.minimum(v, ij.TTR_CAP).mean()),
                         ("winkler", np.mean), ("worst_local_coverage", np.mean), ("mean_width", np.mean),
                         ("det_delay", np.median)):
            src = "ttr" if col == "rmst" else col
            out[col] = s.groupby(["series", "method"])[src].apply(agg).unstack().rename(columns=inv)
        fam = []
        for col in out:
            M = out[col]
            eff = {"signal": 0.5 * ((M.B - M.R) + (M.D - M.A)),
                   "window": 0.5 * ((M.A - M.R) + (M.D - M.B)),
                   "interaction": (M.D - M.B) - (M.A - M.R)}
            for e, d in eff.items():
                d = d.dropna().to_numpy(float)
                p = 1.0 if np.all(d == 0) else float(stats.wilcoxon(d).pvalue)
                lo, hi = cluster_boot_median(d)
                fam.append({"stratum": stratum, "outcome": col, "effect": e, "median": float(np.median(d)),
                            "ci_lo": lo, "ci_hi": hi, "p": p, "rank_biserial": rank_biserial(d),
                            "cell_medians": {k: float(M[k].median()) for k in "DBAR"}})
        prim = [f for f in fam if f["outcome"] == "rmst"]
        for f, pa in zip(prim, holm([f["p"] for f in prim])):
            f["p_holm_primary"] = float(pa)
        rows += fam
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- sensitivity / ranking
def ttr_sensitivity(df: pd.DataFrame, methods: list[str]) -> dict:
    x = df[(df.layer == "L1_target") & (df.method.isin(methods))]
    cols = [c for c in x.columns if c.startswith("ttr__")]
    ranks = {}
    for c in cols:
        r = x.groupby(["series", "method"])[c].apply(lambda v: np.minimum(v, ij.TTR_CAP).mean()) \
             .groupby("method").median()
        ranks[c[5:]] = r.rank()
    taus = {f"{a}|{b}": float(stats.kendalltau(ranks[a], ranks[b]).statistic)
            for a, b in itertools.combinations(ranks, 2)}
    mn = min(taus.values()) if taus else float("nan")
    return {"min_kendall_tau": mn, "ranking_claim_allowed": bool(mn >= 0.7),
            "ranks": {k: {label(m): float(v) for m, v in r.items()} for k, r in ranks.items()},
            "pairwise_tau": taus}


def friedman_nemenyi(co: pd.DataFrame, col: str, methods: list[str], lower_better: bool = True) -> dict:
    M = co[co.method.isin(methods)].pivot(index="series", columns="method", values=col)[methods].dropna()
    st = stats.friedmanchisquare(*[M[m] for m in methods])
    R = (M if lower_better else -M).rank(axis=1).mean()
    k, n = len(methods), len(M)
    cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
    return {"outcome": col, "chi2": float(st.statistic), "p": float(st.pvalue), "n": n, "k": k,
            "critical_difference": float(cd), "avg_rank": {label(m): float(R[m]) for m in methods}}


def achieved_sizes(df: pd.DataFrame) -> pd.DataFrame:
    x = df[(df.layer == "L1_target") & (df.method == MAIN)]
    rows = []
    for (kind, size), g in x.groupby(["kind", "size"]):
        col = {"variance": "achieved_ratio", "level": "achieved_mean_shift_sd",
               "gradual": "achieved_ramp_end_sd"}[kind]
        v = g[col].dropna()
        target = size ** 2 if kind == "variance" else size
        med = float(v.median())
        rows.append({"kind": kind, "size": size, "metric": col, "target": target, "median": med,
                     "q25": float(v.quantile(.25)), "q75": float(v.quantile(.75)),
                     "relabel": bool(kind == "variance" and abs(med / target - 1) > 0.10)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Elec2 MBB
def elec2_mbb(block: int = 48, reps: int = 1000) -> pd.DataFrame:
    """Re-replays the deterministic L2 stream for Elec2 to get per-target Winkler, then a
    moving-block bootstrap CI of the mean difference DTCW-R+ACI - baseline."""
    import calib as cb
    import run_test as rt
    from run_calib_val import load_forecasts
    fz = rt.load_frozen()
    rows = []
    rng = np.random.default_rng(SEED)
    for h in HORIZONS["elec2"]:
        df = load_forecasts("elec2", h)
        df = df[df.split.isin(["val", "test"])]
        y, yh = df.y.to_numpy(float), df.yhat.to_numpy(float)
        n_val = int((df.split == "val").sum())
        wk = {}
        for name, make in rt.method_rows("elec2", h, fz).items():
            out = cb.replay(make(), y, yh, clip_inf=True)
            q, s = out["q"][n_val:], out["score"][n_val:]
            wk[name] = 2 * q + (2 / ALPHA) * np.maximum(s - q, 0)
        n = len(wk[MAIN])
        starts = rng.integers(0, n - block + 1, size=(reps, int(np.ceil(n / block))))
        idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
        for b in baselines(list(wk)) + ["dtcwr"]:
            d = wk[MAIN] - wk[b]
            bs = d[idx].mean(axis=1)
            rows.append({"h": h, "baseline": label(b), "mean_diff": float(d.mean()),
                         "rel_diff": float(d.mean() / wk[b].mean()),
                         "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def load_all() -> dict:
    return {(ds, h): pd.read_csv(IN / f"{ds}_h{h}.csv.gz", low_memory=False)
            for ds in HORIZONS for h in HORIZONS[ds] if (IN / f"{ds}_h{h}.csv.gz").exists()}


def main(do_mbb: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    data = load_all()
    co = {k: client_outcomes(v) for k, v in data.items()}
    report = {}

    conf = confirmatory(co)
    conf.to_csv(OUT / "confirmatory.csv", index=False)
    print("\n### CONFIRMATORY (Amendment 1): DTCW-R+ACI vs baselines, Electricity")
    print(conf[["h", "family", "baseline", "median_diff", "ci_lo", "ci_hi", "p_holm", "rank_biserial",
                "cov_err_gap_pp", "verdict"]].round(4).to_string(index=False))

    sec = confirmatory(co, main="dtcwr")
    sec.to_csv(OUT / "secondary_dtcwr_alone.csv", index=False)

    for (ds, h), df in data.items():
        key = f"{ds}_h{h}"
        tb = table_b(df, co[(ds, h)])
        tb.to_csv(OUT / f"table_b_{key}.csv")
        bst = by_shift_type(df)
        bst.to_csv(OUT / f"by_shift_type_{key}.csv", index=False)
        by_shift_type(df, "L1_resid").to_csv(OUT / f"by_shift_type_resid_{key}.csv", index=False)
        ach = achieved_sizes(df)
        ach.to_csv(OUT / f"achieved_injection_{key}.csv", index=False)
        tbm = [m for m in df.method.unique() if not m.startswith("abl_")]
        sens = ttr_sensitivity(df, tbm)
        report[key] = {"ttr_sensitivity": {k: sens[k] for k in ("min_kendall_tau", "ranking_claim_allowed")}}
        (OUT / f"ttr_sensitivity_{key}.json").write_text(json.dumps(sens, indent=2))
        print(f"\n### TABLE B — {key} (Electricity: medians over 50 clients; rel. Winkler = / longest fixed window, L2)")
        print(tb.round(3).to_string())
        print(f"\n### Achieved injection sizes — {key}")
        print(ach.round(3).to_string(index=False))
        print(f"TTR sensitivity: min Kendall tau over 12 cells = {sens['min_kendall_tau']:.3f} -> ranking claim "
              f"{'allowed' if sens['ranking_claim_allowed'] else 'NOT allowed'}")
        if ds == "electricity":
            fr = [friedman_nemenyi(co[(ds, h)], "winkler_L2", tbm),
                  friedman_nemenyi(co[(ds, h)], "rmst", tbm)]
            report[key]["friedman"] = fr
            for f in fr:
                rk = sorted(f["avg_rank"].items(), key=lambda kv: kv[1])
                print(f"Friedman {f['outcome']}: chi2={f['chi2']:.1f} p={f['p']:.2g}; CD={f['critical_difference']:.2f}; "
                      "avg ranks: " + ", ".join(f"{m} {r:.2f}" for m, r in rk))
            ab = ablation(df)
            ab.to_csv(OUT / f"ablation_{key}.csv", index=False)
            print(f"\n### 2x2 ABLATION — {key}" + ("  (PRIMARY: h = 1)" if h == 1 else "  (secondary)"))
            show = ab[ab.outcome.isin(["rmst", "det_delay"])]
            print(show[["stratum", "outcome", "effect", "median", "ci_lo", "ci_hi", "p",
                        "p_holm_primary"]].round(4).to_string(index=False))
        print(f"\n### By shift type (L1 target) — {key}")
        print(bst.round(3).to_string(index=False))

    if do_mbb and ("elec2", 1) in data:
        mb = elec2_mbb()
        mb.to_csv(OUT / "elec2_mbb_winkler.csv", index=False)
        print("\n### Elec2 L2 Winkler, DTCW-R+ACI - baseline (moving-block bootstrap, block 48)")
        print(mb.round(5).to_string(index=False))
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--elec2-mbb", action="store_true")
    main(ap.parse_args().elec2_mbb)
