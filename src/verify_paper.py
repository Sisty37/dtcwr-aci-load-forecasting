"""
Check every number reported in the ICREST 2027 paper against the saved experiment outputs.
Read-only: it opens result files and prints OK / MISMATCH / MISSING for each value. Nothing is rerun
and nothing is written.

Run from anywhere (Windows paths shown):
    python verify_paper.py --c1 D:\\icrest2027 --c3 D:\\icrest2027_c3
Cohort 2 outputs are read from <c1>/results/cohort2 (copied there by combine_cohorts.py).
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = {"ok": 0, "bad": 0, "missing": 0}


def tol_of(paper) -> float:
    """Half a unit in the last reported digit of the paper value."""
    d = Decimal(str(paper)).as_tuple().exponent
    return 0.5 * 10 ** d + 1e-9


def check(name: str, paper, getter, tol: float | None = None):
    try:
        got = getter()
    except Exception as e:  # noqa: BLE001
        RESULTS["missing"] += 1
        print(f"  MISSING   {name}: {type(e).__name__}: {e}")
        return
    if isinstance(paper, str):
        ok = str(got) == paper
        shown = got
    else:
        t = tol_of(paper) if tol is None else tol
        ok = abs(float(got) - float(paper)) <= t
        shown = f"{float(got):.4f}"
    RESULTS["ok" if ok else "bad"] += 1
    print(f"  {'OK      ' if ok else 'MISMATCH'}  {name}: paper {paper}, output {shown}")


def note(name: str, getter):
    """Print a value the paper states as a range or a condition (judged by eye)."""
    try:
        print(f"  INFO      {name}: {getter()}")
    except Exception as e:  # noqa: BLE001
        RESULTS["missing"] += 1
        print(f"  MISSING   {name}: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- loaders
class Cohort:
    def __init__(self, root: Path):
        self.root = root
        self.a = root / "analysis"

    def conf(self) -> pd.DataFrame:
        c = pd.read_csv(self.a / "confirmatory.csv")
        return c[c.dataset == "electricity"] if "dataset" in c else c

    def tb(self, h: int) -> pd.DataFrame:
        return pd.read_csv(self.a / f"table_b_electricity_h{h}.csv", index_col=0)

    def abl(self, h: int) -> pd.DataFrame:
        return pd.read_csv(self.a / f"ablation_electricity_h{h}.csv")

    def bst(self, h: int) -> pd.DataFrame:
        return pd.read_csv(self.a / f"by_shift_type_electricity_h{h}.csv")

    def ach(self, h: int) -> pd.DataFrame:
        return pd.read_csv(self.a / f"achieved_injection_electricity_h{h}.csv")

    def tau(self, h: int) -> float:
        return json.loads((self.a / f"ttr_sensitivity_electricity_h{h}.json").read_text())["min_kendall_tau"]

    # helpers
    def row(self, h, family, baseline):
        c = self.conf()
        r = c[(c.h == h) & (c.family.str.startswith(family)) & (c.baseline == baseline)]
        if len(r) != 1:
            raise KeyError(f"{family} / {baseline} / h={h}: {len(r)} rows")
        return r.iloc[0]

    def n_verdict(self, h, family, verdict):
        c = self.conf()
        return int(((c.h == h) & c.family.str.startswith(family) & (c.verdict == verdict)).sum())

    def tbv(self, h, method, col):
        return float(self.tb(h).loc[method, col])

    def ablv(self, h, effect="signal", stratum="variance", col="median"):
        a = self.abl(h)
        r = a[(a.stratum == stratum) & (a.outcome == "rmst") & (a.effect == effect)]
        return float(r.iloc[0][col])


RMST, WINK, WORST = "RMST", "Winkler", "Worst"
M = "DTCW-R+ACI"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", required=True, help="cohort-1 working copy, e.g. D:\\icrest2027")
    ap.add_argument("--c3", required=True, help="cohort-3 working copy, e.g. D:\\icrest2027_c3")
    a = ap.parse_args()
    R1, R3 = Path(a.c1) / "results", Path(a.c3) / "results"
    c1, c2, c3 = Cohort(R1), Cohort(R1 / "cohort2"), Cohort(R3)
    C12 = (("c1", c1), ("c2", c2))

    print("\n=== V-A  Confirmatory verdicts (Abstract, Sec. V-A, Fig. 2, Table II)")
    check("coverage h=1 c1", 0.900, lambda: c1.tbv(1, M, "coverage_L2"))
    check("coverage h=1 c2", 0.900, lambda: c2.tbv(1, M, "coverage_L2"))
    check("coverage h=24 c1", 0.901, lambda: c1.tbv(24, M, "coverage_L2"))
    check("coverage h=24 c2", 0.903, lambda: c2.tbv(24, M, "coverage_L2"))
    for (tag, c), vals in zip(C12, ((95.9, 98.8, 108.8, 122.5), (88.5, 102.0, 112.5, 122.1))):
        for m, v in zip((M, "Conformal PID", "ECI-cutoff", "ACI"), vals):
            check(f"RMST h=1 {m} {tag}", v, lambda c=c, m=m: c.tbv(1, m, "rmst_L1"))
    check("abstract: median 96 (c1)", 96, lambda: c1.tbv(1, M, "rmst_L1"))
    check("abstract: ACI 122 (c1)", 122, lambda: c1.tbv(1, "ACI", "rmst_L1"))
    for tag, c, v, lo, hi in (("c1", c1, -27.9, -41.8, -17.3), ("c2", c2, -34.8, -43.2, -23.3)):
        check(f"RMST h=1 vs ACI diff {tag}", v, lambda c=c: c.row(1, RMST, "ACI").median_diff)
        check(f"  CI lo {tag}", lo, lambda c=c: c.row(1, RMST, "ACI").ci_lo)
        check(f"  CI hi {tag}", hi, lambda c=c: c.row(1, RMST, "ACI").ci_hi)
    check("RMST h=1 vs ECI-cutoff c1", -17.9, lambda: c1.row(1, RMST, "ECI-cutoff").median_diff)
    check("RMST h=1 vs ECI-cutoff c2", -24.9, lambda: c2.row(1, RMST, "ECI-cutoff").median_diff)
    check("RMST h=1 vs PID c1", -9.1, lambda: c1.row(1, RMST, "Conformal PID").median_diff)
    note("  PID c1 CI (paper: includes 0)", lambda: (round(c1.row(1, RMST, "Conformal PID").ci_lo, 2),
                                                     round(c1.row(1, RMST, "Conformal PID").ci_hi, 2)))
    check("RMST h=1 vs PID c2", -18.4, lambda: c2.row(1, RMST, "Conformal PID").median_diff)
    check("h=1 recovery 'better' count c1", 8, lambda: c1.n_verdict(1, RMST, "better"), tol=0)
    check("h=1 recovery 'better' count c2", 9, lambda: c2.n_verdict(1, RMST, "better"), tol=0)
    check("h=1 recovery 'better' count c3", 7, lambda: c3.n_verdict(1, RMST, "better"), tol=0)
    check("rel Winkler h=1 DTCW-R+ACI c1", 0.989, lambda: c1.tbv(1, M, "rel_winkler_L2"))
    check("rel Winkler h=1 DtACI c1", 0.976, lambda: c1.tbv(1, "DtACI", "rel_winkler_L2"))
    for tag, c in C12:
        for b in ("ACI", "DtACI", "ECI-cutoff", "Fixed 168"):
            check(f"h=1 Winkler vs {b} {tag} = worse", "worse", lambda c=c, b=b: c.row(1, WINK, b).verdict)
    for (tag, c), vals in zip(C12, ((0.962, 1.097, 1.068), (0.943, 1.104, 1.059))):
        for m, v in zip((M, "Conformal PID", "ECI-cutoff"), vals):
            check(f"rel Winkler h=24 {m} {tag}", v, lambda c=c, m=m: c.tbv(24, m, "rel_winkler_L2"))
    for tag, c in (("c1", c1), ("c2", c2), ("c3", c3)):
        check(f"h=24 Winkler 'better' count {tag}", 8, lambda c=c: c.n_verdict(24, WINK, "better"), tol=0)
    check("h=24 Winkler vs ACI c1 = worse", "worse", lambda: c1.row(24, WINK, "ACI").verdict)
    check("h=24 Winkler vs ACI c2 = no claim", "no claim", lambda: c2.row(24, WINK, "ACI").verdict)
    check("RMST h=24 vs PID c1", 124.6, lambda: c1.row(24, RMST, "Conformal PID").median_diff)
    check("RMST h=24 vs PID c2", 112.2, lambda: c2.row(24, RMST, "Conformal PID").median_diff)
    check("RMST h=24 vs ACI c1", -237.6, lambda: c1.row(24, RMST, "ACI").median_diff)
    check("RMST h=24 vs ACI c2", -228.0, lambda: c2.row(24, RMST, "ACI").median_diff)
    for tag, c in C12:
        for b in ("Conformal PID", "ECI-cutoff", "DtACI", "ROCP", "Fixed 168"):
            check(f"h=24 recovery vs {b} {tag} = worse", "worse", lambda c=c, b=b: c.row(24, RMST, b).verdict)
        for b in ("ACI", "DriftOCP", "Fixed 720", "Fixed 2160"):
            check(f"h=24 recovery vs {b} {tag} = better", "better", lambda c=c, b=b: c.row(24, RMST, b).verdict)

    def agree():
        k = ["h", "family", "baseline"]
        m = c1.conf()[k + ["verdict"]].merge(c2.conf()[k + ["verdict"]], on=k, suffixes=("_1", "_2"))
        return int((m.verdict_1 == m.verdict_2).sum()), len(m)
    check("verdicts identical c1 vs c2 (52 of 54)", 52, lambda: agree()[0], tol=0)
    check("  out of", 54, lambda: agree()[1], tol=0)

    print("\n=== V-A  Cohort 3 (95% coverage)")
    check("coverage h=1 c3", 0.950, lambda: c3.tbv(1, M, "coverage_L2"))
    check("coverage h=24 c3", 0.955, lambda: c3.tbv(24, M, "coverage_L2"))
    check("RMST h=1 DTCW-R+ACI c3", 25.9, lambda: c3.tbv(1, M, "rmst_L1"))
    check("RMST h=1 ACI c3", 71.8, lambda: c3.tbv(1, "ACI", "rmst_L1"))
    check("RMST h=1 PID c3", 41.8, lambda: c3.tbv(1, "Conformal PID", "rmst_L1"))
    check("h=1 recovery vs ROCP c3 = no claim", "no claim", lambda: c3.row(1, RMST, "ROCP").verdict)
    check("h=1 recovery vs ECI-cutoff c3", 7.5, lambda: c3.row(1, RMST, "ECI-cutoff").median_diff)
    check("  CI lo", 3.4, lambda: c3.row(1, RMST, "ECI-cutoff").ci_lo)
    check("  CI hi", 11.5, lambda: c3.row(1, RMST, "ECI-cutoff").ci_hi)
    check("rel Winkler h=24 DTCW-R+ACI c3", 0.950, lambda: c3.tbv(24, M, "rel_winkler_L2"))
    check("rel Winkler h=24 ACI c3", 0.934, lambda: c3.tbv(24, "ACI", "rel_winkler_L2"))
    check("RMST h=24 PID c3", 92, lambda: c3.tbv(24, "Conformal PID", "rmst_L1"))
    check("RMST h=24 ECI-cutoff c3", 80, lambda: c3.tbv(24, "ECI-cutoff", "rmst_L1"))
    check("RMST h=24 DTCW-R+ACI c3", 166, lambda: c3.tbv(24, M, "rmst_L1"))

    print("\n=== Table II  predictions")
    for tag, c in (("c1", c1), ("c2", c2), ("c3", c3)):
        for b in ("Conformal PID", "ECI-cutoff"):
            check(f"R3 h=24 recovery vs {b} {tag} = worse", "worse", lambda c=c, b=b: c.row(24, RMST, b).verdict)
        for b in ("ACI", "DtACI"):
            check(f"R4 h=1 Winkler vs {b} {tag} = worse", "worse", lambda c=c, b=b: c.row(1, WINK, b).verdict)
    check("R5 c1 (h=1 variance signal effect)", 16.8, lambda: c1.ablv(1))
    note("  R5 c1 Holm p (paper < 0.05)", lambda: round(c1.ablv(1, col="p_holm_primary"), 4))
    check("R5 c2", 6.5, lambda: c2.ablv(1))
    check("  R5 c2 CI lo", 0.8, lambda: c2.ablv(1, col="ci_lo"))
    check("  R5 c2 CI hi", 14.4, lambda: c2.ablv(1, col="ci_hi"))
    # pre-registered rule (check_replication.py): Holm over the two predictions R5 and R6
    check("  R5 c2 Holm p (R5+R6 family)", 0.071,
          lambda: json.loads((c2.a / "replication_verdicts.json").read_text())["R5"]["p_holm_2"])
    check("R6 c1 (h=24)", -13.7, lambda: c1.ablv(24))
    note("  R6 c1 Holm p (paper < 0.05)", lambda: round(c1.ablv(24, col="p_holm_primary"), 4))
    check("R6 c2", 3.4, lambda: c2.ablv(24))
    check("V-D: R5 c1 CI lo", 10.0, lambda: c1.ablv(1, col="ci_lo"))
    check("V-D: R5 c1 CI hi", 26.5, lambda: c1.ablv(1, col="ci_hi"))
    check("V-D: cohort 3 signal h=1 (exploratory)", 17.4, lambda: c3.ablv(1))
    check("  CI lo", 0.2, lambda: c3.ablv(1, col="ci_lo"))
    check("  CI hi", 41.6, lambda: c3.ablv(1, col="ci_hi"))

    print("\n=== V-B  trade-off")
    note("PID/ECI rel Winkler h=24, c1-c2 (paper: 6-10% worse)",
         lambda: [round(c.tbv(24, m, "rel_winkler_L2"), 3) for _, c in C12 for m in ("Conformal PID", "ECI-cutoff")])

    def var_rmst(c, m):
        b = c.bst(1)
        return float(b[(b.method == m) & (b.kind == "variance")].rmst_median_over_clients.iloc[0])
    check("variance RMST h=1 DTCW-R+ACI c1", 108.3, lambda: var_rmst(c1, M))
    check("variance RMST h=1 DTCW-R+ACI c2", 100.6, lambda: var_rmst(c2, M))
    check("variance RMST h=1 PID c1", 157.1, lambda: var_rmst(c1, "Conformal PID"))
    check("variance RMST h=1 PID c2", 162.2, lambda: var_rmst(c2, "Conformal PID"))

    def level_range():
        v = [c.ach(h).query("kind == 'level'")["median"] for _, c in C12 for h in (1, 24)]
        v = pd.concat(v)
        return round(v.min(), 3), round(v.max(), 3)
    note("achieved level shift, residual mean in sigma (paper 0.02-0.39)", level_range)
    note("min Kendall tau h=1 c1, c2, c3 (paper: all < 0.7)", lambda: [round(c.tau(1), 3) for c in (c1, c2, c3)])

    print("\n=== V-C  natural changes (Table III)")
    nat = R1 / "natural"

    def nm(h, method, col):
        t = pd.read_csv(nat / "summary_methods.csv")
        return float(t[(t.h == h) & (t.cohort == "pooled") & (t.method == method)][col].iloc[0])

    def nb(h, base, col):
        t = pd.read_csv(nat / "summary_vs_baselines.csv")
        return float(t[(t.h == h) & (t.cohort == "pooled") & (t.baseline == base)][col].iloc[0])
    check("changes h=1", 141, lambda: nm(1, M, "change_points"), tol=0)
    check("changes h=24", 384, lambda: nm(24, M, "change_points"), tol=0)
    note("clients with changes h=1, h=24 (100 clients in total; 64/95 earlier)",
         lambda: (nm(1, M, "clients"), nm(24, M, "clients")))
    tab3 = {"ACI": (184, 1.14, 183, 1.04), "DtACI": (173, 1.14, 183, 1.19),
            "Conformal PID": (122, 1.22, 116, 1.48), "ECI-cutoff": (137, 1.21, 127, 1.39),
            "DTCW-R": (160, 1.22, 183, 1.16), M: (120, 1.26, 165, 1.18)}
    for m, (r1, w1, r24, w24) in tab3.items():
        check(f"Table III {m} recovery h=1", r1, lambda m=m: nm(1, m, "median_ttr"))
        check(f"Table III {m} width h=1", w1, lambda m=m: nm(1, m, "median_rel_width"))
        check(f"Table III {m} recovery h=24", r24, lambda m=m: nm(24, m, "median_ttr"))
        check(f"Table III {m} width h=24", w24, lambda m=m: nm(24, m, "median_rel_width"))
    check("paired h=1 vs ACI (31 sooner)", -31, lambda: nb(1, "ACI", "median_diff_ttr"))
    check("  CI (paper [17,45] as positive)", -45, lambda: nb(1, "ACI", "ci_lo"))
    check("  CI", -17, lambda: nb(1, "ACI", "ci_hi"))
    check("paired h=1 vs DtACI (36 sooner)", -36, lambda: nb(1, "DtACI", "median_diff_ttr"))
    check("paired h=1 vs PID", 1.8, lambda: nb(1, "Conformal PID", "median_diff_ttr"))
    check("  CI lo", -4.3, lambda: nb(1, "Conformal PID", "ci_lo"))
    check("  CI hi", 8.3, lambda: nb(1, "Conformal PID", "ci_hi"))
    note("paired h=1 vs ECI-cutoff (paper: tied)", lambda: (nb(1, "ECI-cutoff", "median_diff_ttr"),
                                                          nb(1, "ECI-cutoff", "ci_lo"), nb(1, "ECI-cutoff", "ci_hi")))
    check("paired h=24 vs PID (39 sooner)", 39, lambda: nb(24, "Conformal PID", "median_diff_ttr"))
    check("  CI lo", 26, lambda: nb(24, "Conformal PID", "ci_lo"))
    check("  CI hi", 65, lambda: nb(24, "Conformal PID", "ci_hi"))
    check("paired h=24 vs ECI-cutoff (36 sooner)", 36, lambda: nb(24, "ECI-cutoff", "median_diff_ttr"))
    check("  CI lo", 19, lambda: nb(24, "ECI-cutoff", "ci_lo"))
    check("  CI hi", 48, lambda: nb(24, "ECI-cutoff", "ci_hi"))
    check("paired h=24 vs ACI (22 sooner)", -22, lambda: nb(24, "ACI", "median_diff_ttr"))
    check("  CI (paper [8,37] as positive)", -37, lambda: nb(24, "ACI", "ci_lo"))
    check("  CI", -8, lambda: nb(24, "ACI", "ci_hi"))
    check("width h=24 PID vs ours (25% wider)", 25,
          lambda: 100 * (nm(24, "Conformal PID", "median_rel_width") / nm(24, M, "median_rel_width") - 1))
    check("width h=24 ECI vs ours (18% wider)", 18,
          lambda: 100 * (nm(24, "ECI-cutoff", "median_rel_width") / nm(24, M, "median_rel_width") - 1))
    note("width h=1 ours vs PID, ECI in % (paper: 3-4% wider)",
         lambda: [round(100 * (nm(1, M, "median_rel_width") / nm(1, b, "median_rel_width") - 1), 1)
                  for b in ("Conformal PID", "ECI-cutoff")])
    check("DTCW-R alarm within a week, natural h=1 (42%)", 42, lambda: 100 * nm(1, M, "dtcwr_alarm_within_week"))
    check("DTCW-R alarm within a week, natural h=24 (30%)", 30, lambda: 100 * nm(24, M, "dtcwr_alarm_within_week"))

    print("\n=== V-C  SAOCP (post hoc)")
    def s(h, tag, col):
        sa = pd.read_csv(R1 / "test_extra" / "saocp_vs_dtcwr_aci.csv")
        return float(sa[(sa.h == h) & (sa.cohort == tag)][col].iloc[0])
    check("SAOCP rel Winkler h=1 c1", 0.975, lambda: s(1, "c1", "saocp_rel_winkler_L2"))
    check("SAOCP rel Winkler h=1 c2", 0.972, lambda: s(1, "c2", "saocp_rel_winkler_L2"))
    check("ours - SAOCP RMST h=1 c1 (48 later)", -48, lambda: s(1, "c1", "rmst_median_diff"))
    check("  CI (paper [28,80] as positive)", -80, lambda: s(1, "c1", "rmst_ci_lo"))
    check("  CI", -28, lambda: s(1, "c1", "rmst_ci_hi"))
    check("ours - SAOCP RMST h=1 c2 (75 later)", -75, lambda: s(1, "c2", "rmst_median_diff"))
    check("  CI (paper [58,92] as positive)", -92, lambda: s(1, "c2", "rmst_ci_lo"))
    check("  CI", -58, lambda: s(1, "c2", "rmst_ci_hi"))
    check("SAOCP rel Winkler h=24 c1", 1.008, lambda: s(24, "c1", "saocp_rel_winkler_L2"))
    check("SAOCP rel Winkler h=24 c2", 1.005, lambda: s(24, "c2", "saocp_rel_winkler_L2"))
    check("SAOCP coverage h=24 c1", 0.874, lambda: s(24, "c1", "saocp_coverage_L2"))
    check("SAOCP coverage h=24 c2", 0.884, lambda: s(24, "c2", "saocp_coverage_L2"))
    check("ours - SAOCP RMST h=24 c1 (30 later)", -30, lambda: s(24, "c1", "rmst_median_diff"))
    check("  CI (paper [6,57] as positive)", -57, lambda: s(24, "c1", "rmst_ci_lo"))
    check("  CI", -6, lambda: s(24, "c1", "rmst_ci_hi"))
    note("ours - SAOCP RMST h=24 c2 (paper: as fast)",
         lambda: (s(24, "c2", "rmst_median_diff"), s(24, "c2", "rmst_ci_lo"), s(24, "c2", "rmst_ci_hi")))
    note("ours - SAOCP worst local coverage (paper: SAOCP higher, i.e. negative)",
         lambda: [round(s(h, t, "worst_local_cov_L1_median_diff"), 4) for h in (1, 24) for t in ("c1", "c2")])

    print("\n=== V-C  detector, clipping, cost")
    rv1, rv3 = R1 / "review", R3 / "review"

    def alarms():
        a = pd.concat([pd.read_csv(rv1 / "alarms_L2.csv"), pd.read_csv(rv3 / "alarms_L2.csv")])
        return a[a.method == "dtcwr_aci"]
    note("alarm medians per cohort and horizon (paper 0.19-0.57)",
         lambda: alarms()[["h", "cohort", "median", "ci_lo", "ci_hi"]].round(3).to_dict("records"))
    note("every CI includes 0.5 or lies below (ci_lo <= 0.5)", lambda: bool((alarms().ci_lo <= 0.5).all()))

    def delay():
        d = pd.concat([pd.read_csv(rv1 / "delay_variance.csv"), pd.read_csv(rv3 / "delay_variance.csv")])
        return d[d.method == "dtcwr_aci"]
    note("variance detection rate, % (paper 72-86)", lambda: sorted((100 * delay().detect_rate).round(1)))
    note("median detection delay (paper 31-82)", lambda: sorted(delay().median_delay_detected))

    def clip(m, h):
        c = pd.concat([pd.read_csv(rv1 / "clipping.csv"), pd.read_csv(rv3 / "clipping.csv")])
        return float(c[(c.method == m) & (c.h == h) & (c.layer == "L2")].pct_targets_clipped.max())
    check("max clipping DTCW-R+ACI h=1, %", 0.054, lambda: clip("dtcwr_aci", 1))
    check("max clipping DTCW-R+ACI h=24, %", 0.31, lambda: clip("dtcwr_aci", 24))
    check("max clipping ACI, %", 0.83, lambda: max(clip("aci", 1), clip("aci", 24)))
    check("max clipping DtACI, %", 1.9, lambda: max(clip("dtaci", 1), clip("dtaci", 24)))

    def bench():
        b = pd.read_csv(rv1 / "bench_methods.csv")
        return b.groupby(["method"]).agg(ms=("ms_per_step", "median"), mib=("peak_kib", lambda v: v.median() / 1024))
    note("ms per step and MiB by method (paper: ours 0.011-0.013, others 0.005-0.10, SAOCP 0.21-0.23, 1.0-1.2 MiB)",
         lambda: bench().round(3).to_dict("index"))

    print("\n=== V-D  components (Table IV) and ACI layer")
    tab4 = {"ACI": ((122, 122), (467, 472), (.982, .983), (.950, .941)),
            "DTCW-R": ((142, 136), (206, 211), (.987, .990), (1.004, .996)),
            "abl_B_magnitude_reset": ((143, 132), (199, 211), (.986, .988), (1.000, .992)),
            "abl_A_miss_anchored": ((138, 128), (206, 211), (.989, .990), (1.005, 1.003)),
            "abl_R_miss_reset": ((140, 133), (210, 207), (.986, .989), (1.005, 1.000)),
            M: ((96, 88), (226, 222), (.989, .990), (.962, .943))}
    for m, (r1, r24, w1, w24) in tab4.items():
        for i, (tag, c) in enumerate(C12):
            check(f"Table IV {m} RMST h=1 {tag}", r1[i], lambda c=c, m=m: c.tbv(1, m, "rmst_L1"))
            check(f"Table IV {m} RMST h=24 {tag}", r24[i], lambda c=c, m=m: c.tbv(24, m, "rmst_L1"))
            check(f"Table IV {m} Winkler h=1 {tag}", w1[i], lambda c=c, m=m: c.tbv(1, m, "rel_winkler_L2"))
            check(f"Table IV {m} Winkler h=24 {tag}", w24[i], lambda c=c, m=m: c.tbv(24, m, "rel_winkler_L2"))
    al = R1 / "combined" / "aci_layer.csv"

    def layer(h, comp, col="median_diff_rmst"):
        t = pd.read_csv(al)
        return float(t[(t.h == h) & (t.cohort == "pooled") & (t.comparison == f"DTCW-R+ACI - {comp}")][col].iloc[0])
    check("ACI layer h=1 vs DTCW-R (49.4 sooner)", -49.4, lambda: layer(1, "DTCW-R"))
    check("ACI layer h=1 vs ACI (30.3 sooner)", -30.3, lambda: layer(1, "ACI"))
    check("ACI layer h=24 vs ACI (233.8 sooner)", -233.8, lambda: layer(24, "ACI"))
    check("ACI layer h=24 vs DTCW-R", 7.3, lambda: layer(24, "DTCW-R"))
    check("  CI lo", -1.5, lambda: layer(24, "DTCW-R", "ci_lo"))
    check("  CI hi", 17.8, lambda: layer(24, "DTCW-R", "ci_hi"))

    print("\n=== V-D  Elec2")
    note("Elec2 Winkler, DTCW-R+ACI vs baselines (paper: h=48 3.7-42.5% lower than all; h=1 1.4-2.8% higher than five)",
         lambda: pd.read_csv(R1 / "analysis" / "elec2_mbb_winkler.csv").round(4).to_string(index=False))

    print(f"\nSUMMARY: {RESULTS['ok']} OK, {RESULTS['bad']} MISMATCH, {RESULTS['missing']} MISSING "
          "(INFO lines are ranges or conditions to read by eye)")


if __name__ == "__main__":
    main()
