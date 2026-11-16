"""
POST HOC sensitivity analyses requested in the ICREST'27 reviews (cohort 1, Electricity).
Nothing here changes an earlier result; every run starts from the frozen configurations.

Part A (meta-review 4, Reviewer 2 point 4): one-at-a-time sensitivity of DTCW-R+ACI to
    the CUSUM threshold H (x0.5, x0.75, x1.5, x2; i.e. a looser or stricter false-alarm target),
    the ACI step size gamma (x0.5, x2), the long window W_L (x0.5, x2) and the short window W_S (x0.5, x2).
Part B (Reviewer 2 point 6): the delayed-feedback rule of the adaptive baselines at h = 24.
    Published rule used in the paper (one state, updated when the score of target t arrives) versus
    an interleaved rule: h independent copies of the published h = 1 algorithm, copy j running on the
    targets t = j (mod h) only, the same idea as DTCW-R+ACI's interleaved ACI copies. Also ACI the
    other way round: its h copies (paper) versus one shared level.
Part C (Reviewer 2 point 4, shift magnitudes): RMST by shift type and size, read from the
    existing test results (no rerun).

For every variant, per client: false alarms per 1,000 test steps (untouched stream), test coverage,
Winkler score relative to the 2,160 h window (untouched stream), RMST over the 35 target-level shifts
(frozen TTR cell), and the share of variance shifts detected.

Run from the cohort-1 project root (D:\\icrest2027):
    python src/sensitivity.py --workers 3
Outputs in results/posthoc/: sensitivity_clients_h<h>.csv, sensitivity_summary.csv,
    feedback_clients_h24.csv, feedback_summary.csv, shift_size_summary.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib as cb  # noqa: E402
import inject as ij  # noqa: E402
import inject_target as it  # noqa: E402
import run_test as rt  # noqa: E402
import tune as tn  # noqa: E402
from run_calib_val import ALPHA, LOCAL_W, load_forecasts  # noqa: E402

DS = "electricity"
OUT = Path("results/posthoc")
VARIANTS = [("frozen", {}), ("H x0.5", {"H": 0.5}), ("H x0.75", {"H": 0.75}), ("H x1.5", {"H": 1.5}),
            ("H x2", {"H": 2.0}), ("gamma x0.5", {"gamma": 0.5}), ("gamma x2", {"gamma": 2.0}),
            ("W_L x0.5", {"W_L": 0.5}), ("W_L x2", {"W_L": 2.0}), ("W_S x0.5", {"W_S": 0.5}),
            ("W_S x2", {"W_S": 2.0})]
FEEDBACK_FAMS = ("pid", "eci_cutoff", "dtaci")


# ----------------------------------------------------------------------------- method wrappers
class Interleaved:
    """h independent copies of an h = 1 method; copy j sees and serves only targets t = j (mod h).
    Uses the replay engine's timing: score t arrives at origin t, the interval for t + h is issued
    at origin t, and for copy j these are consecutive steps of an ordinary h = 1 problem."""

    def __init__(self, make1, h: int):
        self.h = h
        self.copies = [make1() for _ in range(h)]
        self.q_issued: dict = {}

    def setup(self, warm):
        warm = np.asarray(warm, float)
        for j, c in enumerate(self.copies):
            c.setup(warm[j::self.h])

    def observe(self, t, s):
        self.copies[t % self.h].observe(t // self.h, s)

    def predict(self, tau):
        q = self.copies[tau % self.h].predict(tau // self.h)
        self.q_issued[tau] = q
        return q


class _OneLevel:
    """Makes ACI keep one shared level instead of h interleaved copies."""

    def __init__(self, v):
        self.v = v

    def __getitem__(self, i):
        return self.v

    def __setitem__(self, i, v):
        self.v = v


# ----------------------------------------------------------------------------- workers
_W: dict = {}


def _init(h):
    fz = rt.load_frozen()
    key = f"{DS}_h{h}"
    _W.update(fz=fz, h=h, key=key, inj=it.TargetInjector(DS, h),
              pw=fz["choice"][key]["winner"] == "prewhiten")
    df = load_forecasts(DS, h)
    df = df[df["split"].isin(["val", "test"])]
    _W["streams"] = {s: (g["y"].to_numpy(float), g["yhat"].to_numpy(float), int((g["split"] == "val").sum()))
                     for s, g in df.groupby("series", sort=True)}


def _cfg(fam):
    return dict(_W["fz"]["sel"][_W["key"]][fam]["config"])


def dtcwr_variant(mult: dict):
    cfg = _cfg("dtcwr_aci")
    for k, m in mult.items():
        cfg[k] = cfg[k] * m
        if k in ("W_S", "W_L", "c"):
            cfg[k] = int(round(cfg[k]))
    if cfg["W_S"] > cfg["W_L"]:
        cfg["W_S"] = cfg["W_L"]
    return lambda: tn.make_method("dtcwr_aci", cfg, _W["h"], {"prewhiten": _W["pw"]})


def feedback_makers(h):
    mk = {}
    for fam in FEEDBACK_FAMS:
        cfg = _cfg(fam)
        mk[f"{fam} | published (one state)"] = lambda fam=fam, cfg=cfg: tn.make_method(fam, cfg, h)
        mk[f"{fam} | interleaved copies"] = (lambda fam=fam, cfg=cfg:
                                            Interleaved(lambda: tn.make_method(fam, cfg, 1), h))
    g = _cfg("aci")
    mk["aci | h copies (paper)"] = lambda: tn.make_method("aci", g, h)

    def aci_one():
        m = tn.make_method("aci", g, h)
        m.copies = _OneLevel(ALPHA)
        return m
    mk["aci | one level"] = aci_one
    return mk


def _streams(name):
    h, fz, key = _W["h"], _W["fz"], _W["key"]
    y, yhat, n_val = _W["streams"][name]
    n_test, week = len(y) - n_val, LOCAL_W[DS]
    cell = rt.frozen_cell(DS, fz)
    taus = ij.test_positions(n_val, n_test, L=cell["L"])
    sig = float(np.std((y - yhat)[:cb.W0]))
    shifted = []
    for tau in taus:
        for kind, size in ij.SETTINGS:
            se = fz["sig"][key][name][str(size)]["m"] * sig if kind == "variance" else None
            _, y_t, yh_t, _ = _W["inj"].inject(name, kind, size, tau, week, test=True, sigma_eps=se,
                                                base=(y, yhat))
            shifted.append((kind, size, tau, y_t, yh_t))
    return y, yhat, n_val, cell, shifted


def evaluate(make, y, yhat, n_val, cell, shifted, w_ref):
    n, week = len(y), LOCAL_W[DS]
    test = np.zeros(n, bool)
    test[n_val:] = True
    m = make()
    out = cb.replay(m, y, yhat, clip_inf=True)
    met = cb.interval_metrics(y, yhat, out["q"], test, ALPHA, week)
    al = [a[0] for a in getattr(m, "alarms", [])]
    row = {"coverage_L2": met["coverage"], "rel_winkler_L2": met["winkler"] / w_ref,
           "fa_test_per_1000": (1000 * sum(t >= n_val for t in al) / (n - n_val)) if hasattr(m, "alarms") else np.nan}
    ttrs, det = [], []
    by = {}
    for kind, size, tau, y_t, yh_t in shifted:
        mm = make()
        o = cb.replay(mm, y_t, yh_t, clip_inf=True)
        ttr, _ = ij.time_to_recover(o["score"] <= o["q"], tau, ALPHA, cell["w"], cell["delta"], cell["L"])
        ttrs.append(min(ttr, ij.TTR_CAP))
        by.setdefault(f"rmst_{kind}", []).append(min(ttr, ij.TTR_CAP))
        if kind == "variance" and hasattr(mm, "alarms"):
            det.append(ij.detection_delay([a[0] for a in mm.alarms], tau)[1])
    row["rmst_L1"] = float(np.mean(ttrs))
    row.update({k: float(np.mean(v)) for k, v in by.items()})
    row["detect_rate_variance"] = float(np.mean(det)) if det else np.nan
    return row


def _work(args):
    name, part = args
    h = _W["h"]
    y, yhat, n_val, cell, shifted = _streams(name)
    test = np.zeros(len(y), bool)
    test[n_val:] = True
    ref = cb.replay(tn.make_method("fixed", {"W": 2160}, h), y, yhat, clip_inf=True)
    w_ref = cb.interval_metrics(y, yhat, ref["q"], test, ALPHA, LOCAL_W[DS])["winkler"]
    rows = []
    if part == "A":
        for label, mult in VARIANTS:
            rows.append({"h": h, "series": name, "variant": label,
                         **evaluate(dtcwr_variant(mult), y, yhat, n_val, cell, shifted, w_ref)})
    else:
        for label, make in feedback_makers(h).items():
            rows.append({"h": h, "series": name, "variant": label,
                         **evaluate(make, y, yhat, n_val, cell, shifted, w_ref)})
    return name, rows


# ----------------------------------------------------------------------------- summaries
def boot_ci(d, B=2000, seed=0):
    d = np.asarray(d, float)
    if len(d) == 0:
        return np.nan, np.nan
    meds = np.median(np.random.default_rng(seed).choice(d, (B, len(d)), replace=True), axis=1)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def summarize(df, base_of):
    out = []
    for (h, v), g in df.groupby(["h", "variant"], sort=False):
        b = df[(df.h == h) & (df.variant == base_of(v))].set_index("series")
        gi = g.set_index("series")
        row = {"h": h, "variant": v, "coverage_L2": g.coverage_L2.mean(),
               "rel_winkler_L2": g.rel_winkler_L2.median(), "rmst_L1": g.rmst_L1.median(),
               "fa_test_per_1000": g.fa_test_per_1000.median(),
               "detect_rate_variance": g.detect_rate_variance.mean()}
        for col in ("rmst_L1", "rel_winkler_L2"):
            d = (gi[col] - b[col]).dropna().to_numpy()
            lo, hi = boot_ci(d)
            row[f"{col}_diff_vs_ref"] = float(np.median(d)) if len(d) else np.nan
            row[f"{col}_ci_lo"], row[f"{col}_ci_hi"] = lo, hi
        out.append(row)
    return pd.DataFrame(out)


def run_part(part, hs, workers):
    frames = []
    for h in hs:
        f = OUT / (f"sensitivity_clients_h{h}.csv" if part == "A" else f"feedback_clients_h{h}.csv")
        if f.exists():
            print(f"=== part {part}, h={h}: reusing {f}", flush=True)
            frames.append(pd.read_csv(f))
            continue
        names = sorted(load_forecasts(DS, h).series.unique())
        t0, rows = time.time(), []
        print(f"=== part {part}, h={h}: {len(names)} clients", flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(h,)) as pool:
            futs = [pool.submit(_work, (nm, part)) for nm in names]
            for i, fu in enumerate(as_completed(futs), 1):
                nm, r = fu.result()
                rows += r
                print(f"  {i}/{len(names)} {nm}  [{time.time() - t0:.0f} s]", flush=True)
        d = pd.DataFrame(rows)
        d.to_csv(f, index=False)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def shift_sizes():
    rows = []
    for h in (1, 24):
        f = Path("results/test") / f"{DS}_h{h}.csv.gz"
        d = pd.read_csv(f, low_memory=False)
        d = d[(d.layer == "L1_target") & d.method.isin(["dtcwr_aci", "aci", "dtaci", "pid", "eci_cutoff"])]
        cl = d.groupby(["method", "kind", "size", "series"])["ttr"].apply(
            lambda v: np.minimum(v, ij.TTR_CAP).mean()).reset_index()
        s = cl.groupby(["method", "kind", "size"])["ttr"].median().unstack("method")
        s["h"] = h
        rows.append(s.reset_index())
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    A = run_part("A", (1, 24), a.workers)
    sa = summarize(A, lambda v: "frozen")
    sa.to_csv(OUT / "sensitivity_summary.csv", index=False)
    print("\n### PART A (post hoc): DTCW-R+ACI one-at-a-time sensitivity, cohort 1; diffs vs frozen config")
    print(sa.round(4).to_string(index=False))

    B = run_part("B", (24,), a.workers)

    def ref_b(v):
        fam = v.split(" | ")[0]
        return "aci | h copies (paper)" if fam == "aci" else f"{fam} | published (one state)"
    sb = summarize(B, ref_b)
    sb.to_csv(OUT / "feedback_summary.csv", index=False)
    print("\n### PART B (post hoc): delayed-feedback rule at h = 24, cohort 1; diffs vs the rule used in the paper")
    print(sb.round(4).to_string(index=False))
    main_rows = A[(A.variant == "frozen") & (A.h == 24)].set_index("series")
    cmp = []
    for v, g in B.groupby("variant"):
        gi = g.set_index("series")
        for col in ("rmst_L1", "rel_winkler_L2"):
            d = (main_rows[col] - gi[col]).dropna().to_numpy()
            lo, hi = boot_ci(d)
            cmp.append({"baseline_variant": v, "outcome": col, "dtcwr_aci_minus_baseline": float(np.median(d)),
                        "ci_lo": lo, "ci_hi": hi})
    cmp = pd.DataFrame(cmp)
    cmp.to_csv(OUT / "feedback_vs_dtcwr_aci.csv", index=False)
    print("\n### PART B: DTCW-R+ACI (frozen) minus each baseline variant, h = 24 (negative = DTCW-R+ACI lower)")
    print(cmp.round(4).to_string(index=False))

    try:
        ss = shift_sizes()
        ss.to_csv(OUT / "shift_size_summary.csv", index=False)
        print("\n### PART C: median client RMST by shift type and size (from the test run, no rerun)")
        print(ss.round(1).to_string(index=False))
    except FileNotFoundError as e:
        print(f"\nPART C skipped: {e}")
    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
