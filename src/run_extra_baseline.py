"""
Decision D17 (after both tests were opened): SAOCP (Bhatnagar et al., ICML 2023) as an ADDITIONAL
baseline, requested in the pre-submission review. It is not part of the pre-registered
confirmatory families; it is reported separately as post hoc.

Fairness rules (identical to every other family):
  - 16 configurations (lifetime g in {4, 8, 16, 32} x scale multiplier in {0.25, 0.5, 1, 2}),
    scored on the same 22 VALIDATION streams per client with the D10 selection rule;
  - the selected configuration is frozen and committed BEFORE the test streams are run;
  - the test streams (L2 and L1 target/residual layers) are rebuilt exactly as in run_test.py,
    the TTR cell and metrics are the frozen ones, and the test is run once.
Electricity only (both cohorts). Run in each cohort working copy:

    python src/run_extra_baseline.py --stage tune
    git add . ; git commit -m "D17: SAOCP tuned on validation (frozen before test)"
    python src/run_extra_baseline.py --stage test
Then, in the main copy (after copying the cohort-2 results/test_extra to results/cohort2/test_extra):
    python src/run_extra_baseline.py --stage compare
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as an  # noqa: E402
import run_test as rt  # noqa: E402
import tune as tn  # noqa: E402
from run_calib_val import HORIZONS, load_forecasts  # noqa: E402

DS = "electricity"
SEL = Path("results/tuning/selected_extra.json")
OUT = Path("results/test_extra")


def _extra_rows(ds, h, fz):
    cfg = json.loads(SEL.read_text())[f"{ds}_h{h}"]["saocp"]["config"]
    return {"saocp": (lambda: tn.make_method("saocp", cfg, h))}


rt.method_rows = _extra_rows          # also executed in every worker process (module import)


def stage_tune(workers):
    sel = json.loads(SEL.read_text()) if SEL.exists() else {}
    for h in HORIZONS[DS]:
        key = f"{DS}_h{h}"
        val = load_forecasts(DS, h)
        val = val[val.split == "val"]
        sids = sorted(val.series.unique())
        T_val = int(val.groupby("series").size().min())
        configs = tn.grids(DS, h, T_val)["saocp"]
        d = tn.OUT / key
        d.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=tn._init, initargs=(DS, h)) as pool:
            df = tn.run_family(pool, "saocp", configs, sids, d / "saocp.csv")
        sel.setdefault(key, {})["saocp"] = s = tn.select(df)
        print(f"  {key} saocp selected {s['config']} score {s['score']:.3f} cov {s['cov_unshifted']:.3f} "
              f"eligible {s['n_eligible']}/16 {s['flag']}")
        SEL.write_text(json.dumps(sel, indent=2))


def stage_test(workers, allow_dirty):
    prov = rt.check_provenance(allow_dirty)
    if not SEL.exists():
        raise SystemExit("run --stage tune and commit first")
    fz = rt.load_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    for h in HORIZONS[DS]:
        t0 = time.time()
        key = f"{DS}_h{h}"
        names = json.loads((Path("results/models") / f"{key}.json").read_text())["series"]
        rows = []
        with ProcessPoolExecutor(max_workers=workers, initializer=rt._init, initargs=(DS, h, fz)) as pool:
            futs = [pool.submit(_work, n) for n in names]
            for i, f in enumerate(as_completed(futs), 1):
                name, r = f.result()
                rows += r
                print(f"  {key} {i}/{len(names)} {name} [{time.time() - t0:.0f} s]", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"{key}.csv.gz", index=False)
        (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
        print(f"=== {key}: {len(rows)} rows -> {OUT / (key + '.csv.gz')}")


def _work(name):
    return rt._work(name)


def stage_compare():
    roots = {"c1": (Path("results/test"), Path("results/test_extra")),
             "c2": (Path("results/cohort2/test"), Path("results/cohort2/test_extra"))}
    out = []
    for h in HORIZONS[DS]:
        for tag, (main, extra) in roots.items():
            fm, fe = main / f"{DS}_h{h}.csv.gz", extra / f"{DS}_h{h}.csv.gz"
            if not (fm.exists() and fe.exists()):
                continue
            a = pd.read_csv(fm, low_memory=False)
            a = a[a.method.isin(["dtcwr_aci", "fixed_2160"])]
            b = pd.read_csv(fe, low_memory=False)
            co = an.client_outcomes(pd.concat([a, b], ignore_index=True))
            piv = {k: co.pivot(index="series", columns="method", values=k)
                   for k in ("rmst", "winkler_L2", "coverage_L2", "worst_local_cov_L1")}
            rel = piv["winkler_L2"].div(piv["winkler_L2"]["fixed_2160"], axis=0)
            row = {"h": h, "cohort": tag,
                   "saocp_coverage_L2": float(piv["coverage_L2"]["saocp"].median()),
                   "saocp_rel_winkler_L2": float(rel["saocp"].median()),
                   "dtcwr_aci_rel_winkler_L2": float(rel["dtcwr_aci"].median()),
                   "saocp_rmst": float(piv["rmst"]["saocp"].median()),
                   "dtcwr_aci_rmst": float(piv["rmst"]["dtcwr_aci"].median()),
                   "saocp_worst_cov_L1": float(piv["worst_local_cov_L1"]["saocp"].median())}
            for k in ("rmst", "winkler_L2", "worst_local_cov_L1"):
                d = (piv[k]["dtcwr_aci"] - piv[k]["saocp"]).dropna().to_numpy(float)
                lo, hi = an.cluster_boot_median(d)
                row.update({f"{k}_median_diff": float(np.median(d)), f"{k}_ci_lo": lo, f"{k}_ci_hi": hi,
                            f"{k}_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0})
            out.append(row)
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "saocp_vs_dtcwr_aci.csv", index=False)
    pd.set_option("display.width", 250)
    print("### POST HOC (D17): SAOCP as an additional baseline; DTCW-R+ACI minus SAOCP (client medians)")
    print(df.round(4).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("tune", "test", "compare"), required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if a.stage == "tune":
        stage_tune(a.workers)
    elif a.stage == "test":
        stage_test(a.workers, a.allow_dirty)
    else:
        stage_compare()
