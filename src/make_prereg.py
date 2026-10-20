"""
Write PREREGISTRATION.md from the frozen validation-stage outputs (plan v5 Section 12, step 7).
Everything the test run will use is listed with a SHA-256 hash; the commit that adds the file
(and the git tag prereg-v1) is the pre-registration. The test segment has not been opened.

Run from the project root, on a clean working tree (everything else committed):
    python src/make_prereg.py
Then:
    git add PREREGISTRATION.md && git commit -m "Pre-registration" && git tag prereg-v1
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inject as ij  # noqa: E402
from run_calib_val import ALPHA, HORIZONS, LOCAL_W  # noqa: E402

OUT = Path("PREREGISTRATION.md")
FROZEN = [
    "data/processed/electricity_hourly.parquet", "data/processed/selected_clients.csv",
    "data/processed/excluded_clients.csv", "data/processed/preprocessing_summary.json",
    "data/processed/elec2_main.parquet", "data/processed/elec2_pub.parquet",
    "data/processed/elec2_summary.json",
    "results/forecasts/forecast_summary.json",
    *[f"results/forecasts/{d}_h{h}.parquet" for d in HORIZONS for h in HORIZONS[d]],
    *[f"results/models/{d}_h{h}.{x}" for d in HORIZONS for h in HORIZONS[d] for x in ("txt", "json")],
    "results/detector_calibration.json", "results/detector_choice.json",
    "results/tuning/selected.json",
    *[f"results/tuning/{d}_h{h}/dtcwr_H.json" for d in HORIZONS for h in HORIZONS[d]],
    "results/prereg/ttr_sensitivity.json", "results/prereg/ablation_H.json",
    "results/prereg/sigma_eps.json", "DECISIONS.md",
]
LABEL = {"fixed": "Fixed windows", "aci": "ACI", "rocp": "ROCP", "dtaci": "DtACI",
         "pid": "Conformal PID", "eci_cutoff": "ECI-cutoff", "driftocp": "DriftOCP",
         "dtcwr": "DTCW-R", "dtcwr_aci": "DTCW-R+ACI"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def fmt(cfg: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))


def main():
    status = git("status", "--porcelain")
    dirty = [l for l in status.splitlines() if l and not l.endswith("PREREGISTRATION.md")] \
        if status != "unavailable" else []
    if dirty:
        raise SystemExit("Working tree not clean; commit first:\n" + "\n".join(dirty))
    missing = [p for p in FROZEN if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing frozen files:\n" + "\n".join(missing))

    sel = json.loads(Path("results/tuning/selected.json").read_text())
    ttr = json.loads(Path("results/prereg/ttr_sensitivity.json").read_text())
    abl = json.loads(Path("results/prereg/ablation_H.json").read_text())
    sig = json.loads(Path("results/prereg/sigma_eps.json").read_text())
    choice = json.loads(Path("results/detector_choice.json").read_text())
    cell = ttr["chosen"]
    L = []
    w = L.append

    w("# Pre-registration — ICREST 2027 (drift-triggered adaptive conformal calibration)\n")
    w(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from code commit "
      f"`{git('rev-parse', 'HEAD')}`. The commit that adds this file, tagged `prereg-v1`, is the "
      "pre-registration. **The test segment has not been opened.** Every choice below was made on "
      "validation data; the decision log is `DECISIONS.md` (D1–D12).\n")

    w("## 1. Hypotheses and decision rules\n")
    w("- **H1:** at matched false-alarm rate, the magnitude signal detects variance shifts sooner "
      "than the miss-indicator signal.")
    w("- **H2:** for mean (level, gradual) shifts this advantage is smaller.")
    w("- **H3:** a change-point-anchored window gives narrower post-alarm intervals than a hard "
      "reset, with similar recovery.")
    w("- **Main comparison:** DTCW-R and DTCW-R+ACI against each baseline on the primary metrics "
      "of Section 5. Validation suggested an advantage at the day-ahead horizon (h = 24 / 48) and "
      "none at h = 1; this is stated now so that neither outcome can be reframed after the test.")
    w("- **2×2 decision rules (plan 9.1):** signal effect significant → evidence for H1/H2; window "
      "effect significant → evidence for H3; neither → report the null result under the "
      "benchmark title.\n")

    w("## 2. Data, splits, forecasts (frozen)\n")
    w("- Electricity (UCI LD2011–2014): 2012–2014 hourly, DST repair D1, 50 clients drawn with "
      "seed 2027 from 314 eligible. Elec2: all 45,312 half-hours (P-main); first 2,000 rows (P-pub, "
      "Table C only, D2).")
    w("- Chronological 70/10/20 split per series. Horizons: Electricity h ∈ {1, 24}, Elec2 h ∈ {1, 48} (D3).")
    w("- LightGBM direct, frozen after training on the train segment; never refit.")
    w(f"- Miscoverage α = {ALPHA}. Score |y − ŷ|, symmetric interval.\n")

    w("## 3. Frozen method configurations\n")
    w("Selected on validation by the D10 rule (16 configurations per family, same streams, same "
      "criterion). Infinite intervals are clipped to the largest known score for every method (D10).\n")
    for key in sorted(sel):
        pw = choice.get(key, {}).get("winner", "?")
        w(f"**{key}** (DTCW-R detector variant: {pw}, D8)\n")
        w("| Family | Configuration | Validation score | Unshifted coverage |")
        w("|---|---|---|---|")
        for fam in ("fixed", "aci", "rocp", "dtaci", "pid", "eci_cutoff", "driftocp", "dtcwr", "dtcwr_aci"):
            s = sel[key].get(fam)
            if s is None:
                continue
            if fam == "fixed":
                w(f"| {LABEL[fam]} | W ∈ {{{', '.join(str(c['W']) for c in s['configs'])}}} (a priori) | — | — |")
            else:
                flag = f" ({s['flag']})" if s.get("flag") else ""
                w(f"| {LABEL[fam]} | {fmt(s['config'])} | {s['score']:.3f} | {s['cov_unshifted']:.3f}{flag} |")
        w("")

    w("## 4. Test protocol\n")
    w(f"- One stream per series = validation targets followed by test targets. Warm-up W0 = 500; "
      "methods run from the start of validation; **only test targets are evaluated**. Local "
      f"metrics use one calendar week (w = {LOCAL_W['electricity']} hourly, {LOCAL_W['elec2']} half-hourly) "
      "on full windows only.")
    w(f"- **L1 controlled shifts.** One shift per replay at {', '.join(f'{int(100 * f)}%' for f in ij.TEST_FRACS)} "
      f"of the test segment; settings {', '.join(f'{k} {v}' for k, v in ij.SETTINGS)} "
      "(variance = κ, level = δ, gradual = ramp height, in σ̂ units). **Primary:** target-level "
      "injection with the frozen forecaster (D12). **Control:** residual-level injection (separate panel). "
      f"TTR cap {ij.TTR_CAP}; follow-up ≥ cap + L asserted for every position.")
    w("- **L2** untouched test segments. **L3** (exploratory): PELT with BIC penalty on the full "
      "series, offline, in a module no method can import; TTR around detected points; no confirmatory claims.")
    w("- Variance injection uses the σ_ε calibrated on validation (D12). On test the achieved ratio "
      "is reported per client; a setting whose test ratio differs from κ² by more than 10% is "
      "labelled by its achieved ratio.\n")
    w("| Stream | κ = 1.5 | κ = 2 | κ = 3 |")
    w("|---|---|---|---|")
    for key in sorted(sig):
        cells = []
        for k in ("1.5", "2.0", "3.0"):
            ms = sorted(v[k]["m"] for v in sig[key].values())
            ok = sum(v[k]["within_tol"] for v in sig[key].values())
            cells.append(f"σ_ε/σ̂ median {ms[len(ms) // 2]:.3f}; {ok}/{len(ms)} within 2%")
        w(f"| {key} | " + " | ".join(cells) + " |")
    w("")

    w("## 5. Metrics and TTR\n")
    w("- Primary: coverage, coverage error, Winkler, TTR (RMST up to the cap, % recovered), worst "
      "local coverage. Secondary: longest under-coverage run, width at matched coverage (±1 pp), "
      "detection delay, false alarms per 1,000 steps, runtime.")
    w("- Reading order: coverage error and worst local coverage → Winkler → width at matched coverage.")
    w(f"- **TTR cell (D11):** w = {cell['w_h']} h, δ = {cell['delta']}, L = {cell['L_over_w']}·w "
      f"({ttr['reason']}). All 12 cells are reported on test; a method ranking is claimed only if "
      "Kendall τ ≥ 0.7 between the rankings of all cells.")
    w("- Secondary TTR: recovery relative to the method's own pre-shift local coverage.\n")

    w("## 6. Statistics\n")
    w("- Unit = client (Electricity, 50). Per client, outcomes are averaged over the 5 positions "
      "within a shift setting (RMST for TTR).")
    w("- DTCW-R vs each baseline: Wilcoxon signed-rank on client differences; Holm within each "
      "metric family; median difference, 95% cluster-bootstrap CI (2,000 client resamples), "
      "rank-biserial effect size.")
    w("- Elec2 (one series): moving-block bootstrap (block 48, 1,000 resamples), effect sizes "
      "with CIs only, no significance claims.")
    w("- Overall ranking: Friedman test + Nemenyi critical-difference diagram on client-level "
      "Winkler and RMST.\n")

    w("## 7. 2×2 ablation (thresholds frozen, D11)\n")
    w("Primary outcome: client-level RMST of TTR, L1 target-level, h = 1, averaged over positions "
      "and settings within shift type. Signal = ½[(B−R)+(D−A)], window = ½[(A−R)+(D−B)], "
      "interaction = (D−B)−(A−R); Wilcoxon + cluster bootstrap; the 3 effects on the primary "
      "outcome form one Holm family. Mixed model as a sensitivity check only. Reported per shift type.\n")
    w("| Stream | D magnitude+anchored | B magnitude+reset | A miss+anchored | R miss+reset |")
    w("|---|---|---|---|---|")
    for key in sorted(abl):
        a = abl[key]
        c = lambda x: f"H {a[x]['H']} (fa {a[x].get('fa_at_H', float('nan')):.2f})"   # noqa: E731
        w(f"| {key} | {c('D_magnitude_anchored')} | {c('B_magnitude_reset')} | "
          f"{c('A_miss_anchored')} | {c('R_miss_reset')} |")
    w("\nAll four cells share the selected DTCW-R configuration (W_S, W_L, k, c); the plan's "
      "16-configuration budget per cell is replaced by this shared configuration so that only the "
      "two factors differ (deviation, recorded here). Achieved false-alarm rates are reported.\n")

    w("## 8. Deviations from plan v5 and known limitations (declared before the test)\n")
    w("- D6/D7/D8: H calibrated by block bootstrap; prewhitened detector chosen for three of four "
      "streams; prewhitening misses persistent level shifts at Elec2 h = 48.")
    w("- D9: ECI follows the authors' code, not eq. 5 as printed; PID without scorecaster in P-main; "
      "all baselines use the PID code's delayed-feedback rule.")
    w("- D10: DTCW-R grid has no H axis (H calibrated per configuration).")
    w("- D11: the definition of the 'strictest' TTR cell was written in code before the run but "
      "committed together with its result (commit 3f22f5e). From D12 on, code is committed before it is run.")
    w("- GRU robustness forecaster (plan 7.1) is not part of this pre-registration; if run later it "
      "is exploratory.")
    w("- The test is run once. Any later change to code or configuration after the test is "
      "opened is reported as a post-hoc analysis.\n")

    w("## 9. Frozen files (SHA-256)\n")
    w("| File | SHA-256 |")
    w("|---|---|")
    for p in FROZEN:
        w(f"| `{p}` | `{sha(Path(p))}` |")
    w("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT} ({len(FROZEN)} frozen files hashed). Next:\n"
          "  git add PREREGISTRATION.md\n  git commit -m \"Pre-registration\"\n  git tag prereg-v1")


if __name__ == "__main__":
    main()
