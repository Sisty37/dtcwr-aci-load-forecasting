# Pre-registration — ICREST 2027 (drift-triggered adaptive conformal calibration)

Cohort / datasets: **electricity**; tag to create: `prereg-c3`.

Generated 2026-10-21 18:07 UTC from code commit `34424d941200bcc993cd998185c8aa92b00ce006`. The commit that adds this file, tagged `prereg-v1`, is the pre-registration. **The test segment has not been opened.** Every choice below was made on validation data; the decision log is `DECISIONS.md` (D1–D12).

## 1. Hypotheses and decision rules

- **H1:** at matched false-alarm rate, the magnitude signal detects variance shifts sooner than the miss-indicator signal.
- **H2:** for mean (level, gradual) shifts this advantage is smaller.
- **H3:** a change-point-anchored window gives narrower post-alarm intervals than a hard reset, with similar recovery.
- **Main comparison:** DTCW-R and DTCW-R+ACI against each baseline on the primary metrics of Section 5. Validation suggested an advantage at the day-ahead horizon (h = 24 / 48) and none at h = 1; this is stated now so that neither outcome can be reframed after the test.
- **2×2 decision rules (plan 9.1):** signal effect significant → evidence for H1/H2; window effect significant → evidence for H3; neither → report the null result under the benchmark title.

## 2. Data, splits, forecasts (frozen)

- **Cohort 3 (robustness at 95% coverage, D18):** Electricity only; 50 NEW clients drawn with seed 2029 from the 214 eligible clients used by neither cohort 1 nor cohort 2 (lists in data/processed/selected_clients.csv, cohort1_clients.csv, cohort2_clients.csv). Same preprocessing (D1); LightGBM re-trained on these clients' train segments; alpha = 0.05. Predictions S1-S4: PREREG_C3_PREDICTIONS.md.
- Chronological 70/10/20 split per series. Horizons: Electricity h ∈ {1, 24}, Elec2 h ∈ {1, 48} (D3).
- LightGBM direct, frozen after training on the train segment; never refit.
- Miscoverage α = 0.05. Score |y − ŷ|, symmetric interval.

## 3. Frozen method configurations

Selected on validation by the D10 rule (16 configurations per family, same streams, same criterion). Infinite intervals are clipped to the largest known score for every method (D10).

**electricity_h1** (DTCW-R detector variant: prewhiten, D8)

| Family | Configuration | Validation score | Unshifted coverage |
|---|---|---|---|
| Fixed windows | W ∈ {168, 2160, 720} (a priori) | — | — |
| ACI | gamma=0.00398 | 14.799 | 0.949 |
| ROCP | W=51 | 14.631 | 0.942 |
| DtACI | I=100, gscale=0.25 | 14.564 | 0.949 |
| Conformal PID | KI=0, lr=0.05 | 14.542 | 0.949 |
| ECI-cutoff | h_cut=0.25, lr=0.1 | 14.537 | 0.948 |
| DriftOCP | min_len=5, sigma=0.5 | 15.407 | 0.942 |
| DTCW-R | H=7, W_L=720, W_S=168, c=72, k=0.5 | 14.927 | 0.951 |
| DTCW-R+ACI | H=7, W_L=720, W_S=168, c=72, gamma=0.00631, k=0.5 | 14.831 | 0.950 |

**electricity_h24** (DTCW-R detector variant: plain, D8)

| Family | Configuration | Validation score | Unshifted coverage |
|---|---|---|---|
| Fixed windows | W ∈ {168, 2160, 720} (a priori) | — | — |
| ACI | gamma=0.01 | 15.656 | 0.953 |
| ROCP | W=364 | 16.212 | 0.940 |
| DtACI | I=1000, gscale=0.25 | 16.647 | 0.946 |
| Conformal PID | KI=0, lr=0.05 | 17.360 | 0.949 |
| ECI-cutoff | h_cut=0.25, lr=0.05 | 17.624 | 0.952 |
| DriftOCP | min_len=40, sigma=0.5 | 17.119 | 0.930 |
| DTCW-R | H=20, W_L=720, W_S=168, c=72, k=0.5 | 16.140 | 0.946 |
| DTCW-R+ACI | H=20, W_L=720, W_S=168, c=72, gamma=0.00631, k=0.5 | 15.759 | 0.953 |

## 4. Test protocol

- One stream per series = validation targets followed by test targets. Warm-up W0 = 500; methods run from the start of validation; **only test targets are evaluated**. Local metrics use one calendar week (w = 168 hourly, 336 half-hourly) on full windows only.
- **L1 controlled shifts.** One shift per replay at 15%, 30%, 45%, 60%, 75% of the test segment; settings variance 1.5, variance 2.0, variance 3.0, level 1.0, level 2.0, level 3.0, gradual 2.0 (variance = κ, level = δ, gradual = ramp height, in σ̂ units). **Primary:** target-level injection with the frozen forecaster (D12). **Control:** residual-level injection (separate panel). TTR cap 1000; follow-up ≥ cap + L asserted for every position.
- **L2** untouched test segments. **L3** (exploratory): PELT with BIC penalty on the full series, offline, in a module no method can import; TTR around detected points; no confirmatory claims.
- Variance injection uses the σ_ε calibrated on validation (D12). On test the achieved ratio is reported per client; a setting whose test ratio differs from κ² by more than 10% is labelled by its achieved ratio.

| Stream | κ = 1.5 | κ = 2 | κ = 3 |
|---|---|---|---|
| electricity_h1 | σ_ε/σ̂ median 0.906; 50/50 within 2% | σ_ε/σ̂ median 1.406; 50/50 within 2% | σ_ε/σ̂ median 2.250; 50/50 within 2% |
| electricity_h24 | σ_ε/σ̂ median 0.938; 50/50 within 2% | σ_ε/σ̂ median 1.469; 50/50 within 2% | σ_ε/σ̂ median 2.406; 50/50 within 2% |

## 5. Metrics and TTR

- Primary: coverage, coverage error, Winkler, TTR (RMST up to the cap, % recovered), worst local coverage. Secondary: longest under-coverage run, width at matched coverage (±1 pp), detection delay, false alarms per 1,000 steps, runtime.
- Reading order: coverage error and worst local coverage → Winkler → width at matched coverage.
- **TTR cell (D11):** w = 168 h, δ = 0.05, L = 1.0·w (primary cell meets the rule). All 12 cells are reported on test; a method ranking is claimed only if Kendall τ ≥ 0.7 between the rankings of all cells.
- Secondary TTR: recovery relative to the method's own pre-shift local coverage.

## 6. Statistics

- Unit = client (Electricity, 50). Per client, outcomes are averaged over the 5 positions within a shift setting (RMST for TTR).
- DTCW-R vs each baseline: Wilcoxon signed-rank on client differences; Holm within each metric family; median difference, 95% cluster-bootstrap CI (2,000 client resamples), rank-biserial effect size.
- Elec2 (one series): moving-block bootstrap (block 48, 1,000 resamples), effect sizes with CIs only, no significance claims.
- Overall ranking: Friedman test + Nemenyi critical-difference diagram on client-level Winkler and RMST.

## 7. 2×2 ablation (thresholds frozen, D11)

Primary outcome: client-level RMST of TTR, L1 target-level, h = 1, averaged over positions and settings within shift type. Signal = ½[(B−R)+(D−A)], window = ½[(A−R)+(D−B)], interaction = (D−B)−(A−R); Wilcoxon + cluster bootstrap; the 3 effects on the primary outcome form one Holm family. Mixed model as a sensitivity check only. Reported per shift type.

| Stream | D magnitude+anchored | B magnitude+reset | A miss+anchored | R miss+reset |
|---|---|---|---|---|
| electricity_h1 | H 7 (fa 0.28) | H 7 (fa 0.28) | H 1.35 (fa 0.48) | H 1.45 (fa 0.33) |
| electricity_h24 | H 20 (fa 0.42) | H 20 (fa 0.42) | H 2.4 (fa 0.45) | H 2.5 (fa 0.48) |

All four cells share the selected DTCW-R configuration (W_S, W_L, k, c); the plan's 16-configuration budget per cell is replaced by this shared configuration so that only the two factors differ (deviation, recorded here). Achieved false-alarm rates are reported.

## 8. Deviations from plan v5 and known limitations (declared before the test)

- D6/D7/D8: H calibrated by block bootstrap; prewhitened detector chosen for three of four streams; prewhitening misses persistent level shifts at Elec2 h = 48.
- D9: ECI follows the authors' code, not eq. 5 as printed; PID without scorecaster in P-main; all baselines use the PID code's delayed-feedback rule.
- D10: DTCW-R grid has no H axis (H calibrated per configuration).
- D11: the definition of the 'strictest' TTR cell was written in code before the run but committed together with its result (commit 3f22f5e). From D12 on, code is committed before it is run.
- GRU robustness forecaster (plan 7.1) is not part of this pre-registration; if run later it is exploratory.
- The test is run once. Any later change to code or configuration after the test is opened is reported as a post-hoc analysis.

## 9. Frozen files (SHA-256)

| File | SHA-256 |
|---|---|
| `data/processed/electricity_hourly.parquet` | `2fe2336bc7504392f041c63e69a0837b0fc66803d35ba6b5f05192d299ada94a` |
| `data/processed/selected_clients.csv` | `d066de9478372d1282cb014764af6f35ac52528b3666f34aef31207452b175a8` |
| `data/processed/excluded_clients.csv` | `310560aa1e6d5904a404c4452f90ae471405c1291a68a9e93a60a1d7500a87fe` |
| `data/processed/preprocessing_summary.json` | `6f75eda335f3e770858d693b4a46570942b8a66f6a008c361bef9ddd0b2023cc` |
| `data/processed/cohort1_clients.csv` | `c7c44a17f3669c1de61dc78bde57a731a04f37ac74cd0348305fbf310879508f` |
| `PREREG_C2_PREDICTIONS.md` | `7d82a0dd82b92bf69044bbfdb01f52114f92b3c7ceab7df984a44e75a038b559` |
| `data/processed/cohort2_clients.csv` | `60c6f074cc3dc63c6e64e20945704c2e5f6fcf3d197b828b14908a0c611f5d94` |
| `PREREG_C3_PREDICTIONS.md` | `15419bae8c08e7d0ef4f7bf234f74bf4ee8531ed806954b09391f2d0f3610cbe` |
| `results/forecasts/forecast_summary.json` | `9e76ee3b4b8167d86bb5c952ad68c884b5facb170beeadacb9300782638becc7` |
| `results/forecasts/electricity_h1.parquet` | `ef6152a96e37799139ff26cdb8d206cc46cab556c0741957cc7a63d7d50ca6b8` |
| `results/forecasts/electricity_h24.parquet` | `6210379d5e9744b6e71c595f1c3baef7999ef89fa1896792066053fad1c9578b` |
| `results/models/electricity_h1.txt` | `c82ca1e88ca6d6183c79ce91e5cf006af5c6d5dd0ff34dbae9e41f6a8646474c` |
| `results/models/electricity_h1.json` | `02120fe0e32106138ec89a66a5ee15ad40bc5e73a715562cf76021107072e20a` |
| `results/models/electricity_h24.txt` | `d54665f7a0973e9c0e9c553bee2f4f12be245be472c501f4b542d7fa347195c6` |
| `results/models/electricity_h24.json` | `0941d5bd61f277aff0ff0d71ed90db4a5d2fb8c428d52faa0968ccd16e885b6b` |
| `results/detector_calibration.json` | `a25f439d21c49f44de3e1ef7722a287b3613bff77389c0a6ffc445675f29f7ea` |
| `results/detector_choice.json` | `8734b262b723cd462f9b500509bbe7b513009f6caa17e457cf9572c3da5bd689` |
| `results/tuning/selected.json` | `7ebe9271ecb12a159b3141157f9d982a8c3add44642c2e1ffc1e36ed193968a3` |
| `results/tuning/electricity_h1/dtcwr_H.json` | `79d5bee0b2f1d92ee049c85a5a22ab5f5e7eee2a76b1ba184462a216ffac07ce` |
| `results/tuning/electricity_h24/dtcwr_H.json` | `37f5c79b8769d8aeb86404960d356033e33ac4b4415a731eeff12a4a1d4c325b` |
| `results/prereg/ttr_sensitivity.json` | `15557655b62253edb86081d6cfdce28841decc980d8eec07c4191d6dcb199dc0` |
| `results/prereg/ablation_H.json` | `acf5f62546745937a3a910ac831e5bc6084e432607d08139a07ddc661b9a3ccb` |
| `results/prereg/sigma_eps.json` | `16e8d10012d9dc90e412958a6678a2c73f3cf2fc514bd4ee392ece5eb76b0d81` |
| `DECISIONS.md` | `834428812dd63378909c921cc3bf2ce64c14e82977b3abec85e5b0f05d96e2ef` |
