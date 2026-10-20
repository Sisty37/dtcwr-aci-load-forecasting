# Pre-registration — ICREST 2027 (drift-triggered adaptive conformal calibration)

Generated 2026-10-20 12:30 UTC from code commit `b96495d27496545c77520423b1d7e8b0451d04bb`. The commit that adds this file, tagged `prereg-v1`, is the pre-registration. **The test segment has not been opened.** Every choice below was made on validation data; the decision log is `DECISIONS.md` (D1–D12).

## 1. Hypotheses and decision rules

- **H1:** at matched false-alarm rate, the magnitude signal detects variance shifts sooner than the miss-indicator signal.
- **H2:** for mean (level, gradual) shifts this advantage is smaller.
- **H3:** a change-point-anchored window gives narrower post-alarm intervals than a hard reset, with similar recovery.
- **Main comparison:** DTCW-R and DTCW-R+ACI against each baseline on the primary metrics of Section 5. Validation suggested an advantage at the day-ahead horizon (h = 24 / 48) and none at h = 1; this is stated now so that neither outcome can be reframed after the test.
- **2×2 decision rules (plan 9.1):** signal effect significant → evidence for H1/H2; window effect significant → evidence for H3; neither → report the null result under the benchmark title.

## 2. Data, splits, forecasts (frozen)

- Electricity (UCI LD2011–2014): 2012–2014 hourly, DST repair D1, 50 clients drawn with seed 2027 from 314 eligible. Elec2: all 45,312 half-hours (P-main); first 2,000 rows (P-pub, Table C only, D2).
- Chronological 70/10/20 split per series. Horizons: Electricity h ∈ {1, 24}, Elec2 h ∈ {1, 48} (D3).
- LightGBM direct, frozen after training on the train segment; never refit.
- Miscoverage α = 0.1. Score |y − ŷ|, symmetric interval.

## 3. Frozen method configurations

Selected on validation by the D10 rule (16 configurations per family, same streams, same criterion). Infinite intervals are clipped to the largest known score for every method (D10).

**elec2_h1** (DTCW-R detector variant: prewhiten, D8)

| Family | Configuration | Validation score | Unshifted coverage |
|---|---|---|---|
| Fixed windows | W ∈ {1440, 336, 4320} (a priori) | — | — |
| ACI | gamma=0.00631 | 8.662 | 0.899 |
| ROCP | W=57 | 8.648 | 0.896 |
| DtACI | I=100, gscale=0.25 | 8.618 | 0.901 |
| Conformal PID | KI=0, lr=0.05 | 8.710 | 0.899 |
| ECI-cutoff | h_cut=0.25, lr=0.05 | 8.686 | 0.900 |
| DriftOCP | min_len=20, sigma=0.5 | 8.738 | 0.904 |
| DTCW-R | H=7, W_L=672, W_S=96, c=0, k=0.5 | 8.648 | 0.899 |
| DTCW-R+ACI | H=7, W_L=672, W_S=96, c=0, gamma=0.0001, k=0.5 | 8.649 | 0.898 |

**elec2_h48** (DTCW-R detector variant: plain, D8)

| Family | Configuration | Validation score | Unshifted coverage |
|---|---|---|---|
| Fixed windows | W ∈ {1440, 336, 4320} (a priori) | — | — |
| ACI | gamma=0.0631 | 7.400 | 0.910 |
| ROCP | W=196 | 7.527 | 0.886 |
| DtACI | I=1000, gscale=0.5 | 7.720 | 0.882 |
| Conformal PID | KI=0, lr=0.05 | 11.098 | 0.896 |
| ECI-cutoff | h_cut=0.25, lr=0.05 | 9.407 | 0.901 |
| DriftOCP | min_len=10, sigma=2 | 8.108 | 0.885 |
| DTCW-R | H=60, W_L=1440, W_S=336, c=144, k=0.5 | 7.482 | 0.890 |
| DTCW-R+ACI | H=60, W_L=1440, W_S=336, c=144, gamma=0.0158, k=0.5 | 7.365 | 0.891 |

**electricity_h1** (DTCW-R detector variant: prewhiten, D8)

| Family | Configuration | Validation score | Unshifted coverage |
|---|---|---|---|
| Fixed windows | W ∈ {168, 2160, 720} (a priori) | — | — |
| ACI | gamma=0.00631 | 12.190 | 0.900 |
| ROCP | W=51 | 12.191 | 0.885 |
| DtACI | I=100, gscale=0.25 | 12.080 | 0.898 |
| Conformal PID | KI=0, lr=0.05 | 12.105 | 0.899 |
| ECI-cutoff | h_cut=0.25, lr=0.05 | 12.094 | 0.898 |
| DriftOCP | min_len=10, sigma=1 | 12.579 | 0.889 |
| DTCW-R | H=7, W_L=336, W_S=96, c=0, k=0.5 | 12.272 | 0.903 |
| DTCW-R+ACI | H=7, W_L=336, W_S=96, c=0, gamma=0.01, k=0.5 | 12.270 | 0.900 |

**electricity_h24** (DTCW-R detector variant: prewhiten, D8)

| Family | Configuration | Validation score | Unshifted coverage |
|---|---|---|---|
| Fixed windows | W ∈ {168, 2160, 720} (a priori) | — | — |
| ACI | gamma=0.01 | 12.674 | 0.901 |
| ROCP | W=174 | 13.000 | 0.891 |
| DtACI | I=100, gscale=0.25 | 13.360 | 0.892 |
| Conformal PID | KI=0, lr=0.05 | 14.978 | 0.900 |
| ECI-cutoff | h_cut=0.25, lr=0.05 | 14.134 | 0.899 |
| DriftOCP | min_len=20, sigma=1 | 13.457 | 0.880 |
| DTCW-R | H=20, W_L=720, W_S=168, c=24, k=0.25 | 13.060 | 0.898 |
| DTCW-R+ACI | H=20, W_L=720, W_S=168, c=24, gamma=0.0158, k=0.25 | 12.522 | 0.911 |

## 4. Test protocol

- One stream per series = validation targets followed by test targets. Warm-up W0 = 500; methods run from the start of validation; **only test targets are evaluated**. Local metrics use one calendar week (w = 168 hourly, 336 half-hourly) on full windows only.
- **L1 controlled shifts.** One shift per replay at 15%, 30%, 45%, 60%, 75% of the test segment; settings variance 1.5, variance 2.0, variance 3.0, level 1.0, level 2.0, level 3.0, gradual 2.0 (variance = κ, level = δ, gradual = ramp height, in σ̂ units). **Primary:** target-level injection with the frozen forecaster (D12). **Control:** residual-level injection (separate panel). TTR cap 1000; follow-up ≥ cap + L asserted for every position.
- **L2** untouched test segments. **L3** (exploratory): PELT with BIC penalty on the full series, offline, in a module no method can import; TTR around detected points; no confirmatory claims.
- Variance injection uses the σ_ε calibrated on validation (D12). On test the achieved ratio is reported per client; a setting whose test ratio differs from κ² by more than 10% is labelled by its achieved ratio.

| Stream | κ = 1.5 | κ = 2 | κ = 3 |
|---|---|---|---|
| elec2_h1 | σ_ε/σ̂ median 0.750; 1/1 within 2% | σ_ε/σ̂ median 1.156; 1/1 within 2% | σ_ε/σ̂ median 1.875; 1/1 within 2% |
| elec2_h48 | σ_ε/σ̂ median 0.625; 1/1 within 2% | σ_ε/σ̂ median 1.000; 1/1 within 2% | σ_ε/σ̂ median 1.688; 1/1 within 2% |
| electricity_h1 | σ_ε/σ̂ median 0.938; 50/50 within 2% | σ_ε/σ̂ median 1.469; 50/50 within 2% | σ_ε/σ̂ median 2.375; 49/50 within 2% |
| electricity_h24 | σ_ε/σ̂ median 1.062; 50/50 within 2% | σ_ε/σ̂ median 1.688; 50/50 within 2% | σ_ε/σ̂ median 2.750; 50/50 within 2% |

## 5. Metrics and TTR

- Primary: coverage, coverage error, Winkler, TTR (RMST up to the cap, % recovered), worst local coverage. Secondary: longest under-coverage run, width at matched coverage (±1 pp), detection delay, false alarms per 1,000 steps, runtime.
- Reading order: coverage error and worst local coverage → Winkler → width at matched coverage.
- **TTR cell (D11):** w = 84 h, δ = 0.025, L = 1.0·w (strictest cell meeting the rule). All 12 cells are reported on test; a method ranking is claimed only if Kendall τ ≥ 0.7 between the rankings of all cells.
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
| elec2_h1 | H 7 (fa 0.49) | H 7 (fa 0.49) | H 1.7 (fa 0.29) | H 1.7 (fa 0.31) |
| elec2_h48 | H 60 (fa 0.27) | H 60 (fa 0.27) | H 4.7 (fa 0.48) | H 4.85 (fa 0.49) |
| electricity_h1 | H 7 (fa 0.30) | H 7 (fa 0.30) | H 1.7 (fa 0.50) | H 1.8 (fa 0.45) |
| electricity_h24 | H 20 (fa 0.25) | H 20 (fa 0.25) | H 2.85 (fa 0.49) | H 3.0 (fa 0.41) |

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
| `data/processed/selected_clients.csv` | `3d61c7514b83fb74d4406f64aff426e778fcb1214fc5bed7c7bb4ba332974449` |
| `data/processed/excluded_clients.csv` | `310560aa1e6d5904a404c4452f90ae471405c1291a68a9e93a60a1d7500a87fe` |
| `data/processed/preprocessing_summary.json` | `6f75eda335f3e770858d693b4a46570942b8a66f6a008c361bef9ddd0b2023cc` |
| `data/processed/elec2_main.parquet` | `ade4ebe73f601d576c8c1f76cbe7a08dbcf48d09b1eb03ef080c6e47083f90f4` |
| `data/processed/elec2_pub.parquet` | `c375211a1e277ee4d95f0dffd3194b74812ef2e19c99601a01fa0bd39cfae83e` |
| `data/processed/elec2_summary.json` | `02355146e82303d9e20e024dd2ee7b8ce3c896a5dbd0118f34b21b283d262872` |
| `results/forecasts/forecast_summary.json` | `a0736b2d139aa7bb73a17b2a2b7ce9684280dcb6b8c9514d6284d911e6a86864` |
| `results/forecasts/electricity_h1.parquet` | `0bfb106a86c3dc5111279eaa26d7960ed5ad3238a28efe879cdef3019ab6dced` |
| `results/forecasts/electricity_h24.parquet` | `c4b524d7b003e25c35d211f66a5d0bf469debfeb19af99054e7494a4419ceff3` |
| `results/forecasts/elec2_h1.parquet` | `35a0d86fb5055b18dbd2066eb806fd1f76d29e77cd0b265129b9405c377acfaa` |
| `results/forecasts/elec2_h48.parquet` | `dfe6eaf5db7cb2868bbdca09a9bd8a6b92662678c1a662f4e0bcc6e163242c3b` |
| `results/models/electricity_h1.txt` | `20edd3b15a7e6f20b5097a0d89df3a0680ab9fb0a9436b100817b701050a92de` |
| `results/models/electricity_h1.json` | `824dea224a948bfbf7bd6f2681f1c07878703a584e840c24b92ce7d41544d015` |
| `results/models/electricity_h24.txt` | `4345e5dddbdbbb565677fdb7fa004f6fa0b8eb7169b83a7b0516579fa2f1fe2c` |
| `results/models/electricity_h24.json` | `96b942801d61fe85acc3a83c559e5075222f04b2c6cc267fa30eb9738cf7e701` |
| `results/models/elec2_h1.txt` | `5eb8557c437ddda9babea3d3801d0df5d96c4c36ef1ca115ceac5fd87f11f056` |
| `results/models/elec2_h1.json` | `133235cfb34ecd2ca8cdd07fedd8c665658ef9a7039df9674728814d76bb93e6` |
| `results/models/elec2_h48.txt` | `523d7d1d9cc29542ed1b234a35d51203281d1dae1d3b4883f5bd0b98e650ac02` |
| `results/models/elec2_h48.json` | `94beda6e17a4591758ca03c4a0551cb11beda0b601e44b3d5ed61e6e03068121` |
| `results/detector_calibration.json` | `7384d0496a4855cd8be03cf3cc4f8f26faca86c3791f98b3e82fbf3bba45a73a` |
| `results/detector_choice.json` | `dd87f09ac303035eb58f2377725939786d773982c0254699b5b47ae0920e0ae0` |
| `results/tuning/selected.json` | `2f80a652a9d2d1ca97206641a669564d54618c34e054676707d7ac4cc26b6b56` |
| `results/tuning/electricity_h1/dtcwr_H.json` | `c0fa878808a13e101131c0040c116169c75a3710a5190df86cd990c5395173f6` |
| `results/tuning/electricity_h24/dtcwr_H.json` | `2e8edfdfe52d0a776aa11e9ea0c35a4921a0f8f6f19c3c1ab972893f262c422b` |
| `results/tuning/elec2_h1/dtcwr_H.json` | `5d9b7668c128edbea10471643b54ac400d1f858a4ce020ff215dd33fe0ebf9eb` |
| `results/tuning/elec2_h48/dtcwr_H.json` | `b22f55badcf777144d2c5d018ce81e2752ec0034a3f9f5e1f8753929651f6c6a` |
| `results/prereg/ttr_sensitivity.json` | `e9f317c5ad5b0ac2ec5b61d703a43b66fbefa65c9e225fed4b44e9f9b68a377a` |
| `results/prereg/ablation_H.json` | `b72142fd1ada2496b4ba6a7c22d1377b767143daddcf225b1ee74f644a0821c6` |
| `results/prereg/sigma_eps.json` | `53fb0e7e7fe3cfde8cd916762d1ff2fc5f0bcea14385a4a8595b284564325b26` |
| `DECISIONS.md` | `f90151a2f9e598eee71bd4a49a2d5dfe2c43234d654ea1536f25c7d2e6be0123` |
