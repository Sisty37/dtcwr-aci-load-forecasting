# Pre-registration — ICREST 2027 (drift-triggered adaptive conformal calibration)

Cohort / datasets: **electricity**; tag to create: `prereg-c2`.

Generated 2026-10-20 18:52 UTC from code commit `0d566cc9329f6a03be65cb45d11f1129996dc565`. The commit that adds this file, tagged `prereg-v1`, is the pre-registration. **The test segment has not been opened.** Every choice below was made on validation data; the decision log is `DECISIONS.md` (D1–D12).

## 1. Hypotheses and decision rules

- **H1:** at matched false-alarm rate, the magnitude signal detects variance shifts sooner than the miss-indicator signal.
- **H2:** for mean (level, gradual) shifts this advantage is smaller.
- **H3:** a change-point-anchored window gives narrower post-alarm intervals than a hard reset, with similar recovery.
- **Main comparison:** DTCW-R and DTCW-R+ACI against each baseline on the primary metrics of Section 5. Validation suggested an advantage at the day-ahead horizon (h = 24 / 48) and none at h = 1; this is stated now so that neither outcome can be reframed after the test.
- **2×2 decision rules (plan 9.1):** signal effect significant → evidence for H1/H2; window effect significant → evidence for H3; neither → report the null result under the benchmark title.

## 2. Data, splits, forecasts (frozen)

- **Cohort 2 (replication, D14):** Electricity only; 50 FRESH clients drawn with seed 2028 from the 264 eligible clients not used in cohort 1 (lists in data/processed/selected_clients.csv and data/processed/cohort1_clients.csv). Same preprocessing (D1). LightGBM re-trained on these clients' train segments with the same procedure.
- Chronological 70/10/20 split per series. Horizons: Electricity h ∈ {1, 24}, Elec2 h ∈ {1, 48} (D3).
- LightGBM direct, frozen after training on the train segment; never refit.
- Miscoverage α = 0.1. Score |y − ŷ|, symmetric interval.

## 3. Frozen method configurations

Selected on validation by the D10 rule (16 configurations per family, same streams, same criterion). Infinite intervals are clipped to the largest known score for every method (D10).

**electricity_h1** (DTCW-R detector variant: prewhiten, D8)

| Family | Configuration | Validation score | Unshifted coverage |
|---|---|---|---|
| Fixed windows | W ∈ {168, 2160, 720} (a priori) | — | — |
| ACI | gamma=0.00631 | 13.215 | 0.899 |
| ROCP | W=51 | 13.171 | 0.885 |
| DtACI | I=100, gscale=0.5 | 13.117 | 0.898 |
| Conformal PID | KI=0, lr=0.05 | 13.129 | 0.899 |
| ECI-cutoff | h_cut=0.25, lr=0.05 | 13.115 | 0.898 |
| DriftOCP | min_len=40, sigma=0.5 | 13.627 | 0.891 |
| DTCW-R | H=6, W_L=336, W_S=96, c=0, k=0.5 | 13.275 | 0.903 |
| DTCW-R+ACI | H=6, W_L=336, W_S=96, c=0, gamma=0.01, k=0.5 | 13.273 | 0.900 |

**electricity_h24** (DTCW-R detector variant: plain, D8)

| Family | Configuration | Validation score | Unshifted coverage |
|---|---|---|---|
| Fixed windows | W ∈ {168, 2160, 720} (a priori) | — | — |
| ACI | gamma=0.0158 | 13.019 | 0.906 |
| ROCP | W=174 | 13.438 | 0.890 |
| DtACI | I=100, gscale=0.25 | 14.019 | 0.891 |
| Conformal PID | KI=0, lr=0.05 | 15.339 | 0.899 |
| ECI-cutoff | h_cut=0.25, lr=0.05 | 14.273 | 0.896 |
| DriftOCP | min_len=40, sigma=1 | 14.031 | 0.874 (no config within coverage tolerance) |
| DTCW-R | H=20, W_L=720, W_S=168, c=72, k=0.5 | 13.546 | 0.897 |
| DTCW-R+ACI | H=20, W_L=720, W_S=168, c=72, gamma=0.0158, k=0.5 | 12.928 | 0.913 |

## 4. Test protocol

- One stream per series = validation targets followed by test targets. Warm-up W0 = 500; methods run from the start of validation; **only test targets are evaluated**. Local metrics use one calendar week (w = 168 hourly, 336 half-hourly) on full windows only.
- **L1 controlled shifts.** One shift per replay at 15%, 30%, 45%, 60%, 75% of the test segment; settings variance 1.5, variance 2.0, variance 3.0, level 1.0, level 2.0, level 3.0, gradual 2.0 (variance = κ, level = δ, gradual = ramp height, in σ̂ units). **Primary:** target-level injection with the frozen forecaster (D12). **Control:** residual-level injection (separate panel). TTR cap 1000; follow-up ≥ cap + L asserted for every position.
- **L2** untouched test segments. **L3** (exploratory): PELT with BIC penalty on the full series, offline, in a module no method can import; TTR around detected points; no confirmatory claims.
- Variance injection uses the σ_ε calibrated on validation (D12). On test the achieved ratio is reported per client; a setting whose test ratio differs from κ² by more than 10% is labelled by its achieved ratio.

| Stream | κ = 1.5 | κ = 2 | κ = 3 |
|---|---|---|---|
| electricity_h1 | σ_ε/σ̂ median 1.000; 50/50 within 2% | σ_ε/σ̂ median 1.531; 50/50 within 2% | σ_ε/σ̂ median 2.438; 50/50 within 2% |
| electricity_h24 | σ_ε/σ̂ median 1.156; 50/50 within 2% | σ_ε/σ̂ median 1.812; 50/50 within 2% | σ_ε/σ̂ median 2.938; 50/50 within 2% |

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
| electricity_h1 | H 6 (fa 0.40) | H 6 (fa 0.40) | H 1.7 (fa 0.45) | H 1.7 (fa 0.49) |
| electricity_h24 | H 20 (fa 0.35) | H 20 (fa 0.35) | H 3.0 (fa 0.38) | H 3.0 (fa 0.43) |

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
| `data/processed/selected_clients.csv` | `60c6f074cc3dc63c6e64e20945704c2e5f6fcf3d197b828b14908a0c611f5d94` |
| `data/processed/excluded_clients.csv` | `310560aa1e6d5904a404c4452f90ae471405c1291a68a9e93a60a1d7500a87fe` |
| `data/processed/preprocessing_summary.json` | `6f75eda335f3e770858d693b4a46570942b8a66f6a008c361bef9ddd0b2023cc` |
| `data/processed/cohort1_clients.csv` | `3d61c7514b83fb74d4406f64aff426e778fcb1214fc5bed7c7bb4ba332974449` |
| `PREREG_C2_PREDICTIONS.md` | `7d82a0dd82b92bf69044bbfdb01f52114f92b3c7ceab7df984a44e75a038b559` |
| `results/forecasts/forecast_summary.json` | `6f9f1dc7ecc5985d293697d67aae68738ce21756740e0630b688c12ead42407e` |
| `results/forecasts/electricity_h1.parquet` | `1257fb0c2e25ab21600ff7b296f1455d97fe1ca74c611af2321ea84cefe1fa00` |
| `results/forecasts/electricity_h24.parquet` | `84bc666f851a57f2b84ffcfe73e909b4d17e1ce082e6d7ddbed8058f7a5b9c2b` |
| `results/models/electricity_h1.txt` | `41eae37778f854f85feab680fffd053e61b198faf1205290404abd5c82fcc5c3` |
| `results/models/electricity_h1.json` | `12257ef323b7f5e579c64ccd2e128669a3f15f70e3d8b6725cffb112804a1f70` |
| `results/models/electricity_h24.txt` | `fca801c2cdcbfe458bae7a0c2441a046319a7f717f4c7e286f4c97d33e7ad28a` |
| `results/models/electricity_h24.json` | `4addb9c74cdd24bfc91860537c837c7c71efeddb30afffee55088db074605b49` |
| `results/detector_calibration.json` | `ec54aafe3b3af8c90c6a835bdc0fe91409c8e174215f7d3ab4d2a809ad179580` |
| `results/detector_choice.json` | `aba01f76175cc94faf6bc749fb05f9056febc43354acc609fb391cd274d88a7f` |
| `results/tuning/selected.json` | `575330cc94692710b1237cd163beb13dca628c6674a3a102b4af97c143875b63` |
| `results/tuning/electricity_h1/dtcwr_H.json` | `f966af135bc5766210d377df7a3248a4d00dc6f269800cd69ccac980c8236cdb` |
| `results/tuning/electricity_h24/dtcwr_H.json` | `57621220696c597b0e5dc5e490ad42c56a7fa87eef07ea31d486e8614aca7d11` |
| `results/prereg/ttr_sensitivity.json` | `0dea2f33f8dafe6a88a5a9b2949d9a7eedf650025beb2262d395f27299c7303c` |
| `results/prereg/ablation_H.json` | `bb48a12503745f51221c7a0933c5ccf645df94c11d38c85ff22a4586e68b77fb` |
| `results/prereg/sigma_eps.json` | `056d3e36d57bf6c939c365fd82b3f0856d4060e055dee1d3000dc06d91fb3eab` |
| `DECISIONS.md` | `27d6014e8f89754fced61038290b3398a193a5ab26ae5ef9cb6852b6ad980730` |
