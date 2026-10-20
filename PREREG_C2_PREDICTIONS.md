# Cohort 2: pre-registered replication predictions

Written and committed **before any cohort-2 data step** (client draw, forecasting, tuning). The
cohort-2 pre-registration (`PREREGISTRATION.md` regenerated in the cohort-2 working copy, tag
`prereg-c2`) adds the frozen configurations and hashes. When the two disagree, this file
governs the replication claims.

## Why a second cohort

- The cohort-1 test (tag `prereg-v1.1`) gave a horizon-dependent pattern.
- A post-hoc check at an exactly matched false-alarm rate (D13a) kept the same 2×2 directions.
- The v2 design search (D13b, `results/v2/design.json`) selected **v1** over every v2 variant on
  validation. Cohort 2 therefore tests **the unchanged v1 method**, so this is a pure,
  pre-registered replication on 50 fresh clients.

## Data and procedure

- **Data.** Electricity only: 50 clients drawn with seed 2028 from the 264 eligible clients that
  cohort 1 did not use (`src/make_cohort2.py`). Elec2 has no fresh data and is not part of cohort 2.
- **Procedure.** Identical code and rules to cohort 1:
  - LightGBM re-trained on the new clients' train segments;
  - detector calibration and choice (D6–D8);
  - validation tuning with the same 16-configuration grids and selection rule (D10);
  - TTR cell and 2×2 thresholds by the D11 rules;
  - σ_ε calibration (D12);
  - the same test protocol and analysis (`src/run_test.py`, `src/analysis.py`).
- **Confirmatory method and families.** The method is DTCW-R+ACI. The families are the 6 of
  Amendment 1 (RMST L1, Winkler L2 and worst local coverage L1, at h = 1 and h = 24, each against
  the 9 baselines). The tests are the same: Wilcoxon, Holm, cluster bootstrap, and the
  coverage-error guard for Winkler.

## Replication predictions

Each prediction is checked with the cohort-2 test results.
- "**better**" or "**worse**" means Holm p < 0.05 and a CI on the stated side, as in Amendment 1.
- "**not better**" means "worse" or "no claim".

| # | Prediction | Cohort-1 result |
|---|---|---|
| R1 | h = 24, Winkler (L2): DTCW-R+ACI **better** than at least 7 of the 9 baselines | better than 8, worse than ACI |
| R2 | h = 1, RMST of TTR (L1): DTCW-R+ACI **better** than at least 7 of the 9 baselines | better than 8, no claim vs PID |
| R3 | h = 24, RMST of TTR (L1): DTCW-R+ACI **worse** than Conformal PID and than ECI-cutoff (the price of their wider intervals) | worse than both |
| R4 | h = 1, Winkler (L2): DTCW-R+ACI **not better** than ACI and not better than DtACI | worse than both |
| R5 | 2×2, h = 1, variance shifts, RMST: signal effect **> 0** (the miss-indicator detector recovers sooner) | +16.8, Holm p = 0.001 |
| R6 | 2×2, h = 24, variance shifts, RMST: signal effect **< 0** (the magnitude detector recovers sooner) | −13.7, p = 0.027 (secondary in cohort 1) |

- R5 and R6 form one Holm family of 2. Each uses a Wilcoxon test with a cluster-bootstrap CI.
- **Overall verdict, fixed now:**
  - The main claim (a horizon-dependent coverage–width–recovery trade-off) is **replicated** if
    R1, R2 and R3 all hold.
  - It is **partially replicated** if one or two of them hold.
  - It is **not replicated** if none holds.
  - R4–R6 are reported with their own verdicts.
- **What the paper reports:**
  - The cohort-1 and cohort-2 results side by side.
  - Every prediction, whatever its outcome.
  - A pooled (100-client) analysis as exploratory only.
