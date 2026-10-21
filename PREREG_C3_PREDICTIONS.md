# Cohort 3: pre-registered robustness study at 95% coverage (decision D18)

Written and committed **before any cohort-3 data step** (client draw, forecasting, tuning). Tag:
`c3-predictions`. The cohort-3 pre-registration (`PREREGISTRATION.md` regenerated in the cohort-3
working copy, tag `prereg-c3`) adds the frozen configurations and hashes.

## Why
The pre-submission review asked for an independent robustness test. Cohort 3 changes two things
at once relative to cohorts 1-2, both fixed now: **new clients** (50 Electricity clients drawn with
seed 2029 from the 214 eligible clients used by neither earlier cohort) and **a different target
coverage, 95% (alpha = 0.05)**. Everything else is unchanged: the code at the commit that adds this
file (plus the alpha switch), the D6-D12 rules, the 16-configuration tuning budget and selection
rule (eligible = unshifted validation coverage within 0.95 +/- 0.02), the TTR cell rule (recovery
level 1 - alpha - delta), the 7 shift settings and 5 positions, the 9 baselines and the analysis.
SAOCP (D17) is not part of cohort 3.

## Confirmatory families
DTCW-R+ACI against the 9 baselines; RMST of TTR (L1 target), Winkler (L2), worst local coverage (L1
target) at h = 1 and h = 24; Wilcoxon + Holm within family + 95% cluster-bootstrap CI; "better" /
"worse" as in Amendment 1, including the 1 pp coverage-error guard.

## Predictions (same as R1-R4, at 95% coverage)
| # | Prediction |
|---|---|
| S1 | h = 24, Winkler (L2): DTCW-R+ACI **better** than at least 7 of the 9 baselines |
| S2 | h = 1, RMST of TTR (L1): DTCW-R+ACI **better** than at least 7 of the 9 baselines |
| S3 | h = 24, RMST of TTR (L1): DTCW-R+ACI **worse** than Conformal PID and than ECI-cutoff |
| S4 | h = 1, Winkler (L2): DTCW-R+ACI **not better** than ACI and not better than DtACI |

- The horizon-dependent trade-off is said to **hold at 95% coverage** if S1, S2 and S3 all hold,
  **partly** if one or two hold, and **not** if none holds. S4 is reported with its own verdict.
- All outcomes are reported, whatever they are. No further cohort will be run to rescue a failed
  prediction.
