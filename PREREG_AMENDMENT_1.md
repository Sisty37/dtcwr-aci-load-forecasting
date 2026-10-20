# Pre-registration amendment 1 (before the test segment is opened)

Added on top of `prereg-v1` (PREREGISTRATION.md). Tag: `prereg-v1.1`. **The test segment has not
been opened.** Nothing in PREREGISTRATION.md is changed; this file only removes an ambiguity in
Sections 1 and 6 about which comparisons are confirmatory. When the two files differ, this file
applies.

## A1. Confirmatory method

The confirmatory method is **DTCW-R+ACI** (the full method). This is the configuration with the
best validation score at the day-ahead horizon in both datasets (PREREGISTRATION.md, Section 3).
DTCW-R without the ACI layer is reported in every table, but its comparisons are secondary.

## A2. Confirmatory comparisons (Electricity, 50 clients)

DTCW-R+ACI against each of the **9 baselines**: fixed windows (3), ACI, ROCP, DtACI, Conformal PID,
ECI-cutoff and DriftOCP.

| Family | Outcome per client | Layer | Horizons |
|---|---|---|---|
| F1 | RMST of TTR (frozen cell), averaged over the 5 positions and the 7 settings | L1 target-level | h = 1 and h = 24, separate families |
| F2 | Winkler score | L2 | h = 1 and h = 24, separate families |
| F3 | Worst local coverage, averaged over the 5 positions and the 7 settings | L1 target-level | h = 1 and h = 24, separate families |

- That is 6 families of 9 comparisons each. Each family uses a Wilcoxon signed-rank test on the 50
  client differences and Holm's correction within the family, at a familywise level of 0.05.
- Each comparison reports the median difference, a 95% cluster-bootstrap CI (2,000 client
  resamples) and the rank-biserial effect size.
- A claim of the form "better than X" needs the Holm-adjusted p < 0.05 **and** the CI excluding 0
  in the favourable direction.
- Coverage is read first (PREREGISTRATION.md, Section 5). A Winkler or width advantage is not
  claimed for a method pair if DTCW-R+ACI's L2 coverage error is more than 1 pp larger than the
  baseline's.

## A3. Everything else is secondary or exploratory

The following are reported in full, but none of them supports a confirmatory claim:

- **Per-shift-type breakdown:** variance, level and gradual reported separately.
- **Other layers and metrics:** the residual-level control panel, the relative TTR, the 12-cell
  TTR sensitivity, detection delay and false alarms.
- **Elec2:** effect sizes with block-bootstrap CIs only.
- **DTCW-R alone** against the baselines.
- **The overall ranking** (Friedman/Nemenyi).

The 2×2 ablation keeps its own pre-specified analysis (PREREGISTRATION.md, Section 7; primary
outcome at h = 1).

## A4. Expected direction, stated before the test

On validation, DTCW-R+ACI led at the day-ahead horizon and did not lead at h = 1 (DtACI, ECI-cutoff
and PID were about 1.5% better on the validation score). If the test agrees, the paper reports an
advantage at the day-ahead horizon only. If the test disagrees, the paper reports the test result.
