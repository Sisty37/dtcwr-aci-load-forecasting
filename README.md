# Drift-Triggered Recalibration with Adaptive Conformal Inference for Load Forecasting

Code, frozen configurations and pre-registration records for the paper

> A. J. Sisty, N. Roy, K. Mansur, A. Bhowmik and S. M. A. Shafi, "Drift-Triggered Recalibration with
> Adaptive Conformal Inference for Load Forecasting: A Pre-Registered Three-Cohort Evaluation," in *Proc. 5th Int. Conf.
> Robotics, Electrical and Signal Processing Techniques (ICREST)*, Dhaka, Bangladesh, 2027.

The method, **DTCW-R+ACI**, runs a CUSUM on the log magnitude of forecast residuals, restarts the
conformal calibration window after an alarm, and keeps adaptive conformal inference (ACI) on top,
with one ACI copy per forecast offset for h-step-ahead feedback. It is compared with nine baselines
(fixed windows, ACI, DtACI, Conformal PID, ECI-cutoff, ROCP, DriftOCP) under a pre-registered
protocol on hourly UCI Electricity load (three cohorts of 50 clients) and Elec2.

## Pre-registration

Every confirmatory choice was committed and tagged before the corresponding test data were opened.
The git tags are the time stamps:

| Tag | What it froze |
|---|---|
| `prereg-v1`, `prereg-v1.1` | Cohort 1: configurations, rules, file hashes, confirmatory families |
| `c2-predictions` | Predictions R1-R6, before any cohort-2 step |
| `prereg-c2` | Cohort 2 frozen before its test |
| `c3-predictions` | Predictions S1-S4 (95% coverage), before the cohort-3 client draw |
| `prereg-c3` | Cohort 3 frozen before its test |

`git log --tags --simplify-by-decoration --format="%ci %d"` lists them with dates. Every decision
taken after seeing data is logged in `DECISIONS.md`; post hoc analyses are labelled there and in the
paper. Branches: `master` (cohort 1 and combined analyses), `cohort2`, `cohort3`.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | All pipeline, method, baseline, analysis and figure scripts |
| `tests/` | Unit tests, including the look-ahead (leakage) test |
| `checks/` | Prototype and verification scripts |
| `results/` | Frozen configurations, test metrics, analyses and figures (cohort 1 on `master`) |
| `results/posthoc/` | Post hoc analyses, including the camera-ready robustness checks |
| `PREREGISTRATION.md`, `PREREG_*.md` | Pre-registration records and predictions |
| `DECISIONS.md` | Every decision taken after seeing data, with date and reason |

## Data (not included)

Download the raw files and place them as follows:

- UCI ElectricityLoadDiagrams20112014 (doi:10.24432/C58C86): `data/LD2011_2014.txt`
- Elec2, as distributed with the Conformal PID code
  (`tests/datasets/elec2.csv` of https://github.com/aangelopoulos/conformal-time-series):
  `data/conformal-time-series/tests/datasets/elec2.csv`

`data/processed/preprocessing_summary.json` (created by the first step) contains SHA-256 hashes
that can be compared with those recorded in `PREREGISTRATION.md`.

## Reproducing cohort 1

```
pip install -r requirements.txt
python src/data_electricity.py        # cleaning, DST repair, client selection
python src/data_elec2.py
python src/forecast.py                # frozen LightGBM (and seasonal naive) forecasts
python src/run_test.py                # test run (refuses to start unless HEAD descends from prereg-v1)
python src/analysis.py --elec2-mbb    # confirmatory statistics, tables, 2x2 ablation, Elec2
python src/figures.py                 # figures
pytest tests                          # unit tests, including the look-ahead (leakage) test
```

Cohorts 2 and 3 use the same scripts on their branches; `src/cohort.py` reads the cohort from
environment variables (`ICREST_DATASETS`, `ICREST_PREREG_TAG`, `ICREST_ALPHA`). Tuning, threshold
calibration and TTR selection scripts (`tune.py`, `run_detector_calibration.py`,
`run_prereg_selection.py`, `run_sigma_calibration.py`) regenerate the frozen files in `results/`.

Post hoc analyses: `natural_drift.py` (natural changes), `run_extra_baseline.py` (SAOCP),
`review_stats.py`, `bench_methods.py`, `sensitivity.py` (threshold, step size, windows, feedback
rule) and `second_forecaster.py` (seasonal-naive forecaster). `verify_paper.py` checks every number
reported in the paper against the saved outputs.

## Data licenses

The datasets keep their own licenses (UCI Electricity: CC BY 4.0) and are not redistributed here.
