"""
Decision D18 helper: in a NEW cohort working copy (git worktree), remove every cohort-specific
output inherited from cohort 1 so that each pipeline step is recomputed (several scripts are
resumable and would otherwise skip existing files). Data files, code and pre-registration
documents of earlier cohorts are kept. Dry run by default:

    python src/reset_for_new_cohort.py            # list what would be removed
    python src/reset_for_new_cohort.py --apply    # remove it (then commit)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REMOVE = ["results/forecasts", "results/models", "results/calib_val", "results/tuning", "results/prereg",
          "results/test", "results/test_extra", "results/analysis", "results/posthoc", "results/v2",
          "results/natural", "results/review", "results/cohort2", "results/combined", "results/figures",
          "results/detector_calibration.json", "results/detector_choice.json", "PREREGISTRATION.md"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not Path("data/processed/cohort2_clients.csv").exists():
        raise SystemExit("run src/make_cohort3.py first (this must be the cohort-3 working copy)")
    for p in map(Path, REMOVE):
        if p.exists():
            print(("removing " if a.apply else "would remove ") + str(p))
            if a.apply:
                shutil.rmtree(p) if p.is_dir() else p.unlink()


if __name__ == "__main__":
    main()
