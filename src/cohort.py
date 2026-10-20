"""
Which client cohort / datasets this working copy runs (decision D14).

Cohort 1 (default): Electricity 50 clients + Elec2, tag prereg-v1.
Cohort 2 (fresh replication): set in PowerShell before running any script
    $env:ICREST_DATASETS = "electricity"
    $env:ICREST_PREREG_TAG = "prereg-c2"
Defaults reproduce cohort 1 exactly.
"""
import os

DATASETS = tuple(d.strip() for d in os.environ.get("ICREST_DATASETS", "elec2,electricity").split(",") if d.strip())
PREREG_TAG = os.environ.get("ICREST_PREREG_TAG", "prereg-v1")
