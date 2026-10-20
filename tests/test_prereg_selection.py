"""Tests for src/run_prereg_selection.py (TTR cell rule and 2x2 thresholds, decision D11).
Run:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import calib as cb  # noqa: E402
import detector_calibration as dc  # noqa: E402
import run_prereg_selection as ps  # noqa: E402


def test_twelve_cells_one_primary_and_scaling():
    c = ps.ttr_cells("electricity")
    assert len(c) == 12 and sum(ps.is_primary(x) for x in c) == 1
    p = next(x for x in c if ps.is_primary(x))
    assert p["w"] == 168 and p["L"] == 168 and p["delta"] == 0.05
    p2 = next(x for x in ps.ttr_cells("elec2") if ps.is_primary(x))
    assert p2["w"] == 336 and p2["L"] == 336


def test_strictness_order():
    c = sorted(ps.ttr_cells("electricity"), key=ps.strictness_key)
    assert (c[0]["delta"], c[0]["L_over_w"], c[0]["w_h"]) == (0.025, 1.0, 168)
    assert c[-1]["delta"] == 0.05 and c[-1]["L_over_w"] == 0.5


def _df(rate_by_cell):
    rows = []
    for c in ps.ttr_cells("electricity"):
        r = rate_by_cell(c)
        for i in range(100):
            rows.append({"dataset": "electricity", "h": 1, "method": "aci", "series": i, "tau": 0,
                         "w_h": c["w_h"], "delta": c["delta"], "L_over_w": c["L_over_w"],
                         "nonrecovered": i < 100 * r})
    return pd.DataFrame(rows)


def test_choose_cell_keeps_primary_when_it_passes():
    res = ps.choose_cell(_df(lambda c: 0.01), ps.ttr_cells("electricity"))
    assert res["chosen"] == {"w_h": 168, "delta": 0.05, "L_over_w": 1.0}


def test_choose_cell_falls_back_to_strictest_passing():
    # only delta = 0.05 cells pass; primary fails -> strictest among the passing ones
    f = lambda c: 0.02 if (c["delta"] == 0.05 and not ps.is_primary(c)) else 0.2   # noqa: E731
    res = ps.choose_cell(_df(f), ps.ttr_cells("electricity"))
    assert res["chosen"]["delta"] == 0.05 and res["chosen"]["L_over_w"] == 1.0
    assert res["chosen"]["w_h"] != 168 and res["reason"].startswith("strictest")
    res2 = ps.choose_cell(_df(lambda c: 0.5), ps.ttr_cells("electricity"))
    assert res2["chosen"] is None


def test_nonrecovery_is_rare_on_iid_stream_for_primary_cell():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 2630)
    rows = ps.nonrecovery_rows("electricity", 1, 0, y, np.zeros_like(y), {"gamma": 0.005},
                               ps.ttr_cells("electricity"))
    d = pd.DataFrame(rows)
    prim = d[(d.w_h == 168) & (d.delta == 0.05) & (d.L_over_w == 1)]
    assert len(prim) == 6 and prim["nonrecovered"].mean() <= 1 / 3


def test_miss_detector_calibration_meets_target():
    rng = np.random.default_rng(1)
    sc = [np.abs(rng.normal(0, 1, 2000)) for _ in range(3)]
    make = lambda H: cb.DTCWR(0.1, 1, H=H, signal="miss", W_S=48, W_L=336)   # noqa: E731
    res = ps.calibrate_H_bisect(sc, 24, make, length=6000)
    assert res["fa_at_H"] <= dc.TARGET_FA and res["H"] in ps.MISS_GRID
    i = ps.MISS_GRID.index(res["H"])
    assert i == 0 or res["curve"][ps.MISS_GRID[i - 1]] > dc.TARGET_FA       # smallest passing H
    assert len(res["curve"]) <= 8                                           # bisection, not a walk


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("PASS", k)
