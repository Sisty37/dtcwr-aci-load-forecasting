"""Tests for src/tune.py (validation tuning, decision D10).
Run:  python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import baselines as bl  # noqa: E402
import calib as cb  # noqa: E402
import tune as tn  # noqa: E402


def test_every_tuned_family_has_16_configs_and_fixed_has_3():
    for ds, T in (("electricity", 2630), ("elec2", 4531)):
        g = tn.grids(ds, 24, T)
        assert len(g["fixed"]) == 3
        for fam in ("aci", "rocp", "dtaci", "pid", "eci_cutoff", "driftocp", "dtcwr"):
            assert len(g[fam]) == 16, (ds, fam)
            assert len({json.dumps(c, sort_keys=True) for c in g[fam]}) == 16, (ds, fam)
    assert len(tn.ACI_GAMMAS) == 16 and tn.ACI_GAMMAS[0] == 1e-4 and tn.ACI_GAMMAS[-1] == 0.1


def test_dtcwr_draw_is_fixed_and_scaled_for_half_hourly():
    a, b = tn.dtcwr_draw(), tn.dtcwr_draw()
    assert a == b and len(a) == 16
    e = tn.grids("electricity", 1, 2630)["dtcwr"]
    e2 = tn.grids("elec2", 1, 4531)["dtcwr"]
    for x, y in zip(e, e2):
        assert y["W_S"] == 2 * x["W_S"] and y["W_L"] == 2 * x["W_L"] and y["c"] == 2 * x["c"]
        assert y["k"] == x["k"]
    assert tn.grids("elec2", 1, 4531)["fixed"][0]["W"] == 336


def test_make_method_builds_and_runs_every_config():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 900)
    yhat = np.zeros_like(y)
    for fam, cfgs in tn.grids("electricity", 24, 2630).items():
        for cfg in cfgs[:2] + cfgs[-1:]:
            if fam == "dtcwr":
                cfg = dict(cfg, H=10)
            out = cb.replay(tn.make_method(fam, cfg, 24, {"prewhiten": True}), y, yhat)
            assert np.isfinite(out["q"][cb.W0 + 24:]).any(), (fam, cfg)
    m = tn.make_method("dtcwr_aci", dict(tn.dtcwr_draw()[0], H=10, gamma=0.01), 1)
    assert m.gamma == 0.01


def test_pid_KI_is_scale_equivariant():
    rng = np.random.default_rng(1)
    e = rng.standard_t(5, 3000)
    m1, m2 = (tn.make_method("pid", {"lr": 0.1, "KI": 10}, 1) for _ in range(2))
    q1 = cb.replay(m1, e, np.zeros_like(e))["q"]
    q2 = cb.replay(m2, 1000 * e, np.zeros_like(e))["q"]
    assert np.isclose(m2.KI, 1000 * m1.KI) and np.isclose(m1.KI, 10 * bl.robust_scale(e[:cb.W0]) / bl.PUB_SCALE)
    # equal up to the repo's first step, whose lr is not range-scaled (lr_t = lr at t = 0)
    assert np.median(np.abs(1000 * q1[cb.W0:] - q2[cb.W0:]) / q2[cb.W0:]) < 0.1


def test_streams_count_and_unshifted_first():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 2630)
    s = tn.streams(y, np.zeros_like(y), 24, 168)
    assert len(s) == 22 and s[0][0] == "none" and np.array_equal(s[0][1], y)


def test_infinite_intervals_are_clipped_to_max_known_score():
    class Inf(cb.FixedWindow):
        def _q(self, w, p):
            return np.inf
    y = np.random.default_rng(3).normal(0, 1, 800)
    cov, w, nc = tn.stream_metrics(Inf(0.1, 1, 100), y, np.zeros_like(y))
    out = cb.replay(Inf(0.1, 1, 100), y, np.zeros_like(y), clip_inf=True)
    s = np.abs(y)
    assert nc == len(y) - cb.W0 - 1 and np.isfinite(w)
    assert all(out["q"][t] == s[:t].max() for t in (cb.W0 + 1, 700, 799))   # max known at origin t-1
    assert np.isinf(cb.replay(Inf(0.1, 1, 100), y, np.zeros_like(y))["q"][cb.W0 + 1])


def test_selection_rule():
    rows = []
    for cfg, cov, score in (({"a": 1}, 0.90, 5.0), ({"a": 2}, 0.85, 3.0), ({"a": 3}, 0.91, 4.0)):
        for sid in range(3):
            rows.append({"config": json.dumps(cfg), "series": sid, "cov_unshifted": cov,
                         "nwink_all": score, "nwink_unshifted": score, "cov_shifted": cov})
    s = tn.select(pd.DataFrame(rows))
    assert s["config"] == {"a": 3} and s["n_eligible"] == 2 and s["flag"] == ""
    bad = pd.DataFrame([dict(r, cov_unshifted=r["cov_unshifted"] - 0.1) for r in rows])
    s2 = tn.select(bad)
    assert s2["config"] == {"a": 3} and s2["flag"]                 # closest coverage (0.81)


def test_eval_config_series_uses_normalised_winkler():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 2630)
    r1 = tn.eval_config_series("fixed", {"W": 720}, 1, y, np.zeros_like(y), 168)
    r2 = tn.eval_config_series("fixed", {"W": 720}, 1, 50 * y, np.zeros_like(y), 168)
    assert r1["n_streams"] == 22 and 0.87 < r1["cov_unshifted"] < 0.93
    assert np.isclose(r1["nwink_all"], r2["nwink_all"])            # client scale removed


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("PASS", k)
