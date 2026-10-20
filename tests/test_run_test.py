"""Tests for src/run_test.py on SYNTHETIC data only (the real test segment is opened once,
by running the script after the prereg-v1 tag).
Run:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import calib as cb  # noqa: E402
import inject as ij  # noqa: E402
import run_test as rt  # noqa: E402
import tune as tn  # noqa: E402
from test_inject_target import make_injector  # noqa: E402


def fake_frozen(ds="electricity", h=24, names=("MT_a", "MT_b")):
    key = f"{ds}_h{h}"
    d = {"W_S": 96, "W_L": 720, "k": 0.5, "c": 24}
    sel = {"fixed": {"configs": [{"W": 168}, {"W": 720}, {"W": 2160}]},
           "aci": {"config": {"gamma": 0.005}}, "rocp": {"config": {"W": 190}},
           "dtaci": {"config": {"gscale": 1, "I": 500}}, "pid": {"config": {"lr": 0.05, "KI": 0}},
           "eci_cutoff": {"config": {"lr": 0.05, "h_cut": 0.25}},
           "driftocp": {"config": {"sigma": 1, "min_len": 10}},
           "dtcwr": {"config": dict(d, H=20)}, "dtcwr_aci": {"config": dict(d, H=20, gamma=0.01)}}
    abl = {"config": d, "D_magnitude_anchored": {"H": 20}, "B_magnitude_reset": {"H": 20},
           "A_miss_anchored": {"H": 2.5}, "R_miss_reset": {"H": 2.5}}
    sig = {n: {k: {"m": m} for k, m in (("1.5", 1.0), ("2.0", 1.6), ("3.0", 2.6))} for n in names}
    return {"sel": {key: sel}, "abl": {key: abl}, "sig": {key: sig},
            "ttr": {"chosen": {"w_h": 84, "delta": 0.025, "L_over_w": 1.0}},
            "choice": {key: {"winner": "plain"}}}


def test_fourteen_method_rows_build_and_run():
    fz = fake_frozen()
    rows = rt.method_rows("electricity", 24, fz)
    assert len(rows) == 14
    assert sum(k.startswith("fixed_") for k in rows) == 3 and sum(k.startswith("abl_") for k in rows) == 3
    y = np.random.default_rng(0).normal(0, 1, 1200)
    cell = rt.frozen_cell("electricity", fz)
    assert cell == {"w": 84, "delta": 0.025, "L": 84}
    for name, make in rows.items():
        met = rt.eval_stream(make, y, np.zeros_like(y), 800, "electricity", [], cell)
        assert 0.5 < met["coverage"] <= 1.0 and met["n"] == 400, name
    m = rows["abl_R_miss_reset"]()
    assert m.signal == "miss" and m.window_rule == "reset" and m.cs.H == 2.5


def test_eval_stream_evaluates_only_test_targets():
    y = np.random.default_rng(1).normal(0, 1, 3000)
    y[:2000] *= 10                                         # validation part very different
    make = lambda: tn.make_method("fixed", {"W": 168}, 1)  # noqa: E731
    met = rt.eval_stream(make, y, np.zeros_like(y), 2000, "electricity", [], rt.frozen_cell("electricity", fake_frozen()))
    assert met["n"] == 1000 and 0.85 < met["coverage"] < 0.95


def test_run_client_layers_rows_and_injection():
    inj = make_injector(h=24)
    fz = fake_frozen()
    name = "MT_a"
    tg = inj.stream_targets(test=True)
    y = inj.y(name)[tg]
    yhat = inj.forecast(name, inj.y(name), tg)
    n_val = int((inj.split[tg] == "val").sum())
    keep = ("fixed_720", "dtcwr", "abl_A_miss_anchored")
    orig = rt.method_rows
    rt.method_rows = lambda ds, h, f: {k: v for k, v in orig(ds, h, f).items() if k in keep}
    try:
        rows = pd.DataFrame(rt.run_client("electricity", 24, name, y, yhat, n_val, inj, fz))
    finally:
        rt.method_rows = orig
    n_runs = 1 + 2 * len(ij.TEST_FRACS) * len(ij.SETTINGS)
    assert len(rows) == len(keep) * n_runs
    assert set(rows["layer"]) == {"L2", "L1_target", "L1_resid"}
    l2 = rows[rows.layer == "L2"]
    assert l2["ttr"].isna().all() and len(l2) == len(keep)
    l1 = rows[rows.layer == "L1_target"]
    assert l1["ttr"].notna().all() and set(l1["pos_frac"]) == set(ij.TEST_FRACS)
    assert l1.filter(like="ttr__").shape[1] == 12
    v = l1[(l1.kind == "variance") & (l1.method == "dtcwr")]
    assert v["achieved_ratio"].notna().all() and (v["achieved_ratio"] > 1).all()
    det = rows[(rows.method == "dtcwr") & (rows.layer != "L2")]
    assert det["det_delay"].notna().all()
    assert (rows.loc[rows.method == "fixed_720", "det_delay"].isna()).all()


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("PASS", k)
