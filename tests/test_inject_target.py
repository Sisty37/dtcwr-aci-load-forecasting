"""Tests for src/inject_target.py (target-level injection, plan 6.3, decision D12).
Run:  python -m pytest tests -q
Synthetic tests use a stand-in for the frozen booster; the last test uses the REAL frozen
LightGBM model and data if present and checks that it reproduces the saved forecasts.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import calib as cb  # noqa: E402
import forecast as fc  # noqa: E402
import inject as ij  # noqa: E402
import inject_target as it  # noqa: E402

P = 24


class LinearBooster:
    """predict = 0.6 * lag_0 + 0.4 * same_time_yesterday (scaled units)."""
    def predict(self, X):
        return 0.6 * X[:, 0] + 0.4 * X[:, P]


def make_injector(h=24, n=26304):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2012-01-01", periods=n, freq="h")
    t = np.arange(n)
    cols = {}
    for k, name in enumerate(("MT_a", "MT_b")):
        base = 50 + 10 * np.sin(2 * np.pi * t / 24) + 5 * np.sin(2 * np.pi * t / 168)
        cols[name] = base * (1 + k) + rng.normal(0, 2, n)
    wide = pd.DataFrame(cols, index=idx)
    n_tr, _, _ = fc.split_sizes(n)
    meta = {"period": P, "series": list(cols), "scales": {c: float(wide[c][:n_tr].mean()) for c in cols}}
    return it.TargetInjector("electricity", h, booster=LinearBooster(), wide=wide, meta=meta)


def test_forecast_matches_manual_feature_path():
    inj = make_injector()
    tg = inj.stream_targets(False)
    y = inj.y("MT_b")
    s = inj.scales["MT_b"]
    X, _, _, tgt = fc.make_features(y / s, inj.wide.index, inj.spec)
    rows = np.searchsorted(tgt, tg)
    manual = (0.6 * X[rows, 0] + 0.4 * X[rows, P]) * s
    assert np.allclose(inj.forecast("MT_b", y, tg), manual)
    assert (inj.split[tg] == "val").all() and len(inj.stream_targets(True)) > len(tg)


def test_level_injection_and_no_effect_before_origin():
    inj = make_injector(h=24)
    tau = 1200
    tg, y2, yh2, sig = inj.inject("MT_a", "level", 2.0, tau, 168)
    y0 = inj.y("MT_a")[tg]
    yh0 = inj.forecast("MT_a", inj.y("MT_a"), tg)
    assert np.isclose(sig, np.std((y0 - yh0)[:cb.W0]))
    assert np.array_equal(y2[:tau], y0[:tau]) and np.allclose(y2[tau:] - y0[tau:], 2 * sig)
    # targets whose ORIGIN is before the shift keep their forecast; later ones react
    assert np.allclose(yh2[:tau + 24], yh0[:tau + 24])
    assert not np.allclose(yh2[tau + 24:tau + 48], yh0[tau + 24:tau + 48])
    # lags absorb part of a level shift: mean residual shift is below delta
    a = it.achieved("level", 2.0, y0 - yh0, y2 - yh2, tau, sig, 168)
    assert 0 < a["achieved_mean_shift_sd"] < 2.0


def test_gradual_ramp_reaches_full_height():
    inj = make_injector(h=1)
    tau = 1000
    tg, y2, _, sig = inj.inject("MT_a", "gradual", 2.0, tau, 168)
    d = y2 - inj.y("MT_a")[tg]
    assert d[tau - 1] == 0 and np.isclose(d[tau], 2 * sig / 168) and np.allclose(d[tau + 200:], 2 * sig)


def test_eps_is_fixed_per_client_and_differs_between_clients():
    inj = make_injector()
    assert np.array_equal(inj.eps("MT_a"), inj.eps("MT_a"))
    assert not np.allclose(inj.eps("MT_a")[:100], inj.eps("MT_b")[:100])


def test_sigma_eps_bisection_hits_target_ratio():
    inj = make_injector(h=24)
    for kappa in (1.5, 3.0):
        r = it.calibrate_sigma_eps(inj, "MT_a", kappa, 168)
        assert r["within_tol"] and abs(r["achieved_ratio"] / kappa ** 2 - 1) <= it.SIGMA_TOL
    r1 = it.calibrate_sigma_eps(inj, "MT_a", 1.5, 168)["m"]
    r2 = it.calibrate_sigma_eps(inj, "MT_a", 3.0, 168)["m"]
    assert r2 > r1 > 0


def test_test_positions_have_full_follow_up():
    taus = ij.test_positions(2630, 5261, L=84)
    assert len(taus) == 5 and taus[0] == 2630 + round(0.15 * 5261)
    assert 2630 + 5261 - taus[-1] >= ij.TTR_CAP + 84
    t2 = ij.test_positions(4531, 9063, L=336)
    assert t2[-1] == 4531 + round(0.75 * 9063)
    try:
        ij.test_positions(100, 1000, L=84)
        raise AssertionError("short test segment accepted")
    except ValueError:
        pass


def test_real_frozen_model_reproduces_saved_forecasts():
    model = ROOT / "results" / "models" / "electricity_h24.txt"
    fcst = ROOT / "results" / "forecasts" / "electricity_h24.parquet"
    if not (model.exists() and fcst.exists()) or importlib.util.find_spec("lightgbm") is None:
        try:
            import pytest
            pytest.skip("frozen model / forecasts / lightgbm not available")
        except ImportError:
            print("SKIP real-model test")
            return
    inj = it.TargetInjector("electricity", 24, model_dir=ROOT / "results" / "models",
                            data_dir=ROOT / "data" / "processed")
    saved = pd.read_parquet(fcst)
    name = inj.series[0]
    g = saved[(saved["series"] == name) & (saved["split"] == "val")].sort_values("target")
    tg = inj.stream_targets(False)
    assert len(g) == len(tg)
    yhat = inj.forecast(name, inj.y(name), tg)
    assert np.allclose(yhat, g["yhat"].to_numpy(), rtol=1e-6, atol=1e-6)
    assert np.allclose(inj.y(name)[tg], g["y"].to_numpy())


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("PASS", k)
