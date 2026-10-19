"""Tests for src/forecast.py.
Run:  python -m pytest tests -q
The end-to-end test needs lightgbm (skipped automatically if it is not installed).
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import forecast as fc  # noqa: E402


def _series(n=24 * 7 * 30, P=24, seed=0, freq="h"):
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    y = 100 + 30 * np.sin(2 * np.pi * (k % P) / P) + 10 * (k // P % 7 >= 5) + rng.normal(0, 2, n)
    ts = pd.date_range("2012-01-01", periods=n, freq=freq)
    return y, ts


def test_features_use_only_past_values():
    y, ts = _series()
    for h in (1, 24):
        spec = fc.FeatureSpec(24, h)
        X1, _, o1, _ = fc.make_features(y, ts, spec)
        cut = 3000
        y2 = y.copy()
        y2[cut + 1:] += 1e6                                   # change the future only
        X2, _, o2, _ = fc.make_features(y2, ts, spec)
        rows = o1 <= cut
        assert np.array_equal(X1[rows], X2[rows]), f"leakage at h={h}"


def test_target_and_seasonal_alignment():
    y, ts = _series()
    spec = fc.FeatureSpec(24, 24)
    X, tgt, o, t = fc.make_features(y, ts, spec)
    names = spec.names()
    assert np.array_equal(t, o + 24) and np.allclose(tgt, y[t])
    assert np.allclose(X[:, names.index("same_time_yesterday")], y[t - 24].astype("float32"))
    assert np.allclose(X[:, names.index("same_time_last_week")], y[t - 168].astype("float32"))
    assert np.allclose(X[:, names.index("lag_0")], y[o].astype("float32"))


def test_half_hourly_slot_feature():
    y, ts = _series(n=48 * 7 * 10, P=48, freq="30min")
    spec = fc.FeatureSpec(48, 1)
    X, _, _, t = fc.make_features(y, ts, spec)
    slot = X[:, spec.names().index("slot_of_day")].astype(int)
    assert np.array_equal(slot, ts[t].hour * 2 + ts[t].minute // 30)
    assert slot.min() == 0 and slot.max() == 47


def test_horizon_longer_than_a_day_rejected():
    try:
        fc.FeatureSpec(24, 25)
    except ValueError:
        return
    raise AssertionError("h > P must be rejected")


def test_split_labels_chronological():
    n = 26304
    lab = fc.split_label(np.arange(n), n)
    assert (lab[:18413] == "train").all() and (lab[18413:18413 + 2630] == "val").all()
    assert (lab[-5261:] == "test").all()


def _write_fake_electricity(d: Path, n_clients=3, n=24 * 7 * 40):
    cols = {}
    for i in range(n_clients):
        y, ts = _series(n=n, seed=i)
        cols[f"MT_{i + 1:03d}"] = y * (1 + i)
    wide = pd.DataFrame(cols, index=ts)
    wide.index.name = "hour_start"
    wide.to_csv(d / "electricity_hourly.csv.gz")
    pd.DataFrame({"client": list(cols)}).to_csv(d / "selected_clients.csv", index=False)
    return wide


def test_panel_scaling_uses_train_only():
    d = Path(tempfile.mkdtemp())
    wide = _write_fake_electricity(d)
    w, series = fc.load_series("electricity", d)
    w2 = w.copy()
    n_tr = fc.split_sizes(len(w2))[0]
    w2.iloc[n_tr:] *= 50                                       # change val/test only
    _, s1 = fc.build_panel(w, series, fc.FeatureSpec(24, 1))
    _, s2 = fc.build_panel(w2, series, fc.FeatureSpec(24, 1))
    assert s1 == s2


def test_end_to_end_small():
    if importlib.util.find_spec("lightgbm") is None:
        import pytest
        pytest.skip("lightgbm not installed")
    d = Path(tempfile.mkdtemp())
    _write_fake_electricity(d)
    r = fc.run_one("electricity", 24, d, d / "fc", d / "models",
                   configs=fc.CONFIGS[:2], max_trees=200, early_stop=20)
    out = pd.read_parquet(r["file"]) if r["file"].endswith(".parquet") else pd.read_csv(r["file"])
    assert set(out["split"]) == {"val", "test"}
    assert np.isfinite(out["yhat"]).all()
    assert np.allclose(out["resid"], out["y"] - out["yhat"])
    assert r["share_series_lgbm_better_val"] >= 0.0 and (d / "models" / "electricity_h24.txt").exists()
    # frozen model reproduces its own predictions
    import lightgbm as lgb
    b = lgb.Booster(model_file=str(d / "models" / "electricity_h24.txt"))
    assert b.num_trees() >= 1


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("PASS", k)
