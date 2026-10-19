"""Tests for src/detector_calibration.py and the prewhitening option of DTCWR.
Run:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import calib as cb  # noqa: E402
import detector_calibration as dc  # noqa: E402


def _ar_scores(n, phi=0.9, seed=0, shift_at=None, factor=3.0):
    """|residual| whose LOG scale follows an AR(1): mimics autocorrelated h-step errors."""
    rng = np.random.default_rng(seed)
    g = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        g[t] = phi * g[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
    s = np.abs(rng.normal(size=n)) * np.exp(0.6 * g)
    if shift_at is not None:
        s[shift_at:] *= factor
    return s


def test_block_bootstrap_shape_and_seed():
    x = np.arange(100.0)
    a = dc.block_bootstrap(x, 250, 10, np.random.default_rng(1))
    b = dc.block_bootstrap(x, 250, 10, np.random.default_rng(1))
    assert len(a) == 250 and np.array_equal(a, b) and set(a) <= set(x)
    assert np.all(np.diff(a[:10]) % 100 == 1)          # inside a block consecutive values


def test_autocorrelation_inflates_alarms_and_prewhiten_helps():
    s = _ar_scores(30_000, phi=0.9, seed=1)
    plain = dc.alarm_rate(s, lambda: cb.DTCWR(0.1, 1, H=6))
    pw = dc.alarm_rate(s, lambda: cb.DTCWR(0.1, 1, H=6, prewhiten=True))
    assert plain > 1.0                                  # far above the 0.5 / 1,000 target
    assert pw < plain


def test_calibrated_H_meets_target_on_fresh_data():
    val = [_ar_scores(4000, phi=0.9, seed=s) for s in range(4)]
    for pw in (False, True):
        cal = dc.calibrate_H(val, block=48, make_with_H=lambda H: cb.DTCWR(0.1, 1, H=H, prewhiten=pw),
                             reps_per_series=2, length=8000)
        fresh = _ar_scores(40_000, phi=0.9, seed=99)
        rate = dc.alarm_rate(fresh, lambda: cb.DTCWR(0.1, 1, H=cal["H"], prewhiten=pw))
        assert rate <= 3 * dc.TARGET_FA, (pw, cal["H"], rate)   # same order as the target


def test_calibrated_detector_still_detects_large_shift():
    val = [_ar_scores(4000, phi=0.9, seed=s) for s in range(4)]
    cal = dc.calibrate_H(val, block=48, make_with_H=lambda H: cb.DTCWR(0.1, 1, H=H, prewhiten=True),
                         reps_per_series=2, length=8000)
    s = _ar_scores(6000, phi=0.9, seed=7, shift_at=3000, factor=3.0)
    m = cb.DTCWR(0.1, 1, H=cal["H"], prewhiten=True)
    m.setup(s[:cb.W0])
    for t, v in enumerate(s):
        m.observe(t, float(v))
    post = [a for a in m.alarms if a[0] >= 3000]
    assert post and post[0][0] - 3000 <= 400, (cal["H"], m.alarms)


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("PASS", k)
