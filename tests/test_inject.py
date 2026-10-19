"""Tests for src/inject.py (plan v5 Sections 6.3 and 8.2).
Run:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import calib as cb  # noqa: E402
import inject as ij  # noqa: E402

R = np.random.default_rng(0).normal(0, 2.0, 4000)


def test_variance_exact_ratio_and_prefix_untouched():
    for k in (1.5, 2.0, 3.0):
        sh = ij.Shift("variance", k, 2000, 168)
        r2 = ij.inject_residuals(R, sh)
        assert np.array_equal(r2[:2000], R[:2000])
        assert np.isclose(np.var(r2[2000:]) / np.var(R[2000:]), k ** 2, rtol=0, atol=1e-12)
        assert np.allclose(np.abs(r2[2000:]), k * np.abs(R[2000:]))


def test_level_shift_size_uses_warmup_sd():
    sh = ij.Shift("level", 2.0, 2000, 168)
    r2 = ij.inject_residuals(R, sh)
    sig = np.std(R[:cb.W0])
    assert np.allclose(r2[2000:] - R[2000:], 2.0 * sig)


def test_gradual_ramp():
    sh = ij.Shift("gradual", 2.0, 2000, 168)
    d = ij.inject_residuals(R, sh) - R
    sig = np.std(R[:cb.W0])
    assert d[1999] == 0 and np.isclose(d[2000], 2 * sig / 168)
    assert np.isclose(d[2000 + 167], 2 * sig) and np.allclose(d[2400:], 2 * sig)


def test_positions_have_full_follow_up():
    for n, h in ((2630, 1), (2630, 24), (4531, 48)):
        ps = ij.positions(n, h, follow=1168)
        assert len(ps) == 3 and ps[0] >= cb.W0 + h and all(n - p >= 1168 for p in ps)


def test_detection_delay():
    assert ij.detection_delay([100, 2050, 2600], 2000) == (50, True)
    assert ij.detection_delay([100], 2000) == (ij.TTR_CAP, False)
    assert ij.detection_delay([3500], 2000) == (ij.TTR_CAP, False)       # later than the cap


def test_time_to_recover_hand_example():
    n, tau, w = 3000, 1000, 50
    cov = np.ones(n, bool)
    cov[1000:1300] = False                     # coverage collapses for 300 steps
    ttr, ok = ij.time_to_recover(cov, tau, alpha=0.1, w=w, delta=0.05, L=50)
    # rolling(50) >= 0.85 first again when <= 7 misses in the window: t = 1300 + 42 = 1342
    assert ok and ttr == 342
    ttr2, ok2 = ij.time_to_recover(np.zeros(n, bool), tau, alpha=0.1, w=w)
    assert not ok2 and ttr2 == ij.TTR_CAP


def test_rmst_counts_censored_at_cap():
    assert ij.rmst(np.array([100, 1000]), np.array([True, False])) == 550.0


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("PASS", k)
