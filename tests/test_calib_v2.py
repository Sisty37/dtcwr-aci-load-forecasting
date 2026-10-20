"""Tests for src/calib_v2.py (DTCW-R v2, decision D13).
Run:  python -m pytest tests -q
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import calib as cb  # noqa: E402
import calib_v2 as v2  # noqa: E402


def _stream(seed=0, n=6000, k=2.5, at=3000):
    rng = np.random.default_rng(seed)
    s = np.abs(rng.normal(0, 1, n))
    s[at:] *= k
    return s


def test_v2_without_miss_and_reset_is_exactly_v1():
    s = _stream()
    z = np.zeros_like(s)
    for pw in (False, True):
        for g in (None, 0.01):
            for h in (1, 24):
                a = cb.DTCWR(0.1, h, W_S=96, W_L=720, H=8, c=24, aci_gamma=g, prewhiten=pw)
                b = v2.DTCWR2(0.1, h, W_S=96, W_L=720, H=8, c=24, aci_gamma=g, prewhiten=pw)
                ra, rb = cb.replay(a, s, z), cb.replay(b, s, z)
                assert a.alarms == b.alarms and np.array_equal(ra["q"], rb["q"], equal_nan=True)


def test_dual_fires_on_miss_when_magnitude_is_silent():
    s = _stream()
    d = v2.DTCWR2(0.1, 1, W_S=96, W_L=720, H=1e6, H_miss=1.5)
    cb.replay(d, s, np.zeros_like(s))
    post = [a for a in d.alarms if a[0] >= 3000]
    assert post and post[0][1] == "miss" and abs(post[0][2] - 3000) < 100


def test_aci_reset_sets_copies_to_alpha_at_alarm():
    s = _stream()
    d = v2.DTCWR2(0.1, 24, W_S=96, W_L=720, H=8, aci_gamma=0.05, aci_reset=True)
    seen = []
    orig = d._alarm

    def spy(t, trig, tau_hat):
        orig(t, trig, tau_hat)
        seen.append(d.copies.copy())
    d._alarm = spy
    cb.replay(d, s, np.zeros_like(s))
    assert seen and all(np.allclose(c, 0.1) for c in seen)


def test_calibrate_dual_meets_each_target():
    rng = np.random.default_rng(1)
    sc = [np.abs(rng.normal(0, 1, 2000)) for _ in range(3)]
    r = v2.calibrate_dual(sc, 24, {"W_S": 48, "W_L": 336, "k": 0.5, "c": 24}, 1, length=6000)
    assert r["fa_mag"] <= 0.25 and r["fa_miss"] <= 0.25 and math.isfinite(r["H_miss"])
    assert r["fa_union"] <= 0.25 + 0.25 + 0.15          # union bound, Monte Carlo slack


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("PASS", k)
