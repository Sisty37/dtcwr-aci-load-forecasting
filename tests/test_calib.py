"""Tests for src/calib.py (plan v5 Sections 4.1, 4.4, 4.5, 6.5).
Run:  python -m pytest tests -q
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import calib as cb  # noqa: E402

A = 0.1


def _methods(h):
    return {
        "fixed168": cb.FixedWindow(A, h, 168),
        "fixed720": cb.FixedWindow(A, h, 720),
        "aci": cb.ACI(A, h, gamma=0.005),
        "dtcwr": cb.DTCWR(A, h),
        "dtcwr_aci": cb.DTCWR(A, h, aci_gamma=0.005),
    }


def _run(m, s):
    return cb.replay(m, y=s, yhat=np.zeros_like(s))


# ---------------------------------------------------------------- quantile convention
def test_qhat_convention():
    w = np.random.default_rng(0).exponential(size=50)
    ps = np.linspace(-0.2, 1.2, 1401)
    qs = [cb.qhat(w, p) for p in ps]
    assert all(a <= b for a, b in zip(qs[:-1], qs[1:]))
    assert cb.qhat(w, 0) == -math.inf and cb.qhat(w, 1) == math.inf
    assert cb.qhat(w, 1.01) == math.inf and cb.qhat(w, -0.01) == -math.inf
    assert cb.qhat(w, 0.9) == np.sort(w)[math.ceil(0.9 * 51) - 1]


# ---------------------------------------------------------------- leakage (plan 6.5 test 1)
def test_no_leakage_all_methods():
    rng = np.random.default_rng(1)
    s = np.abs(rng.standard_t(3, size=4000))
    T = 2500
    for h in (1, 24):
        for name, _ in _methods(h).items():
            q1 = _run(_methods(h)[name], s)["q"]
            s2 = s.copy()
            s2[T + 1:] *= 50                       # change scores of targets after T only
            q2 = _run(_methods(h)[name], s2)["q"]
            upto = T + h                           # intervals issued at origins <= T
            assert np.array_equal(q1[:upto + 1], q2[:upto + 1], equal_nan=True), (name, h)


# ---------------------------------------------------------------- iid sanity gate (plan step 5)
def test_iid_coverage_gate():
    rng = np.random.default_rng(2)
    s = np.abs(rng.normal(size=20000))
    for h in (1, 24):
        for name, m in _methods(h).items():
            r = _run(m, s)
            cov = r["covered"][r["evaluated"]].mean()
            assert 0.89 <= cov <= 0.91, (name, h, cov)


# ---------------------------------------------------------------- DTCW-R behaviour (plan 4.5)
def _feed(m, s):
    m.setup(s[:cb.W0])
    for t, v in enumerate(s):
        m.observe(t, float(v))


def _shift_stream(changes, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    sc = np.ones(n)
    for tau, f in changes:
        sc[tau:] = f
    return np.abs(rng.normal(size=n)) * sc


def test_T1_anchor_and_floor_membership():
    s = _shift_stream([(2000, 3.0)])
    m = cb.DTCWR(A, 1, W_S=48, W_L=720, H=6, k=0.5, c=24)
    m.setup(s[:cb.W0])
    first = None
    for t, v in enumerate(s):
        m.observe(t, float(v))
        post = [a for a in m.alarms if a[0] >= 1990]     # pre-shift false alarms are tested in T5
        if post and first is None:
            first = post[0]
        if first and t >= first[0]:
            mm = t + 1
            n_post = m.n_post(mm)
            n = len(m._window(mm))
            pre = n - min(n, n_post)
            if n_post < m.W_S:
                assert pre == m.W_S - n_post
            else:
                assert pre == 0
    assert first is not None and 2000 <= first[0] <= 2100 and abs(first[2] - 2000) <= 15
    assert len([a for a in m.alarms if 2100 < a[0] < 2700]) == 0   # no re-alarm during the shift


def test_T2_cooldown_keeps_anchor():
    m = cb.DTCWR(A, 1, W_S=48, W_L=720, H=6, k=0.5, c=1000)
    warm = np.abs(np.random.default_rng(3).normal(size=cb.W0))
    m.setup(warm)
    for t, v in enumerate(warm):
        m.observe(t, float(v))
    t = cb.W0
    for v in [m.sigma0 * math.e ** 3] * 40 + [m.sigma0] * 100 + [m.sigma0 * math.e ** 4] * 40:
        m.observe(t, float(v))
        t += 1
    assert len(m.alarms) == 1                       # second burst fell inside the cooldown


def test_T3_repeated_alarms_drop_old_scores():
    s = _shift_stream([(2000, 3.0), (3500, 0.5)])
    m = cb.DTCWR(A, 1, W_S=48, W_L=720, H=6, k=0.5, c=24)
    _feed(m, s)
    second = [a for a in m.alarms if a[0] > 3400][0]
    a2 = m.anchor if m.anchor >= 3400 else second[2]
    assert abs(a2 - 3500) <= 40
    mm = len(s)
    assert m.n_post(mm) >= m.W_S and (np.arange(mm - len(m._window(mm)), mm) >= a2).all()


def test_T4_downward_shift_uses_g_minus():
    s = _shift_stream([(2500, 0.3)])
    m = cb.DTCWR(A, 1, W_S=48, W_L=720, H=6, k=0.5, c=24)
    _feed(m, s)
    d = [a for a in m.alarms if a[0] > 2400][0]
    assert d[1] == "-" and abs(d[2] - 2500) <= 30


def test_T5_false_alarm_rate():
    s = _shift_stream([], n=21000, seed=3)
    m = cb.DTCWR(A, 1, W_S=48, W_L=720, H=6, k=0.5, c=24)
    _feed(m, s)
    assert len(m.alarms) / ((len(s) - cb.W0) / 1000) <= 1.0


# ---------------------------------------------------------------- ACI delayed bound (plan 4.4)
def test_interleaved_aci_bound():
    rng = np.random.default_rng(4)
    for h in (1, 24):
        for g in (0.005, 0.05):
            z = np.zeros(6000)
            e = rng.standard_t(2, size=6000)
            for t in range(1, 6000):
                z[t] = 0.9 * z[t - 1] + e[t]
            s = np.abs(z) * np.where(np.arange(6000) > 3000, 5, 1)
            m = cb.ACI(A, h, gamma=g, W=200)
            r = _run(m, s)
            ev = np.zeros(len(s), bool)
            ev[h:] = True                         # every issued interval
            err = ~(s <= r["q"])
            T = int(ev.sum())
            gap = abs(err[ev].mean() - A)
            C = max(A, 1 - A) + g
            assert gap <= h * C / (g * T) + h / T, (h, g, gap)


# ---------------------------------------------------------------- metrics
def test_winkler_hand_example():
    y = np.array([0.0, 2.0, -3.0])
    yhat = np.zeros(3)
    q = np.array([1.0, 1.0, 1.0])
    r = cb.interval_metrics(y, yhat, q, np.ones(3, bool), alpha=0.1, w=2)
    # widths 2; misses: 0, (2-1)=1, (-1-(-3))=2  -> winkler = 2 + 20*[0,1,2] -> mean 22
    assert abs(r["winkler"] - 22.0) < 1e-12 and abs(r["coverage"] - 1 / 3) < 1e-12
    assert r["worst_local_coverage"] == 0.0


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("PASS", k)
