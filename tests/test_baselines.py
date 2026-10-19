"""Tests for src/baselines.py (plan v5 Section 7.2 replication checks + leakage + i.i.d. gate).
Run:  python -m pytest tests -q

The exact-replication tests load the ORIGINAL code of the two repositories:
    data/conformal-time-series/                (Conformal PID, already cloned)
    data/Error-quantified-Conformal-Inference/ (ECI authors' code)
They are skipped if a repository is missing.
"""
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import baselines as bl  # noqa: E402
import calib as cb  # noqa: E402

PID_REPO = ROOT / "data" / "conformal-time-series"
ECI_REPO = ROOT / "data" / "Error-quantified-Conformal-Inference"
ALPHA = 0.1


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:          # script mode
        raise RuntimeError("SKIP " + msg)


def load_repo_methods(repo: Path, name: str):
    """Import <repo>/core/methods.py as-is. Only the imports the tested functions do not use
    (statsmodels for the scorecaster, tqdm progress bars) are stubbed if missing; numpy 2
    removed np.infty, which the repo code uses, so it is restored as an alias of np.inf."""
    path = repo / "core" / "methods.py"
    if not path.exists():
        _skip(f"{repo.name} not cloned")
    if not hasattr(np, "infty"):
        np.infty = np.inf
    if "statsmodels" not in sys.modules and importlib.util.find_spec("statsmodels") is None:
        for mod in ("statsmodels", "statsmodels.tsa", "statsmodels.tsa.api",
                    "statsmodels.tsa.forecasting", "statsmodels.tsa.forecasting.theta"):
            m = types.ModuleType(mod)
            m.ExponentialSmoothing = m.ThetaModel = None
            sys.modules[mod] = m
    if "tqdm" not in sys.modules and importlib.util.find_spec("tqdm") is None:
        t = types.ModuleType("tqdm")
        t.tqdm = lambda x, *a, **k: x
        sys.modules["tqdm"] = t
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ours_q(method, s):
    """Run one of our methods through the replay engine (h = 1, warm-up irrelevant here) and
    return the issued q for targets 1..N-1 (target 0 is never issued; the repo's q[0] = 0)."""
    method.setup(s[:cb.W0])
    q = np.full(len(s), np.nan)
    q[0] = method.q0
    for tau in range(1, len(s)):
        method.observe(tau - 1, float(s[tau - 1]))
        q[tau] = method.predict(tau)
    return q


def _streams():
    rng = np.random.default_rng(1)
    a = np.abs(rng.normal(0, 1, 1500))
    a[800:] *= 3
    return [a, rng.normal(0.2, 1, 1500)]          # absolute and signed scores


def pub_scores():
    """Signed P-pub scores y - yhat (AR forecasts) from the PID repo, if present."""
    csv = PID_REPO / "tests" / "datasets" / "elec2.csv"
    npz = PID_REPO / "tests" / "datasets" / "proc" / "elec2" / "ar.npz"
    if not (csv.exists() and npz.exists()):
        return []
    import pandas as pd
    y = pd.read_csv(csv)["nswdemand"].to_numpy(float)[:2000]
    f = np.asarray(np.load(npz, allow_pickle=True)["forecasts"], float)
    return [y - f, f - y]


# ----------------------------------------------------------------------------- exact replication
def test_pid_and_quantile_match_official_repo_exactly():
    repo = load_repo_methods(PID_REPO, "pid_repo_methods")
    for s in _streams() + pub_scores():
        for lr in (1, 0.5, 0.1, 0.05):
            ref = repo.quantile_integrator_log_scorecaster(
                s, ALPHA, lr, None, 300, 5, 10, True, 1, scorecast=False)["q"]
            mine = ours_q(bl.ConformalPID(ALPHA, 1, lr, Csat=5, KI=10), s)
            assert np.allclose(mine, ref, rtol=1e-10, atol=1e-12, equal_nan=True), lr
            ref0 = repo.quantile(s, ALPHA, lr, 1, T_burnin=300)["q"]
            mine0 = ours_q(bl.ConformalPID(ALPHA, 1, lr, Csat=1.0, KI=0), s)
            assert np.allclose(mine0, ref0, rtol=1e-10, atol=1e-12), lr


def test_eci_and_eci_cutoff_match_authors_code_exactly():
    repo = load_repo_methods(ECI_REPO, "eci_repo_methods")
    for s in _streams() + pub_scores():
        for lr in (1, 0.5, 0.1, 0.05):
            ref = repo.ECI(s, ALPHA, lr, 300, 1)["q"]
            mine = ours_q(bl.ECI(ALPHA, 1, lr, c=1.0), s)
            assert np.allclose(mine, ref, rtol=1e-9, atol=1e-12), lr
            ref = repo.ECI_cutoff(s, ALPHA, lr, 300, 1)["q"]
            mine = ours_q(bl.ECICutoff(ALPHA, 1, lr, c=1.0), s)
            assert np.allclose(mine, ref, rtol=1e-9, atol=1e-12), lr


def test_eci_c_rel_constant_matches_pub_data():
    s = pub_scores()
    if not s:
        _skip("P-pub data not present")
    assert abs(bl.robust_scale(np.abs(s[0][300:800])) - bl.ECI_C_REL) < 5e-4


# ----------------------------------------------------------------------------- leakage + gate
def all_methods(h):
    return {
        "pid": bl.ConformalPID(ALPHA, h, 0.1),
        "quantile": bl.ConformalPID(ALPHA, h, 0.1, KI=0),
        "eci": bl.ECI(ALPHA, h, 0.1, c_rel=bl.ECI_C_REL),
        "eci_cutoff": bl.ECICutoff(ALPHA, h, 0.1, c_rel=bl.ECI_C_REL),
        "dtaci": bl.DtACI(ALPHA, h),
        "rocp": bl.ROCP(ALPHA, h, W=300),
        "driftocp": bl.DriftOCP(ALPHA, h, sigma=1.0),
    }


def test_predict_refuses_unknown_scores():
    for name, m in all_methods(24).items():
        m.setup(np.ones(cb.W0))
        for t in range(30):
            m.observe(t, 1.0)
        try:
            m.predict(30 + 24)            # needs 31 known scores, only 30 known
        except RuntimeError:
            continue
        raise AssertionError(f"{name} issued an interval with unseen scores")


def test_iid_coverage_gate():
    """Plan v5 Section 10 step 5 gate: every method 0.89-0.91 on i.i.d. data (h = 1 and 24)."""
    rng = np.random.default_rng(7)
    for h in (1, 24):
        for name, m in all_methods(h).items():
            covs = []
            for rep in range(3):
                e = rng.standard_t(5, 6000)
                out = cb.replay(m if rep == 0 else all_methods(h)[name], np.zeros_like(e), -e)
                covs.append(out["covered"][out["evaluated"]].mean())
            cov = float(np.mean(covs))
            assert 0.89 <= cov <= 0.91, (name, h, cov)


def test_future_scores_do_not_change_issued_intervals():
    """Changing scores of targets >= k must leave every interval for targets < k + h unchanged."""
    rng = np.random.default_rng(8)
    s = np.abs(rng.normal(0, 1, 1500))
    k = 1100
    s2 = s.copy()
    s2[k:] = 100 * s2[k:] + 7
    for h in (1, 24):
        for name in all_methods(h):
            q1 = cb.replay(all_methods(h)[name], s, np.zeros_like(s))["q"]
            q2 = cb.replay(all_methods(h)[name], s2, np.zeros_like(s))["q"]
            assert np.array_equal(q1[:k + h], q2[:k + h], equal_nan=True), (name, h)


# ----------------------------------------------------------------------------- DtACI
def test_dtaci_defaults_and_beta():
    sig, eta = bl.dtaci_defaults(0.1, 8, 500)
    assert sig == 0.001 and abs(eta - 2.77) < 0.02          # paper's appendix: eta ~ 2.8
    m = bl.DtACI(0.1, 1)
    win = np.arange(1.0, 10.0)                              # 1..9, n = 9
    for s in (0.5, 4.5, 9.5):
        b = m.beta(win, s)
        assert s <= cb.qhat(win, 1 - b + 1e-12)            # covered at beta
        assert s > cb.qhat(win, 1 - b - 0.02)              # not covered just above beta
    assert math.isclose(bl.pinball(0.3, 0.1, 0.1), 0.02) and abs(bl.pinball(0.1, 0.3, 0.1) - 0.18) < 1e-12


def test_dtaci_tracks_best_expert_after_shift():
    """Replication check (plan 7.2, DtACI's main claim): after a shift DtACI's miss rate is close
    to the best single ACI expert of its grid and far below the slowest one."""
    rng = np.random.default_rng(3)
    s = np.abs(rng.normal(0, 1, 6000))
    s[3000:] *= 4

    def miss(m):
        out = cb.replay(m, s, np.zeros_like(s))
        return 1 - out["covered"][3000:3500].mean()

    experts = [miss(cb.ACI(ALPHA, 1, g)) for g in bl.DTACI_GAMMAS]
    d = miss(bl.DtACI(ALPHA, 1))
    assert d <= min(experts) + 0.03 and d < max(experts) - 0.1, (d, experts)


# ----------------------------------------------------------------------------- ROCP
def test_rocp_quantile_has_no_finite_sample_correction():
    w = np.arange(1.0, 11.0)
    assert bl.rocp_quantile(w, 0.9) == 9.0                 # ceil(0.9*10) = 9th
    assert cb.qhat(w, 0.9) == 10.0                         # split conformal: ceil(0.9*11) = 10th
    g = bl.rocp_window_grid(2630)
    assert len(g) == 16 and g[0] == round(0.1 * 2630 ** (2 / 3)) and g[-1] == round(4 * 2630 ** (2 / 3))


def test_rocp_winkler_selection_behaviour():
    """Replication check (plan 7.2): on a locally stationary series the Winkler-optimal window is
    interior; on i.i.d. data it is the largest window (ROCP Sec. 2.3)."""
    rng = np.random.default_rng(11)
    n = 6000
    t = np.arange(n)
    loc = np.abs(rng.normal(0, 1, n)) * (1 + 0.8 * np.sin(2 * np.pi * t / 1500))
    iid = np.abs(rng.normal(0, 1, n))
    grid = bl.rocp_window_grid(n)

    def best(s):
        res = {}
        for W in grid:
            out = cb.replay(bl.ROCP(ALPHA, 1, W=W), s, np.zeros_like(s))
            ev = out["evaluated"]
            res[W] = float(np.mean(bl.winkler(s[ev], -out["q"][ev], out["q"][ev], ALPHA)))
        return min(res, key=res.get)

    b_loc, b_iid = best(loc), best(iid)
    assert grid[0] < b_loc < grid[-1], b_loc
    assert b_iid >= grid[-4], b_iid


# ----------------------------------------------------------------------------- DriftOCP
def test_driftocp_rounds_and_reset_keep_q():
    m = bl.DriftOCP(ALPHA, 1, sigma=1e9)                    # no detection: rounds only
    rng = np.random.default_rng(0)
    for t in range(3 + 9 + 27):
        m.observe(t, float(abs(rng.normal())))
    assert m.r == 4 and m.round_start == 39
    # constant scores: round 1 (3 misses at q0 = 0) sets q = 1; then every score is covered,
    # block error 0.1 per step, which a tiny threshold flags inside round 2
    m2 = bl.DriftOCP(ALPHA, 1, sigma=0.1, min_len=5)
    for t in range(3 + 5):
        m2.observe(t, 1.0)
    assert m2.alarms == [(7, "miss", 3)]
    assert m2.q == 1.0 and m2.stage == 2 and m2.r == 1 and m2.round_start == 8   # q carried over


def test_driftocp_detects_variance_shift_and_recovers():
    rng = np.random.default_rng(5)
    s = np.abs(rng.normal(0, 1, 8000))
    s[4000:] *= 3
    m = bl.DriftOCP(ALPHA, 1, sigma=1.0)
    out = cb.replay(m, s, np.zeros_like(s))
    post = [a for a in m.alarms if a[0] >= 4000]
    assert post and post[0][0] - 4000 < 300
    cov = out["covered"]
    assert cov[6000:].mean() > 0.85


def test_driftocp_regret_shape_setting1():
    """Replication check (plan 7.2, DriftOCP Setting 1, piecewise-constant variance): with drift
    detection the cumulative coverage error grows sub-linearly and below the no-detection run."""
    rng = np.random.default_rng(9)
    T = 9000
    scale = np.repeat([1.0, 3.0, 0.7, 2.0], T // 4)
    s = np.abs(rng.normal(0, 1, T)) * scale

    def cum_err(sigma):
        out = cb.replay(bl.DriftOCP(ALPHA, 1, sigma=sigma), s, np.zeros_like(s))
        miss = 1 - out["covered"][out["evaluated"]]
        return np.abs(np.cumsum(miss - ALPHA))

    with_det, without = cum_err(1.0), cum_err(1e9)
    blocks = [with_det[k * 2000:(k + 1) * 2000].max() for k in range(4)]
    assert with_det[-1] < without[-1]
    assert max(blocks) < 0.05 * len(with_det)


# ----------------------------------------------------------------------------- ECI behaviour
def test_eci_on_authors_changepoint_data():
    """Replication check (plan 7.2) on the authors' own synthetic change-point series and cached
    forecasts: our ECI / ECI-cutoff equal their code, and reach 90% coverage as in their paper
    (two-sided signed residuals at alpha/2 per side, T_burnin = 100, as in their config)."""
    repo = load_repo_methods(ECI_REPO, "eci_repo_methods")
    d = ECI_REPO / "tests" / "datasets"
    import pandas as pd
    y = pd.read_csv(d / "changepoint.csv")["y"].to_numpy(float)
    f = np.asarray(np.load(d / "proc" / "changepoint_test" / "ar.npz", allow_pickle=True)["forecasts"], float)
    for lr in (1, 0.1):
        qs = []
        for s in (y - f, f - y):
            ref = repo.ECI(s, ALPHA / 2, lr, 100, 1)["q"]
            mine = ours_q(bl.ECI(ALPHA / 2, 1, lr, c=1.0, lr_window=100), s)
            assert np.allclose(mine, ref, rtol=1e-9, atol=1e-12)
            assert np.allclose(ours_q(bl.ECICutoff(ALPHA / 2, 1, lr, c=1.0), s),
                               repo.ECI_cutoff(s, ALPHA / 2, lr, 100, 1)["q"], rtol=1e-9, atol=1e-12)
            qs.append(mine)
        cov = ((y <= f + qs[0]) & (y >= f - qs[1]))[101:].mean()
        assert 0.89 <= cov <= 0.91, (lr, cov)


def test_eci_paper_form_is_available_and_differs():
    """eq. 5 as printed (EQ term accumulated) is kept as an option; it is NOT the code form."""
    rng = np.random.default_rng(2)
    s = np.abs(rng.normal(0, 1, 1500))
    a = cb.replay(bl.ECI(ALPHA, 1, 0.05, c=1.0), s, np.zeros_like(s))["q"]
    b = cb.replay(bl.ECI(ALPHA, 1, 0.05, c=1.0, paper_form=True), s, np.zeros_like(s))["q"]
    assert not np.allclose(a[cb.W0:], b[cb.W0:])


# ----------------------------------------------------------------------------- miss-signal detector
def test_dtcwr_miss_signal_detects_coverage_collapse_and_has_tau_hat():
    rng = np.random.default_rng(4)
    s = np.abs(rng.normal(0, 1, 5000))
    s[3000:] *= 3
    m = cb.DTCWR(ALPHA, 1, H=4.0, signal="miss", W_S=48, W_L=720)
    cb.run_stream(m, s)
    post = [a for a in m.alarms if a[0] >= 3000]
    assert post and post[0][0] - 3000 < 200 and abs(post[0][2] - 3000) < 100
    r = cb.DTCWR(ALPHA, 1, H=4.0, signal="miss", window_rule="reset")
    cb.run_stream(r, s)
    assert any(a[0] >= 3000 for a in r.alarms) and r.anchor == r.alarms[-1][0] + 1


def test_max_block_z():
    d = np.r_[np.zeros(10), np.ones(4)]
    z, j = cb.max_block_z(d, 1)
    assert j == 10 and math.isclose(z, 2.0)
    assert cb.max_block_z(np.ones(3), 5) == (0.0, 0)


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            try:
                v()
                print("PASS", k)
            except RuntimeError as e:
                if str(e).startswith("SKIP"):
                    print(e, k)
                else:
                    raise
