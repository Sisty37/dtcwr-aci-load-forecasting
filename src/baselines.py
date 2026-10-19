"""
Baselines for the ICREST 2027 study (plan v5 Section 7.2), all on the Method interface of
calib.py, so they run through the same replay engine (no method sees a score before the
origin at which it becomes known).

Class            Source followed                                           Notes
---------------  --------------------------------------------------------  ------------------------------
ConformalPID     Official repo aangelopoulos/conformal-time-series          quantile tracker (P) +
                 (commit b729c3f), quantile_integrator_log_scorecaster     log-saturated integrator (I);
                                                                           KI = 0 gives "Quantile" (OGD).
                                                                           No scorecaster (decision D9).
ECI / ECICutoff  Authors' repo creator-xi/Error-quantified-Conformal-      ECI() and ECI_cutoff() of the
                 Inference (commit 05086e7), eq. 5 / eq. 6                 code, including its details
                                                                           (see class docstrings).
DtACI            Gibbs & Candes, JMLR 2024 (arXiv 2208.08401), Alg. 2      weighted-average output.
                 (Alg. 1 with the weighted average instead of sampling)
ROCP             Halkiewicz, arXiv 2605.08422 v1, Alg. 1 + Sec. 2.3         fixed window m, empirical
                                                                           quantile WITHOUT the (m+1)
                                                                           correction; m chosen by Winkler.
DriftOCP         Liang, Ren & Chen, arXiv 2602.16537 v2, Algs. 1-2          threshold sigma*sqrt(log 4 tau)
                                                                           (paper: sigma = 24), minimum
                                                                           block length (decision D9).

Delayed feedback (h > 1). None of these methods has its own multi-step rule except the
Conformal PID code, which updates with the score of target t - ahead + 1 (t_pred). We use that
rule for every baseline: the update for target t is applied when its score arrives (origin t),
using the interval that was issued for t. For h = 1 this is exactly the published algorithm,
and tests/test_baselines.py checks that ConformalPID and ECI reproduce the repositories'
quantile sequences to machine precision. (ACI and DTCW-R+ACI keep h interleaved copies,
the variant covered by the delayed-feedback bound.)

Score: P-main uses s = |y - yhat| (symmetric interval). The trackers do not need s >= 0, so
P-pub (Table C) can run them on the signed residuals at alpha/2 per side, as the repos do.
"""
from __future__ import annotations

import math

import numpy as np

import calib as cb

PID_BURNIN = 300            # T_burnin in the PID / ECI Elec2 configs: window of the lr scale
ECI_CUTOFF_WINDOW = 100     # hard-coded in the ECI code (ECI_cutoff sets T_burnin = 100)
# Dimensionless ECI smoothing: in the authors' Elec2 run c = 1 on scores whose robust scale
# sigma0 = median|s| / 0.6745 is 0.0185 (AR forecasts, P-pub rows 300-799; checked in the tests
# when the data are present). On P-main we keep c * sigma0 at that value (decision D9),
# because c = 1 on raw kW scores would switch the error-quantification term off.
ECI_C_REL = 0.0185


def robust_scale(x: np.ndarray) -> float:
    return float(np.median(np.abs(x)) / 0.6745) or 1e-12


# ============================================================================ trackers
class _Tracker(cb.Method):
    """q is a state, not a window quantile. predict() returns the state; observe() updates it
    with the score of target t and the interval that was issued for t."""

    def __init__(self, alpha, h, lr, lr_window: int = PID_BURNIN, proportional_lr: bool = True,
                 q0: float = 0.0):
        super().__init__(alpha, h)
        self.lr, self.lr_window, self.proportional = lr, lr_window, proportional_lr
        self.q0 = q0
        self.qt = q0                 # the accumulated (proportional) part
        self.q = q0                  # what predict() issues
        self.n_miss = 0

    def predict(self, tau):
        m = len(self.scores)
        if m != tau - self.h + 1:
            raise RuntimeError(f"target {tau}: {m} scores known, expected {tau - self.h + 1}")
        self.q_issued[tau] = self.q
        return self.q

    def _lr_t(self, t: int) -> float:
        """lr * range of the scores known BEFORE score t, over the last lr_window of them
        (repo: scores[max(t - T_burnin, 0):t])."""
        if not self.proportional or t == 0:
            return self.lr
        w = self.scores[max(t - self.lr_window, 0):t]
        return self.lr * (max(w) - min(w))

    def _q_for(self, t: int) -> float:
        return self.q_issued.get(t, self.q0)   # targets 0..h-1 were never issued: repo uses q0


class ConformalPID(_Tracker):
    """
    Repo quantile_integrator_log_scorecaster(..., scorecast=False), one step per observed score t:
        err_t  = 1{s_t > q_t}
        qt    <- qt + lr_t * (err_t - alpha)                          (P part)
        I      = KI * tan( (sum_{i<t} err_i - t*alpha) * log(t+1) / (Csat*(t+1)) ), |arg| capped at pi/2
        q_next = qt + I
    KI = 0 gives the plain quantile tracker ("Quantile" in the PID paper, OGD in ECI).
    Defaults are the Elec2 config: Csat = 5, KI = 10.
    """
    name = "pid"

    def __init__(self, alpha, h, lr, Csat=5.0, KI=10.0, lr_window=PID_BURNIN, proportional_lr=True):
        super().__init__(alpha, h, lr, lr_window, proportional_lr)
        self.Csat, self.KI = Csat, KI

    def _integrator(self, t: int) -> float:
        if self.KI == 0:
            return 0.0
        x = (self.n_miss - t * self.alpha) * math.log(t + 1) / (self.Csat * (t + 1))
        if x >= math.pi / 2:
            return math.inf
        if x <= -math.pi / 2:
            return -math.inf
        return self.KI * math.tan(x)

    def _on_observe(self, t, s):
        lr_t = self._lr_t(t)
        err = float(s > self._q_for(t))
        integ = self._integrator(t)                  # uses misses of targets 0..t-1
        self.n_miss += int(err)
        self.qt += lr_t * (err - self.alpha)
        self.q = self.qt + integ


def _eq_term(x: float, c: float) -> float:
    """(s - q) * f'(s - q) for f(x) = 1/(1+exp(-c x)); numerically stable form of the repo's
    x * c * exp(-c x) / (1 + exp(-c x))**2."""
    z = c * x
    if z >= 0:
        e = math.exp(-z)
        return x * c * e / (1 + e) ** 2
    e = math.exp(z)
    return x * c * e / (1 + e) ** 2


class ECI(_Tracker):
    """
    Authors' ECI() code. Note: in the code the error-quantification (EQ) term is NOT accumulated:
        qt     <- qt + lr_t * (err_t - alpha)
        q_next  = qt + lr_t * EQ_t,   EQ_t = (s_t - q_t) f'(s_t - q_t),  f = sigmoid(c x)
    (paper eq. 5 writes q_{t+1} = q_t + eta[...], i.e. the EQ term accumulates. The reported
    numbers come from the code, so the code form is the default; paper_form=True gives eq. 5 /
    eq. 6 as printed, for a sensitivity check. Decision D9.)
    c: absolute smoothing; or c_rel (P-main): c = c_rel / sigma0 of the warm-up scores.
    """
    name = "eci"

    def __init__(self, alpha, h, lr, c: float = 1.0, c_rel: float | None = None,
                 lr_window=PID_BURNIN, proportional_lr=True, paper_form: bool = False):
        super().__init__(alpha, h, lr, lr_window, proportional_lr)
        self.c, self.c_rel, self.paper_form = c, c_rel, paper_form

    def setup(self, warm):
        if self.c_rel is not None:
            self.c = self.c_rel / robust_scale(np.asarray(warm, float))

    def _eq(self, t, s, q):
        return _eq_term(s - q, self.c)

    def _on_observe(self, t, s):
        lr_t = self._lr_t(t)
        q_t = self._q_for(t)
        err = float(s > q_t)
        if not self._update_now(t):
            return
        eq = self._eq(t, s, q_t)
        if self.paper_form:                          # eq. 5 / 6 as printed: q accumulates
            self.q = self.qt = self.q + lr_t * (err - self.alpha + eq)
            return
        self.qt += lr_t * (err - self.alpha)
        self.q = self.qt + lr_t * eq

    def _update_now(self, t):
        return True


class ECICutoff(ECI):
    """
    Authors' ECI_cutoff() code (paper eq. 6): EQ term used only when
    |s_t - q_t| >= h_cut * (max - min of the last 100 known scores); lr window also 100;
    no update at all for the first score (the code `continue`s at t_pred == 0).
    """
    name = "eci_cutoff"

    def __init__(self, alpha, h, lr, c: float = 1.0, c_rel: float | None = None,
                 h_cut: float = 1.0, window: int = ECI_CUTOFF_WINDOW, proportional_lr=True,
                 paper_form: bool = False):
        super().__init__(alpha, h, lr, c, c_rel, window, proportional_lr, paper_form)
        self.h_cut = h_cut

    def _update_now(self, t):
        return t > 0

    def _eq(self, t, s, q):
        w = self.scores[max(t - self.lr_window, 0):t]
        rng = (max(w) - min(w)) if w else 0.0
        return 0.0 if abs(s - q) < self.h_cut * rng else _eq_term(s - q, self.c)


# ============================================================================ DtACI
DTACI_GAMMAS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)


def dtaci_defaults(alpha: float, k: int, I: int = 500) -> tuple[float, float]:
    """(sigma, eta) of Gibbs & Candes: sigma = 1/(2I); eta from E[l^2] ~ alpha^2 (1-alpha)^2 / 3."""
    sigma = 1 / (2 * I)
    eta = math.sqrt(3 / I) * math.sqrt(math.log(k * I) + 2) / (alpha * (1 - alpha))
    return sigma, eta


def pinball(beta: float, theta: float, alpha: float) -> float:
    d = beta - theta
    return alpha * d - min(0.0, d)


class DtACI(cb.Method):
    """
    Dynamically-tuned ACI (Alg. 2: output the weighted average of the experts' alphas).
    Each expert i runs ACI with step gamma_i on the quantile of the last W known scores.
    beta_t = largest beta with s_t covered by the issued window at level 1 - beta.
    """
    name = "dtaci"

    def __init__(self, alpha, h, gammas=DTACI_GAMMAS, W: int = 10**7, sigma: float | None = None,
                 eta: float | None = None, I: int = 500):
        super().__init__(alpha, h)
        self.g = np.asarray(gammas, float)
        k = len(self.g)
        s0, e0 = dtaci_defaults(alpha, k, I)
        self.sigma = s0 if sigma is None else sigma
        self.eta = e0 if eta is None else eta
        self.W = W
        self.a = np.full(k, alpha)            # expert alphas
        self.w = np.ones(k)                   # expert weights
        self.pending: dict[int, tuple] = {}   # target -> (m, expert q's) at issue

    def _win(self, m):
        return np.asarray(self.scores[max(0, m - self.W):m])

    def predict(self, tau):
        m = len(self.scores)
        if m != tau - self.h + 1:
            raise RuntimeError(f"target {tau}: {m} scores known, expected {tau - self.h + 1}")
        win = self._win(m)
        p = self.w / self.w.sum()
        a_bar = float(p @ self.a)
        qs = np.array([cb.qhat(win, 1 - a) if len(win) else math.inf for a in self.a])
        q = cb.qhat(win, 1 - a_bar) if len(win) else math.inf
        self.pending[tau] = (m, qs, self.a.copy())
        self.q_issued[tau] = q
        return q

    def beta(self, win: np.ndarray, s: float) -> float:
        """sup{beta : s <= qhat(win, 1 - beta)} = 1 - #{win < s} / (n + 1)."""
        n = len(win)
        return 1.0 - np.count_nonzero(win < s) / (n + 1)

    def _on_observe(self, t, s):
        if t not in self.pending:
            return
        m, qs, a_issue = self.pending.pop(t)
        b = self.beta(self._win(m), s)
        loss = np.array([pinball(b, a, self.alpha) for a in a_issue])
        wbar = self.w * np.exp(-self.eta * loss)
        Wbar = wbar.sum()
        self.w = (1 - self.sigma) * wbar + Wbar * self.sigma / len(self.w)
        self.w /= self.w.sum()                      # rescaling does not change p (homogeneous)
        err = (s > qs).astype(float)
        self.a = self.a + self.g * (self.alpha - err)


# ============================================================================ ROCP
def rocp_quantile(window: np.ndarray, p: float) -> float:
    """Empirical p-quantile inf{x : F_n(x) >= p}, no (n+1) correction (ROCP Alg. 1, Remark 1)."""
    if p <= 0:
        return -math.inf
    n = len(window)
    if p > 1 or n == 0:
        return math.inf
    k = max(1, math.ceil(p * n - 1e-12))
    return float(np.partition(window, k - 1)[k - 1])


class ROCP(cb.FixedWindow):
    name = "rocp"

    def _q(self, window, p):
        return rocp_quantile(window, p)


def rocp_window_grid(T: int, n: int = 16, lo: float = 0.10, hi: float = 4.0) -> list[int]:
    """ROCP Sec. 2.3 / 4.1: m = c * T^(2/3), c on a grid in [0.10, 4.0] (16 points here)."""
    base = T ** (2 / 3)
    return sorted({max(2, int(round(c * base))) for c in np.geomspace(lo, hi, n)})


def winkler(y, lo, hi, alpha):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return (hi - lo) + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


# ============================================================================ DriftOCP
class DriftOCP(cb.Method):
    """
    Liang, Ren & Chen, Algs. 1-2. A stage n runs rounds r = 1, 2, ... of at most 3^r scores.
    Within a round q is fixed. After each score, DRIFTDETECT scans every block [j, now] of the
    current round: Z_j = |sum (1{s_l <= q} - (1 - alpha))| / sqrt(len). If max Z_j exceeds
    sigma * sqrt(log(4 tau_{n,r})) (tau_{n,r} = 1-based start time of the round; the paper's
    theory uses sigma = 24), a new stage starts with round 1 and the SAME q. Otherwise, when the
    round is full, q is set to argmin_q |sum_j (1{s_j > q} - alpha)| over the round's scores.
    min_len: shortest block scanned (decision D9; the paper scans every block).
    """
    name = "driftocp"

    def __init__(self, alpha, h, sigma: float = 1.0, min_len: int = 10, q0: float = 0.0):
        super().__init__(alpha, h)
        self.sigma, self.min_len = sigma, min_len
        self.q = q0
        self.stage, self.r = 1, 1
        self.round_start = 0          # stream index of the first score of the current round
        self.round: list[float] = []
        self.alarms: list[tuple] = []

    def predict(self, tau):
        m = len(self.scores)
        if m != tau - self.h + 1:
            raise RuntimeError(f"target {tau}: {m} scores known, expected {tau - self.h + 1}")
        self.q_issued[tau] = self.q
        return self.q

    def threshold(self) -> float:
        return self.sigma * math.sqrt(math.log(4 * (self.round_start + 1)))

    @staticmethod
    def round_quantile(x: np.ndarray, alpha: float) -> float:
        n = len(x)
        k = min(n, max(1, n - int(round(alpha * n))))     # #{s > q} = n - k  closest to alpha*n
        return float(np.partition(x, k - 1)[k - 1])

    def _on_observe(self, t, s):
        self.round.append(s)
        d = (np.asarray(self.round) <= self.q).astype(float) - (1 - self.alpha)
        z, j = cb.max_block_z(d, self.min_len)
        if z > self.threshold():
            self.alarms.append((t, "miss", self.round_start + j))
            self.stage, self.r = self.stage + 1, 1          # q carried over
            self.round_start, self.round = t + 1, []
            return
        if len(self.round) >= 3 ** self.r:
            self.q = self.round_quantile(np.asarray(self.round), self.alpha)
            self.r += 1
            self.round_start, self.round = t + 1, []
