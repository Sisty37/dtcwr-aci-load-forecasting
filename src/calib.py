"""
Step 5 (batch 1) of the ICREST 2027 plan (v5, Sections 4 and 7.2): calibration methods
and the replay engine every method shares.

Methods in this file
- FixedWindow(W)                 split-conformal quantile of the last W available scores
- ACI(gamma, W)                  Gibbs & Candes (2021) update on the window quantile;
                                 h interleaved copies when feedback is delayed (h > 1)
- DTCWR(W_S, W_L, H, k, c)       residual-magnitude CUSUM + change-point-anchored window
- DTCWR + ACI                    DTCWR(..., aci_gamma=g)

Replay protocol (identical for every method)
- One stream per series = VAL targets followed by TEST targets, in time order.
  Stream index tau = position of the TARGET in that stream.
- Interval for target tau is issued at origin tau - h. At that moment the method has
  observed exactly the scores of targets 0 .. tau - h (each score becomes known when its
  target is observed). The engine enforces this: observe(tau - h) is called BEFORE
  predict(tau), and predict never receives anything newer.
- Score s = |y - yhat| (symmetric interval yhat +/- q).
- Warm-up: the first W0 = 500 targets are observed but never evaluated. Constants a method
  fits (e.g. the CUSUM reference) use only those W0 scores. Evaluation starts at
  tau0 = W0 + h, the first target whose issuing origin has seen all W0 warm-up scores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

W0 = 500


# ----------------------------------------------------------------------------- quantile
def qhat(window: np.ndarray, p: float) -> float:
    """Plan v5 Section 4.4 convention. Non-decreasing in p on all of R."""
    if p <= 0:
        return -math.inf
    if p > 1:
        return math.inf
    n = len(window)
    k = math.ceil(p * (n + 1))
    if k > n:
        return math.inf
    return float(np.partition(window, k - 1)[k - 1])


# ----------------------------------------------------------------------------- base
class Method:
    """Subclasses implement _window(m) and optionally _on_observe(t, s)."""
    name = "base"

    def __init__(self, alpha: float, h: int, aci_gamma: float | None = None):
        self.alpha, self.h, self.gamma = alpha, h, aci_gamma
        self.copies = np.full(h, alpha) if aci_gamma is not None else None
        self.scores: list[float] = []
        self.q_issued: dict[int, float] = {}
        self._buf = np.empty(0)

    def arr(self, lo: int, hi: int) -> np.ndarray:
        """Read-only view of scores[lo:hi] (no copy)."""
        return self._buf[max(0, lo):hi]

    def setup(self, warm: np.ndarray) -> None:
        """Called once with the W0 warm-up scores, before any evaluated interval."""

    def observe(self, t: int, s: float) -> None:
        if t != len(self.scores):
            raise RuntimeError(f"scores must arrive in order: got {t}, expected {len(self.scores)}")
        self.scores.append(s)
        if t >= len(self._buf):                                   # numpy mirror for fast windows
            self._buf = np.concatenate([self._buf, np.empty(max(1024, len(self._buf)))])
        self._buf[t] = s
        if self.copies is not None and t in self.q_issued:          # ACI copy update
            err = float(s > self.q_issued[t])
            j = t % self.h
            self.copies[j] += self.gamma * (self.alpha - err)
        self._on_observe(t, s)

    def predict(self, tau: int) -> float:
        m = len(self.scores)                       # scores of targets 0 .. m-1 are known
        if m != tau - self.h + 1:
            raise RuntimeError(f"target {tau}: {m} scores known, expected {tau - self.h + 1}")
        level = 1 - (self.copies[tau % self.h] if self.copies is not None else self.alpha)
        win = self._window(m)
        q = self._q(win, level) if len(win) else (math.inf if level > 0 else -math.inf)
        self.q_issued[tau] = q
        return q

    # hooks
    def _q(self, window: np.ndarray, p: float) -> float:
        return qhat(window, p)

    def _on_observe(self, t: int, s: float) -> None:
        pass

    def _window(self, m: int) -> np.ndarray:
        raise NotImplementedError


class FixedWindow(Method):
    name = "fixed"

    def __init__(self, alpha, h, W: int, aci_gamma=None):
        super().__init__(alpha, h, aci_gamma)
        self.W = W

    def _window(self, m):
        return self.arr(m - self.W, m)


class ACI(FixedWindow):
    """ACI on the quantile of the last W scores (W = 10**7 means all past, as in the PID repo)."""
    name = "aci"

    def __init__(self, alpha, h, gamma: float, W: int = 10**7):
        super().__init__(alpha, h, W, aci_gamma=gamma)


@dataclass
class _Cusum:
    k: float
    H: float
    mu: float = 0.0
    sd: float = 1.0
    zlo: float = -math.inf
    gp: float = 0.0
    gm: float = 0.0
    zero_p: int = 0
    zero_m: int = 0


class DTCWR(Method):
    """Plan v5 Section 4.1 pseudocode, implemented exactly (see tests/test_calib.py)."""
    name = "dtcwr"

    def __init__(self, alpha, h, W_S=48, W_L=720, H=6.0, k=0.5, c=24, aci_gamma=None,
                 signal: str = "magnitude", window_rule: str = "anchored",
                 prewhiten: bool = False, miss_min: int = 10):
        super().__init__(alpha, h, aci_gamma)
        if signal not in ("magnitude", "miss") or window_rule not in ("anchored", "reset"):
            raise ValueError((signal, window_rule))
        self.needs_predict = signal == "miss"     # the miss signal needs the issued intervals
        self.miss_min = miss_min
        self.scan: list[float] = []               # miss signal: e_t - alpha since the scan start
        self.scan_t0 = 0
        self.prewhiten = prewhiten
        self.phi = 0.0                    # AR(1) coefficient of z, fitted on warm-up (prewhiten only)
        self.z_prev = None
        if W_S > W_L:
            raise ValueError("W_S must be <= W_L")
        self.W_S, self.W_L, self.c = W_S, W_L, c
        self.cs = _Cusum(k=k, H=H)
        self.signal, self.window_rule = signal, window_rule     # for the 2x2 ablation (later)
        self.anchor = -10**12
        self.t_alarm = -10**12
        self.paused = False
        self.post: list[float] = []
        self.sigma0 = 1.0
        self.started_at = 10**12          # set by setup(); detector is off until then
        self.alarms: list[tuple] = []

    def setup(self, warm):
        warm = np.asarray(warm, dtype=float)
        self.sigma0 = float(np.median(np.abs(warm)) / 0.6745) or 1e-12
        r = np.log(warm / self.sigma0 + 1e-6)
        self.cs.mu, self.cs.sd = float(r.mean()), float(r.std() or 1.0)
        z = np.maximum((r - self.cs.mu) / self.cs.sd, np.quantile((r - self.cs.mu) / self.cs.sd, 0.05))
        self.cs.zlo = float(z.min())
        if self.prewhiten and len(z) > 2:
            zc = z - z.mean()
            self.phi = float(np.clip((zc[1:] * zc[:-1]).mean() / (zc * zc).mean(), 0.0, 0.95))
        self.started_at = len(warm)            # detector runs from the first post-warm-up score
        self.scan_t0 = self.started_at         # miss signal: stream index of scan[0]

    def _on_observe(self, t, s):
        if t < self.started_at:
            return
        if self.signal == "miss":
            self._observe_miss(t, s)
            return
        cs = self.cs
        r = math.log(s / self.sigma0 + 1e-6)
        if self.paused:
            self.post.append(r)
            if len(self.post) >= self.W_S:
                cs.mu = float(np.mean(self.post))
                self.paused = False
                cs.gp = cs.gm = 0.0
                cs.zero_p = cs.zero_m = t
                self.z_prev = None
            return
        z = max((r - cs.mu) / cs.sd, cs.zlo)
        if self.prewhiten:                     # AR(1) innovation, rescaled to unit variance
            e = z if self.z_prev is None else (z - self.phi * self.z_prev) / math.sqrt(1 - self.phi ** 2)
            self.z_prev, z = z, e
        cs.gp = max(0.0, cs.gp + z - cs.k)
        if cs.gp == 0.0:
            cs.zero_p = t
        cs.gm = max(0.0, cs.gm - z - cs.k)
        if cs.gm == 0.0:
            cs.zero_m = t
        if cs.gp > cs.H or cs.gm > cs.H:
            if t - self.t_alarm >= self.c:
                trig = "+" if cs.gp >= cs.gm else "-"
                tau_hat = (cs.zero_p if trig == "+" else cs.zero_m) + 1
                self.anchor = tau_hat if self.window_rule == "anchored" else t + 1
                self.t_alarm = t
                self.paused, self.post = True, []
                self.alarms.append((t, trig, tau_hat))
                self.z_prev = None
            cs.gp = cs.gm = 0.0
            cs.zero_p = cs.zero_m = t

    def _observe_miss(self, t, s):
        """2x2 ablation, factor 'signal' = miss indicators (the DriftOCP statistic, Liang et al. Alg. 1):
        max over j of |sum_{l=j..t}(e_l - alpha)| / sqrt(t - j + 1), blocks of length >= miss_min,
        scan restricted to the last W_L steps since the last (re)start. Same pause / cooldown /
        window rule as the magnitude detector, so only the signal differs."""
        if self.paused:
            self.post.append(s)
            if len(self.post) >= self.W_S:
                self.paused, self.scan, self.scan_t0 = False, [], t + 1
            return
        e = float(s > self.q_issued.get(t, math.inf))
        self.scan.append(e - self.alpha)
        if len(self.scan) > self.W_L:
            self.scan.pop(0)
            self.scan_t0 += 1
        z, j = max_block_z(np.asarray(self.scan), self.miss_min)
        if z > self.cs.H and t - self.t_alarm >= self.c:
            tau_hat = self.scan_t0 + j
            self.anchor = tau_hat if self.window_rule == "anchored" else t + 1
            self.t_alarm = t
            self.paused, self.post = True, []
            self.alarms.append((t, "miss", tau_hat))

    def n_post(self, m: int) -> int:
        return max(0, m - self.anchor) if self.anchor <= m else 0

    def _window(self, m):
        n = min(self.W_L, max(self.W_S, self.n_post(m)), m)
        return self.arr(m - n, m)


def max_block_z(d: np.ndarray, min_len: int = 1) -> tuple[float, int]:
    """max_j |sum(d[j:])| / sqrt(len(d) - j) over blocks ending at the last element with
    length >= min_len. Returns (statistic, j). (0, 0) if no block is long enough."""
    n = len(d)
    if n < min_len:
        return 0.0, 0
    tail = np.cumsum(d[::-1])[::-1][: n - min_len + 1]          # sum(d[j:]) for j = 0..n-min_len
    z = np.abs(tail) / np.sqrt(n - np.arange(len(tail)))
    j = int(np.argmax(z))
    return float(z[j]), j


# ----------------------------------------------------------------------------- engine
def run_stream(method: Method, scores: np.ndarray, warm: int = W0) -> Method:
    """Feed a whole score stream (detector studies). Issues the interval for target t + h after
    observing score t when the method needs its own intervals (miss-indicator detectors)."""
    method.setup(np.asarray(scores[:warm], float))
    need = getattr(method, "needs_predict", False)
    for t, v in enumerate(scores):
        method.observe(t, float(v))
        if need:
            method.predict(t + method.h)
    return method


def replay(method: Method, y: np.ndarray, yhat: np.ndarray, warm: int = W0,
           clip_inf: bool = False) -> dict:
    """Run one method over one stream. Returns per-target q and coverage (NaN where not evaluated).
    clip_inf (decision D10): an issued q = +inf is replaced in the OUTPUT by the largest score known
    at the issuing origin, as in the PID repo's "ACI (clipped)". The method's own updates still
    see the interval it issued. n_clipped counts the replaced intervals."""
    s = np.abs(np.asarray(y, float) - np.asarray(yhat, float))
    N, h = len(s), method.h
    tau0 = warm + h
    if N <= tau0:
        raise ValueError("stream shorter than warm-up + h")
    method.setup(s[:warm])
    q = np.full(N, np.nan)
    n_clip = 0
    for tau in range(h, N):
        method.observe(tau - h, s[tau - h])        # the newest score known at origin tau - h
        q[tau] = method.predict(tau)               # intervals before tau0 are warm-up only
        if clip_inf and q[tau] == math.inf:
            q[tau] = float(s[:tau - h + 1].max())
            n_clip += tau >= tau0
    evaluated = np.zeros(N, bool)
    evaluated[tau0:] = True
    covered = np.where(evaluated, s <= q, False)
    return {"q": q, "score": s, "evaluated": evaluated, "covered": covered, "tau0": tau0,
            "n_clipped": int(n_clip)}


# ----------------------------------------------------------------------------- metrics
def interval_metrics(y, yhat, q, mask, alpha: float, w: int) -> dict:
    """Plan v5 Section 8.1. `mask` selects the evaluated targets (contiguous block)."""
    y, yhat, q = np.asarray(y)[mask], np.asarray(yhat)[mask], np.asarray(q)[mask]
    lo, hi = yhat - q, yhat + q
    cov = (y >= lo) & (y <= hi)
    width = np.where(np.isfinite(q), 2 * q, np.nan)
    wink = width + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    roll = np.convolve(cov.astype(float), np.ones(w) / w, mode="valid") if len(cov) >= w else np.array([])
    under = roll < 0.80
    longest = 0
    if under.any():
        d = np.diff(np.concatenate(([0], under.astype(int), [0])))
        longest = int((np.where(d == -1)[0] - np.where(d == 1)[0]).max())
    return {"n": int(len(cov)), "coverage": float(cov.mean()),
            "coverage_error": float(abs(cov.mean() - (1 - alpha))),
            "mean_width": float(np.nanmean(width)), "winkler": float(np.nanmean(wink)),
            "worst_local_coverage": float(roll.min()) if len(roll) else float("nan"),
            "longest_under_run": longest,
            "n_infinite_q": int((~np.isfinite(q)).sum())}
