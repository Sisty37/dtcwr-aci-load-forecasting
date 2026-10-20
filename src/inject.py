"""
Shift injection (plan v5 Section 6.3) and the recovery / detection metrics that use known
shift times (Section 8.2).

This file implements the RESIDUAL-LEVEL injection (the clean control):
forecasts are untouched and the signed residual r = y - yhat is perturbed for t >= tau.
    variance  : r' = kappa * r                 -> score scaled by exactly kappa
    level     : r' = r + delta * sigma_hat
    gradual   : r' = r + ramp(t) * 2 * sigma_hat, ramp 0 -> 1 over one week, then held
sigma_hat = SD of the signed residuals in the warm-up block (first W0 targets) of that stream,
so the injection never uses data from after the warm-up.
Target-level injection (y changed, frozen LightGBM re-run) is a separate step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import calib as cb

SETTINGS = (
    ("variance", 1.5), ("variance", 2.0), ("variance", 3.0),
    ("level", 1.0), ("level", 2.0), ("level", 3.0),
    ("gradual", 2.0),
)
TTR_CAP = 1000


@dataclass(frozen=True)
class Shift:
    kind: str           # variance | level | gradual
    size: float         # kappa, delta, or ramp height in sigma units
    tau: int            # first affected target (stream index)
    week: int           # steps per week (for the gradual ramp)


def inject_residuals(r: np.ndarray, shift: Shift, warm: int = cb.W0) -> np.ndarray:
    r = np.asarray(r, float).copy()
    sig = float(np.std(r[:warm]))
    t = np.arange(len(r))
    after = t >= shift.tau
    if shift.kind == "variance":
        r[after] *= shift.size
    elif shift.kind == "level":
        r[after] += shift.size * sig
    elif shift.kind == "gradual":
        ramp = np.clip((t - shift.tau + 1) / shift.week, 0, 1)
        r[after] += ramp[after] * shift.size * sig
    else:
        raise ValueError(shift.kind)
    return r


def positions(n: int, h: int, follow: int, n_pos: int = 3, lead: int = 150,
              warm: int = cb.W0) -> list[int]:
    """Shift times inside a stream of length n, each with at least `follow` steps after it."""
    first = warm + h + lead
    last = n - follow
    if last < first:
        raise ValueError(f"stream too short: n={n}, need {first + follow}")
    if n_pos == 1:
        return [first]
    return [int(round(first + k * (last - first) / (n_pos - 1))) for k in range(n_pos)]


TEST_FRACS = (0.15, 0.30, 0.45, 0.60, 0.75)


def test_positions(n_val: int, n_test: int, L: int, cap: int = TTR_CAP, fracs=TEST_FRACS) -> list[int]:
    """Plan 6.3: shift times at 15/30/45/60/75% of the TEST segment, as indices of the
    val + test stream. Every position must have cap + L steps of follow-up (asserted)."""
    taus = [n_val + int(round(f * n_test)) for f in fracs]
    follow = n_val + n_test - taus[-1]
    if follow < cap + L:
        raise ValueError(f"last position has {follow} steps of follow-up, needs cap + L = {cap + L}")
    return taus


# ----------------------------------------------------------------------------- metrics
def detection_delay(alarm_times: list[int], tau: int, max_delay: int = TTR_CAP):
    """(delay, detected). An alarm counts if raised at score index >= tau; else censored."""
    later = [a for a in alarm_times if a >= tau]
    if later and later[0] - tau <= max_delay:
        return later[0] - tau, True
    return max_delay, False


def time_to_recover(covered: np.ndarray, tau: int, alpha: float, w: int, delta: float = 0.05,
                    L: int | None = None, cap: int = TTR_CAP):
    """
    Plan v5 Section 8.2. Rolling coverage over the last w targets (window ending at t);
    recovered at the first t >= tau with rolling coverage >= 1 - alpha - delta that stays so for
    L consecutive targets. Returns (ttr, recovered). ttr is counted from tau; censored at `cap`.
    """
    L = w if L is None else L
    c = np.asarray(covered, float)
    roll = np.full(len(c), np.nan)
    cs = np.concatenate(([0.0], np.cumsum(c)))
    roll[w - 1:] = (cs[w:] - cs[:-w]) / w
    ok = roll >= 1 - alpha - delta
    end = min(len(c), tau + cap + L)
    run = 0
    for t in range(tau, end):
        run = run + 1 if ok[t] else 0
        if run >= L:
            start = t - L + 1
            if start - tau <= cap:
                return start - tau, True
            break
    return cap, False


def rmst(ttrs: np.ndarray, recovered: np.ndarray, cap: int = TTR_CAP) -> float:
    """Restricted mean survival time of 'not yet recovered' up to cap (censored runs count as cap)."""
    t = np.minimum(np.asarray(ttrs, float), cap)
    return float(t.mean())
