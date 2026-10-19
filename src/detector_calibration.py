"""
Calibrate the CUSUM threshold H to a target false-alarm rate on autocorrelated residuals
(plan v5 Section 4.2: detectors are compared at MATCHED false-alarm rate).

Why: the first validation look showed many more alarms on real data than on i.i.d. data
(about 6 per 1,000 steps for Electricity h=24 and 11.5 for Elec2 h=48, vs 0.3 on i.i.d.),
because h-step residuals are strongly autocorrelated and CUSUM assumes independence.

How: circular block bootstrap of each series' VALIDATION scores. Blocks of length
B = max(one day, 2h) keep the short-range autocorrelation but scramble long-range order, so
genuine level shifts are mostly broken up. H is the smallest grid value whose pooled alarm
rate on the bootstrap streams is <= TARGET_FA (default 0.5 per 1,000 steps = 1 per 2,000).
Uses validation data only; the test segment stays sealed.
"""
from __future__ import annotations

import numpy as np

import calib as cb

TARGET_FA = 0.5                 # alarms per 1,000 steps
H_GRID = (3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 60)


def block_bootstrap(x: np.ndarray, length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    starts = rng.integers(0, n, size=int(np.ceil(length / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return x[idx.ravel()[:length]]


def alarm_rate(stream: np.ndarray, make, warm: int = cb.W0) -> float:
    """Alarms per 1,000 post-warm-up steps for a detector built by make()."""
    m = make()
    m.setup(stream[:warm])
    for t, v in enumerate(stream):
        m.observe(t, float(v))
    return 1000.0 * len(m.alarms) / max(1, len(stream) - warm)


def calibrate_H(val_scores: list[np.ndarray], block: int, make_with_H, target: float = TARGET_FA,
                grid=H_GRID, length: int = 10_000, reps_per_series: int = 1, seed: int = 2027) -> dict:
    """
    val_scores : one array of validation scores per series
    make_with_H: function H -> fresh detector-bearing method (e.g. lambda H: cb.DTCWR(..., H=H))
    Returns {"H": chosen, "fa_at_H": rate, "curve": {H: rate}} ; walks the grid upward and stops
    at the first H meeting the target (the rate is non-increasing in H up to Monte Carlo noise).
    """
    rng = np.random.default_rng(seed)
    streams = [block_bootstrap(np.asarray(x, float), length, block, rng)
               for x in val_scores for _ in range(reps_per_series)]
    curve = {}
    for H in grid:
        rate = float(np.mean([alarm_rate(s, lambda: make_with_H(H)) for s in streams]))
        curve[H] = rate
        if rate <= target:
            return {"H": H, "fa_at_H": rate, "curve": curve, "n_streams": len(streams)}
    return {"H": grid[-1], "fa_at_H": curve[grid[-1]], "curve": curve, "n_streams": len(streams),
            "warning": "target not reached on grid"}
