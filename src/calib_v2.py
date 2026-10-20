"""
DTCW-R v2 (post-test development; confirmed only on a FRESH client cohort, decision D13).

What the v1 test taught us (results/analysis, commit after prereg-v1.1):
  - at h = 1 the miss-indicator scan detected variance shifts sooner than the magnitude CUSUM
    (2x2, Holm p = 0.001), at h = 24 the magnitude CUSUM was sooner;
  - the magnitude H sat on a coarse grid, so its false-alarm rate was below the miss detector's.
v2 therefore
  1. runs BOTH detectors and raises an alarm when either fires (same pause, cooldown and
     anchored window as v1; the anchor comes from the detector that fired);
  2. calibrates each threshold by bisection on a fine grid to half the false-alarm budget
     (0.25 per 1,000 steps each, so the union stays near 0.5);
  3. optionally resets the ACI copies to alpha at an alarm (aci_reset), so the ACI layer does
     not keep correcting for the pre-shift regime.
With H_miss = inf, aci_reset = False, v2 is exactly v1 (tested).
"""
from __future__ import annotations

import math

import numpy as np

import calib as cb


class DTCWR2(cb.DTCWR):
    name = "dtcwr2"

    def __init__(self, alpha, h, W_S=48, W_L=720, H=6.0, k=0.5, c=24, aci_gamma=None,
                 H_miss: float = math.inf, miss_min: int = 10, prewhiten: bool = False,
                 aci_reset: bool = False):
        super().__init__(alpha, h, W_S=W_S, W_L=W_L, H=H, k=k, c=c, aci_gamma=aci_gamma,
                         signal="magnitude", window_rule="anchored", prewhiten=prewhiten,
                         miss_min=miss_min)
        self.H_miss = H_miss
        self.aci_reset = aci_reset
        self.needs_predict = math.isfinite(H_miss)

    def _alarm(self, t, trig, tau_hat):
        self.anchor = tau_hat
        self.t_alarm = t
        self.paused, self.post = True, []
        self.alarms.append((t, trig, tau_hat))
        self.z_prev = None
        if self.aci_reset and self.copies is not None:
            self.copies[:] = self.alpha

    def _on_observe(self, t, s):
        if t < self.started_at:
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
                self._scan.reset()
                self.scan_t0 = t + 1
            return
        # magnitude CUSUM (identical to v1)
        z = max((r - cs.mu) / cs.sd, cs.zlo)
        if self.prewhiten:
            e = z if self.z_prev is None else (z - self.phi * self.z_prev) / math.sqrt(1 - self.phi ** 2)
            self.z_prev, z = z, e
        cs.gp = max(0.0, cs.gp + z - cs.k)
        if cs.gp == 0.0:
            cs.zero_p = t
        cs.gm = max(0.0, cs.gm - z - cs.k)
        if cs.gm == 0.0:
            cs.zero_m = t
        mag_fire = cs.gp > cs.H or cs.gm > cs.H
        # miss-indicator scan
        miss_fire, j = False, 0
        if self.needs_predict:
            e_t = float(s > self.q_issued.get(t, math.inf))
            self._scan.append(e_t - self.alpha)
            zm, j = self._scan.stat()
            miss_fire = zm > self.H_miss
        cool = t - self.t_alarm >= self.c
        if mag_fire:
            if cool:
                trig = "+" if cs.gp >= cs.gm else "-"
                self._alarm(t, trig, (cs.zero_p if trig == "+" else cs.zero_m) + 1)
            cs.gp = cs.gm = 0.0
            cs.zero_p = cs.zero_m = t
        elif miss_fire and cool:
            self._alarm(t, "miss", self.scan_t0 + j)
            cs.gp = cs.gm = 0.0
            cs.zero_p = cs.zero_m = t


# ----------------------------------------------------------------------------- calibration
MAG_GRID = tuple(float(x) for x in np.round(np.geomspace(2.0, 150.0, 121), 3))
MISS_GRID = tuple(round(0.5 + 0.05 * i, 2) for i in range(91))


def calibrate_dual(val_scores, block, base_kw: dict, h: int, alpha: float = 0.1, prewhiten=False,
                   target_each: float = 0.25, reps_per_series: int = 1, length: int = 10_000):
    """H (magnitude only) and H_miss (miss only), each to target_each alarms / 1,000 steps by
    bisection on fine grids; then the union rate of the dual detector on the same streams."""
    import detector_calibration as dc
    import run_prereg_selection as ps
    mag = ps.calibrate_H_bisect(val_scores, block,
                                lambda H: DTCWR2(alpha, h, H=H, prewhiten=prewhiten, **base_kw),
                                grid=MAG_GRID, target=target_each, length=length,
                                reps_per_series=reps_per_series)
    miss = ps.calibrate_H_bisect(val_scores, block,
                                 lambda H: cb.DTCWR(alpha, h, H=H, signal="miss", **base_kw),
                                 grid=MISS_GRID, target=target_each, length=length,
                                 reps_per_series=reps_per_series)
    rng = np.random.default_rng(2027)
    streams = [dc.block_bootstrap(np.asarray(x, float), length, block, rng)
               for x in val_scores for _ in range(reps_per_series)]
    union = float(np.mean([dc.alarm_rate(s, lambda: DTCWR2(alpha, h, H=mag["H"], H_miss=miss["H"],
                                                           prewhiten=prewhiten, **base_kw))
                           for s in streams]))
    return {"H": mag["H"], "fa_mag": mag["fa_at_H"], "H_miss": miss["H"], "fa_miss": miss["fa_at_H"],
            "fa_union": union}
