"""
Target-level shift injection (plan v5 Section 6.3, PRIMARY injection layer).

The modified y is treated as the observed world: y'_t = y_t + shift_t for t >= tau, later lag
features use the injected values, and the forecasts are recomputed with the FROZEN LightGBM
model saved by src/forecast.py (no refit, no recursion). A level shift is therefore partly
absorbed by the lags, as it would be in reality.

    level     y'_t = y_t + delta * sigma_hat                   delta in {1, 2, 3}
    variance  y'_t = y_t + sigma_eps * eps_t, eps_t ~ N(0, 1)  sigma_eps calibrated on VALIDATION
                                                               so that the achieved residual-
                                                               variance ratio is kappa^2 (D12)
    gradual   y'_t = y_t + ramp_t * 2 * sigma_hat, ramp 0 -> 1 over one week, then held

sigma_hat = SD of the signed residuals of the stream's warm-up block (first W0 targets), the
same constant the residual-level control uses. eps uses a fixed seed per (dataset, h, client),
identical across kappa values and positions' calibration steps (common random numbers).

Stream convention (as everywhere): validation stream = VAL targets; test stream = VAL targets
followed by TEST targets (the methods warm up on validation). Stream index i <-> absolute
target index targets[i].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import calib as cb
import forecast as fc
import inject as ij

SEED = 2027
SIGMA_TOL = 0.02            # bisection tolerance: |achieved ratio / kappa^2 - 1| <= 2%


class TargetInjector:
    def __init__(self, dataset: str, h: int, model_dir=Path("results/models"),
                 data_dir=Path("data/processed"), booster=None, wide=None, meta=None):
        self.ds, self.h = dataset, h
        self.meta = meta or json.loads((Path(model_dir) / f"{dataset}_h{h}.json").read_text())
        if wide is None:
            wide, _ = fc.load_series(dataset, Path(data_dir))
        self.wide = wide
        self.series = list(self.meta["series"])
        self.scales = self.meta["scales"]
        self.spec = fc.FeatureSpec(int(self.meta["period"]), h)
        if booster is None:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(Path(model_dir) / f"{dataset}_h{h}.txt"))
        self.booster = booster
        self.n = len(wide)
        self.split = fc.split_label(np.arange(self.n), self.n)

    # ------------------------------------------------------------------ streams
    def stream_targets(self, test: bool = False) -> np.ndarray:
        """Absolute target indices of the validation stream, or of the val + test stream."""
        first = self.spec.first_origin + self.h
        keep = (self.split == "val") | ((self.split == "test") if test else False)
        t = np.arange(self.n)
        return t[keep & (t >= first)]

    def y(self, name: str) -> np.ndarray:
        return self.wide[name].to_numpy(dtype="float64")

    # ------------------------------------------------------------------ forecasting
    def forecast(self, name: str, y_full: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Frozen-model forecasts for the given absolute targets from the (possibly modified)
        full series y_full. Features use y_full[<= origin] only (forecast.make_features)."""
        scale = float(self.scales[name])
        X, _, origins, tgt = fc.make_features(y_full / scale, self.wide.index, self.spec)
        rows = targets - self.h - self.spec.first_origin
        assert np.array_equal(tgt[rows], targets)
        Xs = np.column_stack([X[rows], np.full(len(rows), self.series.index(name), dtype="float32")])
        try:
            p = self.booster.predict(Xs, num_threads=1)
        except TypeError:                       # test doubles without num_threads
            p = self.booster.predict(Xs)
        return np.asarray(p, float) * scale

    # ------------------------------------------------------------------ injection
    def eps(self, name: str) -> np.ndarray:
        ds_id = {"electricity": 0, "elec2": 1}.get(self.ds, 9)
        rng = np.random.default_rng([SEED, ds_id, self.h, self.series.index(name)])
        return rng.standard_normal(self.n)

    def shift_vector(self, name: str, kind: str, size: float, tau_abs: int, sigma_hat: float,
                     week: int, sigma_eps: float | None = None) -> np.ndarray:
        d = np.zeros(self.n)
        t = np.arange(self.n)
        after = t >= tau_abs
        if kind == "level":
            d[after] = size * sigma_hat
        elif kind == "gradual":
            d[after] = np.clip((t[after] - tau_abs + 1) / week, 0, 1) * size * sigma_hat
        elif kind == "variance":
            if sigma_eps is None:
                raise ValueError("variance injection needs the calibrated sigma_eps")
            d[after] = sigma_eps * self.eps(name)[after]
        else:
            raise ValueError(kind)
        return d

    def inject(self, name: str, kind: str, size: float, tau: int, week: int, test: bool = False,
               sigma_eps: float | None = None, base: tuple | None = None):
        """Returns (targets, y_stream_mod, yhat_stream_mod, sigma_hat). tau is a STREAM index.
        base = (y_stream, yhat_stream) of the un-injected stream, to get sigma_hat."""
        targets = self.stream_targets(test)
        y_full = self.y(name)
        if base is None:
            base = (y_full[targets], self.forecast(name, y_full, targets))
        sigma_hat = float(np.std((base[0] - base[1])[:cb.W0]))
        d = self.shift_vector(name, kind, size, int(targets[tau]), sigma_hat, week, sigma_eps)
        y_mod = y_full + d
        affected = targets[tau:]
        yhat = base[1].copy()
        yhat[tau:] = self.forecast(name, y_mod, affected)       # earlier targets cannot change
        return targets, y_mod[targets], yhat, sigma_hat


# ----------------------------------------------------------------------------- sigma_eps
def variance_ratio(inj: TargetInjector, name: str, m: float, taus, week: int, base, follow: int) -> float:
    """Mean over positions of Var(r' over [tau, tau+follow)) / Var(r over the same targets),
    r = un-injected residual. sigma_eps = m * sigma_hat."""
    r0 = base[0] - base[1]
    sig = float(np.std(r0[:cb.W0]))
    out = []
    for tau in taus:
        _, y2, yh2, _ = inj.inject(name, "variance", 0.0, tau, week, sigma_eps=m * sig, base=base)
        seg = slice(tau, tau + follow)
        out.append(np.var((y2 - yh2)[seg]) / np.var(r0[seg]))
    return float(np.mean(out))


def calibrate_sigma_eps(inj: TargetInjector, name: str, kappa: float, week: int,
                        follow: int = ij.TTR_CAP, tol: float = SIGMA_TOL, max_iter: int = 40) -> dict:
    """Bisection on m (sigma_eps = m * sigma_hat) until |ratio / kappa^2 - 1| <= tol, on the
    VALIDATION stream at the 3 validation positions."""
    targets = inj.stream_targets(False)
    y_full = inj.y(name)
    base = (y_full[targets], inj.forecast(name, y_full, targets))
    taus = ij.positions(len(targets), inj.h, follow=follow)
    goal = kappa ** 2
    f = lambda m: variance_ratio(inj, name, m, taus, week, base, follow)   # noqa: E731
    lo, hi = 0.0, 1.0
    r_hi = f(hi)
    while r_hi < goal and hi < 64:
        lo, hi = hi, 2 * hi
        r_hi = f(hi)
    m, r = hi, r_hi
    for _ in range(max_iter):
        if abs(r / goal - 1) <= tol:
            break
        m = 0.5 * (lo + hi)
        r = f(m)
        if r < goal:
            lo = m
        else:
            hi = m
    return {"m": m, "sigma_hat": float(np.std((base[0] - base[1])[:cb.W0])),
            "achieved_ratio": r, "kappa2": goal, "within_tol": abs(r / goal - 1) <= tol}


# ----------------------------------------------------------------------------- achieved size
def achieved(kind: str, size: float, r0: np.ndarray, r1: np.ndarray, tau: int, sigma_hat: float,
             week: int, follow: int = ij.TTR_CAP) -> dict:
    """Plan 6.3 magnitude controls, computed on the stream (works for val or test)."""
    seg = slice(tau, tau + follow)
    if kind == "variance":
        return {"achieved_ratio": float(np.var(r1[seg]) / np.var(r0[seg]))}
    if kind == "level":
        return {"achieved_mean_shift_sd": float(np.mean(r1[seg] - r0[seg]) / sigma_hat)}
    end = slice(tau + week - 1, tau + week - 1 + min(week, follow))
    return {"achieved_ramp_end_sd": float(np.mean(r1[end] - r0[end]) / sigma_hat)}
