"""
Step 4 of the ICREST 2027 plan (v5, Sections 6.1, 6.3, 7.1): point forecasters.

Run from the project root:
    python src/forecast.py --dataset all            # electricity + elec2, horizons from HORIZONS
    python src/forecast.py --dataset electricity

Models
- Seasonal naive: y_hat(t+h) = y(t+h-P), P = one day (24 hourly / 48 half-hourly). Sanity baseline.
- LightGBM, DIRECT strategy: one model per (dataset, horizon). Features are computed at the
  forecast ORIGIN t from values y(<= t) only, plus the calendar of the target time t+h.
  Electricity uses ONE global model over the selected clients (client id as a categorical
  feature, each client scaled by its own TRAIN mean). Elec2 is a single series.
- Hyperparameters: 4 configurations compared on an inner validation block (last 10% of the
  TRAIN targets, by time) with early stopping; the winner is refit on all train targets with
  the selected number of trees. Validation and test data are never used for fitting.

Outputs (per dataset and horizon)
    results/forecasts/<dataset>_h<h>.parquet   rows for VAL and TEST targets:
        series, origin, target, split, y, yhat, yhat_snaive, resid (= y - yhat)
    results/models/<dataset>_h<h>.txt          frozen LightGBM booster (reused for shift injection)
    results/models/<dataset>_h<h>.json         feature spec, scaling constants, chosen config
    results/forecasts/forecast_summary.json    gate check and per-series validation MAE

Gate (plan Step 4): LightGBM beats seasonal naive on validation MAE for >= 80% of series.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = {"electricity": (1, 24), "elec2": (1, 48)}     # 1 step and 1 day ahead
PERIOD = {"electricity": 24, "elec2": 48}                 # steps per day
SPLIT = (0.70, 0.10, 0.20)
SEED = 2027
CONFIGS = [
    {"num_leaves": 31, "min_data_in_leaf": 100},
    {"num_leaves": 63, "min_data_in_leaf": 100},
    {"num_leaves": 31, "min_data_in_leaf": 500},
    {"num_leaves": 127, "min_data_in_leaf": 200},
]
BASE_PARAMS = {"objective": "l2", "learning_rate": 0.05, "feature_fraction": 0.9,
               "verbosity": -1, "seed": SEED, "deterministic": True, "num_threads": 4,
               "force_row_wise": True}
MAX_TREES, EARLY_STOP = 2000, 100


# ----------------------------------------------------------------------------- splits
def split_sizes(n: int, frac=SPLIT) -> tuple[int, int, int]:
    n_tr, n_va = int(round(frac[0] * n)), int(round(frac[1] * n))
    return n_tr, n_va, n - n_tr - n_va


def split_label(target_idx: np.ndarray, n: int) -> np.ndarray:
    n_tr, n_va, _ = split_sizes(n)
    return np.where(target_idx < n_tr, "train", np.where(target_idx < n_tr + n_va, "val", "test"))


# ----------------------------------------------------------------------------- features
@dataclass
class FeatureSpec:
    period: int
    h: int
    n_lags: int = field(init=False)

    def __post_init__(self):
        if self.h > self.period:
            raise ValueError("h must be <= one day (same-time-yesterday feature needs t+h-P <= t)")
        self.n_lags = self.period

    @property
    def first_origin(self) -> int:
        return max(self.n_lags - 1, 7 * self.period - self.h)

    def names(self) -> list[str]:
        return ([f"lag_{k}" for k in range(self.n_lags)] +
                ["same_time_yesterday", "same_time_last_week", "mean_last_day",
                 "slot_of_day", "weekday"])


def make_features(y: np.ndarray, ts: pd.DatetimeIndex, spec: FeatureSpec):
    """
    Rows = forecast origins t with a valid target t+h. Every feature uses y[<= t] only
    (the calendar columns describe the TARGET time, which is known in advance).
    Returns (X, target, origin_idx, target_idx).
    """
    n, P, h = len(y), spec.period, spec.h
    origins = np.arange(spec.first_origin, n - h)
    targets = origins + h
    lags = np.stack([y[origins - k] for k in range(spec.n_lags)], axis=1)
    yday = y[targets - P]                       # t+h-P <= t because h <= P
    week = y[targets - 7 * P]                   # t+h-7P <= t
    csum = np.concatenate(([0.0], np.cumsum(y, dtype="float64")))
    mean_day = (csum[origins + 1] - csum[origins + 1 - P]) / P
    tt = ts[targets]
    slot = (tt.hour * (P // 24) + tt.minute // (60 // (P // 24))).to_numpy()
    wday = tt.weekday.to_numpy()
    X = np.column_stack([lags, yday, week, mean_day, slot, wday]).astype("float32")
    return X, y[targets], origins, targets


# ----------------------------------------------------------------------------- data
def load_series(dataset: str, data_dir: Path = Path("data/processed")) -> tuple[pd.DataFrame, list[str]]:
    """Wide frame (index = time, one column per series) and the list of series to model."""
    if dataset == "electricity":
        df = _read(data_dir / "electricity_hourly")
        sel = pd.read_csv(data_dir / "selected_clients.csv")["client"].tolist()
        return df[sel].astype("float64"), sel
    if dataset == "elec2":
        df = _read(data_dir / "elec2_main")[["y"]].rename(columns={"y": "elec2"})
        return df.astype("float64"), ["elec2"]
    raise ValueError(dataset)


def _read(base: Path) -> pd.DataFrame:
    p = base.with_suffix(".parquet")
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(base.with_suffix(".csv.gz"), index_col=0, parse_dates=True)


def build_panel(wide: pd.DataFrame, series: list[str], spec: FeatureSpec):
    """Stack all series. Scaling constant = mean of the series over TRAIN time points only."""
    n = len(wide)
    n_tr, _, _ = split_sizes(n)
    parts, scales = [], {}
    for sid, name in enumerate(series):
        y = wide[name].to_numpy()
        scale = float(np.mean(y[:n_tr]))
        if scale <= 0:
            raise ValueError(f"{name}: non-positive train mean")
        scales[name] = scale
        X, tgt, o_idx, t_idx = make_features(y / scale, wide.index, spec)
        parts.append(dict(X=X, y_scaled=tgt, origin=o_idx, target=t_idx,
                          series=np.full(len(tgt), sid, dtype=np.int32),
                          y=y[t_idx], snaive=y[t_idx - spec.period], scale=scale))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in
           ("X", "y_scaled", "origin", "target", "series", "y", "snaive")}
    cat["scale"] = np.concatenate([np.full(len(p["y"]), p["scale"]) for p in parts])
    cat["split"] = split_label(cat["target"], n)
    cat["X"] = np.column_stack([cat["X"], cat["series"].astype("float32")])
    return cat, scales


# ----------------------------------------------------------------------------- model
def fit_lgbm(X: np.ndarray, y: np.ndarray, t_idx: np.ndarray, feat_names: list[str],
             configs=CONFIGS, max_trees: int = MAX_TREES, early_stop: int = EARLY_STOP):
    """Select a config on the last 10% of TRAIN targets (by time), then refit on all train rows."""
    import lightgbm as lgb
    cut = np.quantile(t_idx, 0.90)
    inner_tr, inner_va = t_idx < cut, t_idx >= cut
    cat_feat = [feat_names.index("series_id")]
    results = []
    for cfg in configs:
        params = {**BASE_PARAMS, **cfg}
        dtr = lgb.Dataset(X[inner_tr], y[inner_tr], feature_name=feat_names,
                          categorical_feature=cat_feat, free_raw_data=False)
        dva = lgb.Dataset(X[inner_va], y[inner_va], reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=max_trees, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stop, verbose=False)])
        pred = b.predict(X[inner_va], num_iteration=b.best_iteration)
        results.append({"config": cfg, "best_iter": int(b.best_iteration),
                        "inner_val_mae": float(np.mean(np.abs(pred - y[inner_va])))})
    best = min(results, key=lambda r: r["inner_val_mae"])
    dall = lgb.Dataset(X, y, feature_name=feat_names, categorical_feature=cat_feat)
    booster = lgb.train({**BASE_PARAMS, **best["config"]}, dall,
                        num_boost_round=max(1, best["best_iter"]))
    return booster, best, results


# ----------------------------------------------------------------------------- run
def run_one(dataset: str, h: int, data_dir: Path, out_dir: Path, model_dir: Path,
            configs=CONFIGS, max_trees: int = MAX_TREES, early_stop: int = EARLY_STOP) -> dict:
    t0 = time.time()
    wide, series = load_series(dataset, data_dir)
    spec = FeatureSpec(PERIOD[dataset], h)
    panel, scales = build_panel(wide, series, spec)
    feat_names = spec.names() + ["series_id"]

    tr = panel["split"] == "train"
    booster, best, cv = fit_lgbm(panel["X"][tr], panel["y_scaled"][tr], panel["target"][tr],
                                 feat_names, configs, max_trees, early_stop)

    keep = ~tr
    yhat = booster.predict(panel["X"][keep]) * panel["scale"][keep]
    idx = wide.index
    out = pd.DataFrame({
        "series": np.array(series, dtype=object)[panel["series"][keep]],
        "origin": idx[panel["origin"][keep]],
        "target": idx[panel["target"][keep]],
        "split": panel["split"][keep],
        "y": panel["y"][keep],
        "yhat": yhat,
        "yhat_snaive": panel["snaive"][keep],
    })
    out["resid"] = out["y"] - out["yhat"]
    out = out.sort_values(["series", "target"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{dataset}_h{h}.parquet"
    try:
        out.to_parquet(fpath)
    except ImportError:
        fpath = fpath.with_suffix(".csv.gz")
        out.to_csv(fpath, index=False)
    booster.save_model(str(model_dir / f"{dataset}_h{h}.txt"))
    (model_dir / f"{dataset}_h{h}.json").write_text(json.dumps({
        "dataset": dataset, "h": h, "period": spec.period, "features": feat_names,
        "series": series, "scales": scales, "selected": best, "cv": cv,
        "split_sizes": split_sizes(len(wide))}, indent=2))

    va = out[out["split"] == "val"]
    mae = va.groupby("series").apply(
        lambda g: pd.Series({"mae_lgbm": np.mean(np.abs(g["y"] - g["yhat"])),
                             "mae_snaive": np.mean(np.abs(g["y"] - g["yhat_snaive"]))}),
        include_groups=False)
    share = float((mae["mae_lgbm"] < mae["mae_snaive"]).mean())
    return {"dataset": dataset, "h": h, "file": str(fpath), "n_rows": int(len(out)),
            "n_val": int((out["split"] == "val").sum()), "n_test": int((out["split"] == "test").sum()),
            "selected_config": best, "share_series_lgbm_better_val": round(share, 3),
            "gate_pass": share >= 0.80,
            "median_val_mae_ratio_lgbm_over_snaive": round(float((mae["mae_lgbm"] / mae["mae_snaive"]).median()), 3),
            "seconds": round(time.time() - t0, 1),
            "per_series_val_mae": mae.round(6).reset_index().to_dict(orient="records")}


def main(datasets, data_dir=Path("data/processed"), out_dir=Path("results/forecasts"),
         model_dir=Path("results/models"), **kw):
    summary = []
    for ds in datasets:
        for h in HORIZONS[ds]:
            r = run_one(ds, h, data_dir, out_dir, model_dir, **kw)
            print(f"{ds:11s} h={h:<3d} val={r['n_val']:>7d} test={r['n_test']:>7d}  "
                  f"LGBM better than snaive on {100 * r['share_series_lgbm_better_val']:.0f}% of series "
                  f"(median MAE ratio {r['median_val_mae_ratio_lgbm_over_snaive']}) "
                  f"gate={'PASS' if r['gate_pass'] else 'FAIL'}  [{r['seconds']} s]")
            summary.append(r)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "forecast_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "electricity", "elec2"])
    a = ap.parse_args()
    main(["electricity", "elec2"] if a.dataset == "all" else [a.dataset])
