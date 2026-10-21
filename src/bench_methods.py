"""
Runtime and peak memory per method, one consistent benchmark (pre-submission review).
Every method of the test run (frozen cohort-1 configurations, plus SAOCP if selected_extra.json
exists) replays the same untouched validation+test stream of the first 10 clients, one process,
no other load. Reports median ms per step (untraced run) and median peak traced memory (tracemalloc,
KiB, second run) of the replay, including the method's score buffer.

    python src/bench_methods.py
Output: results/review/bench_methods.csv
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib as cb  # noqa: E402
import run_test as rt  # noqa: E402
import tune as tn  # noqa: E402
from run_calib_val import load_forecasts  # noqa: E402

DS, N_CLIENTS = "electricity", 10


def main():
    fz = rt.load_frozen()
    extra = Path("results/tuning/selected_extra.json")
    rows = []
    for h in (1, 24):
        makers = {k: v for k, v in rt.method_rows(DS, h, fz).items() if not k.startswith("abl_")}
        if extra.exists():
            cfg = json.loads(extra.read_text())[f"{DS}_h{h}"]["saocp"]["config"]
            makers["saocp"] = lambda cfg=cfg, h=h: tn.make_method("saocp", cfg, h)
        df = load_forecasts(DS, h)
        df = df[df.split.isin(["val", "test"])]
        for name in sorted(df.series.unique())[:N_CLIENTS]:
            g = df[df.series == name]
            y, yhat = g.y.to_numpy(float), g.yhat.to_numpy(float)
            for m, make in makers.items():
                t0 = time.perf_counter()                       # timing without tracing
                cb.replay(make(), y, yhat, clip_inf=True)
                dt = time.perf_counter() - t0
                tracemalloc.start()                             # memory in a second, traced run
                cb.replay(make(), y, yhat, clip_inf=True)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rows.append({"h": h, "series": name, "method": m, "ms_per_step": 1000 * dt / len(y),
                             "peak_kib": peak / 1024})
    out = pd.DataFrame(rows)
    Path("results/review").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/review/bench_methods.csv", index=False)
    s = out.groupby(["h", "method"]).agg(ms_per_step=("ms_per_step", "median"),
                                         peak_kib=("peak_kib", "median")).round(3)
    print(s.to_string())


if __name__ == "__main__":
    main()
