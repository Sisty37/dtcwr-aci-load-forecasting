"""Tests for src/data_elec2.py on a synthetic file in the exact Elec2 CSV format.
Run:  python -m pytest tests -q        (or:  python tests/test_data_elec2.py)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_elec2 as de2  # noqa: E402

N_DAYS = 60
N = N_DAYS * 48
TS0, TS1 = "1996-05-07 00:00:00", str(pd.Timestamp("1996-05-07") + pd.Timedelta(days=N_DAYS) - pd.Timedelta(minutes=30))


def make_repo(root: Path) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    k = np.arange(N)
    y = 0.4 + 0.15 * np.sin(2 * np.pi * (k % 48) / 48) + rng.normal(0, 0.01, N)
    df = pd.DataFrame({"date": np.repeat(np.linspace(0, 0.05, N_DAYS), 48),
                       "day": np.repeat((np.arange(N_DAYS) + 1) % 7 + 1, 48),
                       "period": np.tile(np.arange(48) / 47, N_DAYS),
                       "nswprice": 0.05, "nswdemand": np.clip(y, 0, 1),
                       "vicprice": 0.003, "vicdemand": 0.42, "transfer": 0.41,
                       "class": np.where(rng.random(N) > 0.5, "UP", "DOWN")})
    (root / "proc" / "elec2").mkdir(parents=True)
    df.to_csv(root / "elec2.csv", index=False)
    fc = np.r_[np.zeros(10), df["nswdemand"].to_numpy()[9:de2.PUB_ROWS - 1]]   # lag-1 "AR" forecast
    np.savez(root / "proc" / "elec2" / "ar.npz", forecasts=fc)
    return df


TMP = Path(tempfile.mkdtemp())
RAW = make_repo(TMP)
SUMMARY = de2.run(TMP, TMP / "out", expected_rows=N, ts_start=TS0, ts_end=TS1)


def _read(p):
    p = Path(p)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0, parse_dates=True)


def test_rows_and_timestamps():
    main = _read(SUMMARY["P_main"]["file"])
    assert len(main) == N and main.index[0] == pd.Timestamp(TS0)
    assert (main.groupby(main.index.normalize()).size() == 48).all()


def test_target_is_nswdemand():
    main = _read(SUMMARY["P_main"]["file"])
    assert np.allclose(main["y"].to_numpy(), RAW["nswdemand"].to_numpy())


def test_main_split_chronological():
    main = _read(SUMMARY["P_main"]["file"])
    lab = main["split"].to_numpy()
    order = {"train": 0, "val": 1, "test": 2}
    assert all(order[a] <= order[b] for a, b in zip(lab[:-1], lab[1:]))
    assert abs((lab == "test").mean() - 0.20) < 0.01


def test_pub_window_and_forecasts():
    pub = _read(SUMMARY["P_pub"]["file"])
    assert len(pub) == de2.PUB_ROWS
    assert (pub["phase"] == "eval").sum() == de2.PUB_ROWS - de2.PUB_BURNIN
    assert "fc_ar" in pub.columns
    assert SUMMARY["P_pub"]["forecasts"]["theta"] == "missing"


def test_period_mismatch_detected():
    bad = TMP / "bad"
    df = make_repo(bad)
    df["period"] = np.roll(df["period"].to_numpy(), 1)
    df.to_csv(bad / "elec2.csv", index=False)
    try:
        de2.load_elec2(bad / "elec2.csv", N, TS0, TS1)
    except ValueError as e:
        assert "period" in str(e)
    else:
        raise AssertionError("should detect misaligned timestamps")


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} tests passed")
