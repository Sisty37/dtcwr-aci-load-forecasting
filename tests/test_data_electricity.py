"""Tests for src/data_electricity.py on a small synthetic file in the exact UCI format.

Run:  python -m pytest tests -q        (or:  python tests/test_data_electricity.py)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_electricity as de  # noqa: E402

N_CLIENTS = 8


def make_raw(path: Path) -> pd.DataFrame:
    """Write a synthetic LD2011_2014.txt and return the TRUE 15-min kW (start-indexed, period only)."""
    end_labels = pd.date_range("2011-01-01 00:15", "2015-01-01 00:00", freq="15min")
    start = end_labels - pd.Timedelta(minutes=15)
    hours = start.hour + start.minute / 60
    rng = np.random.default_rng(0)
    true = {}
    for i in range(N_CLIENTS):
        base = 50 + 10 * i
        x = base + 20 * np.sin(2 * np.pi * (hours - 6) / 24) + rng.normal(0, 1, len(start))
        true[f"MT_{i+1:03d}"] = np.maximum(x, 1.0)
    true = pd.DataFrame(true, index=start)
    raw = true.copy()
    raw.loc[raw.index < "2012-06-01", "MT_002"] = 0.0                       # created later -> excluded
    raw.loc["2013-05-10 00:00":"2013-05-11 05:45", "MT_003"] = 0.0          # 30 h outage  -> excluded
    raw.loc["2013-05-10 00:00":"2013-05-10 19:45", "MT_004"] = 0.0          # 20 h outage  -> kept
    # As in the real file: block = interval starts 00:45..01:30 (straddles the 01:00 boundary).
    # March: ~2/3 of clients read 0, the rest read ~3% of normal. October: same rows ~2x.
    cols = list(raw.columns)
    for d in de.MARCH_DST:
        blk = slice(d + pd.Timedelta(minutes=45), d + pd.Timedelta(hours=1, minutes=30))
        raw.loc[blk, cols[:6]] = 0.0
        raw.loc[blk, cols[6:]] = raw.loc[blk, cols[6:]] * 0.03
    for d in de.OCTOBER_DST:
        blk = slice(d + pd.Timedelta(minutes=45), d + pd.Timedelta(hours=1, minutes=30))
        raw.loc[blk] = raw.loc[blk] * 2
    out = raw.copy()
    out.index = end_labels.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, sep=";", decimal=",", float_format="%.6f")
    return true.loc[(true.index >= de.PERIOD_START) & (true.index < de.PERIOD_END)]


def _run():
    tmp = Path(tempfile.mkdtemp())
    raw_path = tmp / "LD2011_2014.txt"
    true15 = make_raw(raw_path)
    summary = de.run(raw_path, tmp / "processed", tmp / "figs", make_plots=False, n_select=3)
    hourly_file = Path(summary["outputs"]["hourly"])
    hourly = (pd.read_parquet(hourly_file) if hourly_file.suffix == ".parquet"
              else pd.read_csv(hourly_file, index_col=0, parse_dates=True))
    return tmp, true15, summary, hourly


TMP, TRUE15, SUMMARY, HOURLY = _run()


def test_end_label_converted_to_start():
    raw = de.load_raw(TMP / "LD2011_2014.txt")
    assert raw.index[0] == pd.Timestamp("2011-01-01 00:00")


def test_unit_conversion_hand_computed():
    idx = pd.date_range("2012-01-01", periods=96, freq="15min")
    df = pd.DataFrame({"A": np.tile([1.0, 2.0, 3.0, 4.0], 24)}, index=idx)
    h = de.to_hourly_kwh(df)
    assert np.allclose(h["A"].values, 2.5)          # (1+2+3+4)/4 kWh in every hour


def test_period_and_24_per_day():
    assert HOURLY.index[0] == pd.Timestamp("2012-01-01 00:00")
    assert HOURLY.index[-1] == pd.Timestamp("2014-12-31 23:00")
    assert (HOURLY.groupby(HOURLY.index.normalize()).size() == 24).all()
    assert SUMMARY["n_hours"] == 26304              # 1096 days x 24


def test_march_block_interpolated():
    true_h = de.to_hourly_kwh(TRUE15)
    for d in de.MARCH_DST:
        for h in (d, d + pd.Timedelta(hours=1)):           # block touches hours 00 and 01
            for c in ("MT_001", "MT_008"):                  # a zero client and a tiny-value client
                got, exp = HOURLY.loc[h, c], true_h.loc[h, c]
                assert abs(got - exp) / exp < 0.05, (d, h, c, got, exp)


def test_october_hour_halved():
    true_h = de.to_hourly_kwh(TRUE15)
    for d in de.OCTOBER_DST:
        for h in (d, d + pd.Timedelta(hours=1)):
            assert abs(HOURLY.loc[h, "MT_001"] - true_h.loc[h, "MT_001"]) < 1e-3
    for o in SUMMARY["dst"]["october"]:
        assert 1.6 < o["median_ratio_before"] < 2.4 and 0.8 < o["median_ratio_after"] < 1.2


def test_zero_run_filter():
    excluded = pd.read_csv(SUMMARY["outputs"]["excluded"])
    assert set(excluded["client"]) == {"MT_002", "MT_003"}
    assert "MT_004" in HOURLY.columns                # 20 h outage is allowed


def test_october_refuses_if_not_double():
    raw = de.restrict_period(de.load_raw(TMP / "LD2011_2014.txt"))
    d = de.OCTOBER_DST[0]
    blk = pd.date_range(d + pd.Timedelta(minutes=45), periods=4, freq="15min")
    raw.loc[blk] = raw.loc[blk] / 2                       # remove the doubling
    try:
        de.fix_dst(raw)
    except ValueError as e:
        assert "not ~2x" in str(e)
    else:
        raise AssertionError("should refuse to halve a normal hour")


def test_selection_deterministic():
    hourly = HOURLY
    _, s1, _ = de.select_clients(hourly, n=3)
    _, s2, _ = de.select_clients(hourly, n=3)
    assert list(s1["client"]) == list(s2["client"])
    assert not set(s1["client"]) & {"MT_002", "MT_003"}


def test_summary_written():
    s = json.loads((TMP / "processed" / "preprocessing_summary.json").read_text())
    assert s["dst"]["dst_clock_offset_from_midnight"] == "0 days 00:45:00"
    assert len(s["sha256"]["selected_clients.csv"]) == 64


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} tests passed")
