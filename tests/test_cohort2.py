"""Tests for the cohort-2 switch and draw (decision D14).
Run:  python -m pytest tests -q
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import make_cohort2 as mc  # noqa: E402


def test_draw_is_disjoint_deterministic_and_sized():
    elig = [f"MT_{i:03d}" for i in range(314)]
    c1 = elig[::6][:50]
    a, b = mc.draw(elig, c1), mc.draw(elig, c1)
    assert a == b and len(a) == 50 and not set(a) & set(c1) and a == sorted(a)


def test_cohort_defaults_reproduce_cohort1():
    import os
    old = {k: os.environ.pop(k, None) for k in ("ICREST_DATASETS", "ICREST_PREREG_TAG")}
    try:
        import cohort
        importlib.reload(cohort)
        assert cohort.DATASETS == ("elec2", "electricity") and cohort.PREREG_TAG == "prereg-v1"
        os.environ["ICREST_DATASETS"], os.environ["ICREST_PREREG_TAG"] = "electricity", "prereg-c2"
        importlib.reload(cohort)
        assert cohort.DATASETS == ("electricity",) and cohort.PREREG_TAG == "prereg-c2"
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        importlib.reload(cohort)


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("PASS", k)
