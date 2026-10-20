"""
Paper figures (IEEE two-column: 3.5 in single column, 7.16 in double column), PDF + PNG.

Fig. 1  Coverage-width-recovery trade-off: median relative Winkler on untouched test streams (x)
        vs median RMST of TTR after injected shifts (y), h = 1 and h = 24, both cohorts.
Fig. 2  Rolling coverage (one week) around one injected variance shift (kappa = 2, at 45% of the
        test segment), h = 1 and h = 24, for one REPRESENTATIVE cohort-1 client: the client whose
        DTCW-R+ACI RMST at h = 24 is the median of the 50 (chosen by rule, not by look).

Run from the project root (main working copy, after src/combine_cohorts.py):
    python src/figures.py
Output: results/figures/fig1_tradeoff.(pdf|png), results/figures/fig2_rolling_coverage.(pdf|png)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib as cb  # noqa: E402
import inject as ij  # noqa: E402

OUT = Path("results/figures")
INK, INK2, GRID, SURF = "#1f1e1b", "#6b6a63", "#e6e5e0", "#ffffff"
# categorical slots in fixed order (validated: scripts/validate_palette.js, light surface)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# label placement (points, alignment) chosen so labels do not collide; default right-above
LABEL_POS = {
    (1, "ACI"): (-6, 2, "right"), (1, "ECI-cutoff"): (-6, -3, "right"),
    (1, "Conformal PID"): (-6, -6, "right"), (1, "DTCW-R+ACI"): (8, -4, "left"),
    (1, "Fixed 168"): (36, -10, "left", True), (1, "DTCW-R"): (-2, 8, "center"),
    (24, "DtACI"): (12, -3, "left"), (24, "ROCP"): (-7, 0, "right"),
    (24, "Fixed 168"): (10, -6, "left"), (24, "DTCW-R"): (10, 2, "left"),
    (24, "ECI-cutoff"): (0, 7, "center"), (24, "Conformal PID"): (0, -12, "right"),
}
GROUP = {"DTCW-R+ACI": BLUE, "Conformal PID": ORANGE, "ECI-cutoff": ORANGE, "ACI": AQUA, "DtACI": AQUA}

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 1
def fig1(tb: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)
    for ax, h in zip(axes, (1, 24)):
        t = tb[tb.h == h]
        for _, r in t.iterrows():
            m = r["method"]
            col = GROUP.get(m, INK2)
            main = m == "DTCW-R+ACI"
            x1, y1, x2, y2 = r.rel_winkler_L2_c1, r.rmst_L1_c1, r.rel_winkler_L2_c2, r.rmst_L1_c2
            ax.plot([x1, x2], [y1, y2], color=col, lw=1.0, alpha=0.6, zorder=2)
            ax.scatter([x1], [y1], s=34 if main else 22, color=col, edgecolor=SURF, linewidth=1.0,
                       zorder=3, marker="o")
            ax.scatter([x2], [y2], s=34 if main else 22, facecolor=SURF, edgecolor=col, linewidth=1.2,
                       zorder=3, marker="s")
            dx, dy, ha, *lead = LABEL_POS.get((h, m), (5, 3, "left"))
            arrow = dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0, shrinkB=3) if lead else None
            ax.annotate(m, (x1, y1), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, ha=ha,
                        va="center" if lead else "baseline", arrowprops=arrow,
                        color=INK if col != INK2 else INK2, fontweight="bold" if main else "normal")
        ax.set_yscale("log")
        ax.set_yticks([100, 200, 300, 500])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Relative Winkler score, untouched test (lower is better)")
        ax.set_title(f"h = {h}" + (" (one step ahead)" if h == 1 else " (day ahead)"), color=INK)
    axes[0].set_ylabel("RMST of time-to-recover (steps, lower is better)")
    handles = [plt.Line2D([], [], marker="o", ms=4.5, ls="", color=INK2, label="Cohort 1"),
               plt.Line2D([], [], marker="s", ms=4.5, ls="", mfc=SURF, mec=INK2, label="Cohort 2 (replication)")]
    axes[1].legend(handles=handles, loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, "fig1_tradeoff")


# ----------------------------------------------------------------------------- Fig. 2
def representative_client(t24: pd.DataFrame) -> str:
    x = t24[(t24.layer == "L1_target") & (t24.method == "dtcwr_aci")]
    r = x.groupby("series")["ttr"].apply(lambda v: np.minimum(v, ij.TTR_CAP).mean()).sort_values()
    return r.index[len(r) // 2]


def rolling_runs(name: str, h: int, methods=("dtcwr_aci", "aci", "pid", "eci_cutoff")) -> tuple:
    import inject_target as it
    import run_test as rt
    from run_calib_val import LOCAL_W, load_forecasts
    fz = rt.load_frozen()
    df = load_forecasts("electricity", h)
    g = df[(df.series == name) & df.split.isin(["val", "test"])]
    y, yhat = g.y.to_numpy(float), g.yhat.to_numpy(float)
    n_val = int((g.split == "val").sum())
    cell = rt.frozen_cell("electricity", fz)
    tau = ij.test_positions(n_val, len(y) - n_val, L=cell["L"])[2]
    inj = it.TargetInjector("electricity", h)
    sig = float(np.std((y - yhat)[:cb.W0]))
    se = fz["sig"][f"electricity_h{h}"][name]["2.0"]["m"] * sig
    _, y2, yh2, _ = inj.inject(name, "variance", 2.0, tau, LOCAL_W["electricity"], test=True,
                               sigma_eps=se, base=(y, yhat))
    rows = rt.method_rows("electricity", h, fz)
    w = LOCAL_W["electricity"]
    curves = {}
    for m in methods:
        out = cb.replay(rows[m](), y2, yh2, clip_inf=True)
        cov = (out["score"] <= out["q"]).astype(float)
        roll = pd.Series(cov).rolling(w).mean().to_numpy()
        curves[m] = roll
    return curves, tau, cell


def fig2(name: str, data: dict):
    labels = {"dtcwr_aci": "DTCW-R+ACI", "aci": "ACI", "pid": "Conformal PID", "eci_cutoff": "ECI-cutoff"}
    colors = {"dtcwr_aci": BLUE, "aci": ORANGE, "pid": AQUA, "eci_cutoff": YELLOW}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
    for ax, h in zip(axes, (1, 24)):
        curves, tau, cell = data[h]
        lo, hi = tau - 200, tau + ij.TTR_CAP
        x = np.arange(lo, hi) - tau
        for m, c in curves.items():
            ax.plot(x, c[lo:hi], color=colors[m], lw=1.6 if m == "dtcwr_aci" else 1.1,
                    label=labels[m], zorder=3 if m == "dtcwr_aci" else 2)
        ax.axhline(0.9, color=INK2, lw=0.8, ls="--")
        ax.axhline(1 - 0.1 - cell["delta"], color=INK2, lw=0.6, ls=":")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.text(3, 0.905, "target 0.90", fontsize=6.5, color=INK2, va="bottom")
        ax.set_xlabel("Steps after the shift")
        ax.set_title(f"h = {h}", color=INK)
    axes[0].set_ylabel("Rolling coverage (one week)")
    axes[0].legend(loc="lower right", frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig2_rolling_coverage")
    (OUT / "fig2_client.txt").write_text(f"representative client (median DTCW-R+ACI RMST, h=24): {name}\n")


def main():
    tb = pd.read_csv("results/combined/table_b_side_by_side.csv")
    fig1(tb)
    t24 = pd.read_csv("results/test/electricity_h24.csv.gz", low_memory=False)
    name = representative_client(t24)
    fig2(name, {h: rolling_runs(name, h) for h in (1, 24)})
    print(f"Figures in {OUT} (Fig. 2 client: {name})")


if __name__ == "__main__":
    main()
