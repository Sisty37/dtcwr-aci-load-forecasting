"""
Paper figures, vector PDF only (IEEE two-column: 3.5 in single, 7.16 in double column).

Main paper
  Fig. 1  fig1_method            DTCW-R+ACI pipeline and the anchored calibration window (diagram)
  Fig. 2  fig2_protocol          splits and the pre-registered two-cohort sequence (diagram)
  Fig. 3  fig3_tradeoff          dot plots per method: relative Winkler (untouched test) | RMST of TTR
                                 (shifts), rows of panels h = 1 / 24, both cohorts
  Fig. 4  fig4_rolling_coverage  rolling coverage around one injected variance shift (kappa = 2, 45%),
                                 representative cohort-1 client chosen by rule (median DTCW-R+ACI RMST)
Supplement
  Fig. S1 figS1_cd_cohort1       Nemenyi critical-difference diagrams, cohort 1
  Fig. S2 figS2_cd_cohort2       the same, cohort 2
  Fig. S3 figS3_aci_layer        exploratory: DTCW-R+ACI vs DTCW-R alone and vs ACI alone (RMST)
All of them are also collected in all_figures.pdf. Old file names and PNGs are removed.

Run from the project root (main working copy, after src/combine_cohorts.py):
    python src/figures.py
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
GROUP = {"DTCW-R+ACI": BLUE, "Conformal PID": ORANGE, "ECI-cutoff": ORANGE, "ACI": AQUA, "DtACI": AQUA}

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


_BOOK = None          # PdfPages collecting every figure into all_figures.pdf
EXTS = ("pdf",)       # vector PDF only (what the IEEE template needs)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in EXTS:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    if _BOOK is not None:
        _BOOK.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 3 (CD diagrams)
def cd_panel(ax, avg_rank: dict, cd: float, title: str):
    """Nemenyi critical-difference diagram (Demsar 2006): rank 1 (best) on the left; methods whose
    average ranks differ by less than CD are joined by a thick bar."""
    items = sorted(avg_rank.items(), key=lambda kv: kv[1])
    k = len(items)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1)
    ax.grid(False)
    for sp in ("left", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["top"].set_visible(True)
    ax.xaxis.tick_top()
    ax.set_xticks(range(1, k + 1))
    ax.set_yticks([])
    ax.tick_params(axis="x", length=3, pad=1, labelsize=6)
    half = (k + 1) // 2
    for i, (m, r) in enumerate(items):
        left = i < half
        y = 0.72 - 0.12 * (i if left else (k - 1 - i))
        xt = 0.55 if left else k + 0.45
        main = m == "DTCW-R+ACI"
        ax.plot([r, r, xt], [1.0, y, y], color=BLUE if main else INK2, lw=0.9 if main else 0.6)
        ax.text(xt + (-0.1 if left else 0.1), y, f"{m} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=6.3, color=INK, fontweight="bold" if main else "normal")
    ranks = [r for _, r in items]
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for lvl, (a, b) in enumerate(cliques):
        y = 0.95 - 0.035 * lvl
        ax.plot([ranks[a] - 0.08, ranks[b] + 0.08], [y, y], color=INK, lw=1.8, solid_capstyle="round")
    ax.plot([1, 1 + cd], [0.03, 0.03], color=INK, lw=1.0)
    ax.text(1 + cd + 0.15, 0.03, f"CD = {cd:.2f}", ha="left", va="center", fontsize=6.3, color=INK)
    ax.set_title(title, color=INK, pad=12, fontsize=7.5)


def fig_cd(report: dict, cohort: str, name: str):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))
    lab = {"winkler_L2": "Winkler score (untouched test)", "rmst": "RMST of time-to-recover (shifts)"}
    for col, h in enumerate((1, 24)):
        fr = {f["outcome"]: f for f in report[f"electricity_h{h}"]["friedman"]}
        for row, out in enumerate(("winkler_L2", "rmst")):
            f = fr[out]
            cd_panel(axes[row, col], f["avg_rank"], f["critical_difference"],
                     f"{lab[out]}, h = {h}\nFriedman p = {f['p']:.1e}")
    fig.suptitle(f"Average ranks over 50 clients, {cohort} (lower rank is better)", fontsize=8, color=INK)
    fig.tight_layout()
    save(fig, name)


# ----------------------------------------------------------------------------- Fig. 1
ORDER = ["DTCW-R+ACI", "DTCW-R", "ACI", "DtACI", "Conformal PID", "ECI-cutoff", "ROCP", "DriftOCP",
         "Fixed 168", "Fixed 720", "Fixed 2160"]


def fig1(tb: pd.DataFrame):
    """Fig. 3: the trade-off as aligned dot plots. Rows = methods (same order everywhere);
    columns = width on untouched test streams | recovery after shifts; one row of panels per
    horizon. Filled circle = cohort 1, open square = cohort 2."""
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.3), sharey=True,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.42})
    cols = [("rel_winkler_L2", "Relative Winkler score, untouched test\n(lower = narrower calibrated intervals)", False),
            ("rmst_L1", "RMST of time-to-recover after shifts, steps\n(lower = faster recovery)", True)]
    ypos = {m: i for i, m in enumerate(ORDER)}
    for r, h in enumerate((1, 24)):
        t = tb[tb.h == h].set_index("method")
        for c, (key, xlabel, logx) in enumerate(cols):
            ax = axes[r, c]
            ax.axhspan(-0.5, 0.5, color=LIGHT_BLUE, alpha=0.55, lw=0, zorder=0)
            for m in ORDER:
                if m not in t.index:
                    continue
                y = ypos[m]
                col = GROUP.get(m, INK2)
                v1, v2 = t.loc[m, f"{key}_c1"], t.loc[m, f"{key}_c2"]
                ax.plot([v1, v2], [y, y], color=col, lw=1.0, alpha=0.6, zorder=2)
                ax.plot([v1], [y], "o", ms=4.6, color=col, mec=SURF, mew=0.8, zorder=3)
                ax.plot([v2], [y], "s", ms=4.4, mfc=SURF, mec=col, mew=1.1, zorder=3)
            if logx:
                ax.set_xscale("log")
                ax.set_xticks([100, 200, 300, 500])
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.grid(axis="y", visible=False)
            ax.set_title(f"h = {h}" + (" (one step ahead)" if h == 1 else " (day ahead)") +
                         (" - width" if c == 0 else " - recovery"), color=INK, loc="left", fontsize=7.5)
            if r == 1:
                ax.set_xlabel(xlabel)
    axes[0, 0].set_yticks(range(len(ORDER)))
    axes[0, 0].set_yticklabels(ORDER)
    for lab in axes[0, 0].get_yticklabels() + axes[1, 0].get_yticklabels():
        if lab.get_text() == "DTCW-R+ACI":
            lab.set_fontweight("bold")
            lab.set_color(INK)
    axes[0, 0].set_ylim(len(ORDER) - 0.5, -0.5)
    handles = [plt.Line2D([], [], marker="o", ms=4.5, ls="", color=INK2, label="Cohort 1 (50 clients)"),
               plt.Line2D([], [], marker="s", ms=4.5, ls="", mfc=SURF, mec=INK2,
                          label="Cohort 2, pre-registered replication (50 new clients)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.01))
    fig.subplots_adjust(left=0.14, right=0.99, top=0.90, bottom=0.13)
    save(fig, "fig3_tradeoff")


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
    save(fig, "fig4_rolling_coverage")
    (OUT / "fig4_client.txt").write_text(f"representative client (median DTCW-R+ACI RMST, h=24): {name}\n")


# ----------------------------------------------------------------------------- diagrams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

LIGHT_BLUE = "#cde2fb"
PALE = "#f3f2ee"


def _box(ax, x, y, w, h, text, fc=PALE, ec=INK2, bold=False, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, color=INK2, ls="-", rad=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw,
                                 color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def fig_method():
    """Fig. 1: DTCW-R(+ACI) pipeline (top) and the anchored window around an alarm (bottom)."""
    fig = plt.figure(figsize=(7.16, 3.45))
    ax = fig.add_axes([0, 0.44, 1, 0.56])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y, h, gap = 0.40, 0.42, 0.02
    spec = [(0.13, "Forecast $\\hat y_{t|t-h}$\n(frozen LightGBM);\n$y_t$ arrives\nh steps later"),
            (0.145, "Score\n$s_t=|y_t-\\hat y_{t|t-h}|$,\nstandardised log,\nwinsorised"),
            (0.16, "Magnitude CUSUM\ntwo-sided, threshold H\nat 0.5 false alarms\nper 1,000 steps"),
            (0.19, "Anchored window\nstarts at change point $\\hat\\tau$\n$n_t=\\min(W_L,\\max(W_S,n_{post}))$\nre-baseline, cooldown c"),
            (0.13, "Conformal\nquantile $\\hat q_t$\nof window scores\nat level $1-\\alpha_t$"),
            (0.10, "Interval\n$\\hat y\\pm\\hat q_t$")]
    x, pos = 0.02, []
    for i, (w, t) in enumerate(spec):
        main = i in (2, 3)
        _box(ax, x, y, w, h, t, fc=LIGHT_BLUE if main else PALE, ec=BLUE if main else INK2, fs=6.3)
        if i:
            _arrow(ax, pos[-1][0] + pos[-1][1], y + h / 2, x, y + h / 2)
        pos.append((x, w))
        x += w + gap
    cx, cw = pos[2]
    wx, ww = pos[3]
    qx, qw = pos[4]
    ix, iw = pos[5]
    _arrow(ax, cx + cw * 0.5, y, wx + ww * 0.5, y, color=BLUE, ls="--", rad=0.35)
    ax.text((cx + wx + ww) / 2 + 0.01, 0.13, "alarm at $t_a$: anchor $\\hat\\tau$, pause $W_S$ steps",
            fontsize=6, color=BLUE, ha="center")
    ax_x0 = qx
    _box(ax, ax_x0, 0.02, ix + iw - ax_x0, 0.24,
         "ACI layer: h interleaved copies $\\alpha^{(j)}_t$;\n"
         "$\\alpha^{(j)}\\leftarrow\\alpha^{(j)}+\\gamma(\\alpha-\\mathrm{miss}_t)$ when $y_t$ arrives",
         fc=LIGHT_BLUE, ec=BLUE, fs=6.1)
    _arrow(ax, ix + iw * 0.5, y, ix + iw * 0.5, 0.26)
    _arrow(ax, qx + qw * 0.5, 0.26, qx + qw * 0.5, y)
    ax.text(ix + iw * 0.5 + 0.005, 0.33, "miss, h late", fontsize=5.6, color=INK2, va="center")
    ax.text(qx + qw * 0.5 + 0.005, 0.33, "$\\alpha_t$", fontsize=6.3, color=INK2, va="center")
    ax.text(0.02, 0.92, "(a) DTCW-R+ACI pipeline (every quantity uses only scores known at the forecast origin)",
            fontsize=7, color=INK, fontweight="bold")
    # (b) timeline
    tx = fig.add_axes([0.03, 0.07, 0.95, 0.33])
    t = np.arange(0, 400)
    rng = np.random.default_rng(3)
    sc = np.abs(rng.normal(0, 1, 400)) * np.where(t >= 200, 2.4, 1.0)
    tx.plot(t, sc + 0.3, color=INK2, lw=0.6)
    tau, ta, WS = 200, 238, 48
    tx.axvline(tau, color=ORANGE, lw=1.0)
    tx.axvline(ta, color=BLUE, lw=1.0, ls="--")
    rows = [(-1.4, 60, tau, INK2, 0.30, "window before the alarm (last $W_L$ scores)"),
            (-3.0, tau, 400, BLUE, 0.35, "anchored window after the alarm: starts at $\\hat\\tau$, grows to $W_L$"),
            (-4.6, ta, ta + WS, ORANGE, 0.45, "a hard reset (2x2 ablation) would start at $t_a$ with only $W_S$ scores")]
    for yy, a, b, col, al, lab in rows:
        tx.add_patch(Rectangle((a, yy), b - a, 0.9, color=col, alpha=al, lw=0))
        tx.text(a + 2 if col != INK2 else a, yy + 0.45, "  " + lab, fontsize=5.8, va="center", color=INK,
                ha="left")
    tx.text(tau - 3, 8.3, "true shift $\\tau$ ($\\approx\\hat\\tau$)", fontsize=5.8, ha="right", color=ORANGE)
    tx.text(ta + 3, 8.3, "alarm $t_a$", fontsize=5.8, color=BLUE)
    tx.set_ylim(-5.0, 9.2)
    tx.set_xlim(0, 400)
    tx.set_yticks([])
    tx.set_xticks([])
    tx.grid(False)
    tx.spines["left"].set_visible(False)
    tx.set_xlabel("time (targets) $\\rightarrow$", fontsize=6.5, labelpad=1)
    tx.text(0, 9.6, "(b) Scores $s_t$ around a variance shift, and which scores enter the calibration window",
            fontsize=7, color=INK, fontweight="bold")
    save(fig, "fig1_method")


def fig_protocol():
    """Fig. 2: splits and pre-registered timeline of the two cohorts."""
    fig = plt.figure(figsize=(7.16, 2.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.94, "(a) Every series, in time order (widths not to scale)", fontsize=7,
            fontweight="bold", color=INK)
    x0, W, y0, hh = 0.01, 0.98, 0.70, 0.13
    # schematic widths (not to scale: the true split is 70/10/20)
    parts = [(0.40, "train 70%: fit LightGBM (direct), then frozen", PALE),
             (0.24, "validation 10%: warm-up (500),\ntuning, H / TTR / sigma calibration", LIGHT_BLUE),
             (0.36, "test 20%: sealed until the pre-registration tag;\none shift per replay at 15/30/45/60/75%",
              "#fde0d3")]
    x = x0
    for frac, lab, col in parts:
        ax.add_patch(Rectangle((x, y0), W * frac, hh, fc=col, ec=SURF, lw=2))
        ax.text(x + W * frac / 2, y0 + hh / 2, lab, ha="center", va="center", fontsize=6.3, color=INK)
        x += W * frac
    for f in (0.15, 0.30, 0.45, 0.60, 0.75):
        xx = x0 + W * (0.64 + 0.36 * f)
        ax.plot([xx, xx], [y0 - 0.03, y0], color=ORANGE, lw=1.0)
    ax.text(0.01, 0.55, "(b) Pre-registered sequence (git tags)", fontsize=7, fontweight="bold", color=INK)
    steps = [("Validation-only\ntuning, 16 configs\nper family", PALE, INK2),
             ("prereg-v1(.1)\nconfigs, rules,\nhashes frozen", LIGHT_BLUE, BLUE),
             ("Cohort 1 test\n50 clients\n+ Elec2", PALE, INK2),
             ("c2-predictions\nR1-R6 written\nbefore new data", LIGHT_BLUE, BLUE),
             ("Cohort 2: 50 fresh\nclients, same code,\nprereg-c2", PALE, INK2),
             ("Cohort 2 test\nmain claim\nreplicated", "#d6f0e5", AQUA)]
    n = len(steps)
    bw, gap, by, bh = 0.145, 0.019, 0.10, 0.36
    for i, (lab, fc, ec) in enumerate(steps):
        bx = 0.01 + i * (bw + gap)
        _box(ax, bx, by, bw, bh, lab, fc=fc, ec=ec, fs=6.3, bold=ec == BLUE)
        if i:
            _arrow(ax, bx - gap, by + bh / 2, bx, by + bh / 2)
    save(fig, "fig2_protocol")


def fig_aci_layer(al: pd.DataFrame):
    """Fig. S3 (exploratory): RMST difference of DTCW-R+ACI vs each of its parts."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.1), sharey=True)
    comps = ["DTCW-R+ACI - DTCW-R", "DTCW-R+ACI - ACI"]
    names = {"DTCW-R+ACI - DTCW-R": "vs DTCW-R alone (no ACI layer)",
             "DTCW-R+ACI - ACI": "vs ACI alone (no detector window)"}
    cohorts = [("c1", "Cohort 1", "o"), ("c2", "Cohort 2", "s"), ("pooled", "Pooled (100)", "D")]
    for ax, h in zip(axes, (1, 24)):
        d = al[al.h == h]
        for i, c in enumerate(comps):
            for j, (key, lab, mk) in enumerate(cohorts):
                r = d[(d.comparison == c) & (d.cohort == key)].iloc[0]
                yy = i + (j - 1) * 0.22
                col = BLUE if key == "pooled" else INK2
                ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=1.2)
                ax.plot([r.median_diff_rmst], [yy], marker=mk, ms=4.5, color=col, mfc=col if key != "c2" else SURF,
                        ls="", label=lab if i == 0 else None)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels([names[c] for c in comps])
        ax.set_title(f"h = {h}", color=INK)
        ax.invert_yaxis()
    fig.supxlabel("Median client-level RMST difference in steps (negative = DTCW-R+ACI recovers faster); "
                  "bars: 95% cluster-bootstrap CI", fontsize=7, color=INK, y=0.02)
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="upper center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.6, 1.04))
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, "figS3_aci_layer")


def main():
    global _BOOK
    import json
    from matplotlib.backends.backend_pdf import PdfPages
    OUT.mkdir(parents=True, exist_ok=True)
    for f in list(OUT.glob("fig*.p*")) + [OUT / "all_figures.pdf", OUT / "fig2_client.txt"]:
        if f.exists() and f.suffix in (".pdf", ".png", ".txt"):
            f.unlink()                          # remove old names and PNGs
    _BOOK = PdfPages(OUT / "all_figures.pdf")
    fig_method()
    fig_protocol()
    fig1(pd.read_csv("results/combined/table_b_side_by_side.csv"))
    t24 = pd.read_csv("results/test/electricity_h24.csv.gz", low_memory=False)
    name = representative_client(t24)
    fig2(name, {h: rolling_runs(name, h) for h in (1, 24)})
    fig_cd(json.loads(Path("results/analysis/report.json").read_text()), "cohort 1", "figS1_cd_cohort1")
    fig_cd(json.loads(Path("results/cohort2/analysis/report.json").read_text()), "cohort 2 (replication)",
           "figS2_cd_cohort2")
    fig_aci_layer(pd.read_csv("results/combined/aci_layer.csv"))
    _BOOK.close()
    _BOOK = None
    print(f"PDF figures in {OUT}: fig1_method, fig2_protocol, fig3_tradeoff, fig4_rolling_coverage, "
          f"figS1_cd_cohort1, figS2_cd_cohort2, figS3_aci_layer, and all_figures.pdf (Fig. 4 client: {name})")


if __name__ == "__main__":
    main()
