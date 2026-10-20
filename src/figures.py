"""
Paper figures, vector PDF only (IEEE two-column: 3.5 in single, 7.16 in double column).

Main paper
  Fig. 1  fig1_method            DTCW-R+ACI pipeline and the anchored calibration window (diagram)
  Fig. 2  fig2_replication       single column: every confirmatory verdict (6 families x 9 baselines),
                                 cohort 1 vs cohort 2 side by side
  Fig. 3  fig3_tradeoff          scatter: relative Winkler (untouched test) vs RMST of TTR (shifts),
                                 h = 1 / 24, both cohorts
  Fig. 4  fig4_rolling_coverage  rolling coverage around one injected variance shift (kappa = 2, 45%),
                                 representative cohort-1 client chosen by rule (median DTCW-R+ACI RMST)
Supplement
  Fig. S1 figS1_cd_cohort1       Nemenyi critical-difference diagrams, cohort 1
  Fig. S2 figS2_cd_cohort2       the same, cohort 2
  Fig. S3 figS3_aci_layer        exploratory: DTCW-R+ACI vs DTCW-R alone and vs ACI alone (RMST)
  Fig. S4 figS4_protocol         splits and the pre-registered sequence (the paper describes it in text)
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
# label placement (points, alignment[, leader line]) so labels do not collide; default right-above
LABEL_POS = {
    (1, "ACI"): (-6, 2, "right"), (1, "ECI-cutoff"): (-6, -3, "right"),
    (1, "Conformal PID"): (-6, -6, "right"), (1, "DTCW-R+ACI"): (8, -4, "left"),
    (1, "Fixed 168"): (36, -10, "left", True), (1, "DTCW-R"): (-2, 8, "center"),
    (24, "DtACI"): (12, -3, "left"), (24, "ROCP"): (-7, 0, "right"),
    (24, "Fixed 168"): (10, -6, "left"), (24, "DTCW-R"): (10, 2, "left"),
    (24, "ECI-cutoff"): (0, 7, "center"), (24, "Conformal PID"): (0, -12, "right"),
}


def fig1(tb: pd.DataFrame):
    """Fig. 3: scatter of relative Winkler (untouched test, x) vs RMST of TTR (shifts, y), one panel
    per horizon; filled circle = cohort 1, open square = cohort 2, joined per method."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharey=True)
    for ax, h in zip(axes, (1, 24)):
        t = tb[tb.h == h]
        for _, r in t.iterrows():
            m = r["method"]
            col = GROUP.get(m, INK2)
            main = m == "DTCW-R+ACI"
            x1, y1, x2, y2 = r.rel_winkler_L2_c1, r.rmst_L1_c1, r.rel_winkler_L2_c2, r.rmst_L1_c2
            ax.plot([x1, x2], [y1, y2], color=col, lw=1.0, alpha=0.6, zorder=2)
            ax.scatter([x1], [y1], s=34 if main else 22, color=col, edgecolor=SURF, linewidth=1.0, zorder=3)
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


def _stage_box(ax, x, y, w, h, title, eq, ec, fc=SURF):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.1, zorder=3))
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=7, color=INK,
            fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.33, eq, ha="center", va="center", fontsize=6.6, color=INK, zorder=4,
            linespacing=1.5)


def _bracket(ax, x0, x1, y, text, color, tx=None, ha="center", beside=False):
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="|-|", color=color, lw=1.0, shrinkA=0, shrinkB=0,
                                mutation_scale=3))
    if beside:                                   # label to the right of a short bracket
        ax.text(x1 + 6, y, text, ha="left", va="center", fontsize=6.0, color=color)
    else:
        ax.text((x0 + x1) / 2 if tx is None else tx, y + 0.35, text, ha=ha, va="bottom",
                fontsize=6.0, color=color)


def _toy_stream(seed=7, n=200, shift=120):
    """Synthetic stream used only to draw Fig. 1 (not data)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    yhat = 10 + 3 * np.sin(2 * np.pi * t / 24)
    sd = np.where(t >= shift, 2.3, 1.0)
    y = yhat + rng.normal(0, 1, n) * sd
    s = np.abs(y - yhat)
    r = np.log(s / np.median(s[:100]) * 0.6745 + 1e-6)
    z = np.maximum((r - r[:100].mean()) / r[:100].std(), -1.5)
    g, last0, alarm, tau_hat = np.zeros(n), 0, None, None
    for k in range(1, n):
        g[k] = max(0.0, g[k - 1] + z[k] - 0.5)
        if g[k] == 0:
            last0 = k
        if alarm is None and k > 100 and g[k] > 6:
            alarm, tau_hat = k, last0 + 1
    if alarm is not None:
        g[alarm + 1:] = np.nan
    return t, y, yhat, s, g, alarm, tau_hat


def fig_method():
    """Fig. 1: the method as five illustrated stages (each tile draws its own step on one toy
    stream) plus the ACI feedback loop."""
    t, y, yhat, s, g, ta, th = _toy_stream()
    H, WL, WS = 6.0, 90, 24
    fig = plt.figure(figsize=(7.16, 3.25))
    ov = fig.add_axes([0, 0, 1, 1], zorder=0)
    ov.set_xlim(0, 1)
    ov.set_ylim(0, 1)
    ov.axis("off")
    w, gap, x0 = 0.172, 0.027, 0.015
    xs = [x0 + k * (w + gap) for k in range(5)]
    ty0, ty1 = 0.37, 0.93
    # stage groups
    for a, b, col, lab in ((1, 2, "#f3f2ee", "change detection"), (3, 4, "#e8f1fc", "calibration")):
        gx0, gx1 = xs[a] - 0.009, xs[b] + w + 0.009
        yb = 0.025 if a == 3 else ty0 - 0.02
        ov.add_patch(FancyBboxPatch((gx0, yb), gx1 - gx0, 0.995 - yb, boxstyle="round,pad=0,rounding_size=0.015",
                                    fc=col, ec="none", zorder=0))
        ov.text((gx0 + gx1) / 2, 0.965, lab, ha="center", va="center", fontsize=6.6, color=INK2,
                style="italic")
    titles = ["Score", "1  Detect", "2  Anchor", "3  Calibrate", "Interval"]
    eqs = ["$s_t=|y_t-\\hat y_{t|t-h}|$",
           "$g_t=\\max(0,g_{t-1}+z_t-k)$\nalarm if $g_t>H$",
           "$\\hat\\tau$ = last zero of $g$\nbefore the alarm $t_a$",
           "$n_t=\\mathrm{clip}(t-\\hat\\tau;\\,W_S,W_L)$\n$\\hat q_t=Q_{1-\\alpha_t}$(window)",
           "$\\hat y_{t+h}\\pm\\hat q_t$"]
    edge = [INK2, BLUE, BLUE, BLUE, INK2]
    axs = []
    for k, x in enumerate(xs):
        ov.add_patch(FancyBboxPatch((x, ty0), w, ty1 - ty0, boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=SURF, ec=edge[k], lw=1.0, zorder=1))
        ov.text(x + w / 2, ty1 - 0.035, titles[k], ha="center", va="center", fontsize=7, fontweight="bold",
                color=INK, zorder=2)
        ov.text(x + w / 2, ty0 + 0.065, eqs[k], ha="center", va="center", fontsize=6.4, color=INK,
                zorder=2, linespacing=1.4)
        a = fig.add_axes([x + 0.012, ty0 + 0.15, w - 0.024, 0.33], zorder=2)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        axs.append(a)
        if k:
            _arrow(ov, xs[k - 1] + w, ty0 + 0.31, x, ty0 + 0.31, color=INK, lw=1.0)
    shift = 120
    # tile 0: forecast and outcome
    a = axs[0]
    v = slice(84, 156)
    a.plot(t[v], yhat[v], color=BLUE, lw=0.9, ls="--")
    a.vlines(t[v], np.minimum(y[v], yhat[v]), np.maximum(y[v], yhat[v]), color=ORANGE, lw=0.6, alpha=0.7)
    a.plot(t[v], y[v], ".", ms=2.2, color=INK2)
    a.axvline(shift, color=ORANGE, lw=0.8)
    # tile 1: CUSUM
    a = axs[1]
    a.plot(t, g, color=BLUE, lw=0.9)
    a.axhline(H, color=INK2, lw=0.7, ls="--")
    a.text(2, H + 0.3, "H", fontsize=6, color=INK2, va="bottom")
    a.plot([ta], [g[ta]], marker="o", ms=4, color=BLUE)
    a.text(ta - 4, H - 0.6, "alarm $t_a$", fontsize=6.4, color=BLUE, ha="right", va="top")
    a.set_ylim(-0.5, max(H, np.nanmax(g)) + 1.5)
    # tile 2: anchor
    a = axs[2]
    lo = th - 25
    a.plot(t[lo:ta + 1], g[lo:ta + 1], color=BLUE, lw=0.9)
    a.axhline(H, color=INK2, lw=0.7, ls="--")
    a.plot([th - 1], [0], marker="^", ms=5, color=ORANGE)
    a.plot([ta], [g[ta]], marker="o", ms=4, color=BLUE)
    a.annotate("", xy=(th, 0.6), xytext=(ta - 1, g[ta] - 0.3),
               arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8, mutation_scale=6,
                               connectionstyle="arc3,rad=0.3"))
    a.text(th + 1, -0.4, "$\\hat\\tau$", fontsize=7, color=ORANGE, va="top")
    a.set_ylim(-1.2, max(H, np.nanmax(g)) + 1.0)
    a.set_xlim(lo, ta + 4)
    # tile 3: window
    a = axs[3]
    a.bar(t, s, width=1.0, color=INK2, alpha=0.55, lw=0)
    win_old = (ta - WL, th)
    a.axvspan(*win_old, color=INK2, alpha=0.12, lw=0)
    a.axvspan(th, t[-1], color=BLUE, alpha=0.18, lw=0)
    a.axvspan(ta, ta + WS, ymin=0.9, ymax=1.0, color=ORANGE, alpha=0.6, lw=0)
    qn = np.quantile(s[th:th + 60], 0.9)
    qo = np.quantile(s[win_old[0]:win_old[1]], 0.9)
    a.plot([win_old[0], th], [qo, qo], color=INK2, lw=1.0)
    a.plot([th, t[-1]], [qn, qn], color=BLUE, lw=1.0)
    a.text(t[-1], qn, " $\\hat q$", fontsize=6, color=BLUE, va="center", ha="left")
    a.text((win_old[0] + win_old[1]) / 2, s.max() * 0.98, "old", fontsize=6.2, color=INK2, ha="center", va="top")
    a.text((th + t[-1]) / 2, s.max() * 0.98, "from $\\hat\\tau$", fontsize=6.2, color=BLUE, ha="center", va="top")
    a.set_xlim(win_old[0] - 5, t[-1] + 12)
    # tile 4: interval band
    a = axs[4]
    q = np.full(len(t), np.nan)
    for k in range(40, len(t)):
        if ta is not None and k > ta + 1:
            n = int(min(WL, max(WS, k - th)))
        else:
            n = min(WL, k)
        q[k] = np.quantile(s[k - n:k], 0.9)
    a.fill_between(t, yhat - q, yhat + q, color=BLUE, alpha=0.18, lw=0)
    a.plot(t, yhat, color=BLUE, lw=0.8, ls="--")
    a.plot(t, y, ".", ms=1.6, color=INK2)
    a.axvline(shift, color=ORANGE, lw=0.8)
    a.set_xlim(84, t[-1])
    # ACI strip
    ax_x, ax_w = xs[3], xs[4] + w - xs[3]
    ov.add_patch(FancyBboxPatch((ax_x, 0.045), ax_w, 0.22, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=SURF, ec=BLUE, lw=1.0, zorder=1))
    ov.text(ax_x + 0.012, 0.225, "4  Adapt the level (ACI, h interleaved copies)", fontsize=6.8,
            fontweight="bold", color=INK, va="center", zorder=2)
    ov.text(ax_x + 0.012, 0.10, "$\\alpha^{(j)}\\leftarrow\\alpha^{(j)}+\\gamma(\\alpha-\\mathrm{miss}_t)$\n"
            "$j=t\\,\\mathrm{mod}\\,h$", fontsize=6.6, color=INK, va="center",
            zorder=2, linespacing=1.4)
    aa = fig.add_axes([ax_x + ax_w * 0.58, 0.065, ax_w * 0.39, 0.13], zorder=2)
    rng = np.random.default_rng(1)
    miss = (rng.random(160) < np.where(np.arange(160) > 80, 0.25, 0.1)).astype(float)
    al = [0.1]
    for m_ in miss:
        al.append(al[-1] + 0.01 * (0.1 - m_))
    aa.plot(al, color=BLUE, lw=0.8)
    aa.axhline(0.1, color=INK2, lw=0.6, ls="--")
    aa.text(1, 0.1, "$\\alpha$", fontsize=5.8, color=INK2, va="bottom")
    aa.text(158, al[-1], "$\\alpha_t$", fontsize=5.8, color=BLUE, va="center", ha="left")
    aa.axis("off")
    _arrow(ov, xs[4] + w * 0.5, ty0, xs[4] + w * 0.5, 0.265, color=INK2)
    ov.text(xs[4] + w * 0.5 + 0.005, 0.315, "miss$_t$ (h late)", fontsize=6.3, color=INK2, va="center")
    _arrow(ov, xs[3] + w * 0.5, 0.265, xs[3] + w * 0.5, ty0, color=INK2)
    ov.text(xs[3] + w * 0.5 + 0.005, 0.315, "$\\alpha_t$", fontsize=6.2, color=INK2, va="center")
    # visual key instead of a block of notes (details go in the caption)
    from matplotlib.patches import Patch
    key = [plt.Line2D([], [], marker=".", ls="", ms=6, color=INK2, label="outcome $y_t$"),
           plt.Line2D([], [], color=BLUE, ls="--", lw=1.0, label="forecast $\\hat y$"),
           plt.Line2D([], [], color=ORANGE, lw=0.8, alpha=0.7, label="score $s_t$ (gap)"),
           plt.Line2D([], [], color=ORANGE, lw=1.4, label="true shift"),
           plt.Line2D([], [], marker="o", ls="", ms=5, color=BLUE, label="alarm $t_a$"),
           plt.Line2D([], [], marker="^", ls="", ms=6, color=ORANGE, label="change point $\\hat\\tau$"),
           Patch(fc=INK2, alpha=0.25, label="window before alarm"),
           Patch(fc=BLUE, alpha=0.30, label="window from $\\hat\\tau$ (ours)"),
           Patch(fc=ORANGE, alpha=0.6, label="hard reset (ablation)")]
    fig.legend(handles=key, loc="lower left", bbox_to_anchor=(x0, 0.03), ncol=3, frameon=True,
               fontsize=7, handlelength=1.6, columnspacing=1.2, labelspacing=0.7, borderpad=0.8,
               title="Key", title_fontsize=7.2, edgecolor=GRID, facecolor=SURF)
    save(fig, "fig1_method")


def _pill(ax, x, y, text, fc, ec, color, fs=6.6, mono=True):
    """Text in a rounded badge (used for git tags)."""
    ax.text(x, y, text, ha="left", va="center", fontsize=fs, color=color, fontweight="bold",
            family="monospace" if mono else None,
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc=fc, ec=ec, lw=0.9))


def fig_protocol():
    """Fig. 2 (single IEEE column, 3.5 in): (a) one series to scale with what each part is used
    for; (b) the pre-registered sequence as a vertical timeline (tag badges = choices frozen)."""
    fig = plt.figure(figsize=(3.5, 3.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    TEST_C, GREEN_C = "#fde0d3", "#d6f0e5"
    # (a) splits
    ax.text(0.03, 0.975, "(a) Each series, in time order", fontsize=7.2, fontweight="bold", color=INK, va="top")
    x0, x1, yb, hb = 0.04, 0.96, 0.855, 0.05
    W = x1 - x0
    for a, b, col in ((0, 0.70, PALE), (0.70, 0.80, LIGHT_BLUE), (0.80, 1.0, TEST_C)):
        ax.add_patch(Rectangle((x0 + W * a, yb), W * (b - a), hb, fc=col, ec=SURF, lw=1.2))
    ax.text(x0 + W * 0.35, yb + hb / 2, "train 70%  (LightGBM, then frozen)", ha="center", va="center",
            fontsize=6.4, color=INK)
    ax.text(x0 + W * 0.75, yb + hb + 0.012, "val", ha="center", va="bottom", fontsize=6.3, color=BLUE)
    ax.text(x0 + W * 0.90, yb + hb + 0.012, "test", ha="center", va="bottom", fontsize=6.3, color=ORANGE)
    # zoom boxes
    zy, zh = 0.675, 0.135
    vb = (0.04, 0.43)
    tb = (0.47, 0.96)
    # colour links the zoom boxes to the bar segments; one short arrow from each segment
    for (xa, xb), (sa, sb), col in ((vb, (0.70, 0.80), BLUE), (tb, (0.80, 1.0), ORANGE)):
        _arrow(ax, x0 + W * (sa + sb) / 2, yb - 0.004, (xa + xb) / 2, zy + zh + 0.004, color=col, lw=0.8)
    ax.add_patch(FancyBboxPatch((vb[0], zy), vb[1] - vb[0], zh, boxstyle="round,pad=0,rounding_size=0.02",
                                fc=LIGHT_BLUE, ec=BLUE, lw=0.8))
    ax.text((vb[0] + vb[1]) / 2, zy + zh - 0.022, "Validation 10%", ha="center", va="top", fontsize=6.6,
            fontweight="bold", color=INK)
    ax.text((vb[0] + vb[1]) / 2, zy + 0.045, "warm-up, tuning,\ncalibration", ha="center", va="center",
            fontsize=6.3, color=INK, linespacing=1.25)
    ax.add_patch(FancyBboxPatch((tb[0], zy), tb[1] - tb[0], zh, boxstyle="round,pad=0,rounding_size=0.02",
                                fc=TEST_C, ec=ORANGE, lw=0.8))
    ax.text((tb[0] + tb[1]) / 2, zy + zh - 0.022, "Test 20%", ha="center", va="top", fontsize=6.6,
            fontweight="bold", color=INK)
    ax.text((tb[0] + tb[1]) / 2, zy + 0.060, "sealed until the tag; scored once", ha="center", va="center",
            fontsize=6.2, color=INK)
    for f in (0.15, 0.30, 0.45, 0.60, 0.75):
        ax.plot([tb[0] + (tb[1] - tb[0]) * f], [zy + 0.022], marker="^", ms=3.6, color=ORANGE, ls="")
    ax.text(tb[0] + (tb[1] - tb[0]) * 0.80, zy + 0.022, " shifts", fontsize=6.0, color=ORANGE,
            va="center", ha="left")
    # (b) vertical timeline
    ax.text(0.03, 0.625, "(b) Pre-registered sequence", fontsize=7.2, fontweight="bold", color=INK, va="top")
    events = [("run", "Tuning on validation only", "16 configs per method family"),
              ("tag", "prereg-v1", "configs, rules, file hashes frozen"),
              ("run", "Cohort 1 test", "50 clients + Elec2, opened once"),
              ("tag", "c2-predictions", "R1-R6 written before new data"),
              ("run", "50 fresh clients", "same code, re-tuned on validation"),
              ("tag", "prereg-c2", "frozen again"),
              ("end", "Cohort 2 test", "main claim replicated (R1-R3)")]
    lx, top, bot = 0.20, 0.555, 0.045
    ys = np.linspace(top, bot, len(events))
    ax.plot([lx, lx], [top + 0.015, bot - 0.015], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
    for (kind, name, desc), yy in zip(events, ys):
        if kind == "tag":
            ax.plot([lx], [yy], marker="D", ms=6.2, mfc=LIGHT_BLUE, mec=BLUE, mew=1.2, ls="", zorder=3)
            _pill(ax, lx + 0.05, yy + 0.012, name, LIGHT_BLUE, BLUE, BLUE, fs=6.4)
        else:
            fc, ec = (GREEN_C, AQUA) if kind == "end" else (SURF, INK2)
            ax.plot([lx], [yy], marker="o", ms=6.2, mfc=fc, mec=ec, mew=1.3, ls="", zorder=3)
            ax.text(lx + 0.05, yy + 0.012, name, ha="left", va="center", fontsize=6.8, fontweight="bold",
                    color=INK)
        ax.text(lx + 0.05, yy - 0.026, desc, ha="left", va="center", fontsize=6.2, color=INK2)
    # cohort brackets
    for (a, b), lab, col in (((0, 2), "Cohort 1", INK2), ((3, 6), "Cohort 2", AQUA)):
        ya, yb2 = ys[a] + 0.02, ys[b] - 0.02
        ax.plot([0.085, 0.085], [yb2, ya], color=col, lw=2.0, solid_capstyle="round")
        ax.text(0.05, (ya + yb2) / 2, lab, rotation=90, ha="center", va="center", fontsize=6.6,
                color=col, fontweight="bold")
    save(fig, "figS4_protocol")


def fig_replication(cs: pd.DataFrame):
    """Fig. 2 (single column): every confirmatory verdict of DTCW-R+ACI vs the 9 baselines, in
    cohort 1 (left half of each cell) and the pre-registered replication cohort 2 (right half)."""
    fams = [("RMST of TTR (L1 target)", "Recovery"), ("Winkler (L2)", "Width"),
            ("Worst local coverage (L1 target)", "Worst\ncoverage")]
    cols = [(h, f, lab) for h in (1, 24) for f, lab in fams]
    base = ["Fixed 168", "Fixed 720", "Fixed 2160", "ACI", "DtACI", "Conformal PID", "ECI-cutoff",
            "ROCP", "DriftOCP"]
    style = {"better": (BLUE, "+", SURF), "worse": (ORANGE, "−", SURF), "no claim": ("#e6e5e0", "·", INK2)}
    fig = plt.figure(figsize=(3.5, 3.35))
    ax = fig.add_axes([0.25, 0.115, 0.73, 0.80])
    ax.set_xlim(-0.5, len(cols) - 0.5 + 0.35)
    ax.set_ylim(len(base) - 0.5, -2.35)
    ax.axis("off")
    cw, gap = 0.44, 0.03
    for c, (h, fam, lab) in enumerate(cols):
        xc = c + (0.35 if h == 24 else 0)
        for r, b in enumerate(base):
            row = cs[(cs.h == h) & (cs.family == fam) & (cs.baseline == b)]
            if row.empty:
                continue
            for k, v in enumerate((row.verdict_c1.iloc[0], row.verdict_c2.iloc[0])):
                fc, glyph, tc = style[v]
                x = xc - cw - gap / 2 if k == 0 else xc + gap / 2
                ax.add_patch(Rectangle((x, r - 0.42), cw, 0.84, fc=fc, ec=SURF, lw=0.6))
                ax.text(x + cw / 2, r, glyph, ha="center", va="center", fontsize=7.5, color=tc,
                        fontweight="bold")
        ax.text(xc, -0.75, lab, ha="center", va="bottom", fontsize=6.4, color=INK, linespacing=1.1)
    for h, c0 in ((1, 1), (24, 4.35)):
        ax.text(c0, -1.85, f"h = {h}" + (" (one step)" if h == 1 else " (day ahead)"), ha="center",
                va="bottom", fontsize=7, fontweight="bold", color=INK)
        ax.plot([c0 - 1.45, c0 + 1.45], [-1.6, -1.6], color=INK2, lw=0.8)
    for r, b in enumerate(base):
        ax.text(-0.62, r, b, ha="right", va="center", fontsize=6.6, color=INK)
    ax.text(-0.62, -0.75, "vs baseline", ha="right", va="bottom", fontsize=6.2, color=INK2, style="italic")
    from matplotlib.patches import Patch
    hd = [Patch(fc=BLUE, label="+ better"), Patch(fc=ORANGE, label="− worse"),
          Patch(fc="#e6e5e0", label="· no claim")]
    fig.legend(handles=hd, loc="lower center", ncol=3, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.6, 0.035), handlelength=1.2, columnspacing=1.0)
    fig.text(0.6, 0.01, "each cell: left = cohort 1, right = cohort 2 (replication)", ha="center",
             fontsize=6.2, color=INK2)
    same = (cs.verdict_c1 == cs.verdict_c2).sum()
    fig.text(0.6, 0.975, f"DTCW-R+ACI vs 9 baselines: {same}/{len(cs)} verdicts identical in both cohorts",
             ha="center", va="top", fontsize=6.6, color=INK, fontweight="bold")
    save(fig, "fig2_replication")


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
    cs = pd.read_csv("results/combined/confirmatory_side_by_side.csv")
    fig_replication(cs)
    fig1(pd.read_csv("results/combined/table_b_side_by_side.csv"))
    t24 = pd.read_csv("results/test/electricity_h24.csv.gz", low_memory=False)
    name = representative_client(t24)
    fig2(name, {h: rolling_runs(name, h) for h in (1, 24)})
    fig_cd(json.loads(Path("results/analysis/report.json").read_text()), "cohort 1", "figS1_cd_cohort1")
    fig_cd(json.loads(Path("results/cohort2/analysis/report.json").read_text()), "cohort 2 (replication)",
           "figS2_cd_cohort2")
    fig_aci_layer(pd.read_csv("results/combined/aci_layer.csv"))
    fig_protocol()
    _BOOK.close()
    _BOOK = None
    print(f"PDF figures in {OUT}: fig1_method, fig2_replication, fig3_tradeoff, fig4_rolling_coverage, "
          f"figS1_cd_cohort1, figS2_cd_cohort2, figS3_aci_layer, figS4_protocol, and all_figures.pdf (Fig. 4 client: {name})")


if __name__ == "__main__":
    main()
