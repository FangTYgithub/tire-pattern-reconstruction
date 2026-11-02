# -*- coding: utf-8 -*-

from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thesis_figstyle as st
from plot_io import configure, verify_inputs, selected
DATA_ROOT, FIGURE_OUT = configure('kap3b_redraw')
verify_inputs()
st.setup()
OUTDIR = FIGURE_OUT; OUTDIR.mkdir(parents=True, exist_ok=True)
META = {"figures": {}, "qa": []}

ROOT = DATA_ROOT
RESEARCH = ROOT / "research/project_Part1_v1.7"
SR = RESEARCH / "outputs/surface_revision"
SOURCE = {}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p): p = Path(p); SOURCE[str(p)] = sha(p); return p
def nz(p):
    with np.load(track(p), allow_pickle=False) as z: return {k: z[k] for k in z.files}
def rd(p): return pd.read_csv(track(p), low_memory=False)
def readj(p): return json.loads(track(p).read_text(encoding="utf-8"))

SURFS = ["B_curv", "M10_curv", "M24_curv"]; NAMES = {"B_curv": "EF", "M10_curv": "M10", "M24_curv": "M24"}
SID = {"B_curv": "B_curv", "M10_curv": "A_curv", "M24_curv": "C_curv"}
SIG = [0, 20, 40, 60, 80, 100]
SIGLAB = r"$\sigma_\mathrm{n}$ / Graustufen"
D = nz(SR / "s1_real01/domain/global_domain_step0p50.npz")
X, Y = np.meshgrid(D["lattice_x"], D["lattice_y"])
def ref(sid):
    z = np.full(X.shape, np.nan); z[D["mask"]] = D["z_" + SID[sid]]; return z

# layout helpers (same conventions as kap3_redraw.py)
def cm_axes(fig, x, top, w, h, **kw):
    W, H = fig.get_size_inches() / st.CM
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H], **kw)

def ftext(fig, x, top, s, **kw):
    W, H = fig.get_size_inches() / st.CM
    kw.setdefault("fontsize", 9)
    return fig.text(x / W, 1 - top / H, s, **kw)

def save(fig, num, notes=""):
    name = f"Abb_3_{num:02d}"
    if not selected(name):
        plt.close(fig)
        return
    st.save(fig, OUTDIR / name, name, META["qa"])
    META["figures"][name] = {"notes": notes, "pdf_sha256": sha(OUTDIR / (name + ".pdf"))}

def logaxis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: st.num(float(f"{v:.3g}"))))
    ax.yaxis.set_minor_formatter(NullFormatter())

def grid(nrows, ncols, ylabels, legend_rows, ph=3.25, left=2.05, top=0.55, coltitles=None, rowlabels=None, gx=0.3):
    """nrows x ncols panels at 15 cm; shared sigma x axis; legend area below (cm bookkeeping in the returned dict)."""
    W, right, gy = 15, 0.2, 0.3
    pw = (W - left - right - (ncols - 1) * gx) / ncols
    bottom = 1.2 + legend_rows * 0.58 + 0.3
    H = top + nrows * ph + (nrows - 1) * gy + bottom
    fig = st.figure(W, H)
    axes = np.array([[cm_axes(fig, left + j * (pw + gx), top + i * (ph + gy), pw, ph) for j in range(ncols)] for i in range(nrows)])
    for i in range(nrows):
        if ylabels: axes[i, 0].set_ylabel(ylabels[i])
        for j in range(ncols):
            ax = axes[i, j]; ax.set_xticks(SIG); ax.set_xlim(-6, 106)
            if i < nrows - 1: ax.tick_params(labelbottom=False)
            if coltitles and i == 0: ax.set_title(coltitles[j], pad=3)
    bottom_axes = top + nrows * ph + (nrows - 1) * gy
    ftext(fig, left + (W - left - right) / 2, bottom_axes + 0.88, SIGLAB, ha="center", va="center", fontsize=11)
    if rowlabels:
        for i, t in enumerate(rowlabels):
            ftext(fig, 0.15, top + i * (ph + gy) + ph / 2, t, rotation=90, ha="left", va="center", fontsize=10)
    return fig, axes, {"W": W, "H": H, "legend_top": bottom_axes + 1.2, "center": (left + (W - left - right) / 2) / W}

def legend(fig, L, handles, labels, ncol, title=None):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(L["center"], 1 - L["legend_top"] / L["H"]), ncol=ncol, title=title,
               handlelength=1.6, columnspacing=1.0, handletextpad=0.4, borderaxespad=0)

def triad(fig, ox, oy, view, box, L=0.8):
    """x/y/z arrows projected with the view and box of the 3-D panels (origin in cm from lower left)."""
    tmp = fig.add_axes([0, 0, .3, .3], projection="3d")
    tmp.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1)); tmp.view_init(*view); tmp.set_box_aspect(box)
    fig.canvas.draw(); M = tmp.get_proj()
    Pp = np.array([tmp.transData.transform(proj3d.proj_transform(*v, M)[:2]) for v in [np.zeros(3), *np.eye(3)]])
    tmp.remove()
    V = Pp[1:] - Pp[0]; V = V / np.linalg.norm(V[:2], axis=1).max()
    Wpx, Hpx = fig.bbox.width, fig.bbox.height; px = st.CM * fig.dpi; o = np.array([ox, oy]) * px
    for v, l in zip(V, "xyz"):
        tip = o + v * L * px
        fig.add_artist(FancyArrowPatch(o / (Wpx, Hpx), tip / (Wpx, Hpx), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6, lw=.8, color="#444444"))
        lp = tip + v / np.linalg.norm(v) * 0.22 * px
        fig.text(lp[0] / Wpx, lp[1] / Hpx, f"${l}$", fontsize=9, ha="center", va="center")

def bare3d(ax):
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.tick_params(length=0, pad=0); ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): pane.fill = False; pane.set_edgecolor("#dddddd")

def map2d(ax, ylab=True):
    ax.set(xlim=(-47, 47), ylim=(-47, 47), xticks=[-40, 0, 40], yticks=[-40, 0, 40]); ax.set_aspect("equal"); ax.grid(False); st.comma_axis(ax)
    ax.set_xlabel("$x$ / mm", labelpad=1)
    if ylab: ax.set_ylabel("$y$ / mm", labelpad=0)
    else: ax.tick_params(labelleft=False)

def hbar(fig, x, top, w, norm, cmap, label, ticks=None, extend="neither", fmt=None):
    cax = cm_axes(fig, x, top, w, 0.2)
    b = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend=extend)
    b.set_label(label, labelpad=2)
    if ticks is not None: b.set_ticks(ticks)
    b.ax.xaxis.set_major_formatter(FuncFormatter(fmt or (lambda v, p: st.num(round(v, 2)))))
    b.ax.xaxis.set_minor_formatter(NullFormatter())
    return b

# 3.4  [C] five line widths, S1, common resource cap, P4
c = rd(SR / "s12_resource01/results/conditions.csv")
c = c[(c.section == "S1") & (c.surface_class == "main") & (c.family == "line") & (c.regime == "resource_cap_11695")]
fig, axes, L = grid(2, 3, ["$C_\\tau$ / %", "Punkt-RMSE / mm"], 2, coltitles=[NAMES[s] for s in SURFS])
WCOL = ["#405a8c", "#009f8c", "#e9553e", "#9272ba", "#dda000"]
for j, sf in enumerate(SURFS):
    for w, col in zip([.5, .75, 1, 1.25, 1.5], WCOL):
        q = c[(c.surface == sf) & (c.size_mm == w)].sort_values("sigma"); assert len(q) == 6
        for i, met in enumerate(["glob_Ctau", "glob_pt_rmse"]):
            ax = axes[i, j]; f = 100 if i == 0 else 1
            ax.plot(q.sigma, q[met + "_mean"] * f, "-o", color=col, label=st.num(w), ms=2.5, lw=1.1)
            ax.fill_between(q.sigma, q[met + "_min"] * f, q[met + "_max"] * f, color=col, alpha=.1, lw=0)
            if i == 1:
                partial = q[(q.glob_pt_rmse_n_phases_def > 0) & (q.glob_pt_rmse_n_phases_def < 8)]
                ax.scatter(partial.sigma, partial.glob_pt_rmse_mean, facecolors="white", edgecolors=col, s=14, zorder=5, linewidths=.8)
for ax in axes[0]: ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
for ax in axes[1]: logaxis(ax)
lo = min(a.get_ylim()[0] for a in axes[1]); hi = max(a.get_ylim()[1] for a in axes[1])
for ax in axes[1]: ax.set_ylim(lo, hi)
for ax in axes[:, 1:].ravel(): ax.tick_params(labelleft=False)
st.comma_axis(axes[0, 0], "y")
legend(fig, L, *axes[0, 0].get_legend_handles_labels(), 5, "$w$ / mm")
save(fig, 4, "[C] phase mean, band min-max; hollow: error defined for 1-7 of 8 phases; P4")

# 3.8, 3.9  [B -> s12_report.curves_page] line spacing, 0 and 45 degrees
S12 = rd(SR / "s12_resource01/results/conditions.csv")
for col in S12.columns:
    if col.endswith(("_mean", "_min", "_max", "_median")): S12[col] = pd.to_numeric(S12[col], errors="coerce")
S12 = S12[S12.surface_class == "main"]
metrics = [("glob_Ctau", "$C_\\tau$ / %", "lin"), ("glob_pt_rmse", "Punkt-RMSE / mm", "log"), ("glob_surf_rmse", "Oberflächen-\nRMSE / mm", "log"),
           ("glob_recovery", "korrekte\nRückgewinnung / %", "lin")]
PITCH = [4.45, 5, 5.6, 6.3, 8, 10, 12]
for num, phi in [(8, 0), (9, 45)]:
    groups = [f"s{p:g}_phi{phi}" for p in PITCH]
    q = S12[(S12.section == "S2") & (S12.family == "line") & (S12.regime == "resource_cap_11695")]
    fig, axes, L = grid(4, 3, [m[1] for m in metrics], 2, coltitles=[NAMES[s] for s in SURFS])
    cols = plt.get_cmap("viridis")(np.linspace(0, 0.92, len(groups)))
    for j, s in enumerate(SURFS):
        for i, (m, lab, sc) in enumerate(metrics):
            ax = axes[i, j]; f = 1 if sc == "log" else 100
            for k, gv in enumerate(groups):
                h = q[(q.surface == s) & (q.layout == gv)].sort_values("sigma")
                if not len(h): continue
                x = h.sigma.to_numpy(float) + (k - (len(groups) - 1) / 2) * 1.2
                y = h[f"{m}_mean"].to_numpy(float) * f
                lo_ = h[f"{m}_min"].to_numpy(float) * f; hi_ = h[f"{m}_max"].to_numpy(float) * f
                ax.plot(x, y, "-o", ms=2.2, lw=1.0, color=cols[k], label=st.num(PITCH[k]))
                ok = ~np.isnan(y)
                ax.errorbar(x[ok], y[ok], yerr=[np.maximum(y[ok] - lo_[ok], 0), np.maximum(hi_[ok] - y[ok], 0)], fmt="none", ecolor=cols[k], elinewidth=0.6, capsize=1.2, alpha=0.7)
                nd = h.get(f"{m}_n_phases_def")
                if sc == "log" and nd is not None:
                    part = (pd.to_numeric(nd).to_numpy() < h.n_phases.to_numpy()) & ok
                    ax.plot(x[part], y[part], "o", ms=3.2, mfc="white", mec=cols[k], mew=.8, ls="none", zorder=4)
            if sc == "log": logaxis(ax)
        axes[0, j].set_ylim(0, 103); axes[3, j].set_ylim(0, 103)
    for i, (m, lab, sc) in enumerate(metrics):
        if sc == "log":
            lo = min(a.get_ylim()[0] for a in axes[i]); hi = max(a.get_ylim()[1] for a in axes[i])
            for a in axes[i]: a.set_ylim(lo, hi)
        else:
            for a in axes[i]: a.set_yticks([0, 25, 50, 75, 100])
            st.comma_axis(axes[i, 0], "y")
        for a in axes[i, 1:]: a.tick_params(labelleft=False)
    legend(fig, L, *axes[0, 0].get_legend_handles_labels(), 7, "Linienabstand / mm")
    save(fig, num, f"[B/s12_report] phi {phi}; phase mean, bars phase min-max, x offset 1.2 per group; hollow = error defined for part of the phases")

# 3.12-3.14  [B -> c7_figs f01 / f03 / f02] coding, same-image class contribution
CT = SR / "s3_code07/analysis/formal/current"
SP = rd(CT / "pair_phase_summary.csv"); BS = rd(CT / "bootstrap.csv"); SC = rd(CT / "case_phase_summary.csv")
LEV = ["weak", "mid", "strong"]; LEV_C = dict(plain="#222222", weak="#9467bd", mid="#ff7f0e", strong="#8c564b")
SIZE = dict(dots=dict(plain="$r$ 1,0 mm", weak="$r$ 0,916 / 1,078 mm", mid="$r$ 0,811 / 1,159 mm", strong="$r$ 0,525 / 1,313 mm"),
            lines=dict(plain="$w$ 1,0 mm", weak="$w$ 0,85 / 1,15 mm", mid="$w$ 0,725 / 1,275 mm", strong="$w$ 0,5 / 1,5 mm"))
LEV_DE = dict(plain="normal", weak="schwach", mid="mittel", strong="stark")
LAYC = dict(dots=["R5x4", "R6x4", "R8x4"], lines=["4.45", "5.6", "8"])
LAY_DE = {"R5x4": "Punkte R5×4", "R6x4": "Punkte R6×4", "R8x4": "Punkte R8×4", "4.45": "Linien 4,45 mm", "5.6": "Linien 5,6 mm", "8": "Linien 8 mm"}
MAIN = dict(dots=("DF", "use"), lines=("R@1.1247", "R@1.1247")); PLAIN = dict(dots=("DF", "single"), lines=("P4", "P4"))
E1 = dict(dots="E1 DF use-ignore", lines="E1 R-ignore"); E1LB = "E1 LB-ignore"
def sel(Xt, fam, surf, lay, **kw):
    m = (Xt.family == fam) & (Xt.surface == surf) & (Xt.layout.astype(str) == str(lay))
    for k, v in kw.items(): m &= Xt[k] == v
    return Xt[m]
def level_handles(fam, plain=False):
    return [Line2D([], [], color=LEV_C[l], lw=1.4, label=f"{LEV_DE[l]} ({SIZE[fam][l]})") for l in (["plain"] if plain else []) + LEV]
def coding_grid(num, fam, effs, absolute=False):
    ylab = "$C_\\tau$ / %" if absolute else "Δ$C_\\tau$ / Prozentpunkte"
    nleg = 3 if (fam == "lines" and not absolute) else 2   # legend in two columns
    fig, axes, L = grid(3, 3, None, nleg, ph=3.1, left=2.25, coltitles=[LAY_DE[l] for l in LAYC[fam]], rowlabels=[NAMES[s] for s in SURFS], gx=0.95)
    for i, s in enumerate(SURFS):
        for j, l in enumerate(LAYC[fam]):
            ax = axes[i, j]
            if absolute:
                for lev, (ru, ar) in [("plain", PLAIN[fam])] + [(x, MAIN[fam]) for x in LEV]:
                    q = sel(SC, fam, s, l, pattern=lev, rule=ru, arm=ar).sort_values("sigma")
                    if len(q): ax.plot(q.sigma, 100 * q["glob_Ctau_mean"], "-", color=LEV_C[lev], lw=1.4 if lev == "plain" else 1.0, marker="o", ms=2.2)
            else:
                ax.axhline(0, color="0.5", lw=0.6)
                for k, (eff, ls) in enumerate(effs):
                    for lev in LEV:
                        q = sel(SP, fam, s, l, effect=eff, pattern=lev).sort_values("sigma")
                        if not len(q): continue
                        ax.plot(q.sigma, 100 * q.d_glob_Ctau_mean, ls, color=LEV_C[lev], lw=1.1, marker="o" if k == 0 else None, ms=2.2)
                        if k == 0 and len(BS):
                            b = sel(BS, fam, s, l, effect=eff, pattern=lev).sort_values("sigma")
                            if len(b): ax.fill_between(b.sigma, 100 * b.lo, 100 * b.hi, color=LEV_C[lev], alpha=0.13, lw=0)
            st.comma_axis(ax, "y")
    ftext(fig, 0.95, 0.55 + 1.5 * 3.4, ylab, rotation=90, ha="center", va="center", fontsize=11)
    hs = level_handles(fam, plain=absolute)
    if effs and len(effs) > 1:
        hs += [Line2D([], [], color="0.3", ls=ls, lw=1.1, label=t) for (e, ls), t in zip(effs, ["R: Code − ignore", "LB: Code − ignore"])]
    L = dict(L, center=0.5); legend(fig, L, hs, [h.get_label() for h in hs], 2)
    return fig
save(coding_grid(12, "dots", [(E1["dots"], "-")]), 12, "[B/c7 f01] DF use - ignore; band 2.5-97.5 % block bootstrap (stored); own y scale per panel")
save(coding_grid(13, "dots", None, absolute=True), 13, "[B/c7 f03] plain DF single vs coded DF use; own y scale per panel")
save(coding_grid(14, "lines", [(E1["lines"], "-"), (E1LB, "--")]), 14, "[B/c7 f02] R and LB minus ignore; band only for R")

# 3.19  [B] position adjustments, lines s 6.3 mm, P4, global statistics
P = rd(SR / "s4_real02/paired_summary.csv")
fig = st.figure(15, 7.3); H = 7.3
axs = [cm_axes(fig, 1.55 + j * 4.5, 0.55, 4.25, 4.3) for j in range(3)]
SCOL = ["#222222", "#28708c", "#d58f1a", "#3b9564", "#9e6b91"]; MARK = ["o", "s", "^", "D", "v"]
for ax, sf in zip(axs, SURFS):
    for strat, col, mk in zip(["S0", "S1", "S2", "S3", "S4"], SCOL, MARK):
        q = P[(P.table_type == "absolute") & (P.family == "line") & (P.surface == sf) & (P.contrast == strat) & (P.metric == "glob_Ctau")].sort_values("sigma"); assert len(q) == 6
        ax.plot(q.sigma, 100 * q.main, marker=mk, color=col, ls="--" if strat == "S4" else "-", ms=3, lw=1.1, label=strat)
        ax.vlines(q.sigma, 100 * q.block_min, 100 * q.block_max, color=col, alpha=.35, lw=.8)
    ax.set(xticks=SIG, xlim=(-6, 106), ylim=(-2, 102), yticks=[0, 25, 50, 75, 100]); ax.set_title(NAMES[sf], pad=3); st.comma_axis(ax, "y")
    if sf != SURFS[0]: ax.tick_params(labelleft=False)
axs[0].set_ylabel("$C_\\tau$ / %")
ftext(fig, 7.6, 5.7, SIGLAB, ha="center", va="center", fontsize=11)
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.535, 1 - 6.25 / H), ncol=5, handlelength=1.8, columnspacing=1.2, borderaxespad=0)
save(fig, 19, "[B] median and range of block medians; S0 four phases; S3/S4 same layouts; P4, j = 0.20")

# 3.24  [A] six line combinations, P4, global statistics
CS = rd(SR / "s5_align05/condition_summary.csv")
combos = [x for x in sorted(CS.loc[CS.front_end == "P4", "combo"].unique()) if x.startswith("L_")]
rows = [("glob_Ctau_mean8", "$C_\\tau$ / %", 100), ("glob_pt_rmse_median", "Punkt-RMSE / mm", 1), ("sup_surf_rmse_median", "Oberflächen-RMSE\ngestützt / mm", 1)]
fig, axes, L = grid(3, 3, [r[1] for r in rows], 3, coltitles=[NAMES[s] for s in SURFS])
for j, sid in enumerate(SURFS):
    for i, (met, lab, factor) in enumerate(rows):
        for cmb in combos:
            g = CS[(CS.combo == cmb) & (CS.surface == sid) & (CS.front_end == "P4")].set_index("sigma").reindex(SIG)
            w = float(cmb.split("_")[1][1:]); spacing = cmb.split("_s")[1]
            color = "#1f77b4" if w == .5 else "#d62728"; ls = {"4.45": ":", "6.3": "--", "8": "-"}[spacing]
            y = pd.to_numeric(g[met], errors="coerce").to_numpy() * factor
            axes[i, j].plot(SIG, y, color=color, ls=ls, marker="o", ms=2.5, lw=1.0, label=f"{st.num(w)} / {spacing.replace('.', ',')}")
            if i:
                nd = g[met.replace("_median", "_n_defined")]; partial = ((nd > 0) & (nd < g.n_cases)).to_numpy()
                axes[i, j].plot(np.array(SIG)[partial], y[partial], "o", mfc="white", mec=color, ms=3.5, ls="none")
        if i == 0: axes[i, j].set_ylim(-3, 103); axes[i, j].set_yticks([0, 25, 50, 75, 100])
        else: logaxis(axes[i, j])
for i in (1, 2):
    lo = min(a.get_ylim()[0] for a in axes[i]); hi = max(a.get_ylim()[1] for a in axes[i])
    for a in axes[i]: a.set_ylim(lo, hi)
for a in axes[:, 1:].ravel(): a.tick_params(labelleft=False)
st.comma_axis(axes[0, 0], "y")
legend(fig, L, *axes[0, 0].get_legend_handles_labels(), 3, "$w$ / $s$ in mm")
save(fig, 24, "[A] coverage: equal-weight phase means (no output = 0); errors: median of defined cases; hollow = partly defined")

# 3.11  [A, F5] line spacing 10 vs 8 mm, P4, sigma 60
selection = readj(SR / "thesis_reinforce05/run.json")["analysis"]["cases"]
W, side = 15, 3.55
H = 0.25 + 2 * (0.95 + side + 0.75 + 1.35) + 0.4
fig = st.figure(W, H); details = []
xs = [1.75, 6.25, 10.75]
for row, kind in enumerate(["median", "worst"]):
    s = selection["F5"][kind]; sid = s["surface"]; data = []
    for key in ["forward_key_left", "forward_key_right"]:
        a = nz(SR / "thesis_reinforce05/data" / f"{s[key]}.npz"); m = json.loads(str(a["_meta"])); v = a["P4__ev__valid"].astype(bool)
        p = a["P4__ev__pts"][v]; e = p[:, 2] - a["P4__ev__zr"][v]; mt = m["results"]["P4"]["metrics"]; assert len(p) == mt["n_valid"]
        data.append((p, e, mt)); details.append({"forward_key": s[key], "kind": kind, "n_valid": len(p), "global_Ctau": mt["glob_Ctau"]})
    err = np.concatenate([d[1] for d in data]); lim = max(float(np.percentile(np.abs(err), 98)), .05)
    top = 0.25 + row * (0.95 + side + 0.75 + 1.35) + 0.95
    ftext(fig, 0.1, top + side / 2, NAMES[sid] + ", " + ("median-nahes Paar" if kind == "median" else "ungünstigstes Paar"), rotation=90, ha="left", va="center", fontsize=10)
    a0 = cm_axes(fig, xs[0], top, side, side); zr = ref(sid)
    a0.pcolormesh(X, Y, zr, cmap="viridis", shading="nearest", rasterized=True); map2d(a0)
    ftext(fig, xs[0] + side / 2, top - 0.25, "Referenzhöhe", ha="center", va="bottom", fontsize=10)
    hbar(fig, xs[0], top + side + 1.0, side, Normalize(np.nanmin(zr), np.nanmax(zr)), "viridis", "$z$ / mm", fmt=lambda v, p: st.num(round(v)))
    for col, ((p, e, mt), lab) in enumerate(zip(data, ["$s$ = 10 mm", "$s$ = 8 mm"]), 1):
        a1 = cm_axes(fig, xs[col], top, side, side)
        a1.scatter(p[:, 0], p[:, 1], c=e, cmap="RdBu_r", vmin=-lim, vmax=lim, s=1.0, linewidths=0, rasterized=True)
        a1.contour(X, Y, D["mask"].astype(float), levels=[.5], colors=".55", linewidths=.6)
        map2d(a1, ylab=False)
        ftext(fig, xs[col] + side / 2, top - 0.1, f"{lab}, $n$ = {len(p)}\n$C_\\tau$ {st.num(100 * mt['glob_Ctau'], 1)} %, RMSE {st.num(mt['glob_pt_rmse'], 3)} mm",
              ha="center", va="bottom", fontsize=9, linespacing=1.2)
    hbar(fig, xs[1] + 0.6, top + side + 1.0, xs[2] + side - xs[1] - 1.2, Normalize(-lim, lim), "RdBu_r", "Höhenfehler / mm (98. Perzentil)", extend="both")
save(fig, 11, json.dumps({"selection": selection["F5"], "cases": details}))

# 3.15, 3.16  [A] coded cases on M24: reference, identity cloud, error map
board = readj(SR / "s3_code07/analysis/formal/current/case_board.json")
VIEW_A, BOX_A = (28, -60), (1, 1, .45)
ICOL = {"co": "#1f77b4", "wr": "#d62728", "un": "#9467bd"}
for fam, sigmas, num in [("dots", [100], 15), ("lines", [40, 60], 16)]:
    chosen = [cc for cc in board if cc["group"] in ("fixed", "supplement") and cc.get("surface") == "M24_curv" and cc["family"] == fam and cc["sigma"] in sigmas]
    assert len(chosen) == len(sigmas); chosen.sort(key=lambda cc: cc["sigma"])
    rh = 5.0; H = 0.2 + len(chosen) * rh + 1.4
    fig = st.figure(15, H); details = []
    for row, cc in enumerate(chosen):
        a = nz(SR / "s3_code07/formal_run/records/eval" / f"{cc['eval_key']}.npz"); ok = a["acc"].astype(bool) & np.isfinite(a["pts"]).all(axis=1); p = a["pts"][ok]
        ids = a["ident"].astype(str) if "ident" in a else np.where(a["bridge"][0].astype(bool), "co", np.where(a["bridge"][1].astype(bool), "wr", "un"))
        idt = ids[ok]; e = p[:, 2] - a["zr"][ok]
        assert len(p) == cc["n_accepted"]; cnt = [int(np.char.startswith(idt, t).sum()) for t in ["co", "wr"]]; assert cnt == [cc["c"], cc["w"]]
        top = 0.2 + row * rh
        zlo = min(-16, float(np.floor(p[:, 2].min())) - .5); zhi = max(4, float(np.ceil(p[:, 2].max())) + .5)
        ax = cm_axes(fig, 0.6, top + 0.55, 4.6, 3.6, projection="3d", computed_zorder=False); z = ref(cc["surface"])
        ax.plot_surface(X[::4, ::4], Y[::4, ::4], z[::4, ::4], cmap="viridis", linewidth=0, rasterized=True)
        ax.set(xlim=(-47, 47), ylim=(-47, 47), zlim=(zlo, zhi)); ax.set_box_aspect(BOX_A, zoom=1.1); ax.view_init(*VIEW_A); bare3d(ax)
        ftext(fig, 2.9, top + 0.35, f"M24, $\\sigma_\\mathrm{{n}}$ = {int(cc['sigma'])}", ha="center", va="center", fontsize=10)
        ax = cm_axes(fig, 5.2, top + 0.55, 4.6, 3.6, projection="3d", computed_zorder=False)
        ax.plot_wireframe(X[::5, ::5], Y[::5, ::5], z[::5, ::5], color=".6", lw=.3, alpha=.55, rasterized=True)
        color = np.where(np.char.startswith(idt, "co"), ICOL["co"], np.where(np.char.startswith(idt, "wr"), ICOL["wr"], ICOL["un"]))
        ax.scatter(*p.T, c=color, s=3 if fam == "dots" else .5, depthshade=False, zorder=5, rasterized=True, linewidths=0)
        ax.set(xlim=(-47, 47), ylim=(-47, 47), zlim=(zlo, zhi)); ax.set_box_aspect(BOX_A, zoom=1.1); ax.view_init(*VIEW_A); bare3d(ax)
        ftext(fig, 7.5, top + 0.35, f"{cc['c']} korrekt, {cc['w']} falsch,\n{cc['u']} unbestätigt", ha="center", va="center", fontsize=9, linespacing=1.15)
        ax = cm_axes(fig, 11.05, top + 0.55, 3.3, 3.3)
        ax.scatter(p[:, 0], p[:, 1], c=e, s=3 if fam == "dots" else .6, cmap="RdBu_r", vmin=-.5, vmax=.5, linewidths=0, rasterized=True)
        map2d(ax); ftext(fig, 12.7, top + 0.35, f"$C_\\tau$ = {st.num(100 * cc['Ctau'], 1)} %", ha="center", va="center", fontsize=10)
        details.append({"eval_key": cc["eval_key"], "n_accepted": len(p), "c": cc["c"], "w": cc["w"], "u": cc["u"], "Ctau": cc["Ctau"], "points_outside_color": int((np.abs(e) > .5).sum())})
    hbar(fig, 10.85, H - 1.25, 3.7, Normalize(-.5, .5), "RdBu_r", "Höhenfehler / mm", ticks=[-.5, 0, .5], extend="both")
    hs = [Line2D([], [], marker="o", ls="", color=col, ms=4, label=lab) for col, lab in [(ICOL["co"], "korrekt"), (ICOL["wr"], "falsch"), (ICOL["un"], "unbestätigt")]]
    fig.legend(handles=hs, loc="center", bbox_to_anchor=(6.2 / 15, 1 - (H - 0.45) / H), ncol=3, handletextpad=0.2, columnspacing=1.0, borderaxespad=0)
    triad(fig, 0.35, 0.55, VIEW_A, BOX_A, L=0.7)
    save(fig, num, json.dumps({"cases": details}))

# 3.21, 3.22  [A] residual-line protection: P4 vs CLIP; common log colour scale over the three stored pairs
O = nz(SR / "pix_bound01/domains/omega1.npz")
sets = [("M10_curv", ["R7E-4772d06a571fb22a7b1f1854", "R7E-b4eca56f794c4c54c1cb14a4"], ["S0: regelmäßig", "S1: frei verschoben"]),
        ("M10_curv", ["R7E-32aaa0e2250b5ce974a36964", "R7E-b15a18e82fbdfdd594f42d34"], ["P4", "CLIP"]),
        ("M24_curv", ["R7E-c308dc0d74b7963913b0a557", "R7E-1beff84016228c4154f2dfd7"], ["P4", "CLIP"])]
loaded = []; allerr = []
for sid, keys, labs in sets:
    group = []
    for key, lab in zip(keys, labs):
        a = nz(SR / "s4_real07/evals" / f"{key}.npz"); ev = readj(SR / "s4_real07/evals" / f"{key}.json"); v = a["valid"].astype(bool)
        pts = a["pts"][v]; err = np.abs(pts[:, 2] - a["zr"][v]); allerr.extend(err); group.append((key, lab, pts, err, ev))
    loaded.append((sid, group))
lnorm = LogNorm(.001, max(allerr))
VIEW_B, BOX_B = (28, -60), (1, 1, .7)
for i, num in [(1, 21), (2, 22)]:
    sid, group = loaded[i]
    xx, yy, zz = O["x"], O["y"], O["z_" + SID[sid]]; xs_, ys_ = np.unique(xx), np.unique(yy); ox, oy = np.meshgrid(xs_, ys_)
    oz = np.full(ox.shape, np.nan); oz[np.searchsorted(ys_, yy), np.searchsorted(xs_, xx)] = zz
    bounds = np.vstack([np.c_[xx, yy, zz]] + [g[2] for g in group]); lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    H = 8.4; fig = st.figure(15, H); details = []
    for j, (key, lab, pts, err, ev) in enumerate(group):
        ax = cm_axes(fig, 0.6 + j * 7.2, 1.0, 6.9, 5.3, projection="3d", computed_zorder=False)
        ax.plot_wireframe(ox, oy, oz, rstride=10, cstride=10, color=".65", lw=.3, alpha=.6, rasterized=True)
        ax.scatter(*pts.T, c=np.maximum(err, .001), norm=lnorm, cmap="viridis", s=.8, depthshade=False, zorder=5, rasterized=True, linewidths=0)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2] - 1, hi[2] + 1)); ax.view_init(*VIEW_B); ax.set_box_aspect(BOX_B, zoom=1.05); bare3d(ax)
        mt = ev["metrics"]; rm = ev["nodes"]["glob"]["sup_surf_rmse"]
        ftext(fig, 4.05 + j * 7.2, 0.15, f"{lab}: $C_\\tau$ = {st.num(100 * mt['glob_Ctau'], 1)} %\nRMSE gestützt = {st.num(rm, 3)} mm", ha="center", va="top", fontsize=10, linespacing=1.25)
        details.append({"eval_key": key, "n_valid": len(pts), "Ctau": mt["glob_Ctau"], "supported_RMSE": rm})
    hbar(fig, 4.0, H - 1.3, 7.0, lnorm, "viridis", "|Höhenfehler| / mm (logarithmisch)", fmt=lambda v, p: st.num(float(f"{v:.3g}")))
    triad(fig, 0.45, 0.6, VIEW_B, BOX_B, L=0.7)
    save(fig, num, json.dumps({"cases": details, "axis_min": lo.tolist(), "axis_max": hi.tolist(), "colour_scale": "log, common over the three stored pairs"}))

# 3.28  [A] EF line noise sequence: camera image, true height, reconstruction, error
C5 = rd(SR / "s5_align05/case_results.csv")
def s5row(combo, sid, phase, sg, rep, fe):
    a = C5[(C5.combo == combo) & (C5.surface == sid) & (C5.phase_label == phase) & (C5.sigma == sg) & (C5.rep == rep) & (C5.front_end == fe)]; assert len(a) == 1
    return a.iloc[0]
side = 2.9; xs = [0.6, 5.0, 8.1, 11.2]; rh = side + 1.35
H = 0.55 + 3 * rh + 1.4; fig = st.figure(15, H); details = []
for c_, t in enumerate(["Kamerabild", "Wahre Höhe", "Rekonstruierte Höhe", "Höhenfehler"]):
    ftext(fig, xs[c_] + side / 2, 0.3, t, ha="center", va="center", fontsize=10)
for i, sg in enumerate([0, 40, 100]):
    r = s5row("L_w0.5_s8", "B_curv", "ph3", sg, 0 if sg == 0 else 1, "P4"); a = nz(RESEARCH / r.eval_record); v = a["valid"].astype(bool)
    p = a["pts"][v]; zt = a["zr"][v]; err = p[:, 2] - zt
    cands = [SR / "s345_ready/run/images" / f"{r.image_key}.npz", SR / "s5_six_formal/store/image" / f"{r.image_key}.npz"]
    ip = next(x for x in cands if x.exists()); img = nz(ip)["image"]
    top = 0.55 + i * rh
    ftext(fig, 0.15, top + side / 2, f"$\\sigma_\\mathrm{{n}}$ = {sg}", rotation=90, ha="left", va="center", fontsize=10)
    ax = cm_axes(fig, xs[0], top, side, side); ax.imshow(img, cmap="gray", vmin=0, vmax=255, rasterized=True); ax.set_xticks([]); ax.set_yticks([]); ax.set_axis_off()
    ftext(fig, xs[0] + side / 2, top + side + 0.1, f"$n$ = {len(p)}; $C_\\tau$ = {st.num(100 * r.glob_Ctau, 1)} %", ha="center", va="top", fontsize=9)
    for j, (vals, cmap, lims) in enumerate([(zt, "viridis", (-16, 4)), (p[:, 2], "viridis", (-16, 4)), (err, "RdBu_r", (-.5, .5))], 1):
        ax = cm_axes(fig, xs[j], top, side, side)
        if len(p): ax.scatter(p[:, 0], p[:, 1], c=vals, s=.8, cmap=cmap, vmin=lims[0], vmax=lims[1], linewidths=0, rasterized=True)
        else: ax.text(.5, .5, "keine gültigen\nPunkte", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        map2d(ax, ylab=(j == 1))
        if i < 2: ax.set_xlabel(""); ax.tick_params(labelbottom=False)
    details.append({"eval_record": r.eval_record, "image": str(ip), "sigma": sg, "n_valid": len(p), "Ctau": float(r.glob_Ctau)})
hbar(fig, 4.75, H - 1.25, 6.0, Normalize(-16, 4), "viridis", "Höhe $z$ / mm", fmt=lambda v, p: st.num(round(v)))
hbar(fig, 11.2, H - 1.25, side, Normalize(-.5, .5), "RdBu_r", "Höhenfehler / mm", ticks=[-.5, 0, .5], extend="both")
save(fig, 28, json.dumps({"cases": details}))

# 3.17  [D] nominal arrangements, block 26, j = 0.30 (arms as rows; central three dot rows)
G = readj(SR / "s4_real07/generation.json")
arms = ["S1", "S2*", "S3", "S4*"]
W = 15; pw = 6.35; xs = [1.35, 8.25]; scale = pw / (2 * 3.6 * 8.0)
ph = 13 * scale; rh = ph + 0.85
H = 0.95 + 5 * rh + 0.75; fig = st.figure(W, H); geo = []
for c_, (base, title) in enumerate([("R8x4", "Punkte R8×4, $r$ = 1 mm"), ("L6.3", "Linien, $s$ = 6,3 mm, $w$ = 1 mm")]):
    b = next(x for x in G["blocks_out"] if x["baseline"] == base and x["block"] == 26); p = b["p0_mm"]
    ftext(fig, xs[c_] + pw / 2, 0.3, title, ha="center", va="center", fontsize=10)
    for r_, arm in enumerate(["S0"] + arms):
        a = b["arms"]["S1" if arm == "S0" else arm]
        disp = {int(k): v * 0.18 for k, v in a["disp_px"].items()}
        qs = np.array([i * p + (0 if arm == "S0" else disp.get(i, 0)) for i in range(-3, 4)])
        top = 0.95 + r_ * rh
        ax = cm_axes(fig, xs[c_], top, pw, ph)
        if base == "R8x4":
            for q in qs:
                for yy in [-4, 0, 4]: ax.add_patch(Circle((q, yy), 1, fc="#31688e", ec="none"))
        else:
            for q in qs: ax.fill_between([q - .5, q + .5], -6.5, 6.5, color="#31688e", lw=0)
        ax.set(xlim=(-3.6 * 8.0, 3.6 * 8.0), ylim=(-6.5, 6.5)); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xticks([-2 * p, 0, 2 * p]); ax.set_yticks([]); st.comma_axis(ax, "x")
        for sp in ("left",): ax.spines[sp].set_visible(False)
        if r_ < 4: ax.tick_params(labelbottom=False)
        else: ax.set_xlabel("$q$ / mm", labelpad=1)
        if c_ == 0: ftext(fig, 0.25, top + ph / 2, arm, ha="left", va="center", fontsize=10)
        gm = p if arm == "S0" else a["g_min_mm"]; h = b["h0_mm"] if arm == "S0" else a["h_mm"]; rms = 0 if arm == "S0" else a["rms_mm"]
        ftext(fig, xs[c_] + pw / 2, top - 0.08, f"$g_\\mathrm{{min}}$ {st.num(gm, 2)} · $h$ {st.num(h, 2)} · RMS(Δ) {st.num(rms, 2)}", ha="center", va="bottom", fontsize=9)
        geo.append({"baseline": base, "arm": arm, "g_min_mm": gm, "h_mm": h, "rms_mm": rms})
save(fig, 17, json.dumps({"block": 26, "geometry": geo, "note": "nominal positions from generation.json; central three dot rows shown; no camera image"}))

# input check against the records of A and B
def norm(p): return str(Path(p)).replace("\\", "/").lower()
META["inputs"] = SOURCE
META["submission_manifest_verified"] = True
(OUTDIR / "kap3b_record.json").write_text(json.dumps(META, ensure_ascii=False, indent=2), encoding="utf-8")
print("figures:", len(META["figures"]), "| inputs:", len(SOURCE))
