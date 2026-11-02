# -*- coding: utf-8 -*-

from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thesis_figstyle as st
from plot_io import configure, verify_inputs, selected
DATA_ROOT, FIGURE_OUT = configure('kap4_redraw')
verify_inputs()

ROOT = DATA_ROOT
SRC = ROOT / "outputs/ch4_valid02"
OLD = ROOT / "outputs/ch4_bridge01/analysis"
OUT = FIGURE_OUT; OUT.mkdir(parents=True, exist_ok=True)
st.setup()

def sha(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

# Check the declared numerical inputs.
extra = [SRC / "analysis/surfaces.csv.gz",
         ROOT / "research/project_Part1_v1.7/outputs/surface_revision/s1_real01/domain/global_domain_step0p50.npz"]
META = {"submission_manifest_verified": True, "extra_inputs": {str(p): sha(p) for p in extra},
        "figures": {}, "qa": []}

R = pd.read_csv(SRC / "results.csv.gz"); S = pd.read_csv(SRC / "analysis/surfaces.csv.gz"); SC = pd.read_csv(SRC / "analysis/config_scores.csv.gz")
CH = pd.read_csv(SRC / "analysis/choices_new.csv"); KE = pd.read_csv(SRC / "key_effects.csv"); VT = pd.read_csv(SRC / "validation_table.csv")
RO = pd.read_csv(OLD / "results_all.csv.gz"); SO = pd.read_csv(OLD / "config_scores.csv.gz"); CO = pd.read_csv(OLD / "choices_all.csv")
ENV = ["NOMINAL", "BLUR1_5", "BLUR2", "BLUR2_5", "BLUR3", "UNIFORM", "G05", "GRAD_X", "GRAD_Y", "GRAD_Y_REV"]
DE = ["Nominal", "PSF 1,5 px", "PSF 2 px", "PSF 2,5 px", "PSF 3 px", "$g$ = 0,75", "$g$ = 0,50", "Gradient $x$", "Gradient $y$", "Gradient −$y$"]
EN = dict(zip(ENV, DE)); SIG = [0, 20, 40, 60, 80, 100]
DOT = ["D_r1_p8x2.5", "D_r1_p8x3", "D_r1_p8x4", "D_r1_p8x6", "D_r1_p8x8", "D_r1.5_p8x4", "D_r1.5_p8x6", "D_r0.75_p8x4", "D_r0.75_p8x6", "D_r1_T8x4"]
LINE = ["L_w0.5_s8", "L_w0.75_s8", "L_w1_s8", "L_w1.25_s8", "L_w1.5_s8", "L_w0.5_s10", "L_w1_s10"]
SIGLAB = r"$\sigma_\mathrm{n}$ / Graustufen"

def lab(c):
    """Short label as in the text: r1 T8×4, r0,75 R8×6, w0,5 s8."""
    if c.startswith("D_"):
        r, lay = c[2:].split("_")
        lay = lay.replace("p", "R").replace("x", "×").replace(".", ",")
        return f"$r$ {st.num(float(r[1:]))} {lay}"
    w, s = c[3:].split("_s")
    return f"$w$ {st.num(float(w))} / $a_L$ {s}"

def leglab(c):
    """Legend entry without the parameter names (given in the legend title): '1 R8×2,5', '0,5 / 8'."""
    if c.startswith("D_"):
        r, lay = c[2:].split("_")
        return f"{st.num(float(r[1:]))} " + lay.replace("p", "R").replace("x", "×").replace(".", ",")
    w, s = c[3:].split("_s")
    return f"{st.num(float(w))} / {s}"

def main(t): return t[(t.d_mm == 6) & (t.tau_mm == .15)]
MC = main(CH); MC = MC[MC.floor == .5]; MS = main(SC).set_index(["config", "env", "sigma"])

def save(fig, num, notes=""):
    name = f"Abb_4_{num:02d}"
    if not selected(name):
        plt.close(fig)
        return
    st.save(fig, OUT / name, name, META["qa"])
    META["figures"][name] = {"notes": notes, "pdf_sha256": sha(OUT / (name + ".pdf"))}

def xsig(ax, labels=True):
    ax.set_xticks(SIG); ax.set_xlim(-5, 105)
    if not labels:
        ax.tick_params(labelbottom=False)

def cm_axes(fig, x, top, w, h, **kw):
    """Axes placed in cm, measured from the upper left corner of the canvas."""
    W, H = fig.get_size_inches() / st.CM
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H], **kw)

# 4.1 selection map: two matrices stacked (fits one page with caption)
fig = st.figure(15, 17.9)
axs = [cm_axes(fig, 3.525, 0.65, 11.25, 7.4), cm_axes(fig, 3.525, 9.25, 11.25, 7.4)]
for ax, fam, title in zip(axs, ["point", "line"], ["Punktmuster", "Linienmuster"]):
    for iy, e in enumerate(ENV):
        for ix, z in enumerate(SIG):
            row = MC[(MC.family == fam) & (MC.env == e) & (MC.sigma == z)].iloc[0]; ok = pd.notna(row.preferred)
            ax.add_patch(plt.Rectangle((ix - .5, iy - .5), 1, 1, facecolor="#d0e7da" if ok else "#e3e3df", edgecolor="white", lw=1.5))
            if ok:
                t = lab(row.preferred).replace(" / ", "  ")
                t = t + "\n" + st.num(row.pref_score * 100, 1) + " %" + (" *" if row.pref_score < .52 else "")
            else:
                t = "—"
            ax.text(ix, iy, t, ha="center", va="center", fontsize=9, linespacing=1.05)
    ax.set(xlim=(-.5, 5.5), ylim=(9.5, -.5)); ax.grid(False)
    ax.set_xticks(range(6), SIG); ax.set_yticks(range(10), DE); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, fontsize=11, pad=4)
axs[1].set_xlabel(SIGLAB)
save(fig, 1, "selected representative and its own J; '-' no admissible candidate; '*' < 2 pp above the 50 % floor")

# curves
PH = R.groupby(["config", "env", "sigma", "surface", "phase_block"]).glob_Ctau.mean().rename("ct").reset_index()
AVG = PH.groupby(["config", "env", "sigma", "surface"]).ct.mean().reset_index()
WORST = AVG.sort_values(["ct", "surface"]).drop_duplicates(["config", "env", "sigma"])
BANDS = PH.merge(WORST[["config", "env", "sigma", "surface"]], on=["config", "env", "sigma", "surface"], validate="many_to_one").groupby(["config", "env", "sigma"]).ct.agg(lo="min", hi="max", center="mean")
chk = main(SC).merge(BANDS.reset_index(), on=["config", "env", "sigma"], validate="one_to_one")
assert np.allclose(chk.J, chk.center, atol=1e-12, rtol=0) and (chk.lo <= chk.J + 1e-12).all() and (chk.J <= chk.hi + 1e-12).all()

def curves(num, cfgs, envs):
    n = len(envs)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(cfgs))) if cfgs == DOT else plt.get_cmap("Blues")(np.linspace(.35, .95, 5)).tolist() + ["#d46c24", "#9a3d15"]
    H = 10.6 if n == 5 else 12.4
    fig = st.figure(15, H)
    gs = fig.add_gridspec(2, 3, left=0.09, right=0.985, top=0.95, bottom=0.11 if n == 5 else 0.25, wspace=0.12, hspace=0.32)
    axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    for k, (ax, e) in enumerate(zip(axs, envs)):
        for i, cfg in enumerate(cfgs):
            q = MS.loc[(cfg, e)].reindex(SIG); b = BANDS.loc[(cfg, e)].reindex(SIG)
            ax.plot(SIG, 100 * q.J, color=colors[i], lw=1.2, marker="o", ms=2.5, ls="--" if cfg.endswith("_s10") else "-", label=leglab(cfg))
            ax.fill_between(SIG, 100 * b.lo, 100 * b.hi, color=colors[i], alpha=.07, lw=0)
        ax.axhline(50, color="#555555", lw=.7, ls=":"); ax.axhline(70, color="#999999", lw=.7, ls=":")
        ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100]); xsig(ax, labels=(k >= 3 or (n == 5 and k == 2)))
        if k % 3: ax.tick_params(labelleft=False)
        ax.set_title(EN[e], pad=3); st.comma_axis(ax)
    fig.text(0.012, 0.53 if n == 5 else 0.6, "$J$ / %", rotation=90, va="center", fontsize=11)
    fig.text(0.54, 0.015 if n == 5 else 0.155, SIGLAB, ha="center", fontsize=11)
    h, l = axs[0].get_legend_handles_labels()
    title = "$r$ / mm, Anordnung" if cfgs == DOT else "$w$ / mm, $a_L$ / mm"
    if n == 5:
        axs[5].axis("off"); axs[5].legend(h, l, loc="center", ncol=2 if cfgs == DOT else 1, title=title, handlelength=1.2, columnspacing=0.7, handletextpad=0.4, borderaxespad=0)
    else:
        fig.legend(h, l, loc="lower center", ncol=5 if cfgs == DOT else 4, title=title, handlelength=1.2, columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.54, 0.0))
    save(fig, num, "J = min of surface phase means; band = four phase means on the fixed worst surface; dotted 50/70 %")
curves(2, DOT, ENV[:5]); curves(7, LINE, ENV[:5])   # thesis numbering: the heat map moved to its first mention and became 4.6
curves(9, DOT, [ENV[i] for i in [0, 5, 6, 7, 8, 9]]); curves(10, LINE, [ENV[i] for i in [0, 5, 6, 7, 8, 9]])

# 4.3 sampling vs height substitution (13 x 7,5 cm)
mechanism = OLD / "independent_check/mechanism_summary.csv"; D = pd.read_csv(mechanism); META["extra_inputs"][str(mechanism)] = sha(mechanism)
cfgs = ["D_r1_p8x4", "D_r1_p8x6", "D_r1_T8x4", "L_w0.5_s8"]; dd = D[(D.env == "NOMINAL") & (D.sigma == 0)].set_index("config").loc[cfgs]
fig = st.figure(13, 7.8); ax = fig.add_axes([0.13, 0.30, 0.85, 0.66]); xx = np.arange(4)
for off, col, cl, tx in [(-.26, "Ctau", "#376d9c", "Gemessene Höhen"), (0, "C_ideal", "#cc8a3d", "Wahre Höhen an denselben Positionen"), (.26, "Cvalid", "#8c9a99", "Stützabdeckung")]:
    ax.bar(xx + off, 100 * dd[col], width=.25, color=cl, label=tx)
ax.set(xticks=xx, ylim=(0, 100)); ax.set_xticklabels([lab(c) for c in cfgs]); ax.set_ylabel("Flächenanteil / %"); st.comma_axis(ax, "y")
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, handlelength=1.0, columnspacing=1.0, bbox_to_anchor=(0.55, 0.0))
save(fig, 3, "three-surface and four-phase means (CH4-BRIDGE-01 mechanism_summary)")

# point clouds
domain = ROOT / "research/project_Part1_v1.7/outputs/surface_revision/s1_real01/domain/global_domain_step0p50.npz"
with np.load(domain, allow_pickle=False) as z: DOM = {k: z[k] for k in z.files}
def getpack(row):
    p = SRC / "main/data" / (hashlib.sha256(row.case_key.encode()).hexdigest()[:24] + ".npz")
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files if k != "_meta"}

BOX = (1, 1, 0.45)   # displayed box: z (20 mm) about twice exaggerated against x, y (92 mm); stated in the captions
CASE_NPZ = {}

def cloud(ax, row, a, metric="ctau"):
    pref = row.method + "__ev__"; v = a[pref + "valid"] & a[pref + "in_global"]; p = a[pref + "pts"][v]; err = p[:, 2] - a[pref + "zr"][v]
    sid = {"B_curv": "B_curv", "M10_curv": "A_curv", "M24_curv": "C_curv"}[row.surface]
    lx, ly = DOM["lattice_x"], DOM["lattice_y"]; Z = np.full((len(ly), len(lx)), np.nan); Z[np.searchsorted(ly, DOM["y"]), np.searchsorted(lx, DOM["x"])] = DOM["z_" + sid]; X, Y = np.meshgrid(lx, ly)
    ax.plot_wireframe(X, Y, Z, rstride=14, cstride=14, color="#d2d3cf", lw=.25, rasterized=True)
    if len(p): ax.scatter(*p.T, c=err, cmap="coolwarm", vmin=-.3, vmax=.3, s=2.6 if row.family == "point" else .4, depthshade=False, rasterized=True, linewidths=0)
    ax.set(xlim=(-46, 46), ylim=(-46, 46), zlim=(-16, 4)); ax.view_init(28, -62)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.tick_params(length=0, pad=0)
    ax.grid(False); ax.set_box_aspect(BOX, zoom=1.22)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#dddddd")
    txt = (r"$C_\tau$ = " + st.num(row.glob_Ctau * 100, 1) + " %") if metric == "ctau" else f"$n$ = {int(v.sum())}"
    return txt

def triad(fig, ox, oy, L=0.85):
    """One x/y/z direction marker per figure: 2-D arrows along the projected axes of the point-cloud view
    (same view and box, so the short z arrow shows the display scale). Origin (ox, oy) in cm from lower left."""
    tmp = fig.add_axes([0, 0, .3, .3], projection="3d")
    tmp.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1)); tmp.view_init(28, -62); tmp.set_box_aspect(BOX)
    fig.canvas.draw(); M = tmp.get_proj()
    P = np.array([tmp.transData.transform(proj3d.proj_transform(*v, M)[:2]) for v in [np.zeros(3), *np.eye(3)]])
    tmp.remove()
    V = P[1:] - P[0]; V = V / np.linalg.norm(V[:2], axis=1).max()   # x, y scaled to <= L, z keeps its relative length
    Wpx, Hpx = fig.bbox.width, fig.bbox.height; px = st.CM * fig.dpi   # pixels per cm
    o = np.array([ox, oy]) * px
    for v, l in zip(V, "xyz"):
        tip = o + v * L * px
        fig.add_artist(FancyArrowPatch(o / (Wpx, Hpx), tip / (Wpx, Hpx), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6, lw=.8, color="#444444"))
        lp = tip + v / np.linalg.norm(v) * 0.22 * px
        fig.text(lp[0] / Wpx, lp[1] / Hpx, f"${l}$", fontsize=9, ha="center", va="center")

def getpack_logged(row):
    p = SRC / "main/data" / (hashlib.sha256(row.case_key.encode()).hexdigest()[:24] + ".npz")
    CASE_NPZ[row.case_key] = sha(p)
    return getpack(row)

def board(num, cfgs, conditions, rowlab, row_h, notes, W=15):
    """Point-cloud grid: columns = configurations, rows = conditions; sizes in cm."""
    nc, nr = len(cfgs), len(conditions)
    left, right, top, bottom = 1.55, 0.4, 0.7, 1.35
    H = top + nr * row_h + bottom
    fig = st.figure(W, H); cw = (W - left - right) / nc
    selected = []
    for i, cond in enumerate(conditions):
        env, sig, sf = cond
        y = top + i * row_h
        for j, cfg in enumerate(cfgs):
            row = R[(R.config == cfg) & (R.env == env) & (R.sigma == sig) & (R.surface == sf) & (R.phase_block == "V1") & (R.rep == (0 if sig == 0 else 1))].iloc[0]
            ax = cm_axes(fig, left + j * cw, y, cw, row_h - 0.38, projection="3d")
            txt = cloud(ax, row, getpack_logged(row)); selected.append(row.case_key)
            fig.text((left + (j + .5) * cw) / W, 1 - (y + row_h - 0.24) / H, txt, ha="center", va="center", fontsize=9, zorder=10)
            if i == 0:
                fig.text((left + (j + .5) * cw) / W, 1 - (top - 0.12) / H, lab(cfg), ha="center", va="bottom", fontsize=10)
        fig.text(0.15 / W, 1 - (y + row_h / 2) / H, rowlab(cond), ha="left", va="center", fontsize=10)
    triad(fig, 1.2, 0.45)
    cax = cm_axes(fig, 4.5, H - 0.85, 6.0, 0.22)
    bar = fig.colorbar(ScalarMappable(norm=Normalize(-.3, .3), cmap="coolwarm"), cax=cax, orientation="horizontal", extend="both")
    bar.set_label("Höhenfehler / mm", labelpad=2); bar.ax.xaxis.set_label_position("top"); bar.set_ticks([-.3, -.15, 0, .15, .3]); bar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: st.num(round(x, 2))))
    save(fig, num, notes)
    if f"Abb_4_{num:02d}" in META["figures"]:
        META["figures"][f"Abb_4_{num:02d}"]["cases"] = selected

SURF = lambda s: s.split("_")[0]
board(4, ["D_r1_T8x4", "D_r1_p8x4", "D_r1_p8x6", "L_w0.5_s8"], [("NOMINAL", 20, s) for s in ["M10_curv", "M24_curv"]],
      lambda c: SURF(c[2]), 2.6, "fixed V1/rep1; all valid global points; gray wireframe = true surface")
board(5, ["D_r0.75_p8x4", "D_r0.75_p8x6", "D_r1_p8x6"], [("NOMINAL", sig, s) for sig in [60, 80] for s in ["M10_curv", "M24_curv"]],
      lambda c: r"$\sigma_\mathrm{n}$ = " + f"{c[1]}\n{SURF(c[2])}", 2.65, "fixed V1/rep1; all valid global points")
board(8, ["L_w0.5_s8", "L_w1_s8", "L_w1.25_s8", "L_w1.5_s8"], [(env, 20, s) for env in ["BLUR2_5", "BLUR3"] for s in ["M10_curv", "M24_curv"]],
      lambda c: EN[c[0]].replace(" px", " px") + f"\n{SURF(c[2])}", 2.45, "fixed V1/rep1; all valid global points")

# 4.6  four-cell interactions: three maps stacked (status marker kept)
fig = st.figure(15, 17.0); mark_count = 0
hs = [0.6, 5.35, 10.1]   # map tops in cm, each map 4.2 cm high (10 rows)
for ax_y, name, title in zip(hs, ["dot_r1_r0.75", "dot_r1_r1.5", "line_w0.5_w1"],
                             [r"$r$ 1 → 0,75; R8×4 → R8×6", r"$r$ 1 → 1,5; R8×4 → R8×6", r"$w$ 0,5 → 1; $a_L$ 8 → 10"]):
    ax = cm_axes(fig, 3.525, ax_y, 11.25, 4.2)
    t = KE[(KE.kind == "fourcell") & (KE.name == name)].set_index(["env", "sigma"]); arr = np.array([[100 * t.loc[(e, z)].effect for z in SIG] for e in ENV])
    ax.imshow(arr, cmap="RdBu", vmin=-25, vmax=25, aspect="auto")
    cells = json.loads(t.iloc[0].cells)
    for i, e in enumerate(ENV):
        for j, z in enumerate(SIG):
            allok = all(bool(MS.loc[(cfg, e, z)].safe and MS.loc[(cfg, e, z)].J >= .5) for cfg in cells.values())
            if allok: mark_count += 1
            v = arr[i, j]; s = ("+" if v >= 0 else "−") + st.num(abs(v), 1)
            ax.text(j, i, s + (" ●" if allok else ""), ha="center", va="center", fontsize=9, color="white" if abs(v) > 18 else "#111111")
    ax.set(xticks=range(6), yticks=range(10)); ax.set_xticklabels(SIG); ax.set_yticklabels(DE); ax.grid(False); ax.tick_params(length=0)
    if ax_y != hs[-1]: ax.tick_params(labelbottom=False)
    ax.set_title(title, fontsize=10, pad=3)
fig.text(0.61, 1 - 15.05 / 17.0, SIGLAB, ha="center", va="center", fontsize=11)
cax = cm_axes(fig, 5.7, 16.2, 6.75, 0.2); bar = fig.colorbar(ScalarMappable(norm=Normalize(-25, 25), cmap="RdBu"), cax=cax, orientation="horizontal", extend="both")
bar.set_label("Interaktion / Prozentpunkte", labelpad=4); bar.ax.xaxis.set_label_position("top")
bar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: st.num(round(x))))
save(fig, 6, "four-candidate qualification marker kept")
if "Abb_4_06" in META["figures"]:
    META["figures"]["Abb_4_06"]["marked_cells"] = mark_count

# 4.11 loss decomposition
envs = [ENV[i] for i in [0, 5, 6, 7, 8, 9]]
fig = st.figure(15, 9.4); axs = [fig.add_axes([0.10, 0.415, 0.42, 0.52]), fig.add_axes([0.565, 0.415, 0.42, 0.52])]
for k, (ax, cfg) in enumerate(zip(axs, ["D_r1_p8x6", "L_w0.5_s8"])):
    q = main(S); q = q[(q.config == cfg) & (q.sigma == 40)].groupby("env")[["Ctau", "Cvalid"]].mean().reindex(envs)
    vals = [q.Ctau, q.Cvalid - q.Ctau, 1 - q.Cvalid]; bottom = np.zeros(len(q))
    for v, col, label in zip(vals, ["#459078", "#d9ac61", "#c7cccd"], ["Genauigkeitsabdeckung", "Gestützt, aber über Toleranz", "Ohne gültige Stützung"]):
        ax.bar(range(6), 100 * v, bottom=100 * bottom, color=col, label=label); bottom += v.to_numpy()
    ax.set(xticks=range(6), ylim=(0, 100)); ax.set_xticklabels([EN[e] for e in envs], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_title(lab(cfg), pad=3); st.comma_axis(ax, "y")
    if k: ax.tick_params(labelleft=False)
axs[0].set_ylabel("Flächenanteil / %")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, handlelength=1.0, columnspacing=1.2, bbox_to_anchor=(0.54, 0.0))
save(fig, 11, "three-surface equal means; common 30947-node denominator; sigma_n = 40")

# 4.12 gradients: camera image, reconstruction, error map
fig = st.figure(15, 11.2); chosen = []
W, H = 15, 11.2
for i, e in enumerate(["GRAD_Y", "GRAD_Y_REV"]):
    row = R[(R.config == "L_w0.5_s8") & (R.env == e) & (R.sigma == 40) & (R.surface == "M24_curv") & (R.phase_block == "V1") & (R.rep == 1)].iloc[0]; a = getpack_logged(row); chosen.append(row.case_key)
    y0 = 1 - (0.7 + (i + 1) * 4.25) / H
    ax = fig.add_axes([1.55 / W, y0, 3.9 / W, 3.9 / H]); ax.imshow(a["image"], cmap="gray", vmin=0, vmax=255, rasterized=True); ax.set_axis_off()
    ax = fig.add_axes([5.6 / W, y0 + 0.35 / H, 4.3 / W, 3.6 / H], projection="3d"); txt = cloud(ax, row, a, metric="n")
    fig.text(7.75 / W, y0 + 0.15 / H, txt, ha="center", va="center", fontsize=9, zorder=10)
    px = row.method + "__ev__"; v = a[px + "valid"] & a[px + "in_global"]; pts = a[px + "pts"][v]; err = pts[:, 2] - a[px + "zr"][v]
    ax = fig.add_axes([11.75 / W, y0 + 0.45 / H, 3.15 / W, 3.4 / H]); ax.scatter(pts[:, 0], pts[:, 1], c=err, vmin=-.3, vmax=.3, cmap="coolwarm", s=1.2, linewidths=0, rasterized=True)
    ax.set(xlim=(-46, 46), ylim=(-46, 46), xticks=[-40, 0, 40], yticks=[-40, 0, 40]); ax.set_aspect("equal"); st.comma_axis(ax)
    ax.set_ylabel("$y$ / mm", labelpad=0)
    if i == 1: ax.set_xlabel("$x$ / mm", labelpad=1)
    else: ax.tick_params(labelbottom=False)
    fig.text(0.75 / W, y0 + 1.95 / H, EN[e], ha="center", va="center", rotation=90, fontsize=10)
for x, t in [(3.5, "Kamerabild"), (7.75, "Rekonstruktion"), (13.3, "Punkthöhenfehler")]:
    fig.text(x / W, 1 - 0.45 / H, t, ha="center", va="center", fontsize=10)
triad(fig, 1.2, 0.45)
cax = fig.add_axes([0.32, 0.8 / H, 0.4, 0.2 / H]); bar = fig.colorbar(ScalarMappable(norm=Normalize(-.3, .3), cmap="coolwarm"), cax=cax, orientation="horizontal", extend="both")
bar.set_label("Höhenfehler / mm", labelpad=2); bar.ax.xaxis.set_label_position("top"); bar.set_ticks([-.3, -.15, 0, .15, .3]); bar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: st.num(round(x, 2))))
save(fig, 12, "fixed V1/rep1, all global valid points; camera image 0-255")
if "Abb_4_12" in META["figures"]:
    META["figures"]["Abb_4_12"]["cases"] = chosen

# 4.13 tolerance and family
fig = st.figure(15, 8.3); axs = [fig.add_axes([0.10, 0.385, 0.41, 0.53]), fig.add_axes([0.575, 0.385, 0.41, 0.53])]
for k, (ax, sig) in enumerate(zip(axs, [0, 20])):
    for fam, col in [("point", "#cc792e"), ("line", "#397ba6")]:
        for t, style in [(CO, ":"), (CH, "-")]:
            q = t[(t.family == fam) & (t.env == "NOMINAL") & (t.sigma == sig) & (t.d_mm == 6) & (t.floor == .5)].sort_values("tau_mm")
            ax.plot(q.tau_mm, 100 * q.best_score, style, color=col, marker="o" if style == "-" else None, ms=3)
    ax.set(xticks=[.1, .15, .2, .3], ylim=(50, 100)); ax.set_title("Nominal, " + r"$\sigma_\mathrm{n}$" + f" = {sig}", pad=3); st.comma_axis(ax)
    ax.set_xlabel(r"Höhentoleranz $\tau$ / mm")
    if k: ax.tick_params(labelleft=False)
axs[0].set_ylabel("Höchstes zulässiges $J$ / %")
handles = [Line2D([], [], color="#cc792e", lw=1.2), Line2D([], [], color="#397ba6", lw=1.2), Line2D([], [], color="#555", lw=1.2, marker="o", ms=3), Line2D([], [], color="#555", lw=1.2, ls=":")]
fig.legend(handles, ["Punktmuster", "Linienmuster", "Unabhängige Phasen", "Parametersweep"], loc="lower center", ncol=2, handlelength=1.6, columnspacing=1.5, bbox_to_anchor=(0.54, 0.0))
save(fig, 13, "solid: independent phases; dotted: parameter sweep; d = 6 mm; floor 50 %")

# 4.14 validation of original recommendations
fig = st.figure(15, 8.4); axs = [fig.add_axes([0.11, 0.27, 0.40, 0.66]), fig.add_axes([0.575, 0.27, 0.40, 0.66])]
for k, (ax, fam) in enumerate(zip(axs, ["point", "line"])):
    q = VT[(VT.family == fam) & (VT.d_mm == 6) & (VT.tau_mm == .15) & (VT.floor == .5) & VT.preferred_old.notna()]
    for label, color, text in [("kept", "#39816b", "Weiterhin nahezu optimal"), ("usable_reorder", "#d0903c", "Zulässig, Rangbereich geändert")]:
        v = q[q.label == label]; ax.scatter(100 * v.J_oldpref_old, 100 * v.J_oldpref_new, c=color, s=22, edgecolors="white", linewidths=.4, label=text)
    ax.plot([45, 90], [45, 90], color="#aaa", lw=.8); ax.axhline(50, color="#777", ls=":", lw=.8)
    ax.set(xlim=(45, 90), ylim=(45, 90), xticks=[50, 70, 90], yticks=[50, 70, 90]); ax.set_aspect("equal"); ax.set_title("Punktmuster" if fam == "point" else "Linienmuster", pad=3)
    ax.set_xlabel("$J$ im Parametersweep / %"); st.comma_axis(ax)
    if k: ax.tick_params(labelleft=False)
axs[0].set_ylabel("$J$ in unabh. Phasen / %")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, handlelength=1.0, bbox_to_anchor=(0.54, 0.0))
save(fig, 14, "each point one original recommendation; 49/49 qualified, 45/49 near-optimal")

verify_inputs()
META["case_npz_sha256"] = CASE_NPZ
(OUT / "kap4_record.json").write_text(json.dumps(META, ensure_ascii=False, indent=2), encoding="utf-8")
print("figures:", len(META["figures"]), "| min font:", min(r["min_font_pt"] for r in META["qa"]), "| sizes:", [(r["figure"][-2:], r["width_cm"], r["height_cm"]) for r in META["qa"]])
