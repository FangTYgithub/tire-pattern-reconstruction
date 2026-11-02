# -*- coding: utf-8 -*-

from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thesis_figstyle as st
from plot_io import configure, verify_inputs, selected
DATA_ROOT, FIGURE_OUT = configure('kap2_redraw')
verify_inputs()
st.setup()
OUTDIR = FIGURE_OUT; OUTDIR.mkdir(parents=True, exist_ok=True)
META = {"figures": {}, "qa": []}

ROOT = DATA_ROOT
P = ROOT / "research/project_Part1_v1.7"
SR = P / "outputs/surface_revision"
SOURCE = {}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p): p = Path(p); SOURCE[str(p)] = sha(p); return p
def nz(p):
    with np.load(track(p), allow_pickle=False) as z: return {k: z[k] for k in z.files}

def cm_axes(fig, x, top, w, h, **kw):
    W, H = fig.get_size_inches() / st.CM
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H], **kw)

def ftext(fig, x, top, s, **kw):
    W, H = fig.get_size_inches() / st.CM
    kw.setdefault("fontsize", 10)
    return fig.text(x / W, 1 - top / H, s, **kw)

def save(fig, num, notes=""):
    name = f"Abb_2_{num:02d}"
    if not selected(name):
        plt.close(fig)
        return
    st.save(fig, OUTDIR / name, name, META["qa"])
    META["figures"][name] = {"notes": notes, "pdf_sha256": sha(OUTDIR / (name + ".pdf"))}

def triad(fig, ox, oy, view, box, L=0.8, labels="xyz"):
    """Arrows along the projected axes of a 3-D view (origin in cm from lower left)."""
    tmp = fig.add_axes([0, 0, .3, .3], projection="3d")
    tmp.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1)); tmp.view_init(*view); tmp.set_box_aspect(box)
    fig.canvas.draw(); M = tmp.get_proj()
    Pp = np.array([tmp.transData.transform(proj3d.proj_transform(*v, M)[:2]) for v in [np.zeros(3), *np.eye(3)]])
    tmp.remove()
    V = Pp[1:] - Pp[0]; V = V / np.linalg.norm(V[:2], axis=1).max()
    Wpx, Hpx = fig.bbox.width, fig.bbox.height; px = st.CM * fig.dpi; o = np.array([ox, oy]) * px
    for v, l in zip(V, labels if isinstance(labels, list) else list(labels)):
        tip = o + v * L * px
        fig.add_artist(FancyArrowPatch(o / (Wpx, Hpx), tip / (Wpx, Hpx), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6, lw=.8, color="#444444"))
        lp = tip + v / np.linalg.norm(v) * 0.25 * px
        fig.text(lp[0] / Wpx, lp[1] / Hpx, f"${l}$", fontsize=9, ha="center", va="center")

def bare3d(ax):
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.tick_params(length=0, pad=0); ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): pane.fill = False; pane.set_edgecolor("#dddddd")

def hbar(fig, x, top, w, norm, cmap, label, ticks=None, fmt=None, extend="neither"):
    cax = cm_axes(fig, x, top, w, 0.2)
    b = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend=extend)
    b.set_label(label, labelpad=2)
    if ticks is not None: b.set_ticks(ticks)
    b.ax.xaxis.set_major_formatter(FuncFormatter(fmt or (lambda v, p: st.num(round(v, 2)))))
    return b

# 2.1, 2.2 surfaces ()
items = [("B", "EF", "Fig9_Flat_Default_Dataset.csv", "#2B8072"), ("A", "M10", "1405Fig14M10_Default_Dataset.csv", "#1864A0"),
         ("C", "M24", "Fig13_M24_Default_Dataset.csv", "#D26A26")]
W, H = 15, 11.8
fig = st.figure(W, H); pw, ph = 6.15, 2.45; xs = [1.6, 8.6]
for c_, t in enumerate(["Gesamte Kontur", "Zentralausschnitt"]):
    ftext(fig, xs[c_] + pw / 2, 1.15, t, ha="center", va="center")
for row, (sid, label, csv, color) in enumerate(items):
    raw = np.loadtxt(track(P / "data/raw" / csv), delimiter=",")
    z = nz(SR / f"surf01/surfaces/{sid}_curv_fine.npz")
    ix = np.flatnonzero(z["X_p"][0] == 0).item(); yy = z["Y_p"][:, ix].copy(); hh = z["H_src"][:, ix].copy()
    assert np.array_equal(hh, z["H"][:, ix])
    top = 1.45 + row * (ph + 0.75)
    ftext(fig, 0.15, top + ph / 2, label, rotation=90, ha="left", va="center")
    for col, domain in enumerate([(-125, 125), (-50, 50)]):
        ax = cm_axes(fig, xs[col], top, pw, ph)
        ax.scatter(raw[:, 0], raw[:, 1], s=2.5, color="#9CA7B3", alpha=.9, zorder=2, linewidths=0)
        ax.plot(yy, hh, lw=1.1, color=color, zorder=3); ax.set_xlim(*domain)
        if col == 0: ax.set_ylim(10, 60); ax.set_yticks([10, 30, 50])
        else:
            mask = (raw[:, 0] >= -50) & (raw[:, 0] <= 50); hm = (yy >= -50) & (yy <= 50)
            lo = min(raw[mask, 1].min(), hh[hm].min()); hi = max(raw[mask, 1].max(), hh[hm].max()); margin = .09 * (hi - lo)
            ax.set_ylim(lo - margin, hi + margin); ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xticks([-100, 0, 100] if col == 0 else [-40, 0, 40]); st.comma_axis(ax)
        if col == 0: ax.set_ylabel("$H$ / mm", labelpad=2)
        if row == 2: ax.set_xlabel("$y_p$ / mm", labelpad=1)
fig.legend(handles=[Line2D([], [], ls="", marker="o", markersize=3.5, color="#9CA7B3", label="Digitalisierte Punkte"),
                    Line2D([], [], color="#33495e", lw=1.3, label="Gespeichertes Referenzprofil")],
           loc="center", bbox_to_anchor=(0.53, 1 - 0.4 / H), ncol=2, borderaxespad=0)
save(fig, 1, "x_p = 0 column of the saved fine surfaces; raw digitised points")

W, H = 15, 6.8; fig = st.figure(W, H); norm = Normalize(10, 70); VIEW2, BOX2 = (28, -58), (200, 250, 60)
for i, (sid, label, csv, color) in enumerate(items):
    z = nz(SR / f"surf01/surfaces/{sid}_curv_coarse.npz"); X, Y, Hh = z["X_p"], z["Y_p"], z["H"]; assert np.count_nonzero(z["H_gauss"]) == 0
    ax = cm_axes(fig, 0.1 + i * 4.95, 0.55, 4.9, 4.0, projection="3d")
    ax.plot_surface(X, Y, Hh, cmap="viridis", norm=norm, rstride=1, cstride=1, linewidth=0, antialiased=False, rasterized=True)
    ax.set_xlim(-100, 100); ax.set_ylim(-125, 125); ax.set_zlim(10, 70); ax.set_box_aspect(BOX2, zoom=1.0); ax.view_init(*VIEW2); bare3d(ax)
    ftext(fig, 2.55 + i * 4.95, 0.35, label, ha="center", va="center")
hbar(fig, 4.75, H - 1.4, 5.5, norm, "viridis", "$H$ / mm", ticks=[10, 30, 50, 70], fmt=lambda v, p: st.num(round(v)))
triad(fig, 0.5, 0.55, VIEW2, BOX2, L=0.75, labels=["x_p", "y_p", "H"])
save(fig, 2, "all saved coarse cells; same view and physical aspect")

# 2.3 geometry sketch (redraw_fig23)
blue, dark, amber = "#316f91", "#243f50", "#a75d18"
W, H = 15, 6.1; fig = st.figure(W, H)
axes = [cm_axes(fig, 0.5, 0.75, 6.8, 4.1), cm_axes(fig, 7.7, 0.75, 6.8, 4.1)]
for ax in axes:
    ax.set(xlim=(-12, 12), ylim=(7, -7)); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(r"$\xi$", labelpad=2); ax.set_ylabel(r"$\eta$", labelpad=2); ax.grid(False)
for x in [-8, 0, 8]:
    for y in [-4, 0, 4]:
        axes[0].add_patch(Circle((x, y), 1, fc="#bcd5e3", ec=blue, lw=1.0, zorder=2)); axes[0].plot(x, y, ".", color=dark, ms=2, zorder=3)
ftext(fig, 3.9, 0.35, "(a) Punktraster: Radius und Mittenabstände", ha="center", va="center")
axes[0].annotate("", xy=(0, -6), xytext=(-8, -6), arrowprops=dict(arrowstyle="<->", color=dark, lw=1.0))
for x in [-8, 0]: axes[0].plot([x, x], [-4.5, -6.2], color=dark, lw=0.6)
axes[0].text(-4, -6.3, r"$a_u$", ha="center", va="bottom", color=dark, fontsize=11)
axes[0].annotate("", xy=(10, 0), xytext=(10, -4), arrowprops=dict(arrowstyle="<->", color=dark, lw=1.0))
for y in [-4, 0]: axes[0].plot([8.4, 10.3], [y, y], color=dark, lw=0.6)
axes[0].text(10.55, -2, r"$a_v$", rotation=90, ha="left", va="center", fontsize=11)
axes[0].annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=amber, lw=1.4))
axes[0].annotate(r"$r$", xy=(0.8, 0), xytext=(2.5, 1.8), color=amber, fontsize=11, arrowprops=dict(arrowstyle="-", color=amber, lw=0.8))
for x in [-8, 0, 8]:
    axes[1].add_patch(Rectangle((x - 0.25, -4.8), 0.5, 10.3, fc="#bcd5e3", ec=blue, lw=0.8)); axes[1].plot([x, x], [-4.8, 5.5], color=dark, lw=0.5, ls="--")
ftext(fig, 11.1, 0.35, "(b) Linien: volle Breite und Normalabstand", ha="center", va="center")
axes[1].annotate("", xy=(0, -6), xytext=(-8, -6), arrowprops=dict(arrowstyle="<->", color=dark, lw=1.0))
for x in [-8, 0]: axes[1].plot([x, x], [-4.8, -6.2], color=dark, lw=0.6)
axes[1].text(-4, -6.3, r"$a_L$", ha="center", va="bottom", fontsize=11)
axes[1].annotate(r"$w$", xy=(0.25, -2.6), xytext=(2.1, -2.6), va="center", color=amber, fontsize=11, arrowprops=dict(arrowstyle="->", color=amber, lw=1.0))
axes[1].annotate("", xy=(3, 1.1), xytext=(0, 1.1), arrowprops=dict(arrowstyle="->", color=dark, lw=1.2))
axes[1].text(3.35, 1.1, r"$\mathbf{n}_\theta$", ha="left", va="center", fontsize=11)
ftext(fig, 3.9, H - 0.35, "Kreisfläche: Durchmesser = 2$r$", ha="center", va="center", fontsize=9, color=dark)
ftext(fig, 11.1, H - 0.35, "Linienbreite senkrecht zur Mittellinie", ha="center", va="center", fontsize=9, color=dark)
save(fig, 3, "redraw_fig23 sketch; suptitle and footer moved to the caption")

# 2.4 grey-value budget raster (, Abb_2_4)
uv = (np.arange(512) - 255.5) * .18; u, v = np.meshgrid(uv, uv)
patterns = []
for px_ in [8, 6]:
    dx = (u + px_ / 2) % px_ - px_ / 2; dy = (v + 2) % 4 - 2
    patterns.append((f"Punkte {px_}×4 mm\n$r$ = 1,5 mm", dx * dx + dy * dy <= 1.5 ** 2))
patterns.append(("Linien $s$ = 8 mm\n$w$ = 1,5 mm", np.abs((u + 4) % 8 - 4) <= .75))
budget = 45430; meta = []
W = 15; side = 4.15; xs = [1.0, 5.45, 9.9]; H = 0.95 + 2 * (side + 0.75) + 1.6
fig = st.figure(W, H)
for j, (lab, b) in enumerate(patterns):
    s0 = int(b.sum()); assert s0 >= budget; peak = budget / s0
    ftext(fig, xs[j] + side / 2, 0.45, lab, ha="center", va="center", fontsize=10, linespacing=1.15)
    for i, amp in enumerate([1., peak]):
        top = 0.95 + i * (side + 0.75)
        ax = cm_axes(fig, xs[j], top, side, side); ax.imshow(b * amp, cmap="gray", vmin=0, vmax=1, rasterized=True); ax.set_xticks([]); ax.set_yticks([])
        ftext(fig, xs[j] + side / 2, top + side + 0.1, f"$A$ = {st.num(amp, 3)}; $S$ = {st.num(s0 * amp, 0)}", ha="center", va="top", fontsize=9)
    meta.append({"pattern": lab, "unit_peak_pixel_sum": s0, "illustrative_budget": budget, "normalized_peak": peak})
for i, t in enumerate(["Fester Spitzenwert", "Gleiches Integral"]):
    ftext(fig, 0.4, 0.95 + i * (side + 0.75) + side / 2, t, rotation=90, ha="center", va="center")
hbar(fig, 4.5, H - 1.4, 6.0, Normalize(0, 1), "gray", "Normierter Projektionsgrauwert", ticks=[0, .5, 1], fmt=lambda v_, p: st.num(v_))
save(fig, 4, json.dumps({"patterns": meta, "basis": "Eq. 2-4/2-5, pixel centres, s_nom = 0.18 mm/px, no camera rendering"}))

# 2.5 fixed device (redraw_fig25)
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.signal import savgol_filter
Q = SR / "surf02"
cfg = json.loads(track(Q / "config/surf02_config.json").read_text(encoding="utf-8")); c = cfg["frame"]["c_ref_mm"]
names = {"A": "1405Fig14M10_Default_Dataset.csv", "B": "Fig9_Flat_Default_Dataset.csv", "C": "Fig13_M24_Default_Dataset.csv"}
splines = {}
for vv, name in names.items():
    raw = pd.read_csv(track(P / "data/raw" / name), header=None, names=["y", "H"]).apply(pd.to_numeric, errors="coerce").dropna()
    raw["bin"] = raw.y.round(); env = raw.groupby("bin", as_index=False).H.max().sort_values("bin")
    yy = np.arange(env.bin.min(), env.bin.max() + .0625 * .5, .0625)
    hh = savgol_filter(PchipInterpolator(env.bin.to_numpy(), env.H.to_numpy())(yy), 321, 3)
    splines[vv] = (CubicSpline(yy, hh), float(env.bin.min()), float(env.bin.max()))
W, H = 15, 8.6; fig = st.figure(W, H); ax = cm_axes(fig, 1.5, 0.3, 13.3, 7.0)
yp = np.linspace(-46, 46, 501)
for key, col, lab in zip(["A_curv", "C_curv", "B_curv"], ["#1F5B85", "#BA632D", "#3E7E65"], ["M10", "M24", "EF: ebene Fahrbahn"]):
    ax.plot(yp, splines[cfg["surfaces"][key]["variant"]][0](yp), c=col, lw=1.5, label=lab)
ax.axhspan(c - 4, c + 16, color="#DBE6EE", alpha=.7, label="Suchfenster: $z$ von −16 bis +4 mm")
ax.scatter([-25, 25], [c + 150, c + 150], c=["#BA632D", "#1F5B85"], s=30, zorder=5)
ax.text(-25, c + 150 + 6, "Projektor", ha="center", fontsize=10); ax.text(25, c + 150 + 6, "Kamera", ha="center", fontsize=10)
for yp0 in [-25, 25]: ax.plot([yp0, 0], [c + 150, c], ls="--", lw=0.9, color="#536472")
ax.scatter([0], [c], c="black", s=12, zorder=5); ax.annotate("Gemeinsamer Bezug (0, 0, $c$)", (0, c), xytext=(9, 55), arrowprops={"arrowstyle": "->", "lw": .8}, fontsize=10)
ax.annotate("", (-25, c + 132), (25, c + 132), arrowprops={"arrowstyle": "<->", "color": "#536472", "lw": .8}); ax.text(0, c + 136, "Basislinie 50 mm", ha="center", fontsize=10)
ax.annotate("", (48, c), (48, c + 150), arrowprops={"arrowstyle": "<->", "color": "#536472", "lw": .8}); ax.text(51, c + 75, "150 mm\nnominal", va="center", fontsize=10)
ax.set(xlim=(-57, 69), ylim=(8, 184), xlabel="$y_p$ / mm", ylabel="$H$ / mm"); ax.grid(alpha=.15); ax.legend(loc="center left", bbox_to_anchor=(.02, .55))
st.comma_axis(ax)
save(fig, 5, f"redraw_fig25; cross-section x_p = 0; c_ref = {c} mm")

# 2.6 camera images (fig27)
import csv as _csv
S12 = SR / "s12_resource01"
rows = list(_csv.DictReader(open(track(S12 / "manifest.csv"), encoding="utf-8")))
met = {r["input_key"]: r for r in _csv.DictReader(open(track(S12 / "results/inputs_metrics.csv"), encoding="utf-8"))}
def pick(family, sigma, **cond):
    cc = [r for r in rows if r["surface"] == "M10_curv" and r["family"] == family and r["regime"] == "resource_cap_11695"
          and r["sigma"] == sigma and r["rep"] == "0" and all(r[k] == vv for k, vv in cond.items())]
    assert len(cc) == 1, (family, sigma, cond, len(cc)); return cc[0]
sel = {}
for s in ("0.0", "40.0", "100.0"):
    sel[("Punkte", s)] = pick("point", s, layout="R8x4", diameter_mm="2.0", phase_label="pos23")
    sel[("Linien", s)] = pick("line", s, line_pitch_mm="8.0", size_mm="0.5", orientation_deg="0.0", phase_label="ph0")
imgs = {}
for key, r in sel.items():
    npz = (P / met[r["input_key"]]["image_record"]).with_suffix(".npz")
    im = nz(npz)["image"]; assert im.shape == (512, 512) and im.dtype == np.uint8; imgs[key] = im
W = 15; side = 4.35; xs = [1.45, 5.95, 10.45]; H = 0.6 + 2 * side + 0.15 + 1.6
fig = st.figure(W, H)
rowlab = {"Punkte": "Punkte\n$r$ = 1 mm, 8 × 4 mm", "Linien": "Linien\n$w$ = 0,5 mm, 8 mm"}
for i, fam in enumerate(("Punkte", "Linien")):
    top = 0.6 + i * (side + 0.15)
    ftext(fig, 0.6, top + side / 2, rowlab[fam], rotation=90, ha="center", va="center", linespacing=1.2)
    for k, s in enumerate(("0.0", "40.0", "100.0")):
        ax = cm_axes(fig, xs[k], top, side, side); ax.imshow(imgs[(fam, s)], cmap="gray", vmin=0, vmax=255, interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ftext(fig, xs[k] + side / 2, 0.3, r"$\sigma_\mathrm{n}$ = " + f"{int(float(s))}", ha="center", va="center")
hbar(fig, 5.0, H - 1.4, 5.0, Normalize(0, 255), "gray", "Grauwert", ticks=[0, 128, 255], fmt=lambda v_, p: str(int(round(v_))))
save(fig, 6, json.dumps({k[0] + " " + k[1]: r["input_key"] for k, r in sel.items()}))

# 2.7 correspondence schematic ()
blue_, orange_, dark_ = "#1877A3", "#BA642B", "#233644"
W, H = 15, 5.35; fig = st.figure(W, H)
axs = [cm_axes(fig, 0.15 + j * 4.95, 0.6, 4.75, 3.6) for j in range(3)]
for ax in axs:
    ax.set(xlim=(0, 10), ylim=(0, 5)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values(): spine.set_visible(True); spine.set_color("#CCD5DB")
for j, t in enumerate(["(a) Eindeutiger Punkt", "(b) Mehrdeutiger Punkt", "(c) Kandidaten einer Linie"]):
    ftext(fig, 0.15 + j * 4.95 + 2.375, 0.3, t, ha="center", va="center")
for ax, tracks, xq in [(axs[0], [(1, 4, 2.5), (6, 9, 2.5)], 2.7), (axs[1], [(1, 5, 2.5), (4, 8, 2.5)], 4.5)]:
    for j, (x0, x1, y) in enumerate(tracks):
        col = [blue_, orange_][j]; yy = y + (.2 if j else -.2)
        ax.plot([x0, x1], [yy, yy], color=col, lw=1.5); ax.fill_between([x0, x1], yy - .5, yy + .5, color=col, alpha=.12, lw=0); ax.scatter([x0, x1], [yy, yy], c=col, s=10)
        ax.text((x0 + x1) / 2, yy + (1.03 if j else -1.05), f"Spur {j + 1}", ha="center", color=col, fontsize=9)
    ax.scatter(xq, 2.5, c=dark_, s=25, zorder=5)
    ax.annotate("Bildmerkmal", xy=(xq, 2.5), xytext=(4.5, 4.3), ha="center", arrowprops={"arrowstyle": "->", "color": dark_, "lw": .8}, color=dark_, fontsize=9)
    ax.text(5, .25, "Kamerabild", ha="center", color="#5B6972", fontsize=9)
ax = axs[2]
for x, k in [(2.2, "$k-1$"), (5, "$k$"), (7.8, "$k+1$")]:
    ax.plot([x, x], [.8, 4], color=blue_, lw=1.5); ax.text(x, 4.2, k, ha="center", color=blue_, fontsize=9)
ax.plot([.5, 9.5], [2.4, 2.4], "--", color="#8998A2", lw=1.0); ax.plot([3.3, 6.6], [2.4, 2.4], color=orange_, lw=3)
ax.scatter([2.2, 5, 7.8], [2.4] * 3, c=["#B5BEC4", dark_, "#B5BEC4"], s=22, zorder=5)
ax.text(3.3, 1.6, r"$z_\mathrm{min}$", ha="center", fontsize=9); ax.text(6.6, 1.6, r"$z_\mathrm{max}$", ha="center", fontsize=9)
ax.text(5, .25, "Projektorbild", ha="center", color="#5B6972", fontsize=9)
for j, t in enumerate(["Nur eine Spur im\nToleranzbereich", "Zwei Spuren:\nnicht eindeutig", "Kandidat $k$ liegt\nim Suchfenster"]):
    ftext(fig, 0.15 + j * 4.95 + 2.375, 4.35, t, ha="center", va="top", fontsize=9, linespacing=1.2)
save(fig, 7, "schematic; footer moved to the caption")

# 2.8 reconstruction M10 (redraw_fig28)
D = nz(SR / "s1_real01/domain/global_domain_step0p50.npz")
X, Y = np.meshgrid(D["lattice_x"], D["lattice_y"])
zref = np.full(X.shape, np.nan); zref[D["mask"]] = D["z_A_curv"]
C5 = pd.read_csv(track(SR / "s5_align05/case_results.csv"), low_memory=False)
def s5row(combo, sid, phase, sg, rep, fe):
    a = C5[(C5.combo == combo) & (C5.surface == sid) & (C5.phase_label == phase) & (C5.sigma == sg) & (C5.rep == rep) & (C5.front_end == fe)]
    assert len(a) == 1; return a.iloc[0]
VIEW8, BOX8 = (28, -60), (1, 1, .45)
W, H = 15, 6.6; fig = st.figure(W, H); keys = []
axes = [cm_axes(fig, 0.05 + i * 4.98, 0.55, 4.95, 4.1, projection="3d", computed_zorder=False) for i in range(3)]
axes[0].plot_surface(X[::4, ::4], Y[::4, ::4], zref[::4, ::4], cmap="viridis", vmin=-16, vmax=4, linewidth=0, rasterized=True)
for ax, combo, fe in zip(axes[1:], ["D_r1_p8x4", "L_w0.5_s8"], ["D0", "P4"]):
    r = s5row(combo, "M10_curv", "ph0", 0, 0, fe); a = nz(P / r.eval_record); vmask = a["valid"].astype(bool); p = a["pts"][vmask]; assert len(p) == r.n_valid
    ax.plot_wireframe(X[::5, ::5], Y[::5, ::5], zref[::5, ::5], color=".6", lw=.3, alpha=.55, rasterized=True)
    ax.scatter(*p.T, c=p[:, 2], cmap="viridis", vmin=-16, vmax=4, s=2.5 if fe == "D0" else .3, depthshade=False, zorder=5, rasterized=True, linewidths=0)
    keys.append({"combo": combo, "eval_record": r.eval_record, "n_valid": int(len(p)), "sigma": 0})
for ax, t in zip(axes, ["Referenzfläche M10", "Punktmuster, D0", "Linienmuster, P4"]):
    ax.set(xlim=(-47, 47), ylim=(-47, 47), zlim=(-16, 4)); ax.set_box_aspect(BOX8, zoom=1.15); ax.view_init(*VIEW8); bare3d(ax)
for i, t in enumerate(["Referenzfläche M10", "Punktmuster, D0", "Linienmuster, P4"]):
    ftext(fig, 2.53 + i * 4.98, 0.35, t, ha="center", va="center")
hbar(fig, 4.5, H - 1.4, 6.0, Normalize(-16, 4), "viridis", "Höhe $z$ / mm", ticks=[-16, -11, -6, -1, 4], fmt=lambda v_, p: st.num(round(v_)))
triad(fig, 0.45, 0.6, VIEW8, BOX8, L=0.7)
save(fig, 8, json.dumps({"cases": keys}))

# 2.9 evaluation definitions ()
blue9, orange9, dark9, green9 = "#1C6590", "#C87932", "#233644", "#2B8269"
W, H = 15, 5.9; fig = st.figure(W, H)
axs = [cm_axes(fig, 0.6 + j * 4.85, 0.6, 4.2, 4.2) for j in range(3)]
for j, t in enumerate(["(a) Höhenfehler am Punkt", "(b) Lineare Interpolation", "(c) Stützmaß des Dreiecks"]):
    ftext(fig, 0.6 + j * 4.85 + 2.1, 0.3, t, ha="center", va="center")
x = np.linspace(-1, 1, 220); y = 1.7 + .18 * x - .35 * x * x
ax = axs[0]; ax.plot(x, y, color=dark9, lw=1.5)
xp = .25; yt = 1.7 + .18 * xp - .35 * xp * xp; yp_ = yt + .42
ax.scatter(xp, yp_, c=blue9, s=22, zorder=5); ax.scatter(xp, yt, c=dark9, s=12, zorder=5)
ax.annotate("", xy=(xp, yp_), xytext=(xp, yt), arrowprops=dict(arrowstyle="<->", color=orange9, lw=1.2))
ax.text(xp + .1, (yt + yp_) / 2, "$e_{H,i}$", color=orange9, fontsize=11)
ax.text(xp - .09, yp_ + .1, r"$\widehat H_i$", color=blue9, ha="right", fontsize=11)
ax.text(.55, 1.22, r"$H_\mathrm{ref}(\widehat x_{p,i},\widehat y_{p,i})$", color=dark9, ha="center", fontsize=10)
ax.plot([xp, xp], [.98, yt], ls=":", color="#8E9AA2", lw=0.8); ax.text(xp, 1.00, r"$\widehat x_{p,i}$", ha="center", va="top", fontsize=10)
ax.set(xlim=(-1.1, 1.2), ylim=(.9, 2.38), xticks=[], yticks=[]); ax.set_xlabel("$x_p$", labelpad=12); ax.set_ylabel("$H$", labelpad=2); ax.grid(False)
vtx = np.array([[0., 0.], [8., 0.], [0., 8.]])
for ax in axs[1:]:
    ax.fill(*vtx.T, color=green9, alpha=.1); ax.plot(*np.vstack([vtx, vtx[0]]).T, color=green9, lw=1.3); ax.scatter(*vtx.T, c=blue9, s=22, zorder=5)
    ax.set(xlim=(-1.2, 9.2), ylim=(-1.3, 9.2), xticks=[], yticks=[]); ax.set_xlabel("$x_p$", labelpad=2); ax.set_ylabel("$y_p$", labelpad=2); ax.set_aspect("equal"); ax.grid(False)
ax = axs[1]; q = np.array([2.4, 2.3]); ax.scatter(*q, c=orange9, s=20, zorder=5)
for k, pt in enumerate(vtx):
    ax.plot([q[0], pt[0]], [q[1], pt[1]], ":", color="#8E9AA2", lw=0.8); ax.text(pt[0] - .25, pt[1] + .35, rf"$\widehat H_{k + 1}$", color=blue9, fontsize=10)
ax.text(q[0] + .3, q[1] + .2, r"$\mathbf{q}_j$", color=orange9, fontsize=10)
ax.text(5.6, 6.3, r"$\widehat H(\mathbf{q}_j)$" + "\n" + r"$=\sum_{k=1}^{3}\lambda_{jk}\widehat H_k$", ha="center", fontsize=10)
ax = axs[2]; q = np.array([4., 4.]); ax.scatter(*q, c=orange9, s=20, zorder=5)
for pt in vtx: ax.plot([q[0], pt[0]], [q[1], pt[1]], "--", color=orange9, lw=0.9)
ax.text(4.5, 4.4, r"$\mathbf{q}^*$", color=orange9, fontsize=10); ax.text(1.2, 2.5, r"$\rho_T$", color=orange9, fontsize=11)
ax.text(5.6, 7.0, "Stützbedingung:\n" + r"$\rho_T\leq d_s$", ha="center", fontsize=10)
save(fig, 9, "definition sketch; footer moved to the caption")

(OUTDIR / "kap2_record.json").write_text(json.dumps(dict(META, inputs=SOURCE), ensure_ascii=False, indent=2), encoding="utf-8")
print("figures:", len(META["figures"]), "| inputs:", len(SOURCE), "| min font:", min(r["min_font_pt"] for r in META["qa"]),
      "| sizes:", [(r["figure"][-2:], r["height_cm"]) for r in META["qa"]])
