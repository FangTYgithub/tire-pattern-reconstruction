# -*- coding: utf-8 -*-

from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thesis_figstyle as st
from plot_io import configure, verify_inputs, selected
DATA_ROOT, FIGURE_OUT = configure('kap3_redraw')
verify_inputs()
st.setup()
OUTDIR = FIGURE_OUT; OUTDIR.mkdir(parents=True, exist_ok=True)
META = {"figures": {}, "qa": []}

# data: copied from data_figures.py (lines 12-54, 161-171)
ROOT = DATA_ROOT
P=ROOT/'research/project_Part1_v1.7'
S=P/'outputs/surface_revision'
SOURCE={}
def read(rel):
    p=S/rel; SOURCE[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,low_memory=False)
SM={'B_curv':'EF','M10_curv':'M10','M24_curv':'M24'}
SURFS=['B_curv','M10_curv','M24_curv']; SIG=[0,20,40,60,80,100]
AV=read('s3_fe01/results/A_s12_views.csv'); AV=AV[AV.rule=='DF'].copy()
B=read('s3_fe01/results/B_size02_cases.csv'); B=B[B.rule=='DF'].copy()
SIZE=pd.concat([AV[AV.section=='S1'],B],ignore_index=True)
LAY=AV[AV.section=='S2'].copy()
assert len(SIZE)==1440 and SIZE.input_key.nunique()==1296
assert len(LAY)==2640 and LAY.input_key.nunique()==2640
SZ=SIZE.groupby(['regime','surface','size_mm','sigma','phase_label']).mean(numeric_only=True).reset_index()
LC=LAY.groupby(['layout','surface','sigma','phase_label']).mean(numeric_only=True).reset_index()
C=read('s3_fe01/results/C_s4_cases.csv'); C=C[C.rule=='DF'].copy()
M2=read('s4_real02/manifest.csv'); M2=M2[M2.family=='point']
cols=['image_key','strategy','block','surface','phase','sigma','layout','view_id']
V2=M2[cols].merge(C[C.group=='six_noise'][['image_key','glob_Ctau']],on='image_key',validate='many_to_one')
assert V2.glob_Ctau.notna().all()
R7=read('s4_real07/results.csv')
V7=R7[(R7.family=='point')&(R7.method=='D0')&(R7.record_type=='view')&(R7.domain=='glob')].copy()
assert not V7.duplicated(['surface','sigma','block','phase','arm']).any()
metric=C.set_index('image_key').glob_Ctau
V7['DF_Ctau']=V7.image_key.map(metric);V7['DF_S0']=V7.s0_image_key.map(metric)
assert V7[['DF_Ctau','DF_S0']].notna().all().all()
V7['delta']=V7.DF_Ctau-V7.DF_S0
COM=read('s5_align05/case_results.csv'); COM=COM[COM.front_end.isin(['DF','P4'])].copy()
assert len(COM)==3960
CP=COM.groupby(['surface','combo','sigma','phase_label']).mean(numeric_only=True).reset_index()
EF=read('s5_align06/paired_effects.csv');EF.sigma=pd.to_numeric(EF.sigma,errors='coerce')
EF=EF[(EF.front_end.isin(['DF','P4']))&(EF.metric=='glob_Ctau')&(EF.record_type=='interaction')].copy()

def loadpts(row,mode):
    if mode=='fe':
      stage='size02' if row.size_mm in [.75,1.25] else 's12'
      p=S/f's3_fe01/{stage}/cache/{row.input_key}.npz';pref=''
    elif mode=='s4':p=S/f's3_fe01/s4/cache/{row.image_key}.npz';pref=''
    else:p=P/row.eval_record;pref=row.image_key+'__'
    SOURCE[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p,allow_pickle=False) as z:
      if pref+'pts' not in z:pref=''
      pts=z[pref+'pts'];v=z[pref+'valid'].astype(bool)&z[pref+'in_global'].astype(bool)
      return pts[v],z[pref+'zr'][v]


SIGLAB = r"$\sigma_\mathrm{n}$ / Graustufen"
COLORS = plt.get_cmap("tab10")

def cm_axes(fig, x, top, w, h, **kw):
    W, H = fig.get_size_inches() / st.CM
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H], **kw)

def lay(c):
    """'R7.35x4' -> 'R7,35×4'."""
    return str(c).replace("x", "×").replace(".", ",")

def combo(c):
    """'D_r0.75_p8x4' -> '0,75 R8×4' (legend title gives r / mm, Anordnung)."""
    r, p = c[2:].split("_")
    return st.num(float(r[1:])) + " " + lay(p.replace("p", "R"))

def save(fig, num, notes=""):
    name = f"Abb_3_{num:02d}"
    if not selected(name):
        plt.close(fig)
        return
    st.save(fig, OUTDIR / name, name, META["qa"])
    META["figures"][name] = {"notes": notes, "pdf_sha256": hashlib.sha256((OUTDIR / (name + ".pdf")).read_bytes()).hexdigest()}

def logaxis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: st.num(float(f"{v:.3g}"))))
    ax.yaxis.set_minor_formatter(NullFormatter())

def grid(nrows, ylabels, legend_rows, ph=3.25, ncols=3, left=2.05, legend_h=0.5):
    """Rows of metrics x columns of surfaces; returns fig, axes, layout dict (cm)."""
    W, right, top, gx, gy = 15, 0.2, 0.55, 0.3, 0.3
    pw = (W - left - right - (ncols - 1) * gx) / ncols
    bottom = 1.2 + legend_rows * legend_h + 0.3
    H = top + nrows * ph + (nrows - 1) * gy + bottom
    fig = st.figure(W, H)
    axes = np.array([[cm_axes(fig, left + j * (pw + gx), top + i * (ph + gy), pw, ph) for j in range(ncols)] for i in range(nrows)])
    for i in range(nrows):
        axes[i, 0].set_ylabel(ylabels[i])
        for j in range(ncols):
            ax = axes[i, j]; ax.set_xticks(SIG); ax.set_xlim(-6, 106)
            if i < nrows - 1: ax.tick_params(labelbottom=False)
            if j: ax.tick_params(labelleft=False)
    fig.text((left + (W - left - right) / 2) / W, 1 - (top + nrows * ph + (nrows - 1) * gy + 0.88) / H, SIGLAB, ha="center", va="center", fontsize=11)
    return fig, axes, {"W": W, "H": H, "legend_top": top + nrows * ph + (nrows - 1) * gy + 1.2, "center": (left + (W - left - right) / 2) / W}

def share_rows(axes, logrows=()):
    for i, row in enumerate(axes):
        if i in logrows:
            lo = min(a.get_ylim()[0] for a in row); hi = max(a.get_ylim()[1] for a in row)
            for a in row: a.set_ylim(lo, hi)
        else:
            for a in row: a.set_ylim(0, 103); a.set_yticks([0, 25, 50, 75, 100])

def legend(fig, L, handles, labels, ncol, title=None):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(L["center"], 1 - L["legend_top"] / L["H"]), ncol=ncol, title=title,
               handlelength=1.4, columnspacing=1.0, handletextpad=0.4, borderaxespad=0)

YL = {"glob_Ctau": "$C_\\tau$ / %", "glob_recovery": "korrekte\nRückgewinnung / %", "glob_pt_rmse": "Punkt-RMSE / mm",
      "glob_surf_rmse": "Oberflächen-\nRMSE / mm", "loc_Ctau": "$C_\\tau$ lokal / %", "sup_surf_rmse": "Oberflächen-RMSE\ngestützt / mm"}

def plot_curves(df, group, configs, metrics, num, labeler, legtitle, ncol, notes):
    """Same aggregation as data_figures.plot_curves: phase means, band = min..max of phase means."""
    nleg = int(np.ceil(len(configs) / ncol)) + (1 if legtitle else 0)
    fig, axes, L = grid(len(metrics), [YL[m] for m in metrics], nleg)
    for j, sf in enumerate(SURFS):
        for k, c in enumerate(configs):
            sub = df[(df.surface == sf) & (df[group] == c)]
            for i, col in enumerate(metrics):
                scale = 1 if "rmse" in col else 100
                g = sub.groupby("sigma")[col].agg(["mean", "min", "max"]).reindex(SIG) * scale
                ax = axes[i, j]
                ax.plot(SIG, g["mean"], "o-", ms=2.5, lw=1.1, color=COLORS(k), label=labeler(c))
                ax.fill_between(SIG, g["min"], g["max"], color=COLORS(k), alpha=.09, lw=0)
                if scale == 1: logaxis(ax)
        axes[0, j].set_title(SM[sf], pad=3)
    share_rows(axes, [i for i, m in enumerate(metrics) if "rmse" in m])
    for ax in axes[:, 0]:
        if ax.get_yscale() != "log": st.comma_axis(ax, "y")
    legend(fig, L, *axes[0, 0].get_legend_handles_labels(), ncol, legtitle)
    save(fig, num, notes)

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

def cloud(rows, mode, num, labels, notes):
    """Same points, reference, colour limits and error scale as data_figures.cloud."""
    dom = S / 's1_real01/domain/global_domain_step0p50.npz'
    sf = rows[0].surface; zk = {'B_curv': 'z_B_curv', 'M10_curv': 'z_A_curv', 'M24_curv': 'z_C_curv'}[sf]
    with np.load(dom) as z:
        xyz = np.c_[z['x'], z['y'], z[zk]]; xyz = xyz[np.isfinite(xyz).all(axis=1)]
    SOURCE[str(dom)] = hashlib.sha256(dom.read_bytes()).hexdigest()
    ptslist = [loadpts(r, mode) for r in rows]
    allz = np.concatenate([xyz[:, 2]] + [p[:, 2] for p, _ in ptslist]); lo, hi = float(allz.min()), float(allz.max())
    norm = Normalize(lo, hi); en = Normalize(-.5, .5)
    VIEW, BOX = (25, -58), (90, 90, 40)
    k = len(rows); W = 15; left, right = 1.0, 2.3
    cw = (W - left - right) / k; side = min(cw - 0.9, 3.3); H = 4.75 + side + 1.05 + 1.3
    fig = st.figure(W, H)
    for i, ((pts, zr), label) in enumerate(zip(ptslist, labels)):
        x0 = left + i * cw
        fig.text((x0 + cw / 2) / W, 1 - 0.3 / H, label, ha="center", va="center", fontsize=10)
        ax = cm_axes(fig, x0, 0.55, cw, 3.9, projection="3d")
        thin = xyz[::35]; ax.scatter(*thin.T, c='#90ab95', s=.25, alpha=.25, rasterized=True)
        ax.scatter(*pts.T, c=pts[:, 2], cmap='viridis', norm=norm, s=3.5, depthshade=False, rasterized=True, linewidths=0)
        ax.set(xlim=(-45, 45), ylim=(-45, 45), zlim=(lo, hi)); ax.set_box_aspect(BOX, zoom=0.98); ax.view_init(*VIEW)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.tick_params(length=0, pad=0); ax.grid(False)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): pane.fill = False; pane.set_edgecolor("#dddddd")
        ax = cm_axes(fig, x0 + (cw - side) / 2 + 0.25, 4.75, side, side)
        ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2] - zr, cmap='RdBu_r', norm=en, s=5, linewidths=0, rasterized=True)
        ax.set(xlim=(-45, 45), ylim=(-45, 45), xticks=[-40, 0, 40], yticks=[-40, 0, 40]); ax.set_aspect("equal"); ax.grid(False); st.comma_axis(ax)
        ax.set_xlabel("$x$ / mm", labelpad=1)
        if i == 0: ax.set_ylabel("$y$ / mm", labelpad=0)
        else: ax.tick_params(labelleft=False)
        rmse = np.sqrt(np.mean((pts[:, 2] - zr) ** 2)); out = int((np.abs(pts[:, 2] - zr) > .5).sum())
        fig.text((x0 + cw / 2 + 0.25) / W, 1 - (4.75 + side + 1.05) / H, f"$n$ = {len(pts)}\nRMSE = {st.num(rmse, 3)} mm\n|Δ$z$| > 0,5 mm: {out}",
                 ha="center", va="top", fontsize=9, linespacing=1.25)
    cax = cm_axes(fig, W - right + 0.2, 1.0, 0.22, 2.9)
    b = fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), cax=cax); b.set_label("$z$ / mm"); b.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: st.num(round(v, 1))))
    cax = cm_axes(fig, W - right + 0.2, 4.9, 0.22, 3.0)
    b = fig.colorbar(ScalarMappable(norm=en, cmap='RdBu_r'), cax=cax, extend="both"); b.set_label("Δ$z$ / mm"); b.set_ticks([-.5, -.25, 0, .25, .5])
    b.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: st.num(round(v, 2))))
    triad(fig, 0.35, H - 4.0, VIEW, BOX)
    save(fig, num, notes)

# figures (thesis numbering; reference file number in brackets)
cov = ["glob_Ctau", "glob_recovery"]; err = ["glob_pt_rmse", "glob_surf_rmse"]; rad = [.5, .75, 1, 1.25, 1.5]
rlab = lambda c: st.num(c)
# 3.1 [01], 3.2 [02]
plot_curves(SZ[SZ.regime == 'resource_cap_11695'], 'size_mm', rad, cov, 1, rlab, "$r$ / mm", 5, "band = range of phase means; DF")
plot_curves(SZ[SZ.regime == 'resource_cap_11695'], 'size_mm', rad, err, 2, rlab, "$r$ / mm", 5, "phase means; log axis; DF")

# 3.3 [03] representative clouds (same case selection as reference)
q = SIZE[(SIZE.surface == 'M24_curv') & (SIZE.regime == 'resource_cap_11695') & (SIZE.sigma == 40)]
ref = q[q.size_mm == 1].sort_values(['phase_label', 'rep']).iloc[0]
rows = [q[(q.size_mm == r) & (q.phase_label == ref.phase_label) & (q.rep == ref.rep)].iloc[0] for r in [.75, 1, 1.25]]
cloud(rows, 'fe', 3, [f"$r$ = {st.num(r)} mm" for r in [.75, 1, 1.25]], f"M24, sigma 40, {ref.phase_label}, rep {int(ref.rep)}")

# 3.5 [06] radius choice under two resource conditions
fig = st.figure(15, 7.7)
axs = [cm_axes(fig, 1.55, 0.6, 6.3, 4.6), cm_axes(fig, 8.35, 0.6, 6.3, 4.6)]
for ax, reg, title in zip(axs, ['resource_cap_11695', 'fixed_peak_1'], ['Gemeinsame Obergrenzen', 'Spitzenwert $A$ = 1']):
    qq = SZ[SZ.regime == reg].groupby(['size_mm', 'sigma']).glob_Ctau.mean().unstack().reindex(index=[.5, .75, 1, 1.25, 1.5], columns=SIG) * 100
    for k, s in enumerate([20, 40, 60, 80, 100]): ax.plot(qq.index, qq[s], 'o-', ms=2.5, lw=1.1, color=COLORS(k), label=st.num(s))
    ax.set(xticks=qq.index, ylim=(0, 100), yticks=[0, 25, 50, 75, 100]); ax.set_title(title, pad=3); ax.set_xlabel("Punktradius $r$ / mm"); st.comma_axis(ax)
axs[1].tick_params(labelleft=False); axs[0].set_ylabel("$C_\\tau$ / %")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.535, 1 - 6.3 / 7.7), ncol=5, title=r"$\sigma_\mathrm{n}$ / Graustufen", handlelength=1.4, columnspacing=1.2, borderaxespad=0)
save(fig, 5, "equal-weight mean of three surfaces; DF")

# 3.6 [07], 3.7 [08] layout curves (four metrics)
met = ["glob_Ctau", "glob_pt_rmse", "glob_surf_rmse", "glob_recovery"]
plot_curves(LC, 'layout', ['R6x4', 'R7.35x4', 'R8x4', 'R10x4', 'R12x4'], met, 6, lay, "Anordnung", 5, "phase means, band = range; DF")
plot_curves(LC, 'layout', ['R8x2.5', 'R8x3', 'R8x4', 'R8x6', 'R8x8', 'T8x4'], met, 7, lay, "Anordnung", 6, "phase means, band = range; DF")

# 3.10 [11] ideal sampling reference vs DF
t = read('thesis_check01/_parts/B_results.csv')
qq = t[(t.record_type == 'summary') & (t.section == 'S2') & (t.family == 'point') & (t.domain == 'glob') & (t.sigma == 0)]
if qq.empty:
    qq = t[(t.section == 'S2') & (t.family == 'point') & (t.domain == 'glob') & (t.sigma == 0) & t.ref_C_tau_mean.notna()]
assert not qq.empty
cfgs = ['R6x4', 'R7.35x4', 'R8x4', 'R10x4', 'R12x4', 'R8x2.5', 'R8x3', 'R8x6', 'R8x8', 'T8x4']
fig = st.figure(15, 8.4); W, H = 15, 8.4
axs = [cm_axes(fig, 1.55 + j * 4.5, 0.55, 4.25, 4.9) for j in range(3)]
for ax, sf in zip(axs, SURFS):
    d = qq[qq.surface == sf].groupby('layout').ref_C_tau_mean.mean().reindex(cfgs) * 100
    x = np.arange(len(cfgs)); ax.bar(x, d, color='#c4d2c1', width=.8, label='Ideale Abtastreferenz')
    for s, col in [(0, '#111111'), (40, '#bd4040')]:
        a = LC[(LC.surface == sf) & (LC.sigma == s)].groupby('layout').glob_Ctau.mean().reindex(cfgs) * 100
        ax.plot(x, a, 'o', ms=3.5, c=col, label=r"DF, $\sigma_\mathrm{n}$ = " + str(s))
    ax.set(xticks=x, ylim=(0, 105), yticks=[0, 25, 50, 75, 100]); ax.set_xticklabels([lay(c) for c in cfgs], rotation=90); ax.set_title(SM[sf], pad=3)
    ax.grid(axis='x', visible=False); st.comma_axis(ax, "y")
    if sf != SURFS[0]: ax.tick_params(labelleft=False)
axs[0].set_ylabel("$C_\\tau$ / %")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.535, 1 - 7.6 / H), ncol=3, handlelength=1.2, columnspacing=1.2, borderaxespad=0)
save(fig, 10, "bars: ideal sampling reference (thesis_check01); dots: DF mean")

# 3.18 [19] position variation
fig = st.figure(15, 7.3); H = 7.3
axs = [cm_axes(fig, 1.55 + j * 4.5, 0.55, 4.25, 4.3) for j in range(3)]
for ax, sf in zip(axs, SURFS):
    for j, stg in enumerate(['S0', 'S1', 'S2', 'S3', 'S4']):
        qv = V2[(V2.surface == sf) & (V2.strategy == stg)]
        if stg == 'S0': z = qv.groupby('sigma').glob_Ctau.agg(['median', 'min', 'max'])
        else:
            bb = qv.groupby(['sigma', 'block']).glob_Ctau.median().reset_index()
            z = bb.groupby('sigma').glob_Ctau.agg(['median', 'min', 'max'])
        z = z.reindex(SIG) * 100
        ax.plot(SIG, z['median'], 'o-' if stg != 'S4' else 's--', ms=2.5, lw=1.1, color=COLORS(j), label=stg)
        ax.fill_between(SIG, z['min'], z['max'], color=COLORS(j), alpha=.07, lw=0)
    ax.set(xticks=SIG, xlim=(-6, 106), ylim=(0, 100), yticks=[0, 25, 50, 75, 100]); ax.set_title(SM[sf], pad=3); st.comma_axis(ax, "y")
    if sf != SURFS[0]: ax.tick_params(labelleft=False)
axs[0].set_ylabel("$C_\\tau$ / %")
fig.text(0.535, 1 - 5.7 / H, SIGLAB, ha="center", va="center", fontsize=11)
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.535, 1 - 6.25 / H), ncol=5, handlelength=1.6, columnspacing=1.2, borderaxespad=0)
save(fig, 18, "median over layout blocks; S0 median and range of four phases; DF; j = 0.20")

# 3.20 [21] free adjustment clouds
v = V7[(V7.surface == 'M10_curv') & (V7.sigma == 0) & (V7.arm == 'S1')].sort_values(['delta', 'block', 'phase']).iloc[0]
rows = [C[C.image_key == key].iloc[0] for key in [v.s0_image_key, v.image_key]]
cloud(rows, 's4', 20, ['S0: regelmäßiges Raster', 'S1: freie Verschiebung'], f"M10, sigma 0, {v.block}, {v.phase}")

# 3.23 [24] nine point combinations (coverage: phase means + range; RMSE: median of defined cases, hollow = partly defined)
configs = sorted(COM[COM.front_end == 'DF'].combo.unique())
rws = ["glob_Ctau", "loc_Ctau", "glob_pt_rmse", "sup_surf_rmse"]
YL["glob_Ctau"] = "$C_\\tau$ global / %"
fig, axes, L = grid(4, [YL[r] for r in rws], 4, ph=2.95)
YL["glob_Ctau"] = "$C_\\tau$ / %"
for j, sf in enumerate(SURFS):
    for k, c in enumerate(configs):
        raw = COM[(COM.front_end == 'DF') & (COM.surface == sf) & (COM.combo == c)]
        phase = CP[(CP.surface == sf) & (CP.combo == c)]
        for i, col in enumerate(rws):
            ax = axes[i, j]
            if "rmse" not in col:
                g = phase.groupby('sigma')[col].agg(['mean', 'min', 'max']).reindex(SIG) * 100
                y = g['mean']; ax.fill_between(SIG, g['min'], g['max'], color=COLORS(k), alpha=.09, lw=0)
            else:
                g = raw.groupby('sigma')[col].agg(['median', 'count', 'size']).reindex(SIG)
                y = g['median']
            ax.plot(SIG, y, 'o-', ms=2.5, lw=1.0, color=COLORS(k), label=combo(c))
            if "rmse" in col:
                partial = (g['count'] > 0) & (g['count'] < g['size'])
                ax.plot(np.array(SIG)[partial], y[partial], 'o', ms=3.5, mfc='white', mec=COLORS(k))
                logaxis(ax)
    axes[0, j].set_title(SM[sf], pad=3)
share_rows(axes, [2, 3])
for ax in axes[:2, 0]: st.comma_axis(ax, "y")
legend(fig, L, *axes[0, 0].get_legend_handles_labels(), 3, "$r$ / mm, Anordnung")
save(fig, 23, "coverage phase means + range; RMSE median of defined cases; hollow = partly defined")

# 3.25 [26] size-spacing interaction
sets = ['dots_parallel', 'dots_perp', 'lines', 'lines_6.3_8']
assert set(sets).issubset(set(EF['set']))
rowlab = {'dots_parallel': 'Punkte 6 → 10 mm\n$r$ 1 / 1,5 mm', 'dots_perp': 'Punkte 4 → 6 mm\n$r$ 1 / 1,5 mm',
          'lines': 'Linien 4,45 → 8 mm\n$w$ 0,5 / 1,5 mm', 'lines_6.3_8': 'Linien 6,3 → 8 mm\n$w$ 0,5 / 1,5 mm'}
fig, axes, L = grid(4, [""] * 4, 1, ph=3.0, left=2.95)
for i, se in enumerate(sets):
    for j, sf in enumerate(SURFS):
        ax = axes[i, j]
        for eff, col, labx in [('E_small', '#2d70a6', 'Kleineres Merkmal'), ('E_large', '#c1483c', 'Größeres Merkmal'), ('I', '#202020', 'Interaktion')]:
            an = 'posthoc' if se == 'lines_6.3_8' else 'registered'
            fe = 'DF' if se.startswith('dots_') else 'P4'
            d = EF[(EF['set'] == se) & (EF.surface == sf) & (EF.effect == eff) & EF.sigma.isin(SIG) & EF.analysis.str.startswith(an) & (EF.front_end == fe)].sort_values('sigma')
            assert d.sigma.tolist() == SIG, (se, sf, eff)
            ax.plot(d.sigma, 100 * d['mean'], 'o-', c=col, label=labx, ms=2.5, lw=1.1)
            ax.fill_between(d.sigma, 100 * d.res_lo_5, 100 * d.res_hi_95, color=col, alpha=.1, lw=0)
        ax.axhline(0, color='#888', lw=.7)
        if i == 0: ax.set_title(SM[sf], pad=3)
    lo = min(a.get_ylim()[0] for a in axes[i]); hi = max(a.get_ylim()[1] for a in axes[i])
    for a in axes[i]: a.set_ylim(lo, hi)
    st.comma_axis(axes[i, 0], "y")
    fig.text(0.15 / L["W"], 1 - (0.55 + i * 3.3 + 1.5) / L["H"], rowlab[se], rotation=90, ha="left", va="center", fontsize=10, linespacing=1.2)
fig.text(1.45 / L["W"], 1 - (0.55 + 2 * 3.3 - 0.15) / L["H"], "Δ$C_\\tau$ / Prozentpunkte", rotation=90, ha="center", va="center", fontsize=11)
legend(fig, L, *axes[0, 0].get_legend_handles_labels(), 3)
save(fig, 25, "bands: descriptive 5-95 % block intervals; dots DF, lines P4")

# 3.26 [27] recovery and coverage of representative combinations
plot_curves(CP, 'combo', ['D_r0.75_p8x4', 'D_r1_p8x4', 'D_r1_p8x6', 'D_r1.5_p8x4', 'D_r1.5_p8x6'], ['glob_recovery', 'glob_Ctau'], 26, combo, "$r$ / mm, Anordnung", 5, "phase means, band = range; DF")

# 3.27 [28] same position on M10 at three noise levels
q = COM[(COM.front_end == 'DF') & (COM.combo == 'D_r0.75_p8x4') & (COM.surface == 'M10_curv') & (COM.phase_label == 'ph0')]
rows = [q[(q.sigma == s) & (q.rep == (0 if s == 0 else 1))].iloc[0] for s in [0, 40, 100]]
cloud(rows, 'combo', 27, [r"$\sigma_\mathrm{n}$ = " + str(s) for s in [0, 40, 100]], "M10, r 0.75 R8x4, phase 0")

# input check against the reference record
META["inputs"] = SOURCE
META["submission_manifest_verified"] = True
(OUTDIR / "kap3_record.json").write_text(json.dumps(META, ensure_ascii=False, indent=2), encoding="utf-8")
print("figures:", len(META["figures"]), "| inputs:", len(SOURCE))
