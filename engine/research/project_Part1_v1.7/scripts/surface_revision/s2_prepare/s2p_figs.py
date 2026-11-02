# Plot spacing-layout comparisons and representative reconstructions.

from __future__ import annotations

import numpy as np
import pandas as pd

import s2p_common as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch

FIG = K.OUT / 'figures'
SURF_DE = {'M10_curv': 'M10 + Zusatzkrümmung', 'M24_curv': 'M24 + Zusatzkrümmung', 'B_curv': 'Ebene Fahrbahn B + Zusatzkrümmung', 'M10_curv_gauss': 'M10 + Gauß-Erhebung (Ergänzung)'}
SURF_COL = {'M10_curv': '#2a78d6', 'M24_curv': '#eb6834', 'B_curv': '#1baf7a', 'M10_curv_gauss': '#eda100'}
SURF_MK = {'M10_curv': 'o', 'M24_curv': 's', 'B_curv': '^', 'M10_curv_gauss': 'D'}
DOM_DE = {'loc': 'lokaler Bereich (r ≤ 15 mm)', 'glob': 'gemeinsamer Messbereich'}
MET_DE = {'Ctau': 'Abdeckung C$_\\tau$ (d 6 mm, τ 0,15 mm)', 'pt_rmse': 'Punkt-RMSE [mm]', 'pt_p95': 'Punkt-P95 [mm]', 'surf_rmse': 'Flächen-RMSE [mm]', 'recovery': 'Wiederfindung',
          'cov_interp': 'interpolierbare Abdeckung', 'Cvalid': 'gestützte Abdeckung C$_{valid}$'}
E_LIM = 0.3
C_NOINT, C_NOSUP = '#ecebe6', '#a8a69e'
DOT_SERIES = {'p$_\\parallel$ (entlang Epipolar), p$_\\perp$ = 4 mm': (['R6x4', 'R7.35x4', 'R8x4', 'R10x4', 'R12x4'], 0),
              'p$_\\perp$ (Zeilenabstand), p$_\\parallel$ = 8 mm': (['R8x2.5', 'R8x3', 'R8x4', 'R8x6', 'R8x8'], 1)}


def _style():
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6, 'savefig.dpi': 200,
                         'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.5, 'figure.facecolor': 'white'})


def _de(x, nd=3):
    try:
        return f'{float(x):.{nd}f}'.replace('.', ',')
    except (TypeError, ValueError):
        return 'undef.'


def save(fig, name):
    d = K.guard(FIG); d.mkdir(parents=True, exist_ok=True)
    p = d / f'{name}.png'; fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    return p


def _lat_pitch(lat, i):
    a, b = lat[1:].split('x'); return float(a) if i == 0 else float(b)


def fig_curves(cond: pd.DataFrame, family: str, metric: str, dom: str, sigma: float, surfaces=K.MAIN):
    _style()
    col = f'{dom}_{metric}'; c = cond[(cond.metric == col) & (cond.family == family) & (cond.sigma == sigma) & (cond.regime == 'equal_integral')]
    inc = c[c.state != 'complete']; c = c[c.state == 'complete']
    if family == 'point':
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), gridspec_kw=dict(width_ratios=[1, 1, 0.6]))
        for ax, (title, (lats, i)) in zip(axes[:2], DOT_SERIES.items()):
            for s in surfaces:
                x = c[(c.surface == s) & c.layout.isin(lats)].copy(); x['p'] = [_lat_pitch(l, i) for l in x.layout]; x = x.sort_values('p')
                ax.plot(x.p, pd.to_numeric(x['mean']), marker=SURF_MK[s], color=SURF_COL[s], label=SURF_DE[s])
                ax.fill_between(x.p, pd.to_numeric(x['min']), pd.to_numeric(x['max']), color=SURF_COL[s], alpha=0.10, lw=0)
            ax.set_xlabel(title.split(',')[0] + ' [mm]'); ax.set_title(title, fontsize=6.5)
        ax = axes[2]
        for j, s in enumerate(surfaces):
            for k, lat in enumerate(('R8x4', 'T8x4')):
                v = c[(c.surface == s) & (c.layout == lat)]
                if len(v):
                    ax.plot([k + 0.1 * (j - 1)], [float(v['mean'].iloc[0])], marker=SURF_MK[s], color=SURF_COL[s], ls='none')
        ax.set_xticks([0, 1], ['R (regulär)', 'T (versetzt)']); ax.set_xlim(-0.5, 1.5); ax.set_title('Anordnung 8 × 4 mm', fontsize=6.5)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4))
        for ax, phi in zip(axes, (0.0, 45.0)):
            for s in surfaces:
                x = c[(c.surface == s) & c.layout.str.endswith(f'_phi{phi:g}')].copy(); x['p'] = [float(l.split('_')[0][1:]) for l in x.layout]; x = x.sort_values('p')
                ax.plot(x.p, pd.to_numeric(x['mean']), marker=SURF_MK[s], color=SURF_COL[s], label=SURF_DE[s])
                ax.fill_between(x.p, pd.to_numeric(x['min']), pd.to_numeric(x['max']), color=SURF_COL[s], alpha=0.10, lw=0)
            ax.set_xlabel('Linienabstand s (Normale) [mm]'); ax.set_title(f'Orientierung φ = {phi:g}°', fontsize=6.5)
    axes[0].set_ylabel(MET_DE.get(metric, metric)); axes[0].legend(frameon=False, fontsize=5.5)
    fig.suptitle(f"{'Punkte' if family == 'point' else 'Linien'} · {DOM_DE[dom]} · σ = {sigma:g} · gleiches Integral Q = 11 695 (Mittel über 8 Phasen, Band = min–max; "
                 f"{len(inc)} unvollständige Zellen nicht gezeichnet)", fontsize=7, y=1.02)
    return fig


def fig_budget(bud: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4))
    d = bud[bud.family == 'point']
    for (title, (lats, i)), mk in zip(DOT_SERIES.items(), ('o', 's')):
        x = d[d.layout.isin(lats)].copy(); x['p'] = [_lat_pitch(l, i) for l in x.layout]
        g = x.groupby('p').agg(n=('n_features', 'median'), pk=('peak_equal_integral', 'median'), pmin=('peak_equal_integral', 'min'), pmax=('peak_equal_integral', 'max')).reset_index()
        axes[0].plot(g.p, g.n, marker=mk, label=title.split(',')[0]); axes[1].errorbar(g.p, g.pk, yerr=[g.pk - g.pmin, g.pmax - g.pk], marker=mk, capsize=2, label=title.split(',')[0])
    axes[0].set_xlabel('Abstand [mm]'); axes[0].set_ylabel('Punktanzahl (Muster)'); axes[1].set_xlabel('Abstand [mm]'); axes[1].set_ylabel('Peak = Q / ΣI$_0$ (Punkte)')
    l = bud[bud.family == 'line'].copy(); l['s'] = [float(x.split('_')[0][1:]) for x in l.layout]; l['phi'] = [x.split('_phi')[1] for x in l.layout]
    for phi, g in l.groupby('phi'):
        gg = g.groupby('s').agg(pk=('peak_equal_integral', 'median')).reset_index(); axes[2].plot(gg.s, gg.pk, marker='o', label=f'φ = {phi}°')
    axes[2].set_xlabel('Linienabstand s [mm]'); axes[2].set_ylabel('Peak (Linien, w 1 mm)')
    for ax in axes:
        ax.legend(frameon=False, fontsize=5.5)
    fig.suptitle('Ressourcenprüfung bei fester Größe: Q = 11 695; Median über Positionen / Phasen, Balken = min–max', fontsize=6.5, y=1.02)
    fig.subplots_adjust(wspace=0.38)
    return fig


def _case_data(row) -> dict:

    rp = K.load_json(K.OUT / 'run' / 'raw' / f"{row['decode_key']}.json")
    with np.load(K.OUT / 'run' / 'images' / f"{row['image_key']}.npz") as z:
        img = np.asarray(z['image'])
    with np.load(K.OUT / 'run' / 'eval' / f"{row['decode_key']}.npz") as z:
        E = {k: np.asarray(z[k]) for k in z.files}
    return dict(img=img, E=E, image_source=rp['image_source'])


# Draw camera images and point clouds for the selected spacing cases.
def fig_group(cases: pd.DataFrame, title: str, stem: str, n_draw=5000, band=1.0):

    import s1r_common as S1K
    _style(); S = S1K.s2(); dom = S1K.global_domain(0.5); gx, gy = np.asarray(dom['x']), np.asarray(dom['y'])
    n = len(cases); fig = plt.figure(figsize=(12.0, 3.1 * n)); check = {}
    Hmin, Hmax = np.inf, -np.inf; meshes = {}
    for _, r in cases.iterrows():
        sid = K.PLATFORM_ID[r.surface]; mesh = meshes.setdefault(sid, S.build_mesh(sid))
        m = (np.abs(mesh['X']) <= 46) & (np.abs(mesh['Y']) <= 46); H = K.C_REF - mesh['Z'][m]; Hmin = min(Hmin, H.min()); Hmax = max(Hmax, H.max())
    sc2 = None
    for i, (_, r) in enumerate(cases.iterrows()):
        sid = K.PLATFORM_ID[r.surface]; mesh = meshes[sid]; D = _case_data(r); E = D['E']
        v = E['valid']; P = E['pts'][v]; dH = -(P[:, 2] - E['zr'][v].astype(float))
        rng = np.random.default_rng(0); sel = rng.choice(len(P), min(n_draw, len(P)), replace=False) if len(P) else np.zeros(0, int)
        interp = E['glob_node_interpolable']; sup = E['glob_node_supported']; acc = E['glob_node_accurate']; dHn = -E['glob_node_err'].astype(float)

        n_nodes = len(gx); Cv = float(sup.sum()) / n_nodes; Ct = float(acc.sum()) / n_nodes
        eg = -(E['pts'][E['in_global'], 2] - E['zr'][E['in_global']].astype(float)); rm = float(np.sqrt(np.mean(eg ** 2))) if len(eg) else None

        chk = dict(Cvalid=abs(Cv - float(r.glob_Cvalid or 0.0)) <= 1e-12, Ctau=abs(Ct - float(r.glob_Ctau or 0.0)) <= 1e-12,
                   pt_rmse=(rm is None and pd.isna(r.glob_pt_rmse)) or (rm is not None and abs(rm - float(r.glob_pt_rmse)) <= 1e-6))
        check[r.cand_id] = dict(chk, ok=all(chk.values()))
        ax0 = fig.add_subplot(n, 4, 4 * i + 1); ax0.imshow(D['img'], cmap='gray', vmin=0, vmax=255); ax0.set_axis_off()
        ax0.set_title(f"{r.cand_id} · {SURF_DE[r.surface].split(' +')[0]} · {r.layout} · σ {r.sigma:g}\nKamerabild ({D['image_source'].replace('S2 ', '')})", fontsize=6)
        ax1 = fig.add_subplot(n, 4, 4 * i + 2, projection='3d')
        X, Y, Z = mesh['X'], mesh['Y'], mesh['Z']; ix = np.flatnonzero(np.abs(X[0, :]) <= 46); iy = np.flatnonzero(np.abs(Y[:, 0]) <= 46)
        Xs, Ys, Zs = (A[np.ix_(iy[::16], ix[::16])] for A in (X, Y, Z))
        ax1.plot_wireframe(-Xs, Ys, K.C_REF - Zs, color='#a8a69e', linewidth=0.3, rstride=2, cstride=2)
        if len(sel):
            ax1.scatter(-P[sel, 0], P[sel, 1], K.C_REF - P[sel, 2], s=3, c=np.clip(dH[sel], -E_LIM, E_LIM), cmap='RdBu_r', vmin=-E_LIM, vmax=E_LIM, depthshade=False)
        ax1.set_xlim(-46, 46); ax1.set_ylim(-46, 46); ax1.set_zlim(Hmin - 1, Hmax + 1); ax1.view_init(elev=28, azim=-60)
        ax1.set_xlabel('x$_p$'); ax1.set_ylabel('y$_p$'); ax1.set_title(f'Referenz (Gitter) + Punkte, H [mm]\n{len(sel)} / {len(P)} gültige Punkte', fontsize=6)
        ax2 = fig.add_subplot(n, 4, 4 * i + 3)
        ax2.scatter(-gx[~interp], gy[~interp], s=0.15, c=C_NOINT, marker='s', lw=0)
        ax2.scatter(-gx[interp & ~sup], gy[interp & ~sup], s=0.15, c=C_NOSUP, marker='s', lw=0)
        sc2 = ax2.scatter(-gx[sup], gy[sup], s=0.15, c=np.clip(dHn[sup], -E_LIM, E_LIM), cmap='RdBu_r', vmin=-E_LIM, vmax=E_LIM, marker='s', lw=0)
        ax2.add_patch(Circle((0, 0), 15.0, fill=False, ls='--', lw=0.6, color='#52514e'))
        if len(P) == 0:
            ax2.text(0, 30, 'keine gültige Ausgabe\n(wissenschaftliches Ergebnis, kein technischer Fehler)', ha='center', va='center', fontsize=6, color='#52514e')
        n_out = int((np.abs(dHn[sup]) > E_LIM).sum()); n_tau = int((sup & ~acc).sum())
        ax2.set_aspect('equal'); ax2.set_xlim(-47, 47); ax2.set_ylim(-47, 47); ax2.set_xlabel('x$_p$ [mm]'); ax2.set_ylabel('y$_p$ [mm]')
        ax2.set_title(f"C$_{{valid}}$ {_de(Cv)} · C$_\\tau$ {_de(Ct)}\n|ΔH| > τ: {n_tau} · über ±0,3 mm: {n_out} Knoten\nnicht interp. {int((~interp).sum())} · ohne Stützung {int((interp & ~sup).sum())}", fontsize=6)
        ax3 = fig.add_subplot(n, 4, 4 * i + 4)
        j = int(np.argmin(np.abs(X[0, :]))); ys = Y[:, j]; keep = np.abs(ys) <= 46
        ax3.plot(ys[keep], K.C_REF - Z[keep, j], color='#52514e', lw=1, label='Referenz (Schnitt x$_p$ = 0)')
        col0 = np.abs(gx) < 1e-9; o = np.argsort(gy[col0]); zest = E['glob_node_z_est'][col0][o].astype(float); okn = interp[col0][o]
        ax3.plot(gy[col0][o][okn], K.C_REF - zest[okn], color='#eb6834', lw=0.9, label='interpolierte Rekonstruktion (Knoten)')
        b = np.abs(P[:, 0]) <= band
        ax3.plot(P[b, 1], K.C_REF - P[b, 2], 'o', ms=1.2, color='#2a78d6', label=f'Punkte |x$_p$| ≤ {_de(band, 1)} mm (roh)')
        ax3.set_xlabel('y$_p$ [mm]'); ax3.set_ylabel('H [mm]'); ax3.set_title(f"Mittelschnitt x$_p$ = 0\nPunkt-RMSE (gem. Bereich) {_de(rm)} mm", fontsize=6)
        if i == 0:
            ax3.legend(frameon=False, fontsize=5)
    if sc2 is not None:
        cax = fig.add_axes([0.955, 0.25, 0.008, 0.5])
        cb = fig.colorbar(sc2, cax=cax, extend='both')
        cb.set_label('ΔH = H$_{rek}$ − H$_{ref}$ [mm]  (rot: Höhe überschätzt, blau: unterschätzt)', fontsize=6)
    fig.legend(handles=[Patch(color=C_NOINT, label='nicht interpolierbar'), Patch(color=C_NOSUP, label='interpolierbar, ohne Stützung (> 6 mm)'),
                        Patch(facecolor='white', edgecolor='#52514e', ls='--', label='vorgegebener Bewertungsbereich r ≤ 15 mm (keine Verformungsgrenze)')],
               loc='lower center', ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'{title}\nEinzelfälle (keine Phasenmittel) · gemeinsamer Messbereich · gleiches Integral Q = 11 695 · S2-PREP-02 Verbundtest', fontsize=7.5, y=1.0)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.08, top=0.86, wspace=0.42, hspace=0.75)
    return save(fig, stem), check


def panel_rows(df: pd.DataFrame, surface: str, sigma: float, panel: str) -> pd.DataFrame:

    return df[(df.surface == surface) & (df.sigma == sigma) & df.panel.str.contains(panel, regex=False)].sort_values(['layout'])
