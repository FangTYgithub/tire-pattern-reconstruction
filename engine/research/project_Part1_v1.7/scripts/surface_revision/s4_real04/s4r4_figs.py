# Plot local reconstruction and layout-generation diagnostics.

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import s4r4_common as H

K = H.K
SRC = 'Quelle: S4-REAL-04 (Diagnose ohne neue Bildgebung / Decodierung) auf S4-REAL-02-Daten, s4r4_figs.py'


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7.0, 'axes.titlesize': 7.0, 'axes.labelsize': 6.8, 'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2, 'legend.fontsize': 5.8, 'font.family': 'DejaVu Sans'})
    return plt


def _de(v, nd=2):
    return f'{v:.{nd}f}'.replace('.', ',')


def _save(fig, name):
    p = H._guard(H.FIGS / f'{name}.png'); fig.savefig(p, dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


def fig_local():
    import r2_eval as E
    import rdy_platform as P
    from surface_eval_r3 import covering_radius_exact, dedup_xy, triangulate_xy
    plt = _plt(); r = H.run_load(); A = r['A']['anchor_local']; ek = H.CONFIG['A']['anchor_eval']
    z = K.load_npz(H.reg(K.EVL / f'{ek}.npz')); valid = z['valid'].astype(bool); pts = z['pts'].astype(np.float64)
    anom = valid & z['accepted'].astype(bool) & z['point_unconf'].astype(bool)
    Pd, _ = dedup_xy(pts[valid], None); ka = set(map(tuple, np.round(pts[anom, :2] / 1e-6).astype(np.int64)))
    flag = np.array([tuple(k) in ka for k in np.round(Pd[:, :2] / 1e-6).astype(np.int64)]); tri, _r = triangulate_xy(Pd)
    dl = P.dom_local('M10_curv'); q = np.column_stack([dl['x'], dl['y']]); s = tri.find_simplex(q); ins = s >= 0
    has = np.zeros(len(q), bool); has[ins] = flag[tri.simplices[s[ins]]].any(axis=1)
    T = tri.transform[s[ins]]; rr = q[ins] - T[:, 2, :]; bc = np.einsum('nij,nj->ni', T[:, :2, :], rr); w = np.column_stack([bc, 1 - bc.sum(1)])
    zest = np.full(len(q), np.nan); zest[ins] = (w * Pd[tri.simplices[s[ins]], 2]).sum(1); e = zest - np.asarray(dl['z_ref'], np.float64)
    Rt = covering_radius_exact(Pd[:, :2], tri.simplices); Rn = np.full(len(q), np.inf); Rn[ins] = Rt[s[ins]]; sup = np.isfinite(e) & (Rn <= K.D_SUPPORT)
    fig = plt.figure(figsize=(6.3, 3.8)); gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.35, left=0.08, right=0.97, top=0.82, bottom=0.22)
    ax = fig.add_subplot(gs[0, 0])
    lt = np.flatnonzero(flag[tri.simplices].any(axis=1)); cen = Pd[tri.simplices[lt], :2].mean(axis=1); near = np.hypot(cen[:, 0], cen[:, 1]) < 40
    for t in lt[near][::3]:
        v = Pd[tri.simplices[t], :2]; ax.plot(np.r_[v[:, 0], v[0, 0]], np.r_[v[:, 1], v[0, 1]], color='#e3a35a', lw=0.2, alpha=0.6)
    ax.scatter(q[~has & sup, 0], q[~has & sup, 1], s=0.3, c='#4c72b0', lw=0, label='gestützt (d = 6 mm), ohne Fremdpunkt')
    ax.scatter(q[~has & ~sup & ins, 0], q[~has & ~sup & ins, 1], s=0.3, c='#c8c8c8', lw=0, label='interpolierbar, nicht gestützt')
    ax.scatter(q[has, 0], q[has, 1], s=0.3, c='#d55e00', lw=0, label='in Dreiecken mit Fremdpunkt (alle nicht gestützt)')
    ax.scatter(pts[anom, 0], pts[anom, 1], s=0.6, c='#8b0000', lw=0, label='akzeptierte Punkte aus Randlinie −7')
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(15 * np.cos(th), 15 * np.sin(th), color='#2ca02c', lw=0.9)
    ax.set_xlim(-45, 45); ax.set_ylim(-45, 30); ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title(f"(a) lokaler Kreis r ≤ 15 mm: {A['n_link_nodes']} von {A['n_local_nodes']} Knoten\nin Dreiecken mit Fremdpunkt außerhalb des Kreises")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=5.2, markerscale=8, ncol=2)
    ax = fig.add_subplot(gs[0, 1])
    vals = [('Punkt-RMSE lokal\n(gültige Punkte)', A['stored_loc_pt_rmse']), ('Flächen-RMSE\ngestützte Knoten', A['stored_loc_sup_surf_rmse']),
            ('Flächen-RMSE interpolierbar\nohne Fremddreiecke', A['surf_rmse_interpolable_nonlink']), ('Flächen-RMSE\nalle interpolierbaren', A['surf_rmse_interpolable'])]
    y = np.arange(len(vals)); ax.barh(y, [v for _, v in vals], color=['#55a868', '#4c72b0', '#9a9a9a', '#d55e00'])
    for i, (_, v) in enumerate(vals):
        ax.text(v * 1.15, i, _de(v, 4) + ' mm', va='center', fontsize=5.8)
    ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels([n for n, _ in vals], fontsize=5.8); ax.set_xlim(0.005, 30); ax.set_xlabel('mm (log)')
    ax.set_title(f"(b) Punkt-, Stütz- und Interpolationsfehler\nlokales C_τ = {_de(100 * A['stored_loc_Ctau'], 1)} % (offizieller Wert)")
    fig.suptitle('Linien L6,3, M10, S0, H0, σ = 0: Fremdpunkte außerhalb des Kreises\nverändern die Interpolation im lokalen Kreis', y=0.985)
    fig.text(0.01, 0.01, f"Bewertungsschlüssel {ek}. Anteil der Fremddreieck-Knoten am quadratischen Flächenfehler: {_de(100 * A['share_sq_err_link'], 3)} %.\n"
             'Gleicher Befund in 18 von 76 betroffenen Bildern (M10, H0, σ 0/20/40). ' + SRC, fontsize=5.0, color='#555555')
    return _save(fig, 'G1_Lokale_Interpolation')


def fig_generation():
    plt = _plt()
    D = pd.read_csv(H.reg(H.OUT / '_parts' / 'C1_geometry.csv'), low_memory=False); C = D[(D.table_type == 'C1_cell') & D.alpha.notna()]
    fig, axs = plt.subplots(2, 4, figsize=(6.3, 4.2))
    js = sorted(C.j.unique()); al = sorted(C.alpha.unique()); be = sorted(C.beta.unique())
    import matplotlib.colors as mcolors
    for i, name in enumerate(('R8x4', 'L6.3')):
        x = C[C.baseline == name]
        specs = [('change_S2_vs_S1', 'alpha', al, 'S2 ≠ S1 (β 1,15)', 1.15), ('change_S3_vs_S1', 'beta', be, 'S3 ≠ S1 (α 0,70)', 0.70),
                 ('change_S4_vs_S3', 'beta', be, 'S4 ≠ S3 (α 0,80)', 0.80), ('failed_S4', 'beta', be, 'S4 ohne Layout (α 0,80)', 0.80)]
        for jx, (col, axis_var, vals, ttl, fixed) in enumerate(specs):
            ax = axs[i, jx]
            if axis_var == 'alpha':
                sub = x[x.beta == fixed]; M = sub.pivot_table(index='j', columns='alpha', values=col)
            elif col in ('change_S4_vs_S3', 'failed_S4'):
                sub = x[x.alpha == fixed]; M = sub.pivot_table(index='j', columns='beta', values=col)
            else:
                sub = x[x.alpha == fixed]; M = sub.pivot_table(index='j', columns='beta', values=col)
            vmax = 16 if col == 'failed_S4' else 1
            fail_col = {'change_S3_vs_S1': ['failed_S3'], 'change_S4_vs_S3': ['failed_S3', 'failed_S4']}.get(col, [])
            dead = np.zeros(M.shape, bool)
            for fc in fail_col:
                dead |= (sub.pivot_table(index='j', columns=M.columns.name, values=fc).reindex(index=M.index, columns=M.columns).to_numpy() >= 16)
            Mv = np.where(dead, np.nan, M.to_numpy())
            im = ax.imshow(Mv, cmap='viridis' if col != 'failed_S4' else 'Reds', vmin=0, vmax=vmax, aspect='auto', origin='lower')
            for a_ in range(M.shape[0]):
                for b_ in range(M.shape[1]):
                    v = M.to_numpy()[a_, b_]
                    if dead[a_, b_]:
                        ax.text(b_, a_, 'kein\nLayout', ha='center', va='center', fontsize=4.2, color='black'); continue
                    ax.text(b_, a_, (f'{int(v)}' if col == 'failed_S4' else _de(v, 2)), ha='center', va='center', fontsize=4.8, color=('white' if (v / vmax) < 0.55 and col != 'failed_S4' else 'black'))
            ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([_de(c, 2) for c in M.columns], fontsize=5.2); ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([_de(r_, 2) for r_ in M.index], fontsize=5.2)
            ax.set_xlabel('α' if axis_var == 'alpha' else 'β', fontsize=6)
            if jx == 0:
                ax.set_ylabel(f"{'Punkte R8×4' if name == 'R8x4' else 'Linien L6,3'}\nj", fontsize=6.4)
            ax.set_title(ttl, fontsize=6.0)

            try:
                ri = list(M.index).index(0.20); ci = list(M.columns).index(0.70 if axis_var == 'alpha' else 1.15)
                if (axis_var == 'alpha') or (fixed == 0.70):
                    ax.add_patch(__import__('matplotlib.patches', fromlist=['Rectangle']).Rectangle((ci - 0.5, ri - 0.5), 1, 1, fill=False, ec='red', lw=1.0))
            except ValueError:
                pass
    fig.suptitle('Generierung: Wann ändern Mindestabstand (S2), Lückenobergrenze (S3)\nund Kombination (S4) das gewählte Layout?', y=0.99)
    fig.text(0.01, 0.01, 'Anteil der 16 Blöcke (je 1000 registrierte Vorschläge), deren erstes zulässiges Layout sich unterscheidet; Spalte 4: Blöcke ohne zulässiges S4-Layout.\n'
             'Koordinatenprüfung, validiert gegen S4-REAL-02 (≤ 4·10⁻¹⁴ mm); rot = registrierte Einstellung (j 0,20, α 0,70, β 1,15). Keine Bildgebung, keine Leistungsaussage.\n'
             + SRC, fontsize=4.9, color='#555555')
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    return _save(fig, 'G2_Generierung_Unterscheidbarkeit')


def make_all() -> dict:
    p1 = fig_local(); p2 = fig_generation()
    out = dict(files=[K.rel(p1), K.rel(p2)], hashes={K.rel(p): K.sha_file(p) for p in (p1, p2)}, t_local=K.now_local())
    H.run_put('figures', out); H.flush_reads('figures')
    return out
