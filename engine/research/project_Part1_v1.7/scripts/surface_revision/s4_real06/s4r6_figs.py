# Plot layout and clipping comparisons from saved analysis tables.

from __future__ import annotations

import numpy as np
import pandas as pd

import s4r6_common as H

K = H.K
SRC = 'Quelle: S4-REAL-06 (Technik-Pilot, Entwicklungsbedingungen B000 / H0–H1 / σ 0 und 40), s4r6_figs.py'
SN = {'B_curv': 'B', 'M10_curv': 'M10', 'M24_curv': 'M24'}
LAYLAB = {'regular': 'S0', 'j0.10_B000_c0': 'S1 j0,10', 'j0.30_B000_c7': 'S2* j0,30', 'j0.30_B000_c25': 'S4* j0,30', 'j0.20_B000_c2': 'S1-RMS j0,20', 'j0.20_B000_c1': 'S3 j0,20',
          'j0.30_B000_c39': 'S2* j0,30', 'j0.30_B000_c134': 'S4* j0,30', 'j0.20_B000_c0': 'S1-RMS j0,20', 'j0.20_B000_c63': 'S3 j0,20'}
ORDER = ['S0', 'S1 j0,10', 'S2* j0,30', 'S4* j0,30', 'S1-RMS j0,20', 'S3 j0,20']
COL = dict(zip(ORDER, ['#4d4d4d', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']))


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7.0, 'axes.titlesize': 7.0, 'axes.labelsize': 6.8, 'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2, 'legend.fontsize': 5.8, 'font.family': 'DejaVu Sans'})
    return plt


def _de(v, nd=2):
    return '–' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{nd}f}'.replace('.', ',')


def _res():
    return pd.read_csv(H.OUT / 'results.csv', low_memory=False, float_precision='round_trip')


def fig1() -> str:
    plt = _plt(); R = _res(); P = R[(R.record_type == 'clip') & (R.domain == 'glob')].copy()
    P['lay'] = P.layout.map(LAYLAB); P['removed'] = (P.old_u + P.old_w) - (P.new_u + P.new_w); P['gross_removed'] = P.old_n_gross - P.new_n_gross; P['dC'] = 100 * P.d_Ctau
    surfs = ['B_curv', 'M10_curv', 'M24_curv']; grp = [(s, ph) for s in surfs for ph in ('H0', 'H1')]
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.9))
    for gi, (s, ph) in enumerate(grp):
        for li, lab in enumerate(ORDER):
            for sg, mk in ((0.0, 'o'), (40.0, '^')):
                x = P[(P.surface == s) & (P.phase == ph) & (P.lay == lab) & (P.sigma == sg)]
                if not len(x):
                    continue
                xx = gi + (li - 2.5) * 0.11 + (0.03 if sg else -0.03); r = x.iloc[0]
                axs[0].scatter(xx, r.removed + r.gross_removed, s=9, marker=mk, c=COL[lab], lw=0)
                axs[1].scatter(xx, r.old_correct_lost, s=9, marker=mk, c=COL[lab], lw=0)
                axs[2].scatter(xx, r.dC, s=9, marker=mk, c=COL[lab], lw=0)
    for ax, t in zip(axs, ('(a) abgelehnte unbestätigbare\n+ grobe Punkte (global)', '(b) verlorene bestätigt\nkorrekte Punkte (global)', '(c) Δ C_τ global / pp\n(Kandidat − P4 alt)')):
        ax.set_xticks(range(len(grp))); ax.set_xticklabels([f'{ph}\n{"Band" if ph == "H0" else "ohne"}' for s, ph in grp], fontsize=5.0)
        for si, s in enumerate(surfs):
            ax.text(2 * si + 0.5, -0.2, SN[s], transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.0)
        ax.axhline(0, c='#888888', lw=0.5); ax.set_title(t)
        for b in (1.5, 3.5):
            ax.axvline(b, c='#dddddd', lw=0.5)
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], marker='o', ls='', c=COL[l], ms=3, label=l) for l in ORDER] + [Line2D([], [], marker='o', ls='', c='#999999', ms=3, label='σ = 0'),
                                                                                         Line2D([], [], marker='^', ls='', c='#999999', ms=3, label='σ = 40')]
    fig.legend(handles=hd, loc='lower center', ncol=8, bbox_to_anchor=(0.5, -0.11), frameon=False)
    fig.suptitle('Linien L6,3: Randbandkandidat gegen P4 (gleiches Bild) – Fehlerabbau und Verlust korrekter Abdeckung', y=1.02)
    fig.text(0.01, -0.17,'Jeder Punkt = ein Bildzustand (Layout × σ). H0: Randband −7 beleuchtet; H1: kein Randband. ' + SRC, fontsize=5.0, color='#555555')
    fig.tight_layout()
    p = H.FIGS / 'F1_Abwaegung_Fehler_Abdeckung.png'; H.guard(p); fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    return K.rel(p)


def fig2() -> str:
    plt = _plt(); from matplotlib.colors import LogNorm
    import r2_eval as E
    import rdy_platform as P
    import s4r6_run as R
    R.set_read_only(True)
    ex = H.run_load()['analysis']['examples']; norm = LogNorm(1e-2, 2e1)
    rows = [('Verbesserung (größte Abnahme grober Punkte)', ex['improvement']), ('Kosten (größter C_τ-Verlust)', ex['cost']), ('Kontrolle ohne Randband (H1)', ex['control'])]
    fig = plt.figure(figsize=(6.3, 9.6)); sc = None; row_titles = []
    for i, (title, e) in enumerate(rows):
        c = R.cond(e['image_key']); dom = E.domain('omega1'); sid = K.RK.SURF_MAP[c['surface']]
        X, Y, Z = np.asarray(dom['x']), np.asarray(dom['y']), np.asarray(dom[f'z_{sid}'], np.float64); sub = np.arange(0, len(X), 7)
        dl = P.dom_local(c['surface']); cx, cy = float(np.mean(dl['x'])), float(np.mean(dl['y'])); rr = float(np.max(np.hypot(np.asarray(dl['x']) - cx, np.asarray(dl['y']) - cy)))
        data = []
        for m in ('P4TABLE', 'CLIP'):
            ev = R.evaluate(dict(c, method=m), m); A = K.load_npz(H.EVL6 / f"{ev['identity']}.npz")
            pts = np.asarray(A['pts'], np.float64); val = np.asarray(A['valid'], bool); err = np.abs(pts[:, 2] - np.asarray(A['zr'], np.float64)); data.append((m, ev, pts, val, err))
        allp = np.vstack([np.column_stack([X, Y, Z])] + [d[2][d[3]] for d in data])
        lo, hi = np.nanmin(allp, axis=0), np.nanmax(allp, axis=0)
        for j, (m, ev, pts, val, err) in enumerate(data):
            ax = fig.add_subplot(3, 2, 2 * i + j + 1, projection='3d')
            ax.scatter(X[sub], Y[sub], Z[sub], s=0.05, c='#bbbbbb', alpha=0.35, rasterized=True)
            sc = ax.scatter(pts[val, 0], pts[val, 1], pts[val, 2], s=0.25, c=np.clip(err[val], 1e-2, 2e1), cmap='viridis', norm=norm, rasterized=True)
            th = np.linspace(0, 2 * np.pi, 120); zc = np.full_like(th, float(lo[2]))
            ax.plot(cx + rr * np.cos(th), cy + rr * np.sin(th), zc, c='#d62728', lw=0.6)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2] - 1, hi[2] + 1); ax.view_init(elev=28, azim=-60)
            ax.set_xlabel('x / mm', fontsize=5.5, labelpad=-6); ax.set_ylabel('y / mm', fontsize=5.5, labelpad=-6); ax.set_zlabel('z / mm', fontsize=5.5, labelpad=-6); ax.tick_params(labelsize=4.5, pad=-2)
            g = ev['metrics']
            ax.set_title(f"{'P4 alt' if m == 'P4TABLE' else 'Randbandkandidat'}: C_τ {_de(100 * float(g['glob_Ctau'] or 0), 1)} %, C_valid {_de(100 * float(g['glob_Cvalid'] or 0), 1)} %\n"
                         f"grobe Punkte {g['glob_n_gross']}, gestützte RMSE {_de(ev['nodes']['glob']['sup_surf_rmse'], 3)} mm", fontsize=5.6)
            if j == 0:
                row_titles.append((ax, f"{title}: L6,3 {LAYLAB.get(c['layout'], c['layout'])}, {SN[c['surface']]}, {c['phase']}, σ = {int(float(c['sigma']))}"))
    fig.subplots_adjust(top=0.93, bottom=0.05, hspace=0.5)
    for ax, t in row_titles:
        fig.text(0.02, ax.get_position().y1 + 0.045, t, fontsize=6.4, weight='bold')
    cb = fig.colorbar(sc, ax=fig.axes, shrink=0.4, pad=0.04); cb.set_label('|z_rekonstruiert − z_wahr| / mm (gemeinsame Skala)')
    fig.text(0.01, 0.005, 'Grau: wahre Oberfläche (Ω1-Knoten, ausgedünnt); farbig: rekonstruierte gültige Punkte; rot: lokaler Kreis (Projektion). Gleiche Ansicht und Achsen je Zeile.\n' + SRC,
             fontsize=5.0, color='#555555')
    p = H.FIGS / 'F2_Rekonstruktion_3D.png'; H.guard(p); fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    return K.rel(p)


def fig3() -> str:
    plt = _plt(); R = _res(); C = R[(R.record_type == 'cond') & (R.domain == 'glob')].copy(); C['lay'] = C.layout.map(LAYLAB)
    BR = pd.read_csv(H.R5 / '_parts' / 'B_rasters.csv', low_memory=False)
    fam_lay = {'R8x4': ['regular', 'j0.10_B000_c0', 'j0.30_B000_c7', 'j0.30_B000_c25', 'j0.20_B000_c2', 'j0.20_B000_c1'],
               'L6.3': ['regular', 'j0.10_B000_c0', 'j0.30_B000_c39', 'j0.30_B000_c134', 'j0.20_B000_c0', 'j0.20_B000_c63']}
    BR = BR[[l in fam_lay[b] for b, l in zip(BR.baseline, BR.layout)]].copy(); BR['lay'] = BR.layout.map(LAYLAB)
    C = C[[l in fam_lay[b] for b, l in zip(C.baseline, C.layout)]].copy()
    reg = __import__('json').loads((H.R2 / 'layouts.json').read_text(encoding='utf-8'))
    fig, axs = plt.subplots(3, 2, figsize=(6.3, 6.6))
    for ci, (base, fam) in enumerate((('R8x4', 'point'), ('L6.3', 'line'))):
        p0, h0 = float(reg['baselines'][base]['p0_mm']), float(reg['baselines'][base]['h0_mm'])
        b = BR[(BR.baseline == base) & BR.lay.isin(ORDER) & BR.phase.isin(['H0', 'H1'])]
        ax = axs[0, ci]
        for li, lab in enumerate(ORDER):
            x = b[b.lay == lab]
            if not len(x):
                continue
            ax.scatter([li - 0.12], [x.g_min_mm.iloc[0] / p0], marker='s', s=12, c=COL[lab]); ax.scatter([li + 0.12], [x.h_mm.iloc[0] / h0], marker='D', s=12, c=COL[lab])
            ax.scatter([li - 0.12] * len(x), x.g_min_raster_mm / p0, marker='_', s=30, c='k', lw=0.6); ax.scatter([li + 0.12] * len(x), x.h_raster_mm / h0, marker='_', s=30, c='k', lw=0.6)
        ax.axhline(0.80, c='#ff7f0e', lw=0.5, ls='--'); ax.axhline(1.15, c='#2ca02c', lw=0.5, ls='--')
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=30, fontsize=5.2); ax.set_ylabel('g_min / p0 (■), h / h0 (◆)')
        ax.set_title(f"{'Punkte R8×4' if fam == 'point' else 'Linien L6,3'}: Geometrie (Strich = Raster H0 / H1)")
        for ri, (col, lab_y) in enumerate((('Ctau', 'C_τ global / %'), ('sup_rmse', 'gestützte RMSE / mm'))):
            ax = axs[ri + 1, ci]
            for li, lab in enumerate(ORDER):
                for si, s in enumerate(('B_curv', 'M10_curv', 'M24_curv')):
                    for sg, mk in ((0.0, 'o'), (40.0, '^')):
                        for mi, m in enumerate(['D0'] if fam == 'point' else ['P4TABLE', 'CLIP']):
                            x = C[(C.baseline == base) & (C.lay == lab) & (C.surface == s) & (C.sigma == sg) & (C.method == m)]
                            if not len(x):
                                continue
                            v = (100 * x[col]) if col == 'Ctau' else x[col]
                            xx = li + (si - 1) * 0.22 + (0.05 if sg else -0.05) + (0.03 * mi)
                            ax.scatter([xx] * len(v), v, s=5, marker=mk, c=COL[lab], edgecolors=('none' if mi == 0 else 'k'), lw=0.3)
            ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=30, fontsize=5.2); ax.set_ylabel(lab_y)
            if col == 'sup_rmse':
                ax.set_yscale('log')
            ax.set_title('Abdeckung C_τ' if col == 'Ctau' else 'Fehler gestützter Knoten')
    fig.suptitle('Layouts und Strategien (Entwicklungsblock B000) mit S0 und RMS-Partner S3 – Punkte und Linien getrennt', y=1.0)
    fig.text(0.01, -0.03, 'Je Layout von links: B / M10 / M24, jeweils H0 und H1. Kreis σ = 0, Dreieck σ = 40; Linien: ohne Rand = P4 alt, schwarzer Rand = Randbandkandidat.\n'
             'Gestrichelt: α 0,80 / β 1,15. Keine Gesamtrangfolge. ' + SRC, fontsize=5.0, color='#555555')
    fig.tight_layout()
    p = H.FIGS / 'F3_Layouts_Strategien.png'; H.guard(p); fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    return K.rel(p)


def make_all() -> dict:
    files = [fig1(), fig2(), fig3()]
    out = dict(files=files, hashes={f: K.sha_file(K.ROOT / f) for f in files}, t_local=K.now_local())
    H.run_put('figures', out)
    return out
