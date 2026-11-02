# Plot representative encoding diagnostics and failure cases.

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import c8_common as K

LEV = ['weak', 'mid', 'strong']
LC = dict(plain='#222222', weak='#9467bd', mid='#ff7f0e', strong='#8c564b')
LDE = dict(plain='normal', weak='schwach', mid='mittel', strong='stark')
SIZE = dict(dots=dict(plain='r 1,0', weak='r 0,92/1,08', mid='r 0,81/1,16', strong='r 0,53/1,31'), lines=dict(plain='b 1,0', weak='b 0,85/1,15', mid='b 0,73/1,28', strong='b 0,5/1,5'))
SURF = ['B_curv', 'M10_curv', 'M24_curv']; SC_ = dict(B_curv='#1f77b4', M10_curv='#2ca02c', M24_curv='#d62728')
LAY = dict(dots=['R5x4', 'R6x4', 'R8x4'], lines=['4.45', '5.6', '8'])
LAYDE = {'R5x4': 'R5×4 (dicht)', 'R6x4': 'R6×4 (Übergang)', 'R8x4': 'R8×4 (weit)', '4.45': '4,45 mm (dicht)', '5.6': '5,6 mm (Übergang)', '8': '8 mm (weit)'}
FAMDE = dict(dots='Punkte', lines='Linien')
MAIN = dict(dots=('DF', 'use'), lines=('R@1.1247', 'R@1.1247')); PLAIN = dict(dots=('DF', 'single'), lines=('P4', 'P4'))
SIG = [0, 20, 40, 60, 80, 100]
YLIM_RMSE = (0.02, 15.0)
W_IN = 6.3
INDEX = []


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'font.family': 'DejaVu Sans',
                         'pdf.fonttype': 42})
    return plt


def _num(x, nd=3) -> str:

    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return '–'
    return f'{float(x):.{nd}f}'.replace('.', ',')


def _de(ax, logy=False, pct=False):

    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    f = FuncFormatter(lambda v, _: (f'{v:g}' + (' %' if pct else '')).replace('.', ','))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
    if logy:
        lo, hi = ax.get_ylim(); span = np.log10(hi / lo) if lo > 0 else 9
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,) if span > 3.2 else ((1.0, 3.0) if span > 1.2 else (1.0, 2.0, 5.0))))
        ax.yaxis.set_major_formatter(f); ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_formatter(f)


def _save(fig, name, inputs, change, identity, recommended, cases=''):
    import matplotlib.pyplot as plt
    p = K.guard(K.FIGS / f'{name}.png'); p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=220)
    files = [p]
    if recommended:
        q = K.guard(K.FIGS / f'{name}.pdf'); fig.savefig(q, metadata={'CreationDate': None}); files.append(q)
    plt.close(fig)
    INDEX.append(dict(figure=K.rel(p), pdf=(K.rel(files[1]) if len(files) > 1 else ''), inputs=inputs, cases=cases, entry='scripts/surface_revision/s3_code08/c8_figs.py', change=change,
                      identity=identity, recommended_main_text=recommended, sha256_png=K.sha_file(p), sha256_pdf=(K.sha_file(files[1]) if len(files) > 1 else '')))


def _sel(X, **kw):
    m = pd.Series(True, index=X.index)
    for k, v in kw.items():
        m &= (X[k].astype(str) == str(v)) if k == 'layout' else (X[k] == v)
    return X[m]


def _lev_legend(fam, with_plain=True):
    import matplotlib.lines as ml
    return [ml.Line2D([], [], color=LC[l], marker='o', ms=3, label=f'{LDE[l]} ({SIZE[fam][l]} mm)') for l in (['plain'] if with_plain else []) + LEV]


# Plot encoding performance and response curves from the supplied result tables.
def f05(T, fam):
    plt = _plt(); C, SC = T['C'], T['SC']; YL = YLIM_RMSE
    fig, axs = plt.subplots(3, 3, figsize=(W_IN, 8.4), sharex=True); notes = []; clipped = []
    for i, s in enumerate(SURF):
        for j, lay in enumerate(LAY[fam]):
            ax = axs[i, j]; ax.grid(alpha=0.25); ax.set_xticks(SIG); ax.set_xlim(-10, 110); labs = {}
            for lev, (ru, ar) in [('plain', PLAIN[fam])] + [(x, MAIN[fam]) for x in LEV]:
                q = _sel(SC, family=fam, surface=s, layout=lay, pattern=lev, rule=ru, arm=ar).sort_values('sigma')
                y = q.glob_surf_rmse_mean.to_numpy(float); yc = np.where(y < YL[0], YL[0] * 1.08, y)
                ax.plot(q.sigma, yc, '-', color=LC[lev], lw=1.5 if lev == 'plain' else 1.0, marker='o', ms=2.5, zorder=2)
                for r, v in zip(q.itertuples(), yc):
                    if not pd.isna(r.glob_surf_rmse_mean) and r.glob_surf_rmse_mean < YL[0]:
                        ax.scatter([r.sigma], [v], s=30, marker='v', color=LC[lev], zorder=4); clipped.append(dict(surface=s, layout=lay, pattern=lev, sigma=r.sigma, value=float(r.glob_surf_rmse_mean)))
                    if pd.isna(r.glob_surf_rmse_mean) or r.glob_surf_rmse_n_defined >= r.n_rows:
                        continue
                    cc = _sel(C, family=fam, surface=s, layout=lay, pattern=lev, rule=ru, arm=ar, sigma=r.sigma); dfd = cc[cc.glob_surf_rmse.notna()]
                    nmin = int(dfd.glob_n_interp.min())
                    ax.scatter([r.sigma], [v], s=22, facecolors='white', edgecolors=LC[lev], zorder=3)
                    txt = f'{int(r.glob_surf_rmse_n_defined)}/{int(r.n_rows)}' + (f'\n{nmin} Kn.' if nmin < 1000 else '') + (f'\n{_num(r.glob_surf_rmse_mean, 6)}' if r.glob_surf_rmse_mean < YL[0] else '')
                    labs.setdefault(r.sigma, []).append((lev, txt, float(v)))
                    notes.append(dict(family=fam, surface=s, layout=lay, pattern=lev, sigma=r.sigma, defined=int(r.glob_surf_rmse_n_defined), images=int(r.n_rows), min_nodes=nmin,
                                      rmse=float(r.glob_surf_rmse_mean)))
            ax.set_yscale('log'); ax.set_ylim(*YL); _de(ax, logy=True)
            lo, hi = np.log10(ax.get_ylim())
            for sg, L in labs.items():
                up = np.mean([(np.log10(v) - lo) / (hi - lo) for _, _, v in L]) < 0.5
                st = 0.27 * (int(round(sg / 20)) % 2)
                y = 0.97 - st if up else 0.03 + st
                for lev, txt, _ in (L if up else L[::-1]):
                    a = ax.text(sg, y, txt, transform=ax.get_xaxis_transform(), fontsize=6.3, color=LC[lev], ha='center', va='top' if up else 'bottom', linespacing=0.95,
                                bbox=dict(boxstyle='square,pad=0.08', fc='white', ec='none', alpha=0.75), zorder=4)
                    a.set_in_layout(False); y += (-1 if up else 1) * (0.07 + 0.075 * txt.count('\n'))
            if i == 0:
                ax.set_title(LAYDE[lay])
            if j == 0:
                ax.set_ylabel(f'{s}\nFlächen-RMSE [mm]')
            if i == 2:
                ax.set_xlabel('σ (Eingangsrauschen)')
    import matplotlib.lines as ml
    h = _lev_legend(fam) + [ml.Line2D([], [], color='0.3', marker='o', mfc='white', ls='', label='hohl: nicht alle Bilder definiert'),
                            ml.Line2D([], [], color='none', label='Zahlen: definiert/gesamt Bilder; Kn. = kleinste Knotenzahl (nur < 1000)')]
    if clipped:
        h.append(ml.Line2D([], [], color='0.3', marker='v', ls='', label=f'▼ Wert unter {_num(YL[0], 2)} mm (am Rand; Wert in der Beschriftung): {len(clipped)} Punkt(e)'))
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fam_m = 'DF Code genutzt gegen DF einzelne Klasse' if fam == 'dots' else 'R gegen P4 (normal)'
    extra = '\n5,6 mm, B/M10, σ0–40: Mittel enthält die N1-Randanomalie (1 von 8 Phasen, s. F15)' if fam == 'lines' else ''
    fig.suptitle(f'Globaler Flächenfehler (bedingt), {FAMDE[fam]}: {fam_m}\nPhasen gleich gewichtet; Mittel nur über definierte Bilder; Lücke = in keinem Bild definiert; '
                 f'gemeinsame Skala{extra}', fontsize=8)
    fig.tight_layout(rect=(0, 0.085 if clipped else 0.075, 1, 1), h_pad=0.6)
    _save(fig, f'S3C8_F05{"a" if fam == "dots" else "b"}_Flaechenfehler_{FAMDE[fam]}', 'case_phase_summary.csv; cases.csv (glob_surf_rmse, glob_n_interp)',
          'conditional RMSE with defined images / images and minimum interpolable nodes; common y range (values below marked); split by family; replaces CODE-07 F05',
          'formal summary', True)
    return notes


def f06(T, fam):
    plt = _plt(); R = T['R']; ru, ar = MAIN[fam]
    fig, axs = plt.subplots(3, 3, figsize=(W_IN, 6.6), sharex=True); rows = []; top = [0.0, 0.0]
    for j, lay in enumerate(LAY[fam]):
        for lev in LEV:
            q = _sel(R, family=fam, layout=lay, pattern=lev, rule=ru, arm=ar).groupby('sigma')[['n_records', 'n_flag_wrong_all', 'n_flag_wrong_all_defined', 'n_flag_wrong_new',
                                                                                                'n_flag_wrong_new_defined', 'sum_u', 'sum_acc']].sum()
            a = 100 * q.n_flag_wrong_all / q.n_records; b = 100 * q.n_flag_wrong_new / q.n_records; u = 100 * q.sum_u / q.sum_acc.where(q.sum_acc > 0)
            ua = 100 * (q.n_records - q.n_flag_wrong_all_defined) / q.n_records; ub = 100 * (q.n_records - q.n_flag_wrong_new_defined) / q.n_records
            axs[0, j].plot(q.index, a, '-', color=LC[lev], marker='o', ms=2.5); axs[1, j].plot(q.index, b, '-', color=LC[lev], marker='o', ms=2.5)
            axs[0, j].plot(q.index, ua, ls=':', color=LC[lev], marker='x', ms=3.5, lw=0.9); axs[1, j].plot(q.index, ub, ls=':', color=LC[lev], marker='x', ms=3.5, lw=0.9)
            axs[2, j].plot(q.index, u, '-', color=LC[lev], marker='o', ms=2.5)
            for sg, r in q.iterrows():
                rows.append(dict(family=fam, layout=lay, pattern=lev, sigma=sg, images=int(r.n_records), flagged_all=int(r.n_flag_wrong_all), all_defined=int(r.n_flag_wrong_all_defined),
                                 flagged_new=int(r.n_flag_wrong_new), new_defined=int(r.n_flag_wrong_new_defined), unconf_points=int(r.sum_u), accepted_points=int(r.sum_acc)))
        axs[0, j].set_title(LAYDE[lay]); axs[2, j].set_xlabel('σ (Eingangsrauschen)')
        for i in range(3):
            axs[i, j].grid(alpha=0.25); axs[i, j].set_xticks(SIG)
    for j in range(3):
        for i in range(2):
            axs[i, j].set_ylim(-3, 103); _de(axs[i, j], pct=True)
        _de(axs[2, j], pct=True)
    axs[0, 0].set_ylabel('Bildanteil\n(alle akzeptierten Punkte)'); axs[1, 0].set_ylabel('Bildanteil\n(neu akzeptierte Punkte)')
    axs[2, 0].set_ylabel('unbestätigte Punkte\n(u / akzeptiert)')
    import matplotlib.lines as ml
    h = _lev_legend(fam, False) + [ml.Line2D([], [], color='0.3', marker='o', ms=3, label='f = w/(c+w) > 0,1 % (bestätigt überschritten)'),
                                   ml.Line2D([], [], color='0.3', ls=':', marker='x', ms=3.5, label='f undefiniert (c+w = 0 bzw. keine neuen bestätigten; nicht „sicher“)')]
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False)
    meth = 'DF, Code genutzt' if fam == 'dots' else 'R'
    fig.suptitle(f'Bestätigt überschrittene Fehlzuordnungsschwelle (Bildanteil), {FAMDE[fam]} ({meth})\n'
                 '3 Oberflächen gepoolt; Zeilen 1–2: Nenner = alle Bilder (24 bei σ0, 48 bei σ > 0)\n'
                 'bedingte Anteile s. Bericht; Zeile 3: u / akzeptiert (Skala je Feld); kein Zuverlässigkeitsnachweis', fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.6)
    _save(fig, f'S3C8_F06{"a" if fam == "dots" else "b"}_Risiko_{FAMDE[fam]}', 'risk_summary.csv (n_records, n_flag_wrong_all, n_flag_wrong_new, sum_u, sum_acc)',
          'both image shares over ALL images (CODE-07 F06 divided the new-point line by the defined images); unconfirmed point share added; split by family', 'formal summary', True)
    return rows


def risk_conditional(T) -> pd.DataFrame:

    C = T['C']; out = []
    for fam in ('dots', 'lines'):
        ru, ar = MAIN[fam]
        X = C[(C.family == fam) & (C.rule == ru) & (C.arm == ar) & (C.pattern != 'plain')]
        for (s, lay, lev, sg), g in X.groupby(['surface', 'layout', 'pattern', 'sigma']):
            dfn = g.share_new.notna()
            out.append(dict(family=fam, surface=s, layout=lay, pattern=lev, sigma=sg, images=len(g), flagged_new=int((g.flag_wrong_new == 1).sum()), new_defined=int(dfn.sum()),
                            no_new_points=int((g.new_n == 0).sum()), new_but_no_confirmed_identity=int(((g.new_n > 0) & ((g.new_correct + g.new_wrong) == 0)).sum()),
                            share_all_images=float((g.flag_wrong_new == 1).sum() / len(g)), share_defined=(float((g.flag_wrong_new == 1).sum() / dfn.sum()) if dfn.sum() else np.nan)))
    return pd.DataFrame(out)


def f07(T):
    plt = _plt(); RO = T['RO']
    fig, axs = plt.subplots(2, 3, figsize=(W_IN, 5.6), sharex=True)
    for j, lay in enumerate(LAY['dots']):
        ax = axs[0, j]
        for lev in LEV:
            q = _sel(RO, family='dots', layout=lay, pattern=lev, rule='DF', arm='use').groupby('sigma')[['sum_cls_true_known', 'sum_cls_pre_correct', 'sum_cls_post_assigned', 'sum_cls_post_correct', 'sum_cls_post_abstain', 'sum_cls_sel_known']].sum()
            ax.plot(q.index, q.sum_cls_pre_correct / q.sum_cls_true_known, '-', color=LC[lev], marker='o', ms=2.5)
            ax.plot(q.index, q.sum_cls_post_correct / q.sum_cls_post_assigned.where(q.sum_cls_post_assigned > 0), '--', color=LC[lev])
            ax.plot(q.index, q.sum_cls_post_abstain / q.sum_cls_sel_known, ':', color=LC[lev], lw=1.4)
        ax.set_title(f'Punkte (DF) {LAYDE[lay]}'); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25); _de(ax)
    for j, lay in enumerate(LAY['lines']):
        ax = axs[1, j]
        for lev in LEV:
            for ru, ls in (('R@1.1247', '-'), ('LB@1.0000', '--')):
                q = _sel(RO, family='lines', layout=lay, pattern=lev, rule=ru).groupby('sigma')[['sum_L3_adopted', 'sum_L3_n2w', 'sum_L3_w2n', 'n_image_fallback', 'n_records']].sum()
                ax.plot(q.index, (q.sum_L3_n2w + q.sum_L3_w2n) / q.sum_L3_adopted.where(q.sum_L3_adopted > 0), ls, color=LC[lev], marker='o' if ls == '-' else None, ms=2.5)
                if ru.startswith('R'):
                    ax.plot(q.index, q.n_image_fallback / q.n_records, ls='', marker='s', mfc='none', color=LC[lev], ms=3.5)
        ax.set_title(f'Linien {LAYDE[lay]}'); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25); ax.set_xlabel('σ (Eingangsrauschen)'); ax.set_xticks(SIG); _de(ax)
    axs[0, 0].set_ylabel('Anteil'); axs[1, 0].set_ylabel('Anteil')
    import matplotlib.lines as ml
    h1 = [ml.Line2D([], [], color=LC[l], lw=2, label=LDE[l]) for l in LEV]
    h2 = [ml.Line2D([], [], color='0.3', ls='-', marker='o', ms=3, label='Punkte: Klasse richtig vor Prüfung (/ Kandidaten bekannter Klasse)'),
          ml.Line2D([], [], color='0.3', ls='--', label='Punkte: richtig nach Prüfung (/ eindeutig zugewiesene)'),
          ml.Line2D([], [], color='0.3', ls=':', lw=1.4, label='Punkte: Enthaltung (/ ausgewählte mit bekannter Klasse)'),
          ml.Line2D([], [], color='0.3', ls='-', marker='o', ms=3, label='Linien: Klassenfehler (/ übernommene Segmente), R; gestrichelt LB'),
          ml.Line2D([], [], color='0.3', ls='', marker='s', mfc='none', label='Linien: Bildrückfall von R (/ Bilder)')]
    l1 = fig.legend(handles=h1, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.17))
    fig.legend(handles=h2, loc='lower center', ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.0)); fig.add_artist(l1)
    fig.suptitle('Lesemechanismus, 3 Oberflächen gepoolt\nPunkte: Klassenlesung (DF); Linien: Segmentklassen und Bildrückfall', fontsize=8)
    fig.tight_layout(rect=(0, 0.23, 1, 1), h_pad=0.6)
    _save(fig, 'S3C8_F07_Lesemechanismus', 'readout_summary.csv (sum_cls_*, sum_L3_*, n_image_fallback, n_records)',
          'labels / legend separated; accuracy after the guard added from cls_post_correct / cls_post_assigned (stored); lines denominators segments / images', 'formal summary', True)


def f13_14(T):
    plt = _plt(); SC = T['SC']; tab = []
    for name, base, ylab, logy in (('S3C8_F13_Global_Lokal_Ctau', 'Ctau', 'Cτ', False), ('S3C8_F14_Global_Lokal_Flaechenfehler', 'surf_rmse', 'Flächen-RMSE [mm]', True)):
        fig, axs = plt.subplots(2, 3, figsize=(W_IN, 5.2), sharex=True)
        for i, fam in enumerate(('dots', 'lines')):
            ru, ar = MAIN[fam]
            for j, lay in enumerate(LAY[fam]):
                ax = axs[i, j]
                for s in SURF:
                    q = SC[(SC.family == fam) & (SC.surface == s) & (SC.layout == lay) & (SC.rule == ru) & (SC.arm == ar) & SC.pattern.isin(LEV)]
                    for dom, ls, mk in (('glob', '-', 'o'), ('loc', '--', '^')):
                        col = f'{dom}_{base}_mean'; xs, ys, part = [], [], []
                        for sg, g in q.groupby('sigma'):
                            v = g[col].to_numpy(float); fin = np.isfinite(v)
                            full = bool(fin.all()) and (base == 'Ctau' or bool((g[f'{dom}_{base}_n_defined'] >= g.n_rows).all()))
                            xs.append(sg); ys.append(float(v[fin].mean()) if fin.any() else np.nan); part.append(fin.any() and not full)
                            tab.append(dict(quantity=base, domain=dom, family=fam, layout=lay, surface=s, sigma=sg, value=ys[-1], levels_defined=int(fin.sum()), all_images_defined=full))
                        xs, ys, part = np.array(xs), np.array(ys), np.array(part)
                        ax.plot(xs, ys, ls, color=SC_[s], marker=mk, ms=2.8)
                        if part.any():
                            ax.scatter(xs[part], ys[part], s=26, marker=mk, facecolors='white', edgecolors=SC_[s], zorder=3)
                if logy:
                    ax.set_yscale('log'); ax.set_ylim(*YLIM_RMSE)
                else:
                    ax.set_ylim(-0.03, 1.03)
                _de(ax, logy=logy); ax.set_title(f'{FAMDE[fam]} {LAYDE[lay]}'); ax.grid(alpha=0.25); ax.set_xticks(SIG)
                if i == 1:
                    ax.set_xlabel('σ (Eingangsrauschen)')
            axs[i, 0].set_ylabel(ylab)
        import matplotlib.lines as ml
        h = [ml.Line2D([], [], color=SC_[s], lw=2, label=s) for s in SURF] + [ml.Line2D([], [], color='0.3', ls='-', marker='o', ms=3, label='global (30.947 Knoten)'),
                                                                               ml.Line2D([], [], color='0.3', ls='--', marker='^', ms=3, label='lokal (r ≤ 15 mm, 22.605 Knoten)')]
        if logy:
            h.append(ml.Line2D([], [], color='0.3', ls='', marker='o', mfc='white', label='hohl: nicht alle Stufen/Bilder definiert'))
        fig.legend(handles=h, loc='lower center', ncol=3, frameon=False)
        what = 'Cτ' if base == 'Ctau' else 'Flächen-RMSE (bedingt, Mittel über definierte Bilder)'
        fig.suptitle(f'Global gegen lokal: {what}\nCodierte Hauptmethode (Punkte: DF Code genutzt; Linien: R)\n3 Codestufen und Phasen gleich gewichtet; gemeinsame Skala', fontsize=8)
        fig.tight_layout(rect=(0, 0.1 if logy else 0.08, 1, 1), h_pad=0.6)
        _save(fig, name, 'case_phase_summary.csv (glob/loc_Ctau_mean, glob/loc_surf_rmse_mean, *_n_defined, n_rows)', 'new compact global / local comparison (local measurement task)',
              'formal summary', True)
    return pd.DataFrame(tab)


STATE = dict(normal='normal (Anzeige)', degraded='degradiert (Anzeige)', no_output='keine gültigen Punkte', not_found='kein Datensatz')


def _case_pair(fig, gs, c, arr, ident, dg):
    import matplotlib.patches as mp
    from matplotlib.colors import ListedColormap
    ax3 = fig.add_subplot(gs[0], projection='3d'); axm = fig.add_subplot(gs[1])
    mask = dg['mask']; lx, ly = dg['lattice_x'], dg['lattice_y']; sid = K.K4.RK.SURF_MAP[c['surface']]
    zt = np.full(mask.shape, np.nan); zt[mask] = dg[f'z_{sid}']; X, Y = np.meshgrid(lx, ly)
    ax3.plot_surface(X[::8, ::8], Y[::8, ::8], zt[::8, ::8], color='0.7', alpha=0.25, linewidth=0, antialiased=False)
    if arr is not None:
        pts, acc = arr['pts'], arr['acc']; ok = acc & np.isfinite(pts).all(axis=1)
        col = np.full(len(pts), '#1f77b4', dtype=object)
        if ident is not None and len(ident) == len(pts):
            col[ident == 'wr'] = '#d62728'; col[ident == 'un'] = '#9467bd'
        if ok.any():
            ax3.scatter(pts[ok, 0], pts[ok, 1], pts[ok, 2], s=0.6 if c['family'] == 'lines' else 2.0, c=list(col[ok]), depthshade=False)
        stm = np.full(mask.shape, np.nan); stm[mask] = arr['state_glob'].astype(float)
        axm.imshow(stm, origin='lower', extent=(lx[0], lx[-1], ly[0], ly[-1]), cmap=ListedColormap(['#d9d9d9', '#ff7f0e', '#2ca02c']), vmin=0, vmax=2)
    axm.add_patch(mp.Circle((0, 0), 15.0, fill=False, ls='--', lw=0.8, color='k'))
    if c['state'] == 'no_output':
        ax3.text2D(0.5, 0.55, 'keine gültigen Punkte', transform=ax3.transAxes, ha='center', color='#b22222', fontsize=7.5)
    ax3.set_xlim(-46, 46); ax3.set_ylim(-46, 46); ax3.set_zlim(-16, 4); ax3.view_init(elev=28, azim=-60); ax3.set_box_aspect((1, 1, 0.35))
    ax3.set_xticks([-40, 0, 40]); ax3.set_yticks([-40, 0, 40]); ax3.set_zticks([]); ax3.tick_params(labelsize=6, pad=-2)
    axm.tick_params(labelsize=6.5); axm.set_xticks([-40, 0, 40]); axm.set_yticks([-40, 0, 40])
    st = STATE[c['state']]
    if c.get('group') == 'posthoc':
        lay = LAYDE[str(c['layout'])].split(' (')[0]; ru = 'R' if str(c['rule']).startswith('R@') else str(c['rule'])
        st = f"{FAMDE[c['family']]} {lay}, {LDE[c['pattern']]}, {ru}\n" + ('höchster Anteil f' if c['posthoc'] == 'risk' else 'Kollaps σ0 → σ20')
    ax3.set_title(f"{c['surface']} {c['phase_id']} W{c['rep']} σ{int(c['sigma'])}\n{st}", fontsize=7.5)
    t = (f"k {c['c']} · f {c['w']} · u {c['u']}\nCτ {_num(c['Ctau'])} · Cvalid {_num(c['Cvalid'])}\nRMSE Pkt {_num(c.get('rmse'))} · Fl. {_num(c.get('surf_rmse'))}")
    if c.get('n_interp') is not None and c.get('surf_rmse') is not None and int(c['n_interp']) < 1000:
        t += f" ({c['n_interp']} Kn.)"
    axm.set_title(t, fontsize=7.5)


# Draw representative encoding images, point clouds and coverage maps.
def case_figs(T):
    plt = _plt(); import matplotlib.gridspec as gridspec
    import matplotlib.patches as mp
    import c7_config as CF7
    import c7_figs as F7
    import rdy_platform as P
    from c6_store import FileStore6
    cfg = CF7.load(K.K7.CONFIGS / 'formal.json'); store = FileStore6(cfg['data_root'], K.K7.Hashes()); board, arrays = F7.case_board(T['C'], cfg, store); dg = P.dom_global()

    def ident_of(c):
        st_, rec = store.inspect('eval', c['eval_key']); ea = store.arrays('eval', c['eval_key'], rec)
        if 'ident' in ea:
            i = np.asarray(ea['ident']).astype(str); return np.where(np.char.startswith(i, 'co'), 'co', np.where(np.char.startswith(i, 'wr'), 'wr', 'un'))
        b = np.asarray(ea['bridge'], bool); return np.where(b[0], 'co', np.where(b[1], 'wr', 'un'))

    def fig_of(cases, title, name, change, identity, recommended):
        n = len(cases); nrow = int(np.ceil(n / 2)); odd = n % 2 == 1; hrow = 2.3; H = hrow * nrow + (0.75 if odd else 1.15)
        fig = plt.figure(figsize=(W_IN, H)); outer = gridspec.GridSpec(nrow, 2, figure=fig, wspace=0.1, hspace=0.42)
        for k, c in enumerate(cases):
            g = outer[k // 2, k % 2].subgridspec(1, 2, wspace=0.18); arr = arrays.get(c.get('eval_key'))
            idt = ident_of(c) if (arr is not None) else None
            if idt is not None and len(idt) != len(arr['pts']):
                idt = None
            _case_pair(fig, g, c, arr, idt, dg)
        h = [mp.Patch(color='#1f77b4', label='Punkt: Identität korrekt (k)'), mp.Patch(color='#d62728', label='Punkt: bestätigt falsch (f)'), mp.Patch(color='#9467bd', label='Punkt: unbestätigt (u)'),
             mp.Patch(color='#2ca02c', label='Knoten: gestützt, |Δz| ≤ 0,15 mm'), mp.Patch(color='#ff7f0e', label='Knoten: gestützt, |Δz| > 0,15 mm'), mp.Patch(color='#d9d9d9', label='Knoten: ohne Stütze')]
        fig.suptitle(title + '\nKarte: globales Gebiet (30.947 Knoten); gestrichelt: lokales Gebiet r ≤ 15 mm; RMSE in mm', fontsize=8)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=(0.2 if odd else 0.75) / H, top=1 - 0.8 / H)
        if odd:
            bb = outer[nrow - 1, 1].get_position(fig)
            fig.legend(handles=h, loc='center', ncol=1, frameon=False, fontsize=7.5, bbox_to_anchor=(bb.x0 + bb.width / 2, bb.y0 + bb.height / 2))
        else:
            fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, fontsize=7.5)
        _save(fig, name, 'cases.csv; stored decode records (float64 recomputation c7_cases); evaluation fields ident / bridge', change, identity, recommended,
              cases=';'.join(c.get('eval_key', '') for c in cases))
    fx = [c for c in board if c.get('group') == 'fixed']; sp = [c for c in board if c.get('group') == 'supplement']; ph = [c for c in board if c.get('group') == 'posthoc']
    dots_m24 = [c for c in fx if c['rule_def']['family'] == 'dots']; lines_m24 = [c for c in fx if c['rule_def']['family'] == 'lines'] + [c for c in sp if c['rule_def']['family'] == 'lines' and c['surface'] == 'M24_curv']
    lines_m24 = sorted(lines_m24, key=lambda c: c['sigma'])
    fig_of(dots_m24, 'Registrierte Fälle, M24-Rauschserie: Punkte R6×4, mittel, N3, DF Code genutzt (σ0 W0, sonst W1)', 'S3C8_F11a_Faelle_M24_Punkte',
           'split from CODE-07 F11; same registered cases', 'registered cases', True)
    fig_of(lines_m24, 'Registrierte Fälle, M24-Rauschserie: Linien 5,6 mm, stark, N5, R (W0)', 'S3C8_F11b_Faelle_M24_Linien', 'split from CODE-07 F12; same registered cases',
           'registered cases', True)
    order = {'B_curv': 0, 'M10_curv': 1, 'M24_curv': 2}
    for fam, nm, ttl in (('dots', 'S3C8_F12a_Faelle_Oberflaechen_Punkte', 'Drei Oberflächen, Punkte R6×4, mittel, N3, DF Code genutzt: σ0 (W0) und σ40 (W1)'),
                         ('lines', 'S3C8_F12b_Faelle_Oberflaechen_Linien', 'Drei Oberflächen, Linien 5,6 mm, stark, N5, R (W0): σ0 und σ40')):
        cs = [c for c in fx + sp if c['rule_def']['family'] == fam and float(c['sigma']) in (0.0, 40.0)]
        cs = sorted(cs, key=lambda c: (order[c['surface']], c['sigma']))
        fig_of(cs, ttl, nm, 'three-surface comparison from the registered fixed + supplement cases', 'registered cases', True)
    fig_of(ph, 'Post-hoc-Diagnosefälle (Auswahlregel vor dem Lauf festgelegt)\nhöchster Fehlzuordnungsanteil f = w/(c+w) und stärkster Cτ-Abfall σ0 → σ20', 'S3C8_F12c_Fehlerfaelle_PostHoc',
           'post-hoc diagnostic cases kept as failures (no output / risk)', 'post-hoc diagnosis', False)
    return board


# Plot selected detection and reconstruction diagnostics from saved arrays.
def f15(T):
    plt = _plt()
    import c2_decode as CD
    import c3_store as ST
    import c4_store as S4
    import c6_check6x as X6
    import c7_cases as CS
    import rdy_platform as P
    from scipy.spatial import Delaunay
    from surface_eval_r3 import DEDUP_TOL_MM
    C = T['C']; M = pd.read_csv(K.K4.OUT / 'formal' / 'manifest.csv', low_memory=False); M['layout'] = M.layout.astype(str); Mi = M.set_index('image_key')
    dg = P.dom_global(); lx, ly = np.asarray(dg['lattice_x'], float), np.asarray(dg['lattice_y'], float)
    fig, axs = plt.subplots(2, 4, figsize=(W_IN, 5.3), gridspec_kw=dict(width_ratios=[0.75, 0.8, 1.15, 1.05])); keys = []; facts = []
    for i, lev in enumerate(('strong', 'weak')):
        r = _sel(C, family='lines', layout='5.6', phase_id='N1', sigma=0.0, surface='B_curv', pattern=lev, rule='R@1.1247').iloc[0]; keys.append(r.eval_key)
        row = dict(Mi.loc[r.image_key].to_dict(), image_key=r.image_key); inp = S4.inputs_of(row); sid = K.K4.RK.SURF_MAP[row['surface']]
        E = K.load_npz(K.C7_REC / 'eval' / f'{r.eval_key}.npz'); dj = K.load_json(K.C7_REC / 'dec' / f'{r.decode_key}.json'); A = K.load_npz(K.C7_REC / 'dec' / f'{r.decode_key}.npz')
        D = ST.unpack('lines', A, dj['meta']); feat = np.asarray(D['features'], float); acc = np.asarray(D['accepted'], bool); cor, wr, un = np.asarray(E['bridge'], bool)
        img = K.load_npz(K.C7_REC / 'images' / f'{r.image_key}.npz')['image']
        pat = np.asarray(inp['pattern'], float)
        ax = axs[i, 0]; prof = pat[256, :36]; ax.bar(np.arange(36), prof, width=1.0, color='0.3'); ax.set_xlim(-0.8, 35.5); ym = max(prof.max(), 1e-6)
        ax.set_ylim(0, 1.5 * ym); ax.set_title(f'Projektormuster\n({LDE[lev]}), Zeile 256', fontsize=7.5); ax.set_xlabel('Projektorspalte')
        ax.text(0.2, 1.06 * ym, 'Rand-\nrest', fontsize=6.5, va='bottom', ha='left'); ax.text(np.argmax(prof[10:]) + 10, 1.06 * ym, 'Linie\n−8', fontsize=6.5, va='bottom', ha='center')
        ax.set_yticks([]); _de(ax); ax.set_ylabel('Intensität (rel.)')
        ax = axs[i, 1]; ax.imshow(img[:, :90], cmap='gray', aspect='auto', origin='upper', vmin=0, vmax=max(1, np.percentile(img, 99.5)))
        m1 = acc & cor & (feat[:, 0] < 90); m2 = acc & un
        ax.scatter(feat[m1, 0], feat[m1, 1], s=0.3, c='#1f77b4'); ax.scatter(feat[m2, 0], feat[m2, 1], s=0.3, c='#9467bd')
        ax.set_title('Kamerabild\n(Spalten 0–89)', fontsize=7.5); ax.set_xlabel('Kameraspalte'); ax.set_ylabel('Bildzeile')
        S = P.s2(); mesh = P.mesh(row['surface']); dec = CD.line_dec(D); V = S.point_validity(dec['pts'], dec['accepted'], mesh); pts, valid = V['pts'], V['valid']
        ax = axs[i, 2]; ok = valid & acc
        zt = np.full(dg['mask'].shape, np.nan); zt[dg['mask']] = dg[f'z_{sid}']
        lo, hi = np.nanmin(zt, axis=1), np.nanmax(zt, axis=1); fin = np.isfinite(lo)
        ax.fill_between(ly[fin], lo[fin], hi[fin], color='0.75', alpha=0.6, lw=0, label='Referenzfläche (Bereich über x)')
        ax.axvspan(ly[fin].min(), ly[fin].max(), color='0.9', alpha=0.4, lw=0, zorder=0); ax.axvline(-15, color='k', ls='--', lw=0.7); ax.axvline(15, color='k', ls='--', lw=0.7)
        ax.scatter(pts[ok & cor, 1], pts[ok & cor, 2], s=0.15, c='#1f77b4', zorder=2); ax.scatter(pts[ok & un, 1], pts[ok & un, 2], s=0.4, c='#9467bd', zorder=3)
        ez = pts[ok & un, 2] - np.asarray(E['zr'], float)[ok & un]
        ax.annotate(f'{int((ok & un).sum())} unbestätigt\nΔz ≈ {_num(np.sqrt(np.mean(ez ** 2)), 1)} mm', (float(np.median(pts[ok & un, 1])), float(np.median(pts[ok & un, 2]))),
                    xytext=(-15, -11.5), fontsize=6.5, color='#5b2c83', arrowprops=dict(arrowstyle='->', lw=0.6, color='#5b2c83'))
        ax.text(0, 1.2, 'lokal', ha='center', va='bottom', fontsize=6.5)
        ax.set_xlim(-46, 46); ax.set_ylim(-17, 3); ax.set_xlabel('y [mm]'); ax.set_ylabel('z [mm]'); _de(ax)
        ax.set_title('Seitenansicht y–z\n(grau: Referenzfläche)', fontsize=7.5)
        dl = P.dom_local(row['surface']); Pv = X6._dedup(pts[valid].astype(np.float64), DEDUP_TOL_MM); tri = Delaunay(Pv[:, :2])
        L_ = CS._nodes(Pv, tri, dl['x'], dl['y'], dl['z_ref'], float(K.K4.RK.D_SUPPORT), float(K.K4.RK.TAU)); e = np.abs(L_['z64'] - np.asarray(dl['z_ref'], float))
        ax = axs[i, 3]; ok2 = np.isfinite(e)
        sc2 = ax.scatter(np.asarray(dl['x'])[ok2], np.asarray(dl['y'])[ok2], s=0.2, c=e[ok2], cmap='viridis', vmin=0, vmax=5, rasterized=True)
        ax.set_xlim(-16, 16); ax.set_ylim(-16, 16); ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); _de(ax)
        ax.set_title(f'lokal interpoliert |Δz| [mm]\nFl.-RMSE {_num(float(r.loc_surf_rmse))} mm', fontsize=7.5)
        cb = fig.colorbar(sc2, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=6.5)
        facts.append(dict(level=lev, eval_key=r.eval_key, n_unconf_valid=int((ok & un).sum()), unconf_depth_rmse=float(np.sqrt(np.mean(ez ** 2))), loc_surf_rmse_table=float(r.loc_surf_rmse),
                          loc_surf_rmse_recomputed=float(L_['surf'][1]) if L_['surf'][1] is not None else None))
    fig.suptitle('Post-hoc-Diagnose der N1-Randanomalie: B, Linien 5,6 mm, σ0, R (oben stark, unten schwach)\n'
                 'Randrest → linker Kamerarand → als Kennung −8 trianguliert → Tiefenfehler ≈ 13 mm\n'
                 'blau: Identität korrekt; violett: unbestätigt; schwach: Dreiecke reichen vom Randband ins lokale Gebiet', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1), w_pad=0.3, h_pad=0.8)
    _save(fig, 'S3C8_F15_N1_Randanomalie', 'stored camera images, decode and evaluation records; projector raster (inputs_of); float64 local interpolation (c7_cases)',
          'new post-hoc diagnosis of the N1 edge anomaly (B; M10 analogous, see diagnostic_cases.csv)', 'post-hoc diagnosis', True, cases=';'.join(keys))
    return facts


def run() -> dict:
    T = K.tables(); res = {}
    res['f05_partly_defined'] = f05(T, 'dots') + f05(T, 'lines')
    res['f06_rows'] = f06(T, 'dots') + f06(T, 'lines')
    rc = risk_conditional(T); res['risk_conditional'] = rc.to_dict('records')
    f07(T); gl = f13_14(T); res['global_local'] = gl.to_dict('records')
    case_figs(T); res['f15'] = f15(T)
    reused = [('S3C7_F01_E1_Klassenbeitrag_Punkte.png', 'E1 class contribution, dots'), ('S3C7_F02_E1_Klassenbeitrag_Linien.png', 'E1 class contribution, lines'),
              ('S3C7_F03_Schema_Ctau_Punkte.png', 'plain vs coded C_tau, dots'), ('S3C7_F04_Schema_Ctau_Linien.png', 'plain vs coded C_tau, lines'),
              ('S3C7_F08_Frontend_Vergleich.png', 'front-end contribution'), ('S3C7_F09_Bedingungskarte.png', 'condition map'), ('S3C7_F10_Diagnose_Referenzen.png', 'conditional diagnostic references')]
    for f, what in reused:
        p = K.C7_CUR / 'figures' / f
        INDEX.append(dict(figure=K.rel(p), pdf='', inputs='CODE-07 analysis tables', cases='', entry='scripts/surface_revision/s3_code07/c7_figs.py', change=f'reused unchanged ({what})',
                          identity='formal summary', recommended_main_text=f in ('S3C7_F01_E1_Klassenbeitrag_Punkte.png', 'S3C7_F02_E1_Klassenbeitrag_Linien.png', 'S3C7_F03_Schema_Ctau_Punkte.png',
                                                                                   'S3C7_F04_Schema_Ctau_Linien.png'), sha256_png=K.sha_file(p), sha256_pdf=''))
    for f, why in (('S3C7_F05_Flaechenfehler.png', 'replaced by S3C8_F05a/b'), ('S3C7_F06_Identitaetsrisiko.png', 'replaced by S3C8_F06a/b (denominator)'), ('S3C7_F07_Lesemechanismus.png', 'replaced by S3C8_F07'),
                   ('S3C7_F11_Faelle_Punkte.png', 'split into S3C8_F11a / F12a'), ('S3C7_F12_Faelle_Linien_PostHoc.png', 'split into S3C8_F11b / F12b / F12c')):
        p = K.C7_CUR / 'figures' / f
        INDEX.append(dict(figure=K.rel(p), pdf='', inputs='', cases='', entry='scripts/surface_revision/s3_code07/c7_figs.py', change=why, identity='superseded for the thesis (kept)',
                          recommended_main_text=False, sha256_png=K.sha_file(p), sha256_pdf=''))
    FI = pd.DataFrame(INDEX); K.save_csv(FI, K.OUT / 'figure_index.csv')
    res['figures'] = FI[['figure', 'pdf', 'identity', 'recommended_main_text', 'sha256_png', 'sha256_pdf']].to_dict('records')
    a = rc[(rc.family == 'lines') & (rc.surface == 'B_curv') & (rc.layout == '4.45') & (rc.pattern == 'mid') & (rc.sigma == 20.0)].iloc[0]
    b = [x for x in res['f05_partly_defined'] if x['family'] == 'lines' and x['surface'] == 'M10_curv' and x['layout'] == '5.6' and x['pattern'] == 'plain' and x['sigma'] == 100.0]
    res['anchors'] = dict(f06=dict(images=int(a.images), flagged_new=int(a.flagged_new), new_defined=int(a.new_defined), share_all_images=float(a.share_all_images), share_defined=float(a.share_defined)),
                          f05=(b[0] if b else None))
    res['ok'] = bool(res['anchors']['f06'] == dict(images=16, flagged_new=1, new_defined=1, share_all_images=0.0625, share_defined=1.0) and b and b[0]['defined'] == 1 and b[0]['images'] == 16
                     and b[0]['min_nodes'] == 1)
    return res


if __name__ == '__main__':
    r = run()
    if len(sys.argv) > 1:
        K.checks_add(sys.argv[1], {k: v for k, v in r.items() if k not in ('f06_rows', 'global_local')})
    print(json.dumps(K.clean({k: v for k, v in r.items() if k in ('anchors', 'ok', 'f15')}), indent=1, ensure_ascii=False)[:5000])
