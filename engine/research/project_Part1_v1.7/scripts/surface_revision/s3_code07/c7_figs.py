# Plot encoding performance, class contribution and identity risks.

from __future__ import annotations

import numpy as np
import pandas as pd

import c7_cases as CS
import c7_common as K

FIGS = {}
LEV = ['weak', 'mid', 'strong']
LEV_C = dict(plain='#222222', weak='#9467bd', mid='#ff7f0e', strong='#8c564b')
SIZE = dict(dots=dict(plain='r 1,0 mm', weak='r 0,916 / 1,078 mm', mid='r 0,811 / 1,159 mm', strong='r 0,525 / 1,313 mm'),
            lines=dict(plain='b 1,0 mm', weak='b 0,85 / 1,15 mm', mid='b 0,725 / 1,275 mm', strong='b 0,5 / 1,5 mm'))
LEV_DE = dict(plain='normal', weak='schwach', mid='mittel', strong='stark')
SURF = ['B_curv', 'M10_curv', 'M24_curv']
LAY = dict(dots=['R5x4', 'R6x4', 'R8x4'], lines=['4.45', '5.6', '8'])
LAY_DE = {'R5x4': 'Punkte R5×4', 'R6x4': 'Punkte R6×4', 'R8x4': 'Punkte R8×4', '4.45': 'Linien 4,45 mm', '5.6': 'Linien 5,6 mm', '8': 'Linien 8 mm'}
SIG = [0, 20, 40, 60, 80, 100]
XLAB = 'σ Eingangsrauschen (Std.-abw., Graustufen)'
CTAU = 'Cτ global (Anteil Referenzknoten mit |Δz| ≤ 0,15 mm)'
MAIN = dict(dots=('DF', 'use'), lines=('R@1.1247', 'R@1.1247')); PLAIN = dict(dots=('DF', 'single'), lines=('P4', 'P4'))
E1 = dict(dots='E1 DF use-ignore', lines='E1 R-ignore'); E1LB = 'E1 LB-ignore'
E2 = dict(dots='E2 DF coded-plain (scheme difference incl. size-dependent front end)', lines='E3 lines LC(R)-P4 whole scheme')


def reg(name):
    def deco(f):
        FIGS[name] = f; return f
    return deco


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 7.5, 'legend.fontsize': 5.8})
    return plt


def _finish(fig, cfg, stage, name, sources, note) -> dict:
    import matplotlib.pyplot as plt
    ev = f"S3-CODE-07 · {cfg['name']} · {cfg['evidence']} · Skript scripts/surface_revision/s3_code07/c7_figs.py · Quellen: {sources}"
    fig.text(0.005, 0.004, ev, fontsize=5.2, color='0.35', va='bottom', wrap=True)
    if cfg['evidence_kind'] == 'test':
        fig.text(0.5, 0.5, 'TEST – kein Forschungsergebnis', fontsize=40, color='red', alpha=0.18, ha='center', va='center', rotation=25, transform=fig.transFigure)
    p = K.guard(stage / 'figures' / name); p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=150); plt.close(fig)
    return dict(figure=name, batch=cfg['name'], evidence_kind=cfg['evidence_kind'], sources=sources, note=note, script='scripts/surface_revision/s3_code07/c7_figs.py', sha256=K.sha_file(p))


def _grid(plt, families=('dots', 'lines'), h=2.0, w=2.35):
    cols = [(f, l) for f in families for l in LAY[f]]
    fig, axs = plt.subplots(len(SURF), len(cols), figsize=(w * len(cols) + 0.6, h * len(SURF) + 1.3), squeeze=False, sharex=True)
    for i, s in enumerate(SURF):
        for j, (f, l) in enumerate(cols):
            ax = axs[i, j]; ax.grid(alpha=0.25); ax.set_xticks(SIG)
            if i == 0:
                ax.set_title(LAY_DE[l], fontsize=8)
            if j == 0:
                ax.set_ylabel(s, fontsize=8, fontweight='bold')
            if i == len(SURF) - 1:
                ax.set_xlabel('σ', fontsize=7)
    return fig, axs, cols


def _sel(X, fam=None, surf=None, lay=None, **kw):
    m = pd.Series(True, index=X.index)
    if fam is not None:
        m &= X.family == fam
    if surf is not None:
        m &= X.surface == surf
    if lay is not None:
        m &= X.layout.astype(str) == str(lay)
    for k, v in kw.items():
        m &= X[k] == v
    return X[m]


def _legend_levels(fig, fam_list, extra=None):
    import matplotlib.lines as mlines
    def lab(l):
        if len(fam_list) == 1:
            return f"{LEV_DE[l]} ({SIZE[fam_list[0]][l]})"
        return f"{LEV_DE[l]} (Punkte {SIZE['dots'][l]}; Linien {SIZE['lines'][l]})"
    h = [mlines.Line2D([], [], color=LEV_C[l], lw=1.6, label=lab(l)) for l in (['plain'] if 'plain' in (extra or []) else []) + LEV]
    h += [e for e in (extra or []) if e != 'plain']
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=min(len(h), 3 if len(fam_list) == 1 else 4), fontsize=6.0, frameon=False)


TOP, BOTTOM = 0.915, 0.10


def _head(fig, title, note):
    fig.suptitle(title, fontsize=9, y=0.995)
    fig.text(0.5, 0.955, note, ha='center', va='top', fontsize=6.2, color='0.25', wrap=True)


def _effect_grid(T, cfg, stage, name, fam, effs, title, note):
    plt = _plt(); fig, axs, cols = _grid(plt, (fam,), h=2.1, w=3.1); SP, B = T['SP'], T['B']
    import matplotlib.lines as mlines
    for i, s in enumerate(SURF):
        for j, (f, l) in enumerate(cols):
            ax = axs[i, j]; ax.axhline(0, color='0.5', lw=0.6)
            for k, (eff, ls) in enumerate(effs):
                for lev in LEV:
                    q = _sel(SP, f, s, l, effect=eff, pattern=lev).sort_values('sigma')
                    if not len(q):
                        continue
                    ax.plot(q.sigma, q.d_glob_Ctau_mean, ls, color=LEV_C[lev], lw=1.2, marker='o' if k == 0 else None, ms=2.2)
                    if k == 0 and len(B):
                        b = _sel(B, f, s, l, effect=eff, pattern=lev).sort_values('sigma')
                        if len(b):
                            ax.fill_between(b.sigma, b.lo, b.hi, color=LEV_C[lev], alpha=0.13, lw=0)
            if j == 0:
                ax.set_ylabel(f'{s}\nΔCτ', fontsize=7.5)
    extra = [mlines.Line2D([], [], color='0.3', ls=ls, label=lab) for (eff, ls), lab in zip(effs, [e[2] for e in effs_labels(effs)])]
    _legend_levels(fig, (fam,), extra + [mlines.Line2D([], [], color='0.6', lw=6, alpha=0.3, label='Band: 2,5–97,5 % Block-Bootstrap (8 Phasen, 2.000×; deskriptiv)')])
    _head(fig, title, note)
    fig.tight_layout(rect=(0, BOTTOM, 1, TOP))
    return _finish(fig, cfg, stage, name, 'pair_phase_summary.csv; bootstrap.csv', title)


def effs_labels(effs):
    return [(e, ls, {'E1 DF use-ignore': 'DF: Code − ohne Klasse', 'E1 R-ignore': 'R: Code − ignore', 'E1 LB-ignore': 'LB: Code − ignore'}.get(e, e)) for e, ls in effs]


@reg('F01')
def f01(T, cfg, stage, ctx):
    return _effect_grid(T, cfg, stage, 'S3C7_F01_E1_Klassenbeitrag_Punkte.png', 'dots', [(E1['dots'], '-')],
                        'F01  Klassenbeitrag im gleichen Bild, Punkte (DF): ΔCτ = Cτ(Code genutzt) − Cτ(Klasse ignoriert)',
                        'je Punkt: 8 Phasen gleich gewichtet, je Phase Mittel über 2 Rauschrealisierungen (σ > 0) bzw. 1 (σ 0); Paare je Punkt 16 (σ > 0) / 8 (σ 0); y-Achsen je Feld eigene Skala; x: ' + XLAB)


@reg('F02')
def f02(T, cfg, stage, ctx):
    return _effect_grid(T, cfg, stage, 'S3C7_F02_E1_Klassenbeitrag_Linien.png', 'lines', [(E1['lines'], '-'), (E1LB, '--')],
                        'F02  Klassenbeitrag im gleichen Bild, Linien: ΔCτ = Cτ(R bzw. LB) − Cτ(ignore); Band nur für R',
                        'je Punkt: 8 Phasen gleich gewichtet, je Phase Mittel über 2 Rauschrealisierungen (σ > 0); Paare je Punkt 16 / 8; y-Achsen je Feld eigene Skala; x: ' + XLAB)


def _abs_grid(T, cfg, stage, name, fam, value, title, note, ylabel, logy=False, cond=False):
    plt = _plt(); fig, axs, cols = _grid(plt, (fam,), h=2.1, w=3.1); SC = T['SC']
    for i, s in enumerate(SURF):
        for j, (f, l) in enumerate(cols):
            ax = axs[i, j]
            for lev, (ru, ar) in [('plain', PLAIN[fam])] + [(x, MAIN[fam]) for x in LEV]:
                q = _sel(SC, f, s, l, pattern=lev, rule=ru, arm=ar).sort_values('sigma')
                if not len(q):
                    continue
                y = q[f'{value}_mean']
                ax.plot(q.sigma, y, '-', color=LEV_C[lev], lw=1.6 if lev == 'plain' else 1.1, marker='o', ms=2.2)
                if cond and f'{value}_phases_defined' in q.columns:
                    part = q[(q[f'{value}_phases_defined'] < q.n_phases) & y.notna()]
                    ax.scatter(part.sigma, part[f'{value}_mean'], s=18, facecolors='white', edgecolors=LEV_C[lev], zorder=3)
            if logy:
                ax.set_yscale('log')
            if j == 0:
                ax.set_ylabel(f'{s}\n{ylabel}', fontsize=7)
    _legend_levels(fig, (fam,), ['plain'])
    _head(fig, title, note)
    fig.tight_layout(rect=(0, BOTTOM, 1, TOP))
    return _finish(fig, cfg, stage, name, 'case_phase_summary.csv', title)


@reg('F03')
def f03(T, cfg, stage, ctx):
    return _abs_grid(T, cfg, stage, 'S3C7_F03_Schema_Ctau_Punkte.png', 'dots', 'glob_Ctau', 'F03  Punkte: normales Muster (DF, einzelne Klasse) als Referenz gegen codierte Muster (DF, Code genutzt) – Cτ',
                     'Cτ = Anteil der festen Referenzknoten mit |Δz| ≤ 0,15 mm (ohne Ausgabe = 0, im Nenner); 8 Phasen gleich gewichtet; Unterschied enthält Größen- und Frontend-Effekte; y-Achsen je Feld eigene Skala; x: ' + XLAB, 'Cτ')


@reg('F04')
def f04(T, cfg, stage, ctx):
    return _abs_grid(T, cfg, stage, 'S3C7_F04_Schema_Ctau_Linien.png', 'lines', 'glob_Ctau', 'F04  Linien: normales Muster (P4) als Referenz gegen codierte Muster (R) – Cτ',
                     'Cτ = Anteil der festen Referenzknoten mit |Δz| ≤ 0,15 mm (ohne Ausgabe = 0, im Nenner); 8 Phasen gleich gewichtet; Gesamtunterschied der Schemata; y-Achsen je Feld eigene Skala; x: ' + XLAB, 'Cτ')


@reg('F05')
def f05(T, cfg, stage, ctx):
    plt = _plt(); fig, axs, cols = _grid(plt, ('dots', 'lines'), h=2.0, w=2.35); SC = T['SC']
    import matplotlib.lines as mlines
    for i, s in enumerate(SURF):
        for j, (f, l) in enumerate(cols):
            ax = axs[i, j]
            for lev, (ru, ar) in [('plain', PLAIN[f])] + [(x, MAIN[f]) for x in LEV]:
                q = _sel(SC, f, s, l, pattern=lev, rule=ru, arm=ar).sort_values('sigma')
                if not len(q):
                    continue
                for v, ls in (('glob_surf_rmse', '-'), ('glob_surf_p95', ':')):
                    ax.plot(q.sigma, q[f'{v}_mean'], ls, color=LEV_C[lev], lw=1.4 if lev == 'plain' else 1.0, marker='o' if ls == '-' else None, ms=2)
                part = q[(q.glob_surf_rmse_phases_defined < q.n_phases) & q.glob_surf_rmse_mean.notna()]
                ax.scatter(part.sigma, part.glob_surf_rmse_mean, s=16, facecolors='white', edgecolors=LEV_C[lev], zorder=3)
            ax.set_yscale('log')
            if j == 0:
                ax.set_ylabel(f'{s}\nFlächenfehler [mm]', fontsize=7)
    _legend_levels(fig, ('dots', 'lines'), ['plain', mlines.Line2D([], [], color='0.3', ls='-', label='RMSE'), mlines.Line2D([], [], color='0.3', ls=':', label='P95 |Δz|'),
                                            mlines.Line2D([], [], color='0.3', marker='o', mfc='white', ls='', label='hohl: nicht in allen Phasen definiert')])
    ttl = 'F05  Flächenfehler des interpolierten Knotengitters (global): normal (DF einzeln / P4) gegen codiert (DF Code / R)'
    _head(fig, ttl, 'Nenner: interpolierbare Knoten je Fall; Mittel über definierte Fälle (Phasen gleich gewichtet); undefiniert (keine interpolierbaren Knoten) = Lücke, nie 0; y-Achsen je Feld eigene Skala; x: ' + XLAB)
    fig.tight_layout(rect=(0, BOTTOM, 1, TOP))
    return _finish(fig, cfg, stage, 'S3C7_F05_Flaechenfehler.png', 'case_phase_summary.csv', ttl)


@reg('F06')
def f06(T, cfg, stage, ctx):
    plt = _plt(); fig, axs, cols = _grid(plt, ('dots', 'lines'), h=2.0, w=2.35); R = T['R']
    import matplotlib.lines as mlines
    for i, s in enumerate(SURF):
        for j, (f, l) in enumerate(cols):
            ax = axs[i, j]; ru, ar = MAIN[f]
            for lev in LEV:
                q = _sel(R, f, s, l, pattern=lev, rule=ru, arm=ar).sort_values('sigma')
                if not len(q):
                    continue
                n = q.n_records
                ax.plot(q.sigma, q.n_flag_wrong_all / n, '-', color=LEV_C[lev], marker='o', ms=2.2, lw=1.2)
                ax.plot(q.sigma, q.n_flag_wrong_new / q.n_flag_wrong_new_defined.where(q.n_flag_wrong_new_defined > 0), '--', color=LEV_C[lev], lw=0.9)
                ax.plot(q.sigma, q.n_flag_unconf / n, ':', color=LEV_C[lev], lw=0.9)
                ax.plot(q.sigma, q.n_flag_no_output / n, ls='', marker='x', color=LEV_C[lev], ms=3.5)
            ax.set_ylim(-0.03, 1.03)
            if j == 0:
                ax.set_ylabel(f'{s}\nAnteil Datensätze', fontsize=7)
    _legend_levels(fig, ('dots', 'lines'), [mlines.Line2D([], [], color='0.3', ls='-', label='bestätigte Fehlzuordnung > 0,1 % (alle akzeptierten Punkte)'),
                                            mlines.Line2D([], [], color='0.3', ls='--', label='> 0,1 % unter neu akzeptierten Punkten'),
                                            mlines.Line2D([], [], color='0.3', ls=':', label='unbestätigte Punkte vorhanden'),
                                            mlines.Line2D([], [], color='0.3', ls='', marker='x', label='ohne Ausgabe (0 akzeptierte Punkte)')])
    ttl = 'F06  Identitätsrisiko der Hauptmethoden (Punkte DF Code genutzt, Linien R): Anteil der Datensätze je Zelle'
    _head(fig, ttl, 'Nenner: Datensätze je Zelle (16 bei σ > 0, 8 bei σ 0; 8 Phasen); Marken gleichzeitig möglich; Schwellen 0,05 % / 0,5 % in risk_summary.csv; x: ' + XLAB)
    fig.tight_layout(rect=(0, BOTTOM, 1, TOP))
    return _finish(fig, cfg, stage, 'S3C7_F06_Identitaetsrisiko.png', 'risk_summary.csv', ttl)


@reg('F07')
def f07(T, cfg, stage, ctx):
    plt = _plt(); fig, axs, cols = _grid(plt, ('dots', 'lines'), h=2.0, w=2.35); RO = T['RO']
    import matplotlib.lines as mlines
    for i, s in enumerate(SURF):
        for j, (f, l) in enumerate(cols):
            ax = axs[i, j]
            for lev in LEV:
                if f == 'dots':
                    q = _sel(RO, f, s, l, pattern=lev, rule='DF', arm='use').sort_values('sigma')
                    if len(q):
                        ax.plot(q.sigma, q.pre_guard_class_accuracy, '-', color=LEV_C[lev], marker='o', ms=2.2, lw=1.2)
                        ax.plot(q.sigma, q.abstention_share, '--', color=LEV_C[lev], lw=0.9)
                else:
                    for ru, ls, mk in (('R@1.1247', '-', 'o'), ('LB@1.0000', '--', None)):
                        q = _sel(RO, f, s, l, pattern=lev, rule=ru).sort_values('sigma')
                        if len(q):
                            ax.plot(q.sigma, q.L3_class_error_share_of_adopted, ls, color=LEV_C[lev], marker=mk, ms=2.2, lw=1.1 if ls == '-' else 0.9)
                            ax.plot(q.sigma, q.n_image_fallback / q.n_records, ls='', marker='s' if ru.startswith('R') else 'D', mfc='none', color=LEV_C[lev], ms=3)
            ax.set_ylim(-0.03, 1.03)
            if j == 0:
                ax.set_ylabel(f'{s}\nAnteil', fontsize=7)
    _legend_levels(fig, ('dots', 'lines'), [mlines.Line2D([], [], color='0.3', ls='-', label='Punkte DF: Klassenrichtigkeit vor der Prüfung (/ Kandidaten mit bekannter Klasse)'),
                                            mlines.Line2D([], [], color='0.3', ls='--', label='Punkte DF: Enthaltung (/ ausgewählte Kandidaten mit bekannter Klasse)'),
                                            mlines.Line2D([], [], color='0.3', ls='-', marker='o', label='Linien R / LB (--): Klassenfehler der übernommenen Segmente (/ übernommene Segmente)'),
                                            mlines.Line2D([], [], color='0.3', ls='', marker='s', mfc='none', label='Linien R (□) / LB (◇): Bildrückfall (/ Bilder)')])
    ttl = 'F07  Lesemechanismus: Klassenlesung vor / nach der Prüfung (Punkte) und Segmentklassen, Rückfall (Linien)'
    _head(fig, ttl, 'gepoolt über 8 Phasen und Realisierungen je Zelle; Linien-Nenner sind Segmente bzw. Bilder, nicht Punkte; x: ' + XLAB)
    fig.tight_layout(rect=(0, BOTTOM, 1, TOP))
    return _finish(fig, cfg, stage, 'S3C7_F07_Lesemechanismus.png', 'readout_summary.csv', ttl)


def _lines_frontend(C):
    L = C[(C.family == 'lines') & C.rule.isin(['R@1.1247', 'LB@1.0000'])]
    w = L.pivot_table(index=['image_key', 'surface', 'layout', 'pattern', 'phase_id', 'sigma', 'rep'], columns='rule', values='glob_Ctau').reset_index()
    w['d'] = w['LB@1.0000'] - w['R@1.1247']
    ph = w.groupby(['surface', 'layout', 'pattern', 'sigma', 'phase_id']).d.mean().reset_index()
    return ph.groupby(['surface', 'layout', 'pattern', 'sigma']).d.mean().reset_index()


@reg('F08')
def f08(T, cfg, stage, ctx):
    plt = _plt(); fig, axs, cols = _grid(plt, ('dots', 'lines'), h=2.0, w=2.35); SP = T['SP']; LF = _lines_frontend(T['C'])
    import matplotlib.lines as mlines
    for i, s in enumerate(SURF):
        for j, (f, l) in enumerate(cols):
            ax = axs[i, j]; ax.axhline(0, color='0.5', lw=0.6)
            for lev in LEV:
                if f == 'dots':
                    for eff, ls in (('E3 DF-B0 use', '-'), ('E3 DS-B0 use', '--')):
                        q = _sel(SP, f, s, l, effect=eff, pattern=lev).sort_values('sigma')
                        ax.plot(q.sigma, q.d_glob_Ctau_mean, ls, color=LEV_C[lev], lw=1.1, marker='o' if ls == '-' else None, ms=2)
                else:
                    q = LF[(LF.surface == s) & (LF.layout.astype(str) == l) & (LF.pattern == lev)].sort_values('sigma')
                    ax.plot(q.sigma, q.d, '-', color=LEV_C[lev], lw=1.1, marker='o', ms=2)
            if j == 0:
                ax.set_ylabel(f'{s}\nΔCτ', fontsize=7)
    _legend_levels(fig, ('dots', 'lines'), [mlines.Line2D([], [], color='0.3', ls='-', label='Punkte: DF − B0 (Code genutzt, gleiches Bild)'),
                                            mlines.Line2D([], [], color='0.3', ls='--', label='Punkte: DS − B0 (Glättung kann auch die Lokalisierung ändern)'),
                                            mlines.Line2D([], [], color='0.3', ls='-', marker='o', label='Linien: LB − R (gleiches Bild, gleiche Detektion)')])
    ttl = 'F08  Frontend- bzw. Lesemethoden-Beitrag bei gleichem Eingangsbild: ΔCτ zwischen Methoden'
    _head(fig, ttl, '8 Phasen gleich gewichtet, je Phase Mittel der Realisierungen; R ist die vorab festgelegte Hauptmethode (keine nachträgliche Umstellung); y-Achsen je Feld eigene Skala; x: ' + XLAB)
    fig.tight_layout(rect=(0, BOTTOM, 1, TOP))
    return _finish(fig, cfg, stage, 'S3C7_F08_Frontend_Vergleich.png', 'pair_phase_summary.csv; cases.csv (LB − R je Bild)', ttl)


@reg('F09')
def f09(T, cfg, stage, ctx):
    plt = _plt(); CM = T['CM']
    from matplotlib.colors import TwoSlopeNorm
    fig, axs = plt.subplots(2, 2, figsize=(15, 9.2), squeeze=False)
    for i, fam in enumerate(('dots', 'lines')):
        X = CM[CM.family == fam]; rows = [(s, l) for s in SURF for l in LAY[fam]]; cols = [(lev, sg) for lev in LEV for sg in SIG]
        for j, (v, nm) in enumerate((('E1_d_glob_Ctau', 'E1 Klassenbeitrag (Code − ohne Klasse)'), ('E2_d_glob_Ctau', 'E2 Schema (codiert − normal)'))):
            A = np.full((len(rows), len(cols)), np.nan); wr = np.zeros_like(A); no = np.zeros_like(A); ci0 = np.zeros_like(A)
            for r_, (s, l) in enumerate(rows):
                for c_, (lev, sg) in enumerate(cols):
                    q = X[(X.surface == s) & (X.layout.astype(str) == l) & (X.pattern == lev) & (X.sigma == sg)]
                    if len(q) == 1:
                        A[r_, c_] = q[v].iloc[0]; wr[r_, c_] = q.risk_n_flag_wrong_all.iloc[0]; no[r_, c_] = q.risk_n_flag_no_output.iloc[0]
                        lo, hi = q[v.split('_')[0] + '_lo'].iloc[0], q[v.split('_')[0] + '_hi'].iloc[0]
                        ci0[r_, c_] = 1 if (pd.notna(lo) and lo <= 0 <= hi) else 0
            ax = axs[i, j]; lim = max(0.05, np.nanmax(np.abs(A)) if np.isfinite(A).any() else 0.05)
            im = ax.imshow(A, cmap='RdBu', norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
            for r_ in range(len(rows)):
                for c_ in range(len(cols)):
                    if np.isfinite(A[r_, c_]):
                        ax.text(c_, r_, f'{A[r_, c_]:+.2f}', ha='center', va='center', fontsize=4.6, color='k')
                    if wr[r_, c_] > 0:
                        ax.plot(c_ + 0.33, r_ - 0.3, marker='x', color='#d62728', ms=3.5)
                    if no[r_, c_] > 0:
                        ax.plot(c_ - 0.33, r_ - 0.3, marker='o', mfc='none', color='0.15', ms=3)
                    if ci0[r_, c_]:
                        ax.plot(c_, r_ + 0.32, marker='_', color='0.3', ms=4)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{s} · {LAY_DE[l]}' for s, l in rows], fontsize=5.8)
            ax.set_xticks(range(len(cols))); ax.set_xticklabels([f'{LEV_DE[lev][:3]} σ{sg}' for lev, sg in cols], fontsize=5.2, rotation=90)
            for c_ in (5.5, 11.5):
                ax.axvline(c_, color='k', lw=0.8)
            for r_ in (2.5, 5.5):
                ax.axhline(r_, color='k', lw=0.8)
            ax.set_title(f"{'Punkte (DF)' if fam == 'dots' else 'Linien (R)'}: {nm}, ΔCτ", fontsize=8); fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    ttl = 'F09  Bedingungskarte: Gewinn (Farbe, ΔCτ) und Risiko je Oberfläche × Abstand × Codestufe × σ'
    fig.suptitle(ttl, fontsize=9.5)
    fig.text(0.5, 0.945, '× rot: mindestens ein Datensatz mit bestätigter Fehlzuordnung > 0,1 % (Hauptmethode, codiert) · ○: mindestens ein Datensatz ohne Ausgabe · _ unter dem Wert: '
             'deskriptives Bootstrap-Band schließt 0 ein · keine Empfehlungskarte', ha='center', fontsize=6.4, color='0.25')
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _finish(fig, cfg, stage, 'S3C7_F09_Bedingungskarte.png', 'condition_map.csv (pair_phase_summary, bootstrap, risk_summary)', ttl)


@reg('F10')
def f10(T, cfg, stage, ctx):
    plt = _plt(); fig, axs, cols = _grid(plt, ('dots', 'lines'), h=2.0, w=2.35); DS = T['DS']
    import matplotlib.lines as mlines
    for i, s in enumerate(SURF):
        for j, (f, l) in enumerate(cols):
            ax = axs[i, j]; ax.axhline(0, color='0.5', lw=0.6)
            for lev in LEV:
                if f == 'dots':
                    spec = (('TI_src', 'source', '-'), ('TC_src', 'source', '--'))
                    rule = 'DF'
                else:
                    spec = (('TI', 'lines_v1', '-'), ('TC', 'lines_v1', '--'))
                    rule = 'ignore'
                for arm, ref, ls in spec:
                    q = _sel(DS, f, s, l, pattern=lev, rule=rule, arm=arm, reference=ref).sort_values('sigma')
                    if len(q):
                        ax.plot(q.sigma, q.d_glob_Ctau_mean, ls, color=LEV_C[lev], lw=1.1, marker='o' if ls == '-' else None, ms=2)
            if j == 0:
                ax.set_ylabel(f'{s}\nΔCτ Diagnose − Methode', fontsize=6.5)
    _legend_levels(fig, ('dots', 'lines'), [mlines.Line2D([], [], color='0.3', ls='-', label='Punkte TI_src / Linien TI: richtige Zuordnung als Referenz'),
                                            mlines.Line2D([], [], color='0.3', ls='--', label='Punkte TC_src / Linien TC: wahre Klasse als Referenz')])
    ttl = 'F10  Bedingte Diagnose-Referenzen (Wahrheit nur in der Auswertung): ΔCτ gegenüber der Methode desselben Decodes'
    _head(fig, ttl, 'Punkte: Quellenzuordnung (source) auf DF; Linien: lines_v1 auf ignore; keine Informationsgrenze des Gesamtsystems, keine additive Verlustzerlegung; y-Achsen je Feld eigene Skala; x: ' + XLAB)
    fig.tight_layout(rect=(0, BOTTOM, 1, TOP))
    return _finish(fig, cfg, stage, 'S3C7_F10_Diagnose_Referenzen.png', 'diag_summary.csv', ttl)


STATE_DE = dict(normal='normal (Anzeige)', degraded='degradiert (Anzeige)', no_output='keine gültigen Punkte (0 akzeptiert)', not_found='kein Datensatz unter der registrierten Bedingung')


def _case_axes(fig, gs, c, arr, ident, dg, mask, lx, ly):
    from matplotlib.colors import ListedColormap
    ax3 = fig.add_subplot(gs[0], projection='3d'); axm = fig.add_subplot(gs[1])
    import rdy_platform as P
    rr = c['rule_def'] if c.get('group') != 'posthoc' else dict(family=c['family'], layout=c['layout'], pattern=c['pattern'], surface=c['surface'], phase_id=c['phase_id'])
    fam = c.get('family', rr.get('family'))
    ph = ('Post-hoc-Diagnose (' + c['posthoc'] + ')\n') if c.get('group') == 'posthoc' else ''
    head = (f"{ph}{LAY_DE.get(str(c.get('layout', rr.get('layout'))), '')} {LEV_DE.get(c.get('pattern', rr.get('pattern')), '')} · {c.get('rule', rr.get('rule'))}\n"
            f"{c.get('surface', rr.get('surface'))} {c.get('phase_id', rr.get('phase_id'))} Wdh. {c.get('rep')} · σ {int(c['sigma'])}")
    if c['state'] == 'not_found':
        ax3.set_axis_off(); axm.set_axis_off(); axm.text(0.5, 0.5, f'{head}\n{STATE_DE["not_found"]}', ha='center', va='center', fontsize=6, transform=axm.transAxes); return
    sid = K.K4.RK.SURF_MAP[c['surface']]; zt = np.full(mask.shape, np.nan); zt[mask] = dg[f'z_{sid}']; X, Y = np.meshgrid(lx, ly)
    ax3.plot_surface(X[::8, ::8], Y[::8, ::8], zt[::8, ::8], color='0.75', alpha=0.3, linewidth=0, antialiased=False)
    if arr is not None:
        pts, acc = arr['pts'], arr['acc']; ok = acc & np.isfinite(pts).all(axis=1)
        col = np.full(len(pts), '#1f77b4', dtype=object)
        if ident is not None and len(ident) == len(pts):
            col[ident == 'wr'] = '#d62728'; col[ident == 'un'] = '#9467bd'
        if ok.any():
            ax3.scatter(pts[ok, 0], pts[ok, 1], pts[ok, 2], s=1.0 if fam == 'lines' else 2.5, c=list(col[ok]), depthshade=False)
        stm = np.full(mask.shape, np.nan); stm[mask] = arr['state_glob'].astype(float)
        axm.imshow(stm, origin='lower', extent=(lx[0], lx[-1], ly[0], ly[-1]), cmap=ListedColormap(['#d9d9d9', '#ff7f0e', '#2ca02c']), vmin=0, vmax=2)
    if c['state'] == 'no_output':
        ax3.text2D(0.5, 0.5, 'keine gültigen Punkte', transform=ax3.transAxes, ha='center', color='#b22222', fontsize=7)
    ax3.set_xlim(-46, 46); ax3.set_ylim(-46, 46); ax3.set_zlim(-16, 4); ax3.view_init(elev=28, azim=-60); ax3.set_box_aspect((1, 1, 0.35)); ax3.tick_params(labelsize=4, pad=-2)
    ax3.set_title(f"{head}\nakzeptiert {c['n_accepted']}, gültig {c['n_valid']}\nkorrekt {c['c']}, falsch {c['w']}, unbestätigt {c['u']}\n{STATE_DE[c['state']] if (c.get('group') != 'posthoc' or c['state'] == 'no_output') else 'Post-hoc-Fall: keine relative Anzeigeklasse'}", fontsize=5.8)
    rm = '–' if c.get('rmse') is None else f"{c['rmse']:.3f}"; sr = '–' if c.get('surf_rmse') is None else f"{c['surf_rmse']:.3f}"
    axm.set_title(f"Cvalid {c['Cvalid']:.3f} · Cτ {c['Ctau']:.3f}\n(Nenner 30.947 Referenzknoten)\nPunkt-RMSE {rm} mm (/ {c['n_valid']} gültige Pkt.)\n"
                  f"Flächen-RMSE {sr} mm (/ {c.get('n_interp')} interp. Knoten)", fontsize=5.8)
    axm.tick_params(labelsize=4.5); axm.set_xlabel('x [mm]', fontsize=5, labelpad=1); axm.set_ylabel('y [mm]', fontsize=5, labelpad=1)
    ax3.set_xlabel('x [mm]', fontsize=4.5, labelpad=-8); ax3.set_ylabel('y [mm]', fontsize=4.5, labelpad=-8); ax3.set_zlabel('z [mm]', fontsize=4.5, labelpad=-8)


def _case_fig(T, cfg, stage, ctx, name, title, sel):
    plt = _plt(); import rdy_platform as P
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    board = [c for c in ctx['board'] if sel(c)]; dg = P.dom_global(); mask = dg['mask']; lx, ly = dg['lattice_x'], dg['lattice_y']
    order = {'M24_curv': 0, 'B_curv': 1, 'M10_curv': 2}

    def rowkey(c):
        return 3 if c.get('group') == 'posthoc' else order.get(c.get('surface') or c['rule_def'].get('surface'), 0)
    rows_ = {}
    for c in sorted(board, key=lambda c: (rowkey(c), float(c['sigma']))):
        rows_.setdefault(rowkey(c), []).append(c)
    nrow = max(1, len(rows_)); ncol = max([len(v) for v in rows_.values()] + [1])
    fig = plt.figure(figsize=(6.4 * ncol, 4.2 * nrow + 1.3)); outer = gridspec.GridSpec(nrow, ncol, figure=fig, wspace=0.18, hspace=0.75)
    slots = [(i, j, c) for i, (_, v) in enumerate(sorted(rows_.items())) for j, c in enumerate(v)]
    store = ctx.get('store')
    for i, j, c in slots:
        g = outer[i, j].subgridspec(1, 2, wspace=0.1); arr = ctx['arrays'].get(c.get('eval_key')); ident = None
        if arr is not None and store is not None and c.get('eval_key'):
            st_, rec = store.inspect('eval', c['eval_key']); ea = store.arrays('eval', c['eval_key'], rec)
            if 'ident' in ea:
                ident = np.asarray(ea['ident']).astype(str); ident = np.where(np.char.startswith(ident, 'co'), 'co', np.where(np.char.startswith(ident, 'wr'), 'wr', 'un'))
            elif 'bridge' in ea:
                b = np.asarray(ea['bridge'], bool); ident = np.where(b[0], 'co', np.where(b[1], 'wr', 'un'))
            if ident is not None and len(ident) != len(arr['pts']):
                ident = None
        _case_axes(fig, g, c, arr, ident, dg, mask, lx, ly)
    h = [mpatches.Patch(color='#1f77b4', label='Punkt: Identität korrekt'), mpatches.Patch(color='#d62728', label='Punkt: bestätigte Fehlzuordnung'),
         mpatches.Patch(color='#9467bd', label='Punkt: Identität unbestätigt'), mpatches.Patch(color='#2ca02c', label='Knoten: gestützt, |Δz| ≤ 0,15 mm'),
         mpatches.Patch(color='#ff7f0e', label='Knoten: gestützt, |Δz| > 0,15 mm'), mpatches.Patch(color='#d9d9d9', label='Knoten: ohne Stütze (d = 6 mm)')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=6, fontsize=6.8, frameon=False)
    fig.suptitle(title, fontsize=9.5)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.1, top=1 - 1.7 / (4.2 * nrow + 1.3))
    return _finish(fig, cfg, stage, name, 'cases.csv; gespeicherte Decodes (float64-Neuberechnung c7_cases); Auswertungsfelder ident / bridge (Anzeige)', title)


@reg('F11')
def f11(T, cfg, stage, ctx):
    return _case_fig(T, cfg, stage, ctx, 'S3C7_F11_Faelle_Punkte.png',
                     'F11  Registrierte Fälle Punkte R6×4, mittel, N3, DF Code genutzt (σ0 Wdh. 0, sonst Wdh. 1): Wahrheit (grau), Punktwolke, Knotenzustand – vor der Aufnahme festgelegt',
                     lambda c: c.get('group') in ('fixed', 'supplement') and c['rule_def'].get('family') == 'dots')


@reg('F12')
def f12(T, cfg, stage, ctx):
    return _case_fig(T, cfg, stage, ctx, 'S3C7_F12_Faelle_Linien_PostHoc.png',
                     'F12  Registrierte Fälle Linien 5,6 mm, stark, N5, R (Wdh. 0) und zwei Post-hoc-Diagnosefälle (Regeln vor der Aufnahme festgelegt; ersetzen keine festen Fälle)',
                     lambda c: (c.get('group') in ('fixed', 'supplement') and c['rule_def'].get('family') == 'lines') or c.get('group') == 'posthoc')


def case_board(C: pd.DataFrame, cfg: dict, store) -> tuple[list, dict]:

    sel = CS.select(C, cfg); arrays = {}
    if store is None or not hasattr(store, 'root') or cfg.get('record_source') != 'c34_files':
        return [dict(c, recompute='n/a (no stored decode records)') for c in sel], arrays
    import c7_config as CF
    M = CF.manifest(cfg); Ci = C.set_index('eval_key'); out = []
    for c in sel:
        if c['state'] == 'not_found':
            out.append(dict(c, recompute='n/a (not found)')); continue
        r = CS.recompute(c, cfg['data_root'], M); cmp_ = CS.compare(r['metrics'], Ci.loc[c['eval_key']])
        ok = cmp_['max_diff'] <= 1e-9 and r['undecided_nodes'] == 0
        out.append(dict(c, recompute=('ok' if ok else 'differs'), max_diff=cmp_['max_diff'], worst=cmp_['worst'], n_fields=cmp_['n_fields'], undecided_nodes=r['undecided_nodes'],
                        tau_margin_glob=r['tau_margin_glob']))
        arrays[c['eval_key']] = r['arrays']
        if not ok:
            raise K.Stop('check', f"registered case {c['eval_key']} does not reproduce its record (max diff {cmp_['max_diff']:.3g} in {cmp_['worst']})")
    return out, arrays


def make(tables: dict, cfg: dict, stage, store) -> tuple[list, list]:
    board, arrays = case_board(tables['C'], cfg, store)
    ctx = dict(board=board, arrays=arrays, store=store)
    figs = []
    names = list(FIGS) if cfg.get('figures') == ['*'] else cfg.get('figures', [])
    for name in names:
        if name not in FIGS:
            raise K.Stop('config', f'figure {name} is not implemented')
        r = FIGS[name](tables, cfg, stage, ctx)
        if r is not None:
            figs.append(r)
    return figs, board
