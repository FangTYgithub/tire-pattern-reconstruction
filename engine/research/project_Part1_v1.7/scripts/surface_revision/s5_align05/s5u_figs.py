# Plot joint size-spacing curves, interactions and point clouds.

from __future__ import annotations

import numpy as np
import pandas as pd

import s5u_common as K

SEL_RULES = dict(statistic='median of global C_tau over the 16 sigma-40 records of a surface x combination cell (dots: D0, lines: P4)',
                 cells='highest and lowest median per family; ties by (surface, combination) key order; coinciding ends -> next cell of the other end',
                 case='phase / repetition with C_tau closest to the cell median; ties by (phase, rep)', series='sigma 0 (rep 0), 40 and 100 (selected rep), same phase',
                 diagnostic='M10_curv, L_w0.5_s6.3, ph1, sigma 20, rep 1, P4 (ALIGN-04 edge anomaly)', purpose='post-hoc illustration, not an independent validation')
SURF = ('B_curv', 'M10_curv', 'M24_curv'); SH = {'B_curv': 'B', 'M10_curv': 'M10', 'M24_curv': 'M24'}
SIG = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
DOTS = ('D_r0.75_p8x4', 'D_r1_p6x4', 'D_r1_p8x4', 'D_r1_p10x4', 'D_r1_p8x6', 'D_r1.5_p6x4', 'D_r1.5_p8x4', 'D_r1.5_p10x4', 'D_r1.5_p8x6')
LINES = ('L_w0.5_s4.45', 'L_w0.5_s6.3', 'L_w0.5_s8', 'L_w1.5_s4.45', 'L_w1.5_s6.3', 'L_w1.5_s8')
COL = {0.75: '#2ca02c', 1.0: '#1f77b4', 1.5: '#d62728', 0.5: '#1f77b4'}
LS_D = {'p6x4': ':', 'p8x4': '-', 'p10x4': '--', 'p8x6': '-.'}; LS_L = {'s4.45': ':', 's6.3': '--', 's8': '-'}


def lab(c: str) -> str:
    if c.startswith('D_'):
        r = float(c.split('_')[1][1:]); sp = c.split('_p')[1].replace('x', ' × ')
        return f'r {str(r).replace(".", ",")} (Ø {str(2 * r).replace(".", ",")}) · {sp}'
    w = c.split('_')[1][1:]; s = c.split('_s')[1]
    return f'b {w.replace(".", ",")} · s {s.replace(".", ",")}'


def style(c: str) -> dict:
    if c.startswith('D_'):
        r = float(c.split('_')[1][1:]); return dict(color=COL[r], ls=LS_D[c.split('_')[2]])
    w = float(c.split('_')[1][1:]); return dict(color=COL[w] if w == 0.5 else '#d62728', ls=LS_L['s' + c.split('_s')[1]])


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3})
    return plt, FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))


def _save(fig, out_dir, name) -> str:
    p = out_dir / name; K.root_of(p); p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=300); return K.rel(p)


def f_range(S: pd.DataFrame, fam: str, out_dir, tag='') -> list:
    plt, CF = _plt(); combos = DOTS if fam == 'point' else LINES; fe = 'D0' if fam == 'point' else 'P4'
    from matplotlib.ticker import LogLocator, NullFormatter
    rows = [('glob_Ctau_mean8', 'C$_τ$ global / %', 100, 'lin'), ('loc_Ctau_mean8', 'C$_τ$ lokal (r ≤ 15 mm) / %', 100, 'lin'),
            ('glob_pt_rmse_median', 'Punkt-RMSE global\n(Median def. Fälle) / mm', 1, 'log'), ('sup_surf_rmse_median', 'Flächen-RMSE gestützt\n(Median def. Fälle) / mm', 1, 'log')]
    fig, axs = plt.subplots(len(rows), 3, figsize=(6.5, 8.6), sharex=True)
    for j, s in enumerate(SURF):
        for i, (m, ylab, f, sc) in enumerate(rows):
            ax = axs[i, j]
            for c in combos:
                g = S[(S.combo == c) & (S.surface == s) & (S.front_end == fe)].set_index('sigma').reindex(SIG)
                y = pd.to_numeric(g[m], errors='coerce') * f; st = style(c)
                ax.plot(SIG, y, marker='o', ms=2.6, lw=0.9, label=lab(c), **st)
                if m.endswith('_median'):
                    nd = g[m.replace('_median', '_n_defined')]; part = (nd > 0) & (nd < g.n_cases)
                    ax.plot(np.array(SIG)[part.to_numpy()], y[part].to_numpy(), 'o', mfc='white', mec=st['color'], ms=3.2, lw=0)
            if sc == 'log':
                ax.set_yscale('log'); ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0))); ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylim(-3, 103)
            ax.grid(lw=0.3, alpha=0.5); ax.yaxis.set_major_formatter(CF)
            if i == 0:
                ax.set_title(SH[s])
            if j == 0:
                ax.set_ylabel(ylab, fontsize=6.3)
            if i == len(rows) - 1:
                ax.set_xlabel('Rauschen σ'); ax.set_xticks(SIG)
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, ncol=3 if fam == 'point' else 3, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6)
    txt = ('Punktmuster, Frontend D0; r = Radius, Ø = Durchmesser, Abstand längs × quer in mm.' if fam == 'point' else 'Linienmuster, P4; b = Linienbreite, s = Linienabstand in mm.')
    fig.text(0.01, 0.004, f'{txt}\nC$_τ$: 8-Phasen-Mittel (Wiederholungen je Phase gemittelt, kein Ausgang = 0). Fehler: Median definierter Fälle,\n'
             'hohler Marker = nicht alle Fälle definiert, fehlender Punkt = kein definierter Fall. Gemessene Stufen, keine Interpolation.' + tag, fontsize=5.6)
    fig.tight_layout(rect=(0, 0.05, 1, 0.945 if fam == 'point' else 0.96))
    return [_save(fig, out_dir, f"F{1 if fam == 'point' else 2}_{'Punkte' if fam == 'point' else 'Linien'}_Gesamtbereich.png")]


def f_fourcell(E: pd.DataFrame, out_dir, tag='') -> list:
    plt, CF = _plt()
    sets = [('dots_parallel', 'registered', 'D0', 'Punkte: Abstand längs 6 → 10 mm (r 1,0 / 1,5)'), ('dots_perp', 'registered', 'D0', 'Punkte: Abstand quer 4 → 6 mm (r 1,0 / 1,5)'),
            ('lines', 'registered', 'P4', 'Linien: Abstand 4,45 → 8 mm (b 0,5 / 1,5)'), ('lines_6.3_8', 'posthoc', 'P4', 'Linien: 6,3 → 8 mm (b 0,5 / 1,5) – nachträglich')]
    fig, axs = plt.subplots(4, 3, figsize=(6.5, 8.4), sharex=True)
    for i, (setn, an, fe, title) in enumerate(sets):
        for j, s in enumerate(SURF):
            ax = axs[i, j]
            for eff, col, dx in (('E_small', '#1f77b4', -2.2), ('E_large', '#d62728', 0.0), ('I', 'k', 2.2)):
                g = E[(E.set == setn) & E.analysis.str.startswith(an) & (E.front_end == fe) & (E.surface == s) & (E.effect == eff) & (E.metric == 'glob_Ctau') & (E.record_type == 'interaction')]
                g = g[g.sigma.isin([f'{x:.1f}' for x in SIG])].copy(); g['sg'] = g.sigma.astype(float); g = g.set_index('sg').reindex(SIG)
                y = 100 * pd.to_numeric(g['mean'], errors='coerce'); lo = 100 * pd.to_numeric(g['res_lo_5'], errors='coerce'); hi = 100 * pd.to_numeric(g['res_hi_95'], errors='coerce')
                ax.errorbar(np.array(SIG) + dx, y, yerr=[y - lo, hi - y], fmt='o-' if eff == 'I' else 'o--', color=col, ms=2.6, lw=0.9, capsize=1.5,
                            label={'E_small': 'E$_{klein}$', 'E_large': 'E$_{groß}$', 'I': 'I = E$_{groß}$ − E$_{klein}$'}[eff])
            ax.axhline(0, color='0.4', lw=0.5); ax.grid(lw=0.3, alpha=0.5); ax.yaxis.set_major_formatter(CF)
            if j == 0:
                ax.set_ylabel('Δ C$_τ$ global / pp', fontsize=6.3)
            ax.set_title(f'{title}\n{SH[s]}' if j == 1 else f'\n{SH[s]}', fontsize=6.3)
            if i == 3:
                ax.set_xlabel('Rauschen σ'); ax.set_xticks(SIG)
    h, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l_, ncol=3, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.004, 'E = C$_τ$(dünn) − C$_τ$(dicht) je Größe; Phasenmittel (Wiederholungen je Phase gemittelt); Balken = deskriptives 5–95-%-Intervall\n'
             '(Phasen-Resampling 10 000×, gleiche Indizes für alle Zellen); Punkte D0, Linien P4; kein Signifikanztest.' + tag, fontsize=5.6)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    return [_save(fig, out_dir, 'F3_Vierfelder_Abwaegung.png')]


def f_frontend(FC: pd.DataFrame, C: pd.DataFrame, out_dir, tag='') -> list:
    plt, CF = _plt()
    fig, axs = plt.subplots(4, 3, figsize=(6.5, 8.4), sharex=True)
    d = C[C.family == 'point']
    for j, s in enumerate(SURF):
        for c in DOTS:
            g = FC[(FC.combo == c) & (FC.surface == s)].set_index('sigma').reindex(SIG); st = style(c)
            axs[0, j].plot(SIG, 100 * g.d_glob_Ctau_mean8, marker='o', ms=2.4, lw=0.8, label=lab(c), **st)
            axs[1, j].plot(SIG, g.d_n_valid_mean8, marker='o', ms=2.4, lw=0.8, **st)
        for fe, ls in (('D0', '-'), ('DF', '--')):
            for k, (num, den, row) in enumerate(((('id_wrong',), ('id_correct', 'id_wrong'), 2), (('id_unconf',), ('id_correct', 'id_wrong', 'id_unconf'), 3))):
                y = []
                for sg in SIG:
                    g = d[(d.surface == s) & (d.front_end == fe) & (d.sigma == sg)]
                    nu = sum(pd.to_numeric(g[x]).sum() for x in num); de = sum(pd.to_numeric(g[x]).sum() for x in den)
                    y.append(100 * nu / de if de else np.nan)
                axs[row, j].plot(SIG, y, ls, marker='s', ms=2.6, color='k' if fe == 'D0' else '#ff7f0e', lw=1, label=fe)
        axs[0, j].set_title(SH[s])
    for i, yl in enumerate(('Δ C$_τ$ global (DF − D0) / pp', 'Δ gültige Punkte (DF − D0)', 'falsch bestätigt w/(c+w) / %', 'unbestimmbar u/(c+w+u) / %')):
        axs[i, 0].set_ylabel(yl, fontsize=6.2)
        for j in range(3):
            axs[i, j].grid(lw=0.3, alpha=0.5); axs[i, j].axhline(0, color='0.4', lw=0.5); axs[i, j].yaxis.set_major_formatter(CF)
    for j in range(3):
        axs[3, j].set_xlabel('Rauschen σ'); axs[3, j].set_xticks(SIG)
    h, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l_, ncol=3, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=5.8)
    axs[2, 0].legend(frameon=False, fontsize=6)
    fig.text(0.01, 0.004, 'Gleiche Bilder, nur das Frontend unterscheidet sich (D0 / eingefrorenes DF). Zeilen 1–2: Phasenmittel der Paardifferenzen je Konfiguration;\n'
             'Zeilen 3–4: über alle 9 Punktkonfigurationen gepoolte Punktanteile (Nenner = akzeptierte Punkte), leerer Punkt = Nenner 0.' + tag, fontsize=5.6)
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    return [_save(fig, out_dir, 'F4_Frontend_Identitaetsrisiko.png')]


def select_cases(C: pd.DataFrame) -> dict:
    out = {}
    for fam, fe in (('point', 'D0'), ('line', 'P4')):
        d = C[(C.family == fam) & (C.front_end == fe) & (C.sigma == 40.0)]
        med = d.groupby(['surface', 'combo']).glob_Ctau.median().reset_index().sort_values(['surface', 'combo']).reset_index(drop=True)
        hi = med.iloc[med.glob_Ctau.values.argmax()] if len(med) else None
        lo_cands = med[med.glob_Ctau == med.glob_Ctau.min()]
        lo = lo_cands.iloc[0]
        note = ''
        if (lo.surface, lo.combo) == (hi.surface, hi.combo):
            lo = med.drop(index=hi.name).iloc[med.drop(index=hi.name).glob_Ctau.values.argmin()]; note = 'ends coincided; next cell of the low end used'
        sel = []
        for role, cell in (('hoch', hi), ('niedrig', lo)):
            g = d[(d.surface == cell.surface) & (d.combo == cell.combo)].copy(); g['dist'] = (g.glob_Ctau - cell.glob_Ctau).abs()
            r = g.sort_values(['dist', 'phase_label', 'rep']).iloc[0]
            sel.append(dict(role=role, surface=cell.surface, combo=cell.combo, cell_median_Ctau=float(cell.glob_Ctau), phase=r.phase_label, rep=int(r.rep), front_end=fe,
                            n_cells_at_min=int(len(lo_cands)) if role == 'niedrig' else None,
                            series=[[cell.combo, cell.surface, r.phase_label, 0.0, 0], [cell.combo, cell.surface, r.phase_label, 40.0, int(r.rep)], [cell.combo, cell.surface, r.phase_label, 100.0, int(r.rep)]]))
        out[fam] = dict(cells=sel, note=note)
    out['diagnostic'] = dict(series=[['L_w0.5_s6.3', 'M10_curv', 'ph1', 20.0, 1]], front_end='P4')
    return out


def _arrays(r) -> tuple:
    if str(r.source).startswith('S5-ALIGN-04/05'):
        img = K.load_npz(K.FORMAL_STORE / 'image' / f'{r.image_key}.npz')['image']
    else:
        img = K.load_npz(K.SR / 's345_ready' / 'run' / 'images' / f'{r.image_key}.npz')['image']
    z = K.load_npz(K.ROOT / r.eval_record)
    return img, z


def f_cases(C: pd.DataFrame, sel: dict, out_dir, tag='') -> list:
    plt, CF = _plt(); files = []
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    nz, ne = Normalize(-16, 4), Normalize(-0.5, 0.5)
    jobs = [(fam, c['role'], c['series'], c['front_end']) for fam in ('point', 'line') for c in sel[fam]['cells']] + [('line', 'Diagnose', sel['diagnostic']['series'], 'P4')]
    for fam, role, series, fe in jobs:
        n = len(series); H = 1.75 * n + 0.95; fig = plt.figure(figsize=(6.5, H))
        gs = fig.add_gridspec(n, 7, width_ratios=[1, 1, 1, 0.05, 0.32, 1, 0.05], wspace=0.36, hspace=0.66,
                              left=0.03, right=0.92, top=1 - 0.42 / H, bottom=0.62 / H)
        cax_z = cax_e = None
        for i, key in enumerate(series):
            g = C[(C.combo == key[0]) & (C.surface == key[1]) & (C.phase_label == key[2]) & (C.sigma == float(key[3])) & (C.rep == int(key[4])) & (C.front_end == fe)]
            r = g.iloc[0]; img, z = _arrays(r)
            val = z['valid'].astype(bool); P = z['pts'][val]; zt = z['zr'][val].astype(float); e = P[:, 2] - zt if len(P) else np.zeros(0)
            a0 = fig.add_subplot(gs[i, 0]); a0.imshow(img, cmap='gray', vmin=0, vmax=255); a0.set_xticks([]); a0.set_yticks([])
            a0.set_title(f'{SH[key[1]]} · {lab(key[0])}\n{key[2]} · σ {key[3]:g} · Wdh. {key[4]} · Kamerabild', fontsize=5.4, loc='left')
            axk = [fig.add_subplot(gs[i, c]) for c in (1, 2, 5)]
            for k, (vals, norm, cmap, t) in enumerate(((zt, nz, 'viridis', 'wahre Höhe\n(gültige Punkte)'), (P[:, 2] if len(P) else zt, nz, 'viridis', f'rekonstruiert\n({fe})'),
                                                      (np.clip(e, -0.5, 0.5), ne, 'RdBu_r', 'z$_{rek}$ − z$_{wahr}$\n(±0,5 mm begrenzt)'))):
                ax = axk[k]; ax.set_facecolor('#dcdcdc')
                if len(P):
                    ax.scatter(P[:, 0], P[:, 1], c=vals, cmap=cmap, norm=norm, s=(3 if fam == 'point' else 0.5), linewidths=0, rasterized=True)
                else:
                    ax.text(0.5, 0.5, 'keine gültigen\nPunkte', ha='center', va='center', fontsize=6, transform=ax.transAxes)
                ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.set_aspect('equal'); ax.tick_params(labelsize=4.8, labelleft=(k == 0)); ax.set_title(t, fontsize=5.4)
            n_over = int((np.abs(e) > 0.5).sum()); emax = float(np.abs(e).max()) if len(e) else None
            ct = (lambda v: '–' if pd.isna(v) else f'{100 * float(v):.1f}'.replace('.', ','))
            rm = '–' if pd.isna(r.glob_pt_rmse) else f'{float(r.glob_pt_rmse):.3f}'.replace('.', ',')
            axk[0].set_xlabel(f'gültig {int(r.n_valid)} · C$_τ$ gl. {ct(r.glob_Ctau)} % / lok. {ct(r.loc_Ctau)} % · Punkt-RMSE {rm} mm\n'
                              f'c/w/u {int(r.id_correct)}/{int(r.id_wrong)}/{int(r.id_unconf)} · |Fehler| > 0,5 mm: {n_over} · max |Fehler| {"–" if emax is None else f"{emax:.2f}".replace(".", ",")} mm',
                              fontsize=5.0)
            if i == 0:
                cax_z = fig.add_subplot(gs[:, 3]); cax_e = fig.add_subplot(gs[:, 6])
        for cax, norm, cmap, lab_ in ((cax_z, nz, 'viridis', 'Höhe z / mm (Fenster −16…4)'), (cax_e, ne, 'RdBu_r', 'z$_{rek}$ − z$_{wahr}$ / mm')):
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.set_label(lab_, fontsize=5.3); cb.ax.tick_params(labelsize=5); cb.ax.yaxis.set_major_formatter(CF)
        fam_t = 'Punkte (D0)' if fam == 'point' else 'Linien (P4)'
        fig.suptitle(f'{fam_t} – {role}: ' + ('Zelle mit ' + ('höchstem' if role == 'hoch' else 'niedrigstem') + ' Median-C$_τ$ bei σ 40' if role != 'Diagnose' else 'Randanomalie aus S5-ALIGN-04'),
                     fontsize=7, x=0.01, ha='left')
        fig.text(0.01, 0.003, 'Draufsicht der gültigen Punkte (x, y in mm); gleiche Achsen und Farbskalen in allen Fällen; nachträgliche Auswahl nach festgelegter Regel, keine Validierung.' + tag, fontsize=5.3)
        files.append(_save(fig, out_dir, f"{'F5_Punkte' if fam == 'point' else 'F6_Linien'}_{role}.png")); plt.close(fig)
    return files
