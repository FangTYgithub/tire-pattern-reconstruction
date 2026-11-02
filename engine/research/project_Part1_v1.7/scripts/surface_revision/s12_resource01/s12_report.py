# Assemble size-spacing results, tables and figures into a report.

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_common as K
import s12_figs as F

SIG_COL = {0.0: '#1b9e77', 20.0: '#66a61e', 40.0: '#e6ab02', 60.0: '#d95f02', 80.0: '#e7298a', 100.0: '#7570b3'}
REG_SHORT = {'resource_cap_11695': 'gemeinsame Obergrenze', 'fixed_peak_1': 'fester Peak 1', 'equal_integral_5850': 'Integral 5850'}
S2_SERIES = {'p_par': ['R6x4', 'R7.35x4', 'R8x4', 'R10x4', 'R12x4'], 'p_perp': ['R8x2.5', 'R8x3', 'R8x4', 'R8x6', 'R8x8'], 'RT': ['R8x4', 'T8x4']}
METRICS_G = [('glob_Ctau', 'C_τ gesamt', 'lin'), ('glob_pt_rmse', 'Punkt-RMSE gesamt [mm]', 'log'), ('glob_surf_rmse', 'Flächen-RMSE gesamt [mm]', 'log'), ('glob_recovery', 'Wiederfindung gesamt (richtig)', 'lin')]
METRICS_L = [('loc_Ctau', 'C_τ lokal (r ≤ 15 mm)', 'lin'), ('loc_pt_rmse', 'Punkt-RMSE lokal [mm]', 'log'), ('loc_surf_rmse', 'Flächen-RMSE lokal [mm]', 'log'), ('loc_recovery', 'Wiederfindung lokal', 'lin')]


def size_label(fam, size):
    return f'd {2 * float(size):g} mm' if fam == 'point' else f'w {float(size):g} mm'


def lay_label(lay):
    if lay.startswith('s'):
        p, phi = lay[1:].split('_phi')
        return f's {p} mm · {phi}°'
    return lay.replace('x', '×')


def load_tables():
    C = pd.read_csv(K.RESULTS / 'conditions.csv', keep_default_na=False, low_memory=False)
    for c in C.columns:
        if c.endswith(('_mean', '_min', '_max', '_median')) or c in ('sigma', 'size_mm', 'p_parallel_mm', 'p_perp_mm', 'line_pitch_mm', 'orientation_deg'):
            C[c] = pd.to_numeric(C[c], errors='coerce')
    return C


def _axes_metric(ax, sc):
    if sc == 'log':
        ax.set_yscale('log')
        lo, hi = ax.get_ylim()
        if lo > 0:
            ax.set_ylim(lo, hi * (hi / lo) ** 0.1)
    else:
        top = max(ax.get_ylim()[1], 0.05); ax.set_ylim(-0.03 * top, top * 1.04)
    ax.grid(alpha=0.3, which='both')


def curves_page(C, sel: dict, groups: list, gcol: str, glabel, metrics, title, out, index, note=''):

    plt = F.plt_(); surfs = [s for s in K.SURFACES if s in set(C.surface)] if sel.get('surface') is None else [sel['surface']]
    q = C.copy()
    for k, v in sel.items():
        if k != 'surface':
            q = q[q[k] == v]
    surfs = [s for s in surfs if s in set(q.surface)]
    fig, axs = plt.subplots(len(metrics), len(surfs), figsize=(4.0 * len(surfs) + 1.0, 2.35 * len(metrics) + 1.3), squeeze=False)
    import matplotlib
    cols = matplotlib.colormaps['viridis'](np.linspace(0, 0.92, len(groups)))
    for j, s in enumerate(surfs):
        for i, (m, lab, sc) in enumerate(metrics):
            ax = axs[i, j]
            for k, gv in enumerate(groups):
                h = q[(q.surface == s) & (q[gcol] == gv)].sort_values('sigma')
                if not len(h):
                    continue
                x = h.sigma.to_numpy(float) + (k - (len(groups) - 1) / 2) * 1.2
                y = h[f'{m}_mean'].to_numpy(float)
                lo = h[f'{m}_min'].to_numpy(float) if f'{m}_min' in h else y; hi = h[f'{m}_max'].to_numpy(float) if f'{m}_max' in h else y
                ax.plot(x, y, '-o', ms=3, lw=1.0, color=cols[k], label=glabel(gv))
                ok = ~np.isnan(y)
                ax.errorbar(x[ok], y[ok], yerr=[np.maximum(y[ok] - lo[ok], 0), np.maximum(hi[ok] - y[ok], 0)], fmt='none', ecolor=cols[k], elinewidth=0.6, capsize=1.5, alpha=0.7)
                nd = h.get(f'{m}_n_phases_def')
                if sc == 'log' and nd is not None:
                    for xx, n_, ntot in zip(x, pd.to_numeric(nd), h.n_phases):
                        if n_ < ntot:
                            ax.text(xx, 0.02, f'{int(n_)}/{int(ntot)}', transform=ax.get_xaxis_transform(), fontsize=4.3, color=cols[k], ha='center', rotation=90)
            _axes_metric(ax, sc); ax.set_xticks(list(K.SIGMAS))
            if i == 0:
                ax.set_title(K.SHORT[s], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
            if i == len(metrics) - 1:
                ax.set_xlabel('Rauschen σ (Graustufen 0–255)', fontsize=7)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=min(len(groups), 7), frameon=False, fontsize=6.5)
    fig.suptitle(title + '\nPunkt = Mittel über Phasen (Realisierungen je Phase zuerst gemittelt), Balken = min–max der Phasenwerte; Lücke = undefiniert (keine Ausgabe), Zahl a/n = Phasen mit definiertem Fehler',
                 fontsize=7.8, y=0.998)
    F.foot(fig, note)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.07, hspace=0.25, wspace=0.22)
    fig.savefig(out, dpi=150); plt.close(fig)
    index.append(dict(figure=K.rel(out), panel='alle', selection='Statistik', scope=title.split('\n')[0]))


def heatmaps(C, section_rows, metric, title, out, index):

    plt = F.plt_(); surfs = [s for s in K.SURFACES if s in set(C.surface)]
    fig, axs = plt.subplots(len(section_rows), len(surfs), figsize=(3.6 * len(surfs) + 1.2, 1.9 * len(section_rows) + 1.4), squeeze=False)
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(0, max(0.05, float(C[f'{metric}_mean'].max())))
    for i, (fam, reg) in enumerate(section_rows):
        for j, s in enumerate(surfs):
            ax = axs[i, j]; h = C[(C.section == 'S1') & (C.family == fam) & (C.regime == reg) & (C.surface == s)]
            sizes = sorted(h.size_mm.unique()); M = np.full((len(sizes), len(K.SIGMAS)), np.nan)
            for a, sz in enumerate(sizes):
                for b, sg in enumerate(K.SIGMAS):
                    r = h[(h.size_mm == sz) & (h.sigma == sg)]
                    if len(r):
                        M[a, b] = r[f'{metric}_mean'].iloc[0]
            im = ax.imshow(M, cmap='viridis', norm=norm, aspect='auto', origin='lower')
            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    if not np.isnan(M[a, b]):
                        ax.text(b, a, f'{100 * M[a, b]:.0f}', ha='center', va='center', fontsize=5.6, color=('white' if norm(M[a, b]) < 0.55 else 'black'))
            ax.set_xticks(range(len(K.SIGMAS))); ax.set_xticklabels([f'{s:g}' for s in K.SIGMAS], fontsize=6)
            ax.set_yticks(range(len(sizes))); ax.set_yticklabels([size_label(fam, z) for z in sizes], fontsize=6)
            if i == 0:
                ax.set_title(K.SHORT[s], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{'Punkte' if fam == 'point' else 'Linien'}\n{REG_SHORT[reg]}", fontsize=6.6)
            if i == len(section_rows) - 1:
                ax.set_xlabel('σ', fontsize=7)
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5]); cb = fig.colorbar(im, cax=cax); cb.set_label(f'{metric} (Phasenmittel; Zahlen in %)', fontsize=6.5)
    fig.suptitle(title, fontsize=8.2, y=0.995)
    F.foot(fig, 'gemeinsame Obergrenze: Integral ≤ 11695 und Peak ≤ 1; fester Peak: Mechanismuskontrolle; Integral 5850: bestehende Punkte (Niedrigbudget)')
    fig.subplots_adjust(left=0.1, right=0.91, top=0.9, bottom=0.06, hspace=0.38, wspace=0.25)
    fig.savefig(out, dpi=150); plt.close(fig)
    index.append(dict(figure=K.rel(out), panel='alle', selection='Statistik', scope=title))


def regime_page(C, D, out, index):
    plt = F.plt_(); surfs = list(K.MAIN)
    rows = [('point', 'S1_regime_cap_vs_fixed', 'Punkte: Obergrenze − fester Peak'), ('point', 'S1_regime_cap_vs_5850', 'Punkte: Obergrenze − Integral 5850'),
            ('line', 'S1_regime_cap_vs_fixed', 'Linien: Obergrenze − fester Peak')]
    fig, axs = plt.subplots(len(rows), len(surfs), figsize=(13.0, 8.4), squeeze=False)
    import matplotlib
    for i, (fam, cn, lab) in enumerate(rows):
        for j, s in enumerate(surfs):
            ax = axs[i, j]; d = D[(D.contrast == cn) & (D.a_surface == s) & (D.a_family == fam) & (D.metric == 'glob_Ctau')]
            sizes = sorted(pd.to_numeric(d.a_size_mm).unique()); cols = matplotlib.colormaps['viridis'](np.linspace(0, 0.9, max(1, len(sizes))))
            for k, sz in enumerate(sizes):
                h = d[np.isclose(pd.to_numeric(d.a_size_mm), sz)].copy(); h['sig'] = pd.to_numeric(h.a_sigma); h = h.sort_values('sig')
                y = pd.to_numeric(h['mean'], errors='coerce'); lo = pd.to_numeric(h['min'], errors='coerce'); hi = pd.to_numeric(h['max'], errors='coerce')
                same = h.identical_inputs.astype(str).eq('True').all()
                ax.plot(h.sig + (k - 1) * 1.2, y, '-o', ms=3, lw=1, color=cols[k], label=size_label(fam, sz) + (' (gleiche Eingabe)' if same else ''))
                ax.errorbar(h.sig + (k - 1) * 1.2, y, yerr=[(y - lo).clip(lower=0), (hi - y).clip(lower=0)], fmt='none', ecolor=cols[k], elinewidth=0.6, alpha=0.7)
            ax.axhline(0, color='k', lw=0.6); ax.grid(alpha=0.3); ax.set_xticks(list(K.SIGMAS))
            if i == 0:
                ax.set_title(K.SHORT[s], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(lab + '\nΔ C_τ gesamt', fontsize=6.8)
            ax.legend(fontsize=5.4, frameon=False)
            if i == len(rows) - 1:
                ax.set_xlabel('σ', fontsize=7)
    fig.suptitle('Abschnitt 1 – Helligkeitsregime im Vergleich (gepaart je Phase und Realisierung; Mittel, Balken min–max der Phasendifferenzen)\n'
                 'gleiche Eingabe = beide Regime liefern dasselbe Bild (Peak-begrenzt bzw. Q0 = 5850), Differenz 0, keine zusätzliche Evidenz', fontsize=7.8, y=0.998)
    F.foot(fig); fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.07, hspace=0.3, wspace=0.22)
    fig.savefig(out, dpi=150); plt.close(fig); index.append(dict(figure=K.rel(out), panel='alle', selection='Statistik', scope='S1 Regimevergleich'))


def spacing_scan(C, metric, lab, out, index, surfs=None, title_extra=''):

    plt = F.plt_(); surfs = surfs or list(K.MAIN)
    rows = [('point', 'p_par', 'p∥ [mm] (p⊥ = 4)'), ('point', 'p_perp', 'p⊥ [mm] (p∥ = 8)'), ('line', 0.0, 'Linienabstand s [mm] · 0°'), ('line', 45.0, 'Linienabstand s [mm] · 45°')]
    fig, axs = plt.subplots(len(rows), len(surfs), figsize=(4.1 * len(surfs) + 1.0, 9.6), squeeze=False)
    for i, (fam, ser, xl) in enumerate(rows):
        for j, s in enumerate(surfs):
            ax = axs[i, j]; q = C[(C.section == 'S2') & (C.surface == s) & (C.family == fam)]
            for sg in K.SIGMAS:
                if fam == 'point':
                    lays = S2_SERIES[ser]; xs = [float(l[1:].split('x')[0]) if ser == 'p_par' else float(l.split('x')[1]) for l in lays]
                else:
                    lays = [f's{p:g}_phi{ser:g}' for p in K.S2_PITCHES]; xs = list(K.S2_PITCHES)
                y = [q[(q.layout == l) & (q.sigma == sg)][f'{metric}_mean'].iloc[0] if len(q[(q.layout == l) & (q.sigma == sg)]) else np.nan for l in lays]
                ax.plot(xs, y, '-o', ms=3, lw=1.0, color=SIG_COL[sg], label=f'σ = {sg:g}')
            ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.0) if 'Ctau' in metric else None
            if i == 0:
                ax.set_title(K.SHORT[s], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{'Punkte r 1 mm' if fam == 'point' else 'Linien w 1 mm'}\n{lab}", fontsize=6.8)
            ax.set_xlabel(xl, fontsize=6.5)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=6, frameon=False, fontsize=6.8)
    fig.suptitle(f'Abschnitt 2 – Abstand × Rauschen{title_extra}: {lab} je Abstand (Phasenmittel; nur gemessene Abstände, Verbindungen ohne Interpolation)\n'
                 'zeigt, ob sich die Reihenfolge der Abstände mit σ ändert; gemeinsame Obergrenze (Integral 11695, Peak 1)', fontsize=7.8, y=0.998)
    F.foot(fig); fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.06, hspace=0.42, wspace=0.22)
    fig.savefig(out, dpi=150); plt.close(fig); index.append(dict(figure=K.rel(out), panel='alle', selection='Statistik', scope=f'S2 Abstandsscan {metric}'))


def pair_effects(D, out, index):
    plt = F.plt_(); surfs = list(K.MAIN)
    rows = [('S2_dots_T_vs_R', None, 'T8×4 − R8×4'), ('S2_lines_phi45_vs_0', [4.45, 6.3, 8.0, 12.0], '45° − 0° (gleicher Abstand)')]
    fig, axs = plt.subplots(2, len(surfs), figsize=(13.0, 6.4), squeeze=False)
    for i, (cn, pitches, lab) in enumerate(rows):
        for j, s in enumerate(surfs):
            ax = axs[i, j]; d = D[(D.contrast == cn) & (D.a_surface == s) & (D.metric == 'glob_Ctau')].copy(); d['sig'] = pd.to_numeric(d.a_sigma)
            groups = [None] if pitches is None else [f's{p:g}_phi0' for p in pitches]
            import matplotlib
            cols = matplotlib.colormaps['plasma'](np.linspace(0, 0.85, len(groups)))
            for k, gname in enumerate(groups):
                h = d if gname is None else d[d.a_layout == gname]
                h = h.sort_values('sig'); y = pd.to_numeric(h['mean'], errors='coerce'); lo = pd.to_numeric(h.res_lo_5, errors='coerce'); hi = pd.to_numeric(h.res_hi_95, errors='coerce')
                xx = h.sig + (k - (len(groups) - 1) / 2) * 1.5
                ax.plot(xx, y, '-o', ms=3, lw=1, color=cols[k], label=('T − R' if gname is None else lay_label(gname).replace(' · 0°', '')))
                ax.errorbar(xx, y, yerr=[(y - lo).clip(lower=0), (hi - y).clip(lower=0)], fmt='none', ecolor=cols[k], elinewidth=0.7, alpha=0.8)
            ax.axhline(0, color='k', lw=0.6); ax.grid(alpha=0.3); ax.set_xticks(list(K.SIGMAS)); ax.legend(fontsize=5.6, frameon=False)
            if i == 0:
                ax.set_title(K.SHORT[s], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(lab + '\nΔ C_τ gesamt', fontsize=6.8)
            ax.set_xlabel('σ', fontsize=7)
    fig.suptitle('Abschnitt 2 – gepaarte Effekte (8 Phasen, Realisierungen zuerst gemittelt): Mittel und deskriptives Intervall 5–95 % (Phasenblock-Resampling, PCG64 2026101401)',
                 fontsize=7.8, y=0.995)
    F.foot(fig); fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.08, hspace=0.35, wspace=0.22)
    fig.savefig(out, dpi=150); plt.close(fig); index.append(dict(figure=K.rel(out), panel='alle', selection='Statistik', scope='S2 R/T und Orientierung'))


def resource_page(out, index):
    plt = F.plt_(); RT = pd.read_csv(K.RESULTS / 'resource_table.csv')
    cap = RT[RT.regime == 'resource_cap_11695']
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 8.0))
    blocks = [('S1', 'point', 'Abschnitt 1 · Punkte (Raster 8×4)'), ('S1', 'line', 'Abschnitt 1 · Linien (s 8 mm, 0°)'), ('S2', 'point', 'Abschnitt 2 · Punkte r 1 mm'), ('S2', 'line', 'Abschnitt 2 · Linien w 1 mm')]
    LC = {'peak_limited': '#1f77b4', 'integral_limited': '#d62728', 'both_limits': '#9467bd'}
    for ax, (sec, fam, title) in zip(axs.ravel(), blocks):
        q = cap[(cap.section == sec) & (cap.family == fam)]
        keys = (sorted(q.size_mm.unique()) if sec == 'S1' else (S2_SERIES['p_par'] + [l for l in S2_SERIES['p_perp'] if l != 'R8x4'] + ['T8x4'] if fam == 'point' else
                                                               [f's{p:g}_phi{f:g}' for f in (0.0, 45.0) for p in K.S2_PITCHES]))
        x = np.arange(len(keys)); ax2 = ax.twinx()
        for k, key in enumerate(keys):
            h = q[(q.size_mm == key)] if sec == 'S1' else q[q.layout == key]
            ax.bar(k - 0.2, h.Q0.mean(), 0.4, color='#bbbbbb', yerr=[[h.Q0.mean() - h.Q0.min()], [h.Q0.max() - h.Q0.mean()]], error_kw=dict(lw=0.6))
            lim = h.active_limit.iloc[0]
            ax.bar(k + 0.2, h.Q_actual.mean(), 0.4, color=LC.get(lim, 'k'))
            ax2.plot([k], [h.peak_f32.mean()], 'kD', ms=3.5)
        ax.axhline(K.Q_BUDGET, color='#d62728', lw=0.8, ls='--'); ax2.axhline(1.0, color='k', lw=0.6, ls=':'); ax2.set_ylim(0, 1.1); ax2.set_ylabel('Peak (♦, Graustufe/255)', fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels([size_label(fam, k) if sec == 'S1' else lay_label(k) for k in keys], rotation=60, fontsize=5.8)
        ax.set_ylabel('Q0 (grau) / Q_tatsächlich (farbig)', fontsize=6.5); ax.set_title(title, fontsize=7.5); ax.grid(alpha=0.3, axis='y')
    import matplotlib.patches as mp
    import matplotlib.lines as ml
    h = [mp.Patch(color='#bbbbbb', label='Q0 = Σ I0 (Rasterpixel)'), mp.Patch(color=LC['peak_limited'], label='Q_tatsächlich: Peak-begrenzt (Peak 1, Q < 11695)'),
         mp.Patch(color=LC['integral_limited'], label='Q_tatsächlich: Integral-begrenzt (= 11695)'), ml.Line2D([], [], color='#d62728', ls='--', label='Budget 11695'), ml.Line2D([], [], color='k', marker='D', ls='', label='Peak')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=5, frameon=False, fontsize=6.5)
    fig.suptitle('Ressourcen unter der gemeinsamen Obergrenze (normierte Grauwertsumme der Musterebene; keine Watt- oder Photonenzahl)\n'
                 'Balken = Mittel über Phasen, Fehlerbalken = min–max von Q0; Rauschen ändert das Musterbudget nicht (Kamerahintergrund ist kein Ressourcenzuwachs)', fontsize=7.8, y=0.998)
    F.foot(fig); fig.subplots_adjust(left=0.07, right=0.94, top=0.88, bottom=0.12, hspace=0.6, wspace=0.3)
    fig.savefig(out, dpi=150); plt.close(fig); index.append(dict(figure=K.rel(out), panel='alle', selection='Statistik', scope='Ressourcen'))


def reliability_page(C, sec, out, index, surfs=None):

    plt = F.plt_(); surfs = surfs or list(K.MAIN)
    q0 = C[(C.section == sec) & (C.regime == 'resource_cap_11695')]
    fams = ['point', 'line']
    fig, axs = plt.subplots(len(fams) * 2, len(surfs), figsize=(4.4 * len(surfs) + 1.0, 13.5 if sec == 'S2' else 9.5), squeeze=False)
    import matplotlib.colors as mcolors
    for a, fam in enumerate(fams):
        for b, (kind, lab) in enumerate((('fail', 'Ansichten ohne Ausgabe / ohne richtige Wiederfindung'), ('ident', 'Anteil nicht bestimmbar / falsch (Fälle)'))):
            for j, s in enumerate(surfs):
                ax = axs[2 * a + b, j]; q = q0[(q0.family == fam) & (q0.surface == s)]
                cfg = sorted(q.size_mm.unique()) if sec == 'S1' else ([l for l in S2_SERIES['p_par'] + S2_SERIES['p_perp'] + ['T8x4'] if l in set(q.layout)] if fam == 'point' else
                                                                       [f's{p:g}_phi{f:g}' for f in (0.0, 45.0) for p in K.S2_PITCHES])
                cfg = list(dict.fromkeys(cfg))
                M = np.full((len(cfg), 6), np.nan); T = [[''] * 6 for _ in cfg]
                for i_, c in enumerate(cfg):
                    for k, sg in enumerate(K.SIGMAS):
                        r = q[((q.size_mm == c) if sec == 'S1' else (q.layout == c)) & (q.sigma == sg)]
                        if not len(r):
                            continue
                        r = r.iloc[0]
                        if kind == 'fail':
                            M[i_, k] = r.n_zero_correct_recovery / r.n_views; T[i_][k] = f'{int(r.n_no_output)}|{int(r.n_zero_correct_recovery)}'
                        else:
                            u = pd.to_numeric(r.unconf_share_case_mean_def, errors='coerce'); w = pd.to_numeric(r.wrong_share_case_mean_def, errors='coerce')
                            M[i_, k] = 0 if np.isnan(u) else u; T[i_][k] = ('–' if np.isnan(u) else f'{100 * u:.1f}') + ('' if np.isnan(w) or w == 0 else f'/{100 * w:.1f}')
                ax.imshow(M, cmap='Reds', norm=mcolors.Normalize(0, 1 if kind == 'fail' else max(0.05, np.nanmax(M) if np.isfinite(M).any() else 0.05)), aspect='auto')
                for i_ in range(len(cfg)):
                    for k in range(6):
                        ax.text(k, i_, T[i_][k], ha='center', va='center', fontsize=4.6)
                ax.set_xticks(range(6)); ax.set_xticklabels([f'{s_:g}' for s_ in K.SIGMAS], fontsize=5.6)
                ax.set_yticks(range(len(cfg))); ax.set_yticklabels([size_label(fam, c) if sec == 'S1' else lay_label(c) for c in cfg], fontsize=5.2)
                if 2 * a + b == 0:
                    ax.set_title(K.SHORT[s], fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(f"{'Punkte' if fam == 'point' else 'Linien'}\n{lab}", fontsize=6.2)
    fig.suptitle(f'Abschnitt {sec[1]} – Zuverlässigkeit je Konfiguration und σ (gemeinsame Obergrenze; alle Phasen und Realisierungen)\n'
                 'oben je Familie: "ohne Ausgabe | ohne richtige Wiederfindung" (Anzahl Ansichten; Farbe = Anteil ohne richtige Wiederfindung); unten: nicht bestimmbar [/ falsch] in % der angenommenen Punkte',
                 fontsize=7.6, y=0.998)
    F.foot(fig); fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.04, hspace=0.35, wspace=0.28)
    fig.savefig(out, dpi=150); plt.close(fig); index.append(dict(figure=K.rel(out), panel='alle', selection='Statistik', scope=f'{sec} Zuverlässigkeit'))


def phase_strips(C, out, index):

    plt = F.plt_(); sel = [('S1', 'point', 'size_mm', [0.5, 1.0, 1.5]), ('S1', 'line', 'size_mm', [0.5, 1.0, 1.5]), ('S2', 'point', 'layout', ['R6x4', 'R8x4', 'R12x4', 'R8x2.5', 'R8x8']),
                           ('S2', 'line', 'layout', ['s4.45_phi0', 's6.3_phi0', 's12_phi0', 's6.3_phi45'])]
    fig, axs = plt.subplots(len(sel), 3, figsize=(13.5, 12.0), squeeze=False)
    for i, (sec, fam, col, cfgs) in enumerate(sel):
        for j, s in enumerate(K.MAIN):
            ax = axs[i, j]; q = C[(C.section == sec) & (C.family == fam) & (C.surface == s) & (C.regime == 'resource_cap_11695')]
            xt = []
            for k, cfg in enumerate(cfgs):
                for l_, sg in enumerate((0.0, 40.0, 80.0)):
                    r = q[((q[col] == cfg) if col == 'layout' else np.isclose(q[col], cfg)) & (q.sigma == sg)]
                    if not len(r):
                        continue
                    v = np.array([np.nan if x is None else x for x in json.loads(r.glob_Ctau_phases.iloc[0])], float)
                    xx = k * 4 + l_
                    ax.scatter(np.full(len(v), xx) + np.linspace(-0.25, 0.25, len(v)), v, s=6, color=SIG_COL[sg], linewidths=0)
                    ax.plot([xx - 0.35, xx + 0.35], [np.nanmean(v)] * 2, color='k', lw=1.2)
                xt.append((k * 4 + 1, size_label(fam, cfg) if col == 'size_mm' else lay_label(cfg)))
            ax.set_xticks([a for a, _ in xt]); ax.set_xticklabels([b for _, b in xt], fontsize=5.8, rotation=20); ax.set_ylim(-0.03, 1.0); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(K.SHORT[s], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{sec} {'Punkte' if fam == 'point' else 'Linien'}\nC_τ gesamt je Phase", fontsize=6.6)
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=SIG_COL[s], marker='o', ls='', label=f'σ = {s:g}') for s in (0.0, 40.0, 80.0)] + [ml.Line2D([], [], color='k', label='Mittel')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=6.8)
    fig.suptitle('Mittel und Phasenstreuung (gemeinsame Obergrenze): jeder Punkt eine Phase (Realisierungen gemittelt); Abschnitt 1 Punkte nur 3 Phasen\n'
                 'Ausreißerphasen und Totalausfälle bleiben sichtbar; keine Glättung', fontsize=7.8, y=0.998)
    F.foot(fig); fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.06, hspace=0.45, wspace=0.2)
    fig.savefig(out, dpi=150); plt.close(fig); index.append(dict(figure=K.rel(out), panel='alle', selection='Statistik', scope='Phasenwerte'))


def sensitivity_page(out, index):
    plt = F.plt_(); C = pd.read_csv(K.RESULTS / 'sens_conditions.csv'); R = pd.read_csv(K.RESULTS / 'sens_rankings.csv')
    combos = [(d, t) for d in K.SENS_D for t in K.SENS_TAU]
    rows = [('S1', 'point', 'size_mm', [0.5, 1.0, 1.5]), ('S1', 'line', 'size_mm', list(K.LINE_WIDTHS)), ('S2', 'point', 'layout', S2_SERIES['p_par'] + ['R8x2.5', 'R8x3', 'R8x6', 'R8x8', 'T8x4']),
            ('S2', 'line', 'layout', [f's{p:g}_phi0' for p in K.S2_PITCHES])]
    for sg in (0.0, 40.0, 80.0):
        fig, axs = plt.subplots(len(rows), 3, figsize=(13.5, 12.0), squeeze=False)
        import matplotlib
        cols = matplotlib.colormaps['coolwarm'](np.linspace(0, 1, len(combos)))
        for i, (sec, fam, col, cfgs) in enumerate(rows):
            for j, s in enumerate(K.MAIN):
                ax = axs[i, j]; q = C[(C.section == sec) & (C.family == fam) & (C.surface == s) & (C.sigma == sg)]
                for k, (d, t) in enumerate(combos):
                    y = [q[((q[col] == c) if col == 'layout' else np.isclose(q[col], c))][f'Ctau_d{d:g}_t{t:g}'].iloc[0] if len(q[((q[col] == c) if col == 'layout' else np.isclose(q[col], c))]) else np.nan for c in cfgs]
                    main = (d == 6.0 and t == 0.15)
                    ax.plot(range(len(cfgs)), y, '-o' if main else '-', ms=3, lw=(2.0 if main else 0.8), color=('k' if main else cols[k]), label=f'd {d:g} · τ {t:g}' + (' (Haupt)' if main else ''), zorder=(5 if main else 2))
                r = R[(R.section == sec) & (R.family == fam) & (R.surface == s) & (R.sigma == sg)]
                ax.set_title(f'{K.SHORT[s]} · Spitze wechselt in {int(r.top_changed.sum())}/{len(r)} Kombinationen', fontsize=7)
                ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([size_label(fam, c) if col == 'size_mm' else lay_label(c) for c in cfgs], fontsize=5.6, rotation=35)
                ax.set_ylim(-0.02, 1.0); ax.grid(alpha=0.3)
                if j == 0:
                    ax.set_ylabel(f"{sec} {'Punkte' if fam == 'point' else 'Linien'}\nC_τ gesamt", fontsize=6.6)
        h_, l_ = axs[0, 0].get_legend_handles_labels()
        fig.legend(h_, l_, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=9, frameon=False, fontsize=6)
        fig.suptitle(f'Bewertungsanforderung – Sensitivität von C_τ gesamt für d = 5/6/7 mm × τ = 0,10/0,15/0,20 mm, σ = {sg:g} (gemeinsame Obergrenze, gespeicherte Punktwolken)\n'
                     'lockerere Anforderungen verbessern nicht die Rekonstruktion; Hauptkennzahl d 6 / τ 0,15 schwarz', fontsize=7.8, y=0.998)
        F.foot(fig); fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.07, hspace=0.55, wspace=0.2)
        o = Path(str(out).replace('.png', f'_s{int(sg)}.png')); fig.savefig(o, dpi=150); plt.close(fig)
        index.append(dict(figure=K.rel(o), panel='alle', selection='Statistik', scope=f'Sensitivität d/τ σ {sg:g}'))


def gauss_amplitude(C, out, index):
    plt = F.plt_(); X = pd.read_csv(K.RESULTS / 'views.csv', keep_default_na=False, low_memory=False)
    X = X[(X.surface == K.GAUSS) & (X.regime == 'resource_cap_11695')].copy()
    X['amp'] = pd.to_numeric(X.amp_err_peak_mm, errors='coerce'); X['sigma'] = pd.to_numeric(X.sigma)
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.6))
    for ax, (sec, col, fam) in zip(axs, (('S1', 'size_mm', None), ('S2', 'layout', None))):
        q = X[X.section == sec]
        for fam in ('point', 'line'):
            h = q[q.family == fam]
            cfgs = sorted(h[col].unique(), key=lambda v: float(v) if col == 'size_mm' else str(v))
            import matplotlib
            cols = matplotlib.colormaps['viridis' if fam == 'point' else 'magma'](np.linspace(0, 0.85, max(1, len(cfgs))))
            for k, c in enumerate(cfgs):
                g = h[h[col] == c].groupby('sigma').amp.agg(['median', 'count']).reset_index()
                ax.plot(g.sigma + (k - len(cfgs) / 2) * 0.8, g['median'], '-o' if fam == 'point' else '--s', ms=2.5, lw=0.8, color=cols[k],
                        label=(size_label(fam, c) if col == 'size_mm' else lay_label(str(c))))
        ax.axhline(0, color='k', lw=0.6); ax.grid(alpha=0.3); ax.set_xticks(list(K.SIGMAS)); ax.set_xlabel('σ', fontsize=7); ax.set_ylabel('Amplitudenfehler Peak [mm] (Median der definierten Fälle)', fontsize=6.6)
        ax.set_title(f'{sec}: M10 + Gauß-Erhebung (wahre Amplitude 5 mm)', fontsize=7.5); ax.legend(fontsize=4.8, frameon=False, ncol=3)
    fig.suptitle('Zusatzfläche M10 + Gauß – Amplitudenfehler der Erhebung (Definition: dip = z_base − z_est auf ROI-Knoten, Peak = max in r ≤ 10 mm); getrennt von den Hauptflächen',
                 fontsize=7.8, y=0.995)
    F.foot(fig); fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.12, wspace=0.2)
    fig.savefig(out, dpi=150); plt.close(fig); index.append(dict(figure=K.rel(out), panel='alle', selection='Statistik', scope='Gauß Amplitude'))


def stats(which='main'):
    C = load_tables(); D = pd.read_csv(K.RESULTS / 'contrasts.csv', keep_default_na=False, low_memory=False)
    od = K.FIG / 'statistik'; od.mkdir(parents=True, exist_ok=True); idx = []
    if which in ('main', 'all'):
        Cm = C[C.surface_class == 'main']
        heatmaps(Cm, [('point', 'resource_cap_11695'), ('point', 'fixed_peak_1'), ('point', 'equal_integral_5850'), ('line', 'resource_cap_11695'), ('line', 'fixed_peak_1')], 'glob_Ctau',
                 'Abschnitt 1 – C_τ gesamt: Größe × Rauschen (Phasenmittel in %)', od / 'S12_S1_Heatmap_Ctau_gesamt.png', idx)
        heatmaps(Cm, [('point', 'resource_cap_11695'), ('point', 'fixed_peak_1'), ('point', 'equal_integral_5850'), ('line', 'resource_cap_11695'), ('line', 'fixed_peak_1')], 'loc_Ctau',
                 'Abschnitt 1 – C_τ lokal (r ≤ 15 mm): Größe × Rauschen (Phasenmittel in %)', od / 'S12_S1_Heatmap_Ctau_lokal.png', idx)
        for fam in ('point', 'line'):
            regs = ['resource_cap_11695', 'fixed_peak_1'] + (['equal_integral_5850'] if fam == 'point' else [])
            for reg in regs:
                sizes = sorted(Cm[(Cm.section == 'S1') & (Cm.family == fam)].size_mm.unique())
                for dom, mets in (('gesamt', METRICS_G), ('lokal', METRICS_L)):
                    curves_page(Cm, dict(section='S1', family=fam, regime=reg), sizes, 'size_mm', lambda v, f=fam: size_label(f, v), mets,
                                f"Abschnitt 1 – {'Punkte' if fam == 'point' else 'Linien'} · {REG_SHORT[reg]} · Rauschkurven je Größe ({dom})",
                                od / f"S12_S1_Rauschkurven_{'Punkte' if fam == 'point' else 'Linien'}_{reg}_{dom}.png", idx,
                                note=('Integral 5850: bestehende Niedrigbudget-Ergänzung, nicht Teil der gemeinsamen Obergrenze' if reg == 'equal_integral_5850' else ''))
        regime_page(Cm, D, od / 'S12_S1_Regimevergleich.png', idx)
        for ser, lays in S2_SERIES.items():
            for dom, mets in (('gesamt', METRICS_G), ('lokal', METRICS_L)):
                curves_page(Cm, dict(section='S2', family='point'), lays, 'layout', lay_label, mets, f'Abschnitt 2 – Punkte r 1 mm · Serie {ser} · Rauschantwort ({dom})',
                            od / f'S12_S2_Rauschantwort_Punkte_{ser}_{dom}.png', idx)
        for phi in (0.0, 45.0):
            lays = [f's{p:g}_phi{phi:g}' for p in K.S2_PITCHES]
            for dom, mets in (('gesamt', METRICS_G), ('lokal', METRICS_L)):
                curves_page(Cm, dict(section='S2', family='line'), lays, 'layout', lay_label, mets, f'Abschnitt 2 – Linien w 1 mm · {phi:g}° · Rauschantwort ({dom})',
                            od / f'S12_S2_Rauschantwort_Linien_phi{phi:g}_{dom}.png', idx)
        spacing_scan(Cm, 'glob_Ctau', 'C_τ gesamt', od / 'S12_S2_Abstandsscan_Ctau_gesamt.png', idx)
        spacing_scan(Cm, 'loc_Ctau', 'C_τ lokal', od / 'S12_S2_Abstandsscan_Ctau_lokal.png', idx)
        spacing_scan(Cm, 'glob_pt_p95', 'Punkt-P95 gesamt [mm]', od / 'S12_S2_Abstandsscan_P95_gesamt.png', idx)
        pair_effects(D, od / 'S12_S2_Paareffekte_RT_Orientierung.png', idx)
        resource_page(od / 'S12_Ressourcen.png', idx)
        reliability_page(Cm, 'S1', od / 'S12_S1_Zuverlaessigkeit.png', idx)
        reliability_page(Cm, 'S2', od / 'S12_S2_Zuverlaessigkeit.png', idx)
        phase_strips(Cm, od / 'S12_Phasenwerte.png', idx)
        if (K.RESULTS / 'sens_conditions.csv').exists():
            sensitivity_page(od / 'S12_Sensitivitaet_d_tau.png', idx)
    if which in ('gauss', 'all'):
        Cg = C[C.surface_class == 'supplement']
        for fam in ('point', 'line'):
            sizes = sorted(Cg[(Cg.section == 'S1') & (Cg.family == fam)].size_mm.unique())
            for reg in ['resource_cap_11695', 'fixed_peak_1'] + (['equal_integral_5850'] if fam == 'point' else []):
                curves_page(Cg, dict(section='S1', family=fam, regime=reg, surface=K.GAUSS), sizes, 'size_mm', lambda v, f=fam: size_label(f, v), METRICS_G + METRICS_L[:1],
                            f"Zusatz M10 + Gauß – Abschnitt 1 · {'Punkte' if fam == 'point' else 'Linien'} · {REG_SHORT[reg]}", od / f"S12_Gauss_S1_{'Punkte' if fam == 'point' else 'Linien'}_{reg}.png", idx)
        spacing_scan(Cg, 'glob_Ctau', 'C_τ gesamt', od / 'S12_Gauss_S2_Abstandsscan_Ctau_gesamt.png', idx, surfs=[K.GAUSS], title_extra=' (Zusatz M10 + Gauß)')
        spacing_scan(Cg, 'loc_Ctau', 'C_τ lokal', od / 'S12_Gauss_S2_Abstandsscan_Ctau_lokal.png', idx, surfs=[K.GAUSS], title_extra=' (Zusatz M10 + Gauß)')
        gauss_amplitude(Cg, od / 'S12_Gauss_Amplitude.png', idx)
    p = K.RESULTS / f'figure_index_stats_{which}.csv'; K.save_csv(pd.DataFrame(idx), p)
    return len(idx)


def _case_rows():
    X = pd.read_csv(K.RESULTS / 'views.csv', keep_default_na=False, low_memory=False)
    U = pd.read_csv(K.INPUTS, keep_default_na=False, float_precision='round_trip').set_index('input_key')
    for c in ('sigma', 'size_mm', 'rep', 'peak_f32', 'glob_Ctau', 'glob_pt_rmse', 'n_valid'):
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, U


def _row(X, U, **sel):
    q = X
    for k, v in sel.items():
        q = q[np.isclose(q[k], v)] if isinstance(v, float) else q[q[k] == v]
    if len(q) != 1:
        raise K.Stop('technical_stop', f'figure case not unique: {sel} -> {len(q)}')
    v = q.iloc[0]; u = U.loc[v.input_key].copy()
    for k in ('size_mm', 'layout', 'phase_label', 'sigma', 'rep', 'surface', 'family', 'regime', 'section', 'view_id', 'peak_f32', 'active_limit'):
        u[k] = v[k]
    u['input_key'] = v.input_key
    return u


def board_job(job):
    import matplotlib
    matplotlib.use('Agg')
    U = pd.read_csv(K.INPUTS, keep_default_na=False, float_precision='round_trip').set_index('input_key')
    cases = []
    for lab, d in job['cases']:
        s = pd.Series(d); s.name = d['input_key']; cases.append((lab, s))
    idx = []
    r = F.board(cases, job['title'], K.ROOT / job['out'], idx, job['meta'], foot_text=job.get('foot', ''))
    return dict(r, index=idx)


def atlas(X, U, rows_sel, cols, title, out, index, meta):

    plt = F.plt_(); nc = len(cols)
    fig, axs = plt.subplots(len(K.SIGMAS), nc, figsize=(2.15 * nc + 0.8, 2.3 * len(K.SIGMAS) + 1.0), squeeze=False)
    import s12_reuse as RU
    for i, sg in enumerate(K.SIGMAS):
        for j, (lab, sel) in enumerate(cols):
            ax = axs[i, j]; sel2 = dict(rows_sel, **sel, sigma=sg, rep=(0 if sg == 0 else rows_sel.get('rep_noisy', 0)))
            sel2.pop('rep_noisy', None)
            u = _row(X, U, **sel2)
            with np.load(RU.image_record_path(u).with_suffix('.npz')) as z:
                img = np.asarray(z['image'])
            v = X[X.view_id == u.view_id].iloc[0]
            ax.imshow(img, cmap='gray', vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
            st = 'Keine Rekonstruktion' if v.state == 'no_output' else f"C_τ {v.glob_Ctau:.2f} · n {int(v.n_valid)}"
            ax.set_title(f"{lab}\nPeak {u.peak_f32:.3f} · {st}", fontsize=5.6, color=('#b00020' if v.state == 'no_output' else 'k'))
            if j == 0:
                ax.set_ylabel(f'σ = {sg:g}', fontsize=7.5)
            index.append(dict(figure=K.rel(out), panel=f'σ {sg:g} / {lab}', input_key=u.input_key, view_id=u.view_id, surface=u.surface, family=u.family, size_mm=u.size_mm, layout=u.layout,
                              phase=u.phase_label, sigma=sg, rep=int(sel2['rep']), state=v.state, colour_scale='Graustufen 0–255 (alle Bilder gleich)', **meta))
    fig.suptitle(title + '\nKamerabilder 512 × 512, gleiche Grauskala 0–255; feste Beispielphase und Realisierung 0 (σ > 0), kein Mittelwert', fontsize=7.6, y=0.998)
    F.foot(fig); fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.03, hspace=0.42, wspace=0.08)
    fig.savefig(out, dpi=120); plt.close(fig)


# Select representative stored cases and generate image and point-cloud comparisons.
def visuals(which='main', workers=8):
    X, U = _case_rows(); idx = []; jobs = []
    od = K.FIG / 'visuell'; od.mkdir(parents=True, exist_ok=True)

    def case(**sel):
        u = _row(X, U, **sel); d = u.to_dict(); d['input_key'] = u.input_key
        return d
    surfs = list(K.MAIN) if which == 'main' else [K.GAUSS]

    for s in surfs:
        for fam, ph, sizes in (('point', 'sn_phase0', [0.5, 1.0, 1.5]), ('line', 'ph0', list(K.LINE_WIDTHS))):
            for reg in ('resource_cap_11695', 'fixed_peak_1'):
                base = dict(section='S1', surface=s, family=fam, regime=reg, phase_label=ph)
                tag = f"{K.SHORT[s].replace('+', '_').replace('ß', 'ss')}_{'Punkte' if fam == 'point' else 'Linien'}_{'Obergrenze' if reg.startswith('resource') else 'FesterPeak'}"
                if which == 'main':
                    atlas(X, U, base, [(size_label(fam, z), dict(size_mm=float(z))) for z in sizes],
                          f"Abschnitt 1 – {K.SHORT[s]} · {'Punkte (Raster 8×4)' if fam == 'point' else 'Linien s 8 mm, 0°'} · {REG_SHORT[reg]} · {ph}", od / f'S12_S1_Kamera_{tag}.png', idx,
                          dict(selection=f'feste Beispielphase {ph}, Realisierung 0'))
                sig_list = (0.0, 40.0, 80.0) if which == 'main' else ((40.0, 80.0) if reg.startswith('resource') else ())
                for sg in sig_list:
                    cs = [(f"{size_label(fam, z)} · Peak {case(**base, size_mm=float(z), sigma=sg, rep=0)['peak_f32']:.3f}", case(**base, size_mm=float(z), sigma=sg, rep=0)) for z in sizes]
                    jobs.append(dict(cases=cs, out=K.rel(od / f'S12_S1_3D_{tag}_s{int(sg)}.png'), meta=dict(selection=f'feste Beispielphase {ph}, Realisierung 0'),
                                     title=f"Abschnitt 1 – {K.SHORT[s]} · {'Punkte (Raster 8×4)' if fam == 'point' else 'Linien s 8 mm, 0°'} · {REG_SHORT[reg]} · σ = {sg:g} · {ph} · Realisierung 0"))

    series = [('p_par', 'point', [('R6×4', 'R6x4'), ('R8×4', 'R8x4'), ('R12×4', 'R12x4')]), ('p_perp', 'point', [('R8×2,5', 'R8x2.5'), ('R8×4', 'R8x4'), ('R8×8', 'R8x8')]),
              ('RT', 'point', [('R8×4', 'R8x4'), ('T8×4', 'T8x4')]), ('phi0', 'line', [(f's {p:g} mm', f's{p:g}_phi0') for p in (4.45, 6.3, 8.0, 10.0, 12.0)]),
              ('phi45', 'line', [(f's {p:g} mm', f's{p:g}_phi45') for p in (4.45, 6.3, 8.0, 10.0, 12.0)])]
    for s in surfs:
        for name, fam, lays in series:
            if which != 'main' and name not in ('p_par', 'phi0'):
                continue
            ph = 'pos33' if fam == 'point' else 'ph0'
            base = dict(section='S2', surface=s, family=fam, regime='resource_cap_11695', phase_label=ph)
            tag = f"{K.SHORT[s].replace('+', '_').replace('ß', 'ss')}_{name}"
            if which == 'main':
                atlas(X, U, base, [(lab, dict(layout=lay)) for lab, lay in lays], f"Abschnitt 2 – {K.SHORT[s]} · Serie {name} · gemeinsame Obergrenze · {ph}", od / f'S12_S2_Kamera_{tag}.png', idx,
                      dict(selection=f'feste Beispielposition {ph}, Realisierung 0'))
            for sg in ((0.0, 40.0, 80.0) if which == 'main' else (40.0, 80.0)):
                cs = [(f"{lab} · Peak {case(**base, layout=lay, sigma=sg, rep=0)['peak_f32']:.3f}", case(**base, layout=lay, sigma=sg, rep=0)) for lab, lay in lays]
                jobs.append(dict(cases=cs, out=K.rel(od / f'S12_S2_3D_{tag}_s{int(sg)}.png'), meta=dict(selection=f'feste Beispielposition {ph}, Realisierung 0'),
                                 title=f"Abschnitt 2 – {K.SHORT[s]} · Serie {name} · gemeinsame Obergrenze · σ = {sg:g} · {ph} · Realisierung 0"))

    if which == 'main':
        M = X[(X.surface_class == 'main') & (X.regime == 'resource_cap_11695')].copy(); M['order'] = np.arange(len(M))
        for sec in ('S1', 'S2'):
            for fam in ('point', 'line'):
                q = M[(M.section == sec) & (M.family == fam)]
                picks = [('niedrigstes C_τ gesamt', q.sort_values(['glob_Ctau', 'order']).iloc[0]), ('höchster definierter Punkt-RMSE', q[q.glob_pt_rmse.notna()].sort_values(['glob_pt_rmse', 'order'], ascending=[False, True]).iloc[0])]
                for rule, v in picks:
                    if sec == 'S1':
                        sizes = sorted(q.size_mm.unique()); k = sizes.index(v.size_mm); nb = sizes[max(0, k - 1):k + 2]
                        cs = [(f"{size_label(fam, z)}{' (Auswahl)' if z == v.size_mm else ''}", case(section=sec, surface=v.surface, family=fam, regime=v.regime, phase_label=v.phase_label,
                                                                                                     size_mm=float(z), sigma=float(v.sigma), rep=int(v.rep))) for z in nb]
                    else:
                        ser = next((s_ for s_ in (S2_SERIES['p_par'], S2_SERIES['p_perp'], [f's{p:g}_phi0' for p in K.S2_PITCHES], [f's{p:g}_phi45' for p in K.S2_PITCHES]) if v.layout in s_), [v.layout])
                        k = ser.index(v.layout); nb = ser[max(0, k - 1):k + 2]
                        cs = [(f"{lay_label(l)}{' (Auswahl)' if l == v.layout else ''}", case(section=sec, surface=v.surface, family=fam, regime=v.regime, phase_label=v.phase_label, layout=l,
                                                                                                sigma=float(v.sigma), rep=int(v.rep))) for l in nb]
                    jobs.append(dict(cases=cs, out=K.rel(od / f"S12_Diagnose_{sec}_{'Punkte' if fam == 'point' else 'Linien'}_{'Ctau_min' if 'C_τ' in rule else 'RMSE_max'}.png"),
                                     meta=dict(selection=f'nachträglich nach Regel: {rule} (gemeinsame Obergrenze, Hauptflächen; Gleichstand nach Manifestreihenfolge)'),
                                     title=f"Diagnose (nachträglich, Regel: {rule}) – Abschnitt {sec[1]} · {K.SHORT[v.surface]} · σ = {v.sigma:g} · {v.phase_label} · Realisierung {int(v.rep)} · Nachbarn gleicher Phase / gleichen Seeds",
                                     foot='Extremfall nach registrierter Regel, nicht repräsentativ; Nachbarn: gleiche Phase und Rauschrealisierung'))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(board_job, jobs))
    for r in res:
        idx += r['index']
    p = K.RESULTS / f'figure_index_visual_{which}.csv'; K.save_csv(pd.DataFrame(idx), p)
    return dict(boards=len(res), all_values_match=all(r['all_values_match'] for r in res), atlases=len({i['figure'] for i in idx if 'Kamera' in i['figure']}))


def finalize() -> dict:

    parts = []
    for name in ('stats_main', 'stats_gauss', 'visual_main', 'visual_gauss'):
        f = K.RESULTS / f'figure_index_{name}.csv'
        if f.exists():
            parts.append(pd.read_csv(f, keep_default_na=False).assign(group=name)); f.unlink()
    ff = K.CHECKS / 'first_figure.json'
    if ff.exists():
        parts.append(pd.DataFrame(K.load_json(ff)['index']).assign(group='first_batch'))
    idx = pd.concat(parts, ignore_index=True); K.save_csv(idx, K.RESULTS / 'figure_index.csv')
    figs = sorted({str(p_) for p_ in idx.figure})
    on_disk = sorted(K.rel(p_) for p_ in K.FIG.rglob('*.png'))
    ck = {n: K.load_json(K.CHECKS / f'{n}.json') for n in ('manifest_check', 'reuse_check', 'drills', 'replay_check', 'first_batch', 'stats_run', 'sensitivity_check', 'independent_check',
                                                           'reuse_sources_end') if (K.CHECKS / f'{n}.json').exists()}
    tm = K.load_json(K.CHECKS / 'run_timing.json') if (K.CHECKS / 'run_timing.json').exists() else {}
    summ = dict(t_local=K.now_local(), counts=ck['manifest_check']['counts'], manifest_problems=ck['manifest_check']['problems'],
                reuse=dict(n=ck['reuse_check']['n_reused'], all_three_layers=ck['reuse_check']['all_three_layers'], sources_unchanged_end=ck.get('reuse_sources_end', {}).get('ok'),
                           n_source_files=ck.get('reuse_sources_end', {}).get('n_files')),
                drills={k: v['ok'] for k, v in ck['drills'].items() if isinstance(v, dict)}, replay=dict(n=ck['replay_check']['n'], all_equal=ck['replay_check']['all_equal']),
                first_batch=dict(ok=ck['first_batch']['ok'], n=ck['first_batch']['timing']['n']), run_timing={k: dict(n=v['n'], wall_s=v['wall_s'], workers=v['workers']) for k, v in tm.items()},
                stats=ck.get('stats_run'), sensitivity=ck.get('sensitivity_check'),
                independent=dict(all_ok=ck['independent_check']['all_ok'], n=ck['independent_check']['n_cases'], ctau=ck['independent_check']['n_ctau_checked'],
                                 aggregation_ok=ck['independent_check']['aggregation_ok'], shared=ck['independent_check']['shared_inputs']),
                figures=dict(indexed=len(figs), png_on_disk=len(on_disk), not_indexed=[f for f in on_disk if f not in figs][:10]),
                note='Completeness and numerical consistency checks using a second code path.')
    K.save_json(K.CHECKS / 'selfcheck_summary.json', summ)
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['stats', 'visuals', 'finalize']); ap.add_argument('--set', default='main', choices=['main', 'gauss', 'all'])
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == 'stats':
        print(stats(a.set))
    elif a.cmd == 'finalize':
        print(json.dumps(K.clean(finalize()), ensure_ascii=False, indent=1)[:3000])
    else:
        print(json.dumps(K.clean(visuals(a.set, a.workers)), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
