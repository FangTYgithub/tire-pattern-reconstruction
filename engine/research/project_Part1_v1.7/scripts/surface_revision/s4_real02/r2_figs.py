# Plot layout effects, generation costs and reconstruction examples.

from __future__ import annotations

import numpy as np
import pandas as pd

import r2_common as K

W_IN = 6.3
STY = {'S0': dict(c='#222222', m='o', ls='-'), 'S1': dict(c='#1f77b4', m='s', ls='-'), 'S2': dict(c='#ff7f0e', m='^', ls='-'), 'S3': dict(c='#2ca02c', m='D', ls='-'),
       'S4': dict(c='#d62728', m='v', ls='--')}
LAB = {'S0': 'S0 regulär', 'S1': 'S1 frei', 'S2': 'S2 Mindestabstand', 'S3': 'S3 Lückenobergrenze', 'S4': 'S4 kombiniert'}
DODGE = {'S0': -4.0, 'S1': -2.0, 'S2': 0.0, 'S3': 2.0, 'S4': 4.0}
SN = {'B_curv': 'B', 'M10_curv': 'M10', 'M24_curv': 'M24'}
FAM = {'point': 'Punktmuster R8×4 (r = 1 mm, D0)', 'line': 'Linienmuster L6,3 (w = 1 mm, 0°, P4-Tabelle)'}
BASE = {'point': 'R8x4', 'line': 'L6.3'}
CASE_BLOCK = {'point': 'B021', 'line': 'B000'}
SRC = 'Quelle: S4-REAL-02 (symmetric_footprint, Ω₁), r2_figs.py'
CALLS = dict(figures=0, board_checks=[])


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7.2, 'axes.titlesize': 7.2, 'axes.labelsize': 7.0, 'xtick.labelsize': 6.6, 'ytick.labelsize': 6.6, 'legend.fontsize': 6.4, 'font.family': 'DejaVu Sans'})
    return plt


def _fmt(ax, x=True, y=True):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))
    if x:
        ax.xaxis.set_major_formatter(f)
    if y:
        ax.yaxis.set_major_formatter(f)


def _save(fig, out_dir, name):
    p = K.guard(out_dir / f'{name}.png'); p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig); CALLS['figures'] += 1
    return p


def _same_s3_s4(reg: dict, fam: str) -> tuple[int, int]:
    V = [v for v in reg['baselines'][BASE[fam]]['views'] if v['strategy'] in ('S3', 'S4')]
    by = {}
    for v in V:
        by.setdefault(v['block'], {})[v['strategy']] = v['layout']
    return sum(1 for b in by.values() if b.get('S3') == b.get('S4')), len(by)


def _labels(reg, fam):
    n, t = _same_s3_s4(reg, fam); lab = dict(LAB)
    lab['S4'] = f'S4 kombiniert (Layout = S3 in {n}/{t} Blöcken)'
    return lab


def _abs(P, fam, s, k, metric):
    return P[(P.table_type == 'absolute') & (P.family == fam) & (P.surface == s) & (P.contrast == k) & (P.metric == metric)].sort_values('sigma')


def _curve(ax, P, fam, s, metric, scale, strategies, lab, rng=True, defined=None):
    for k in strategies:
        d = _abs(P, fam, s, k, metric)
        if not len(d):
            continue
        st = STY[k]; x = d.sigma.to_numpy(float) + DODGE[k]; y = pd.to_numeric(d['main'], errors='coerce').to_numpy(float) * scale
        ax.plot(x, y, color=st['c'], ls=st['ls'], lw=0.9, label=lab[k], zorder=3)
        full = (d.n_pairs_defined.to_numpy(float) >= d.n_views.to_numpy(float)) if defined else np.ones(len(d), bool)
        ax.scatter(x[full], y[full], s=10, marker=st['m'], color=st['c'], zorder=4)
        ax.scatter(x[~full], y[~full], s=10, marker=st['m'], facecolors='white', edgecolors=st['c'], linewidths=0.8, zorder=4)
        if rng:
            ax.vlines(x, pd.to_numeric(d.block_min, errors='coerce') * scale, pd.to_numeric(d.block_max, errors='coerce') * scale, color=st['c'], lw=0.6, alpha=0.55, zorder=2)
    ax.set_xticks(K.SIGMAS); _fmt(ax); ax.grid(alpha=0.25, lw=0.4)


def fig_ctau(fam, P, reg, out_dir, name, strategies=K.STRATS):
    plt = _plt(); lab = _labels(reg, fam)
    fig, axs = plt.subplots(2, 3, figsize=(W_IN, 4.6), sharex=True)
    for j, s in enumerate(K.SURF):
        for i, (mtr, rt) in enumerate((('glob_Ctau', 'Gesamtbereich Ω₁'), ('loc_Ctau', 'lokal (r ≤ 15 mm)'))):
            ax = axs[i, j]; _curve(ax, P, fam, s, mtr, 100.0, strategies, lab)
            ax.set_title(f'{SN[s]} – {rt}')
            if j == 0:
                ax.set_ylabel('C_τ [%]')
            if i == 1:
                ax.set_xlabel('Rauschen σ [Grauwerte]')
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.055))
    fig.suptitle(f'{FAM[fam]}: C_τ (d = 6 mm, τ = 0,15 mm)', y=0.995)
    fig.text(0.01, 0.005, 'Linie: Median über 8 Blöcke der Blockmediane (H0–H3); S0: Median der 4 Phasen. Balken: Spannweite der Blockmediane (S0: der Phasen).\n'
             'Werte leicht horizontal versetzt. ' + SRC + ', paired_summary.csv', fontsize=5.6, color='#555555')
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    return _save(fig, out_dir, name)


def fig_err(fam, P, reg, out_dir, name, strategies=K.STRATS):
    plt = _plt(); lab = _labels(reg, fam)
    rows = (('glob_pt_rmse', 'Punkt-RMSE [mm]\n(gültige Punkte in Ω₁)', 1.0), ('glob_surf_rmse', 'Flächen-RMSE [mm]\n(interpolierbare Knoten)', 1.0),
            ('glob_sup_surf_rmse', 'Flächen-RMSE [mm]\n(gestützte Knoten, d = 6 mm)', 1.0))
    fig, axs = plt.subplots(4, 3, figsize=(W_IN, 7.4), sharex=True)
    for j, s in enumerate(K.SURF):
        for i, (mtr, yl, sc) in enumerate(rows):
            ax = axs[i, j]; _curve(ax, P, fam, s, mtr, sc, strategies, lab, defined=True)
            ax.set_yscale('log')
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
            if j == 0:
                ax.set_ylabel(yl)
            if i == 0:
                ax.set_title(SN[s])
        ax = axs[3, j]
        for k in strategies:
            d = _abs(P, fam, s, k, 'glob_surf_rmse')
            if len(d):
                ax.plot(d.sigma + DODGE[k], 100.0 * d.n_pairs_defined / d.n_views, color=STY[k]['c'], ls=STY[k]['ls'], marker=STY[k]['m'], ms=3, lw=0.9)
        ax.set_ylim(-5, 105); ax.set_xticks(K.SIGMAS); _fmt(ax); ax.grid(alpha=0.25, lw=0.4); ax.set_xlabel('Rauschen σ [Grauwerte]')
        if j == 0:
            ax.set_ylabel('Fälle mit definiertem\nFlächenfehler [%]')
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle(f'{FAM[fam]}: Punkt- und Flächenfehler mit gültigem Nenner', y=0.995)
    fig.text(0.01, 0.004, 'Hohle Marker: nicht alle Ansichten definiert (ohne Ausgabe / < 3 Punkte); fehlende Werte sind Lücken, keine Nullfehler.\n'
             'Flächen-RMSE über interpolierbare Knoten ist nicht an die Stützbedingung von C_τ gebunden. ' + SRC, fontsize=5.6, color='#555555')
    fig.tight_layout(rect=(0, 0.09, 1, 0.975))
    return _save(fig, out_dir, name)


def fig_risk(P, reg, out_dir, name, strategies=K.STRATS):
    plt = _plt()
    rows = (('wrong_rate', 'Fehlzuordnungen\nw/(c+w) [%]'), ('unconf_share', 'nicht bestätigbar\nu/(c+w+u) [%]'), ('no_output', 'ohne Ausgabe\n[% der Ansichten]'))
    fig, axs = plt.subplots(3, 6, figsize=(W_IN, 5.0), sharex=True)
    for jf, fam in enumerate(('point', 'line')):
        lab = _labels(reg, fam)
        for js, s in enumerate(K.SURF):
            j = 3 * jf + js
            for i, (mtr, yl) in enumerate(rows):
                ax = axs[i, j]
                for k in strategies:
                    d = _abs(P, fam, s, k, mtr)
                    if len(d):
                        ax.plot(d.sigma + DODGE[k], 100.0 * pd.to_numeric(d.mean_pairs, errors='coerce'), color=STY[k]['c'], ls=STY[k]['ls'], marker=STY[k]['m'], ms=2.6, lw=0.8,
                                label=(lab[k] if jf == 0 else None))
                ax.set_xticks([0, 50, 100]); _fmt(ax); ax.grid(alpha=0.25, lw=0.4)
                if i == 0:
                    ax.set_title(f"{'Punkte' if fam == 'point' else 'Linien'} {SN[s]}")
                if j == 0:
                    ax.set_ylabel(yl)
                if i == 2:
                    ax.set_xlabel('σ')
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.05))
    fig.suptitle('Risiken je Strategie: Fehlzuordnung, nicht bestätigbare Punkte, keine Ausgabe', y=0.995)
    fig.text(0.01, 0.004, 'Mittelwert über alle Ansichten mit definiertem Nenner (Median wäre meist 0). Identität: Punkte = Quellidentität aus dem σ0-Bild derselben Version,\n'
             'Linien = Tabellenwahrheit. Nenner 0 = undefiniert (Lücke). ' + SRC + ', paired_summary.csv', fontsize=5.6, color='#555555')
    fig.tight_layout(rect=(0, 0.12, 1, 0.975))
    return _save(fig, out_dir, name)


def fig_cost(reg, out_dir, name):
    plt = _plt()
    fig, axs = plt.subplots(5, 2, figsize=(W_IN, 8.9))
    blocks = [f'B{b:03d}' for b in K.BLOCKS]; ss = ('S1', 'S2', 'S3', 'S4')
    for jf, fam in enumerate(('point', 'line')):
        B = reg['baselines'][BASE[fam]]; L = B['layouts']; V = [v for v in B['views'] if v['strategy'] != 'S0']
        lay = {(v['block'], v['strategy']): v['layout'] for v in V}; cost = {(v['block'], v['strategy']): v['n_mask_candidates'] for v in V}
        ax = axs[0, jf]; M = np.zeros((len(ss), len(blocks)))
        for ib, b in enumerate(blocks):
            names = []
            for k in ss:
                if lay[(b, k)] not in names:
                    names.append(lay[(b, k)])
            for ik, k in enumerate(ss):
                M[ik, ib] = names.index(lay[(b, k)])
                ax.text(ib, ik, f"c{L[lay[(b, k)]]['cand']}", ha='center', va='center', fontsize=5.6)
        ax.imshow(M, cmap='Pastel1', vmin=0, vmax=8, aspect='auto')
        ax.set_xticks(range(len(blocks))); ax.set_xticklabels(blocks, fontsize=5.6); ax.set_yticks(range(len(ss))); ax.set_yticklabels(ss)
        ax.set_title(f"{'Punkte R8×4' if fam == 'point' else 'Linien L6,3'}: gewähltes Layout je Block")
        p0, h0 = B['p0_mm'], B['h0_mm']
        for i, (key, ref, yl, lim) in enumerate((('g_min_mm', p0, 'g_min / p₀', (0.70, 'α = 0,70')), ('h_mm', h0, 'h / h₀', (1.15, 'β = 1,15')),
                                                  ('delta_rms_mm', p0, 'RMS-Verschiebung / p₀', None))):
            ax = axs[1 + i, jf]
            for ik, k in enumerate(ss):
                xs = ik + np.linspace(-0.25, 0.25, len(blocks))
                ax.scatter(xs, [L[lay[(b, k)]][key] / ref for b in blocks], s=9, color=STY[k]['c'], marker=STY[k]['m'])
            ax.axhline(L['regular'][key] / ref, color='#222222', lw=0.7, ls=':', label='S0 regulär')
            if lim is not None:
                ax.axhline(lim[0], color='#888888', lw=0.7, ls='--', label=lim[1])
            ax.set_xticks(range(len(ss))); ax.set_xticklabels(ss); ax.set_ylabel(yl); _fmt(ax, x=False); ax.grid(alpha=0.25, lw=0.4); ax.legend(loc='best', frameon=False)
        ax = axs[4, jf]
        for ik, k in enumerate(ss):
            xs = ik + np.linspace(-0.25, 0.25, len(blocks))
            ax.scatter(xs, [cost[(b, k)] for b in blocks], s=9, color=STY[k]['c'], marker=STY[k]['m'])
        ax.set_xticks(range(len(ss))); ax.set_xticklabels(ss); ax.set_ylabel('gültige Kandidaten\nvon 1000'); ax.set_ylim(0, 1050); ax.grid(alpha=0.25, lw=0.4)
    fig.suptitle('Generierungsaufwand und Layoutunterschiede (j = 0,20, 1000 Kandidaten je Block)', y=0.997)
    fig.text(0.01, 0.004, 'Oben: gleiche Farbe in einer Blockspalte = identisches Layout, c = Kandidatenindex. Darunter je Strategie 8 Blöcke B000–B021 (horizontal verteilt).\n'
             'Gewählt wird der erste Kandidat der registrierten Reihenfolge, der alle Bedingungen erfüllt (ausgewertete Raster je Auswahl = 1); keine Generierungsfehler.\n'
             + SRC + ', layouts.json', fontsize=5.6, color='#555555')
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    return _save(fig, out_dir, name)


def _node_state(pts_valid: np.ndarray, dom: dict, z_ref: np.ndarray) -> dict:
    from surface_eval_r3 import surface_eval_r3_dev
    n = len(dom['x'])
    if len(pts_valid) < 3:
        return dict(state=np.zeros(n, np.int8), err=np.full(n, np.nan), Ctau=0.0, note='fewer than 3 valid points')
    D = dict(x=dom['x'], y=dom['y'], z_ref=z_ref, area=np.full(n, dom['step'] ** 2), hash=dom['hash'])
    ev = surface_eval_r3_dev(pts_valid, None, D, [K.D_SUPPORT], [K.TAU], return_fields=True, support_rule='exact')
    f = ev['fields']; interp = (np.asarray(f['simplex']) >= 0) & np.isfinite(np.asarray(f['z_est'], np.float64))
    sup = interp & (np.asarray(f['support_measure'], np.float64) <= K.D_SUPPORT)
    st = np.zeros(n, np.int8); st[interp] = 1; st[sup] = 2
    err = np.asarray(f['z_est'], np.float64) - z_ref
    Ctau = float(ev['subdomains']['global']['by_d'][str(float(K.D_SUPPORT))]['C_tau'][str(float(K.TAU))] or 0.0)
    return dict(state=st, err=err, Ctau=Ctau, note=ev['triangulation'])


def _grid(dom: dict, vals: np.ndarray, fill=np.nan) -> np.ndarray:
    G = np.full((dom['ny'], dom['nx']) + np.shape(vals)[1:], fill, dtype=np.float64)
    ix = np.rint((dom['x'] - dom['x0']) / dom['step']).astype(int); iy = np.rint((dom['y'] - dom['y0']) / dom['step']).astype(int)
    G[iy, ix] = vals
    return G


# Draw images and reconstruction maps for a selected layout comparison.
def fig_board(fam: str, surface: str, C: pd.DataFrame, reg: dict, out_dir, name: str, sigmas=(0.0, 60.0), strategies=('S0', 'S1', 'S2', 'S3'), eval_dir=None, emax=0.30):
    import matplotlib.colors as mcolors
    import r2_eval as E
    plt = _plt(); eval_dir = eval_dir or K.EVL
    dom = E.domain('omega1'); sid = K.RK.SURF_MAP[surface]; z_ref = np.asarray(dom[f'z_{sid}'], np.float64)
    blk = CASE_BLOCK[fam]; base = BASE[fam]
    views = {v['strategy']: v for v in reg['baselines'][base]['views'] if v['strategy'] == 'S0' or v['block'] == blk}
    cols = list(strategies)
    s4_same = views['S4']['layout'] == views['S3']['layout']
    if not s4_same and 'S4' not in cols:
        cols.append('S4')
    ext = (dom['x0'] - dom['step'] / 2, dom['x0'] + (dom['nx'] - 0.5) * dom['step'], dom['y0'] - dom['step'] / 2, dom['y0'] + (dom['ny'] - 0.5) * dom['step'])
    nr, nc = len(sigmas), len(cols) + 1
    fig, axs = plt.subplots(nr, nc, figsize=(W_IN, 1.32 * nr + 1.05), squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.885, top=0.86, bottom=0.12, wspace=0.07, hspace=0.24)
    cmap = plt.get_cmap('RdBu_r'); norm = mcolors.Normalize(-emax, emax)
    zr = _grid(dom, z_ref); a0 = axs[0, 0]
    im0 = a0.imshow(zr, origin='lower', extent=ext, cmap='viridis'); a0.set_title('Referenzfläche z [mm]'); a0.tick_params(labelsize=5.2)
    cax0 = a0.inset_axes([0.0, -0.36, 1.0, 0.06]); cb0 = fig.colorbar(im0, cax=cax0, orientation='horizontal'); cb0.ax.tick_params(labelsize=5.0)
    for i in range(1, nr):
        axs[i, 0].axis('off')
    from matplotlib.patches import Patch
    axs[-1, 0].legend(handles=[Patch(color='#d9d9d9', label='außerhalb der Hülle'), Patch(color='#e6cf9c', label='interpolierbar,\nnicht gestützt'),
                               Patch(color=cmap(0.88), label='gestützt, Farbe =\nFehler z_est − z_ref'), plt.Line2D([], [], color='k', marker='.', ls='', ms=2.5, label='gültige Punkte')],
                      loc='lower center', bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=5.5)
    checks = []
    for i, sg in enumerate(sigmas):
        for j, k in enumerate(cols):
            ax = axs[i, j + 1]; v = views[k]
            row = C[(C.family == fam) & (C.surface == surface) & (C.strategy == k) & (C.phase == 'H0') & (C.sigma == float(sg)) & ((C.block == (blk if k != 'S0' else '')) | (k == 'S0'))]
            if k == 'S0':
                row = row[row.layout == 'regular']
            ax.set_xticks([]); ax.set_yticks([])
            title = f'{k}' + (' (= S4)' if (k == 'S3' and s4_same) else '') + f' · σ = {sg:g}'
            if not len(row):
                ax.set_title(title); ax.text(0.5, 0.5, 'nicht im Teilsatz', ha='center', va='center', transform=ax.transAxes, fontsize=6); continue
            r = row.iloc[0]; z = K.load_npz(eval_dir / f'{r.eval_key}.npz'); Pv = np.asarray(z['pts'], np.float64)[np.asarray(z['valid'], bool)]
            ns = _node_state(Pv, dom, z_ref)
            ok = abs(ns['Ctau'] - float(r.glob_Ctau)) <= 1e-12
            checks.append(dict(board=name, eval_key=r.eval_key, strategy=k, sigma=float(sg), Ctau_board=ns['Ctau'], Ctau_table=float(r.glob_Ctau), equal=bool(ok)))
            if not ok:
                raise K.Stop('check', f'board {name} {k} sigma {sg}: recomputed C_tau {ns["Ctau"]} != table {r.glob_Ctau}')
            rgba = np.zeros((len(dom['x']), 4)); rgba[:] = mcolors.to_rgba('#d9d9d9')
            rgba[ns['state'] == 1] = mcolors.to_rgba('#e6cf9c'); s2 = ns['state'] == 2
            rgba[s2] = cmap(norm(np.clip(ns['err'][s2], -emax, emax)))
            G = _grid(dom, rgba, fill=1.0)
            ax.imshow(G, origin='lower', extent=ext, interpolation='nearest')
            if len(Pv):
                ax.scatter(Pv[:, 0], Pv[:, 1], s=(1.2 if len(Pv) < 2000 else 0.06), c='k', marker='.', linewidths=0, alpha=0.7)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_title(title)
            txt = 'keine Ausgabe' if r.state == 'no_output' else f"C_τ = {100 * float(r.glob_Ctau):.1f} %\nn = {int(r.n_valid)}".replace('.', ',')
            ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=5.3, va='bottom', bbox=dict(facecolor='white', alpha=0.8, lw=0, pad=0.8))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.905, 0.2, 0.012, 0.6]); cb = fig.colorbar(sm, cax=cax); cb.set_label('Fehler z_est − z_ref [mm]\n(gestützte Knoten)', fontsize=5.8); cb.ax.tick_params(labelsize=5.2)
    _fmt(cb.ax, x=False); _fmt(cb0.ax, y=False)
    note = 'S4 = S3 (identisches Layout, keine eigene Spalte)' if s4_same else 'S4 ≠ S3 (eigene Spalte)'
    fig.suptitle(f"{'Punkte R8×4 (D0)' if fam == 'point' else 'Linien L6,3 (P4-Tabelle)'} – {SN[surface]}: Phase H0, Block {blk}, Draufsicht auf Ω₁", y=0.985)
    fig.text(0.01, 0.008, f'Vor der Bildgebung festgelegte Darstellungsauswahl (kein Häufigkeitsbefund); {note}.\n'
             f'Farbskala ±{str(emax).replace(".", ",")} mm, τ = 0,15 mm; Achsen x, y in mm. Knotenstatus aus der gespeicherten Punktwolke neu berechnet, C_τ gegen Tabelle geprüft.\n'
             + SRC + ', eval/*.npz', fontsize=5.2, color='#555555')
    p = _save(fig, out_dir, name); CALLS['board_checks'].extend(checks)
    return p


def make_all(C: pd.DataFrame, P: pd.DataFrame, reg: dict, out_dir, entry: bool = False) -> dict:
    CALLS['figures'] = 0; CALLS['board_checks'] = []; out = []
    if entry:
        st = ('S0', 'S2', 'S3')
        out.append(fig_ctau('point', P, reg, out_dir, 'E01_Punkte_Ctau_Eingang', strategies=st))
        out.append(fig_board('point', 'M24_curv', C, reg, out_dir, 'E02_Fallbrett_Punkte_M24_Eingang', sigmas=(0.0, 100.0), strategies=st))
        out.append(fig_board('line', 'M24_curv', C, reg, out_dir, 'E03_Fallbrett_Linien_M24_Eingang', sigmas=(0.0, 100.0), strategies=st))
    else:
        out.append(fig_ctau('point', P, reg, out_dir, 'F01_Punkte_Ctau'))
        out.append(fig_ctau('line', P, reg, out_dir, 'F02_Linien_Ctau'))
        out.append(fig_err('point', P, reg, out_dir, 'F03_Punkte_Fehler'))
        out.append(fig_err('line', P, reg, out_dir, 'F04_Linien_Fehler'))
        out.append(fig_risk(P, reg, out_dir, 'F05_Risiken'))
        out.append(fig_cost(reg, out_dir, 'F06_Generierung_Layouts'))
        n = 7
        for fam in ('point', 'line'):
            for s in K.SURF:
                out.append(fig_board(fam, s, C, reg, out_dir, f"F{n:02d}_Fallbrett_{'Punkte' if fam == 'point' else 'Linien'}_{SN[s]}")); n += 1
    return dict(n_figures=CALLS['figures'], files=[K.rel(p) for p in out], board_checks=list(CALLS['board_checks']))
