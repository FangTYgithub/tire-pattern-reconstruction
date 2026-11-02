# Plot new-phase performance and representative reconstructions.

from __future__ import annotations

import numpy as np
import pandas as pd

import tr5_common as K

SURF = (('B_curv', 'B'), ('M10_curv', 'M10'), ('M24_curv', 'M24'))
SID = {'B_curv': 'B_curv', 'M10_curv': 'A_curv', 'M24_curv': 'C_curv'}
DOTCFG = ('r0.5/R8x4', 'r1.0/R8x4', 'r1.0/T8x4', 'r1.0/R8x6')
CFG_DE = {'r0.5/R8x4': 'r0,5 R8×4', 'r1.0/R8x4': 'r1 R8×4', 'r1.0/T8x4': 'r1 T8×4', 'r1.0/R8x6': 'r1 R8×6', 's8': 's8', 's10': 's10'}
COL = dict(D0='#1f77b4', DF='#d62728', P4='#2ca02c')
TAG = 'B2-Kern · 1 Strahl je Projektorpixel · neue registrierte Phasen'
LABEL_DE = dict(supported='wiederholt gestützt', qualified='nur eingeschränkt', insufficient='Evidenz unzureichend', reversed='Richtung umgekehrt',
                exploratory='explorativ (keine vorab registrierte Richtung)')
CONTRAST_DE = dict(C1='C1: r1 R8×4 − r0,5 R8×4', C2='C2: r1 T8×4 − r1 R8×4', C3='C3: r1 R8×6 − r1 R8×4', C4='C4: r1 T8×4 − r1 R8×6', C5='C5: s10 − s8 (w1, 0°)')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5})
    return plt


def _block_means(R: pd.DataFrame, keys: list, metric: str) -> pd.DataFrame:
    g = R.groupby(keys + ['block'])[metric].agg(lambda s: pd.to_numeric(s, errors='coerce').mean()).reset_index()
    return g


# Plot new-phase coverage and error curves for the selected pattern family.
def _perf_grid(R: pd.DataFrame, fam: str, out, fname: str, title: str, technical: bool) -> str:
    plt = _plt()
    from matplotlib.lines import Line2D
    D = R[R.family == fam].copy()
    cfgs = [c for c in (DOTCFG if fam == 'point' else ('s8', 's10')) if c in set(D.config)]
    sigmas = sorted(set(D.sigma)); meths = ['D0', 'DF'] if fam == 'point' else ['P4']
    rows = [('glob_Ctau', 100, 'Cτ global [%]'), ('loc_Ctau', 100, 'Cτ lokal [%]'), ('glob_pt_rmse', 1000, 'Punkt-RMSE global [µm]\n(bedingt, log. Achse bei Punkten)'),
            ('u_share', 100, 'u/(c+w+u) [%]\n(Text: Summe absolute w)')]
    if fam == 'line':
        rows[2] = ('glob_pt_rmse', 1000, 'Punkt-RMSE (●) / Flächen-RMSE (◆)\nglobal [µm], bedingt')
    fig, ax = plt.subplots(len(rows), 3, figsize=(15.5, 12.5 if fam == 'point' else 11.5), sharex=True)
    xs = {}; x = 0; labels = []
    for c in cfgs:
        for sg in sigmas:
            xs[(c, sg)] = x; labels.append(f'{CFG_DE[c]}\nσ{sg:g}'); x += 1
    off = {m: (i - (len(meths) - 1) / 2) * 0.32 for i, m in enumerate(meths)}
    for j, (sid, sname) in enumerate(SURF):
        S = D[D.surface == sid]
        for i, (met, sc, lab) in enumerate(rows):
            a = ax[i, j]
            for (c, sg), x0 in xs.items():
                for m in meths:
                    g = S[(S.config == c) & (S.sigma == sg) & (S.method == m)]
                    if g.empty:
                        continue
                    bm = _block_means(g, [], met)
                    v = pd.to_numeric(bm[met], errors='coerce') * sc
                    xx = x0 + off[m]
                    a.scatter(np.full(v.notna().sum(), xx) + np.linspace(-0.07, 0.07, v.notna().sum()), v.dropna(), s=9, color=COL[m], alpha=0.55, linewidths=0)
                    if v.notna().any():
                        a.plot([xx - 0.12, xx + 0.12], [v.mean()] * 2, color=COL[m], lw=2.2)
                    nd = int(pd.to_numeric(g[met], errors='coerce').isna().sum())
                    if nd:
                        a.annotate(f'{nd}×undef.', (xx, 0), xytext=(0, 2), textcoords='offset points', fontsize=6, ha='center', color='0.3', rotation=90, va='bottom')
                    if met == 'glob_Ctau':
                        no = int((g.state == 'no_output').sum())
                        if no:
                            a.annotate(f'{no}×kein Output', (xx, 0), xytext=(0, 12), textcoords='offset points', fontsize=6, ha='center', color='k', rotation=90, va='bottom')
                    if met == 'u_share':
                        w = int(g.full_w.sum())
                        a.annotate(f'w={w}', (xx, 1.0), xycoords=('data', 'axes fraction'), xytext=(0, -2 - 8 * meths.index(m)), textcoords='offset points', fontsize=5.5,
                                   ha='center', va='top', color=('#b00' if w else COL[m]))
                    if fam == 'line' and met == 'glob_pt_rmse':
                        bs = _block_means(g, [], 'glob_surf_rmse'); vs = pd.to_numeric(bs['glob_surf_rmse'], errors='coerce') * 1000
                        a.scatter(np.full(vs.notna().sum(), xx + 0.1), vs.dropna(), s=12, marker='D', color=COL[m], alpha=0.4, linewidths=0)
            a.grid(axis='y', color='0.9', lw=0.6)
            if j == 0:
                a.set_ylabel(lab)
            if i == 0:
                a.set_title(sname)
            if met in ('glob_Ctau', 'loc_Ctau'):
                a.set_ylim(bottom=0)
            if met == 'glob_pt_rmse' and fam == 'point':
                a.set_yscale('log')
        ax[-1, j].set_xticks(list(xs.values())); ax[-1, j].set_xticklabels(labels, fontsize=7)
    h = [Line2D([], [], marker='o', color=COL[m], ls='none', label=f'{m}: Phasenblock (Mittel der 2 Rauschrealisierungen)') for m in meths]
    h += [Line2D([], [], color='0.3', lw=2.2, label='Mittel über Phasenblöcke (gleichgewichtet)')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(title + ('\nTECHNISCHE PRÜFUNG (Checkpoint, ein Block)' if technical else '') + '\n' + TAG, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    p = K.guard(out / fname); fig.savefig(p, dpi=150); plt.close(fig)
    return K.rel(p)


def collapse_note(R: pd.DataFrame) -> str:

    out = []
    for (m, sg), g in R[R.family == 'point'].groupby(['method', 'sigma']):
        cs = g.groupby(['config', 'surface'])[['glob_Ctau', 'loc_Ctau']].mean() * 100
        mx, ml, ms = cs.glob_Ctau.max(), cs.loc_Ctau.max(), g.glob_Ctau.max() * 100
        if mx < 0.1:
            out.append(f'{m} σ{sg:g}: Mittelwert je Konfiguration und Fläche Cτ global ≤ {mx:.3f} %, lokal ≤ {ml:.3f} % (Einzelbild global max. {ms:.3f} %);\n'
                       f'beide Arme nahezu ausgefallen – Kennzeichnungen beruhen auf Hundertstel-pp, keine Musterempfehlung')
    return '\n'.join(out)


# Plot paired comparison curves from the supplied validation results.
def f3(C: pd.DataFrame, D: pd.DataFrame, out, technical: bool, note: str = '') -> str:
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, 3, figsize=(16, 11.0)); axs = axs.ravel()
    for i, cid in enumerate(('C1', 'C2', 'C3', 'C4', 'C5')):
        a = axs[i]; g = C[C.contrast == cid]
        if g.empty:
            a.set_visible(False); continue
        meths = ['D0', 'DF'] if cid != 'C5' else ['P4']; sigmas = sorted(set(g.sigma)); x = 0; ticks = []; tl = []
        for sid, sname in SURF:
            for sg in sigmas:
                for m in meths:
                    c = g[(g.surface == sid) & (g.sigma == sg) & (g.method == m)]
                    if c.empty:
                        x += 1; continue
                    c = c.iloc[0]
                    bv = [float(t.split(':')[1]) for t in str(c['blocks_glob_Ctau']).split(';') if ':' in t]
                    a.scatter(np.full(len(bv), x) + np.linspace(-0.12, 0.12, len(bv)), bv, s=10, color=COL[m], alpha=0.5, linewidths=0)
                    mu = c['mean_glob_Ctau']; lo, hi = c.get('lo_glob_Ctau'), c.get('hi_glob_Ctau')
                    if lo is not None and not pd.isna(lo):
                        a.errorbar(x, mu, yerr=[[mu - lo], [hi - mu]], fmt='o', color=COL[m], ms=5.5, capsize=3, lw=1.4, mec='k', mew=0.6)
                    else:
                        a.plot(x, mu, 'o', color=COL[m], ms=5.5, mec='k', mew=0.6)
                    e = c.get('expected_sign')
                    if e is not None and not pd.isna(e):
                        a.annotate('▲' if e > 0 else '▼', (x, 1.0), xycoords=('data', 'axes fraction'), xytext=(0, -2), textcoords='offset points', ha='center', va='top', fontsize=8,
                                   color='0.45')
                    ticks.append(x); tl.append(f'{sname}\nσ{sg:g}' + (f'\n{m}' if len(meths) > 1 else '')); x += 1
            x += 0.6
        a.axhline(0, color='0.3', lw=0.8); a.set_xticks(ticks); a.set_xticklabels(tl, fontsize=6.5); a.grid(axis='y', color='0.9', lw=0.6)
        nb = int(g.n_blocks.max()) if len(g) else 0
        lab = ''
        if D is not None and len(D):
            dd = D[D.contrast == cid]
            lab = '\n'.join(f"{r.method} σ{r.sigma:g}: {LABEL_DE.get(r.label, r.label)}" for r in dd.itertuples())
        a.set_title(f'{CONTRAST_DE[cid]}  ({nb} Phasenblöcke)', fontsize=9)
        if lab:
            a.text(0.0, -0.30, lab, transform=a.transAxes, fontsize=7, va='top')
        a.set_ylabel('ΔCτ global [pp] (vorne − hinten)')
    axs[5].axis('off')
    h = [Line2D([], [], marker='o', color=c, ls='none', label=m) for m, c in COL.items()]
    h += [Line2D([], [], marker='o', color='0.5', ls='none', ms=3, label='Phasenblock (2 Realisierungen gemittelt)'),
          Line2D([], [], marker='o', color='0.2', ls='-', label='Mittel ± deskriptives Block-Bootstrap-Intervall (2,5–97,5 %)'),
          Line2D([], [], marker='$▲$', color='0.45', ls='none', label='registrierte Richtung aus alten Daten (▲ +, ▼ −)')]
    axs[5].legend(handles=h, loc='center', frameon=False, fontsize=8)
    axs[5].text(0.02, 0.02, 'Nur gemessene σ-Stufen; Intervall ist deskriptiv (keine Signifikanz).\nAbdeckung mit vollem Nenner (kein Output = 0); keine Fehlerdifferenzen gezeigt.'
                + (('\n' + note) if note else ''), transform=axs[5].transAxes, fontsize=7.2, va='bottom')
    fig.suptitle('F3  Fünf Hauptvergleiche: gepaarte Differenz der globalen Präzisionsabdeckung Cτ' + (' — TECHNISCHE PRÜFUNG' if technical else '') + '\n' + TAG, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=7.5)
    p = K.guard(out / 'F3_Hauptvergleiche_C1_C5.png'); fig.savefig(p, dpi=150); plt.close(fig)
    return K.rel(p)


# Draw one case's image, point cloud, coverage map and comparison profiles.
def _case_panels(fig, gs_row, sel: dict, root, method: str, labels: tuple, title: str, FP: pd.DataFrame | None):
    import rdy_platform as RP
    import tr5_chain as CH
    dg = RP.dom_global(); S = None
    z = np.full((dg['ny'], dg['nx']), np.nan); ix = np.rint((dg['x'] - dg['x0']) / dg['step']).astype(int); iy = np.rint((dg['y'] - dg['y0']) / dg['step']).astype(int)
    z[iy, ix] = dg[f"z_{SID[sel['surface']]}"]
    ext = (dg['x0'] - dg['step'] / 2, dg['x0'] + dg['step'] * (dg['nx'] - 0.5), dg['y0'] - dg['step'] / 2, dg['y0'] + dg['step'] * (dg['ny'] - 0.5))
    axs = [fig.add_subplot(gs_row[k]) for k in range(4)]
    im = axs[0].imshow(z, origin='lower', extent=ext, cmap='viridis'); fig.colorbar(im, ax=axs[0], orientation='horizontal', fraction=0.05, pad=0.17, label='z Referenz / mm')
    axs[0].set_title('Referenzfläche (globaler Bewertungsbereich)', fontsize=8)
    data = []
    for fk in (sel['forward_key_left'], sel['forward_key_right']):
        meta, A, _ = CH.pack_read(root, fk); r = meta['results'][method]; mt = r['metrics']
        v = np.asarray(A[f'{method}__ev__valid'], bool); P = np.asarray(A[f'{method}__ev__pts'], float)[v]; e = P[:, 2] - np.asarray(A[f'{method}__ev__zr'], float)[v]
        data.append((P, e, mt, A['image'], r['state']))
    allerr = np.concatenate([d[1] for d in data if d[1].size]) if any(d[1].size for d in data) else np.zeros(0)
    lim = max(float(np.percentile(np.abs(allerr), 98)) if allerr.size else 0.0, 0.05)
    for k, ((P, e, mt, img, st), lab) in enumerate(zip(data, labels)):
        a = axs[k + 1]
        if P.size:
            sc = a.scatter(P[:, 0], P[:, 1], c=np.clip(e, -lim, lim), s=4, cmap='RdBu_r', vmin=-lim, vmax=lim, linewidths=0)
        else:
            a.text(0.5, 0.5, f'keine gültigen Punkte\n(Zustand {st})', ha='center', va='center', transform=a.transAxes, fontsize=9)
        a.contour(np.isfinite(z).astype(float), levels=[0.5], origin='lower', extent=ext, colors='0.45', linewidths=0.8)
        t = np.linspace(0, 2 * np.pi, 200); a.plot(15 * np.cos(t), 15 * np.sin(t), 'k--', lw=0.7)
        a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3]); a.set_aspect('equal')
        f = lambda x, s=1.0: ('undef.' if x is None else f'{x * s:.1f}')
        a.set_title(f"{lab}: {mt['n_valid']} gültige Ausgabepunkte, davon {mt['glob_n_valid']} im globalen Bereich\n"
                    f"Cτ glob {f(mt['glob_Ctau'], 100)} / lok {f(mt['loc_Ctau'], 100)} %, Punkt-RMSE glob. Bereich {f(mt['glob_pt_rmse'], 1000)} µm", fontsize=7.3)
        if len(gs_row) > 4:
            cam = gs_row[4].subgridspec(2, 1, hspace=0.35)[k]
            ca = fig.add_subplot(cam); ca.imshow(img, cmap='gray', vmin=0, vmax=255); ca.set_xticks([]); ca.set_yticks([])
            ca.set_title(f'Kamerabild {lab}', fontsize=7, pad=2)
        else:
            ins = a.inset_axes([0.70, 0.70, 0.29, 0.29]); ins.imshow(img, cmap='gray', vmin=0, vmax=255); ins.set_xticks([]); ins.set_yticks([])
            ins.set_title('Kamera', fontsize=6, pad=1)
    if any(d[1].size for d in data):
        fig.colorbar(sc, ax=axs[1:3], orientation='horizontal', fraction=0.05, pad=0.17,
                     label='Fehler z − z_ref / mm (rot: über, blau: unter der Referenz; gemeinsame Skala, Farbe begrenzt auf ±98.-Perzentil |Fehler|)')
    bins = np.linspace(-lim, lim, 41)
    for (P, e, mt, img, st), lab, col in zip(data, labels, ('#d62728', '#1f77b4')):
        if e.size:
            axs[3].hist(np.clip(e, -lim, lim), bins=bins, histtype='step', color=col, lw=1.4, label=f'{lab} (n={e.size})')
    axs[3].set_xlabel('Fehler z − z_ref / mm'); axs[3].set_ylabel('Anzahl gültiger Ausgabepunkte'); axs[3].legend(fontsize=6.5)
    axs[3].set_title('Fehlerverteilung (alle gültigen Ausgabepunkte)', fontsize=8)
    for a in axs[:3]:
        a.set_xlabel('x / mm'); a.set_ylabel('y / mm')
    extra = ''
    if FP is not None and len(FP) and method == 'D0':
        q = FP[(FP.forward_key_left.isin([sel['forward_key_left'], sel['forward_key_right']]))]
        extra = ' | DF gleiches Bild: ' + ', '.join(f"{CFG_DE.get(r.config, r.config)} ΔCτ(DF−D0) {r.d_glob_Ctau:+.2f} pp" for r in q.itertuples())
    axs[0].text(0.0, 1.32, title + extra, transform=axs[0].transAxes, fontsize=8.5, va='bottom')


def case_figure(sel: dict, root, FP, out, fname: str, labels: tuple, head: str, method: str, technical: bool, caption: str = '') -> str:
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(19.5, 12)); gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 0.8, 0.5], hspace=0.8, wspace=0.45)
    for i, kind in enumerate(('median', 'worst')):
        s = sel.get(kind)
        if s is None:
            continue
        nm = 'Median-nächster Fall' if kind == 'median' else 'ungünstigster Fall'
        t = (f"{nm}: {dict(SURF)[s['surface']]}, Block {s['block']}, Realisierung k={s['noise_k']}, σ{s['sigma']:g}, {method} — orientierte ΔCτ {s['oriented_diff_pp']:+.2f} pp "
             f"(Median aller {sel['n_candidates']} Paare {sel['median_value']:+.2f} pp)")
        _case_panels(fig, [gs[i, k] for k in range(5)], s, root, method, labels, t, FP)
    fig.suptitle(head + (' — TECHNISCHE PRÜFUNG' if technical else '') + '\n' + TAG + (f"\nHinweis: {sel['note']}" if sel.get('note') else ''), fontsize=10, y=1.06)
    if caption:
        fig.text(0.01, -0.02, caption, fontsize=8.2, va='top', ha='left')
    p = K.guard(out / fname); fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
    return K.rel(p)


def make_all(R, P, FP, C, CF, D, sel: dict, root, out, technical: bool = False) -> dict:
    figs = dict(F1=_perf_grid(R, 'point', out, 'F1_Punkte_Leistung_Risiko.png', 'F1  Punktraster: Abdeckung, Fehler und Identitätsrisiko (σ60 / σ80, D0 und DF getrennt)', technical),
                F2=_perf_grid(R, 'line', out, 'F2_Linien_Leistung_Risiko.png', 'F2  Linienraster w1, 0°: Abdeckung, Fehler und Identitätsrisiko (s8 / s10, σ40 / σ60, P4)', technical),
                F3=f3(C, D, out, technical, collapse_note(R)))
    if sel.get('F4') and sel['F4'].get('median'):
        figs['F4'] = case_figure(sel['F4'], root, FP, out, 'F4_Punkte_Rekonstruktion_C1.png', ('r1 R8×4', 'r0,5 R8×4'),
                                 f"F4  Punktraster C1 (r1 R8×4 vs. r0,5 R8×4), σ{sel['F4']['sigma']:g}, D0 — Median-nächster und ungünstigster Realisierungspaar", 'D0', technical)
    if sel.get('F5') and sel['F5'].get('median'):
        figs['F5'] = case_figure(sel['F5'], root, None, out, 'F5_Linien_Rekonstruktion_C5.png', ('s10', 's8'),
                                 f"F5  Linienraster C5 (s10 vs. s8, w1, 0°), σ{sel['F5']['sigma']:g}, P4 — Median-nächster und ungünstigster Realisierungspaar", 'P4', technical)
    return figs


def tables_md(C: pd.DataFrame, D: pd.DataFrame) -> str:

    lines = ['| Contrast | Method | sigma | Surface | Mean global delta Ctau / pp | Interval | Same-direction blocks | Local delta Ctau / pp | Global delta point RMSE / um (defined blocks) | Delta wrong share / pp | Delta unconfirmed share / pp | Delta wrong count | Delta no-output count |', '|' + '---|' * 13]
    for r in C.sort_values(['contrast', 'method', 'sigma', 'surface']).to_dict('records'):
        f = lambda x, s=1.0, d=2: ('—' if x is None or pd.isna(x) else f'{x * s:+.{d}f}')
        iv = '—' if r.get('lo_glob_Ctau') is None or pd.isna(r.get('lo_glob_Ctau')) else f"[{r['lo_glob_Ctau']:+.2f}, {r['hi_glob_Ctau']:+.2f}]"
        ns = '—' if r.get('expected_sign') is None or pd.isna(r.get('expected_sign')) else f"{int(r.get('n_blocks_expected_sign') or 0)}/{r['n_blocks']}"
        lines.append(f"| {r['contrast']} | {r['method']} | {r['sigma']:g} | {dict(SURF)[r['surface']]} | {f(r['mean_glob_Ctau'])} | {iv} | {ns} | {f(r['mean_loc_Ctau'])} | "
                     f"{f(r['mean_glob_pt_rmse'], 1000, 1)} ({r['nblocks_defined_glob_pt_rmse']}) | {f(r['mean_w_share'])} | {f(r['mean_u_share'])} | {f(r['mean_full_w'], 1, 1)} | {f(r['mean_no_output'], 1, 2)} |")
    lab = ['', '| Contrast | Method | sigma | Registered direction | Verdict | Surfaces with increased risk in the favoured arm |', '|---|---|---|---|---|---|']
    for r in (D.to_dict('records') if D is not None and len(D) else []):
        lab.append(f"| {r['contrast']} | {r['method']} | {r['sigma']:g} | {r['expected']} | {r['label_zh']} | {r['risk_higher_on_favoured_arm'] or '—'} |")
    return '\n'.join(lines + lab)
