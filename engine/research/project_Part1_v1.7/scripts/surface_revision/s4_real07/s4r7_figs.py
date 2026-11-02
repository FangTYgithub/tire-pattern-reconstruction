# Plot held-out layout comparisons and representative point clouds.

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import s4r7_common as H

K = H.K
SRC = 'Quelle: S4-REAL-07 (Validierung Blöcke B024–B029, j 0,30, Phasen R0/R1; H0-Anker getrennt), s4r7_figs.py'
SRC6 = 'Quelle: S4-REAL-06-Daten (unverändert), neu ausgewertet in S4-REAL-07 A1, s4r7_figs.py'
SN = {'B_curv': 'B', 'M10_curv': 'M10', 'M24_curv': 'M24'}
SURFS = ('B_curv', 'M10_curv', 'M24_curv')
ARMS = ('S1', 'S2*', 'S3', 'S4*')
ACOL = {'S0': '#4d4d4d', 'S1': '#1f77b4', 'S2*': '#ff7f0e', 'S3': '#8c564b', 'S4*': '#2ca02c'}


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


def _save(fig, name: str) -> str:
    p = H.FIGS / name; H.guard(p); fig.savefig(p, dpi=300, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return K.rel(p)


LAY6 = {'regular': 'S0', 'j0.10_B000_c0': 'S1 j0,10', 'j0.30_B000_c39': 'S2* j0,30', 'j0.30_B000_c134': 'S4* j0,30', 'j0.20_B000_c0': 'S1-RMS j0,20', 'j0.20_B000_c63': 'S3 j0,20'}
ORD6 = ['S0', 'S1 j0,10', 'S2* j0,30', 'S4* j0,30', 'S1-RMS j0,20', 'S3 j0,20']
COL6 = dict(zip(ORD6, ['#4d4d4d', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']))


def fig1() -> str:
    plt = _plt(); D = pd.read_csv(H.OUT / '_parts' / 'A1_union.csv', low_memory=False, float_precision='round_trip'); D['lay'] = D.layout.map(LAY6)
    a1 = H.run_load()['A1']['anchors']; grp = [(s, ph) for s in SURFS for ph in ('H0', 'H1')]
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.9))
    for gi, (s, ph) in enumerate(grp):
        for li, lab in enumerate(ORD6):
            for sg, mk in ((0.0, 'o'), (40.0, '^')):
                x = D[(D.surface == s) & (D.phase == ph) & (D.lay == lab) & (D.sigma == sg)]
                if not len(x):
                    continue
                xx = gi + (li - 2.5) * 0.11 + (0.03 if sg else -0.03); r = x.iloc[0]
                axs[0].scatter(xx, r.union_rejected, s=9, marker=mk, c=COL6[lab], lw=0)
                if r.both_rejected > 0:
                    axs[0].scatter(xx, r.both_rejected, s=22, marker='_', c='k', lw=0.6)
                axs[1].scatter(xx, r.correct_lost, s=9, marker=mk, c=COL6[lab], lw=0)
                axs[2].scatter(xx, r.dCtau_pp, s=9, marker=mk, c=COL6[lab], lw=0)
    for ax, t in zip(axs, ('(a) abgelehnte Punkte: unbestätigbar ∪ grob\n(global; Strich = Schnittmenge)', '(b) verlorene bestätigt\nkorrekte Punkte (global)', '(c) Δ C_τ global / pp\n(Kandidat − P4 alt)')):
        ax.set_xticks(range(len(grp))); ax.set_xticklabels([f'{ph}\n{"Band" if ph == "H0" else "ohne"}' for s, ph in grp], fontsize=5.0)
        for si, s in enumerate(SURFS):
            ax.text(2 * si + 0.5, -0.2, SN[s], transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.0)
        ax.axhline(0, c='#888888', lw=0.5); ax.set_title(t)
        for b in (1.5, 3.5):
            ax.axvline(b, c='#dddddd', lw=0.5)
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], marker='o', ls='', c=COL6[l], ms=3, label=l) for l in ORD6] + [Line2D([], [], marker='o', ls='', c='#999999', ms=3, label='σ = 0'),
                                                                                       Line2D([], [], marker='^', ls='', c='#999999', ms=3, label='σ = 40')]
    fig.legend(handles=hd, loc='lower center', ncol=8, bbox_to_anchor=(0.5, -0.11), frameon=False)
    fig.suptitle('Korrektur zu S4-REAL-06 F1: Randbandkandidat gegen P4 (gleiches Bild) – Vereinigung statt Summe', y=1.02)
    fig.text(0.01, -0.2, f'Unbestätigbare und grobe Punkte können zusammenfallen; die frühere Summe zählte sie doppelt. Vereinigung: B {a1["B_union"]}, M10 {a1["M10_union"]}, '
             f'gesamt {a1["total_union"]} Punkte (typischer M10-Fall σ = 0: {a1["typical_M10_sigma0"][0]});\nin allen 72 Paaren fallen beide Klassen vollständig zusammen (Schnittmenge = Vereinigung). Gehört zu S4-REAL-06 (Entwicklungsblock B000, H0/H1); keine neue Messung. ' + SRC6,
             fontsize=5.0, color='#555555')
    fig.tight_layout()
    return _save(fig, 'F1_Korrektur_S4-REAL-06_Vereinigung.png')


def fig2() -> str:
    plt = _plt(); R = _res(); P = R[R.record_type == 'clip'].copy()
    groups = [('main', 'R0', 'Haupt R0'), ('main', 'R1', 'Haupt R1'), ('anchor', 'H0', 'Anker H0')]
    fig, axs = plt.subplots(3, 3, figsize=(6.3, 6.3), sharex=True)
    for ci, s in enumerate(SURFS):
        for gi, (st, ph, _lab) in enumerate(groups):
            g = P[(P.surface == s) & (P.set == st) & (P.phase == ph)]
            lays = sorted(g.layout.unique(), key=lambda l: (l != 'regular', l))
            for li, lay in enumerate(lays):
                for sg, col in ((0.0, '#1f77b4'), (40.0, '#d62728')):
                    x = g[(g.layout == lay) & (g.sigma == sg)]
                    if not len(x):
                        continue
                    gl, lo = x[x.domain == 'glob'].iloc[0], x[x.domain == 'loc'].iloc[0]
                    xx = gi + (li - (len(lays) - 1) / 2) * (0.6 / max(len(lays), 1)) + (0.03 if sg else -0.03); mk = 's' if lay == 'regular' else 'o'
                    axs[0, ci].scatter(xx, gl.union_rejected, s=8, marker=mk, c=col, lw=0)
                    if gl.both_rejected > 0:
                        axs[0, ci].scatter(xx, gl.both_rejected, s=18, marker='_', c='k', lw=0.5)
                    axs[1, ci].scatter(xx, gl.correct_lost, s=8, marker=mk, c=col, lw=0)
                    axs[2, ci].scatter(xx, 100 * gl.d_Ctau, s=8, marker=mk, c=col, lw=0)
                    axs[2, ci].scatter(xx + 0.08, 100 * lo.d_Ctau, s=8, marker=mk, facecolors='none', edgecolors=col, lw=0.5)
        axs[0, ci].set_title(f'Oberfläche {SN[s]}' + (' (eigene Spalte, nicht verrechnet)' if s == 'M24_curv' else ''))
    for ri, yl in enumerate(('abgelehnt: unbestätigbar ∪ grob\n(global; Strich = Schnittmenge)', 'verlorene korrekte Punkte\n(global)', 'Δ C_τ / pp (CLIP − P4)\nvoll: global, hohl: lokal')):
        axs[ri, 0].set_ylabel(yl)
        for ax in axs[ri]:
            ax.axhline(0, c='#888888', lw=0.5); ax.axvline(1.5, c='#bbbbbb', lw=0.6, ls='--')
    for ax in axs[2]:
        ax.set_xticks(range(3)); ax.set_xticklabels([g[2] for g in groups], fontsize=5.6)
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], marker='s', ls='', c='#777777', ms=3, label='S0'), Line2D([], [], marker='o', ls='', c='#777777', ms=3, label='Strategie-Layout'),
          Line2D([], [], marker='o', ls='', c='#1f77b4', ms=3, label='σ = 0'), Line2D([], [], marker='o', ls='', c='#d62728', ms=3, label='σ = 40')]
    fig.legend(handles=hd, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.suptitle('Linien L6,3: Randbandschutz (CLIP) gegen P4 auf demselben Bild – Hauptvalidierung und H0-Anker getrennt', y=1.0)
    fig.text(0.01, -0.07, 'R0/R1: in keinem Raster ist ein Randband beleuchtet – CLIP ist dort per Konstruktion wirkungslos (geprüft: identische Ausgabe). Der Schutzeffekt wird nur am H0-Anker\n'
             '(S0 und Blöcke B024/B025) sichtbar; Anker und Hauptvalidierung werden nicht zusammengefasst. ' + SRC, fontsize=5.0, color='#555555')
    fig.tight_layout()
    return _save(fig, 'F2_Schutz_P4_CLIP.png')


def fig3() -> str:
    plt = _plt(); R = _res(); V = R[(R.record_type == 'view') & (R.set == 'main') & (R.domain == 'glob')].copy()
    V['d_sup_rmse'] = pd.to_numeric(V.d_sup_rmse, errors='coerce')
    VV = pd.read_csv(H.OUT / '_parts' / 'views.csv', low_memory=False, keep_default_na=False)
    rows = [('point', 'D0', 'd_Ctau', 'Punkte R8×4 (D0)\nΔ C_τ global / pp'), ('point', 'D0', 'd_sup_rmse', 'Punkte R8×4 (D0)\nΔ gestützte RMSE / mm'),
            ('line', 'CLIP', 'd_Ctau', 'Linien L6,3 (CLIP; P4 hohl)\nΔ C_τ global / pp'), ('line', 'CLIP', 'd_sup_rmse', 'Linien L6,3 (CLIP; P4 hohl)\nΔ gestützte RMSE / mm')]
    fig, axs = plt.subplots(4, 3, figsize=(6.3, 8.2), sharex=True)
    for ri, (fam, meth, col, yl) in enumerate(rows):
        for ci, s in enumerate(SURFS):
            ax = axs[ri, ci]
            for ai, a in enumerate(ARMS):
                for sg, off in ((0.0, -0.17), (40.0, 0.17)):
                    for m, filled in ((meth, True),) + ((('P4TABLE', False),) if fam == 'line' else ()):
                        x = V[(V.family == fam) & (V.arm == a) & (V.surface == s) & (V.sigma == sg) & (V.method == m)]
                        if not len(x):
                            continue
                        bm = x.groupby('block')[col].mean()
                        sc = 100.0 if col == 'd_Ctau' else 1.0
                        xx = ai + off + (0.07 if not filled else 0.0)
                        ax.scatter([xx] * len(bm), sc * bm.values, s=7, marker=('o' if sg == 0 else '^'), facecolors=(ACOL[a] if filled else 'none'), edgecolors=ACOL[a], lw=0.5)
                        if filled and bm.notna().any():
                            ax.plot([xx - 0.07, xx + 0.07], [sc * bm.median()] * 2, c='k', lw=0.8)
            ax.axhline(0, c='#888888', lw=0.5)
            if ri == 0:
                ax.set_title(f'Oberfläche {SN[s]}')
            if ci == 0:
                ax.set_ylabel(yl)
            if ri == 3:
                ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
    fails = VV[(VV.status != 'selected') & (VV.phase != 'H0')][['baseline', 'block', 'arm']].drop_duplicates()
    ftxt = '; '.join(f"{'Linien' if r.baseline == 'L6.3' else 'Punkte'} {r.arm} {r.block}: kein zulässiger Kandidat in 1000 Vorschlägen (Generierungsfehler, im Nenner)" for r in fails.itertuples()) or 'keine Generierungsfehler'
    fig.suptitle('Strategie-Layouts gegen S0 (gleiche Phase, Oberfläche, σ und Methode) – je Block ein Punkt (Mittel über R0/R1)', y=1.0)
    fig.text(0.01, -0.035, f'Kreis σ = 0, Dreieck σ = 40; schwarzer Strich: Median der Blöcke. Gleiche Layouts mehrerer Strategien sind nur einmal abgebildet/dekodiert; in der Hauptvalidierung sind CLIP und P4 identisch (kein Randband). {ftxt}.\n'
             'Gestützte RMSE nur für Bedingungen mit gestützten Knoten; keine Gesamtrangfolge. ' + SRC, fontsize=5.0, color='#555555')
    fig.tight_layout()
    return _save(fig, 'F3_Strategien_gegen_S0.png')


# Draw held-out layout point clouds with a shared spatial and height scale.
def fig4() -> str:
    plt = _plt(); from matplotlib.colors import LogNorm
    import r2_eval as E
    import rdy_platform as P
    import s4r7_run as R
    R.set_read_only(True)
    ex = H.run_load()['analysis']['examples']; dom = E.domain('omega1')
    LAB = {'D0': 'D0', 'P4TABLE': 'P4 (Referenz)', 'CLIP': 'CLIP (Randbandschutz)'}
    rows = [('Punkte M10: ' + ('größter C_τ-Verlust gegen S0' if ex['dot_m10']['d_Ctau'] < 0 else 'kleinste Änderung gegen S0'), [(ex['dot_m10']['image_key'], 'D0', f"Strategie {ex['dot_m10']['arm']} ({ex['dot_m10']['block']})"), (ex['dot_m10']['s0_image_key'], 'D0', 'S0 (regulär)')]),
            ('Linien: ' + ('größte Abnahme grober Punkte' if ex['line_improvement']['gross_reduction'] > 0 else 'größte Änderung (keine Abnahme grober Punkte)'),
             [(ex['line_improvement']['image_key'], 'P4TABLE', ''), (ex['line_improvement']['image_key'], 'CLIP', '')]),
            ('Linien: ' + ('größter C_τ-Verlust durch CLIP' if ex['line_cost']['d_Ctau'] < 0 else 'kleinste C_τ-Änderung (kein Verlust)'),
             [(ex['line_cost']['image_key'], 'P4TABLE', ''), (ex['line_cost']['image_key'], 'CLIP', '')])]
    data = []
    for title, panels in rows:
        rd = []
        for key, m, tag in panels:
            c = R.cond(key); ev = R.evaluate(dict(c, method=m), m); A = K.load_npz(H.EVL7 / f"{ev['identity']}.npz")
            pts = np.asarray(A['pts'], np.float64); val = np.asarray(A['valid'], bool); err = np.abs(pts[:, 2] - np.asarray(A['zr'], np.float64)); rd.append((c, m, tag, ev, pts, val, err))
        data.append((title, rd))
    allerr = np.concatenate([e[v] for _t, rd in data for (_c, _m, _g, _e, _p, v, e) in rd if v.any()] + [np.array([1e-3])])
    norm = LogNorm(1e-3, max(float(np.nanmax(allerr)), 1e-2))
    fig = plt.figure(figsize=(6.3, 9.6)); sc = None; titles = []
    for i, (title, rd) in enumerate(data):
        c0 = rd[0][0]; sid = K.RK.SURF_MAP[c0['surface']]
        X, Y, Z = np.asarray(dom['x']), np.asarray(dom['y']), np.asarray(dom[f'z_{sid}'], np.float64); sub = np.arange(0, len(X), 7)
        dl = P.dom_local(c0['surface']); cx, cy = float(np.mean(dl['x'])), float(np.mean(dl['y'])); rr = float(np.max(np.hypot(np.asarray(dl['x']) - cx, np.asarray(dl['y']) - cy)))
        allp = np.vstack([np.column_stack([X, Y, Z])] + [p[v] for (_c, _m, _g, _e, p, v, _r) in rd])
        lo, hi = np.nanmin(allp, axis=0), np.nanmax(allp, axis=0)
        for j, (c, m, tag, ev, pts, val, err) in enumerate(rd):
            ax = fig.add_subplot(3, 2, 2 * i + j + 1, projection='3d')
            ax.scatter(X[sub], Y[sub], Z[sub], s=0.05, c='#bbbbbb', alpha=0.35, rasterized=True)
            if val.any():
                sc = ax.scatter(pts[val, 0], pts[val, 1], pts[val, 2], s=(2.5 if c['family'] == 'point' else 0.25), c=np.maximum(err[val], 1e-3), cmap='viridis', norm=norm, rasterized=True)
            else:
                ax.text2D(0.5, 0.5, 'keine Ausgabe', transform=ax.transAxes, ha='center', fontsize=7, color='#d62728')
            th = np.linspace(0, 2 * np.pi, 120); ax.plot(cx + rr * np.cos(th), cy + rr * np.sin(th), np.full_like(th, float(lo[2])), c='#d62728', lw=0.6)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2] - 1, hi[2] + 1); ax.view_init(elev=28, azim=-60)
            ax.set_xlabel('x / mm', fontsize=5.5, labelpad=-6); ax.set_ylabel('y / mm', fontsize=5.5, labelpad=-6); ax.set_zlabel('z / mm', fontsize=5.5, labelpad=-6); ax.tick_params(labelsize=4.5, pad=-2)
            g = ev['metrics']; head = (f"{tag}: {c['layout']}" if tag.startswith('Strategie') else (tag or LAB[m]))
            ax.set_title(f"{head}\nC_τ {_de(100 * float(g['glob_Ctau'] or 0), 1)} %, C_valid {_de(100 * float(g['glob_Cvalid'] or 0), 1)} %\n"
                         f"grob {g['glob_n_gross']}, gestützte RMSE {_de(ev['nodes']['glob']['sup_surf_rmse'], 3)} mm", fontsize=5.4)
            if j == 0:
                fam = 'Punkte R8×4' if c['family'] == 'point' else 'Linien L6,3'
                titles.append((ax, f"{title} – {fam}, {SN[c['surface']]}, {c['phase']} ({'Anker' if c['set'] == 'anchor' else 'Hauptvalidierung'}), σ = {int(float(c['sigma']))}"))
    fig.subplots_adjust(top=0.93, bottom=0.12, hspace=0.55, wspace=0.25)
    for ax, t in titles:
        fig.text(0.02, ax.get_position().y1 + 0.045, t, fontsize=6.2, weight='bold')
    if sc is not None:
        cax = fig.add_axes([0.25, 0.075, 0.5, 0.008]); cb = fig.colorbar(sc, cax=cax, orientation='horizontal'); cb.ax.tick_params(labelsize=5); cb.set_label('|z_rekonstruiert − z_wahr| / mm (eine Skala für alle Felder; < 0,001 mm als 0,001 gezeigt)', fontsize=6)
    fig.text(0.01, 0.005, 'Grau: wahre Oberfläche (Ω1-Knoten, ausgedünnt); farbig: gültige rekonstruierte Punkte; rot: lokaler Kreis. Gleiche Ansicht und Achsen je Zeile; Achsen umfassen alle Punkte.\n'
             'Auswahlregeln vor der Bildgebung festgelegt (Gleichstand: Bildschlüssel); Diagnosefälle, keine Häufigkeitsaussage. ' + SRC, fontsize=5.0, color='#555555')
    return _save(fig, 'F4_Rekonstruktion_3D.png')


def drill_figure(P: pd.DataFrame, path) -> str:
    plt = _plt(); g = P[P.domain == 'glob'].sort_values('image_key'); fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.bar(np.arange(len(g)) - 0.2, g.union_rejected, width=0.4, label='abgelehnt (∪)'); ax.bar(np.arange(len(g)) + 0.2, g.correct_lost, width=0.4, label='korrekt verloren')
    ax.set_xticks(np.arange(len(g))); ax.set_xticklabels([f'{p} {SN[s]} {int(sg)}' for p, s, sg in zip(g.phase, g.surface, g.sigma)], rotation=90, fontsize=4.5); ax.legend(fontsize=5)
    ax.set_title('Probelauf (isoliert, S4-REAL-06-Material) – nur Technikprüfung', fontsize=6)
    p = H.guard(path); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return str(p)


def make_all() -> dict:
    files = [fig1(), fig2(), fig3(), fig4()]
    out = dict(files=files, hashes={f: K.sha_file(K.ROOT / f) for f in files}, display_code=H.code_hashes(H.DISPLAY_DEPS), t_local=K.now_local())
    H.run_put('figures', out)
    return out


def register_display() -> dict:

    r = H.run_load(); hist = r.get('display_versions', [])
    v = dict(n=len(hist) + 1, display_code=H.code_hashes(H.DISPLAY_DEPS), figures={f: K.sha_file(K.ROOT / f) for f in r.get('figures', {}).get('files', [])}, t_local=K.now_local())
    r['display_versions'] = hist + [v]; r['updated'] = K.now_local(); H.save_json(H.RUN, r)
    return v
