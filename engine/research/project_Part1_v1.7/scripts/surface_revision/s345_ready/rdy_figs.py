# Plot shared pattern-study metrics and reconstruction examples.

from __future__ import annotations

import numpy as np

import rdy_common as K

NOTE = 'Integrationstestfall – kein formales Forschungsergebnis (S345-PREP-03)'
DH_LIM = 0.5
GREY_NI, GREY_NS = '#e6e6e6', '#b3b3b3'


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams.update({'font.size': 7.5, 'font.family': 'DejaVu Sans', 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5, 'figure.dpi': 150})
    return plt


def load_case(ctx, dk: str) -> dict:
    p = ctx.eval / f'{dk}.json'
    r = K.load_json(p)
    with np.load(p.with_suffix('.npz'), allow_pickle=False) as z:
        a = {k: np.asarray(z[k]) for k in ('pts', 'valid', 'zr') if k in z.files}
    return dict(rec=r, **a)


def node_fields(case: dict) -> dict:

    import rdy_platform as P
    from surface_eval_r3 import surface_eval_r3_dev
    r = case['rec']; sid = r['platform_surface']; dom = P.dom_global()
    v = case['valid'].astype(bool); pts = np.asarray(case['pts'], np.float64)[v]
    z_ref = np.asarray(dom[f'z_{sid}'], np.float64); n = len(z_ref)
    if len(pts) < 3:
        f = dict(z_est=np.full(n, np.nan), err=np.full(n, np.nan), interp=np.zeros(n, bool), sup=np.zeros(n, bool), acc=np.zeros(n, bool), Cvalid=0.0, Ctau=0.0)
    else:
        D = dict(x=dom['x'], y=dom['y'], z_ref=z_ref, area=np.full(n, dom['step'] ** 2), hash=dom['hash'])
        ev = surface_eval_r3_dev(pts, None, D, [K.D_SUPPORT], [K.TAU], return_fields=True, support_rule='exact')
        fl = ev['fields']; err = np.asarray(fl['err'], np.float64); Rm = np.asarray(fl['support_measure'], np.float64)
        interp = (np.asarray(fl['simplex']) >= 0) & np.isfinite(err); sup = interp & (Rm <= K.D_SUPPORT); acc = sup & (np.abs(err) <= K.TAU)
        bd = ev['subdomains']['global']['by_d'][str(float(K.D_SUPPORT))]
        f = dict(z_est=np.asarray(fl['z_est'], np.float64), err=err, interp=interp, sup=sup, acc=acc, Cvalid=float(bd['C_valid'] or 0.0), Ctau=float(bd['C_tau'][str(float(K.TAU))] or 0.0))
    m = r['metrics']
    if abs(f['Cvalid'] - (m['glob_Cvalid'] or 0.0)) > 1e-12 or abs(f['Ctau'] - (m['glob_Ctau'] or 0.0)) > 1e-12:
        raise K.Stop('check', f"figure node fields differ from the record {r['decode_key']}: Cvalid {f['Cvalid']} / {m['glob_Cvalid']}, Ctau {f['Ctau']} / {m['glob_Ctau']}")
    f.update(x=dom['x'], y=dom['y'], z_ref=z_ref, check=dict(Cvalid_equal=True, Ctau_equal=True))
    return f


def class_map(ax, f: dict, title: str):
    import matplotlib.colors as mcolors
    x, y = f['x'], f['y']; dH = -(f['err'])
    ax.scatter(x[~f['interp']], y[~f['interp']], s=0.6, c=GREY_NI, marker='s', linewidths=0, rasterized=True)
    ns = f['interp'] & ~f['sup']
    ax.scatter(x[ns], y[ns], s=0.6, c=GREY_NS, marker='s', linewidths=0, rasterized=True)
    sc = ax.scatter(x[f['sup']], y[f['sup']], s=0.6, c=np.clip(dH[f['sup']], -DH_LIM, DH_LIM), cmap='RdBu_r', norm=mcolors.Normalize(-DH_LIM, DH_LIM), marker='s', linewidths=0, rasterized=True)
    t = np.linspace(0, 2 * np.pi, 200); ax.plot(15 * np.cos(t), 15 * np.sin(t), color='k', lw=0.6, ls='--')
    ax.set_aspect('equal'); ax.set_xlim(-47, 47); ax.set_ylim(-47, 47); ax.set_xticks([-40, -20, 0, 20, 40]); ax.set_yticks([-40, -20, 0, 20, 40])
    ax.set_title(title, fontsize=6.8)
    return sc


def _footer(fig, extra=''):
    fig.text(0.01, 0.005, NOTE + (' · ' + extra if extra else ''), fontsize=6, color='#555555', ha='left', va='bottom')


def _legend_classes(fig, x=0.70):
    import matplotlib.patches as mpatches
    h = [mpatches.Patch(color=GREY_NI, label='nicht interpolierbar'), mpatches.Patch(color=GREY_NS, label='interpolierbar, Stützweite > d = 6 mm'),
         mpatches.Patch(color='#ffffff', ec='k', lw=0.4, label='gestrichelt: lokaler Bereich r ≤ 15 mm (vorgegeben, keine Verformungsgrenze)')]
    fig.legend(handles=h, loc='lower right', bbox_to_anchor=(0.995, 0.02), ncol=1, frameon=False, fontsize=6)


def _cbar(fig, sc, rect=(0.955, 0.30, 0.008, 0.45)):
    cax = fig.add_axes(rect); cb = fig.colorbar(sc, cax=cax)
    cb.set_label('ΔH = H_rec − H_ref [mm] (rot: Höhe überschätzt)', fontsize=6); cb.ax.tick_params(labelsize=6)


def _mt(m, pre='glob'):
    def f(v):
        return '–' if v is None else f'{v:.3f}'
    return f"C_τ {f(m.get(pre + '_Ctau'))} · C_valid {f(m.get(pre + '_Cvalid'))}"


def fig_s3(ctx, f: 'pd.DataFrame', out_png) -> dict:
    plt = _plt(); checks = []
    d = f[(f.section == 'S3')]
    pick = d[(d.family == 'point') & (d.surface == 'M10_curv') & (d.regime == 'coded_high') & (d.state.isin(['ok', 'no_output']))]
    fig = plt.figure(figsize=(7.4, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.5, wspace=0.3, left=0.07, right=0.93, top=0.89, bottom=0.15)
    sc = None
    if len(pick):
        ik = pick.image_key.iloc[0]
        with np.load(ctx.images / f'{ik}.npz') as z:
            img = np.asarray(z['image'])
        ax = fig.add_subplot(gs[0, 0]); ax.imshow(img, cmap='gray', vmin=0, vmax=255); ax.set_title('Kamerabild M10, coded2 R5×4, σ 30\n(ein Bild für beide Arme)', fontsize=6.5); ax.axis('off')
        for j, arm in enumerate(('ENC_coded', 'ENC_ignore')):
            row = pick[pick.arm == arm]
            if row.empty:
                continue
            c = load_case(ctx, row.decode_key.iloc[0]); nf = node_fields(c); checks.append(dict(decode_key=row.decode_key.iloc[0], **nf['check']))
            lab = 'Code genutzt' if arm == 'ENC_coded' else 'Code ignoriert'
            sc = class_map(fig.add_subplot(gs[0, 1 + j]), nf, f"{arm} ({lab})\ngesamt: {_mt(c['rec']['metrics'])}\nlokal r ≤ 15: {_mt(c['rec']['metrics'], 'loc')}")
    ax = fig.add_subplot(gs[1, 0:2])
    pts = d[(d.family == 'point') & d.state.isin(['ok', 'no_output'])]
    mk = {'ENC_coded': 'o', 'ENC_ignore': 's'}
    for (surf, code, arm), g in pts.groupby(['surface', pts.code.str.split().str[0], 'arm']):
        g = g.sort_values('sigma'); off = {'M10_curv': -0.6, 'M24_curv': 0.6}[surf] if surf in ('M10_curv', 'M24_curv') else 0
        ax.plot(g.sigma.astype(float) + off, g.glob_Ctau, marker=mk.get(arm, 'o'), ms=3.5, lw=0.8, ls='-' if code == 'coded2' else ':',
                label=f"{K.SHORT[surf]} {code} {'Code' if arm == 'ENC_coded' else 'ohne Code'}")
    ax.set_xlabel('Rauschen σ [Grauwert]'); ax.set_ylabel('C_τ gesamt (d 6 mm, τ 0,15 mm)'); ax.set_title('Punkte R5×4, Phase val0, Realisierung 0: je Bild ein Wert (keine Phasenmittel)', fontsize=6.5)
    ax.legend(ncol=2, frameon=False, fontsize=5.6); ax.grid(alpha=0.3); ax.set_xticks([0, 25, 30, 35])
    ax = fig.add_subplot(gs[1, 2])
    ln = d[(d.family == 'line') & d.state.isin(['ok', 'no_output'])]
    labels = []; xs = 0
    for (surf, view, sig), g in ln.groupby(['surface', 'view', 'sigma']):
        for arm in ('LC_ignore', 'LC_use', 'LC_R', 'LC_A', 'P4_ref'):
            h = g[g.arm == arm]
            if len(h):
                ax.scatter([xs], [h.glob_Ctau.iloc[0]], s=10, marker={'LC_ignore': 'o', 'LC_use': '^', 'LC_R': 's', 'LC_A': 'D', 'P4_ref': '*'}[arm], color={'M10_curv': '#1f77b4', 'M24_curv': '#d62728'}.get(surf, 'k'))
        labels.append(f"{K.SHORT[surf]} {view[-2:]} σ{int(sig)}"); xs += 1
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=80, fontsize=5.0); ax.set_ylabel('C_τ gesamt'); ax.grid(alpha=0.3)
    ax.set_title('Linien s 4,45 mm (blau M10, rot M24)\no ignore ^ use ■ R ◆ A ★ P4_ref 45°', fontsize=6)
    if sc is not None:
        _cbar(fig, sc, (0.945, 0.56, 0.008, 0.30))
    fig.suptitle('Abschnitt 3 – Codierung: gleiches Bild, Arme unterscheiden sich nur in der Decodierung', fontsize=8)
    _footer(fig, 'V1 fester Peak, V2/V3 gleiches Integral, V5 45° uncodiert = Schemavergleich')
    fig.savefig(out_png, dpi=200); plt.close(fig)
    return dict(path=K.rel(out_png), value_check=checks)


def fig_s4(ctx, f, out_png) -> dict:
    plt = _plt(); checks = []
    d = f[(f.section == 'S4') & (f.surface == 'M10_curv') & (f.phase_label == 'H0') & f.state.isin(['ok', 'no_output'])].copy()
    d['baseline'] = d.layout_id.str.split(':').str[0]
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], hspace=0.5, wspace=0.62, left=0.06, right=0.92, top=0.9, bottom=0.1)
    sc = None
    for i, b in enumerate(('R5x4', 'L4.45')):
        g = d[(d.baseline == b) & (d.sigma.astype(float) == 0)]
        for j, s0 in enumerate((True, False)):
            h = g[g.strategy.str.startswith('S0') == s0]
            if h.empty:
                continue
            dk = h.decode_key.iloc[0]; c = load_case(ctx, dk); nf = node_fields(c); checks.append(dict(decode_key=dk, **nf['check']))
            dsp = (c['rec']['extras'].get('displacement') or {})
            nm = 'S0 (regulär)' if s0 else f"verschoben j {dsp.get('j', 0):.2f}, max |Δ| {dsp.get('max_abs_disp_mm', 0):.2f} mm"
            sc = class_map(fig.add_subplot(gs[i, j]), nf, f"{b} {nm}, σ 0\ngesamt: {_mt(c['rec']['metrics'])}")
        ax = fig.add_subplot(gs[i, 2]); bars = []
        for sig in (0.0, 20.0):
            gg = d[(d.baseline == b) & (d.sigma.astype(float) == sig)]
            s0 = gg[gg.strategy.str.startswith('S0')]
            for strat, h in gg[~gg.strategy.str.startswith('S0')].groupby('strategy'):
                if len(s0) and len(h):
                    bars.append((f"σ{int(sig)} {strat.split('_')[0]}", float(h.glob_Ctau.iloc[0] - s0.glob_Ctau.iloc[0])))
        if bars:
            ax.barh(range(len(bars)), [v for _, v in bars], color=['#4c72b0' if v >= 0 else '#c44e52' for _, v in bars], height=0.6)
            ax.set_yticks(range(len(bars))); ax.set_yticklabels([n for n, _ in bars], fontsize=5.6); ax.axvline(0, color='k', lw=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.tick_params(axis='x', labelsize=5.5)
        ax.set_xlabel('Δ C_τ gesamt (Strategie − S0)'); ax.set_title(f'{b}: H0, 1 von 4 Phasen\n(kein Blockmedian; S1–S4 teilen\nein Layout = eine Decodierung)', fontsize=5.8)
        ax.grid(alpha=0.3, axis='x')
    if sc is not None:
        _cbar(fig, sc, (0.945, 0.30, 0.008, 0.45))
    fig.suptitle('Abschnitt 4 – Positionstabelle: S0 und registriertes verschobenes Layout (M10, H0)', fontsize=8)
    _footer(fig, 'Generierung / Akzeptanz aus SPAT-03 (projektorseitig), hier nicht neu erzeugt')
    fig.savefig(out_png, dpi=200); plt.close(fig)
    return dict(path=K.rel(out_png), value_check=checks)


def fig_s5(ctx, f, out_png) -> dict:
    import rdy_stats as ST
    plt = _plt()
    d = f[(f.section == 'S5') & f.state.isin(['ok', 'no_output'])]
    sets = [('dots_parallel', 'M10_curv', 0.0, 'Punkte, p∥ variiert (p⊥ 4 mm)'), ('lines', 'M10_curv', 0.0, 'Linien (w, s)'), ('dots_perp', 'M24_curv', 40.0, 'Punkte, p⊥ variiert (p∥ 8 mm)')]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0)); fig.subplots_adjust(left=0.07, right=0.88, top=0.76, bottom=0.2, wspace=0.45)
    vals = []
    for ax, (s, surf, sig, title) in zip(axs, sets):
        cells = ST.FOUR[s]; M = np.full((2, 2), np.nan)
        for (ri, size), (ci, dens) in [((0, 's'), (0, 'd')), ((0, 's'), (1, 's')), ((1, 'l'), (0, 'd')), ((1, 'l'), (1, 's'))]:
            g = d[(d.surface == surf) & (d.sigma.astype(float) == sig) & (d.pair_role == cells[size + dens])]
            if len(g):
                M[ri, ci] = float(g.glob_Ctau.mean())
        vals.append(M)
    vmax = np.nanmax([np.nanmax(m) for m in vals if np.isfinite(m).any()] + [0.01])
    for ax, (s, surf, sig, title), M in zip(axs, sets, vals):
        cells = ST.FOUR[s]
        im = ax.imshow(M, cmap='viridis', vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, '–' if not np.isfinite(M[i, j]) else f'{M[i, j]:.3f}', ha='center', va='center', color='w' if (np.isfinite(M[i, j]) and M[i, j] < 0.6 * vmax) else 'k', fontsize=7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['dicht', 'weit']); ax.set_yticks([0, 1]); ax.set_yticklabels(['klein', 'groß'])
        I = (M[1, 1] - M[1, 0]) - (M[0, 1] - M[0, 0]) if np.isfinite(M).all() else np.nan
        ax.set_title(f"{title}\n{K.SHORT[surf]}, σ {int(sig)}, Phase ph0\n{cells['sd']} … {cells['ls']}\nI (1 Phase) = {('–' if not np.isfinite(I) else f'{I:+.3f}')}", fontsize=5.8)
    cax = fig.add_axes([0.9, 0.2, 0.012, 0.56]); cb = fig.colorbar(im, cax=cax); cb.set_label('C_τ gesamt', fontsize=6); cb.ax.tick_params(labelsize=6)
    fig.suptitle('Abschnitt 5 – Vier Zellen Größe × Abstand (eine Phase von 8: Kontrast formal unvollständig)', fontsize=8)
    _footer(fig)
    fig.savefig(out_png, dpi=200); plt.close(fig)
    return dict(path=K.rel(out_png), value_check=[])


def fig_formal_s3(stats: dict, out_png) -> dict:

    import pandas as pd
    plt = _plt(); s = stats['S3']; dev = pd.DataFrame(s['development'])
    blocks = pd.DataFrame(s['same_image']); n_ok = int((blocks.state == 'ok').sum()) if len(blocks) else 0
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0)); fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.25, wspace=0.4)
    for ax, (con, title) in zip(axs[:2], (('ENC_coded - ENC_ignore', 'Punkte coded2: Code − ohne Code'), (None, 'Linien: use / R / A − ignore'))):
        x = dev[dev.contrast == con] if con else dev[dev.contrast.str.startswith('LC_')] if len(dev) else dev
        if len(x):
            keys = sorted(set(zip(x.surface, x.sigma)))
            for i, k in enumerate(keys):
                v = x[(x.surface == k[0]) & (x.sigma == k[1])]['d_glob_Ctau'].astype(float)
                ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=3, alpha=0.6)
                ax.scatter([i], [v.mean()], marker='_', s=120, color='k'); ax.scatter([i], [v.median()], marker='x', s=20, color='r')
            ax.set_xticks(range(len(keys))); ax.set_xticklabels([f'{K.SHORT.get(a, a)} σ{int(b)}' for a, b in keys], rotation=60, fontsize=5.5)
        ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('Δ C_τ gesamt (gleiches Bild)'); ax.set_title(title + '\n— Mittel, x Median', fontsize=6.5); ax.grid(alpha=0.3)
    ev = pd.DataFrame(s['events'])
    if len(ev):
        e = ev[ev.event.str.contains('fallback')]
        lab = [f"{K.SHORT.get(a, a)} σ{int(b)} {c.split()[0]}" for a, b, c in zip(e.surface, e.sigma, e.event)]
        axs[2].barh(range(len(e)), e.n_events, color='#c44e52'); axs[2].barh(range(len(e)), e.n_images - e.n_events, left=e.n_events, color='#dddddd')
        axs[2].set_yticks(range(len(e))); axs[2].set_yticklabels(lab, fontsize=5); axs[2].set_xlabel('Bilder (rot: Rückfall)')
    axs[2].set_title('Schutz-Rückfall (Anzahl, kein Intervall)', fontsize=6.5)
    fig.suptitle(f"Abschnitt 3 – Codierung ({stats['set']}): {n_ok} von {len(blocks)} Blöcken vollständig", fontsize=8)
    _footer(fig, 'formale Auswertung' if n_ok == len(blocks) and len(blocks) else 'UNVOLLSTÄNDIG – keine Schlussfolgerung')
    fig.savefig(out_png, dpi=200); plt.close(fig)
    return dict(path=K.rel(out_png), blocks_ok=n_ok, blocks=int(len(blocks)))


def fig_formal_s4(stats: dict, out_png) -> dict:
    import pandas as pd
    plt = _plt(); bl = pd.DataFrame(stats['S4']['blocks'])
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0)); fig.subplots_adjust(left=0.18, right=0.98, top=0.8, bottom=0.18, wspace=0.6)
    for ax, sig in zip(axs, (0.0, 20.0)):
        x = bl[(bl.sigma == sig) & (bl.state == 'ok')] if len(bl) else bl
        keys = sorted(set(zip(x.baseline, x.strategy))) if len(x) else []
        for i, k in enumerate(keys):
            v = x[(x.baseline == k[0]) & (x.strategy == k[1])]['median4_glob_Ctau'].astype(float)
            ax.scatter(v, np.full(len(v), i), s=3, alpha=0.6); ax.scatter([v.median()], [i], marker='|', s=150, color='k')
        ax.set_yticks(range(len(keys))); ax.set_yticklabels([f'{a} {b.split("@")[0]}' for a, b in keys], fontsize=5.5); ax.axvline(0, color='k', lw=0.5)
        sm = pd.DataFrame(stats['S4']['summary']); sm = sm[(sm.surface == 'all') & (sm.sigma == sig)] if len(sm) else sm
        for i, k in enumerate(keys):
            r = sm[(sm.baseline == k[0]) & (sm.strategy == k[1])]
            if len(r) and r.iloc[0].get('state') == 'ok':
                ax.text(ax.get_xlim()[1], i, f" +{int(r.iloc[0]['n_blocks_improved'])}/−{int(r.iloc[0]['n_blocks_worsened'])}", fontsize=5, va='center')
        ax.set_xlabel('Blockmedian Δ C_τ gesamt (Layout − S0, 4 Phasen)'); ax.set_title(f'σ {int(sig)}: | = Median über vollständige Blöcke; +/− verbessert / verschlechtert', fontsize=6); ax.grid(alpha=0.3)
    n_ok = int((bl.state == 'ok').sum()) if len(bl) else 0
    fig.suptitle(f"Abschnitt 4 – Strategien ({stats['set']}): {n_ok} von {len(bl)} Blöcken vollständig", fontsize=8)
    _footer(fig, 'formale Auswertung' if n_ok == len(bl) and len(bl) else 'UNVOLLSTÄNDIG – keine Schlussfolgerung')
    fig.savefig(out_png, dpi=200); plt.close(fig)
    return dict(path=K.rel(out_png), blocks_ok=n_ok, blocks=int(len(bl)))


def fig_formal_s5(stats: dict, out_png) -> dict:
    import pandas as pd
    plt = _plt(); c = pd.DataFrame(stats['S5']['contrasts']); cb = pd.DataFrame(stats['S5']['combos'])
    surfs = sorted(cb.surface.unique()) if len(cb) else []
    fig, axs = plt.subplots(1, max(1, len(surfs)), figsize=(7.4, 3.2), squeeze=False); fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.3, wspace=0.5)
    for ax, s in zip(axs[0], surfs):
        x = cb[cb.surface == s].pivot_table(index='geometry_id', columns='sigma', values='mean_over_phases', aggfunc='first')
        ax.imshow(x.to_numpy(float), cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_yticks(range(len(x.index))); ax.set_yticklabels(x.index, fontsize=5); ax.set_xticks(range(len(x.columns))); ax.set_xticklabels([f'σ{int(v)}' for v in x.columns], fontsize=6)
        ax.set_title(K.SHORT.get(s, s), fontsize=6.5)
    n_ok = int((c.state == 'ok').sum()) if len(c) else 0
    fig.suptitle(f"Abschnitt 5 – Kombinationen (Mittel über Phasen, C_τ gesamt) ({stats['set']}): {n_ok} von {len(c)} Kontrasten vollständig", fontsize=7.5)
    _footer(fig, 'formale Auswertung' if n_ok == len(c) and len(c) else 'UNVOLLSTÄNDIG – keine Schlussfolgerung')
    fig.savefig(out_png, dpi=200); plt.close(fig)
    return dict(path=K.rel(out_png), contrasts_ok=n_ok, contrasts=int(len(c)))


EXAMPLE_NOTE = 'Beispiel einer registrierten Phase (Realisierung 0) – keine Phasenmittel; Gesamtstatistik siehe Statistik-Abbildung'


def fig_example_s3(ctx, f, out_png, surface='M10_curv', phase='val0', line_phase='x32_-16') -> dict:

    plt = _plt(); C = ('ok', 'no_output')
    d = f[(f.section == 'S3') & (f.surface == surface) & f.state.isin(C)]
    pt = d[(d.family == 'point') & (d.phase_label == phase)]; ln = d[(d.family == 'line') & (d.phase_label == line_phase)]
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6)); fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.2, hspace=0.6, wspace=0.28)
    for j, (m, lab) in enumerate((('glob_Ctau', 'C_τ gesamt'), ('id_wrong', 'falsche Identitäten (Anzahl)'))):
        ax = axs[0, j]
        for (lay, code, arm, reg), g in pt.groupby(['layout_id', pt.code.str.split().str[0], 'arm', pt.regime.str.replace('_ref', '', regex=False)]):
            g = g.sort_values('sigma')
            ax.plot(g.sigma.astype(float), g[m], marker='o' if arm == 'ENC_coded' else 's', ms=3, lw=0.8, ls='-' if reg in ('main_common_budget', 'coded_high') else '--',
                    label=f"{lay} {code} {reg.replace('main_common_budget', 'Budget').replace('coded_', '')} {'Code' if arm == 'ENC_coded' else 'ohne'}")
        ax.set_xlabel('σ'); ax.set_ylabel(lab); ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=4.6, ncol=2, frameon=False)
        ax.set_title(f'Punkte, {K.SHORT[surface]}, Phase {phase}', fontsize=6.5)
        ax = axs[1, j]; labels = []; xs = 0
        for (view, reg, sig), g in ln.groupby(['view', 'regime', 'sigma']):
            for k, arm in enumerate(('LC_ignore', 'LC_use', 'LC_R', 'LC_A', 'P4_ref')):
                h = g[g.arm == arm]
                if len(h):
                    ax.scatter([xs + 0.12 * (k - 2)], [h[m].iloc[0]], s=9, marker={'LC_ignore': 'o', 'LC_use': '^', 'LC_R': 's', 'LC_A': 'D', 'P4_ref': '*'}[arm], label=arm if xs == 0 else None)
            labels.append(f"{view[-2:]} {'fest' if reg == 'fixed_peak' else 'Integral'} σ{int(sig)}"); xs += 1
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=70, fontsize=5); ax.set_ylabel(lab); ax.grid(alpha=0.3)
        ax.set_title(f'Linien, {K.SHORT[surface]}, Phase {line_phase}', fontsize=6.5)
        if j == 0 and labels:
            ax.legend(fontsize=5, ncol=5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.55))
    fig.suptitle('Abschnitt 3 – Beispiel aller Bedingungen einer Phase (gleiches Bild: Code genutzt / ignoriert; Helligkeit; Rauschen)', fontsize=7.5)
    _footer(fig, EXAMPLE_NOTE)
    fig.savefig(out_png, dpi=200); plt.close(fig)
    return dict(path=K.rel(out_png), n_point_rows=int(len(pt)), n_line_rows=int(len(ln)))


def fig_example_s4(ctx, f, out_png, surface='M10_curv', phase='H0', block='B000', baseline='R5x4', sigma=0.0) -> dict:

    import rdy_platform as P
    plt = _plt(); checks = []; S = P.s2()
    d = f[(f.section == 'S4') & (f.surface == surface) & (f.phase_label == phase) & (f.sigma.astype(float) == float(sigma)) & f.layout_id.astype(str).str.startswith(baseline + ':')]
    order = ['S0', 'S1', 'S2', 'S3', 'S4']
    fig = plt.figure(figsize=(9.0, 5.8)); gs = fig.add_gridspec(3, 5, hspace=0.45, wspace=0.3, left=0.04, right=0.93, top=0.86, bottom=0.06)
    sc = None; seen = {}
    for j, s in enumerate(order):
        h = d[d.strategy.str.startswith('S0')] if s == 'S0' else d[d.strategy.str.startswith(s + '_') & (d.s4_block == block)]
        ok = h[h.state.isin(['ok', 'no_output'])]
        if ok.empty:
            for i in range(3):
                ax = fig.add_subplot(gs[i, j]); ax.axis('off'); ax.text(0.5, 0.5, f'{s}: kein vollständiger Datensatz', ha='center', fontsize=6)
            continue
        r = ok.iloc[0]; c = load_case(ctx, r.decode_key); nf = node_fields(c); checks.append(dict(decode_key=r.decode_key, **nf['check']))
        same = seen.get(r.image_key); seen.setdefault(r.image_key, s)
        with np.load(ctx.images / f'{r.image_key}.npz') as z:
            img = np.asarray(z['image'])
        ax = fig.add_subplot(gs[0, j]); ax.imshow(img, cmap='gray', vmin=0, vmax=255); ax.axis('off')
        ax.set_title(f"{s}{' (gleiches Layout wie ' + same + ')' if same else ''}\n{str(r.layout_id).split(':')[-1][-14:]}", fontsize=5.5)
        ax = fig.add_subplot(gs[1, j], projection='3d'); v = c['valid'].astype(bool); p = c['pts'][v]
        ax.scatter(p[::2, 0], p[::2, 1], S.C_REF - p[::2, 2], s=0.3, c='#d62728', depthshade=False)
        ax.set_xlim(-46, 46); ax.set_ylim(-46, 46); ax.tick_params(labelsize=4, pad=-3); ax.set_title(f'Punkte (n = {int(v.sum())}), H = c − z', fontsize=5.5)
        sc = class_map(fig.add_subplot(gs[2, j]), nf, f"gesamt {_mt(c['rec']['metrics'])}")
    if sc is not None:
        _cbar(fig, sc, (0.945, 0.08, 0.008, 0.25))
    fig.suptitle(f'Abschnitt 4 – Beispiel {K.SHORT[surface]}, {phase}, Block {block}, {baseline}, σ {int(sigma)}: fünf Strategien (S0 = gemeinsame Referenz)', fontsize=7.5)
    _footer(fig, 'Beispiel, keine Statistik; Strategien mit gleichem Layout teilen eine Aufnahme / Decodierung')
    fig.savefig(out_png, dpi=200); plt.close(fig)
    return dict(path=K.rel(out_png), value_check=checks)


def fig_surfaces(ctx, f, out_png) -> dict:
    import rdy_platform as P
    plt = _plt(); checks = []
    d = f[f.joint_block.astype(str).str.startswith('surface_') & f.state.isin(['ok', 'no_output'])]
    fig = plt.figure(figsize=(7.4, 5.8))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42, left=0.03, right=0.92, top=0.84, bottom=0.08)
    sc = None; S = P.s2()
    for i, surf in enumerate(('B_curv', 'M10_curv_gauss')):
        g = d[d.surface == surf]
        if g.empty:
            continue
        dk = g.decode_key.iloc[0]; c = load_case(ctx, dk); nf = node_fields(c); checks.append(dict(decode_key=dk, **nf['check'])); m = c['rec']['metrics']
        ax = fig.add_subplot(gs[i, 0], projection='3d'); mesh = P.mesh(surf)
        X, Y, Z = mesh['X'], mesh['Y'], mesh['Z']; sel = (np.abs(X[0]) <= 46); ys = (np.abs(Y[:, 0]) <= 46)
        Xs, Ys, Zs = X[np.ix_(ys, sel)][::24, ::24], Y[np.ix_(ys, sel)][::24, ::24], (S.C_REF - Z[np.ix_(ys, sel)])[::24, ::24]
        ax.plot_wireframe(Xs, Ys, Zs, color='#999999', lw=0.3)
        v = c['valid'].astype(bool); p = c['pts'][v]
        ax.scatter(p[::3, 0], p[::3, 1], S.C_REF - p[::3, 2], s=0.4, c='#d62728', depthshade=False)
        ax.set_title(f"{K.SHORT[surf]}: Referenz (Gitter), Punkte (rot)\nH = c − z [mm], c = {S.C_REF:.3f} mm", fontsize=6)
        ax.set_xlabel('x [mm]', fontsize=5.5, labelpad=-4); ax.set_ylabel('y [mm]', fontsize=5.5, labelpad=-4); ax.tick_params(labelsize=4.5, pad=-2)
        amp = (f"Amplitude: Fehler Spitze {m.get('amp_err_peak_mm'):+.3f} mm (Soll {m.get('amp_true_mm')} mm)" if m.get('amp_applicable') and m.get('amp_err_peak_mm') is not None
               else ('Amplitude: nicht anwendbar (kein Gauß-Term)' if not m.get('amp_applicable') else 'Amplitude: nicht bestimmbar'))
        sc = class_map(fig.add_subplot(gs[i, 1]), nf, f"gesamt: {_mt(m)}\nlokal r ≤ 15: {_mt(m, 'loc')}\n{amp}")
        ax = fig.add_subplot(gs[i, 2])
        row = np.isclose(nf['y'], 0.0); o = np.argsort(nf['x'][row])
        ax.plot(nf['x'][row][o], (S.C_REF - nf['z_ref'][row])[o], color='k', lw=0.9, label='Referenz H_ref (y = 0)')
        ax.plot(nf['x'][row][o], (S.C_REF - nf['z_est'][row])[o], color='#d62728', lw=0.8, label='interpoliert H_rec (Knoten)')
        band = v & (np.abs(c['pts'][:, 1]) <= 0.5)
        ax.scatter(c['pts'][band, 0], S.C_REF - c['pts'][band, 2], s=2.0, color='#1f77b4', label='Punkte |y| ≤ 0,5 mm', zorder=4)
        ax.axvspan(-15, 15, color='#cccccc', alpha=0.25, lw=0); ax.set_xlabel('x [mm]'); ax.set_ylabel('H [mm]'); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=5.2)
        ax.set_title('Profil y = 0 (grau: lokaler Bereich, vorgegeben)', fontsize=6)
    if sc is not None:
        _cbar(fig, sc, (0.945, 0.30, 0.008, 0.45))
    fig.suptitle('Weitere Flächen – Linien w 1,5 mm, s 8 mm, σ 0, Phase ph0:\nFlächenzuordnung, Auswertebereiche, Amplitude', fontsize=8, y=0.985)
    _footer(fig)
    fig.savefig(out_png, dpi=200); plt.close(fig)
    return dict(path=K.rel(out_png), value_check=checks)
