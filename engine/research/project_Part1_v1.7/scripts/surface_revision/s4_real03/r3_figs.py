# Plot boundary and detection diagnostics from stored layout arrays.

from __future__ import annotations

import numpy as np
import pandas as pd

import r3_diag as D

K = D.K
W_IN = 6.3
SRC = 'Quelle: S4-REAL-03 (Diagnose ohne neue Bildgebung / Decodierung) auf S4-REAL-02-Daten, r3_figs.py'


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7.0, 'axes.titlesize': 7.0, 'axes.labelsize': 6.8, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'legend.fontsize': 6.0, 'font.family': 'DejaVu Sans'})
    return plt


def _de(v, nd=1):
    return f'{v:.{nd}f}'.replace('.', ',')


def _fmt(ax, x=True, y=True):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))
    if x:
        ax.xaxis.set_major_formatter(f)
    if y:
        ax.yaxis.set_major_formatter(f)


def _save(fig, name):
    p = D._guard(D.FIGS / f'{name}.png'); fig.savefig(p, dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


# Plot clipped-line geometry, detections and reconstructed samples for a diagnostic case.
def fig_clipped():
    import l01_common as C0
    import r2_eval as E
    import r2_run as R
    import rdy_platform as P
    plt = _plt(); run = D.run_load(); an = run['lines']['anchor']
    M = R.manifest(); row = M[M.eval_key == an['eval_key']].iloc[0].to_dict(); inp = R.inputs_of(row); spec = inp['table']['spec']
    z = K.load_npz(K.EVL / f"{row['eval_key']}.npz"); img = K.load_npz(K.IMG / f"{row['image_key']}.npz")['image']
    acc = z['accepted'].astype(bool); unc = z['point_unconf'].astype(bool); sel = acc & unc
    mesh = P.mesh(row['surface']); c = D.classify_line_features(mesh, z['feats'][sel], spec, inp['binary'])
    pts = z['pts'].astype(np.float64); B = np.asarray(inp['binary']) > 0
    fig = plt.figure(figsize=(W_IN, 8.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85], hspace=0.75, wspace=0.32, left=0.08, right=0.97, top=0.93, bottom=0.12)

    ax = fig.add_subplot(gs[0, 0])
    import sp_lines_table as LT
    fam = LT.line_family_table(spec); owner = np.full(B.shape, np.nan)
    for u in range(0, 45):
        if B[:, u].any():
            qq = np.array([u - fam['c'][0]]); k1, _d1, _k2, _d2 = __import__('sp2_lines_table').nearest_table_line(fam, qq); owner[:, u] = k1[0]
    show = np.where(B[:, :45], owner[:, :45], np.nan)
    from matplotlib.colors import ListedColormap
    ax.imshow(np.where(show == -7, 1.0, np.where(show == -6, 0.0, np.nan)), cmap=ListedColormap(['#4c72b0', '#dd8452']), vmin=0, vmax=1, aspect='auto', origin='upper',
              extent=(-0.5, 44.5, 511.5, -0.5), interpolation='nearest')
    for k, col in ((-7, '#dd8452'), (-6, '#4c72b0')):
        u0 = 255.5 + fam['q_of'][k]; ax.axvline(u0, color=col, ls='--', lw=0.8)
    ax.axvline(-0.5, color='k', lw=0.8)
    pu = c['proj_uv']; ax.scatter(pu[:, 0], pu[:, 1], s=1.5, c='#c44e52', lw=0, label='wahre Projektorpunkte\nder 466 Punkte')
    ax.set_xlim(-6, 44.5); ax.set_ylim(511.5, -0.5); ax.set_xlabel('Projektor u [px]'); ax.set_ylabel('Projektor v [px]')
    ax.text(-0.98, 30, 'Linie −7\n(abgeschnitten,\nnicht nummeriert)\nMitte u = −0,98', fontsize=5.4, color='#a0522d', ha='left', va='top', bbox=dict(fc='white', alpha=0.8, lw=0, pad=0.6))
    ax.text(34.0, 30, 'Linie −6\nMitte u = 34,02', fontsize=5.4, color='#2f4f8f', ha='center', va='top', bbox=dict(fc='white', alpha=0.8, lw=0, pad=0.6))
    ax.set_title('(a) Projektorraster am linken Rand (H0)'); ax.legend(loc='lower right', frameon=False, fontsize=5.4); _fmt(ax)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img, cmap='gray', vmin=0, vmax=255, origin='upper', extent=(-0.5, 511.5, 511.5, -0.5), aspect='auto')
    f = z['feats'].astype(np.float64); conf = acc & ~unc
    ax.scatter(f[conf, 0], f[conf, 1], s=0.15, c='#55a868', lw=0, label='bestätigt korrekt')
    ax.scatter(f[sel, 0], f[sel, 1], s=0.6, c='#c44e52', lw=0, label='Kette 13 (nicht bestätigbar,\nals Linie −6 decodiert)')
    ax.set_xlim(-0.5, 200); ax.set_ylim(511.5, -0.5); ax.set_xlabel('Kamera u [px] (Ausschnitt u ≤ 200)'); ax.set_ylabel('Kamera v [px]')
    ax.set_title('(b) Kamerabild σ = 0 mit Merkmalen'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=5.2, markerscale=6, ncol=2); _fmt(ax)

    ax = fig.add_subplot(gs[1, 0])
    X, Y, Zs = mesh['X'], mesh['Y'], np.asarray(mesh['Z'], np.float64)
    j0 = int(np.argmin(np.abs(X[0, :] - 0.0)))
    ax.plot(Y[:, j0], Zs[:, j0], color='#888888', lw=0.8, label='Referenzfläche bei x = 0')
    xt = c['xyz']; mid = np.abs(pts[sel, 0]) <= 3.0
    ax.scatter(xt[:, 1], xt[:, 2], s=1.2, c='#4c72b0', lw=0, label='wahre Oberflächenpunkte (Kamerastrahl)')
    ax.scatter(pts[sel, 1], pts[sel, 2], s=1.2, c='#c44e52', lw=0, label='rekonstruiert (Linie −6)')
    for i in np.flatnonzero(mid)[::12]:
        ax.annotate('', xy=(pts[sel][i, 1], pts[sel][i, 2]), xytext=(xt[i, 1], xt[i, 2]), arrowprops=dict(arrowstyle='->', lw=0.4, color='#555555'))
    ax.axhspan(-16.0, 4.0, color='#f2f2f2', zorder=0); ax.axhline(-16.0, color='#999999', lw=0.6, ls=':'); ax.axhline(4.0, color='#999999', lw=0.6, ls=':')
    ax.axvline(-15.0, color='#2ca02c', lw=0.7, ls='--'); ax.axvline(15.0, color='#2ca02c', lw=0.7, ls='--')
    ax.text(-14.6, -17.3, 'Rand lokaler Kreis\n(y = −15 mm bei x = 0)', fontsize=5.0, color='#2ca02c', va='bottom', ha='left')
    ax.text(-51.5, -15.4, 'Suchfenster [−16, +4] mm (grau)', fontsize=5.0, color='#777777', va='bottom')
    ax.axvline(-43.5, color='#999999', lw=0.6, ls='-.'); ax.text(-43.3, -9, 'Ω₁-Rand\ny = −43,5', fontsize=5.0, color='#777777')
    ax.set_xlim(-52, -5); ax.set_ylim(-18, 5); ax.set_aspect('equal'); ax.set_xlabel('y [mm]'); ax.set_ylabel('z [mm]')
    e0, e1 = an['err_z_minus_zr']
    ax.set_title(f'(c) Seitenprofil 1:1, Tiefenfehler {_de(e0, 2)} … {_de(e1, 2)} mm'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=5.0, markerscale=4, ncol=2); _fmt(ax)

    ax = fig.add_subplot(gs[1, 1])
    dom = E.domain('omega1'); ext = (dom['x0'] - 0.25, dom['x0'] + (dom['nx'] - 0.5) * 0.5, dom['y0'] - 0.25, dom['y0'] + (dom['ny'] - 0.5) * 0.5)
    ax.imshow(np.where(dom['mask'], 1.0, np.nan), origin='lower', extent=ext, cmap=ListedColormap(['#e8e8e8']), vmin=0, vmax=1)
    ok = acc & ~unc
    ax.scatter(pts[ok, 0][::6], pts[ok, 1][::6], s=0.15, c='#9a9a9a', lw=0, label='übrige akzeptierte Punkte (jeder 6.)')
    ax.scatter(xt[:, 0], xt[:, 1], s=0.8, c='#4c72b0', lw=0, label='wahrer Ort der 466 Punkte')
    ax.scatter(pts[sel, 0], pts[sel, 1], s=0.8, c='#c44e52', lw=0, label='rekonstruierter Ort')
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(15 * np.cos(th), 15 * np.sin(th), color='#2ca02c', lw=0.8)
    ax.set_xlim(-48, 48); ax.set_ylim(-50, 48); ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title('(d) Draufsicht: Ω₁ (grau), lokaler Kreis (grün)\nwahre Orte y ≈ −45 mm liegen außerhalb von Ω₁'); ax.legend(loc='upper center', frameon=True, fontsize=5.0, markerscale=5, ncol=1); _fmt(ax)

    ax = fig.add_subplot(gs[2, 0])
    by = pd.DataFrame(run['lines']['summary']['by_condition']); by = by[by.phase == 'H0']
    surf = list(K.SURF); sig = list(K.SIGMAS); Mx = np.zeros((3, 6))
    for i, s in enumerate(surf):
        for jj, sg in enumerate(sig):
            r = by[(by.surface == s) & (by.sigma == sg)]
            Mx[i, jj] = r.accepted_clipped_points.iloc[0] if len(r) else 0
    ax.imshow(np.where(Mx > 0, Mx, np.nan), cmap='Reds', aspect='auto', vmin=0, vmax=Mx.max())
    for i, s in enumerate(surf):
        for jj, sg in enumerate(sig):
            r = by[(by.surface == s) & (by.sigma == sg)].iloc[0]
            txt = f"{int(r.accepted_clipped_points)}\n({int(r.images_with_accepted_clipped)}/{int(r.n_images)})" if r.accepted_clipped_points else f"0\n(0/{int(r.n_images)})"
            ax.text(jj, i, txt, ha='center', va='center', fontsize=5.0, color=('white' if Mx[i, jj] > 0.5 * Mx.max() else 'black'))
    ax.set_xticks(range(6)); ax.set_xticklabels([f'{s:g}' for s in sig]); ax.set_yticks(range(3)); ax.set_yticklabels(['B', 'M10', 'M24'])
    ax.set_xlabel('Rauschen σ [Grauwerte]'); ax.set_title('(e) akzeptierte Punkte aus Linie −7, H0\n(Punkte; in Klammern betroffene / alle eindeutigen Bilder)')

    ax = fig.add_subplot(gs[2, 1])
    W = [w for w in run['lines']['window_explanation'] if w['sigma'] == 0.0]
    for i, w in enumerate(W):
        a0, a1 = w['true_z_mm']; b0, b1 = w['neighbour_candidate_depth_approx_mm']
        ax.plot([a0, a1], [i + 0.12] * 2, color='#4c72b0', lw=3, solid_capstyle='butt')
        ax.plot([b0, b1], [i - 0.12] * 2, color='#c44e52', lw=3, solid_capstyle='butt')
        st = ', '.join(f'{k}: {v}' for k, v in w['status'].items())
        ax.text(5.5, i, st.replace('segment_unknown', 'Segment unbekannt').replace('no_candidate', 'kein Kandidat').replace('accepted', 'akzeptiert'), fontsize=5.0, va='center')
    ax.axvspan(-16, 4, color='#f2f2f2', zorder=0); ax.axvline(-16, color='#999999', lw=0.6, ls=':'); ax.axvline(4, color='#999999', lw=0.6, ls=':')
    sn = {'B_curv': 'B', 'M10_curv': 'M10', 'M24_curv': 'M24'}; ax.set_yticks(range(len(W))); ax.set_yticklabels([sn[w['surface']] for w in W])
    ax.set_xlim(-30, 30); ax.set_ylim(-0.6, len(W) - 0.4); ax.set_xlabel('z [mm]')
    ax.plot([], [], color='#4c72b0', lw=3, label='wahre Höhe am Rand'); ax.plot([], [], color='#c44e52', lw=3, label='Tiefe des Nachbarlinien-Kandidaten (≈ −15,5 mm)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), frameon=False, fontsize=5.0, ncol=1); ax.set_title('(f) Fenster entscheidet über Annahme (S0, H0, σ = 0)'); _fmt(ax, y=False)
    fig.suptitle('Linien L6,3 – abgeschnittene Randlinie −7 wird als Linie −6 rekonstruiert (M10, S0, H0, σ = 0)', y=0.985)
    fig.text(0.01, 0.006, f"Bewertungsschlüssel {an['eval_key']}, Decodierschlüssel {an['decode_key']}.\nOffizielle Identität (Tabellenwahrheit) unverändert: nicht bestätigbar; "
             'die Quellklasse ist eine getrennte Diagnose.\n' + SRC, fontsize=5.0, color='#555555')
    return _save(fig, 'D1_Randlinie_Diagnose')


# Plot a selected dot-detection counterexample and its reconstruction map.
def fig_dot_exception():
    import matplotlib.colors as mcolors
    import r2_eval as E
    import r2_figs as F2
    plt = _plt(); run = D.run_load(); comp = run['dots']['comparison']
    order = ['S0 same noise', 'exception'] + [c['role'] for c in comp if c['role'].startswith('S1') and c['sigma'] in (60.0, 100.0)]
    cols = [next(c for c in comp if c['role'] == o) for o in order]
    title = {'S0 same noise': 'S0 regulär, σ = 80\n(gleiches Rauschfeld)', 'exception': 'S1 B000, σ = 80\n(Ausnahme)'}
    dom = E.domain('omega1'); sid = K.RK.SURF_MAP['B_curv']; z_ref = np.asarray(dom[f'z_{sid}'], np.float64)
    ext = (dom['x0'] - 0.25, dom['x0'] + (dom['nx'] - 0.5) * 0.5, dom['y0'] - 0.25, dom['y0'] + (dom['ny'] - 0.5) * 0.5)
    import r2_run as R
    Mn = R.manifest()
    fig, axs = plt.subplots(3, len(cols), figsize=(W_IN, 5.6), gridspec_kw=dict(height_ratios=[1.0, 0.75, 1.0]))
    cmap = plt.get_cmap('RdBu_r'); norm = mcolors.Normalize(-0.3, 0.3); checks = []
    for j, c in enumerate(cols):
        img = K.load_npz(K.IMG / f"{c['image_key']}.npz")['image']; z = K.load_npz(K.EVL / f"{c['eval_key']}.npz")
        f = z['feats'].astype(np.float64); conf = z['feat_confirmable'].astype(bool)
        ax = axs[0, j]; x0, y0, wpx = 150, 150, 210
        ax.imshow(img[y0:y0 + wpx, x0:x0 + wpx], cmap='gray', vmin=0, vmax=255, extent=(x0 - 0.5, x0 + wpx - 0.5, y0 + wpx - 0.5, y0 - 0.5))
        inb = (f[:, 0] >= x0) & (f[:, 0] < x0 + wpx) & (f[:, 1] >= y0) & (f[:, 1] < y0 + wpx)
        ax.scatter(f[inb & conf, 0], f[inb & conf, 1], s=9, facecolors='none', edgecolors='#2ca02c', lw=0.6)
        ax.scatter(f[inb & ~conf, 0], f[inb & ~conf, 1], s=9, marker='x', c='#d62728', lw=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title.get(c['role'], f"S1 B000, σ = {c['sigma']:g}"), fontsize=6.4)
        ax.text(0.02, 0.02, f"gewählt {c['n_selected']} / bestätigbar {c['n_confirmable']}", transform=ax.transAxes, fontsize=4.8, color='k', bbox=dict(fc='white', alpha=0.8, lw=0, pad=0.5))

        ax = axs[1, j]; area = z['cand_area'].astype(np.float64); code = z['cand_code'].astype(int)
        ref = float(c['area_ref_px']); mrow = Mn[Mn.eval_key == c['eval_key']].iloc[0]
        ref_ek = Mn[Mn.image_key == mrow['sigma0_ref_key']].iloc[0]['eval_key']
        a0 = float(K.load_json(K.EVL / f'{ref_ek}.json')['info']['det_info']['blob_area_median_px'])
        dsz = (area >= 0.5 * a0) & (area <= 1.5 * a0); share = float(area[dsz].sum() / area.sum()) if area.sum() else None
        bins = np.arange(0, 181, 5)
        ax.hist(area[code != 0], bins=bins, color='#bbbbbb', label='verworfen')
        ax.hist(area[code == 0], bins=bins, color='#4c72b0', label='angenommen')
        ax.axvspan(0.5 * ref, 1.5 * ref, color='#4c72b0', alpha=0.12)
        ax.axvline(ref, color='#4c72b0', lw=0.8)
        if a0:
            ax.axvline(a0, color='#2ca02c', lw=0.8, ls='--')
        ax.set_yscale('log'); ax.set_xlim(0, 180); ax.set_xlabel('Komponentenfläche [px]'); _fmt(ax, y=False)
        ax.text(0.98, 0.95, f"Referenz {ref:g} px\nσ0-Punkt {a0:g} px" + (f"\npunktgroß: {_de(100 * share, 1)} %\nder Fläche" if share is not None else ''), transform=ax.transAxes, fontsize=5.0, ha='right', va='top')
        if j == 0:
            ax.set_ylabel('Anzahl'); ax.legend(loc='center right', frameon=False, fontsize=5.0)

        ax = axs[2, j]; Pv = np.asarray(z['pts'], np.float64)[np.asarray(z['valid'], bool)]
        ns = F2._node_state(Pv, dom, z_ref); checks.append(dict(eval_key=c['eval_key'], Ctau_fig=ns['Ctau'], Ctau_table=c['glob_Ctau'], equal=bool(abs(ns['Ctau'] - c['glob_Ctau']) <= 1e-12)))
        rgba = np.zeros((len(dom['x']), 4)); rgba[:] = mcolors.to_rgba('#d9d9d9'); rgba[ns['state'] == 1] = mcolors.to_rgba('#e6cf9c'); s2 = ns['state'] == 2
        rgba[s2] = cmap(norm(np.clip(ns['err'][s2], -0.3, 0.3)))
        ax.imshow(F2._grid(dom, rgba, fill=1.0), origin='lower', extent=ext, interpolation='nearest')
        if len(Pv):
            ax.scatter(Pv[:, 0], Pv[:, 1], s=1.0, c='k', lw=0)
        ax.set_xticks([]); ax.set_yticks([])
        s = c['src']
        ax.text(0.02, 0.02, f"C_τ {_de(100 * c['glob_Ctau'], 2)} % (lokal {_de(100 * c['loc_Ctau'], 2)} %)\nkorrekt {s['c']} · unbest. {s['u']} · falsch {s['w']}",
                transform=ax.transAxes, fontsize=4.6, bbox=dict(fc='white', alpha=0.8, lw=0, pad=0.5))
    fig.suptitle('Punkte R8×4 – Ausnahme bei σ = 80 (B, H2):\ndie Flächenreferenz entscheidet, ob echte Punkte die Auswahl passieren', y=0.995)
    fig.text(0.01, 0.005, 'Zeile 1: Kameraausschnitt, Kreis = geometrisch bestätigbares Merkmal, × = nicht bestätigbar.\n'
             'Zeile 2: blaue Linie = flächengewichteter Median (Referenz), Band = Annahmefenster 0,5…1,5 × Referenz, grün gestrichelt = Punktfläche im σ0-Bild.\n'
             'Zeile 3: Knotenstatus auf Ω₁ wie F07–F12 (grau außerhalb der Hülle, beige nicht gestützt, Farbe = Fehler ±0,3 mm); Quelle = Quellidentität aus dem σ0-Bild.\n'
             'Nachträgliche Diagnose eines Einzelfalls, keine Häufigkeitsaussage. ' + SRC, fontsize=4.8, color='#555555')
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    p = _save(fig, 'D2_Punkte_Ausnahme_sigma80')
    return p, checks


def make_all() -> dict:
    p1 = fig_clipped(); p2, checks = fig_dot_exception()
    out = dict(files=[K.rel(p1), K.rel(p2)], hashes={K.rel(p): D.sha(p) for p in (p1, p2)}, d2_ctau_checks=checks, all_equal=all(x['equal'] for x in checks), t_local=K.now_local())
    D.run_put('figures', out)
    return out
