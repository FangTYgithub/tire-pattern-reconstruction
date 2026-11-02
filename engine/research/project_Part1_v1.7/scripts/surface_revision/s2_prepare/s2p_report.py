# Summarize spacing-layout evidence and generate its report figures.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2p_common as K

K.setup()
import s2p_stats as ST

KEEP = ('state', 'n_out', 'n_accepted', 'n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'id_denominator', 'mismatch_rate', 'unconf_share', 'img_n_expected', 'img_n_recovered', 'img_recovery',
        'n_features_design', 'n_candidates', 'n_selected', 'n_matched', 'n_ambiguous', 'n_no_correspondence', 'n_detected',
        'loc_n_valid', 'loc_pt_rmse', 'loc_pt_mae', 'loc_pt_p95', 'loc_pt_bias', 'loc_gross_rate', 'loc_id_correct', 'loc_id_wrong', 'loc_id_unconf', 'loc_mismatch_rate', 'loc_n_expected', 'loc_n_recovered',
        'loc_recovery', 'loc_n_nodes', 'loc_n_interp', 'loc_cov_interp', 'loc_Cvalid', 'loc_Ctau', 'loc_surf_rmse', 'loc_surf_mae', 'loc_surf_p95', 'loc_surf_bias',
        'glob_n_valid', 'glob_pt_rmse', 'glob_pt_mae', 'glob_pt_p95', 'glob_pt_bias', 'glob_gross_rate', 'glob_id_correct', 'glob_id_wrong', 'glob_id_unconf', 'glob_mismatch_rate', 'glob_n_expected',
        'glob_n_recovered', 'glob_recovery', 'glob_n_nodes', 'glob_n_interp', 'glob_cov_interp', 'glob_Cvalid', 'glob_Ctau', 'glob_surf_rmse', 'glob_surf_mae', 'glob_surf_p95', 'glob_surf_bias',
        'amp_applicable', 'amp_true_mm', 'amp_peak_mm', 'amp_centre_mm', 'amp_err_peak_mm', 'amp_err_centre_mm', 'amp_err_peak_pct')
EXTRA_METRICS = ('loc_pt_bias', 'glob_pt_bias', 'loc_n_valid', 'glob_n_valid', 'loc_surf_mae', 'glob_surf_mae', 'loc_surf_p95', 'glob_surf_p95', 'loc_mismatch_rate', 'glob_mismatch_rate')
AMP_METRICS = ('amp_peak_mm', 'amp_centre_mm', 'amp_err_peak_pct')
RES = K.RESULTS
FIG = K.OUT / 'figures'


def _tag(set_name):
    return set_name.replace('-', '_')


def records(set_name: str, A: pd.DataFrame | None = None, parts: dict | None = None) -> pd.DataFrame:
    import s1r_common as S1K
    import s2p_case as CS
    import s2p_formal as FM
    import s2p_reuse as RU
    import s2p_run as RN
    if parts is None:
        parts = K.load_json(FM.FREEZE_FORMAL)['part_digests']
    if A is None:
        A = FM.plan(set_name, parts)
    if FM._B.version is None:
        FM._B.version = K.platform_state()['version']
    T, _ = RU.build(RN.load_list()); idx = {r['image_key']: r for r in T.to_dict('records')}
    out = []
    for _, t in A.iterrows():
        t = t.to_dict()
        rec = dict(cand_id=t['cand_id'], decode_key=t['decode_key'], image_key=t['image_key'], noise_seed=K.norm_seed(t['noise_seed']), rep=int(t['rep']), phase_label=t['phase_label'],
                   surface=t['surface'], family=t['family'], regime=t['regime'], layout=t['layout'], sigma=float(t['sigma']), peak=float(t['peak']), action=t['action'])
        if t['action'] == 'reuse_s1':
            u = idx.get(t['image_key']) or {}
            lay = RU.verify_layers(u) if u else dict(image_ok=False, decode_ok=False, eval_ok=False, reasons=['no reuse match'])
            src = K.S1OUT / 'decodes' / f"{u.get('s1_decode_key')}.json"; ok = bool(lay['image_ok'] and lay['decode_ok'] and lay['eval_ok'] and S1K.complete(src))
            m = K.load_json(src)['metrics'] if ok else {}
            rec.update(source='S1 reuse (image + decode + evaluation, read only)', record=K.rel(src), image_record=K.rel(K.S1OUT / 'images' / f"{u.get('s1_image_key')}.json"),
                       complete=ok, state=(m.get('state') if ok else 'technical'), **{k: m.get(k) for k in KEEP if k != 'state'})
        else:
            ep = K.RUN_DIR / 'eval' / f"{t['decode_key']}.json"
            R = FM._raster(t); peak = CS.peak_for(t, R); ids = CS.identities(t, FM._B, parts, peak)
            ok = CS.complete(ep, ids['eval']); m = {}
            if ok:
                er = K.load_json(ep); m = er['metrics']
                if K.norm_seed(er['noise_seed']) != rec['noise_seed'] or int(er['rep']) != rec['rep'] or er['phase_label'] != rec['phase_label']:
                    ok = False
            src = 'S2 stored (joint10, S2-PREP-02)' if t['cand_id'] in K.JOINT10['new'] else 'S2 formal (S2-REAL-01)'
            rec.update(source=src, record=K.rel(ep), image_record=K.rel(K.RUN_DIR / 'images' / f"{t['image_key']}.json"), complete=bool(ok), state=(m.get('state') if ok else 'technical'),
                       **{k: m.get(k) for k in KEEP if k != 'state'})
        out.append(rec)
    return pd.DataFrame(out)


def tables(set_name: str) -> dict:
    import s2p_formal as FM
    parts = K.load_json(FM.FREEZE_FORMAL)['part_digests']; A = FM.plan(set_name, parts); rec = records(set_name, A, parts)
    RES.mkdir(parents=True, exist_ok=True); tg = _tag(set_name)
    cols = ['cand_id', 'surface', 'family', 'regime', 'layout', 'lattice', 'arrangement', 'p_parallel_mm', 'p_perp_mm', 'line_pitch_mm', 'orientation_deg', 'size_mm', 'sigma', 'phase_label', 'rep',
            'noise_seed', 'seed_rule', 'raster_key', 'integral_I0', 'Q', 'peak', 'peak_f32', 'integral_actual', 'n_features', 'image_key', 'decode_key', 'eval_key', 'panel', 'action']
    C = A[cols].merge(rec.drop(columns=['decode_key', 'image_key', 'noise_seed', 'rep', 'phase_label', 'surface', 'family', 'regime', 'layout', 'sigma', 'peak', 'action']), on='cand_id')
    C.to_csv(K.guard(RES / f'{tg}_cases.csv'), index=False)
    metrics = tuple(dict.fromkeys(tuple(ST.METRICS) + EXTRA_METRICS + (AMP_METRICS if set_name == 'core-gauss' else ())))
    old = ST.METRICS; ST.METRICS = metrics
    try:
        An = ST.analyse(A, rec)
    finally:
        ST.METRICS = old
    An['conditions'].to_csv(K.guard(RES / f'{tg}_conditions.csv'), index=False)
    An['contrasts'].to_csv(K.guard(RES / f'{tg}_contrasts.csv'), index=False)
    An['surface_contrasts'].to_csv(K.guard(RES / f'{tg}_surface_contrasts.csv'), index=False)
    n = len(A); ok = int((rec.complete & rec.state.isin(['ok', 'no_output'])).sum())
    tal = FM.SetLedger(set_name, {}).tally()
    tech = dict(set=set_name, denominator_registered_cases=n, technically_complete=ok, technical_missing=n - ok, scientific_no_output=int((rec.state == 'no_output').sum()),
                sources={k: int(v) for k, v in rec.source.value_counts().items()}, by_surface={s: dict(cases=int(len(g)), complete=int(g.complete.sum()), no_output=int((g.state == 'no_output').sum()))
                                                                                             for s, g in rec.groupby('surface')},
                no_output_by_layout_sigma={f'{s}|{l}|σ{g:g}': int(c) for (s, l, g), c in rec[rec.state == 'no_output'].groupby(['surface', 'layout', 'sigma']).size().items()},
                calls=dict(this_set=tal, note='normal = first attempt of a key in this set; retry = further attempts; joint10 = the 8 + 8 calls of S2-PREP-02 (separate)'),
                formal_statistics=dict(planned_decodes=An['validation']['n_planned'], ok=An['validation']['n_ok'], no_output=An['validation']['n_no_output'], formal_ok=An['validation']['formal_ok'],
                                       problems=dict(list(An['validation']['problems'].items())[:20]), cells_complete=int((An['conditions'].drop_duplicates(ST.CELL).state == 'complete').sum()),
                                       cells_total=int(An['conditions'].drop_duplicates(ST.CELL).shape[0]),
                                       contrasts=An['contrasts'].state.value_counts().to_dict() if len(An['contrasts']) else {},
                                       surface_contrasts=An['surface_contrasts'].state.value_counts().to_dict() if len(An['surface_contrasts']) else {}),
                denominators=dict(errors='point errors over valid points of the domain; surface errors over interpolable nodes; both undefined for no-output cases (n_phases_defined / total stated)',
                                  coverage='cov_interp, C_valid, C_tau over all domain nodes (22605 local, 30947 global); no-output = 0',
                                  recovery='image: unique expected features; domain: expected features whose true point lies in the domain', phases='8 registered phases per cell; reps averaged inside a phase'),
                ok=bool(ok == n), t_local=K.now_local())
    K.save_json(RES / f'{tg}_technical.json', tech)
    return tech


SURF_DE = {'M10_curv': 'M10 + Zusatzkrümmung', 'M24_curv': 'M24 + Zusatzkrümmung', 'B_curv': 'Ebene Fahrbahn B + Zusatzkrümmung', 'M10_curv_gauss': 'M10 + Zusatzkrümmung + Gauß-Erhebung'}
SURF_COL = {'M10_curv': '#2a78d6', 'M24_curv': '#eb6834', 'B_curv': '#1baf7a', 'M10_curv_gauss': '#9b59b6'}
SURF_MK = {'M10_curv': 'o', 'M24_curv': 's', 'B_curv': '^', 'M10_curv_gauss': 'D'}
CURVE_METRICS = [('pt_rmse', 'Punkt-RMSE [mm]', 'log'), ('surf_rmse', 'Flächen-RMSE [mm]', 'log'), ('Ctau', 'C$_\\tau$ (d 6 mm, τ 0,15 mm)', 'lin01'),
                 ('Cvalid', 'gestützte Abdeckung C$_{valid}$', 'lin01'), ('cov_interp', 'interpolierbare Abdeckung', 'lin01'), ('recovery', 'Wiederfindung im Bereich', 'lin01')]
DOM_DE = {'loc': 'lokaler Bereich (r ≤ 15 mm, vorgegeben)', 'glob': 'gemeinsamer Messbereich'}
DOT_SERIES = [('p$_\\parallel$ (entlang Epipolar) [mm], p$_\\perp$ = 4 mm', ['R6x4', 'R7.35x4', 'R8x4', 'R10x4', 'R12x4'], 0),
              ('p$_\\perp$ (Zeilenabstand) [mm], p$_\\parallel$ = 8 mm', ['R8x2.5', 'R8x3', 'R8x4', 'R8x6', 'R8x8'], 1)]


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6, 'axes.grid': True,
                         'grid.color': '#e1e0d9', 'grid.linewidth': 0.5, 'figure.facecolor': 'white', 'axes.formatter.use_locale': False})
    return plt


def _comma(fig):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, p: f'{v:g}'.replace('.', ','))
    for ax in fig.axes:
        if ax.name != '3d':
            if ax.get_xscale() == 'linear' and not getattr(ax, '_keep_xticks', False):
                ax.xaxis.set_major_formatter(f)
            if ax.get_yscale() == 'linear':
                ax.yaxis.set_major_formatter(f)


def _series(c, s, layouts, getx):
    x = c[(c.surface == s) & c.layout.isin(layouts)].copy()
    if x.empty:
        return x
    x['p'] = [getx(l) for l in x.layout]
    return x.sort_values('p')


# Plot coverage and error curves for the selected spacing and layout groups.
def fig_curves(cond: pd.DataFrame, set_name: str, family: str, dom: str, sigma: float, surfaces):
    plt = _plt()
    c0 = cond[(cond.family == family) & (cond.sigma == sigma) & (cond.regime == 'equal_integral')]
    inc = c0[c0.state != 'complete'].drop_duplicates(ST.CELL)
    ncol = 3 if family == 'point' else 2
    fig, axes = plt.subplots(len(CURVE_METRICS), ncol, figsize=(4.0 * ncol, 1.9 * len(CURVE_METRICS)), squeeze=False)
    for i, (met, lab, sc) in enumerate(CURVE_METRICS):
        c = c0[(c0.metric == f'{dom}_{met}') & (c0.state == 'complete')]
        if family == 'point':
            for j, (title, lats, k) in enumerate(DOT_SERIES):
                ax = axes[i, j]
                for s in surfaces:
                    x = _series(c, s, lats, lambda l, k=k: float(l[1:].split('x')[k]))
                    if x.empty:
                        continue
                    y = pd.to_numeric(x['mean'], errors='coerce').astype(float)
                    ax.plot(x.p, y, marker=SURF_MK[s], ms=3, lw=1, color=SURF_COL[s], label=SURF_DE[s])
                    ax.fill_between(x.p, pd.to_numeric(x['min'], errors='coerce').astype(float), pd.to_numeric(x['max'], errors='coerce').astype(float), color=SURF_COL[s], alpha=0.12, lw=0)
                if i == len(CURVE_METRICS) - 1:
                    ax.set_xlabel(title)
                if i == 0:
                    ax.set_title(title.split(' [')[0] + (' (p$_\\perp$ 4 mm)' if k == 0 else ' (p$_\\parallel$ 8 mm)'))
            ax = axes[i, 2]
            for jj, s in enumerate(surfaces):
                for kk, lat in enumerate(('R8x4', 'T8x4')):
                    v = c[(c.surface == s) & (c.layout == lat)]
                    if len(v) and pd.notna(v['mean'].iloc[0]):
                        y = float(v['mean'].iloc[0]); xx = kk + 0.12 * (jj - (len(surfaces) - 1) / 2)
                        ax.errorbar([xx], [y], yerr=[[y - float(v['min'].iloc[0])], [float(v['max'].iloc[0]) - y]], marker=SURF_MK[s], ms=3, color=SURF_COL[s], capsize=1.5, lw=0.8)
            ax.set_xticks([0, 1], ['R 8 × 4 (regulär)', 'T 8 × 4 (versetzt)']); ax.set_xlim(-0.5, 1.5); ax._keep_xticks = True
            if i == 0:
                ax.set_title('Anordnung')
        else:
            for j, phi in enumerate((0.0, 45.0)):
                ax = axes[i, j]
                for s in surfaces:
                    lays = [l for l in c.layout.unique() if l.endswith(f'_phi{phi:g}')]
                    x = _series(c, s, lays, lambda l: float(l.split('_')[0][1:]))
                    if x.empty:
                        continue
                    ax.plot(x.p, pd.to_numeric(x['mean'], errors='coerce').astype(float), marker=SURF_MK[s], ms=3, lw=1, color=SURF_COL[s], label=SURF_DE[s])
                    ax.fill_between(x.p, pd.to_numeric(x['min'], errors='coerce').astype(float), pd.to_numeric(x['max'], errors='coerce').astype(float), color=SURF_COL[s], alpha=0.12, lw=0)
                if i == 0:
                    ax.set_title(f'Orientierung φ = {phi:g}° (Linienbreite 1 mm)')
                if i == len(CURVE_METRICS) - 1:
                    ax.set_xlabel('Linienabstand s (normal zur Linie) [mm]')
        for j in range(ncol):
            ax = axes[i, j]
            if sc == 'log':
                ax.set_yscale('log')
            elif sc == 'lin01':
                ax.set_ylim(-0.03, 1.03)
        axes[i, 0].set_ylabel(lab)
    axes[0, 0].legend(frameon=False, fontsize=6)
    fam = 'Punkte (r 1 mm)' if family == 'point' else 'Linien (w 1 mm)'
    fig.suptitle(f"S2-REAL-01 · {fam} · {DOM_DE[dom]} · σ = {sigma:g} · gleiches Musterintegral Q = 11 695\nLinie = Mittel über 8 registrierte Phasen (Realisierungen je Phase gemittelt)\n"
                 f"Band = min–max der Phasenwerte, kein Konfidenzintervall; undefinierte Werte nicht verbunden; {len(inc)} unvollständige Zellen nicht gezeichnet", fontsize=7.5, y=0.998)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.915, bottom=0.05, hspace=0.32, wspace=0.25)
    _comma(fig)
    stem = f"S2R_{_tag(set_name)}_Kurven_{'Punkte' if family == 'point' else 'Linien'}_{'lokal' if dom == 'loc' else 'gesamt'}_s{int(sigma)}"
    FIG.mkdir(parents=True, exist_ok=True); p = K.guard(FIG / f'{stem}.png'); fig.savefig(p, dpi=170); plt.close(fig)
    return p


def curves(set_name: str) -> list:
    cond = pd.read_csv(RES / f'{_tag(set_name)}_conditions.csv', keep_default_na=False)
    cond['sigma'] = cond.sigma.astype(float)
    surfaces = K.MAIN if set_name == 'core-main' else (K.GAUSS,)
    return [K.rel(fig_curves(cond, set_name, f, d, s, surfaces)) for f in ('point', 'line') for d in ('loc', 'glob') for s in (0.0, 40.0)]


E_LIM = 0.3
C_NOINT, C_NOSUP = '#ecebe6', '#a8a69e'
PANEL_ORDER = {'panel_dots_p_parallel': ['R6x4', 'R8x4', 'R12x4'], 'panel_dots_p_perp': ['R8x2.5', 'R8x4', 'R8x8'], 'panel_dots_RT': ['R8x4', 'T8x4'],
               'panel_lines_phi0': ['s4.45_phi0', 's6.3_phi0', 's12_phi0'], 'panel_lines_phi45': ['s4.45_phi45', 's6.3_phi45', 's12_phi45']}
PANEL_DE = {'panel_dots_p_parallel': 'Punkte, p$_\\parallel$ 6 / 8 / 12 mm (p$_\\perp$ 4 mm)', 'panel_dots_p_perp': 'Punkte, p$_\\perp$ 2,5 / 4 / 8 mm (p$_\\parallel$ 8 mm)',
            'panel_dots_RT': 'Punkte 8 × 4 mm, regulär (R) / versetzt (T)', 'panel_lines_phi0': 'Linien φ = 0°, s 4,45 / 6,3 / 12 mm', 'panel_lines_phi45': 'Linien φ = 45°, s 4,45 / 6,3 / 12 mm'}


def _case_arrays(r) -> dict:

    import s1r_common as S1K
    import s2p_case as CS
    if str(r['source']).startswith('S1 reuse'):
        dk = Path(r['record']).stem; ik = Path(r['image_record']).stem
        with np.load(K.S1OUT / 'images' / f'{ik}.npz') as z:
            img = np.asarray(z['image'])
        with np.load(K.S1OUT / 'decodes' / f'{dk}.npz') as z:
            E = {k: np.asarray(z[k]) for k in z.files}
        dom = S1K.global_domain(0.5); sid = K.PLATFORM_ID[r['surface']]
        f = CS.node_fields(sid, E['pts'][E['valid'].astype(bool)], dict(x=dom['x'], y=dom['y'], area=np.full(len(dom['x']), dom['step'] ** 2), hash=dom['hash']), np.asarray(dom[f'z_{sid}'], float))
        E.update({f'glob_node_{k}': f[k] for k in ('z_est', 'err', 'support_measure', 'interpolable', 'supported', 'accurate')})
        return dict(img=img, E=E, src='S1-Wiederverwendung (Knotenfelder aus gespeicherten Punkten nachgerechnet)')
    with np.load(K.RUN_DIR / 'images' / f"{r['image_key']}.npz") as z:
        img = np.asarray(z['image'])
    with np.load(K.RUN_DIR / 'eval' / f"{r['decode_key']}.npz") as z:
        E = {k: np.asarray(z[k]) for k in z.files}
    return dict(img=img, E=E, src=('S2 gespeichert (joint10)' if str(r['source']).startswith('S2 stored') else 'S2 formal'))


# Draw representative images, reconstruction clouds and coverage maps.
def fig_group(cases: pd.DataFrame, title: str, stem: str, n_draw=5000, band=1.0):
    import s1r_common as S1K
    plt = _plt()
    from matplotlib.patches import Circle, Patch
    from matplotlib.lines import Line2D
    S = S1K.s2(); dom = S1K.global_domain(0.5); gx, gy = np.asarray(dom['x']), np.asarray(dom['y']); M = dom['mask']
    LX = dom['x0'] + dom['step'] * np.arange(dom['nx']); LY = dom['y0'] + dom['step'] * np.arange(dom['ny'])
    ix = np.rint((gx - dom['x0']) / dom['step']).astype(int); iy = np.rint((gy - dom['y0']) / dom['step']).astype(int)
    n = len(cases); fig = plt.figure(figsize=(12.6, 3.2 * n + 0.6)); check = {}; meshes = {}
    Hmin, Hmax = np.inf, -np.inf
    for _, r in cases.iterrows():
        sid = K.PLATFORM_ID[r.surface]; mesh = meshes.setdefault(sid, S.build_mesh(sid))
        m = (np.abs(mesh['X']) <= 46) & (np.abs(mesh['Y']) <= 46); H = K.C_REF - mesh['Z'][m]; Hmin = min(Hmin, H.min()); Hmax = max(Hmax, H.max())
    sc2 = None
    for i, (_, r) in enumerate(cases.iterrows()):
        sid = K.PLATFORM_ID[r.surface]; mesh = meshes[sid]; D = _case_arrays(r); E = D['E']
        v = E['valid'].astype(bool); P = E['pts'][v]; dH = -(P[:, 2] - E['zr'][v].astype(float))
        rng = np.random.default_rng(0); sel = np.sort(rng.choice(len(P), min(n_draw, len(P)), replace=False)) if len(P) else np.zeros(0, int)
        interp = E['glob_node_interpolable'].astype(bool); sup = E['glob_node_supported'].astype(bool); acc = E['glob_node_accurate'].astype(bool); dHn = -E['glob_node_err'].astype(float)
        nn = len(gx); Cv = float(sup.sum()) / nn; Ct = float(acc.sum()) / nn
        eg = -(E['pts'][E['in_global'].astype(bool), 2] - E['zr'][E['in_global'].astype(bool)].astype(float)); rm = float(np.sqrt(np.mean(eg ** 2))) if len(eg) else None
        chk = dict(Cvalid=abs(Cv - float(r.glob_Cvalid or 0.0)) <= 1e-12, Ctau=abs(Ct - float(r.glob_Ctau or 0.0)) <= 1e-12,
                   pt_rmse=(rm is None and pd.isna(pd.to_numeric(r.glob_pt_rmse, errors='coerce'))) or (rm is not None and abs(rm - float(r.glob_pt_rmse)) <= 1e-6))
        check[r.cand_id] = dict(chk, ok=all(chk.values()))
        lay = r.layout if r.family == 'point' else f"s {str(r.layout).split('_')[0][1:].replace('.', ',')} mm, φ {str(r.layout).split('phi')[1]}°"
        ax0 = fig.add_subplot(n, 4, 4 * i + 1); ax0.imshow(D['img'], cmap='gray', vmin=0, vmax=255); ax0.set_axis_off()
        ax0.set_title(f"{r.cand_id} · {lay} · σ {r.sigma:g} · {r.phase_label}, Wdh. {int(r.rep)}\nKamerabild ({D['src']}) · Peak {str(round(float(r.peak), 3)).replace('.', ',')}", fontsize=6)
        ax1 = fig.add_subplot(n, 4, 4 * i + 2, projection='3d')
        X, Y, Z = mesh['X'], mesh['Y'], mesh['Z']; jx = np.flatnonzero(np.abs(X[0, :]) <= 46); jy = np.flatnonzero(np.abs(Y[:, 0]) <= 46)
        Xs, Ys, Zs = (A_[np.ix_(jy[::16], jx[::16])] for A_ in (X, Y, Z))
        ax1.plot_wireframe(-Xs, Ys, K.C_REF - Zs, color='#a8a69e', linewidth=0.3, rstride=2, cstride=2)
        if len(sel):
            ax1.scatter(-P[sel, 0], P[sel, 1], K.C_REF - P[sel, 2], s=(6 if r.family == 'point' else 2.5), c=np.clip(dH[sel], -E_LIM, E_LIM), cmap='RdBu_r', vmin=-E_LIM, vmax=E_LIM,
                        depthshade=False, edgecolors='k', linewidths=(0.25 if r.family == 'point' else 0.05))
        else:
            ax1.text2D(0.25, 0.5, 'keine gültigen Punkte', transform=ax1.transAxes, fontsize=7, color='#8b0000')
        ax1.set_xlim(-46, 46); ax1.set_ylim(-46, 46); ax1.set_zlim(Hmin - 1, Hmax + 1); ax1.view_init(elev=28, azim=-60); ax1.tick_params(labelsize=5, pad=0)
        ax1.set_xlabel('x$_p$', labelpad=-6); ax1.set_ylabel('y$_p$', labelpad=-6)
        ax1.set_title(f"Referenz (Gitter) + gültige Punkte, H [mm]\n{len(sel)} von {len(P)} gezeigt\n|ΔH| > 0,3 mm: {int((np.abs(dH) > E_LIM).sum())} Punkte", fontsize=6)
        ax2 = fig.add_subplot(n, 4, 4 * i + 3)
        import matplotlib
        from matplotlib.colors import to_rgba
        rgb = np.ones(M.shape + (4,)); rgb[iy[~interp], ix[~interp]] = to_rgba(C_NOINT); rgb[iy[interp & ~sup], ix[interp & ~sup]] = to_rgba(C_NOSUP)
        ext = (-(LX[-1] + dom['step'] / 2), -(LX[0] - dom['step'] / 2), LY[0] - dom['step'] / 2, LY[-1] + dom['step'] / 2)
        ax2.imshow(rgb[:, ::-1], origin='lower', extent=ext, interpolation='nearest')
        G = np.full(M.shape, np.nan); G[iy[sup], ix[sup]] = dHn[sup]
        sc2 = ax2.imshow(np.ma.masked_invalid(G[:, ::-1]), origin='lower', extent=ext, cmap='RdBu_r', vmin=-E_LIM, vmax=E_LIM, interpolation='nearest')
        if np.isfinite(G).sum() > 4:
            ax2.contour(-LX, LY, np.abs(G), levels=[0.15], colors='k', linewidths=0.35)
        ax2.add_patch(Circle((0, 0), 15.0, fill=False, ls='--', lw=0.6, color='#52514e'))
        if len(P) == 0:
            ax2.text(0, 30, 'keine gültige Ausgabe\n(wissenschaftliches Ergebnis)', ha='center', va='center', fontsize=6, color='#52514e')
        n_out = int((np.abs(dHn[sup]) > E_LIM).sum()); n_tau = int((sup & ~acc).sum())
        ax2.set_aspect('equal'); ax2.set_xlim(-47, 47); ax2.set_ylim(-47, 47); ax2.set_xlabel('x$_p$ [mm]'); ax2.set_ylabel('y$_p$ [mm]')
        ax2.set_title(f"C$_{{valid}}$ {str(round(Cv, 3)).replace('.', ',')} · C$_\\tau$ {str(round(Ct, 3)).replace('.', ',')} (gemeinsamer Messbereich)\nKnoten: nicht interp. {int((~interp).sum())} · ohne Stützung {int((interp & ~sup).sum())}"
                      f"\ngestützt, |ΔH| > τ: {n_tau} · jenseits ±0,3 mm: {n_out}", fontsize=6)
        ax3 = fig.add_subplot(n, 4, 4 * i + 4)
        j = int(np.argmin(np.abs(X[0, :]))); ys = Y[:, j]; keep = np.abs(ys) <= 46
        ax3.plot(ys[keep], K.C_REF - Z[keep, j], color='#52514e', lw=1)
        col0 = np.abs(gx) < 1e-9; o = np.argsort(gy[col0]); yy = gy[col0][o]; zz = E['glob_node_z_est'][col0][o].astype(float); ok_ = interp[col0][o]
        yfull = LY; hfull = np.full(len(LY), np.nan); pos = np.rint((yy - dom['y0']) / dom['step']).astype(int); hfull[pos[ok_]] = K.C_REF - zz[ok_]
        ax3.plot(yfull, hfull, color='#eb6834', lw=0.9)
        b = np.abs(P[:, 0]) <= band
        ax3.plot(P[b, 1], K.C_REF - P[b, 2], 'o', ms=1.2, color='#2a78d6')
        ax3.set_xlabel('y$_p$ [mm]'); ax3.set_ylabel('H [mm]'); ax3.set_xlim(-46, 46)
        ax3.set_title(f"Mittelschnitt x$_p$ = 0\nPunkt-RMSE (gemeinsamer Bereich) {('–' if rm is None else str(round(rm, 3)).replace('.', ','))} mm", fontsize=6)
    if sc2 is not None:
        cax = fig.add_axes([0.955, 0.25, 0.008, 0.5]); cb = fig.colorbar(sc2, cax=cax, extend='both')
        cb.set_label('ΔH = H$_{rek}$ − H$_{ref}$ = −Δz [mm] (rot: Höhe überschätzt); Werte jenseits ±0,3 mm gezählt', fontsize=6)
    fig.legend(handles=[Patch(color=C_NOINT, label='nicht interpolierbar'), Patch(color=C_NOSUP, label='interpolierbar, ohne Stützung (Stützmaß > 6 mm)'),
                        Line2D([], [], color='k', lw=0.6, label='Kontur |ΔH| = τ = 0,15 mm (gestützt; außerhalb = nicht genau)'),
                        Line2D([], [], color='#52514e', lw=1, label='Schnitt: Referenz'), Line2D([], [], color='#eb6834', lw=0.9, label='Schnitt: interpolierte Rekonstruktion (Lücke = nicht interpolierbar)'),
                        Line2D([], [], color='#2a78d6', marker='o', ms=2, ls='none', label=f'Schnitt: Rohpunkte |x$_p$| ≤ {str(band).replace(".", ",")} mm'),
                        Patch(facecolor='white', edgecolor='#52514e', ls='--', label='vorgegebener lokaler Bewertungsbereich r ≤ 15 mm (keine Verformungsgrenze)')],
               loc='lower center', ncol=4, frameon=False, fontsize=5.8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'{title}\nS2-REAL-01 formale Serie · feste registrierte Darstellungsposition (Einzelfälle, keine Phasenmittel; die Kurven enthalten alle registrierten Werte) · '
                 f'gemeinsamer Messbereich · gleiches Musterintegral Q = 11 695', fontsize=7.5, y=0.998)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.07, top=0.88 if n <= 2 else 0.9, wspace=0.42, hspace=0.75)
    FIG.mkdir(parents=True, exist_ok=True); p = K.guard(FIG / f'{stem}.png'); fig.savefig(p, dpi=170); plt.close(fig)
    return p, check


def panels(set_name: str) -> dict:
    C = pd.read_csv(RES / f'{_tag(set_name)}_cases.csv', keep_default_na=False, float_precision='round_trip')
    C['sigma'] = C.sigma.astype(float)
    surfaces = K.MAIN if set_name == 'core-main' else (K.GAUSS,)
    made, checks, index = [], {}, []
    for s in surfaces:
        for sg in (0.0, 40.0):
            for tag, order in PANEL_ORDER.items():
                G = C[(C.surface == s) & (C.sigma == sg) & C.panel.str.contains(tag, regex=False)]
                G = G.set_index('layout').loc[order].reset_index()
                if len(G) != len(order) or (G.rep.astype(int) != 0).any():
                    raise RuntimeError(f'panel group {s} {sg} {tag} incomplete: {len(G)}')
                stem = f"S2R_{_tag(set_name)}_Tafel_{s.replace('_curv', '').replace('_gauss', 'Gauss')}_s{int(sg)}_{tag.replace('panel_', '')}"
                p, chk = fig_group(G, f'{SURF_DE[s]} · {PANEL_DE[tag]} · σ = {sg:g}', stem)
                made.append(K.rel(p)); checks[stem] = chk
                index += [dict(figure=Path(p).name, group=tag, surface=s, sigma=sg, row=i + 1, cand_id=g.cand_id, layout=g.layout, phase_label=g.phase_label, rep=int(g.rep), image_key=g.image_key,
                               decode_key=g.decode_key, source=g.source, state=g.state, rule='registered panel (s2_config.panels): first registered position pos33 / phase ph0, rep 0; failures shown in place')
                          for i, (_, g) in enumerate(G.iterrows())]
    pd.DataFrame(index).to_csv(K.guard(FIG / f'S2R_{_tag(set_name)}_figure_index.csv'), index=False)
    ok = all(v['ok'] for g in checks.values() for v in g.values())
    K.save_json(RES / f'{_tag(set_name)}_figure_check.json', dict(n_figures=len(made), all_values_match=ok, checks=checks, t_local=K.now_local()))
    return dict(figures=made, values_match=ok)


def spotcheck(set_name: str) -> dict:
    import s1r_common as S1K
    import s2p_case as CS
    S = S1K.s2(); dom = S1K.global_domain(0.5)
    C = pd.read_csv(RES / f'{_tag(set_name)}_cases.csv', keep_default_na=False, float_precision='round_trip')
    F = C[C.source.str.startswith('S2 formal')]
    pick = {}
    for lab, q in (('dot', (F.family == 'point') & (F.state == 'ok')), ('line', (F.family == 'line') & (F.state == 'ok')), ('no_output', F.state == 'no_output')):
        x = F[q]
        if len(x):
            pick[lab] = x.iloc[0]
    out = {}
    for lab, r in pick.items():
        ep = K.RUN_DIR / 'eval' / f'{r.decode_key}.json'; m = K.load_json(ep)['metrics']
        with np.load(ep.with_suffix('.npz')) as z:
            E = {k: np.asarray(z[k]) for k in z.files}
        v = E['valid'].astype(bool); e = E['pts'][:, 2] - E['zr'].astype(float)
        loc = v & S1K.in_local(E['pts'][:, :2]); glob = v & S1K.in_global(E['pts'][:, :2], dom)
        d = dict(masks_equal=bool(np.array_equal(loc, E['in_local']) and np.array_equal(glob, E['in_global'])),
                 identity_closure=int(E['point_correct'].sum() + E['point_wrong'].sum() + E['point_unconf'].sum()) == int(E['accepted'].sum()))
        for pre, mk in (('loc', loc), ('glob', glob)):
            ee = e[mk]
            for name, val in (('pt_rmse', (np.sqrt(np.mean(ee ** 2)) if len(ee) else None)), ('pt_p95', (np.percentile(np.abs(ee), 95) if len(ee) else None))):
                rv = m[f'{pre}_{name}']; tv = pd.to_numeric(r[f'{pre}_{name}'], errors='coerce')
                d[f'{pre}_{name}'] = bool((val is None and rv is None and pd.isna(tv)) or (val is not None and rv is not None and abs(val - rv) <= 1e-6 and abs(float(tv) - rv) <= 1e-12))
        sid = K.PLATFORM_ID[r.surface]; mesh = S.build_mesh(sid); dl = S.roi_domain(mesh)
        fl = CS.node_fields(sid, E['pts'][v], dict(x=dl['x'], y=dl['y'], area=dl['area'], hash=dl['hash']), np.asarray(dl['z_ref'], float))
        fg = CS.node_fields(sid, E['pts'][v], dict(x=dom['x'], y=dom['y'], area=np.full(len(dom['x']), dom['step'] ** 2), hash=dom['hash']), np.asarray(dom[f'z_{sid}'], float))
        for pre, f in (('loc', fl), ('glob', fg)):
            d[f'{pre}_node_masks_equal'] = bool(np.array_equal(f['supported'], E[f'{pre}_node_supported']) and np.array_equal(f['accurate'], E[f'{pre}_node_accurate'])
                                                 and np.array_equal(f['interpolable'], E[f'{pre}_node_interpolable']))
            d[f'{pre}_Ctau_record_table'] = bool(abs(f['Ctau'] - (m[f'{pre}_Ctau'] or 0.0)) <= 1e-12 and abs(float(r[f'{pre}_Ctau'] or 0.0) - (m[f'{pre}_Ctau'] or 0.0)) <= 1e-12)
        if lab == 'no_output':
            d['no_output_semantics'] = bool(m['n_valid'] == 0 and (m['glob_Ctau'] or 0.0) == 0.0 and m['glob_pt_rmse'] is None and pd.isna(pd.to_numeric(r.glob_pt_rmse, errors='coerce')))
        out[lab] = dict(cand_id=r.cand_id, surface=r.surface, layout=r.layout, sigma=float(r.sigma), state=r.state, **d, ok=all(x for x in d.values()))
    res = dict(set=set_name, cases=out, all_ok=bool(out) and all(x['ok'] for x in out.values()), missing_kinds=[k for k in ('dot', 'line', 'no_output') if k not in out], t_local=K.now_local(),
               note='recomputed from the stored NPZ (point stats tolerance 1e-6 mm: zr stored as float32; masks / coverage exact)')
    K.save_json(RES / f'{_tag(set_name)}_spotcheck.json', res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['tables', 'curves', 'panels', 'spotcheck', 'all'])
    ap.add_argument('--set', required=True, choices=['core-main', 'core-gauss'])
    a = ap.parse_args(argv)
    try:
        out = {}
        if a.cmd in ('tables', 'all'):
            out['tables'] = tables(a.set)
            if a.cmd == 'all' and not out['tables']['ok']:
                raise RuntimeError(f"set {a.set} not technically complete ({out['tables']['technically_complete']} / {out['tables']['denominator_registered_cases']}): no figures")
        if a.cmd in ('curves', 'all'):
            out['curves'] = curves(a.set)
        if a.cmd in ('panels', 'all'):
            out['panels'] = panels(a.set)
        if a.cmd in ('spotcheck', 'all'):
            out['spotcheck'] = spotcheck(a.set)
        print(json.dumps(K.clean(out), ensure_ascii=False, indent=1)[:4000])
    except Exception:
        import traceback
        traceback.print_exc(); return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
