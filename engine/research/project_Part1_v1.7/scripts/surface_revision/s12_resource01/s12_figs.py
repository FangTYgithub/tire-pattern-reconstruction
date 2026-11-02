# Plot size and spacing effects from saved case tables.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_common as K

DH_LIM = 0.5
GREY_NI, GREY_NS = '#e6e6e6', '#9e9e9e'
FOOT = 'S12-RESOURCE-01 · SURF-02 v1.1 (nominelles Gerät; Literaturprofile, keine Messflächen)'
ZLIM = None


def plt_():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams.update({'font.size': 7.5, 'font.family': 'DejaVu Sans', 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5, 'figure.dpi': 150})
    return plt


def foot(fig, text=''):
    fig.text(0.01, 0.004, FOOT + (' · ' + text if text else ''), fontsize=6, color='#555555', ha='left', va='bottom')


_DOM = {}


def dom_global():
    if 'g' not in _DOM:
        _DOM['g'] = K.platform_ids()['s1k'].global_domain(0.5)
    return _DOM['g']


def zlim():
    global ZLIM
    if ZLIM is None:
        S = K.s2(); lo, hi = np.inf, -np.inf
        for s in K.SURFACES:
            m = S.build_mesh(K.PLATFORM_ID[s]); w = (np.abs(m['X']) <= 46) & (np.abs(m['Y']) <= 46); H = K.C_REF_MM - m['Z'][w]
            lo, hi = min(lo, float(H.min())), max(hi, float(H.max()))
        ZLIM = (lo - 0.5, hi + 0.5)
    return ZLIM


def load_case(u) -> dict:

    import s12_reuse as RU
    import s2p_case as SC
    ns = u if hasattr(u, 'source') else pd.Series(u)
    ip, ep = RU.image_record_path(ns), RU.eval_record_path(ns)
    with np.load(ip.with_suffix('.npz')) as z:
        img = np.asarray(z['image'])
    rec = K.load_json(ep)
    with np.load(ep.with_suffix('.npz')) as z:
        A = {k: np.asarray(z[k]) for k in ('pts', 'valid', 'zr', 'accepted') if k in z.files}
    m = rec['metrics']; sid = ns.platform_surface; dom = dom_global()
    v = A['valid'].astype(bool); P = np.asarray(A['pts'], np.float64)[v]
    nf = SC.node_fields(sid, P, dict(x=dom['x'], y=dom['y'], area=np.full(len(dom['x']), dom['step'] ** 2), hash=dom['hash']), np.asarray(dom[f'z_{sid}'], np.float64))
    if abs(nf['Cvalid'] - (m['glob_Cvalid'] or 0.0)) > 1e-12 or abs(nf['Ctau'] - (m['glob_Ctau'] or 0.0)) > 1e-12:
        raise K.Stop('technical_stop', f"node fields differ from the record {ns.input_key}: {nf['Cvalid']} / {m['glob_Cvalid']}, {nf['Ctau']} / {m['glob_Ctau']}")
    return dict(u=ns, img=img, A=A, m=m, info=rec.get('info') or {}, nf=nf, dom=dom, check=dict(input_key=ns.input_key, Cvalid_equal=True, Ctau_equal=True))


def wire(ax3, sid, color='#2f2f2f', step=20):
    m = K.s2().build_mesh(sid); X, Y, Z = m['X'], m['Y'], m['Z']
    jx = np.flatnonzero(np.abs(X[0, :]) <= 46); jy = np.flatnonzero(np.abs(Y[:, 0]) <= 46)
    Xs, Ys, Zs = (a[np.ix_(jy[::step], jx[::step])] for a in (X, Y, Z))
    ax3.plot_wireframe(-Xs, Ys, K.C_REF_MM - Zs, color=color, linewidth=0.25, zorder=1)


def detect_text(c) -> str:
    m = c['m']
    if c['u'].family == 'point':
        return (f"Komp. {m.get('n_components')} · Kand. {m.get('n_candidates')} · Merkm. {m.get('n_selected')}\nzugeordnet {m.get('n_matched')} · mehrdeutig {m.get('n_ambiguous')} · "
                f"angen. {m['n_accepted']} · gültig {m['n_valid']}")
    st = {k.replace('n_status_', ''): v for k, v in m.items() if k.startswith('n_status_') and v}
    top = ', '.join(f'{k} {v}' for k, v in sorted(st.items(), key=lambda kv: -kv[1])[:3])
    return f"erkannt {m.get('n_detected')} · angen. {m['n_accepted']} · gültig {m['n_valid']}\nStatus: {top}"


def f3(x, nd=3):
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def panels(fig, gs_cam, gs_3d, gs_map, c, title):
    import matplotlib.colors as mcolors
    u, m, nf, dom = c['u'], c['m'], c['nf'], c['dom']
    ax = fig.add_subplot(gs_cam); ax.imshow(c['img'], cmap='gray', vmin=0, vmax=255); ax.axis('off')
    ax.set_title(title, fontsize=6.2)
    ax3 = fig.add_subplot(gs_3d, projection='3d'); ax3.computed_zorder = False; wire(ax3, u.platform_surface)
    for a_ in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        a_.set_pane_color((0.62, 0.62, 0.62, 1.0))
    v = c['A']['valid'].astype(bool); P = np.asarray(c['A']['pts'], np.float64)[v]
    dH = -(P[:, 2] - np.asarray(c['A']['zr'], np.float64)[v]) if len(P) else np.zeros(0)
    n_sat = int((np.abs(dH) > DH_LIM).sum())
    if len(P) == 0:
        ax3.text2D(0.5, 0.62, 'Keine Rekonstruktion', transform=ax3.transAxes, ha='center', fontsize=8.5, color='#b00020', weight='bold')
        ax3.text2D(0.5, 0.42, 'Abdeckung 0 · RMSE undefiniert', transform=ax3.transAxes, ha='center', fontsize=6, color='#b00020')
    else:
        sel = np.sort(np.random.default_rng(0).choice(len(P), min(6000, len(P)), replace=False)); sel = np.union1d(sel, np.flatnonzero(np.abs(dH) > DH_LIM))
        ax3.scatter(-P[sel, 0], P[sel, 1], K.C_REF_MM - P[sel, 2], s=(9 if u.family == 'point' else 3.5), zorder=5, c=np.clip(dH[sel], -DH_LIM, DH_LIM), cmap='RdBu_r', vmin=-DH_LIM,
                    vmax=DH_LIM, depthshade=False, edgecolors='k', linewidths=(0.25 if u.family == 'point' else 0.0))
    zl = zlim(); ax3.set_xlim(-46, 46); ax3.set_ylim(-46, 46); ax3.set_zlim(*zl); ax3.view_init(elev=28, azim=-60); ax3.tick_params(labelsize=4.5, pad=-1)
    ax3.set_xlabel('x_p', fontsize=5, labelpad=-6); ax3.set_ylabel('y_p', fontsize=5, labelpad=-6); ax3.set_zlabel('H [mm]', fontsize=5, labelpad=-6)
    ax3.set_title(f"C_τ {f3(m.get('glob_Ctau'))} · Pkt-RMSE {f3(m.get('glob_pt_rmse'))} mm\n{int(m['n_valid'])} gültig · {n_sat} gesättigt (|ΔH| > 0,5 mm)", fontsize=5.6)
    ax = fig.add_subplot(gs_map)
    x, y = -np.asarray(dom['x']), np.asarray(dom['y']); dHn = -(nf['err'].astype(np.float64)); interp = nf['interpolable']; sup = nf['supported']
    ax.scatter(x[~interp], y[~interp], s=0.6, c=GREY_NI, marker='s', linewidths=0, rasterized=True)
    ns = interp & ~sup
    ax.scatter(x[ns], y[ns], s=0.6, c=GREY_NS, marker='s', linewidths=0, rasterized=True)
    sc = ax.scatter(x[sup], y[sup], s=0.6, c=np.clip(dHn[sup], -DH_LIM, DH_LIM), cmap='RdBu_r', norm=mcolors.Normalize(-DH_LIM, DH_LIM), marker='s', linewidths=0, rasterized=True)
    if sup.sum() > 4:
        G = np.full(dom['mask'].shape, np.nan); ix = np.rint((np.asarray(dom['x']) - dom['x0']) / dom['step']).astype(int); iy = np.rint((y - dom['y0']) / dom['step']).astype(int)
        G[iy[sup], ix[sup]] = np.abs(dHn[sup]); LX = -(dom['x0'] + dom['step'] * np.arange(dom['nx'])); LY = dom['y0'] + dom['step'] * np.arange(dom['ny'])
        ax.contour(LX, LY, G, levels=[K.TAU], colors='k', linewidths=0.3)
    t = np.linspace(0, 2 * np.pi, 200); ax.plot(15 * np.cos(t), 15 * np.sin(t), color='k', lw=0.6, ls='--')
    ax.set_aspect('equal'); ax.set_xlim(-47, 47); ax.set_ylim(-47, 47); ax.tick_params(labelsize=5); ax.set_xlabel('x_p [mm]', fontsize=5.5, labelpad=0)
    ax.set_title(f"C_τ lok. {f3(m.get('loc_Ctau'))} · Wf. {f3(m.get('glob_recovery'))}\n{detect_text(c)}\nfalsch {m['id_wrong']} · unbest. {m['id_unconf']} / {m['id_denominator']}", fontsize=5.2)
    if len(P) == 0:
        ax.text(0, 0, 'Keine Rekonstruktion', ha='center', va='center', fontsize=7.5, color='#b00020', weight='bold', bbox=dict(fc='white', ec='none', alpha=0.85))
    return sc, n_sat


def legend_nodes(fig, ncol=4, y=0.012):
    import matplotlib.lines as ml
    import matplotlib.patches as mp
    h = [mp.Patch(color=GREY_NI, label='nicht interpolierbar'), mp.Patch(color=GREY_NS, label='interpolierbar, Stützweite > d = 6 mm'),
         ml.Line2D([], [], color='k', lw=0.6, label='gestützt: Kontur |ΔH| = τ = 0,15 mm (außen > τ)'),
         ml.Line2D([], [], color='k', lw=0.6, ls='--', label='Kreis: lokaler Auswertebereich r = 15 mm (globale Kennzahlen: 30 947 Knoten)')]
    fig.legend(handles=h, loc='lower left', bbox_to_anchor=(0.01, y + 0.012), ncol=ncol, frameon=False, fontsize=5.8)


def cbar(fig, sc, rect):
    cax = fig.add_axes(rect); cb = fig.colorbar(sc, cax=cax)
    cb.set_label('ΔH = H_rec − H_ref [mm]\n(rot: zu hoch; ±0,5 abgeschnitten)', fontsize=6); cb.ax.tick_params(labelsize=6)


def profile_ref(sid):
    m = K.s2().build_mesh(sid); X, Y, Z = m['X'], m['Y'], m['Z']
    j = int(np.argmin(np.abs(X[0, :]))); ys = Y[:, j]; keep = np.abs(ys) <= 46
    return ys[keep], K.C_REF_MM - Z[keep, j]


def board(cases: list[tuple[str, object]], title: str, out_png, index: list, meta: dict, ncols=None, foot_text='') -> dict:

    plt = plt_(); n = len(cases); ncols = ncols or n
    W = max(10.0, 2.45 * ncols + 1.4); H = 11.4
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(5, ncols, height_ratios=[0.95, 1.25, 1.3, 0.85, 0.55], left=0.04, right=0.9, top=0.9, bottom=0.08, hspace=0.62, wspace=0.18)
    sc = None; rows = []; prof = []; checks = []
    for j, (lab, u) in enumerate(cases):
        c = load_case(u); checks.append(c['check'])
        sc_, n_sat = panels(fig, gs[0, j], gs[1, j], gs[2, j], c, lab); sc = sc_ or sc
        m = c['m']; prof.append((lab, c))
        rows.append([lab.split('\n')[0], f"{m['glob_Ctau']:.3f}", f"{m['loc_Ctau']:.3f}", f3(m.get('glob_pt_rmse')), f3(m.get('glob_surf_rmse')), str(int(m['n_valid'])),
                     f"{m['id_wrong']}/{m['id_unconf']}/{m['id_denominator']}", str(n_sat)])
        index.append(dict(figure=K.rel(out_png), panel=f'Spalte {j + 1}', label=lab.replace('\n', ' '), input_key=u.input_key, decode_key=u.decode_key, source=u.source, surface=u.surface,
                          family=u.family, size_mm=u.size_mm, layout=u.layout, phase=u.phase_label, sigma=u.sigma, rep=u.rep, state=m['state'], **meta))
    import matplotlib
    ax = fig.add_subplot(gs[3, :]); sid = cases[0][1].platform_surface; ys, hs = profile_ref(sid)
    ax.plot(ys, hs, color='#52514e', lw=1.5, label='Referenz (wahr)')
    dom = dom_global(); LY = dom['y0'] + dom['step'] * np.arange(dom['ny']); cols = matplotlib.colormaps['tab10']
    for k, (lab, c) in enumerate(prof):
        nf = c['nf']; xx = np.asarray(dom['x']); col0 = np.abs(xx) < 1e-9; o = np.argsort(np.asarray(dom['y'])[col0]); yy = np.asarray(dom['y'])[col0][o]
        zz = np.asarray(nf['z_est'], np.float64)[col0][o]; ok = np.asarray(nf['interpolable'])[col0][o]
        h = np.full(len(LY), np.nan); pos = np.rint((yy - dom['y0']) / dom['step']).astype(int); h[pos[ok]] = K.C_REF_MM - zz[ok]
        ax.plot(LY, h, color=cols(k % 10), lw=0.7, label=lab.split('\n')[0])
    ax.set_xlim(-46, 46); ax.set_ylim(*zlim()); ax.set_xlabel('y_p [mm]', fontsize=6.5, labelpad=1); ax.set_ylabel('H [mm]', fontsize=6.5); ax.grid(alpha=0.3)
    ax.legend(fontsize=5.4, frameon=False, ncol=min(6, len(prof) + 1), loc='upper left'); ax.set_title('Mittelschnitt x_p = 0 (interpoliert; Lücke = nicht interpolierbar, keine Verbindung)', fontsize=6.4)
    ax = fig.add_subplot(gs[4, :]); ax.axis('off')
    cl = ['Fall', 'C_τ ges.', 'C_τ lok.', 'Pkt-RMSE', 'Fl.-RMSE', 'n gültig', 'falsch/unbest./n', 'gesättigt']
    tb = ax.table(cellText=rows, colLabels=cl, loc='center', cellLoc='center', colWidths=[0.24] + [0.76 / 7] * 7); tb.auto_set_font_size(False); tb.set_fontsize(5.2); tb.scale(1.0, 1.15)
    if sc is not None:
        cbar(fig, sc, (0.925, 0.42, 0.009, 0.3))
    legend_nodes(fig, ncol=(4 if ncols >= 5 else 2))
    fig.suptitle(title, fontsize=7.8, y=0.995)
    foot(fig, foot_text or 'feste Beispielphase, kein Mittelwert; Kennzahlen aus allen Punkten; 3-D ≤ 6000 + alle gesättigten')
    fig.savefig(out_png, dpi=135); plt.close(fig)
    return dict(path=K.rel(out_png), value_check=checks, all_values_match=all(c['Cvalid_equal'] and c['Ctau_equal'] for c in checks))


def first_figure() -> dict:
    U = pd.read_csv(K.INPUTS, keep_default_na=False, float_precision='round_trip')
    V = pd.read_csv(K.MANIFEST, keep_default_na=False, float_precision='round_trip')
    sel = V[(V.section == 'S1') & (V.surface == 'M24_curv') & (V.family == 'line') & (V.regime == 'resource_cap_11695') & (V.sigma == 100.0) & (V.phase_label == 'ph0') & (V.rep == 0) &
            V.size_mm.isin([0.5, 1.0, 1.5])].sort_values('size_mm')
    Ui = U.set_index('input_key')
    cases = [(f"w {r.size_mm:g} mm · σ 100\nPeak {r.peak_f32:.3f} ({'Integral' if r.active_limit == 'integral_limited' else 'Peak'} begrenzt)", Ui.loc[r.input_key].rename(r.input_key)) for r in sel.itertuples()]
    for _, u in cases:
        u['input_key'] = u.name
    out = K.FIG / 'erstcharge' / 'S12_Erstcharge_M24_Linien_s100.png'; out.parent.mkdir(parents=True, exist_ok=True)
    idx = []
    r = board(cases, 'S12 – Erstcharge (formal): M24 · Linien 0°, s 8 mm · gemeinsame Obergrenze (Integral 11695, Peak 1) · σ = 100 · ph0 · Realisierung 0',
              out, idx, dict(selection='Erstcharge (vorab festgelegt)'))
    K.save_json(K.CHECKS / 'first_figure.json', dict(r, index=idx, t_local=K.now_local()))
    return r


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'first':
        print(first_figure())
