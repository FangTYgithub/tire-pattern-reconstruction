# Draw 3D comparisons using the stored reconstructed points and reference surfaces.

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import s5v_common as K
import s5u_figs as FG

_RDY = K.ROOT / 'scripts' / 'surface_revision' / 's345_ready'
if str(_RDY) not in sys.path:
    sys.path.insert(0, str(_RDY))
FOV = 46.08
STRIDE = dict(x=4, y=8)
ZLIM = (-17.0, 5.0)
VIEW = dict(elev=32, azim=-62)
DOT_CASES = [('D_r0.75_p8x4', 'M10_curv', 'ph7', 0.0, 0), ('D_r0.75_p8x4', 'M10_curv', 'ph7', 40.0, 1), ('D_r0.75_p8x4', 'M10_curv', 'ph7', 100.0, 1)]
LINE_CASE = ('L_w0.5_s6.3', 'M10_curv', 'ph1', 20.0, 1)


def bound_mesh(sid: str) -> tuple:
    import rdy_platform as P
    m = P.mesh(sid); S = P.s2(); h = S.mesh_sha(m)
    if h != K.REGISTERED_MESH[sid]:
        raise K.Stop('refused', f'mesh {sid}: hash {h} != registered {K.REGISTERED_MESH[sid]}')
    return m, S, h


def _grid(m: dict) -> tuple:
    X, Y, Z = m['X'], m['Y'], m['Z']
    cx = np.flatnonzero(np.abs(X[0]) <= FOV)[::STRIDE['x']]; cy = np.flatnonzero(np.abs(Y[:, 0]) <= FOV)[::STRIDE['y']]
    return X[np.ix_(cy, cx)], Y[np.ix_(cy, cx)], Z[np.ix_(cy, cx)]


def _case(C: pd.DataFrame, key, fe):
    r = C[(C.combo == key[0]) & (C.surface == key[1]) & (C.phase_label == key[2]) & (C.sigma == float(key[3])) & (C.rep == int(key[4])) & (C.front_end == fe)]
    if len(r) != 1:
        raise K.Stop('refused', f'3-D case {key} {fe}: {len(r)} rows')
    return r.iloc[0]


def _points(r, z, m, S) -> dict:
    v = z['valid'].astype(bool); P = z['pts'][v].astype(float); zt = z['zr'][v].astype(float)
    if len(P) != int(r.n_valid):
        raise K.Stop('refused', f'{r.eval_record}: {len(P)} saved valid points != table n_valid {int(r.n_valid)}')
    dz = float(np.nanmax(np.abs(S.truth_interp(m)(np.column_stack([P[:, 1], P[:, 0]])) - zt))) if len(P) else 0.0
    if dz > 1e-5:
        raise K.Stop('refused', f'{r.eval_record}: saved true heights differ from the bound mesh by {dz:.3g} mm')
    e = P[:, 2] - zt
    return dict(P=P, e=e, n_over=int((np.abs(e) > 0.5).sum()), emax=(float(np.abs(e).max()) if len(e) else None), zr_vs_mesh_max_mm=dz)


def _axes(fig, spec):
    ax = fig.add_subplot(spec, projection='3d', computed_zorder=False)
    ax.set_xlim(-FOV, FOV); ax.set_ylim(-FOV, FOV); ax.set_zlim(*ZLIM); ax.set_box_aspect((1, 1, 0.45)); ax.view_init(**VIEW)
    plt, CF = FG._plt()
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_major_formatter(CF)
    ax.tick_params(labelsize=4.6, pad=0); ax.set_xlabel('x / mm', fontsize=5.2, labelpad=-6); ax.set_ylabel('y / mm', fontsize=5.2, labelpad=-6); ax.set_zlabel('z / mm', fontsize=5.2, labelpad=-7)
    return ax


def _panel(ax, G, pts, mode, title):
    Xs, Ys, Zs = G
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    if mode == 'height':
        nz = Normalize(-16, 4); ax.plot_surface(Xs, Ys, Zs, facecolors=cm.viridis(nz(Zs)), linewidth=0, antialiased=False, shade=False, rcount=Zs.shape[0], ccount=Zs.shape[1], zorder=1)
    else:
        ax.plot_surface(Xs, Ys, Zs, color='0.55', alpha=0.4, linewidth=0, antialiased=False, shade=True, rcount=Zs.shape[0], ccount=Zs.shape[1], zorder=1)
        if pts is not None and len(pts['P']):
            P = pts['P']; few = len(P) < 1000
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.clip(pts['e'], -0.5, 0.5), cmap='RdBu_r', vmin=-0.5, vmax=0.5, s=(4.0 if few else 0.45),
                       edgecolors=('0.15' if few else 'none'), linewidths=(0.2 if few else 0), depthshade=False, zorder=3)
        elif pts is not None:
            ax.text2D(0.5, 0.55, 'keine gültigen Punkte', transform=ax.transAxes, ha='center', fontsize=6)
    ax.set_title(title, fontsize=5.4, pad=0)


def _title(r, key, fe, pts) -> str:
    pct = (lambda v: '–' if pd.isna(v) else f'{100 * float(v):.1f}'.replace('.', ','))
    rm = '–' if pd.isna(r.glob_pt_rmse) else f'{float(r.glob_pt_rmse):.3f}'.replace('.', ',')
    em = '–' if pts['emax'] is None else f"{pts['emax']:.2f}".replace('.', ',')
    return (f'σ {key[3]:g} · Wdh. {key[4]} · {fe}\ngültig (alle) {int(r.n_valid)}, global {int(r.glob_n_valid)} · C$_τ$ gl. {pct(r.glob_Ctau)} %\n'
            f'Punkt-RMSE gl. {rm} mm · |Fehler| > 0,5 mm: {pts["n_over"]} · max {em} mm')


def _colorbars(fig, rect_h, rect_e):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    plt, CF = FG._plt()
    for rect, norm, cmap, lab in ((rect_h, Normalize(-16, 4), 'viridis', 'Referenzhöhe z / mm'), (rect_e, Normalize(-0.5, 0.5), 'RdBu_r', 'z$_{rek}$ − z$_{wahr}$ / mm (±0,5 begrenzt)')):
        cax = fig.add_axes(rect); cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
        cb.set_label(lab, fontsize=5.2); cb.ax.tick_params(labelsize=4.8); cb.ax.xaxis.set_major_formatter(CF)


FOOT = ('Darstellung gespeicherter Daten (keine neue Bildgebung, Dekodierung oder Auswertung). Referenzfläche: gebundenes SURF-02-Netz der Fläche (Hash geprüft) im registrierten\n'
        'Sichtfeld |x|, |y| ≤ 46,08 mm, zur Darstellung auf 1 × 1 mm ausgedünnt (jeder 4. x-, jeder 8. y-Knoten); z überhöht. Rekonstruktion: alle gültigen Punkte, ohne Ausdünnung\n'
        'oder Fehlerfilter, stets über der Fläche gezeichnet (keine Verdeckung). Wahre Höhe = gespeichertes z$_{wahr}$, stimmt mit dem Netz überein. Gleiche Achsen, Blickrichtung und Farbskalen.')


def draw(C: pd.DataFrame, arrays) -> list:
    plt, CF = FG._plt()
    m, S, h = bound_mesh('M10_curv'); G = _grid(m); out = []

    fig = plt.figure(figsize=(6.5, 6.6)); gs = fig.add_gridspec(2, 2, left=0.0, right=1.0, top=0.885, bottom=0.17, wspace=0.0, hspace=0.16)
    _panel(_axes(fig, gs[0, 0]), G, None, 'height', 'Referenzfläche M10 (Höhe farbcodiert)')
    cases = []
    for i, key in enumerate(DOT_CASES):
        r = _case(C, key, 'D0'); _, z = arrays(r, 'D0'); pts = _points(r, z, m, S)
        _panel(_axes(fig, gs[(i + 1) // 2, (i + 1) % 2]), G, pts, 'error', _title(r, key, 'D0', pts))
        cases.append(dict(key=key, front_end='D0', eval_record=r.eval_record, n_valid=int(r.n_valid), n_abs_err_gt_0p5=pts['n_over'], max_abs_err_mm=pts['emax'], zr_vs_mesh_max_mm=pts['zr_vs_mesh_max_mm']))
    fig.suptitle('Punkte (D0) M10 · r 0,75 (Ø 1,5) · 8 × 4 · ph7 – Referenzfläche und rekonstruierte Punktwolke (3D)', fontsize=7, x=0.01, y=0.995, ha='left')
    _colorbars(fig, [0.08, 0.135, 0.36, 0.012], [0.56, 0.135, 0.36, 0.012])
    fig.text(0.01, 0.008, FOOT, fontsize=4.7, va='bottom')
    out.append(dict(file=_save(fig, 'F7_3D_Punkte_M10.png'), mesh_sha16=h, cases=cases))

    fig = plt.figure(figsize=(6.5, 4.1)); gs = fig.add_gridspec(1, 2, left=0.0, right=1.0, top=0.83, bottom=0.25, wspace=0.0)
    _panel(_axes(fig, gs[0, 0]), G, None, 'height', 'Referenzfläche M10 (Höhe farbcodiert)')
    r = _case(C, LINE_CASE, 'P4'); _, z = arrays(r, 'P4'); pts = _points(r, z, m, S)
    lr = f'{float(r.loc_pt_rmse):.3f}'.replace('.', ',')
    t = _title(r, LINE_CASE, 'P4', pts) + f'\nlokal (r ≤ 15 mm): {int(r.loc_n_valid)} Punkte, Punkt-RMSE {lr} mm'
    _panel(_axes(fig, gs[0, 1]), G, pts, 'error', t)
    fig.suptitle('Linien (P4) M10 · b 0,5 · s 6,3 · ph1 – Randanomalie: Referenzfläche und rekonstruierte Punktwolke (3D)', fontsize=7, x=0.01, y=0.995, ha='left')
    _colorbars(fig, [0.08, 0.215, 0.36, 0.02], [0.56, 0.215, 0.36, 0.02])
    fig.text(0.01, 0.008, FOOT, fontsize=4.7, va='bottom')
    out.append(dict(file=_save(fig, 'F8_3D_Linien_Diagnose.png'), mesh_sha16=h,
                    cases=[dict(key=LINE_CASE, front_end='P4', eval_record=r.eval_record, n_valid=int(r.n_valid), glob_n_valid=int(r.glob_n_valid), loc_n_valid=int(r.loc_n_valid),
                                n_abs_err_gt_0p5=pts['n_over'], max_abs_err_mm=pts['emax'], zr_vs_mesh_max_mm=pts['zr_vs_mesh_max_mm'])]))
    return out


def _save(fig, name) -> str:
    import s5v_figs as F2D
    return F2D._save(fig, name)
