# Compare fitted deformation amplitudes with exact-height sampling references.

import sys, json, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

import r4_common as C
import r4_wp1 as W1
SL = C.SL
sys.path.insert(0, str(C.ROOT / 'src'))
from m5_evaluation import compute_indentation_recovery as IND
import pm014_surface as S14

OUT = C.OUT / 'wp2'
A_MM = 5.0
SIG_DENT = 20.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
RHO10 = SIG_DENT * np.sqrt(2.0 * np.log(1.0 / 0.9))
QWINS, QMIN = (8.0, 10.0, 12.0), 6
_CTX = {}


def ctx(surf):
    if surf not in _CTX:
        mesh, cfg, ref = SL.L15.surface(surf)
        X = np.asarray(mesh['X'], float); Y = np.asarray(mesh['Y'], float); Z = np.asarray(mesh['Z'], float); Zb = np.asarray(mesh['Z_base'], float)
        zmin = float(Z.min())
        base = RegularGridInterpolator((Y[:, 0], X[0, :]), Zb - zmin, bounds_error=False, fill_value=np.nan)
        _CTX[surf] = dict(mesh=mesh, cfg=cfg, ref=ref, zmin=zmin, base=base, half=float(cfg.get('fov_half_width_mm', 46.0)))
    return _CTX[surf]


def exact_points(c, xy):

    z = S14.z_ref_at(c['ref'], c['cfg'], xy[:, 0], xy[:, 1]) - c['zmin']
    return np.column_stack([xy[:, 0], xy[:, 1], z])


def official(c, pts, amb=None):
    o = IND(points_3d=pts, mesh=c['mesh'], m1_config=c['cfg'], ambiguous_mask=amb)
    pk = o['ind_depth_rec_mm']
    return (float(pk) if np.isfinite(pk) else None), o


def dips(c, pts):
    d = c['base'](np.column_stack([pts[:, 1], pts[:, 0]])) - pts[:, 2]
    fov = (np.abs(pts[:, 0]) <= c['half']) & (np.abs(pts[:, 1]) <= c['half'])
    return np.where(fov & np.isfinite(d), d, np.nan)


def quad_peak(xy, d, QWIN):

    ok = np.isfinite(d)
    if ok.sum() < QMIN:
        return None, 'too_few_points'
    xy, d = xy[ok], d[ok]
    i0 = int(np.argmax(d)); c0 = xy[i0]
    w = np.hypot(*(xy - c0).T) <= QWIN
    if w.sum() < QMIN:
        return None, 'too_few_in_window'
    u, v = (xy[w] - c0).T
    Am = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
    a, *_ = np.linalg.lstsq(Am, d[w], rcond=None)
    H = np.array([[2 * a[3], a[4]], [a[4], 2 * a[5]]])
    if not np.all(np.linalg.eigvalsh(H) < 0):
        return None, 'not_concave'
    s = np.linalg.solve(H, -a[1:3])
    if np.hypot(*s) > QWIN:
        return None, 'vertex_outside_window'
    return float(a[0] + a[1] * s[0] + a[2] * s[1] + a[3] * s[0] ** 2 + a[4] * s[0] * s[1] + a[5] * s[1] ** 2), 'ok'


def ideal(surf, ph, key):

    c = ctx(surf)
    with np.load(C.CROSS_CASES / f'{key}.npz') as z:
        exy = z['exp_xy'].astype(float); euv = z['exp_uv'].astype(float)
    hit = np.isfinite(exy).all(axis=1); vis = hit & np.isfinite(euv).all(axis=1)
    pb = exact_points(c, exy[vis]); B, _ = official(c, pb)
    pall = exact_points(c, exy[hit]); B_hit, _ = official(c, pall)
    db = dips(c, pb); i = int(np.nanargmax(db))
    rho = np.hypot(exy[vis][:, 0], exy[vis][:, 1]); rmin = float(rho.min())
    g = np.arange(-15, 15.0001, 0.25); GX, GY = np.meshgrid(g, g); m = GX ** 2 + GY ** 2 <= 15 ** 2
    dn, _ = cKDTree(exy[vis]).query(np.column_stack([GX[m], GY[m]]))
    qd = {}
    for wq in QWINS:
        q, qf = quad_peak(pb[:, :2], db, wq); qd[f'Q{wq:g}_ideal_mm'] = q; qd[f'Q{wq:g}_ideal_flag'] = qf
    return dict(surface=surf, phase=ph, n_visible=int(vis.sum()), n_hit=int(hit.sum()), rho_min_mm=rmin, nearest_x=float(exy[vis][np.argmin(rho), 0]),
                nearest_y=float(exy[vis][np.argmin(rho), 1]), B_pts_mm=float(np.nanmax(db)), B_mm=B, B_hit_mm=B_hit,
                B_analytic_mm=float(A_MM * np.exp(-rmin ** 2 / (2 * SIG_DENT ** 2))), S_mm=B - A_MM, S_pct=100 * (B - A_MM) / A_MM,
                S_nearest_mm=float(np.nanmax(db)) - A_MM, S_grid_mm=B - float(np.nanmax(db)), argmax_rho_mm=float(rho[i]),
                disc_frac_beyond_rho10=float((dn > RHO10).mean()), disc_cover_radius_mm=float(dn.max()), **qd)


def work(args):
    slot, Bref = args
    sp = slot['spec']; k = slot['key']; f = OUT / 'cases' / f'{k}.json'
    if f.exists():
        return json.loads(f.read_text(encoding='utf-8'))
    t0 = time.monotonic()
    c = ctx(sp['surface'])
    with np.load(C.CROSS_CASES / f'{k}.npz') as z:
        pts = z['pts'].astype(np.float64); amb = z['amb'].astype(bool); exy = z['exp_xy'].astype(float)
        gref = z['gt_ref'].astype(float); eref = z['err_ref'].astype(float)
    fin = np.isfinite(pts).all(axis=1); un = fin & ~amb
    Cv, oc = official(c, pts, amb)
    pdx = pts[un].copy(); pdx[:, 2] = gref[un]
    Dv, _ = official(c, pdx) if un.sum() else (None, None)
    finE = np.isfinite(exy).all(axis=1); E = exy[finE]; P = pts[un][:, :2]
    match, _, _ = W1.associate(E, P, W1.GATE_XY)
    rec = E[match >= 0]
    B1, _ = official(c, exact_points(c, rec)) if len(rec) else (None, None)

    dc = dips(c, pts[un]) if un.sum() else np.zeros(0); dd = dips(c, pdx) if un.sum() else np.zeros(0)
    assoc_out = np.zeros(un.sum(), bool); assoc_out[match[match >= 0]] = True
    info = {}
    for nm, d in (('C', dc), ('D', dd)):
        if len(d) and np.isfinite(d).any():
            i = int(np.nanargmax(d)); x, y = P[i]
            info.update({f'{nm}_argmax_rho_mm': float(np.hypot(x, y)), f'{nm}_argmax_err_ref_mm': float(eref[un][i]), f'{nm}_argmax_associated': bool(assoc_out[i]),
                         f'{nm}_max_sample_dip_mm': float(d[i])})
    qd = {}
    for wq in QWINS:
        q, qf = quad_peak(P, dc, wq) if len(dc) else (None, 'no_points')
        qd[f'Q{wq:g}_actual_mm'] = q; qd[f'Q{wq:g}_actual_flag'] = qf
    res = dict(key=k, surface=sp['surface'], phase=sp['phase'], r_mm=sp['r_mm'], v=sp['v'], sigma=sp['sigma'], seed=sp['seed'],
               n_unflagged=int(un.sum()), n_recovered_expected=int(len(rec)), A_mm=A_MM, B_mm=Bref, B1_mm=B1, D_mm=Dv, C_mm=Cv,
               C_coverage=oc.get('ind_coverage_frac') if oc else None, **qd, seconds=round(time.monotonic() - t0, 3), **info)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding='utf-8')
    return res


def nz(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


def main():
    t0 = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    slots = C.cross_slots()
    s0 = {(s['spec']['surface'], s['spec']['phase']): s['key'] for s in slots if s['spec']['sigma'] == 0}
    I = pd.DataFrame([ideal(su, ph, k) for (su, ph), k in sorted(s0.items())])
    I.to_csv(OUT / 'ideal_sampling.csv', index=False)
    Bmap = {(r.surface, r.phase): r.B_mm for r in I.itertuples()}
    with Pool(12) as pool:
        res = list(pool.imap_unordered(work, [(s, Bmap[(s['spec']['surface'], s['spec']['phase'])]) for s in slots], chunksize=10))
    R = pd.DataFrame(res)
    X = pd.read_csv(C.ROUND3 / 'analysis' / 'X_cases.csv', dtype={'vlabel': str}).set_index('key')
    R['C_saved_mm'] = [A_MM + X.loc[k, 'dent_err_mm'] if np.isfinite(X.loc[k, 'dent_err_mm']) else np.nan for k in R.key]
    R['C_check_abs_mm'] = (R.C_mm.astype(float) - R.C_saved_mm).abs()
    for col in ('status', 'L2_geometric', 'L3_research', 'L23_both', 'dent_rel_pct', 'vlabel'):
        R[col] = [X.loc[k, col] for k in R.key]
    for a, b, nm in (('B_mm', 'A_mm', 'S'), ('B1_mm', 'B_mm', 'M'), ('D_mm', 'B1_mm', 'P'), ('C_mm', 'D_mm', 'Z'), ('C_mm', 'A_mm', 'total')):
        R[f'{nm}_mm'] = R[a].astype(float) - R[b].astype(float)
        R[f'{nm}_pct'] = 100 * R[f'{nm}_mm'] / A_MM
    for wq in QWINS:
        R[f'Q{wq:g}_actual_err_pct'] = 100 * (pd.to_numeric(R[f'Q{wq:g}_actual_mm'], errors='coerce') - A_MM) / A_MM
    R = R.sort_values(['surface', 'phase', 'r_mm', 'v', 'sigma', 'seed']); R.to_csv(OUT / 'amplitude_cases.csv', index=False)
    C.save(OUT / 'summary.json', dict(generated=C.now(), n_slots=int(len(R)), seconds=round(time.monotonic() - t0, 1),
                                      C_check_max_abs_mm=nz(float(R.C_check_abs_mm.max())), n_C_undefined=int(R.C_mm.isna().sum()),
                                      n_D_undefined=int(R.D_mm.isna().sum()), n_B1_undefined=int(R.B1_mm.isna().sum()), rho10_mm=RHO10, sigma_dent_mm=SIG_DENT,
                                      order='S=B-A, M=B1-B, P=D-B1, Z=C-D', estimator='src/m5_evaluation.compute_indentation_recovery (unchanged)'))
    C.log(f'WP2 amplitude decomposition: {len(R)} slots, max |C recomputed - saved| = {R.C_check_abs_mm.max():.2e} mm, {time.monotonic() - t0:.0f} s')
    print(I.round(4).to_string())


if __name__ == '__main__':
    main()
