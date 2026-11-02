# Check repeat-aware pairing and metric rules on synthetic tables.

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import c6_common as K

N_COARSE, N_FINE = 24, 400


def _st(e):
    e = np.asarray(e, np.float64); e = e[np.isfinite(e)]
    return (int(e.size), None, None) if e.size == 0 else (int(e.size), float(np.sqrt(np.mean(e * e))), float(np.percentile(np.abs(e), 95)))


def _d(a, b):
    na = a is None or (isinstance(a, float) and np.isnan(a)); nb = b is None or pd.isna(b)
    if na and nb:
        return 0.0
    if na or nb:
        return float('inf')
    return abs(float(a) - float(b))


def _dedup(P: np.ndarray, tol: float) -> np.ndarray:
    key = np.round(P[:, :2] / tol).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return P[np.sort(first)]


def _rho_bounds(V: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:

    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij'); m = (i + j) <= n
    l1 = (i[m] / n)[:, None]; l2 = (j[m] / n)[:, None]; l0 = 1.0 - l1 - l2
    L = np.empty(len(V)); edge = np.max(np.stack([np.linalg.norm(V[:, 1] - V[:, 0], axis=1), np.linalg.norm(V[:, 2] - V[:, 1], axis=1),
                                                    np.linalg.norm(V[:, 2] - V[:, 0], axis=1)], 1), axis=1)
    step = max(1, int(4_000_000 // (len(l0) * 3)))
    for a in range(0, len(V), step):
        W = V[a:a + step]
        X = l0[None] * W[:, None, 0] + l1[None] * W[:, None, 1] + l2[None] * W[:, None, 2]
        f = np.sqrt(((X[:, :, None, :] - W[:, None, :, :]) ** 2).sum(-1)).min(-1)
        L[a:a + step] = f.max(1)
    return L, L + edge / n


def _rho_exact(V: np.ndarray) -> np.ndarray:

    A_, B_, C_ = V[:, 0], V[:, 1], V[:, 2]
    a = np.linalg.norm(B_ - C_, axis=1); b = np.linalg.norm(C_ - A_, axis=1); c = np.linalg.norm(A_ - B_, axis=1)
    F = 0.5 * np.abs((B_[:, 0] - A_[:, 0]) * (C_[:, 1] - A_[:, 1]) - (B_[:, 1] - A_[:, 1]) * (C_[:, 0] - A_[:, 0]))
    S = np.sort(np.stack([a, b, c], 1), axis=1); s1, s2, L = S[:, 0], S[:, 1], S[:, 2]
    rho = np.full(len(V), np.inf); ok = F > 0
    acute = ok & (L * L <= s1 * s1 + s2 * s2)
    rho[acute] = (a * b * c)[acute] / (4.0 * F[acute])
    ob = ok & ~acute
    with np.errstate(divide='ignore', invalid='ignore'):
        cos1 = (s1 * s1 + L * L - s2 * s2) / (2.0 * s1 * L)
        cos2 = (s2 * s2 + L * L - s1 * s1) / (2.0 * s2 * L)
        rho[ob] = np.maximum(s1 / (2.0 * cos1), s2 / (2.0 * cos2))[ob]
    return rho


def _domain(P: np.ndarray, tri, xq, yq, z_ref, d: float, tau: float) -> dict:
    q = np.column_stack([np.asarray(xq, np.float64), np.asarray(yq, np.float64)]); z_ref = np.asarray(z_ref, np.float64); n = len(q)
    s = tri.find_simplex(q); inside = s >= 0; z64 = np.full(n, np.nan)
    if inside.any():
        vx = P[tri.simplices[s[inside]]]
        A = np.stack([vx[:, 1, :2] - vx[:, 0, :2], vx[:, 2, :2] - vx[:, 0, :2]], axis=2)
        lam = np.linalg.solve(A, (q[inside] - vx[:, 0, :2])[:, :, None])[:, :, 0]
        w = np.column_stack([1.0 - lam.sum(1), lam]); z64[inside] = (w * vx[:, :, 2]).sum(1)
    used = np.unique(s[inside]); V = P[tri.simplices[used]][:, :, :2]
    rho = _rho_exact(V); L, U = _rho_bounds(V, N_COARSE); fin = np.isfinite(rho)
    amb = fin & ~((rho >= L - 1e-9) & (rho <= U + 1e-9))
    if amb.any():
        L2, U2 = _rho_bounds(V[amb], N_FINE); L[amb], U[amb] = L2, U2
    bound_violations = int((fin & ~((rho >= L - 1e-9) & (rho <= U + 1e-9))).sum())
    sup_t = fin & (rho <= d); undecided_t = fin & (np.abs(rho - d) < 1e-9)
    tri_sup = np.zeros(len(tri.simplices), bool); tri_sup[used[sup_t]] = True
    tri_und = np.zeros(len(tri.simplices), bool); tri_und[used[undecided_t]] = True
    e64 = z64 - z_ref; interp = inside & np.isfinite(e64)
    sup = interp & tri_sup[np.where(inside, s, 0)]; und = interp & tri_und[np.where(inside, s, 0)]
    e32 = np.asarray(z64.astype(np.float32), np.float64) - z_ref
    return dict(n_nodes=n, n_interp=int(interp.sum()), Cvalid=float(sup.mean()), Ctau=float((sup & (np.abs(np.nan_to_num(e64, nan=1e9)) <= tau)).mean()),
                surf32=_st(e32[interp]), surf64=_st(e64[interp]), undecided_nodes=int(und.sum()), refined_triangles=int(amb.sum()), bound_violations=bound_violations,
                min_rho_margin=float(np.min(np.abs(rho[fin] - d))) if fin.any() else None,
                min_tau_margin=float(np.min(np.abs(np.abs(e64[sup]) - tau))) if sup.any() else None)


# Recompute selected coverage and error values from saved reconstruction arrays.
def run() -> dict:
    import c2_decode as CD
    import c3_eval as EV
    import c3_store as ST
    import c4_store as S4
    import rdy_platform as P
    import s1r_eval as SE
    from scipy.spatial import Delaunay
    from surface_eval_r3 import DEDUP_TOL_MM
    cfg = K.load_json(K.CONFIGS / 'b_entry.json'); out = K.ROOT / cfg['output'] / 'current'; rec = K.ROOT / cfg['data_root']
    C = pd.read_csv(out / 'cases.csv', low_memory=False); C['layout'] = C.layout.astype(str)
    M = pd.read_csv(K.ROOT / cfg['inputs']['manifest']['file'], low_memory=False); M['layout'] = M.layout.astype(str); Mi = M.set_index('image_key')
    S = P.s2(); dg = P.dom_global(); d, tau = float(K.K4.RK.D_SUPPORT), float(K.K4.RK.TAU); rows = []
    for rr in cfg['cases']:
        for sg in rr['sigmas']:
            rep = 0 if float(sg) == 0 else rr['rep']
            q = C[(C.family == rr['family']) & (C.layout == rr['layout']) & (C.pattern == rr['pattern']) & (C.surface == rr['surface']) & (C.phase_id == rr['phase_id']) & (C.rep == rep)
                  & (C.rule == rr['rule']) & (C.arm == rr['arm']) & (C.sigma == sg)]
            if len(q) != 1:
                rows.append(dict(family=rr['family'], sigma=sg, rep=rep, status=f'{len(q)} rows for the registered case')); continue
            c = q.iloc[0]; row = dict(Mi.loc[c.image_key].to_dict(), image_key=c.image_key)
            dj = K.load_json(rec / 'dec' / f'{c.decode_key}.json'); A = K.load_npz(rec / 'dec' / f'{c.decode_key}.npz')
            mesh = P.mesh(row['surface']); sid = K.K4.RK.SURF_MAP[row['surface']]; dl = P.dom_local(row['surface'])
            D = ST.unpack(row['family'], A, dj['meta']); dec = CD.dot_dec(D, rr['arm']) if row['family'] == 'dots' else CD.line_dec(D)
            V = S.point_validity(dec['pts'], dec['accepted'], mesh); pts, valid, zr = V['pts'], V['valid'], V['zr']; ep = pts[:, 2] - zr
            gin = valid & SE.K.in_global(pts[:, :2], dg); lin = valid & SE.K.in_local(pts[:, :2])
            diffs = {}; info = {}
            for nm, sel in (('glob', gin), ('loc', lin)):
                h = _st(ep[sel]); diffs[f'{nm}_n_valid'] = _d(h[0], c.get(f'{nm}_n_valid')); diffs[f'{nm}_pt_rmse'] = _d(h[1], c.get(f'{nm}_pt_rmse')); diffs[f'{nm}_pt_p95'] = _d(h[2], c.get(f'{nm}_pt_p95'))
            Pv = _dedup(pts[valid].astype(np.float64), DEDUP_TOL_MM)
            tri = Delaunay(Pv[:, :2]) if len(Pv) >= 3 else None
            for nm, (xq, yq, zref) in (('glob', (dg['x'], dg['y'], dg[f'z_{sid}'])), ('loc', (dl['x'], dl['y'], dl['z_ref']))):
                if tri is None:
                    r = dict(n_nodes=len(xq), n_interp=0, Cvalid=0.0, Ctau=0.0, surf32=(0, None, None), surf64=(0, None, None), undecided_nodes=0, refined_triangles=0, bound_violations=0, min_rho_margin=None, min_tau_margin=None)
                else:
                    r = _domain(Pv, tri, xq, yq, zref, d, tau)
                diffs[f'{nm}_n_interp'] = _d(r['n_interp'], c.get(f'{nm}_n_interp')); diffs[f'{nm}_Cvalid'] = _d(r['Cvalid'], c.get(f'{nm}_Cvalid'))
                diffs[f'{nm}_Ctau'] = _d(r['Ctau'], c.get(f'{nm}_Ctau'))
                diffs[f'{nm}_surf_rmse'] = _d(r['surf32'][1], c.get(f'{nm}_surf_rmse')); diffs[f'{nm}_surf_p95'] = _d(r['surf32'][2], c.get(f'{nm}_surf_p95'))
                info[nm] = dict(undecided_nodes=r['undecided_nodes'], refined_triangles=r['refined_triangles'], bound_violations=r['bound_violations'], min_rho_margin=r['min_rho_margin'],
                                min_tau_margin=r['min_tau_margin'],
                                surf_rmse_f64_minus_stored=_d(r['surf64'][1], c.get(f'{nm}_surf_rmse')))
            no = int(c.n_accepted) == 0
            out_r = dict(family=rr['family'], sigma=sg, rep=rep, phase_id=rr['phase_id'], image_key=c.image_key, eval_key=c.eval_key, n_accepted=int(c.n_accepted), n_valid=int(c.n_valid),
                         glob_Cvalid=float(c.glob_Cvalid), glob_Ctau=float(c.glob_Ctau), loc_Cvalid=float(c.loc_Cvalid), loc_Ctau=float(c.loc_Ctau), max_diff=max(diffs.values()),
                         worst=max(diffs, key=diffs.get), undecided_nodes=info['glob']['undecided_nodes'] + info['loc']['undecided_nodes'],
                         refined_triangles=info['glob']['refined_triangles'] + info['loc']['refined_triangles'],
                         bound_violations=info['glob']['bound_violations'] + info['loc']['bound_violations'], glob_min_rho_margin=info['glob']['min_rho_margin'], glob_min_tau_margin=info['glob']['min_tau_margin'],
                         loc_min_tau_margin=info['loc']['min_tau_margin'], glob_surf_rmse_f64_diff=info['glob']['surf_rmse_f64_minus_stored'],
                         loc_surf_rmse_f64_diff=info['loc']['surf_rmse_f64_minus_stored'], scientific_no_output=no)
            out_r['status'] = 'ok' if (out_r['max_diff'] <= 1e-9 and out_r['undecided_nodes'] == 0 and out_r['bound_violations'] == 0) else 'differs'
            rows.append(out_r)
    T = pd.DataFrame(rows); K.save_csv(T, K.OUT / 'results' / 'B_check6_f64.csv')
    return dict(ok=bool(len(T) == 6 and (T.status == 'ok').all()), n=len(T), tolerance='1e-9 (counts / coverage exact to 1e-12 in practice)', table=K.rel(K.OUT / 'results' / 'B_check6_f64.csv'),
                script=dict(file=K.rel(K.HERE / 'c6_check6x.py'), sha256=K.sha_file(K.HERE / 'c6_check6x.py'), in_frozen_package=False), cases=T.to_dict('records'))


if __name__ == '__main__':
    import json
    r = run()
    if len(sys.argv) > 1:
        K.checks_add(sys.argv[1], r)
    print(json.dumps(K.clean({k: v for k, v in r.items() if k != 'cases'}), indent=1)); print(pd.DataFrame(r['cases']).drop(columns=['image_key', 'eval_key']).to_string())
