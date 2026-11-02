# Select representative encoded-pattern and reconstruction examples.

from __future__ import annotations

import numpy as np
import pandas as pd

import c6_check6x as X6
import c7_common as K

POSTHOC = ('risk', 'collapse')
MAIN = dict(dots=('DF', 'use'), lines=('R@1.1247', 'R@1.1247'))


def _state(n_acc: int, ct: float, ref_ct: float | None) -> str:
    if n_acc == 0:
        return 'no_output'
    return 'degraded' if (ref_ct is not None and ct < 0.5 * ref_ct) else 'normal'


def _rec(r, extra) -> dict:
    return dict(extra, phase_id=r.phase_id, rep=int(r.rep), sigma=float(r.sigma), family=r.family, surface=r.surface, layout=str(r.layout), pattern=r.pattern, rule=r.rule, arm=r.arm,
                image_key=r.image_key, eval_key=r.eval_key, decode_key=r.decode_key, n_accepted=int(r.n_accepted), n_valid=int(r.n_valid), Ctau=float(r.glob_Ctau),
                Cvalid=float(r.glob_Cvalid), rmse=(None if pd.isna(r.glob_pt_rmse) else float(r.glob_pt_rmse)),
                surf_rmse=(None if pd.isna(r.get('glob_surf_rmse')) else float(r.get('glob_surf_rmse'))), n_interp=(None if pd.isna(r.get('glob_n_interp')) else int(r.get('glob_n_interp'))),
                c=int(r.c), w=int(r.w), u=int(r.u))


def select(C: pd.DataFrame, cfg: dict) -> list:
    out = []; C = C.copy(); C['layout'] = C.layout.astype(str)
    for i, rr in enumerate(cfg.get('cases', [])):
        ref_ct = None
        for sg in rr['sigmas']:
            rep = 0 if float(sg) == 0 else int(rr['rep'])
            q = C[(C.family == rr['family']) & (C.layout == rr['layout']) & (C.pattern == rr['pattern']) & (C.surface == rr['surface']) & (C.phase_id == rr['phase_id'])
                  & (C.rep == rep) & (C.rule == rr['rule']) & (C.arm == rr['arm']) & (C.sigma == float(sg))]
            if len(q) > 1:
                raise K.Stop('technical', f'case rule {rr} sigma {sg} is ambiguous ({len(q)} records)')
            base = dict(row=i, group=rr.get('group', 'fixed'), rule_def=rr)
            if not len(q):
                out.append(dict(base, sigma=float(sg), rep=rep, state='not_found')); continue
            r = q.iloc[0]
            if ref_ct is None:
                ref_ct = float(r.glob_Ctau)
            out.append(_rec(r, dict(base, state=_state(int(r.n_accepted), float(r.glob_Ctau), ref_ct))))
    for rule in (cfg.get('posthoc') or []):
        out.append(posthoc(C, rule))
    return out


def posthoc(C: pd.DataFrame, rule: str) -> dict:
    base = dict(row=None, group='posthoc', posthoc=rule, rule_def=dict(rule=rule))
    m = pd.Series(False, index=C.index)
    for fam, (ru, ar) in MAIN.items():
        m |= (C.family == fam) & (C.rule == ru) & (C.arm == ar)
    X = C[m & (C.pattern != 'plain')]
    if rule == 'risk':
        X = X[(X.acc >= 20) & (X.w > 0)].copy()
        if not len(X):
            return dict(base, state='not_found')
        X['share'] = X.w / (X.c + X.w); r = X.sort_values(['share', 'w', 'eval_key'], ascending=[False, False, True]).iloc[0]
    elif rule == 'collapse':
        z = X[X.sigma == 0].set_index(['family', 'surface', 'layout', 'pattern', 'phase_id', 'rule', 'arm']).glob_Ctau
        X = X[X.sigma == 20].copy()
        X['drop'] = [z.get((a.family, a.surface, a.layout, a.pattern, a.phase_id, a.rule, a.arm), np.nan) - a.glob_Ctau for a in X.itertuples()]
        X = X[X['drop'].notna()]
        if not len(X):
            return dict(base, state='not_found')
        r = X.sort_values(['drop', 'eval_key'], ascending=[False, True]).iloc[0]
    else:
        raise K.Stop('config', f'unknown post-hoc rule {rule}')
    return _rec(r, dict(base, state=_state(int(r.n_accepted), float(r.glob_Ctau), None)))


def _nodes(Pv: np.ndarray, tri, xq, yq, zref, d: float, tau: float) -> dict:
    q = np.column_stack([np.asarray(xq, np.float64), np.asarray(yq, np.float64)]); zref = np.asarray(zref, np.float64); n = len(q)
    s = tri.find_simplex(q) if tri is not None else np.full(n, -1); inside = s >= 0; z64 = np.full(n, np.nan)
    state = np.zeros(n, np.int8)
    if tri is None or not inside.any():
        return dict(n_nodes=n, n_interp=0, Cvalid=0.0, Ctau=0.0, surf=(0, None, None), state=state, z64=z64, undecided=0)
    vx = Pv[tri.simplices[s[inside]]]
    A = np.stack([vx[:, 1, :2] - vx[:, 0, :2], vx[:, 2, :2] - vx[:, 0, :2]], axis=2)
    lam = np.linalg.solve(A, (q[inside] - vx[:, 0, :2])[:, :, None])[:, :, 0]
    w = np.column_stack([1.0 - lam.sum(1), lam]); z64[inside] = (w * vx[:, :, 2]).sum(1)
    rho = X6._rho_exact(Pv[tri.simplices][:, :, :2]); fin = np.isfinite(rho)
    tri_sup = fin & (rho <= d); und_t = fin & (np.abs(rho - d) < 1e-9)
    e64 = z64 - zref; interp = inside & np.isfinite(e64); si = np.where(inside, s, 0)
    sup = interp & tri_sup[si]; acc = sup & (np.abs(np.nan_to_num(e64, nan=1e9)) <= tau)
    state[sup] = 1; state[acc] = 2
    e32 = np.asarray(z64.astype(np.float32), np.float64) - zref
    return dict(n_nodes=n, n_interp=int(interp.sum()), Cvalid=float(sup.mean()), Ctau=float(acc.mean()), surf=X6._st(e32[interp]), state=state, z64=z64,
                undecided=int((interp & und_t[si]).sum()), tau_margin=(float(np.min(np.abs(np.abs(e64[sup]) - tau))) if sup.any() else None))


def recompute(case: dict, rec_root, M: pd.DataFrame) -> dict:

    import c2_decode as CD
    import c3_store as ST
    import rdy_platform as P
    import s1r_eval as SE
    from scipy.spatial import Delaunay
    from surface_eval_r3 import DEDUP_TOL_MM
    rec_root = K.abspath(rec_root)
    row = dict(M.drop_duplicates('image_key').set_index('image_key').loc[case['image_key']].to_dict(), image_key=case['image_key'])
    dp = rec_root / 'dec' / f"{case['decode_key']}.json"; dj = K.load_json(dp)
    if dj.get('kind') == 'ref':
        A = K.load_npz(K.ROOT / dj['source_npz']); dj = dict(dj, meta=K.load_json(K.ROOT / dj['source_json'])['meta'])
    else:
        A = K.load_npz(dp.with_suffix('.npz'))
    S = P.s2(); dg = P.dom_global(); mesh = P.mesh(row['surface']); sid = K.K4.RK.SURF_MAP[row['surface']]; dl = P.dom_local(row['surface'])
    D = ST.unpack(row['family'], A, dj['meta']); dec = CD.dot_dec(D, case['arm']) if row['family'] == 'dots' else CD.line_dec(D)
    V = S.point_validity(dec['pts'], dec['accepted'], mesh); pts, valid, zr = V['pts'], V['valid'], V['zr']; ep = pts[:, 2] - zr
    gin = valid & SE.K.in_global(pts[:, :2], dg); lin = valid & SE.K.in_local(pts[:, :2])
    Pv = X6._dedup(pts[valid].astype(np.float64), DEDUP_TOL_MM); tri = Delaunay(Pv[:, :2]) if len(Pv) >= 3 else None
    d, tau = float(K.K4.RK.D_SUPPORT), float(K.K4.RK.TAU)
    g = _nodes(Pv, tri, dg['x'], dg['y'], dg[f'z_{sid}'], d, tau); l_ = _nodes(Pv, tri, dl['x'], dl['y'], dl['z_ref'], d, tau)
    hp = dict(glob=X6._st(ep[gin]), loc=X6._st(ep[lin]))
    met = {}
    for nm, h, nd in (('glob', hp['glob'], g), ('loc', hp['loc'], l_)):
        met.update({f'{nm}_n_valid': h[0], f'{nm}_pt_rmse': h[1], f'{nm}_pt_p95': h[2], f'{nm}_n_interp': nd['n_interp'], f'{nm}_n_nodes': nd['n_nodes'], f'{nm}_Cvalid': nd['Cvalid'],
                    f'{nm}_Ctau': nd['Ctau'], f'{nm}_surf_rmse': nd['surf'][1], f'{nm}_surf_p95': nd['surf'][2]})
    met['n_accepted'] = int(np.asarray(dec['accepted'], bool).sum()); met['n_valid'] = int(valid.sum())
    return dict(metrics=met, undecided_nodes=g['undecided'] + l_['undecided'], tau_margin_glob=g.get('tau_margin'), arrays=dict(pts=pts, valid=valid, acc=np.asarray(dec['accepted'], bool),
                state_glob=g['state'], z64_glob=g['z64'], sid=sid))


def compare(met: dict, C_row: pd.Series) -> dict:
    diffs = {f: X6._d(v, C_row.get(f)) for f, v in met.items() if f in C_row.index}
    return dict(max_diff=max(diffs.values()) if diffs else float('inf'), worst=(max(diffs, key=diffs.get) if diffs else None), n_fields=len(diffs), diffs=diffs)
