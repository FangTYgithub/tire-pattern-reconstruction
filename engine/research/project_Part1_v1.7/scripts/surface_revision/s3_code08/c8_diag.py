# Inspect encoded-pattern failure mechanisms from saved arrays.

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import c8_common as K


def _rows(C: pd.DataFrame) -> pd.DataFrame:
    L = C[C.family == 'lines']
    base = (L.sigma == 0)
    sel = []

    def add(m, tag, rules):
        q = L[m & L.rule.isin(rules)].copy(); q['selection'] = tag; sel.append(q)
    n1 = (L.layout == '5.6') & (L.phase_id == 'N1') & L.surface.isin(['B_curv', 'M10_curv'])
    add(n1 & base & L.pattern.isin(['weak', 'mid', 'strong']), 'A required', ['R@1.1247', 'ignore', 'LB@1.0000'])
    add(n1 & base & (L.pattern == 'plain'), 'B plain same phase', ['P4'])
    add((L.layout == '5.6') & (L.phase_id == 'N1') & (L.surface == 'M24_curv') & base & L.pattern.isin(['weak', 'mid', 'strong']), 'C M24 same condition', ['R@1.1247'])
    add((L.layout == '5.6') & L.phase_id.isin(['N0', 'N2']) & L.surface.isin(['B_curv', 'M10_curv']) & base & (L.pattern == 'strong'), 'D neighbour phases', ['R@1.1247'])
    add(n1 & L.sigma.isin([20.0, 40.0]) & (L.rep == 0) & (L.pattern == 'strong'), 'E noise persistence', ['R@1.1247'])
    flagged = L[base & (L.rule == 'R@1.1247') & (L.flag_wrong_all == 1)].image_key
    add(L.image_key.isin(flagged) & base, 'F sigma-0 confirmed wrong', ['R@1.1247', 'LB@1.0000', 'ignore'])
    X = pd.concat(sel).drop_duplicates(['eval_key'])
    return X


def _nodes_touch(Pv, anomal_xy, dl, d, tau):
    import c6_check6x as X6
    import c7_cases as CS
    from scipy.spatial import Delaunay
    from surface_eval_r3 import DEDUP_TOL_MM
    tri = Delaunay(Pv[:, :2]); L_ = CS._nodes(Pv, tri, dl['x'], dl['y'], dl['z_ref'], d, tau)
    key = lambda A: set(map(tuple, np.round(A / DEDUP_TOL_MM).astype(np.int64).tolist()))
    ak = key(anomal_xy); vk = [tuple(x) for x in np.round(Pv[:, :2] / DEDUP_TOL_MM).astype(np.int64).tolist()]
    is_anom = np.array([v in ak for v in vk], bool)
    s = tri.find_simplex(np.column_stack([dl['x'], dl['y']])); ins = s >= 0
    touch = np.zeros(len(s), bool); touch[ins] = is_anom[tri.simplices[s[ins]]].any(axis=1)
    e = L_['z64'] - np.asarray(dl['z_ref'], float)
    rm = lambda m: (float(np.sqrt(np.nanmean(e[m] ** 2))) if m.any() else None)
    return dict(loc_nodes_interp=int(ins.sum()), loc_nodes_touching_anomaly=int(touch.sum()), loc_rmse_touching=rm(touch), loc_rmse_not_touching=rm(ins & ~touch))


# Extract detection, identity and reconstruction diagnostics from selected cases.
def diagnose() -> pd.DataFrame:
    import c2_decode as CD
    import c3_eval as EV
    import c3_store as ST
    import c4_store as S4
    import c6_check6x as X6
    import c7_cases as CS
    import l01_truth as LT
    import rdy_platform as P
    import s1r_eval as SE
    from scipy.spatial import Delaunay
    from surface_eval_r3 import DEDUP_TOL_MM
    T = K.tables(); C = T['C']; X = _rows(C)
    M = pd.read_csv(K.K4.OUT / 'formal' / 'manifest.csv', low_memory=False); M['layout'] = M.layout.astype(str); Mi = M.set_index('image_key')
    S = P.s2(); dg = P.dom_global(); d, tau = float(K.K4.RK.D_SUPPORT), float(K.K4.RK.TAU); out = []
    for r in X.itertuples():
        row = dict(Mi.loc[r.image_key].to_dict(), image_key=r.image_key)
        E = K.load_npz(K.C7_REC / 'eval' / f'{r.eval_key}.npz'); dj = K.load_json(K.C7_REC / 'dec' / f'{r.decode_key}.json'); A = K.load_npz(K.C7_REC / 'dec' / f'{r.decode_key}.npz')
        D = ST.unpack('lines', A, dj['meta']); feat = np.asarray(D['features'], float); lid = np.asarray(D['line_id']); seg = np.asarray(D['segment_id']); acc = np.asarray(D['accepted'], bool)
        cor, wr, un = np.asarray(E['bridge'], bool); pts = np.asarray(E['pts'], float); zr = np.asarray(E['zr'], float)
        inp = S4.inputs_of(row); ctx = EV.ctx_of(row, inp); mesh = P.mesh(row['surface']); fam = LT._family(ctx['meta_full']['olines'])
        ti = LT.truth_identity(mesh, feat, ctx['meta_full']['olines'])
        q = (ti['proj_uv'] - fam['c']) @ fam['n']; kk = np.round(q / fam['s'] - fam['phase'])
        dec = CD.line_dec(D); V = S.point_validity(dec['pts'], dec['accepted'], mesh); vpts, valid = V['pts'], V['valid']
        gin = valid & SE.K.in_global(vpts[:, :2], dg); lin = valid & SE.K.in_local(vpts[:, :2])
        an = acc & (wr | un); cm = acc & cor
        rec = dict(selection=r.selection, family='lines', surface=r.surface, layout=r.layout, pattern=r.pattern, phase_id=r.phase_id, sigma=r.sigma, rep=int(r.rep), rule=r.rule,
                   image_key=r.image_key, decode_key=r.decode_key, eval_key=r.eval_key, accepted=int(acc.sum()), c=int((acc & cor).sum()), w=int((acc & wr).sum()), u=int((acc & un).sum()),
                   table_c=int(r.c), table_w=int(r.w), table_u=int(r.u), registered_ids=f"{min(fam['ids'])}..{max(fam['ids'])}", decision=D.get('decision'), image_fallback=bool(D.get('image_fallback')),
                   n_anomalous=int(an.sum()), anomalous_segments=int(len(np.unique(seg[an]))) if an.any() else 0,
                   anomalous_decoded_ids=';'.join(str(int(x)) for x in np.unique(lid[an])) if an.any() else '',
                   anomalous_truth_k=';'.join(str(int(x)) for x in np.unique(kk[an][np.isfinite(kk[an])])) if an.any() else '',
                   anomalous_truth_k_registered=bool(np.isin(kk[an], fam['ids']).all()) if an.any() else None,
                   anomalous_img_col=(f'{feat[an, 0].min():.1f}-{feat[an, 0].max():.1f}' if an.any() else ''), anomalous_img_rows=(f'{feat[an, 1].min():.0f}-{feat[an, 1].max():.0f}' if an.any() else ''),
                   anomalous_proj_col=(f'{np.nanmin(ti["proj_uv"][an, 0]):.2f}-{np.nanmax(ti["proj_uv"][an, 0]):.2f}' if an.any() else ''),
                   anomalous_x_mm=(f'{pts[an, 0].min():.1f}..{pts[an, 0].max():.1f}' if an.any() else ''), anomalous_y_mm=(f'{pts[an, 1].min():.1f}..{pts[an, 1].max():.1f}' if an.any() else ''),
                   anomalous_z_mm=(f'{pts[an, 2].min():.1f}..{pts[an, 2].max():.1f}' if an.any() else ''),
                   anomalous_depth_rmse_mm=(float(np.sqrt(np.mean((pts[an, 2] - zr[an]) ** 2))) if an.any() else None),
                   correct_depth_rmse_mm=(float(np.sqrt(np.mean((pts[cm, 2] - zr[cm]) ** 2))) if cm.any() else None),
                   anomalous_in_local=int((an & lin).sum()), anomalous_in_global=int((an & gin).sum()), anomalous_valid=int((an & valid).sum()),
                   glob_Ctau=float(r.glob_Ctau), loc_Ctau=float(r.loc_Ctau), glob_Cvalid=float(r.glob_Cvalid), loc_Cvalid=float(r.loc_Cvalid),
                   glob_surf_rmse=r.glob_surf_rmse, loc_surf_rmse=r.loc_surf_rmse, glob_n_interp=r.glob_n_interp, loc_n_interp=r.loc_n_interp)
        rec['counts_match_table'] = bool(rec['c'] == rec['table_c'] and rec['w'] == rec['table_w'] and rec['u'] == rec['table_u'])
        if r.rule in ('R@1.1247', 'P4') and an.any():
            dl = P.dom_local(row['surface']); Pv = X6._dedup(vpts[valid].astype(np.float64), DEDUP_TOL_MM)
            rec.update(_nodes_touch(Pv, vpts[an & valid][:, :2], dl, d, tau))
            keep = valid & ~an; Pw = X6._dedup(vpts[keep].astype(np.float64), DEDUP_TOL_MM)
            if len(Pw) >= 3:
                tri = Delaunay(Pw[:, :2]); L_ = CS._nodes(Pw, tri, dl['x'], dl['y'], dl['z_ref'], d, tau); G_ = CS._nodes(Pw, tri, dg['x'], dg['y'], dg[f"z_{K.K4.RK.SURF_MAP[row['surface']]}"], d, tau)
                rec.update(diag_loc_surf_rmse_without_anomalous=L_['surf'][1], diag_loc_n_interp_without=L_['n_interp'], diag_glob_surf_rmse_without_anomalous=G_['surf'][1])

        edge = an.any() and np.nanmax(ti['proj_uv'][an, 0]) < 6.0
        if not an.any():
            st = 'no anomalous point (control)'
        elif edge and not rec['anomalous_truth_k_registered']:
            st = 'proven: projector-raster edge remnant (unregistered line index) imaged at the camera border, accepted with the adjacent registered id -> unconfirmable by the registered truth rule'
        elif edge:
            st = 'proven: projector-raster edge partial line (registered index) imaged at the camera border, decoded one id off -> confirmed wrong'
        elif rec['w'] > 0:
            st = 'proven (by ids): whole-line numbering shift (decoded = truth + 1) inside the image; mechanism of the read-out step not traced further'
        else:
            st = 'anomalous points not at the raster edge: cause not determined from stored fields'
        rec['cause_status'] = st
        out.append(rec)
    return pd.DataFrame(out)


def edge_scan() -> pd.DataFrame:
    import c3_eval as EV
    import c4_store as S4
    import l01_truth as LT
    import rdy_platform as P
    M = pd.read_csv(K.K4.OUT / 'formal' / 'manifest.csv', low_memory=False); M['layout'] = M.layout.astype(str)
    g = LT.C.geom(); meshes = {s: P.mesh(s) for s in ('B_curv', 'M10_curv', 'M24_curv')}; rows = []
    for (lay, pat, ph), q in M[(M.family == 'lines') & (M.sigma == 0)].groupby(['layout', 'pattern', 'phase_id']):
        row = dict(q.iloc[0].to_dict()); inp = S4.inputs_of(row); ctx = EV.ctx_of(row, inp); fam = LT._family(ctx['meta_full']['olines'])
        rast = np.asarray(inp['pattern'], float); colsum = rast.sum(axis=0); on = (colsum > 0).astype(int); edges = np.flatnonzero(np.diff(on)) + 1
        runs = [(int(x[0]), int(x[-1])) for x in np.split(np.arange(rast.shape[1]), edges) if on[x[0]]]
        for a, b in runs:
            if a > 0 and b < rast.shape[1] - 1:
                continue
            c_ = (a + b) / 2.0; kc = float((np.array([c_, 255.5]) - fam['c']) @ fam['n'] / fam['s'] - fam['phase']); k = int(np.round(kc))
            r = dict(layout=lay, pattern=pat, phase_id=ph, side=('left' if a == 0 else 'right'), raster_cols=f'{a}-{b}', width_px=b - a + 1, design_width_px=round(float(fam['w']), 2),
                     k_continuous=round(kc, 2), k=k, registered=bool(k in set(fam['ids'])), raster_energy=round(float(colsum[a:b + 1].sum()), 1))
            uvp = np.column_stack([np.full(rast.shape[0], c_), np.arange(float(rast.shape[0]))])
            for s, mm in meshes.items():
                hit, Xw = LT.C.cast(uvp, g['proj'], mm); uvc, ins = LT.C.project(Xw, g['cam'])
                r[f'camera_col_{s}'] = round(float(np.nanmedian(uvc[hit, 0])), 1) if hit.any() else None
            rows.append(r)
    return pd.DataFrame(rows)


def scope(C: pd.DataFrame) -> dict:
    L = C[C.family == 'lines']
    n1 = L[(L.layout == '5.6') & (L.phase_id == 'N1') & L.surface.isin(['B_curv', 'M10_curv'])]
    by = n1.groupby(['sigma']).agg(records=('u', 'size'), records_u_ge_100=('u', lambda x: int((x >= 100).sum())), u_max=('u', 'max'), u_sum=('u', 'sum'))
    e445 = L[(L.layout == '4.45') & (L.phase_id == 'N1') & L.surface.isin(['B_curv', 'M10_curv']) & (L.w >= 400) & (L.sigma <= 20)]
    sig = L[((L.u >= 300) | (L.w >= 300)) & (L.sigma <= 40)]
    return dict(n1_5p6_BM10_by_sigma=by.reset_index().to_dict('records'),
                n1_5p6_BM10_records_with_u_ge_100=int((n1.u >= 100).sum()), n1_5p6_BM10_records=len(n1),
                n1_4p45_BM10_records_w_ge_400_sigma_le_20=e445[['surface', 'pattern', 'rule', 'sigma', 'w']].to_dict('records'),
                m24_n1_5p6_u_max=float(L[(L.layout == '5.6') & (L.phase_id == 'N1') & (L.surface == 'M24_curv')].u.max()),
                lines_sigma_le_40_records_u_or_w_ge_300=int(len(sig)),
                signature_by_condition=sig.groupby(['layout', 'surface', 'phase_id', 'pattern', 'sigma']).agg(rules=('rule', lambda x: ','.join(sorted(set(x)))), w=('w', 'max'), u=('u', 'max')).reset_index().to_dict('records'))


def run() -> dict:
    T = K.tables(); D = diagnose(); K.save_csv(D, K.OUT / 'diagnostic_cases.csv')
    ES = edge_scan(); sc = scope(T['C'])
    req = D[D.selection == 'A required']
    six = req[req.rule == 'R@1.1247'].sort_values(['surface', 'pattern'])
    res = dict(diagnostic_table=dict(file=K.rel(K.OUT / 'diagnostic_cases.csv'), sha256=K.sha_file(K.OUT / 'diagnostic_cases.csv'), rows=len(D), unique_images=int(D.image_key.nunique())),
               counts_match_table=bool(D.counts_match_table.all()), six_required=six[['surface', 'pattern', 'c', 'w', 'u', 'anomalous_segments', 'anomalous_depth_rmse_mm', 'correct_depth_rmse_mm',
                                                                                     'glob_surf_rmse', 'loc_surf_rmse', 'anomalous_in_local', 'cause_status']].to_dict('records'),
               edge_scan=ES.to_dict('records'), scope=sc)
    res['ok'] = bool(res['counts_match_table'] and len(six) == 6 and res['diagnostic_table']['unique_images'] <= 24)
    return res


if __name__ == '__main__':
    r = run()
    if len(sys.argv) > 1:
        K.checks_add(sys.argv[1], r)
    print(json.dumps(K.clean({k: v for k, v in r.items() if k not in ('edge_scan', 'scope')}), indent=1, ensure_ascii=False)[:6000])
