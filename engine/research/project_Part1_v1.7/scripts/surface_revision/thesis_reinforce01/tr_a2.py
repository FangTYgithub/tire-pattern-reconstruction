# Predict candidate ambiguity from surface geometry and the depth search window.

from __future__ import annotations

import hashlib
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import tr_common as K

S3_MANIFEST = K.ROOT / 'outputs/surface_revision/s3_code04/formal/manifest.csv'
S3_KEYS = K.ROOT / 'outputs/surface_revision/s3_code07/plan/expected_keys_formal.csv'
S3_REC = K.ROOT / 'outputs/surface_revision/s3_code07/formal_run/records'
_C: dict = {}


def _s2():
    import rdy_platform as P
    return P.s2()


def _mesh(surface):
    import rdy_platform as P
    if ('mesh', surface) not in _C:
        _C[('mesh', surface)] = P.mesh(surface)
    return _C[('mesh', surface)]


def _h(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes())
    return h.hexdigest()[:16]


def _bucket(n: np.ndarray) -> np.ndarray:
    return np.where(n == 0, '0', np.where(n == 1, '1', np.where(n == 2, '2', '>2')))


def dot_cfg(kind: str, radii=None):
    import m4_reconstruction as MR
    if kind == 'binding':
        return MR._resolve_sensor_cfg(_s2().dot_cfg())
    import ep_detect as ED
    import rdy_decode as DC
    import ep_patterns as EP
    base = dict(ED.CFG) if radii is None or len(radii) == 1 else ED.coded_cfg(list(radii), len(radii), EP.mmpp())
    return MR._resolve_sensor_cfg(DC.window_cfg_dots(base))


def dot_geometry(binary, centres, classes, surface, cfg) -> dict:
    import m4_reconstruction as MR
    S = _s2(); mesh = _mesh(surface); c = np.asarray(centres, np.float64)
    depth = float(cfg.get('measurement_volume_mm')); z0, z1 = MR.search_window(cfg, depth)
    proj_pts, ids = MR.pattern_features(np.asarray(binary, np.float32), 'dots')
    A, B = MR.projector_tracks(proj_pts, cfg, z0, z1)
    tol, atol, tm = float(cfg.get('epi_tol_px', MR.EPI_TOL_PX)), float(cfg.get('epi_ambig_tol_px', MR.EPI_AMBIG_TOL_PX)), float(cfg.get('epi_t_margin', MR.EPI_T_MARGIN))
    exp = S.SL.expected_dots(mesh, c); uv = exp['uv']; hit = exp['hit']
    in_cam = hit & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= -0.5) & (uv[:, 0] <= 511.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= 511.5)
    zt = np.full(len(c), np.nan)
    if hit.any():
        zt[hit] = S.truth_interp(mesh)(np.column_stack([exp['xy'][hit, 1], exp['xy'][hit, 0]]))
    g = S.C0.geom(); occl = np.zeros(len(c), bool)
    if in_cam.any():
        h2, X = S.C0.cast(uv[in_cam], g['cam'], mesh)
        xyz = np.column_stack([exp['xy'][in_cam], zt[in_cam]])
        occl[np.flatnonzero(in_cam)] = ~np.asarray(h2, bool) | (np.linalg.norm(np.asarray(X) - xyz, axis=1) > 0.05)
    pdot, _ = S.RD.projector_dot_index(np.asarray(binary, np.float32), c)
    use = in_cam & ~occl; n_geo = np.full(len(c), -1); true_in = np.zeros(len(c), bool); n_cls = np.full(len(c), -1)
    if use.any():
        within = MR._tracks_within(uv[use], A, B, max(tol, atol), tm)
        for i, (js, _t) in zip(np.flatnonzero(use), within):
            js = np.asarray(js, np.int64); n_geo[i] = len(js); true_in[i] = bool((pdot[js] == i).any()) if len(js) else False
            if classes is not None:
                n_cls[i] = int((np.asarray(classes)[pdot[js]] == classes[i]).sum()) if len(js) else 0
    return dict(in_cam=in_cam, occl=occl, z_true=zt, hit=hit, n_geo=n_geo, true_in=true_in, n_cls=n_cls, window=(z0, z1), tol=(tol, atol, tm), n_design=len(c))


def line_geometry(ref_u, ref_v, ref_k, binary, olines, cfg) -> dict:
    import m4_lines as ML
    fam = ML.line_family(olines); cl = ML.raster_centrelines(np.asarray(binary, np.float32), fam)
    cand = ML.line_candidates(np.column_stack([ref_u, ref_v]), fam, cfg, centrelines=cl)
    adm = cand['admissible']; ids = [int(k) for k in cand['ids']]; col = {k: j for j, k in enumerate(ids)}
    jt = np.array([col.get(int(k), -1) for k in ref_k]); n = adm.sum(axis=1)
    true_adm = np.array([bool(adm[i, j]) if j >= 0 else False for i, j in enumerate(jt)])
    chain = np.unique(np.asarray(ref_k), return_inverse=True)[1]
    p = ML.default_params(fam, cfg)
    lid, st, _, table = ML.number_segments(cand, chain, p['vote_frac'])
    seg_ok = (st == 0) & (lid == np.asarray(ref_k)); seg_wrong = (st == 0) & (lid != np.asarray(ref_k)); seg_unknown = st == 2; seg_none = st == 1
    return dict(n_geo=n, true_in=true_adm, seg_ok=seg_ok, seg_wrong=seg_wrong, seg_unknown=seg_unknown, seg_none=seg_none,
                window=(float(cand['window_z0_mm']), float(cand['window_z1_mm'])), margin=float(cand['depth_margin_mm']), n_lines=len(ids),
                n_segments=len(table), seg_numbered=sum(1 for s in table if s['decision'] == 'numbered'), seg_unknown_n=sum(1 for s in table if s['decision'] == 'unknown'))


def _summ(unit: dict, G: dict, status: np.ndarray, sel: np.ndarray, extra: dict) -> dict:
    n = G['n_geo']; b = _bucket(np.where(n < 0, 0, n))
    out = dict(unit, n_expected=int(len(sel)), **extra)
    for bb in ('0', '1', '2', '>2'):
        m = sel & (b == bb)
        out[f'cand_{bb}'] = int(m.sum())
        for s in ('correct', 'other', 'missing'):
            out[f'cand_{bb}_{s}'] = int((m & (status == s)).sum())
    out['true_candidate_present'] = int((sel & G['true_in']).sum()); out['true_candidate_absent'] = int((sel & ~G['true_in']).sum())
    out['window_z0'], out['window_z1'] = G['window']
    return out


def _zq(z):
    z = np.asarray(z, float); z = z[np.isfinite(z)]
    return dict(z_p05=float(np.percentile(z, 5)) if len(z) else None, z_med=float(np.median(z)) if len(z) else None, z_p95=float(np.percentile(z, 95)) if len(z) else None)


def work_s12_dot(row: dict) -> dict:
    try:
        import s12_case as SC12
        R = SC12.raster_of(row); c = np.asarray(R['centres'], np.float64)
        key = ('dotgeo', row['surface'], _h(c, R['binary'].shape))
        if key not in _C:
            _C[key] = dot_geometry(R['binary'], c, None, row['surface'], dot_cfg('binding'))
        G = _C[key]
        z = K.load_npz(K.ROOT / row['eval_record'].replace('.json', '.npz'))
        acc, val = z['accepted'].astype(bool), z['valid'].astype(bool)
        pf = z['point_feature'].astype(np.int64); ftd = z['feat_truth_dot'].astype(np.int64)
        ptd = np.where(pf >= 0, ftd[np.clip(pf, 0, None)], -1)
        good = acc & val
        status = np.full(len(c), 'missing', object)
        wr = np.zeros(len(c), bool); wr[ptd[good & z['point_wrong'].astype(bool) & (ptd >= 0)]] = True
        det = np.zeros(len(c), bool); det[ftd[ftd >= 0]] = True
        status[det] = 'other'; status[wr] = 'other'; status[z['exp_recovered'].astype(bool)] = 'correct'
        sel = G['in_cam'] & ~G['occl']
        unit = dict(source='S12', family='dots', section=row['section'], surface=row['surface'], layout=row['layout'], size_mm=row['size_mm'], phase_label=row['phase_label'],
                    arm='D0', input_key=row['input_key'], geometry=key[2])
        extra = dict(n_design=G['n_design'], no_hit=int((~G['hit']).sum()), out_of_camera=int((G['hit'] & ~G['in_cam']).sum()), occluded=int(G['occl'].sum()),
                     unlinked_accepted=int((good & z['point_unconf'].astype(bool)).sum()), **_zq(G['z_true'][sel]))
        return _summ(unit, G, status, sel, extra)
    except Exception as e:
        return dict(source='S12', family='dots', input_key=row.get('input_key'), error=repr(e)[:300])


def work_s12_line(row: dict) -> dict:
    try:
        import s12_case as SC12
        R = SC12.raster_of(row); o = R['meta_full']['olines']
        z = K.load_npz(K.ROOT / row['eval_record'].replace('.json', '.npz'))
        import m4_lines as ML
        cfg = ML._resolve_sensor_cfg(_s2().line_cfg(float(row['peak_f32'])))
        G = line_geometry(z['ref_u'], z['ref_v'], z['ref_k'], R['binary'], o, cfg)
        status = np.where(z['ref_rec_id'].astype(bool), 'correct', np.where(z['ref_rec_any'].astype(bool), 'other', 'missing')).astype(object)
        sel = np.ones(len(status), bool); acc, val = z['accepted'].astype(bool), z['valid'].astype(bool)
        unit = dict(source='S12', family='lines', section=row['section'], surface=row['surface'], layout=row['layout'], size_mm=row['size_mm'], phase_label=row['phase_label'],
                    arm='P4', input_key=row['input_key'], geometry=_h(z['ref_u'], z['ref_v'], z['ref_k']))
        extra = dict(unlinked_accepted=int((acc & val & z['point_unconf'].astype(bool)).sum()), seg_rows_ok=int(G['seg_ok'].sum()), seg_rows_wrong=int(G['seg_wrong'].sum()),
                     seg_rows_unknown=int(G['seg_unknown'].sum()), seg_rows_none=int(G['seg_none'].sum()), n_segments=G['n_segments'], seg_numbered=G['seg_numbered'],
                     seg_unknown=G['seg_unknown_n'], **_zq(z['ref_xyz'][:, 2]))
        return _summ(unit, G, status, sel, extra)
    except Exception as e:
        return dict(source='S12', family='lines', input_key=row.get('input_key'), error=repr(e)[:300])


def _s3_raster(row):
    import sys
    q = str(K.SR / 's3_code04')
    if q not in sys.path:
        sys.path.append(q)
    import c3_raster as CR
    import c4_parity as PA
    fam = 'dots' if row['family'] == 'dots' else 'lines'
    ph = dict(u=row['u'], v=row['v']) if fam == 'dots' else dict(phase_frac=float(row['phase_frac']))
    R = PA.build(fam, str(row['layout']), row['pattern'], ph, CR.levels(), int(row['code_parity']))
    if CR.raster_key(R['binary']) != row['raster_key']:
        raise K.Stop('technical', f"S3 raster {row['raster_key']} not reproduced")
    return R


def work_s3_dot(row: dict) -> dict:
    try:
        R = _s3_raster(row); c = np.asarray(R['centres'], np.float64); radii = R['radii']; cls = None if row['pattern'] == 'plain' else np.asarray(R['classes'])
        G = dot_geometry(R['binary'], c, cls, row['surface'], dot_cfg('enc', radii))
        S = _s2(); mesh = _mesh(row['surface']); g = S.C0.geom(); sel = G['in_cam'] & ~G['occl']
        res = []
        for arm, ek, dk in row['arms']:
            z = K.load_npz(S3_REC / 'eval' / f'{ek}.npz'); d = K.load_npz(S3_REC / 'dec' / f'{dk}.npz')
            acc, val = z['acc'].astype(bool), z['valid'].astype(bool); cor, wr, un = (z['bridge'][i].astype(bool) for i in range(3))
            a = 'a1' if arm.endswith('ignore') else 'a0'
            if len(d[f'{a}_pts']) != len(acc):
                raise K.Stop('technical', f'{arm}: stored arm points {len(d[a + "_pts"])} != evaluated points {len(acc)}')
            pidx = np.asarray(d[f'{a}_tr_point_idx'], np.int64); has = pidx >= 0
            f_of_p = np.full(len(acc), -1, np.int64); f_of_p[pidx[has]] = np.flatnonzero(has)
            feats = np.full((len(acc), 2), np.nan); ok_f = f_of_p >= 0; feats[ok_f] = np.asarray(d['feats'], np.float64).reshape(-1, 2)[f_of_p[ok_f]]
            truth = np.full(len(feats), -1)
            m = acc & val & (cor | wr) & ok_f
            if m.any():
                h, Pp = S.C0.cast(feats[m], g['cam'], mesh); uvp, _ = S.C0.project(Pp, g['proj'])
                dd = np.hypot(uvp[:, None, 0] - c[None, :, 0], uvp[:, None, 1] - c[None, :, 1]); truth[m] = dd.argmin(axis=1)
            status = np.full(len(c), 'missing', object)
            status[truth[m & wr]] = 'other'; status[truth[m & cor]] = 'correct'
            unit = dict(source='S3', family='dots', section='S3', surface=row['surface'], layout=row['layout'], size_mm=row['pattern'], phase_label=row['phase_id'], arm=arm,
                        input_key=row['image_key'], geometry=_h(c))
            extra = dict(n_design=G['n_design'], no_hit=int((~G['hit']).sum()), out_of_camera=int((G['hit'] & ~G['in_cam']).sum()), occluded=int(G['occl'].sum()),
                         unlinked_accepted=int((acc & val & un).sum()), **_zq(G['z_true'][sel]))
            if cls is not None:
                nc = G['n_cls']
                extra.update({f'cls_cand_{b}': int((sel & (_bucket(np.where(nc < 0, 0, nc)) == b)).sum()) for b in ('0', '1', '2', '>2')})
            res.append(_summ(unit, G, status, sel, extra))
        return res
    except Exception as e:
        return [dict(source='S3', family='dots', input_key=row.get('image_key'), error=repr(e)[:300])]


def work_s3_line(row: dict) -> list:
    try:
        import m4_lines as ML
        import l01_truth as T
        R = _s3_raster(row); o = R['meta_full']['olines']; mesh = _mesh(row['surface']); S = _s2()
        ref = T.line_reference(mesh, o, 1.0, pattern=np.asarray(R['binary'], np.float32)); Rr = ref['rows']
        cfg = ML._resolve_sensor_cfg(S.line_cfg(float(row['peak_f32'])))
        G = line_geometry(Rr['u'], Rr['v'], Rr['k'], R['binary'], o, cfg)
        res = []
        for arm, ek, dk in row['arms']:
            z = K.load_npz(S3_REC / 'eval' / f'{ek}.npz'); d = K.load_npz(S3_REC / 'dec' / f'{dk}.npz')
            feats = np.asarray(d['features'], np.float64).reshape(-1, 2); acc, val = z['acc'].astype(bool), z['valid'].astype(bool); lid = z['lid'].astype(np.int64)
            rec, ident = S._row_flags(ref, feats, lid, acc & val)
            status = np.where(ident, 'correct', np.where(rec, 'other', 'missing')).astype(object)
            unit = dict(source='S3', family='lines', section='S3', surface=row['surface'], layout=str(row['layout']), size_mm=row['pattern'], phase_label=row['phase_id'], arm=arm,
                        input_key=row['image_key'], geometry=_h(Rr['u'], Rr['v'], Rr['k']))
            extra = dict(unlinked_accepted=int((acc & val & z['bridge'][2].astype(bool)).sum()), seg_rows_ok=int(G['seg_ok'].sum()), seg_rows_wrong=int(G['seg_wrong'].sum()),
                         seg_rows_unknown=int(G['seg_unknown'].sum()), seg_rows_none=int(G['seg_none'].sum()), n_segments=G['n_segments'], seg_numbered=G['seg_numbered'],
                         seg_unknown=G['seg_unknown_n'], **_zq(Rr['xyz'][:, 2]))
            res.append(_summ(unit, G, status, np.ones(len(status), bool), extra))
        return res
    except Exception as e:
        return [dict(source='S3', family='lines', input_key=row.get('image_key'), error=repr(e)[:300])]


def plan():
    V = K.read_csv(K.S12 / 'results' / 'views.csv')
    V = V[(V.surface_class == 'main') & (V.regime == 'resource_cap_11695') & (V.sigma == 0.0)].drop_duplicates('input_key')
    d12 = V[V.family == 'point'].to_dict('records'); l12 = V[V.family == 'line'].to_dict('records')
    M = K.read_csv(S3_MANIFEST); M = M[M.sigma == 0.0]
    E = K.read_csv(S3_KEYS); E = E[E.layer == 'eval']
    want = {('dots', 'plain'): [('DF', 'single')], ('dots', 'coded'): [('DF', 'ignore'), ('DF', 'use')],
            ('lines', 'plain'): [('P4', 'P4')], ('lines', 'coded'): [('ignore', 'ignore'), ('R@1.1247', 'R@1.1247')]}
    rows3 = {'dots': [], 'lines': []}
    for r in M.to_dict('records'):
        kind = 'plain' if r['pattern'] == 'plain' else 'coded'
        e = E[E.image_key == r['image_key']]; arms = []
        for rule, arm in want[(r['family'], kind)]:
            x = e[(e.rule == rule) & (e.arm == arm)]
            if len(x) != 1:
                raise K.Stop('refused', f"S3 {r['image_key']} {rule}/{arm}: {len(x)} eval keys")
            arms.append((f'{rule}|{arm}', x.iloc[0]['key'], x.iloc[0]['decode_key']))
        rows3[r['family']].append(dict(r, arms=arms))
    return d12, l12, rows3['dots'], rows3['lines']


def node_z() -> list:
    import rdy_platform as P
    import s1r_common as S1K
    dom = P.dom_global(); out = []
    for s in ('B_curv', 'M10_curv', 'M24_curv'):
        sid = K.RK.SURF_MAP[s]; zg = np.asarray(dom[f'z_{sid}'], float)
        loc = P.dom_local(s); zl = np.asarray(loc.get('z_ref', loc.get('z', [])), float) if isinstance(loc, dict) else np.zeros(0)
        out.append(dict(source='domain', family='nodes', surface=s, layout='global Omega0', n_expected=len(zg), **_zq(zg)))
        if len(zl):
            out.append(dict(source='domain', family='nodes', surface=s, layout='local r<=15 mm', n_expected=len(zl), **_zq(zl)))
    return out


ALIAS_DOTS = ('R6x4', 'R7.35x4', 'R8x4')
ALIAS_LINES = ('s4.45_phi0', 's5_phi0', 's5.6_phi0', 's6.3_phi0', 's8_phi0')
EXT = (-60.0, 50.0)


def _alias_summ(unit: dict, zt, zb, ib, za, ia, z0, z1, n_any_in) -> dict:

    zt, zb, za = (np.asarray(x, float) for x in (zt, zb, za)); ok = np.isfinite(zt); ib, ia = np.asarray(ib, bool) & ok, np.asarray(ia, bool) & ok
    anyin = ib | ia; hb, ha = ok & np.isfinite(zb), ok & np.isfinite(za)

    def med(x, m):
        return float(np.median(x[m])) if m.any() else None
    return dict(unit, n_expected=int(ok.sum()), window_z0=z0, window_z1=z1, alias_below_in=int(ib.sum()), alias_above_in=int(ia.sum()), alias_any_in=int(anyin.sum()),
                candidates_ge2=int(n_any_in), dz_below_med=med(zb - zt, hb), dz_above_med=med(za - zt, ha),
                below_out_by_med=med(z0 - zb, hb & ~ib), above_out_by_med=med(za - z1, ha & ~ia),
                zt_med_alias_in=med(zt, anyin), zt_med_alias_out=med(zt, ok & ~anyin), **_zq(zt[ok]))


def alias_dot(row: dict) -> dict:
    try:
        import m4_reconstruction as MR
        import s12_case as SC12
        R = SC12.raster_of(row); c = np.asarray(R['centres'], np.float64); cfg = dot_cfg('binding')
        G = dot_geometry(R['binary'], c, None, row['surface'], cfg); S = _s2(); mesh = _mesh(row['surface'])
        exp = S.SL.expected_dots(mesh, c); uv = exp['uv']; z0, z1 = G['window']
        proj_pts, _ = MR.pattern_features(np.asarray(R['binary'], np.float32), 'dots'); pp = np.asarray(proj_pts, np.float64)
        A, B = MR.projector_tracks(proj_pts, cfg, *EXT); Aw, Bw = MR.projector_tracks(proj_pts, cfg, z0, z1)
        pdot, _ = S.RD.projector_dot_index(np.asarray(R['binary'], np.float32), c)
        tol, tm = max(G['tol'][0], G['tol'][1]), G['tol'][2]; use = G['in_cam'] & ~G['occl']; n = len(c)
        zb, za = np.full(n, np.nan), np.full(n, np.nan); ib, ia = np.zeros(n, bool), np.zeros(n, bool)
        idx = np.flatnonzero(use)
        for i, (js, _t), (jw, _tw) in zip(idx, MR._tracks_within(uv[use], A, B, tol, 0.0), MR._tracks_within(uv[use], Aw, Bw, tol, tm)):
            js = np.asarray(js, np.int64); js = js[pdot[js] != i]; win = set(int(j) for j in np.asarray(jw, np.int64) if pdot[j] != i)
            if not len(js):
                continue
            z = MR.triangulate(np.repeat(uv[i][None, :], len(js), axis=0), pp[js], cfg)[:, 2]; zt_i = G['z_true'][i]
            lo, hi = z < zt_i, z > zt_i
            if lo.any():
                zb[i] = z[lo].max(); ib[i] = any(int(j) in win for j in js[lo])
            if hi.any():
                za[i] = z[hi].min(); ia[i] = any(int(j) in win for j in js[hi])
        unit = dict(source='S12', family='dots', surface=row['surface'], layout=row['layout'], phase_label=row['phase_label'], geometry=_h(c, R['binary'].shape))
        return _alias_summ(unit, np.where(use, G['z_true'], np.nan), zb, ib, za, ia, z0, z1, int((use & (G['n_geo'] >= 2)).sum()))
    except Exception as e:
        return dict(source='S12', family='dots', input_key=row.get('input_key'), error=repr(e)[:300])


def alias_line(row: dict) -> dict:
    try:
        import m4_lines as ML
        import s12_case as SC12
        R = SC12.raster_of(row); o = R['meta_full']['olines']
        z = K.load_npz(K.ROOT / row['eval_record'].replace('.json', '.npz'))
        cfg = ML._resolve_sensor_cfg(_s2().line_cfg(float(row['peak_f32'])))
        fam = ML.line_family(o); cl = ML.raster_centrelines(np.asarray(R['binary'], np.float32), fam)
        cand = ML.line_candidates(np.column_stack([z['ref_u'], z['ref_v']]), fam, cfg, centrelines=cl)
        ids = [int(k) for k in cand['ids']]; col = {k: j for j, k in enumerate(ids)}; Z = np.asarray(cand['z'], float)
        jt = np.array([col.get(int(k), -1) for k in z['ref_k']]); ok = jt >= 0; r = np.flatnonzero(ok); adm = np.asarray(cand['admissible'], bool)
        zt = np.full(len(jt), np.nan); zt[ok] = Z[r, jt[ok]]
        other = np.isfinite(Z); other[r, jt[ok]] = False
        lo, hi = other & (Z < zt[:, None]), other & (Z > zt[:, None])
        zb = np.where(lo.any(axis=1), np.where(lo, Z, -np.inf).max(axis=1), np.nan); za = np.where(hi.any(axis=1), np.where(hi, Z, np.inf).min(axis=1), np.nan)
        ib, ia = (lo & adm).any(axis=1), (hi & adm).any(axis=1)
        n_any = int((ok & (adm.sum(axis=1) >= 2)).sum())
        unit = dict(source='S12', family='lines', surface=row['surface'], layout=row['layout'], phase_label=row['phase_label'], geometry=_h(z['ref_u'], z['ref_v'], z['ref_k']))
        return _alias_summ(unit, zt, zb, ib, za, ia, float(cand['window_z0_mm']), float(cand['window_z1_mm']), n_any)
    except Exception as e:
        return dict(source='S12', family='lines', input_key=row.get('input_key'), error=repr(e)[:300])


def run_alias(workers: int = 12) -> pd.DataFrame:

    t0 = time.time(); d12, l12, _, _ = plan()
    d = list({(r['surface'], r['raster_key']): r for r in d12 if r['layout'] in ALIAS_DOTS}.values())
    ln = list({(r['surface'], r['raster_key']): r for r in l12 if r['layout'] in ALIAS_LINES}.values())
    with Pool(workers) as pool:
        T = pd.DataFrame(pool.map(alias_dot, d, chunksize=2) + pool.map(alias_line, ln, chunksize=2))
    T = T.drop_duplicates([c for c in ('source', 'family', 'surface', 'layout', 'geometry') if c in T]) if 'geometry' in T else T
    print(f'A2 alias: {len(T)} rows, {round((time.time() - t0) / 60, 1)} min, errors {int(T["error"].notna().sum()) if "error" in T else 0}', flush=True)
    return T


def run(workers: int = 12) -> pd.DataFrame:
    t0 = time.time(); d12, l12, d3, l3 = plan()
    with Pool(workers) as pool:
        a = pool.map(work_s12_dot, d12, chunksize=4); b = pool.map(work_s12_line, l12, chunksize=4)
        c = [x for r in pool.map(work_s3_dot, d3, chunksize=2) for x in r]; d = [x for r in pool.map(work_s3_line, l3, chunksize=2) for x in r]
    T = pd.DataFrame(a + b + c + d + node_z())
    print(f'A2: {len(T)} rows, {round((time.time() - t0) / 60, 1)} min, errors {int(T.get("error", pd.Series(dtype=object)).notna().sum()) if "error" in T else 0}', flush=True)
    return T


if __name__ == '__main__':
    import sys
    T = run(); T.to_csv(sys.argv[1] if len(sys.argv) > 1 else 'a2.csv', index=False)
