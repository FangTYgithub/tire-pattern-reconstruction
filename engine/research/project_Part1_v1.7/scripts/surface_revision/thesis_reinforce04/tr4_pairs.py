# Compare corresponding features and image-level metrics between ray-sampling settings.

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import tr4_chain as CH
import tr4_common as K
import tr4_prep as P

REF_PP, REF_MM = 0.2, 0.005
METRICS = [
    ('glob_Ctau', 'pp', 100.0, 'higher'), ('loc_Ctau', 'pp', 100.0, 'higher'), ('glob_Cvalid', 'pp', 100.0, 'higher'), ('loc_Cvalid', 'pp', 100.0, 'higher'),
    ('glob_recovery', 'pp', 100.0, 'higher'), ('loc_recovery', 'pp', 100.0, 'higher'),
    ('glob_pt_rmse', 'mm', 1.0, 'lower'), ('glob_pt_p95', 'mm', 1.0, 'lower'), ('loc_pt_rmse', 'mm', 1.0, 'lower'), ('loc_pt_p95', 'mm', 1.0, 'lower'),
    ('glob_surf_rmse', 'mm', 1.0, 'lower'), ('loc_surf_rmse', 'mm', 1.0, 'lower'),
    ('w_share', 'pp', 100.0, 'lower'), ('u_share', 'pp', 100.0, 'lower'),
    ('n_detected', 'count', 1.0, 'none'), ('n_accepted', 'count', 1.0, 'none'), ('n_valid', 'count', 1.0, 'none'), ('id_correct', 'count', 1.0, 'none'),
    ('id_wrong', 'count', 1.0, 'none'), ('id_unconf', 'count', 1.0, 'none'), ('glob_n_valid', 'count', 1.0, 'none'), ('loc_n_valid', 'count', 1.0, 'none')]
SIZE_METRICS = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'loc_pt_rmse', 'glob_surf_rmse')
SLOPE_ROWS, SLOPE_MIN = 3, 3


def _undef(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _ratio(a, b):
    return (float(a) / float(b)) if b else None


def load_old(c: dict) -> tuple[dict, dict, str]:
    p = K.ROOT / c['eval_record']; E = K.load_json(p)
    with np.load(p.with_suffix('.npz')) as z:
        A = {k: np.asarray(z[k]) for k in z.files}
    return E['metrics'], A, K.rel(p)


def load_new(store: K.Store, ik: str, rk: str) -> tuple[dict, dict, str]:
    meta, A, _ = CH.pack_read(store, ik)
    r = meta['results'][rk]
    return r['metrics'], CH._sub(A, f'{rk}__ev__'), f'{K.rel(CH.pack_path(store, ik))}#{rk}'


def case_results(c: dict, store: K.Store) -> dict:

    out = {}
    if c['family'] == 'point':
        out[('D0', 1)] = load_old(c); out[('D0', 3)] = load_new(store, c['input_key'], 'n3_D0')
        out[('DF', 1)] = load_new(store, c['input_key'], 'n1_DF'); out[('DF', 3)] = load_new(store, c['input_key'], 'n3_DF')
    else:
        out[('P4', 1)] = load_old(c); out[('P4', 3)] = load_new(store, c['input_key'], 'n3_P4')
    return out


def line_identity_full(c: dict, a: dict) -> dict:
    import s1r_common as S1K
    import tr_a4 as A4
    R = S1K.load_raster(c['raster_key']); o = R['meta_full']['olines']; s = A4.scan_raster(R['binary'], o)
    acc = np.asarray(a['accepted'], bool); lid = np.asarray(a['line_id'], np.int64)
    r = A4.reidentify(A4._mesh(c['surface']), a['feats'], o, s['extra'], np.asarray(R['binary']) > 0, acc, lid, np.asarray(a['confirmable'], bool),
                      np.asarray(a['truth_k'], np.int64), np.asarray(a['valid'], bool), np.asarray(a['in_global'], bool), np.asarray(a['in_local'], bool))
    return dict(extra=s['extra'], closure_old_identity=r['closure_old_identity'], n_new_labels=r['n_new_labels'], c=r['all_new_c'], w=r['all_new_w'], u=r['all_new_u'],
                stored_c=int((acc & np.asarray(a['point_correct'], bool)).sum()), stored_w=int((acc & np.asarray(a['point_wrong'], bool)).sum()),
                stored_u=int((acc & np.asarray(a['point_unconf'], bool)).sum()))


def result_row(c: dict, method: str, rays: int, m: dict, a: dict, src: str, img_stats: dict) -> dict:
    fam = c['family']
    c_, w_, u_ = int(m['id_correct']), int(m['id_wrong']), int(m['id_unconf'])
    row = dict(record_type='result', input_key=c['input_key'], family=fam, surface=c['surface'], size_mm=float(c['size_mm']), phase_label=c['phase_label'],
               method=method, rays=rays, sigma=0.0, source=src, state=m['state'], n_detected=int(m.get('n_selected', m.get('n_detected', len(a.get('feats', []))))),
               n_candidates=m.get('n_candidates'), n_accepted=int(m['n_accepted']), n_valid=int(m['n_valid']), id_correct=c_, id_wrong=w_, id_unconf=u_,
               w_share=_ratio(w_, c_ + w_), u_share=_ratio(u_, c_ + w_ + u_), accepted_but_all_invalid=bool(int(m['n_accepted']) > 0 and int(m['n_valid']) == 0))
    for k in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_cov_interp', 'loc_cov_interp', 'glob_recovery', 'loc_recovery', 'glob_pt_rmse', 'glob_pt_p95',
              'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'loc_surf_rmse', 'glob_n_valid', 'loc_n_valid', 'glob_n_interp', 'loc_n_interp', 'glob_n_nodes', 'loc_n_nodes',
              'img_n_expected', 'img_n_recovered', 'n_out_of_window'):
        row[k] = m.get(k)
    if fam == 'line':
        li = line_identity_full(c, a)
        row.update(full_reg_extra_lines=';'.join(map(str, li['extra'])), full_reg_new_labels=li['n_new_labels'], full_reg_closure=li['closure_old_identity'],
                   full_reg_c=li['c'], full_reg_w=li['w'], full_reg_u=li['u'], full_reg_equal_stored=(li['c'], li['w'], li['u']) == (li['stored_c'], li['stored_w'], li['stored_u']))
    row.update({f'img_{k}': v for k, v in img_stats.items()})
    return row


def _unique_sources(keys: np.ndarray, valid_src: np.ndarray) -> tuple[dict, set]:

    idx = {}; amb = set()
    for f in np.flatnonzero(valid_src):
        k = tuple(int(x) for x in np.atleast_1d(keys[f]))
        if k in idx or k in amb:
            amb.add(k); idx.pop(k, None)
        else:
            idx[k] = int(f)
    return idx, amb


def _point_of_feature(a: dict, n_feat: int) -> np.ndarray:

    pf = np.asarray(a['point_feature'], np.int64); out = np.full(n_feat, -1, np.int64)
    ok = pf >= 0; out[pf[ok]] = np.flatnonzero(ok)
    return out


def _q(x, q):
    x = np.asarray(x, np.float64)
    return float(np.percentile(x, q)) if x.size else None


def feature_pair(c: dict, a1: dict, a3: dict) -> dict:
    fam = c['family']
    f1 = np.asarray(a1['feats'], np.float64).reshape(-1, 2); f3 = np.asarray(a3['feats'], np.float64).reshape(-1, 2)
    if fam == 'point':
        k1 = np.asarray(a1['feat_truth_dot'], np.int64); k3 = np.asarray(a3['feat_truth_dot'], np.int64)
        s1, amb1 = _unique_sources(k1, np.asarray(a1['feat_confirmable'], bool) & (k1 >= 0)); s3, amb3 = _unique_sources(k3, np.asarray(a3['feat_confirmable'], bool) & (k3 >= 0))
        p1 = _point_of_feature(a1, len(f1)); p3 = _point_of_feature(a3, len(f3))
    else:
        v1 = np.rint(f1[:, 1]).astype(np.int64); v3 = np.rint(f3[:, 1]).astype(np.int64)
        if len(f1) and np.abs(f1[:, 1] - v1).max() > 1e-6 or len(f3) and np.abs(f3[:, 1] - v3).max() > 1e-6:
            raise K.Stop('technical', f"{c['input_key']}: line samples are not on integer camera rows")
        k1 = np.column_stack([np.asarray(a1['truth_k'], np.int64), v1]); k3 = np.column_stack([np.asarray(a3['truth_k'], np.int64), v3])
        s1, amb1 = _unique_sources(k1, np.asarray(a1['confirmable'], bool)); s3, amb3 = _unique_sources(k3, np.asarray(a3['confirmable'], bool))
        p1 = np.arange(len(f1)); p3 = np.arange(len(f3))
    all1 = set(s1) | amb1; all3 = set(s3) | amb3
    common = sorted(set(s1) & set(s3))
    out = dict(record_type='feature_pair', input_key=c['input_key'], family=fam, surface=c['surface'], size_mm=float(c['size_mm']), phase_label=c['phase_label'],
               n_feat_1x1=len(f1), n_feat_3x3=len(f3), n_src_1x1=len(all1), n_src_3x3=len(all3), n_ambiguous_1x1=len(amb1), n_ambiguous_3x3=len(amb3),
               n_common_any=len(all1 & all3), n_lost=len(all1 - all3), n_new=len(all3 - all1), n_common_unique=len(common))
    if fam == 'point':
        d = np.array([np.hypot(*(f3[s3[k]] - f1[s1[k]])) for k in common]) if common else np.zeros(0)
        out.update(shift_kind='centroid', shift_median_px=_q(d, 50), shift_p95_px=_q(d, 95), shift_max_px=(float(d.max()) if d.size else None), n_slope_undefined=None)
    else:
        by_line = {}
        for (kk, vv), f in s1.items():
            by_line.setdefault(kk, []).append((vv, f1[f, 0]))
        slope = {}
        for kk, lst in by_line.items():
            arr = np.array(sorted(lst)); slope[kk] = arr
        d = []; nund = 0
        for kk, vv in common:
            arr = slope[kk]; sel = np.abs(arr[:, 0] - vv) <= SLOPE_ROWS
            if sel.sum() < SLOPE_MIN:
                nund += 1; continue
            b = np.polyfit(arr[sel, 0], arr[sel, 1], 1)[0]
            du = f3[s3[(kk, vv)], 0] - f1[s1[(kk, vv)], 0]
            d.append(abs(du) / math.sqrt(1.0 + b * b))
        d = np.asarray(d)
        out.update(shift_kind='normal', shift_median_px=_q(d, 50), shift_p95_px=_q(d, 95), shift_max_px=(float(d.max()) if d.size else None), n_slope_undefined=nund)
    acc1 = np.asarray(a1['accepted'], bool) & np.asarray(a1['valid'], bool); acc3 = np.asarray(a3['accepted'], bool) & np.asarray(a3['valid'], bool)
    P1, P3 = np.asarray(a1['pts'], np.float64), np.asarray(a3['pts'], np.float64)
    dz = []
    for k in common:
        i1, i3 = p1[s1[k]], p3[s3[k]]
        if i1 >= 0 and i3 >= 0 and acc1[i1] and acc3[i3]:
            dz.append(P3[i3, 2] - P1[i1, 2])
    dz = np.asarray(dz)
    out.update(n_common_both_valid=int(dz.size), dz_median_mm=_q(dz, 50), dz_abs_median_mm=_q(np.abs(dz), 50), dz_abs_p95_mm=_q(np.abs(dz), 95),
               dz_abs_max_mm=(float(np.abs(dz).max()) if dz.size else None))
    return out


def image_pair(c: dict, img1, img3, m3_1: dict, m3_3: dict) -> dict:
    d = img3.astype(np.int16) - img1.astype(np.int16)
    g1, g3 = int(img1.astype(np.int64).sum()), int(img3.astype(np.int64).sum())
    return dict(record_type='image_pair', input_key=c['input_key'], family=c['family'], surface=c['surface'], size_mm=float(c['size_mm']), phase_label=c['phase_label'],
                n_px_changed=int((d != 0).sum()), share_px_changed=float((d != 0).mean()), mean_abs_diff=float(np.abs(d).mean()), max_abs_diff=int(np.abs(d).max()),
                mean_abs_diff_changed=(float(np.abs(d[d != 0]).mean()) if (d != 0).any() else 0.0),
                grey_sum_1x1=g1, grey_sum_3x3=g3, grey_sum_rel_change=(g3 - g1) / g1 if g1 else None, sat_share_1x1=float((img1 == 255).mean()), sat_share_3x3=float((img3 == 255).mean()),
                n_rays_cast_1x1=m3_1['n_rays_cast'], n_rays_cast_3x3=m3_3['n_rays_cast'], n_hits_1x1=m3_1['n_hits'], n_hits_3x3=m3_3['n_hits'],
                n_hit_pixels_1x1=m3_1['n_valid_pixels'], n_hit_pixels_3x3=m3_3['n_valid_pixels'])


def metric_pairs(c: dict, method: str, r1: dict, r3: dict) -> list:
    out = []
    for name, unit, sc, better in METRICS:
        v1, v3 = r1.get(name), r3.get(name)
        if _undef(v1) or _undef(v3):
            st = 'undefined_both' if (_undef(v1) and _undef(v3)) else ('undefined_1x1' if _undef(v1) else 'undefined_3x3'); dl = None
        else:
            st = 'defined'; dl = (float(v3) - float(v1)) * sc
        ref = REF_PP if unit == 'pp' else (REF_MM if unit == 'mm' else None)
        direction = None
        if dl is not None and better != 'none':
            direction = 'equal' if dl == 0 else (('better' if dl > 0 else 'worse') if better == 'higher' else ('better' if dl < 0 else 'worse'))
        out.append(dict(record_type='metric_pair', input_key=c['input_key'], family=c['family'], surface=c['surface'], size_mm=float(c['size_mm']), phase_label=c['phase_label'],
                        method=method, metric=name, unit=unit, v_1x1=(None if _undef(v1) else float(v1) * sc), v_3x3=(None if _undef(v3) else float(v3) * sc), delta=dl,
                        delta_state=st, direction=direction, exceeds_reference=(None if dl is None or ref is None else bool(abs(dl) > ref)), state_1x1=r1['state'], state_3x3=r3['state']))
    return out


def size_effects(R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    D = R[(R.record_type == 'result') & (R.family == 'point')]
    for (surf, ph, meth), g in D.groupby(['surface', 'phase_label', 'method']):
        for name in SIZE_METRICS:
            unit, sc = next((u, s) for n, u, s, _ in METRICS if n == name)
            val = {}
            for rays in (1, 3):
                for r in (0.5, 1.0, 1.5):
                    x = g[(g.rays == rays) & (g.size_mm == r)][name]
                    v = x.iloc[0] if len(x) == 1 else None
                    val[(rays, r)] = None if _undef(v) else float(v) * sc
            for a, b in ((0.5, 1.0), (1.0, 1.5)):
                s1 = None if val[(1, a)] is None or val[(1, b)] is None else val[(1, b)] - val[(1, a)]
                s3 = None if val[(3, a)] is None or val[(3, b)] is None else val[(3, b)] - val[(3, a)]
                sg = lambda x: None if x is None else int(np.sign(x))
                rows.append(dict(record_type='size_effect', surface=surf, phase_label=ph, method=meth, metric=name, unit=unit, step=f'r{b}-r{a}', S_1x1=s1, S_3x3=s3,
                                 drift=(None if s1 is None or s3 is None else s3 - s1), sign_1x1=sg(s1), sign_3x3=sg(s3),
                                 sign_change=(None if s1 is None or s3 is None else bool(sg(s1) != sg(s3)))))
            for rays in (1, 3):
                vs = [(val[(rays, r)], r) for r in (0.5, 1.0, 1.5)]
                if any(v is None for v, _ in vs):
                    order = None
                else:
                    better = next(b for n, _, _, b in METRICS if n == name)
                    order = '>'.join(f'{r:g}' for _, r in sorted(vs, key=lambda t: (-t[0] if better == 'higher' else t[0], t[1])))
                rows.append(dict(record_type='size_rank', surface=surf, phase_label=ph, method=meth, metric=name, rays=rays, order_best_first=order))
    T = pd.DataFrame(rows)
    rk = T[T.record_type == 'size_rank']
    if len(rk):
        rk = rk.assign(rays=rk['rays'].astype(int))
        piv = rk.pivot_table(index=['surface', 'phase_label', 'method', 'metric'], columns='rays', values='order_best_first', aggfunc='first').reset_index()
        piv.columns = [str(x) for x in piv.columns]
        ch = [dict(record_type='size_rank_change', surface=r['surface'], phase_label=r['phase_label'], method=r['method'], metric=r['metric'], order_1x1=r.get('1'),
                   order_3x3=r.get('3'), rank_changed=(None if r.get('1') is None or r.get('3') is None or pd.isna(r.get('1')) or pd.isna(r.get('3')) else bool(r.get('1') != r.get('3'))))
              for r in piv.to_dict('records')]
        T = pd.concat([T[T.record_type != 'size_rank'], pd.DataFrame(ch)], ignore_index=True, sort=False)
    _rank_consistency(T)
    return T


def _rank_consistency(T: pd.DataFrame) -> None:

    E = T[T.record_type == 'size_effect']; C = T[T.record_type == 'size_rank_change'].set_index(['surface', 'phase_label', 'method', 'metric'])
    for r in E.to_dict('records'):
        k = (r['surface'], r['phase_label'], r['method'], r['metric']); c = C.loc[k]
        both = r['S_1x1'] is not None and r['S_3x3'] is not None and not pd.isna(r['S_1x1']) and not pd.isna(r['S_3x3'])
        if both and (pd.isna(c['order_1x1']) or pd.isna(c['order_3x3'])):
            raise K.Stop('technical', f'size rank order missing for {k}')
        if both and r['sign_1x1'] * r['sign_3x3'] == -1 and not bool(c['rank_changed']):
            raise K.Stop('technical', f'sign change without rank change for {k} {r["step"]}')


def surface_summary(M: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (surf, fam, meth, size, name), g in M.groupby(['surface', 'family', 'method', 'size_mm', 'metric']):
        d = g[g.delta_state == 'defined'].sort_values('phase_label')
        vals = d.delta.astype(float).tolist()
        rows.append(dict(record_type='surface_summary', surface=surf, family=fam, method=meth, size_mm=size, metric=name, unit=g.unit.iloc[0], n_phases=len(g), n_defined=len(vals),
                         delta_phases=';'.join(f'{p}:{v:.6g}' for p, v in zip(d.phase_label, vals)), delta_mean=(float(np.mean(vals)) if vals else None),
                         delta_min=(min(vals) if vals else None), delta_max=(max(vals) if vals else None), states=';'.join(sorted(set(g.delta_state)))))
    return pd.DataFrame(rows)


def size_drift_summary(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    E = S[S.record_type == 'size_effect']
    for (surf, meth, name, step), g in E.groupby(['surface', 'method', 'metric', 'step']):
        d = g.dropna(subset=['drift'])
        rows.append(dict(record_type='size_effect_surface', surface=surf, method=meth, metric=name, unit=g.unit.iloc[0], step=step, n_phases=len(g), n_defined=len(d),
                         S_1x1_mean=(float(d.S_1x1.mean()) if len(d) else None), S_3x3_mean=(float(d.S_3x3.mean()) if len(d) else None),
                         drift_mean=(float(d.drift.mean()) if len(d) else None), drift_min=(float(d.drift.min()) if len(d) else None), drift_max=(float(d.drift.max()) if len(d) else None),
                         n_sign_change=int(d.sign_change.astype(bool).sum()) if len(d) else 0))
    return pd.DataFrame(rows)


def build(C: pd.DataFrame, store: K.Store, only: list | None = None) -> dict:
    compat = CH.compat_load(store); res = []; pairs = []; cases = {}
    for c in C.to_dict('records'):
        ik = c['input_key']
        if only is not None and ik not in only:
            continue
        meta, A, st = CH.pack_read(store, ik)
        p1 = P.prepare(c, 1)
        img1 = np.asarray(K.load_npz(p1['stored_image'])['image']); img3 = A['img3']
        m3_1 = compat[ik]['m3']; m3_3 = meta['image3']['m3']
        s1 = dict(CH.image_stats(img1), **{k: m3_1[k] for k in ('n_rays_cast', 'n_hits', 'n_valid_pixels')})
        s3 = dict(CH.image_stats(img3), **{k: m3_3[k] for k in ('n_rays_cast', 'n_hits', 'n_valid_pixels')})
        R = case_results(c, store); cases[ik] = dict(cond=c, results=R, img1=img1, img3=img3)
        pairs.append(image_pair(c, img1, img3, m3_1, m3_3))
        for meth in P.METHODS[c['family']]:
            (m1, a1, src1), (m3, a3, src3) = R[(meth, 1)], R[(meth, 3)]
            r1 = result_row(c, meth, 1, m1, a1, src1, s1); r3 = result_row(c, meth, 3, m3, a3, src3, s3)
            res += [r1, r3]
            fp = feature_pair(c, a1, a3); fp['method'] = meth; pairs.append(fp)
            pairs += metric_pairs(c, meth, r1, r3)
    Rdf = pd.DataFrame(res); Pdf = pd.DataFrame(pairs)
    M = Pdf[Pdf.record_type == 'metric_pair']
    extra = [surface_summary(M)]
    if only is None:
        S = size_effects(Rdf); extra += [S, size_drift_summary(S)]
    Pdf = pd.concat([Pdf] + extra, ignore_index=True, sort=False)
    return dict(results=Rdf, pairs=Pdf, cases=cases)


def select_cases(pairs: pd.DataFrame) -> dict:

    M = pairs[pairs.record_type == 'metric_pair']; out = {}
    for fam in ('point', 'line'):
        cand = []
        for (ik, meth), g in M[M.family == fam].groupby(['input_key', 'method']):
            g = g.set_index('metric')
            st_changed = g.loc['glob_Ctau', 'state_1x1'] != g.loc['glob_Ctau', 'state_3x3']
            dc = g.loc['glob_Ctau', 'delta']; dr = g.loc['glob_pt_rmse', 'delta']
            cand.append(((0 if st_changed else 1), (-abs(dc) if not _undef(dc) else math.inf), (-abs(dr) if not _undef(dr) else math.inf), ik, meth,
                         dict(input_key=ik, method=meth, state_changed=bool(st_changed), d_glob_Ctau_pp=(None if _undef(dc) else float(dc)), d_glob_pt_rmse_mm=(None if _undef(dr) else float(dr)))))
        cand.sort(key=lambda t: t[:5])
        out[fam] = cand[0][5] if cand else None
    return out


RULES = dict(
    pairing='same surface / pattern / phase / stored peak / method; 1x1 (stored D0 / P4, new DF 1x1) vs 3x3; D0 never vs DF',
    features='dots: unique confirmable true dot per feature; lines: unique (truth_k, camera row); ambiguous sources excluded and counted; '
             'line normal shift with local slope (+-3 rows, >= 3 samples); dz on common unique sources accepted and valid in both',
    identity='w / (c + w), u / (c + w + u), zero denominators undefined; lines: A4 extension on the same raster for 1x1 and 3x3',
    size_effect='S(r1.0 - r0.5), S(r1.5 - r1.0) per surface / phase / method / rays; drift = S3 - S1; sign change; rank order',
    reference='0.005 mm, 0.2 pp = explanation reference lines only; all pairs reported; undefined never 0',
    f2_rule='per family: state changed > |global dC_tau| > |global d point RMSE| > input_key (then method); undefined last',
    metrics=[m[0] for m in METRICS], size_metrics=list(SIZE_METRICS))
