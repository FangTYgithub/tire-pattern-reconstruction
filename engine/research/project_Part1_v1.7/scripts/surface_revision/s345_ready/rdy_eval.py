# Evaluate reconstructed points and classify their source identities.

from __future__ import annotations

import numpy as np

import rdy_common as K
import rdy_platform as P


def ctx_eval(row: dict, inp: dict) -> dict:

    if row['family'] == 'point':
        if row['section'] == 'S3':
            size = max(inp['enc_spec']['radii_mm'])
        elif row['section'] == 'S4':
            import sp2_common as S2
            size = float(S2.R_MM)
        else:
            size = float(row['radius_mm'])
    else:
        size = None
    meta = inp['meta_full']
    if row['section'] == 'S4' and row['family'] == 'line':
        meta = dict(olines=inp['table']['spec'])
    return dict(binary=inp['binary'], centres=(None if inp['centres'] is None else np.asarray(inp['centres'], np.float64)), pattern=inp['pattern'], peak=float(inp['peak']),
                size_mm=size, meta_full=meta, row=dict(radius_mm=size))


def evaluate_common(row: dict, dec: dict, ctx: dict):
    import s1r_eval as E
    a = row['surface']; sid = K.SURF_MAP[a]
    if row['section'] == 'S4' and row['family'] == 'line':
        return evaluate_lines_table(sid, dec, ctx, P.mesh(a), P.dom_local(a), P.dom_global(), sid == 'A_curv_gauss')
    m, arrays, info, z = E.evaluate_case(sid, row['family'], dec, ctx, P.mesh(a), P.dom_local(a), P.dom_global(), sid == 'A_curv_gauss')
    return m, arrays, info


def _truth_table(mesh, feat, spec):
    import sp2_lines_table as LT2
    return LT2.truth_identity_table(mesh, feat, spec)


def _ref_table(mesh, spec, pattern):
    import sp2_lines_table as LT2
    return LT2.line_reference_table(mesh, spec, 1.0, pattern=np.asarray(pattern, np.float32))


# Evaluate table-positioned line reconstructions and their source identities.
def evaluate_lines_table(sid, dec, ctx, mesh, dom_loc, dom_glob, gauss, truth_fn=_truth_table, ref_fn=_ref_table):

    import s1r_common as S1K
    import s1r_eval as E
    S = P.s2()
    V = S.point_validity(dec['pts'], dec['accepted'], mesh)
    pts, acc, valid, zr = V['pts'], V['acc'], V['valid'], V['zr']
    e_pts = pts[:, 2] - zr
    loc_pts = valid & S1K.in_local(pts[:, :2]); glob_pts = valid & S1K.in_global(pts[:, :2], dom_glob)
    aux = dec['aux']; info = {}
    arrays = dict(pts=pts, accepted=acc, valid=valid, zr=zr.astype(np.float32), in_local=loc_pts, in_global=glob_pts)
    m = dict(n_out=int(len(pts)), n_accepted=int(acc.sum()), n_valid=int(valid.sum()), n_nonfinite_acc=int((acc & ~V['fin']).sum()), n_out_of_window=int((acc & V['fin'] & ~V['in_win']).sum()),
             n_out_of_grid=int((acc & V['fin'] & ~V['in_grid']).sum()))
    r = aux['r']; feat = np.asarray(r['features'], np.float64).reshape(-1, 2); lid = np.asarray(r['line_id'], np.int64); st = np.asarray(r['status'], np.int64)
    meta = ctx['meta_full']['olines']
    ident = truth_fn(mesh, feat, meta); kO = np.asarray(ident['k'], np.int64); conf = np.asarray(ident['confirmable'], bool)
    if len(feat) != len(pts):
        raise K.Stop('technical', f'line output: {len(feat)} features vs {len(pts)} points')
    correct = acc & conf & (lid == kO); wrong = acc & conf & (lid != kO); unconf = acc & ~conf
    ref = ref_fn(mesh, meta, ctx['pattern']); R = ref['rows']
    rec_any, rec_id = S._row_flags(ref, feat, lid, acc & valid)
    xyz_r = np.asarray(R['xyz'], np.float64).reshape(-1, 3)
    exp_loc = S1K.in_local(xyz_r[:, :2]); exp_glob = S1K.in_global(xyz_r[:, :2], dom_glob)
    rec_dots = rec_id
    from m4_lines import STATUS
    m.update(n_detected=int(len(feat)), n_confirmable_features=int(conf.sum()), **{f'n_status_{STATUS[k]}': int((st == k).sum()) for k in STATUS},
             id_correct=int(correct.sum()), id_wrong=int(wrong.sum()), id_unconf=int(unconf.sum()), id_denominator=int(acc.sum()), mismatch_rate=E._ratio(int(wrong.sum()), int(correct.sum() + wrong.sum())),
             unconf_share=E._ratio(int(unconf.sum()), int(acc.sum())), img_n_expected=int(len(xyz_r)), img_n_recovered=int(rec_id.sum()), img_recovery=E._ratio(int(rec_id.sum()), len(xyz_r)),
             img_n_recovered_any=int(rec_any.sum()), img_recovery_any=E._ratio(int(rec_any.sum()), len(xyz_r)), n_features_design=int(meta['n_lines']),
             n_ref_rows_inner=int(np.asarray(R['inner'], bool).sum()))
    if 'multi_candidate' in ident:
        m['n_multi_candidate'] = int((acc & np.asarray(ident['multi_candidate'], bool)).sum())
    info.update(detect_info={k: v for k, v in (r.get('detect_info') or {}).items() if not isinstance(v, np.ndarray)}, counts=r.get('counts'), checks=r.get('checks'),
                reference=dict(n_rows=int(len(xyz_r)), per_line=ref['per_line'], centre_source=ref['centre_source']))
    arrays.update(feats=feat.astype(np.float32), status=st.astype(np.int8), line_id=lid.astype(np.int32), truth_k=kO.astype(np.int32), confirmable=conf, chain=np.asarray(r['chain']).astype(np.int32),
                  segment_id=np.asarray(r['segment_id']).astype(np.int32), n_candidates=np.asarray(r['n_candidates']).astype(np.int8), guard_reason=np.asarray(r['guard_reason']).astype(np.int8),
                  point_correct=correct, point_wrong=wrong, point_unconf=unconf, ref_k=np.asarray(R['k'], np.int32), ref_v=np.asarray(R['v'], np.int32), ref_u=np.asarray(R['u'], np.float32),
                  ref_xyz=xyz_r.astype(np.float32), ref_inner=np.asarray(R['inner'], bool), ref_rec_any=rec_any, ref_rec_id=rec_id)
    m.update(loc_n_recovered_any=int((exp_loc & rec_any).sum()), glob_n_recovered_any=int((exp_glob & rec_any).sum()))
    dl = S.roi_domain(mesh) if dom_loc is None else dom_loc
    ev = S.evaluate(pts, acc, mesh, dl, gauss, fields=True)
    ev.pop('_z_est')
    if ev['n_valid'] != int(valid.sum()) or ev['n_valid_roi'] != int(loc_pts.sum()):
        raise K.Stop('technical', 'validity bookkeeping differs from s2.evaluate')
    r3l = ev.get('r3_dev') or {}; bdl = ((r3l.get('subdomains') or {}).get('global') or {}).get('by_d', {}).get(str(float(K.D_SUPPORT)))
    nl = ev['nodes_roi']
    surf_loc = dict(n_nodes=int(nl['n_nodes']), n_interp=int(nl.get('n', 0) or 0), cov_interp=float(nl['coverage']), Cvalid=(float(bdl['C_valid']) if bdl else 0.0),
                    Ctau=(float(bdl['C_tau'][str(float(K.TAU))]) if bdl else 0.0), surf=dict(rmse=nl.get('rmse'), mae=nl.get('mae'), p95=nl.get('p95_abs'), bias=nl.get('bias')))
    m.update(E._domain_block('loc', loc_pts, e_pts, float(S.GROSS_MM), correct, wrong, unconf, exp_loc, rec_dots, surf_loc))
    if abs((m['loc_pt_rmse'] or 0.0) - (ev['pts_roi']['rmse'] or 0.0)) > 1e-12 or m['loc_n_gross'] != ev['n_gross_roi']:
        raise K.Stop('technical', 'local point statistics differ from s2.evaluate')
    zg = dom_glob[f'z_{sid}']
    sg, _z = E.global_surface(pts[valid], dom_glob, zg)
    m.update(E._domain_block('glob', glob_pts, e_pts, float(S.GROSS_MM), correct, wrong, unconf, exp_glob, rec_dots, sg))
    am = ev.get('amplitude') or {}
    if gauss:
        m.update(amp_applicable=True, amp_true_mm=am.get('true_mm'), amp_peak_mm=am.get('rec_peak_mm'), amp_centre_mm=am.get('rec_centre_mm'), amp_err_peak_mm=am.get('err_peak_mm'),
                 amp_err_centre_mm=am.get('err_centre_mm'), amp_err_peak_pct=am.get('err_peak_pct'))
    else:
        m.update(amp_applicable=False)
    m['state'] = 'no_output' if m['n_valid'] == 0 else 'ok'
    info['local_r3'] = dict(version=r3l.get('version'), triangulation=r3l.get('triangulation'), status=r3l.get('status')); info['global_r3'] = sg['r3']
    info['amplitude_definition'] = am.get('definition') if gauss else 'not applicable (no Gaussian reference term)'
    info['truth'] = 'table truth (sp2_lines_table.truth_identity_table / line_reference_table, same table spec)'
    return m, arrays, info


def enc_case_arrays(img, inp: dict, out: dict, mesh: dict) -> dict:

    import r5_detect as RD
    import sn_lib as SL
    S = P.s2(); D = out['shared']; c = np.asarray(inp['centres'], np.float64); binpat = np.asarray(inp['binary'], np.float32)
    exp = SL.expected_dots(mesh, c)
    proj_dot, _ = RD.projector_dot_index(binpat, c)
    cands = D['cd']['cands']
    a = dict(cam_img=np.asarray(img, np.uint8), pattern_bin=(binpat > 0).astype(np.uint8), centres_px=c.astype(np.float32), centre_class=np.asarray(inp['classes'], np.int8),
             exp_xy=exp['xy'].astype(np.float32), exp_uv=exp['uv'].astype(np.float32), exp_in_disc=exp['in_disc'], exp_cam_inner=exp['cam_inner'],
             cand_uv=np.asarray([[q[0], q[1]] for q in cands], np.float32).reshape(-1, 2), cand_area=np.asarray([q[2] for q in cands], np.int32),
             cand_label=np.asarray([q[5] for q in cands], np.int32), cand_code=np.asarray(D['code'], np.int8), cand_fidx=np.asarray(D['fidx'], np.int32),
             cand_class=np.asarray(D['cinfo'].get('cls_all', np.zeros(len(cands), np.int64)), np.int8), comp_reason=np.asarray(D['cd']['reason'], np.int8),
             feat_class=np.asarray(D['det_info'].get('feature_classes', np.zeros(len(D['feats']), np.int64)), np.int8), feat_allowed=np.asarray(D['allowed'], bool),
             feat_n_geom=np.asarray(D['cand']['n_geom'], np.int16), feat_n_class=np.asarray(D['cand']['n_class'], np.int16), proj_dot=np.asarray(proj_dot, np.int32),
             proj_class=np.asarray(D['proj']['cls'], np.int8), proj_ids=np.asarray(D['proj']['ids'], np.int32),
             feat_geom_ids=np.array([','.join(map(str, np.asarray(g).tolist())) for g in D['cand']['geom_ids']], dtype=str),
             feat_class_ids=np.array([','.join(map(str, np.asarray(g).tolist())) for g in D['cand']['class_ids']], dtype=str))
    f = S.truth_interp(mesh)
    for arm, d in out['arms'].items():
        suf = '' if arm == 'ENC_coded' else '_ig'; tr = d['aux']['tr']; pts = np.asarray(d['pts'], np.float32).reshape(-1, 3)
        amb = np.asarray((d['aux']['info'] or {}).get('ambiguous_mask', np.zeros(len(pts), bool)), bool)
        amb = amb if len(amb) == len(pts) else np.zeros(len(pts), bool)
        fin = np.isfinite(pts).all(axis=1); zr = np.full(len(pts), np.nan)
        if fin.any():
            zr[fin] = f(np.column_stack([pts[fin, 1].astype(np.float64), pts[fin, 0].astype(np.float64)]))
        a.update({f'feat_matched{suf}': np.asarray(tr['matched'], bool), f'feat_ambiguous{suf}': np.asarray(tr['ambiguous'], bool), f'feat_duplicate{suf}': np.asarray(tr['duplicate'], bool),
                  f'feat_best_idx{suf}': np.asarray(tr['best_idx']).astype(np.int32), f'feat_point_idx{suf}': np.asarray(tr['point_idx']).astype(np.int32), f'pts{suf}': pts, f'amb{suf}': amb,
                  f'gt_ref{suf}': zr.astype(np.float32), f'err_ref{suf}': np.where(fin, pts[:, 2] - zr, np.nan).astype(np.float32)})
    return a


def enc_labels(img, ref_img, inp: dict, out: dict, row: dict, mesh: dict, label_dir) -> tuple[dict, dict]:

    import json
    import ep_lib as EL
    arrays = enc_case_arrays(img, inp, out, mesh)
    modes = {}
    for arm in out['arms']:
        suf = '' if arm == 'ENC_coded' else '_ig'
        pts = arrays[f'pts{suf}']; amb = arrays[f'amb{suf}']; err = arrays[f'err_ref{suf}']; fin = np.isfinite(pts).all(axis=1)
        rel_ = fin & ~amb; r_rel = err[rel_ & np.isfinite(err)]
        info = out['arms'][arm]['aux']['info'] or {}; n_det = int(info.get('n_detected', 0) or 0); n_rej = int(info.get('n_rejected', 0) or 0)
        off = float(np.median(r_rel)) if len(r_rel) else None; neg = float((pts[fin, 2] < 0).mean()) if fin.any() else None
        alarms = dict(rej=bool(n_det and n_rej / n_det > EL.ALARM['rej_frac']), offset=bool(off is not None and abs(off) > EL.ALARM['offset_mm']), neg_share_reported=neg,
                      note='negative world z is valid under the common window: reported only')
        alarms['any'] = bool(alarms['rej'] or alarms['offset'])
        modes['coded' if arm == 'ENC_coded' else 'ignore_class'] = dict(alarms=alarms, extras=dict(recovery=None))
    key = row['image_key']
    rec = dict(spec=dict(inp['enc_spec']), key=key, detection=dict(size_code_readable=out['shared']['det_info'].get('size_code_readable')), modes=modes)
    p = K.guard(label_dir / f'{key}.json')
    K.save_npz(p.with_suffix('.npz'), **arrays)
    K.save_text(p, json.dumps(K.clean(rec)))
    _r, summ, per_dot = EL.labels(p, ref_img)
    return K.clean(summ), {f'lab_{k}': (np.asarray(v).astype(str) if np.asarray(v).dtype == object else np.asarray(v)) for k, v in per_dot.items()}


def lc_bookkeeping(arms_eval: dict, out: dict) -> dict:

    import lc_pattern as LP
    if 'LC_ignore' not in arms_eval:
        return {}
    ig = arms_eval['LC_ignore']; acc_i = np.asarray(ig['accepted'], bool); lid_i = np.asarray(ig['line_id']); kO = np.asarray(ig['truth_k']); conf = np.asarray(ig['confirmable'], bool)
    seg = np.asarray(ig['segment_id']); true_cls = np.where(conf, LP.class_of_id(np.maximum(kO, 0)), -1)
    res = {}
    for arm, a in arms_eval.items():
        r = out['arms'][arm]['aux']['r']
        acc = np.asarray(a['accepted'], bool); lid = np.asarray(a['line_id'])
        new = acc & ~acc_i; lost = acc_i & ~acc; common = acc & acc_i; changed = common & (lid != lid_i)
        sc = np.asarray(r.get('segment_class', np.full(len(acc), -1)))
        segs = np.unique(seg[seg >= 0]); conf_mat = np.zeros((2, 3), np.int64)
        for s in segs:
            m = (seg == s) & conf
            if not m.any():
                continue
            t = int(round(float(true_cls[m].mean()))); rd = int(sc[seg == s][0]) if (sc[seg == s] >= 0).any() else -1
            conf_mat[t, rd if rd >= 0 else 2] += 1
        res[arm] = dict(n_accepted=int(acc.sum()), correct=int((acc & conf & (lid == kO)).sum()), wrong=int((acc & conf & (lid != kO)).sum()), unconfirmable=int((acc & ~conf).sum()),
                        new=dict(n=int(new.sum()), correct=int((new & conf & (lid == kO)).sum()), wrong=int((new & conf & (lid != kO)).sum()), unconfirmable=int((new & ~conf).sum())),
                        lost=int(lost.sum()), changed=dict(n=int(changed.sum()), to_correct=int((changed & conf & (lid == kO)).sum()), to_wrong=int((changed & conf & (lid != kO)).sum())),
                        image_fallback=bool(r.get('image_fallback', False)), threshold_px=K.clean(r.get('threshold_px')), decision=r.get('decision'),
                        segments_true_x_read=conf_mat.tolist(), segments_true_x_read_def='rows: true class 0 / 1 (majority of confirmable features); columns: read class 0 / 1 / unread')
    return res


def source_identity(img, ref_img, inp: dict, dec: dict, mesh: dict) -> tuple[dict, np.ndarray]:
    import r5_detect as RD
    import sl_lib as L
    import sn_lib as SL
    c = np.asarray(inp['centres'], np.float64); binpat = np.asarray(inp['binary'], np.float32)
    aux = dec['aux']; pts = np.asarray(dec['pts'], np.float32).reshape(-1, 3)
    amb = np.asarray((aux['info'] or {}).get('ambiguous_mask', np.zeros(len(pts), bool)), bool)
    amb = amb if len(amb) == len(pts) else np.zeros(len(pts), bool)
    fin = np.isfinite(pts).all(axis=1) if len(pts) else np.zeros(0, bool); rel_ = fin & ~amb
    code = np.zeros(len(pts), np.int8)
    if not rel_.any():
        return dict(correct=0, wrong=0, undeterminable=0, n_reliable=0, note='no accepted points'), code
    cd = RD.candidates(np.asarray(img, np.uint8))
    if int(cd['n_labels']) != int(aux['cd']['n_labels']) or len(cd['cands']) != len(aux['cd']['cands']):
        raise K.Stop('technical', 'candidate re-detection differs from the stored decode (non-deterministic detection)')
    exp = SL.expected_dots(mesh, c)
    Sm, rinfo = L.reference_map(np.asarray(ref_img, np.uint8), exp['uv'])
    cl = L.classify(cd['label_map'], cd['n_labels'], Sm)
    proj_dot, _ = RD.projector_dot_index(binpat, c)
    tr = aux['tr']
    z = dict(pts=pts, cand_label=np.array([q[5] for q in cd['cands']], np.int32), cand_fidx=np.asarray(aux['sel_fidx']), feat_matched=np.asarray(tr['matched'], bool),
             feat_best_idx=np.asarray(tr['best_idx']), feat_point_idx=np.asarray(tr['point_idx']), proj_dot=np.asarray(proj_dot))
    idn = np.asarray(L.output_identity(z, cl), dtype=object)
    code[idn == 'correct'] = 1; code[idn == 'wrong'] = 2
    lab = idn[rel_]
    sub = {str(k): int((lab == k).sum()) for k in sorted(set(lab.tolist())) if k not in ('correct', 'wrong')}
    return dict(correct=int((lab == 'correct').sum()), wrong=int((lab == 'wrong').sum()), undeterminable=int((~np.isin(lab, ['correct', 'wrong'])).sum()), n_reliable=int(rel_.sum()),
                undeterminable_breakdown=sub, source_labels=dict(n_components=rinfo['n_components'], n_owner_components=rinfo['n_owner_components'], n_ref_merged_dots=rinfo['n_ref_merged_dots'],
                                                                 n_unlabelled_components=rinfo['n_unlabelled_components']),
                definition='sl_lib labels of the sigma-0 image of the same layout / phase / surface (sp5_dot_eval identity); population = finite non-ambiguous points'), code


def displacement(inp: dict) -> dict:
    import sp2_common as S2
    T = inp['table']
    if T is None:
        return {}
    mx = max((abs(v) for v in T['disp'].values()), default=0.0)
    return dict(baseline=T['baseline'], proposal_id=T['proposal_id'], j=T['j'], n_displaced=int(len(T['disp'])), max_abs_disp_px=float(mx), max_abs_disp_mm=float(mx * S2.mmpp()),
                mean_abs_disp_mm=(float(np.mean([abs(v) for v in T['disp'].values()]) * S2.mmpp()) if T['disp'] else 0.0))
