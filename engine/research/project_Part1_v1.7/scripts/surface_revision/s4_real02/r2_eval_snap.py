# Provide the surface-evaluation implementation used by layout comparisons.

from __future__ import annotations

import difflib
import inspect
import math

import numpy as np

import s1r_common as K
import s1r_eval as _E
import sp2_lines_table as _LT
from s1r_eval import _domain_block, _pstats, _ratio, global_surface
from sp2_lines_table import C, S2, inner_margin_px, line_family_table, raster_centrelines_table


def _project(points, ext, boundary: str):

    import l01_common as C0
    import pb_m3 as PM
    g = C0.geom()
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    uv, inside = PM.project_to_camera(pts, g['K'], ext, C0.IMG, C0.IMG, boundary)
    return np.asarray(uv, np.float64), np.asarray(inside, bool)


# Evaluate reconstructed heights, sampling support and source-identity counts.
def evaluate_case(sid: str, family: str, dec: dict, ctx: dict, mesh: dict, dom_loc: dict, dom_glob: dict, gauss: bool, boundary: str = 'legacy'):

    S = K.s2()
    V = S.point_validity(dec['pts'], dec['accepted'], mesh)
    pts, acc, valid, zr = V['pts'], V['acc'], V['valid'], V['zr']
    e_pts = pts[:, 2] - zr
    loc_pts = valid & K.in_local(pts[:, :2]); glob_pts = valid & K.in_global(pts[:, :2], dom_glob)
    aux = dec['aux']; g = S.C0.geom(); info = {}
    arrays = dict(pts=pts, accepted=acc, valid=valid, zr=zr.astype(np.float32), in_local=loc_pts, in_global=glob_pts)
    m = dict(n_out=int(len(pts)), n_accepted=int(acc.sum()), n_valid=int(valid.sum()), n_nonfinite_acc=int((acc & ~V['fin']).sum()), n_out_of_window=int((acc & V['fin'] & ~V['in_win']).sum()),
             n_out_of_grid=int((acc & V['fin'] & ~V['in_grid']).sum()))
    if family == 'point':
        feats = np.asarray(aux['feats'], np.float64).reshape(-1, 2); tr = aux['tr']; cd = aux['cd']
        c = np.asarray(ctx['centres'], np.float64); r_px = float(ctx['size_mm']) / S.SL.mmpp()
        n = len(feats); truth = np.full(n, -1, np.int64); conf = np.zeros(n, bool); dmin = np.full(n, np.nan)
        if n:
            hit, P = S.C0.cast(feats, g['cam'], mesh); uvp, inside = _project(P, g['proj'], boundary)
            d = np.hypot(uvp[:, None, 0] - c[None, :, 0], uvp[:, None, 1] - c[None, :, 1]); j = d.argmin(axis=1); dmin = d[np.arange(n), j]
            conf = hit & inside & (dmin <= r_px + 1.0); truth = np.where(conf, j, -1)
        proj_dot, _ = S.RD.projector_dot_index(np.asarray(ctx['binary'], np.float32), c)
        best = np.asarray(tr['best_idx'], np.int64); matched = np.where(best >= 0, proj_dot[np.clip(best, 0, None)] if len(proj_dot) else -1, -1)
        pidx = np.asarray(tr['point_idx'], np.int64); has = pidx >= 0
        if int(has.sum()) != len(pts):
            raise K.Stop('technical_stop', f'dot correspondence trace: {int(has.sum())} matched features vs {len(pts)} points')
        f_of_p = np.full(len(pts), -1, np.int64); f_of_p[pidx[has]] = np.flatnonzero(has)
        p_truth, p_conf, p_match = truth[f_of_p], conf[f_of_p], matched[f_of_p]
        correct = acc & p_conf & (p_match == p_truth); wrong = acc & p_conf & (p_match != p_truth); unconf = acc & ~p_conf
        exp = S.SL.expected_dots(mesh, c); uv_e = np.asarray(exp['uv'], np.float64); xy_e = np.asarray(exp['xy'], np.float64)
        with np.errstate(invalid='ignore'):
            in_img = np.asarray(exp['hit'], bool) & np.isfinite(uv_e).all(axis=1) & (uv_e[:, 0] >= 0) & (uv_e[:, 0] <= S.IMG - 1) & (uv_e[:, 1] >= 0) & (uv_e[:, 1] <= S.IMG - 1)
        rec_dots = np.zeros(len(c), bool)
        ok_ids = p_truth[correct & valid]
        if len(ok_ids):
            rec_dots[np.unique(ok_ids)] = True
        exp_loc = in_img & K.in_local(xy_e); exp_glob = in_img & K.in_global(xy_e, dom_glob)
        reason = np.asarray(cd['reason']); comp_hist = {str(k): int((reason == k).sum()) for k in (0, 1, 2, 3, 4, 5)}
        code = np.asarray(aux['sel_code']); sel_hist = {str(k): int((code == k).sum()) for k in (0, 1, 2, 3, 4)}
        m.update(n_components=int(cd['n_labels']) - 1, n_candidates=int(len(cd['cands'])), n_selected=int(n), n_matched=int(has.sum()), n_ambiguous=int((has & np.asarray(tr['ambiguous'], bool)).sum()),
                 n_no_correspondence=int(n - has.sum()), n_duplicate=int((has & np.asarray(tr.get('duplicate', np.zeros(n, bool)), bool)).sum()), n_confirmable_features=int(conf.sum()),
                 id_correct=int(correct.sum()), id_wrong=int(wrong.sum()), id_unconf=int(unconf.sum()), id_denominator=int(acc.sum()), mismatch_rate=_ratio(int(wrong.sum()), int(correct.sum() + wrong.sum())),
                 unconf_share=_ratio(int(unconf.sum()), int(acc.sum())), img_n_expected=int(in_img.sum()), img_n_recovered=int((rec_dots & in_img).sum()),
                 img_recovery=_ratio(int((rec_dots & in_img).sum()), int(in_img.sum())), n_recovered_outside_expected=int((rec_dots & ~in_img).sum()), n_features_design=int(len(c)))
        info.update(component_reasons={'0_candidate': comp_hist['0'], '1_abs_area_small': comp_hist['1'], '2_abs_area_large': comp_hist['2'], '3_border': comp_hist['3'], '4_intensity': comp_hist['4'], '5_m00': comp_hist['5']},
                    selection_codes={'0_accepted': sel_hist['0'], '1_rel_area_small': sel_hist['1'], '2_rel_area_large': sel_hist['2'], '3_elongation': sel_hist['3'], '4_group_unavailable': sel_hist['4']},
                    det_info={k: v for k, v in (aux.get('det_info') or {}).items() if not isinstance(v, np.ndarray)},
                    recon_info={k: v for k, v in (aux.get('info') or {}).items() if not isinstance(v, np.ndarray)})
        cands = cd['cands']
        arrays.update(feats=feats.astype(np.float32), cand_uv=np.asarray([[a[0], a[1]] for a in cands], np.float32).reshape(-1, 2), cand_area=np.asarray([a[2] for a in cands], np.int32),
                      cand_elong=np.asarray([a[3] for a in cands], np.float32), cand_label=np.asarray([a[5] for a in cands], np.int32), cand_code=code.astype(np.int8), cand_fidx=np.asarray(aux['sel_fidx'], np.int32),
                      feat_matched=np.asarray(tr['matched'], bool), feat_ambiguous=np.asarray(tr['ambiguous'], bool), feat_duplicate=np.asarray(tr.get('duplicate', np.zeros(n, bool)), bool),
                      feat_best_idx=best.astype(np.int32), feat_point_idx=pidx.astype(np.int32), feat_truth_dot=truth.astype(np.int32), feat_confirmable=conf, feat_matched_dot=matched.astype(np.int32),
                      feat_truth_dist_px=np.asarray(dmin, np.float32), proj_dot=np.asarray(proj_dot, np.int32), point_feature=f_of_p.astype(np.int32), point_correct=correct, point_wrong=wrong, point_unconf=unconf,
                      exp_xy=xy_e.astype(np.float32), exp_uv=uv_e.astype(np.float32), exp_in_image=in_img, exp_recovered=rec_dots)
    else:
        r = aux['r']; feat = np.asarray(r['features'], np.float64).reshape(-1, 2); lid = np.asarray(r['line_id'], np.int64); st = np.asarray(r['status'], np.int64)
        meta = ctx['meta_full']['olines']
        ident = S.T.truth_identity(mesh, feat, meta); kO = np.asarray(ident['k'], np.int64); conf = np.asarray(ident['confirmable'], bool)
        if len(feat) != len(pts):
            raise K.Stop('technical_stop', f'line output: {len(feat)} features vs {len(pts)} points')
        correct = acc & conf & (lid == kO); wrong = acc & conf & (lid != kO); unconf = acc & ~conf
        ref = S.T.line_reference(mesh, meta, 1.0, pattern=np.asarray(ctx['pattern'], np.float32)); R = ref['rows']
        rec_any, rec_id = S._row_flags(ref, feat, lid, acc & valid)
        xyz_r = np.asarray(R['xyz'], np.float64).reshape(-1, 3); exp_all = np.ones(len(xyz_r), bool)
        exp_loc = K.in_local(xyz_r[:, :2]); exp_glob = K.in_global(xyz_r[:, :2], dom_glob)
        rec_dots = rec_id; in_img = exp_all
        from m4_lines import STATUS
        m.update(n_detected=int(len(feat)), n_confirmable_features=int(conf.sum()), **{f'n_status_{STATUS[k]}': int((st == k).sum()) for k in STATUS},
                 id_correct=int(correct.sum()), id_wrong=int(wrong.sum()), id_unconf=int(unconf.sum()), id_denominator=int(acc.sum()), mismatch_rate=_ratio(int(wrong.sum()), int(correct.sum() + wrong.sum())),
                 unconf_share=_ratio(int(unconf.sum()), int(acc.sum())), img_n_expected=int(len(xyz_r)), img_n_recovered=int(rec_id.sum()), img_recovery=_ratio(int(rec_id.sum()), len(xyz_r)),
                 img_n_recovered_any=int(rec_any.sum()), img_recovery_any=_ratio(int(rec_any.sum()), len(xyz_r)), n_features_design=int(meta['n_lines']),
                 n_ref_rows_inner=int(np.asarray(R['inner'], bool).sum()))
        info.update(detect_info={k: v for k, v in (r.get('detect_info') or {}).items() if not isinstance(v, np.ndarray)}, counts=r.get('counts'), checks=r.get('checks'),
                    reference=dict(n_rows=int(len(xyz_r)), per_line=ref['per_line'], centre_source=ref['centre_source']))
        arrays.update(feats=feat.astype(np.float32), status=st.astype(np.int8), line_id=lid.astype(np.int32), truth_k=kO.astype(np.int32), confirmable=conf, chain=np.asarray(r['chain']).astype(np.int32),
                      segment_id=np.asarray(r['segment_id']).astype(np.int32), n_candidates=np.asarray(r['n_candidates']).astype(np.int8), guard_reason=np.asarray(r['guard_reason']).astype(np.int8),
                      point_correct=correct, point_wrong=wrong, point_unconf=unconf, ref_k=np.asarray(R['k'], np.int32), ref_v=np.asarray(R['v'], np.int32), ref_u=np.asarray(R['u'], np.float32),
                      ref_xyz=xyz_r.astype(np.float32), ref_inner=np.asarray(R['inner'], bool), ref_rec_any=rec_any, ref_rec_id=rec_id)
        exp_loc_any, exp_glob_any = exp_loc, exp_glob
        m.update(loc_n_recovered_any=int((exp_loc_any & rec_any).sum()), glob_n_recovered_any=int((exp_glob_any & rec_any).sum()))

    dl = S.roi_domain(mesh) if dom_loc is None else dom_loc
    ev = S.evaluate(pts, acc, mesh, dl, gauss, fields=True)
    z_loc = ev.pop('_z_est')
    if ev['n_valid'] != int(valid.sum()) or ev['n_valid_roi'] != int(loc_pts.sum()):
        raise K.Stop('technical_stop', f"validity bookkeeping differs from s2.evaluate: {ev['n_valid']} / {int(valid.sum())}, {ev['n_valid_roi']} / {int(loc_pts.sum())}")
    r3l = ev.get('r3_dev') or {}; bdl = ((r3l.get('subdomains') or {}).get('global') or {}).get('by_d', {}).get(str(float(K.D_SUPPORT)))
    nl = ev['nodes_roi']
    surf_loc = dict(n_nodes=int(nl['n_nodes']), n_interp=int(nl.get('n', 0) or 0), cov_interp=float(nl['coverage']), Cvalid=(float(bdl['C_valid']) if bdl else 0.0),
                    Ctau=(float(bdl['C_tau'][str(float(K.TAU))]) if bdl else 0.0), surf=dict(rmse=nl.get('rmse'), mae=nl.get('mae'), p95=nl.get('p95_abs'), bias=nl.get('bias')))
    m.update(_domain_block('loc', loc_pts, e_pts, float(S.GROSS_MM), correct, wrong, unconf, exp_loc, rec_dots, surf_loc))
    if abs((m['loc_pt_rmse'] or 0.0) - (ev['pts_roi']['rmse'] or 0.0)) > 1e-12 or m['loc_n_gross'] != ev['n_gross_roi']:
        raise K.Stop('technical_stop', 'local point statistics differ from s2.evaluate')

    zg = dom_glob[f'z_{sid}']
    sg, z_glob = global_surface(pts[valid], dom_glob, zg)
    m.update(_domain_block('glob', glob_pts, e_pts, float(S.GROSS_MM), correct, wrong, unconf, exp_glob, rec_dots, sg))
    am = ev.get('amplitude') or {}
    if gauss:
        m.update(amp_applicable=True, amp_true_mm=am.get('true_mm'), amp_peak_mm=am.get('rec_peak_mm'), amp_centre_mm=am.get('rec_centre_mm'), amp_err_peak_mm=am.get('err_peak_mm'),
                 amp_err_centre_mm=am.get('err_centre_mm'), amp_err_peak_pct=am.get('err_peak_pct'))
    else:
        m.update(amp_applicable=False)
    m['state'] = 'no_output' if m['n_valid'] == 0 else 'ok'
    info['local_r3'] = dict(version=r3l.get('version'), triangulation=r3l.get('triangulation'), status=r3l.get('status')); info['global_r3'] = sg['r3']
    info['amplitude_definition'] = am.get('definition') if gauss else 'not applicable (no Gaussian reference term)'
    return m, arrays, info, dict(z_loc=z_loc, z_glob=z_glob)


def line_reference_table(mesh: dict, spec: dict, step_px: float = 1.0, border_px: int = 3, pattern: np.ndarray | None = None, inner_px: float | None = None, boundary: str = 'legacy') -> dict:

    fam = line_family_table(spec); g = C.geom()
    inner_px = inner_margin_px(spec) if inner_px is None else float(inner_px)
    centrelines = raster_centrelines_table(pattern, fam) if pattern is not None else None
    rk, rv, ru, rxyz, rt, rw, rin = [], [], [], [], [], [], []; g_xyz = []; per_line = {}
    for k in [int(x) for x in fam['ids']]:
        seg = spec['segments'][str(k)]
        a = np.asarray(seg['start_uv'], np.float64); b = np.asarray(seg['end_uv'], np.float64); L = float(seg['length_px'])
        if L <= 0:
            continue
        if centrelines is not None:
            cl = centrelines[int(k)]
            if len(cl['rows']) < 3:
                continue
            P = np.column_stack([cl['u'], cl['rows'].astype(np.float64)])
        else:
            lam = np.arange(0.0, L + 1e-9, step_px); P = a + lam[:, None] * (b - a) / L
        hit, pts = C.cast(P, g['proj'], mesh)
        uv, inside = _project(pts, g['cam'], boundary)
        ok = hit & inside
        rec = dict(n_samples=int(len(P)), n_hit=int(hit.sum()), n_in_image=int(ok.sum()), n_rows=0)
        if ok.sum() >= 3:
            uvh, xyzh = uv[ok], pts[ok]; g_xyz.append(xyzh)
            order = np.argsort(uvh[:, 1]); uvh, xyzh = uvh[order], xyzh[order]
            if np.all(np.diff(uvh[:, 1]) > 0) and (uvh[-1, 1] - uvh[0, 1]) >= 2.0:
                rows = np.arange(math.ceil(uvh[0, 1]), math.floor(uvh[-1, 1]) + 1)
                u = np.interp(rows, uvh[:, 1], uvh[:, 0])
                xyz = np.column_stack([np.interp(rows, uvh[:, 1], xyzh[:, j]) for j in range(3)])
                d = np.gradient(uvh, axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                t = np.column_stack([np.interp(rows, uvh[:, 1], d[:, j]) for j in range(2)]); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
                w_arc = 1.0 / np.maximum(np.abs(t[:, 1]), 1e-6)
                inner = (u >= inner_px) & (u <= C.IMG - 1 - inner_px) & (rows >= border_px) & (rows <= C.IMG - 1 - border_px)
                rk.append(np.full(len(rows), k)); rv.append(rows); ru.append(u); rxyz.append(xyz); rt.append(t); rw.append(w_arc); rin.append(inner)
                rec['n_rows'] = int(len(rows))
        per_line[str(k)] = rec
    cat = lambda L_, shape=None: (np.concatenate(L_) if L_ else np.zeros((0,) + (shape or ())))
    rows = dict(k=cat(rk).astype(np.int64), v=cat(rv).astype(np.int64), u=cat(ru), xyz=cat(rxyz, (3,)), t=cat(rt, (2,)), w_arc=cat(rw), inner=cat(rin).astype(bool))
    return dict(rows=rows, g_xyz=(np.concatenate(g_xyz) if g_xyz else np.zeros((0, 3))), per_line=per_line, step_px=float(step_px), n_lines=int(len(fam['ids'])),
                centre_source=('raster' if centrelines is not None else 'analytic'), inner_margin_px=inner_px, adapter=S2.ADAPTER_VERSION)


def snapshot_diff() -> dict:
    def lines(f):
        return [l.rstrip() for l in inspect.getsource(f).splitlines()]
    out = {}
    for name, a, b in (('evaluate_case', _E.evaluate_case, evaluate_case), ('line_reference_table', _LT.line_reference_table, line_reference_table)):
        out[name] = [l for l in difflib.unified_diff(lines(a), lines(b), lineterm='', n=0) if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    return out
