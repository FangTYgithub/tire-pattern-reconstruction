# Render and decode encoding cases and store feature-level diagnostics.

import json, time, traceback
from pathlib import Path
import numpy as np

import ep_common as C
import ep_patterns as EP
import ep_detect as ED
import sl_lib as L
import sn_lib as SL
import r5_detect as RD
import sn_posthoc as PH
import sa_defs as SAD
import pm014_surface as S14

NOISE_BASE = 9460000
KEY_FIELDS = ('enc', 'layout', 'arm', 'p_par', 'p_perp', 'k', 'ratio', 'rotation_deg', 'code_shift', 'r_mm', 'pos_index', 'off_u', 'off_v', 'surface', 'fwhm',
              'sigma', 'kseed', 'psf_px', 'regime', 'budget', 'noise_base', 'code_ref')
STEP = L.STEP
ALARM = dict(rej_frac=0.10, offset_mm=1.5, neg_frac=0.30)


def spec(a, pos_index, surface, sigma, kseed, psf_px, regime, fwhm=20.0, budget=EP.BUDGET_LIT_PX, code_ref='global'):
    u, v = L.positions()[pos_index]
    return dict(enc=1, layout=a['layout'], arm=a['arm'], p_par=a['p_par'], p_perp=a['p_perp'], k=int(a['k']), ratio=float(a['ratio']), rotation_deg=float(a['rotation_deg']),
                code_shift=int(a['code_shift']), r_mm=EP.R_L, radii_mm=list(a['radii_mm']), pos_index=int(pos_index), off_u=float(u), off_v=float(v), surface=surface,
                fwhm=float(fwhm), sigma=float(sigma), kseed=int(kseed) if sigma > 0 else 0, psf_px=float(psf_px), regime=regime, budget=float(budget),
                noise_base=NOISE_BASE, code_ref=code_ref)


def key(s):
    return C.digest({kk: s[kk] for kk in KEY_FIELDS})


def noise_seed(s):
    return int(s['noise_base'] + 10 * s['pos_index'] + s['kseed'])


def _f(v):
    return float(v) if v is not None and np.isfinite(v) else None


def image(mesh, pat, sigma, seed, psf_px):
    from m3_virtual_camera import synthesise_camera_image
    import preliminary_analysis as PA
    m3_cfg = {'batch_size': PA.BATCH_SIZE, 'noise_sigma': float(sigma), 'psf_sigma_px': float(psf_px), 'albedo': 1.0, 'albedo_exposure': 1.0,
              'specular_k': 0.0, 'mottle_amp': 0.0, 'outlier_frac': 0.0, 'mottle_seed': int(seed)}
    img, _, _, m3_info = synthesise_camera_image(mesh_dict=mesh, pattern=pat, cfg=m3_cfg)
    return img, m3_info


def decode(img, binpat, s):

    t0 = time.monotonic()
    cd = RD.candidates(img)
    k = int(s['k']); radii_px = [r / EP.mmpp() for r in s['radii_mm']]
    if k == 1:
        feats, det_info, code, fidx, _ref = RD.select(cd, 'D0')
        cfg = dict(ED.CFG); ptype = 'dots'; allowed = np.ones((len(feats), 1), bool); cinfo = dict(readable=None)
    else:
        feats, det_info, code, fidx, cinfo = ED.select_coded(cd, k, radii_px, s.get('code_ref', 'global'))
        cfg = ED.coded_cfg(s['radii_mm'], k, EP.mmpp()); ptype = 'coded_dots'; allowed = det_info['feature_allowed']
    t_det = time.monotonic() - t0
    modes = {}
    t1 = time.monotonic()
    pts, info, tr = ED.reconstruct_from(feats, det_info, binpat, cfg, ptype, allowed if k > 1 else None)
    modes['coded'] = dict(pts=pts, info=info, tr=tr, seconds=time.monotonic() - t1)
    if k > 1:
        t2 = time.monotonic()
        pts_i, info_i, tr_i = ED.reconstruct_from(feats, det_info, binpat, cfg, ptype, np.ones_like(allowed))
        modes['ignore_class'] = dict(pts=pts_i, info=info_i, tr=tr_i, seconds=time.monotonic() - t2)
    proj = ED.projector_side(binpat, cfg, ptype)
    cand = ED.track_candidates(feats, proj, cfg, allowed if k > 1 else None)
    return dict(cd=cd, feats=feats, det_info=det_info, code=code, fidx=fidx, cinfo=cinfo, cfg=cfg, ptype=ptype, allowed=allowed, modes=modes, proj=proj, cand=cand,
                seconds_detect=t_det)


# Calculate reconstruction errors and recovery descriptors for the decoded case.
def extras(pts, info, s, mesh, cfg, ref, exp, img, m3_info, binpat, pat, v, c):

    from m5_evaluation import evaluate_all
    import preliminary_analysis as PA
    t0 = time.monotonic()
    m = evaluate_all(points_3d=pts, mesh=mesh, camera_image=img, m1_config=cfg, m3_info=m3_info, uniq_window_px=PA.UNIQ_WINDOW_PX, uniq_stride_px=PA.UNIQ_STRIDE_PX,
                     uniq_n_samples=PA.UNIQ_N_SAMPLES, spacing_mm=s['p_par'], theta_deg=None, recon_info=info)
    t_m5 = time.monotonic() - t0
    pts = np.asarray(pts, np.float32)
    amb = np.asarray(info.get('ambiguous_mask', np.zeros(len(pts), bool)), bool)
    if len(amb) != len(pts):
        amb = np.zeros(len(pts), bool)
    fin = np.isfinite(pts).all(axis=1); rel = fin & ~amb; n_rel = int(rel.sum())
    gt = SL.truth_depth(pts, mesh).astype(np.float32)
    zmin = float(np.asarray(mesh['Z']).min())
    dref = np.full(len(pts), np.nan)
    if fin.any():
        dref[fin] = S14.z_ref_at(ref, cfg, pts[fin, 0], pts[fin, 1]) - zmin
    err_ref = np.where(fin, pts[:, 2] - dref, np.nan).astype(np.float32)
    r_rel = err_ref[rel & np.isfinite(err_ref)]
    mc = SL.clean(m)
    in_disc = fin & ((pts[:, 0].astype(np.float64) ** 2 + pts[:, 1].astype(np.float64) ** 2) <= SL.DENT_R_MM ** 2)
    ok_ref = np.isfinite(err_ref) & (np.abs(np.nan_to_num(err_ref, nan=1e9)) <= SL.OUTLIER_MM)
    n_exp_disc = int(exp['in_disc'].sum())
    d_err = mc.get('ind_depth_err_mm'); d_gt = mc.get('ind_depth_gt_mm'); d_pts = mc.get('ind_depth_rec_pts_mm')
    exp_disc = exp['xy'][exp['in_disc']].astype(np.float64)
    rec, nn_rec, n_q = PH.recovery(pts, amb, err_ref, exp_disc, L.GATE_XY, L.GATE_Z)
    good = rel & ok_ref
    dn_real, _ = L.disc_gaps(pts[good & in_disc][:, :2].astype(np.float64))
    ex = dict(n_detected=int(info.get('n_detected', 0) or 0), n_matched=int(info.get('n_matched', 0) or 0), n_ambiguous=int(info.get('n_ambiguous', 0) or 0),
              n_rejected=int(info.get('n_rejected', 0) or 0), n_duplicate_claims=int(info.get('n_duplicate_claims', 0) or 0), n_finite=int(fin.sum()), n_rel=n_rel,
              A=_f(mc.get('A_ambiguity')), data_valid=bool(mc.get('data_valid')), n_rel_disc=int((in_disc & rel).sum()), n_good_disc=int((in_disc & good).sum()),
              completeness=(int((in_disc & rel).sum()) / n_exp_disc) if n_exp_disc else None,
              ref_rmse_rel_mm=float(np.sqrt(np.mean(r_rel ** 2))) if len(r_rel) else None, ref_p95_rel_mm=float(np.percentile(np.abs(r_rel), 95)) if len(r_rel) else None,
              ref_bias_rel_mm=float(np.mean(r_rel)) if len(r_rel) else None, ref_median_rel_mm=float(np.median(r_rel)) if len(r_rel) else None,
              gross_rate_rel=float((np.abs(r_rel) > SL.OUTLIER_MM).mean()) if len(r_rel) else None, n_gross_rel=int((np.abs(r_rel) > SL.OUTLIER_MM).sum()),
              recovery=None if not np.isfinite(rec) else float(rec), recovery_nn_dedup=None if not np.isfinite(nn_rec) else float(nn_rec), n_qualified_outputs=int(n_q),
              cover_radius_realised_mm=float(dn_real.max()) if dn_real is not None else None,
              neg_depth_frac=float((pts[fin, 2] < 0).mean()) if fin.any() else None,
              dent_err_mm=d_err, dent_rel_pct=(100.0 * d_err / d_gt) if (d_err is not None and d_gt) else None,
              dent_pts_rel_pct=(100.0 * (d_pts - d_gt) / d_gt) if (d_pts is not None and d_gt) else None, dent_interp_coverage=mc.get('ind_coverage_frac'),
              search_depth_mm=_f(info.get('search_depth_mm')), search_depth_source=info.get('search_depth_source'), seconds_m5=round(t_m5, 3))
    ex.update({kk + '_rel': val for kk, val in SL.occupancy(pts, rel).items()})
    st = SL.status_of(mc, n_rel)
    T = L.task_criterion(ex, ex['data_valid'], s['sigma'])

    off = ex['ref_median_rel_mm']
    alarms = dict(rej=bool(ex['n_detected'] and ex['n_rejected'] / ex['n_detected'] > ALARM['rej_frac']),
                  offset=bool(off is not None and abs(off) > ALARM['offset_mm']),
                  neg=bool(ex['neg_depth_frac'] is not None and ex['neg_depth_frac'] > ALARM['neg_frac']))
    alarms['any'] = bool(alarms['rej'] or alarms['offset'] or alarms['neg'])
    return dict(metrics=mc, extras=ex, status=st, legacy=SAD.layers(st, ex, ex['recovery'], s['sigma']), task=T, alarms=alarms,
                arrays=dict(pts=pts, amb=amb, gt_depth=gt, gt_ref=dref.astype(np.float32), err_ref=err_ref))


# Render an encoding case and run its requested decoding arms.
def work(s, base=None):
    t0 = time.monotonic(); k = key(s); base = base or C.CASES
    try:
        mesh, cfg, ref = L.surface(s['surface'], s['fwhm'])
        a = dict(layout=s['layout'], arm=s['arm'], p_par=s['p_par'], p_perp=s['p_perp'], k=s['k'], ratio=s['ratio'], rotation_deg=s['rotation_deg'], code_shift=s['code_shift'],
                 radii_mm=s['radii_mm'])
        c, cls, binpat = EP.build(a, s['off_u'], s['off_v'])
        res = EP.resources(c, cls, s['radii_mm'], binpat, s['budget'])
        v = 1.0 if s['regime'] == 'A' else float(res['v_equal_integral'])
        if v > 1.0 + 1e-9:
            raise RuntimeError(f'dot value {v} exceeds 1 (budget not realisable)')
        pat = (binpat * np.float32(v)).astype(np.float32)
        pchk = SL.pattern_checks(binpat, c); gp = EP.gaps(c, cls, s['radii_mm'])
        exp = SL.expected_dots(mesh, c)
        t1 = time.monotonic()
        img, m3_info = image(mesh, pat, s['sigma'], noise_seed(s), s['psf_px'])
        t_m3 = time.monotonic() - t1
        D = decode(img, binpat, s)
        proj_dot, proj_dist = RD.projector_dot_index(binpat, c)
        out_modes = {}
        for mode, M in D['modes'].items():
            out_modes[mode] = extras(M['pts'], M['info'], s, mesh, cfg, ref, exp, img, m3_info, binpat, pat, v, c)
            out_modes[mode]['seconds_match'] = round(M['seconds'], 3)
        di = D['det_info']
        det = dict(image_sat_frac=float((img >= 255).mean()), otsu_threshold=_f(di.get('otsu_threshold')), n_components=int(D['cd']['n_labels'] - 1),
                   n_candidates=int(len(D['cd']['cands'])), n_detected=int(len(D['feats'])), n_area_rejected=int(di.get('n_area_rejected', 0) or 0),
                   n_elong_rejected=int(di.get('n_elong_rejected', 0) or 0), n_filtered_noise=int(di.get('n_filtered_noise', 0) or 0),
                   n_border_excluded=int(di.get('n_border_excluded', 0) or 0), blob_area_median_px=_f(di.get('blob_area_median_px')),
                   size_code_readable=di.get('size_code_readable'), size_class_dprime=_f(di.get('size_class_dprime')) if di.get('size_class_dprime') is not None else None,
                   size_class_dprime_adjacent=di.get('size_class_dprime_adjacent'), size_class_balance=di.get('size_class_balance'), size_class_ratio=di.get('size_class_ratio'),
                   size_class_counts=di.get('size_class_counts'), n_size_uncertain=di.get('n_size_uncertain'), size_log_span_design=di.get('size_log_span_design'),
                   size_log_span_measured=di.get('size_log_span_measured'), n_proj_features=int(len(proj_dot)), proj_dot_max_dist_px=float(proj_dist.max()) if len(proj_dist) else None,
                   cand_n_geom_median=float(np.median(D['cand']['n_geom'])) if len(D['feats']) else None, cand_n_class_median=float(np.median(D['cand']['n_class'])) if len(D['feats']) else None,
                   cand_n_geom_gt1=int((D['cand']['n_geom'] > 1).sum()), cand_n_class_gt1=int((D['cand']['n_class'] > 1).sum()), cand_n_geom_0=int((D['cand']['n_geom'] == 0).sum()))
        timing = dict(seconds_m3=round(t_m3, 3), seconds_detect=round(D['seconds_detect'], 3), seconds_total=None, peak_rss_mb=SL.peak_rss_mb())
        arrays = dict(cam_img=img.astype(np.uint8), pattern_bin=(binpat > 0).astype(np.uint8), centres_px=c.astype(np.float32), centre_class=cls.astype(np.int8),
                      exp_xy=exp['xy'].astype(np.float32), exp_uv=exp['uv'].astype(np.float32), exp_in_disc=exp['in_disc'], exp_cam_inner=exp['cam_inner'],
                      cand_uv=np.array([[q[0], q[1]] for q in D['cd']['cands']], np.float32).reshape(-1, 2), cand_area=np.array([q[2] for q in D['cd']['cands']], np.int32),
                      cand_label=np.array([q[5] for q in D['cd']['cands']], np.int32), cand_code=D['code'].astype(np.int8), cand_fidx=D['fidx'].astype(np.int32),
                      cand_class=np.asarray(D['cinfo'].get('cls_all', np.zeros(len(D['cd']['cands']), np.int64)), np.int8),
                      comp_reason=D['cd']['reason'].astype(np.int8), feat_class=np.asarray(D['det_info'].get('feature_classes', np.zeros(len(D['feats']), np.int64)), np.int8),
                      feat_allowed=np.asarray(D['allowed'], bool), feat_n_geom=D['cand']['n_geom'].astype(np.int16), feat_n_class=D['cand']['n_class'].astype(np.int16),
                      proj_dot=proj_dot.astype(np.int32), proj_class=D['proj']['cls'].astype(np.int8), proj_ids=D['proj']['ids'].astype(np.int32))
        feat_geom_ids = D['cand']['geom_ids']; feat_class_ids = D['cand']['class_ids']
        arrays['feat_geom_ids'] = np.array([','.join(map(str, g)) for g in feat_geom_ids], dtype=object) if len(D['feats']) else np.zeros(0, dtype=object)
        arrays['feat_class_ids'] = np.array([','.join(map(str, g)) for g in feat_class_ids], dtype=object) if len(D['feats']) else np.zeros(0, dtype=object)
        for mode, M in D['modes'].items():
            suf = '' if mode == 'coded' else '_ig'
            tr = M['tr']
            arrays.update({f'feat_matched{suf}': tr['matched'], f'feat_ambiguous{suf}': tr['ambiguous'], f'feat_duplicate{suf}': tr['duplicate'],
                           f'feat_best_idx{suf}': tr['best_idx'].astype(np.int32), f'feat_point_idx{suf}': tr['point_idx'].astype(np.int32)})
            arrays.update({f'{kk}{suf}': val for kk, val in out_modes[mode].pop('arrays').items()})
        res_all = dict(spec=s, key=k, status=out_modes['coded']['status'], legacy=out_modes['coded']['legacy'], task=out_modes['coded']['task'],
                       unavailable=bool(out_modes['coded']['extras']['n_finite'] == 0 or out_modes['coded']['extras']['dent_rel_pct'] is None),
                       seconds=round(time.monotonic() - t0, 3), finished_utc=C.utcnow(), pattern=dict(n_dots=int(len(c)), n_per_class=res['n_per_class'], lit_px=res['lit_px'],
                       lit_px_per_class=res['lit_px_per_class'], dot_value=float(v), pattern_integral=float(pat.sum(dtype=np.float64)), budget_lit_px=float(s['budget']),
                       regime=s['regime'], radii_mm=s['radii_mm'], radii_px=[r / EP.mmpp() for r in s['radii_mm']], pattern_checks=pchk, gaps=gp,
                       n_exp_disc=int(exp['in_disc'].sum()), n_exp_cam_inner=int(exp['cam_inner'].sum()), n_exp_visible=int(np.isfinite(exp['uv']).all(axis=1).sum()),
                       noise_seed_m3=noise_seed(s), noise_rng_seed=noise_seed(s) + 104729, psf_px=s['psf_px']),
                       detection=det, modes=out_modes, timing=timing, arrays=arrays)
        res_all['timing']['seconds_total'] = round(time.monotonic() - t0, 3)
        return res_all
    except Exception:
        return dict(spec=s, key=k, status='error', seconds=time.monotonic() - t0, finished_utc=C.utcnow(), error=traceback.format_exc(), metrics=None)


def store(r, base=None):
    base = base or C.CASES
    p = base / f"{r['key']}.json"; p.parent.mkdir(parents=True, exist_ok=True)
    arrays = r.pop('arrays', None)
    if arrays is not None:
        tmp = p.with_suffix('.npz.tmp')
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(p.with_suffix('.npz')); r['npz'] = p.with_suffix('.npz').name
    C.save_json(p, r)
    return p


def ref_key(s):
    return key(dict(s, sigma=0.0, kseed=0))


def _view(z, suf):
    return dict(exp_uv=z['exp_uv'], cand_label=z['cand_label'], cand_code=z['cand_code'], cand_fidx=z['cand_fidx'], feat_matched=z['feat_matched' + suf],
                feat_best_idx=z['feat_best_idx' + suf], feat_ambiguous=z['feat_ambiguous' + suf], feat_point_idx=z['feat_point_idx' + suf], proj_dot=z['proj_dot'],
                err_ref=z['err_ref' + suf], pts=z['pts' + suf])


# Compare detected and assigned classes with the projected source identities.
def labels(case_json, ref_img):

    r = json.loads(Path(case_json).read_text(encoding='utf-8'))
    z = dict(np.load(Path(case_json).with_suffix('.npz'), allow_pickle=True))
    cd = RD.candidates(z['cam_img'])
    Sm, rinfo = L.reference_map(ref_img, z['exp_uv'].astype(float))
    cl = L.classify(cd['label_map'], cd['n_labels'], Sm)
    n = len(z['exp_uv']); disc = z['exp_in_disc'].astype(bool); fin_uv = np.isfinite(z['exp_uv']).all(axis=1); inner = z['exp_cam_inner'].astype(bool) & fin_uv
    true_cls = z['centre_class'].astype(int)
    lab_to_cand = {int(l): j for j, l in enumerate(z['cand_label'])}
    read_cls = np.full(n, -1, np.int8); uncertain = np.zeros(n, bool); comp_cls = np.full(n, -1, np.int8); cand_of_dot = np.full(n, -1, np.int32)
    k = int(r['spec']['k'])
    for i in range(n):
        if i in cl['cover']:
            Lc, _ = cl['cover'][i]; comp_cls[i] = cl['cls'][Lc]
            j = lab_to_cand.get(Lc)
            if j is not None:
                cand_of_dot[i] = j
                if k > 1:
                    read_cls[i] = z['cand_class'][j]
                    f = int(z['cand_fidx'][j])
                    if f >= 0:
                        uncertain[i] = bool(z['feat_allowed'][f].sum() > 1)
    modes = ['coded'] + (['ignore_class'] if 'feat_matched_ig' in z else [])
    summ = dict(reference=rinfo, component_classes={L.CLS[c]: int((cl['cls'][1:] == c).sum()) for c in L.CLS}, n_exp=n, n_exp_visible=int(inner.sum()), n_exp_disc=int(disc.sum()),
                size_code_readable=r['detection'].get('size_code_readable'), guard_fallback=(bool(r['detection'].get('size_code_readable') == 0) if k > 1 else None))
    arrays = dict(true_class=true_cls.astype(np.int8), read_class=read_cls, class_uncertain=uncertain, comp_class=comp_cls, cand_of_dot=cand_of_dot)

    percls = {}
    for cc in range(k):
        sel = inner & (true_cls == cc)
        acc = sel & (cand_of_dot >= 0) & np.isin(cand_of_dot, np.where(z['cand_fidx'] >= 0)[0])
        single = sel & (comp_cls == 1)
        percls[int(cc)] = dict(n_visible=int(sel.sum()), n_component=int((sel & (comp_cls >= 1)).sum()), n_single_source=int(single.sum()), n_merged=int((sel & (comp_cls == 3)).sum()),
                               n_partial=int((sel & (comp_cls == 2)).sum()), n_no_component=int((sel & (comp_cls < 1)).sum()), n_candidate_accepted=int(acc.sum()),
                               n_rejected_by_window=int((sel & (cand_of_dot >= 0) & ~acc).sum()),
                               n_read_correct=int((single & acc & (read_cls == cc)).sum()) if k > 1 else None, n_read_wrong=int((single & acc & (read_cls >= 0) & (read_cls != cc)).sum()) if k > 1 else None,
                               n_uncertain=int((single & acc & uncertain).sum()) if k > 1 else None)
    summ['per_true_class'] = percls
    if k > 1:
        conf = np.zeros((k, k), np.int64)
        m = inner & (comp_cls == 1) & (read_cls >= 0)
        for t, rr in zip(true_cls[m], read_cls[m]):
            conf[t, rr] += 1
        summ['confusion_true_x_read_single_source'] = conf.tolist()
    for mode in modes:
        suf = '' if mode == 'coded' else '_ig'
        zz = _view(z, suf)
        out, amb_dot, err_dot = L.dot_outcomes(zz, cd, cl)
        ident = L.output_identity(zz, cl)
        good = (out == STEP.index('correct')) & ~amb_dot & np.isfinite(err_dot) & (np.abs(err_dot) <= SL.OUTLIER_MM)
        pts = zz['pts']; fin = np.isfinite(pts).all(axis=1); amb = z['amb' + suf].astype(bool); rel = fin & ~amb
        n_acc = int(fin.sum()); n_rel = int(rel.sum())
        wrong_pt = (ident == 'wrong')
        idc = {kk: int((ident == kk).sum()) for kk in ('correct', 'wrong', 'background', 'undeterminable_partial', 'undeterminable_multi', 'undeterminable_unlabelled', 'none')}
        steps_full = {nm: int(((out == i) & inner).sum()) for i, nm in enumerate(STEP)}
        steps_disc = {nm: int(((out == i) & disc).sum()) for i, nm in enumerate(STEP)}
        alarms = r['modes'][mode]['alarms']
        n_wrong_rel = int((wrong_pt & rel).sum())
        summ[mode] = dict(steps_full_visible=steps_full, steps_disc=steps_disc, closure_full=bool(sum(steps_full.values()) == int(inner.sum())),
                          closure_disc=bool(sum(steps_disc.values()) == int((disc & fin_uv).sum())),
                          identity_recovery_full=(float((good & inner).sum() / inner.sum()) if inner.sum() else None),
                          identity_recovery_disc=(float((good & disc).sum() / disc.sum()) if disc.sum() else None),
                          positional_recovery_disc=r['modes'][mode]['extras'].get('recovery'),
                          n_accepted=n_acc, n_reliable=n_rel, n_wrong_identity=int(wrong_pt.sum()), n_wrong_identity_reliable=n_wrong_rel,
                          wrong_over_accepted=(float(wrong_pt.sum() / n_acc) if n_acc else None), wrong_over_reliable=(float(n_wrong_rel / n_rel) if n_rel else None),
                          output_identity=idc, frame_alarm=alarms, silent_frame=bool(n_wrong_rel > 0 and not alarms['any']),
                          n_truth_excluded_by_class=None)
        arrays.update({f'dot_step{suf}': out, f'dot_ambiguous{suf}': amb_dot, f'dot_err_ref{suf}': err_dot, f'output_identity{suf}': ident.astype(str)})

    if k > 1 and 'feat_geom_ids' in z:
        proj_dot = z['proj_dot'].astype(int); dot_to_proj = {int(d): p for p, d in enumerate(proj_dot)}
        pid = z['proj_ids'].astype(int)
        n_excl = n_geom_has = 0
        for i in range(n):
            j = int(cand_of_dot[i]); f = int(z['cand_fidx'][j]) if j >= 0 else -1
            if f < 0 or comp_cls[i] != 1 or not inner[i] or i not in dot_to_proj:
                continue
            tid = int(pid[dot_to_proj[i]])
            g = set(int(x) for x in str(z['feat_geom_ids'][f]).split(',') if x != ''); a = set(int(x) for x in str(z['feat_class_ids'][f]).split(',') if x != '')
            if tid in g:
                n_geom_has += 1
                if tid not in a:
                    n_excl += 1
        summ['coded']['n_truth_in_geometric_candidates'] = n_geom_has; summ['coded']['n_truth_excluded_by_class'] = n_excl
    return r, summ, arrays
