# Apply D0, DF and smoothed dot detection and diagnose class readability.

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

import s3c_common as K

RULES_DOTS = ('B0', 'DF', 'DF025', 'DS')
FLOOR = {'DF': 0.5, 'DF025': 0.25}
SMOOTH_PX = 1.0
LB_SIZES = {'LB': 8, 'LB4': 4, 'LB16': 16}
LG_RATIO = (0.5, 1.5)
LG_MARGIN = 0.15
THR_OLD = 6.05
W_SNAP = dict(narrow=0.75 / 0.18, wide=1.25 / 0.18)


def _P():
    import rdy_platform as P
    return P


def smooth_u8(img, s=SMOOTH_PX):
    from scipy.ndimage import gaussian_filter
    return np.clip(np.rint(gaussian_filter(np.asarray(img, np.float64), s)), 0, 255).astype(np.uint8)


def dot_candidates(img, rule: str, r_min_mm: float) -> dict:
    import r5_detect as RD
    im = smooth_u8(img) if rule == 'DS' else np.asarray(img, np.uint8)
    cd = RD.candidates(im)
    cd['n_floor'], cd['a_floor'] = 0, None
    if rule in FLOOR:
        a = FLOOR[rule] * math.pi * (float(r_min_mm) / K.NOMINAL_MM_PER_PX) ** 2
        keep = [c for c in cd['cands'] if c[2] >= a]
        cd = dict(cd, cands=keep, n_floor=len(cd['cands']) - len(keep), a_floor=a)
    return cd


def dot_decode(img, binpat, k: int, radii_mm, rule: str, path: str) -> dict:

    import ep_detect as ED
    import ep_patterns as EP
    import r5_detect as RD
    import rdy_decode as DC
    binpat = np.asarray(binpat, np.float32)
    cd = dot_candidates(img, rule, min(radii_mm))
    out = dict(cd=cd, rule=rule)
    if k == 1:
        feats, det_info, code, fidx, _ = RD.select(cd, 'D0')
        if path == 'binding':
            cfg = _P().s2().dot_cfg(); pts, info, tr = RD.reconstruct_from(feats, det_info, binpat, cfg=cfg)
        else:
            cfg = DC.window_cfg_dots(dict(ED.CFG)); pts, info, tr = ED.reconstruct_from(feats, det_info, binpat, cfg, 'dots', None)
        out.update(feats=feats, det_info=det_info, code=code, fidx=fidx, cinfo=None, cfg=cfg, ptype='dots', allowed=None, arms={'single': (pts, info, tr)})
    else:
        radii_px = [float(r) / EP.mmpp() for r in radii_mm]
        feats, det_info, code, fidx, cinfo = ED.select_coded(cd, k, radii_px, 'global')
        cfg = DC.window_cfg_dots(ED.coded_cfg(radii_mm, k, EP.mmpp())); allowed = det_info['feature_allowed']
        arms = {'use': ED.reconstruct_from(feats, det_info, binpat, cfg, 'coded_dots', allowed),
                'ignore': ED.reconstruct_from(feats, det_info, binpat, cfg, 'coded_dots', np.ones_like(allowed))}
        out.update(feats=feats, det_info=det_info, code=code, fidx=fidx, cinfo=cinfo, cfg=cfg, ptype='coded_dots', allowed=allowed, arms=arms,
                   guard=DC.guard_reasons(cinfo, radii_px))
    return out


def dot_dec(D: dict, arm: str) -> dict:
    import rdy_decode as DC
    pts, info, tr = D['arms'][arm]
    return dict(pts=np.asarray(pts, np.float64).reshape(-1, 3), accepted=DC._accept(pts, tr),
                aux=dict(feats=D['feats'], tr=tr, cd=D['cd'], sel_code=D['code'], sel_fidx=D['fidx'], det_info=D['det_info'], info=info))


def dot_truth(feats, ctx: dict, mesh) -> tuple[np.ndarray, np.ndarray]:

    S = _P().s2(); g = S.C0.geom(); c = np.asarray(ctx['centres'], np.float64); r_px = float(ctx['size_mm']) / S.SL.mmpp()
    n = len(feats); truth = np.full(n, -1, np.int64); conf = np.zeros(n, bool)
    if n:
        hit, Pp = S.C0.cast(np.asarray(feats, np.float64).reshape(-1, 2), g['cam'], mesh); uvp, inside = S.C0.project(Pp, g['proj'])
        d = np.hypot(uvp[:, None, 0] - c[None, :, 0], uvp[:, None, 1] - c[None, :, 1]); j = d.argmin(axis=1); dmin = d[np.arange(n), j]
        conf = hit & inside & (dmin <= r_px + 1.0); truth = np.where(conf, j, -1)
    return truth, conf


def dot_truth_arms(D: dict, binpat, ctx: dict, mesh, classes) -> dict:

    import ep_detect as ED
    import m4_reconstruction as M4
    import r5_detect as RD
    feats = D['feats']; truth, conf = dot_truth(feats, ctx, mesh); out = {}
    binpat = np.asarray(binpat, np.float32)
    if D['ptype'] == 'coded_dots':
        k = D['allowed'].shape[1]; allowed = np.ones((len(feats), k), bool)
        cls = np.asarray(classes, np.int64)
        allowed[conf] = False; allowed[np.flatnonzero(conf), cls[truth[conf]]] = True
        out['TC'] = ED.reconstruct_from(feats, D['det_info'], binpat, D['cfg'], 'coded_dots', allowed)
    proj_pts, _ = M4.pattern_features(binpat, 'dots'); proj_dot, _ = RD.projector_dot_index(binpat, np.asarray(ctx['centres'], np.float64))
    dot2p = {int(d): p for p, d in enumerate(proj_dot)}
    sel = np.array([bool(conf[i]) and int(truth[i]) in dot2p for i in range(len(feats))], bool)
    n = len(feats); tr = dict(matched=sel.copy(), ambiguous=np.zeros(n, bool), duplicate=np.zeros(n, bool), best_idx=np.full(n, -1, np.int64), point_idx=np.full(n, -1, np.int64))
    pts = np.zeros((0, 3), np.float32)
    if sel.any():
        p = np.array([dot2p[int(truth[i])] for i in np.flatnonzero(sel)]); tr['best_idx'][sel] = p; tr['point_idx'][sel] = np.arange(int(sel.sum()))
        pts = M4.triangulate(np.asarray(feats, np.float32)[sel], np.asarray(proj_pts, np.float32)[p], D['cfg']).astype(np.float32)
    out['TI'] = (pts, dict(n_detected=n, n_matched=int(sel.sum()), ambiguous_mask=np.zeros(int(sel.sum()), bool)), tr)
    return out, conf


def fwhm_profile(prof, ts, prm) -> dict:

    m = len(prof); k = max(2, int(prm['bg_outer_frac'] * m)); bg = float(np.quantile(np.r_[prof[:k], prof[-k:]], prm['bg_quantile']))
    c = m // 2; r = max(1, int(round(1.5 / prm['step_px']))); ipk = c - r + int(np.argmax(prof[c - r:c + r + 1])); peak = float(prof[ipk])
    if peak - bg < 1e-6:
        return dict(valid=False, reason='no_contrast')
    if ipk <= c - r or ipk >= c + r or not (prof[ipk] >= prof[ipk - 1] and prof[ipk] >= prof[ipk + 1]):
        return dict(valid=False, reason='peak_not_central')
    half = bg + prm['half_frac'] * (peak - bg)
    il = ipk
    while il > 0 and prof[il] >= half:
        il -= 1
    ir = ipk
    while ir < m - 1 and prof[ir] >= half:
        ir += 1
    if il == 0 or ir == m - 1:
        return dict(valid=False, reason='half_level_not_reached')
    tl = ts[il] + (half - prof[il]) / (prof[il + 1] - prof[il] + 1e-12) * prm['step_px']; tr = ts[ir] - (half - prof[ir]) / (prof[ir - 1] - prof[ir] + 1e-12) * prm['step_px']
    fwhm = float(tr - tl)
    cont = bool((prof[:il].max() if il > 0 else -np.inf) >= half or (prof[ir + 1:].max() if ir + 1 < m else -np.inf) >= half)
    if cont:
        return dict(valid=False, reason='neighbour_contamination', fwhm_px=fwhm)
    return dict(valid=True, fwhm_px=fwhm)


def block_widths(img, front: dict, Pm: dict, n_b: int) -> dict:

    import lc_readout as LR
    from scipy.ndimage import map_coordinates
    prm = {**LR.DEFAULT, **Pm['readout']}; fam = front['fam']; widths = Pm['LC']['class_widths_px']
    L = LR.profile_params(fam, Pm['LC']['steger_sigma_px'], widths)['L_px']; ts = np.arange(-L, L + 1e-9, prm['step_px'])
    img = np.asarray(img, np.float64); H, W = img.shape; feat = front['feat']; segs = front['segments']; valid = front['guard_reason'] == 0; fam_t = np.asarray(fam['t'], np.float64)
    fw = np.full(len(feat), np.nan); per_seg = {}
    for sid in np.unique(segs[(segs >= 0) & valid]):
        idx = np.flatnonzero((segs == sid) & valid)
        if len(idx) > prm['max_rows_per_segment']:
            idx = idx[np.linspace(0, len(idx) - 1, prm['max_rows_per_segment']).astype(int)]
        sf = feat[idx].astype(np.float64); profs, rows = [], []
        for j, i in enumerate(idx):
            t = LR._tangent(sf, j, fam_t); n = np.array([-t[1], t[0]]); xs = sf[j, 0] + ts * n[0]; ys = sf[j, 1] + ts * n[1]
            if xs.min() < 0 or ys.min() < 0 or xs.max() > W - 1 or ys.max() > H - 1:
                continue
            profs.append(map_coordinates(img, [ys, xs], order=1, mode='nearest')); rows.append(i)
        order = np.argsort(feat[rows, 1]) if rows else np.zeros(0, int)
        profs = [profs[o] for o in order]; rows = [rows[o] for o in order]; vals = []
        for b0 in range(0, len(rows), n_b):
            blk = list(range(b0, min(b0 + n_b, len(rows))))
            if len(blk) < max(2, n_b // 2):
                continue
            r = fwhm_profile(np.mean([profs[q] for q in blk], axis=0), ts, prm)
            if r['valid']:
                vals.append(r['fwhm_px']); fw[[rows[q] for q in blk]] = r['fwhm_px']
        per_seg[int(sid)] = vals
    return dict(fw=fw, per_seg=per_seg)


def table_from_blocks(per_seg: dict, thr, mode: str, rho_d: float, margin_px=0.5) -> dict:
    out = {}
    for sid, vals in per_seg.items():
        if not vals:
            out[sid] = dict(n_blocks=0, median_px=None, cls=None, readable=False, reason='no_valid_block'); continue
        med = float(np.median(vals))
        if thr is None:
            out[sid] = dict(n_blocks=len(vals), median_px=med, cls=None, readable=False, reason='image_fallback'); continue
        sep = abs(med - thr) >= margin_px if mode == 'px' else abs(math.log(med) - math.log(thr)) >= LG_MARGIN * math.log(rho_d)
        readable = bool(len(vals) >= 2 and sep)
        out[sid] = dict(n_blocks=len(vals), median_px=med, cls=int(med > thr), readable=readable, reason='ok' if readable else ('too_few_blocks' if len(vals) < 2 else 'ambiguous_width'))
    return out


def lg_threshold(peaks: dict, rho_d: float):
    p = peaks.get('pair')
    if p is None:
        return None, 'fallback:' + str(peaks.get('reason'))
    lr = math.log(p['high_px'] / p['low_px']); lo, hi = LG_RATIO[0] * math.log(rho_d), LG_RATIO[1] * math.log(rho_d)
    if not (lo <= lr <= hi):
        return None, f'fallback:ratio_outside_design ({math.exp(lr):.3f})'
    return float(math.sqrt(p['low_px'] * p['high_px'])), 'design_ratio_geometric_mean'


def line_rule_record(front, tab: dict, name: str, thr, why: str) -> dict:
    import lc_pipeline as LC
    import lc_r4_runtime as RT
    adm, fc, seg_class = RT.filter_admissible(front['cand'], front['segments'], tab)
    r = LC.finish(front, dict(front['cand'], admissible=adm), dict(class_mode=name, **fc))
    r.update(segment_class=seg_class, threshold_px=thr, decision=why, segment_table=tab, image_fallback=(thr is None))
    return r


def lines_decode(img, inp: dict) -> dict:

    import lc_pipeline as LC
    import lc_r3_reliability as REL
    import lc_readout as LR
    import lc_r4_runtime as RT
    import m4_lines as ML
    import rdy_decode as DC
    Pm = DC.lc_params(); meta = inp['meta_full']['olines']; cfg = _P().s2().line_cfg(float(inp['peak'])); fam = ML.line_family(meta)
    wpx = (meta.get('code') or {}).get('widths_px')
    thr_old = THR_OLD
    if wpx:
        Pm = dict(Pm, LC=dict(Pm['LC'], class_widths_px=dict(narrow=float(wpx['narrow']), wide=float(wpx['wide']))))
        thr_old = THR_OLD * float(np.sqrt(float(wpx['narrow']) * float(wpx['wide']) / (W_SNAP['narrow'] * W_SNAP['wide'])))
        if abs(thr_old - THR_OLD) > 1e-12:
            Pm = dict(Pm, old_threshold_px=thr_old, readout=dict(Pm['readout'], threshold_px=thr_old))
    front = LC.shared_front_end(img, meta, cfg, inp['pattern'], Pm['LC'])
    readout = LR.read_segments(np.asarray(img, np.float64), front['feat'], front['segments'], front['guard_reason'] == 0, fam, Pm['LC']['steger_sigma_px'], Pm['LC']['class_widths_px'], Pm['readout'])
    runtime = RT.run_arms(RT.inputs_from_front(front, readout), Pm)
    recs = {f'LC_{k}': v for k, v in runtime['arms'].items()}
    w = meta['code']['widths_mm']; rho_d = float(w['wide']) / float(w['narrow'])
    for name, nb in LB_SIZES.items():
        bw = block_widths(img, front, Pm, nb)
        ok = np.isfinite(bw['fw']); peaks = REL.image_peaks(bw['fw'], ok, front['chain'], Pm['reliability'])
        thr, why = REL.choose_threshold('R', peaks, thr_old)
        recs[name] = line_rule_record(front, table_from_blocks(bw['per_seg'], thr, 'px', rho_d), name, thr, why)
        recs[name]['peaks'] = {k: v for k, v in peaks.items() if k in ('pair', 'reason', 'n_in_domain')}
        if name == 'LB':
            thr_g, why_g = lg_threshold(peaks, rho_d)
            recs['LG'] = line_rule_record(front, table_from_blocks(bw['per_seg'], thr_g, 'log', rho_d), 'LG', thr_g, why_g)
            recs['LG']['peaks'] = recs['LB']['peaks']; recs['_LB_fw'] = bw['fw']
    return dict(front=front, readout=readout, recs=recs, fam=fam, cfg=cfg, rho_d=rho_d)


def lines_truth_arms(Lr: dict, kO, conf) -> dict:
    import lc_pattern as LP
    import m4_lines as ML
    front = Lr['front']; segs = front['segments']; tc = np.full(len(kO), -1); tc[conf] = LP.class_of_id(kO[conf])
    tab = {}
    for s in np.unique(segs[segs >= 0]):
        c = np.unique(tc[(segs == s) & conf])
        tab[int(s)] = dict(readable=bool(len(c) == 1), cls=(int(c[0]) if len(c) == 1 else None))
    out = {'TC': line_rule_record(front, tab, 'TC', None, 'truth_class_diagnostic')}
    use = conf & (front['guard_reason'] == 0) & (segs >= 0) & np.isin(kO, front['cand']['ids'])
    line_id = np.where(use, kO, ML.UNKNOWN_ID).astype(np.int64); status = np.where(use, 0, 2).astype(np.int64)
    pts, proj, st, checks = ML.reconstruct_numbered(front['feat'], line_id, status, front['cand'], Lr['fam'], Lr['cfg'])
    base = out['TC']
    out['TI'] = dict(base, line_id=line_id, status=st, points=pts, accepted=(st == 0), checks=checks, decision='truth_identity_diagnostic', segment_class=np.full(len(kO), -1))
    return out


def evaluate(sid: str, family: str, dec: dict, ctx: dict, mesh) -> dict:
    import s1r_eval as E
    P = _P(); a = {v: k for k, v in K.RK.SURF_MAP.items()}[sid]
    m, arrays, info, _z = E.evaluate_case(sid, family, dec, ctx, mesh, P.dom_local(a), P.dom_global(), False)
    return m, arrays


KEEP = ('state', 'n_out', 'n_accepted', 'n_valid', 'n_selected', 'n_detected', 'n_confirmable_features', 'id_correct', 'id_wrong', 'id_unconf', 'glob_Ctau', 'loc_Ctau', 'glob_Cvalid',
        'loc_Cvalid', 'glob_recovery', 'loc_recovery', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'glob_gross_rate')


def mrow(m: dict) -> dict:
    return {k: m.get(k) for k in KEEP}


def new_vs(acc, ok_c, ok_w, ok_u, acc_ref) -> dict:

    if len(acc) != len(acc_ref):
        raise K.Stop('technical', 'new_vs needs aligned per-feature arrays (lines); use new_vs_features for dots')
    new = acc & ~acc_ref
    return dict(new_n=int(new.sum()), new_correct=int((new & ok_c).sum()), new_wrong=int((new & ok_w).sum()), new_unconf=int((new & ok_u).sum()))


def new_vs_features(arr: dict, arr_ref: dict) -> dict:

    def acc_feats(a):
        pf = np.asarray(a['point_feature'], np.int64); acc = np.asarray(a['accepted'], bool)
        st = np.where(np.asarray(a['point_correct'], bool), 'c', np.where(np.asarray(a['point_wrong'], bool), 'w', np.where(np.asarray(a['point_unconf'], bool), 'u', '-')))
        return {int(f): s for f, s, ok in zip(pf, st, acc) if ok}
    a, r = acc_feats(arr), acc_feats(arr_ref); new = {f: s for f, s in a.items() if f not in r}
    return dict(new_n=len(new), new_correct=sum(s == 'c' for s in new.values()), new_wrong=sum(s == 'w' for s in new.values()), new_unconf=sum(s == 'u' for s in new.values()))


def status_r4(m: dict, nv: dict, ref_acc: int) -> str:
    import lc_r4_rules as RR
    return RR.identity_status(int(m['id_correct'] + m['id_wrong']), int(m['id_wrong']), int(m['id_unconf']), int(nv['new_correct'] + nv['new_wrong']), int(nv['new_wrong']), int(nv['new_unconf']),
                              int(m['n_accepted']), int(ref_acc))
