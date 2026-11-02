# Share line detection between coded and uncoded decoding, then apply class constraints.

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
sys.path.insert(0, str(HERE))
import m4_lines as ML
import lc_readout as LR
from lc_pattern import class_of_id


def shared_front_end(camera_image, spec, cfg, pattern, params):

    cfg = ML._resolve_sensor_cfg(cfg); fam = ML.line_family(spec); centrelines = ML.raster_centrelines(pattern, fam) if pattern is not None else None
    p = {**ML.default_params(fam, cfg), **{k: v for k, v in params.items() if v is not None}}
    p.setdefault('steger_threshold_rule', 'design'); p.setdefault('guard_min_chain_px', 0.0); p.setdefault('vote_valid_only', True); p.setdefault('steger_noise_coef', None); p.setdefault('steger_noise_coef_id', None)
    widths = p.get('class_widths_px')
    resp = {k: float(ML.steger_design_response(v, p['steger_sigma_px'], p['psf_sigma_px'], 255.0 * p['pattern_peak'])) for k, v in widths.items()} if widths else None
    weaker = min(resp, key=resp.get) if resp else None
    w_design = widths[weaker] if widths else fam['w_px']
    thr_design = p['steger_threshold_frac'] * ML.steger_design_response(w_design, p['steger_sigma_px'], p['psf_sigma_px'], 255.0 * p['pattern_peak'])
    sig_hat, nstatus = None, 'not_estimated'; thr = thr_design
    if p['steger_threshold_rule'] == 'design_or_noise':
        sig_hat, nstatus, _, _ = ML.estimate_noise_sigma(camera_image); thr = max(thr_design, ML.steger_noise_threshold(sig_hat, p['steger_sigma_px'], p['steger_noise_coef']))
    feat, comp, dinfo = ML.detect_steger(camera_image, p['steger_sigma_px'], thr)
    dinfo = dict(dinfo, steger_threshold_rule=p['steger_threshold_rule'], steger_threshold_design=float(thr_design), noise_sigma_hat=sig_hat, noise_status=nstatus, steger_scale_effective=float(p['steger_sigma_px']), design_width_px_for_threshold=float(w_design),
                 design_class_for_threshold=weaker, response_by_class=resp)
    chain = ML.chain_features(feat, comp, p['du_max_px'])
    min_chain = float(p['guard_min_chain_px'] or 0.0); guard_reason = np.zeros(len(feat), np.int64)
    segments, n_split = ML.split_segments_at_gaps(feat, chain, guard_reason == 0)
    r8, _ = ML.apply_guards(feat, segments, None, min_chain, None); guard_reason = np.where((guard_reason == 0) & (r8 == 8), 8, guard_reason); segments = np.where(guard_reason == 0, segments, -1)
    guard_info = dict(n_guard_chain_short=int((guard_reason == 8).sum()), n_guard_width=0, min_chain_px=min_chain, width_window=None, n_chains_split=int(n_split), guard_rule='width_then_split_then_chain_v2', vote_rule='valid_only', width_model=None)
    cand = ML.line_candidates(feat, fam, cfg, centrelines=centrelines)
    ident = hashlib.sha256(feat.astype(np.float32).tobytes() + segments.astype(np.int64).tobytes() + guard_reason.astype(np.int64).tobytes() + cand['admissible'].astype(np.uint8).tobytes()).hexdigest()[:16]
    return dict(cfg=cfg, fam=fam, p=p, feat=feat, comp=comp, chain=chain, segments=segments, guard_reason=guard_reason, cand=cand, dinfo=dinfo, guard_info=guard_info, centrelines=centrelines, shared_identity=ident)


def finish(front, cand_for_vote, mode_info):

    fam, cfg, p, feat = front['fam'], front['cfg'], front['p'], front['feat']; cand = front['cand']; guard_reason = front['guard_reason']; segments = front['segments']
    line_id, status, n_cand, table = ML.number_segments(cand_for_vote, segments, p['vote_frac'], valid=(guard_reason == 0))
    status = np.where(guard_reason > 0, guard_reason, status)
    pts, proj, status, checks = ML.reconstruct_numbered(feat, line_id, status, cand, fam, cfg)
    acc = status == 0; counts = {ML.STATUS[k]: int((status == k).sum()) for k in ML.STATUS}
    raster_shift = np.full(len(feat), np.nan)
    if acc.any():
        col = {int(k): j for j, k in enumerate(cand['ids'])}; jj = np.array([col[int(k)] for k in line_id[acc]]); raster_shift[acc] = cand['Q'][np.flatnonzero(acc), jj, 0] - cand['Q_design'][np.flatnonzero(acc), jj, 0]
    checks = dict(checks, centre_source=cand['centre_source'], raster_shift_px=dict(mean=float(np.nanmean(raster_shift)) if acc.any() else None, max_abs=float(np.nanmax(np.abs(raster_shift))) if acc.any() else None))
    return dict(detector='steger', params=p, features=feat.astype(np.float32), comp=front['comp'].astype(np.int64), chain=front['chain'], segment_id=segments, line_id=line_id, status=status, run_width_px=None, admissible=cand['admissible'], candidate_ids=cand['ids'], guard_reason=guard_reason,
                n_candidates=n_cand.astype(np.int64), points=pts, proj_uv=proj, accepted=acc, segments=table, detect_info=dict(front['dinfo'], **front['guard_info'], **mode_info), checks=checks, raster_shift_px=raster_shift,
                counts=dict(n_detected=int(len(feat)), n_chains=int(len(np.unique(front['chain'][front['chain'] >= 0]))), n_segments_numbered=sum(1 for s in table if s['decision'] == 'numbered'), n_segments_unknown=sum(1 for s in table if s['decision'] == 'unknown'), n_segments_no_candidate=sum(1 for s in table if s['decision'] == 'no_candidate'), n_ambiguous_features=int((n_cand >= 2).sum()), **counts),
                sin_alpha=dict(min=float(cand['sin_alpha'].min()) if len(feat) else None, mean=float(cand['sin_alpha'].mean()) if len(feat) else None), search_volume_mm=cand['search_volume_mm'], depth_margin_mm=cand['depth_margin_mm'], window_z0_mm=cand['window_z0_mm'], window_z1_mm=cand['window_z1_mm'], shared_identity=front['shared_identity'])


def run_lc(camera_image, spec, cfg, pattern, params, readout_params=None, modes=('ignore', 'use')):

    front = shared_front_end(camera_image, spec, cfg, pattern, params)
    widths = front['p'].get('class_widths_px'); out = {}
    ro = None
    if widths:
        ro = LR.read_segments(np.asarray(camera_image, np.float64), front['feat'], front['segments'], front['guard_reason'] == 0, front['fam'], front['p']['steger_sigma_px'], widths, readout_params)
    if 'ignore' in modes:
        out['ignore'] = finish(front, front['cand'], dict(class_mode='ignore'))
    if 'use' in modes:
        cand = front['cand']; adm = cand['admissible'].copy(); ids_cls = class_of_id(cand['ids']); segs = front['segments']; n_filtered = n_fallback = n_conflict = n_readable = 0; seg_class = np.full(len(front['feat']), -1, np.int64)
        if ro is not None:
            for sid, s in ro['per_segment'].items():
                if not s['readable']:
                    continue
                n_readable += 1; idx = np.flatnonzero(segs == sid); seg_class[idx] = s['cls']
                for i in idx:
                    row = adm[i]; k = int(row.sum())
                    if k == 0:
                        continue
                    if k == 1:
                        if ids_cls[np.flatnonzero(row)[0]] != s['cls']:
                            n_conflict += 1
                        continue
                    new = row & (ids_cls == s['cls'])
                    if new.sum() == 0:
                        n_fallback += 1; continue
                    if new.sum() < k:
                        n_filtered += 1
                    adm[i] = new
        cand_use = dict(cand, admissible=adm)
        out['use'] = finish(front, cand_use, dict(class_mode='use', n_readable_segments=n_readable, n_features_filtered=n_filtered, n_features_fallback=n_fallback, n_conflicts_unique_candidate=n_conflict))
        out['use']['segment_class'] = seg_class
    return dict(out, readout=ro, shared_identity=front['shared_identity'], front_counts=dict(n_detected=int(len(front['feat'])), n_valid_after_guards=int((front['guard_reason'] == 0).sum()), n_segments=int(len(np.unique(front['segments'][front['segments'] >= 0])))))
