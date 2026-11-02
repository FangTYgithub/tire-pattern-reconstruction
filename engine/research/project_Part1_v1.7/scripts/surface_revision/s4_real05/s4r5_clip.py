# Apply optional abstention near clipped line bands before line reconstruction.

from __future__ import annotations

import numpy as np

import sp2_common as S2
import sp2_lines_table as LT2
import m4_lines as ML
from m4_lines import STATUS, _resolve_sensor_cfg

ADAPTER = 's4r5_clip_v1'
ABSTAIN = 11
STATUS_EXT = {**STATUS, ABSTAIN: 'clipped_source_abstain'}
OPTIONS = ('off', 'abstain')
assert ABSTAIN not in STATUS


def check_option(spec: dict, option: str) -> None:
    if option not in OPTIONS:
        raise ValueError(f'clipped option {option!r} not in {OPTIONS}')
    if option == 'off':
        return
    if abs(float(spec.get('angle_deg', np.nan))) > 1e-12:
        raise ValueError(f"clipped option implemented for 0 deg lines only (angle {spec.get('angle_deg')})")
    if 'table' not in spec or spec['table'].get('generator') not in ('sp_lines_table_v1',):
        raise ValueError('clipped option implemented for the uncoded position-table path only')
    if spec.get('classes') is not None or spec.get('code') is not None:
        raise ValueError('coded patterns are not supported by the clipped option')


def clipped_bands(spec: dict, pattern: np.ndarray | None, fam: dict) -> dict:

    clipped = sorted(int(k) for k in (spec.get('clipped_lines') or {}))
    if not clipped or pattern is None:
        return dict(ids=[], bands={}, empty=[], famx=None, centrelines=None)
    famx = dict(fam, ids=np.asarray(list(fam['ids']) + clipped, np.int64), q=np.asarray(list(fam['q']) + [float(LT2.line_q_table(fam, k)) for k in clipped], np.float64))
    cl = LT2.raster_centrelines_table(pattern, famx)
    bands = {}; empty = []
    for k in clipped:
        c = cl[k]
        if len(c['rows']) == 0:
            empty.append(k); continue
        bands[k] = dict(nominal_q_px=float(LT2.line_q_table(fam, k)), nominal_u=float(fam['c'][0] + LT2.line_q_table(fam, k) * fam['n'][0]), actual_u_mean=float(np.mean(c['u'])),
                        n_rows=int(len(c['rows'])), lit_px_meta=int(spec['clipped_lines'][str(k)]['lit_px']))
    if empty:
        keep = [k for k in clipped if k in bands]
        famx = dict(fam, ids=np.asarray(list(fam['ids']) + keep, np.int64), q=np.asarray(list(fam['q']) + [float(LT2.line_q_table(fam, k)) for k in keep], np.float64))
        cl = LT2.raster_centrelines_table(pattern, famx) if keep else None
        if not keep:
            famx = None
    return dict(ids=sorted(bands), bands=bands, empty=empty, famx=famx, centrelines=cl)


def apply_abstain(line_id: np.ndarray, status: np.ndarray, table: list, clipped_ids) -> tuple[np.ndarray, list, int]:

    clip = np.isin(line_id, np.asarray(list(clipped_ids), np.int64)) if len(clipped_ids) else np.zeros(len(line_id), bool)
    st = status.copy(); sel = clip & (st == 0); st[sel] = ABSTAIN
    cs = set(int(k) for k in clipped_ids); tab = []
    for s in table:
        s = dict(s)
        if s.get('decision') == 'numbered' and s.get('line_id') in cs:
            s['decision'] = 'clipped_source_abstain'
        tab.append(s)
    return st, tab, int(sel.sum())


def run_p4_table_clip(camera_image: np.ndarray, spec: dict, cfg: dict, pattern: np.ndarray | None, params: dict, option: str = 'off') -> dict:

    check_option(spec, option)
    p_in = LT2.check_p4_params(params)
    cfg = _resolve_sensor_cfg(cfg); fam = LT2.line_family_table(spec); centrelines = LT2.raster_centrelines_table(pattern, fam) if pattern is not None else None
    p = {**ML.default_params(fam, cfg), **{k: v for k, v in p_in.items() if v is not None}}
    p.setdefault('steger_threshold_rule', 'design'); p.setdefault('guard_min_chain_px', 0.0); p.setdefault('vote_valid_only', True); p.setdefault('steger_noise_coef', None); p.setdefault('steger_noise_coef_id', None)
    thr_design = p['steger_threshold_frac'] * ML.steger_design_response(fam['w_px'], p['steger_sigma_px'], p['psf_sigma_px'], 255.0 * p['pattern_peak'])
    sig_hat, nstatus = None, 'not_estimated'; thr = thr_design
    if p['steger_threshold_rule'] == 'design_or_noise':
        sig_hat, nstatus, _, _ = ML.estimate_noise_sigma(camera_image); thr = max(thr_design, ML.steger_noise_threshold(sig_hat, p['steger_sigma_px'], p['steger_noise_coef']))
    feat, comp, dinfo = ML.detect_steger(camera_image, p['steger_sigma_px'], thr)
    dinfo = dict(dinfo, steger_threshold_rule=p['steger_threshold_rule'], steger_threshold_design=float(thr_design), noise_sigma_hat=sig_hat, noise_status=nstatus, steger_scale_effective=float(p['steger_sigma_px']),
                 steger_noise_coef_id=(p['steger_noise_coef_id'] or ('registered_fp1e-4' if p['steger_noise_coef'] is None else 'custom')), adapter=S2.ADAPTER_VERSION)
    chain = ML.chain_features(feat, comp, p['du_max_px'])
    min_chain = float(p['guard_min_chain_px'] or 0.0); guard_reason = np.zeros(len(feat), np.int64)
    segments, n_split = ML.split_segments_at_gaps(feat, chain, guard_reason == 0)
    r8, _ = ML.apply_guards(feat, segments, None, min_chain, None); guard_reason = np.where((guard_reason == 0) & (r8 == 8), 8, guard_reason); segments = np.where(guard_reason == 0, segments, -1)
    guard_info = dict(n_guard_chain_short=int((guard_reason == 8).sum()), n_guard_width=0, min_chain_px=min_chain, width_window=None, n_chains_split=int(n_split), guard_rule='width_then_split_then_chain_v2', vote_rule='valid_only', width_model=None)
    clip = dict(ids=[], bands={}, empty=[], famx=None, centrelines=None)
    if option != 'off':
        clip = clipped_bands(spec, pattern, fam)
    famc = clip['famx'] if clip['famx'] is not None else fam; clc = clip['centrelines'] if clip['famx'] is not None else centrelines
    cand = LT2.line_candidates_table(feat, famc, cfg, centrelines=clc)
    line_id, status, n_cand, table = ML.number_segments(cand, segments, p['vote_frac'], valid=(guard_reason == 0))
    status = np.where(guard_reason > 0, guard_reason, status)
    n_abstain = 0
    if clip['ids']:
        status, table, n_abstain = apply_abstain(line_id, status, table, clip['ids'])
    pts, proj, status, checks = LT2.reconstruct_numbered_table(feat, line_id, status, cand, famc, cfg)
    acc = status == 0; counts = {STATUS_EXT[k]: int((status == k).sum()) for k in STATUS_EXT}
    raster_shift = np.full(len(feat), np.nan)
    if acc.any():
        col = {int(k): j for j, k in enumerate(cand['ids'])}; jj = np.array([col[int(k)] for k in line_id[acc]]); raster_shift[acc] = cand['Q'][np.flatnonzero(acc), jj, 0] - cand['Q_design'][np.flatnonzero(acc), jj, 0]
    checks = dict(checks, centre_source=cand['centre_source'], raster_shift_px=dict(mean=float(np.nanmean(raster_shift)) if acc.any() else None, max_abs=float(np.nanmax(np.abs(raster_shift))) if acc.any() else None))
    K0 = len(fam['ids'])
    adm_clip = cand['admissible'][:, K0:] if cand['admissible'].shape[1] > K0 else np.zeros((len(feat), 0), bool)
    return dict(detector='steger', params=p, features=feat.astype(np.float32), comp=comp.astype(np.int64), chain=chain, segment_id=segments, line_id=line_id, status=status, run_width_px=None,
                admissible=cand['admissible'], candidate_ids=cand['ids'], guard_reason=guard_reason, n_candidates=n_cand.astype(np.int64), points=pts, proj_uv=proj, accepted=acc, segments=table,
                detect_info=dict(dinfo, **guard_info), checks=checks, raster_shift_px=raster_shift,
                counts=dict(n_detected=int(len(feat)), n_chains=int(len(np.unique(chain[chain >= 0]))), n_segments_numbered=sum(1 for s in table if s['decision'] == 'numbered'),
                            n_segments_unknown=sum(1 for s in table if s['decision'] == 'unknown'), n_segments_no_candidate=sum(1 for s in table if s['decision'] == 'no_candidate'),
                            n_segments_clipped_abstain=sum(1 for s in table if s['decision'] == 'clipped_source_abstain'), n_ambiguous_features=int((n_cand >= 2).sum()), **counts),
                sin_alpha=dict(min=float(cand['sin_alpha'].min()) if len(feat) else None, mean=float(cand['sin_alpha'].mean()) if len(feat) else None),
                search_volume_mm=cand['search_volume_mm'], depth_margin_mm=cand['depth_margin_mm'], window_z0_mm=cand['window_z0_mm'], window_z1_mm=cand['window_z1_mm'], family_source=fam['source'],
                clip=dict(adapter=ADAPTER, option=option, ids=clip['ids'], bands=clip['bands'], empty=clip['empty'], n_abstain=n_abstain,
                          n_features_with_clip_candidate=int(adm_clip.any(axis=1).sum()), n_complete_candidates=cand['admissible'][:, :K0].sum(axis=1).astype(np.int64)))
