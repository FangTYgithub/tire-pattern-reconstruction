# Decode position-table lines with explicit parameter checks and nearest-line lookup.

from __future__ import annotations

import math

import numpy as np

import sp2_common as S2
import sp_lines_table as V1
import l01_common as C
import m4_lines as ML
from m4_lines import UNKNOWN_ID, STATUS, _resolve_sensor_cfg

table_spec = V1.table_spec
regular_table = V1.regular_table
line_family_table = V1.line_family_table
line_q_table = V1.line_q_table
line_candidates_table = V1.line_candidates_table
light_plane_points_table = V1.light_plane_points_table
reconstruct_numbered_table = V1.reconstruct_numbered_table
table_hash = V1.table_hash
pattern_identity = V1.pattern_identity
spec_from_m2 = V1.spec_from_m2
inner_margin_px = V1.inner_margin_px


EFFECTIVE_KEYS = {'detector': ('steger',), 'steger_threshold_rule': ('design', 'design_or_noise'), 'vote_valid_only': (True,)}
NUMERIC_KEYS = ('guard_min_chain_px', 'steger_sigma_px', 'steger_threshold_frac', 'du_max_px', 'vote_frac', 'pattern_peak', 'psf_sigma_px')
TABLE_KEYS = ('steger_noise_coef',)
METADATA_KEYS = ('steger_noise_coef_id', 'sg_window_rows')
DISABLED_ONLY = {'guard_width_window': (None,), 'guard_end_rows': (None,), 'binarise_rule': (None, 'otsu'), 'noise_k': (None, ML.NOISE_K_DEFAULT), 'guard_width_mult': None}


def check_p4_params(params: dict) -> dict:

    p = dict(params or {})
    unknown = [k for k in p if k not in EFFECTIVE_KEYS and k not in NUMERIC_KEYS and k not in TABLE_KEYS and k not in METADATA_KEYS and k not in DISABLED_ONLY]
    if unknown:
        raise ValueError(f'parameter(s) not supported by the table adapter (would be silently ignored): {unknown}')
    for k, allowed in EFFECTIVE_KEYS.items():
        if k in p and p[k] not in allowed:
            raise ValueError(f'{k}={p[k]!r} not supported by the table adapter (allowed {allowed})')
    for k, allowed in DISABLED_ONLY.items():
        if k in p and allowed is not None and p[k] not in allowed:
            raise ValueError(f'{k}={p[k]!r} would change the P4 behaviour and is not implemented in the table adapter')
    if 'guard_width_mult' in p and p.get('guard_width_window') is not None:
        raise ValueError('guard_width_window is not implemented in the table adapter')
    for k in NUMERIC_KEYS:
        if k in p and p[k] is not None and not isinstance(p[k], (int, float)):
            raise ValueError(f'{k} must be numeric')
    if 'steger_noise_coef' in p and p['steger_noise_coef'] is not None:
        if not isinstance(p['steger_noise_coef'], dict):
            raise ValueError('steger_noise_coef must be a dict scale -> coefficient or None')
        p['steger_noise_coef'] = {float(k): float(v) for k, v in p['steger_noise_coef'].items()}
    return p


def nearest_table_line(fam: dict, q):

    qs = np.asarray(fam['q_all'], np.float64); ks = np.asarray(fam['ids_all'], np.int64)
    q = np.asarray(q, np.float64).ravel(); n = len(qs)
    k1 = np.full(len(q), UNKNOWN_ID, np.int64); k2 = np.full(len(q), UNKNOWN_ID, np.int64); d1 = np.full(len(q), np.inf); d2 = np.full(len(q), np.inf)
    if n == 0 or len(q) == 0:
        return k1, d1, k2, d2
    if np.any(np.diff(qs) <= 0):
        raise ValueError('table positions must be strictly increasing (duplicate line positions)')
    fin = np.isfinite(q)
    if not fin.any():
        return k1, d1, k2, d2
    qf = q[fin]
    j = np.searchsorted(qs, qf)
    idx = np.stack([j - 2, j - 1, j, j + 1], axis=1)
    valid = (idx >= 0) & (idx < n)
    idc = np.clip(idx, 0, n - 1)
    d = np.where(valid, np.abs(qf[:, None] - qs[idc]), np.inf)

    order = np.argsort(d, axis=1, kind='stable')
    rows = np.arange(len(qf))
    i1 = idc[rows, order[:, 0]]; dd1 = d[rows, order[:, 0]]
    i2 = idc[rows, order[:, 1]]; dd2 = d[rows, order[:, 1]]
    k1[fin] = ks[i1]; d1[fin] = dd1
    has2 = np.isfinite(dd2)
    k2f = np.where(has2, ks[i2], UNKNOWN_ID); k2[fin] = k2f; d2[fin] = np.where(has2, dd2, np.inf)
    return k1, d1, k2, d2


def nearest_reference(fam: dict, q):

    qs = np.asarray(fam['q_all'], np.float64); ks = np.asarray(fam['ids_all'], np.int64)
    q = np.asarray(q, np.float64).ravel()
    k1 = np.full(len(q), UNKNOWN_ID, np.int64); k2 = np.full(len(q), UNKNOWN_ID, np.int64); d1 = np.full(len(q), np.inf); d2 = np.full(len(q), np.inf)
    for i, x in enumerate(q):
        if not np.isfinite(x) or len(qs) == 0:
            continue
        d = np.abs(x - qs)
        order = sorted(range(len(qs)), key=lambda t: (d[t], qs[t]))
        k1[i] = ks[order[0]]; d1[i] = d[order[0]]
        if len(order) > 1:
            k2[i] = ks[order[1]]; d2[i] = d[order[1]]
    return k1, d1, k2, d2


def raster_centrelines_table(pattern: np.ndarray, fam: dict) -> dict:

    pat = np.asarray(pattern, np.float64); H, W = pat.shape
    lit = pat > 0.5 * pat.max() if pat.max() > 0 else np.zeros_like(pat, bool)
    vv, uu = np.nonzero(lit); n, c = fam['n'], fam['c']
    q = (uu - c[0]) * n[0] + (vv - c[1]) * n[1]
    k, _, _, _ = nearest_table_line(fam, q)
    ids = fam['ids']; dense = np.full((len(ids), H), np.nan); out = {}
    for j, kk in enumerate(ids):
        sel = k == kk
        if not sel.any():
            out[int(kk)] = dict(rows=np.zeros(0, np.int64), u=np.zeros(0), q_offset_px=None, row_dev_std_px=None, row_dev_max_px=None); continue
        vs, us = vv[sel], uu[sel]
        cnt = np.bincount(vs, minlength=H); su = np.bincount(vs, weights=us, minlength=H)
        rows = np.flatnonzero(cnt); u_mean = su[rows] / cnt[rows]; dense[j, rows] = u_mean
        qk = float(line_q_table(fam, kk))
        q_rows = (u_mean - c[0]) * n[0] + (rows - c[1]) * n[1] - qk
        out[int(kk)] = dict(rows=rows, u=u_mean, q_offset_px=float(np.mean(q[sel]) - qk), row_dev_std_px=float(np.std(q_rows)), row_dev_max_px=float(np.abs(q_rows).max()))
    out['_u_of_row'] = dense
    return out


def truth_identity_table(mesh: dict, feats: np.ndarray, spec: dict, tol_px: float = 1.0) -> dict:

    fam = line_family_table(spec); g = C.geom()
    feats = np.asarray(feats, np.float64).reshape(-1, 2); N = len(feats)
    out = dict(k=np.full(N, UNKNOWN_ID, np.int64), confirmable=np.zeros(N, bool), dist_px=np.full(N, np.nan), dist2_px=np.full(N, np.nan), k2=np.full(N, UNKNOWN_ID, np.int64), hit=np.zeros(N, bool), proj_uv=np.full((N, 2), np.nan),
               xyz=np.full((N, 3), np.nan), in_proj=np.zeros(N, bool), multi_candidate=np.zeros(N, bool), n_within_band=np.zeros(N, np.int64), n_confirmable=0, n_unconfirmable=0, n_multi_candidate=0,
               band_px=float(fam['w_px'] / 2.0 + tol_px), adapter=S2.ADAPTER_VERSION)
    if N == 0:
        return out
    hit, pts = C.cast(feats, g['cam'], mesh)
    uv, _ = C.project(pts, g['proj'])
    finite_uv = np.isfinite(uv).all(axis=1)
    inside = finite_uv & (uv[:, 0] >= -0.5) & (uv[:, 0] <= C.IMG - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= C.IMG - 0.5)
    out['hit'] = hit; out['xyz'][hit] = pts[hit]; out['proj_uv'][hit] = uv[hit]; out['in_proj'] = hit & inside
    q = np.where(finite_uv, (np.nan_to_num(uv, nan=0.0) - fam['c']) @ fam['n'], np.nan)
    k1, d1, k2, d2 = nearest_table_line(fam, q)
    band = fam['w_px'] / 2.0 + tol_px
    in_ids = np.isin(k1, fam['ids'])
    within1 = out['in_proj'] & np.isfinite(d1) & (d1 <= band); within2 = out['in_proj'] & np.isfinite(d2) & (d2 <= band)
    multi = within1 & within2
    conf = within1 & in_ids & ~multi
    out['k'][conf] = k1[conf]; out['confirmable'] = conf; out['multi_candidate'] = multi
    out['dist_px'][hit] = d1[hit]; out['dist2_px'][hit] = d2[hit]; out['k2'][hit] = k2[hit]
    out['n_within_band'] = within1.astype(np.int64) + within2.astype(np.int64)
    out['n_confirmable'] = int(conf.sum()); out['n_unconfirmable'] = int(N - conf.sum()); out['n_multi_candidate'] = int(multi.sum())
    return out


def line_reference_table(mesh: dict, spec: dict, step_px: float = 1.0, border_px: int = 3, pattern: np.ndarray | None = None, inner_px: float | None = None) -> dict:

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
        uv, inside = C.project(pts, g['cam'])
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


def run_p4_table(camera_image: np.ndarray, spec: dict, cfg: dict, pattern: np.ndarray | None, params: dict) -> dict:

    p_in = check_p4_params(params)
    cfg = _resolve_sensor_cfg(cfg); fam = line_family_table(spec); centrelines = raster_centrelines_table(pattern, fam) if pattern is not None else None
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
    cand = line_candidates_table(feat, fam, cfg, centrelines=centrelines)
    line_id, status, n_cand, table = ML.number_segments(cand, segments, p['vote_frac'], valid=(guard_reason == 0))
    status = np.where(guard_reason > 0, guard_reason, status)
    pts, proj, status, checks = reconstruct_numbered_table(feat, line_id, status, cand, fam, cfg)
    acc = status == 0; counts = {STATUS[k]: int((status == k).sum()) for k in STATUS}
    raster_shift = np.full(len(feat), np.nan)
    if acc.any():
        col = {int(k): j for j, k in enumerate(cand['ids'])}; jj = np.array([col[int(k)] for k in line_id[acc]]); raster_shift[acc] = cand['Q'][np.flatnonzero(acc), jj, 0] - cand['Q_design'][np.flatnonzero(acc), jj, 0]
    checks = dict(checks, centre_source=cand['centre_source'], raster_shift_px=dict(mean=float(np.nanmean(raster_shift)) if acc.any() else None, max_abs=float(np.nanmax(np.abs(raster_shift))) if acc.any() else None))
    return dict(detector='steger', params=p, features=feat.astype(np.float32), comp=comp.astype(np.int64), chain=chain, segment_id=segments, line_id=line_id, status=status, run_width_px=None,
                admissible=cand['admissible'], candidate_ids=cand['ids'], guard_reason=guard_reason, n_candidates=n_cand.astype(np.int64), points=pts, proj_uv=proj, accepted=acc, segments=table,
                detect_info=dict(dinfo, **guard_info), checks=checks, raster_shift_px=raster_shift,
                counts=dict(n_detected=int(len(feat)), n_chains=int(len(np.unique(chain[chain >= 0]))), n_segments_numbered=sum(1 for s in table if s['decision'] == 'numbered'),
                            n_segments_unknown=sum(1 for s in table if s['decision'] == 'unknown'), n_segments_no_candidate=sum(1 for s in table if s['decision'] == 'no_candidate'), n_ambiguous_features=int((n_cand >= 2).sum()), **counts),
                sin_alpha=dict(min=float(cand['sin_alpha'].min()) if len(feat) else None, mean=float(cand['sin_alpha'].mean()) if len(feat) else None),
                search_volume_mm=cand['search_volume_mm'], depth_margin_mm=cand['depth_margin_mm'], window_z0_mm=cand['window_z0_mm'], window_z1_mm=cand['window_z1_mm'], family_source=fam['source'], adapter=S2.ADAPTER_VERSION)
