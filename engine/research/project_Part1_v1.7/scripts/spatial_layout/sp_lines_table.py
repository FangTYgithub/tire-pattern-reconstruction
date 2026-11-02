# Use an explicit line-position table for rasterization, matching and triangulation.

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

import sp_common as S
import m4_lines as ML
import l01_common as C
from m4_lines import UNKNOWN_ID, STATUS, _resolve_sensor_cfg, _m3_cfg, _geometry, _dirs, triangulate64, epipolar_in_projector

GENERATOR = 'sp_lines_table_v1'
PROVENANCE = {f: S.sha_file(S.ROOT / f) for f in ('src/m4_lines.py', 'scripts/line01r/lc_pipeline.py', 'scripts/line01/l01_truth.py', 'src/m2_pattern.py')}


def regular_table(s_mm: float, phase_frac: float, w_mm: float = S.W_MM, image_size_px: int = S.IMG, phi_deg: float = S.PHI_DEG):

    s_px = S.px(s_mm); w_px = S.px(w_mm); half = (image_size_px - 1) / 2.0
    phi = math.radians(float(phi_deg)); reach = half * (abs(math.cos(phi)) + abs(math.sin(phi)))
    kmax = int(math.ceil((reach + w_px / 2.0 + 1.0) / s_px)) + 1
    ks = np.arange(-kmax, kmax + 1)
    q = (ks.astype(np.float64) + float(phase_frac)) * s_px
    return {int(k): float(v) for k, v in zip(ks, q)}


# Rasterize the explicit line positions and record illuminated line identifiers.
def table_spec(s_mm: float, w_mm: float, phase_frac: float, disp_px: dict | None = None, phi_deg: float = S.PHI_DEG, image_size_px: int = S.IMG) -> tuple[np.ndarray, dict]:

    H = W = int(image_size_px)
    s_px = S.px(s_mm); w_px = S.px(w_mm)
    disp = {int(k): float(v) for k, v in (disp_px or {}).items()}
    base = regular_table(s_mm, phase_frac, w_mm, image_size_px, phi_deg)
    q_tab = {k: base[k] + disp.get(k, 0.0) for k in base}
    ks = np.array(sorted(q_tab, key=lambda k: q_tab[k]), np.int64); qs = np.array([q_tab[int(k)] for k in ks], np.float64)
    if np.any(np.diff(qs) <= 1e-9):
        raise ValueError('duplicate or non-increasing table positions (two lines at the same position)')
    cu, cv = (W - 1) / 2.0, (H - 1) / 2.0
    phi = math.radians(float(phi_deg)); nx, ny = math.cos(phi), math.sin(phi); tx, ty = -math.sin(phi), math.cos(phi)
    vv, uu = np.mgrid[0:H, 0:W]
    q = ((uu - cu) * nx + (vv - cv) * ny).ravel()


    r = q / s_px - float(phase_frac)
    rs = np.array([float(k) + disp.get(int(k), 0.0) / s_px for k in ks], np.float64)
    j = np.clip(np.searchsorted(rs, r), 1, len(rs) - 1)
    left, right = rs[j - 1], rs[j]
    use_right = np.abs(r - right) < np.abs(r - left)
    k_near = np.where(use_right, ks[j], ks[j - 1]); d_near = np.where(use_right, np.abs(r - right), np.abs(r - left)) * s_px
    lit = (d_near <= w_px / 2.0).reshape(H, W); k_img = k_near.reshape(H, W)
    pattern = lit.astype(np.float32)
    ids, counts = np.unique(k_img[lit], return_counts=True)
    segments: dict = {}; clipped: dict = {}
    for kk, cnt in zip(ids.tolist(), counts.tolist()):
        qk = q_tab[int(kk)]
        p0 = np.array([cu + qk * nx, cv + qk * ny]); t = np.array([tx, ty])
        lo, hi = -math.inf, math.inf
        for axis, (a, b) in enumerate(((-0.5, W - 0.5), (-0.5, H - 0.5))):
            if abs(t[axis]) < 1e-12:
                if not (a <= p0[axis] <= b):
                    lo, hi = math.inf, -math.inf
            else:
                l1, l2 = (a - p0[axis]) / t[axis], (b - p0[axis]) / t[axis]
                lo, hi = max(lo, min(l1, l2)), min(hi, max(l1, l2))
        length = max(hi - lo, 0.0) if hi > lo else 0.0
        if length > 0:
            segments[str(kk)] = {'q_px': float(qk), 'start_uv': (p0 + lo * t).tolist(), 'end_uv': (p0 + hi * t).tolist(), 'length_px': float(length), 'lit_px': int(cnt), 'mean_lit_width_px': float(cnt / length)}
        else:
            clipped[str(kk)] = {'q_px': float(qk), 'lit_px': int(cnt)}
    ids_in = [int(x) for x in ids if str(int(x)) in segments]
    ids_all = [int(x) for x in ids]
    info = {'angle_deg': float(phi_deg), 'phase_frac': float(phase_frac), 'spacing_px': float(s_px), 'width_px': float(w_px), 'centre_uv': (cu, cv), 'normal_uv': (nx, ny), 'direction_uv': (tx, ty),
            'ids': ids_in, 'n_lines': int(len(ids_in)), 'lit_px_total': int(lit.sum()), 'segments': segments, 'clipped_lines': clipped,
            'rule': 'pixel lit when its distance to the NEAREST table line centre <= line_width_px / 2 (table version of m2 olines); ids = lines whose centre segment lies inside the image',
            'table': {'generator': GENERATOR, 'q_px': {str(k): float(q_tab[k]) for k in ids_all}, 'disp_px': {str(k): float(disp.get(k, 0.0)) for k in ids_all}, 'ids_all': ids_all,
                      'nominal_pitch_px': float(s_px), 'nominal_pitch_mm': float(s_mm), 'w_mm': float(w_mm), 'max_abs_disp_px': float(max([abs(v) for v in disp.values()] or [0.0])),
                      'n_displaced': int(sum(1 for v in disp.values() if v != 0.0))}}
    info['table']['hash'] = table_hash(info)
    return pattern, {'olines': info, 'pattern_kind': 'uncoded_table', 'generator': GENERATOR}


def table_hash(olines: dict) -> str:
    t = olines['table']
    body = dict(q_px={k: round(v, 9) for k, v in t['q_px'].items()}, ids_all=t['ids_all'], s=round(olines['spacing_px'], 9), w=round(olines['width_px'], 9), phi=olines['angle_deg'], phase=olines['phase_frac'], gen=t['generator'])
    return hashlib.sha256(S.digest(body).encode()).hexdigest()[:16]


def pattern_identity(meta: dict, raster_sha: str) -> dict:

    o = meta['olines']
    return dict(type='olines_table', s_px=float(o['spacing_px']), w_px=float(o['width_px']), phi_deg=float(o['angle_deg']), phase_frac=float(o['phase_frac']), raster_sha256=raster_sha,
                table_hash=(o['table']['hash'] if 'table' in o else None), generator=(o['table']['generator'] if 'table' in o else 'm2_pattern.olines'))


def line_family_table(spec: dict) -> dict:

    phi = math.radians(float(spec['angle_deg']))
    fam = dict(phi_deg=float(spec['angle_deg']), n=np.array([math.cos(phi), math.sin(phi)]), t=np.array([-math.sin(phi), math.cos(phi)]),
               s_px=float(spec['spacing_px']), w_px=float(spec['width_px']), phase=float(spec['phase_frac']), c=np.asarray(spec['centre_uv'], np.float64), ids=np.asarray(spec['ids'], np.int64))
    if 'table' in spec:
        qt = {int(k): float(v) for k, v in spec['table']['q_px'].items()}
        ids_all = np.asarray(spec['table']['ids_all'], np.int64)
        fam['source'] = 'table'
    else:
        ids_all = np.asarray(sorted(set(int(k) for k in spec['ids']) | set(int(k) for k in spec.get('clipped_lines', {}))), np.int64)
        qt = {int(k): float((np.float64(k) + fam['phase']) * fam['s_px']) for k in ids_all}
        fam['source'] = 'regular_arithmetic'
    missing = [int(k) for k in fam['ids'] if int(k) not in qt]
    if missing:
        raise ValueError(f'ids without table position: {missing}')
    fam['q'] = np.array([qt[int(k)] for k in fam['ids']], np.float64)
    order = np.argsort([qt[int(k)] for k in ids_all]); fam['ids_all'] = ids_all[order]; fam['q_all'] = np.array([qt[int(k)] for k in fam['ids_all']], np.float64)
    fam['q_of'] = qt; fam['table'] = True
    return fam


def line_q_table(fam: dict, k):

    qt = fam['q_of']
    if np.ndim(k) == 0:
        return np.float64(qt[int(k)])
    return np.array([qt[int(x)] for x in np.asarray(k).ravel()], np.float64).reshape(np.shape(k))


def nearest_table_line(fam: dict, q: np.ndarray):

    qs, ks = fam['q_all'], fam['ids_all']
    q = np.asarray(q, np.float64).ravel()
    if len(qs) == 1:
        return np.full(len(q), ks[0]), np.abs(q - qs[0]), np.full(len(q), UNKNOWN_ID), np.full(len(q), np.inf)


    j = np.searchsorted(qs, q)
    idx = np.stack([np.clip(j - 1, 0, len(qs) - 1), np.clip(j, 0, len(qs) - 1), np.clip(j + 1, 0, len(qs) - 1)], axis=1)
    d = np.abs(q[:, None] - qs[idx])
    d = np.where(np.diff(np.concatenate([np.full((len(q), 1), -1), idx], axis=1), axis=1) == 0, np.inf, d)
    order = np.argsort(d, axis=1, kind='stable')
    rows = np.arange(len(q))
    i1, i2 = idx[rows, order[:, 0]], idx[rows, order[:, 1]]
    d1, d2 = d[rows, order[:, 0]], d[rows, order[:, 1]]
    return ks[i1], d1, np.where(np.isfinite(d2), ks[i2], UNKNOWN_ID), d2


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


def light_plane_points_table(cam_uv: np.ndarray, line_id: np.ndarray, fam: dict, cfg: dict) -> np.ndarray:

    K, proj, cam, m3 = _geometry(cfg)
    c, n, t = fam['c'], fam['n'], fam['t']
    d_c = _dirs(cam_uv, K, cam); o_c = cam['center']; o_p = proj['center']
    out = np.full((len(cam_uv), 3), np.nan)
    for k in np.unique(line_id):
        sel = line_id == k
        q = float(line_q_table(fam, k))
        p_a = c + q * n - 200.0 * t; p_b = c + q * n + 200.0 * t
        da, db = _dirs(p_a[None, :], K, proj)[0], _dirs(p_b[None, :], K, proj)[0]
        N = np.cross(da, db); N /= np.linalg.norm(N)
        den = d_c[sel] @ N
        with np.errstate(divide='ignore', invalid='ignore'):
            lam = ((o_p - o_c) @ N) / den
        pts = o_c + lam[:, None] * d_c[sel]
        pts[~np.isfinite(lam) | (lam <= 0)] = np.nan
        out[sel] = pts
    return out


def line_candidates_table(cam_uv: np.ndarray, fam: dict, cfg: dict, centrelines: dict | None = None) -> dict:

    cfg = _resolve_sensor_cfg(cfg)
    vol = float(cfg.get('measurement_volume_mm', ML.MEASUREMENT_VOLUME_MM))
    z0 = float(cfg.get('search_z_min_mm', 0.0) or 0.0)
    margin = float(cfg['search_margin_mm']) if cfg.get('search_margin_mm') is not None else float(cfg.get('depth_margin_frac', ML.EPI_T_MARGIN)) * vol
    cam_uv = np.asarray(cam_uv, np.float64).reshape(-1, 2)
    ids = fam['ids']; N, K = len(cam_uv), len(ids)
    if N == 0:
        return dict(Q=np.zeros((0, K, 2)), Q_design=np.zeros((0, K, 2)), z=np.zeros((0, K)), admissible=np.zeros((0, K), bool), mu=np.zeros((0, K)), dq=np.zeros(0), ids=ids,
                    sin_alpha=np.zeros(0), search_volume_mm=vol, depth_margin_mm=margin, window_z0_mm=z0, window_z1_mm=z0 + vol, centre_source='raster' if centrelines is not None else 'analytic')
    P0, P1 = epipolar_in_projector(cam_uv, cfg, z0, z0 + vol)
    n, c, t = fam['n'], fam['c'], fam['t']
    q0 = (P0 - c) @ n; q1 = (P1 - c) @ n; dq = q1 - q0
    e = P1 - P0; e_len = np.linalg.norm(e, axis=1); e_unit = e / np.where(e_len[:, None] > 0, e_len[:, None], 1.0)
    sin_alpha = np.abs(e_unit[:, 0] * t[1] - e_unit[:, 1] * t[0])
    qk = np.asarray(fam['q'], np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = (qk[None, :] - q0[:, None]) / dq[:, None]
    Q_design = P0[:, None, :] + mu[..., None] * e[:, None, :]
    W, H = _m3_cfg(cfg)['image_width_px'], _m3_cfg(cfg)['image_height_px']
    Q = Q_design.copy(); has_centre = np.ones((N, K), bool)
    if centrelines is not None:
        dense = centrelines['_u_of_row']
        vd = np.nan_to_num(Q_design[..., 1], nan=-10.0)
        in_rows = np.isfinite(Q_design[..., 1]) & (vd >= 0.0) & (vd <= H - 1.0)
        kk = np.arange(K)[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            m = ((P1[:, 1] - P0[:, 1]) / (P1[:, 0] - P0[:, 0]))[:, None]
        P0u, P0v = P0[:, 0][:, None], P0[:, 1][:, None]
        v0 = np.clip(np.floor(vd).astype(np.int64), 0, H - 2)
        u_lit = np.full((N, K), np.nan); v_lit = np.full((N, K), np.nan)
        for _ in range(2):
            ua, ub = dense[kk, v0], dense[kk, v0 + 1]
            both = np.isfinite(ua) & np.isfinite(ub)
            a = np.where(np.isfinite(ua), ua, ub); b = np.where(both, ub - ua, 0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                u_lit = (a + b * (P0v - v0 - P0u * m)) / (1.0 - b * m)
            v_lit = P0v + (u_lit - P0u) * m
            v0 = np.clip(np.floor(np.nan_to_num(v_lit, nan=-10.0)).astype(np.int64), 0, H - 2)
        u_lit = np.where(in_rows, u_lit, np.nan)
        has_centre = np.isfinite(u_lit) & np.isfinite(v_lit)
        Q = np.where(has_centre[..., None], np.stack([u_lit, v_lit], axis=-1), np.nan)
    cam_rep = np.repeat(cam_uv, K, axis=0)
    pts, resid, valid = triangulate64(cam_rep, np.nan_to_num(Q.reshape(-1, 2), nan=0.0), cfg)
    z = np.where(has_centre, pts[:, 2].reshape(N, K), np.nan)
    inside = has_centre & (Q[..., 0] >= -0.5) & (Q[..., 0] <= W - 0.5) & (Q[..., 1] >= -0.5) & (Q[..., 1] <= H - 0.5)
    adm = np.isfinite(mu) & inside & valid.reshape(N, K) & np.isfinite(z) & (z >= z0 - margin) & (z <= z0 + vol + margin)
    return dict(Q=Q, Q_design=Q_design, z=z, admissible=adm, mu=mu, dq=dq, ids=ids, sin_alpha=sin_alpha, search_volume_mm=vol, depth_margin_mm=margin, window_z0_mm=z0, window_z1_mm=z0 + vol,
                centre_source='raster' if centrelines is not None else 'analytic')


def reconstruct_numbered_table(cam_uv: np.ndarray, line_id: np.ndarray, status: np.ndarray, cand: dict, fam: dict, cfg: dict):

    N = len(cam_uv)
    pts = np.full((N, 3), np.nan); proj = np.full((N, 2), np.nan); resid = np.full(N, np.nan); plane_diff = np.full(N, np.nan)
    acc = np.flatnonzero(status == 0)
    if len(acc) == 0:
        return pts, proj, status, dict(n_accepted=0, ray_residual_max_mm=None, plane_check_max_mm=None)
    col = {int(k): j for j, k in enumerate(cand['ids'])}
    jj = np.array([col[int(k)] for k in line_id[acc]])
    Q = cand['Q'][acc, jj]; Qd = cand.get('Q_design', cand['Q'])[acc, jj]
    cam_acc = np.asarray(cam_uv, np.float64)[acc]
    P, r, valid = triangulate64(cam_acc, Q, cfg)
    Pd, _, valid_d = triangulate64(cam_acc, Qd, cfg)
    PL = light_plane_points_table(cam_acc, line_id[acc], fam, cfg)
    bad = ~valid
    status = status.copy(); status[acc[bad]] = 4
    good = acc[valid]
    pts[good] = P[valid]; proj[good] = Q[valid]; resid[good] = r[valid]
    ok_d = valid & valid_d
    plane_diff[acc[ok_d]] = np.linalg.norm(Pd[ok_d] - PL[ok_d], axis=1)
    raster_dz = np.full(N, np.nan); raster_dz[acc[ok_d]] = P[ok_d, 2] - Pd[ok_d, 2]
    checks = dict(n_accepted=int(len(good)), ray_residual_max_mm=float(np.nanmax(resid)) if len(good) else None,
                  plane_check_max_mm=float(np.nanmax(plane_diff)) if np.isfinite(plane_diff).any() else None,
                  raster_depth_shift_mm=dict(mean=float(np.nanmean(raster_dz)) if np.isfinite(raster_dz).any() else None, max_abs=float(np.nanmax(np.abs(raster_dz))) if np.isfinite(raster_dz).any() else None))
    return pts, proj, status, checks


def run_p4_table(camera_image: np.ndarray, spec: dict, cfg: dict, pattern: np.ndarray | None, params: dict) -> dict:

    cfg = _resolve_sensor_cfg(cfg); fam = line_family_table(spec); centrelines = raster_centrelines_table(pattern, fam) if pattern is not None else None
    p = {**ML.default_params(fam, cfg), **{k: v for k, v in params.items() if v is not None}}
    if p.get('detector', 'steger') != 'steger':
        raise ValueError('the table adapter implements the registered P4 (steger) path only')
    p.setdefault('steger_threshold_rule', 'design'); p.setdefault('guard_min_chain_px', 0.0); p.setdefault('vote_valid_only', True); p.setdefault('steger_noise_coef', None); p.setdefault('steger_noise_coef_id', None)
    if not p['vote_valid_only']:
        raise ValueError('P4 uses vote_valid_only=True')
    thr_design = p['steger_threshold_frac'] * ML.steger_design_response(fam['w_px'], p['steger_sigma_px'], p['psf_sigma_px'], 255.0 * p['pattern_peak'])
    sig_hat, nstatus = None, 'not_estimated'; thr = thr_design
    if p['steger_threshold_rule'] == 'design_or_noise':
        sig_hat, nstatus, _, _ = ML.estimate_noise_sigma(camera_image); thr = max(thr_design, ML.steger_noise_threshold(sig_hat, p['steger_sigma_px'], p['steger_noise_coef']))
    feat, comp, dinfo = ML.detect_steger(camera_image, p['steger_sigma_px'], thr)
    dinfo = dict(dinfo, steger_threshold_rule=p['steger_threshold_rule'], steger_threshold_design=float(thr_design), noise_sigma_hat=sig_hat, noise_status=nstatus, steger_scale_effective=float(p['steger_sigma_px']),
                 steger_noise_coef_id=(p['steger_noise_coef_id'] or ('registered_fp1e-4' if p['steger_noise_coef'] is None else 'custom')))
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
                search_volume_mm=cand['search_volume_mm'], depth_margin_mm=cand['depth_margin_mm'], window_z0_mm=cand['window_z0_mm'], window_z1_mm=cand['window_z1_mm'], family_source=fam['source'])


def inner_margin_px(spec: dict, psf_px: float = C.PSF_PX, border_px: int = 3) -> float:
    return float(border_px + 0.5 * float(spec['width_px']) + 2.0 * psf_px + 1.0)


def truth_identity_table(mesh: dict, feats: np.ndarray, spec: dict, tol_px: float = 1.0) -> dict:

    fam = line_family_table(spec); g = C.geom()
    feats = np.asarray(feats, np.float64).reshape(-1, 2); N = len(feats)
    out = dict(k=np.full(N, UNKNOWN_ID, np.int64), confirmable=np.zeros(N, bool), dist_px=np.full(N, np.nan), hit=np.zeros(N, bool), proj_uv=np.full((N, 2), np.nan), xyz=np.full((N, 3), np.nan), in_proj=np.zeros(N, bool),
               multi_candidate=np.zeros(N, bool), n_within_band=np.zeros(N, np.int64), n_confirmable=0, n_unconfirmable=0, n_multi_candidate=0, band_px=float(fam['w_px'] / 2.0 + tol_px))
    if N == 0:
        return out
    hit, pts = C.cast(feats, g['cam'], mesh)
    uv, _ = C.project(pts, g['proj'])
    inside = (uv[:, 0] >= -0.5) & (uv[:, 0] <= C.IMG - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= C.IMG - 0.5)
    out['hit'] = hit; out['xyz'][hit] = pts[hit]; out['proj_uv'][hit] = uv[hit]; out['in_proj'] = hit & inside
    q = (uv - fam['c']) @ fam['n']
    k1, d1, k2, d2 = nearest_table_line(fam, q)
    band = fam['w_px'] / 2.0 + tol_px
    in_ids = np.isin(k1, fam['ids'])
    multi = out['in_proj'] & (d1 <= band) & (d2 <= band)
    conf = out['in_proj'] & (d1 <= band) & in_ids & ~multi
    out['k'][conf] = k1[conf]; out['confirmable'] = conf; out['dist_px'][hit] = d1[hit]; out['multi_candidate'] = multi
    out['n_within_band'][hit] = ((d1 <= band).astype(int) + (d2 <= band).astype(int))[hit]
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
                centre_source=('raster' if centrelines is not None else 'analytic'), inner_margin_px=inner_px)


def spec_from_m2(meta: dict) -> dict:

    return copy.deepcopy(meta['olines'])
