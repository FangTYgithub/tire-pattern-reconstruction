# Provide the corrected surfaces, fixed device geometry and common evaluation routines.

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts' / 'pm014', ROOT / 'scripts' / 'spotnoise', ROOT / 'scripts' / 'spotnoise_round5',
          ROOT / 'scripts' / 'spacing_layout', ROOT / 'scripts' / 'line01', ROOT / 'scripts' / 'line01r', ROOT / 'scripts' / 'surface_revision' / 'surf01'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import s1_model as M
import sn_lib as SL
import sl_lib as SLL
import r5_detect as RD
import l01_common as C0
import l01rb_common as B
import l01_truth as T
import m4_reconstruction as M4

OUT = ROOT / 'outputs' / 'surface_revision' / 'surf02'
DOCS = ROOT / 'docs' / 'surface_revision'
for d in ('config', 'cases', 'metrics', 'checks', 'figures'):
    (OUT / d).mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

VERSION = 'SURF-02 v1.1 (2026-10-12)'
IMG = 512
Z0_MM, Z1_MM = -16.0, 4.0  # Common reconstruction-depth limits [mm].
VOL_MM = Z1_MM - Z0_MM
MARGIN_MM = 1e-6
GRID_SCI = (0.25, 0.125)  # Scientific surface-mesh steps (x, y) [mm]; distinct from evaluation-node spacing.
GRID_CHECK = (0.5, 0.25)  # Alternative surface-mesh steps (x, y) [mm] for numerical checks.
X_HALF_MM = 100.0
ROI_R_MM = 15.0  # Radius of the optional local diagnostic region [mm].
FOV_HALF_MM = 46.08  # Reference-plane field-of-view half-width [mm].
GROSS_MM = 1.0
SIGMAS = (0.0, 40.0)  # Noise levels for platform checks; formal studies supply their own case levels.
SEED_BASE = 10300000
SURFACES = {'A_curv': dict(variant='A', gauss=False, de='A: M10 + Zusatzkrümmung', zh='M10+弧'),
            'C_curv': dict(variant='C', gauss=False, de='C: M24 + Zusatzkrümmung', zh='M24+弧'),
            'B_curv': dict(variant='B', gauss=False, de='B: ebene Fahrbahn (Reifenkontur) + Zusatzkrümmung', zh='平地B+弧'),
            'A_curv_gauss': dict(variant='A', gauss=True, de='A: M10 + Zusatzkrümmung + Gauß-Erhebung (5 mm)', zh='M10+弧+5 mm高斯')}
SURFACE_ORDER = ('A_curv', 'C_curv', 'B_curv', 'A_curv_gauss')
DOT = dict(lattice='R8x4', r_mm=1.0, peak=1.0, group='D0', positions=tuple(SLL.latin_indices()[:2]))
LINE = dict(w_mm=0.5, s_mm=8.0, phi_deg=0.0, peak=1.0, regime='fixed_peak', path='P4_C3_cal', phases=tuple(B.PHASES))
P4_FILE = ROOT / 'outputs' / 'line01r' / 'c3' / 'config' / 'paths_c3_validation.json'  # Calibrated P4 settings; the study adapter supplies effective line parameters.
P4 = json.loads(P4_FILE.read_text(encoding='utf-8'))['P4_C3_cal']
clean = C0.clean

_PROF: dict = {}
_MESH: dict = {}


def profile(variant: str) -> dict:
    if variant not in _PROF:
        _PROF[variant] = M.source_profile(variant)
    return _PROF[variant]


def c_ref() -> float:

    return float(M.h_base(profile('B'), 0.0, 0.0))


C_REF = c_ref()


def h_physical(key: str, x_c, y_c, base_only: bool = False):

    S = SURFACES[key]; prof = profile(S['variant'])
    xp = -np.asarray(x_c, np.float64); yp = np.asarray(y_c, np.float64)
    H = M.h_base(prof, xp, yp)
    if S['gauss'] and not base_only:
        H = H + M.h_gauss(xp, yp)
    return H


def z_true(key: str, x_c, y_c, base_only: bool = False):

    return C_REF - h_physical(key, x_c, y_c, base_only)


# Sample the corrected surface on the requested grid and attach its coordinate metadata.
def build_mesh(key: str, grid=GRID_SCI) -> dict:

    ck = (key, tuple(grid))
    if ck in _MESH:
        return _MESH[ck]
    dx, dy = grid; prof = profile(SURFACES[key]['variant'])
    nx = int(round(2 * X_HALF_MM / dx)); x = -X_HALF_MM + dx * np.arange(nx + 1)
    y0, y1 = prof['y_range']; ny = int(round((y1 - y0) / dy)); y = y0 + dy * np.arange(ny + 1)
    X, Y = np.meshgrid(x, y)
    Z = z_true(key, X, Y); Zb = z_true(key, X, Y, base_only=True)
    mesh = {'X': X, 'Y': Y, 'Z': Z, 'Z_base': Zb, 'z_alignment': 'absolute',
            'meta': dict(surface=key, grid=list(grid), c_ref_mm=C_REF, frame='x=-x_p, y=y_p, z=c-H', shape=list(Z.shape), **{k: v for k, v in SURFACES[key].items() if k != 'de'})}
    _MESH[ck] = mesh
    return mesh


def mesh_sha(mesh: dict) -> str:
    h = hashlib.sha256()
    for k in ('X', 'Y', 'Z'):
        h.update(np.ascontiguousarray(mesh[k], np.float64).tobytes())
    return h.hexdigest()[:16]


def truth_interp(mesh: dict):

    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator((mesh['Y'][:, 0], mesh['X'][0, :]), np.asarray(mesh['Z'], np.float64), method='linear', bounds_error=False, fill_value=np.nan)


def roi_domain(mesh: dict) -> dict:

    X = mesh['X']; Y = mesh['Y']; Z = np.asarray(mesh['Z'], np.float64); Zb = np.asarray(mesh['Z_base'], np.float64)
    m = (X ** 2 + Y ** 2) <= ROI_R_MM ** 2
    idx = np.flatnonzero(m.ravel()); x = X.ravel()[idx]; y = Y.ravel()[idx]
    dx = float(X[0, 1] - X[0, 0]); dy = float(Y[1, 0] - Y[0, 0])
    fov = (np.abs(X) <= FOV_HALF_MM) & (np.abs(Y) <= FOV_HALF_MM)
    return dict(idx=idx, x=x, y=y, z_ref=Z.ravel()[idx], z_base=Zb.ravel()[idx], area=np.full(len(idx), dx * dy), r=np.hypot(x, y), n_nodes=int(len(idx)),
                area_total_mm2=float(len(idx) * dx * dy), hash=C0.sha_bytes(idx.astype(np.int64).tobytes() + Z.ravel()[idx].tobytes()),
                z_range=(float(Z.ravel()[idx].min()), float(Z.ravel()[idx].max())),
                window_cov_roi=float(np.mean((Z.ravel()[idx] >= Z0_MM) & (Z.ravel()[idx] <= Z1_MM))),
                window_cov_fov=float(np.mean((Z[fov] >= Z0_MM) & (Z[fov] <= Z1_MM))), z_range_fov=(float(Z[fov].min()), float(Z[fov].max())))


def geometry_record() -> dict:
    g = C0.geom(); cfg = g['cfg']
    return dict(convention='fixed nominal simulation device (SENSOR_DEFAULTS); not measured extrinsics', image_px=[cfg['image_width_px'], cfg['image_height_px']],
                focal_length_mm=cfg['focal_length_mm'], pixel_pitch_mm=cfg['pixel_pitch_mm'], baseline_mm=cfg['baseline_mm'], working_dist_mm=cfg['working_dist_mm'],
                K=np.asarray(g['K']).tolist(), camera_center_mm=np.asarray(g['cam']['center']).tolist(), projector_center_mm=np.asarray(g['proj']['center']).tolist(),
                camera_R=np.asarray(g['cam']['R']).tolist(), projector_R=np.asarray(g['proj']['R']).tolist(), look_at_mm=[0.0, 0.0, 0.0], mm_per_px_at_wd=g['mm_per_px'],
                triangulation_angle_deg=g['theta_deg'])


def m3_cfg(sigma: float, seed: int) -> dict:
    return C0.m3_cfg(sigma, seed)


# Combine dot-detector settings with the common depth search window.
def dot_cfg() -> dict:
    return {**RD.CFG, 'search_z_min_mm': Z0_MM, 'measurement_volume_mm': VOL_MM}


# Combine the chosen line brightness and P4 settings with the common depth window.
def line_cfg(peak: float | None = None) -> dict:

    pk = float(LINE['peak'] if peak is None else peak)
    return {**B.m4_cfg(pk), 'search_z_min_mm': Z0_MM, 'measurement_volume_mm': VOL_MM, 'search_margin_mm': MARGIN_MM}


def seed_for(si: int, fi: int, pi: int, sigma: float) -> int:
    return SEED_BASE + 1000 * si + 100 * fi + 10 * pi + (1 if sigma > 0 else 0)


def cases(grid=GRID_SCI, surfaces=SURFACE_ORDER, families=('dots', 'lines'), sigmas=SIGMAS, phases=(0, 1), tag='main') -> list[dict]:
    out = []
    for si, s in enumerate(SURFACE_ORDER):
        if s not in surfaces:
            continue
        for fi, fam in enumerate(('dots', 'lines')):
            if fam not in families:
                continue
            for pi in phases:
                for sg in sigmas:
                    cid = f'{s}_{fam}_ph{pi}_s{int(sg)}' + ('' if tag == 'main' else f'_{tag}')
                    out.append(dict(case_id=cid, tag=tag, surface=s, family=fam, phase_index=pi, sigma=float(sg), seed=seed_for(si, fi, pi, sg), grid=list(grid)))
    return out


def dot_pattern(pi: int):
    u, v = SLL.positions()[DOT['positions'][pi]]
    c = SLL.centres(DOT['lattice'], u, v); binpat = SL.binary_pattern(c, DOT['r_mm'])
    pat = (binpat * np.float32(DOT['peak'])).astype(np.float32)
    return c, binpat, pat, dict(position_index=int(DOT['positions'][pi]), off_u=u, off_v=v, n_centres=int(len(c)), lit_px=int((binpat > 0).sum()), r_px=float(DOT['r_mm'] / SL.mmpp()))


def dot_case(mesh: dict, pi: int, sigma: float, seed: int) -> dict:
    from m3_virtual_camera import synthesise_camera_image
    g = C0.geom()
    c, binpat, pat, pinfo = dot_pattern(pi)
    t0 = time.monotonic()
    img, _, _, m3 = synthesise_camera_image(mesh, pat, m3_cfg(sigma, seed))
    t_img = time.monotonic() - t0; t0 = time.monotonic()
    cd = RD.candidates(img); feats, det_info, code, fidx, _ = RD.select(cd, DOT['group'])
    cfg = dot_cfg()
    pts, info, tr = RD.reconstruct_from(feats, det_info, binpat, cfg=cfg)
    t_dec = time.monotonic() - t0
    z0, z1 = M4.search_window(cfg, float(cfg['measurement_volume_mm']))

    n = len(feats)
    truth = np.full(n, -1, np.int64); conf = np.zeros(n, bool); dist = np.full(n, np.nan)
    if n:
        hit, P = C0.cast(feats, g['cam'], mesh); uvp, inside = C0.project(P, g['proj'])
        d = np.hypot(uvp[:, None, 0] - c[None, :, 0], uvp[:, None, 1] - c[None, :, 1]); j = d.argmin(axis=1); dmin = d[np.arange(n), j]
        conf = hit & inside & (dmin <= pinfo['r_px'] + 1.0); truth = np.where(conf, j, -1); dist = dmin
    proj_dot, _ = RD.projector_dot_index(binpat, c)
    matched = np.where(tr['best_idx'] >= 0, proj_dot[np.clip(tr['best_idx'], 0, None)], -1)
    pidx = tr['point_idx']; has_pt = pidx >= 0
    acc = np.zeros(len(pts), bool); acc[pidx[has_pt]] = True
    amb = np.zeros(len(pts), bool); amb[pidx[has_pt & tr['ambiguous']]] = True
    acc &= ~amb & np.isfinite(pts).all(axis=1)
    corr_f = has_pt & conf & (matched == truth); wrong_f = has_pt & conf & (matched != truth); unconf_f = has_pt & ~conf
    exp = SL.expected_dots(mesh, c)
    uv_e = np.asarray(exp['uv']); in_img = exp['hit'] & np.isfinite(uv_e).all(axis=1) & (uv_e[:, 0] >= 0) & (uv_e[:, 0] <= IMG - 1) & (uv_e[:, 1] >= 0) & (uv_e[:, 1] <= IMG - 1)
    n_exp = int(in_img.sum())
    ident = dict(n_detected=int(n), n_matched=int(has_pt.sum()), n_ambiguous=int(amb.sum()), n_confirmable=int(conf.sum()),
                 correct=int((corr_f & ~tr['ambiguous']).sum()), wrong=int((wrong_f & ~tr['ambiguous']).sum()), unconfirmable=int((unconf_f & ~tr['ambiguous']).sum()),
                 n_expected_visible=n_exp, n_expected_inner=int(exp['cam_inner'].sum()), recovery=(float((corr_f & ~tr['ambiguous']).sum() / n_exp) if n_exp else None),
                 recovery_def='correct-identity accepted (non-ambiguous) points / dots whose centre ray hits the surface and projects into the camera image')
    eff = dict(window_z0_mm=z0, window_z1_mm=z1, measurement_volume_mm=float(cfg['measurement_volume_mm']), search_z_min_mm=float(cfg['search_z_min_mm']), working_dist_mm=float(cfg['working_dist_mm']),
               baseline_mm=float(cfg['baseline_mm']), pattern=dict(**pinfo, lattice=DOT['lattice'], r_mm=DOT['r_mm'], peak=DOT['peak'], group=DOT['group'], binary_sha=C0.sha_array(binpat)),
               m3=dict(noise_sigma=float(m3['noise_sigma']), mottle_seed=int(m3['mottle_seed']), psf_sigma_px=float(m3['psf_sigma_px']), z_alignment=m3.get('z_alignment'), n_hits=int(m3['n_hits']),
                       n_rays_cast=int(m3['n_rays_cast']), surface_z_range_mm=list(m3['surface_z_range_mm'])))
    det = {k: v for k, v in info.items() if not isinstance(v, np.ndarray)}
    arrays = dict(img=img, pts=pts.astype(np.float32), acc=acc, amb=amb, feats=feats.astype(np.float32), truth_dot=truth.astype(np.int32), matched_dot=matched.astype(np.int32),
                  confirmable=conf, point_idx=pidx.astype(np.int32), centres=c.astype(np.float32), exp_uv=np.asarray(exp['uv'], np.float32), exp_in_image=in_img, exp_xy=np.asarray(exp['xy'], np.float32))
    return dict(points=pts, accept=acc, img=img, effective=eff, identity=ident, detect=det, seconds=dict(image=round(t_img, 2), decode=round(t_dec, 2)), arrays=arrays)


def line_pattern(pi: int):
    spec = dict(s_mm=LINE['s_mm'], w_mm=LINE['w_mm'], phi_deg=LINE['phi_deg'], phase_frac=float(LINE['phases'][pi]))
    pat_bin, meta = B.pattern_for(spec); pat = (pat_bin * np.float32(LINE['peak'])).astype(np.float32)
    return spec, pat_bin, pat, meta


def _row_flags(ref, feats, ids, acc, gate=3.0):
    Rr = ref['rows']; n = len(Rr['v']); by_row: dict = {}
    for i in np.flatnonzero(acc):
        by_row.setdefault(int(round(feats[i, 1])), []).append(i)
    rec = np.zeros(n, bool); ident = np.zeros(n, bool)
    for j in range(n):
        cand = by_row.get(int(Rr['v'][j]), [])
        if not cand:
            continue
        du = np.array([feats[i, 0] - Rr['u'][j] for i in cand]); near = np.abs(du) <= gate
        if near.any():
            rec[j] = True
            if (ids[np.array(cand)[near]] == Rr['k'][j]).any():
                ident[j] = True
    return rec, ident


def line_case(mesh: dict, pi: int, sigma: float, seed: int) -> dict:
    from m3_virtual_camera import synthesise_camera_image
    from m4_lines import run_olines, STATUS
    spec, pat_bin, pat, meta = line_pattern(pi)
    t0 = time.monotonic()
    img, _, _, m3 = synthesise_camera_image(mesh, pat, B.m3_cfg(sigma, seed))
    t_img = time.monotonic() - t0; t0 = time.monotonic()
    cfg = line_cfg(LINE['peak'])
    r = run_olines(img, meta['olines'], cfg, pattern=pat, **P4)
    t_dec = time.monotonic() - t0
    if abs(float(r['window_z0_mm']) - Z0_MM) > 1e-9 or abs(float(r['window_z1_mm']) - Z1_MM) > 1e-9 or abs(float(r['depth_margin_mm']) - MARGIN_MM) > 1e-12:
        raise RuntimeError(f"effective line window {r['window_z0_mm']}, {r['window_z1_mm']}, {r['depth_margin_mm']} != registered [{Z0_MM}, {Z1_MM}], {MARGIN_MM}")
    feat = r['features']; acc = np.asarray(r['accepted'], bool); st = np.asarray(r['status']); pts = np.asarray(r['points'], np.float64)
    ident = T.truth_identity(mesh, feat, meta['olines']); kO = ident['k']; conf = ident['confirmable']
    ref = T.line_reference(mesh, meta['olines'], 1.0, pattern=pat)
    lm = T.line_metrics(ref, feat, r['line_id'], acc, chain=r['chain'], truth_ids=kO)
    rec, idr = _row_flags(ref, feat, r['line_id'], acc)
    n_ref = int(len(ref['rows']['v']))
    identity = dict(n_detected=int(len(feat)), n_accepted=int(acc.sum()), n_confirmable=int(conf.sum()), correct=int((acc & conf & (r['line_id'] == kO)).sum()), wrong=int((acc & conf & (r['line_id'] != kO)).sum()),
                    unconfirmable=int((acc & ~conf).sum()), rejected_by_reason={STATUS[k]: int((st == k).sum()) for k in STATUS if k != 0}, ref_rows=n_ref, ref_rows_recovered=int(rec.sum()),
                    ref_rows_identity=int(idr.sum()), recovery=(float(rec.sum() / n_ref) if n_ref else None), recovery_def='reference rows with an accepted feature within 3 px / reference rows',
                    line_metrics=clean(lm))
    eff = dict(window_z0_mm=float(r['window_z0_mm']), window_z1_mm=float(r['window_z1_mm']), depth_margin_mm=float(r['depth_margin_mm']), measurement_volume_mm=float(cfg['measurement_volume_mm']),
               search_z_min_mm=float(cfg['search_z_min_mm']), search_margin_mm=float(cfg['search_margin_mm']), working_dist_mm=float(cfg['working_dist_mm']), baseline_mm=float(cfg['baseline_mm']),
               pattern_peak=float(cfg['pattern_peak']), psf_sigma_px=float(cfg['psf_sigma_px']), path=dict(name=LINE['path'], **P4),
               pattern=dict(**spec, peak=LINE['peak'], regime=LINE['regime'], n_lines=int(meta['olines']['n_lines']), width_px=float(meta['olines']['width_px']), spacing_px=float(meta['olines']['spacing_px']),
                            lit_px=int((pat_bin > 0).sum()), binary_sha=C0.sha_array(pat_bin)),
               m3=dict(noise_sigma=float(m3['noise_sigma']), mottle_seed=int(m3['mottle_seed']), psf_sigma_px=float(m3['psf_sigma_px']), z_alignment=m3.get('z_alignment'), n_hits=int(m3['n_hits']),
                       n_rays_cast=int(m3['n_rays_cast']), surface_z_range_mm=list(m3['surface_z_range_mm'])))
    det = {k: v for k, v in r['detect_info'].items() if not isinstance(v, np.ndarray)}
    arrays = dict(img=img, pts=pts.astype(np.float32), acc=acc, feats=np.asarray(feat, np.float32), status=st.astype(np.int8), line_id=np.asarray(r['line_id']).astype(np.int32), truth_k=kO.astype(np.int32),
                  confirmable=conf, ref_rows_v=ref['rows']['v'].astype(np.int32), ref_rows_u=ref['rows']['u'].astype(np.float32), ref_rows_k=ref['rows']['k'].astype(np.int32), ref_recovered=rec)
    return dict(points=pts, accept=acc, img=img, effective=eff, identity=identity, detect=dict(det, counts=clean(r['counts'])), seconds=dict(image=round(t_img, 2), decode=round(t_dec, 2)), arrays=arrays)


def _stats(e):
    e = np.asarray(e, np.float64); e = e[np.isfinite(e)]
    if len(e) == 0:
        return dict(n=0, rmse=None, mae=None, p95_abs=None, bias=None, max_abs=None)
    return dict(n=int(len(e)), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), p95_abs=float(np.percentile(np.abs(e), 95)), bias=float(np.mean(e)), max_abs=float(np.abs(e).max()))


def point_validity(points, accept, mesh: dict) -> dict:

    pts = np.asarray(points, np.float64).reshape(-1, 3); acc = np.asarray(accept, bool)
    fin = np.isfinite(pts).all(axis=1)
    f = truth_interp(mesh)
    zr = np.full(len(pts), np.nan)
    if fin.any():
        zr[fin] = f(np.column_stack([pts[fin, 1], pts[fin, 0]]))
    in_grid = np.isfinite(zr)
    in_win = fin & (pts[:, 2] >= Z0_MM - MARGIN_MM) & (pts[:, 2] <= Z1_MM + MARGIN_MM)
    return dict(pts=pts, acc=acc, fin=fin, zr=zr, in_grid=in_grid, in_win=in_win, valid=acc & fin & in_grid & in_win)


# Evaluate reconstructed points against the corrected surface and fixed node domain.
def evaluate(points, accept, mesh: dict, dom: dict, gauss: bool, fields: bool = False) -> dict:

    from surface_eval_r3 import surface_eval_r3_dev
    V = point_validity(points, accept, mesh)
    pts, acc, fin, zr, in_grid, in_win, valid = V['pts'], V['acc'], V['fin'], V['zr'], V['in_grid'], V['in_win'], V['valid']
    n_out = int(len(pts))
    r = np.hypot(pts[:, 0], pts[:, 1]); roi = valid & (r <= ROI_R_MM)
    e_pts = pts[:, 2] - zr
    gross = roi & (np.abs(e_pts) > GROSS_MM)
    out = dict(n_output=n_out, n_accepted=int(acc.sum()), n_nonfinite=int((acc & ~fin).sum()), n_out_of_grid=int((acc & fin & ~in_grid).sum()), n_out_of_window=int((acc & fin & ~in_win).sum()),
               n_valid=int(valid.sum()), n_valid_roi=int(roi.sum()), n_gross_roi=int(gross.sum()), gross_rate_roi=(float(gross.sum() / roi.sum()) if roi.sum() else None),
               no_output=bool(valid.sum() == 0), pts_roi=_stats(e_pts[roi]), pts_all_valid=_stats(e_pts[valid]))
    P = pts[valid]
    dm = dict(x=dom['x'], y=dom['y'], z_ref=dom['z_ref'], area=dom['area'], hash=dom['hash'])
    z_est = np.full(dom['n_nodes'], np.nan); ev = None
    if len(P) >= 3:
        ev = surface_eval_r3_dev(P, None, dm, [6.0], [0.15], return_fields=True, support_rule='exact')
        z_est = np.asarray(ev.pop('fields')['z_est'], np.float64)
    e_nodes = z_est - dom['z_ref']; cov = np.isfinite(e_nodes)
    out['nodes_roi'] = dict(coverage=float(cov.mean()), **_stats(e_nodes[cov]), n_nodes=int(dom['n_nodes']), coverage_def='ROI nodes inside the hull of the valid points (linear interpolation) / ROI nodes')
    out['r3_dev'] = clean({k: v for k, v in ev.items()} if ev else dict(status='fewer than 3 valid points'))
    if gauss:
        dip = dom['z_base'] - z_est
        core = cov & (dom['r'] <= 10.0); centre = cov & (dom['r'] <= 1.5)
        a_peak = float(np.nanmax(dip[core])) if core.any() else None; a_cen = float(np.nanmean(dip[centre])) if centre.any() else None
        out['amplitude'] = dict(applicable=True, true_mm=float(M.GAUSS['d_mm']), rec_peak_mm=a_peak, rec_centre_mm=a_cen,
                                err_peak_mm=(a_peak - M.GAUSS['d_mm'] if a_peak is not None else None), err_centre_mm=(a_cen - M.GAUSS['d_mm'] if a_cen is not None else None),
                                err_peak_pct=(100.0 * (a_peak - M.GAUSS['d_mm']) / M.GAUSS['d_mm'] if a_peak is not None else None),
                                definition='dip = z_base - z_est on ROI nodes; peak = max over r <= 10 mm, centre = mean over r <= 1.5 mm')
    else:
        out['amplitude'] = dict(applicable=False, note='not applicable (no Gaussian reference term); no old T criterion')
    if fields:
        out['_z_est'] = z_est
    return out


def flat_row(case: dict, rec: dict) -> dict:

    ev = rec['evaluation']; idn = rec['identity']; eff = rec['effective']; am = ev['amplitude']
    row = dict(case_id=case['case_id'], tag=case['tag'], surface=case['surface'], family=case['family'], phase_index=case['phase_index'], sigma=case['sigma'], seed=case['seed'],
               grid_x_mm=case['grid'][0], grid_y_mm=case['grid'][1], c_ref_mm=C_REF, window_z0_mm=eff['window_z0_mm'], window_z1_mm=eff['window_z1_mm'],
               window_cov_roi=rec['domain']['window_cov_roi'], window_cov_fov=rec['domain']['window_cov_fov'], roi_z_min_mm=rec['domain']['z_range'][0], roi_z_max_mm=rec['domain']['z_range'][1],
               n_output=ev['n_output'], n_accepted=ev['n_accepted'], n_valid=ev['n_valid'], n_valid_roi=ev['n_valid_roi'], n_nonfinite=ev['n_nonfinite'], n_out_of_grid=ev['n_out_of_grid'],
               n_out_of_window=ev['n_out_of_window'], n_gross_roi=ev['n_gross_roi'], gross_rate_roi=ev['gross_rate_roi'], no_output=ev['no_output'],
               pts_roi_rmse_mm=ev['pts_roi']['rmse'], pts_roi_mae_mm=ev['pts_roi']['mae'], pts_roi_p95_mm=ev['pts_roi']['p95_abs'], pts_roi_bias_mm=ev['pts_roi']['bias'],
               nodes_roi_coverage=ev['nodes_roi']['coverage'], nodes_roi_rmse_mm=ev['nodes_roi']['rmse'], nodes_roi_mae_mm=ev['nodes_roi']['mae'], nodes_roi_p95_mm=ev['nodes_roi']['p95_abs'],
               n_detected=idn['n_detected'], n_confirmable=idn['n_confirmable'], id_correct=idn['correct'], id_wrong=idn['wrong'], id_unconfirmable=idn['unconfirmable'],
               recovery=idn['recovery'], recovery_denominator=(idn.get('n_expected_visible') if case['family'] == 'dots' else idn.get('ref_rows')),
               amplitude_applicable=am['applicable'], amp_true_mm=am.get('true_mm'), amp_rec_peak_mm=am.get('rec_peak_mm'), amp_rec_centre_mm=am.get('rec_centre_mm'),
               amp_err_peak_mm=am.get('err_peak_mm'), amp_err_centre_mm=am.get('err_centre_mm'), amp_err_peak_pct=am.get('err_peak_pct'),
               m3_n_hits=eff['m3']['n_hits'], m3_z_alignment=eff['m3']['z_alignment'], seconds_image=rec['seconds']['image'], seconds_decode=rec['seconds']['decode'], seconds_eval=rec['seconds']['eval'])
    if case['family'] == 'dots':
        row.update(n_ambiguous=idn['n_ambiguous'], n_expected_visible=idn['n_expected_visible'])
    return row


def run_case(case: dict) -> dict:
    t0 = time.monotonic()
    mesh = build_mesh(case['surface'], tuple(case['grid'])); dom = roi_domain(mesh)
    gauss = SURFACES[case['surface']]['gauss']
    rec = dot_case(mesh, case['phase_index'], case['sigma'], case['seed']) if case['family'] == 'dots' else line_case(mesh, case['phase_index'], case['sigma'], case['seed'])
    t1 = time.monotonic()
    ev = evaluate(rec['points'], rec['accept'], mesh, dom, gauss, fields=True)
    z_est = ev.pop('_z_est')
    rec['seconds']['eval'] = round(time.monotonic() - t1, 2); rec['seconds']['total'] = round(time.monotonic() - t0, 2)
    arrays = rec.pop('arrays'); arrays['z_est_roi'] = z_est.astype(np.float32); arrays['roi_idx'] = dom['idx'].astype(np.int64)
    rec.pop('points'); rec.pop('accept'); rec.pop('img')
    rec['evaluation'] = ev
    rec['domain'] = dict(n_nodes=dom['n_nodes'], area_total_mm2=dom['area_total_mm2'], hash=dom['hash'], z_range=list(dom['z_range']), window_cov_roi=dom['window_cov_roi'], window_cov_fov=dom['window_cov_fov'],
                         z_range_fov=list(dom['z_range_fov']), roi_r_mm=ROI_R_MM)
    rec['mesh'] = dict(sha16=mesh_sha(mesh), **mesh['meta'])
    rec['case'] = case; rec['version'] = VERSION; rec['finished_local'] = C0.now_local()
    np.savez_compressed(OUT / 'cases' / f"{case['case_id']}.npz", **arrays)
    (OUT / 'cases' / f"{case['case_id']}.json").write_text(json.dumps(clean(rec), ensure_ascii=False, indent=1), encoding='utf-8')
    return clean(rec)


def load_case(case_id: str):
    rec = json.loads((OUT / 'cases' / f'{case_id}.json').read_text(encoding='utf-8'))
    z = np.load(OUT / 'cases' / f'{case_id}.npz')
    return rec, z
