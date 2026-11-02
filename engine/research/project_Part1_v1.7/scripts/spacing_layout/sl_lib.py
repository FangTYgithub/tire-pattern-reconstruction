# Generate dot lattices and evaluate sampling, detection and correspondence outcomes.

import sys, json, time, hashlib, traceback
from pathlib import Path

import numpy as np

import sl_common as S

ROOT = S.ROOT
sys.path[:0] = [str(ROOT / 'scripts' / 'spotnoise_round5'), str(ROOT / 'scripts' / 'spotnoise_round4'), str(ROOT / 'scripts' / 'spotnoise_round3'),
                str(ROOT / 'scripts' / 'spotnoise_suppA'), str(ROOT / 'scripts' / 'spotnoise'), str(ROOT / 'scripts' / 'pm015'),
                str(ROOT / 'scripts' / 'pm014'), str(ROOT / 'scripts' / 'pm013'), str(ROOT / 'src'), str(ROOT)]
import sn_lib as SL
import r5_detect as RD
import r4_wp2 as W2
import sn_posthoc as PH
import sa_defs as SAD
import pm014_surface as S14
import pm014_lib as L14
import pm015_lib as L15

OUT = ROOT / 'outputs' / 'spacing_layout'
CASES = OUT / 'cases'
S.OUT = OUT
VERSION = 2
R_MM = 1.0
P_PAR = (5.0, 6.0, 7.35, 8.0, 10.0, 12.0)
P_PERP = (2.5, 3.0, 4.0, 6.0, 8.0)
ARR = ('R', 'T')
SURFACES = ('baseline', 'curv_x_strong')
FWHM_MAIN = 20.0
FWHM_SENS = (15.0, 30.0)
N_GRID = 16
LATIN8 = [((2 * k + 1 + 0.5) / N_GRID, (((5 * k + 2) % N_GRID) + 0.5) / N_GRID) for k in range(8)]
B_LATTICES = ['R6x4', 'R7.35x4', 'R8x4', 'R10x4', 'R12x4', 'R8x2.5', 'R8x3', 'R8x6', 'R8x8', 'T8x4']
BUDGET_REF = 'R8x8'
NOISE = {'A': ((0.0, (0,)), (40.0, (0, 1)), (80.0, (0, 1))), 'B': ((0.0, (0,)), (20.0, (0, 1)), (40.0, (0, 1)))}
NOISE_BASE = 9420000
PSF_PX = 1.0
VOLUME_MM = SL.VOLUME_MM
GATE_XY, GATE_Z = 1.5, 1.0
T_CRIT = dict(gross_max=0.01, recovery_min=0.9, dent_abs_max_pct=10.0, rmse_sigma0=0.06, rmse_noisy=0.15)
STEP = ['no_component', 'abs_area_small', 'abs_area_large', 'border', 'intensity', 'm00', 'rel_area_small', 'rel_area_large', 'elongation',
        'unmatched', 'correct', 'wrong', 'merged_matched', 'partial_matched', 'unlabelled_matched']
CLS = {0: 'background', 1: 'single_source', 2: 'partial', 3: 'multi_source', 4: 'ref_unlabelled'}
_SURF = {}


def lattice(name):
    arr = name[0]; a, b = name[1:].split('x')
    return dict(name=name, p_par=float(a), p_perp=float(b), stagger=0.5 if arr == 'T' else 0.0, arrangement=arr)


def all_lattices():
    return [f'{arr}{a:g}x{b:g}' for a in P_PAR for b in P_PERP for arr in ARR]


def positions():
    g = (np.arange(N_GRID) + 0.5) / N_GRID
    return [(float(u), float(v)) for v in g for u in g]


def latin_indices():
    pos = positions()
    return [pos.index((u, v)) for u, v in LATIN8]


def centres(name, u, v):

    from m2_pattern import _lattice_centres
    L = lattice(name); k = SL.mmpp()
    cx, cy, _, _ = _lattice_centres(SL.IMG, L['p_par'] / k, L['p_perp'] / k, u * L['p_par'] / k, v * L['p_perp'] / k, stagger_frac=L['stagger'])
    cx, cy = np.ravel(cx), np.ravel(cy); m = SL.edge_margin_px()
    keep = (cx >= m) & (cx <= SL.IMG - 1 - m) & (cy >= m) & (cy <= SL.IMG - 1 - m)
    c = np.column_stack([cx[keep], cy[keep]]).astype(np.float64)
    return c[np.lexsort((c[:, 0], c[:, 1]))]


def lit_pixels(name, u, v, r_mm=R_MM):
    return float(SL.binary_pattern(centres(name, u, v), r_mm).sum())


def surface(surf, fwhm=FWHM_MAIN):

    key = (surf, float(fwhm))
    if key not in _SURF:
        _SURF[key] = S14.build_fixed_surface('A', L15.SURFACES[surf], L14.GRIDS[L15.GRID], {**L14.BASE, 'indent_width_mm': float(fwhm)}, L15.L_MM)
    return _SURF[key]


def ctx(surf, fwhm=FWHM_MAIN):

    from scipy.interpolate import RegularGridInterpolator
    mesh, cfg, ref = surface(surf, fwhm)
    X = np.asarray(mesh['X'], float); Y = np.asarray(mesh['Y'], float); Z = np.asarray(mesh['Z'], float); Zb = np.asarray(mesh['Z_base'], float)
    zmin = float(Z.min())
    base = RegularGridInterpolator((Y[:, 0], X[0, :]), Zb - zmin, bounds_error=False, fill_value=np.nan)
    return dict(mesh=mesh, cfg=cfg, ref=ref, zmin=zmin, base=base, half=float(cfg.get('fov_half_width_mm', 46.0)), fwhm=float(fwhm))


def rho_crit(fwhm, eps):

    sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return float(sig * np.sqrt(2.0 * np.log(1.0 / (1.0 - eps))))


def disc_gaps(xy, r_disc=SL.DENT_R_MM, step=0.25):
    from scipy.spatial import cKDTree
    g = np.arange(-r_disc, r_disc + 1e-9, step); GX, GY = np.meshgrid(g, g); m = GX ** 2 + GY ** 2 <= r_disc ** 2
    if len(xy) == 0:
        return None, None
    dn, _ = cKDTree(xy).query(np.column_stack([GX[m], GY[m]]))
    return dn, float(np.hypot(xy[:, 0], xy[:, 1]).min())


def stage_a(c_ctx, exp):

    xy, uv = exp['xy'], exp['uv']
    hit = np.isfinite(xy).all(axis=1); vis = hit & np.isfinite(uv).all(axis=1)
    pb = W2.exact_points(c_ctx, xy[vis]) if vis.any() else np.empty((0, 3))
    B, o = (W2.official(c_ctx, pb) if len(pb) >= 4 else (None, {}))
    db = W2.dips(c_ctx, pb) if len(pb) else np.array([])
    dn, rmin = disc_gaps(xy[vis]) if vis.any() else (None, None)
    fw = c_ctx['fwhm']
    r10, r5 = rho_crit(fw, 0.10), rho_crit(fw, 0.05)
    return dict(n_visible=int(vis.sum()), n_hit=int(hit.sum()), n_disc=int((vis & ((xy[:, 0] ** 2 + xy[:, 1] ** 2) <= SL.DENT_R_MM ** 2)).sum()),
                B_mm=B, S_pct=(100.0 * (B - 5.0) / 5.0) if B is not None else None,
                B_pts_mm=float(np.nanmax(db)) if len(db) and np.isfinite(db).any() else None,
                S_pts_pct=(100.0 * (float(np.nanmax(db)) - 5.0) / 5.0) if len(db) and np.isfinite(db).any() else None,
                centre_err_mm=o.get('ind_centre_err_mm') if o else None, iou=o.get('ind_iou') if o else None,
                rho_min_mm=rmin, rho_min_over_fwhm=(rmin / fw) if rmin is not None else None,
                cover_radius_mm=float(dn.max()) if dn is not None else None, cover_over_fwhm=(float(dn.max()) / fw) if dn is not None else None,
                frac_beyond_rho10=float((dn > r10).mean()) if dn is not None else None, frac_beyond_rho5=float((dn > r5).mean()) if dn is not None else None,
                rho10_mm=r10, rho5_mm=r5)


def stage_a_eval(name, pos_index, surf, fwhm):
    u, v = positions()[pos_index]
    cx = ctx(surf, fwhm)
    c = centres(name, u, v)
    exp = SL.expected_dots(cx['mesh'], c)
    out = dict(lattice=name, **{k: lattice(name)[k] for k in ('p_par', 'p_perp', 'stagger', 'arrangement')}, pos_index=pos_index, off_u=u, off_v=v,
               surface=surf, fwhm=fwhm, n_dots=int(len(c)))
    out.update(stage_a(cx, exp))
    return out


def spec(group, name, pos_index, surf, sigma, k):
    u, v = positions()[pos_index]
    return dict(sl=VERSION, group=group, lattice=name, **{kk: lattice(name)[kk] for kk in ('p_par', 'p_perp', 'stagger', 'arrangement')},
                pos_index=int(pos_index), off_u=u, off_v=v, r_mm=R_MM, surface=surf, fwhm=FWHM_MAIN, sigma=float(sigma),
                k=int(k) if sigma > 0 else 0, noise_base=NOISE_BASE, psf_px=PSF_PX)


def key(s):
    return SL.digest({kk: s[kk] for kk in ('sl', 'group', 'lattice', 'off_u', 'off_v', 'r_mm', 'surface', 'fwhm', 'sigma', 'k', 'noise_base', 'psf_px')})


def noise_seed(s):
    return int(s['noise_base'] + 10 * s['pos_index'] + s['k'])


def design_b():
    out = []
    for g in ('A', 'B'):
        for name in B_LATTICES:
            for pi in latin_indices():
                for surf in SURFACES:
                    for sg, ks in NOISE[g]:
                        for k in ks:
                            out.append(spec(g, name, pi, surf, sg, k))
    return out


def budget(design_meta):
    return float(design_meta['budget_lit_px'])


def dot_value(s, binpat, bud):

    if s['group'] == 'A':
        return 1.0
    return float(bud) / float(binpat.sum())


def chain(mesh, cfg, pat, binpat, sigma, seed, p_par):

    from m3_virtual_camera import synthesise_camera_image
    from m5_evaluation import evaluate_all
    import preliminary_analysis as PA
    m3_cfg = {'batch_size': PA.BATCH_SIZE, 'noise_sigma': float(sigma), 'psf_sigma_px': PSF_PX, 'albedo': 1.0, 'albedo_exposure': 1.0,
              'specular_k': 0.0, 'mottle_amp': 0.0, 'outlier_frac': 0.0, 'mottle_seed': int(seed)}
    img, _, _, m3_info = synthesise_camera_image(mesh_dict=mesh, pattern=pat, cfg=m3_cfg)
    cd = RD.candidates(img)
    feats, det_info, code, fidx, _ref = RD.select(cd, 'D0')
    pts, info, tr = RD.reconstruct_from(feats, det_info, binpat)
    m = evaluate_all(points_3d=pts, mesh=mesh, camera_image=img, m1_config=cfg, m3_info=m3_info, uniq_window_px=PA.UNIQ_WINDOW_PX,
                     uniq_stride_px=PA.UNIQ_STRIDE_PX, uniq_n_samples=PA.UNIQ_N_SAMPLES, spacing_mm=p_par, theta_deg=None, recon_info=info)
    return img, cd, code, fidx, pts, info, tr, m, m3_info


def _f(v):
    return float(v) if v is not None and np.isfinite(v) else None


def task_criterion(ex, status_valid, sigma):

    def le(x, lim):
        return x is not None and np.isfinite(x) and x <= lim
    g, rec, dent, rmse = ex.get('gross_rate_rel'), ex.get('recovery'), ex.get('dent_rel_pct'), ex.get('ref_rmse_rel_mm')
    lim = T_CRIT['rmse_sigma0'] if sigma == 0 else T_CRIT['rmse_noisy']
    parts = dict(data_valid=bool(status_valid), gross=le(g, T_CRIT['gross_max']),
                 recovery=(rec is not None and np.isfinite(rec) and rec >= T_CRIT['recovery_min']),
                 dent=(dent is not None and np.isfinite(dent) and abs(dent) <= T_CRIT['dent_abs_max_pct']), rmse=le(rmse, lim))
    parts['T_pass'] = all(parts.values())
    parts['undefined'] = [k for k, x in (('gross', g), ('recovery', rec), ('dent', dent), ('rmse', rmse)) if x is None or not np.isfinite(x)]
    return parts


# Build a dot layout, run reconstruction and evaluate expected-position recovery.
def work(s, bud):
    t0 = time.monotonic(); k = key(s)
    try:
        mesh, cfg, ref = surface(s['surface'], s['fwhm'])
        c = centres(s['lattice'], s['off_u'], s['off_v'])
        binpat = SL.binary_pattern(c, s['r_mm'])
        v = dot_value(s, binpat, bud)
        if v > 1.0 + 1e-9:
            raise RuntimeError(f'dot value {v} exceeds 1 (budget not realisable)')
        pat = (binpat * np.float32(v)).astype(np.float32)
        pchk = SL.pattern_checks(binpat, c)
        exp = SL.expected_dots(mesh, c)
        t1 = time.monotonic()
        img, cd, code, fidx, pts, info, tr, m, m3_info = chain(mesh, cfg, pat, binpat, s['sigma'], noise_seed(s), s['p_par'])
        t_run = time.monotonic() - t1
        pts = np.asarray(pts, np.float32)
        amb = np.asarray(info.get('ambiguous_mask', np.zeros(len(pts), bool)), bool)
        if len(amb) != len(pts):
            amb = np.zeros(len(pts), bool)
        fin = np.isfinite(pts).all(axis=1); rel = fin & ~amb; n_rel = int(rel.sum())
        gt = SL.truth_depth(pts, mesh).astype(np.float32)
        err = np.where(fin, pts[:, 2] - gt, np.nan).astype(np.float32)
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
        rec, nn_rec, n_q = PH.recovery(pts, amb, err_ref, exp_disc, GATE_XY, GATE_Z)
        rec10, _, _ = PH.recovery(pts, amb, err_ref, exp_disc, 1.0, GATE_Z)
        good = rel & ok_ref
        dn_real, _ = disc_gaps(pts[good & in_disc][:, :2].astype(np.float64))
        wnn = _nn(exp_disc); cnn = _nn(exp['uv'][exp['cam_inner']])
        proj_dot, proj_dist = RD.projector_dot_index(binpat, c)
        ex = dict(

            n_dots=int(len(c)), n_exp_disc=n_exp_disc, n_exp_cam_inner=int(exp['cam_inner'].sum()), dot_value=float(v), lit_px=float(binpat.sum()),
            pattern_integral=float(pat.sum(dtype=np.float64)), budget_lit_px=float(bud), pattern_lit_frac=float((binpat > 0).mean()),
            r_px=float(s['r_mm'] / SL.mmpp()), d_mm=float(2 * s['r_mm']), pattern_checks=pchk, world_nn_min_disc_mm=wnn[0], world_nn_median_disc_mm=wnn[1],
            cam_nn_min_px=cnn[0], cam_nn_median_px=cnn[1], noise_seed_m3=noise_seed(s), noise_rng_seed=noise_seed(s) + 104729,

            image_sat_frac=float((img >= 255).mean()), otsu_threshold=_f(info.get('otsu_threshold')), n_components=int(cd['n_labels'] - 1),
            n_candidates=int(len(cd['cands'])), n_detected=int(info.get('n_detected', 0) or 0), n_area_rejected=int(info.get('n_area_rejected', 0) or 0),
            n_elong_rejected=int(info.get('n_elong_rejected', 0) or 0), n_filtered_noise=int(info.get('n_filtered_noise', 0) or 0),
            n_border_excluded=int(info.get('n_border_excluded', 0) or 0), blob_area_median_px=_f(info.get('blob_area_median_px')),
            n_proj_features=int(len(proj_dot)), proj_dot_max_dist_px=float(proj_dist.max()) if len(proj_dist) else None,

            n_matched=int(info.get('n_matched', 0) or 0), n_ambiguous=int(info.get('n_ambiguous', 0) or 0), n_finite=int(fin.sum()), n_rel=n_rel,
            A=_f(mc.get('A_ambiguity')), data_valid=bool(mc.get('data_valid')), n_rel_disc=int((in_disc & rel).sum()), n_good_disc=int((in_disc & good).sum()),
            completeness=(int((in_disc & rel).sum()) / n_exp_disc) if n_exp_disc else None,
            ref_rmse_rel_mm=float(np.sqrt(np.mean(r_rel ** 2))) if len(r_rel) else None, ref_p95_rel_mm=float(np.percentile(np.abs(r_rel), 95)) if len(r_rel) else None,
            ref_bias_rel_mm=float(np.mean(r_rel)) if len(r_rel) else None, gross_rate_rel=float((np.abs(r_rel) > SL.OUTLIER_MM).mean()) if len(r_rel) else None,
            n_gross_rel=int((np.abs(r_rel) > SL.OUTLIER_MM).sum()), recovery=None if not np.isfinite(rec) else float(rec),
            recovery_nn_dedup=None if not np.isfinite(nn_rec) else float(nn_rec), recovery_gate1=None if not np.isfinite(rec10) else float(rec10),
            n_qualified_outputs=int(n_q), cover_radius_realised_mm=float(dn_real.max()) if dn_real is not None else None,

            dent_err_mm=d_err, dent_rel_pct=(100.0 * d_err / d_gt) if (d_err is not None and d_gt) else None,
            dent_pts_rel_pct=(100.0 * (d_pts - d_gt) / d_gt) if (d_pts is not None and d_gt) else None, dent_interp_coverage=mc.get('ind_coverage_frac'),
            seconds_pipeline=round(t_run, 3), peak_rss_mb=SL.peak_rss_mb())
        ex.update({kk + '_rel': val for kk, val in SL.occupancy(pts, rel).items()})
        st = SL.status_of(mc, n_rel)
        legacy = SAD.layers(st, ex, ex['recovery'], s['sigma'])
        T = task_criterion(ex, ex['data_valid'], s['sigma'])
        unavailable = ex['n_finite'] == 0 or ex['dent_rel_pct'] is None
        return dict(spec=s, key=k, status=st, legacy=legacy, task=T, unavailable=bool(unavailable), seconds=time.monotonic() - t0, finished=S.now(),
                    metrics=mc, extras=ex,
                    arrays=dict(cam_img=img.astype(np.uint8), pattern_bin=(binpat > 0).astype(np.uint8), centres_px=c.astype(np.float32),
                                exp_xy=exp['xy'].astype(np.float32), exp_uv=exp['uv'].astype(np.float32), exp_in_disc=exp['in_disc'], exp_cam_inner=exp['cam_inner'],
                                pts=pts, amb=amb, gt_depth=gt, gt_ref=dref.astype(np.float32), err=err, err_ref=err_ref,
                                cand_uv=np.array([[q[0], q[1]] for q in cd['cands']], np.float32).reshape(-1, 2), cand_area=np.array([q[2] for q in cd['cands']], np.int32),
                                cand_label=np.array([q[5] for q in cd['cands']], np.int32), cand_code=code.astype(np.int8), cand_fidx=fidx.astype(np.int32),
                                comp_reason=cd['reason'].astype(np.int8), feat_matched=tr['matched'], feat_ambiguous=tr['ambiguous'], feat_best_idx=tr['best_idx'].astype(np.int32),
                                feat_point_idx=tr['point_idx'].astype(np.int32), proj_dot=proj_dot.astype(np.int32)))
    except Exception:
        return dict(spec=s, key=k, status='error', seconds=time.monotonic() - t0, finished=S.now(), error=traceback.format_exc(), metrics=None)


def _nn(p):
    from scipy.spatial import cKDTree
    p = np.asarray(p, float); p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 2:
        return None, None
    d, _ = cKDTree(p).query(p, k=2)
    return float(d[:, 1].min()), float(np.median(d[:, 1]))


def store(r, base=CASES):
    p = base / f"{r['key']}.json"; p.parent.mkdir(parents=True, exist_ok=True)
    arrays = r.pop('arrays', None)
    if arrays is not None:
        tmp = p.with_suffix('.npz.tmp')
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(p.with_suffix('.npz')); r['npz'] = p.with_suffix('.npz').name
    S.save_json(p, r)
    return p


def reference_map(img0, uv):

    import cv2
    t0, b0 = cv2.threshold(img0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n0, lab0, _, _ = cv2.connectedComponentsWithStats(b0)
    H, W = img0.shape
    owners = {}
    for i, (u, v) in enumerate(uv):
        if np.isfinite(u) and np.isfinite(v):
            iu, iv = int(round(u)), int(round(v))
            if 0 <= iu < W and 0 <= iv < H and lab0[iv, iu] > 0:
                owners.setdefault(int(lab0[iv, iu]), []).append(i)
    Sm = np.full(lab0.shape, -1, np.int64)
    Sm[lab0 > 0] = -3
    merged = set()
    for L, ids in owners.items():
        pix = np.argwhere(lab0 == L)
        if len(ids) == 1:
            Sm[pix[:, 0], pix[:, 1]] = ids[0]
        else:
            P = uv[ids]
            d = np.hypot(pix[:, 1][:, None] - P[None, :, 0], pix[:, 0][:, None] - P[None, :, 1])
            Sm[pix[:, 0], pix[:, 1]] = np.asarray(ids)[d.argmin(axis=1)]
            merged.update(ids)
    return Sm, dict(t0=float(t0), n_components=int(n0 - 1), n_owner_components=len(owners), n_ref_merged_dots=len(merged),
                    n_unlabelled_components=int(n0 - 1 - len(owners)), merged_dots=sorted(merged))


def classify(label_map, n_labels, Sm):

    tl = label_map.ravel(); sv = Sm.ravel()
    m = (tl > 0) & (sv >= 0)
    n_dot = int(Sm.max()) + 1 if (Sm >= 0).any() else 1
    size_i = np.bincount(sv[sv >= 0], minlength=n_dot)
    key_ = tl[m].astype(np.int64) * (n_dot + 1) + sv[m]
    uk, cnt = np.unique(key_, return_counts=True)
    comp, dot = uk // (n_dot + 1), uk % (n_dot + 1)
    n_touch = np.bincount(comp, minlength=n_labels)
    best_dot = np.full(n_labels, -1, np.int64); best_ov = np.zeros(n_labels, np.int64)
    for c_, d_, n_ in zip(comp, dot, cnt):
        if n_ > best_ov[c_]:
            best_dot[c_], best_ov[c_] = d_, n_
    touch_unl = np.bincount(tl[(tl > 0) & (sv == -3)], minlength=n_labels) > 0
    cls = np.zeros(n_labels, np.int8)
    one = n_touch == 1
    frac = np.where(best_dot >= 0, best_ov / np.maximum(size_i[np.clip(best_dot, 0, None)], 1), 0.0)
    cls[one & (frac >= 0.5)] = 1; cls[one & (frac < 0.5)] = 2; cls[n_touch >= 2] = 3; cls[(n_touch == 0) & touch_unl] = 4; cls[0] = -1
    cover = {}
    for c_, d_, n_ in zip(comp, dot, cnt):
        if d_ not in cover or n_ > cover[d_][1] or (n_ == cover[d_][1] and c_ < cover[d_][0]):
            cover[int(d_)] = (int(c_), int(n_))
    return dict(cls=cls, dot=np.where(one, best_dot, -1), overlap=best_ov, n_touch=n_touch, cover=cover, size_i=size_i)


def dot_outcomes(z, cd, cl):

    n = len(z['exp_uv']); out = np.full(n, -1, np.int8); amb_dot = np.zeros(n, bool); err_dot = np.full(n, np.nan, np.float32)
    lab_to_cand = {int(l): j for j, l in enumerate(z['cand_label'])}
    for i in range(n):
        if not np.isfinite(z['exp_uv'][i]).all():
            continue
        if i not in cl['cover']:
            out[i] = STEP.index('no_component'); continue
        L, _ = cl['cover'][i]
        r = int(cd['reason'][L])
        if r > 0:
            out[i] = r; continue
        j = lab_to_cand.get(L)
        if j is None:
            out[i] = STEP.index('no_component'); continue
        code = int(z['cand_code'][j])
        if code in (1, 2, 3):
            out[i] = STEP.index(['rel_area_small', 'rel_area_large', 'elongation'][code - 1]); continue
        f = int(z['cand_fidx'][j])
        if f < 0 or not z['feat_matched'][f]:
            out[i] = STEP.index('unmatched'); continue
        c = int(cl['cls'][L])
        if c == 3:
            out[i] = STEP.index('merged_matched')
        elif c == 2:
            out[i] = STEP.index('partial_matched')
        elif c == 4 or c == 0:
            out[i] = STEP.index('unlabelled_matched')
        else:
            md = int(z['proj_dot'][int(z['feat_best_idx'][f])]) if z['feat_best_idx'][f] >= 0 else -1
            out[i] = STEP.index('correct') if md == i else STEP.index('wrong')
        amb_dot[i] = bool(z['feat_ambiguous'][f])
        pidx = int(z['feat_point_idx'][f])
        if pidx >= 0:
            err_dot[i] = z['err_ref'][pidx]
    return out, amb_dot, err_dot


def output_identity(z, cl):

    n = len(z['pts']); ident = np.full(n, 'none', dtype=object)
    cand_of_feat = {int(f): j for j, f in enumerate(z['cand_fidx']) if f >= 0}
    for f in range(len(z['feat_matched'])):
        p = int(z['feat_point_idx'][f])
        if p < 0:
            continue
        j = cand_of_feat.get(f); L = int(z['cand_label'][j]) if j is not None else -1
        c = int(cl['cls'][L]) if L >= 0 else 0
        if c == 1:
            md = int(z['proj_dot'][int(z['feat_best_idx'][f])]) if z['feat_best_idx'][f] >= 0 else -1
            ident[p] = 'correct' if md == int(cl['dot'][L]) else 'wrong'
        else:
            ident[p] = {0: 'background', 2: 'undeterminable_partial', 3: 'undeterminable_multi', 4: 'undeterminable_unlabelled'}.get(c, 'none')
    return ident
