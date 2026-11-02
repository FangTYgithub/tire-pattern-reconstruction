# Run dot-size and noise cases with controlled brightness and geometric references.

import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import sys, json, time, hashlib, traceback, ctypes
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'scripts' / 'pm015'), str(ROOT / 'scripts' / 'pm014'), str(ROOT / 'scripts' / 'pm013'), str(ROOT), str(ROOT / 'src')]
import pm015_lib as L15
import pm014_surface as S

OUT = ROOT / 'outputs' / 'spotnoise'
VERSION = 1
IMG = 512
RADII = (0.5, 0.75, 1.0, 1.25, 1.5)
R_MIN, R_MAX = min(RADII), max(RADII)
SIGMAS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
SEEDS = (0, 1, 2)
PHASES = (0, 1, 2)
SURFACES = ('baseline', 'curv_x_strong')
BRIGHT = ('A', 'B')
LATTICES = {'S1': (8.0, 4.0), 'S2': (8.0, 6.0)}
PSF_PX = 1.0
VOLUME_MM = 20.0
DENT_R_MM = 15.0
OUTLIER_MM = 1.0
CAM_INNER_PX = 16.0
NOISE_BASE = 9205000
KEY_FIELDS = ('v', 'lattice', 'surface', 'phase', 'r_mm', 'bright', 'sigma', 'seed')
clean, save, digest, now, quantiles, occupancy, status_of, truth_depth = (L15.clean, L15.save, L15.digest, L15.now, L15.quantiles,
                                                                          L15.occupancy, L15.status_of, L15.truth_depth)
_MMPP = None


def mmpp():
    global _MMPP
    if _MMPP is None:
        from m2_pattern import _compute_fov_mm, FOCAL_LENGTH_MM, PIXEL_PITCH_MM, WORKING_DIST_MM
        _MMPP = float(_compute_fov_mm(IMG, FOCAL_LENGTH_MM, PIXEL_PITCH_MM, WORKING_DIST_MM)[2])
    return _MMPP


def edge_margin_px():
    return R_MAX / mmpp() + 2.0


def centres(lattice, phase):

    from m2_pattern import _lattice_centres
    k = mmpp(); sp, pp = LATTICES[lattice]; u, v = quantiles(phase)
    cx, cy, _, _ = _lattice_centres(IMG, sp / k, pp / k, u * sp / k, v * pp / k)
    cx, cy = np.ravel(cx), np.ravel(cy); m = edge_margin_px()
    keep = (cx >= m) & (cx <= IMG - 1 - m) & (cy >= m) & (cy <= IMG - 1 - m)
    c = np.column_stack([cx[keep], cy[keep]]).astype(np.float64)
    return c[np.lexsort((c[:, 0], c[:, 1]))]


def binary_pattern(c, r_mm):
    from m2_pattern import _draw_dots
    return np.asarray(_draw_dots(IMG, c, r_mm / mmpp()), np.float32)


def render(binpat, p_min, bright):

    v = 1.0 if bright == 'A' else float(p_min) / float(binpat.sum())
    return (binpat * np.float32(v)).astype(np.float32), v


def pattern_checks(binpat, c):
    from scipy import ndimage
    _, n = ndimage.label(binpat > 0)
    frame = np.concatenate([binpat[0, :], binpat[-1, :], binpat[:, 0], binpat[:, -1]])
    marg = float(np.min(np.minimum.reduce([c[:, 0], c[:, 1], IMG - 1 - c[:, 0], IMG - 1 - c[:, 1]]))) if len(c) else None
    return dict(n_components=int(n), n_centres=int(len(c)), frame_zero=bool((frame == 0).all()), min_centre_margin_px=marg)


def expected_dots(mesh, c):

    from m3_virtual_camera import SENSOR_DEFAULTS, build_intrinsics, build_extrinsics, heightfield_intersect, project_to_camera
    cfg = dict(SENSOR_DEFAULTS)
    K = build_intrinsics(cfg); proj, cam = build_extrinsics(cfg)
    x_n = (c[:, 0] - K[0, 2]) / K[0, 0]; y_n = (c[:, 1] - K[1, 2]) / K[1, 1]
    d = (proj['R_inv'] @ np.stack([x_n, y_n, np.ones(len(c))], axis=1).T).T
    o = np.tile(proj['center'], (len(c), 1))
    hit, pts, _, _ = heightfield_intersect(o, d, mesh)
    hit = np.asarray(hit, bool)
    xy = np.full((len(c), 2), np.nan); uv = np.full((len(c), 2), np.nan)
    if hit.any():
        xy[hit] = pts[hit][:, :2]
        uvh, inf = project_to_camera(pts[hit], K, cam, IMG, IMG)
        uvh = np.where(np.asarray(inf)[:, None], uvh, np.nan); uv[hit] = uvh
    in_disc = hit & ((xy[:, 0] ** 2 + xy[:, 1] ** 2) <= DENT_R_MM ** 2)
    m = CAM_INNER_PX
    inner = hit & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= m) & (uv[:, 0] <= IMG - 1 - m) & (uv[:, 1] >= m) & (uv[:, 1] <= IMG - 1 - m)
    return dict(xy=xy, uv=uv, hit=hit, in_disc=in_disc, cam_inner=inner)


def spec(lattice, surface, phase, r_mm, bright, sigma, seed):
    return dict(v=VERSION, lattice=lattice, surface=surface, phase=int(phase), r_mm=float(r_mm), bright=bright, sigma=float(sigma), seed=int(seed))


def key(s):
    return digest({k: s[k] for k in KEY_FIELDS})


def path(stage, s):
    return OUT / stage / 'cases' / (key(s) + '.json')


def noise_seed(s):
    return int(NOISE_BASE + 10 * s['phase'] + s['seed'])


def design_pilot(lattice):
    out = [spec(lattice, surf, 0, r, b, 0.0, 0) for surf in SURFACES for r in (R_MIN, R_MAX) for b in BRIGHT]
    out += [spec(lattice, 'baseline', 0, r, b, 100.0, 0) for r in (R_MIN, R_MAX) for b in BRIGHT]
    return out


def design_main(lattice):
    return [spec(lattice, surf, ph, r, b, sg, k) for surf in SURFACES for ph in PHASES for r in RADII for b in BRIGHT
            for sg in SIGMAS for k in (SEEDS if sg > 0 else (0,))]


def run_chain(mesh, cfg, pat_render, pat_bin, sigma, seed_m3, spacing_mm):
    from m3_virtual_camera import synthesise_camera_image
    from m4_reconstruction import reconstruct
    from m5_evaluation import evaluate_all
    import preliminary_analysis as PA
    m3_cfg = {'batch_size': PA.BATCH_SIZE, 'noise_sigma': float(sigma), 'psf_sigma_px': PSF_PX, 'albedo': 1.0, 'albedo_exposure': 1.0,
              'specular_k': 0.0, 'mottle_amp': 0.0, 'outlier_frac': 0.0, 'mottle_seed': int(seed_m3)}
    img, _, _, m3_info = synthesise_camera_image(mesh_dict=mesh, pattern=pat_render, cfg=m3_cfg)
    pts, info = reconstruct(img, pat_bin, 'dots', method='epipolar', sensor_cfg={'measurement_volume_mm': VOLUME_MM})
    metrics = evaluate_all(points_3d=pts, mesh=mesh, camera_image=img, m1_config=cfg, m3_info=m3_info, uniq_window_px=PA.UNIQ_WINDOW_PX,
                           uniq_stride_px=PA.UNIQ_STRIDE_PX, uniq_n_samples=PA.UNIQ_N_SAMPLES, spacing_mm=spacing_mm, theta_deg=None, recon_info=info)
    return img, pts, info, metrics, m3_info


def _f(v):
    return float(v) if v is not None and np.isfinite(v) else None


# Render one dot-size and noise case and evaluate its reconstruction.
def work(s):
    t0 = time.monotonic()
    try:
        mesh, cfg, ref = L15.surface(s['surface'])
        sp, _pp = LATTICES[s['lattice']]
        c = centres(s['lattice'], s['phase'])
        binpat = binary_pattern(c, s['r_mm'])
        p_min = float(binary_pattern(c, R_MIN).sum())
        pat, v = render(binpat, p_min, s['bright'])
        pchk = pattern_checks(binpat, c)
        exp = expected_dots(mesh, c)
        t1 = time.monotonic()
        img, pts, info, m, m3_info = run_chain(mesh, cfg, pat, binpat, s['sigma'], noise_seed(s), sp)
        t_run = time.monotonic() - t1
        pts = np.asarray(pts, np.float32)
        amb = np.asarray(info.get('ambiguous_mask', np.zeros(len(pts), bool)), bool)
        if len(amb) != len(pts):
            amb = np.zeros(len(pts), bool)
        fin = np.isfinite(pts).all(axis=1); rel = fin & ~amb; n_rel = int(rel.sum())
        gt = truth_depth(pts, mesh).astype(np.float32)
        err = np.where(fin, pts[:, 2] - gt, np.nan).astype(np.float32)
        zmin = float(np.asarray(mesh['Z']).min())
        dref = np.full(len(pts), np.nan)
        if fin.any():
            dref[fin] = S.z_ref_at(ref, cfg, pts[fin, 0], pts[fin, 1]) - zmin
        err_ref = np.where(fin, pts[:, 2] - dref, np.nan).astype(np.float32)
        r_rel = err_ref[rel & np.isfinite(err_ref)]; r_all = err_ref[fin & np.isfinite(err_ref)]
        mc = clean(m)
        in_disc = fin & ((pts[:, 0].astype(np.float64) ** 2 + pts[:, 1].astype(np.float64) ** 2) <= DENT_R_MM ** 2)
        ok_ref = np.isfinite(err_ref) & (np.abs(np.nan_to_num(err_ref, nan=1e9)) <= OUTLIER_MM)
        n_exp_disc = int(exp['in_disc'].sum()); n_exp_inner = int(exp['cam_inner'].sum())
        d_err = mc.get('ind_depth_err_mm'); d_gt = mc.get('ind_depth_gt_mm'); d_pts = mc.get('ind_depth_rec_pts_mm')
        ex = dict(

            n_dots=int(len(c)), dot_value=float(v), p_min_px=p_min, pattern_peak=float(pat.max()), pattern_integral=float(pat.sum(dtype=np.float64)),
            pattern_lit_px=int((pat > 0).sum()), r_px=float(s['r_mm'] / mmpp()), d_mm=float(2 * s['r_mm']), pattern_checks=pchk,
            n_rays_cast=int(m3_info.get('n_rays_cast', 0) or 0), n_hits=int(m3_info.get('n_hits', 0) or 0),
            noise_seed_m3=noise_seed(s), noise_rng_seed=noise_seed(s) + 104729, phase_quantiles=list(quantiles(s['phase'])),

            image_sum=float(img.astype(np.float64).sum() / 255.0), image_max=int(img.max()), image_sat_frac=float((img >= 255).mean()),
            image_zero_frac=float((img == 0).mean()),

            otsu_threshold=_f(info.get('otsu_threshold')), n_detected=int(info.get('n_detected', 0) or 0), n_matched=int(info.get('n_matched', 0) or 0),
            n_rejected=int(info.get('n_rejected', 0) or 0), n_ambiguous=int(info.get('n_ambiguous', 0) or 0),
            n_area_rejected=int(info.get('n_area_rejected', 0) or 0), n_elong_rejected=int(info.get('n_elong_rejected', 0) or 0),
            n_filtered_noise=int(info.get('n_filtered_noise', 0) or 0), n_border_excluded=int(info.get('n_border_excluded', 0) or 0),
            blob_area_median_px=_f(info.get('blob_area_median_px')), n_finite=int(fin.sum()), n_rel=n_rel,
            A=_f(mc.get('A_ambiguity')), n_exp_cam_inner=n_exp_inner, n_exp_disc=n_exp_disc,

            n_rel_disc=int((in_disc & rel).sum()), n_rel_ok_disc=int((in_disc & rel & ok_ref).sum()), n_all_disc=int(in_disc.sum()),
            completeness=(int((in_disc & rel).sum()) / n_exp_disc) if n_exp_disc else None,
            completeness_ok=(int((in_disc & rel & ok_ref).sum()) / n_exp_disc) if n_exp_disc else None,

            zmin=zmin, ref_rmse_rel_mm=float(np.sqrt(np.mean(r_rel ** 2))) if len(r_rel) else None,
            ref_rmse_all_mm=float(np.sqrt(np.mean(r_all ** 2))) if len(r_all) else None,
            ref_bias_rel_mm=float(np.mean(r_rel)) if len(r_rel) else None, ref_p95_rel_mm=float(np.percentile(np.abs(r_rel), 95)) if len(r_rel) else None,
            gross_rate_rel=float((np.abs(r_rel) > OUTLIER_MM).mean()) if len(r_rel) else None, n_gross_rel=int((np.abs(r_rel) > OUTLIER_MM).sum()),

            dent_err_mm=d_err, dent_gt_mm=d_gt, dent_rel_pct=(100.0 * d_err / d_gt) if (d_err is not None and d_gt) else None,
            dent_pts_err_mm=(d_pts - d_gt) if (d_pts is not None and d_gt) else None,
            dent_pts_rel_pct=(100.0 * (d_pts - d_gt) / d_gt) if (d_pts is not None and d_gt) else None,
            dent_interp_coverage=mc.get('ind_coverage_frac'),

            centres_hash=hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest(),
            binpattern_hash=hashlib.sha256(np.ascontiguousarray(binpat).tobytes()).hexdigest(),
            pattern_hash=hashlib.sha256(np.ascontiguousarray(pat).tobytes()).hexdigest(), image_hash=hashlib.sha256(img.tobytes()).hexdigest(),
            seconds_pipeline=round(t_run, 3), peak_rss_mb=peak_rss_mb())
        ex.update({k + '_rel': val for k, val in occupancy(pts, rel).items()})
        st = status_of(mc, n_rel)
        return dict(spec=s, key=key(s), status=st, no_result=bool(ex['n_finite'] == 0), seconds=time.monotonic() - t0, metrics=mc, extras=ex,
                    arrays=dict(cam_img=img.astype(np.uint8), pattern_bin=(binpat > 0).astype(np.uint8), centres_px=c.astype(np.float32),
                                exp_xy=exp['xy'].astype(np.float32), exp_uv=exp['uv'].astype(np.float32), exp_in_disc=exp['in_disc'], exp_cam_inner=exp['cam_inner'],
                                pts=pts, amb=amb, gt_depth=gt, gt_ref=dref.astype(np.float32), err=err, err_ref=err_ref))
    except Exception:
        return dict(spec=s, key=key(s), status='error', no_result=True, seconds=time.monotonic() - t0, error=traceback.format_exc(), metrics=None)


def store(stage, r):
    s = r['spec']; p = path(stage, s); arrays = r.pop('arrays', None)
    p.parent.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        npz = p.with_suffix('.npz'); tmp = p.with_suffix('.npz.tmp')
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(npz); r['npz'] = npz.name
    save(p, r)
    return p


SIM_SOURCES = ['preliminary_analysis.py', 'scripts/pm013/pm013_lib.py', 'scripts/pm014/pm014_lib.py', 'scripts/pm014/pm014_surface.py',
               'scripts/pm015/pm015_lib.py', 'scripts/spotnoise/sn_lib.py', 'scripts/spotnoise/sn_run.py']


def sources():
    files = [*sorted((ROOT / 'src').glob('*.py')), *[ROOT / p for p in SIM_SOURCES], *sorted((ROOT / 'data' / 'raw').glob('*'))]
    return {str(Path(p).relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files if Path(p).is_file()}


def progress(**kw):
    p = OUT / 'progress.json'
    cur = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    cur.update(kw); cur['updated'] = now(); save(p, cur)
    return cur


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f'{now()} {msg}'
    with (OUT / 'run.log').open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


class _PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def peak_rss_mb():

    try:
        pmc = _PMC(); pmc.cb = ctypes.sizeof(_PMC)
        k32 = ctypes.windll.kernel32; k32.GetCurrentProcess.restype = ctypes.c_void_p
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.byref(pmc), pmc.cb)
        return round(pmc.PeakWorkingSetSize / 2 ** 20, 1) if ok else None
    except Exception:
        return None


class _MEMSTAT(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def ram_gb():

    try:
        st = _MEMSTAT(); st.dwLength = ctypes.sizeof(_MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return round(st.ullTotalPhys / 2 ** 30, 2), round(st.ullAvailPhys / 2 ** 30, 2)
    except Exception:
        return None, None
