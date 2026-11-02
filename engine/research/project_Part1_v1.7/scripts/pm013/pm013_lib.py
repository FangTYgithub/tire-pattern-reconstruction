# Run controlled density and ambiguity cases with point and surface diagnostics.

import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import sys, json, time, math, hashlib, platform, traceback
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
OUT = ROOT / 'outputs' / 'pm013'
CASES = OUT / 'cases'
SURF_DIR = OUT / 'surfaces'
FIG = ROOT.parents[1] / 'figures' / '05_sensitivity' / 'PM013_density_control'


VERSION = 1
SURFACES = {'baseline': {}, 'asym_half_0': {'indent_asym': 0.5}, 'curv_x_strong': {'curv_x_1_per_mm': -0.0025}}
BASE = dict(indent_depth_mm=5.0, indent_width_mm=20.0)
COMMON = dict(pattern_type='dots', feature_size_mm=1.0, renderer='psf', decoder='epipolar', measurement_volume_mm=20.0)
GRIDS = {'coarse': (2.0, 1.0), 'medium': (0.5, 0.25), 'fine': (0.25, 0.125), 'finest': (0.125, 0.0625)}
CONDITIONS = {'clean': dict(psf_sigma_px=1.0, noise_sigma=0.0),
              'noise': dict(psf_sigma_px=1.0, noise_sigma=30.0),
              'blur':  dict(psf_sigma_px=2.5, noise_sigma=0.0),
              'both':  dict(psf_sigma_px=2.5, noise_sigma=30.0),
              'stress': dict(psf_sigma_px=4.0, noise_sigma=60.0)}
MAIN_CONDS = ('clean', 'noise', 'blur', 'both')
SCAN_A = dict(spacing_perp_mm=3.0, spacing_mm=[4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0])
SCAN_B = dict(spacing_mm=8.0, spacing_perp_mm=[2.2, 2.5, 3.0, 4.0, 5.0, 6.0])
GRID_PILOT = {'sparse': (10.0, 3.0), 'boundary': (6.0, 3.0), 'rowmerge': (8.0, 2.2)}
GRID_PILOT_PHASES, GRID_PILOT_FINEST_PHASES = 4, 2
PHASES = 5
PHASE_SEED = 20261003
NOISE_SEED_BASE = 910000
LIGHT = {'A': dict(ref=(10.0, 3.0), targets=[(7.0, 3.0), (4.0, 3.0)]),
         'B': dict(ref=(8.0, 6.0), targets=[(8.0, 3.0), (8.0, 2.2)])}
LIGHT_CONDS = ('noise', 'both')
ROI_HALF_MM, ROI_CELL_MM = 40.0, 2.0
DENT_R_MM, DENT_CELL_MM = 15.0, 1.0
STATUS = dict(min_reliable=50, a_fail=0.10, a_degraded=0.01)


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items() if not str(k).startswith('_') and k != 'contour_mask'}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, Path):
        return str(v)
    return v


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(v):
    return hashlib.sha256(json.dumps(clean(v), sort_keys=True).encode()).hexdigest()


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def quantiles(phase):

    if int(phase) == 0:
        return 0.0, 0.0
    r = np.random.default_rng([PHASE_SEED, int(phase)]).random(2)
    return float(r[0]), float(r[1])


def spec(block, surface, grid, cond, sp, pp, phase, variant='A', albedo=1.0, **extra):
    s = dict(v=VERSION, block=block, surface=surface, grid=grid, cond=cond, spacing_mm=float(sp),
             spacing_perp_mm=float(pp), phase=int(phase), variant=variant, albedo=round(float(albedo), 6))
    if extra:
        s['extra'] = extra
    return s


KEY_FIELDS = ('v', 'block', 'surface', 'grid', 'cond', 'spacing_mm', 'spacing_perp_mm', 'phase', 'variant', 'albedo')


def key(s):
    return digest({k: s[k] for k in KEY_FIELDS})


def path(s):
    return CASES / s['block'] / (key(s) + '.json')


def pattern_sum(sp, pp, phase):

    from m2_pattern import generate_pattern
    u, v = quantiles(phase)
    pat, _ = generate_pattern('dots', spacing_mm=float(sp), feature_size_mm=COMMON['feature_size_mm'], depth_range_mm=20.0,
                              phase_offset_x_mm=u * sp, phase_offset_y_mm=v * pp, spacing_perp_mm=float(pp))
    pat = np.asarray(pat, np.float64)
    if pat.max() > 1.0:
        pat = pat / 255.0
    return float(pat.sum()), int((pat > 0).sum())


def design_grid_pilot():
    items = []
    for name, (sp, pp) in GRID_PILOT.items():
        for surf in SURFACES:
            for g in ('coarse', 'medium', 'fine'):
                for ph in range(GRID_PILOT_PHASES):
                    items.append(spec('grid', surf, g, 'clean', sp, pp, ph, config=name))
        for ph in range(GRID_PILOT_FINEST_PHASES):
            items.append(spec('grid', 'baseline', 'finest', 'clean', sp, pp, ph, config=name))
    return items


def design_main(grid):
    items = []
    for surf in SURFACES:
        for cond in MAIN_CONDS:
            for ph in range(PHASES):
                for sp in SCAN_A['spacing_mm']:
                    items.append(spec('main', surf, grid, cond, sp, SCAN_A['spacing_perp_mm'], ph))
                for pp in SCAN_B['spacing_perp_mm']:
                    items.append(spec('main', surf, grid, cond, SCAN_B['spacing_mm'], pp, ph))
    return items


def design_stress(grid):
    items = []
    for ph in range(PHASES):
        for sp in SCAN_A['spacing_mm']:
            items.append(spec('stress', 'baseline', grid, 'stress', sp, SCAN_A['spacing_perp_mm'], ph))
        for pp in SCAN_B['spacing_perp_mm']:
            items.append(spec('stress', 'baseline', grid, 'stress', SCAN_B['spacing_mm'], pp, ph))
    return items


def design_varc(grid):
    return [spec('varc', 'baseline', grid, 'clean', sp, SCAN_A['spacing_perp_mm'], ph, variant='C')
            for ph in range(PHASES) for sp in SCAN_A['spacing_mm']]


def design_light(grid):

    items = []
    for scan, d in LIGHT.items():
        for ph in range(PHASES):
            s_ref, n_ref = pattern_sum(*d['ref'], ph)
            for sp, pp in d['targets']:
                s_t, n_t = pattern_sum(sp, pp, ph)
                alb = min(1.0, s_ref / s_t)
                for cond in LIGHT_CONDS:
                    items.append(spec('light', 'baseline', grid, cond, sp, pp, ph, albedo=alb,
                                      ref=list(d['ref']), scan=scan, pattern_sum_ref=s_ref, pattern_sum=s_t,
                                      pattern_px_ref=n_ref, pattern_px=n_t))
    return items


def scan_of(s):
    a = math.isclose(s['spacing_perp_mm'], SCAN_A['spacing_perp_mm']) and any(math.isclose(s['spacing_mm'], x) for x in SCAN_A['spacing_mm'])
    b = math.isclose(s['spacing_mm'], SCAN_B['spacing_mm']) and any(math.isclose(s['spacing_perp_mm'], x) for x in SCAN_B['spacing_perp_mm'])
    return 'AB' if (a and b) else ('A' if a else ('B' if b else ''))


_CACHE = {}


def surface(variant, surf, grid):
    k = (variant, surf, grid)
    if k not in _CACHE:
        while len(_CACHE) >= 2:
            _CACHE.pop(next(iter(_CACHE)))
        from m1_tire_surface import build_surface
        gx, gy = GRIDS[grid]
        mesh, _, cfg = build_surface(variant, raw_data_dir=ROOT / 'data' / 'raw', grid_step_x_mm=gx, grid_step_y_mm=gy,
                                     **BASE, **SURFACES[surf])
        _CACHE[k] = (mesh, cfg)
    return _CACHE[k]


def save_surface(variant, surf, grid):

    p = SURF_DIR / f'{variant}_{surf}_{grid}.npz'
    if p.exists():
        return p
    mesh, cfg = surface(variant, surf, grid)
    SURF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, x=np.asarray(mesh['X'][0, :], np.float32), y=np.asarray(mesh['Y'][:, 0], np.float32),
                        Z=np.asarray(mesh['Z'], np.float32), Z_base=np.asarray(mesh.get('Z_base', mesh['Z']), np.float32),
                        indent_mask=np.asarray(mesh.get('indent_mask', np.zeros_like(mesh['Z'], bool)), bool))
    save(p.with_suffix('.json'), dict(variant=variant, surface=surf, grid=grid, steps=GRIDS[grid], shape=list(mesh['Z'].shape),
                                      config=clean(cfg)))
    return p


def truth_depth(pts, mesh):

    from scipy.interpolate import RegularGridInterpolator
    Z = mesh['Z'].astype(np.float64)
    y = mesh['Y'][:, 0].astype(np.float64)
    x = mesh['X'][0, :].astype(np.float64)
    D = Z - Z.min()
    if y[0] > y[-1]:
        y, D = y[::-1], D[::-1, :]
    if x[0] > x[-1]:
        x, D = x[::-1], D[:, ::-1]
    f = RegularGridInterpolator((y, x), D, method='linear', bounds_error=False, fill_value=np.nan)
    out = np.full(len(pts), np.nan)
    ok = np.isfinite(pts).all(axis=1)
    if ok.any():
        out[ok] = f(np.column_stack([pts[ok, 1], pts[ok, 0]]))
    return out


def occupancy(pts, mask):

    p = pts[mask & np.isfinite(pts).all(axis=1)]
    out = {}
    n = int(round(2 * ROI_HALF_MM / ROI_CELL_MM))
    if len(p):
        ix = np.floor((p[:, 0] + ROI_HALF_MM) / ROI_CELL_MM).astype(int)
        iy = np.floor((p[:, 1] + ROI_HALF_MM) / ROI_CELL_MM).astype(int)
        ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        out['occ_roi'] = len(np.unique(ix[ok] * n + iy[ok])) / float(n * n)
        out['n_roi'] = int(ok.sum())
    else:
        out['occ_roi'], out['n_roi'] = 0.0, 0
    m = int(round(2 * DENT_R_MM / DENT_CELL_MM))
    cx = (np.arange(m) + 0.5) * DENT_CELL_MM - DENT_R_MM
    CX, CY = np.meshgrid(cx, cx, indexing='ij')
    disc = (CX ** 2 + CY ** 2) <= DENT_R_MM ** 2
    if len(p):
        ix = np.floor((p[:, 0] + DENT_R_MM) / DENT_CELL_MM).astype(int)
        iy = np.floor((p[:, 1] + DENT_R_MM) / DENT_CELL_MM).astype(int)
        ok = (ix >= 0) & (ix < m) & (iy >= 0) & (iy < m)
        ok[ok] &= disc[ix[ok], iy[ok]]
        out['occ_dent'] = len(np.unique(ix[ok] * m + iy[ok])) / float(disc.sum())
        out['n_dent'] = int(ok.sum())
    else:
        out['occ_dent'], out['n_dent'] = 0.0, 0
    return out


def status_of(m, n_reliable):

    if m is None:
        return 'failed'
    A = m.get('A_ambiguity')
    if not m.get('data_valid') or n_reliable < STATUS['min_reliable'] or A is None or not np.isfinite(A) or A >= STATUS['a_fail']:
        return 'failed'
    return 'degraded' if A >= STATUS['a_degraded'] else 'ok'


def work(s):
    t0 = time.monotonic()
    try:
        from preliminary_analysis import _single_run
        from m4_reconstruction import reconstruct
        mesh, cfg = surface(s['variant'], s['surface'], s['grid'])
        t_surf = time.monotonic() - t0
        u, v = quantiles(s['phase'])
        kw = {**COMMON, **CONDITIONS[s['cond']], 'spacing_mm': s['spacing_mm'], 'spacing_perp_mm': s['spacing_perp_mm'],
              'phase_offset_x_mm': u * s['spacing_mm'], 'phase_offset_y_mm': v * s['spacing_perp_mm'],
              'phase_idx': int(s['phase']), 'pattern_seed': NOISE_SEED_BASE + int(s['phase']), 'albedo': float(s['albedo'])}
        t1 = time.monotonic()
        rec = _single_run(mesh, cfg, **kw)
        t_run = time.monotonic() - t1
        if rec is None:
            return dict(spec=s, key=key(s), status='m2_skip', seconds=time.monotonic() - t0, run_kwargs=kw, metrics=None)
        pts = np.asarray(rec['_pts_3d'], np.float32)
        img = np.asarray(rec['_cam_img'])
        pat = np.asarray(rec['_pattern'])

        pts2, info = reconstruct(img, pat, 'dots', sensor_cfg={'measurement_volume_mm': COMMON['measurement_volume_mm']}, method='epipolar')
        same = (pts2.shape == pts.shape) and np.array_equal(np.nan_to_num(pts2, nan=-9e9), np.nan_to_num(pts, nan=-9e9))
        amb = np.asarray(info.get('ambiguous_mask', np.zeros(len(pts), bool)), bool) if same else np.zeros(len(pts), bool)
        finite = np.isfinite(pts).all(axis=1)
        gt = truth_depth(pts, mesh).astype(np.float32)
        err = np.where(finite, pts[:, 2] - gt, np.nan).astype(np.float32)
        rel = finite & ~amb
        n_rel = int(rel.sum())
        m = clean(rec)
        occ_all = occupancy(pts, finite)
        occ_rel = occupancy(pts, rel)
        e_rel, e_all = err[rel & np.isfinite(err)], err[finite & np.isfinite(err)]
        extras = dict(
            n_finite=int(finite.sum()), n_reliable_mask=n_rel, n_ambiguous_mask=int((amb & finite).sum()), amb_mask_consistent=bool(same),
            rmse_rel_recomputed=float(np.sqrt(np.mean(e_rel ** 2))) if len(e_rel) else None,
            rmse_all_recomputed=float(np.sqrt(np.mean(e_all ** 2))) if len(e_all) else None,
            n_err_defined_all=int(len(e_all)), n_err_defined_rel=int(len(e_rel)),
            occ_roi_all=occ_all['occ_roi'], occ_roi_rel=occ_rel['occ_roi'], occ_dent_all=occ_all['occ_dent'], occ_dent_rel=occ_rel['occ_dent'],
            n_roi_rel=occ_rel['n_roi'], n_dent_rel=occ_rel['n_dent'], n_dent_all=occ_all['n_dent'],
            image_sum=float(img.astype(np.float64).sum() / 255.0), image_sat_frac=float((img >= 255).mean()), image_max=int(img.max()),
            image_lit_frac=float((img > 0).mean()), pattern_sum=float(np.asarray(pat, np.float64).sum() / (255.0 if pat.max() > 1 else 1.0)),
            pattern_px=int((pat > 0).sum()), image_hash=hashlib.sha256(img.tobytes()).hexdigest(),
            pattern_hash=hashlib.sha256(np.ascontiguousarray(pat).tobytes()).hexdigest(),
            phase_offset_mm=[u * s['spacing_mm'], v * s['spacing_perp_mm']], quantiles=[u, v], noise_seed=NOISE_SEED_BASE + int(s['phase']) + 104729,
            seconds_surface=round(t_surf, 3), seconds_pipeline=round(t_run, 3))
        st = status_of(m, n_rel)
        return dict(spec=s, key=key(s), status=st, seconds=time.monotonic() - t0, run_kwargs=kw, metrics=m, extras=extras,
                    arrays=dict(cam_img=img.astype(np.uint8), pts=pts, amb=amb, gt_depth=gt, err=err))
    except Exception:
        return dict(spec=s, key=key(s), status='error', seconds=time.monotonic() - t0, error=traceback.format_exc(), metrics=None)


def store(r):

    s = r['spec']
    p = path(s)
    arrays = r.pop('arrays', None)
    if arrays is not None:
        npz = p.with_suffix('.npz')
        tmp = npz.with_suffix('.npz.tmp')
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(npz)
        r['npz'] = npz.name
    save(p, r)
    return p


def sources():
    files = [*sorted((ROOT / 'src').glob('*.py')), ROOT / 'preliminary_analysis.py', *sorted((ROOT / 'scripts' / 'pm013').glob('*.py'))]
    raw = sorted((ROOT / 'data' / 'raw').glob('*'))
    return {str(Path(p).relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in files + raw if Path(p).is_file()}


def environment():
    import numpy, scipy
    try:
        import pandas, matplotlib
        pv, mv = pandas.__version__, matplotlib.__version__
    except Exception:
        pv = mv = None
    try:
        import subprocess
        head = subprocess.run(['git', '-c', 'safe.directory=*', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None
    return dict(python=platform.python_version(), platform=platform.platform(), cpu_count=os.cpu_count(), numpy=numpy.__version__,
                scipy=scipy.__version__, pandas=pv, matplotlib=mv, git_head=head, executable=sys.executable)


def progress(**kw):
    p = OUT / 'progress.json'
    cur = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    cur.update(kw)
    cur['updated'] = now()
    save(p, cur)
    return cur


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f'{now()} {msg}'
    with (OUT / 'run.log').open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    print(line, flush=True)
