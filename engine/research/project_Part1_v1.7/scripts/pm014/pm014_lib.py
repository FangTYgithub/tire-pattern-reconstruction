# Run spacing, surface and illumination comparisons on the reference surface model.

import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import sys, json, time, math, hashlib, traceback
from pathlib import Path
from datetime import datetime
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'scripts' / 'pm013'), str(ROOT), str(ROOT / 'src')]
import pm013_lib as P
import pm014_surface as S

OUT = ROOT / 'outputs' / 'pm014'
CASES = OUT / 'cases'
FIG = ROOT.parents[1] / 'figures' / '05_sensitivity' / 'PM014_surface_grid'

VERSION = 1
GRIDS = P.GRIDS
SURFACES = P.SURFACES
BASE = P.BASE
COMMON = P.COMMON
CONDITIONS = P.CONDITIONS
CONFIGS = {'sparse': (10.0, 3.0), 'boundary': (6.0, 3.0), 'dense': (5.0, 3.0), 'rowmerge': (8.0, 2.5)}
PHASES = 5
L_MAIN = 20.0
L_SENS = (10.0, 40.0)
SCAN_A = P.SCAN_A
SCAN_B = P.SCAN_B
KEY_FIELDS = ('v', 'model', 'L', 'surface', 'grid', 'cond', 'spacing_mm', 'spacing_perp_mm', 'phase', 'variant', 'albedo')
clean, save, digest, now, quantiles, occupancy, status_of, truth_depth = P.clean, P.save, P.digest, P.now, P.quantiles, P.occupancy, P.status_of, P.truth_depth


def spec(block, model, L, surface, grid, cond, sp, pp, phase, variant='A', albedo=1.0, **extra):
    s = dict(v=VERSION, block=block, model=model, L=(None if model == 'old' else float(L)), surface=surface, grid=grid, cond=cond, spacing_mm=float(sp),
             spacing_perp_mm=float(pp), phase=int(phase), variant=variant, albedo=float(albedo))
    if extra:
        s['extra'] = extra
    return s


def key(s):
    return digest({k: s[k] for k in KEY_FIELDS})


def path(s):
    return CASES / (key(s) + '.json')


def design_C_clean():
    items = []
    for model in ('old', 'fixed'):
        for g in GRIDS:
            for surf in SURFACES:
                for name, (sp, pp) in CONFIGS.items():
                    for ph in range(PHASES):
                        items.append(spec('C_clean', model, L_MAIN, surf, g, 'clean', sp, pp, ph, config=name))
    return items


def design_C_dist():
    items = []
    for model, grids in (('fixed', list(GRIDS)), ('old', ['coarse', 'fine'])):
        for g in grids:
            for surf in ('baseline', 'curv_x_strong'):
                for name in ('sparse', 'boundary'):
                    sp, pp = CONFIGS[name]
                    for cond in ('noise', 'blur', 'both'):
                        for ph in range(PHASES):
                            items.append(spec('C_dist', model, L_MAIN, surf, g, cond, sp, pp, ph, config=name))
    return items


def design_B_sens():
    items = []
    for L in L_SENS:
        for g in GRIDS:
            for name in ('sparse', 'boundary'):
                sp, pp = CONFIGS[name]
                for ph in range(PHASES):
                    items.append(spec('B_sens', 'fixed', L, 'baseline', g, 'clean', sp, pp, ph, config=name))
    return items


def design_D_scan():
    items = []
    for surf in ('baseline', 'curv_x_strong'):
        for cond in ('clean', 'both'):
            for ph in range(PHASES):
                for sp in SCAN_A['spacing_mm']:
                    items.append(spec('D_scan', 'fixed', L_MAIN, surf, 'fine', cond, sp, SCAN_A['spacing_perp_mm'], ph, scan='A'))
    for ph in range(PHASES):
        for pp in SCAN_B['spacing_perp_mm']:
            items.append(spec('D_scan', 'fixed', L_MAIN, 'baseline', 'fine', 'clean', SCAN_B['spacing_mm'], pp, ph, scan='B'))
    return items


DESIGNS = {'C_clean': design_C_clean, 'C_dist': design_C_dist, 'B_sens': design_B_sens, 'D_scan': design_D_scan}


_CACHE = {}


def surface_for(s):

    k = (s['model'], s['L'], s['surface'], s['grid'], s['variant'])
    if k not in _CACHE:
        while len(_CACHE) >= 2:
            _CACHE.pop(next(iter(_CACHE)))
        if s['model'] == 'old':
            mesh, cfg = P.surface(s['variant'], s['surface'], s['grid'])
            _CACHE[k] = (mesh, cfg, None)
        else:
            mesh, cfg, ref = S.build_fixed_surface(s['variant'], SURFACES[s['surface']], GRIDS[s['grid']], BASE, s['L'])
            _CACHE[k] = (mesh, cfg, ref)
    return _CACHE[k]


def save_surface(s):
    tag = f"{s['model']}_{'L%g' % s['L'] if s['L'] else 'prod'}_{s['variant']}_{s['surface']}_{s['grid']}"
    p = OUT / 'surfaces' / (tag + '.npz')
    if p.exists():
        return p
    mesh, cfg, ref = surface_for(s)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, x=np.asarray(mesh['X'][0, :], np.float32), y=np.asarray(mesh['Y'][:, 0], np.float32), Z=np.asarray(mesh['Z'], np.float32),
                        Z_base=np.asarray(mesh.get('Z_base', mesh['Z']), np.float32), indent_mask=np.asarray(mesh.get('indent_mask', np.zeros_like(mesh['Z'], bool)), bool))
    meta = dict(model=s['model'], L=s['L'], surface=s['surface'], grid=s['grid'], steps=GRIDS[s['grid']], shape=list(mesh['Z'].shape), zmin=float(np.asarray(mesh['Z']).min()),
                config=clean(cfg))
    if ref is not None:
        meta['reference'] = {k: v for k, v in ref.items() if k not in ('spline', 'y_ref', 'z_ref')}
        np.savez_compressed(p.with_name(tag + '_refnodes.npz'), y_ref=ref['y_ref'], z_ref=ref['z_ref'])
    save(p.with_suffix('.json'), meta)
    return p


# Run one spacing and illumination case and calculate reconstruction diagnostics.
def work(s):
    t0 = time.monotonic()
    try:
        from preliminary_analysis import _single_run
        from m4_reconstruction import reconstruct
        mesh, cfg, ref = surface_for(s)
        t_surf = time.monotonic() - t0
        u, v = quantiles(s['phase'])
        kw = {**COMMON, **CONDITIONS[s['cond']], 'spacing_mm': s['spacing_mm'], 'spacing_perp_mm': s['spacing_perp_mm'],
              'phase_offset_x_mm': u * s['spacing_mm'], 'phase_offset_y_mm': v * s['spacing_perp_mm'], 'phase_idx': int(s['phase']),
              'pattern_seed': P.NOISE_SEED_BASE + int(s['phase']), 'albedo': float(s['albedo'])}
        t1 = time.monotonic()
        rec = _single_run(mesh, cfg, **kw)
        t_run = time.monotonic() - t1
        if rec is None:
            return dict(spec=s, key=key(s), status='m2_skip', seconds=time.monotonic() - t0, run_kwargs=kw, metrics=None)
        pts = np.asarray(rec['_pts_3d'], np.float32); img = np.asarray(rec['_cam_img']); pat = np.asarray(rec['_pattern'])
        pts2, info = reconstruct(img, pat, 'dots', sensor_cfg={'measurement_volume_mm': COMMON['measurement_volume_mm']}, method='epipolar')
        same = (pts2.shape == pts.shape) and np.array_equal(np.nan_to_num(pts2, nan=-9e9), np.nan_to_num(pts, nan=-9e9))
        amb = np.asarray(info.get('ambiguous_mask', np.zeros(len(pts), bool)), bool) if same else np.zeros(len(pts), bool)
        finite = np.isfinite(pts).all(axis=1)
        gt = truth_depth(pts, mesh).astype(np.float32)
        err = np.where(finite, pts[:, 2] - gt, np.nan).astype(np.float32)
        rel = finite & ~amb
        n_rel = int(rel.sum())
        m = clean(rec)
        zmin = float(np.asarray(mesh['Z']).min())
        extras = dict(n_finite=int(finite.sum()), n_reliable_mask=n_rel, n_ambiguous_mask=int((amb & finite).sum()), amb_mask_consistent=bool(same), zmin=zmin,
                      seconds_surface=round(t_surf, 3), seconds_pipeline=round(t_run, 3), phase_offset_mm=[u * s['spacing_mm'], v * s['spacing_perp_mm']],
                      noise_seed=P.NOISE_SEED_BASE + int(s['phase']) + 104729, image_hash=hashlib.sha256(img.tobytes()).hexdigest(),
                      image_sum=float(img.astype(np.float64).sum() / 255.0), image_sat_frac=float((img >= 255).mean()))
        extras.update({k + '_all': v for k, v in occupancy(pts, finite).items()})
        extras.update({k + '_rel': v for k, v in occupancy(pts, rel).items()})
        e_rel, e_all = err[rel & np.isfinite(err)], err[finite & np.isfinite(err)]
        extras.update(rmse_rel_recomputed=float(np.sqrt(np.mean(e_rel ** 2))) if len(e_rel) else None,
                      rmse_all_recomputed=float(np.sqrt(np.mean(e_all ** 2))) if len(e_all) else None, n_err_defined_rel=int(len(e_rel)), n_err_defined_all=int(len(e_all)))
        arrays = dict(cam_img=img.astype(np.uint8), pts=pts, amb=amb, gt_depth=gt, err=err)
        if ref is not None:
            dref = np.full(len(pts), np.nan)
            dref[finite] = S.z_ref_at(ref, cfg, pts[finite, 0], pts[finite, 1]) - zmin
            err_ref = np.where(finite, pts[:, 2] - dref, np.nan).astype(np.float32)
            interp = np.where(finite, gt - dref, np.nan).astype(np.float32)
            r_rel, r_all = err_ref[rel & np.isfinite(err_ref)], err_ref[finite & np.isfinite(err_ref)]
            i_rel = interp[rel & np.isfinite(interp)]
            extras.update(ref_rmse_reliable_mm=float(np.sqrt(np.mean(r_rel ** 2))) if len(r_rel) else None,
                          ref_rmse_all_mm=float(np.sqrt(np.mean(r_all ** 2))) if len(r_all) else None,
                          ref_mae_reliable_mm=float(np.mean(np.abs(r_rel))) if len(r_rel) else None,
                          ref_p95_reliable_mm=float(np.percentile(np.abs(r_rel), 95)) if len(r_rel) else None,
                          ref_bias_reliable_mm=float(np.mean(r_rel)) if len(r_rel) else None,
                          ref_outlier_rate_reliable=float((np.abs(r_rel) > 1.0).mean()) if len(r_rel) else None,
                          interp_rms_at_points_mm=float(np.sqrt(np.mean(i_rel ** 2))) if len(i_rel) else None,
                          interp_max_at_points_mm=float(np.max(np.abs(i_rel))) if len(i_rel) else None,
                          ref_L_mm=ref['L_mm'], ref_nodes_sha256=ref['nodes_sha256'])
            arrays.update(gt_ref=dref.astype(np.float32), err_ref=err_ref)
        st = status_of(m, n_rel)
        return dict(spec=s, key=key(s), status=st, seconds=time.monotonic() - t0, run_kwargs=kw, metrics=m, extras=extras, arrays=arrays)
    except Exception:
        return dict(spec=s, key=key(s), status='error', seconds=time.monotonic() - t0, error=traceback.format_exc(), metrics=None)


def store(r):
    s = r['spec']
    p = path(s)
    arrays = r.pop('arrays', None)
    if arrays is not None:
        npz = p.with_suffix('.npz'); tmp = npz.with_suffix('.npz.tmp')
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(npz)
        r['npz'] = npz.name
    save(p, r)
    return p


def sources():
    files = [*sorted((ROOT / 'src').glob('*.py')), ROOT / 'preliminary_analysis.py', ROOT / 'scripts' / 'pm013' / 'pm013_lib.py', *sorted(HERE.glob('*.py'))]
    raw = sorted((ROOT / 'data' / 'raw').glob('*'))
    return {str(Path(p).relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files + raw if Path(p).is_file()}


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
