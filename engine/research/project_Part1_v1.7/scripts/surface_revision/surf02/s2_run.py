# Run surface-platform cases and numerical consistency checks.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_common as S

C0 = S.C0


def write_config(cases):
    src = {str(p.relative_to(S.ROOT)).replace('\\', '/'): C0.sha_file(p) for p in [S.ROOT / 'src' / f for f in ('m1_tire_surface.py', 'm2_pattern.py', 'm3_virtual_camera.py', 'm4_reconstruction.py', 'm4_lines.py', 'surface_eval_r3.py', 'm5_evaluation.py')]
           + [S.HERE / 's2_common.py', S.HERE / 's2_run.py', S.ROOT / 'scripts' / 'surface_revision' / 'surf01' / 's1_model.py', S.ROOT / 'scripts' / 'pm014' / 'pm014_surface.py', S.P4_FILE]}
    meshes = {k: dict(sha16=S.mesh_sha(S.build_mesh(k)), shape=list(S.build_mesh(k)['Z'].shape), z_range_mm=[float(S.build_mesh(k)['Z'].min()), float(S.build_mesh(k)['Z'].max())],
                      roi=S.clean({kk: v for kk, v in S.roi_domain(S.build_mesh(k)).items() if kk in ('n_nodes', 'area_total_mm2', 'hash', 'z_range', 'window_cov_roi', 'window_cov_fov', 'z_range_fov')})) for k in S.SURFACE_ORDER}
    cfg = dict(version=S.VERSION, created_local=C0.now_local(), frame=dict(mapping='x = -x_p, y = y_p, z = c - H', c_ref_mm=S.C_REF, c_ref_definition='H_B,base(0, 0): flat-ground profile B + arc at the axis, read once, shared by all surfaces; no per-surface re-zeroing',
                                                                             kappa_H_1_per_mm=S.M.KAPPA_H, gauss=S.M.GAUSS, m3_mode='mesh["z_alignment"] = "absolute" (new explicit mode; default "min" unchanged)'),
               device=S.geometry_record(), window=dict(z0_mm=S.Z0_MM, z1_mm=S.Z1_MM, measurement_volume_mm=S.VOL_MM, line_search_margin_mm=S.MARGIN_MM, dot_cfg_keys=dict(search_z_min_mm=S.Z0_MM, measurement_volume_mm=S.VOL_MM),
                                                       line_cfg_keys=dict(search_z_min_mm=S.Z0_MM, measurement_volume_mm=S.VOL_MM, search_margin_mm=S.MARGIN_MM), note='one window for every surface and both families; coverage reported, never adapted per case'),
               grid=dict(scientific=list(S.GRID_SCI), check_only=list(S.GRID_CHECK), x_half_mm=S.X_HALF_MM, y_range='profile range of the variant'), roi=dict(r_mm=S.ROI_R_MM, gross_mm=S.GROSS_MM, fov_half_mm=S.FOV_HALF_MM),
               surfaces={k: dict(S.SURFACES[k], **meshes[k]) for k in S.SURFACE_ORDER}, dots=S.clean(dict(S.DOT, positions_uv=[S.SLL.positions()[i] for i in S.DOT['positions']], detector='r5_detect candidates + D0 window', correspondence='epipolar (M4), not anchored')),
               lines=S.clean(dict(LINE=S.LINE, P4_C3_cal=S.P4, m4_cfg=S.line_cfg())), imaging=dict(S.m3_cfg(0.0, 0), seed_rule='SEED_BASE + 1000 surface + 100 family + 10 phase + (sigma > 0)', seed_base=S.SEED_BASE),
               cases=cases, source_sha256=src)
    (S.OUT / 'config' / 'surf02_config.json').write_text(json.dumps(S.clean(cfg), ensure_ascii=False, indent=1), encoding='utf-8')
    return cfg


def run_batch(cases, log):
    rows = []
    for i, c in enumerate(cases):
        t = time.monotonic(); rec = S.run_case(c); rows.append(S.flat_row(c, rec))
        log(f"[{i + 1}/{len(cases)}] {c['case_id']} {round(time.monotonic() - t, 2)} s  valid {rec['evaluation']['n_valid']}  roi-rmse {rec['evaluation']['pts_roi']['rmse']}  wrong {rec['identity']['wrong']}")
    return rows


def check_triangulation():

    import m4_reconstruction as M4
    g = C0.geom(); out = {}
    for k in S.SURFACE_ORDER:
        mesh = S.build_mesh(k); c, binpat, pat, _ = S.dot_pattern(0)
        hit, P = C0.cast(c, g['proj'], mesh); uv, inside = C0.project(P, g['cam']); ok = hit & inside
        tri = M4.triangulate(uv[ok].astype(np.float64), c[ok].astype(np.float64), S.dot_cfg())
        d = np.linalg.norm(tri - P[ok], axis=1)
        tri32 = M4.triangulate(uv[ok].astype(np.float32), c[ok].astype(np.float32), S.dot_cfg()); d32 = np.linalg.norm(tri32 - P[ok], axis=1)
        out[k] = dict(n=int(ok.sum()), max_abs_mm=float(d.max()), rms_mm=float(np.sqrt(np.mean(d ** 2))), max_abs_mm_float32_features=float(d32.max()), z_hit_range_mm=[float(P[ok, 2].min()), float(P[ok, 2].max())],
                      pass_1e_5=bool(d.max() < 1e-5))
    return dict(passed=all(v['pass_1e_5'] for v in out.values()), tolerance_mm=1e-5, note='M4.triangulate returns float32 points: resolution ~3e-6 mm at |x, y| ~ 46 mm, so agreement is at float32 output precision', per_surface=out)


def check_hits_vs_analytic():

    g = C0.geom(); out = {}
    for k in S.SURFACE_ORDER:
        mesh = S.build_mesh(k); c, _, _, _ = S.dot_pattern(1)
        hit, P = C0.cast(c, g['proj'], mesh)
        za = S.z_true(k, P[hit, 0], P[hit, 1]); zb = S.truth_interp(mesh)(np.column_stack([P[hit, 1], P[hit, 0]]))

        uu, vv = np.meshgrid(np.arange(200, 312, 8), np.arange(200, 312, 8)); uv = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float64)
        h2, P2 = C0.cast(uv, g['cam'], mesh); za2 = S.z_true(k, P2[h2, 0], P2[h2, 1])
        out[k] = dict(n_proj_rays=int(hit.sum()), max_abs_vs_analytic_mm=float(np.abs(P[hit, 2] - za).max()), max_abs_vs_grid_bilinear_mm=float(np.abs(P[hit, 2] - zb).max()),
                      n_cam_rays=int(h2.sum()), cam_max_abs_vs_analytic_mm=float(np.abs(P2[h2, 2] - za2).max()), pass_1e_3=bool(np.abs(P[hit, 2] - za).max() < 1e-3 and np.abs(P2[h2, 2] - za2).max() < 1e-3))
    return dict(passed=all(v['pass_1e_3'] for v in out.values()), tolerance_mm=1e-3, note='a-priori bilinear representation error of the 0.25 x 0.125 mm grid: curvature * dx^2 / 8; Gaussian d / sigma^2 = 5 / 8.49^2 = 0.069 1/mm -> 0.069 * 0.0625 / 8 = 5.4e-4 mm (observed 5.6e-4); the grid truth itself is reproduced to ~1e-10', per_surface=out)


def check_old_path_regression(log, prev=None):
    if prev is not None:
        return dict(prev, reused_from_previous_run=True)

    import glob
    from m3_virtual_camera import synthesise_camera_image
    out = {}

    p = sorted(glob.glob(str(S.ROOT / 'outputs' / 'line01r' / 'a' / 'cases' / '*.json')))[0]
    j = json.loads(Path(p).read_text(encoding='utf-8')); sp = j['spec']
    mesh, cfg, ref_fn, desc = C0.surface(sp['surface']); assert 'z_alignment' not in mesh
    pat_bin, meta = S.B.pattern_for(dict(s_mm=sp['s_mm'], w_mm=sp['w_mm'], phi_deg=sp['phi_deg'], phase_frac=sp['phase_frac'])); pat = (pat_bin * np.float32(sp['peak'])).astype(np.float32)
    img, _, _, m3 = synthesise_camera_image(mesh, pat, C0.m3_cfg(float(sp['sigma']), int(sp['noise_seed'])))
    old = np.load(S.ROOT / j['image']['source'])['image']
    out['line01_plane_sigma40'] = dict(case=Path(p).stem, spec=sp, identical=bool(np.array_equal(img, old)), n_diff_px=int((img != old).sum()), max_abs_diff=int(np.abs(img.astype(int) - old.astype(int)).max()), z_alignment=m3.get('z_alignment'))
    log(f"regression (a) {out['line01_plane_sigma40']['identical']}")

    q = sorted(glob.glob(str(S.ROOT / 'outputs' / 'spacing_layout' / 'cases' / '*.json')))[0]
    jj = json.loads(Path(q).read_text(encoding='utf-8')); s = jj['spec']; v = float(jj['extras']['dot_value'])
    c = S.SLL.centres(s['lattice'], s['off_u'], s['off_v']); binpat = S.SL.binary_pattern(c, s['r_mm']); pat = (binpat * np.float32(v)).astype(np.float32)
    mesh2, cfg2, _ = S.SLL.surface(s['surface'], s['fwhm']); assert 'z_alignment' not in mesh2
    seed = int(S.SLL.noise_seed(s))
    img2, _, _, m32 = synthesise_camera_image(mesh2, pat, S.m3_cfg(float(s['sigma']), seed))
    z = np.load(q[:-5] + '.npz'); old2 = z['cam_img']
    out['spacing_layout_R8x4_curv_sigma40'] = dict(case=Path(q).stem, spec=s, dot_value=v, seed=seed, identical=bool(np.array_equal(img2, old2)), n_diff_px=int((img2 != old2).sum()),
                                                   max_abs_diff=int(np.abs(img2.astype(int) - old2.astype(int)).max()), pattern_identical=bool(np.array_equal(binpat > 0, z['pattern_bin'] > 0)), z_alignment=m32.get('z_alignment'))
    log(f"regression (b) {out['spacing_layout_R8x4_curv_sigma40']['identical']}")
    out['n_images_rendered'] = 2
    out['passed'] = bool(out['line01_plane_sigma40']['identical'] and out['spacing_layout_R8x4_curv_sigma40']['identical'])
    return out


def check_metric_recompute(case_ids):

    from scipy.interpolate import LinearNDInterpolator
    out = {}; worst = 0.0
    for cid in case_ids:
        rec, z = S.load_case(cid); c = rec['case']; mesh = S.build_mesh(c['surface'], tuple(c['grid'])); dom = S.roi_domain(mesh)
        pts = np.asarray(z['pts'], np.float64); acc = np.asarray(z['acc'], bool)
        fin = np.isfinite(pts).all(axis=1); zr = np.full(len(pts), np.nan); f = S.truth_interp(mesh)
        zr[fin] = f(np.column_stack([pts[fin, 1], pts[fin, 0]]))
        valid = acc & fin & np.isfinite(zr) & (pts[:, 2] >= S.Z0_MM - S.MARGIN_MM) & (pts[:, 2] <= S.Z1_MM + S.MARGIN_MM)
        roi = valid & (np.hypot(pts[:, 0], pts[:, 1]) <= S.ROI_R_MM); e = pts[roi, 2] - zr[roi]
        P = pts[valid]

        key = np.round(P[:, :2] / 1e-9).astype(np.int64); _, first = np.unique(key, axis=0, return_index=True); Pu = P[np.sort(first)]
        zi = LinearNDInterpolator(Pu[:, :2], Pu[:, 2])(np.column_stack([dom['x'], dom['y']])) if len(Pu) >= 3 else np.full(dom['n_nodes'], np.nan)
        en = zi - dom['z_ref']; cov = np.isfinite(en)
        prim = rec['evaluation']
        d = dict(pts_rmse=abs(float(np.sqrt(np.mean(e ** 2))) - prim['pts_roi']['rmse']) if len(e) else None, pts_n=int(len(e)) - prim['pts_roi']['n'],
                 nodes_cov=abs(float(cov.mean()) - prim['nodes_roi']['coverage']), nodes_rmse=(abs(float(np.sqrt(np.mean(en[cov] ** 2))) - prim['nodes_roi']['rmse']) if cov.any() and prim['nodes_roi']['rmse'] is not None else None),
                 nodes_mae=(abs(float(np.mean(np.abs(en[cov]))) - prim['nodes_roi']['mae']) if cov.any() and prim['nodes_roi']['mae'] is not None else None),
                 nodes_p95=(abs(float(np.percentile(np.abs(en[cov]), 95)) - prim['nodes_roi']['p95_abs']) if cov.any() and prim['nodes_roi']['p95_abs'] is not None else None))
        if prim['amplitude']['applicable'] and cov.any():
            dip = dom['z_base'] - zi; core = cov & (dom['r'] <= 10.0); cen = cov & (dom['r'] <= 1.5)
            d['amp_peak'] = abs(float(np.nanmax(dip[core])) - prim['amplitude']['rec_peak_mm']); d['amp_centre'] = abs(float(np.nanmean(dip[cen])) - prim['amplitude']['rec_centre_mm'])
        mx = max([abs(v) for v in d.values() if v is not None] + [0.0]); worst = max(worst, mx)
        out[cid] = dict(d, max_abs_diff=mx)
    return dict(passed=bool(worst < 1e-6), tolerance=1e-6, worst_abs_diff=worst, note='node interpolation: both are piecewise-linear on a Delaunay triangulation of the same points (dedup tolerance differs: 1e-9 here vs the evaluator rule); differences at 1e-6 or below = same numbers', per_case=out)


def check_grid(log):

    extra = S.cases(grid=S.GRID_CHECK, surfaces=('A_curv_gauss', 'C_curv'), sigmas=(0.0,), phases=(0,), tag='grid05')
    if all((S.OUT / 'cases' / f"{c['case_id']}.json").exists() for c in extra):
        rows = [S.flat_row(c, json.loads((S.OUT / 'cases' / f"{c['case_id']}.json").read_text(encoding='utf-8'))) for c in extra]; log('grid check: stored extra cases reused (no render)')
    else:
        rows = run_batch(extra, log)
    out = dict(n_extra_images=len(extra), scientific_grid=list(S.GRID_SCI), check_grid=list(S.GRID_CHECK), per_case={}, geometry={})
    for c, r in zip(extra, rows):
        main_id = c['case_id'].replace('_grid05', ''); rm = json.loads((S.OUT / 'cases' / f'{main_id}.json').read_text(encoding='utf-8'))['evaluation']
        rc = json.loads((S.OUT / 'cases' / f"{c['case_id']}.json").read_text(encoding='utf-8'))['evaluation']
        out['per_case'][main_id] = dict(pts_roi_rmse_sci=rm['pts_roi']['rmse'], pts_roi_rmse_check=rc['pts_roi']['rmse'], nodes_rmse_sci=rm['nodes_roi']['rmse'], nodes_rmse_check=rc['nodes_roi']['rmse'],
                                        nodes_cov_sci=rm['nodes_roi']['coverage'], nodes_cov_check=rc['nodes_roi']['coverage'], n_valid_sci=rm['n_valid'], n_valid_check=rc['n_valid'],
                                        amp_peak_sci=rm['amplitude'].get('rec_peak_mm'), amp_peak_check=rc['amplitude'].get('rec_peak_mm'),
                                        abs_diff_pts_rmse_mm=(abs(rm['pts_roi']['rmse'] - rc['pts_roi']['rmse']) if rm['pts_roi']['rmse'] is not None and rc['pts_roi']['rmse'] is not None else None))
    for k in ('A_curv_gauss', 'C_curv'):
        ms = S.build_mesh(k); mc = S.build_mesh(k, S.GRID_CHECK); dom = S.roi_domain(ms)
        zc = S.truth_interp(mc)(np.column_stack([dom['y'], dom['x']])); za = S.z_true(k, dom['x'], dom['y'])
        out['geometry'][k] = dict(max_abs_check_grid_vs_analytic_on_roi_mm=float(np.nanmax(np.abs(zc - za))), max_abs_sci_grid_vs_analytic_on_roi_mm=float(np.nanmax(np.abs(dom['z_ref'] - za))))
    diffs = [v['abs_diff_pts_rmse_mm'] for v in out['per_case'].values() if v['abs_diff_pts_rmse_mm'] is not None]
    out['max_abs_diff_pts_rmse_mm'] = max(diffs) if diffs else None
    out['passed'] = bool(diffs and max(diffs) < 0.01)
    out['note'] = 'pass = point-RMSE difference between grids < 0.01 mm (i.e. the scientific grid is converged at the reported precision); the 0.5 x 0.25 grid is for plotting only'
    return out


def check_effective(case_ids):

    recs = [json.loads((S.OUT / 'cases' / f'{cid}.json').read_text(encoding='utf-8')) for cid in case_ids]
    win = {(r['effective']['window_z0_mm'], r['effective']['window_z1_mm']) for r in recs}; modes = {r['effective']['m3']['z_alignment'] for r in recs}
    cref = {r['mesh']['c_ref_mm'] for r in recs}; wd = {(r['effective']['working_dist_mm'], r['effective']['baseline_mm']) for r in recs}
    cov = {r['case']['surface']: (r['domain']['window_cov_roi'], r['domain']['window_cov_fov']) for r in recs}
    return dict(passed=bool(win == {(S.Z0_MM, S.Z1_MM)} and modes == {'absolute'} and len(cref) == 1 and wd == {(150.0, 50.0)}), windows=sorted(win), z_alignment_modes=sorted(modes), c_ref_values=sorted(cref),
                device_wd_baseline=sorted(wd), window_coverage_by_surface=cov, n_cases=len(recs))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checks', action='store_true'); a = ap.parse_args()
    logf = S.OUT / 'run.log'

    def log(msg):
        line = f'{C0.now_local()} {msg}'; print(line, flush=True)
        with open(logf, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')

    t0 = time.monotonic(); main_cases = S.cases()
    if not a.checks:
        log(f'SURF-02 batch start: {len(main_cases)} images, c_ref {S.C_REF}, window [{S.Z0_MM}, {S.Z1_MM}]')
        write_config(main_cases)
        rows = run_batch(main_cases, log)
        pd.DataFrame(rows).to_csv(S.OUT / 'metrics' / 'surf02_metrics.csv', index=False, encoding='utf-8-sig')
        log(f'batch done in {round(time.monotonic() - t0, 1)} s')
    ids = [c['case_id'] for c in main_cases]
    t1 = time.monotonic(); checks = dict(version=S.VERSION, started_local=C0.now_local())
    checks['effective_parameters'] = check_effective(ids); log(f"effective ok {checks['effective_parameters']['passed']}")
    checks['c1_triangulation_consistency'] = check_triangulation(); log(f"c1 {checks['c1_triangulation_consistency']['passed']}")
    checks['c2_ray_hits_vs_analytic'] = check_hits_vs_analytic(); log(f"c2 {checks['c2_ray_hits_vs_analytic']['passed']}")
    prevf = S.OUT / 'checks' / 'surf02_checks.json'; prev = json.loads(prevf.read_text(encoding='utf-8')).get('c3_old_default_path_regression') if (a.checks and prevf.exists()) else None
    checks['c3_old_default_path_regression'] = check_old_path_regression(log, prev); log(f"c3 {checks['c3_old_default_path_regression']['passed']}")
    checks['c4_metric_recompute'] = check_metric_recompute(ids); log(f"c4 {checks['c4_metric_recompute']['passed']} worst {checks['c4_metric_recompute']['worst_abs_diff']}")
    checks['c5_grid_resolution'] = check_grid(log); log(f"c5 {checks['c5_grid_resolution']['passed']}")
    checks['image_budget'] = dict(main=len(main_cases), regression=checks['c3_old_default_path_regression']['n_images_rendered'], grid=checks['c5_grid_resolution']['n_extra_images'],
                                  development_trials=2, note='2 development renders (one dot, one line case) were made while debugging and deleted; they repeated main-case configurations')
    checks['all_passed'] = all(checks[k]['passed'] for k in ('effective_parameters', 'c1_triangulation_consistency', 'c2_ray_hits_vs_analytic', 'c3_old_default_path_regression', 'c4_metric_recompute', 'c5_grid_resolution'))
    checks['seconds_checks'] = round(time.monotonic() - t1, 1); checks['finished_local'] = C0.now_local()
    (S.OUT / 'checks' / 'surf02_checks.json').write_text(json.dumps(S.clean(checks), ensure_ascii=False, indent=1), encoding='utf-8')

    rows = []
    for p in sorted((S.OUT / 'cases').glob('*.json')):
        rec = json.loads(p.read_text(encoding='utf-8')); rows.append(S.flat_row(rec['case'], rec))
    df = pd.DataFrame(rows); df = df.sort_values(['tag', 'surface', 'family', 'phase_index', 'sigma']); df.to_csv(S.OUT / 'metrics' / 'surf02_metrics.csv', index=False, encoding='utf-8-sig')
    log(f"all passed: {checks['all_passed']}; total {round(time.monotonic() - t0, 1)} s; csv rows {len(df)}")


if __name__ == '__main__':
    main()
