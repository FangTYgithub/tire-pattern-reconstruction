# Build candidate patterns, apply dropout and evaluate reconstruction performance.

import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import sys, json, time, math, hashlib, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'scripts' / 'pm014'), str(ROOT / 'scripts' / 'pm013'), str(ROOT), str(ROOT / 'src')]
import pm013_lib as P
import pm014_lib as L14
import pm014_surface as S

OUT = ROOT / 'outputs' / 'pm015'
CASES = OUT / 'cases'
FIG = ROOT.parents[1] / 'figures' / '05_sensitivity' / 'PM015_pattern_screening'
VERSION = 1
GRID = 'fine'
L_MM = 20.0
SURFACES = {'baseline': {}, 'curv_x_strong': {'curv_x_1_per_mm': -0.0025}}
CONDITIONS = {'clean': dict(psf_sigma_px=1.0, noise_sigma=0.0), 'both': dict(psf_sigma_px=2.5, noise_sigma=30.0)}
PHASES = 3
PATTERN_SEED_BASE = 20261500
NOISE_SEED_BASE = P.NOISE_SEED_BASE
R_MM = 1.0
MM_PER_PX = 0.18
DENT_R_MM = 15.0

FAMILIES = {
    'REG':  dict(kind='dots', label='regular lattice (reference)', configs={'r7x3': dict(spacing_mm=7.0, spacing_perp_mm=3.0), 'r8x4': dict(spacing_mm=8.0, spacing_perp_mm=4.0)}, realisations=1, catalogue='validated (blocks G-J)'),
    'RAND': dict(kind='random_dots', label='constrained random dots (min sep 0.6 s)', configs={'s6': dict(spacing_mm=6.0, min_sep_frac=0.6), 's7': dict(spacing_mm=7.0, min_sep_frac=0.6), 's8': dict(spacing_mm=8.0, min_sep_frac=0.6)}, realisations=4, catalogue='family in P-1 validation catalogue (J-3); new configurations'),
    'EPI':  dict(kind='epi_random_dots', label='epipolar-aware random dots', configs={'s5': dict(spacing_mm=5.0, min_sep_frac=0.9), 's6': dict(spacing_mm=6.0, min_sep_frac=0.9), 's7': dict(spacing_mm=7.0, min_sep_frac=0.9)}, realisations=4, catalogue='family in catalogue (J-8); new configurations'),
    'GRAD': dict(kind='graded_dots', label='locally graded density (centre -> 2x edge)', configs={'c5': dict(spacing_mm=5.0, min_sep_frac=0.6), 'c6': dict(spacing_mm=6.0, min_sep_frac=0.6), 'c7': dict(spacing_mm=7.0, min_sep_frac=0.6)}, realisations=4, catalogue='family in catalogue (J-3); new configurations'),
    'JIT':  dict(kind='dots', label='position-perturbed lattice 7 x 3', configs={'j10': dict(spacing_mm=7.0, spacing_perp_mm=3.0, jitter_frac=0.10), 'j20': dict(spacing_mm=7.0, spacing_perp_mm=3.0, jitter_frac=0.20), 'j30': dict(spacing_mm=7.0, spacing_perp_mm=3.0, jitter_frac=0.30)}, realisations=4, catalogue='family in catalogue (J-3 jitter); new configurations'),
    'DROP': dict(kind='dots_dropout', label='lattice 7 x 3 with randomly missing dots (NEW generator, this directory)', configs={'d10': dict(spacing_mm=7.0, spacing_perp_mm=3.0, dropout=0.10), 'd25': dict(spacing_mm=7.0, spacing_perp_mm=3.0, dropout=0.25), 'd40': dict(spacing_mm=7.0, spacing_perp_mm=3.0, dropout=0.40)}, realisations=4, catalogue='NEW family: not in any validation catalogue -> predictions marked out-of-domain (exploratory)'),
}
DENT_RULE_FAMILIES = ('REG',)

TH = dict(A_pass=0.01, A_reject=0.10, blur_review_px=1.4, noise_review_sigma=20.0, dent_pct=10.0, density_baseline_n_visible=200,
          eng=dict(A=0.01, n_rel_min=50, occ_dent_min=0.02, dent_abs_max=10.0, outlier_max=0.01, ref_rmse_max={'clean': 0.06, 'both': 0.15}))
KEY_FIELDS = ('v', 'family', 'config', 'realisation', 'surface', 'cond', 'phase', 'albedo')
clean, save, digest, now, quantiles, occupancy, status_of, truth_depth = P.clean, P.save, P.digest, P.now, P.quantiles, P.occupancy, P.status_of, P.truth_depth


def pattern_seed(family, config, k):
    return PATTERN_SEED_BASE + 1000 * list(FAMILIES).index(family) + 100 * list(FAMILIES[family]['configs']).index(config) + int(k)


def make_pattern(family, config, k, phase):

    from m2_pattern import generate_pattern
    f = FAMILIES[family]; a = dict(f['configs'][config]); seed = pattern_seed(family, config, k)
    sp = a['spacing_mm']; pp = a.get('spacing_perp_mm', sp)
    u, v = quantiles(phase); ox, oy = u * sp, v * pp
    kind = f['kind']
    if kind == 'dots_dropout':
        pat, meta = generate_pattern('dots', spacing_mm=sp, feature_size_mm=R_MM, depth_range_mm=20.0, phase_offset_x_mm=ox, phase_offset_y_mm=oy, spacing_perp_mm=pp)
        pat, drop_meta = dropout(np.asarray(pat), a['dropout'], seed)
        meta = {**meta, **drop_meta, 'generator': 'pm015 dropout post-processor on M2 dots'}
        return pat.astype(np.float32), meta, 'dots'
    kw = dict(spacing_mm=sp, feature_size_mm=R_MM, depth_range_mm=20.0, phase_offset_x_mm=ox, phase_offset_y_mm=oy)
    if kind == 'dots':
        kw.update(spacing_perp_mm=pp, jitter_frac=a.get('jitter_frac', 0.0), seed=seed)
    else:
        kw.update(seed=seed, min_sep_frac=a.get('min_sep_frac', 0.6))
        if kind == 'epi_random_dots':
            kw.update(epi_along_mm=6.5, epi_across_mm=1.08)
    pat, meta = generate_pattern(kind, **kw)
    return np.asarray(pat, np.float32), meta, kind


def dropout(pat, frac, seed):

    from scipy import ndimage
    lab, n = ndimage.label(pat > 0)
    rng = np.random.default_rng(int(seed))
    drop = rng.random(n) < frac
    keep = np.ones(n + 1, bool); keep[1:] = ~drop
    out = np.where(keep[lab], pat, 0).astype(pat.dtype)
    return out, dict(n_dots_before=int(n), n_dropped=int(drop.sum()), dropout_frac=float(frac), dropout_seed=int(seed))


def features(pat, ptype):

    from pattern_evaluator import evaluate_pattern, features_and_classes
    from m4_reconstruction import _resolve_sensor_cfg
    from scipy.spatial import cKDTree
    cfg = _resolve_sensor_cfg({'measurement_volume_mm': 20.0})
    t0 = time.monotonic()
    ev = evaluate_pattern(pat, ptype, sensor_cfg=cfg, depth_range_mm=20.0, volumes=(20.0,), drifts_px=(5.8,), fast=True)
    t_p1 = time.monotonic() - t0
    pts, ids, classes, areas = features_and_classes(pat, ptype, cfg)
    xy = (np.asarray(pts, float) - (pat.shape[1] - 1) / 2.0) * MM_PER_PX
    d, _ = cKDTree(xy).query(xy, k=2); nn = d[:, 1]

    g = np.arange(-DENT_R_MM, DENT_R_MM + 0.5, 0.5); GX, GY = np.meshgrid(g, g); disc = (GX ** 2 + GY ** 2) <= DENT_R_MM ** 2
    hole, _ = cKDTree(xy).query(np.column_stack([GX[disc], GY[disc]]))
    n_disc = int(((xy[:, 0] ** 2 + xy[:, 1] ** 2) <= DENT_R_MM ** 2).sum())
    return dict(n_features=int(len(xy)), n_visible=int(ev['n_visible']), C_D=float(ev['C_D']), A_design=float(ev['A_design']), blur_px=float(ev['blur_px']),
                gap_mm=float(ev.get('gap_mm', float('nan'))), noise_sigma_max=float(ev.get('noise_sigma_max', float('nan'))), p1_rho=float(ev['rho']) if np.isfinite(ev['rho']) else None,
                p1_dent_pct=float(ev['dent_pct']) if np.isfinite(ev['dent_pct']) else None, p1_p_par_mm=float(ev['p_par_mm']), p1_p_perp_mm=float(ev['p_perp_mm']),
                drift_silent_at_ref=float(ev.get('silent_at_ref', float('nan'))), nn_median_mm=float(np.median(nn)), nn_p10_mm=float(np.percentile(nn, 10)), nn_min_mm=float(nn.min()),
                hole_max_mm=float(hole.max()), hole_p95_mm=float(np.percentile(hole, 95)), n_features_in_dent_disc=n_disc, pattern_sum=float(pat.sum()), pattern_px=int((pat > 0).sum()),
                radius_px_median=float(np.median(np.sqrt(np.asarray(areas, float) / math.pi))) if areas is not None and len(areas) else None, seconds_p1=round(t_p1, 3))


def rule_rho(sp, pp, fwhm=20.0):
    return 0.5 * math.sqrt((sp / fwhm) ** 2 + (pp / fwhm) ** 2)


def predict(family, config, feat, cond):

    reasons, cls = [], []
    A = feat['A_design']
    if A < TH['A_pass']:
        cls.append('pass'); reasons.append(f'A_design {A*100:.2f} % < 1 %')
    elif A >= TH['A_reject']:
        cls.append('reject'); reasons.append(f'A_design {A*100:.1f} % >= 10 %')
    else:
        cls.append('review'); reasons.append(f'A_design {A*100:.1f} % in [1, 10) %')
    if cond == 'both':
        b = feat['blur_px']
        if b < 2.5:
            cls.append('reject'); reasons.append(f'blur limit {b:.2f} px < PSF 2.5 px')
        elif b < 2.5 + TH['blur_review_px']:
            cls.append('review'); reasons.append(f'blur limit {b:.2f} px within 1.4 px of PSF 2.5')
        else:
            cls.append('pass'); reasons.append(f'blur limit {b:.2f} px')
        s = feat['noise_sigma_max']
        if np.isfinite(s):
            if s < 30:
                cls.append('reject'); reasons.append(f'noise limit sigma {s:.0f} < 30')
            elif s < 30 + TH['noise_review_sigma']:
                cls.append('review'); reasons.append(f'noise limit sigma {s:.0f} within 20 of 30')
            else:
                cls.append('pass'); reasons.append(f'noise limit sigma {s:.0f}')
    if family in DENT_RULE_FAMILIES:
        a = FAMILIES[family]['configs'][config]; rho = rule_rho(a['spacing_mm'], a.get('spacing_perp_mm', a['spacing_mm'])); pred = max(0.0, 52 * rho - 3)
        if pred <= TH['dent_pct']:
            cls.append('pass'); reasons.append(f'rule A: rho {rho:.3f} -> {pred:.1f} % <= 10 %')
        else:
            cls.append('reject'); reasons.append(f'rule A: rho {rho:.3f} -> {pred:.1f} % > 10 %')
    else:
        cls.append('review'); reasons.append('dent criterion: rectangular rho not defined for this family (P-1 generalised value recorded as exploratory)')
    final = 'reject' if 'reject' in cls else ('review' if 'review' in cls else 'pass')
    domain = 'new_family_out_of_domain' if family == 'DROP' else 'new_configuration_in_domain'
    if family == 'DROP':
        final_reported = 'out_of_domain'
    else:
        final_reported = final
    return dict(prediction=final_reported, prediction_if_applied=final, rule_classes=cls, reasons=reasons, domain=domain,
                baseline_density=('pass' if feat['n_visible'] >= TH['density_baseline_n_visible'] else 'reject'), baseline_all='pass')


def patterns():
    out = []
    for fam, f in FAMILIES.items():
        for cfg in f['configs']:
            for k in range(f['realisations']):
                out.append(dict(family=fam, config=cfg, realisation=k, seed=pattern_seed(fam, cfg, k)))
    return out


def spec(family, config, k, surface, cond, phase, albedo=1.0, **extra):
    s = dict(v=VERSION, family=family, config=config, realisation=int(k), surface=surface, cond=cond, phase=int(phase), albedo=round(float(albedo), 6))
    if extra:
        s['extra'] = extra
    return s


def key(s):
    return digest({k: s[k] for k in KEY_FIELDS})


def path(s):
    return CASES / (key(s) + '.json')


def design_main():
    return [spec(p['family'], p['config'], p['realisation'], surf, cond, ph) for p in patterns() for surf in SURFACES for cond in CONDITIONS for ph in range(PHASES)]


def design_light(pattern_sums):

    items = []
    for fam, f in FAMILIES.items():
        cfg = max(f['configs'], key=lambda c: pattern_sums[(fam, c, 0, 0)])
        for ph in range(PHASES):
            s_ref = pattern_sums[('REG', 'r8x4', 0, ph)]; s_t = pattern_sums[(fam, cfg, 0, ph)]
            alb = min(1.0, s_ref / s_t)
            if alb < 0.999:
                items.append(spec(fam, cfg, 0, 'baseline', 'both', ph, albedo=alb, ref='REG r8x4', pattern_sum_ref=s_ref, pattern_sum=s_t))
    return items


_SURF = {}


def surface(surf):
    if surf not in _SURF:
        _SURF[surf] = S.build_fixed_surface('A', SURFACES[surf], L14.GRIDS[GRID], L14.BASE, L_MM)
    return _SURF[surf]


def run_chain(mesh, cfg, pat, ptype, meta, cond, phase, albedo, spacing_mm):

    from m3_virtual_camera import synthesise_camera_image
    from m4_reconstruction import reconstruct
    from m5_evaluation import evaluate_all
    import preliminary_analysis as PA
    c = CONDITIONS[cond]
    m3_cfg = {'batch_size': PA.BATCH_SIZE, 'noise_sigma': float(c['noise_sigma']), 'psf_sigma_px': float(c['psf_sigma_px']), 'albedo': float(albedo), 'albedo_exposure': 1.0,
              'specular_k': 0.0, 'mottle_amp': 0.0, 'outlier_frac': 0.0, 'mottle_seed': int(NOISE_SEED_BASE + phase)}
    img, _, _, m3_info = synthesise_camera_image(mesh_dict=mesh, pattern=pat, cfg=m3_cfg)
    pts, info = reconstruct(img, pat, ptype, method='epipolar', sensor_cfg={'measurement_volume_mm': 20.0})
    metrics = evaluate_all(points_3d=pts, mesh=mesh, camera_image=img, m1_config=cfg, m3_info=m3_info, uniq_window_px=PA.UNIQ_WINDOW_PX, uniq_stride_px=PA.UNIQ_STRIDE_PX,
                           uniq_n_samples=PA.UNIQ_N_SAMPLES, spacing_mm=spacing_mm, theta_deg=None, recon_info=info)
    return img, pts, info, metrics, m3_info


def work(s):
    t0 = time.monotonic()
    try:
        mesh, cfg, ref = surface(s['surface'])
        pat, meta, ptype = make_pattern(s['family'], s['config'], s['realisation'], s['phase'])
        a = FAMILIES[s['family']]['configs'][s['config']]
        t1 = time.monotonic()
        img, pts, info, m, m3_info = run_chain(mesh, cfg, pat, ptype, meta, s['cond'], s['phase'], s['albedo'], a['spacing_mm'])
        t_run = time.monotonic() - t1
        pts = np.asarray(pts, np.float32); amb = np.asarray(info.get('ambiguous_mask', np.zeros(len(pts), bool)), bool)
        if len(amb) != len(pts):
            amb = np.zeros(len(pts), bool)
        finite = np.isfinite(pts).all(axis=1); rel = finite & ~amb; n_rel = int(rel.sum())
        gt = truth_depth(pts, mesh).astype(np.float32); err = np.where(finite, pts[:, 2] - gt, np.nan).astype(np.float32)
        zmin = float(np.asarray(mesh['Z']).min())
        dref = np.full(len(pts), np.nan); dref[finite] = S.z_ref_at(ref, cfg, pts[finite, 0], pts[finite, 1]) - zmin
        err_ref = np.where(finite, pts[:, 2] - dref, np.nan).astype(np.float32)
        r_rel, r_all = err_ref[rel & np.isfinite(err_ref)], err_ref[finite & np.isfinite(err_ref)]
        mc = clean(m)
        ex = dict(n_finite=int(finite.sum()), n_reliable_mask=n_rel, n_ambiguous_mask=int((amb & finite).sum()), zmin=zmin, n_detected=int(info.get('n_detected', 0) or 0),
                  ref_rmse_reliable_mm=float(np.sqrt(np.mean(r_rel ** 2))) if len(r_rel) else None, ref_rmse_all_mm=float(np.sqrt(np.mean(r_all ** 2))) if len(r_all) else None,
                  ref_p95_reliable_mm=float(np.percentile(np.abs(r_rel), 95)) if len(r_rel) else None, ref_outlier_rate_reliable=float((np.abs(r_rel) > 1.0).mean()) if len(r_rel) else None,
                  image_sum=float(img.astype(np.float64).sum() / 255.0), image_sat_frac=float((img >= 255).mean()), pattern_sum=float(pat.sum()), pattern_px=int((pat > 0).sum()),
                  pattern_hash=hashlib.sha256(np.ascontiguousarray(pat).tobytes()).hexdigest(), image_hash=hashlib.sha256(img.tobytes()).hexdigest(), pattern_type_m4=ptype,
                  pattern_seed=pattern_seed(s['family'], s['config'], s['realisation']), noise_seed=int(NOISE_SEED_BASE + s['phase'] + 104729), phase_quantiles=list(quantiles(s['phase'])),
                  seconds_pipeline=round(t_run, 3), pattern_meta={k: v for k, v in clean(meta).items() if k not in ('dot_centres',)})
        ex.update({k + '_all': v for k, v in occupancy(pts, finite).items()}); ex.update({k + '_rel': v for k, v in occupancy(pts, rel).items()})
        st = status_of(mc, n_rel)
        eng = engineering_pass(st, mc, ex, s['cond'])
        return dict(spec=s, key=key(s), status=st, engineering=eng, seconds=time.monotonic() - t0, metrics=mc, extras=ex,
                    arrays=dict(cam_img=img.astype(np.uint8), pattern=(pat > 0).astype(np.uint8), pts=pts, amb=amb, gt_depth=gt, gt_ref=dref.astype(np.float32), err=err, err_ref=err_ref))
    except Exception:
        return dict(spec=s, key=key(s), status='error', seconds=time.monotonic() - t0, error=traceback.format_exc(), metrics=None)


def engineering_pass(status, m, ex, cond):

    T = TH['eng']
    dent = (m.get('ind_depth_err_mm') / m['ind_depth_gt_mm'] * 100.0) if (m.get('ind_depth_err_mm') is not None and m.get('ind_depth_gt_mm')) else None
    c = dict(status_ok=(status == 'ok'), occ_dent=(ex['occ_dent_rel'] is not None and ex['occ_dent_rel'] >= T['occ_dent_min']),
             dent=(dent is not None and abs(dent) <= T['dent_abs_max']), outlier=(ex['ref_outlier_rate_reliable'] is not None and ex['ref_outlier_rate_reliable'] <= T['outlier_max']),
             rmse=(ex['ref_rmse_reliable_mm'] is not None and ex['ref_rmse_reliable_mm'] <= T['ref_rmse_max'][cond]))
    return dict(criteria=c, dent_pct=dent, passed=all(c.values()))


def store(r):
    s = r['spec']; p = path(s); arrays = r.pop('arrays', None)
    if arrays is not None:
        npz = p.with_suffix('.npz'); tmp = npz.with_suffix('.npz.tmp'); p.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(npz); r['npz'] = npz.name
    save(p, r)
    return p


def sources():
    files = [*sorted((ROOT / 'src').glob('*.py')), ROOT / 'preliminary_analysis.py', ROOT / 'scripts' / 'pm013' / 'pm013_lib.py', ROOT / 'scripts' / 'pm014' / 'pm014_lib.py',
             ROOT / 'scripts' / 'pm014' / 'pm014_surface.py', *sorted(HERE.glob('*.py'))]
    raw = sorted((ROOT / 'data' / 'raw').glob('*'))
    return {str(Path(p).relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files + raw if Path(p).is_file()}


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
