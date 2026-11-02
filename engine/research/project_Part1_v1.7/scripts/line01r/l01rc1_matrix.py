# Build the continuous-line experiment matrix and check detector calibration coverage.

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l01rc1_common as C
import l01rb_common as B

MX = C.OUT / 'matrix'

FORMAL_PHASES = tuple(k / 32 for k in (-15, -11, -7, -3, 1, 5, 9, 13))
C2_PHASES = tuple(k / 32 for k in (-9, 7))
SEEN_PHASES = (-3 / 8, -1 / 8, 1 / 8, 3 / 8, -7 / 16, -3 / 16, 1 / 16, 5 / 16) + tuple(B.PHASES)
SEED_FORMAL, SEED_C2 = 10300000, 10400000
L1_WIDTHS, L1_PITCH = (0.5, 0.75, 1.0, 1.25, 1.5), 8.0
L2_PITCHES, L2_WIDTH = (4.45, 5.0, 5.6, 6.3, 8.0, 10.0, 12.0), 1.0
SIGMAS_FORMAL = (0.0, 20.0, 40.0, 60.0)
REGIMES = ('fixed_peak', 'equal_integral')
C2_GEOMS = (('L1', 0.5, 8.0), ('L1', 0.75, 8.0), ('L1', 1.5, 8.0), ('L2', 1.0, 4.45), ('L2', 1.0, 5.6), ('L2', 1.0, 12.0))
C2_SIGMAS = (0.0, 40.0, 60.0)
C2_HARD_CAP, C2_SUGGESTED = 384, (192, 320)


def seed_for(base, surface, phase_index, sigma, rep):
    sig_levels = (0.0, 20.0, 40.0, 60.0)
    idx = ((B.SURFACES.index(surface) * 8 + int(phase_index)) * len(sig_levels) + sig_levels.index(float(sigma))) * 2 + int(rep)
    return base + idx, f'{surface}_ph{phase_index}_s{int(sigma)}_r{rep}'


def build(block_rows, phases, seed_base, label, reps_sigma=2):

    rows = []
    cache = {}
    for r in block_rows:
        for pi, ph in enumerate(phases):
            key = (r['phi_deg'], r['w_mm'], r['s_mm'], ph)
            if key not in cache:
                pat, meta = B.pattern_for(dict(s_mm=r['s_mm'], w_mm=r['w_mm'], phi_deg=r['phi_deg'], phase_frac=ph))
                cache[key] = dict(integral=float(pat.sum()), raster=B.sha_array(pat), lit=int((pat > 0).sum()), n_lines=int(meta['olines']['n_lines']), width_px=float(meta['olines']['width_px']), spacing_px=float(meta['olines']['spacing_px']))
            pc = cache[key]
            reps = (0,) if r['sigma'] == 0 else tuple(range(reps_sigma))
            for rep in reps:
                s = dict(block=r['block'], surface=r['surface'], phi_deg=r['phi_deg'], w_mm=r['w_mm'], s_mm=r['s_mm'], regime=r['regime'], sigma=r['sigma'], phase_index=pi, phase_frac=float(ph), purpose=r['purpose'], noise_rep=rep)
                s['noise_seed'], s['noise_realisation'] = seed_for(seed_base, r['surface'], pi, r['sigma'], rep)
                s.update(B.budget_of(s, pc['integral'])); s.update(raster_sha256=pc['raster'], lit_px=pc['lit'], n_lines=pc['n_lines'], width_px=pc['width_px'], spacing_px=pc['spacing_px'])
                s['regime_pair_id'] = hashlib.sha256(json.dumps([s['surface'], s['phi_deg'], s['w_mm'], s['s_mm'], s['sigma'], pi, rep]).encode()).hexdigest()[:12]; s['brightness_pair_id'] = None; s['phase_block'] = f'ph{pi}'; s['trial'] = False
                s['forward_key'] = B.forward_key(s, pc['raster'], s['peak'] if s['peak'] is not None else -1.0); s['case_id'] = s['forward_key']
                rows.append(s)
    df = pd.DataFrame(rows)
    n_before = len(df); df = df.drop_duplicates('forward_key').reset_index(drop=True); df['planned_index'] = range(len(df))
    return df, n_before


def formal_rows():
    rows = []
    for surf, phi, w, reg, sg in itertools.product(B.SURFACES, B.PHIS, L1_WIDTHS, REGIMES, SIGMAS_FORMAL):
        rows.append(dict(block='L1', surface=surf, phi_deg=phi, w_mm=w, s_mm=L1_PITCH, regime=reg, sigma=sg, purpose='L1: line-width boundary, brightness cost and noise at the fixed pitch 8 mm'))
    for surf, phi, s, reg, sg in itertools.product(B.SURFACES, B.PHIS, L2_PITCHES, REGIMES, SIGMAS_FORMAL):
        rows.append(dict(block='L2', surface=surf, phi_deg=phi, w_mm=L2_WIDTH, s_mm=s, regime=reg, sigma=sg, purpose='L2: pitch, orientation, sampling and uniqueness trade-off at the fixed width 1.0 mm'))
    return rows


def c2_rows():
    rows = []
    for (blk, w, s), surf, phi, reg, sg in itertools.product(C2_GEOMS, B.SURFACES, B.PHIS, REGIMES, C2_SIGMAS):
        rows.append(dict(block=f'C2_{blk}', surface=surf, phi_deg=phi, w_mm=w, s_mm=s, regime=reg, sigma=sg, purpose='C2 unseen-material validation (frozen methods, no tuning)'))
    return rows


def geometry_K():

    from m4_lines import epipolar_in_projector
    g = B.geom(); cfg = B.m4_cfg(1.0)
    uu, vv = np.meshgrid(np.arange(16, 512, 32), np.arange(16, 512, 32)); cam = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float64)
    P0, P1 = epipolar_in_projector(cam, cfg, B.Z0_MM, B.Z1_MM); e = P1 - P0; L = np.linalg.norm(e, axis=1); eu = e / L[:, None]
    rows = []
    for phi in B.PHIS:
        t = np.array([-math.sin(math.radians(phi)), math.cos(math.radians(phi))]); sa = np.abs(eu[:, 0] * t[1] - eu[:, 1] * t[0])
        for s in L2_PITCHES:
            pitch = s / g['mm_per_px'] / np.maximum(sa, 1e-9); K = np.ceil(L / pitch)
            rows.append(dict(phi=phi, s_mm=s, along_epipolar_pitch_mm=float(np.median(pitch) * g['mm_per_px']), track_px_max=float(L.max()), K_upper_bound_max=int(K.max()), K_upper_bound_median=float(np.median(K)),
                             unique_depth_mm_min=float((pitch / L * (B.Z1_MM - B.Z0_MM)).min()), note='upper bound from the track length; the actual candidate sets of B2 were smaller (45 deg / 4.45 and 0 deg / 6.3 were not hard)'))
    return pd.DataFrame(rows)


# Check noise estimates and line-filter responses using calibration images.
def calibration_checks():

    from scipy.ndimage import gaussian_filter
    from m4_lines import STEGER_SUBPX_TOL, STEGER_NOISE_COEF, steger_design_response, estimate_noise_sigma
    g = B.geom(); rng = np.random.default_rng(20261010)
    scales = sorted({max(1.0, w / g['mm_per_px'] / (2 * math.sqrt(3))) for w in L1_WIDTHS} | {1.0, 1.6, 2.4})

    base_imgs = []
    for s in B.manifest():
        if s['block'] == 'B1' and s['regime'] == 'fixed_peak' and s['sigma'] == 0.0 and s['surface'] == 'baseline' and s['phi_deg'] == 0.0 and s['phase_index'] == 0:
            with np.load(B.CASES / f"{s['case_id']}.npz") as z:
                base_imgs.append((s['w_mm'], np.asarray(z['image'])))
    rows = []
    for sig in (20.0, 40.0, 60.0):
        noise = rng.normal(0.0, sig / 255.0, (512, 512)); img = np.clip(np.clip(noise, 0.0, 1.0) * 255.0, 0, 255).astype(np.uint8); f = img.astype(np.float64)
        sh = float(np.quantile(f, 0.8413)); zf = float((f == 0).mean()); st = 'q841_on_pure_background'
        line_est = {}
        for w, bimg in base_imgs:
            noisy = np.clip(np.clip(bimg.astype(np.float64) / 255.0 + rng.normal(0.0, sig / 255.0, bimg.shape), 0.0, 1.0) * 255.0, 0, 255).astype(np.uint8)
            v, st_l, frac_l, zf_l = estimate_noise_sigma(noisy); line_est[f'w{w:g}'] = dict(sigma_hat=v, status=st_l, bg_frac=frac_l)
        for sf in scales:
            rx = gaussian_filter(f, sf, order=(0, 1)); ry = gaussian_filter(f, sf, order=(1, 0)); rxx = gaussian_filter(f, sf, order=(0, 2)); ryy = gaussian_filter(f, sf, order=(2, 0)); rxy = gaussian_filter(f, sf, order=(1, 1))
            disc = np.sqrt(((rxx - ryy) / 2.0) ** 2 + rxy ** 2); lam = (rxx + ryy) / 2.0 - disc
            ex = rxy.copy(); ey = lam - rxx; small = np.abs(rxy) < 1e-12
            ex[small] = np.where(rxx[small] <= ryy[small], 1.0, 0.0); ey[small] = np.where(rxx[small] <= ryy[small], 0.0, 1.0)
            nrm = np.hypot(ex, ey); nrm[nrm == 0] = 1.0; nx_, ny_ = ex / nrm, ey / nrm
            den = rxx * nx_ ** 2 + 2 * rxy * nx_ * ny_ + ryy * ny_ ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                t = -(rx * nx_ + ry * ny_) / den
            cond = np.isfinite(t) & (np.abs(t * nx_) <= STEGER_SUBPX_TOL) & (np.abs(t * ny_) <= STEGER_SUBPX_TOL); resp = -lam[cond]; resp = resp[resp > 0]
            r = 1e-4; thr = float(np.quantile(resp, 1.0 - r * img.size / max(resp.size, 1))) if resp.size and r * img.size < resp.size else float(resp.max())
            c_meas = thr / sig; ks = sorted(STEGER_NOISE_COEF); c_tab = float(np.interp(sf, ks, [STEGER_NOISE_COEF[k] for k in ks]))
            row = dict(sigma=sig, sigma_hat=sh, sigma_hat_status=st, bg_zero_frac=zf, scale=round(sf, 3), thr_fp1e4=thr, c_measured=c_meas, c_table_interp=c_tab, c_rel_dev=(c_meas - c_tab) / c_tab,
                       **{f'sigma_hat_line_{k}': v['sigma_hat'] for k, v in line_est.items()}, **{f'sigma_hat_line_{k}_status': v['status'] for k, v in line_est.items()})
            for w in L1_WIDTHS:
                row[f'margin_w{w:g}_P1'] = steger_design_response(w / g['mm_per_px'], sf, 1.0, 255.0) / thr
            rows.append(row)
    df = pd.DataFrame(rows)

    ext = {}
    rng2 = np.random.default_rng(20261009)
    for sig in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0):
        noise = rng2.normal(0.0, sig / 255.0, (512, 512)); f2 = np.clip(np.clip(noise, 0.0, 1.0) * 255.0, 0, 255).astype(np.uint8).astype(np.float64)
        for sf in sorted({round(s, 3) for s in scales}):
            rx = gaussian_filter(f2, sf, order=(0, 1)); ry = gaussian_filter(f2, sf, order=(1, 0)); rxx = gaussian_filter(f2, sf, order=(0, 2)); ryy = gaussian_filter(f2, sf, order=(2, 0)); rxy = gaussian_filter(f2, sf, order=(1, 1))
            disc = np.sqrt(((rxx - ryy) / 2.0) ** 2 + rxy ** 2); lam = (rxx + ryy) / 2.0 - disc
            ex = rxy.copy(); ey = lam - rxx; small = np.abs(rxy) < 1e-12
            ex[small] = np.where(rxx[small] <= ryy[small], 1.0, 0.0); ey[small] = np.where(rxx[small] <= ryy[small], 0.0, 1.0)
            nrm = np.hypot(ex, ey); nrm[nrm == 0] = 1.0; nx_, ny_ = ex / nrm, ey / nrm
            den = rxx * nx_ ** 2 + 2 * rxy * nx_ * ny_ + ryy * ny_ ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                t = -(rx * nx_ + ry * ny_) / den
            cond = np.isfinite(t) & (np.abs(t * nx_) <= STEGER_SUBPX_TOL) & (np.abs(t * ny_) <= STEGER_SUBPX_TOL); resp = -lam[cond]; resp = resp[resp > 0]
            thr = float(np.quantile(resp, 1.0 - 1e-4 * f2.size / max(resp.size, 1))) if resp.size and 1e-4 * f2.size < resp.size else float(resp.max())
            ext.setdefault(sf, []).append(thr / sig)
    ext_table = {str(k): dict(c_mean=float(np.mean(v)), c_min=float(np.min(v)), c_max=float(np.max(v)), n_sigma=len(v)) for k, v in sorted(ext.items())}
    line_cols = [c for c in df.columns if c.startswith('sigma_hat_line_') and not c.endswith('_status')]
    summ = dict(scales_checked=[round(s, 3) for s in scales], sigma_hat={str(int(s)): dict(q841_pure_background=float(df[df.sigma == s].sigma_hat.iloc[0]), **{c: (float(df[df.sigma == s][c].iloc[0]) if df[df.sigma == s][c].notna().iloc[0] else None) for c in line_cols}) for s in (20.0, 40.0, 60.0)},
                c_rel_dev_max_abs=float(df.c_rel_dev.abs().max()),
                c_rel_dev_by_scale={str(round(s, 3)): float(df[df.scale == round(s, 3)].c_rel_dev.abs().max()) for s in scales},
                coverage_ok=bool(df.c_rel_dev.abs().max() <= 0.15), rule='interpolated registered coefficient within 15 % of the direct calibration at the new scales and sigma 60 -> table kept; otherwise register an extended table (option, default unchanged)',
                extended_table_proposal=dict(method='A3 procedure (fp 1e-4, mean of thr / sigma over sigma_true 10..60, seed 20261009) at the registered and the two new scales', table=ext_table,
                                             registered_knots_unchanged={'1.0': STEGER_NOISE_COEF[1.0], '1.6': STEGER_NOISE_COEF[1.6], '2.4': STEGER_NOISE_COEF[2.4]},
                                             effect_on_B='none: the B widths map exactly onto the registered knots 1.0 / 1.6 / 2.4 (np.interp returns the knot values); new knots only change w 0.75 (scale 1.20) and w 1.25 (scale 2.00)',
                                             status='freeze-candidate option steger_noise_coef (to be approved by reference); module default unchanged'),
                line_margin_at_own_scale_P1={f'sigma{int(sg)}_w{w:g}': float(df[(df.sigma == sg) & (df.scale == round(max(1.0, w / g['mm_per_px'] / (2 * math.sqrt(3))), 3))][f'margin_w{w:g}_P1'].iloc[0]) for sg in (20.0, 40.0, 60.0) for w in L1_WIDTHS},
                note='margin = full design response / calibrated noise threshold at peak 1.0 and the filter scale of that width (sigma_f = max(1, w_px / (2 sqrt 3))); the detector threshold is max(0.3 x design, noise threshold), so a margin below 1 / 0.3 = 3.3 means the noise term dominates')
    return df, summ


def main():
    MX.mkdir(parents=True, exist_ok=True); C.progress(state='matrix_running')
    assert not (set(FORMAL_PHASES) | set(C2_PHASES)) & set(SEEN_PHASES) and not set(FORMAL_PHASES) & set(C2_PHASES)
    formal, n_formal_raw = build(formal_rows(), FORMAL_PHASES, SEED_FORMAL, 'formal'); C.save_csv(formal, MX / 'formal_manifest_candidate.csv')
    c2, n_c2_raw = build(c2_rows(), C2_PHASES, SEED_C2, 'c2', reps_sigma=1); C.save_csv(c2, MX / 'c2_manifest_candidate.csv')

    b_keys = {s['forward_key'] for s in B.manifest()}; f_keys = set(formal.forward_key); c_keys = set(c2.forward_key)
    old = B.design_old(); old_phase_seed = {(s['phase_frac'], s['noise_seed']) for s in old}
    disj = dict(B_vs_formal=len(b_keys & f_keys), B_vs_c2=len(b_keys & c_keys), formal_vs_c2=len(f_keys & c_keys), phases_seen_vs_formal=len(set(SEEN_PHASES) & set(FORMAL_PHASES)), phases_seen_vs_c2=len(set(SEEN_PHASES) & set(C2_PHASES)),
                seeds_overlap_line01=len({int(x) for x in formal.noise_seed} & {s['noise_seed'] for s in old}) + len({int(x) for x in c2.noise_seed} & {s['noise_seed'] for s in old}), seeds_overlap_B=len({int(x) for x in formal.noise_seed} & {s['noise_seed'] for s in B.manifest()}) + len({int(x) for x in c2.noise_seed} & {s['noise_seed'] for s in B.manifest()}),
                seeds_overlap_formal_c2=len({int(x) for x in formal.noise_seed} & {int(x) for x in c2.noise_seed}), n_old_phase_seed_pairs=len(old_phase_seed))
    geo = geometry_K(); C.save_csv(geo, MX / 'formal_geometry_K.csv')
    cal, cal_s = calibration_checks(); C.save_csv(cal, MX / 'calibration_new_scales_sigma60.csv')

    tb = B.load_json(B.OUT / 'b_summary.json')['timing']; pm = B.load_json(B.progress_file('main'))['stages']['full']
    per_case_single = tb['render_s_mean'] + 0.3 + sum(tb['decode_s_mean_by_path'][p] + 2 * tb['eval_s_mean_by_path'][p] for p in ('P2_C1_guard', 'P4_C3_cal'))
    wall_b = pm['s_per_case_wall']; speedup = (tb['render_s_mean'] + 0.3 + sum(tb['decode_s_mean_by_path'][p] + 2 * tb['eval_s_mean_by_path'][p] for p in ('P1_C1_hist', 'P2_C1_guard', 'P4_C3_cal'))) / wall_b
    n_formal = len(formal)
    cost = dict(basis='B full run: %d cases, %.1f s / case wall with 6 workers for 3 paths x (P + O) evaluation; per-path decode and evaluation seconds from b_summary.json' % (pm['n_run'], wall_b),
                per_case_single_process_s_two_paths=round(per_case_single, 2), measured_speedup_6_workers=round(speedup, 2), wall_s_per_case_two_paths=round(per_case_single / speedup, 2),
                formal_images=n_formal, formal_hours_two_paths=dict(expected=round(n_formal * per_case_single / speedup / 3600, 2), with_safety_x1_5=round(1.5 * n_formal * per_case_single / speedup / 3600, 2)),
                formal_hours_three_paths_with_P1=round(n_formal * (per_case_single + tb['decode_s_mean_by_path']['P1_C1_hist'] + 2 * tb['eval_s_mean_by_path']['P1_C1_hist']) / speedup / 3600, 2),
                c2_images=len(c2), c2_minutes_two_paths=round(len(c2) * per_case_single / speedup / 60, 1), disk_mb_per_case_B=round(135.0 / 176, 2), formal_disk_gb=round(n_formal * 135.0 / 176 / 1000, 2),
                note='the evaluation (two layers x two paths) dominates; dropping the O-layer evaluation in the formal run would save ~40 %; the estimate is a derivation from measured B throughput, not a guarantee')
    feas = dict(formal=dict(rows_before_dedup=n_formal_raw, unique=n_formal, feasible=int(formal.feasible.sum()), infeasible=int((~formal.feasible).sum()), infeasible_rows=formal[~formal.feasible][['block', 'phi_deg', 'w_mm', 's_mm', 'regime', 'phase_index', 'required_peak']].to_dict(orient='records'),
                            by_block=formal.groupby('block').size().to_dict(), L1_L2_overlap_removed=n_formal_raw - n_formal, equal_integral_peak_by_geometry=formal[formal.regime == 'equal_integral'].groupby(['block', 'w_mm', 's_mm']).peak.agg(['min', 'max']).reset_index().to_dict(orient='records')),
                c2=dict(rows_before_dedup=n_c2_raw, unique=len(c2), feasible=int(c2.feasible.sum()), hard_cap=C2_HARD_CAP, suggested_range=list(C2_SUGGESTED), within_suggested=(C2_SUGGESTED[0] <= len(c2) <= C2_SUGGESTED[1]), by_block=c2.groupby('block').size().to_dict(),
                        coverage=dict(new_width_scales=['0.75 (sigma_f 1.20)'], fixed_peak_thin_line='w 0.5 / 8 mm', equal_integral_dim_wide='w 1.5 / 8 mm', dense_multi_candidate='0 deg / 4.45 mm (and 45 deg)', boundary_pitch='5.6 mm', sparse_gap='12 mm', surfaces=2, orientations=2, noise_upper_bound='sigma 60', phases=list(C2_PHASES), noise_realisations='1 per sigma > 0')))
    checklist = [
        dict(item='no truth leak', gate='structural audit of the P path (truth only in O / G / evaluation) + the m4_lines unit tests pass', source='task 4 / A O-P contract', failure='stop; fix; the batch becomes development material'),
        dict(item='default-path regression', gate='10 representative patterns array-identical to the LINE-01 before snapshot; tests 53/53 and 9/9', source='A / B regression protocol', failure='stop; locate the change; re-run'),
        dict(item='ledger closure', gate='research attempts = ok + failed; unique images = feasible planned; attempts <= 384; test renders <= 64', source='task 5 / 6', failure='stop new imaging; reconcile the ledger'),
        dict(item='easy-condition fidelity (both surfaces, sigma 0, fixed peak, w 1.0, s 8)', gate='recovered_inner >= 0.99, normal error P95 <= 0.20 px (P2) / 0.10 px (P4), confirmable wrong ids = 0, gross <= 0.2 %', source='B sigma-0 values (0.997-0.999, 0.17 / 0.08 px, 0, <= 0.1 %) with margin', failure='stop; implementation or reference error suspected'),
        dict(item='identity / failure ledger per case', gate='accepted = correct + wrong + unconfirmable; rejected by reason sums to detected - accepted; class recorded', source='B denominators', failure='stop; locate'),
        dict(item='geometry and reference recomputation', gate='>= 8 representative clouds: independent RMSE / amplitude / D95 within 1e-6 mm (RMSE) and stored rounding', source='reference audits of A and B', failure='stop'),
        dict(item='resume / corruption branches', gate='NPZ missing, truncated, stale eval -> reeval, stale decode -> redecode, program error, no output: each actually triggered in the C2 rehearsal dir', source='B rehearsal', failure='stop; fix runner'),
        dict(item='O / P consistency', gate='consistency.all_ok for every record', source='B 3.4', failure='stop'),
        dict(item='window effective', gate='every decode window = [-1, 21], tolerance 1e-6 (assertion)', source='B 2.1', failure='stop'),
        dict(item='throughput and resources', gate='wall time per case <= 2 x planned (planned from this estimate); parent peak RSS < 4 GB; disk growth <= 1.3 x estimate', source='B run resources', failure='pause; adjust workers; no data change'),
        dict(item='difficult-domain performance (reported, not gated)', gate='report recovered / gross / unknown / C_tau / amplitude for dense pitch, dim wide line, sigma 60; no threshold', source='task 5.4', failure='n/a: research result'),
    ]
    out = dict(generated_local=C.now_local(), phases=dict(formal=list(FORMAL_PHASES), c2=list(C2_PHASES), seen=list(SEEN_PHASES)), seeds=dict(formal_base=SEED_FORMAL, c2_base=SEED_C2, rule='seed = base + index(surface, phase, sigma level, realisation)'), disjointness=disj,
               feasibility=feas, geometry_K=geo.round(3).to_dict(orient='records'), calibration=cal_s, cost=cost, c2_release_checklist=checklist,
               formal_paths=dict(main='P4_C3_cal', control='P2_C1_guard', historical_reference='P1_C1_hist (optional, cost +40 %)'), window=B.window_cfg())
    C.save_json(MX / 'matrix_summary.json', out)
    C.stage('matrix', done=True, formal=n_formal, c2=len(c2), disjoint=all(v == 0 for k, v in disj.items() if k != 'n_old_phase_seed_pairs'), calibration_ok=cal_s['coverage_ok'])
    print(json.dumps(dict(formal=n_formal, feasible=feas['formal']['feasible'], c2=len(c2), disj=disj, cal=cal_s['c_rel_dev_by_scale'], cost_h=cost['formal_hours_two_paths']), ensure_ascii=False))


if __name__ == '__main__':
    main()
