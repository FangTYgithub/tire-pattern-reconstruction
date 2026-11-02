# Prepare bounded sub-ray comparisons and their rendering inputs.

from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pandas as pd

import tr3_common as K

REAL_CALLS = dict(render=0, decode=0)
PERMIT = K.OUT / 'B1_PERMIT.json'
EXIT = dict(ok=0, refused_no_permit=20, compat_stop=30, backend_error=40, input_check=50)


def validate_rays(x) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise ValueError(f'rays_per_pixel must be an int, got {type(x).__name__} {x!r}')
    if isinstance(x, float) and math.isnan(x):
        raise ValueError('rays_per_pixel NaN')
    x = int(x)
    if x < 1 or x > 7:
        raise ValueError(f'rays_per_pixel {x} outside 1..7')
    return x


def prepare(row: dict, rays) -> dict:

    import s1r_common as S1K
    import rdy_platform as P
    n = validate_rays(rays)
    S = P.s2(); J = K.load_json(K.ROOT / row['image_record'])
    R = S1K.load_raster(row['raster_key']); pattern = S1K.pattern_of(R['binary'], float(J['peak_f32']))
    seed = int(J['noise_seed']) if float(J['sigma']) > 0 else 0
    if seed != int(J['m3_seed_passed']):
        raise ValueError(f"seed rule {seed} differs from the stored m3_seed_passed {J['m3_seed_passed']}")
    cfg = dict(S.m3_cfg(float(J['sigma']), seed)); base = dict(cfg)
    if abs(float(cfg.get('psf_sigma_px', -1)) - float(J['psf_px'])) > 1e-12:
        raise ValueError(f"platform PSF {cfg.get('psf_sigma_px')} != stored {J['psf_px']}")
    if 'rays_per_pixel' in cfg:
        raise ValueError('platform m3_cfg already sets rays_per_pixel (unexpected)')
    cfg['rays_per_pixel'] = n
    integ = float(pattern.sum(dtype=np.float64))
    if abs(integ - float(J['integral_actual'])) > 1e-6 * max(1.0, integ):
        raise ValueError(f"pattern integral {integ} differs from the stored {J['integral_actual']} (double peak scaling?)")
    return dict(condition=row['input_key'], sid=J['platform_surface'], pattern=pattern, pattern_sha=K2_content(pattern),
                cfg=cfg, cfg_without_rays=base, rays=n, seed=seed, stored_image=K.ROOT / row['image_record'].replace('.json', '.npz'),
                stored_npz_sha=J['npz_sha256'], platform_version=J['platform_version'], integral=integ)


def real_render(prep: dict) -> np.ndarray:

    if not permit_ok():
        raise PermissionError('B1 real render refused: no valid permit')
    import rdy_platform as P
    from m3_virtual_camera import synthesise_camera_image
    REAL_CALLS['render'] += 1
    S = P.s2(); img, _, _, m3 = synthesise_camera_image(S.build_mesh(prep['sid']), np.asarray(prep['pattern'], np.float32), prep['cfg'])
    if m3.get('z_alignment') != 'absolute':
        raise RuntimeError(f"M3 alignment {m3.get('z_alignment')}")
    return np.asarray(img, np.uint8)


def permit_ok() -> bool:
    if not PERMIT.exists():
        return False
    p = K.load_json(PERMIT)
    return bool(p.get('authorised') and p.get('task') and p.get('max_renders', 0) >= 1)


def _ledger(state_dir, rec: dict) -> None:
    q = K.guard(state_dir / 'ledger.jsonl')
    with open(q, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(K.clean(rec), ensure_ascii=False, default=str) + chr(10))


def _ledger_rows(state_dir) -> list:
    p = state_dir / 'ledger.jsonl'
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def run(conditions: list, backend, decoder, method_id, state_dir, real: bool, prepare_fn=prepare) -> dict:

    if real and not permit_ok():
        return dict(exit=EXIT['refused_no_permit'], stage='permit', calls=0)
    done = {(r['stage'], r['key']): r for r in _ledger_rows(state_dir) if r.get('status') == 'ok'}
    calls = 0; compat = {}
    try:
        for c in conditions:
            p1 = prepare_fn(c, 1)
            img = backend(p1); calls += 1
            stored = np.asarray(K.load_npz(p1['stored_image'])['image'])
            ok = img.shape == stored.shape and img.dtype == stored.dtype and bool(np.array_equal(img, stored))
            compat[c['input_key']] = ok
            _ledger(state_dir, dict(stage='compat', key=c['input_key'], status='ok' if ok else 'mismatch', rays=p1['cfg']['rays_per_pixel']))
        if not all(compat.values()):
            return dict(exit=EXIT['compat_stop'], stage='compat', calls=calls, mismatched=[k for k, v in compat.items() if not v], decodes=0, renders_3x3=0)
        imgs3 = {}
        for c in conditions:
            p3 = prepare_fn(c, 3)
            img = backend(p3); calls += 1
            fp = K2_content(img); imgs3[c['input_key']] = (img, fp)
            _ledger(state_dir, dict(stage='render3', key=c['input_key'], status='ok', image_fp=fp, rays=p3['cfg']['rays_per_pixel']))
        dec = 0; reused = 0
        for c in conditions:
            for m in c['methods']:
                img, fp = imgs3[c['input_key']]; mid = method_id(m); key = f"{c['input_key']}|{m}"
                prev = done.get(('decode', key))
                if prev and prev.get('image_fp') == fp and prev.get('method_fp') == mid:
                    reused += 1; continue
                decoder(img, c, m); dec += 1
                _ledger(state_dir, dict(stage='decode', key=key, status='ok', image_fp=fp, method_fp=mid))
        return dict(exit=EXIT['ok'], stage='done', calls=calls, decodes=dec, decodes_reused=reused, renders_3x3=len(imgs3))
    except Exception as e:
        _ledger(state_dir, dict(stage='error', key='-', status='failed', error=repr(e)[:200]))
        return dict(exit=EXIT['backend_error'], stage='error', calls=calls, error=repr(e)[:200])


def K2_content(img) -> str:
    import tr3_gate as G
    return G.content_sha(np.asarray(img))


# Test sub-ray input preparation, rendering dispatch and failure handling.
def drills(B1: pd.DataFrame) -> dict:

    tmp = K.guard(K.OUT / '_b1_drill' / 'x').parent; res = {}; seen = []
    conds = []
    for r in B1.head(2).to_dict('records'):
        conds.append(dict(r, methods=(['D0', 'DF'] if r['family'] == 'point' else ['P4'])))

    def stub_ok(p):
        seen.append((p['condition'], p['cfg']['rays_per_pixel'], {k: v for k, v in p['cfg'].items() if k != 'rays_per_pixel'}))
        img = np.asarray(K.load_npz(p['stored_image'])['image']).copy()
        if p['rays'] == 3:
            img[0, 0] = (int(img[0, 0]) + 7) % 256
        return img

    def stub_fail(p):
        img = stub_ok(p)
        if p['rays'] == 1:
            img[100, 100] = (int(img[100, 100]) + 1) % 256
        return img

    def stub_error(p):
        raise RuntimeError('stub backend error')
    decoded = []
    dec = lambda img, c, m: decoded.append((c['input_key'], m))
    mid = lambda m: f'method-{m}-v1'
    try:
        r = run(conds, stub_fail, dec, mid, tmp / 'a', real=False)
        res['compat_fail_stops_before_3x3'] = r['exit'] == EXIT['compat_stop'] and r['renders_3x3'] == 0 and r['decodes'] == 0 and not decoded
        seen.clear(); r = run(conds, stub_ok, dec, mid, tmp / 'b', real=False)
        rays = [s[1] for s in seen]; base1 = [s[2] for s in seen if s[1] == 1]; base3 = [s[2] for s in seen if s[1] == 3]
        res['parameter_passed_1_then_3'] = r['exit'] == 0 and rays == [1] * len(conds) + [3] * len(conds)
        res['only_rays_differ_between_1x1_and_3x3'] = all(json.dumps(K.clean(a), sort_keys=True) == json.dumps(K.clean(b), sort_keys=True) for a, b in zip(base1, base3))
        n_dec_first = r['decodes']
        r2 = run(conds, stub_ok, dec, mid, tmp / 'b', real=False)
        res['resume_reuses_when_image_and_method_unchanged'] = r2['decodes'] == 0 and r2['decodes_reused'] == n_dec_first
        mid2 = lambda m: f'method-{m}-v2' if m == 'DF' else f'method-{m}-v1'
        r3 = run(conds, stub_ok, dec, mid2, tmp / 'b', real=False)
        n_df = sum(1 for c in conds for m in c['methods'] if m == 'DF')
        res['method_change_invalidates_only_that_method'] = r3['decodes'] == n_df and r3['decodes_reused'] == n_dec_first - n_df
        def stub_changed(p):
            img = stub_ok(p)
            if p['rays'] == 3:
                img[1, 1] = (int(img[1, 1]) + 3) % 256
            return img
        r4 = run(conds, stub_changed, dec, mid2, tmp / 'b', real=False)
        res['image_change_invalidates_all_decodes'] = r4['decodes'] == n_dec_first and r4['decodes_reused'] == 0
        r5 = run(conds, stub_error, dec, mid, tmp / 'c', real=False)
        res['backend_error_nonzero_exit'] = r5['exit'] == EXIT['backend_error']
        before = dict(REAL_CALLS)
        r6 = run(conds, real_render, dec, mid, tmp / 'd', real=True)
        res['no_permit_refuses_real_execution'] = r6['exit'] == EXIT['refused_no_permit'] and REAL_CALLS == before
        try:
            real_render(prepare(conds[0], 1)); res['real_render_guard'] = False
        except PermissionError:
            res['real_render_guard'] = REAL_CALLS == before
        bad = []
        for v in (0, -1, float('nan'), 3.0, True, '3', 9):
            try:
                validate_rays(v); bad.append(v)
            except ValueError:
                pass
        res['invalid_rays_rejected'] = bad == []
        res['valid_rays_accepted'] = [validate_rays(v) for v in (1, 3, 5)] == [1, 3, 5]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dict(passed=all(res.values()), checks=res, real_calls=dict(REAL_CALLS), temp_removed=not tmp.exists(),
                note='flow test with stub backends only - NOT a numerical regression of the real renderer')


def _files_equal(pairs: dict) -> tuple[int, list]:
    diff = [f for f, h in pairs.items() if not (K.ROOT / f).exists() or K.sha(K.ROOT / f) != h]
    return len(pairs), diff


def method_identity_of_record(eval_record: str, E: dict) -> tuple[str, bool, tuple]:

    R = K.ROOT / 'outputs' / 'surface_revision'
    if '/s1_real01/' in eval_record.replace('\\', '/'):
        man = R / 's1_real01' / 'config' / 's1_dependencies.json'; M = K.load_json(man)
        n, diff = _files_equal({str(f).replace('\\', '/'): h for f, h in M['files'].items()})
        ok = bool(E.get('s1_dependencies_sha256') == K.sha(man)) and not diff
        return 'S1-REAL-01 dependency manifest (pre-run; its sha is recorded in the decode record)', ok, (n, len(diff))
    if '/s12_resource01/' in eval_record.replace('\\', '/'):
        D = K.load_json(R / 's12_resource01' / 'protocol.json')['dependencies']; pairs = {}
        for key in ('release', 'platform_dependency_manifest', 's1_dependencies', 'binding', 'global_domain'):
            if isinstance(D.get(key), dict) and D[key].get('file'):
                pairs[D[key]['file']] = D[key]['sha256']
        pairs.update({f'scripts/surface_revision/s12_resource01/{f}': h for f, h in D['run_code'].items()})
        pairs.update({str(f).replace('\\', '/'): h for f, h in D['reused_modules'].items()})
        for key in ('platform_dependency_manifest', 's1_dependencies'):
            M = K.load_json(K.ROOT / D[key]['file']); pairs.update({str(f).replace('\\', '/'): h for f, h in M['files'].items()})
        n, diff = _files_equal(pairs)
        return 'S12-RESOURCE-01 protocol.json dependencies (pre-run 2026-10-14 17:29) incl. platform and S1 manifests', not diff, (n, len(diff))
    return 'unknown origin', False, (0, 0)


def inventory(B1: pd.DataFrame) -> pd.DataFrame:
    import tr3_gate as G
    C4 = K.read_csv(K.ROOT / 'outputs/surface_revision/s3_code04/results/C1_dots_cases.csv'); C4 = C4[C4.rule == 'DF'].set_index('input_key')
    s1dep = K.load_json(K.ROOT / 'outputs/surface_revision/s1_real01/config/s1_dependencies.json')
    rows = []
    for r in B1.to_dict('records'):
        J = K.load_json(K.ROOT / r['image_record']); img_ok = K.sha(K.ROOT / r['image_record'].replace('.json', '.npz')) == J['npz_sha256']
        methods = ['D0', 'DF'] if r['family'] == 'point' else ['P4']
        for m in methods:
            rec = dict(record_type='B1_baseline', input_key=r['input_key'], family=r['family'], size_mm=r['size_mm'], surface=r['surface'], phase_label=r['phase_label'], method=m,
                       image_key=J.get('image_key'), image_npz_sha256=J['npz_sha256'], image_file_matches_record=img_ok)
            if m in ('D0', 'P4'):
                E = K.load_json(K.ROOT / r['eval_record']); npz = K.ROOT / r['eval_record'].replace('.json', '.npz')
                z = K.load_npz(npz) if npz.exists() else {}
                have_arr = [k for k in ('pts', 'accepted', 'valid', 'feats') if k in z]
                src, ok, nd = method_identity_of_record(r['eval_record'], E)
                rec.update(result_source=K.rel(K.ROOT / r['eval_record']), result_complete=bool(E.get('complete')) and len(have_arr) == 4, result_arrays=';'.join(have_arr),
                           method_identity_source=src, method_identity_ok=ok, method_identity_files_checked=nd[0], method_identity_files_differ=nd[1],
                           platform_version=E.get('platform_version'))
                rec['reusable_if_1x1_identical'] = bool(rec['result_complete'] and rec['method_identity_ok'])
                rec['decodes_needed_1x1'] = 0 if rec['reusable_if_1x1_identical'] else 1
            else:
                cache = K.ROOT / 'outputs/surface_revision/s3_code04/cache/C1' / f"{r['input_key']}.npz"
                z = K.load_npz(cache) if cache.exists() else {}
                arr = [k for k in ('DF_pts', 'DF_acc', 'DF_zr', 'DF_ident') if k in z]
                ok_row = r['input_key'] in C4.index
                fz = K.load_json(K.ROOT / 'outputs/surface_revision/s3_code04/formal/frozen.json')['closure']['files']
                n, diff = _files_equal({str(f).replace('\\', '/'): h for f, h in fz.items()})
                rec.update(result_source=(K.rel(cache) if cache.exists() else None), result_complete=bool(ok_row and len(arr) == 4), result_arrays=';'.join(arr),
                           result_missing='camera feature positions, valid / domain masks and per-point evaluator identity of the DF run (not stored by S3-CODE-04)',
                           method_identity_source='S3-CODE-04 frozen closure (pre-run 2026-10-15) + THESIS-REINFORCE-01 replay of 6 DF cases (exact)',
                           method_identity_ok=bool(ok_row and not diff), method_identity_files_checked=n, method_identity_files_differ=len(diff),
                           platform_version=None)
                rec['reusable_if_1x1_identical'] = 'metrics and 3-D points only'
                rec['decodes_needed_1x1'] = 1
            rows.append(rec)
    return pd.DataFrame(rows)


def budget(inv: pd.DataFrame) -> dict:
    n_img = inv.input_key.nunique()
    dec3 = int(len(inv))
    dec1 = int(inv.decodes_needed_1x1.sum())
    return dict(renders_1x1_compat=n_img, renders_3x3=n_img, renders_total_max=2 * n_img, decodes_3x3=dec3, decodes_1x1_baseline=dec1,
                decodes_1x1_detail=inv.groupby('method').decodes_needed_1x1.sum().to_dict(), evaluations_case_level=dec3 + dec1,
                optional_5x5_max_conditions=6, this_round=dict(renders=0, decodes=0, evaluations=0), note='future proposal only - not authorised')


CONTRACT = dict(
    image=['max_abs_pixel_diff', 'n_pixels_changed', 'sum_grey_1x1', 'sum_grey_3x3', 'n_hits_1x1', 'n_hits_3x3', 'n_rays_cast'],
    detection=['n_detected', 'matched_feature_pairs', 'centroid_shift_px_median/p95 (dots, camera px)', 'normal_position_shift_px_median/p95 (lines, camera px)'],
    acceptance=['n_accepted', 'n_valid', 'c', 'w', 'u (registered identity version)'],
    reconstruction=['depth_error_change_mm (point-wise, matched outputs)', 'glob/loc pt RMSE and P95 (defined counts)', 'glob/loc C_tau, C_valid (d 6 mm, tau 0.15 mm)'],
    size_effect=['paired differences r0.5-r1.0 and r1.0-r1.5 per surface and phase, 1x1 vs 3x3, with sign'],
    rules=['report effect size and sign always; 0.005 mm / 0.2 pp are engineering triggers for further explanation, not physical acceptance limits',
           'camera px shifts are not converted with the nominal 0.18 mm/px into depth; depth effects come from the reconstruction',
           '1x1 / 3x3 is a limited stability check, not convergence; sigma-0 results do not prove sigma-60 orderings'],
    obtainable=['uint8 image', 'hit map', '3-D hit points', 'n_rays_cast / n_hits / n_valid_pixels (out_config)', 'pure ray weights (offsets, 1/n^2)'],
    not_obtainable=['float image before clipping / noise / quantisation', 'per-ray irradiance'],
    minimal_interface_suggestion='optional return of the float image before saturation in a derived copy of synthesise_camera_image (for review; not implemented)')
