# Prepare sub-ray cases and check their imaging scale and platform configuration.

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import tr4_common as K

SOURCES = {
    'S1-REAL-01': dict(prefix='outputs/surface_revision/s1_real01/images/', integral='integral_actual', peak='peak'),
    'S12-RESOURCE-01': dict(prefix='outputs/surface_revision/s12_resource01/images/', integral='Q_actual', peak='peak_target'),
}
INTEGRAL_FIELDS = ('integral_actual', 'Q_actual')
ALLOWED_RAYS = (1, 3)
METHODS = dict(point=('D0', 'DF'), line=('P4',))
ARM = dict(D0='D0', P4='P4_C3_cal')
COND_COLS = ('input_key', 'family', 'layout', 'size_mm', 'surface', 'phase_label', 'raster_key', 'image_record', 'eval_record', 'Q_actual', 'peak_f32', 'active_limit')
CHAIN_MODULES = ('m1_tire_surface', 'm2_pattern', 'm3_virtual_camera', 'm4_reconstruction', 'm4_lines', 'surface_eval_r3', 'adp_common', 'adp_binding', 'adp_surf02',
                 'adp_patterns', 's1r_common', 's1r_eval', 's2_common', 'r5_detect', 'rdy_platform', 'rdy_common', 'rdy_decode', 's3c_dev', 's3c_common', 'c2_decode',
                 'ep_detect', 'ep_patterns', 's2p_case', 'l01_common', 'l01_truth', 'tr_a4', 'tr3_gate', 'tr3_b1')


def _paths():
    import tr_common
    for q in ('s3_code01', 's3_code02', 's3_code03', 's345_ready', 'sections_adapter', 'surf02', 's2_prepare', 's12_resource01'):
        p = str(K.SR / q)
        if p not in sys.path:
            sys.path.append(p)


_paths()


def s2():
    import rdy_platform as P
    return P.s2()


def conditions() -> pd.DataFrame:
    P3 = K.read_csv(K.OUT3 / 'plan.csv'); B = P3[P3.record_type == 'B1_baseline']
    C = B.drop_duplicates('input_key')[list(COND_COLS)].sort_values('input_key').reset_index(drop=True)
    P2 = K.read_csv(K.OUT2 / 'plan.csv'); C2 = P2[P2.record_type == 'B1_condition'].set_index('input_key')
    if len(C) != 24 or set(C.input_key) != set(C2.index) or len(C2) != 24:
        raise K.Stop('preflight', f'B1 conditions: {len(C)} in TR03, {len(C2)} in TR02, key sets equal {set(C.input_key) == set(C2.index)}')
    for r in C.to_dict('records'):
        q = C2.loc[r['input_key']]
        for k in ('raster_key', 'image_record', 'eval_record', 'surface', 'phase_label', 'family'):
            if str(q[k]) != str(r[k]):
                raise K.Stop('preflight', f"{r['input_key']}: {k} differs between TR03 ({r[k]}) and TR02 ({q[k]})")
        for k in ('Q_actual', 'peak_f32', 'size_mm'):
            if float(q[k]) != float(r[k]):
                raise K.Stop('preflight', f"{r['input_key']}: {k} differs between TR03 and TR02")
    meth = B.groupby('input_key').method.apply(lambda s: tuple(sorted(s))).to_dict()
    for r in C.to_dict('records'):
        if meth[r['input_key']] != tuple(sorted(METHODS[r['family']])):
            raise K.Stop('preflight', f"{r['input_key']}: registered methods {meth[r['input_key']]}")
    C['methods'] = [';'.join(METHODS[f]) for f in C.family]
    return C


def manifest_digest(C: pd.DataFrame) -> str:
    return K.digest(C[list(COND_COLS) + ['methods']].to_dict('records'))


def checkpoint_order(C: pd.DataFrame) -> list[str]:

    keys = sorted(C.input_key)
    d = sorted(C[(C.family == 'point') & (C.size_mm == 0.5)].input_key)[0]; l = sorted(C[C.family == 'line'].input_key)[0]
    return [d, l] + [k for k in keys if k not in (d, l)]


def source_of(image_record: str) -> str:
    p = str(image_record).replace('\\', '/')
    hit = [s for s, d in SOURCES.items() if p.startswith(d['prefix'])]
    if len(hit) != 1:
        raise K.Stop('preflight', f'image record {p}: unknown source version')
    return hit[0]


def stored_integral(J: dict, src: str) -> tuple[float, str]:
    f = SOURCES[src]['integral']
    vals = {k: float(J[k]) for k in INTEGRAL_FIELDS if J.get(k) is not None}
    if not vals:
        raise K.Stop('preflight', f'{src} record without any integral field ({INTEGRAL_FIELDS})')
    if f not in vals:
        raise K.Stop('preflight', f'{src} record lacks its registered integral field {f!r} (present: {sorted(vals)})')
    if len(vals) == 2:
        a, b = vals['integral_actual'], vals['Q_actual']
        if abs(a - b) > 1e-9 * max(1.0, abs(a)):
            raise K.Stop('preflight', f'integral_actual {a} and Q_actual {b} disagree')
    return vals[f], f


def validate_rays(n) -> int:
    import tr3_b1 as B3
    n = B3.validate_rays(n)
    if n not in ALLOWED_RAYS:
        raise ValueError(f'rays_per_pixel {n} not registered for this round {ALLOWED_RAYS}')
    return n


def scaling_check(key: str, binary, pattern, peak: float, integ_stored: float) -> tuple[int, float]:

    lit = int((np.asarray(binary) > 0).sum()); f32 = float(np.float32(peak))
    if pattern.dtype != np.float32 or not np.array_equal(pattern > 0, np.asarray(binary) > 0) or float(pattern.max()) != f32:
        raise K.Stop('preflight', f'{key}: pattern scaling check failed (max {float(pattern.max())} vs float32(peak) {f32})')
    integ_now = float(pattern.sum(dtype=np.float64)); integ_formula = lit * f32
    if abs(integ_now - integ_stored) > 1e-6 * max(1.0, integ_stored) or abs(integ_formula - integ_stored) > 1e-6 * max(1.0, integ_stored):
        raise K.Stop('preflight', f'{key}: pattern integral {integ_now} / lit x f32 peak {integ_formula} vs stored {integ_stored} (double scaling?)')
    return lit, integ_now


def prepare(cond: dict, rays, record: dict | None = None) -> dict:

    import adp_surf02 as SA
    import s1r_common as S1K
    n = validate_rays(rays); S = s2()
    J = record if record is not None else K.load_json(K.ROOT / cond['image_record'])
    src = source_of(cond['image_record']); integ, ifield = stored_integral(J, src)
    pfield = SOURCES[src]['peak']
    if J.get(pfield) is None or J.get('peak_f32') is None:
        raise K.Stop('preflight', f"{cond['input_key']}: {src} record lacks {pfield!r} / 'peak_f32'")
    peak = float(J[pfield]); peak_f32 = float(J['peak_f32'])
    if float(np.float32(peak)) != peak_f32 or abs(peak_f32 - float(cond['peak_f32'])) > 0:
        raise K.Stop('preflight', f"{cond['input_key']}: float32({pfield}) {float(np.float32(peak))} / record peak_f32 {peak_f32} / plan {cond['peak_f32']}")
    if abs(integ - float(cond['Q_actual'])) > 1e-9 * max(1.0, integ):
        raise K.Stop('preflight', f"{cond['input_key']}: stored integral {integ} differs from the registered plan value {cond['Q_actual']}")
    if J.get('raster_key') != cond['raster_key'] or float(J.get('sigma', -1)) != 0.0 or int(J.get('m3_seed_passed', -1)) != 0 or J.get('noise_seed') is not None:
        raise K.Stop('preflight', f"{cond['input_key']}: raster / sigma / seed of the record differ from the sigma-0 condition")
    if SA.ID_MAP[cond['surface']] != J['platform_surface']:
        raise K.Stop('preflight', f"{cond['input_key']}: surface {cond['surface']} -> {SA.ID_MAP[cond['surface']]} != record {J['platform_surface']}")
    if J.get('platform_version') != S.VERSION:
        raise K.Stop('preflight', f"{cond['input_key']}: record platform {J.get('platform_version')} != {S.VERSION}")
    R = S1K.load_raster(cond['raster_key'])
    binary = np.asarray(R['binary'], np.float32)
    pattern = S1K.pattern_of(binary, peak)
    lit, integ_now = scaling_check(cond['input_key'], binary, pattern, peak, integ)
    cfg = dict(S.m3_cfg(0.0, 0))
    if 'rays_per_pixel' in cfg:
        raise K.Stop('preflight', 'platform m3_cfg already sets rays_per_pixel')
    if abs(float(cfg.get('psf_sigma_px', -1)) - float(J['psf_px'])) > 1e-12:
        raise K.Stop('preflight', f"platform PSF {cfg.get('psf_sigma_px')} != stored {J['psf_px']}")
    cfg['rays_per_pixel'] = n
    size = float(cond['size_mm'])
    ctx = dict(binary=binary, centres=R['centres'], pattern=pattern, peak=peak, size_mm=size, meta_full=R['meta_full'], row=dict(radius_mm=size))
    stored = (K.ROOT / cond['image_record']).with_suffix('.npz')
    import m3_virtual_camera as M3
    fwd = dict(sid=J['platform_surface'], raster_key=cond['raster_key'], pattern=K.content_sha(pattern), cfg=cfg, platform=S.VERSION,
               m3_file=K.sha(M3.__file__))
    return dict(input_key=cond['input_key'], family=cond['family'], size_mm=size, surface=cond['surface'], sid=J['platform_surface'], source=src,
                integral_field=ifield, integral_stored=integ, integral_now=integ_now, peak_field=pfield, peak=peak, peak_f32=peak_f32, lit_px=lit,
                pattern=pattern, pattern_sha=K.content_sha(pattern), cfg=cfg, rays=n, ctx=ctx, stored_image=stored, stored_npz_sha=J['npz_sha256'],
                forward_id=K.digest(fwd), methods=METHODS[cond['family']])


def prepare_all(C: pd.DataFrame) -> dict:

    rows = []
    for c in C.to_dict('records'):
        p1, p3 = prepare(c, 1), prepare(c, 3)
        a = {k: v for k, v in p1['cfg'].items() if k != 'rays_per_pixel'}; b = {k: v for k, v in p3['cfg'].items() if k != 'rays_per_pixel'}
        only_rays = K.canon(a) == K.canon(b) and p1['cfg']['rays_per_pixel'] == 1 and p3['cfg']['rays_per_pixel'] == 3
        same_pattern = p1['pattern_sha'] == p3['pattern_sha'] and np.array_equal(p1['pattern'], p3['pattern'])
        rows.append(dict(input_key=c['input_key'], family=c['family'], size_mm=c['size_mm'], surface=c['surface'], phase_label=c['phase_label'], source=p1['source'],
                         integral_field=p1['integral_field'], integral_stored=p1['integral_stored'], integral_now=p1['integral_now'], peak_field=p1['peak_field'],
                         peak=p1['peak'], peak_f32=p1['peak_f32'], lit_px=p1['lit_px'], pattern_sha=p1['pattern_sha'], only_rays_differ=only_rays,
                         same_pattern=same_pattern, forward_id_1=p1['forward_id'], forward_id_3=p3['forward_id'], stored_npz_sha=p1['stored_npz_sha']))
    T = pd.DataFrame(rows)
    ok = bool(len(T) == 24 and T.only_rays_differ.all() and T.same_pattern.all() and (T.forward_id_1 != T.forward_id_3).all())
    return dict(passed=ok, n=len(T), by_source=T.groupby(['source', 'integral_field']).size().reset_index(name='n').to_dict('records'), table=T)


def input_files(C: pd.DataFrame) -> dict:
    R = K.ROOT / 'outputs' / 'surface_revision'
    f = {'tr03_plan': K.OUT3 / 'plan.csv', 'tr03_run': K.OUT3 / 'run.json', 'tr02_plan': K.OUT2 / 'plan.csv', 'tr01_run': K.OUT1 / 'run.json',
         'task_book': K.TASK, 'review_tr03': K.REVIEW3, 's1_dependencies': R / 's1_real01/config/s1_dependencies.json', 's12_protocol': R / 's12_resource01/protocol.json',
         'c4_frozen': R / 's3_code04/formal/frozen.json', 'c4_dots_cases': R / 's3_code04/results/C1_dots_cases.csv',
         'surf02_release': R / 'surf02/release/release_surf02.json', 'surf02_dependencies': R / 'surf02/release/dependencies_surf02_v1_1.json',
         'binding': R / 'sections_adapter/binding.json', 'tr01_identity_geometry': K.OUT1 / 'identity_geometry.csv'}
    for p in sorted((K.OUT1 / 'shards').glob('*.npz')):
        f[f'tr01_shard:{p.name}'] = p
    import s1r_common as S1K
    for c in C.to_dict('records'):
        ik = c['input_key']; ir = K.ROOT / c['image_record']; er = K.ROOT / c['eval_record']
        f[f'{ik}:image_json'] = ir; f[f'{ik}:image_npz'] = ir.with_suffix('.npz')
        f[f'{ik}:raster_json'] = S1K.RASTER_DIR / f"{c['raster_key']}.json"; f[f'{ik}:raster_npz'] = S1K.RASTER_DIR / f"{c['raster_key']}.npz"
        f[f'{ik}:old_eval_json'] = er; f[f'{ik}:old_eval_npz'] = er.with_suffix('.npz')
        if c['family'] == 'point':
            f[f'{ik}:old_DF_cache'] = R / 's3_code04/cache/C1' / f'{ik}.npz'
    return f


def hash_inputs(files: dict) -> dict:
    return {k: dict(path=K.rel(p), sha256=(K.sha(p) if p.exists() else None)) for k, p in files.items()}


def recorded_input_checks(C: pd.DataFrame, H: dict) -> dict:

    bad = []
    r3 = K.load_json(K.OUT3 / 'run.json'); out3 = (r3.get('final') or {}).get('outputs') or {}
    if H['tr03_plan']['sha256'] != out3.get('plan.csv'):
        bad.append('TR03 plan.csv differs from the TR03 final record')
    r2 = K.load_json(K.OUT2 / 'run.json'); out2 = (r2.get('final') or {}).get('outputs') or {}
    if H['tr02_plan']['sha256'] != out2.get('plan.csv'):
        bad.append('TR02 plan.csv differs from the TR02 final record')
    f1 = (K.load_json(K.OUT1 / 'run.json').get('final') or {}).get('shards') or {}
    shard_bad = [k for k in H if k.startswith('tr01_shard:') and H[k]['sha256'] != f1.get(k.split(':', 1)[1])]
    if shard_bad or len([k for k in H if k.startswith('tr01_shard:')]) != len(f1):
        bad.append(f'TR01 DF shards differ from the TR01 final record: {shard_bad[:3]}')
    for c in C.to_dict('records'):
        ik = c['input_key']; J = K.load_json(K.ROOT / c['image_record']); E = K.load_json(K.ROOT / c['eval_record'])
        if H[f'{ik}:image_npz']['sha256'] != J['npz_sha256']:
            bad.append(f'{ik}: image npz differs from its record')
        if H[f'{ik}:old_eval_npz']['sha256'] != E['npz_sha256']:
            bad.append(f'{ik}: old evaluation npz differs from its record')
        if not E.get('complete') or not J.get('complete'):
            bad.append(f'{ik}: record not complete')
    P2 = K.read_csv(K.OUT2 / 'plan.csv').set_index('input_key')
    for c in C.to_dict('records'):
        if str(P2.loc[c['input_key']]['image_npz_sha256']) != H[f"{c['input_key']}:image_npz"]['sha256']:
            bad.append(f"{c['input_key']}: image npz differs from the TR02 registration")
    missing = [k for k, v in H.items() if v['sha256'] is None]
    return dict(passed=not bad and not missing, problems=bad, missing=missing, n_files=len(H))


def platform_checks() -> dict:

    import adp_surf02 as SA
    import rdy_platform as P
    rel_state = P.release_state()
    bd = P.bound()
    S = s2()
    surf = P.surfaces()
    dg = P.dom_global()
    import tr3_gate as G3
    base = K.load_json(K.OUT3 / 'run.json')['baseline_posthoc']
    dom = {s: G3.domain_content(s) for s in ('B_curv', 'M10_curv', 'M24_curv')} | dict(glob=G3.domain_content(None))
    dom_ok = all(dom[k] == base['domains_content'][k] for k in dom)
    out = dict(release=rel_state, released=bool(rel_state['released']), drift=rel_state['drift'], version=bd.version, window=list(bd.window()),
               methods={a: bd.method(m) for a, m in ARM.items()}, surface_ids={s: bd.sid(s) for s in ('B_curv', 'M10_curv', 'M24_curv')},
               id_map_matches=all(bd.sid(s) == SA.ID_MAP[s] for s in ('B_curv', 'M10_curv', 'M24_curv')), surfaces=surf,
               global_domain=dict(n_nodes=int(dg['rec']['n_nodes']), hash=dg['hash']), domains_content=dom, domains_equal_tr03_baseline=dom_ok,
               constants=dict(c_ref=float(S.C_REF), gross=float(S.GROSS_MM), roi=float(S.ROI_R_MM), z0=float(S.Z0_MM), z1=float(S.Z1_MM)))
    out['passed'] = bool(out['released'] and not out['drift'] and out['version'] == S.VERSION and tuple(out['window']) == (-16.0, 4.0) and out['id_map_matches'] and dom_ok
                         and out['methods'] == dict(D0='D0', P4='P4_C3_cal'))
    return out


def _manifest_files(path, key='files') -> dict:
    M = K.load_json(path)
    files = M[key] if key == 'files' else M['closure']['files']
    return {str(f).replace('\\', '/'): h for f, h in files.items()}


def _current(files: dict) -> tuple[dict, list]:
    cur = {}; diff = []
    for f, h in files.items():
        q = K.ROOT / f; c = K.sha(q) if q.exists() else None; cur[f] = c
        if c != h:
            diff.append(f)
    return cur, diff


def method_bindings(C: pd.DataFrame) -> dict:
    import s1r_common as S1K
    import tr3_b1 as B3
    import tr3_gate as G3
    S = s2(); R = K.ROOT / 'outputs' / 'surface_revision'
    s1 = _manifest_files(R / 's1_real01/config/s1_dependencies.json'); s1_now, s1_diff = _current(s1)
    c4 = _manifest_files(R / 's3_code04/formal/frozen.json', 'closure'); c4_now, c4_diff = _current(c4)
    base = K.load_json(K.OUT3 / 'run.json')['baseline_posthoc']
    peaks = sorted({float(K.load_json(K.ROOT / r['image_record'])[SOURCES[source_of(r['image_record'])]['peak']]) for r in C[C.family == 'line'].to_dict('records')})
    dcfg_df = G3.live_decode_cfg()
    thr = G3.live_thresholds()
    dom = {s: G3.domain_content(s) for s in ('B_curv', 'M10_curv', 'M24_curv')} | dict(glob=G3.domain_content(None))
    scan = line_registration_scan(C)
    B = dict(
        D0=dict(entry='adp_surf02.decode("D0") through adp_binding (s1r_common.ARM point)', cfg=S.dot_cfg(), closure='S1-REAL-01 dependency manifest (39 files, pre-run 2026-10-12)',
                closure_files=s1_now, closure_differs=s1_diff),
        P4=dict(entry='adp_surf02.decode("P4_C3_cal") through adp_binding (s1r_common.ARM line); cfg = s2.line_cfg(actual decode peak) + s2.P4',
                cfg={str(p): S.line_cfg(p) for p in peaks} | dict(P4=S.P4), closure='S1-REAL-01 dependency manifest (39 files)', closure_files=s1_now, closure_differs=s1_diff),
        DF=dict(entry='s3c_dev.dot_decode(img, binary, 1, [radius], "DF", "binding") + s3c_dev.dot_dec(D, "single")', cfg=dcfg_df,
                cfg_equals_tr03_baseline=(G3.digest(dcfg_df) == base['decode_cfg_content']), closure='S3-CODE-04 frozen closure (97 files, pre-run 2026-10-15)',
                closure_files=c4_now, closure_differs=c4_diff),
        EVAL=dict(entry='s1r_eval.evaluate_case(sid, family, dec, ctx, mesh, rdy_platform.dom_local(surface), rdy_platform.dom_global(), gauss=False)',
                  thresholds=thr, thresholds_equal_tr03_baseline=(thr == base['thresholds']), domains=dom, domains_equal_tr03_baseline=all(dom[k] == base['domains_content'][k] for k in dom),
                  identity_rule='evaluator identity (s1r_eval); lines: complete registration rule = evaluator identity + THESIS-REINFORCE-01 A4 extension for lit but '
                                'unregistered lines (tr_a4.reidentify), applied identically to 1x1 and 3x3', line_registration_scan=scan,
                  closure='S1-REAL-01 dependency manifest (39 files)', closure_files=s1_now, closure_differs=s1_diff))
    for m, b in B.items():
        b['fingerprint'] = K.digest(dict(method=m, entry=b['entry'], cfg=b.get('cfg'), thresholds=b.get('thresholds'), domains=b.get('domains'),
                                         identity_rule=b.get('identity_rule'), code=b['closure_files']))

    old = []
    for c in C.to_dict('records'):
        E = K.load_json(K.ROOT / c['eval_record'])
        src, ok, nd = B3.method_identity_of_record(c['eval_record'], E)
        m = 'D0' if c['family'] == 'point' else 'P4'
        arm_ok = (E.get('arm') == ARM[m]) if 'arm' in E else (K.load_json(K.ROOT / c['eval_record'].replace('/eval/', '/raw/')).get('method') == ARM[m]
                                                                if (K.ROOT / c['eval_record'].replace('/eval/', '/raw/')).exists() else None)
        peak_ok = True
        if m == 'P4':
            eff = (E.get('effective') or {}).get('pattern_peak_passed')
            if eff is None and (K.ROOT / c['eval_record'].replace('/eval/', '/raw/')).exists():
                eff = (K.load_json(K.ROOT / c['eval_record'].replace('/eval/', '/raw/')).get('effective') or {}).get('pattern_peak_passed')
            peak_ok = eff is not None and float(eff) == float(K.load_json(K.ROOT / c['image_record'])[SOURCES[source_of(c['image_record'])]['peak']])
        old.append(dict(input_key=c['input_key'], method=m, record=c['eval_record'], manifest=src, manifest_ok=ok, files_checked=nd[0], files_differ=nd[1], arm_ok=arm_ok, decode_peak_ok=peak_ok))
    O = pd.DataFrame(old)
    passed = bool(not s1_diff and not c4_diff and B['DF']['cfg_equals_tr03_baseline'] and B['EVAL']['thresholds_equal_tr03_baseline'] and B['EVAL']['domains_equal_tr03_baseline']
                  and O.manifest_ok.all() and (O.arm_ok != False).all() and O.decode_peak_ok.all() and scan['passed']
                  and len({b['fingerprint'] for b in B.values()}) == 4)
    slim = {m: {k: v for k, v in b.items() if k != 'closure_files'} | dict(n_closure_files=len(b['closure_files'])) for m, b in B.items()}
    return dict(passed=passed, methods=slim, old_results=O.to_dict('records'), s1_manifest_differs=s1_diff, c4_closure_differs=c4_diff)


def line_registration_scan(C: pd.DataFrame) -> dict:

    import s1r_common as S1K
    import tr_a4 as A4
    G = K.read_csv(K.OUT1 / 'identity_geometry.csv'); G = G[G.record_type == 'A4_raster'].set_index('raster_key')
    rows = []
    for rk in sorted(set(C[C.family == 'line'].raster_key)):
        R = S1K.load_raster(rk); s = A4.scan_raster(R['binary'], R['meta_full']['olines'])
        stored = json.loads(G.loc[rk]['extra']) if rk in G.index else None
        rows.append(dict(raster_key=rk, extra=s['extra'], lit_px_extra=s['lit_px_extra'], registered_without_lit=s['registered_without_lit'], tr01_extra=stored,
                         equal_to_tr01=(stored == s['extra'])))
    return dict(passed=all(r['equal_to_tr01'] for r in rows), rasters=rows, a4_applies=any(r['extra'] for r in rows))


def df_overlap(C: pd.DataFrame) -> dict:
    import tr_a3 as A3
    keys = (K.load_json(K.OUT1 / 'run.json')['plan']['a3_keys'])
    new = {k['input_key'] for k in keys if k['reuse_class'] == 'new_DF'}; reuse = {k['input_key'] for k in keys if k['reuse_class'] == 'reuse_DF'}
    A = A3.plan().set_index('input_key')
    img_new = {K.load_json(K.ROOT / A.loc[k]['image_record'])['npz_sha256'] for k in new}
    b1 = C[C.family == 'point']; ks = set(b1.input_key)
    img_b1 = {K.load_json(K.ROOT / r['image_record'])['npz_sha256'] for r in b1.to_dict('records')}
    inter_key = sorted(ks & new); inter_img = sorted(img_b1 & img_new)
    return dict(n_new_DF_stored=len(new), n_reuse_DF_stored=len(reuse), n_b1_dot_inputs=len(ks), intersection_input_keys=inter_key, intersection_image_sha=inter_img,
                b1_in_reuse_DF=len(ks & reuse), b1_not_in_a3_list=len(ks - reuse - new), passed=(len(new) == 513 and not inter_key and not inter_img and len(ks) == 18),
                note='all 18 B1 DF baseline inputs have S3-CODE-04 C1 cached DF results (metrics + 3-D points, no feature arrays); 9 (phase 0) were THESIS-REINFORCE-01 '
                     'reuse_DF entries, 9 (phase 1) were not in its A3 list; none is one of the 513 THESIS-REINFORCE-01 new DF decodes')


UNFROZEN = {K.rel(K.HERE / f) for f in K.LAYOUT_FILES + K.DRILL_FILES}


def code_closure() -> dict:
    import importlib
    import s1r_common as S1K
    for m in tuple(S1K.LAZY_MODULES) + CHAIN_MODULES:
        importlib.import_module(m)
    root = K.ROOT.resolve(); files = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if not f or not str(f).endswith('.py'):
            continue
        q = __import__('pathlib').Path(f).resolve()
        if root in q.parents and K.rel(q) not in UNFROZEN:
            files[K.rel(q)] = K.sha(q)
    return dict(sorted(files.items()))


def loaded_project_files() -> set:
    root = K.ROOT.resolve(); out = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and str(f).endswith('.py'):
            q = __import__('pathlib').Path(f).resolve()
            if root in q.parents and K.rel(q) not in UNFROZEN:
                out.add(K.rel(q))
    return out


def closure_check(frozen: dict) -> dict:

    changed = [f for f, h in frozen.items() if not (K.ROOT / f).exists() or K.sha(K.ROOT / f) != h]
    extra = sorted(loaded_project_files() - set(frozen))
    return dict(passed=not changed and not extra, changed=changed, not_frozen_but_loaded=extra, n_frozen=len(frozen))
