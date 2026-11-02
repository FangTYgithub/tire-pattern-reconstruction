# Build and compare rendering, decoding and evaluation fingerprints.

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

import tr3_common as K

DECODE_MODULES = ('s3c_dev', 'r5_detect', 'ep_detect', 'ep_patterns', 'rdy_decode', 'm4_reconstruction', 's2_common')
EVAL_MODULES = ('s1r_eval', 's1r_common', 's2_common', 'surface_eval_r3', 'rdy_platform', 'rdy_common', 'sn_lib')
ARR = ('pts', 'accepted', 'valid', 'zr', 'in_global', 'in_local', 'point_correct', 'point_wrong', 'point_unconf', 'point_feature', 'feat_truth_dot', 'feat_confirmable', 'exp_recovered')
PER_POINT = ('accepted', 'valid', 'zr', 'in_global', 'in_local', 'point_correct', 'point_wrong', 'point_unconf', 'point_feature')
REQ_METRICS = ('state', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_cov_interp', 'loc_cov_interp',
               'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'loc_surf_rmse', 'id_correct', 'id_wrong', 'id_unconf')
IDENTITY_VERSION = 'evaluator geometric identity s1r_eval (THESIS-REINFORCE-01 A3 scope; A4 correction does not apply to dot records)'


def _canon(obj) -> str:
    return json.dumps(K.clean(obj), sort_keys=True, ensure_ascii=False, default=str)


def digest(parts: dict) -> str:
    return hashlib.sha256(_canon(parts).encode('utf-8')).hexdigest()


def content_sha(data) -> str:
    if isinstance(data, np.ndarray):
        h = hashlib.sha256(); h.update(str(data.dtype).encode()); h.update(str(data.shape).encode()); h.update(np.ascontiguousarray(data).tobytes()); return h.hexdigest()
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def forward_fp(image_sha: str | None, gen: dict) -> str | None:
    if not image_sha or not gen or any(gen.get(k) in (None, '') for k in ('surface', 'raster_key', 'peak_f32', 'psf_px', 'platform_version')):
        return None
    return digest(dict(layer='forward', image_sha=image_sha, gen={k: gen.get(k) for k in ('surface', 'raster_key', 'peak_f32', 'sigma', 'm3_seed_passed', 'psf_px', 'platform_version')}))


def decode_fp(method: str, code: dict, cfg: dict | None) -> str | None:
    if not method or not code or any(not v for v in code.values()) or not cfg:
        return None
    return digest(dict(layer='decode', method=method, code=code, cfg=cfg))


def eval_fp(code: dict, domains: dict, thresholds: dict, identity_version: str) -> str | None:
    if not code or any(not v for v in code.values()) or not domains or any(not v for v in domains.values()) or not thresholds or not identity_version:
        return None
    return digest(dict(layer='eval', code=code, domains=domains, thresholds=thresholds, identity=identity_version))


def _module_files(names) -> dict:
    import importlib
    import tr_common
    for q in ('s3_code01', 's3_code03', 's345_ready', 'sections_adapter', 'surf02'):
        p = str(K.SR / q)
        import sys
        if p not in sys.path:
            sys.path.append(p)
    out = {}
    for n in names:
        m = importlib.import_module(n); out[n] = K.rel(m.__file__)
    return out


_CACHE: dict = {}


def _cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def live_decode_cfg() -> dict:
    import r5_detect as RD
    import s3c_dev as DV
    import s3c_common as SC
    import rdy_platform as P
    return dict(dot_cfg=P.s2().dot_cfg(), floor=dict(DV.FLOOR), nominal_mm_per_px=float(SC.NOMINAL_MM_PER_PX), r5_cfg=dict(RD.CFG), path='binding', rule='DF')


def live_thresholds() -> dict:
    import s1r_common as S1K
    import rdy_platform as P
    return dict(d_support=float(S1K.D_SUPPORT), tau=float(S1K.TAU), gross=float(P.s2().GROSS_MM))


def domain_content(sid_surface: str | None = None, override=None) -> str:

    import rdy_platform as P
    if override is not None:
        d = override
    elif sid_surface is None:
        d = P.dom_global()
    else:
        d = P.dom_local(sid_surface)
    keys = sorted(k for k in d if isinstance(d[k], np.ndarray) and d[k].ndim >= 1 and d[k].dtype.kind in 'fiub')
    return digest({k: content_sha(np.asarray(d[k])) for k in keys})


def live_code(names) -> dict:
    files = _module_files(names)
    return {f: K.sha(K.ROOT / f) for f in files.values()}


def target_identity(u: dict, overrides: dict | None = None) -> dict:

    o = overrides or {}
    img = o.get('image_array')
    if img is None:
        img = K.load_npz(K.ROOT / u['image_record'].replace('.json', '.npz'))['image']
    gen = dict(surface=u['platform_surface'], raster_key=u['raster_key'], peak_f32=float(np.float32(u['peak_f32'])), sigma=float(u['sigma']),
               m3_seed_passed=(int(u['noise_seed']) if float(u['sigma']) > 0 else 0), psf_px=float(u.get('psf_px', 1.0)), platform_version=o.get('platform_version', u.get('platform_version')))
    dcode = o.get('decode_code') or _cached('dcode', lambda: live_code(DECODE_MODULES))
    dcfg = o.get('decode_cfg') or _cached('dcfg', live_decode_cfg)
    ecode = o.get('eval_code') or _cached('ecode', lambda: live_code(EVAL_MODULES))
    dom = o.get('domains') or dict(glob=_cached('dom_glob', lambda: domain_content(None)), loc=_cached('dom_' + u['surface'], lambda: domain_content(u['surface'])))
    thr = o.get('thresholds') or _cached('thr', live_thresholds)
    return dict(forward=forward_fp(content_sha(np.asarray(img)), gen), decode=decode_fp('DF', dcode, dict(content=digest(dcfg))),
                eval=eval_fp(ecode, dom, thr, IDENTITY_VERSION))


def persisted_code_hashes() -> dict:

    out = {}
    fz = K.load_json(K.ROOT / 'outputs/surface_revision/s3_code04/formal/frozen.json')['closure']['files']
    for f, h in fz.items():
        out.setdefault(str(f).replace('\\', '/'), (h, 'S3-CODE-04 frozen closure (pre-run, 2026-10-15)'))
    s1 = K.load_json(K.ROOT / 'outputs/surface_revision/s1_real01/config/s1_dependencies.json')['files']
    for f, h in s1.items():
        out.setdefault(str(f).replace('\\', '/'), (h, 'S1-REAL-01 dependency manifest (pre-run, 2026-10-12)'))
    b2 = (K.load_json(K.OUT2 / 'run.json').get('bind') or {}).get('code', {}).get('hashes') or {}
    for f, h in b2.items():
        out.setdefault(f, (h, 'THESIS-REINFORCE-02 binding (post-hoc, 2026-10-18 17:25)'))
    return out


def write_baseline() -> dict:

    r = K.run_load()
    if r.get('baseline_posthoc'):
        return r['baseline_posthoc']
    b = dict(t=K.now(), kind='POST-HOC baseline (no earlier persisted fingerprint exists for these contents); valid as a reuse baseline from now on, '
                           'NOT a record of the state at the original execution', decode_cfg_content=digest(live_decode_cfg()), thresholds=live_thresholds(),
             domains_content={s: domain_content(s) for s in ('B_curv', 'M10_curv', 'M24_curv')} | dict(glob=domain_content(None)))
    K.run_put('baseline_posthoc', b)
    return b


def history_identity(u: dict) -> dict:

    J = K.load_json(K.ROOT / u['image_record'])
    reg = _cached('reg', lambda: K.load_json(K.OUT1 / 'run.json')['plan']['inputs_sha256'])
    npz_rel = K.rel(K.ROOT / u['image_record'].replace('.json', '.npz'))
    img_sha_record = J.get('npz_sha256'); img_sha_reg = reg.get(npz_rel)

    gen = dict(surface=J.get('platform_surface'), raster_key=J.get('raster_key'), peak_f32=(float(np.float32(J['peak_f32'])) if J.get('peak_f32') is not None else None),
               sigma=J.get('sigma'), m3_seed_passed=J.get('m3_seed_passed'), psf_px=J.get('psf_px'), platform_version=J.get('platform_version'))
    base = _cached('base', lambda: K.run_load().get('baseline_posthoc') or {})
    pc = _cached('persisted', persisted_code_hashes)
    files_d = _cached('files_d', lambda: _module_files(DECODE_MODULES)); files_e = _cached('files_e', lambda: _module_files(EVAL_MODULES))
    dcode = {f: (pc.get(f) or (None, None))[0] for f in files_d.values()}
    ecode = {f: (pc.get(f) or (None, None))[0] for f in files_e.values()}
    import rdy_common as RC
    dom = dict(glob=base.get('domains_content', {}).get('glob'), loc=base.get('domains_content', {}).get(u['surface']))
    out = dict(forward_link=dict(record_npz_sha=img_sha_record, preregistered_npz_sha=img_sha_reg, consistent=bool(img_sha_record and img_sha_record == img_sha_reg)),
               provenance=dict(decode={f: (pc.get(f) or (None, 'none'))[1] for f in files_d.values()}, eval={f: (pc.get(f) or (None, 'none'))[1] for f in files_e.values()},
                               decode_cfg='post-hoc baseline (this round)', domains='post-hoc baseline (this round); global platform hash registered ' + str(RC.GLOBAL_DOMAIN_HASH)[:12]))


    if img_sha_record and img_sha_record == img_sha_reg and K.sha(K.ROOT / npz_rel) == img_sha_record:
        img_content = content_sha(np.asarray(K.load_npz(K.ROOT / npz_rel)['image']))
    else:
        img_content = None
    out['forward'] = forward_fp(img_content, gen)
    out['decode'] = decode_fp('DF', dcode, dict(content=base.get('decode_cfg_content')) if base.get('decode_cfg_content') else None)
    out['eval'] = eval_fp(ecode, dom, base.get('thresholds'), IDENTITY_VERSION)
    return out


class Defect(Exception):
    pass


def _undef(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def check_point_cloud(a: dict) -> dict:
    miss = [k for k in ARR if k not in a]
    if miss:
        raise Defect(f'arrays missing: {miss}')
    n = len(a['pts'])
    if np.asarray(a['pts']).ndim != 2 or np.asarray(a['pts']).shape[1] != 3:
        raise Defect('pts not (n, 3)')
    bad = [k for k in PER_POINT if len(a[k]) != n]
    if bad:
        raise Defect(f'per-point arrays of wrong length: {bad}')
    acc, val = np.asarray(a['accepted'], bool), np.asarray(a['valid'], bool); ig, il = np.asarray(a['in_global'], bool), np.asarray(a['in_local'], bool)
    if (val & ~acc).any():
        raise Defect('valid outputs that are not accepted')
    if not np.isfinite(np.asarray(a['pts'], float)[val]).all():
        raise Defect('valid outputs with non-finite coordinates')
    if (ig & ~val).any() or (il & ~val).any():
        raise Defect('domain flags outside the valid outputs')
    c, w, u = (np.asarray(a[k], bool) for k in ('point_correct', 'point_wrong', 'point_unconf'))
    s = c.astype(int) + w + u
    if (s[acc] != 1).any():
        raise Defect(f'accepted outputs whose c / w / u is not exactly one ({int((s[acc] != 1).sum())})')
    if (s[~acc] != 0).any():
        raise Defect('identity flags on non-accepted outputs')
    return dict(n=n, n_accepted=int(acc.sum()), n_valid=int(val.sum()), n_glob=int(ig.sum()), n_loc=int(il.sum()), c=int(c.sum()), w=int(w.sum()), u=int(u.sum()))


def check_metrics(m: dict, pc: dict) -> dict:
    if not isinstance(m, dict):
        raise Defect('metrics record missing')
    miss = [k for k in REQ_METRICS if k not in m]
    if miss:
        raise Defect(f'metric fields missing: {miss}')
    for k, v in (('n_accepted', pc['n_accepted']), ('n_valid', pc['n_valid']), ('glob_n_valid', pc['n_glob']), ('loc_n_valid', pc['n_loc']),
                 ('id_correct', pc['c']), ('id_wrong', pc['w']), ('id_unconf', pc['u'])):
        if m[k] is None or int(m[k]) != v:
            raise Defect(f'{k} = {m[k]} differs from the arrays ({v})')
    if m['state'] not in ('ok', 'no_output'):
        raise Defect(f'illegal state {m["state"]}')
    if (m['state'] == 'no_output') != (pc['n_valid'] == 0):
        raise Defect(f"state {m['state']} with {pc['n_valid']} valid outputs (registered definition: no_output <=> n_valid == 0)")
    for dn, nn in (('glob', pc['n_glob']), ('loc', pc['n_loc'])):
        for e in (f'{dn}_pt_rmse', f'{dn}_pt_p95'):
            v = m[e]
            if nn > 0 and (_undef(v) or not math.isfinite(float(v))):
                raise Defect(f'{e} undefined although {nn} valid outputs lie in the domain')
            if nn == 0 and not _undef(v):
                raise Defect(f'{e} = {v} defined without any valid output in the domain')
        ci, cv, ct = m[f'{dn}_cov_interp'], m[f'{dn}_Cvalid'], m[f'{dn}_Ctau']
        for nm, v in ((f'{dn}_cov_interp', ci), (f'{dn}_Cvalid', cv), (f'{dn}_Ctau', ct)):
            if _undef(v) or not (0.0 <= float(v) <= 1.0):
                raise Defect(f'{nm} = {v} outside [0, 1]')
        if not (float(ct) <= float(cv) + 1e-12 <= float(ci) + 2e-12):
            raise Defect(f'{dn}: C_tau {ct} <= C_valid {cv} <= C_int {ci} violated')
        sv = m[f'{dn}_surf_rmse']
        if float(ci) > 0 and _undef(sv):
            raise Defect(f'{dn}_surf_rmse undefined with interpolation coverage {ci}')
        if float(ci) == 0 and not _undef(sv):
            raise Defect(f'{dn}_surf_rmse = {sv} defined without interpolation')
    if m['state'] == 'no_output' and (float(m['glob_Ctau']) != 0 or float(m['glob_Cvalid']) != 0 or float(m['loc_Ctau']) != 0 or float(m['loc_Cvalid']) != 0):
        raise Defect('no_output with coverage > 0')
    return dict(state=m['state'], legal_no_output=m['state'] == 'no_output', domain_without_valid=[dn for dn, nn in (('glob', pc['n_glob']), ('loc', pc['n_loc'])) if nn == 0])


def decide(have: dict, want: dict, arrays: dict | None, metrics: dict | None) -> dict:

    def conf(layer):
        h, w = (have or {}).get(layer), (want or {}).get(layer)
        if not h or not w:
            return 'unconfirmed'
        return 'equal' if h == w else 'changed'
    st = {l: conf(l) for l in ('forward', 'decode', 'eval')}
    out = dict(layers=st, reuse=[], refused=None, notes=[])
    if st['forward'] != 'equal':
        out['refused'] = 'forward identity ' + st['forward'] + ': no automatic image equivalence (data kept, to be located)'
        return out
    out['reuse'].append('image')
    if st['decode'] != 'equal':
        out['notes'].append('decode identity ' + st['decode'] + ': point cloud is not an output of the target method')
        return out
    try:
        pc = check_point_cloud(arrays or {})
    except Defect as e:
        out['refused'] = f'point cloud defect: {e}'
        return out
    out['reuse'].append('point_cloud')
    if st['eval'] != 'equal':
        out['notes'].append('evaluation identity ' + st['eval'] + ': point cloud may be re-evaluated later; stored metrics not reusable')
        return out
    try:
        mi = check_metrics(metrics, pc)
    except Defect as e:
        out['refused'] = f'metrics defect: {e}'
        return out
    out['reuse'].append('metrics'); out['metric_info'] = mi
    return out


def drills(real_key: str, real_arrays: dict, real_meta: dict, u: dict, have: dict, want: dict, loc0: tuple | None) -> dict:
    import copy
    res = {}
    A, M = real_arrays, real_meta['metrics']
    full = decide(have, want, A, M); res['real_entry_full_reuse'] = full['reuse'] == ['image', 'point_cloud', 'metrics'] and full['refused'] is None

    r = decide({}, {}, A, M); res['bad1_all_identities_empty_refused'] = r['reuse'] == [] and r['refused'] is not None
    A2 = dict(A, point_correct=np.zeros_like(A['point_correct']), point_wrong=np.zeros_like(A['point_wrong']), point_unconf=np.zeros_like(A['point_unconf']))
    M2 = dict(M, id_correct=0, id_wrong=0, id_unconf=0)
    r = decide(have, want, A2, M2); res['bad2_accepted_without_cwu_refused'] = r['refused'] is not None and 'c / w / u' in r['refused']
    r = decide(have, want, A, dict(M, state='no_output')); res['bad3_valid_outputs_but_no_output_refused'] = r['refused'] is not None and 'state' in r['refused']
    r = decide(have, want, A, dict(M, glob_pt_rmse=None)); res['bad4_domain_samples_but_rmse_missing_refused'] = r['refused'] is not None and 'glob_pt_rmse' in r['refused']

    n = len(A['pts']); z = np.zeros(n, bool)
    E = dict(A, accepted=z.copy(), valid=z.copy(), in_global=z.copy(), in_local=z.copy(), point_correct=z.copy(), point_wrong=z.copy(), point_unconf=z.copy())
    ME = dict(M, state='no_output', n_accepted=0, n_valid=0, glob_n_valid=0, loc_n_valid=0, id_correct=0, id_wrong=0, id_unconf=0, glob_Ctau=0.0, loc_Ctau=0.0,
              glob_Cvalid=0.0, loc_Cvalid=0.0, glob_cov_interp=0.0, loc_cov_interp=0.0, glob_pt_rmse=None, glob_pt_p95=None, loc_pt_rmse=None, loc_pt_p95=None,
              glob_surf_rmse=None, loc_surf_rmse=None)
    r = decide(have, want, E, ME); res['ok1_legal_no_output_kept'] = r['refused'] is None and 'metrics' in r['reuse'] and r['metric_info']['legal_no_output']
    acc = np.asarray(A['accepted'], bool)
    I = dict(A, valid=z.copy(), in_global=z.copy(), in_local=z.copy())
    MI = dict(ME, n_accepted=int(acc.sum()), id_correct=M['id_correct'], id_wrong=M['id_wrong'], id_unconf=M['id_unconf'])
    r = decide(have, want, I, MI); res['ok2_accepted_but_all_invalid_is_legal_no_output'] = r['refused'] is None and 'metrics' in r['reuse']
    O = dict(A, in_global=z.copy(), in_local=z.copy())
    MO = dict(M, glob_n_valid=0, loc_n_valid=0, glob_pt_rmse=None, glob_pt_p95=None, loc_pt_rmse=None, loc_pt_p95=None, glob_Ctau=0.0, loc_Ctau=0.0, glob_Cvalid=0.0,
              loc_Cvalid=0.0, glob_cov_interp=0.0, loc_cov_interp=0.0, glob_surf_rmse=None, loc_surf_rmse=None)
    r = decide(have, want, O, MO); res['ok3_only_out_of_domain_outputs_kept'] = r['refused'] is None and 'metrics' in r['reuse'] and set(r['metric_info']['domain_without_valid']) == {'glob', 'loc'}
    if loc0:
        r = decide(loc0[3], loc0[4], loc0[1], loc0[2]['metrics']); res['ok4_real_record_without_local_valid_kept'] = r['refused'] is None and r['metric_info']['domain_without_valid'] == ['loc']
    r = decide(have, dict(want, eval='other-eval-fingerprint'), A, M); res['ok5_eval_change_keeps_point_cloud'] = r['reuse'] == ['image', 'point_cloud'] and r['refused'] is None
    r = decide(have, dict(want, eval=None), A, M); res['ok6_eval_identity_missing_keeps_point_cloud'] = r['reuse'] == ['image', 'point_cloud']
    r = decide(have, dict(want, decode='x'), A, M); res['ok7_decode_change_image_only'] = r['reuse'] == ['image']

    img = K.load_npz(K.ROOT / u['image_record'].replace('.json', '.npz'))['image']
    same = target_identity(u, overrides=dict(image_array=img.copy()))
    res['drift0_unchanged_copies_equal'] = same == want
    im2 = img.copy(); im2[256, 256] = (int(im2[256, 256]) + 1) % 256
    t = target_identity(u, overrides=dict(image_array=im2)); res['drift_forward_pixel_change_detected'] = t['forward'] != want['forward'] and t['decode'] == want['decode']
    cfg = copy.deepcopy(live_decode_cfg()); cfg['dot_cfg']['epi_tol_px'] = float(cfg['dot_cfg'].get('epi_tol_px', 3.0)) + 0.5
    t = target_identity(u, overrides=dict(decode_cfg=cfg)); res['drift_decode_cfg_change_detected'] = t['decode'] != want['decode'] and t['forward'] == want['forward']
    src_file = _module_files(['s3c_dev'])['s3c_dev']; txt = (K.ROOT / src_file).read_text(encoding='utf-8').replace("FLOOR = {'DF': 0.5", "FLOOR = {'DF': 0.6")
    code = live_code(DECODE_MODULES); code[src_file] = content_sha(txt)
    t = target_identity(u, overrides=dict(decode_code=code)); res['drift_decode_code_copy_change_detected'] = t['decode'] != want['decode']
    import rdy_platform as P
    dg = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in P.dom_global().items()}
    zk = [k for k in dg if str(k).startswith('z_') and isinstance(dg[k], np.ndarray)][0]; dg[zk][0] = dg[zk][0] + 1e-3
    t = target_identity(u, overrides=dict(domains=dict(glob=domain_content(override=dg), loc=domain_content(u['surface'])))); res['drift_eval_domain_copy_change_detected'] = t['eval'] != want['eval'] and t['decode'] == want['decode']
    thr = dict(live_thresholds(), tau=0.2); t = target_identity(u, overrides=dict(thresholds=thr)); res['drift_eval_threshold_change_detected'] = t['eval'] != want['eval']
    return dict(passed=all(res.values()), checks=res, note='isolated in-memory copies; no file written, no renderer or decoder imported for calling')
