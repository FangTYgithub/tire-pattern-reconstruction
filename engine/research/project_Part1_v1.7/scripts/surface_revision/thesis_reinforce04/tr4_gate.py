# Validate raw point clouds, derived arrays and metric records for reuse.

from __future__ import annotations

import math

import numpy as np

import tr4_common as K

RAW = ('pts', 'accepted')
DERIVED_COMMON = ('valid', 'zr', 'in_global', 'in_local', 'point_correct', 'point_wrong', 'point_unconf')
DERIVED_FAMILY = dict(point=('point_feature',), line=('line_id', 'truth_k', 'confirmable'))
REQ_METRICS = ('state', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_cov_interp', 'loc_cov_interp',
               'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'loc_surf_rmse', 'id_correct', 'id_wrong', 'id_unconf')


class Defect(Exception):
    pass


def _undef(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _defined_error(name: str, v) -> None:

    if _undef(v):
        raise Defect(f'{name} undefined although the current rule defines it')
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise Defect(f'{name} = {v!r} is not a number') from None
    if not math.isfinite(f):
        raise Defect(f'{name} = {f} is not finite')
    if f < 0.0:
        raise Defect(f'{name} = {f} is negative')


def check_raw(a: dict) -> dict:

    miss = [k for k in RAW if k not in a]
    if miss:
        raise Defect(f'raw output fields missing: {miss}')
    P = np.asarray(a['pts']); acc = np.asarray(a['accepted'])
    if P.ndim != 2 or P.shape[1] != 3 or P.dtype.kind != 'f':
        raise Defect(f'pts is not a float (n, 3) array: {P.dtype} {P.shape}')
    if acc.dtype != bool or acc.ndim != 1 or len(acc) != len(P):
        raise Defect(f'accepted is not a bool (n,) array matching pts: {acc.dtype} {acc.shape} vs {P.shape}')
    if 'feats' in a and np.asarray(a['feats']).ndim != 2:
        raise Defect('feats is not a 2-D array')
    return dict(n=int(len(P)), n_accepted=int(acc.sum()), n_accepted_finite=int((acc & np.isfinite(P).all(axis=1)).sum()))


def check_derived(a: dict, family: str) -> dict:

    need = DERIVED_COMMON + DERIVED_FAMILY.get(family, ())
    miss = [k for k in need if k not in a]
    if miss:
        raise Defect(f'derived evaluation arrays missing: {miss}')
    n = len(a['pts'])
    bad = [k for k in need if len(np.asarray(a[k])) != n]
    if bad:
        raise Defect(f'per-point evaluation arrays of wrong length: {bad}')
    acc = np.asarray(a['accepted'], bool); val = np.asarray(a['valid'], bool); ig = np.asarray(a['in_global'], bool); il = np.asarray(a['in_local'], bool)
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
            if nn > 0:
                _defined_error(e, m[e])
            elif not _undef(m[e]):
                raise Defect(f'{e} = {m[e]} defined without any valid output in the domain')
        ci, cv, ct = m[f'{dn}_cov_interp'], m[f'{dn}_Cvalid'], m[f'{dn}_Ctau']
        for nm, v in ((f'{dn}_cov_interp', ci), (f'{dn}_Cvalid', cv), (f'{dn}_Ctau', ct)):
            if _undef(v) or not math.isfinite(float(v)) or not (0.0 <= float(v) <= 1.0):
                raise Defect(f'{nm} = {v} outside [0, 1]')
        if not (float(ct) <= float(cv) + 1e-12 <= float(ci) + 2e-12):
            raise Defect(f'{dn}: C_tau {ct} <= C_valid {cv} <= C_int {ci} violated')
        for e in (f'{dn}_surf_rmse', f'{dn}_surf_p95'):
            if e not in m and e.endswith('p95'):
                continue
            if float(ci) > 0:
                _defined_error(e, m[e])
            elif not _undef(m[e]):
                raise Defect(f'{e} = {m[e]} defined without interpolation')
    if m['state'] == 'no_output' and (float(m['glob_Ctau']) != 0 or float(m['glob_Cvalid']) != 0 or float(m['loc_Ctau']) != 0 or float(m['loc_Cvalid']) != 0):
        raise Defect('no_output with coverage > 0')
    return dict(state=m['state'], legal_no_output=m['state'] == 'no_output', no_accepted_output=pc['n_accepted'] == 0,
                accepted_but_all_invalid=(pc['n_accepted'] > 0 and pc['n_valid'] == 0),
                domain_without_valid=[dn for dn, nn in (('glob', pc['n_glob']), ('loc', pc['n_loc'])) if nn == 0])


def decide(have: dict, want: dict, arrays: dict | None, metrics: dict | None, family: str = 'point') -> dict:

    def conf(layer):
        h, w = (have or {}).get(layer), (want or {}).get(layer)
        if not h or not w:
            return 'unconfirmed'
        return 'equal' if h == w else 'changed'
    st = {l: conf(l) for l in ('forward', 'decode', 'eval')}
    out = dict(layers=st, reuse=[], refused=None, refused_layer=None, notes=[])
    if st['forward'] != 'equal':
        out.update(refused='forward identity ' + st['forward'] + ': no automatic image equivalence (data kept, to be located)', refused_layer='forward')
        return out
    out['reuse'].append('image')
    if st['decode'] != 'equal':
        out['notes'].append('decode identity ' + st['decode'] + ': point cloud is not an output of the target method')
        return out
    try:
        raw = check_raw(arrays or {})
    except Defect as e:
        out.update(refused=f'raw output defect: {e}', refused_layer='point_cloud')
        return out
    out['reuse'].append('point_cloud'); out['raw_info'] = raw
    if st['eval'] != 'equal':
        out['notes'].append('evaluation identity ' + st['eval'] + ': point cloud is re-evaluated; stored evaluation not reusable (its arrays are not required)')
        return out
    try:
        pc = check_derived(arrays, family)
        mi = check_metrics(metrics, pc)
    except Defect as e:
        out.update(refused=f'evaluation defect: {e}', refused_layer='metrics')
        out['notes'].append('point cloud kept for a re-evaluation')
        return out
    out['reuse'].append('metrics'); out['metric_info'] = mi
    return out


def real_examples() -> dict:

    import tr2_bind as B2
    import tr3_gate as G3
    import tr_a3 as A3
    import rdy_platform as P
    A = A3.plan().set_index('input_key'); ver = P.s2().VERSION; first = None; loc0 = None
    for p in sorted((K.OUT1 / 'shards').glob('*.npz')):
        for ik, (arr, meta) in B2.read_shard(p).items():
            m = meta.get('metrics') or {}
            if first is None or (loc0 is None and m.get('loc_n_valid') == 0):
                u = A.loc[ik].to_dict(); u['input_key'] = ik; u['platform_version'] = ver
                h = G3.history_identity(u); w = G3.target_identity(u)
                e = dict(key=ik, shard=p.name, arrays=arr, metrics=m, have={k: h[k] for k in ('forward', 'decode', 'eval')}, want=w)
                if first is None:
                    first = e
                elif m.get('loc_n_valid') == 0:
                    loc0 = e
            if first is not None and loc0 is not None:
                return dict(first=first, loc0=loc0)
    return dict(first=first, loc0=loc0)


# Test consistency checks for raw points, derived arrays and evaluated metrics.
def drills(ex: dict) -> dict:
    f = ex['first']; A, M, have, want = f['arrays'], f['metrics'], f['have'], f['want']
    res = {}
    full = decide(have, want, A, M); res['real_entry_full_reuse'] = full['reuse'] == ['image', 'point_cloud', 'metrics'] and full['refused'] is None

    for dn in ('glob', 'loc'):
        for bad in (float('inf'), float('-inf'), -0.01, float('nan')):
            r = decide(have, want, A, dict(M, **{f'{dn}_surf_rmse': bad}))
            res[f'fix1_{dn}_surf_rmse_{bad}_refused'] = r['refused_layer'] == 'metrics' and f'{dn}_surf_rmse' in r['refused'] and r['reuse'] == ['image', 'point_cloud']
        r = decide(have, want, A, dict(M, **{f'{dn}_pt_rmse': -1e-3})); res[f'fix1_{dn}_pt_rmse_negative_refused'] = r['refused_layer'] == 'metrics'
        r = decide(have, want, A, dict(M, **{f'{dn}_pt_p95': float('inf')})); res[f'fix1_{dn}_pt_p95_inf_refused'] = r['refused_layer'] == 'metrics'

    A_noig = {k: v for k, v in A.items() if k != 'in_global'}
    r = decide(have, dict(want, eval='new-evaluation-version'), A_noig, M)
    res['fix2_eval_changed_in_global_missing_keeps_point_cloud'] = r['reuse'] == ['image', 'point_cloud'] and r['refused'] is None
    A_noid = {k: v for k, v in A.items() if k not in ('point_correct', 'point_wrong', 'point_unconf', 'point_feature', 'valid', 'in_local', 'in_global', 'zr')}
    r = decide(have, dict(want, eval='new-evaluation-version'), A_noid, None)
    res['fix2_eval_changed_only_raw_fields_keeps_point_cloud'] = r['reuse'] == ['image', 'point_cloud'] and r['refused'] is None
    r = decide(have, want, A_noig, M)
    res['fix2_eval_equal_in_global_missing_refuses_metrics_only'] = r['reuse'] == ['image', 'point_cloud'] and r['refused_layer'] == 'metrics'
    r = decide(have, dict(want, eval=None), A_noig, M); res['fix2_eval_identity_missing_keeps_point_cloud'] = r['reuse'] == ['image', 'point_cloud']

    r = decide({}, {}, A, M); res['bad1_all_identities_empty_refused'] = r['reuse'] == [] and r['refused_layer'] == 'forward'
    A2 = dict(A, point_correct=np.zeros_like(A['point_correct']), point_wrong=np.zeros_like(A['point_wrong']), point_unconf=np.zeros_like(A['point_unconf']))
    r = decide(have, want, A2, dict(M, id_correct=0, id_wrong=0, id_unconf=0)); res['bad2_accepted_without_cwu_refused'] = r['refused_layer'] == 'metrics' and 'c / w / u' in r['refused']
    r = decide(have, want, A, dict(M, state='no_output')); res['bad3_valid_outputs_but_no_output_refused'] = r['refused_layer'] == 'metrics' and 'state' in r['refused']
    r = decide(have, want, A, dict(M, glob_pt_rmse=None)); res['bad4_domain_samples_but_rmse_missing_refused'] = r['refused_layer'] == 'metrics' and 'glob_pt_rmse' in r['refused']
    r = decide(have, want, dict(A, pts=np.asarray(A['pts'])[:, :2]), M); res['bad5_raw_coordinates_not_n3_refused'] = r['refused_layer'] == 'point_cloud' and r['reuse'] == ['image']
    r = decide(have, want, dict(A, accepted=np.asarray(A['accepted'])[:-1]), M); res['bad6_raw_acceptance_length_refused'] = r['refused_layer'] == 'point_cloud'

    n = len(A['pts']); z = np.zeros(n, bool)
    E = dict(A, accepted=z.copy(), valid=z.copy(), in_global=z.copy(), in_local=z.copy(), point_correct=z.copy(), point_wrong=z.copy(), point_unconf=z.copy())
    ME = dict(M, state='no_output', n_accepted=0, n_valid=0, glob_n_valid=0, loc_n_valid=0, id_correct=0, id_wrong=0, id_unconf=0, glob_Ctau=0.0, loc_Ctau=0.0,
              glob_Cvalid=0.0, loc_Cvalid=0.0, glob_cov_interp=0.0, loc_cov_interp=0.0, glob_pt_rmse=None, glob_pt_p95=None, loc_pt_rmse=None, loc_pt_p95=None,
              glob_surf_rmse=None, loc_surf_rmse=None, glob_surf_p95=None, loc_surf_p95=None)
    r = decide(have, want, E, ME); res['ok1_no_accepted_output_is_legal_no_output'] = r['refused'] is None and r['metric_info']['legal_no_output'] and r['metric_info']['no_accepted_output']
    I = dict(A, valid=z.copy(), in_global=z.copy(), in_local=z.copy())
    MI = dict(ME, n_accepted=int(np.asarray(A['accepted'], bool).sum()), id_correct=M['id_correct'], id_wrong=M['id_wrong'], id_unconf=M['id_unconf'])
    r = decide(have, want, I, MI)
    res['ok2_accepted_but_all_invalid_is_legal_no_output'] = r['refused'] is None and r['metric_info']['accepted_but_all_invalid'] and not r['metric_info']['no_accepted_output']
    O = dict(A, in_global=z.copy(), in_local=z.copy())
    MO = dict(M, glob_n_valid=0, loc_n_valid=0, glob_pt_rmse=None, glob_pt_p95=None, loc_pt_rmse=None, loc_pt_p95=None, glob_Ctau=0.0, loc_Ctau=0.0, glob_Cvalid=0.0,
              loc_Cvalid=0.0, glob_cov_interp=0.0, loc_cov_interp=0.0, glob_surf_rmse=None, loc_surf_rmse=None, glob_surf_p95=None, loc_surf_p95=None)
    r = decide(have, want, O, MO); res['ok3_only_out_of_domain_outputs_kept'] = r['refused'] is None and set(r['metric_info']['domain_without_valid']) == {'glob', 'loc'}
    l0 = ex.get('loc0')
    if l0:
        r = decide(l0['have'], l0['want'], l0['arrays'], l0['metrics']); res['ok4_real_record_without_local_valid_kept'] = r['refused'] is None and r['metric_info']['domain_without_valid'] == ['loc']
    r = decide(have, dict(want, eval='other'), A, M); res['ok5_eval_change_keeps_point_cloud'] = r['reuse'] == ['image', 'point_cloud'] and r['refused'] is None
    r = decide(have, dict(want, decode='x'), A, M); res['ok6_decode_change_image_only'] = r['reuse'] == ['image'] and r['refused'] is None
    r = decide(have, want, A, dict(M, glob_surf_rmse=0.0)); res['ok7_zero_surface_rmse_is_legal'] = r['refused'] is None
    return dict(passed=all(res.values()), n=len(res), checks=res, real_record=f['key'], real_record_shard=f['shard'], loc0_record=(l0 or {}).get('key'),
                note='real gate functions on isolated in-memory copies of real THESIS-REINFORCE-01 DF records; nothing written, no renderer / decoder / evaluator called; '
                     'the 513 stored DF entries are NOT re-checked (their acceptance and hashes are referenced)')
