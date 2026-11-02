# Run paired layout and clipping cases with checked image identities.

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import s4r6_common as H

K = H.K
ARM = {'D0': 'SPAT_D0', 'P4TABLE': 'SPAT_P4table', 'CLIP': 'SPAT_P4table'}
REQUIRED = ('image_key', 'baseline', 'family', 'layout', 'phase', 'surface', 'sigma', 'render_seed', 'raster_key', 'peak_f32', 'image_source')
_CACHE: dict = {}


def set_read_only(flag: bool = True) -> None:
    _CACHE['ro'] = bool(flag)


def manifest() -> pd.DataFrame:
    if 'M' not in _CACHE:
        _CACHE['M'] = pd.read_csv(H.reg(H.OUT / 'manifest.csv'), low_memory=False, keep_default_na=False, float_precision='round_trip')
    return _CACHE['M']


def results_plan() -> pd.DataFrame:
    if 'R' not in _CACHE:
        _CACHE['R'] = pd.read_csv(H.reg(H.OUT / '_parts' / 'B_results_plan.csv'), low_memory=False, keep_default_na=False)
    return _CACHE['R']


def validate_row(c: dict) -> dict:
    miss = [k for k in REQUIRED if k not in c or (c[k] == '' and k not in ('noise_seed',))]
    if miss:
        raise H.Stop('input', f"condition {c.get('image_key', '?')}: missing required field(s) {miss}")
    return c


def cond(key: str) -> dict:
    M = manifest(); x = M[M.image_key == key]
    if len(x) != 1:
        raise H.Stop('input', f'condition {key} not in the frozen manifest')
    return validate_row(x.iloc[0].to_dict())


def lay_of(baseline: str, layout: str) -> dict:
    if 'L' not in _CACHE:
        import s4r6_bind as B
        _CACHE['L'] = B.layouts()
    return _CACHE['L'][(baseline, layout)]


def inputs(c: dict) -> dict:
    import r2_layouts as L
    validate_row(c)
    key = ('inp', c['baseline'], c['layout'], c['phase'])
    if key not in _CACHE:
        inp = L.build_input(c['baseline'], lay_of(c['baseline'], c['layout']), c['phase'], c['layout'] == 'regular')
        inp['table'] = dict(inp['table'], j=float(c['j']))
        _CACHE[key] = inp
    inp = _CACHE[key]
    if inp['raster_key'] != c['raster_key'] or float(np.float32(inp['peak'])) != float(c['peak_f32']):
        raise H.Stop('check', f"{c['image_key']}: rebuilt raster / peak differ from the frozen manifest")
    return inp


def image(c: dict, purpose: str = 'science'):
    validate_row(c)
    if c['image_source'] != 'new':
        ip = H.R2 / 'images' / f"{c['image_key']}.json"
        if not K.complete(ip, c['image_key']):
            raise H.Stop('reference', f"{c['image_key']}: stored image record incomplete or hash mismatch")
        r = json.loads(ip.read_text(encoding='utf-8'))
        return K.load_npz(ip.with_suffix('.npz'))['image'], r['npz_sha256'], 'S4-REAL-02'
    ip = H.IMG6 / f"{c['image_key']}.json"
    if H.complete(ip, c['image_key'], required=('render_seed', 'raster_key', 'peak')):
        r = H.load_json(ip)
        return K.load_npz(ip.with_suffix('.npz'))['image'], r['npz_sha256'], 'S4-REAL-06 (stored)'
    if _CACHE.get('ro'):
        raise H.Stop('missing', f"{c['image_key']}: image not rendered (read-only analysis)")
    import pb_m3 as PM
    import rdy_platform as P
    inp = inputs(c); seed = int(c['render_seed'])
    B = H.render_budget(); n = B.reserve(c['image_key'], purpose); t0 = time.monotonic()
    try:
        bd = P.bound()
        res = PM.render(bd.sid(c['surface']), inp['pattern'], float(inp['peak']), float(c['sigma']), seed, K.RK.PSF_PX, K.BOUNDARY)
        img = np.asarray(res['image'])
        if img.dtype != np.uint8 or img.shape != (512, 512):
            raise H.Stop('technical', f'render returned {img.dtype} {img.shape}')
        h = H.save_npz(ip.with_suffix('.npz'), image=img)
        H.save_json(ip, dict(identity=c['image_key'], baseline=c['baseline'], family=c['family'], surface=c['surface'], layout=c['layout'], phase=c['phase'], sigma=float(c['sigma']),
                             noise_seed=c['noise_seed'], render_seed=seed, raster_key=inp['raster_key'], peak=float(inp['peak']), lit_px=inp['lit_px'], integral_f32=inp['integral_f32'],
                             peak_cap_triggered=bool(inp['peak'] > 1.0), boundary=K.BOUNDARY, psf_px=K.RK.PSF_PX, render_meta=K.clean(res['meta']), budget_n=n,
                             seconds=round(time.monotonic() - t0, 3), npz_sha256=h, complete=True, t_local=K.now_local()))
        B.log(dict(n=n, key=c['image_key'], purpose=purpose, event='ok'))
        return img, h, 'S4-REAL-06 (rendered)'
    except Exception as e:
        B.log(dict(n=n, key=c['image_key'], purpose=purpose, event='failed', error=f'{type(e).__name__}: {e}'[:300]))
        raise


def _params(c: dict, inp: dict, method: str):
    import rdy_platform as P
    import sp_regression as R1
    if method == 'D0':
        import sp2_common as S2
        return dict(section='S4', radius_mm=float(S2.R_MM)), None
    return R1.p4_params(), P.s2().line_cfg(float(inp['peak']))


def decode(c: dict, method: str, purpose: str, store: Path | None = None) -> dict:
    import rdy_decode as DC
    store = store or H.DEC6
    img, img_sha, _src = image(c)
    inp = inputs(c); prm, cfg = _params(c, inp, method)
    table_hash = inp['table']['spec']['table']['hash'] if c['family'] == 'line' else None
    option = 'abstain' if method == 'CLIP' else None
    dk = H.decode_key(c['image_key'], img_sha, c['raster_key'], table_hash, method, option, prm, cfg)
    rp = store / f'{dk}.json'
    if H.complete(rp, dk, required=('meta', 'method')):
        return dict(H.load_json(rp), stored=True)
    if _CACHE.get('ro'):
        raise H.Stop('missing', f'{dk}: decode missing (read-only analysis)')
    B = H.decode_budget(); n = B.reserve(dk, purpose); t0 = time.monotonic()
    try:
        extra = {}
        if method in ('D0', 'P4TABLE'):
            row = dict(section='S4', family=c['family'], arm=ARM[method], radius_mm=(prm['radius_mm'] if method == 'D0' else ''), strategy=c['strategy_view'])
            out = DC.decode(method, img, inp, row)
            if sorted(out['arms']) != [ARM[method]]:
                raise H.Stop('technical', f'{method}: arms {sorted(out["arms"])}')
        else:
            import s4r5_clip as CL
            import sp2_lines_table as LT2
            import m4_lines as ML
            spec, pattern = inp['table']['spec'], inp['pattern']
            r = CL.run_p4_table_clip(img, spec, cfg, pattern, prm, 'abstain')
            DC._check_window(r['window_z0_mm'], r['window_z1_mm'], r['depth_margin_mm'])
            if r['clip']['ids']:
                fam = LT2.line_family_table(spec); b = CL.clipped_bands(spec, pattern, fam); feat = np.asarray(r['features'], np.float64); rc = ML._resolve_sensor_cfg(cfg)
                c0 = LT2.line_candidates_table(feat, fam, rc, centrelines=LT2.raster_centrelines_table(pattern, fam)); c1 = LT2.line_candidates_table(feat, b['famx'], rc, centrelines=b['centrelines'])
                K0 = len(fam['ids'])
                extra['complete_columns_identical'] = {k: bool(np.array_equal(np.asarray(c0[k]), np.asarray(c1[k])[:, :K0], equal_nan=np.asarray(c0[k]).dtype.kind == 'f'))
                                                       for k in ('Q', 'Q_design', 'z', 'mu', 'admissible')}
            extra['clip'] = K.clean({k: v for k, v in r['clip'].items() if k != 'n_complete_candidates'})
            out = dict(method='P4TABLE', arms={ARM[method]: dict(pts=np.asarray(r['points'], np.float64).reshape(-1, 3), accepted=np.asarray(r['accepted'], bool), aux=dict(r=r))},
                       shared={}, window=K.RK.WINDOW_MM)
            n_complete = np.asarray(r['clip']['n_complete_candidates'])
        A, meta = DC.pack(out)
        if method == 'CLIP':
            A['clip_n_complete_candidates'] = n_complete
        h = H.save_npz(rp.with_suffix('.npz'), **A)
        rec = dict(identity=dk, image_key=c['image_key'], image_npz_sha=img_sha, method=method, option=option, purpose=purpose, budget_n=n, meta=meta, extra=extra,
                   code=H.code_hashes(H.DECODE_DEPS[method]), seconds=round(time.monotonic() - t0, 3), npz_sha256=h, complete=True, t_local=K.now_local())
        H.save_json(rp, rec); B.log(dict(n=n, key=dk, purpose=purpose, event='ok', seconds=rec['seconds']))
        return dict(rec, stored=False)
    except Exception as e:
        B.log(dict(n=n, key=dk, purpose=purpose, event='failed', error=f'{type(e).__name__}: {e}'[:300], tb=traceback.format_exc()[-1500:]))
        raise


def _dec_from_r5(ident: str) -> dict:
    A = K.load_npz(H.reg(H.R5 / 'decodes' / f'{ident}.npz')); A = {k[2:]: v for k, v in A.items()}
    r = dict(features=A['features'], line_id=A['line_id'], status=A['status'], chain=A['chain'], segment_id=A['segment_id'], n_candidates=A['n_candidates'], guard_reason=A['guard_reason'],
             points=A['points'], accepted=A['accepted'], detect_info={}, counts={}, checks={})
    return dict(pts=np.asarray(A['points'], np.float64).reshape(-1, 3), accepted=np.asarray(A['accepted'], bool), aux=dict(r=r), _clip_n_complete=A.get('n_complete_candidates'))


def result(c: dict, method: str) -> dict:

    import rdy_decode as DC
    R = results_plan(); x = R[(R.cond_id == c['image_key']) & (R.method == method)]
    if len(x) != 1:
        raise H.Stop('plan', f"{c['image_key']} {method}: not in the frozen results plan")
    x = x.iloc[0]
    if x.result_source == 'reuse_S4-REAL-02':
        rp = H.R2 / 'raw' / f'{x.reuse_ref}.json'
        if not K.complete(rp, x.reuse_ref):
            raise H.Stop('reference', f'{x.reuse_ref}: stored decode incomplete')
        out = DC.unpack(K.load_npz(H.reg(rp.with_suffix('.npz'))), json.loads(H.reg(rp).read_text(encoding='utf-8'))['meta'])
        return dict(dkey=x.reuse_ref, dsha=K.sha_file(rp.with_suffix('.npz')), dec=out['arms'][ARM[method]], source='reuse S4-REAL-02', record=K.rel(rp))
    if x.result_source == 'reuse_S4-REAL-05':
        rp = H.R5 / 'decodes' / f'{x.reuse_ref}.json'
        if not H.complete(rp, x.reuse_ref):
            raise H.Stop('reference', f'{x.reuse_ref}: stored CLIP decode incomplete')
        r5 = json.loads((H.R5 / 'run.json').read_text(encoding='utf-8'))
        if r5['config_C']['code'] != H.code_hashes(list(r5['config_C']['code'])):
            raise H.Stop('reference', 'S4-REAL-05 CLIP dependencies changed since its freeze')
        return dict(dkey=x.reuse_ref, dsha=K.sha_file(rp.with_suffix('.npz')), dec=_dec_from_r5(x.reuse_ref), source='reuse S4-REAL-05', record=K.rel(rp))
    rec = decode(c, method, x.decode_purpose)
    A = K.load_npz(H.DEC6 / f"{rec['identity']}.npz"); out = DC.unpack(A, rec['meta']); dec = out['arms'][ARM[method]]
    if method == 'CLIP':
        dec['_clip_n_complete'] = A.get('clip_n_complete_candidates')
    return dict(dkey=rec['identity'], dsha=rec['npz_sha256'], dec=dec, source=('new decode' if not rec.get('stored') else 'new decode (stored)'), record=K.rel(H.DEC6 / f"{rec['identity']}.json"))


def _ref_cond(c: dict) -> dict:
    M = manifest(); x = M[(M.baseline == c['baseline']) & (M.layout == c['layout']) & (M.phase == c['phase']) & (M.surface == c['surface']) & (M.sigma.astype(float) == 0.0)]
    if len(x) != 1:
        raise H.Stop('dependency', f"{c['image_key']}: sigma-0 reference condition missing")
    return validate_row(x.iloc[0].to_dict())


def evaluate(c: dict, method: str, store: Path | None = None, eval_deps: dict | None = None, domains: dict | None = None, res: dict | None = None, counter: dict | None = None) -> dict:
    import s4r6_eval as EV
    store = store or H.EVL6
    res = res or result(c, method)
    ref = None; img = ref_img = None
    if c['family'] == 'point':
        img, isha, _ = image(c); rc = _ref_cond(c); ref_img, rsha, _ = image(rc); ref = [rc['image_key'], rsha]
    ek = H.eval_key(res['dkey'], res['dsha'], eval_deps or EV.eval_deps(), domains or EV.domains(c['surface']), EV.mesh_hash(c['surface']), EV.IDENT_RULE[c['family']], ref)
    ep = store / f'{ek}.json'
    if H.complete(ep, ek, required=('metrics', 'nodes')):
        return H.load_json(ep)
    if _CACHE.get('ro'):
        raise H.Stop('missing', f'{ek}: evaluation missing (read-only analysis)')
    t0 = time.monotonic()
    dec = {k: v for k, v in res['dec'].items() if not k.startswith('_')}
    m, arr, ex = EV.evaluate(c['family'], c['surface'], dec, inputs(c), img, ref_img)
    if counter is not None:
        counter['computed'] = counter.get('computed', 0) + 1
    if res['dec'].get('_clip_n_complete') is not None:
        arr['clip_n_complete_candidates'] = np.asarray(res['dec']['_clip_n_complete'])
    h = H.save_npz(ep.with_suffix('.npz'), **{k: v for k, v in arr.items() if isinstance(v, np.ndarray)})
    x = dict(identity=ek, image_key=c['image_key'], method=method, decode_key=res['dkey'], decode_npz_sha=res['dsha'], result_source=res['source'], decode_record=res['record'], ref_image=ref,
             metrics=K.clean(m), nodes=K.clean(ex['nodes']), source_identity=K.clean(ex.get('source_identity')), displacement=K.clean(ex.get('displacement')),
             seconds=round(time.monotonic() - t0, 3), npz_sha256=h, complete=True, t_local=K.now_local())
    H.save_json(ep, x)
    return x


def _frozen() -> dict:
    r = H.run_load()
    if 'config' not in r.get('frozen', {}):
        raise H.Stop('freeze', 'run configuration not frozen')
    H.run_verify_frozen(); cfg = r['config']
    if K.sha_file(H.OUT / 'manifest.csv') != cfg['manifest']['sha256']:
        raise H.Stop('freeze', 'manifest changed after the freeze')
    for m, deps in cfg['decode_deps'].items():
        if H.code_hashes(list(deps)) != deps:
            raise H.Stop('freeze', f'{m} decode dependencies changed after the freeze')
    return cfg


def gate_raster() -> dict:
    _frozen(); M = manifest(); rows = []
    reg = json.loads(H.reg(H.R2 / 'layouts.json').read_text(encoding='utf-8'))
    for (b, lay, ph), g in M.groupby(['baseline', 'layout', 'phase']):
        c = g.iloc[0].to_dict(); inp = inputs(c); T = inp['table']; B = reg['baselines'][b]
        rows.append(dict(record_type='gate_raster', baseline=b, layout=lay, phase=ph, raster_key=inp['raster_key'], raster_equal=bool(inp['raster_key'] == c['raster_key']),
                         peak_equal=bool(float(np.float32(inp['peak'])) == float(c['peak_f32'])), peak=inp['peak'], lit_px=inp['lit_px'], integral_rel_err=inp['integral_rel_err'],
                         peak_cap_triggered=bool(inp['peak'] > 1.0), protected_present=bool(B['protected_L'] in T['present'] and B['protected_R'] in T['present']),
                         movable_all_present=bool(all(mv in T['present'] for mv in B['movable'])), n_elements=int(T['n_elements']),
                         clipped_lit=(len(T['spec']['clipped_lines']) if T['spec'] else 0), n_conditions=len(g)))
    D = pd.DataFrame(rows)
    ok = bool(D.raster_equal.all() and D.peak_equal.all() and D.protected_present.all() and D.movable_all_present.all() and (~D.peak_cap_triggered).all() and (D.integral_rel_err <= 1e-6).all())
    band = D[D.clipped_lit > 0]
    out = dict(ok=ok, n_rasters=len(D), n_conditions=int(D.n_conditions.sum()), peak_range=[float(D.peak.min()), float(D.peak.max())], integral_rel_err_max=float(D.integral_rel_err.max()),
               band_rasters=band[['baseline', 'layout', 'phase']].to_dict('records'), t_local=K.now_local())
    H.part('gate_raster', D); H.run_put('gate_raster', out); H.flush_reads('gate_raster')
    if not ok:
        raise H.Stop('gate', 'raster gate failed')
    return out


def _old_row(spec: dict) -> dict:
    M = pd.read_csv(H.R2 / 'manifest.csv', low_memory=False, keep_default_na=False, float_precision='round_trip')
    x = M[(M.baseline == spec['baseline']) & (M.layout == spec['layout']) & (M.surface == spec['surface']) & (M.phase == spec['phase']) & (M.sigma.astype(float) == spec['sigma'])].drop_duplicates('image_key')
    r = x.iloc[0].to_dict()
    return r


def gate_entry() -> dict:

    import s4r6_bind as Bd
    cfg = _frozen(); res = []
    for e in Bd.ENTRY:
        r = _old_row(e); key = r['image_key']
        c = cond(key) if key in set(manifest().image_key) else None
        if c is None:
            raise H.Stop('plan', f'entry image {key} not in the manifest')
        rec = decode(c, e['method'], 'entry_regression', store=H.OUT / '_entry' / 'decodes')
        A = K.load_npz(H.OUT / '_entry' / 'decodes' / f"{rec['identity']}.npz")
        if e['method'] in ('D0', 'P4TABLE'):
            O = K.load_npz(H.R2 / 'raw' / f"{r['decode_key']}.npz")
            eq = {k: _eq(A[k], O[k]) for k in O}; ref = f"S4-REAL-02 raw {r['decode_key']}"
        else:
            r5 = _r5_record(key); O = K.load_npz(H.R5 / 'decodes' / f'{r5}.npz')
            eq = {k: _eq(A[f'a0_r_{k[2:]}'], O[k]) for k in O if k.startswith('r_') and k != 'r_n_complete_candidates'}
            eq['n_complete_candidates'] = _eq(A['clip_n_complete_candidates'], O['r_n_complete_candidates']); ref = f'S4-REAL-05 decode {r5}'
            if not rec['extra'].get('clip', {}).get('ids'):
                P4 = K.load_npz(H.R2 / 'raw' / f"{r['decode_key']}.npz"); eq.update({f'noop_vs_P4_{k[5:]}': _eq(A[k], P4[k]) for k in P4})

        dk = rec['identity']; import rdy_decode as DC
        out = DC.unpack(A, rec['meta']); dec = out['arms'][ARM[e['method']]]
        if e['method'] == 'CLIP':
            dec['_clip_n_complete'] = A['clip_n_complete_candidates']
        ev = evaluate(c, e['method'], store=H.OUT / '_entry' / 'evals', res=dict(dkey=dk, dsha=rec['npz_sha256'], dec=dec, source='entry', record=''))
        if e['method'] == 'CLIP':
            old = json.loads((H.R5 / 'evals' / f'{_r5_record(key)}.json').read_text(encoding='utf-8'))['metrics']
        else:
            old = json.loads((H.R2 / 'eval' / f"{r['eval_key']}.json").read_text(encoding='utf-8'))['metrics']
        mdiff = {k: (old[k], ev['metrics'].get(k)) for k in old if old[k] != ev['metrics'].get(k)}
        extra_ok = True
        if c['family'] == 'line':
            extra_ok = bool(ev['metrics']['status_closed'] and ev['metrics']['n_status_clipped_source_abstain'] == int((np.asarray(A['a0_r_status']) == 11).sum()))
        ok = bool(all(eq.values()) and not mdiff and extra_ok and (rec['extra'].get('complete_columns_identical') is None or all(rec['extra']['complete_columns_identical'].values())))
        res.append(dict(role=e['role'], image_key=key, method=e['method'], decode=dk, reference=ref, arrays_equal=bool(all(eq.values())), unequal=[k for k, v in eq.items() if not v],
                        metric_diffs=K.clean(mdiff), new_fields_ok=extra_ok, clip_ids=rec['extra'].get('clip', {}).get('ids'), ok=ok))
    out = dict(ok=bool(all(x['ok'] for x in res)), cases=res, t_local=K.now_local())
    H.run_put('gate_entry', out); H.flush_reads('gate_entry')
    if not out['ok']:
        raise H.Stop('gate', 'entry gate failed: ' + json.dumps([x for x in res if not x['ok']], default=str)[:600])
    return dict(ok=True, n=len(res))


def _eq(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == 'f' or b.dtype.kind == 'f':
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        return bool(np.array_equal(np.isnan(a64), np.isnan(b64)) and np.array_equal(a64, b64, equal_nan=True))
    return bool(np.array_equal(a, b))


def _r5_record(image_key: str) -> str:
    import s4r6_bind as Bd
    r = Bd._r5_clip_records().get(image_key)
    if r is None:
        raise H.Stop('reference', f'no S4-REAL-05 CLIP record for {image_key}')
    return r['identity']


def _iso_fail_with_finally() -> None:
    try:
        raise H.Stop('technical', 'injected failure (isolation test)')
    finally:
        (H.ISO / 'finally_ran.txt').parent.mkdir(parents=True, exist_ok=True); (H.ISO / 'finally_ran.txt').write_text('finally block ran', encoding='utf-8')


# Test result consistency and rejection of altered records in an isolated run.
def a4() -> dict:

    import s4r6_eval as EV
    import s4r6_analysis as N
    import shutil
    _frozen(); iso = H.ISO
    if iso.exists():
        shutil.rmtree(H.guard(iso))
    out = {}
    c = cond(_iso_cond())
    res = result(c, 'P4TABLE'); cnt = {}
    e1 = evaluate(c, 'P4TABLE', store=iso / 'evals', res=res, counter=cnt); e2 = evaluate(c, 'P4TABLE', store=iso / 'evals', res=res, counter=cnt)
    out['resume_same_deps'] = dict(ok=bool(cnt.get('computed') == 1 and e1['identity'] == e2['identity']), computed=cnt.get('computed'))
    dep2 = dict(EV.eval_deps(), **{'scripts/surface_revision/s4_real06/s4r6_eval.py': 'changed-for-test'})
    e3 = evaluate(c, 'P4TABLE', store=iso / 'evals', eval_deps=dep2, res=res, counter=cnt)
    dom2 = dict(EV.domains(c['surface']), omega1='changed-for-test')
    e4 = evaluate(c, 'P4TABLE', store=iso / 'evals', domains=dom2, res=res, counter=cnt)
    out['eval_version_change_misses'] = dict(ok=bool(e3['identity'] != e1['identity'] and e4['identity'] not in (e1['identity'], e3['identity']) and cnt['computed'] == 3), computed=cnt['computed'])
    inp = inputs(c); prm, cfg = _params(c, inp, 'P4TABLE')
    dk_a = H.decode_key(c['image_key'], 'sha', c['raster_key'], 'tab', 'P4TABLE', None, prm, cfg); dk_b = H.decode_key(c['image_key'], 'sha', c['raster_key'], 'tab', 'P4TABLE', None, prm, cfg)
    out['decode_key_independent_of_eval'] = dict(ok=bool(dk_a == dk_b and 'scripts/surface_revision/s4_real06/s4r6_eval.py' not in str(H.DECODE_DEPS)))
    out['label_only_change'] = dict(ok=bool('s4r6_figs.py' not in json.dumps(H.EVAL_DEPS) + json.dumps(H.DECODE_DEPS)), note='figure code is in no image / decode / eval key')

    p = iso / 'evals' / f"{e1['identity']}.npz"; b = bytearray(p.read_bytes()); b[len(b) // 2] ^= 0xFF; p.write_bytes(bytes(b))
    rej_corrupt = not H.complete(p.with_suffix('.json'), e1['identity'], required=('metrics', 'nodes'))
    j = H.load_json(p.with_suffix('.json')); j['complete'] = False; H.save_json(p.with_suffix('.json'), j)
    rej_partial = not H.complete(p.with_suffix('.json'), e1['identity'])
    j['complete'] = True; j.pop('nodes'); H.save_json(p.with_suffix('.json'), j)
    rej_missing = not H.complete(p.with_suffix('.json'), e1['identity'], required=('metrics', 'nodes'))
    e5 = evaluate(c, 'P4TABLE', store=iso / 'evals', res=res, counter=cnt)
    out['corrupt_partial_missing_rejected'] = dict(ok=bool(rej_corrupt and rej_partial and rej_missing and cnt['computed'] == 4 and e5['identity'] == e1['identity']), recomputed=cnt['computed'])

    code1 = subprocess.run([sys.executable, '-c', 'import s4r6_run as R, s4r6_common as H, sys\ntry:\n    R.validate_row({"image_key": "x"})\nexcept H.Stop as e:\n    print(e); sys.exit(1)'],
                           cwd=str(H.HERE), capture_output=True, text=True)
    code2 = subprocess.run([sys.executable, '-c', 'import s4r6_run as R, s4r6_common as H, sys\ntry:\n    R._iso_fail_with_finally()\nexcept H.Stop as e:\n    print(e); sys.exit(1)'],
                           cwd=str(H.HERE), capture_output=True, text=True)
    out['missing_field_explicit_failure'] = dict(ok=bool(code1.returncode == 1 and 'missing required field' in code1.stdout), exit=code1.returncode, msg=code1.stdout.strip()[:160])
    out['failure_exit_status_kept'] = dict(ok=bool(code2.returncode == 1 and (iso / 'finally_ran.txt').exists()), exit=code2.returncode)

    dec0 = {k: v for k, v in res['dec'].items() if not k.startswith('_')}
    dec0 = dict(dec0, accepted=np.zeros_like(np.asarray(dec0['accepted'], bool)), pts=np.full_like(np.asarray(dec0['pts'], np.float64), np.nan))
    r0 = dict(dec0['aux']['r']); r0['accepted'] = dec0['accepted']; r0['points'] = dec0['pts']; r0['status'] = np.where(np.asarray(r0['status']) == 0, 2, r0['status']); dec0['aux'] = dict(r=r0)
    e6 = evaluate(c, 'P4TABLE', store=iso / 'evals_nooutput', res=dict(res, dkey=res['dkey'] + '-nooutput-test', dec=dec0), counter=cnt)
    row = N.domain_rows(dict(c, method='P4TABLE'), e6)
    g = [x for x in row if x['domain'] == 'glob'][0]
    out['no_output_chain'] = dict(ok=bool(e6['metrics']['state'] == 'no_output' and g['Ctau'] == 0.0 and g['pt_rmse'] is None and e6['metrics']['status_closed']), state=e6['metrics']['state'],
                                  Ctau=g['Ctau'], pt_rmse=g['pt_rmse'])

    Bi = H.Budget('decode', 2, dict(test=1), base=iso / 'budget'); Bi.reserve('a', 'test')
    try:
        Bi.reserve('b', 'test'); capped = False
    except H.Stop:
        capped = True
    out['budget_cap'] = dict(ok=capped)
    res_all = dict(ok=bool(all(v['ok'] for v in out.values())), tests=out, decode_calls=0, evaluations=cnt.get('computed'), t_local=K.now_local())
    H.run_put('A4', res_all)
    if not res_all['ok']:
        raise H.Stop('gate', 'A4 isolation tests failed: ' + json.dumps({k: v for k, v in out.items() if not v['ok']}, default=str)[:600])
    return res_all


def _iso_cond() -> str:
    M = manifest(); x = M[(M.baseline == 'L6.3') & (M.layout == 'regular') & (M.surface == 'B_curv') & (M.phase == 'H1') & (M.sigma.astype(float) == 0.0)]
    return x.iloc[0].image_key


def _run(keys: list, label: str) -> dict:
    cfg = _frozen(); r = H.run_load()
    for g in ('gate_raster', 'gate_entry', 'A4'):
        if not r.get(g, {}).get('ok'):
            raise H.Stop('gate', f'{g} not passed')
    M = manifest(); done = []
    rows = [cond(k) for k in keys]
    rows.sort(key=lambda c: (float(c['sigma']), c['family'], c['image_key']))
    for c in rows:
        for m in (['D0'] if c['family'] == 'point' else ['P4TABLE', 'CLIP']):
            ev = evaluate(c, m)
            done.append(dict(image_key=c['image_key'], method=m, eval=ev['identity'], source=ev['result_source'], state=ev['metrics']['state']))
    chk = pair_checks([c for c in rows if c['family'] == 'line'])
    out = dict(label=label, n_conditions=len(rows), n_results=len(done), states={s: sum(1 for d in done if d['state'] == s) for s in set(d['state'] for d in done)},
               pair_checks_ok=chk['ok'], pair_checks=chk, render_budget=H.render_budget().state(), decode_budget=H.decode_budget().state(), t_local=K.now_local())
    H.run_put(f'run_{label}', out); H.flush_reads(f'run_{label}')
    if not chk['ok']:
        raise H.Stop('gate', f'line pair invariants failed in {label}: ' + json.dumps([x for x in chk['cases'] if not x['ok']], default=str)[:800])
    return {k: v for k, v in out.items() if k != 'pair_checks'}


def pair_checks(line_conds: list) -> dict:
    cases = []
    for c in line_conds:
        p4 = result(c, 'P4TABLE'); cl = result(c, 'CLIP')
        a, b = p4['dec']['aux']['r'], cl['dec']['aux']['r']
        front = {k: _eq(a[k], b[k]) for k in ('features', 'chain', 'segment_id', 'guard_reason')}
        band = bool(c['clipped_band_lit'] in (True, 'True'))
        x = dict(image_key=c['image_key'], band=band, front_equal=bool(all(front.values())))
        st = np.asarray(b['status']); acc_b = np.asarray(b['accepted'], bool); acc_a = np.asarray(a['accepted'], bool)
        x['abstain_not_accepted'] = bool(not (acc_b & (st == 11)).any())
        both = acc_a & acc_b
        pa, pb = np.asarray(a['points'], np.float64)[both], np.asarray(b['points'], np.float64)[both]
        x['co_accepted_points_identical'] = bool(np.array_equal(pa, pb) and np.array_equal(np.asarray(a['line_id'])[both], np.asarray(b['line_id'])[both]))
        ncc = cl['dec'].get('_clip_n_complete')
        x['complete_candidates_equal'] = bool(ncc is not None and _eq(np.asarray(a['n_candidates']), np.asarray(ncc)))
        if not band:
            full = {k: _eq(a[k], b[k]) for k in ('line_id', 'status', 'n_candidates', 'accepted', 'points')}
            x['noop_equal'] = bool(all(full.values()))
        else:
            rec = H.load_json(Path(K.ROOT / cl['record'])) if cl['record'] else {}
            cci = (rec.get('extra') or {}).get('complete_columns_identical')
            x['complete_columns_identical'] = (None if cci is None else bool(all(cci.values())))
        x['ok'] = bool(x['front_equal'] and x['abstain_not_accepted'] and x['co_accepted_points_identical'] and x['complete_candidates_equal'] and x.get('noop_equal', True)
                       and x.get('complete_columns_identical', True) is not False)
        cases.append(x)
    return dict(ok=bool(all(x['ok'] for x in cases)), n=len(cases), cases=cases)


def subset() -> dict:
    M = manifest()
    keys = list(M[(M.set == 'pilot') & (M.strategy_view == 'S1') & (M.j.astype(float) == 0.10) & (M.sigma.astype(float) == 0.0)].image_key)
    if len(keys) != 12 or not (M.set_index('image_key').loc[keys, 'image_source'] == 'new').all():
        raise H.Stop('plan', f'subset is not the 12 planned new images ({len(keys)})')
    return _run(keys, 'subset')


def rest() -> dict:
    r = H.run_load()
    if not r.get('run_subset', {}).get('pair_checks_ok'):
        raise H.Stop('gate', 'subset not passed')
    M = manifest()
    return _run(list(M.image_key), 'all')
