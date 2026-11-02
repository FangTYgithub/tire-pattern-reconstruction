# Render sub-ray comparisons and evaluate D0, DF and P4 results.

from __future__ import annotations

import numpy as np

import tr4_common as K
import tr4_gate as G
import tr4_prep as P

IMG_SHAPE = (512, 512)
RESULT_KEYS = dict(point=(('n3', 'D0'), ('n3', 'DF'), ('n1', 'DF')), line=(('n3', 'P4'),))
GROUPS = dict(compat=('render_compat', 'render'), render3=('render_3x3', 'render'), n3=('decode_3x3', 'decode'), n1=('decode_1x1_DF', 'decode'), eval=('evaluation',))


class RealBackend:
    name = 'real'

    def __init__(self):
        self.calls = dict(render=0, decode=0, evaluation=0)

    def render(self, prep: dict):
        from m3_virtual_camera import synthesise_camera_image
        S = P.s2(); self.calls['render'] += 1
        img, _hit_map, _hits3d, m3 = synthesise_camera_image(S.build_mesh(prep['sid']), np.asarray(prep['pattern'], np.float32), prep['cfg'])
        if m3.get('z_alignment') != 'absolute':
            raise K.Stop('technical', f"M3 alignment {m3.get('z_alignment')} (expected 'absolute')")
        return np.asarray(img, np.uint8), m3_meta(m3)

    def decode(self, method: str, img, prep: dict):
        self.calls['decode'] += 1
        if method in P.ARM:
            import rdy_platform as RP
            bd = RP.bound()
            return bd.call('decode', bd.method(P.ARM[method]), img, prep['ctx'])
        if method == 'DF':
            import s3c_dev as DV
            D = DV.dot_decode(img, prep['ctx']['binary'], 1, [prep['size_mm']], 'DF', 'binding')
            dec = DV.dot_dec(D, 'single')
            dec['aux']['df_floor'] = dict(n_floor=int(D['cd'].get('n_floor', 0)), a_floor=D['cd'].get('a_floor'))
            return dec
        raise K.Stop('technical', f'unknown method {method}')

    def evaluate(self, family: str, dec: dict, prep: dict):
        import rdy_platform as RP
        import s1r_eval as E
        self.calls['evaluation'] += 1
        sid = prep['sid']; mesh = P.s2().build_mesh(sid)
        m, arrays, info, _ = E.evaluate_case(sid, family, dec, prep['ctx'], mesh, RP.dom_local(prep['surface']), RP.dom_global(), sid == 'A_curv_gauss')
        return m, arrays, info


def m3_meta(m3: dict) -> dict:
    return dict(n_rays_cast=int(m3['n_rays_cast']), n_hits=int(m3['n_hits']), n_valid_pixels=int(m3['n_valid_pixels']), z_alignment=str(m3.get('z_alignment')),
                rays_per_pixel=int(m3.get('rays_per_pixel', 1) or 1), noise_sigma=float(m3['noise_sigma']), mottle_seed=int(m3['mottle_seed']),
                psf_sigma_px=float(m3['psf_sigma_px']))


def image_stats(img) -> dict:
    a = np.asarray(img)
    return dict(grey_sum=int(a.astype(np.int64).sum()), frac_255=float((a == 255).mean()), frac_0=float((a == 0).mean()), content_sha=K.content_sha(a))


def validate_dec(dec, family: str) -> None:
    if not isinstance(dec, dict):
        raise K.Stop('result_integrity', f'decoder returned no result ({type(dec).__name__})')
    for k in ('pts', 'accepted', 'aux'):
        if k not in dec or dec[k] is None:
            raise K.Stop('result_integrity', f'decoder result lacks {k!r}')
    pts = np.asarray(dec['pts']); acc = np.asarray(dec['accepted'])
    if pts.ndim != 2 or pts.shape[1] != 3 or acc.dtype != bool or len(acc) != len(pts):
        raise K.Stop('result_integrity', f'decoder arrays malformed: pts {pts.shape}, accepted {acc.dtype} {acc.shape}')
    aux = dec['aux']
    need = ('feats', 'tr', 'cd', 'sel_code', 'sel_fidx') if family == 'point' else ('r',)
    miss = [k for k in need if k not in aux]
    if miss:
        raise K.Stop('result_integrity', f'decoder aux lacks {miss}')
    if family == 'point':
        miss = [k for k in ('point_idx', 'best_idx', 'ambiguous', 'matched') if k not in aux['tr']] + [k for k in ('cands', 'reason', 'n_labels', 'stats') if k not in aux['cd']]
    else:
        miss = [k for k in ('features', 'line_id', 'status', 'chain', 'segment_id', 'n_candidates', 'guard_reason') if k not in aux['r']]
    if miss:
        raise K.Stop('result_integrity', f'decoder trace lacks {miss}')


def raw_pack(family: str, dec: dict) -> tuple[dict, dict]:
    import s2p_case as SC
    a, meta = SC.raw_arrays(family, dec)
    if family == 'point' and 'df_floor' in dec['aux']:
        meta['df_floor'] = K.clean(dec['aux']['df_floor'])
    back = raw_unpack(family, a, meta)
    same = np.array_equal(np.asarray(back['pts']), np.asarray(dec['pts'], np.float64), equal_nan=True) and np.array_equal(back['accepted'], np.asarray(dec['accepted'], bool))
    if family == 'point':
        same = same and np.array_equal(np.asarray(back['aux']['feats'], np.float32).reshape(-1, 2), np.asarray(dec['aux']['feats'], np.float32).reshape(-1, 2))
        same = same and all(np.array_equal(np.asarray(back['aux']['tr'][k]), np.asarray(v)) for k, v in dec['aux']['tr'].items() if isinstance(v, np.ndarray))
    else:
        same = same and all(np.array_equal(np.asarray(back['aux']['r'][k]), np.asarray(dec['aux']['r'][k])) for k in ('features', 'line_id', 'status', 'chain', 'segment_id', 'n_candidates', 'guard_reason'))
    if not same:
        raise K.Stop('result_integrity', 'raw decoder arrays do not reproduce the decoder output')
    return a, meta


def raw_unpack(family: str, A: dict, m: dict) -> dict:

    if family == 'point':
        tr = {k[3:]: A[k] for k in A if k.startswith('tr_')}
        cd = dict(cands=[tuple(row) for row in A['cand_table']], reason=A['comp_reason'], n_labels=int(m['n_labels']), stats=A['comp_stats'])
        info = dict(m['info'])
        if 'info_ambiguous_mask' in A:
            info['ambiguous_mask'] = A['info_ambiguous_mask']
        aux = dict(feats=A['feats'], tr=tr, cd=cd, sel_code=A['sel_code'], sel_fidx=A['sel_fidx'], det_info=m['det_info'], info=info)
    else:
        r = {k[2:]: A[k] for k in A if k.startswith('r_')}
        r.update(detect_info=m['detect_info'], counts=m['counts'], checks=m['checks'])
        aux = dict(r=r)
    return dict(pts=A['pts'], accepted=A['accepted'], aux=aux)


def decode_identity(method: str, prep: dict, image_sha: str, binding: dict) -> str:
    return K.digest(dict(method=method, method_fp=binding['methods'][method if method != 'P4' else 'P4']['fingerprint'], image=image_sha,
                         ctx=dict(raster=prep['pattern_sha'], peak=prep['peak'], size=prep['size_mm'])))


def eval_identity(binding: dict) -> str:
    return binding['methods']['EVAL']['fingerprint']


def pack_path(store: K.Store, ik: str):
    return store.data / 'cases' / f'{ik}.npz'


def pack_read(store: K.Store, ik: str) -> tuple[dict, dict, str]:

    p = pack_path(store, ik)
    if not p.exists():
        return dict(input_key=ik, results={}), {}, 'missing'
    try:
        meta, A = K.load_pack(p)
    except Exception as e:
        return dict(input_key=ik, results={}, error=repr(e)[:200]), {}, 'unreadable'
    rel = K.rel(p); ix = store.index().get(rel)
    if ix is None or ix['sha256'] != K.sha(p):
        return meta, A, ('index_mismatch' if ix is not None else 'index_missing')
    return meta, A, 'ok'


def pack_write(store: K.Store, ik: str, meta: dict, A: dict) -> str:
    p = pack_path(store, ik); h = K.save_pack(p, meta, A); store.index_put(K.rel(p), h, kind='case_pack')
    return h


def _sub(A: dict, prefix: str) -> dict:
    n = len(prefix)
    return {k[n:]: v for k, v in A.items() if k.startswith(prefix)}


def verify_image3(meta: dict, A: dict, prep3: dict) -> bool:
    im = meta.get('image3') or {}
    if 'img3' not in A or im.get('forward_id') != prep3['forward_id']:
        return False
    a = A['img3']
    return a.dtype == np.uint8 and a.shape == IMG_SHAPE and K.content_sha(a) == im.get('content_sha') and int((im.get('m3') or {}).get('rays_per_pixel', 0)) == 3


def verify_result(meta: dict, A: dict, rk: str, family: str, dec_id: str, ev_id: str) -> str:

    r = (meta.get('results') or {}).get(rk)
    if not r or r.get('decode_id') != dec_id:
        return 'invalid'
    raw = _sub(A, f'{rk}__raw__')
    if not raw or K.arrays_digest(raw) != r.get('raw_digest'):
        return 'invalid'
    try:
        G.check_raw(raw)
    except G.Defect:
        return 'invalid'
    ev = _sub(A, f'{rk}__ev__')
    if r.get('eval_id') != ev_id or not ev or K.arrays_digest(ev) != r.get('eval_digest'):
        return 'eval_stale'
    try:
        G.check_metrics(r.get('metrics'), G.check_derived(ev, family))
    except G.Defect:
        return 'eval_stale'
    return 'valid'


def compat_load(store: K.Store) -> dict:
    p = store.data / 'compat.json'
    return K.load_json(p) if p.exists() else {}


def stage_compat(C, store: K.Store, backend, log) -> dict:

    entries = compat_load(store)
    for c in C.to_dict('records'):
        ik = c['input_key']; p1 = P.prepare(c, 1)
        stored = np.asarray(K.load_npz(p1['stored_image'])['image']); st_sha = K.content_sha(stored)
        if K.sha(p1['stored_image']) != p1['stored_npz_sha']:
            raise K.Stop('preflight', f'{ik}: stored image file changed')
        e = entries.get(ik)
        if e and e.get('equal') and e.get('forward_id') == p1['forward_id'] and e.get('stored_content_sha') == st_sha:
            log(f'compat {ik}: verified earlier, not re-rendered'); continue
        rid = store.reserve(GROUPS['compat'], ik, stage='compat', rays=1)
        try:
            img, m3 = backend.render(p1)
        except BaseException as ex:
            store.end(rid, _fail(ex), error=repr(ex)[:300]); raise
        equal = bool(img.shape == stored.shape and img.dtype == stored.dtype and np.array_equal(img, stored))
        d = (img.astype(np.int16) - stored.astype(np.int16)) if img.shape == stored.shape else None
        e = dict(forward_id=p1['forward_id'], rays=1, equal=equal, new_content_sha=K.content_sha(img), stored_content_sha=st_sha, stored_npz=K.rel(p1['stored_image']),
                 n_pixels_different=(int((d != 0).sum()) if d is not None else None), max_abs_diff=(int(np.abs(d).max()) if d is not None else None),
                 shape_new=list(img.shape), dtype_new=str(img.dtype), m3=m3, **{f'new_{k}': v for k, v in image_stats(img).items() if k != 'content_sha'}, t=K.now())
        if not equal:
            h = K.save_pack(store.data / 'compat_mismatch' / f'{ik}.npz', dict(input_key=ik, entry=e), dict(image_new=img, image_stored=stored, diff=(d if d is not None else np.zeros(0))))
            e['mismatch_pack_sha256'] = h
        entries[ik] = e; K.save_json(store.data / 'compat.json', entries)
        store.end(rid, 'ok' if equal else 'mismatch')
        log(f'compat {ik}: equal={equal}')
        if not equal:
            raise K.Stop('compat_stop', f'1x1 image of {ik} differs from the stored image ({e["n_pixels_different"]} pixels, max {e["max_abs_diff"]}); no 3x3, no decode')
    ok = len(entries) == len(C) and all(entries[k]['equal'] for k in C.input_key)
    return dict(passed=ok, n=len(entries), n_equal=sum(1 for v in entries.values() if v['equal']))


def run_result(store, backend, meta: dict, A: dict, ik: str, rk: str, family: str, method: str, prep: dict, img, img_sha: str, binding: dict, log) -> tuple[dict, dict, str]:
    dec_id = decode_identity(method, prep, img_sha, binding); ev_id = eval_identity(binding)
    st = verify_result(meta, A, rk, family, dec_id, ev_id)
    if st == 'valid':
        log(f'{ik} {rk}: verified, reused'); return meta, A, 'reused'
    r = dict(meta.get('results', {}).get(rk) or {}) if st == 'eval_stale' else {}
    if st != 'eval_stale':
        rid = store.reserve(GROUPS[rk.split('_')[0]], f'{ik}|{rk}', stage=rk, method=method)
        try:
            dec = backend.decode(method, img, prep)
            validate_dec(dec, family)
            raw, rmeta = raw_pack(family, dec)
        except BaseException as ex:
            store.end(rid, _fail(ex), error=repr(ex)[:300]); raise
        A = {k: v for k, v in A.items() if not k.startswith(f'{rk}__')}
        A.update({f'{rk}__raw__{k}': v for k, v in raw.items()})
        r = dict(method=method, rays=int(prep['rays']), decode_id=dec_id, image_sha=img_sha, raw_digest=K.arrays_digest(raw), raw_meta=rmeta, t_decode=K.now())
        meta.setdefault('results', {})[rk] = r
        pack_write(store, ik, meta, A)
        store.end(rid, 'ok', raw_digest=r['raw_digest'])
    else:
        log(f'{ik} {rk}: decode verified, evaluation re-done only')
    raw = _sub(A, f'{rk}__raw__'); dec = raw_unpack(family, raw, r['raw_meta'])
    if 'df_floor' in r['raw_meta']:
        dec['aux']['df_floor'] = r['raw_meta']['df_floor']
    rid = store.reserve(GROUPS['eval'], f'{ik}|{rk}', stage='eval', method=method)
    try:
        m, ev, info = backend.evaluate(family, dec, prep)
        mi = G.check_metrics(m, G.check_derived(ev, family))
        ev = {k: np.asarray(v) for k, v in ev.items()}
    except G.Defect as ex:
        store.end(rid, _fail(ex), error=repr(ex)[:300]); raise K.Stop('result_integrity', f'{ik} {rk}: evaluation result defect: {ex}') from None
    except BaseException as ex:
        store.end(rid, _fail(ex), error=repr(ex)[:300]); raise
    A = {k: v for k, v in A.items() if not k.startswith(f'{rk}__ev__')}
    A.update({f'{rk}__ev__{k}': v for k, v in ev.items()})
    r.update(eval_id=ev_id, eval_digest=K.arrays_digest(ev), metrics=K.clean(m), metric_info=mi, state=m['state'],
             info=K.clean({k: v for k, v in (info or {}).items() if k in ('component_reasons', 'selection_codes', 'det_info', 'detect_info', 'counts', 'checks', 'reference')}),
             t_eval=K.now())
    meta['results'][rk] = r
    pack_write(store, ik, meta, A)
    store.end(rid, 'ok', state=m['state'])
    log(f'{ik} {rk}: {m["state"]}, glob C_tau {m.get("glob_Ctau")}')
    return meta, A, 'new'


def _fail(ex) -> str:
    return 'interrupted' if isinstance(ex, KeyboardInterrupt) else 'failed'


def df_regression(c: dict, m: dict, ev: dict) -> dict:

    ik = c['input_key']; z = K.load_npz(K.ROOT / 'outputs/surface_revision/s3_code04/cache/C1' / f'{ik}.npz')
    C4 = K.read_csv(K.ROOT / 'outputs/surface_revision/s3_code04/results/C1_dots_cases.csv'); rows = C4[(C4.rule == 'DF') & (C4.input_key == ik)]
    pts = np.asarray(ev['pts'], np.float32)
    eq_pts = bool(pts.shape == z['DF_pts'].shape and np.array_equal(pts, z['DF_pts'], equal_nan=True))
    eq_acc = bool(np.array_equal(np.asarray(ev['accepted'], bool), z['DF_acc']))
    eq_zr = bool(np.array_equal(np.asarray(ev['zr'], np.float32), z['DF_zr'], equal_nan=True))
    diff = {}
    for k in ('state', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse',
              'loc_pt_p95', 'glob_surf_rmse', 'loc_surf_rmse', 'id_correct', 'id_wrong', 'id_unconf'):
        for v0 in rows[k].tolist():
            v1 = m.get(k)
            if k == 'state':
                if str(v0) != str(v1):
                    diff[k] = (v0, v1)
                continue
            u0 = v0 is None or (isinstance(v0, float) and np.isnan(v0)); u1 = v1 is None or (isinstance(v1, float) and np.isnan(v1))
            if u0 != u1 or (not u0 and abs(float(v0) - float(v1)) > 1e-9):
                diff[k] = (v0, v1)
    return dict(ok=bool(len(rows) >= 1 and eq_pts and eq_acc and eq_zr and not diff), n_reference_rows=len(rows), points_equal=eq_pts, accepted_equal=eq_acc,
                zr_equal=eq_zr, metric_diffs=diff)


def baseline_inventory(C, binding: dict) -> dict:

    import pandas as pd
    rows = []; old = {r['input_key']: r for r in binding['old_results']}
    for c in C.to_dict('records'):
        p = K.ROOT / c['eval_record']; E = K.load_json(p); A = dict(K.load_npz(p.with_suffix('.npz')))
        m = 'D0' if c['family'] == 'point' else 'P4'
        try:
            G.check_raw(A); pc = G.check_derived(A, c['family']); mi = G.check_metrics(E['metrics'], pc); ok, why = True, None
        except G.Defect as e:
            ok, why, mi = False, str(e), None
        o = old[c['input_key']]
        idok = bool(o['manifest_ok'] and o['arm_ok'] is not False and o['decode_peak_ok'])
        rows.append(dict(input_key=c['input_key'], family=c['family'], method=m, source=K.rel(p), integrity_ok=ok, defect=why, state=(mi or {}).get('state'),
                         accepted_but_all_invalid=(mi or {}).get('accepted_but_all_invalid'), method_manifest=o['manifest'], method_identity_ok=idok,
                         eval_identity='unchanged: S1-REAL-01 closure (incl. s1r_eval) unchanged, domains / thresholds equal to the TR03 baseline',
                         decision=('reuse image + point cloud + metrics (after G1)' if ok and idok else 'NOT reusable'), common_evaluation_needed=0))
        if c['family'] == 'point':
            rows.append(dict(input_key=c['input_key'], family='point', method='DF', source=f"outputs/surface_revision/s3_code04/cache/C1/{c['input_key']}.npz", integrity_ok=None,
                             defect='feature positions / per-feature identity not stored by S3-CODE-04', state=None, accepted_but_all_invalid=None,
                             method_manifest='S3-CODE-04 frozen closure', method_identity_ok=True, eval_identity='-',
                             decision='new 1x1 DF decode (budget 18); the cached DF result is the regression reference', common_evaluation_needed=0))
    T = pd.DataFrame(rows)
    return dict(passed=bool(T[T.method != 'DF'].integrity_ok.all() and T.method_identity_ok.all()), table=T, old_common_evaluations=0,
                n_reuse=int(T.decision.str.startswith('reuse').sum()), n_df_new=int((T.method == 'DF').sum()))


def process_condition(c: dict, store: K.Store, backend, binding: dict, log, df_reference=df_regression) -> dict:
    ik = c['input_key']; fam = c['family']
    p3 = P.prepare(c, 3); p1 = P.prepare(c, 1)
    meta, A, st = pack_read(store, ik)
    if st in ('unreadable',):
        log(f'{ik}: pack unreadable -> rebuilt'); meta, A = dict(input_key=ik, results={}), {}
    elif st in ('index_mismatch', 'index_missing'):
        log(f'{ik}: pack {st}; components are verified by their embedded digests')
    meta['input_key'] = ik
    if verify_image3(meta, A, p3):
        img3 = A['img3']; log(f'{ik}: 3x3 image verified, reused')
    else:
        rid = store.reserve(GROUPS['render3'], ik, stage='render3', rays=3)
        try:
            img3, m3 = backend.render(p3)
            if img3.dtype != np.uint8 or img3.shape != IMG_SHAPE or m3['rays_per_pixel'] != 3:
                raise K.Stop('result_integrity', f'3x3 render returned {img3.dtype} {img3.shape}, rays {m3["rays_per_pixel"]}')
        except BaseException as ex:
            store.end(rid, _fail(ex), error=repr(ex)[:300]); raise
        meta = dict(input_key=ik, results={}, image3=dict(forward_id=p3['forward_id'], content_sha=K.content_sha(img3), m3=m3, **{k: v for k, v in image_stats(img3).items() if k != 'content_sha'}, t=K.now()))
        A = dict(img3=img3)
        pack_write(store, ik, meta, A); store.end(rid, 'ok')
        log(f'{ik}: 3x3 rendered (rays {m3["n_rays_cast"]}, hits {m3["n_hits"]})')
    sha3 = meta['image3']['content_sha']
    stored1 = np.asarray(K.load_npz(p1['stored_image'])['image']); sha1 = K.content_sha(stored1)
    out = {}
    for rays, method in RESULT_KEYS[fam]:
        rk = f'{rays}_{method}'
        prep, img, s = (p3, img3, sha3) if rays == 'n3' else (p1, stored1, sha1)
        meta, A, how = run_result(store, backend, meta, A, ik, rk, fam, method, prep, img, s, binding, log)
        out[rk] = how
        if rk == 'n1_DF' and df_reference is not None:
            r = meta['results'][rk]; reg = df_reference(c, r['metrics'], _sub(A, f'{rk}__ev__'))
            out['df_regression'] = reg
            if not reg['ok']:
                raise K.Stop('technical', f'{ik}: new 1x1 DF does not reproduce the S3-CODE-04 DF result ({reg}); method binding inconsistent')
    return out


def stage_3x3(C, order: list, store: K.Store, backend, binding: dict, log, checkpoint_fn=None, only: list | None = None, df_reference=df_regression) -> dict:
    rows = C.set_index('input_key'); done = {}
    keys = [k for k in order if only is None or k in only]
    for i, ik in enumerate(keys):
        done[ik] = process_condition(dict(rows.loc[ik].to_dict(), input_key=ik), store, backend, binding, log, df_reference=df_reference)
        if checkpoint_fn is not None and i == 1 and only is None:
            checkpoint_fn(keys[:2])
    return done


def results_complete(C, store: K.Store, binding: dict) -> dict:

    miss = []; states = {}
    for c in C.to_dict('records'):
        ik = c['input_key']; meta, A, st = pack_read(store, ik)
        if st == 'missing' or st == 'unreadable':
            miss.append(f'{ik}: pack {st}'); continue
        p3 = P.prepare(c, 3); p1 = P.prepare(c, 1)
        if not verify_image3(meta, A, p3):
            miss.append(f'{ik}: 3x3 image'); continue
        sha1 = K.content_sha(np.asarray(K.load_npz(p1['stored_image'])['image']))
        for rays, method in RESULT_KEYS[c['family']]:
            rk = f'{rays}_{method}'; prep, s = (p3, meta['image3']['content_sha']) if rays == 'n3' else (p1, sha1)
            v = verify_result(meta, A, rk, c['family'], decode_identity(method, prep, s, binding), eval_identity(binding))
            if v != 'valid':
                miss.append(f'{ik}: {rk} {v}')
            else:
                states[f'{ik}|{rk}'] = meta['results'][rk]['state']
    return dict(complete=not miss, missing=miss, n_results=len(states), states=states)
