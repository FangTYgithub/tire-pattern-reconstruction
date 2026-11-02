# Run new-phase rendering, decoding and evaluation with complete line-identity counts.

"""THESIS-REINFORCE-05 A3 / A4: real chain prepare -> M3 render -> save image -> D0 / DF / P4 -> common evaluation + complete line identity -> save.

Backends
  render    rdy_platform.bound().call('render', sid, pattern, peak, sigma, seed, PSF 1 px) (= adp_surf02.render, the S12 path; 1 ray per projector
            pixel = M3 default, no override); z_alignment 'absolute', noise sigma and seed echoed by M3 are checked
  decode    D0 / P4: bound().call('decode', bd.method(arm), img, ctx) with the row context (actual peak, pitch / orientation from the raster meta);
            DF: s3c_dev.dot_decode(img, binary, 1, [radius of the row], 'DF', 'binding') + dot_dec('single')
  evaluate  s1r_eval.evaluate_case (platform domains); lines additionally the complete identity: THESIS-REINFORCE-01 A4 extension for the lit unregistered
            lines of THIS raster (tr_a4 geometry helpers; counts cross-checked against tr_a4.reidentify); dots: evaluator identity (A4 not applicable)
Storage   data/<forward_key>.npz per image: image (uint8, once), per method the raw decoder arrays (s2p_case.raw_arrays; re-evaluable without decoding),
          the evaluation arrays (+ full_* identity arrays for lines) and an embedded JSON meta with forward / decode / evaluation identities, digests and
          metrics. data/index.json = sha256 at commit (written by the scheduling process only).
Ledger    one writer (the scheduling process). Before an image is processed every needed step is reserved (caps checked). After the work returns - or
          raises - the scheduler inspects the ACTUAL pack: a verified product ends its reservation 'ok', the failing step 'failed' / 'interrupted', steps
          never reached 'aborted'. A ledger entry never stands in for a missing or corrupt product.
Resume    image reused when the forward identity and content digest verify; a method with the same decode identity and valid raw arrays is not decoded
          again; a changed evaluation identity re-runs only the evaluation; a changed method identity re-decodes only that method.
"""
from __future__ import annotations

import importlib
import time
import traceback

import numpy as np

import tr5_common as K
import tr5_manifest as MF

IMG_SHAPE = (512, 512)
PSF_PX = 1.0
GROUPS = dict(render=('render',), D0=('decode', 'decode_D0'), DF=('decode', 'decode_DF'), P4=('decode', 'decode_P4'), eval=('evaluation',))
FULL_ARRAYS = ('full_confirmable', 'full_truth_k', 'full_point_correct', 'full_point_wrong', 'full_point_unconf')


_FWD_STATIC: dict = {}


def _fwd_static() -> dict:
    if not _FWD_STATIC:
        import adp_surf02 as SA
        import m3_virtual_camera as M3
        S = MF.s2()
        _FWD_STATIC.update(platform=S.VERSION, m3_file=K.sha(M3.__file__), adp_file=K.sha(SA.__file__), m3_default_rays=int(M3.SENSOR_DEFAULTS.get('rays_per_pixel', 1)),
                           psf=PSF_PX, camera=K.digest(S.geometry_record()))
    return _FWD_STATIC


def forward_id(row: dict, ctx: dict) -> str:
    S = MF.s2(); cfg = S.m3_cfg(float(row['sigma']), int(row['seed']))
    if 'rays_per_pixel' in cfg:
        raise K.Stop('preflight', 'platform m3_cfg overrides rays_per_pixel')
    return K.digest(dict(forward_key=row['forward_key'], sid=row['platform_surface'], raster=row['raster_key'], pattern=K.content_sha(ctx['pattern']),
                         peak_f32=float(row['peak_f32']), integral=float(row['integral_actual']), block=int(row['block']), sigma=float(row['sigma']),
                         noise_k=int(row['noise_k']), seed=int(row['seed']), rays=1, m3_cfg=cfg, static=_fwd_static()))


def decode_id(method: str, row: dict, image_sha: str, fps: dict) -> str:
    return K.digest(dict(method=method, fp=fps[method], image=image_sha, ctx=dict(raster=row['raster_key'], peak=float(row['peak']), size=float(row['size_mm']),
                                                                                    pitch=row.get('line_pitch_mm'))))


def eval_id(family: str, fps: dict) -> str:
    return K.digest(dict(fp=fps['EVAL'], family=family))


class RealBackend:
    name = 'real'

    def __init__(self):
        self.calls = dict(render=0, decode=0, evaluation=0)

    def render(self, row: dict, ctx: dict):
        import rdy_platform as RP
        bd = RP.bound(); self.calls['render'] += 1
        res = bd.call('render', bd.sid(row['surface']), ctx['pattern'], float(row['peak']), float(row['sigma']), int(row['seed']), PSF_PX)
        img = np.asarray(res['image']); meta = dict(res.get('meta') or {})
        if meta.get('z_alignment') != 'absolute' or abs(float(meta.get('noise_sigma', -1)) - float(row['sigma'])) > 0 or int(meta.get('mottle_seed', -1)) != int(row['seed']):
            raise K.Stop('technical', f"{row['forward_key']}: M3 echo differs (alignment / sigma / seed): {meta}")
        meta['rays_per_pixel'] = _fwd_static()['m3_default_rays']
        return img, meta

    def decode(self, method: str, img, row: dict, ctx: dict):
        self.calls['decode'] += 1
        if method in ('D0', 'P4'):
            import rdy_platform as RP
            bd = RP.bound()
            return bd.call('decode', bd.method({'D0': 'D0', 'P4': 'P4_C3_cal'}[method]), img, ctx)
        if method == 'DF':
            import s3c_dev as DV
            D = DV.dot_decode(img, ctx['binary'], 1, [float(row['size_mm'])], 'DF', 'binding')
            dec = DV.dot_dec(D, 'single'); dec['aux']['df_floor'] = dict(n_floor=int(D['cd'].get('n_floor', 0)), a_floor=D['cd'].get('a_floor'))
            return dec
        raise K.Stop('technical', f'unknown method {method}')

    def evaluate(self, family: str, dec: dict, row: dict, ctx: dict):
        import rdy_platform as RP
        import s1r_eval as E
        self.calls['evaluation'] += 1
        sid = row['platform_surface']; mesh = MF.s2().build_mesh(sid)
        m, arrays, info, _ = E.evaluate_case(sid, family, dec, ctx, mesh, RP.dom_local(row['surface']), RP.dom_global(), sid == 'A_curv_gauss')
        return m, arrays, info


def make_backend(spec):
    if spec == 'real':
        return RealBackend()
    mod = importlib.import_module(spec['module'])
    return getattr(mod, spec['cls'])(**spec.get('args', {}))


def full_identity(row: dict, ctx: dict, ev: dict) -> tuple[dict, dict]:
    if row['family'] != 'line':
        return {}, dict(applied=False, rule='dots: evaluator identity (A4 not applicable)')
    import tr_a4 as A4
    o = ctx['meta_full']['olines']; B = np.asarray(ctx['binary']) > 0; sc = A4.scan_raster(ctx['binary'], o); extra = sc['extra']
    mesh = A4._mesh(row['surface'])
    acc = np.asarray(ev['accepted'], bool); lid = np.asarray(ev['line_id'], np.int64); conf0 = np.asarray(ev['confirmable'], bool); k0 = np.asarray(ev['truth_k'], np.int64)
    val = np.asarray(ev['valid'], bool); ig = np.asarray(ev['in_global'], bool); il = np.asarray(ev['in_local'], bool); feats = np.asarray(ev['feats'], np.float64)
    if len(feats):
        G = A4._geom(mesh, feats, o); lit = A4._lit_at(B, G['uv'])
        new = (~conf0) & G['in_proj'] & (G['dist'] <= G['band']) & np.isin(G['k'], extra) & lit
    else:
        new = np.zeros(0, bool)
    conf = conf0 | new; k = np.where(new, (G['k'] if len(feats) else k0), k0)
    c = acc & conf & (lid == k); w = acc & conf & (lid != k); u = acc & ~conf
    arr = dict(full_confirmable=conf, full_truth_k=k.astype(np.int32), full_point_correct=c, full_point_wrong=w, full_point_unconf=u)
    cnt = {}
    for nm, sel in (('all', acc), ('glob', acc & val & ig), ('loc', acc & val & il)):
        cnt[nm] = dict(c=int((c & sel).sum()), w=int((w & sel).sum()), u=int((u & sel).sum()))
    ref = A4.reidentify(mesh, feats, o, extra, B, acc, lid, conf0, k0, val, ig, il) if len(feats) else None
    if ref is not None:
        same = all(ref[f'{nm}_new_{x}'] == cnt[nm][x] for nm in ('all', 'glob', 'loc') for x in 'cwu') and ref['closure_old_identity']
        if not same:
            raise K.Stop('technical', f"{row['forward_key']}: complete identity differs from tr_a4.reidentify / closure ({ref})")
    info = dict(applied=True, extra_lines=extra, n_new_labels=int(new.sum()), n_new_labels_accepted=int((new & acc).sum()), counts=cnt,
                original=dict(c=int((acc & np.asarray(ev['point_correct'], bool)).sum()), w=int((acc & np.asarray(ev['point_wrong'], bool)).sum()),
                              u=int((acc & np.asarray(ev['point_unconf'], bool)).sum())), reidentify_crosscheck=(ref is not None))
    return arr, info


def pack_path(root, fk: str):
    return K.Store(root).data / f'{fk}.npz'


def pack_read(root, fk: str) -> tuple[dict, dict, str]:
    p = pack_path(root, fk)
    if not p.exists():
        return dict(forward_key=fk, results={}), {}, 'missing'
    try:
        meta, A = K.load_pack(p)
    except Exception as e:
        return dict(forward_key=fk, results={}, error=repr(e)[:200]), {}, 'unreadable'
    return meta, A, 'read'


def _sub(A: dict, prefix: str) -> dict:
    n = len(prefix)
    return {k[n:]: v for k, v in A.items() if k.startswith(prefix)}


def verify_image(meta: dict, A: dict, fid: str) -> bool:
    im = meta.get('image') or {}
    if 'image' not in A or im.get('forward_id') != fid:
        return False
    a = A['image']
    return a.dtype == np.uint8 and a.shape == IMG_SHAPE and K.content_sha(a) == im.get('content_sha')


def verify_result(meta: dict, A: dict, m: str, family: str, did: str, eid: str) -> str:
    import tr4_gate as G
    r = (meta.get('results') or {}).get(m)
    if not r or r.get('decode_id') != did:
        return 'invalid'
    raw = _sub(A, f'{m}__raw__')
    if not raw or K.arrays_digest(raw) != r.get('raw_digest'):
        return 'invalid'
    try:
        G.check_raw(raw)
    except G.Defect:
        return 'invalid'
    ev = _sub(A, f'{m}__ev__')
    if r.get('eval_id') != eid or not ev or K.arrays_digest(ev) != r.get('eval_digest'):
        return 'eval_stale'
    try:
        pc = G.check_derived(ev, family); G.check_metrics(r.get('metrics'), pc)
        if family == 'line':
            check_full(ev, r.get('full_identity') or {})
    except (G.Defect, K.Stop):
        return 'eval_stale'
    return 'valid'


def check_full(ev: dict, fi: dict) -> None:
    import tr4_gate as G
    miss = [k for k in FULL_ARRAYS if k not in ev]
    if miss or not fi.get('applied'):
        raise G.Defect(f'complete line identity missing: {miss}')
    n = len(ev['pts']); acc = np.asarray(ev['accepted'], bool)
    if any(len(ev[k]) != n for k in FULL_ARRAYS):
        raise G.Defect('complete identity arrays of wrong length')
    c, w, u = (np.asarray(ev[k], bool) for k in ('full_point_correct', 'full_point_wrong', 'full_point_unconf'))
    s = c.astype(int) + w + u
    if (s[acc] != 1).any() or (s[~acc] != 0).any():
        raise G.Defect('complete identity is not a c / w / u partition of the accepted outputs')
    if dict(c=int(c.sum()), w=int(w.sum()), u=int(u.sum())) != fi['counts']['all']:
        raise G.Defect('complete identity counts differ from the arrays')


def needs(root, row: dict, ctx: dict, fps: dict) -> dict:

    meta, A, st = pack_read(root, row['forward_key'])
    fid = forward_id(row, ctx)
    if st != 'read' or not verify_image(meta, A, fid):
        return dict(render=True, methods={m: 'decode' for m in MF.METHODS[row['family']]}, fid=fid, pack_state=st)
    sha = meta['image']['content_sha']; out = {}
    for m in MF.METHODS[row['family']]:
        v = verify_result(meta, A, m, row['family'], decode_id(m, row, sha, fps), eval_id(row['family'], fps))
        out[m] = {'valid': 'none', 'eval_stale': 'eval', 'invalid': 'decode'}[v]
    return dict(render=False, methods=out, fid=fid, pack_state=st)


# Run missing new-phase stages and add complete-line identity diagnostics.
def work(job: dict) -> dict:

    import tr4_chain as C4
    import tr4_gate as G
    row = job['row']; fam = row['family']; root = job['root']; fps = job['fps']; need = job['need']; fk = row['forward_key']
    be = make_backend(job['backend']); steps = []; t0 = time.time()
    try:
        ctx = MF.context(row); fid = forward_id(row, ctx)
        if fid != need['fid']:
            raise K.Stop('technical', f'{fk}: forward identity changed between scheduling and work')
        meta, A, _ = pack_read(root, fk)
        if need['render']:
            steps.append(['render', 'started'])
            img, rm = be.render(row, ctx)
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.shape != IMG_SHAPE:
                raise K.Stop('result_integrity', f'{fk}: render returned {type(img).__name__} {getattr(img, "dtype", None)} {getattr(img, "shape", None)}')
            meta = dict(forward_key=fk, results={}, image=dict(forward_id=fid, content_sha=K.content_sha(img), render_meta=K.clean(rm), grey_sum=int(img.astype(np.int64).sum()),
                                                                frac_255=float((img == 255).mean()), frac_0=float((img == 0).mean()), t=K.now()))
            A = dict(image=img)
            K.save_pack(pack_path(root, fk), meta, A); steps[-1][1] = 'done'
        img = A['image']; sha = meta['image']['content_sha']
        for m in MF.METHODS[fam]:
            what = need['methods'].get(m, 'none')
            if what == 'none':
                continue
            did = decode_id(m, row, sha, fps); eid = eval_id(fam, fps)
            if what == 'decode':
                steps.append([f'decode:{m}', 'started'])
                dec = be.decode(m, img, row, ctx)
                C4.validate_dec(dec, fam)
                raw, rmeta = C4.raw_pack(fam, dec)
                A = {k: v for k, v in A.items() if not k.startswith(f'{m}__')}
                A.update({f'{m}__raw__{k}': v for k, v in raw.items()})
                meta.setdefault('results', {})[m] = dict(method=m, decode_id=did, image_sha=sha, raw_digest=K.arrays_digest(raw), raw_meta=rmeta, t_decode=K.now())
                K.save_pack(pack_path(root, fk), meta, A); steps[-1][1] = 'done'
            r = meta['results'][m]
            steps.append([f'eval:{m}', 'started'])
            dec = C4.raw_unpack(fam, _sub(A, f'{m}__raw__'), r['raw_meta'])
            mtr, ev, info = be.evaluate(fam, dec, row, ctx)
            ev = {k: np.asarray(v) for k, v in ev.items()}
            pc = G.check_derived(ev, fam); mi = G.check_metrics(mtr, pc)
            farr, finfo = full_identity(row, ctx, ev)
            ev.update(farr)
            if fam == 'line':
                check_full(ev, finfo)
            A = {k: v for k, v in A.items() if not k.startswith(f'{m}__ev__')}
            A.update({f'{m}__ev__{k}': v for k, v in ev.items()})
            r.update(eval_id=eid, eval_digest=K.arrays_digest(ev), metrics=K.clean(mtr), metric_info=mi, state=mtr['state'], full_identity=K.clean(finfo),
                     info=K.clean({k: v for k, v in (info or {}).items() if k in ('component_reasons', 'selection_codes', 'det_info', 'detect_info', 'counts', 'checks')}),
                     t_eval=K.now())
            meta['results'][m] = r
            K.save_pack(pack_path(root, fk), meta, A); steps[-1][1] = 'done'
        return dict(forward_key=fk, ok=True, steps=steps, seconds=round(time.time() - t0, 2), calls=getattr(be, 'calls', None))
    except KeyboardInterrupt as e:
        if steps and steps[-1][1] == 'started':
            steps[-1][1] = 'interrupted'
        if job.get('inline'):
            e.steps = steps
            raise
        return dict(forward_key=fk, ok=False, steps=steps, error='KeyboardInterrupt', kind='interrupted')
    except (Exception, G.Defect) as e:
        if steps and steps[-1][1] == 'started':
            steps[-1][1] = 'failed'
        return dict(forward_key=fk, ok=False, steps=steps, error=repr(e)[:400], kind=(e.code if isinstance(e, K.Stop) else ('integrity' if isinstance(e, G.Defect) else 'error')),
                    trace=traceback.format_exc()[-1500:])


def _reserve_steps(store: K.Store, row: dict, need: dict) -> dict:
    rids = {}; fk = row['forward_key']
    if need['render']:
        rids['render'] = store.reserve(GROUPS['render'], fk, step='render')
    for m, what in need['methods'].items():
        if what == 'decode':
            rids[f'decode:{m}'] = store.reserve(GROUPS[m], f'{fk}|{m}', step='decode', method=m)
        if what in ('decode', 'eval'):
            rids[f'eval:{m}'] = store.reserve(GROUPS['eval'], f'{fk}|{m}', step='eval', method=m)
    return rids


def _close(store: K.Store, row: dict, need: dict, rids: dict, res: dict | None, fps: dict) -> dict:

    meta, A, st = pack_read(store.root, row['forward_key'])
    fid = need['fid']; img_ok = st == 'read' and verify_image(meta, A, fid)
    steps = {s: v for s, v in ((res or {}).get('steps') or [])}
    out = {}
    for step, rid in rids.items():
        if step == 'render':
            ok = img_ok
        else:
            kind, m = step.split(':')
            v = verify_result(meta, A, m, row['family'], decode_id(m, row, meta['image']['content_sha'], fps), eval_id(row['family'], fps)) if img_ok else 'invalid'
            ok = (v in ('valid', 'eval_stale')) if kind == 'decode' else (v == 'valid')
        if ok:
            status = 'ok'
        else:
            reported = steps.get(step)
            status = {'failed': 'failed', 'interrupted': 'interrupted', 'started': 'interrupted', 'done': 'failed'}.get(reported, 'aborted')
        store.end(rid, status, **({'error': (res or {}).get('error')} if status in ('failed', 'interrupted') else {}))
        out[step] = status
    if img_ok:
        p = pack_path(store.root, row['forward_key']); store.index_put(K.rel(p), K.sha(p), kind='image_pack')
    return out


# Schedule pending cases while tracking stage limits, completions and failures.
def schedule(rows: list, store: K.Store, fps: dict, backend='real', workers: int = 0, log=print, stop_on_first=True) -> dict:

    done, failures, reused = {}, [], 0
    jobs = []
    for row in rows:
        ctx = MF.context(row); need = needs(store.root, row, ctx, fps)
        if not need['render'] and all(v == 'none' for v in need['methods'].values()):
            reused += 1; done[row['forward_key']] = 'verified'
            continue
        jobs.append((row, need))
    if workers <= 0:
        for row, need in jobs:
            rids = _reserve_steps(store, row, need)
            try:
                res = work(dict(row=row, need=need, fps=fps, root=str(store.root), backend=backend, inline=True))
            except KeyboardInterrupt as e:
                st = _close(store, row, need, rids, dict(steps=(getattr(e, 'steps', None) or [[s, 'started'] for s in rids]), error='KeyboardInterrupt'), fps)
                store.note('interrupted', forward_key=row['forward_key'], steps=st)
                raise
            st = _close(store, row, need, rids, res, fps); done[row['forward_key']] = st
            if not res['ok']:
                failures.append(dict(forward_key=row['forward_key'], kind=res.get('kind'), error=res.get('error'), steps=st))
                if stop_on_first:
                    break
            else:
                log(f"{row['forward_key']} {row['config']} {row['surface']} b{row['block']} s{row['sigma']:g} k{row['noise_k']}: {st}")
    else:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        pending = list(jobs); running = {}; stop = False
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            while (pending and not stop) or running:
                while pending and not stop and len(running) < workers * 2:
                    row, need = pending.pop(0)
                    rids = _reserve_steps(store, row, need)
                    fut = ex.submit(work, dict(row=row, need=need, fps=fps, root=str(store.root), backend=backend))
                    running[fut] = (row, need, rids)
                fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for f in fin:
                    row, need, rids = running.pop(f)
                    try:
                        res = f.result()
                    except BaseException as e:
                        res = dict(ok=False, steps=[[s, 'started'] for s in rids], error=repr(e)[:400], kind='worker_crash')
                    st = _close(store, row, need, rids, res, fps); done[row['forward_key']] = st
                    if not res['ok']:
                        failures.append(dict(forward_key=row['forward_key'], kind=res.get('kind'), error=res.get('error'), steps=st, trace=res.get('trace')))
                        if stop_on_first:
                            stop = True
                    else:
                        log(f"{row['forward_key']} {row['config']} {row['surface']} b{row['block']} s{row['sigma']:g} k{row['noise_k']}: {res['seconds']} s")
    return dict(n_rows=len(rows), n_jobs=len(jobs), reused=reused, done=done, failures=failures, stopped=bool(failures and stop_on_first))


def _init_worker():
    import tr5_bind as BD
    BD.force_imports()


def completeness(rows: list, root, fps: dict) -> dict:
    miss = []; states = {}; n = 0
    for row in rows:
        ctx = MF.context(row); need = needs(root, row, ctx, fps)
        if need['render']:
            miss.append(f"{row['forward_key']}: image ({need['pack_state']})"); continue
        for m, what in need['methods'].items():
            if what != 'none':
                miss.append(f"{row['forward_key']}|{m}: {what}")
            else:
                n += 1
                meta, _, _ = pack_read(root, row['forward_key']); states[f"{row['forward_key']}|{m}"] = meta['results'][m]['state']
    return dict(complete=not miss, missing=miss[:50], n_missing=len(miss), n_results=n, states_count={s: list(states.values()).count(s) for s in set(states.values())})
