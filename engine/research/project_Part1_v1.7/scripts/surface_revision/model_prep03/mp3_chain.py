# Prepare, render and evaluate environment cases through the registered processing chain.

from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np

import mp3_common as K
import mp3_forward as F

IMG_SHAPE = (512, 512)
DECODER_FILES = ('src/m4_reconstruction.py', 'src/m4_lines.py', 'scripts/spotnoise_round5/r5_detect.py', 'scripts/surface_revision/s3_code01/s3c_dev.py',
                 'scripts/surface_revision/s3_code01/s3c_common.py', 'scripts/surface_revision/s345_ready/rdy_decode.py', 'scripts/surface_revision/s345_ready/rdy_inputs.py',
                 'scripts/surface_revision/s345_ready/rdy_common.py', 'scripts/surface_revision/s345_ready/rdy_platform.py', 'scripts/surface_revision/sections_adapter/adp_surf02.py',
                 'scripts/surface_revision/sections_adapter/adp_binding.py', 'scripts/surface_revision/sections_adapter/adp_patterns.py', 'scripts/surface_revision/surf02/s2_common.py',
                 'scripts/surface_revision/s5_align04/s5t_platform.py')
EVAL_FILES = ('scripts/surface_revision/s345_ready/rdy_eval.py', 'scripts/surface_revision/s345_ready/rdy_figs.py', 'scripts/surface_revision/sections_adapter/s1r_eval.py',
              'scripts/surface_revision/sections_adapter/s1r_common.py', 'src/surface_eval_r3.py', 'scripts/surface_revision/surf02/s2_common.py', 'scripts/line01/l01_truth.py',
              'scripts/surface_revision/thesis_reinforce05/tr5_chain.py', 'scripts/surface_revision/thesis_reinforce01/tr_a4.py', 'scripts/surface_revision/s5_align04/s5t_platform.py')
FORWARD_FILES = ('src/m3_virtual_camera.py', 'scripts/surface_revision/model_prep02/mp2_forward.py', 'scripts/surface_revision/model_prep03/mp3_forward.py',
                 'scripts/surface_revision/surf02/s2_common.py', 'scripts/surface_revision/surf01/s1_model.py', 'scripts/pm014/pm014_surface.py', 'scripts/line01/l01_common.py')


def _fp(files) -> dict:
    return {f: K.sha(K.ROOT / f) for f in files}


_FP: dict = {}


def fingerprints() -> dict:
    if not _FP:
        import rdy_platform as RP
        dg = RP.dom_global()
        _FP.update(forward=K.digest(_fp(FORWARD_FILES)), decode=K.digest(_fp(DECODER_FILES)),
                   eval=K.digest(dict(code=_fp(EVAL_FILES), global_nodes=int(dg['rec']['n_nodes']), global_hash=dg['hash'], d_mm=6.0, tau_mm=0.15, gross_mm=1.0,
                                      line_identity='A4 complete identity (tr5_chain.full_identity)', support='rdy_figs.node_fields')))
    return _FP


_QX = {}


def realq():

    if 'q' not in _QX:
        p = str(K.SR / 's5_align04')
        if p not in sys.path:
            sys.path.insert(0, p)
        import s5t_platform as PF
        q = PF.RealQ()
        sb = K.P0.sandbox
        for k, sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
            os.environ[k] = str(sb / sub)
        os.environ['LINE01RB_RENDER_CAP'] = '0'
        _QX['q'] = q; _QX['PF'] = PF
    return _QX['q']


def prepare(row: dict) -> dict:

    import rdy_platform as RP
    q = realq(); S = RP.s2()
    inp = q.inputs(dict(cand_id_sigma0=row['cand_id'], family=row['family']))
    pat = np.asarray(inp['pattern'], np.float64)
    chk = dict(raster=inp['raster_key'] == row['raster_key'], peak_f32=float(np.float32(inp['peak'])) == float(row['peak_f32']), lit_px=float(inp['lit_px']) == float(row['lit_px']),
               integral=abs(float(pat.sum()) - float(row['integral'])) <= 1e-6, peak_le_1=float(inp['peak']) <= 1.0,
               peak_rule=abs(float(inp['peak']) - min(1.0, 11695.0 / float(inp['lit_px']))) <= 1e-9 * max(1.0, min(1.0, 11695.0 / float(inp['lit_px']))),
               integral_cap=float(pat.sum()) <= 11695.0 * (1.0 + 2.0 ** -23))
    if not all(chk.values()):
        raise K.Stop('preflight', f"{row['case_id']}: inherited raster / resource check failed {chk}")
    seed = None if (row['seed'] is None or (isinstance(row['seed'], float) and np.isnan(row['seed']))) else int(row['seed'])
    ns = F.noise_spec(row['sigma'], seed, 'compat')
    base = dict(S.m3_cfg(float(row['sigma']), int(ns['m3_seed'])))
    import mp3_design as D
    spec = F.forward_spec(base, float(row['psf_true_px']), D.field_of(row['env']), ns)
    import m3_virtual_camera as M3
    mesh = S.build_mesh(row['sid'])
    fwd = dict(sid=row['sid'], mesh_sha16=S.mesh_sha(mesh), raster_key=inp['raster_key'], pattern=K.content_sha(np.asarray(inp['pattern'], np.float32)), peak_f32=float(np.float32(inp['peak'])),
               integral=float(pat.sum()), spec=spec, platform=S.VERSION, m3_file=K.sha(M3.__file__), adapter_file=K.sha(K.HERE / 'mp3_forward.py'))
    return dict(row=row, case_id=row['case_id'], family=row['family'], sid=row['sid'], surface=row['surface'], pattern=np.asarray(inp['pattern'], np.float32), spec=spec, noise=ns,
                forward_key=K.digest(fwd), forward_record=fwd, resource=chk, ctx=dict(meta_full=inp['meta_full'], binary=inp['binary']))


def decoder_row(prep: dict) -> dict:
    r = prep['row']
    return dict(cand_id_sigma0=r['cand_id'], radius_mm=r['radius_mm'], family=r['family'])


def decode_id(method: str, prep: dict, image_sha: str) -> str:
    return K.digest(dict(method=method, fp=fingerprints()['decode'], forward_key=prep['forward_key'], image=image_sha, design=decoder_row(prep), peak_f32=prep['forward_record']['peak_f32']))


def eval_id(family: str, did: str) -> str:
    return K.digest(dict(fp=fingerprints()['eval'], family=family, decode_id=did))


class RealBackend:
    name = 'real'

    def __init__(self):
        self.q = realq(); self.calls = dict(render=0, decode=0, evaluation=0)

    def render(self, prep: dict) -> dict:
        import rdy_platform as RP
        self.calls['render'] += 1
        return F.render(RP.s2().build_mesh(prep['sid']), prep['pattern'], prep['spec'], keep=bool(prep['row']['keep_arrays']))

    def decode(self, prep: dict, method: str, img):
        self.calls['decode'] += 1
        return self.q.decode(decoder_row(prep), method, dict(image=np.asarray(img, np.uint8)))

    def evaluate(self, prep: dict, method: str, A: dict, dmeta: dict):
        self.calls['evaluation'] += 1
        r = dict(decoder_row(prep), surface=prep['surface'], key=prep['case_id'], _decode_meta=dmeta)
        return self.q.evaluate(r, method, A)


def nominal_record(method: str, dmeta: dict) -> dict:

    pm = dmeta.get('pack_meta') or {}
    if method == 'P4':
        di = (pm.get('a0_r') or {}).get('detect_info') or {}
        return dict(window=pm.get('window'), steger_threshold_design=di.get('steger_threshold_design'), noise_sigma_hat_image=di.get('noise_sigma_hat'),
                    steger_scale_px=di.get('steger_scale_effective'), noise_status=di.get('noise_status'))
    return dict(window=pm.get('window'), otsu_threshold=(pm.get('cd') or {}).get('thresh'), a_floor_px2=dmeta.get('a_floor_px2'), n_floor=dmeta.get('n_floor'))


def pack_path(P: K.Paths, cid: str):
    return P.data / (cid.replace('|', '__') + '.npz')


def pack_read(P: K.Paths, cid: str):
    p = pack_path(P, cid)
    if not p.exists():
        return dict(case_id=cid, results={}), {}, 'missing'
    try:
        meta, A = K.load_pack(p)
    except Exception as e:
        return dict(case_id=cid, results={}, error=repr(e)[:200]), {}, 'unreadable'
    return meta, A, 'read'


def _sub(A: dict, prefix: str) -> dict:
    n = len(prefix)
    return {k[n:]: v for k, v in A.items() if k.startswith(prefix)}


def verify_image(meta: dict, A: dict, fk: str) -> bool:
    im = meta.get('image') or {}
    if 'image' not in A or im.get('forward_key') != fk:
        return False
    a = A['image']
    if not (a.dtype == np.uint8 and a.shape == IMG_SHAPE and K.content_sha(a) == im.get('content_sha')):
        return False
    return K.arrays_digest(_sub(A, 'fwd__')) == im.get('fwd_arrays_digest')


def check_raw(A: dict, dmeta: dict) -> dict:

    import rdy_decode as DC
    import tr4_gate as G
    U = DC.unpack(A, dmeta['pack_meta'])
    if len(U['arms']) != 1:
        raise G.Defect(f"decode record holds {len(U['arms'])} arms")
    arm = next(iter(U['arms'].values()))
    return G.check_raw(dict(pts=np.asarray(arm['pts']), accepted=np.asarray(arm['accepted'])))


def verify_result(meta: dict, A: dict, m: str, family: str, did: str, eid: str) -> str:
    import tr4_gate as G
    import tr5_chain as T5
    r = (meta.get('results') or {}).get(m)
    if not r or r.get('decode_id') != did:
        return 'invalid'
    raw = _sub(A, f'{m}__raw__')
    if not raw or K.arrays_digest(raw) != r.get('raw_digest'):
        return 'invalid'
    try:
        check_raw(raw, r['decode_meta'])
    except (G.Defect, KeyError, ValueError, TypeError):
        return 'invalid'
    ev = _sub(A, f'{m}__ev__')
    if r.get('eval_id') != eid or not ev or K.arrays_digest(ev) != r.get('eval_digest'):
        return 'eval_stale'
    try:
        pc = G.check_derived(ev, family); G.check_metrics(r.get('metrics'), pc)
        if family == 'line':
            T5.check_full(ev, r.get('full_identity') or {})
    except (G.Defect, K.Stop):
        return 'eval_stale'
    return 'valid'


def needs(P: K.Paths, prep: dict) -> dict:
    meta, A, st = pack_read(P, prep['case_id']); fk = prep['forward_key']; methods = prep['row']['methods'].split(';')
    if st != 'read' or not verify_image(meta, A, fk):
        return dict(render=True, methods={m: 'decode' for m in methods}, pack_state=st)
    sha = meta['image']['content_sha']; out = {}
    for m in methods:
        did = decode_id(m, prep, sha)
        v = verify_result(meta, A, m, prep['family'], did, eval_id(prep['family'], did))
        out[m] = {'valid': 'none', 'eval_stale': 'eval', 'invalid': 'decode'}[v]
    return dict(render=False, methods=out, pack_state=st)


# Run missing rendering, decoding and evaluation stages after checking their identities.
def work(P: K.Paths, prep: dict, need: dict, backend) -> dict:
    import tr4_gate as G
    import tr5_chain as T5
    cid = prep['case_id']; fam = prep['family']; methods = prep['row']['methods'].split(';'); steps = []; t0 = time.time(); secs = {}
    try:
        meta, A, _ = pack_read(P, cid)
        if need['render']:
            steps.append(['render', 'started']); t = time.time()
            out = backend.render(prep); img = out['image']
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.shape != IMG_SHAPE:
                raise K.Stop('result_integrity', f'{cid}: render returned {type(img).__name__}')
            fw = {f'fwd__{k}': np.asarray(v) for k, v in (out.get('arrays') or {}).items()}
            meta = dict(case_id=cid, stage=prep['row']['stage'], results={}, image=dict(forward_key=prep['forward_key'], content_sha=K.content_sha(img), diag=K.clean(out['diag']),
                                                                                       fwd_arrays_digest=K.arrays_digest({k[5:]: v for k, v in fw.items()}), t=K.now()))
            A = dict(image=img, **fw)
            P.save_pack(pack_path(P, cid), meta, A); steps[-1][1] = 'done'; secs['render'] = round(time.time() - t, 2)
        img = A['image']; sha = meta['image']['content_sha']
        for m in methods:
            what = need['methods'].get(m, 'none')
            if what == 'none':
                continue
            did = decode_id(m, prep, sha); eid = eval_id(fam, did)
            if what == 'decode':
                steps.append([f'decode:{m}', 'started']); t = time.time()
                res = backend.decode(prep, m, img)
                if not isinstance(res, tuple) or len(res) != 2 or not isinstance(res[0], dict) or not isinstance(res[1], dict):
                    raise K.Stop('result_integrity', f'{cid} {m}: decoder returned {type(res).__name__}')
                raw, dmeta = res
                raw = {k: np.asarray(v) for k, v in raw.items()}
                check_raw(raw, dmeta)
                A = {k: v for k, v in A.items() if not k.startswith(f'{m}__')}
                A.update({f'{m}__raw__{k}': v for k, v in raw.items()})
                meta.setdefault('results', {})[m] = dict(method=m, decode_id=did, image_sha=sha, raw_digest=K.arrays_digest(raw), decode_meta=K.clean(dmeta),
                                                         nominal=K.clean(nominal_record(m, dmeta)), t_decode=K.now())
                P.save_pack(pack_path(P, cid), meta, A); steps[-1][1] = 'done'; secs[f'decode:{m}'] = round(time.time() - t, 2)
            r = meta['results'][m]
            steps.append([f'eval:{m}', 'started']); t = time.time()
            ev, rec = backend.evaluate(prep, m, _sub(A, f'{m}__raw__'), r['decode_meta'])
            ev = {k: np.asarray(v) for k, v in ev.items()}
            mtr = rec['metrics']; pc = G.check_derived(ev, fam); mi = G.check_metrics(mtr, pc)
            farr, finfo = T5.full_identity(dict(family=fam, surface=prep['surface'], forward_key=cid), prep['ctx'], ev)
            ev.update(farr)
            if fam == 'line':
                T5.check_full(ev, finfo)
            A = {k: v for k, v in A.items() if not k.startswith(f'{m}__ev__')}
            A.update({f'{m}__ev__{k}': v for k, v in ev.items()})
            r.update(eval_id=eid, eval_digest=K.arrays_digest(ev), metrics=K.clean(mtr), support=K.clean(rec.get('support')), metric_info=mi, state=mtr['state'],
                     full_identity=K.clean(finfo), info=K.clean({k: v for k, v in (rec.get('info') or {}).items() if k in ('component_reasons', 'selection_codes', 'det_info', 'detect_info', 'counts')}),
                     t_eval=K.now())
            meta['results'][m] = r
            P.save_pack(pack_path(P, cid), meta, A); steps[-1][1] = 'done'; secs[f'eval:{m}'] = round(time.time() - t, 2)
        return dict(case_id=cid, ok=True, steps=steps, seconds=round(time.time() - t0, 2), step_seconds=secs)
    except KeyboardInterrupt:
        if steps and steps[-1][1] == 'started':
            steps[-1][1] = 'interrupted'
        raise
    except (Exception, G.Defect) as e:
        if steps and steps[-1][1] == 'started':
            steps[-1][1] = 'failed'
        return dict(case_id=cid, ok=False, steps=steps, error=repr(e)[:400], kind=(e.code if isinstance(e, K.Stop) else ('integrity' if isinstance(e, G.Defect) else 'error')),
                    trace=traceback.format_exc()[-1500:], step_seconds=secs)


def _groups(prep: dict, step: str, m: str | None = None) -> tuple:
    if step == 'render':
        return ('forward', 'forward_entry') if prep['row']['stage'] == 'entry' else ('forward',)
    if step == 'decode':
        return ('decode', f'decode_{m}')
    return ('evaluation',)


def _reserve(store: K.Store, prep: dict, need: dict) -> dict:
    cid = prep['case_id']; rids = {}
    if need['render']:
        rids['render'] = store.reserve(_groups(prep, 'render'), cid, step='render', forward_key=prep['forward_key'])
    for m, what in need['methods'].items():
        if what == 'decode':
            rids[f'decode:{m}'] = store.reserve(_groups(prep, 'decode', m), f'{cid}|{m}', step='decode', method=m)
        if what in ('decode', 'eval'):
            rids[f'eval:{m}'] = store.reserve(_groups(prep, 'eval'), f'{cid}|{m}', step='eval', method=m)
    return rids


def _close(store: K.Store, prep: dict, rids: dict, res: dict | None) -> dict:

    meta, A, st = pack_read(store.P, prep['case_id'])
    img_ok = st == 'read' and verify_image(meta, A, prep['forward_key'])
    steps = {s: v for s, v in ((res or {}).get('steps') or [])}; out = {}
    for step, rid in rids.items():
        if step == 'render':
            ok = img_ok
        else:
            kind, m = step.split(':')
            if img_ok:
                did = decode_id(m, prep, meta['image']['content_sha']); v = verify_result(meta, A, m, prep['family'], did, eval_id(prep['family'], did))
            else:
                v = 'invalid'
            ok = (v in ('valid', 'eval_stale')) if kind == 'decode' else (v == 'valid')
        status = 'ok' if ok else {'failed': 'failed', 'interrupted': 'interrupted', 'started': 'interrupted', 'done': 'failed'}.get(steps.get(step), 'aborted')
        store.end(rid, status, **({'error': (res or {}).get('error')} if status in ('failed', 'interrupted') else {}))
        out[step] = status
    if img_ok:
        p = pack_path(store.P, prep['case_id']); store.index_put(K.rel(p), K.sha(p), kind='case_pack', stage=prep['row']['stage'])
    return out


def run_cases(preps: list, store: K.Store, backend, launch_check, log=print) -> dict:

    launch_check()
    done, failures, reused, timing = {}, [], 0, {}
    for prep in preps:
        cid = prep['case_id']; need = needs(store.P, prep)
        if not need['render'] and all(v == 'none' for v in need['methods'].values()):
            reused += 1; done[cid] = 'verified'; continue
        rids = _reserve(store, prep, need)
        try:
            res = work(store.P, prep, need, backend)
        except KeyboardInterrupt:
            st = _close(store, prep, rids, dict(steps=[[s, 'started'] for s in rids], error='KeyboardInterrupt')); store.note('interrupted', case_id=cid, steps=st)
            raise
        st = _close(store, prep, rids, res); done[cid] = st; timing[cid] = res.get('step_seconds')
        if not res['ok']:
            failures.append(dict(case_id=cid, kind=res.get('kind'), error=res.get('error'), steps=st, trace=res.get('trace')))
            log(f'{cid}: FAILED {res.get("error")}'); break
        log(f'{cid}: {st} ({res["seconds"]} s)')
    return dict(n=len(preps), reused=reused, done=done, failures=failures, timing=timing, stopped=bool(failures))
