# Run rendering, decoding and evaluation with explicit case identities.

from __future__ import annotations

import time
import traceback

import numpy as np

import mp5_common as K
import mp3_chain as CH3

ADAPTER_FILE = 'scripts/surface_revision/model_prep05/mp5_inputs.py'
CHAIN_FILE = 'scripts/surface_revision/model_prep05/mp5_chain.py'
FORWARD_FILES = tuple(CH3.FORWARD_FILES) + (ADAPTER_FILE,)
DECODER_FILES = tuple(CH3.DECODER_FILES) + (ADAPTER_FILE,)
EVAL_FILES = tuple(CH3.EVAL_FILES) + (ADAPTER_FILE,)
IMG_SHAPE = CH3.IMG_SHAPE
_FP: dict = {}


def fingerprints() -> dict:
    if not _FP:
        import rdy_platform as RP
        dg = RP.dom_global()
        fp = lambda fs: {f: K.sha(K.ROOT / f) for f in fs}
        _FP.update(forward=K.digest(fp(FORWARD_FILES)), decode=K.digest(fp(DECODER_FILES)),
                   eval=K.digest(dict(code=fp(EVAL_FILES), global_nodes=int(dg['rec']['n_nodes']), global_hash=dg['hash'], d_mm=6.0, tau_mm=0.15, gross_mm=1.0,
                                      line_identity='A4 complete identity (tr5_chain.full_identity)', support='rdy_figs.node_fields')))
        _FP['execution'] = K.digest(dict(forward=_FP['forward'], decode=_FP['decode'], eval=_FP['eval'], chain=K.sha(K.ROOT / CHAIN_FILE)))
    return _FP


def decode_id(method: str, prep: dict, image_sha: str) -> str:
    return K.digest(dict(method=method, fp=fingerprints()['decode'], forward_key=prep['forward_key'], image=image_sha, design=prep['decoder_row'], peak_f32=prep['forward_record']['peak_f32']))


def eval_id(family: str, did: str) -> str:
    return K.digest(dict(fp=fingerprints()['eval'], family=family, decode_id=did))


class RealBackend:
    name = 'real'

    def __init__(self):
        import mp5_inputs as IN
        self.q = IN.adapter(); self.calls = dict(render=0, decode=0, evaluation=0); self._mesh = {}

    def _m(self, sid):
        import rdy_platform as RP
        if sid not in self._mesh:
            self._mesh[sid] = RP.s2().build_mesh(sid)
        return self._mesh[sid]

    def render(self, prep: dict) -> dict:
        import mp3_forward as F
        self.calls['render'] += 1
        return F.render(self._m(prep['sid']), prep['pattern'], prep['spec'], keep=bool(prep['row'].get('keep_arrays')))

    def decode(self, prep: dict, method: str, img):
        self.calls['decode'] += 1
        return self.q.decode(dict(prep['decoder_row']), method, dict(image=np.asarray(img, np.uint8)))

    def evaluate(self, prep: dict, method: str, A: dict, dmeta: dict):
        self.calls['evaluation'] += 1
        r = dict(prep['decoder_row'], surface=prep['surface'], key=prep['case_id'], _decode_meta=dmeta)
        return self.q.evaluate(r, method, A)


def image_ok(meta: dict, A: dict, prep: dict) -> bool:
    im = meta.get('image') or {}
    return bool(CH3.verify_image(meta, A, prep['forward_key']) and im.get('execution_fp') == fingerprints()['execution'] and im.get('input_key') == prep['input_key'])


def needs(P, prep: dict) -> dict:
    meta, A, st = CH3.pack_read(P, prep['case_id']); methods = prep['row']['methods'].split(';')
    if st != 'read' or not image_ok(meta, A, prep):
        return dict(render=True, methods={m: 'decode' for m in methods}, pack_state=st)
    sha = meta['image']['content_sha']; out = {}
    for m in methods:
        did = decode_id(m, prep, sha)
        v = CH3.verify_result(meta, A, m, prep['family'], did, eval_id(prep['family'], did))
        out[m] = {'valid': 'none', 'eval_stale': 'eval', 'invalid': 'decode'}[v]
    return dict(render=False, methods=out, pack_state=st)


# Run missing rendering, decoding and evaluation stages with explicit case identities.
def work(P, prep: dict, need: dict, backend) -> dict:
    import tr4_gate as G
    import tr5_chain as T5
    cid = prep['case_id']; fam = prep['family']; methods = prep['row']['methods'].split(';'); steps = []; t0 = time.time(); secs = {}
    try:
        meta, A, _ = CH3.pack_read(P, cid)
        if need['render']:
            steps.append(['render', 'started']); t = time.time()
            out = backend.render(prep); img = out['image']
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.shape != IMG_SHAPE:
                raise K.Stop('result_integrity', f'{cid}: render returned {type(img).__name__}')
            fw = {f'fwd__{k}': np.asarray(v) for k, v in (out.get('arrays') or {}).items()}
            meta = dict(case_id=cid, stage=prep['row']['stage'], results={},
                        image=dict(forward_key=prep['forward_key'], execution_fp=fingerprints()['execution'], input_key=prep['input_key'], input=K.clean(prep['input_record']),
                                   content_sha=K.content_sha(img), diag=K.clean(out['diag']), fwd_arrays_digest=K.arrays_digest({k[5:]: v for k, v in fw.items()}), t=K.now()))
            A = dict(image=img, **fw)
            P.save_pack(CH3.pack_path(P, cid), meta, A); steps[-1][1] = 'done'; secs['render'] = round(time.time() - t, 2)
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
                CH3.check_raw(raw, dmeta)
                A = {k: v for k, v in A.items() if not k.startswith(f'{m}__')}
                A.update({f'{m}__raw__{k}': v for k, v in raw.items()})
                meta.setdefault('results', {})[m] = dict(method=m, decode_id=did, image_sha=sha, raw_digest=K.arrays_digest(raw), decode_meta=K.clean(dmeta),
                                                         nominal=K.clean(CH3.nominal_record(m, dmeta)), t_decode=K.now())
                P.save_pack(CH3.pack_path(P, cid), meta, A); steps[-1][1] = 'done'; secs[f'decode:{m}'] = round(time.time() - t, 2)
            r = meta['results'][m]
            steps.append([f'eval:{m}', 'started']); t = time.time()
            ev, rec = backend.evaluate(prep, m, CH3._sub(A, f'{m}__raw__'), r['decode_meta'])
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
            P.save_pack(CH3.pack_path(P, cid), meta, A); steps[-1][1] = 'done'; secs[f'eval:{m}'] = round(time.time() - t, 2)
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


def _close(store, prep: dict, rids: dict, res: dict | None) -> dict:
    meta, A, st = CH3.pack_read(store.P, prep['case_id'])
    img_ok = st == 'read' and image_ok(meta, A, prep)
    steps = {s: v for s, v in ((res or {}).get('steps') or [])}; out = {}
    for step, rid in rids.items():
        if step == 'render':
            ok = img_ok
        else:
            kind, m = step.split(':')
            if img_ok:
                did = decode_id(m, prep, meta['image']['content_sha']); v = CH3.verify_result(meta, A, m, prep['family'], did, eval_id(prep['family'], did))
            else:
                v = 'invalid'
            ok = (v in ('valid', 'eval_stale')) if kind == 'decode' else (v == 'valid')
        status = 'ok' if ok else {'failed': 'failed', 'interrupted': 'interrupted', 'started': 'interrupted', 'done': 'failed'}.get(steps.get(step), 'aborted')
        store.end(rid, status, **({'error': (res or {}).get('error')} if status in ('failed', 'interrupted') else {}))
        out[step] = status
    if img_ok:
        p = CH3.pack_path(store.P, prep['case_id']); store.index_put(K.rel(p), K.sha(p), kind='case_pack', stage=prep['row']['stage'])
    return out


def run_cases(preps: list, store, backend, launch_check, log=print) -> dict:

    launch_check()
    done, failures, reused, timing = {}, [], 0, {}
    t0 = time.time()
    for i, prep in enumerate(preps):
        cid = prep['case_id']; need = needs(store.P, prep)
        if not need['render'] and all(v == 'none' for v in need['methods'].values()):
            reused += 1; done[cid] = 'verified'; continue
        rids = CH3._reserve(store, prep, need)
        try:
            res = work(store.P, prep, need, backend)
        except KeyboardInterrupt:
            st = _close(store, prep, rids, dict(steps=[[s, 'started'] for s in rids], error='KeyboardInterrupt')); store.note('interrupted', case_id=cid, steps=st)
            raise
        st = _close(store, prep, rids, res); done[cid] = st; timing[cid] = res.get('step_seconds')
        if not res['ok']:
            failures.append(dict(case_id=cid, kind=res.get('kind'), error=res.get('error'), steps=st, trace=res.get('trace')))
            log(f'{cid}: FAILED {res.get("error")}'); break
        log(f'[{i + 1}/{len(preps)} {round(time.time() - t0)} s] {cid}: {st} ({res["seconds"]} s)')
    return dict(n=len(preps), reused=reused, done=done, failures=failures, timing=timing, stopped=bool(failures), seconds=round(time.time() - t0, 1))


REPLAY_PICKS = tuple(
    [f'{c}|M10_curv|ph0|s40|r1|BLUR3' for c in ('D_r1_p8x4', 'D_r1_p8x6', 'D_r1.5_p8x4', 'D_r1.5_p8x6', 'L_w0.5_s6.3', 'L_w0.5_s8', 'L_w1.5_s6.3', 'L_w1.5_s8')]
    + [f'{c}|B_curv|ph1|s20|r1|GRAD_Y' for c in ('D_r1_p8x4', 'D_r1_p8x6', 'D_r1.5_p8x4', 'D_r1.5_p8x6', 'L_w0.5_s6.3', 'L_w0.5_s8', 'L_w1.5_s6.3', 'L_w1.5_s8')]
    + ['L_w1.5_s6.3|B_curv|ph1|s20|r1|BLUR3', 'L_w0.5_s8|M10_curv|ph0|s0|r0|BLUR3', 'D_r1.5_p8x4|M10_curv|ph0|s20|r1|BLUR3', 'L_w0.5_s8|M24_curv|ph1|s40|r1|UNIFORM'])
REPLAY_NOTE = ('16 rasters = 8 configurations x the two registered phases (ph0 M10 sigma 40 BLUR3, ph1 B sigma 20 GRAD_Y incl. the known edge wrong-id case '
               'L_w1.5_s6.3 GRAD_Y) + the second edge case (BLUR3), a sigma-0 line without detection, a DF no-output dot case, a UNIFORM line case')


class _MP3Data:
    data = K.MP3_OUT / 'data'


def replay(store, picks=REPLAY_PICKS) -> dict:
    import mp5_inputs as IN
    import tr4_gate as G
    import tr5_chain as T5
    if len(picks) > K.REPLAY_MAX_IMAGES:
        raise K.Stop('budget', f'replay: {len(picks)} images > {K.REPLAY_MAX_IMAGES}')
    M = K.read_csv(K.MP3_OUT / 'manifest.csv').set_index('case_id'); q = IN.adapter(); out = {}
    for cid in picks:
        r = M.loc[cid].to_dict(); fam = r['family']
        meta, A, st = CH3.pack_read(_MP3Data, cid)
        if st != 'read':
            raise K.Stop('technical', f'replay {cid}: stored pack {st}')
        src = IN.template_source(r['cand_id'])
        ph = dict(u=float(src['phase_u']), v=float(src['phase_v'])) if fam == 'point' else dict(phi=float(src['phase_frac']))
        inp = IN.inputs_for(r['cand_id'], fam, ph); ik = IN.input_key(r['cand_id'], fam, ph, inp); q.register(ik, inp)
        rk_ok = inp['raster_key'] == r['raster_key'] and float(np.float32(inp['peak'])) == float(r['peak_f32'])
        drow = dict(input_key=ik, radius_mm=(float(r['radius_mm']) if fam == 'point' else float('nan')), family=fam)
        img = A['image']; rec = {}
        for m in r['methods'].split(';'):
            old = meta['results'][m]
            rid = store.reserve(('replay_decode',), f'{cid}|{m}', step='replay_decode', method=m)
            try:
                raw, dmeta = q.decode(dict(drow), m, dict(image=np.asarray(img, np.uint8)))
                raw = {k: np.asarray(v) for k, v in raw.items()}
                dec_same = K.arrays_digest(raw) == old['raw_digest'] and K.canon(K.clean(dmeta)) == K.canon(old['decode_meta'])
                store.end(rid, 'ok')
            except BaseException as e:
                store.end(rid, 'failed', error=repr(e)[:300]); raise
            rid = store.reserve(('replay_evaluation',), f'{cid}|{m}', step='replay_evaluation', method=m)
            try:
                ev, er = q.evaluate(dict(drow, surface=r['surface'], key=cid, _decode_meta=dmeta), m, raw)
                ev = {k: np.asarray(v) for k, v in ev.items()}
                pc = G.check_derived(ev, fam); G.check_metrics(er['metrics'], pc)
                farr, finfo = T5.full_identity(dict(family=fam, surface=r['surface'], forward_key=cid), dict(meta_full=inp['meta_full'], binary=inp['binary']), ev)
                ev.update(farr)
                ev_same = K.arrays_digest(ev) == old['eval_digest']
                met_same = K.canon(K.clean(er['metrics'])) == K.canon(old['metrics']) and K.canon(K.clean(er.get('support'))) == K.canon(old['support'])
                fid_same = K.canon(K.clean(finfo)) == K.canon(old['full_identity'])
                store.end(rid, 'ok')
            except BaseException as e:
                store.end(rid, 'failed', error=repr(e)[:300]); raise
            rec[m] = dict(decode_arrays_and_meta=dec_same, eval_arrays=ev_same, metrics_support=met_same, full_identity=fid_same, state=er['metrics'].get('state'))
        out[cid] = dict(raster_peak=rk_ok, methods=rec)
    ok = all(v['raster_peak'] and all(all(x for k, x in mm.items() if k != 'state') for mm in v['methods'].values()) for v in out.values())
    n_dec = sum(len(v['methods']) for v in out.values())
    return dict(ok=ok, n_images=len(out), n_decodes=n_dec, n_evaluations=n_dec, cases=out, note=REPLAY_NOTE)
