# Store complete rendering, decoding and evaluation records for joint cases.

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

import s5r_common as K

DEC_DEPS = {
    'D0': ['scripts/spotnoise_round5/r5_detect.py', 'src/m4_reconstruction.py', 'scripts/surface_revision/surf02/s2_common.py', 'scripts/surface_revision/sections_adapter/adp_surf02.py',
           'scripts/surface_revision/s345_ready/rdy_decode.py'],
    'DF': ['scripts/spotnoise_round5/r5_detect.py', 'src/m4_reconstruction.py', 'scripts/surface_revision/surf02/s2_common.py', 'scripts/surface_revision/s3_code01/s3c_dev.py',
           'scripts/surface_revision/s3_code01/s3c_common.py', 'scripts/surface_revision/s345_ready/rdy_decode.py'],
    'P4': ['src/m4_lines.py', 'scripts/surface_revision/surf02/s2_common.py', 'scripts/surface_revision/sections_adapter/adp_surf02.py', 'scripts/surface_revision/s345_ready/rdy_decode.py']}
EVAL_DEPS = ['scripts/surface_revision/sections_adapter/s1r_eval.py', 'scripts/surface_revision/sections_adapter/s1r_common.py', 'scripts/surface_revision/surf02/s2_common.py',
             'src/surface_eval_r3.py', 'scripts/surface_revision/s345_ready/rdy_eval.py']
RENDER_DEPS = ['src/m3_virtual_camera.py', 'scripts/surface_revision/sections_adapter/adp_surf02.py', 'scripts/surface_revision/s345_ready/rdy_inputs.py',
               'scripts/surface_revision/sections_adapter/adp_patterns.py']
CRITERIA = dict(d_mm=6.0, tau_mm=0.15, gross_mm=1.0, global_nodes=30947, local_nodes=22605, local_rule='SURF-02 roi r <= 15 mm')
DF_PARAMS = dict(front_end='DF', rule='s3c_dev.dot_decode(k=1, rule DF, path binding)', floor_factor=0.5, nominal_mm_per_px=0.18, smoothing=None)
_FH: dict = {}


def fhash(p: str) -> str:
    if p not in _FH:
        _FH[p] = K.sha(K.ROOT / p)
    return _FH[p]


def deps(files) -> dict:
    return {f: fhash(f) for f in files}


def platform_refs() -> dict:
    fz = K.load_json(K.reg(K.S345 / 'freeze_s345_ready.json'))
    man = K.SR / 'surf02' / 'release' / 'dependencies_surf02_v1_1.json'
    gd = K.load_json(K.reg(K.SR / 's1_real01' / 'domain' / 'global_domain_step0p50.json'))
    return dict(mesh=fz['meshes'], release_manifest_sha=K.sha(K.reg(man)), global_domain_hash=gd['hash'])


def image_identity(f: dict, pr: dict) -> str:

    return 'I-' + K.digest(dict(raster=f['raster'], peak_f32=float(f['peak_f32']), surface=f['surface'], mesh=pr['mesh'][f['surface']], psf=float(f['psf']), sigma=float(f['sigma']),
                                 seed=f['seed'], m3=f['m3'], platform=f['platform'], release=pr['release_manifest_sha'], render_code=deps(RENDER_DEPS)), 24)


def method_params(method: str, peak_f32: float, radius_mm=None) -> dict:
    if method == 'D0':
        return dict(front_end='D0', call='adp_surf02.decode("D0")', cfg='s2_common.dot_cfg (window -16 / 4 mm)')
    if method == 'DF':
        return dict(DF_PARAMS, radius_mm=float(radius_mm), area_floor_px2=0.5 * np.pi * (float(radius_mm) / 0.18) ** 2, cfg='s2_common.dot_cfg (window -16 / 4 mm)')
    if method == 'P4':
        return dict(method='P4_C3_cal', call='adp_surf02.decode("P4_C3_cal")', cfg='s2_common.line_cfg(actual peak)', peak_f32=float(peak_f32))
    raise K.Stop('scope', f'unknown method {method!r}')


def decode_identity(img_id: str, img_sha: str, method: str, params: dict) -> str:
    return 'D-' + K.digest(dict(image=img_id, image_content=img_sha, method=method, params=params, code=deps(DEC_DEPS[method])), 24)


def eval_identity(dec_content: str, pr: dict) -> str:
    return 'E-' + K.digest(dict(decode_content=dec_content, code=deps(EVAL_DEPS), global_domain=pr['global_domain_hash'], release=pr['release_manifest_sha'], criteria=CRITERIA), 24)


def analysis_identity(consumed: list, pairing: list, rules: dict, code: dict) -> str:
    return 'A-' + K.digest(dict(consumed=sorted(map(tuple, consumed)), pairing=pairing, rules=rules, code=code), 24)


def content_digest(arrays: dict) -> str:
    h = []
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.append(f'{k}|{a.dtype.str}|{a.shape}|' + K.sha_bytes(a.tobytes()))
    return K.sha_bytes('\n'.join(h).encode())


# Store rendering, decoding and evaluation outputs with checked stage identities.
class Store:

    STAGES = ('image', 'decode', 'eval')

    def __init__(self, root: Path, base: Path = K.OUT):
        self.root = Path(root); self.base = base

    def _p(self, stage, key):
        return self.root / stage / f'{key}.json'

    def status(self, stage: str, key: str, identity: str) -> tuple:
        p = self._p(stage, key)
        if not p.exists():
            half = list(p.parent.glob(f'{key}.*.tmp*')) if p.parent.exists() else []
            return ('todo', 'half-written leftovers ignored' if half else 'absent')
        try:
            r = K.load_json(p)
        except Exception:
            return ('corrupt', 'record json unreadable')
        if r.get('complete') is not True:
            return ('todo', 'record incomplete')
        z = p.with_suffix('.npz')
        if not z.exists() or K.sha(z) != r.get('npz_sha256'):
            return ('corrupt', 'npz missing or hash differs from the record')
        if r.get('identity') != identity:
            return ('mismatch', f"stored identity {r.get('identity')} != current {identity}")
        return ('reuse', 'identity and content match')

    def commit(self, stage: str, key: str, identity: str, arrays: dict, meta: dict) -> str:
        p = self._p(stage, key)
        def w(t):
            with open(t, 'wb') as fh:
                np.savez_compressed(fh, **arrays)
        K.atomic(p.with_suffix('.npz'), w, self.base)
        h = K.sha(p.with_suffix('.npz'))
        K.save_json(p, dict(identity=identity, stage=stage, key=key, npz_sha256=h, content=content_digest(arrays), meta=meta, complete=True, t=K.now()), self.base)
        return h

    def content(self, stage: str, key: str) -> str:
        return K.load_json(self._p(stage, key))['content']

    def arrays(self, stage: str, key: str) -> dict:
        return K.load_npz(self._p(stage, key).with_suffix('.npz'))

    def lock(self):
        store = self

        class _L:
            def __enter__(self_):
                p = K.guard(store.root / 'run.lock', store.base)
                try:
                    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    raise K.Stop('lock', f'{K.rel(p)} exists: another run is active (or crashed; inspect before removing)') from None
                os.write(fd, f'{os.getpid()} {K.now()}'.encode()); os.close(fd); self_.p = p
                return store

            def __exit__(self_, *exc):
                try:
                    os.remove(self_.p)
                except OSError:
                    pass
                return False
        return _L()

    def set_status(self, state: str, **kw) -> None:
        K.save_json(self.root / 'status.json', dict(state=state, t=K.now(), **kw), self.base)


class Forbidden:

    def __init__(self):
        self.calls = []

    def _no(self, what, row):
        self.calls.append(what)
        raise K.Stop('forbidden', f'{what} requested for {row.get("manifest_key", row.get("key"))}: not authorised in this round')

    def render(self, row):
        self._no('render', row)

    def decode(self, row, method, img):
        self._no(f'decode {method}', row)

    def evaluate(self, row, method, dec):
        self._no(f'evaluate {method}', row)


class Stub:

    def __init__(self, fail_after=None):
        self.calls = []; self.fail_after = fail_after

    def _tick(self, what):
        self.calls.append(what)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError(f'injected failure at call {len(self.calls)} ({what})')

    def render(self, row):
        self._tick('render'); rng = np.random.default_rng(int(K.digest(row['manifest_key'], 8), 16))
        return dict(image=rng.integers(0, 255, (8, 8), dtype=np.uint8)), dict(stub=True)

    def decode(self, row, method, img):
        self._tick(f'decode {method}')
        return dict(pts=np.asarray(img['image'], np.float64)[:3, :3].reshape(-1, 3), accepted=np.ones(3, bool)), dict(stub=True)

    def evaluate(self, row, method, dec):
        self._tick(f'evaluate {method}')
        return dict(valid=np.asarray(dec['accepted'], bool)), dict(stub=True, metrics=dict(state='ok', n_valid=3))


class Real:

    def __init__(self, permit: dict | None):
        if not permit or not permit.get('task_book') or not permit.get('caps'):
            raise K.Stop('permit', 'real imaging / decoding needs a permit of a new task book (pending: next round, after review)')
        self.permit = permit; self.calls = []

    def _inp(self, row):
        import rdy_inputs as IN
        return IN.build(dict(cand_id=row['cand_id_sigma0'], section='S5', family=row['family'], view='S5_size_spacing', strategy=''))

    def render(self, row):
        import rdy_platform as P
        inp = self._inp(row); bd = P.bound(); seed = row['seed']
        res = bd.call('render', bd.sid(row['surface']), inp['pattern'], float(inp['peak']), float(row['sigma']), int(seed or 0), K.PSF_PX)
        self.calls.append('render')
        return dict(image=np.asarray(res['image'], np.uint8)), dict(meta=K.clean(res.get('meta', {})))

    def decode(self, row, method, img):
        import rdy_decode as DC
        inp = self._inp(row); self.calls.append(f'decode {method}')
        if method == 'DF':
            import s3c_dev as DV
            D = DV.dot_decode(img['image'], np.asarray(inp['binary'], np.float32), 1, [float(row['radius_mm'])], 'DF', 'binding')
            out = dict(method='D0', arms={'DF': DV.dot_dec(D, 'single')}, shared={}, window=(-16.0, 4.0))
        else:
            out = DC.decode(method, img['image'], inp, dict(section='S5', radius_mm=row['radius_mm'], arm=('D0' if method == 'D0' else 'P4_C3_cal')))
        A, meta = DC.pack(out)
        return A, dict(pack_meta=meta, method_label=method)

    def evaluate(self, row, method, dec_arrays):
        import rdy_decode as DC
        import rdy_eval as EV
        meta = row['_decode_meta']; U = DC.unpack(dec_arrays, meta['pack_meta']); arm = next(iter(U['arms'].values()))
        r = dict(section='S5', family=row['family'], surface=row['surface'], radius_mm=row['radius_mm'])
        m, arrays, info = EV.evaluate_common(r, arm, EV.ctx_eval(r, self._inp(row)))
        self.calls.append(f'evaluate {method}')
        return {k: np.asarray(v) for k, v in arrays.items() if not isinstance(v, dict)}, dict(metrics=K.clean(m), info=K.clean(info))


def run_rows(store: Store, rows: list, ex, pr: dict, methods_of) -> dict:

    log = dict(reuse=0, computed=0, stages={})
    with store.lock():
        store.set_status('running', n_rows=len(rows))
        try:
            for row in rows:
                k = row['key']; iid = image_identity(row['f'], pr)
                st, why = store.status('image', k, iid)
                if st in ('mismatch', 'corrupt'):
                    raise K.Stop(st, f'image {k}: {why}')
                if st == 'todo':
                    a, m = ex.render(row); store.commit('image', k, iid, a, m); log['computed'] += 1
                else:
                    log['reuse'] += 1
                log['stages'][f'image|{k}'] = st
                isha = store.content('image', k)
                for method in methods_of(row):
                    did = decode_identity(iid, isha, method, method_params(method, row['f']['peak_f32'], row.get('radius_mm')))
                    dk = f'{k}__{method}'; st, why = store.status('decode', dk, did)
                    if st in ('mismatch', 'corrupt'):
                        raise K.Stop(st, f'decode {dk}: {why}')
                    if st == 'todo':
                        a, m = ex.decode(row, method, store.arrays('image', k)); store.commit('decode', dk, did, a, m); log['computed'] += 1
                    else:
                        log['reuse'] += 1
                    log['stages'][f'decode|{dk}'] = st
                    eid = eval_identity(store.content('decode', dk), pr); st, why = store.status('eval', dk, eid)
                    if st in ('mismatch', 'corrupt'):
                        raise K.Stop(st, f'eval {dk}: {why}')
                    if st == 'todo':
                        row['_decode_meta'] = K.load_json(store._p('decode', dk))['meta']
                        a, m = ex.evaluate(row, method, store.arrays('decode', dk)); store.commit('eval', dk, eid, a, m); log['computed'] += 1
                    else:
                        log['reuse'] += 1
                    log['stages'][f'eval|{dk}'] = st
            store.set_status('complete', **{k: v for k, v in log.items() if k != 'stages'})
        except K.Stop as e:
            store.set_status('failed', code=e.code, reason=e.msg[:300]); raise
        except Exception as e:
            store.set_status('failed', code='technical', reason=repr(e)[:300])
            raise K.Stop('technical', repr(e)[:300]) from e
    return log


def batch_declare(store: Store, name: str, keys: list) -> dict:

    ent = {k: K.load_json(store._p('eval', k))['npz_sha256'] for k in sorted(keys)}
    rec = dict(name=name, keys=sorted(keys), keys_digest=K.digest(sorted(keys), 32), eval_npz=ent, complete=True, t=K.now())
    K.save_json(store.root / f'batch_{name}.json', rec, store.base)
    return rec


def batch_verify(store: Store, name: str, declared_keys: list | None = None) -> dict:

    p = store.root / f'batch_{name}.json'
    if not p.exists():
        raise K.Stop('missing', f'batch {name} not declared')
    b = K.load_json(p); bad = []
    if declared_keys is not None and sorted(declared_keys) != b['keys']:
        bad.append('declared key set differs from the expected key set')
    if K.digest(b['keys'], 32) != b['keys_digest']:
        bad.append('key list edited after declaration')
    for k, h in b['eval_npz'].items():
        q = store._p('eval', k).with_suffix('.npz')
        if not q.exists() or K.sha(q) != h:
            bad.append(f'member {k} missing or changed')
    if bad:
        raise K.Stop('corrupt', f'batch {name} refused', detail=bad)
    return dict(ok=True, n=len(b['keys']))


def wait_free(p: Path, timeout=5.0) -> None:
    t0 = time.time()
    while p.exists() and time.time() - t0 < timeout:
        time.sleep(0.05)
