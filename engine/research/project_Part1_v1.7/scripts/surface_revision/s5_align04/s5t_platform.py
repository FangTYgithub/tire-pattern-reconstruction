# Load the common platform and run registered joint-study cases.

from __future__ import annotations

import inspect
import os
import sys

import numpy as np

import s5t_common as K

S345_CODE = K.ROOT / 'scripts' / 'surface_revision' / 's345_ready'
S3C1_CODE = K.ROOT / 'scripts' / 'surface_revision' / 's3_code01'
EXPECTED = {
    'rdy_common': 'scripts/surface_revision/s345_ready/rdy_common.py', 'rdy_platform': 'scripts/surface_revision/s345_ready/rdy_platform.py',
    'rdy_inputs': 'scripts/surface_revision/s345_ready/rdy_inputs.py', 'rdy_decode': 'scripts/surface_revision/s345_ready/rdy_decode.py',
    'rdy_eval': 'scripts/surface_revision/s345_ready/rdy_eval.py', 'rdy_figs': 'scripts/surface_revision/s345_ready/rdy_figs.py',
    's3c_common': 'scripts/surface_revision/s3_code01/s3c_common.py', 's3c_dev': 'scripts/surface_revision/s3_code01/s3c_dev.py',
    'adp_surf02': 'scripts/surface_revision/sections_adapter/adp_surf02.py', 'adp_binding': 'scripts/surface_revision/sections_adapter/adp_binding.py',
    'adp_patterns': 'scripts/surface_revision/sections_adapter/adp_patterns.py', 's1r_eval': 'scripts/surface_revision/sections_adapter/s1r_eval.py',
    's1r_common': 'scripts/surface_revision/sections_adapter/s1r_common.py', 's2_common': 'scripts/surface_revision/surf02/s2_common.py',
    'm3_virtual_camera': 'src/m3_virtual_camera.py', 'm4_reconstruction': 'src/m4_reconstruction.py', 'm4_lines': 'src/m4_lines.py', 'surface_eval_r3': 'src/surface_eval_r3.py',
    'r5_detect': 'scripts/spotnoise_round5/r5_detect.py'}
SIGS = {('adp_surf02', 'render'): ['sid', 'pattern', 'peak', 'sigma', 'seed', 'psf_px'], ('adp_surf02', 'decode'): ['method', 'image', 'ctx'],
        ('s3c_dev', 'dot_decode'): ['img', 'binpat', 'k', 'radii_mm', 'rule', 'path'], ('s3c_dev', 'dot_dec'): ['D', 'arm'],
        ('rdy_decode', 'decode'): ['method', 'img', 'inp', 'row'], ('rdy_decode', 'pack'): ['out'], ('rdy_decode', 'unpack'): ['A', 'meta'],
        ('rdy_eval', 'ctx_eval'): ['row', 'inp'], ('rdy_eval', 'evaluate_common'): ['row', 'dec', 'ctx'], ('rdy_inputs', 'build'): ['row'],
        ('rdy_figs', 'node_fields'): ['case'], ('rdy_platform', 'bound'): [], ('s1r_eval', 'evaluate_case'): ['sid', 'family', 'dec', 'ctx', 'mesh', 'dom_loc', 'dom_glob', 'gauss'],
        ('m3_virtual_camera', 'synthesise_camera_image'): None}
_M: dict = {}


def init_platform() -> dict:

    if _M:
        return _M
    for p in (S345_CODE, S3C1_CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import importlib
    mods = {}
    for name in ('rdy_common', 's3c_common'):
        mods[name] = importlib.import_module(name)
    sb = K.OUT / 'sandbox'
    for k, sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
        os.environ[k] = str(sb / sub)
    os.environ['LINE01RB_RENDER_CAP'] = '0'
    for name in EXPECTED:
        if name not in mods:
            mods[name] = importlib.import_module(name)
    _M.update(mods)
    return _M


def platform_check() -> dict:
    M = init_platform(); bad = []
    files = {}
    for name, exp in EXPECTED.items():
        f = K.rel(M[name].__file__)
        files[name] = dict(file=f, sha256=K.sha(K.ROOT / exp))
        if f != exp:
            bad.append(f'{name}: imported from {f}, expected {exp}')
    sigs = {}
    for (mod, fn), want in SIGS.items():
        obj = getattr(M[mod], fn, None)
        if obj is None or not callable(obj):
            bad.append(f'{mod}.{fn} missing'); continue
        got = list(inspect.signature(obj).parameters)
        sigs[f'{mod}.{fn}'] = got
        if want is not None and got != want:
            bad.append(f'{mod}.{fn} signature {got} != {want}')
    from adp_binding import Bound
    for meth in ('call', 'sid', 'method', 'window'):
        if not callable(getattr(Bound, meth, None)):
            bad.append(f'adp_binding.Bound.{meth} missing')
    return dict(ok=not bad, problems=bad, files=files, signatures=sigs, sandbox_env=os.environ.get('LINE01RB_OUT_DIR'),
                sys_path_head=[K.rel(p) for p in sys.path[:4]], computation_calls=0)


# Run joint-study candidates through the bound rendering, decoding and evaluation modules.
class RealQ:

    def __init__(self):
        self.M = init_platform(); self.calls = []; self._inp = {}

    def inputs(self, row) -> dict:
        c = row['cand_id_sigma0']
        if c not in self._inp:
            self._inp[c] = self.M['rdy_inputs'].build(dict(cand_id=c, section='S5', family=row['family'], view='S5_size_spacing', strategy=''))
        return self._inp[c]

    def check_inputs(self, row, man: dict) -> dict:
        inp = self.inputs(row); f = row['f']
        pat = np.asarray(inp['pattern'], np.float64)
        out = dict(raster=inp['raster_key'] == f['raster'], peak_f32=float(np.float32(inp['peak'])) == float(f['peak_f32']), lit_px=float(inp['lit_px']) == float(man['lit_px']),
                   integral=abs(float(pat.sum()) - float(man['integral'])) <= 1e-6, peak_le_1=float(inp['peak']) <= 1.0)
        out['ok'] = all(out.values())
        return out

    def render(self, row):
        P = self.M['rdy_platform']; inp = self.inputs(row); bd = P.bound()
        res = bd.call('render', bd.sid(row['surface']), inp['pattern'], float(inp['peak']), float(row['sigma']), int(row['seed'] or 0), K.K3.PSF_PX)
        img = np.asarray(res['image']); meta = K.clean(res.get('meta', {})); self.calls.append('render')
        if img.dtype != np.uint8 or img.shape != (512, 512):
            raise K.Stop('technical', f'render returned {img.dtype} {img.shape}')
        if float(meta.get('noise_sigma', -1)) != float(row['sigma']) or (row['sigma'] > 0 and int(meta.get('mottle_seed', -1)) != int(row['seed'])) or meta.get('z_alignment') != 'absolute':
            raise K.Stop('technical', f'render meta differs from the registered sigma / seed / alignment: {meta}')
        return dict(image=img), dict(render_meta=meta, peak=float(inp['peak']), integral=float(np.asarray(inp['pattern'], np.float64).sum()), raster_key=inp['raster_key'])

    def decode(self, row, method, img):
        DC = self.M['rdy_decode']; inp = self.inputs(row); im = np.asarray(img['image'], np.uint8); self.calls.append(f'decode {method}')
        extra = {}
        if method == 'DF':
            DV = self.M['s3c_dev']
            D = DV.dot_decode(im, np.asarray(inp['binary'], np.float32), 1, [float(row['radius_mm'])], 'DF', 'binding')
            M4 = self.M['m4_reconstruction']; z0, z1 = M4.search_window(D['cfg'], float(D['cfg']['measurement_volume_mm']))
            if (float(z0), float(z1)) != (-16.0, 4.0):
                raise K.Stop('technical', f'DF window {(z0, z1)}')
            out = dict(method='D0', arms={'DF': DV.dot_dec(D, 'single')}, shared={}, window=(-16.0, 4.0))
            extra = dict(n_floor=int(D['cd'].get('n_floor', 0)), a_floor_px2=D['cd'].get('a_floor'))
        else:
            out = DC.decode(method, im, inp, dict(section='S5', radius_mm=row['radius_mm'], arm=('D0' if method == 'D0' else 'P4_C3_cal')))
        A, meta = DC.pack(out)
        return A, dict(pack_meta=K.clean(meta), method_label=method, **extra)

    def evaluate(self, row, method, dec_arrays):
        DC, EV, FG = self.M['rdy_decode'], self.M['rdy_eval'], self.M['rdy_figs']
        meta = row['_decode_meta']; U = DC.unpack(dec_arrays, meta['pack_meta'])
        if len(U['arms']) != 1:
            raise K.Stop('technical', f"decode record holds {len(U['arms'])} arms")
        arm = next(iter(U['arms'].values()))
        r = dict(section='S5', family=row['family'], surface=row['surface'], radius_mm=row['radius_mm'])
        m, arrays, info = EV.evaluate_common(r, arm, EV.ctx_eval(r, self.inputs(row)))
        self.calls.append(f'evaluate {method}')
        nf = FG.node_fields(dict(rec=dict(platform_surface=K.K3.PLATFORM[row['surface']], metrics=m, decode_key=row['key']), pts=arrays['pts'], valid=arrays['valid']))
        e = np.asarray(nf['err'])[np.asarray(nf['sup'], bool)]; n_int = int(np.asarray(nf['interp'], bool).sum())
        if n_int != int(m.get('glob_n_interp') or 0):
            raise K.Stop('technical', f"support fields: interpolable nodes {n_int} != record {m.get('glob_n_interp')}")
        sup = dict(glob_n_supported_nodes=int(np.asarray(nf['sup'], bool).sum()), glob_n_tau_nodes=int(np.asarray(nf['acc'], bool).sum()),
                   sup_surf_rmse=(float(np.sqrt(np.mean(e ** 2))) if len(e) else None), sup_surf_mae=(float(np.mean(np.abs(e))) if len(e) else None),
                   sup_surf_p95=(float(np.percentile(np.abs(e), 95)) if len(e) else None), definition='rdy_figs.node_fields = S5 supplement (rdy_s4_report._sup_one)')
        self.calls.append(f'support {method}')
        arr = {k: np.asarray(v) for k, v in arrays.items() if not isinstance(v, dict)}
        return arr, dict(metrics=K.clean(m), support=sup, info=K.clean(info))


def run_q(store, rows: list, ex, pr: dict, ledger, methods_of, S) -> dict:
    log = dict(reuse=0, computed=0, stages={})

    def stage(kind, quota_kind, rec_key, ident, fn, row, why_prefix):
        st, why = store.status(kind, rec_key, ident)
        if st in ('mismatch', 'corrupt'):
            raise K.Stop(st, f'{why_prefix} {rec_key}: {why}')
        if st == 'reuse':
            log['reuse'] += 1; log['stages'][f'{kind}|{rec_key}'] = 'reuse'; return
        tok = ledger.reserve(quota_kind, rec_key, row['manifest_key'])
        try:
            a, m = fn()
            store.commit(kind, rec_key, ident, a, m)
        except BaseException as e:
            ledger.close(tok, quota_kind, rec_key, 'failed', repr(e)[:300]); raise
        ledger.close(tok, quota_kind, rec_key, 'ok'); log['computed'] += 1; log['stages'][f'{kind}|{rec_key}'] = 'computed'
    with store.lock():
        store.set_status('running', n_rows=len(rows))
        try:
            for row in rows:
                k = row['key']; iid = S.image_identity(row['f'], pr)
                stage('image', 'render', k, iid, lambda: ex.render(row), row, 'image')
                isha = store.content('image', k)
                for method in methods_of(row):
                    dk = f'{k}__{method}'
                    did = S.decode_identity(iid, isha, method, S.method_params(method, row['f']['peak_f32'], row.get('radius_mm')))
                    stage('decode', f'decode_{method}', dk, did, lambda: ex.decode(row, method, store.arrays('image', k)), row, 'decode')
                    eid = S.eval_identity(store.content('decode', dk), pr)

                    def _ev():
                        row['_decode_meta'] = K.load_json(store._p('decode', dk))['meta']
                        return ex.evaluate(row, method, store.arrays('decode', dk))
                    stage('eval', 'evaluate', dk, eid, _ev, row, 'eval')
            store.set_status('complete', reuse=log['reuse'], computed=log['computed'])
        except K.Stop as e:
            store.set_status('failed', code=e.code, reason=e.msg[:300]); raise
        except Exception as e:
            store.set_status('failed', code='technical', reason=repr(e)[:300])
            raise K.Stop('technical', repr(e)[:300]) from e
    return log
