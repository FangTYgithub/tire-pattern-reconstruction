# Execute held-out layout cases and the optional clipping control arm.

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import s4r7_common as H
import s4r7_phase as PH

K = H.K
ARM = {'D0': 'SPAT_D0', 'P4TABLE': 'SPAT_P4table', 'CLIP': 'SPAT_P4table'}
REQUIRED = ('image_key', 'set', 'baseline', 'family', 'layout', 'phase', 'surface', 'sigma', 'render_seed', 'raster_key', 'peak_f32', 'image_source')
_C: dict = {}


def set_read_only(flag: bool = True) -> None:
    _C['ro'] = bool(flag)


def read_manifest(path) -> pd.DataFrame:
    M = pd.read_csv(H.reg(path), low_memory=False, keep_default_na=False, float_precision='round_trip')
    miss = [c for c in REQUIRED if c not in M.columns]
    if miss:
        raise H.Stop('input', f'manifest {path}: missing column(s) {miss}')
    if M.image_key.duplicated().any():
        raise H.Stop('input', f'manifest {path}: duplicate image key(s) {list(M[M.image_key.duplicated()].image_key)[:3]}')
    return M


def manifest() -> pd.DataFrame:
    if 'M' not in _C:
        _C['M'] = read_manifest(H.OUT / 'manifest.csv')
    return _C['M']


def cond(key: str) -> dict:
    x = manifest()[manifest().image_key == key]
    if len(x) != 1:
        raise H.Stop('input', f'condition {key} not in the frozen manifest')
    return x.iloc[0].to_dict()


def lay_of(baseline: str, layout: str, source: str = 's4_real07') -> dict:
    key = ('L', source)
    if key not in _C:
        if source == 's4_real07':
            import s4r7_bind as B
            _C[key] = B.layouts()
        else:
            import s4r6_bind as B6
            _C[key] = B6.layouts()
    return _C[key][(baseline, layout)]


def inputs(c: dict) -> dict:
    key = ('inp', c.get('layout_source', 's4_real07'), c['baseline'], c['layout'], c['phase'])
    if key not in _C:
        _C[key] = PH.build_input(c['baseline'], lay_of(c['baseline'], c['layout'], c.get('layout_source', 's4_real07')), c['phase'])
    inp = _C[key]
    if inp['raster_key'] != c['raster_key'] or float(np.float32(inp['peak'])) != float(c['peak_f32']):
        raise H.Stop('check', f"{c['image_key']}: rebuilt raster / peak differ from the manifest")
    return inp


def image(c: dict, purpose: str = 'science'):
    src = c['image_source']
    if src != 'new':
        base = H.R6 / 'images' if 'S4-REAL-06 (new' in src else H.R2 / 'images'
        ip = base / f"{c['image_key']}.json"
        if not K.complete(ip, c['image_key']):
            raise H.Stop('reference', f"{c['image_key']}: stored image record incomplete or hash mismatch")
        return K.load_npz(H.reg(ip.with_suffix('.npz')))['image'], json.loads(ip.read_text(encoding='utf-8'))['npz_sha256']
    ip = H.IMG7 / f"{c['image_key']}.json"
    if H.complete(ip, c['image_key'], required=('render_seed', 'raster_key', 'peak', 'phase_values')):
        return K.load_npz(ip.with_suffix('.npz'))['image'], H.load_json(ip)['npz_sha256']
    if _C.get('ro'):
        raise H.Stop('missing', f"{c['image_key']}: image not rendered (read-only)")
    try:
        return _render(c, ip, purpose)
    except H.Stop:
        raise
    except Exception:
        if purpose == 'retry':
            raise
        return _render(c, ip, 'retry')


def _render(c: dict, ip: Path, purpose: str):
    import pb_m3 as PM
    import rdy_platform as P
    inp = inputs(c); seed = int(c['render_seed'])
    B = H.render_budget(); n = B.reserve(c['image_key'], purpose); t0 = time.monotonic()
    try:
        res = PM.render(P.bound().sid(c['surface']), inp['pattern'], float(inp['peak']), float(c['sigma']), seed, K.RK.PSF_PX, K.BOUNDARY)
        img = np.asarray(res['image'])
        if img.dtype != np.uint8 or img.shape != (512, 512):
            raise H.Stop('technical', f'render returned {img.dtype} {img.shape}')
        h = H.save_npz(ip.with_suffix('.npz'), image=img)
        H.save_json(ip, dict(identity=c['image_key'], set=c['set'], baseline=c['baseline'], family=c['family'], surface=c['surface'], layout=c['layout'], phase=c['phase'],
                             phase_values=dict(u=c.get('phase_u'), v=c.get('phase_v'), phase_frac=c.get('phase_frac'), source=c.get('phase_source')), sigma=float(c['sigma']),
                             noise_seed=c['noise_seed'], render_seed=seed, raster_key=inp['raster_key'], peak=float(inp['peak']), lit_px=inp['lit_px'], integral_f32=inp['integral_f32'],
                             peak_cap_triggered=bool(inp['peak'] > 1.0), boundary=K.BOUNDARY, psf_px=K.RK.PSF_PX, render_meta=K.clean(res['meta']), budget_n=n,
                             seconds=round(time.monotonic() - t0, 3), npz_sha256=h, complete=True, t_local=K.now_local()))
        B.log(dict(n=n, key=c['image_key'], purpose=purpose, event='ok'))
        return img, h
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


def decode(c: dict, method: str, purpose: str, store: Path | None = None, hasher=None) -> dict:
    import rdy_decode as DC
    store = store or H.DEC7
    img, img_sha = image(c); inp = inputs(c); prm, cfg = _params(c, inp, method)
    th = inp['table']['spec']['table']['hash'] if c['family'] == 'line' else None
    dk = H.decode_key(c['image_key'], img_sha, c['raster_key'], th, method, ('abstain' if method == 'CLIP' else None), prm, cfg, hasher)
    rp = store / f'{dk}.json'
    if H.complete(rp, dk, required=('meta', 'method')):
        return dict(H.load_json(rp), stored=True)
    if _C.get('ro'):
        raise H.Stop('missing', f'{dk}: decode missing (read-only)')
    try:
        return _decode_once(c, method, purpose, dk, rp, img, img_sha, inp, prm, cfg)
    except H.Stop:
        raise
    except Exception:
        if purpose != 'science':
            raise
        return _decode_once(c, method, 'technical', dk, rp, img, img_sha, inp, prm, cfg)


def _decode_once(c, method, purpose, dk, rp, img, img_sha, inp, prm, cfg) -> dict:
    import rdy_decode as DC
    B = H.decode_budget(); n = B.reserve(dk, purpose); t0 = time.monotonic()
    try:
        extra = {}
        if method in ('D0', 'P4TABLE'):
            out = DC.decode(method, img, inp, dict(section='S4', family=c['family'], arm=ARM[method], radius_mm=(prm['radius_mm'] if method == 'D0' else ''), strategy=c.get('layout')))
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
                extra['complete_columns_identical'] = {k: bool(np.array_equal(np.asarray(c0[k]), np.asarray(c1[k])[:, :K0], equal_nan=np.asarray(c0[k]).dtype.kind == 'f')) for k in ('Q', 'Q_design', 'z', 'mu', 'admissible')}
            extra['clip'] = K.clean({k: v for k, v in r['clip'].items() if k != 'n_complete_candidates'})
            out = dict(method='P4TABLE', arms={ARM[method]: dict(pts=np.asarray(r['points'], np.float64).reshape(-1, 3), accepted=np.asarray(r['accepted'], bool), aux=dict(r=r))}, shared={}, window=K.RK.WINDOW_MM)
            ncc = np.asarray(r['clip']['n_complete_candidates'])
        A, meta = DC.pack(out)
        if method == 'CLIP':
            A['clip_n_complete_candidates'] = ncc
        h = H.save_npz(rp.with_suffix('.npz'), **A)
        rec = dict(identity=dk, image_key=c['image_key'], image_npz_sha=img_sha, method=method, purpose=purpose, budget_n=n, meta=meta, extra=extra, code=H.code_hashes(H.DECODE_DEPS[method]),
                   seconds=round(time.monotonic() - t0, 3), npz_sha256=h, complete=True, t_local=K.now_local())
        H.save_json(rp, rec); B.log(dict(n=n, key=dk, purpose=purpose, event='ok', seconds=rec['seconds']))
        return dict(rec, stored=False)
    except Exception as e:
        B.log(dict(n=n, key=dk, purpose=purpose, event='failed', error=f'{type(e).__name__}: {e}'[:300], tb=traceback.format_exc()[-1500:]))
        raise


def _dec_r5(ident: str) -> dict:
    A = K.load_npz(H.reg(H.R5 / 'decodes' / f'{ident}.npz')); A = {k[2:]: v for k, v in A.items()}
    r = dict(features=A['features'], line_id=A['line_id'], status=A['status'], chain=A['chain'], segment_id=A['segment_id'], n_candidates=A['n_candidates'], guard_reason=A['guard_reason'],
             points=A['points'], accepted=A['accepted'], detect_info={}, counts={}, checks={})
    return dict(pts=np.asarray(A['points'], np.float64).reshape(-1, 3), accepted=np.asarray(A['accepted'], bool), aux=dict(r=r), _clip_n_complete=A.get('n_complete_candidates'))


def result(c: dict, method: str, plan=None, store: Path | None = None) -> dict:
    import rdy_decode as DC
    pk = ('plan', str(plan or 'default'))
    if pk not in _C:
        _C[pk] = pd.read_csv(H.reg(plan or (H.OUT / '_parts' / 'results_plan.csv')), low_memory=False, keep_default_na=False)
    Rp = _C[pk]; x = Rp[(Rp.cond_id == c['image_key']) & (Rp.method == method)]
    if len(x) != 1:
        raise H.Stop('plan', f"{c['image_key']} {method}: not exactly one entry in the result plan")
    x = x.iloc[0]
    if x.result_source == 'reuse_S4-REAL-02':
        rp = H.R2 / 'raw' / f'{x.reuse_ref}.json'
        if not K.complete(rp, x.reuse_ref):
            raise H.Stop('reference', f'{x.reuse_ref}: stored decode incomplete')
        out = DC.unpack(K.load_npz(H.reg(rp.with_suffix('.npz'))), json.loads(H.reg(rp).read_text(encoding='utf-8'))['meta'])
        return dict(dkey=x.reuse_ref, dsha=K.sha_file(rp.with_suffix('.npz')), dec=out['arms'][ARM[method]], source='reuse S4-REAL-02', record=K.rel(rp))
    if x.result_source == 'reuse_S4-REAL-05':
        rp = H.R5 / 'decodes' / f'{x.reuse_ref}.json'
        if not K.complete(rp, x.reuse_ref):
            raise H.Stop('reference', f'{x.reuse_ref}: stored CLIP decode incomplete')
        r5 = json.loads((H.R5 / 'run.json').read_text(encoding='utf-8'))
        if r5['config_C']['code'] != H.code_hashes(list(r5['config_C']['code'])):
            raise H.Stop('reference', 'S4-REAL-05 CLIP dependencies changed')
        return dict(dkey=x.reuse_ref, dsha=K.sha_file(rp.with_suffix('.npz')), dec=_dec_r5(x.reuse_ref), source='reuse S4-REAL-05', record=K.rel(rp))
    if x.result_source == 'reuse_S4-REAL-06':
        rp = H.R6 / 'decodes' / f'{x.reuse_ref}.json'
        if not K.complete(rp, x.reuse_ref):
            raise H.Stop('reference', f'{x.reuse_ref}: stored S4-REAL-06 decode incomplete')
        r = json.loads(rp.read_text(encoding='utf-8')); A = K.load_npz(rp.with_suffix('.npz')); dec = DC.unpack(A, r['meta'])['arms'][ARM[method]]
        if method == 'CLIP':
            dec['_clip_n_complete'] = A.get('clip_n_complete_candidates')
        return dict(dkey=x.reuse_ref, dsha=r['npz_sha256'], dec=dec, source='reuse S4-REAL-06', record=K.rel(rp))
    rec = decode(c, method, 'science', store=store)
    A = K.load_npz((store or H.DEC7) / f"{rec['identity']}.npz"); dec = DC.unpack(A, rec['meta'])['arms'][ARM[method]]
    if method == 'CLIP':
        dec['_clip_n_complete'] = A.get('clip_n_complete_candidates')
    return dict(dkey=rec['identity'], dsha=rec['npz_sha256'], dec=dec, source=('new decode' if not rec.get('stored') else 'new decode (stored)'), record=K.rel((store or H.DEC7) / f"{rec['identity']}.json"))


def _ref_cond(c: dict, M: pd.DataFrame) -> dict:
    x = M[(M.baseline == c['baseline']) & (M.layout == c['layout']) & (M.phase == c['phase']) & (M.surface == c['surface']) & (M.sigma.astype(float) == 0.0)]
    if len(x) != 1:
        raise H.Stop('dependency', f"{c['image_key']}: sigma-0 reference condition missing")
    return x.iloc[0].to_dict()


def evaluate(c: dict, method: str, store: Path | None = None, plan=None, eval_deps=None, domains=None, res=None, counter=None, M: pd.DataFrame | None = None) -> dict:
    import s4r7_eval as EV
    store = store or H.EVL7
    res = res or result(c, method, plan)
    ref = None; img = ref_img = None
    if c['family'] == 'point':
        img, _s = image(c); rc = _ref_cond(c, M if M is not None else _manifest_for(plan)); ref_img, rs = image(rc); ref = [rc['image_key'], rs]
    ek = H.eval_key(res['dkey'], res['dsha'], eval_deps or EV.eval_deps(), domains or EV.domains(c['surface']), EV.mesh_hash(c['surface']), EV.IDENT_RULE[c['family']], ref)
    ep = store / f'{ek}.json'
    if H.complete(ep, ek, required=('metrics', 'nodes')):
        return H.load_json(ep)
    if _C.get('ro'):
        raise H.Stop('missing', f'{ek}: evaluation missing (read-only) for {c["image_key"]} {method}')
    t0 = time.monotonic(); dec = {k: v for k, v in res['dec'].items() if not k.startswith('_')}
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


def _manifest_for(plan) -> pd.DataFrame:
    if plan is None:
        return manifest()
    p = Path(plan).parent.parent / 'manifest.csv' if Path(plan).parent.name == '_parts' else Path(plan).parent / 'manifest.csv'
    key = ('Mfor', str(p))
    if key not in _C:
        _C[key] = read_manifest(p)
    return _C[key]


def _start(stage: str, hasher=None) -> dict:

    import s4r7_bind as B
    return H.freeze_check(stage, hasher=hasher, data_digests=B.data_digests(), record=hasher is None)


def _eq(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == 'f' or b.dtype.kind == 'f':
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        return bool(np.array_equal(np.isnan(a64), np.isnan(b64)) and np.array_equal(a64, b64, equal_nan=True))
    return bool(np.array_equal(a, b))


def image_key(c: dict, hasher=None) -> str:

    import r2_manifest as RM
    import pb_m3 as PM
    inp = inputs(c); sg = float(c['sigma']); seed = None if str(c['noise_seed']) == '' else int(c['noise_seed'])
    fpd = K.digest(H.code_hashes(RM.IMAGE_CODE, hasher), 32)
    f = dict(raster=inp['raster_key'], peak_f32=float(np.float32(inp['peak'])), surface=K.RK.SURF_MAP[c['surface']], psf=K.RK.PSF_PX, sigma=sg, seed=seed, m3=RM.m3_digest(),
             platform=K.RK.PLATFORM_VERSION, boundary=K.BOUNDARY, snapshot=PM.SNAPSHOT_VERSION, image_code=fpd)
    return 'R2I-' + K.digest(f, 24)


def _fake(*changed):
    tg = {(K.ROOT / p).resolve() for p in changed}; real = K.sha_file
    return lambda p: ('changed-in-memory' if Path(p).resolve() in tg else real(p))


# Test image, decoding and evaluation identities with isolated altered inputs.
def a2() -> dict:

    import s4r7_bind as B
    import s4r7_eval as EV
    import s4r7_analysis as N
    if H.ISO.exists():
        shutil.rmtree(H.guard(H.ISO))
    dd = B.data_digests(); t = {}

    def rejects(**kw):
        try:
            H.freeze_check('isolation', record=False, **kw); return False
        except H.Stop:
            return True

    def stops(fn, code=None):
        try:
            fn(); return False
        except H.Stop as e:
            return code is None or e.code == code
    t['baseline_freeze_passes'] = not rejects(data_digests=dd)
    for lab, p in (('manifest', 'outputs/surface_revision/s4_real07/manifest.csv'), ('results_plan', 'outputs/surface_revision/s4_real07/_parts/results_plan.csv'),
                   ('strategy_views', 'outputs/surface_revision/s4_real07/_parts/views.csv'), ('generation_record', 'outputs/surface_revision/s4_real07/generation.json'),
                   ('generator_code', 'scripts/surface_revision/s4_real07/s4r7_gen.py'), ('phase_adapter', 'scripts/surface_revision/s4_real07/s4r7_phase.py'),
                   ('forward_m3', 'src/m3_virtual_camera.py'), ('decode_clip', 'scripts/surface_revision/s4_real05/s4r5_clip.py'), ('decode_d0', 'src/m4_reconstruction.py'),
                   ('surface_eval_r3', 'src/surface_eval_r3.py'), ('eval_adapter', 'scripts/surface_revision/s4_real07/s4r7_eval.py'),
                   ('scheduler_run', 'scripts/surface_revision/s4_real07/s4r7_run.py'), ('run_entry_main', 'scripts/surface_revision/s4_real07/s4r7_main.py'),
                   ('analysis', 'scripts/surface_revision/s4_real07/s4r7_analysis.py'), ('omega1_domain_file', 'outputs/surface_revision/pix_bound01/domains/omega1.npz')):
        t[f'{lab}_change_rejected'] = rejects(hasher=_fake(p), data_digests=dd)
    for k in ('omega1', 'local', 'mesh', 'sensor_cfg', 'p4_params', 'd0_params', 'phases', 'seeds', 'gross_mm', 'd', 'tau'):
        t[f'data_{k}_change_rejected'] = rejects(data_digests=dict(dd, **{k: 'changed'}))
    t['figure_code_change_not_science'] = not rejects(hasher=_fake('scripts/surface_revision/s4_real07/s4r7_figs.py'), data_digests=dd)

    rb0, db0 = H.render_budget().state(), H.decode_budget().state()
    t['stage_run_stops_before_science'] = stops(lambda: _run(['none'], 'isolation', hasher=_fake('outputs/surface_revision/s4_real07/_parts/results_plan.csv')), 'freeze')
    t['stage_a3_stops_before_science'] = stops(lambda: a3(hasher=_fake('src/surface_eval_r3.py')), 'freeze')
    t['stage_entry_change_stops'] = stops(lambda: _run(['none'], 'isolation', hasher=_fake('scripts/surface_revision/s4_real07/s4r7_main.py')), 'freeze')
    t['budgets_untouched_by_rejected_stages'] = bool(H.render_budget().state() == rb0 and H.decode_budget().state() == db0)

    M = manifest(); c = cond(M[(M.family == 'line') & (M.image_source == 'new') & (M.sigma.astype(float) == 40.0)].sort_values('image_key').iloc[0].image_key)
    inp = inputs(c); prm, cfg = _params(c, inp, 'CLIP'); th = inp['table']['spec']['table']['hash']
    ik0 = image_key(c)
    t['image_key_recomputed_equal'] = bool(ik0 == c['image_key'])
    t['image_key_ignores_decode_and_eval'] = bool(image_key(c, _fake('scripts/surface_revision/s4_real05/s4r5_clip.py', 'scripts/surface_revision/s345_ready/rdy_decode.py', 'src/m4_lines.py',
                                                                   'src/m4_reconstruction.py', 'src/surface_eval_r3.py', 'scripts/surface_revision/s4_real02/r2_eval.py')) == ik0)
    t['image_key_follows_forward'] = bool(image_key(c, _fake('src/m3_virtual_camera.py')) != ik0)
    dk0 = H.decode_key(c['image_key'], 'img', c['raster_key'], th, 'CLIP', 'abstain', prm, cfg)
    dk_code = H.decode_key(c['image_key'], 'img', c['raster_key'], th, 'CLIP', 'abstain', prm, cfg, hasher=_fake('scripts/surface_revision/s4_real05/s4r5_clip.py'))
    dk_prm = H.decode_key(c['image_key'], 'img', c['raster_key'], th, 'CLIP', 'abstain', dict(prm, changed=1), cfg)
    dk_cfg = H.decode_key(c['image_key'], 'img', c['raster_key'], th, 'CLIP', 'abstain', prm, dict(cfg, changed=1))
    dk_eval = H.decode_key(c['image_key'], 'img', c['raster_key'], th, 'CLIP', 'abstain', prm, cfg, hasher=_fake('src/surface_eval_r3.py', 'scripts/surface_revision/s4_real07/s4r7_eval.py'))
    ed, dm, mh, ir = EV.eval_deps(), EV.domains(c['surface']), EV.mesh_hash(c['surface']), EV.IDENT_RULE['line']
    ek0 = H.eval_key(dk0, 'dsha', ed, dm, mh, ir)
    t['decode_change_invalidates_decode_and_eval'] = bool(len({dk0, dk_code, dk_prm, dk_cfg}) == 4 and H.eval_key(dk_code, 'dsha', ed, dm, mh, ir) != ek0)
    t['decode_key_ignores_eval'] = bool(dk_eval == dk0)
    t['eval_change_invalidates_eval_only'] = bool(len({ek0, H.eval_key(dk0, 'dsha', ed, dict(dm, omega1='x'), mh, ir), H.eval_key(dk0, 'dsha', ed, dm, mh, 'changed rule'),
                                                      H.eval_key(dk0, 'dsha', EV.eval_deps(hasher=_fake('src/surface_eval_r3.py')), dm, mh, ir), H.eval_key(dk0, 'dsha', ed, dm, 'mesh', ir)}) == 5
                                                 and image_key(c, _fake('src/surface_eval_r3.py')) == ik0 and dk_eval == dk0)

    p1 = H.ISO / 'missing_col.csv'; H.save_csv(M.drop(columns=['raster_key']), p1); p2 = H.ISO / 'dup_key.csv'; H.save_csv(pd.concat([M.head(2), M.head(1)]), p2)
    t['missing_column_rejected'] = stops(lambda: read_manifest(p1), 'input'); t['duplicate_key_rejected'] = stops(lambda: read_manifest(p2), 'input')
    c0 = cond(M[(M.image_source != 'new')].sort_values('image_key').iloc[0].image_key)
    res = result(c0, 'P4TABLE'); cnt = {}
    e1 = evaluate(c0, 'P4TABLE', store=H.ISO / 'evals', res=res, counter=cnt); e2 = evaluate(c0, 'P4TABLE', store=H.ISO / 'evals', res=res, counter=cnt)
    t['resume_reuses_complete_record'] = bool(cnt.get('computed') == 1 and e1['identity'] == e2['identity'])
    z = H.ISO / 'evals' / f"{e1['identity']}.npz"; b = bytearray(z.read_bytes()); b[len(b) // 2] ^= 0xFF; z.write_bytes(bytes(b))
    t['corrupt_npz_rejected_and_recomputed'] = bool(not H.complete(z.with_suffix('.json'), e1['identity']) and evaluate(c0, 'P4TABLE', store=H.ISO / 'evals', res=res, counter=cnt)['identity'] == e1['identity']
                                                    and cnt['computed'] == 2)
    j = H.load_json(z.with_suffix('.json')); j['complete'] = False; H.save_json(z.with_suffix('.json'), j)
    t['partial_record_rejected'] = bool(not H.complete(z.with_suffix('.json'), e1['identity']))
    proc = subprocess.run([sys.executable, 's4r7_main.py', 'nonexistent_stage'], cwd=str(H.HERE), capture_output=True, text=True)
    fin = H.ISO / 'finally_ran.txt'
    code = ('import s4r7_common as H, sys\ntry:\n    try:\n        raise H.Stop("technical", "injected failure")\n    finally:\n        open(sys.argv[1], "w").write("x")\n'
            'except H.Stop as e:\n    print(e); sys.exit(1)')
    proc2 = subprocess.run([sys.executable, '-c', code, str(fin)], cwd=str(H.HERE), capture_output=True, text=True)
    t['failure_exit_status_nonzero'] = bool(proc.returncode != 0 and proc2.returncode == 1 and fin.exists())
    dec0 = {k: v for k, v in res['dec'].items() if not k.startswith('_')}
    r0 = dict(dec0['aux']['r']); r0['accepted'] = np.zeros_like(np.asarray(r0['accepted'], bool)); r0['points'] = np.full_like(np.asarray(r0['points'], np.float64), np.nan)
    r0['status'] = np.where(np.asarray(r0['status']) == 0, 2, r0['status'])
    dec0 = dict(dec0, accepted=r0['accepted'], pts=r0['points'], aux=dict(r=r0))
    e6 = evaluate(c0, 'P4TABLE', store=H.ISO / 'evals_nooutput', res=dict(res, dkey=res['dkey'] + '-nooutput-test', dec=dec0), counter=cnt)
    g = [x for x in N.domain_rows(dict(c0, method='P4TABLE'), e6) if x['domain'] == 'glob'][0]
    t['legal_no_output'] = bool(e6['metrics']['state'] == 'no_output' and g['Ctau'] == 0.0 and g['pt_rmse'] is None and e6['metrics']['status_closed'])
    Bi = H.Budget('decode', 2, dict(test=1), base=H.ISO / 'budget'); Bi.reserve('a', 'test')
    t['budget_cap_stops'] = stops(lambda: Bi.reserve('b', 'test'), 'budget')
    out = dict(ok=bool(all(t.values())), n_tests=len(t), tests=t, decode_calls=0, evaluations=cnt.get('computed', 0),
               note='real freeze_check / key functions with an in-memory hash access layer (changed files simulated, nothing modified); no decode, no imaging', t_local=K.now_local())
    H.run_put('A2', out)
    if not out['ok']:
        raise H.Stop('gate', 'A2 isolation tests failed: ' + json.dumps({k: v for k, v in t.items() if not v}))
    return dict(ok=True, n_tests=len(t), evaluations=out['evaluations'])


REGRESSION = [('D0 S0 sigma 0', 'R8x4', 'regular', 'M10_curv', 'H0', 0.0, 'D0'), ('D0 S0 sigma 40', 'R8x4', 'regular', 'M10_curv', 'H0', 40.0, 'D0'),
              ('D0 S2* worst case (S4-REAL-06)', 'R8x4', 'j0.30_B000_c7', 'M10_curv', 'H1', 0.0, 'D0'), ('P4 S0 no band', 'L6.3', 'regular', 'B_curv', 'H1', 0.0, 'P4TABLE'),
              ('P4 S2* H0', 'L6.3', 'j0.30_B000_c39', 'M24_curv', 'H0', 40.0, 'P4TABLE'), ('CLIP S0 band', 'L6.3', 'regular', 'M24_curv', 'H0', 0.0, 'CLIP'),
              ('CLIP S2* H1', 'L6.3', 'j0.30_B000_c39', 'B_curv', 'H1', 0.0, 'CLIP'), ('CLIP S2* H0', 'L6.3', 'j0.30_B000_c39', 'M10_curv', 'H0', 0.0, 'CLIP')]


def _r6_cond(r: dict) -> dict:
    pv = PH.ph('dots' if r['family'] == 'point' else 'lines', r['phase'])
    return dict(r, set='regression', layout_source='s4_real06', phase_u=pv.get('u'), phase_v=pv.get('v'), phase_frac=pv.get('phase_frac'),
                image_source=('S4-REAL-06 (new)' if r['image_source'] == 'new' else 'S4-REAL-02'))


def a3(hasher=None) -> dict:
    import rdy_decode as DC
    _start('regression (before any decode)', hasher)
    R6M = pd.read_csv(H.reg(H.R6 / 'manifest.csv'), low_memory=False, keep_default_na=False, float_precision='round_trip')
    R6R = pd.read_csv(H.reg(H.R6 / 'results.csv'), low_memory=False, float_precision='round_trip')
    M6 = pd.DataFrame([_r6_cond(r) for r in R6M.to_dict('records')]); rows = []
    for role, b, lay, s, ph, sg, m in REGRESSION:
        x = M6[(M6.baseline == b) & (M6.layout == lay) & (M6.surface == s) & (M6.phase == ph) & (M6.sigma.astype(float) == sg)]
        if len(x) != 1:
            raise H.Stop('plan', f'regression case {role} not unique in S4-REAL-06')
        c = x.iloc[0].to_dict(); inp = inputs(c)
        rec = decode(c, m, 'regression', store=H.ENTRY / 'decodes'); A = K.load_npz(H.ENTRY / 'decodes' / f"{rec['identity']}.npz")
        e = R6R[(R6R.record_type == 'cond') & (R6R.image_key == c['image_key']) & (R6R.method == m) & (R6R.domain == 'glob')].iloc[0]
        e6 = json.loads(H.reg(H.R6 / 'evals' / f'{e.eval_key}.json').read_text(encoding='utf-8')); rp6 = K.ROOT / e6['decode_record']; O = K.load_npz(H.reg(rp6.with_suffix('.npz')))
        if m == 'CLIP' and '/s4_real05/' in e6['decode_record']:
            eq = {k: (f'a0_r_{k[2:]}' in A and _eq(A[f'a0_r_{k[2:]}'], O[k])) for k in O if k.startswith('r_') and k != 'r_n_complete_candidates'}
            eq['n_complete_candidates'] = _eq(A['clip_n_complete_candidates'], O['r_n_complete_candidates'])
        else:
            eq = {k: (k in A and _eq(A[k], O[k])) for k in O}
        dec = DC.unpack(A, rec['meta'])['arms'][ARM[m]]
        if m == 'CLIP':
            dec['_clip_n_complete'] = A['clip_n_complete_candidates']
        ev = evaluate(c, m, store=H.ENTRY / 'evals', res=dict(dkey=rec['identity'], dsha=rec['npz_sha256'], dec=dec, source='regression', record=K.rel(H.ENTRY / 'decodes' / f"{rec['identity']}.json")), M=M6)
        mdiff = {k: (e6['metrics'][k], ev['metrics'].get(k)) for k in e6['metrics'] if e6['metrics'][k] != ev['metrics'].get(k)}
        ndiff = {d: {k: (e6['nodes'][d][k], ev['nodes'][d][k]) for k in e6['nodes'][d] if e6['nodes'][d][k] != ev['nodes'][d].get(k)} for d in ('glob', 'loc')}
        sdiff = (e6.get('source_identity') != ev.get('source_identity')) if c['family'] == 'point' else False
        rows.append(dict(role=role, image_key=c['image_key'], method=m, phase=ph, layout=lay, surface=s, sigma=sg, raster_equal=bool(inp['raster_key'] == c['raster_key']),
                         peak_equal=bool(float(np.float32(inp['peak'])) == float(c['peak_f32'])), table_hash=(inp['table']['spec']['table']['hash'] if c['family'] == 'line' else None),
                         clip_ids=((rec.get('extra') or {}).get('clip') or {}).get('ids'), n_arrays=len(eq), arrays_equal=bool(all(eq.values())), unequal=[k for k, v in eq.items() if not v],
                         metric_diffs=K.clean(mdiff), node_diffs=K.clean(ndiff), source_identity_diff=bool(sdiff), compared_with=e6['decode_record'], compared_eval=K.rel(H.R6 / 'evals' / f'{e.eval_key}.json'),
                         ok=bool(all(eq.values()) and not mdiff and not any(ndiff.values()) and not sdiff)))
    out = dict(ok=bool(all(x['ok'] for x in rows)), n_cases=len(rows), cases=rows, decode_calls=len(rows), t_local=K.now_local())
    H.run_put('A3', out); H.flush_reads('A3')
    if not out['ok']:
        raise H.Stop('gate', 'A3 regression failed: ' + json.dumps([x for x in rows if not x['ok']], default=str)[:800])
    return dict(ok=True, n_cases=len(rows))


def drill() -> dict:

    import s4r7_analysis as N
    import s4r7_figs as F
    _start('drill')
    d = H.ISO / 'drill'
    if d.exists():
        shutil.rmtree(H.guard(d))
    R6M = pd.read_csv(H.R6 / 'manifest.csv', low_memory=False, keep_default_na=False, float_precision='round_trip')
    R6P = pd.read_csv(H.R6 / '_parts' / 'B_results_plan.csv', low_memory=False, keep_default_na=False)
    R6R = pd.read_csv(H.R6 / 'results.csv', low_memory=False); R6R = R6R[(R6R.record_type == 'cond') & (R6R.domain == 'glob')]
    x = R6M[(R6M.baseline == 'L6.3') & R6M.layout.isin(['regular', 'j0.30_B000_c39'])]
    x = pd.DataFrame([dict(_r6_cond(r), set='main', block=('' if r['layout'] == 'regular' else 'B000')) for r in x.to_dict('records')])
    plan = []
    for c in x.itertuples():
        for m in ('P4TABLE', 'CLIP'):
            y = R6P[(R6P.cond_id == c.image_key) & (R6P.method == m)].iloc[0]
            if y.result_source == 'new_decode':
                plan.append(dict(cond_id=c.image_key, method=m, result_source='reuse_S4-REAL-06', reuse_ref=R6R[(R6R.image_key == c.image_key) & (R6R.method == m)].iloc[0].decode_key))
            else:
                plan.append(dict(cond_id=c.image_key, method=m, result_source=y.result_source, reuse_ref=y.reuse_ref))
    views = pd.DataFrame([dict(baseline='L6.3', block='B000', arm=a, phase=p, status='selected', layout='j0.30_B000_c39', denominator=None) for a in ('S2*', 'S4*') for p in ('H0', 'H1')])
    H.save_csv(x, d / 'manifest.csv'); H.save_csv(pd.DataFrame(plan), d / '_parts' / 'results_plan.csv'); H.save_csv(views, d / '_parts' / 'views.csv')
    ctx = dict(manifest=d / 'manifest.csv', plan=d / '_parts' / 'results_plan.csv', views=d / '_parts' / 'views.csv', rasters=None, eval_dir=d / 'evals', out=d)
    set_read_only(False); Md = read_manifest(ctx['manifest']); cnt = {}
    for c in Md.to_dict('records'):
        for m in ('P4TABLE', 'CLIP'):
            evaluate(c, m, store=ctx['eval_dir'], plan=ctx['plan'], M=Md, counter=cnt)
    res = N.analyse(ctx, write=False); fr = res['frames']; V = fr['V']; Cn = fr['Cn']
    per_phase = V.groupby(['arm', 'block', 'surface', 'sigma', 'method', 'domain']).phase.agg(['size', 'nunique'])
    chk = dict(unique_images=bool(res['unique_images'] == len(Md) == 24), unique_layouts=bool(res['unique_layouts'].get('line') == 2),
               one_row_per_phase=bool((per_phase['size'] == per_phase['nunique']).all() and (per_phase['size'] == 2).all()),
               shared_layout_contrast_zero=bool(len(Cn) > 0 and Cn.same_layout.all() and (Cn.d_Ctau.abs() == 0).all() and (Cn.d_c.abs() == 0).all()),
               view_rows=bool(len(V) == 2 * 2 * 3 * 2 * 2 * 2))
    figp = F.drill_figure(fr['P'], d / 'drill.png'); chk['figure_written'] = bool(Path(figp).exists())
    victim = sorted((d / 'evals').glob('*.json'))[0]; victim.unlink()
    try:
        N.analyse(ctx, write=False); chk['missing_record_stops'] = False
    except H.Stop as e:
        chk['missing_record_stops'] = bool(e.code == 'missing')
    set_read_only(False)
    out = dict(ok=bool(all(chk.values())), checks=chk, n_conditions=len(Md), evaluations=cnt.get('computed', 0), figure=K.rel(figp),
               note='stub scheduler: stored S4-REAL-02 / -05 / -06 decodes as backend, isolated evaluation store; no decode, no imaging', t_local=K.now_local())
    H.run_put('drill', out)
    if not out['ok']:
        raise H.Stop('gate', f'drill failed: {chk}')
    return out


def _run(keys: list, label: str, hasher=None) -> dict:
    _start(f'science {label} (start / resume)', hasher)
    r = H.run_load()
    for g in ('A2', 'A3', 'drill'):
        if not r.get(g, {}).get('ok'):
            raise H.Stop('gate', f'{g} not passed')
    set_read_only(False)
    rows = sorted([cond(k) for k in keys], key=lambda c: (float(c['sigma']), c['family'], c['image_key'])); done = []
    for c in rows:
        for m in (['D0'] if c['family'] == 'point' else ['P4TABLE', 'CLIP']):
            ev = evaluate(c, m)
            done.append(dict(image_key=c['image_key'], method=m, state=ev['metrics']['state']))
    chk = pair_checks([c for c in rows if c['family'] == 'line'])
    _start(f'science {label} (end)')
    out = dict(label=label, n_conditions=len(rows), n_results=len(done), states={f'{m}|{s}': sum(1 for x in done if x['state'] == s and x['method'] == m) for m, s in {(x['method'], x['state']) for x in done}},
               pair_checks_ok=chk['ok'], pair_checks=chk, render_budget=H.render_budget().state(), decode_budget=H.decode_budget().state(), t_local=K.now_local())
    H.run_put(f'run_{label}', out); H.flush_reads(f'run_{label}')
    if not chk['ok']:
        raise H.Stop('gate', f'line pair invariants failed in {label}: ' + json.dumps([x for x in chk['cases'] if not x['ok']], default=str)[:800])
    return {k: v for k, v in out.items() if k != 'pair_checks'}


def pair_checks(line_conds: list) -> dict:

    cases = []
    for c in line_conds:
        p4 = result(c, 'P4TABLE'); cl = result(c, 'CLIP'); a, b = p4['dec']['aux']['r'], cl['dec']['aux']['r']
        x = dict(image_key=c['image_key'], set=c['set'], band=bool(c['clipped_band_lit'] in (True, 'True')), front_equal=bool(all(_eq(a[k], b[k]) for k in ('features', 'chain', 'segment_id', 'guard_reason'))))
        st = np.asarray(b['status']); acc_b = np.asarray(b['accepted'], bool); acc_a = np.asarray(a['accepted'], bool); both = acc_a & acc_b
        x['abstain_not_accepted'] = bool(not (acc_b & (st == 11)).any())
        x['co_accepted_identical'] = bool(np.array_equal(np.asarray(a['points'], np.float64)[both], np.asarray(b['points'], np.float64)[both]) and np.array_equal(np.asarray(a['line_id'])[both], np.asarray(b['line_id'])[both]))
        ncc = cl['dec'].get('_clip_n_complete'); x['complete_candidates_equal'] = bool(ncc is not None and _eq(np.asarray(a['n_candidates']), np.asarray(ncc)))
        if not x['band']:
            x['noop_equal'] = bool(all(_eq(a[k], b[k]) for k in ('line_id', 'status', 'n_candidates', 'accepted', 'points')))
        else:
            rec = H.load_json(K.ROOT / cl['record']) if cl['record'] else {}
            cci = (rec.get('extra') or {}).get('complete_columns_identical'); x['complete_columns_identical'] = (None if cci is None else bool(all(cci.values())))
        x['closure'] = bool(np.isin(st, np.arange(12)).all())
        x['ok'] = bool(x['front_equal'] and x['abstain_not_accepted'] and x['co_accepted_identical'] and x['complete_candidates_equal'] and x.get('noop_equal', True)
                       and x.get('complete_columns_identical', True) is not False and x['closure'])
        cases.append(x)
    return dict(ok=bool(all(x['ok'] for x in cases)), n=len(cases), cases=cases)


def _b024_s1() -> dict:
    V = pd.read_csv(H.OUT / '_parts' / 'views.csv', low_memory=False, keep_default_na=False)
    x = V[(V.block == 'B024') & (V.arm == 'S1') & (V.status == 'selected')]
    return {b: sorted(set(g.layout)) for b, g in x.groupby('baseline')}


def subset() -> dict:

    M = manifest(); s1 = _b024_s1()
    sel = (M.sigma.astype(float) == 0.0) & ((M.layout == 'regular') | pd.Series([l in s1.get(b, []) for b, l in zip(M.baseline, M.layout)], index=M.index))
    main, anc = M[sel & (M.set == 'main')], M[sel & (M.set == 'anchor')]
    if len(main) != 24 or len(anc) != 6 or (anc.image_source != 'new').sum() != 3:
        raise H.Stop('plan', f'subset sizes {len(main)} / {len(anc)} (old {int((anc.image_source != "new").sum())}) != 24 / 6 (3)')
    return _run(list(main.image_key) + list(anc.image_key), 'subset')


def rest() -> dict:
    if not H.run_load().get('run_subset', {}).get('pair_checks_ok'):
        raise H.Stop('gate', 'subset not passed')
    return _run(list(manifest().image_key), 'all')
