# Prepare and evaluate individual spacing-layout cases.

from __future__ import annotations

import json
import os
import time

import numpy as np

import s2p_common as K

K.setup()


def _save_npz(p, **arrays) -> str:
    p = K.guard(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, p)
    return K.sha_file(p)


def complete(p, identity: str | None = None) -> bool:
    if not p.exists():
        return False
    try:
        r = K.load_json(p)
    except Exception:
        return False
    z = p.with_suffix('.npz')
    ok = r.get('complete') is True and z.exists() and K.sha_file(z) == r.get('npz_sha256')
    return bool(ok and (identity is None or r.get('identity') == identity))


def raster_for(t: dict) -> dict:
    import adp_patterns as PT
    import s1r_common as S1K
    row = dict(generator=('sl_lib.lattice' if t['family'] == 'point' else 'm2.olines'), layout_id=t['lattice'], phase_u=t['phase_u'], phase_v=t['phase_v'], radius_mm=t['size_mm'],
               p_parallel_mm=t['p_parallel_mm'], p_perp_mm=t['p_perp_mm'], line_pitch_mm=t['line_pitch_mm'], line_width_mm=1.0, orientation_deg=(t['orientation_deg'] or 0.0),
               phase_frac=t['phase_frac'], section='S2', cand_id=t['cand_id'], brightness_rule='peak=1.0', raster_sha256_registered='', code='', diameter_mm='')
    b = PT.build(row); binary = np.asarray(b['binary']) > 0
    if S1K.raster_key(binary) != t['raster_key']:
        raise RuntimeError(f"raster of {t['cand_id']} differs from the list ({S1K.raster_key(binary)} vs {t['raster_key']})")
    return dict(binary=binary.astype(np.float32), centres=b['centres'], meta_full=b['meta_full'])


def peak_for(t: dict, R: dict) -> float:
    peak = 1.0 if t['regime'] == 'fixed_peak' else K.Q / float(R['binary'].sum(dtype=np.float64))
    if float(np.float32(peak)) != float(np.float32(float(t['peak_f32']))):
        raise RuntimeError(f"peak of {t['cand_id']}: rule {peak!r} vs list {t['peak_f32']!r}")
    if peak > 1.0 + 1e-12:
        raise RuntimeError(f"peak {peak} > 1 for {t['cand_id']} (not clipped; condition infeasible)")
    return peak


def identities(t: dict, bd, parts: dict, peak: float) -> dict:
    import s1r_common as S1K
    S = S1K.s2()
    m3 = K.digest({k: v for k, v in S.m3_cfg(0.0, 0).items() if k not in ('noise_sigma', 'mottle_seed')})
    seed = K.norm_seed(t['noise_seed']) if float(t['sigma']) > 0 else None
    img = K.digest(dict(raster=t['raster_key'], peak_f32=float(np.float32(peak)), surface=t['platform_surface'], psf=K.PSF_PX, sigma=float(t['sigma']), seed=seed, m3=m3,
                        platform=bd.version, image_part=parts['image']), 24)
    dcfg = (S.line_cfg(float(peak)) if t['family'] == 'line' else S.dot_cfg())
    dec = K.digest(dict(img=img, method=K.ARM[t['family']], cfg={k: v for k, v in dcfg.items() if isinstance(v, (int, float, str, bool)) or v is None},
                        p4=(S.P4 if t['family'] == 'line' else None), decode_part=parts['decode']), 24)
    ev = K.digest(dict(dec=dec, eval_part=parts['eval'], d=K.D_SUPPORT, tau=K.TAU, gross=K.GROSS), 24)
    return dict(image=img, decode=dec, eval=ev, seed=seed)


def raw_arrays(family: str, dec: dict) -> tuple[dict, dict]:
    aux = dec['aux']; a = dict(pts=np.asarray(dec['pts'], np.float64), accepted=np.asarray(dec['accepted'], bool)); meta = {}
    if family == 'point':
        tr = aux['tr']; cd = aux['cd']
        a.update(feats=np.asarray(aux['feats'], np.float32).reshape(-1, 2), **{f'tr_{k}': np.asarray(v) for k, v in tr.items() if isinstance(v, np.ndarray)},
                 cand_table=np.asarray(cd['cands'], np.float64).reshape(-1, 7), comp_reason=np.asarray(cd['reason'], np.int8), comp_stats=np.asarray(cd['stats'], np.int32),
                 sel_code=np.asarray(aux['sel_code'], np.int8), sel_fidx=np.asarray(aux['sel_fidx'], np.int32))
        meta = dict(n_labels=int(cd['n_labels']), thresh=float(cd['thresh']), n_filtered=int(cd['n_filtered']), n_border=int(cd['n_border']),
                    det_info={k: v for k, v in (aux.get('det_info') or {}).items() if not isinstance(v, np.ndarray)}, info={k: v for k, v in (aux.get('info') or {}).items() if not isinstance(v, np.ndarray)},
                    comp_reason_codes={'-1': 'background', '0': 'candidate', '1': 'area < MIN', '2': 'area > MAX', '3': 'touches border', '4': 'mean intensity < MIN', '5': 'zero moment'})
        if 'ambiguous_mask' in (aux.get('info') or {}):
            a['info_ambiguous_mask'] = np.asarray(aux['info']['ambiguous_mask'], bool)
    else:
        r = aux['r']
        for k in ('features', 'line_id', 'status', 'chain', 'segment_id', 'n_candidates', 'guard_reason'):
            a[f'r_{k}'] = np.asarray(r[k])
        meta = dict(detect_info={k: v for k, v in (r.get('detect_info') or {}).items() if not isinstance(v, np.ndarray)}, counts=r.get('counts'), checks=r.get('checks'),
                    window=[float(r['window_z0_mm']), float(r['window_z1_mm'])])
    return a, K.clean(meta)


def load_raw(p, family: str) -> dict:
    rec = K.load_json(p)
    with np.load(p.with_suffix('.npz')) as z:
        A = {k: np.asarray(z[k]) for k in z.files}
    m = rec['meta']
    if family == 'point':
        tr = {k[3:]: A[k] for k in A if k.startswith('tr_')}
        cd = dict(cands=[tuple(row) for row in A['cand_table']], reason=A['comp_reason'], n_labels=int(m['n_labels']), stats=A['comp_stats'])
        info = dict(m['info']);
        if 'info_ambiguous_mask' in A:
            info['ambiguous_mask'] = A['info_ambiguous_mask']
        aux = dict(feats=A['feats'], tr=tr, cd=cd, sel_code=A['sel_code'], sel_fidx=A['sel_fidx'], det_info=m['det_info'], info=info)
    else:
        r = {k[2:]: A[k] for k in A if k.startswith('r_')}
        r.update(detect_info=m['detect_info'], counts=m['counts'], checks=m['checks'])
        aux = dict(r=r)
    return dict(pts=A['pts'], accepted=A['accepted'], aux=aux)


def node_fields(sid: str, pts_valid: np.ndarray, dom: dict, z_ref: np.ndarray) -> dict:
    from surface_eval_r3 import surface_eval_r3_dev
    n = len(z_ref)
    if len(pts_valid) < 3:
        nanf = np.full(n, np.nan, np.float32)
        return dict(z_est=nanf, err=nanf, support_measure=np.full(n, np.inf, np.float32), interpolable=np.zeros(n, bool), supported=np.zeros(n, bool), accurate=np.zeros(n, bool), Cvalid=0.0, Ctau=0.0)
    D = dict(x=dom['x'], y=dom['y'], z_ref=z_ref, area=dom['area'], hash=dom['hash'])
    ev = surface_eval_r3_dev(pts_valid, None, D, [K.D_SUPPORT], [K.TAU], return_fields=True, support_rule='exact')
    f = ev['fields']; err = np.asarray(f['err'], np.float64); R = np.asarray(f['support_measure'], np.float64)
    interp = (np.asarray(f['simplex']) >= 0) & np.isfinite(err); sup = interp & (R <= K.D_SUPPORT); acc = sup & (np.abs(err) <= K.TAU)
    bd = ev['subdomains']['global']['by_d'][str(float(K.D_SUPPORT))]
    return dict(z_est=np.asarray(f['z_est'], np.float32), err=err.astype(np.float32), support_measure=R.astype(np.float32), interpolable=interp, supported=sup, accurate=acc,
                Cvalid=float(bd['C_valid'] or 0.0), Ctau=float(bd['C_tau'][str(float(K.TAU))] or 0.0))


def evaluate(t: dict, sid: str, dec: dict, ctx: dict, mesh: dict, dom_l: dict, dom_g: dict) -> tuple[dict, dict, dict]:
    import s1r_eval as E
    m, arrays, info, _ = E.evaluate_case(sid, t['family'], dec, ctx, mesh, dom_l, dom_g, sid == 'A_curv_gauss')
    v = np.asarray(arrays['valid'], bool); P = np.asarray(arrays['pts'], np.float64)[v]
    fl = node_fields(sid, P, dict(x=dom_l['x'], y=dom_l['y'], area=dom_l['area'], hash=dom_l['hash']), np.asarray(dom_l['z_ref'], np.float64))
    fg = node_fields(sid, P, dict(x=dom_g['x'], y=dom_g['y'], area=np.full(len(dom_g['x']), dom_g['step'] ** 2), hash=dom_g['hash']), np.asarray(dom_g[f'z_{sid}'], np.float64))
    for pre, f in (('loc', fl), ('glob', fg)):
        if abs(f['Cvalid'] - (m[f'{pre}_Cvalid'] or 0.0)) > 1e-12 or abs(f['Ctau'] - (m[f'{pre}_Ctau'] or 0.0)) > 1e-12:
            raise RuntimeError(f"{pre} node fields differ from the evaluation: Cvalid {f['Cvalid']} / {m[f'{pre}_Cvalid']}, Ctau {f['Ctau']} / {m[f'{pre}_Ctau']}")
    for pre, f in (('loc', fl), ('glob', fg)):
        arrays.update({f'{pre}_node_{k}': f[k] for k in ('z_est', 'err', 'support_measure', 'interpolable', 'supported', 'accurate')})
    return m, arrays, info


# Build, render and evaluate one spacing-layout case or reuse compatible results.
def run_case(t: dict, bd, dom_g: dict, parts: dict, reuse: dict | None, out_dir, ledger) -> dict:

    import s1r_common as S1K
    S = S1K.s2(); sid = t['platform_surface']; mesh = S.build_mesh(sid); dom_l = S.roi_domain(mesh)
    R = raster_for(t); peak = peak_for(t, R); pattern = (R['binary'] * np.float32(peak)).astype(np.float32)
    ids = identities(t, bd, parts, peak); calls = dict(render=0, decode=0)
    ip = out_dir / 'images' / f"{t['image_key']}.json"
    if complete(ip, ids['image']):
        with np.load(ip.with_suffix('.npz')) as z:
            img = np.asarray(z['image'])
        img_src = 'S2 stored'
    else:
        seed = ids['seed'] or 0
        tok = ledger.start('render', t['image_key'])
        try:
            res = bd.call('render', bd.sid(t['surface']), pattern, float(peak), float(t['sigma']), seed, K.PSF_PX)
            img = np.asarray(res['image'])
            if img.dtype != np.uint8 or img.shape != (512, 512):
                raise RuntimeError(f'render returned {img.dtype} {img.shape}')
        except Exception as e:
            ledger.end(tok, 'failed', repr(e)[:300]); raise
        h = _save_npz(ip.with_suffix('.npz'), image=img)
        K.save_json(ip, dict(identity=ids['image'], image_key=t['image_key'], cand_id=t['cand_id'], raster_key=t['raster_key'], surface=t['surface'], platform_surface=sid, family=t['family'],
                             sigma=float(t['sigma']), noise_seed=ids['seed'], m3_seed_passed=seed, peak=peak, peak_f32=float(np.float32(peak)), integral_I0=float(R['binary'].sum(dtype=np.float64)),
                             integral_actual=float(pattern.sum(dtype=np.float64)), psf_px=K.PSF_PX, render_meta=K.clean(res.get('meta', {})), platform_version=bd.version, freeze_parts=parts,
                             npz_sha256=h, complete=True, t_local=K.now_local()))
        ledger.end(tok, 'ok', dict(npz_sha256=h)); calls['render'] = 1; img_src = 'S2 new render'
    ctx = dict(binary=R['binary'], centres=R['centres'], pattern=pattern, peak=float(peak), size_mm=float(t['size_mm']), meta_full=R['meta_full'], row=dict(radius_mm=float(t['size_mm'])))
    rp = out_dir / 'raw' / f"{t['decode_key']}.json"
    if complete(rp, ids['decode']):
        dec = load_raw(rp, t['family'])
    else:
        tok = ledger.start('decode', t['decode_key'])
        try:
            dec = bd.call('decode', bd.method(K.ARM[t['family']]), img, ctx)
        except Exception as e:
            ledger.end(tok, 'failed', repr(e)[:300]); raise
        a, meta = raw_arrays(t['family'], dec)
        cfg_eff = (dec.get('aux') or {}).get('cfg') or {}
        h = _save_npz(rp.with_suffix('.npz'), **a)
        K.save_json(rp, dict(identity=ids['decode'], decode_key=t['decode_key'], image_key=t['image_key'], image_identity=ids['image'], image_source=img_src, cand_id=t['cand_id'],
                             method=K.ARM[t['family']], effective=dict(window=list(bd.window()), decode_cfg={k: v for k, v in cfg_eff.items() if isinstance(v, (int, float, str, bool)) or v is None},
                                                                     p4=(K.clean(S.P4) if t['family'] == 'line' else None), pattern_peak_passed=(float(peak) if t['family'] == 'line' else None)),
                             meta=meta, platform_version=bd.version, freeze_parts=parts, npz_sha256=h, complete=True, t_local=K.now_local()))
        ledger.end(tok, 'ok', dict(npz_sha256=h, n_points=int(len(a['pts'])))); calls['decode'] = 1
    ep = out_dir / 'eval' / f"{t['decode_key']}.json"
    if complete(ep, ids['eval']):
        return dict(decode_key=t['decode_key'], state=K.load_json(ep)['metrics']['state'], image_source=img_src, calls=calls, evaluated=False)
    t0 = time.time()
    m, arrays, info = evaluate(t, sid, dec, ctx, mesh, dom_l, dom_g)
    h = _save_npz(ep.with_suffix('.npz'), **arrays)
    K.save_json(ep, dict(identity=ids['eval'], eval_key=ids['eval'], decode_key=t['decode_key'], decode_identity=ids['decode'], image_key=t['image_key'], cand_id=t['cand_id'], surface=t['surface'],
                         platform_surface=sid, family=t['family'], layout=t['layout'], regime=t['regime'], sigma=float(t['sigma']), phase_label=t['phase_label'], rep=int(t['rep']), noise_seed=ids['seed'],
                         peak=peak, metrics=m, info=info, platform_version=bd.version, freeze_parts=parts, seconds=round(time.time() - t0, 2), npz_sha256=h, complete=True, t_local=K.now_local()))
    return dict(decode_key=t['decode_key'], state=m['state'], image_source=img_src, calls=calls, evaluated=True)
