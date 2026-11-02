# Build size-spacing rasters and prepare per-case imaging inputs.

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_common as K

_C2 = None


def _s2_cands():
    global _C2
    if _C2 is None:
        import pandas as pd
        _C2 = pd.read_csv(K.S2OUT / 's2_candidates.csv', keep_default_na=False, float_precision='round_trip').set_index('cand_id')
    return _C2


def raster_of(u: dict) -> dict:

    if u['raster_source'] == 's1_store':
        import s1r_common as S1K
        R = S1K.load_raster(u['raster_key'])
        return dict(binary=np.asarray(R['binary'], np.float32), centres=R['centres'], meta_full=R['meta_full'])
    import s2p_case as SC
    t = _s2_cands().loc[u['template_cand_id']].to_dict(); t['cand_id'] = u['template_cand_id']
    for k in ('phase_u', 'phase_v', 'phase_frac', 'p_parallel_mm', 'p_perp_mm', 'line_pitch_mm', 'orientation_deg', 'size_mm'):
        t[k] = (float(t[k]) if str(t[k]) not in ('', 'nan') else None)
    if t['raster_key'] != u['raster_key']:
        raise K.Stop('technical_stop', f"template {u['template_cand_id']} has raster {t['raster_key']} != {u['raster_key']}")
    R = SC.raster_for(t)
    return dict(binary=np.asarray(R['binary'], np.float32), centres=R['centres'], meta_full=R['meta_full'])


def peak_of(u: dict, R: dict) -> float:

    pk = set()
    for reg in str(u['regimes']).split(';'):
        r = K.resource(R['binary'], reg)
        if K.resource_check(r, reg):
            raise K.Stop('technical_stop', f"{u['input_key']}: resource check {K.resource_check(r, reg)}")
        pk.add(r['peak_f32'])
    if len(pk) != 1 or float(np.float32(float(u['peak_f32']))) not in pk:
        raise K.Stop('technical_stop', f"{u['input_key']}: peaks {sorted(pk)} vs manifest {u['peak_f32']}")
    return float(u['peak_target'])


def context(u: dict, R: dict, peak: float) -> dict:
    pattern = (R['binary'] * np.float32(peak)).astype(np.float32)
    return dict(binary=R['binary'], centres=R['centres'], pattern=pattern, peak=float(peak), size_mm=float(u['size_mm']), meta_full=R['meta_full'], row=dict(radius_mm=float(u['size_mm'])))


# Render missing size-spacing inputs, decode them and save evaluated results.
def run_new(u: dict, bd, dom_g: dict, dom_l: dict, ledger) -> dict:

    import s1r_eval as E
    import s2p_case as SC
    S = K.s2(); sid = u['platform_surface']; mesh = S.build_mesh(sid)
    R = raster_of(u); peak = peak_of(u, R); ctx = context(u, R, peak)
    seed = None if float(u['sigma']) == 0 else int(u['noise_seed'])
    ik, dk, ek = u['input_key'], u['decode_key'], u['eval_key']
    if K.input_key(u['raster_key'], u['peak_f32'], sid, u['sigma'], seed) != ik or K.decode_key(ik, u['family'], u['peak_f32']) != dk or K.eval_key(dk) != ek:
        raise K.Stop('platform_mismatch', f'{ik}: identities differ from the current platform (manifest frozen on another state)')
    calls = dict(render=0, decode=0); t = dict(render=0.0, decode=0.0, eval=0.0)
    ip = K.IMG_DIR / f'{ik}.json'
    if K.complete(ip, ik):
        with np.load(ip.with_suffix('.npz')) as z:
            img = np.asarray(z['image'])
    else:
        t0 = time.time(); tok = ledger.start('render', ik)
        try:
            res = bd.call('render', bd.sid(u['surface']), ctx['pattern'], float(peak), float(u['sigma']), seed or 0, K.PSF_PX)
            img = np.asarray(res['image'])
            if img.dtype != np.uint8 or img.shape != (512, 512):
                raise RuntimeError(f'render returned {img.dtype} {img.shape}')
        except Exception as e:
            ledger.end(tok, 'failed', repr(e)[:300]); raise
        h = K.save_npz(ip.with_suffix('.npz'), image=img)
        K.save_json(ip, dict(identity=ik, input_key=ik, raster_key=u['raster_key'], surface=u['surface'], platform_surface=sid, family=u['family'], size_mm=float(u['size_mm']),
                             layout=u['layout'], phase_label=u['phase_label'], sigma=float(u['sigma']), rep=int(u['rep']), noise_seed=seed, m3_seed_passed=(seed or 0), psf_px=K.PSF_PX,
                             peak_target=float(peak), peak_f32=float(np.float32(peak)), Q0=float(u['Q0']), Q_actual=float(ctx['pattern'].sum(dtype=np.float64)), regimes=u['regimes'],
                             sections=u['sections'], end_grey=dict(frac_0=float((img == 0).mean()), frac_255=float((img == 255).mean())), render_meta=K.clean(res.get('meta', {})),
                             platform_version=bd.version, npz_sha256=h, complete=True, t_local=K.now_local()))
        ledger.end(tok, 'ok', dict(npz_sha256=h)); calls['render'] = 1; t['render'] = time.time() - t0
    rp = K.RAW_DIR / f'{dk}.json'
    if K.complete(rp, dk):
        dec = SC.load_raw(rp, u['family'])
    else:
        t0 = time.time(); tok = ledger.start('decode', dk)
        try:
            dec = bd.call('decode', bd.method(K.ARM[u['family']]), img, ctx)
        except Exception as e:
            ledger.end(tok, 'failed', repr(e)[:300]); raise
        a, meta = SC.raw_arrays(u['family'], dec)
        cfg_eff = (dec.get('aux') or {}).get('cfg') or {}
        h = K.save_npz(rp.with_suffix('.npz'), **a)
        K.save_json(rp, dict(identity=dk, decode_key=dk, input_key=ik, method=K.ARM[u['family']], meta=meta,
                             effective=dict(window=list(bd.window()), decode_cfg={k: v for k, v in cfg_eff.items() if isinstance(v, (int, float, str, bool)) or v is None},
                                            p4=(K.clean(S.P4) if u['family'] == 'line' else None), pattern_peak_passed=(float(peak) if u['family'] == 'line' else None)),
                             platform_version=bd.version, npz_sha256=h, complete=True, t_local=K.now_local()))
        ledger.end(tok, 'ok', dict(npz_sha256=h, n_points=int(len(a['pts'])))); calls['decode'] = 1; t['decode'] = time.time() - t0
    ep = K.EVAL_DIR / f'{dk}.json'
    if K.complete(ep, ek):
        m = K.load_json(ep)['metrics']
        return dict(input_key=ik, state=m['state'], calls=calls, evaluated=False, seconds=t, glob_Ctau=m['glob_Ctau'])
    t0 = time.time()
    m, arrays, info, _ = E.evaluate_case(sid, u['family'], dec, ctx, mesh, dom_l, dom_g, sid == 'A_curv_gauss')
    h = K.save_npz(ep.with_suffix('.npz'), **arrays)
    K.save_json(ep, dict(identity=ek, eval_key=ek, decode_key=dk, input_key=ik, surface=u['surface'], platform_surface=sid, family=u['family'], size_mm=float(u['size_mm']), layout=u['layout'],
                         phase_label=u['phase_label'], sigma=float(u['sigma']), rep=int(u['rep']), noise_seed=seed, peak_f32=float(np.float32(peak)), regimes=u['regimes'], sections=u['sections'],
                         metrics=m, info=info, platform_version=bd.version, npz_sha256=h, complete=True, t_local=K.now_local()))
    t['eval'] = time.time() - t0
    return dict(input_key=ik, state=m['state'], calls=calls, evaluated=True, seconds=t, glob_Ctau=m['glob_Ctau'], pid=os.getpid())


REPLAY_PTS_ATOL = 1e-5
REPLAY_SCALAR_ATOL = 1e-6


def replay(u: dict, bd, dom_g: dict, dom_l: dict, ledger) -> dict:
    import s1r_eval as E
    import s12_reuse as RU
    S = K.s2(); sid = u['platform_surface']; mesh = S.build_mesh(sid)
    R = raster_of(u); peak = peak_of(u, R); ctx = context(u, R, peak)
    ip = RU.image_record_path(_ns(u)); ep = RU.eval_record_path(_ns(u))
    with np.load(ip.with_suffix('.npz')) as z:
        img = np.asarray(z['image'])
    old = K.load_json(ep)
    with np.load(ep.with_suffix('.npz')) as z:
        A0 = {k: np.asarray(z[k]) for k in z.files}
    tok = ledger.start('replay_decode', u['decode_key'])
    try:
        dec = bd.call('decode', bd.method(K.ARM[u['family']]), img, ctx)
    except Exception as e:
        ledger.end(tok, 'failed', repr(e)[:300]); raise
    ledger.end(tok, 'ok')
    m, A1, info, _ = E.evaluate_case(sid, u['family'], dec, ctx, mesh, dom_l, dom_g, sid == 'A_curv_gauss')
    m0 = old['metrics']
    masks = [k for k in A1 if A1[k].dtype == bool and k in A0]
    mask_eq = {k: bool(np.array_equal(A0[k].astype(bool), A1[k].astype(bool))) for k in masks}
    ints = [k for k in A1 if np.issubdtype(A1[k].dtype, np.integer) and k in A0]
    int_eq = {k: bool(np.array_equal(A0[k], A1[k].astype(A0[k].dtype))) for k in ints}
    p0 = np.asarray(A0['pts'], np.float64); p1 = np.asarray(A1['pts'], A0['pts'].dtype).astype(np.float64)
    pts_ok = bool(p0.shape == p1.shape and np.allclose(p0, p1, rtol=0.0, atol=REPLAY_PTS_ATOL, equal_nan=True))
    dmax = float(np.nanmax(np.abs(p0 - p1))) if p0.shape == p1.shape and np.isfinite(p0).any() else (0.0 if p0.size == 0 else None)
    cnt_keys = [k for k, v in m0.items() if isinstance(v, int) and not isinstance(v, bool)]
    cnt_eq = {k: m0[k] == m.get(k) for k in cnt_keys}
    sc_keys = [k for k, v in m0.items() if isinstance(v, float)]
    sc_diff = {k: (abs(m0[k] - m[k]) if (m0[k] is not None and m.get(k) is not None) else (0.0 if m0[k] == m.get(k) else float('inf'))) for k in sc_keys}
    none_eq = all((m0[k] is None) == (m.get(k) is None) for k in m0 if not isinstance(m0[k], (dict, list, str, bool)))
    ok = all(mask_eq.values()) and all(int_eq.values()) and pts_ok and all(cnt_eq.values()) and (max(sc_diff.values()) if sc_diff else 0.0) <= REPLAY_SCALAR_ATOL and none_eq
    return dict(input_key=u['input_key'], source=u['source'], surface=u['surface'], family=u['family'], regimes=u['regimes'], sigma=float(u['sigma']), state_old=m0['state'], state_new=m['state'],
                masks_equal=all(mask_eq.values()), masks_differing=[k for k, v in mask_eq.items() if not v], int_arrays_equal=all(int_eq.values()), int_arrays_differing=[k for k, v in int_eq.items() if not v],
                points_within_atol=pts_ok, points_dtype_stored=str(A0['pts'].dtype), points_max_abs_diff_mm=dmax, counts_equal=all(cnt_eq.values()), counts_differing=[k for k, v in cnt_eq.items() if not v],
                scalar_max_abs_diff=(max(sc_diff.values()) if sc_diff else 0.0), undefined_pattern_equal=none_eq, ok=bool(ok))


def _ns(u: dict):
    from types import SimpleNamespace
    return SimpleNamespace(**u)
