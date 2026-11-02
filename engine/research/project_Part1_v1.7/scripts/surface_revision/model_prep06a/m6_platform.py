# Prepare environment inputs and expose P4 decoding and evaluation controls.

from __future__ import annotations

import math

import numpy as np

import m6_common as K

K5 = K.K5
_ST: dict = {}


def platform():
    if not _ST:
        import mp5_inputs as IN
        import rdy_decode as DC
        import rdy_platform as RP
        import tr4_gate as G
        import tr5_chain as T5
        import m4_lines as ML
        q = IN.adapter(); K.set_sandbox_env()
        S = RP.s2()
        _ST.update(IN=IN, DC=DC, RP=RP, G=G, T5=T5, ML=ML, q=q, S=S, P4=dict(S.P4))
    return _ST


def prep(row: dict) -> dict:
    p = platform()
    out = p['IN'].prepare(row)
    K.set_sandbox_env()
    return out


def p4_nodes() -> tuple[list, list]:
    t = platform()['P4']['steger_noise_coef']
    ks = sorted(t, key=float)
    return [float(k) for k in ks], [float(t[k]) for k in ks]


def coef_at(s: float) -> float:
    ks, cs = p4_nodes()
    return float(np.interp(float(s), ks, cs))


def ridge(w_px: float, s: float, psf: float, height: float = 1.0) -> float:
    return platform()['ML'].steger_design_response(float(w_px), float(s), float(psf), float(height))


def s0_of(w_px: float) -> float:
    return max(1.0, float(w_px) / (2.0 * math.sqrt(3.0)))


def scale_star(w_px: float, psf_true: float) -> dict:

    s0 = s0_of(w_px); ks, cs = p4_nodes()
    if float(psf_true) == 1.0:
        return dict(s0=s0, s_star=s0, rule='PSF 1: s* = s0 (negative control)', scores=None)
    cand = [(k, c) for k, c in zip(ks, cs) if k >= s0 - K.SCALE_MATCH_TOL]
    if not cand:
        return dict(s0=s0, s_star=None, rule='no legal node >= s0', scores=None)
    sc = [(k, ridge(w_px, k, psf_true) / c) for k, c in cand]
    best = max(v for _, v in sc)
    ties = [k for k, v in sc if v >= best * (1.0 - K.SCALE_TIE_REL)]
    s = min(ties)
    if abs(s - s0) <= K.SCALE_MATCH_TOL:
        s = s0
    return dict(s0=s0, s_star=s, rule='max R/c over nodes >= s0, ties -> smaller', scores={f'{k:.16g}': v for k, v in sc})


def p4_decode(img, prep_: dict, steger_sigma_px: float | None = None, psf_sigma_px: float | None = None) -> tuple[dict, dict, dict]:
    p = platform(); S, DC, ML = p['S'], p['DC'], p['ML']
    inp = p['q'].inputs(prep_['decoder_row'])
    cfg = S.line_cfg(float(inp['peak']))
    params = dict(p['P4'])
    if steger_sigma_px is not None:
        params['steger_sigma_px'] = float(steger_sigma_px)
    if psf_sigma_px is not None:
        params['psf_sigma_px'] = float(psf_sigma_px)
    r = ML.run_olines(np.asarray(img, np.uint8), inp['meta_full']['olines'], cfg, pattern=np.asarray(inp['pattern'], np.float32), **params)
    if abs(float(r['window_z0_mm']) - S.Z0_MM) > 1e-9 or abs(float(r['window_z1_mm']) - S.Z1_MM) > 1e-9:
        raise K.Stop('technical', f"effective line window {r['window_z0_mm']}, {r['window_z1_mm']}")
    DC._check_window(r['window_z0_mm'], r['window_z1_mm'], r['depth_margin_mm'])
    dec = dict(pts=np.asarray(r['points'], np.float64), accepted=np.asarray(r['accepted'], bool), aux=dict(r=r, cfg=cfg))
    out = dict(method='P4', arms={'P4_C3_cal': dec}, shared={}, window=DC.K.WINDOW_MM)
    A, meta = DC.pack(out)
    raw = {k: np.asarray(v) for k, v in A.items()}
    di = r['detect_info']
    thr = dict(steger_sigma_px=float(r['params']['steger_sigma_px']), psf_sigma_px=float(r['params']['psf_sigma_px']), pattern_peak=float(r['params']['pattern_peak']),
               thr_design=float(di['steger_threshold_design']), thr_final=float(di['steger_threshold']), sigma_hat=K.num(di.get('noise_sigma_hat')),
               coef=float(di['steger_noise_coef_effective']), coef_id=di.get('steger_noise_coef_id'))
    thr['thr_noise'] = (thr['coef'] * thr['sigma_hat']) if (thr['sigma_hat'] is not None and thr['sigma_hat'] > 0) else 0.0
    thr['dominant'] = 'design' if thr['thr_design'] >= thr['thr_noise'] else 'noise'
    return raw, dict(pack_meta=K5.clean(meta), method_label='P4'), thr


def evaluate(prep_: dict, method: str, raw: dict, dmeta: dict) -> tuple[dict, dict, dict]:
    p = platform(); G, T5, q = p['G'], p['T5'], p['q']
    r = dict(prep_['decoder_row'], surface=prep_['surface'], key=prep_['case_id'], _decode_meta=dmeta)
    ev, rec = q.evaluate(r, method, raw)
    ev = {k: np.asarray(v) for k, v in ev.items()}
    mtr = rec['metrics']; pc = G.check_derived(ev, prep_['family']); mi = G.check_metrics(mtr, pc)
    farr, finfo = T5.full_identity(dict(family=prep_['family'], surface=prep_['surface'], forward_key=prep_['case_id']), prep_['ctx'], ev)
    ev.update(farr)
    if prep_['family'] == 'line':
        T5.check_full(ev, finfo)
    return ev, dict(metrics=K5.clean(mtr), support=K5.clean(rec.get('support')), metric_info=mi, full_identity=K5.clean(finfo), state=mtr['state']), finfo


def cwu(rec: dict, family: str) -> dict:

    m = rec['metrics']; fi = ((rec.get('full_identity') or {}).get('counts') or {}).get('all') if family == 'line' else None
    c, w, u = ((fi['c'], fi['w'], fi['u']) if fi else (m.get('id_correct'), m.get('id_wrong'), m.get('id_unconf')))
    ws = (w / (c + w)) if (c is not None and (c + w) > 0) else None
    us = (u / (c + w + u)) if (c is not None and (c + w + u) > 0) else None
    return dict(c=c, w=w, u=u, w_share=ws, u_share=us)


def r3_multi(pts_valid: np.ndarray, surface: str) -> dict:

    from surface_eval_r3 import surface_eval_r3_dev
    p = platform(); RP = p['RP']; S = p['S']
    out = {}
    dl = RP.dom_local(surface); dg = RP.dom_global(); sid = RP.K.SURF_MAP[surface]
    doms = dict(loc=dict(x=dl['x'], y=dl['y'], z_ref=dl['z_ref'], area=dl['area'], hash=dl['hash']),
                glob=dict(x=dg['x'], y=dg['y'], z_ref=dg[f'z_{sid}'], area=np.full(len(dg['x']), dg['step'] ** 2), hash=dg['hash']))
    for name, D in doms.items():
        if len(pts_valid) < 3:
            out[name] = None
            continue
        ev = surface_eval_r3_dev(pts_valid, None, D, list(K.DS), list(K.TAUS), return_fields=False, support_rule='exact')
        out[name] = ev['subdomains']['global']
    return out
