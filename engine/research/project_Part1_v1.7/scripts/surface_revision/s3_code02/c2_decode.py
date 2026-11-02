# Read coded point or line classes with the selected detector settings.

from __future__ import annotations

import math

import numpy as np

import c2_common as K

W_CAL = dict(narrow=0.75 / 0.18, wide=1.25 / 0.18)
KAPPA_A3 = 6.05 / math.sqrt(W_CAL['narrow'] * W_CAL['wide'])
RULES_DOTS = ('B0', 'DF', 'DS')


def dot_decode(img, binpat, k: int, radii_mm, rule: str):
    import s3c_dev as DV
    return DV.dot_decode(img, binpat, k, radii_mm, rule, 'enc')


def dot_dec(D: dict, arm: str) -> dict:
    import s3c_dev as DV
    return DV.dot_dec(D, arm)


def line_front(img, inp: dict) -> dict:
    import lc_pipeline as LC
    import lc_readout as LR
    import m4_lines as ML
    import rdy_decode as DC
    import rdy_platform as P
    import s3c_dev as DV
    meta = inp['meta_full']['olines']; cfg = P.s2().line_cfg(float(inp['peak'])); fam = ML.line_family(meta); Pm = DC.lc_params()
    w = (meta.get('code') or {}).get('widths_px')
    if not w:
        raise K.Stop('technical', 'line_front needs a coded pattern (code.widths_px)')
    Pm = dict(Pm, LC=dict(Pm['LC'], class_widths_px=dict(narrow=float(w['narrow']), wide=float(w['wide']))))
    gm = math.sqrt(float(w['narrow']) * float(w['wide']))
    front = LC.shared_front_end(img, meta, cfg, inp['pattern'], Pm['LC'])
    ro = LR.read_segments(np.asarray(img, np.float64), front['feat'], front['segments'], front['guard_reason'] == 0, fam, Pm['LC']['steger_sigma_px'], Pm['LC']['class_widths_px'],
                          dict(Pm['readout'], threshold_px=KAPPA_A3 * gm))
    bw = DV.block_widths(img, front, Pm, 8)
    wm = meta['code']['widths_mm']
    return dict(front=front, readout=ro, bw=bw, Pm=Pm, fam=fam, cfg=cfg, gm=gm, rho_d=float(wm['wide']) / float(wm['narrow']))


def line_rule(F: dict, rule: str, kappa: float | None = None) -> dict:
    import lc_pipeline as LC
    import lc_r3_reliability as REL
    import lc_r4_runtime as RT
    import s3c_dev as DV
    front = F['front']
    if rule == 'ignore':
        r = LC.finish(front, front['cand'], dict(class_mode='ignore')); r['segment_class'] = np.full(len(front['segments']), -1, np.int64)
        r.update(threshold_px=None, decision='ignore', segment_table={}, image_fallback=False); return r
    thr = float(kappa) * F['gm']
    if rule == 'R':
        Pm = dict(F['Pm'], old_threshold_px=thr)
        r = RT.run_arms(RT.inputs_from_front(front, F['readout']), Pm, arms=('R',))['arms']['R']
        return r
    if rule == 'LB':
        bw = F['bw']; ok = np.isfinite(bw['fw']); peaks = REL.image_peaks(bw['fw'], ok, front['chain'], F['Pm']['reliability'])
        t, why = REL.choose_threshold('R', peaks, thr)
        r = DV.line_rule_record(front, DV.table_from_blocks(bw['per_seg'], t, 'px', F['rho_d']), 'LB', t, why)
        return r
    raise K.Stop('technical', f'unknown line rule {rule}')


def line_dec(rec: dict) -> dict:
    return dict(pts=np.asarray(rec['points'], np.float64).reshape(-1, 3), accepted=np.asarray(rec['accepted'], bool), aux=dict(r=rec))


def plain_line_decode(img, inp: dict) -> dict:
    import rdy_decode as DC
    return DC.dec_binding(img, dict(inp, centres=None), 'P4', 'P4', None)['arms']['P4']
