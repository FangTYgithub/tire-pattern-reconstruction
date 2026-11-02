# Dispatch registered point and line decoding methods.

from __future__ import annotations

import numpy as np

import rdy_common as K
import rdy_platform as P

METHODS = {
    'ENC': 'r5_detect.candidates -> select D0 (k 1) | ep_detect.select_coded (k > 1) -> ep_detect.reconstruct_from x2 on the same features (coded / ignore_class), platform window injected',
    'LC': 'lc_pipeline.shared_front_end -> lc_readout.read_segments -> lc_r4_runtime.run_arms (ignore / use / R / A), s2.line_cfg(peak), lc_r4_common.runtime_params()',
    'P4REF': 'lc_pipeline.run_lc(P4_ref params of lc_r4_common.runtime_params(), modes=ignore), s2.line_cfg(peak)',
    'D0': 'binding adp_surf02.decode("D0") (s2.dot_cfg, actual raster)',
    'P4': 'binding adp_surf02.decode("P4_C3_cal") (s2.line_cfg(actual peak), s2.P4)',
    'P4TABLE': 'sp2_lines_table.run_p4_table(table spec, s2.line_cfg(peak), pattern, sp_regression.p4_params())',
}
ARM_METHOD = {'ENC_coded': 'ENC', 'ENC_ignore': 'ENC', 'LC_ignore': 'LC', 'LC_use': 'LC', 'LC_R': 'LC', 'LC_A': 'LC', 'P4_ref': 'P4REF', 'SPAT_D0': 'D0', 'SPAT_P4table': 'P4TABLE',
              'D0': 'D0', 'P4_C3_cal': 'P4'}
POINT_METHODS = ('ENC', 'D0')


def _check_window(z0, z1, margin=None):
    if abs(float(z0) - K.WINDOW_MM[0]) > 1e-9 or abs(float(z1) - K.WINDOW_MM[1]) > 1e-9:
        raise K.Stop('platform', f'effective window {(z0, z1)} != {K.WINDOW_MM}')
    if margin is not None and abs(float(margin) - K.MARGIN_MM) > 1e-12:
        raise K.Stop('platform', f'effective line margin {margin} != {K.MARGIN_MM}')


def window_cfg_dots(cfg: dict) -> dict:
    S = P.s2()
    return {**cfg, 'search_z_min_mm': S.Z0_MM, 'measurement_volume_mm': S.VOL_MM}


def _accept(pts, tr) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 3); pidx = np.asarray(tr['point_idx'], np.int64); has = pidx >= 0
    acc = np.zeros(len(pts), bool); acc[pidx[has]] = True
    amb = np.zeros(len(pts), bool); amb[pidx[has & np.asarray(tr['ambiguous'], bool)]] = True
    return acc & ~amb & np.isfinite(pts).all(axis=1)


def projector_side_window(binpat, cfg, ptype):
    import m4_reconstruction as M4
    import ep_detect as ED
    if ptype == 'coded_dots':
        pts, ids, areas = M4.pattern_features(binpat, ptype, return_areas=True); cls = M4.projector_size_classes(areas, cfg)
    else:
        pts, ids = M4.pattern_features(binpat, ptype); cls = np.zeros(len(pts), np.int64)
    z0, z1 = M4.search_window(cfg, ED.search_depth(cfg))
    A, B = M4.projector_tracks(pts, cfg, z0, z1)
    return dict(pts=pts, ids=np.asarray(ids), cls=np.asarray(cls, np.int64), A=A, B=B, window=(z0, z1))


def guard_reasons(cinfo: dict, radii_px) -> dict:
    import math
    import m4_reconstruction as M4
    span_design = 2.0 * math.log(max(radii_px) / min(radii_px)) if radii_px and min(radii_px) > 0 else None
    d, b, s = cinfo.get('dprime', float('nan')), cinfo.get('balance', 0.0), cinfo.get('span', float('nan'))
    r = dict(dprime=bool(np.isfinite(d) and d >= M4.SIZE_DPRIME_MIN), balance=bool(b >= M4.SIZE_BALANCE_MIN),
             span=bool(not span_design or (M4.SIZE_SPAN_MIN_FRAC * span_design <= s <= M4.SIZE_SPAN_MAX_FRAC * span_design)))
    readable = all(r.values())
    if readable != bool(cinfo.get('readable')):
        raise K.Stop('technical', 'guard reason split disagrees with select_coded readability')
    return dict(readable=readable, fallback=(not readable), failed=[k for k, ok in r.items() if not ok], dprime=d, balance=b, span=s, span_design=span_design,
                thresholds=dict(dprime_min=M4.SIZE_DPRIME_MIN, balance_min=M4.SIZE_BALANCE_MIN, span_frac=[M4.SIZE_SPAN_MIN_FRAC, M4.SIZE_SPAN_MAX_FRAC]))


def enc_decode(img, binpat, spec: dict, cfg_fn=window_cfg_dots) -> dict:

    import ep_detect as ED
    import ep_patterns as EP
    import m4_reconstruction as M4
    import r5_detect as RD
    binpat = np.asarray(binpat, np.float32)
    cd = RD.candidates(img)
    k = int(spec['k']); radii_px = [float(r) / EP.mmpp() for r in spec['radii_mm']]
    if k == 1:
        feats, det_info, code, fidx, _ref = RD.select(cd, 'D0')
        cfg = cfg_fn(dict(ED.CFG)); ptype = 'dots'; allowed = np.ones((len(feats), 1), bool); cinfo = dict(readable=None)
    else:
        feats, det_info, code, fidx, cinfo = ED.select_coded(cd, k, radii_px, spec.get('code_ref', 'global'))
        cfg = cfg_fn(ED.coded_cfg(spec['radii_mm'], k, EP.mmpp())); ptype = 'coded_dots'; allowed = det_info['feature_allowed']
    modes = {}
    pts, info, tr = ED.reconstruct_from(feats, det_info, binpat, cfg, ptype, allowed if k > 1 else None)
    modes['coded'] = dict(pts=pts, info=info, tr=tr)
    if k > 1:
        pts_i, info_i, tr_i = ED.reconstruct_from(feats, det_info, binpat, cfg, ptype, np.ones_like(allowed))
        modes['ignore_class'] = dict(pts=pts_i, info=info_i, tr=tr_i)
    proj = projector_side_window(binpat, cfg, ptype)
    cand = ED.track_candidates(feats, proj, cfg, allowed if k > 1 else None)
    return dict(cd=cd, feats=feats, det_info=det_info, code=code, fidx=fidx, cinfo=cinfo, cfg=cfg, ptype=ptype, allowed=allowed, modes=modes, proj=proj, cand=cand,
                window=M4.search_window(cfg, ED.search_depth(cfg)), guard=(guard_reasons(cinfo, radii_px) if k > 1 else None))


def dec_enc(img, inp: dict) -> dict:
    D = enc_decode(img, inp['binary'], inp['enc_spec'])
    _check_window(*D['window'])
    if int(inp['enc_spec']['k']) == 1:
        S = P.s2(); ref = S.dot_cfg()
        if any(D['cfg'].get(k) != v for k, v in ref.items()):
            raise K.Stop('platform', 'ENC plain cfg differs from s2.dot_cfg()')
    arms = {}
    for mode, arm in (('coded', 'ENC_coded'), ('ignore_class', 'ENC_ignore')):
        if mode not in D['modes']:
            continue
        M = D['modes'][mode]
        arms[arm] = dict(pts=np.asarray(M['pts'], np.float64).reshape(-1, 3), accepted=_accept(M['pts'], M['tr']),
                         aux=dict(feats=D['feats'], tr=M['tr'], cd=D['cd'], sel_code=D['code'], sel_fidx=D['fidx'], det_info=D['det_info'], info=M['info']))
    return dict(method='ENC', arms=arms, shared=D, window=tuple(float(x) for x in D['window']))


def lc_params():
    import lc_r4_common as RC
    return RC.runtime_params()


def lc_arms(img, olines_meta: dict, pat, peak: float, coded: bool, cfg=None) -> dict:
    import lc_pipeline as LC
    import lc_readout as LR
    import lc_r4_runtime as RT
    import m4_lines as ML
    Pm = lc_params()
    cfg = P.s2().line_cfg(float(peak)) if cfg is None else cfg
    if not coded:
        r = LC.run_lc(img, olines_meta, cfg, pat, Pm['P4_ref'], modes=('ignore',))['ignore']
        return dict(arms={'P4_ref': r}, readout=None, peaks=None, hist=None, shared_identity=r.get('shared_identity'))
    fam = ML.line_family(olines_meta)
    front = LC.shared_front_end(img, olines_meta, cfg, pat, Pm['LC'])
    readout = LR.read_segments(np.asarray(img, np.float64), front['feat'], front['segments'], front['guard_reason'] == 0, fam, Pm['LC']['steger_sigma_px'], Pm['LC']['class_widths_px'], Pm['readout'])
    runtime = RT.run_arms(RT.inputs_from_front(front, readout), Pm)
    return dict(arms={f'LC_{k}': v for k, v in runtime['arms'].items()}, readout=readout, peaks=runtime['peaks'], hist=runtime['hist'], shared_identity=front['shared_identity'])


def dec_lc(img, inp: dict, coded: bool) -> dict:
    meta = inp['meta_full']['olines']
    R = lc_arms(img, meta, inp['pattern'], inp['peak'], coded)
    arms = {}
    for arm, r in R['arms'].items():
        _check_window(r['window_z0_mm'], r['window_z1_mm'], r['depth_margin_mm'])
        arms[arm] = dict(pts=np.asarray(r['points'], np.float64).reshape(-1, 3), accepted=np.asarray(r['accepted'], bool), aux=dict(r=r))
    return dict(method=('LC' if coded else 'P4REF'), arms=arms, shared=dict(readout=R['readout'], peaks=R['peaks'], shared_identity=R['shared_identity']), window=K.WINDOW_MM)


def ctx_of(inp: dict, size_mm=None) -> dict:
    return dict(binary=inp['binary'], centres=inp['centres'], pattern=inp['pattern'], peak=float(inp['peak']), meta_full=inp['meta_full'],
                size_mm=(float(size_mm) if size_mm is not None else None), row=dict(radius_mm=(float(size_mm) if size_mm is not None else None)))


def dec_binding(img, inp: dict, method: str, arm: str, size_mm=None) -> dict:
    bd = P.bound()
    dec = bd.call('decode', bd.method({'D0': 'D0', 'P4': 'P4_C3_cal'}[method]), img, ctx_of(inp, size_mm))
    if method == 'D0':
        _check_window(*dec['aux']['window'])
    else:
        r = dec['aux']['r']; _check_window(r['window_z0_mm'], r['window_z1_mm'], r['depth_margin_mm'])
    return dict(method=method, arms={arm: dec}, shared={}, window=K.WINDOW_MM)


def dec_p4_table(img, inp: dict) -> dict:
    import sp2_lines_table as LT2
    import sp_regression as R1
    T = inp['table']
    r = LT2.run_p4_table(img, T['spec'], P.s2().line_cfg(float(inp['peak'])), inp['pattern'], R1.p4_params())
    _check_window(r['window_z0_mm'], r['window_z1_mm'], r['depth_margin_mm'])
    return dict(method='P4TABLE', arms={'SPAT_P4table': dict(pts=np.asarray(r['points'], np.float64).reshape(-1, 3), accepted=np.asarray(r['accepted'], bool), aux=dict(r=r))},
                shared={}, window=K.WINDOW_MM)


def decode(method: str, img, inp: dict, row: dict) -> dict:
    if method == 'ENC':
        return dec_enc(img, inp)
    if method == 'LC':
        return dec_lc(img, inp, True)
    if method == 'P4REF':
        return dec_lc(img, inp, False)
    if method == 'P4TABLE':
        return dec_p4_table(img, inp)
    if method in ('D0', 'P4'):
        size = float(row['radius_mm']) if (method == 'D0' and str(row.get('radius_mm', '')) not in ('', 'nan')) else None
        if method == 'D0' and row['section'] == 'S4':
            import sp2_common as S2
            size = float(S2.R_MM)
        return dec_binding(img, inp, method, row['arm'], size)
    raise K.Stop('technical', f'unknown decode method {method!r}')


_TR = ('matched', 'ambiguous', 'duplicate', 'best_idx', 'point_idx')
_LR = ('features', 'line_id', 'status', 'chain', 'segment_id', 'n_candidates', 'guard_reason', 'points', 'accepted')


def _scal(d: dict) -> dict:
    return K.clean({k: v for k, v in (d or {}).items() if not isinstance(v, np.ndarray)})


def pack(out: dict) -> tuple[dict, dict]:

    A: dict = {}; meta = dict(method=out['method'], window=list(out['window']), arms=sorted(out['arms']))
    if out['method'] in POINT_METHODS:
        first = next(iter(out['arms'].values()))['aux']
        cd = first['cd']
        A.update(feats=np.asarray(first['feats'], np.float32).reshape(-1, 2), cand_table=np.asarray(cd['cands'], np.float64).reshape(-1, 7), comp_reason=np.asarray(cd['reason'], np.int8),
                 comp_stats=np.asarray(cd.get('stats', np.zeros((0,))), np.int32), sel_code=np.asarray(first['sel_code'], np.int8), sel_fidx=np.asarray(first['sel_fidx'], np.int32))
        di = first['det_info'] or {}
        for k in ('feature_classes', 'feature_allowed', 'feature_areas_px'):
            if isinstance(di.get(k), np.ndarray):
                A[f'det_{k}'] = di[k]
        meta.update(cd=dict(n_labels=int(cd['n_labels']), thresh=float(cd['thresh']), n_filtered=int(cd['n_filtered']), n_border=int(cd['n_border'])), det_info=_scal(di))
        for i, arm in enumerate(sorted(out['arms'])):
            d = out['arms'][arm]; tr = d['aux']['tr']; info = d['aux']['info'] or {}
            A[f'a{i}_pts'] = np.asarray(d['pts'], np.float64); A[f'a{i}_accepted'] = np.asarray(d['accepted'], bool)
            for k in _TR:
                A[f'a{i}_tr_{k}'] = np.asarray(tr[k])
            if 'proj_classes' in tr:
                A[f'a{i}_tr_proj_classes'] = np.asarray(tr['proj_classes'])
            if isinstance(info.get('ambiguous_mask'), np.ndarray):
                A[f'a{i}_amb'] = np.asarray(info['ambiguous_mask'], bool)
            meta[f'a{i}_info'] = _scal(info)
        if out['method'] == 'ENC':
            D = out['shared']
            A.update(cinfo_cls_all=np.asarray(D['cinfo'].get('cls_all', np.zeros(0)), np.int64), allowed=np.asarray(D['allowed'], bool), proj_ids=np.asarray(D['proj']['ids'], np.int64),
                     proj_cls=np.asarray(D['proj']['cls'], np.int64), proj_pts=np.asarray(D['proj']['pts'], np.float64), cand_n_geom=np.asarray(D['cand']['n_geom'], np.int64),
                     cand_n_class=np.asarray(D['cand']['n_class'], np.int64),
                     cand_geom_ids=np.array([','.join(map(str, np.asarray(g).tolist())) for g in D['cand']['geom_ids']], dtype=str),
                     cand_class_ids=np.array([','.join(map(str, np.asarray(g).tolist())) for g in D['cand']['class_ids']], dtype=str))
            meta.update(cinfo=_scal(D['cinfo']), guard=K.clean(D['guard']), ptype=D['ptype'], cfg=_scal(D['cfg']), proj_window=list(D['proj']['window']))
    else:
        for i, arm in enumerate(sorted(out['arms'])):
            r = out['arms'][arm]['aux']['r']
            for k in _LR:
                A[f'a{i}_r_{k}'] = np.asarray(r[k])
            if 'segment_class' in r:
                A[f'a{i}_r_segment_class'] = np.asarray(r['segment_class'])
            meta[f'a{i}_r'] = dict(detect_info=_scal(r.get('detect_info')), counts=K.clean(r.get('counts')), checks=K.clean(r.get('checks')), window_z0_mm=float(r['window_z0_mm']),
                                   window_z1_mm=float(r['window_z1_mm']), depth_margin_mm=float(r['depth_margin_mm']), threshold_px=K.clean(r.get('threshold_px')), decision=r.get('decision'),
                                   image_fallback=bool(r.get('image_fallback', False)), runtime_hash=r.get('runtime_hash'), family_source=r.get('family_source'),
                                   segment_table=K.clean({str(k): v for k, v in (r.get('segment_table') or {}).items()}))
        sh = out.get('shared') or {}
        ro = sh.get('readout')
        if ro is not None:
            A['ro_fwhm_px'] = np.asarray(ro['fwhm_px'], np.float32); A['ro_reason'] = np.asarray([str(x) for x in ro['reason']], dtype=str)
            meta['readout_per_segment'] = K.clean({str(k): v for k, v in (ro.get('per_segment') or {}).items()})
        meta['peaks'] = K.clean(sh.get('peaks')); meta['shared_identity'] = sh.get('shared_identity')
    return A, meta


def unpack(A: dict, meta: dict) -> dict:
    method = meta['method']; arms = {}
    if method in POINT_METHODS:
        cd = dict(cands=A['cand_table'], reason=A['comp_reason'], stats=A['comp_stats'], **meta['cd'])
        di = dict(meta['det_info']); di.update({k[4:]: A[k] for k in A if k.startswith('det_')})
        for i, arm in enumerate(meta['arms']):
            tr = {k: A[f'a{i}_tr_{k}'] for k in _TR}
            if f'a{i}_tr_proj_classes' in A:
                tr['proj_classes'] = A[f'a{i}_tr_proj_classes']
            info = dict(meta[f'a{i}_info'])
            if f'a{i}_amb' in A:
                info['ambiguous_mask'] = A[f'a{i}_amb']
            arms[arm] = dict(pts=A[f'a{i}_pts'], accepted=A[f'a{i}_accepted'], aux=dict(feats=A['feats'], tr=tr, cd=cd, sel_code=A['sel_code'], sel_fidx=A['sel_fidx'], det_info=di, info=info))
        shared = {}
        if method == 'ENC':
            shared = dict(cinfo=dict(meta['cinfo'], cls_all=A['cinfo_cls_all']), allowed=A['allowed'], proj=dict(ids=A['proj_ids'], cls=A['proj_cls'], pts=A['proj_pts'], window=meta['proj_window']),
                          cand=dict(n_geom=A['cand_n_geom'], n_class=A['cand_n_class'], geom_ids=[np.array([int(x) for x in s.split(',') if x != ''], np.int64) for s in A['cand_geom_ids']],
                                    class_ids=[np.array([int(x) for x in s.split(',') if x != ''], np.int64) for s in A['cand_class_ids']]),
                          guard=meta['guard'], ptype=meta['ptype'], cfg=meta['cfg'], code=A['sel_code'], fidx=A['sel_fidx'], feats=A['feats'], cd=cd, det_info=di)
        return dict(method=method, arms=arms, shared=shared, window=tuple(meta['window']))
    for i, arm in enumerate(meta['arms']):
        m = meta[f'a{i}_r']
        r = {k: A[f'a{i}_r_{k}'] for k in _LR}
        if f'a{i}_r_segment_class' in A:
            r['segment_class'] = A[f'a{i}_r_segment_class']
        r.update(detect_info=m['detect_info'], counts=m['counts'], checks=m['checks'], window_z0_mm=m['window_z0_mm'], window_z1_mm=m['window_z1_mm'], depth_margin_mm=m['depth_margin_mm'],
                 threshold_px=m['threshold_px'], decision=m['decision'], image_fallback=m['image_fallback'], runtime_hash=m['runtime_hash'], family_source=m.get('family_source'),
                 segment_table=m.get('segment_table'))
        arms[arm] = dict(pts=np.asarray(r['points'], np.float64).reshape(-1, 3), accepted=np.asarray(r['accepted'], bool), aux=dict(r=r))
    shared = dict(peaks=meta.get('peaks'), shared_identity=meta.get('shared_identity'))
    if 'ro_fwhm_px' in A:
        shared['readout'] = dict(fwhm_px=A['ro_fwhm_px'], reason=A['ro_reason'], per_segment=meta.get('readout_per_segment'))
    return dict(method=method, arms=arms, shared=shared, window=tuple(meta['window']))
