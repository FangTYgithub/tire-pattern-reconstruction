# Store encoding images, decoding outputs and evaluation arrays.

from __future__ import annotations

import numpy as np

import c2_decode as CD
import c2_common as K
import c3_raster as RS


def inputs_of(row: dict) -> dict:
    lev = RS.levels(); phd = dict(phase_id=row['phase_id'], u=row.get('u'), v=row.get('v'), phase_frac=row.get('phase_frac'))
    B = RS.build(row['family'], str(row['layout']), row['pattern'], phd, lev)
    if RS.raster_key(B['binary']) != row['raster_key']:
        raise K.Stop('technical', f"{row['image_key']}: raster does not reproduce the manifest key")
    peak = float(row['peak_f32']); pattern = (B['binary'] * np.float32(peak)).astype(np.float32)
    return dict(B=B, binary=B['binary'], peak=peak, pattern=pattern, meta_full=B['meta_full'], centres=B['centres'], classes=B['classes'])


def _dots_pack(D: dict) -> tuple[dict, dict]:
    cd = D['cd']; A = dict(feats=np.asarray(D['feats'], np.float32).reshape(-1, 2), cands=np.asarray(cd['cands'], np.float64).reshape(-1, 7), reason=np.asarray(cd['reason'], np.int8),
                          label_map=np.asarray(cd['label_map'], np.int32), code=np.asarray(D['code'], np.int8), fidx=np.asarray(D['fidx'], np.int64))
    meta = dict(cd={k: cd.get(k) for k in ('thresh', 'n_labels', 'n_filtered', 'n_border', 'n_floor', 'a_floor')}, det_info={k: v for k, v in (D['det_info'] or {}).items() if not isinstance(v, np.ndarray)},
                cfg={k: v for k, v in D['cfg'].items() if not isinstance(v, np.ndarray)}, ptype=D['ptype'], arms=list(D['arms']), guard=K.clean(D.get('guard')))
    if D.get('cinfo') is not None:
        A['cls_all'] = np.asarray(D['cinfo']['cls_all'], np.int64); A['allowed'] = np.asarray(D['allowed'], bool)
        meta['cinfo'] = {k: v for k, v in D['cinfo'].items() if not isinstance(v, np.ndarray)}
    for i, arm in enumerate(D['arms']):
        pts, info, tr = D['arms'][arm]; A[f'a{i}_pts'] = np.asarray(pts, np.float64).reshape(-1, 3)
        for k in ('matched', 'ambiguous', 'duplicate', 'best_idx', 'point_idx'):
            A[f'a{i}_tr_{k}'] = np.asarray(tr[k])
        meta[f'a{i}_info'] = {k: v for k, v in (info or {}).items() if not isinstance(v, np.ndarray)}
        if isinstance((info or {}).get('ambiguous_mask'), np.ndarray):
            A[f'a{i}_amb'] = np.asarray(info['ambiguous_mask'], bool)
    return A, meta


def _dots_unpack(A: dict, meta: dict) -> dict:
    cands = [tuple(float(x) if j not in (2, 5) else int(x) for j, x in enumerate(r)) for r in A['cands']]
    cd = dict(meta['cd'], cands=cands, reason=A['reason'], label_map=A['label_map'])
    di = dict(meta['det_info'])
    D = dict(cd=cd, feats=A['feats'], det_info=di, code=A['code'], fidx=A['fidx'], cfg=meta['cfg'], ptype=meta['ptype'], guard=meta.get('guard'), arms={}, cinfo=None, allowed=None)
    if 'cls_all' in A:
        D['cinfo'] = dict(meta['cinfo'], cls_all=A['cls_all']); D['allowed'] = A['allowed']; D['det_info']['feature_allowed'] = A['allowed']
    for i, arm in enumerate(meta['arms']):
        tr = {k: A[f'a{i}_tr_{k}'] for k in ('matched', 'ambiguous', 'duplicate', 'best_idx', 'point_idx')}
        info = dict(meta[f'a{i}_info'])
        if f'a{i}_amb' in A:
            info['ambiguous_mask'] = A[f'a{i}_amb']
        D['arms'][arm] = (A[f'a{i}_pts'], info, tr)
    return D


_LR = ('features', 'line_id', 'status', 'chain', 'segment_id', 'n_candidates', 'guard_reason', 'points', 'accepted', 'segment_class')


def _lines_pack(rec: dict) -> tuple[dict, dict]:
    A = {k: np.asarray(rec[k]) for k in _LR if k in rec}
    meta = dict(detect_info={k: v for k, v in (rec.get('detect_info') or {}).items() if not isinstance(v, np.ndarray)}, counts=K.clean(rec.get('counts')), checks=K.clean(rec.get('checks')),
                threshold_px=K.clean(rec.get('threshold_px')), decision=rec.get('decision'), image_fallback=bool(rec.get('image_fallback', False)),
                segment_table=K.clean({str(k): v for k, v in (rec.get('segment_table') or {}).items()}), window=[rec.get('window_z0_mm'), rec.get('window_z1_mm')])
    return A, meta


def _lines_unpack(A: dict, meta: dict) -> dict:
    r = {k: A[k] for k in A}; r.update(detect_info=meta['detect_info'], counts=meta['counts'], checks=meta['checks'], threshold_px=meta['threshold_px'], decision=meta['decision'],
                                      image_fallback=meta['image_fallback'], segment_table={int(k): v for k, v in (meta['segment_table'] or {}).items()})
    return r


def decode_one(img, inp: dict, row: dict, method: str, params: dict, F_cache: dict) -> tuple[dict, dict]:
    if method == 'dot':
        k = 1 if row['pattern'] == 'plain' else 2; radii = [float(x) for x in str(row['sizes_mm']).split(';')]
        D = CD.dot_decode(img, inp['binary'], k, radii, params['rule']); return _dots_pack(D)
    if method == 'p4':
        dec = CD.plain_line_decode(img, inp); return _lines_pack(dec['aux']['r'])
    if 'F' not in F_cache:
        F_cache['F'] = CD.line_front(img, inp)
    rec = CD.line_rule(F_cache['F'], params['rule'], params.get('kappa')); return _lines_pack(rec)


def unpack(family: str, A: dict, meta: dict) -> dict:
    return _dots_unpack(A, meta) if family == 'dots' else _lines_unpack(A, meta)
