# Evaluate coded point and line outputs against the surface and identity references.

from __future__ import annotations

import numpy as np

import c2_common as K
import c2_decode as CD
import c2_identity as ID


def ctx_of(row: dict, inp: dict) -> dict:
    if row['family'] == 'dots':
        r = max(float(x) for x in str(row['sizes_mm']).split(';'))
        return dict(binary=inp['binary'], centres=inp['centres'], pattern=inp['pattern'], peak=inp['peak'], size_mm=r, meta_full=None, row=dict(radius_mm=r))
    return dict(binary=inp['binary'], centres=None, pattern=inp['pattern'], peak=inp['peak'], size_mm=None, meta_full=inp['meta_full'], row=dict(radius_mm=None))


def eval_dots(img, ref_img, inp: dict, row: dict, D: dict, ctx: dict) -> dict:

    import rdy_platform as P
    import s3c_dev as DV
    import sn_lib as SL
    mesh = P.mesh(row['surface']); sid = K.RK.SURF_MAP[row['surface']]
    exp = SL.expected_dots(mesh, np.asarray(inp['centres'], np.float64)); sm = ID.dot_source_map(D['cd'], ref_img, exp['uv'])
    cr = ID.dot_class_readout(D, sm, inp['classes']) if D.get('cinfo') is not None else {}
    ev = {}
    for arm in D['arms']:
        m, arr = DV.evaluate(sid, 'point', CD.dot_dec(D, arm), ctx, mesh); pi = ID.dot_point_identity(sm, D, arm, inp['binary'], inp['centres']); ev[arm] = (m, arr, pi)
    out = {}; ref_arm = 'ignore' if 'ignore' in ev else None
    for arm, (m, arr, pi) in ev.items():
        acc = np.asarray(arr['accepted'], bool); c = ID.dot_counts(pi['ident'], acc, pi['dot']); nv = dict(new_n=0, new_correct=0, new_wrong=0, new_unconf=0)
        if ref_arm and arm != ref_arm:
            fa = {int(f): s for f, s, a in zip(pi['feat'], pi['ident'], acc) if a and f >= 0}; rp = ev[ref_arm]
            fr = {int(f) for f, a in zip(rp[2]['feat'], np.asarray(rp[1]['accepted'], bool)) if a and f >= 0}; new = {f: s for f, s in fa.items() if f not in fr}
            nv = dict(new_n=len(new), new_correct=sum(s == 'correct' for s in new.values()), new_wrong=sum(s == 'wrong' for s in new.values()), new_unconf=sum(s in ID.UNCONF for s in new.values()))
        ref_acc = int(np.asarray(ev[ref_arm][1]['accepted'], bool).sum()) if ref_arm else int(acc.sum())
        st = {('status' if g == K.GATE else f'status_{g}'): ID.r4_status(c['src_acc'], c['src_correct'], c['src_wrong'], c['src_unconf'], nv['new_correct'], nv['new_wrong'], nv['new_unconf'], ref_acc, g)
              for g in (K.GATE,) + K.GATE_SENS}
        rowm = dict(**_m(m), **c, **nv, **ID.error_bounds(c['src_correct'], c['src_wrong'], c['src_unconf']), **st,
                    status_bridge=ID.r4_status(int(m['n_accepted']), int(m['id_correct']), int(m['id_wrong']), int(m['id_unconf']), 0, 0, 0, int(m['n_accepted']), K.GATE),
                    n_candidates=len(D['cd']['cands']), n_floor=D['cd'].get('n_floor'), n_bg_candidates=int((sm['cand_src'] == 0).sum()), guard_readable=(D['cinfo'] or {}).get('readable'),
                    guard_failed=';'.join((D.get('guard') or {}).get('failed') or []), guard_dprime=(D['cinfo'] or {}).get('dprime'), guard_span=(D['cinfo'] or {}).get('span'), **cr)
        arrays = dict(pts=np.asarray(arr['pts'], np.float32), acc=acc, valid=np.asarray(arr['valid'], bool), zr=np.asarray(arr['zr'], np.float32),
                      ident=np.array([str(x)[:2] for x in pi['ident']]), dot=pi['dot'].astype(np.int32), feat=pi['feat'].astype(np.int32),
                      bridge=np.stack([arr['point_correct'], arr['point_wrong'], arr['point_unconf']]).astype(bool), cand_src=sm['cand_src'], cand_dot=sm['cand_dot'].astype(np.int32))
        if arm == list(ev)[0]:
            arrays['cand_area'] = np.asarray([c_[2] for c_ in D['cd']['cands']], np.int32)
            if D.get('cinfo') is not None:
                arrays['cls_all'] = np.asarray(D['cinfo']['cls_all'], np.int8)
        out[arm] = (rowm, arrays)
    return out


def eval_lines(row: dict, inp: dict, recs: dict, ctx: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    import rdy_platform as P
    import s3c_dev as DV
    mesh = P.mesh(row['surface']); sid = K.RK.SURF_MAP[row['surface']]; ev = {}
    for arm, rec in recs.items():
        m, arr = DV.evaluate(sid, 'line', CD.line_dec(rec), ctx, mesh); ev[arm] = (m, arr, rec)
    first = next(iter(ev.values())); kO = np.asarray(first[1].get('truth_k', np.zeros(0)), np.int64); conf = np.asarray(first[1].get('confirmable', np.zeros(0)), bool)
    acc_i = np.asarray(ev['ignore'][1]['accepted'], bool) if 'ignore' in ev else None; out = {}
    for arm, (m, arr, rec) in ev.items():
        acc = np.asarray(arr['accepted'], bool); cor, wr, un = (np.asarray(arr[x], bool) for x in ('point_correct', 'point_wrong', 'point_unconf'))
        nv = dict(new_n=0, new_correct=0, new_wrong=0, new_unconf=0)
        if acc_i is not None and arm != 'ignore':
            new = acc & ~acc_i; nv = dict(new_n=int(new.sum()), new_correct=int((new & cor).sum()), new_wrong=int((new & wr).sum()), new_unconf=int((new & un).sum()))
        ref = int(acc_i.sum()) if acc_i is not None else int(acc.sum()); a, c_, w_, u_ = int(acc.sum()), int((acc & cor).sum()), int((acc & wr).sum()), int((acc & un).sum())
        st = {('status' if g == K.GATE else f'status_{g}'): ID.r4_status(a, c_, w_, u_, nv['new_correct'], nv['new_wrong'], nv['new_unconf'], ref, g) for g in (K.GATE,) + K.GATE_SENS}
        lay = ID.line_layers(rec, kO, conf, {int(s): v for s, v in (rec.get('segment_table') or {}).items()}, rec.get('threshold_px')) if row['pattern'] != 'plain' else {}
        rowm = dict(**_m(m), **nv, **ID.error_bounds(c_, w_, u_), **st, threshold_px=rec.get('threshold_px'), decision=rec.get('decision'), image_fallback=bool(rec.get('image_fallback', False)), **lay)
        arrays = dict(pts=np.asarray(arr['pts'], np.float32), acc=acc, valid=np.asarray(arr['valid'], bool), zr=np.asarray(arr['zr'], np.float32), lid=np.asarray(rec['line_id'], np.int32),
                      bridge=np.stack([cor, wr, un]).astype(bool))
        out[arm] = (rowm, arrays)
    return out, kO, conf


def _m(m: dict) -> dict:
    keep = ('state', 'n_accepted', 'n_valid', 'n_selected', 'n_detected', 'id_correct', 'id_wrong', 'id_unconf', 'glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_recovery',
            'loc_recovery', 'img_recovery', 'glob_pt_rmse', 'glob_pt_p95', 'glob_pt_mae', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'glob_gross_rate', 'n_confirmable_features')
    return {('b_' + k if k.startswith('id_') else k): m.get(k) for k in keep}
