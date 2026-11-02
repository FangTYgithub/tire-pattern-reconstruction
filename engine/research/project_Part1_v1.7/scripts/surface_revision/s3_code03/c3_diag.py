# Separate detection, class-reading and correspondence losses in coded patterns.

from __future__ import annotations

import numpy as np

import c2_decode as CD
import c2_identity as ID
import c3_eval as EV

SRC_SINGLE = 1


def _feature_source(D: dict, sm: dict) -> tuple[np.ndarray, np.ndarray]:

    n = len(D['feats']); fidx = np.asarray(D['fidx']); cls = np.full(n, -9, np.int64); dot = np.full(n, -1, np.int64)
    for c, f in enumerate(fidx):
        if f >= 0:
            cls[int(f)] = int(sm['cand_src'][c]); dot[int(f)] = int(sm['cand_dot'][c])
    return cls, dot


def source_map(D: dict, ref_img, inp: dict, mesh) -> dict:
    import sn_lib as SL
    exp = SL.expected_dots(mesh, np.asarray(inp['centres'], np.float64))
    return ID.dot_source_map(D['cd'], ref_img, exp['uv'])


def dot_src_arms(D: dict, sm: dict, inp: dict, ctx: dict) -> tuple[dict, dict]:

    import ep_detect as ED
    import m4_reconstruction as M4
    import r5_detect as RD
    feats = np.asarray(D['feats'], np.float32); n = len(feats); binpat = np.asarray(inp['binary'], np.float32)
    fcls, fdot = _feature_source(D, sm); conf = (fcls == SRC_SINGLE) & (fdot >= 0)
    proj_pts, _ = M4.pattern_features(binpat, 'dots'); proj_dot, _ = RD.projector_dot_index(binpat, np.asarray(ctx['centres'], np.float64))
    dot2p = {int(d): p for p, d in enumerate(proj_dot)}
    out, book = {}, dict(n_features=n, n_src_confirmable=int(conf.sum()), n_background=int((fcls == 0).sum()), n_partial=int((fcls == 2).sum()),
                         n_multi=int((fcls == 3).sum()), n_ref_unlabelled=int((fcls == 4).sum()), n_no_mapping=int((fcls == -9).sum()))
    if D['ptype'] == 'coded_dots':
        allowed_use = np.asarray(D['allowed'], bool); allowed = allowed_use.copy(); cls = np.asarray(inp['classes'], np.int64)
        idx = np.flatnonzero(conf); allowed[idx] = False; allowed[idx, cls[fdot[idx]]] = True
        changed = (allowed != allowed_use).any(axis=1)
        out['TC_src'] = ED.reconstruct_from(feats, D['det_info'], binpat, D['cfg'], 'coded_dots', allowed)
        book.update(tc_replaced=int(conf.sum()), tc_changed=int(changed.sum()), tc_unknown_kept=int((~conf).sum()), tc_excluded=0,
                    tc_allowed_equal_use=bool(not changed.any()))
    mapped = conf & np.array([int(d) in dot2p for d in fdot], bool)
    ds, cnt = np.unique(fdot[mapped], return_counts=True); dup = set(int(d) for d, c in zip(ds, cnt) if c > 1)
    dup_f = mapped & np.isin(fdot, list(dup)) if dup else np.zeros(n, bool); sel = mapped & ~dup_f
    tr = dict(matched=sel.copy(), ambiguous=np.zeros(n, bool), duplicate=dup_f.copy(), best_idx=np.full(n, -1, np.int64), point_idx=np.full(n, -1, np.int64))
    pts = np.zeros((0, 3), np.float32)
    if sel.any():
        p = np.array([dot2p[int(fdot[i])] for i in np.flatnonzero(sel)]); tr['best_idx'][sel] = p; tr['point_idx'][sel] = np.arange(int(sel.sum()))
        pts = M4.triangulate(feats[sel], np.asarray(proj_pts, np.float32)[p], D['cfg']).astype(np.float32)
    out['TI_src'] = (pts, dict(n_detected=n, n_matched=int(sel.sum()), ambiguous_mask=np.zeros(int(sel.sum()), bool)), tr)
    book.update(ti_assigned=int(sel.sum()), ti_unmapped_confirmable=int((conf & ~mapped).sum()), ti_duplicate_dots=len(dup), ti_duplicate_features_excluded=int(dup_f.sum()),
                ti_excluded=int(n - sel.sum()))
    return out, book


def _acc_feats(D: dict, arm: str, acc) -> set:
    pidx = np.asarray(D['arms'][arm][2]['point_idx']); feat = np.full(int(np.asarray(acc).size), -1, np.int64); has = pidx >= 0
    feat[pidx[has]] = np.flatnonzero(has)
    return {int(f) for f, a in zip(feat, np.asarray(acc, bool)) if a and f >= 0}


def eval_src(row: dict, inp: dict, D: dict, sm: dict, ctx: dict, mesh) -> dict:

    import s3c_dev as DV
    import c2_common as K2
    arms, book = dot_src_arms(D, sm, inp, ctx); sid = K2.RK.SURF_MAP[row['surface']]
    meth = 'use' if 'use' in D['arms'] else 'single'
    m_meth, a_meth = DV.evaluate(sid, 'point', CD.dot_dec(D, meth), ctx, mesh); acc_meth = _acc_feats(D, meth, a_meth['accepted'])
    out = {}
    for arm, val in arms.items():
        Dt = dict(D, arms={arm: val}); m, arr = DV.evaluate(sid, 'point', CD.dot_dec(Dt, arm), ctx, mesh)
        acc = np.asarray(arr['accepted'], bool); pi = ID.dot_point_identity(sm, Dt, arm, inp['binary'], inp['centres']); c = ID.dot_counts(pi['ident'], acc, pi['dot'])
        af = _acc_feats(Dt, arm, acc)
        same_as_method = None
        if arm == 'TC_src' and book.get('tc_allowed_equal_use'):
            p0, _, t0 = D['arms']['use']; p1, _, t1 = val
            same_as_method = bool(np.array_equal(np.asarray(p0, np.float64).reshape(-1, 3), np.asarray(p1, np.float64).reshape(-1, 3))
                                  and all(np.array_equal(np.asarray(t0[k]), np.asarray(t1[k])) for k in ('matched', 'ambiguous', 'duplicate', 'best_idx', 'point_idx')))
        rowm = dict(**EV._m(m), **c, **ID.error_bounds(c['src_correct'], c['src_wrong'], c['src_unconf']), method_arm=meth, method_Ctau=m_meth.get('glob_Ctau'),
                    method_accepted=len(acc_meth), acc_both=len(af & acc_meth), acc_diag_only=len(af - acc_meth), acc_method_only=len(acc_meth - af),
                    replay_equal_when_allowed_unchanged=same_as_method,
                    **{k: v for k, v in book.items() if (k.startswith('tc_') and arm == 'TC_src') or (k.startswith('ti_') and arm == 'TI_src') or not k.startswith(('tc_', 'ti_'))})
        arrays = dict(pts=np.asarray(arr['pts'], np.float32), acc=acc, ident=np.array([str(x)[:2] for x in pi['ident']]), feat=pi['feat'].astype(np.int32), zr=np.asarray(arr['zr'], np.float32))
        out[arm] = (rowm, arrays)
    return out


def bridge_counts(D: dict, inp: dict, ctx: dict, mesh) -> dict:

    import r5_detect as RD
    import s3c_dev as DV
    truth, conf = DV.dot_truth(D['feats'], ctx, mesh); n = len(D['feats'])
    proj_dot, _ = RD.projector_dot_index(np.asarray(inp['binary'], np.float32), np.asarray(ctx['centres'], np.float64)); dot2p = {int(d) for d in proj_dot}
    sel = np.array([bool(conf[i]) and int(truth[i]) in dot2p for i in range(n)], bool)
    return dict(n_features=n, geom_confirmable=int(conf.sum()), geom_unconfirmable=int((~conf).sum()),
                TC=dict(geom_unconfirmable_kept_all_classes=int((~conf).sum()), excluded=0, class_replaced=int(conf.sum())),
                TI=dict(assigned=int(sel.sum()), excluded=int(n - sel.sum()), geom_unconfirmable_excluded=int((~conf).sum()), confirmable_unmapped_excluded=int((conf & ~sel).sum())))


def bridge_dot_eval(row: dict, inp: dict, D: dict, ctx: dict, mesh) -> dict:

    import c2_common as K2
    import s3c_dev as DV
    sid = K2.RK.SURF_MAP[row['surface']]
    ta, conf = DV.dot_truth_arms(D, inp['binary'], ctx, mesh, inp['classes']); bc = bridge_counts(D, inp, ctx, mesh); out = {}
    for t in ta:
        Dt = dict(D, arms={t: ta[t]}); m, arr = DV.evaluate(sid, 'point', CD.dot_dec(Dt, t), ctx, mesh)
        out[t] = (dict(**EV._m(m), reference='bridge_geom', features_used=bc['n_features'], geom_confirmable=bc['geom_confirmable'], **bc[t]),
                  dict(acc=np.asarray(arr['accepted'], bool)))
    return out


def line_diag(row: dict, inp: dict, img, recs_ignore: dict, kO, conf, ctx: dict, F: dict | None = None) -> tuple[dict, bool]:

    import s3c_dev as DV
    rerun = F is None
    F = F if F is not None else CD.line_front(img, inp)
    ta = DV.lines_truth_arms(dict(front=F['front'], fam=F['fam'], cfg=F['cfg']), kO, conf)
    resd, _, _ = EV.eval_lines(row, inp, dict(ignore=recs_ignore, **ta), ctx)
    return {t: (dict(resd[t][0], features_used=int(len(F['front']['feat'])), confirmable_features=int(np.asarray(conf).sum())), resd[t][1]) for t in ('TC', 'TI')}, rerun
