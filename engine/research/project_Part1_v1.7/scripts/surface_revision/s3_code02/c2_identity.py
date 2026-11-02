# Classify reconstructed source identities and check their tolerances.

from __future__ import annotations

import numpy as np

UNCONF = ('background', 'undeterminable_partial', 'undeterminable_multi', 'undeterminable_unlabelled', 'none')


def dot_source_map(cd: dict, ref_img, exp_uv) -> dict:

    import sl_lib as L
    Sm, rinfo = L.reference_map(np.asarray(ref_img, np.uint8), np.asarray(exp_uv, float))
    cl = L.classify(cd['label_map'], cd['n_labels'], Sm)
    lab = np.array([c[5] for c in cd['cands']], np.int64)
    src = cl['cls'][lab].astype(np.int8) if len(lab) else np.zeros(0, np.int8)
    dom = cl['dot'][lab].astype(np.int64) if len(lab) else np.zeros(0, np.int64)
    return dict(cl=cl, cand_label=lab, cand_src=src, cand_dot=dom, ref=rinfo)


def dot_point_identity(sm: dict, D: dict, arm: str, binpat, centres) -> dict:

    import r5_detect as RD
    import sl_lib as L
    pts, info, tr = D['arms'][arm]
    proj_dot, _ = RD.projector_dot_index(np.asarray(binpat, np.float32), np.asarray(centres, np.float64))
    z = dict(pts=np.asarray(pts, np.float32).reshape(-1, 3), cand_label=sm['cand_label'], cand_fidx=np.asarray(D['fidx']), feat_matched=np.asarray(tr['matched'], bool),
             feat_best_idx=np.asarray(tr['best_idx']), feat_point_idx=np.asarray(tr['point_idx']), proj_dot=np.asarray(proj_dot))
    ident = np.asarray(L.output_identity(z, sm['cl']), dtype=object)

    n = len(z['pts']); dot = np.full(n, -1, np.int64); feat = np.full(n, -1, np.int64)
    pidx = z['feat_point_idx']; has = pidx >= 0; feat[pidx[has]] = np.flatnonzero(has)
    for p in np.flatnonzero(ident == 'correct'):
        f = feat[p]; dot[p] = int(proj_dot[int(z['feat_best_idx'][f])])
    return dict(ident=ident, dot=dot, feat=feat)


def accept_mask(D: dict, arm: str) -> np.ndarray:
    import rdy_decode as DC
    pts, info, tr = D['arms'][arm]
    return DC._accept(pts, tr)


def dot_counts(ident, acc, dots=None) -> dict:
    a = np.asarray(acc, bool); i = np.asarray(ident, dtype=object)
    out = dict(src_acc=int(a.sum()), src_correct=int((a & (i == 'correct')).sum()), src_wrong=int((a & (i == 'wrong')).sum()),
               src_unconf=int((a & np.isin(i, UNCONF)).sum()))
    for k in UNCONF[:-1]:
        out[f'src_{k}'] = int((a & (i == k)).sum())
    if dots is not None:
        out['src_recovered_dots'] = int(len(np.unique(np.asarray(dots)[a & (i == 'correct')])))
    return out


def dot_class_readout(D: dict, sm: dict, classes) -> dict:

    if D.get('cinfo') is None:
        return {}
    cls_all = np.asarray(D['cinfo']['cls_all'], np.int64); src, dot = sm['cand_src'], sm['cand_dot']; ok = (src == 1) & (dot >= 0)
    t = np.full(len(cls_all), -1); t[ok] = np.asarray(classes)[dot[ok]]
    readable = bool(D['cinfo'].get('readable')); fidx = np.asarray(D['fidx']); allowed = np.asarray(D['allowed'], bool)
    sel = ok & (fidx >= 0)
    pre_ok = int((ok & (cls_all == t)).sum()); pre_n = int(ok.sum())
    post_single = np.zeros(len(cls_all), bool); post_cls = np.full(len(cls_all), -1)
    if readable and sel.any():
        rows = allowed[fidx[sel]]; single = rows.sum(axis=1) == 1
        idx = np.flatnonzero(sel); post_single[idx[single]] = True; post_cls[idx[single]] = rows[single].argmax(axis=1)
    return dict(cls_true_known=pre_n, cls_pre_correct=pre_ok, cls_pre_acc=(pre_ok / pre_n if pre_n else None), cls_guard_readable=readable,
                cls_post_assigned=int(post_single.sum()), cls_post_correct=int((post_single & (post_cls == t)).sum()),
                cls_post_abstain=int((sel & ~post_single).sum()), cls_sel_known=int(sel.sum()),
                bg_candidates=int((src == 0).sum()), small_true=int((ok & (t == 0)).sum()), large_true=int((ok & (t == 1)).sum()))


def line_layers(rec: dict, kO, conf, seg_table: dict | None, thr) -> dict:

    import lc_pattern as LP
    seg = np.asarray(rec['segment_id']); sc = np.asarray(rec.get('segment_class', np.full(len(seg), -1))); tc = np.full(len(kO), -1); tc[conf] = LP.class_of_id(kO[conf])
    out = dict(L1_read=0, L1_n2w=0, L1_w2n=0, L1_no_thr=0, L2_readable=0, L3_adopted=0, L3_n2w=0, L3_w2n=0, seg_known=0, seg_mixed=0, seg_unknown=0)
    for s in np.unique(seg[seg >= 0]):
        m = seg == s; c = np.unique(tc[m & conf]); ids = np.unique(kO[m & conf])
        truth = int(c[0]) if (len(c) == 1 and len(ids) >= 1) else None
        if len(c) > 1:
            out['seg_mixed'] += 1
        elif len(c) == 0:
            out['seg_unknown'] += 1
        else:
            out['seg_known'] += 1
        t = (seg_table or {}).get(int(s)) or {}
        med = t.get('median_px')
        if med is not None:
            out['L1_read'] += 1
            if thr is None:
                out['L1_no_thr'] += 1
            elif truth is not None:
                rd = int(med > thr); out['L1_n2w'] += int(truth == 0 and rd == 1); out['L1_w2n'] += int(truth == 1 and rd == 0)
        out['L2_readable'] += int(bool(t.get('readable')))
        if (sc[m] >= 0).any():
            out['L3_adopted'] += 1; a = int(sc[m][sc[m] >= 0][0])
            if truth is not None:
                out['L3_n2w'] += int(truth == 0 and a == 1); out['L3_w2n'] += int(truth == 1 and a == 0)
    return out


def r4_status(n_acc, cor, wr, unc, new_cor, new_wr, new_unc, ref_acc, gate) -> str:

    def rate(w, c):
        return None if c == 0 else w / c
    if n_acc == 0:
        return 'no_output'
    if unc > 0 or new_unc > 0:
        return 'not_accepted_unconfirmable'
    r = rate(wr, cor + wr)
    if r is None or r > gate:
        return 'unsafe'
    if new_cor + new_wr == 0:
        return 'safe_NA_new' if ref_acc > 0 else 'no_output'
    rn = rate(new_wr, new_cor + new_wr)
    return 'unsafe' if (rn is None or rn > gate) else 'safe'


def error_bounds(cor, wr, unc) -> dict:

    den = cor + wr + unc
    return dict(err_lower=(wr / den if den else None), err_upper=((wr + unc) / den if den else None), err_known=(wr / (cor + wr) if cor + wr else None),
                unconf_share=(unc / den if den else None))
