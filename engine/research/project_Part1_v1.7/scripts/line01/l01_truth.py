# Generate ground-truth line identities and evaluate reconstructed line samples.

from __future__ import annotations

import math

import numpy as np

import l01_common as C
from m4_lines import UNKNOWN_ID


def _family(meta_olines: dict) -> dict:
    o = meta_olines
    return dict(n=np.asarray(o['normal_uv'], np.float64), t=np.asarray(o['direction_uv'], np.float64), c=np.asarray(o['centre_uv'], np.float64),
                s=float(o['spacing_px']), w=float(o['width_px']), phase=float(o['phase_frac']), ids=[int(k) for k in o['ids']], segments=o['segments'])


def inner_margin_px(meta_olines: dict, psf_px: float = C.PSF_PX, border_px: int = 3) -> float:

    return float(border_px + 0.5 * float(meta_olines['width_px']) + 2.0 * psf_px + 1.0)


def line_reference(mesh: dict, meta_olines: dict, step_px: float = 1.0, border_px: int = 3, pattern: np.ndarray | None = None, inner_px: float | None = None) -> dict:

    fam = _family(meta_olines)
    g = C.geom()
    inner_px = inner_margin_px(meta_olines) if inner_px is None else float(inner_px)
    centrelines = None
    if pattern is not None:
        from m4_lines import raster_centrelines, line_family
        centrelines = raster_centrelines(pattern, line_family(meta_olines))
    rk, rv, ru, rxyz, rt, rw, rin = [], [], [], [], [], [], []
    g_xyz = []
    per_line = {}
    for k in fam['ids']:
        seg = fam['segments'][str(k)]
        a = np.asarray(seg['start_uv'], np.float64); b = np.asarray(seg['end_uv'], np.float64)
        L = float(seg['length_px'])
        if L <= 0:
            continue
        if centrelines is not None:
            cl = centrelines[int(k)]
            if len(cl['rows']) < 3:
                continue
            P = np.column_stack([cl['u'], cl['rows'].astype(np.float64)])
        else:
            lam = np.arange(0.0, L + 1e-9, step_px)
            P = a + lam[:, None] * (b - a) / L
        hit, pts = C.cast(P, g['proj'], mesh)
        uv, inside = C.project(pts, g['cam'])
        ok = hit & inside
        rec = dict(n_samples=int(len(P)), n_hit=int(hit.sum()), n_in_image=int(ok.sum()), n_rows=0)
        if ok.sum() >= 3:
            uvh, xyzh = uv[ok], pts[ok]
            g_xyz.append(xyzh)
            order = np.argsort(uvh[:, 1]); uvh, xyzh = uvh[order], xyzh[order]
            if np.all(np.diff(uvh[:, 1]) > 0) and (uvh[-1, 1] - uvh[0, 1]) >= 2.0:
                rows = np.arange(math.ceil(uvh[0, 1]), math.floor(uvh[-1, 1]) + 1)
                u = np.interp(rows, uvh[:, 1], uvh[:, 0])
                xyz = np.column_stack([np.interp(rows, uvh[:, 1], xyzh[:, j]) for j in range(3)])
                d = np.gradient(uvh, axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                t = np.column_stack([np.interp(rows, uvh[:, 1], d[:, j]) for j in range(2)]); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
                w_arc = 1.0 / np.maximum(np.abs(t[:, 1]), 1e-6)
                inner = (u >= inner_px) & (u <= C.IMG - 1 - inner_px) & (rows >= border_px) & (rows <= C.IMG - 1 - border_px)
                rk.append(np.full(len(rows), k)); rv.append(rows); ru.append(u); rxyz.append(xyz); rt.append(t); rw.append(w_arc); rin.append(inner)
                rec['n_rows'] = int(len(rows))
        per_line[str(k)] = rec
    cat = lambda L, shape=None: (np.concatenate(L) if L else np.zeros((0,) + (shape or ())))
    rows = dict(k=cat(rk).astype(np.int64), v=cat(rv).astype(np.int64), u=cat(ru), xyz=cat(rxyz, (3,)), t=cat(rt, (2,)), w_arc=cat(rw), inner=cat(rin).astype(bool))
    return dict(rows=rows, g_xyz=(np.concatenate(g_xyz) if g_xyz else np.zeros((0, 3))), per_line=per_line, step_px=float(step_px), n_lines=int(len(fam['ids'])),
                centre_source=('raster' if centrelines is not None else 'analytic'), inner_margin_px=inner_px)


def truth_identity(mesh: dict, feats: np.ndarray, meta_olines: dict, tol_px: float = 1.0) -> dict:

    fam = _family(meta_olines)
    g = C.geom()
    feats = np.asarray(feats, np.float64).reshape(-1, 2)
    N = len(feats)
    out = dict(k=np.full(N, UNKNOWN_ID, np.int64), confirmable=np.zeros(N, bool), dist_px=np.full(N, np.nan), hit=np.zeros(N, bool), proj_uv=np.full((N, 2), np.nan), xyz=np.full((N, 3), np.nan), in_proj=np.zeros(N, bool),
               n_confirmable=0, n_unconfirmable=0)
    if N == 0:
        return out
    hit, pts = C.cast(feats, g['cam'], mesh)
    uv, _ = C.project(pts, g['proj'])
    inside = (uv[:, 0] >= -0.5) & (uv[:, 0] <= C.IMG - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= C.IMG - 0.5)
    out['hit'] = hit; out['xyz'][hit] = pts[hit]; out['proj_uv'][hit] = uv[hit]; out['in_proj'] = hit & inside
    q = (uv - fam['c']) @ fam['n']
    r = q / fam['s'] - fam['phase']
    k = np.round(r)
    dist = np.abs(r - k) * fam['s']
    conf = out['in_proj'] & (dist <= fam['w'] / 2.0 + tol_px) & np.isin(k.astype(np.int64), fam['ids'])
    out['k'][conf] = k[conf].astype(np.int64); out['confirmable'] = conf; out['dist_px'][hit] = dist[hit]
    out['n_confirmable'] = int(conf.sum()); out['n_unconfirmable'] = int(N - conf.sum())
    return out


# Count recovered line samples and summarize their source-identity outcomes.
def line_metrics(ref: dict, feats: np.ndarray, ids: np.ndarray, accepted: np.ndarray, chain: np.ndarray | None = None, truth_ids: np.ndarray | None = None,
                 gate_px: float = 3.0, tol_px: float = 1.0) -> dict:

    R = ref['rows']
    feats = np.asarray(feats, np.float64).reshape(-1, 2); ids = np.asarray(ids, np.int64); acc = np.asarray(accepted, bool)
    n_ref = len(R['v'])
    out = dict(n_ref_rows=int(n_ref), n_ref_inner=int(R['inner'].sum()), arc_total=float(R['w_arc'].sum()), n_features=int(len(feats)), n_accepted=int(acc.sum()))
    if n_ref == 0:
        return out

    by_row: dict = {}
    for i in np.flatnonzero(acc):
        by_row.setdefault(int(round(feats[i, 1])), []).append(i)
    recovered = np.zeros(n_ref, bool); ident = np.zeros(n_ref, bool); nerr = np.full(n_ref, np.nan)
    matched_feature = np.zeros(len(feats), bool)
    wrong_id = np.zeros(len(feats), bool)
    for j in range(n_ref):
        cand = by_row.get(int(R['v'][j]), [])
        if not cand:
            continue
        du = np.array([feats[i, 0] - R['u'][j] for i in cand])
        near = np.abs(du) <= gate_px
        if not near.any():
            continue
        recovered[j] = True
        ci = np.array(cand)[near]; dun = du[near]
        matched_feature[ci] = True
        same = ids[ci] == R['k'][j]
        if same.any():
            ident[j] = True
            nerr[j] = np.min(np.abs(dun[same])) * abs(R['t'][j, 1])
        wrong_id[ci[~same]] = True
    w = R['w_arc']; inner = R['inner']
    def frac(mask, sel=None):
        sel = np.ones(n_ref, bool) if sel is None else sel
        return float((w[sel] * mask[sel]).sum() / w[sel].sum()) if w[sel].sum() > 0 else None
    ok1 = ident & (nerr <= tol_px)
    e = nerr[ident]
    out.update(recovered_frac=frac(recovered), identity_correct_frac=frac(ident), ok_1px_frac=frac(ok1),
               recovered_frac_inner=frac(recovered, inner), identity_correct_frac_inner=frac(ident, inner), ok_1px_frac_inner=frac(ok1, inner),
               normal_err_rms_px=float(np.sqrt(np.mean(e ** 2))) if len(e) else None, normal_err_p95_px=float(np.percentile(e, 95)) if len(e) else None,
               normal_err_median_px=float(np.median(e)) if len(e) else None, n_rows_recovered=int(recovered.sum()), n_rows_identity=int(ident.sum()),
               n_spurious=int((acc & ~matched_feature).sum()), spurious_frac=(float((acc & ~matched_feature).sum() / acc.sum()) if acc.sum() else None),
               n_wrong_id=int(wrong_id.sum()), wrong_id_frac=(float(wrong_id.sum() / matched_feature.sum()) if matched_feature.sum() else None))

    ks = np.unique(R['k'])
    per_line_rec = np.array([recovered[R['k'] == k].mean() for k in ks])
    out.update(n_lines_ref=int(len(ks)), n_lines_recovered_half=int((per_line_rec >= 0.5).sum()), n_lines_missing=int((per_line_rec == 0).sum()))
    if chain is not None:
        chain = np.asarray(chain)
        segs_per_line = []
        for k in ks:
            sel = acc & (ids == k)
            segs_per_line.append(int(len(np.unique(chain[sel]))) if sel.any() else 0)
        out.update(n_segments_accepted=int(len(np.unique(chain[acc]))) if acc.any() else 0, segments_per_line_mean=float(np.mean(segs_per_line)) if segs_per_line else None,
                   segments_per_line_max=int(max(segs_per_line)) if segs_per_line else 0, n_breaks=int(sum(max(s - 1, 0) for s in segs_per_line)))
        if truth_ids is not None:
            truth_ids = np.asarray(truth_ids); mis = 0
            for cid in np.unique(chain[chain >= 0]):
                tk = truth_ids[(chain == cid) & (truth_ids != UNKNOWN_ID)]
                if len(np.unique(tk)) >= 2:
                    mis += 1
            out['n_misconnected_chains'] = int(mis)
    return out
