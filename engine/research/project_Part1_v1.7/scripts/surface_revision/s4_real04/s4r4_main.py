# Compare sampling, detection and displacement mechanisms in layout results.

"""S4-REAL-04 entry: python s4r4_main.py check | a | b1 | b2 | b3 | c1 | c2 | figs | assemble | final

check     S4-REAL-02 anchors, S4-REAL-03 deliverables (reference hashes), S4-REAL-02 freeze chain; writes the registered configuration first
a         work A: anchor local interpolation link, pairing / phase counter-examples, source grading with unclipped coordinates (76 images),
          dot area-median statement (384 images), monotonicity wording, S4-REAL-03 traceability supplement
b1        L6.3 S0 candidate x status cross-tables (sigma 0, + sigma 20 / 40 noise check) and <= 6 geometric cases
b2        sections 1 / 2 / 5 inventory (stored fields + regenerated rasters) and <= 24 pre-registered traced inputs
b3        front-end / numbering versions per section with the existing bridge evidence
c1 / c2   s4r4_scan.py (generation scan + raster checks; hole / support-scale layers)
figs      s4r4_figs.py; assemble -> diagnostics.csv / geometry_scan.csv; final -> input / code / output hashes
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import s4r4_common as H

K = H.K
DIAG = H.OUT / '_parts'


def part(name: str, df: pd.DataFrame) -> None:
    H.save_csv(df, DIAG / f'{name}.csv')


def check() -> dict:
    import r2_main as RM
    H.run_put('config', H.CONFIG)
    a02 = {f: dict(expected=h, actual=K.sha_file(K.OUT / f)) for f, h in H.reference_R02.items()}
    a03 = {f: dict(expected=h, actual=K.sha_file(K.ROOT / f)) for f, h in H.reference_R03.items()}
    pf = RM.preflight()
    ok = all(v['expected'] == v['actual'] for v in a02.values()) and all(v['expected'] == v['actual'] for v in a03.values()) and pf['ok']
    out = dict(ok=bool(ok), s4_real02_anchors=a02, s4_real03_at_reference_review=a03, s4_real02_preflight=pf, t_local=K.now_local())
    H.run_put('check', out)
    if not ok:
        raise SystemExit('anchor check failed')
    return dict(ok=ok)


def local_link(eval_key: str, surface: str) -> dict:

    import rdy_platform as P
    from surface_eval_r3 import covering_radius_exact, dedup_xy, triangulate_xy
    z = K.load_npz(H.reg(K.EVL / f'{eval_key}.npz'))
    valid = z['valid'].astype(bool); pts = z['pts'].astype(np.float64)
    anom = valid & z['accepted'].astype(bool) & z['point_unconf'].astype(bool)
    Pd, _n = dedup_xy(pts[valid], None)
    ka = set(map(tuple, np.round(pts[anom, :2] / 1e-6).astype(np.int64)))
    flag = np.array([tuple(k) in ka for k in np.round(Pd[:, :2] / 1e-6).astype(np.int64)])
    tri, _r = triangulate_xy(Pd); dl = P.dom_local(surface)
    q = np.column_stack([dl['x'], dl['y']]); s = tri.find_simplex(q); ins = s >= 0
    has = np.zeros(len(q), bool); has[ins] = flag[tri.simplices[s[ins]]].any(axis=1)
    T = tri.transform[s[ins]]; rr = q[ins] - T[:, 2, :]; bc = np.einsum('nij,nj->ni', T[:, :2, :], rr); w = np.column_stack([bc, 1 - bc.sum(1)])
    zest = np.full(len(q), np.nan); zest[ins] = (w * Pd[tri.simplices[s[ins]], 2]).sum(1)
    e = zest - np.asarray(dl['z_ref'], np.float64); cov = np.isfinite(e)
    Rt = covering_radius_exact(Pd[:, :2], tri.simplices); Rn = np.full(len(q), np.inf); Rn[ins] = Rt[s[ins]]; sup = cov & (Rn <= K.D_SUPPORT)
    sq = e[cov] ** 2
    return dict(n_local_nodes=int(len(q)), n_interpolable=int(cov.sum()), n_supported=int(sup.sum()), n_link_nodes=int(has.sum()), link_nodes_supported=int((has & sup).sum()),
                share_sq_err_link=float(sq[has[cov]].sum() / sq.sum()) if sq.sum() else None, surf_rmse_interpolable=float(np.sqrt(sq.mean())) if cov.any() else None,
                surf_rmse_interpolable_nonlink=float(np.sqrt(np.mean(e[cov & ~has] ** 2))) if (cov & ~has).any() else None,
                surf_rmse_supported=float(np.sqrt(np.mean(e[sup] ** 2))) if sup.any() else None, n_anomalous_points=int(anom.sum()),
                anomalous_in_local=int((anom & z['in_local'].astype(bool)).sum()))


def source_grading(eval_keys) -> pd.DataFrame:
    import r2_run as R
    import r3_diag as R3D
    M = R.manifest(); rows = []
    for ek in eval_keys:
        row = M[M.eval_key == ek].iloc[0].to_dict(); inp = R.inputs_of(row); z = K.load_npz(H.reg(K.EVL / f'{ek}.npz'))
        sel = z['accepted'].astype(bool) & z['point_unconf'].astype(bool)
        c = R3D.classify_line_features(R3D._mesh(row['surface']), z['feats'][sel], inp['table']['spec'], inp['binary'])
        cl = np.isin(c['cls'], ['clipped_line', 'clipped_line_end'])
        uv = c['proj_uv'][cl]; inf = np.isfinite(uv).all(1) & (uv[:, 0] >= -0.5) & (uv[:, 0] < 511.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < 511.5)
        B = np.asarray(inp['binary']) > 0; uu = np.rint(np.nan_to_num(uv[:, 0], nan=-9)).astype(int); vv = np.rint(np.nan_to_num(uv[:, 1], nan=-9)).astype(int)
        ok = inf & (uu >= 0) & (uu <= 511) & (vv >= 0) & (vv <= 511); lit = np.zeros(len(uv), bool); lit[ok] = B[vv[ok], uu[ok]]
        rows.append(dict(record_type='A2_source_grading', eval_key=ek, surface=row['surface'], sigma=float(row['sigma']), phase=row['phase'], layout=row['layout'],
                         n_accepted_unconf=int(sel.sum()), n_clipped_source=int(cl.sum()), direct_lit_in_frame=int((ok & lit).sum()), out_of_frame=int((~inf).sum()),
                         in_frame_unlit=int((inf & ~lit).sum()), other_classes=int((~cl).sum())))
    return pd.DataFrame(rows)


def dot_area_statement() -> dict:
    import r2_run as R
    M = R.manifest(); U = M[(M.family == 'point') & (M.sigma >= 80)].drop_duplicates('eval_key'); s0 = M[M.sigma == 0].drop_duplicates('image_key').set_index('image_key')
    rows = []; a0c = {}
    for r in U.itertuples():
        z = K.load_npz(H.reg(K.EVL / f'{r.eval_key}.npz')); a = z['cand_area'].astype(np.float64)
        ek0 = s0.loc[r.sigma0_ref_key, 'eval_key']
        if ek0 not in a0c:
            a0c[ek0] = float(K.load_json(H.reg(K.EVL / f'{ek0}.json'))['info']['det_info']['blob_area_median_px'])
        A0 = a0c[ek0]; tot = a.sum()
        rows.append(dict(eval_key=r.eval_key, small_share=float(a[a < 0.5 * A0].sum() / tot) if tot else None))
    D = pd.DataFrame(rows); ex = D[D.eval_key == 'R2E-5c3d44dc96f4b23c6cc5494d'].iloc[0]
    rest = D[D.eval_key != 'R2E-5c3d44dc96f4b23c6cc5494d']
    return dict(n=len(D), exception_small_share=float(ex.small_share), rest_min_small_share=float(rest.small_share.min()), n_small_share_below_half=int((D.small_share < 0.5).sum()),
                note='area-weighted median falls above 0.5 x sigma-0 dot area iff the small-area share is below 50 %; "dot-sized" is an area proxy, not a confirmed source')


def a3_trace() -> dict:

    r3 = json.loads(H.reg(H.R3 / 'run.json').read_text(encoding='utf-8'))
    registered = set(r3['lines']['inputs_read']) | set(r3['dots']['inputs_read'])
    d3 = pd.read_csv(H.reg(H.R3 / 'diagnostics.csv'), low_memory=False)
    actual = set()
    dots = d3[d3.family == 'point']
    actual |= {K.rel(K.EVL / f'{k}.json') for k in dots.eval_key}
    comp = r3['dots']['comparison']
    actual |= {K.rel(K.EVL / f"{c['eval_key']}.json") for c in comp} | {K.rel(K.EVL / f"{c['eval_key']}.npz") for c in comp} | {K.rel(K.IMG / f"{c['image_key']}.npz") for c in comp}
    an = r3['lines']['anchor']; actual |= {K.rel(K.IMG / f"{an['image_key']}.npz")}
    for x in r3['lines']['legacy']:
        p = K.ROOT / x['legacy_eval']; actual |= {K.rel(p), K.rel(p.with_suffix('.npz'))}
    actual |= {K.rel(K.RUN), K.rel(K.OUT / 'paired_summary.csv'), K.rel(K.PIX / 'domains' / 'omega1.npz'), K.rel(K.PIX / 'domains' / 'omega1.json')}
    supp = sorted(actual - registered); rows = []
    for p in supp:
        q = K.ROOT / p; h = K.sha_file(H.reg(q)) if q.exists() else 'MISSING'; created = None
        if p.endswith('.npz'):
            js = q.with_suffix('.json')
            if js.exists():
                created = K.load_json(js).get('npz_sha256')
        rows.append(dict(path=p, sha256_now=h, npz_sha256_in_record=created, consistent_with_record=(None if created is None else created == h)))
    code = {p: K.sha_file(K.ROOT / p) for p in ('scripts/surface_revision/s4_real03/r3_diag.py', 'scripts/surface_revision/s4_real03/r3_figs.py')}
    return dict(nature=f'supplementary registration {K.now_local()[:10]} by S4-REAL-04: hashes taken now, not a pre-run snapshot; S4-REAL-03 run.json left unchanged',
                r3_diag_initial=r3['check']['s4_real03_code'], r3_code_now=code,
                r3_diag_change='after the 22:50 check: window_explanation added, legacy NaN handling, closure section; figure patches in r3_figs (never registered)',
                n_registered=len(registered), n_supplementary=len(rows), supplementary=rows,
                all_npz_consistent_with_records=bool(all(r['consistent_with_record'] in (True, None) for r in rows)))


def work_a() -> dict:
    t0 = time.time()
    C = pd.read_csv(H.reg(K.OUT / 'case_metrics.csv'), low_memory=False); C['block'] = C['block'].fillna('')
    P = pd.read_csv(H.reg(K.OUT / 'paired_summary.csv'), low_memory=False)
    an = H.CONFIG['A']['anchor_eval']; ll = local_link(an, 'M10_curv'); st = C[C.eval_key == an].iloc[0]
    jx = K.load_json(H.reg(K.EVL / f'{an}.json')); nodes = jx['extras']['nodes_omega1']['loc']
    anchor = dict(ll, stored_loc_pt_rmse=float(st.loc_pt_rmse), stored_loc_surf_rmse=float(st.loc_surf_rmse), stored_loc_Ctau=float(st.loc_Ctau), stored_loc_Cvalid=float(st.loc_Cvalid),
                  stored_loc_sup_surf_rmse=nodes.get('sup_surf_rmse'), stored_loc_n_supported=nodes.get('n_supported'),
                  surf_rmse_reproduced=bool(abs(ll['surf_rmse_interpolable'] - float(st.loc_surf_rmse)) < 1e-8))
    X = C[(C.family == 'line') & (C.surface == 'M10_curv') & (C.phase == 'H0') & (C.sigma == 0.0)]
    s0 = X[X.strategy == 'S0'].iloc[0]
    pair = dict(S0_accepted=int(s0.n_accepted), S0_glob_pt_rmse=float(s0.glob_pt_rmse),
                **{f'{k}_accepted_range': [int(X[X.strategy == k].n_accepted.min()), int(X[X.strategy == k].n_accepted.max())] for k in ('S1', 'S3')},
                **{f'{k}_glob_pt_rmse_range': [float(X[X.strategy == k].glob_pt_rmse.min()), float(X[X.strategy == k].glob_pt_rmse.max())] for k in ('S1', 'S3')},
                S1_minus_S0_pt_rmse_range=[float((X[X.strategy == 'S1'].glob_pt_rmse - s0.glob_pt_rmse).min()), float((X[X.strategy == 'S1'].glob_pt_rmse - s0.glob_pt_rmse).max())],
                note='every layout carries the same clipped line (466 points at H0 sigma 0), yet accepted totals and point RMSE differ by layout -> no automatic cancellation')
    Y = C[(C.family == 'line') & (C.surface == 'M10_curv') & (C.sigma == 40.0)]
    s0p = Y[Y.strategy == 'S0'].set_index('phase').glob_Ctau; s3 = Y[Y.strategy == 'S3'].copy(); s3['d'] = s3.glob_Ctau - s3.phase.map(s0p)
    phase = dict(all_phases_pp=float(100 * s3.groupby('block').d.median().median()), H1_H3_pp=float(100 * s3[s3.phase != 'H0'].groupby('block').d.median().median()),
                 note='post-hoc stability check; dropping H0 also changes the phase sample, it is not an estimate of the clipped-line effect and never replaces the main result')
    d3 = pd.read_csv(H.reg(H.R3 / 'diagnostics.csv'), low_memory=False)
    aff = d3[(d3.family == 'line') & ((d3.acc_clipped_line.fillna(0) + d3.acc_clipped_line_end.fillna(0)) > 0)]
    G = source_grading(list(aff.eval_key)); part('A2_source_grading', G)
    grad = dict(n_images=len(G), n_points=int(G.n_clipped_source.sum()), direct_lit_in_frame=int(G.direct_lit_in_frame.sum()), out_of_frame=int(G.out_of_frame.sum()),
                in_frame_unlit=int(G.in_frame_unlit.sum()), anchor=G[G.eval_key == an].iloc[0].to_dict(),
                matches_reference=bool((int(G.direct_lit_in_frame.sum()), int(G.out_of_frame.sum()), int(G.in_frame_unlit.sum())) == (22327, 60, 55)))
    dot = dot_area_statement()
    mono = []
    Pp = P[(P.table_type == 'paired') & (P.contrast == 'S1-S0') & (P.metric == 'glob_Ctau')]
    for (f, s), g in Pp.groupby(['family', 'surface']):
        v = g.sort_values('sigma')['main'].abs().to_numpy(); inc = [int(i) for i in np.flatnonzero(np.diff(v) > 1e-12)]
        mono.append(dict(family=f, surface=s, abs_S1_S0_by_sigma_pp=[round(100 * x, 3) for x in v], increases_at_step=inc))
    tr = a3_trace()
    out = dict(anchor_local=anchor, pairing=pair, phase_sensitivity=phase, source_grading=grad, dot_area=dot, monotonicity_S1_S0=mono, traceability_s4_real03=tr,
               dot_exception_denominators=dict(selected_relative_area=110, geometric_confirmable_selected=108, accepted=96, source_correct=87, source_undeterminable=9, confirmed_wrong=0),
               seconds=round(time.time() - t0, 1), t_local=K.now_local())
    H.run_put('A', out); H.flush_reads('A')
    return dict(anchor={k: anchor[k] for k in ('n_link_nodes', 'share_sq_err_link', 'surf_rmse_interpolable', 'link_nodes_supported', 'surf_rmse_reproduced')}, grading=grad['matches_reference'],
                dot=dot, phase=phase, trace_n=tr['n_supplementary'], trace_ok=tr['all_npz_consistent_with_records'])


def _cast_domain(feats, surface):

    import l01_common as C0
    import r2_eval as E
    import rdy_platform as P
    g = C0.geom(); hit, X = C0.cast(np.asarray(feats, np.float64), g['cam'], P.mesh(surface)); X = np.asarray(X, np.float64)
    dom = E.domain('omega1'); ix = np.rint((np.nan_to_num(X[:, 0], nan=1e9) - dom['x0']) / dom['step']).astype(np.int64); iy = np.rint((np.nan_to_num(X[:, 1], nan=1e9) - dom['y0']) / dom['step']).astype(np.int64)
    ok = hit & (ix >= 0) & (ix < dom['nx']) & (iy >= 0) & (iy < dom['ny']); glob = np.zeros(len(X), bool); glob[ok] = dom['mask'][iy[ok], ix[ok]]
    loc = hit & (np.hypot(np.nan_to_num(X[:, 0], nan=1e9), np.nan_to_num(X[:, 1], nan=1e9)) <= 15.0)
    return hit, X, glob, loc


def b1() -> dict:
    import r2_run as R
    from m4_lines import STATUS
    t0 = time.time(); M = R.manifest()
    base = M[(M.family == 'line') & (M.layout == 'regular')].drop_duplicates('eval_key')
    rows = []; segs = []; per_img = []
    for r in base[base.sigma.isin([0.0, 20.0, 40.0])].sort_values(['surface', 'sigma', 'phase']).itertuples():
        z = K.load_npz(H.reg(K.EVL / f'{r.eval_key}.npz')); st = z['status'].astype(int); nc = z['n_candidates'].astype(int); seg = z['segment_id'].astype(int); gr = z['guard_reason'].astype(int)
        ncc = np.where(nc >= 2, '>=2', nc.astype(str))
        hit, X, glob, loc = _cast_domain(z['feats'], r.surface)
        for dname, msk in (('all', np.ones(len(st), bool)), ('global_true_xy', glob), ('local_true_xy', loc)):
            for c in ('0', '1', '>=2'):
                for k in np.unique(st):
                    n = int(((ncc == c) & (st == k) & msk).sum())
                    if n:
                        rows.append(dict(record_type='B1_crosstab', eval_key=r.eval_key, surface=r.surface, sigma=r.sigma, phase=r.phase, domain=dname, n_candidates=c, status=STATUS[int(k)], n_features=n))
        unk = st == 2
        per_img.append(dict(record_type='B1_image', eval_key=r.eval_key, surface=r.surface, sigma=r.sigma, phase=r.phase, n_features=int(len(st)), n_accepted=int((st == 0).sum()),
                            n_segment_unknown=int(unk.sum()), n_unknown_multi=int((unk & (nc >= 2)).sum()), n_unknown_le1=int((unk & (nc <= 1)).sum()), n_no_candidate=int((st == 1).sum()),
                            n_guard=int((st >= 8).sum()), unknown_share=float(unk.mean()), n_features_global_true=int(glob.sum()), n_features_local_true=int(loc.sum())))
        for sid in np.unique(seg[seg >= 0]):
            s = seg == sid; dec = 'numbered' if np.isin(st[s], [0, 3]).any() else ('unknown' if (st[s] == 2).any() else ('no_candidate' if (st[s] == 1).any() else 'other'))
            segs.append(dict(record_type='B1_segment', eval_key=r.eval_key, surface=r.surface, sigma=r.sigma, phase=r.phase, segment=int(sid), n=int(s.sum()), decision=dec,
                             n_multi=int((nc[s] >= 2).sum()), n_single=int((nc[s] == 1).sum()), n_zero=int((nc[s] == 0).sum())))
    CT = pd.DataFrame(rows); IM = pd.DataFrame(per_img); SG_ = pd.DataFrame(segs)
    s0 = IM[IM.sigma == 0.0]
    m10 = s0[s0.surface == 'M10_curv']
    verify = dict(M10_sigma0_features=int(m10.n_features.sum()), M10_unknown=int(m10.n_segment_unknown.sum()), M10_unknown_multi=int(m10.n_unknown_multi.sum()),
                  M10_unknown_share=float(m10.n_segment_unknown.sum() / m10.n_features.sum()), M10_multi_share_of_unknown=float(m10.n_unknown_multi.sum() / m10.n_segment_unknown.sum()),
                  M10_not_accepted_share=float(1 - m10.n_accepted.sum() / m10.n_features.sum()))
    segsum = SG_.groupby(['surface', 'sigma', 'decision']).agg(n_segments=('segment', 'size'), features=('n', 'sum'), median_support=('n', 'median'),
                                                                 multi=('n_multi', 'sum'), single=('n_single', 'sum'), zero=('n_zero', 'sum')).reset_index()

    cases = []
    for s, g in s0.groupby('surface'):
        g = g.sort_values(['unknown_share', 'phase'], ascending=[False, True]); cases.append(g.iloc[0].eval_key)
        g2 = g.sort_values(['unknown_share', 'phase'], ascending=[True, True]); cases.append(g2.iloc[0].eval_key if g2.iloc[0].eval_key != g.iloc[0].eval_key else g2.iloc[1].eval_key)
    geo = [b1_geometry(ek) for ek in cases]
    GR = pd.concat([x['rows'] for x in geo], ignore_index=True)
    part('B1', pd.concat([CT, IM, segsum.assign(record_type='B1_segment_summary'), GR], ignore_index=True, sort=False))
    out = dict(verify_M10=verify, image_table=IM.to_dict('records'), segment_summary=segsum.to_dict('records'), geometric_cases=[{k: v for k, v in x.items() if k != 'rows'} for x in geo],
               missing_fields=['segment vote tables (raw meta segment_table is empty): votes reconstructed only for the 6 geometric cases'],
               seconds=round(time.time() - t0, 1), t_local=K.now_local())
    H.run_put('B1', out); H.flush_reads('B1')
    return dict(verify=verify, cases=[(x['surface'], x['phase'], x['vote_agreement']) for x in geo])


# Inspect line candidates and segment outcomes against the explicit projector geometry.
def b1_geometry(ek: str) -> dict:

    import r2_run as R
    import rdy_platform as P
    import sp2_lines_table as LT2
    import sp_lines_table as LT
    import sp_regression as R1
    import m4_lines as ML
    import r3_diag as R3D
    M = R.manifest(); row = M[M.eval_key == ek].iloc[0].to_dict(); inp = R.inputs_of(row); spec = inp['table']['spec']
    z = K.load_npz(H.reg(K.EVL / f'{ek}.npz')); feats = z['feats'].astype(np.float64); st = z['status'].astype(int); seg = z['segment_id'].astype(int); gr = z['guard_reason'].astype(int)
    fam = LT2.line_family_table(spec); cfg0 = P.s2().line_cfg(float(inp['peak'])); cfg = LT2._resolve_sensor_cfg(cfg0) if hasattr(LT2, '_resolve_sensor_cfg') else cfg0
    cl = LT2.raster_centrelines_table(inp['pattern'], fam)
    lct = getattr(LT2, 'line_candidates_table', LT.line_candidates_table); cand = lct(feats, fam, cfg, centrelines=cl)
    p = {**ML.default_params(fam, cfg), **{k: v for k, v in R1.p4_params().items() if v is not None}}; vf = float(p['vote_frac'])
    adm = np.asarray(cand['admissible'], bool); ids = np.asarray(cand['ids'], np.int64); zc = np.asarray(cand['z'], np.float64)
    nc_check = bool(np.array_equal(adm.sum(1), z['n_candidates'].astype(int)))
    c = R3D.classify_line_features(P.mesh(row['surface']), feats, spec, inp['binary'])
    import sp2_lines_table as _L
    uv = c['proj_uv']; fin = np.isfinite(uv).all(1); q = np.where(fin, (np.nan_to_num(uv) - fam['c']) @ fam['n'], np.nan)
    k1, d1, k2, d2 = _L.nearest_table_line(fam, q); single = c['cls'] == 'complete_line_in'
    conf = np.asarray(z['confirmable'], bool); true_k = np.where(conf, z['truth_k'].astype(np.int64), np.where(single, k1, -10 ** 6))
    col = {int(k): j for j, k in enumerate(ids)}; jt = np.array([col.get(int(k), -1) for k in true_k])
    has_t = jt >= 0; t_adm = np.zeros(len(feats), bool); t_adm[has_t] = adm[np.flatnonzero(has_t), jt[has_t]]
    n_other = adm.sum(1) - t_adm.astype(int)
    cat = np.where(~has_t, 'source_not_complete_line', np.where(t_adm & (n_other == 0), 'unique_true', np.where(t_adm & (n_other > 0), 'true_plus_alias', np.where(n_other > 0, 'alias_only', 'none'))))
    hit, X, glob, loc = _cast_domain(feats, row['surface']); zt = X[:, 2]
    from m4_lines import STATUS
    rows = []
    zb = np.floor(np.nan_to_num(zt, nan=-99) / H.CONFIG['B1']['height_bins_mm']) * H.CONFIG['B1']['height_bins_mm']
    for (b, ct, s_), n in pd.Series(1, index=pd.MultiIndex.from_arrays([zb, cat, st])).groupby(level=[0, 1, 2]).sum().items():
        rows.append(dict(record_type='B1_height_bin', eval_key=ek, surface=row['surface'], phase=row['phase'], z_bin_mm=float(b), category=ct, status=STATUS[int(s_)], n_features=int(n)))

    steps = []
    for dk in (-1, 1):
        jn = np.array([col.get(int(k) + dk, -1) if hk else -1 for k, hk in zip(true_k, has_t)]); ok = has_t & (jn >= 0)
        dz = zc[np.flatnonzero(ok), jn[ok]] - zc[np.flatnonzero(ok), jt[ok]]; dz = dz[np.isfinite(dz)]
        steps.append(dict(neighbour=dk, n=int(len(dz)), median=float(np.median(dz)) if len(dz) else None, p05=float(np.percentile(dz, 5)) if len(dz) else None, p95=float(np.percentile(dz, 95)) if len(dz) else None))

    agree = 0; tot = 0; unk_single = []; unk_all = []
    for sid in np.unique(seg[seg >= 0]):
        sel = np.flatnonzero((seg == sid) & (gr == 0))
        if not len(sel):
            continue
        votes = adm[sel].sum(0); n = len(sel); o = np.argsort(-votes); top = int(votes[o[0]]) if len(votes) else 0; sec = int(votes[o[1]]) if len(votes) > 1 else 0
        dec = 'no_candidate' if top == 0 else ('numbered' if (top >= vf * n and sec <= (1 - vf) * n) else 'unknown')
        stored = 'numbered' if np.isin(st[sel], [0, 3]).any() else ('unknown' if (st[sel] == 2).any() else ('no_candidate' if (st[sel] == 1).all() else 'other'))
        tot += 1; agree += int(dec == stored)
        if dec == 'unknown':
            comp = {k: int(v) for k, v in zip(*np.unique(cat[sel], return_counts=True))}
            unk_all.append(dict(segment=int(sid), n=n, top_id=int(ids[o[0]]), top_share=round(top / n, 4), second_id=int(ids[o[1]]) if len(o) > 1 else None, second_share=round(sec / n, 4),
                                zero_candidate_share=round(float((adm[sel].sum(1) == 0).mean()), 4), multi_share=round(float((adm[sel].sum(1) >= 2).mean()), 4), truth_categories=comp))
        if dec == 'unknown' and (adm[sel].sum(1) <= 1).all():
            ids_single = sorted({int(ids[np.flatnonzero(adm[i])[0]]) for i in sel if adm[i].any()})
            unk_single.append(dict(segment=int(sid), n=n, top_share=top / n, second_share=sec / n, zero_candidate_share=float((adm[sel].sum(1) == 0).mean()), single_ids=ids_single[:6]))
    summary = dict(eval_key=ek, surface=row['surface'], phase=row['phase'], n_features=int(len(feats)), n_candidates_reproduced=nc_check, vote_frac=vf, vote_agreement=f'{agree}/{tot}',
                   categories={k: int(v) for k, v in zip(*np.unique(cat, return_counts=True))}, neighbour_depth_steps_mm=steps,
                   unknown_segments_all_le1_candidate=len(unk_single), unknown_le1_examples=unk_single[:5], unknown_segments=unk_all,
                   true_z_range_mm=[float(np.nanmin(zt)), float(np.nanmax(zt))], window_mm=list(K.RK.WINDOW_MM))
    df = pd.DataFrame(rows)
    df = pd.concat([df, pd.DataFrame([dict(record_type='B1_geo_case', eval_key=ek, surface=row['surface'], phase=row['phase'], category=k, n_features=v) for k, v in summary['categories'].items()])], ignore_index=True)
    return dict(summary, rows=df)


def _s12_sections(s: str) -> list:
    return [x for x in str(s).split(';') if x]


# Join sampling-reference results with observed outcomes and select diagnostic cases.
def b2() -> dict:
    import r2_run as R
    sys.path.insert(0, str(K.ROOT / 'scripts' / 'surface_revision' / 's12_resource01'))
    import s12_case as SC
    import rdy_inputs as RI
    t0 = time.time()
    I = pd.read_csv(H.reg(H.S12 / 'inputs.csv'), low_memory=False); X = pd.read_csv(H.reg(H.S12 / 'results' / 'inputs_metrics.csv'), low_memory=False)
    D = I.merge(X, on='input_key', how='left')

    rmeta = {}
    for rk, g in D.groupby('raster_key'):
        u = g.iloc[0].to_dict(); R_ = SC.raster_of(u); rmeta[('S12', rk)] = _raster_meta(R_, u['family'])
    P_ = pd.read_csv(H.reg(H.S345 / 'plan_s345.csv'), low_memory=False); S5p = P_[P_.set_formal == 'S5-main']
    S5c = pd.read_csv(H.reg(H.S345 / 'results' / 'S5-main_cases.csv'), low_memory=False)
    for (lid, ph), g in S5p.groupby(['layout_id', 'phase_label']):
        row = g.iloc[0].to_dict(); inp = RI.build(row); rmeta[('S5', f'{lid}|{ph}')] = _raster_meta(dict(binary=inp['binary'], centres=inp.get('centres'), meta_full=inp.get('meta_full')), row['family'])

    D['clipped_ids'] = [rmeta[('S12', rk)]['clipped'] for rk in D.raster_key]; D['n_clipped_lit'] = [rmeta[('S12', rk)]['n_clipped'] for rk in D.raster_key]
    D['n_border_partial_dots'] = [rmeta[('S12', rk)]['n_border_partial'] for rk in D.raster_key]
    inv = []
    for sec in ('S1', 'S2'):
        d = D[D.sections.apply(lambda s: sec in _s12_sections(s))]
        for (fam, lay, sc), g in d.groupby(['family', 'layout', 'surface_class']):
            inv.append(_inv_row(sec, fam, lay, sc, g, 'id_unconf', 'id_wrong', 'n_accepted', 'glob_n_gross', 'unconf_share', 'glob_pt_p95'))
    S5c['surface_class'] = 'main'
    S5c['clipped_ids'] = [rmeta.get(('S5', f'{a}|{b}'), {}).get('clipped') for a, b in zip(S5c.layout_id, S5c.phase_label)]
    S5c['n_clipped_lit'] = [rmeta.get(('S5', f'{a}|{b}'), {}).get('n_clipped') for a, b in zip(S5c.layout_id, S5c.phase_label)]
    S5c['n_border_partial_dots'] = [rmeta.get(('S5', f'{a}|{b}'), {}).get('n_border_partial') for a, b in zip(S5c.layout_id, S5c.phase_label)]
    S5c['unconf_share'] = S5c.id_unconf / S5c.id_denominator.where(S5c.id_denominator > 0)
    S5c['glob_n_gross'] = np.nan
    for (fam, lay), g in S5c.groupby(['family', 'layout_id']):
        inv.append(_inv_row('S5', fam, lay, 'main', g, 'id_unconf', 'id_wrong', 'n_accepted', 'glob_n_gross', 'unconf_share', 'glob_pt_p95'))
    INV = pd.DataFrame(inv)
    split = []
    for sec, d in (('S1', D[D.sections.apply(lambda s: 'S1' in _s12_sections(s))]), ('S2', D[D.sections.apply(lambda s: 'S2' in _s12_sections(s))]), ('S5', S5c)):
        d = d[(d.family == 'line') & (d.sigma <= 40)]
        if 'surface_class' in d:
            d = d[d.surface_class == 'main']
        for has, g in d.groupby(d.n_clipped_lit.fillna(0) > 0):
            split.append(dict(record_type='B2_split', section=sec, raster_has_lit_clipped_line=bool(has), n_inputs=int(len(g)), unconf_sigma_le40=float(g.id_unconf.sum()),
                              inputs_with_unconf=int((g.id_unconf > 0).sum()), wrong_sigma_le40=float(g.id_wrong.sum())))
    INV = pd.concat([INV, pd.DataFrame(split)], ignore_index=True)

    RM_ = pd.DataFrame([dict(record_type='B2_raster', source=a, raster=b, **{k: v for k, v in m.items() if k != 'clipped_detail'}, clipped_detail=json.dumps(m['clipped_detail'])) for (a, b), m in rmeta.items()])

    sel = select_traces(D, S5c)
    T = [trace_case(x) for x in sel]
    TR = pd.DataFrame(T)
    part('B2', pd.concat([INV, RM_, TR], ignore_index=True, sort=False))
    out = dict(n_inputs_S12=int(len(D)), n_S12_main=int((D.surface_class == 'main').sum()) if 'surface_class' in D else None, n_S5_cases=int(len(S5c)),
               n_rasters=len(rmeta), rasters_with_lit_clipped_line=int(sum(m['n_clipped'] > 0 for m in rmeta.values())),
               rasters_with_border_partial_dots=int(sum((m['n_border_partial'] or 0) > 0 for m in rmeta.values())),
               inventory=INV[INV.record_type == 'B2_inventory'].to_dict('records'), split_sigma_le40=split, selection=[{k: v for k, v in x.items() if k not in ('row',)} for x in sel], traces=T,
               missing_fields=['S5 case table: no gross-error count (glob_n_gross) -> NaN', 'segment vote tables not stored anywhere'],
               execution_note='first run missed the registered 45 deg wrong-identity pick (column lookup error, inputs.csv has no orientation column); fixed via the layout name and rerun before any interpretation; the cap of 8 then drops the S2 dot_unconf_top pick',
               seconds=round(time.time() - t0, 1), t_local=K.now_local())
    H.run_put('B2', out); H.flush_reads('B2')
    return dict(n_rasters=len(rmeta), clipped=out['rasters_with_lit_clipped_line'], border_dots=out['rasters_with_border_partial_dots'], n_traces=len(T))


def _raster_meta(R_, family) -> dict:
    from scipy import ndimage
    B = np.asarray(R_['binary']) > 0
    if family == 'line':
        ol = (R_.get('meta_full') or {}).get('olines') or {}
        cl = ol.get('clipped_lines') or {}
        det = {str(k): dict(q_px=v.get('q_px'), lit_px=v.get('lit_px')) for k, v in cl.items()}
        return dict(family='line', n_ids=len(ol.get('ids') or []), angle_deg=ol.get('angle_deg'), spacing_px=ol.get('spacing_px'), width_px=ol.get('width_px'), phase_frac=ol.get('phase_frac'),
                    n_clipped=int(sum(1 for v in cl.values() if (v.get('lit_px') or 0) > 0)), clipped=';'.join(sorted(det)), clipped_detail=det, n_border_partial=None)
    c = np.asarray(R_.get('centres'), np.float64).reshape(-1, 2); lab, n = ndimage.label(B)
    com = np.asarray(ndimage.center_of_mass(B, lab, range(1, n + 1)), np.float64)[:, ::-1] if n else np.zeros((0, 2))
    touch = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    out_c = (c[:, 0] < -0.5) | (c[:, 0] > 511.5) | (c[:, 1] < -0.5) | (c[:, 1] > 511.5)
    from scipy.spatial import cKDTree
    unmatched = int((cKDTree(c).query(com)[0] > 1.0).sum()) if (n and len(c)) else n
    return dict(family='point', n_centres=int(len(c)), n_components=int(n), n_border_touching=int(len(touch)), n_centres_outside=int(out_c.sum()), n_components_off_centre=unmatched,
                n_border_partial=int(len(touch)), n_clipped=0, clipped='', clipped_detail={})


def _inv_row(sec, fam, lay, sc, g, cu, cw, ca, cg, cs, cp) -> dict:
    def s(c):
        return float(g[c].sum()) if c in g and g[c].notna().any() else None
    lo = g[g.sigma <= 40] if 'sigma' in g else g
    return dict(record_type='B2_inventory', section=sec, family=fam, layout=lay, surface_class=sc, n_inputs=int(len(g)), sigmas=';'.join(f'{x:g}' for x in sorted(g.sigma.unique())),
                n_phases=int(g.phase_label.nunique()) if 'phase_label' in g else None, accepted=s(ca), confirmed_wrong=s(cw), unconfirmable=s(cu), gross=s(cg),
                inputs_with_unconf=int((g[cu] > 0).sum()), inputs_with_wrong=int((g[cw] > 0).sum()), unconf_sigma_le40=float(lo[cu].sum()), wrong_sigma_le40=float(lo[cw].sum()),
                max_unconf_share=float(g[cs].max()) if g[cs].notna().any() else None, max_glob_pt_p95=float(g[cp].max()) if g[cp].notna().any() else None,
                rasters_with_lit_clipped=int(g.loc[g.n_clipped_lit.fillna(0) > 0, 'raster_key' if 'raster_key' in g else 'layout_id'].nunique()) if 'n_clipped_lit' in g else None,
                inputs_on_rasters_with_clipped=int((g.n_clipped_lit.fillna(0) > 0).sum()) if 'n_clipped_lit' in g else None)


# Select representative identity failures and reference cases from the result tables.
def select_traces(D: pd.DataFrame, S5c: pd.DataFrame) -> list:
    sel = []
    main = D[D.surface_class == 'main'] if 'surface_class' in D else D
    for sec in ('S1', 'S2'):
        d = main[main.sections.apply(lambda s: sec in _s12_sections(s))]; picks = []
        L_ = d[d.family == 'line']
        cl = L_[L_.n_clipped_lit > 0]
        for lay, g in cl[cl.id_unconf > 0].groupby('layout'):
            g = g.sort_values(['id_unconf', 'input_key'], ascending=[False, True]); picks.append(('line_unconf_top', g.iloc[0]))
        picks = sorted(picks, key=lambda t: -t[1].id_unconf)[:4]
        w45 = L_[L_.layout.astype(str).str.endswith('phi45') & (L_.id_wrong > 0)]
        if len(w45):
            picks.append(('line_45_wrong', w45.sort_values(['id_wrong', 'input_key'], ascending=[False, True]).iloc[0]))
        nc = cl[(cl.id_unconf == 0) & (cl.sigma == 0)].sort_values('input_key')
        if len(nc):
            picks.append(('negative_clipped_zero_unconf', nc.iloc[0]))
        nn = L_[(L_.n_clipped_lit == 0) & (L_.sigma == 0)].sort_values('input_key')
        if len(nn):
            picks.append(('negative_no_clipped', nn.iloc[0]))
        P_ = d[d.family == 'point']
        w = P_[P_.id_wrong > 0].sort_values(['id_wrong', 'input_key'], ascending=[False, True])
        if len(w):
            picks.append(('dot_wrong_top', w.iloc[0]))
        u = P_[(P_.sigma <= 40) & (P_.id_unconf > 0)].sort_values(['id_unconf', 'input_key'], ascending=[False, True])
        if not len(u):
            u = P_[(P_.id_unconf > 0)].sort_values(['id_unconf', 'input_key'], ascending=[False, True])
        if len(u):
            picks.append(('dot_unconf_top', u.iloc[0]))
        for role, r in picks[:8]:
            sel.append(dict(section=sec, role=role, key=r.input_key, family=r.family, layout=r.layout, surface=r.surface, sigma=float(r.sigma), phase=r.phase_label,
                            eval_record=r.eval_record, raster_key=r.raster_key, row=r.to_dict(), source='S12'))
    s5 = S5c[S5c.surface.isin(['B_curv', 'M10_curv', 'M24_curv'])]; picks = []
    L_ = s5[s5.family == 'line']; cl = L_[L_.n_clipped_lit.fillna(0) > 0]
    for lay, g in cl[cl.id_unconf > 0].groupby('layout_id'):
        g = g.sort_values(['id_unconf', 'decode_key'], ascending=[False, True]); picks.append(('line_unconf_top', g.iloc[0]))
    picks = sorted(picks, key=lambda t: -t[1].id_unconf)[:4]
    nc = cl[(cl.id_unconf == 0) & (cl.sigma == 0)].sort_values('decode_key')
    if len(nc):
        picks.append(('negative_clipped_zero_unconf', nc.iloc[0]))
    nn = L_[(L_.n_clipped_lit.fillna(0) == 0) & (L_.sigma == 0)].sort_values('decode_key')
    if len(nn):
        picks.append(('negative_no_clipped', nn.iloc[0]))
    P_ = s5[s5.family == 'point']; w = P_[P_.id_wrong > 0].sort_values(['id_wrong', 'decode_key'], ascending=[False, True])
    if len(w):
        picks.append(('dot_wrong_top', w.iloc[0]))
    u = P_[P_.id_unconf > 0].sort_values(['id_unconf', 'decode_key'], ascending=[False, True])
    if len(u):
        picks.append(('dot_unconf_top', u.iloc[0]))
    for role, r in picks[:8]:
        sel.append(dict(section='S5', role=role, key=r.decode_key, family=r.family, layout=r.layout_id, surface=r.surface, sigma=float(r.sigma), phase=r.phase_label,
                        eval_record=K.rel(H.S345 / 'run' / 'eval' / f'{r.decode_key}.json'), raster_key=None, row=r.to_dict(), source='S5'))
    return sel


def trace_case(x: dict) -> dict:

    import rdy_platform as P
    import r3_diag as R3D
    import rdy_inputs as RI
    sys.path.insert(0, str(K.ROOT / 'scripts' / 'surface_revision' / 's12_resource01'))
    import s12_case as SC
    rec = str(x['eval_record']); jp = K.ROOT / rec if not rec.startswith(('C:', '/')) else Path(rec)
    jp = Path(jp); zp = jp.with_suffix('.npz'); z = K.load_npz(H.reg(zp)); out = dict(record_type='B2_trace', **{k: v for k, v in x.items() if k not in ('row', 'eval_record')}, eval_record=K.rel(jp))
    acc = z['accepted'].astype(bool) if 'accepted' in z else None
    if x['family'] == 'line':
        if x['source'] == 'S12':
            R_ = SC.raster_of(x['row']); spec = R_['meta_full']['olines']; binary = R_['binary']
        else:
            import pandas as _pd
            P_ = _pd.read_csv(H.S345 / 'plan_s345.csv', low_memory=False); row = P_[P_.decode_key == x['key']].iloc[0].to_dict(); inp = RI.build(row); spec = inp['meta_full']['olines']; binary = inp['binary']
        unc = acc & z['point_unconf'].astype(bool); wr = acc & z['point_wrong'].astype(bool)
        mesh = P.mesh(x['surface'])
        for nm, m in (('unconf', unc), ('wrong', wr)):
            if m.any():
                c = R3D.classify_line_features(mesh, z['feats'][m], spec, binary)
                uv = c['proj_uv']; inf = np.isfinite(uv).all(1) & (uv[:, 0] >= -0.5) & (uv[:, 0] < 511.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < 511.5)
                for k, v in zip(*np.unique(c['cls'], return_counts=True)):
                    out[f'{nm}_{k}'] = int(v)
                out[f'{nm}_out_of_frame'] = int((~inf).sum())
                e = (z['pts'][m, 2].astype(float) - z['zr'][m].astype(float)); out[f'{nm}_abs_err_median_mm'] = float(np.nanmedian(np.abs(e))); out[f'{nm}_abs_err_max_mm'] = float(np.nanmax(np.abs(e)))
                out[f'{nm}_in_local'] = int((m & z['in_local'].astype(bool)).sum()); out[f'{nm}_in_global'] = int((m & z['in_global'].astype(bool)).sum())
        out['n_accepted'] = int(acc.sum()); out['n_unconf'] = int(unc.sum()); out['n_wrong'] = int(wr.sum()); out['clipped_ids'] = ';'.join(sorted((spec.get('clipped_lines') or {}).keys()))
        if unc.any():
            ll = _local_link_arrays(z, x['surface'], unc); out.update({f'link_{k}': v for k, v in ll.items()})
        verdict = ('confirmed_clipped_line' if out.get('unconf_clipped_line', 0) + out.get('unconf_clipped_line_end', 0) > 0 else
                   ('not_seen' if not unc.any() else 'other_source'))
        out['verdict'] = verdict
    else:
        unc = acc & z['point_unconf'].astype(bool) if 'point_unconf' in z else np.zeros(0, bool); wr = acc & z['point_wrong'].astype(bool) if 'point_wrong' in z else np.zeros(0, bool)
        pf = z['point_feature'].astype(int) if 'point_feature' in z else None
        dist = np.asarray(z['feat_truth_dist_px'], np.float64) if 'feat_truth_dist_px' in z else None
        out.update(n_accepted=int(acc.sum()), n_unconf=int(unc.sum()), n_wrong=int(wr.sum()))
        if pf is not None and dist is not None:
            for nm, m in (('unconf', unc), ('wrong', wr)):
                f = pf[m]; f = f[f >= 0]
                if len(f):
                    out[f'{nm}_truth_dist_px_median'] = float(np.nanmedian(dist[f])); out[f'{nm}_truth_dist_px_max'] = float(np.nanmax(dist[f]))
                    e = (z['pts'][m, 2].astype(float) - z['zr'][m].astype(float)); out[f'{nm}_abs_err_median_mm'] = float(np.nanmedian(np.abs(e)))
            if wr.any() and 'feat_matched_dot' in z and 'feat_truth_dot' in z:
                f = pf[wr]; f = f[f >= 0]; md = z['feat_matched_dot'][f].astype(int); td = z['feat_truth_dot'][f].astype(int)
                out['wrong_matched_minus_truth_dot'] = ';'.join(str(int(v)) for v in np.unique(md - td)[:8])
        out['verdict'] = 'dot_record_summary'
    return out


def _local_link_arrays(z, surface, anom_mask) -> dict:
    import rdy_platform as P
    from surface_eval_r3 import covering_radius_exact, dedup_xy, triangulate_xy
    valid = z['valid'].astype(bool); pts = z['pts'].astype(np.float64); anom = valid & anom_mask
    if valid.sum() < 3:
        return dict(n_link_nodes=0)
    Pd, _n = dedup_xy(pts[valid], None); ka = set(map(tuple, np.round(pts[anom, :2] / 1e-6).astype(np.int64)))
    flag = np.array([tuple(k) in ka for k in np.round(Pd[:, :2] / 1e-6).astype(np.int64)])
    tri, _r = triangulate_xy(Pd)
    if tri is None:
        return dict(n_link_nodes=0)
    dl = P.dom_local(surface); q = np.column_stack([dl['x'], dl['y']]); s = tri.find_simplex(q); ins = s >= 0
    has = np.zeros(len(q), bool); has[ins] = flag[tri.simplices[s[ins]]].any(axis=1)
    Rt = covering_radius_exact(Pd[:, :2], tri.simplices); Rn = np.full(len(q), np.inf); Rn[ins] = Rt[s[ins]]
    return dict(n_link_nodes=int(has.sum()), link_nodes_supported=int((has & (Rn <= K.D_SUPPORT)).sum()), n_local_nodes=int(len(q)))


def b3() -> dict:

    def find(d, key):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == key:
                    return v
                x = find(v, key)
                if x is not None:
                    return x
    rows = []
    import r2_run as R
    M = R.manifest(); r = M[(M.family == 'line') & (M.sigma == 0)].iloc[0]; m = K.load_json(H.reg(K.RAW / f'{r.decode_key}.json'))['meta']
    rows.append(dict(section='4 (S4-REAL-02)', family='line', method='P4TABLE (sp2_lines_table.run_p4_table)', guard_rule=find(m, 'guard_rule'), steger_rule=find(m, 'steger_threshold_rule'), numbering='table ids (complete lines only), vote_frac 0.9'))
    rows.append(dict(section='4 (S4-REAL-02)', family='point', method='D0 (binding, relative area window = area-weighted median)', guard_rule=None, steger_rule=None, numbering='epipolar tracks, design centres'))
    rows.append(dict(section='1 / 2 (S12)', family='line', method='P4_C3_cal (binding m4_lines.run_olines)', guard_rule='see S12 raw meta', steger_rule='see S12 raw meta', numbering='regular ids (complete lines only)'))
    rows.append(dict(section='1 / 2 (S12)', family='point', method='D0', guard_rule=None, steger_rule=None, numbering='epipolar'))
    rows.append(dict(section='3 (S3-CODE-07)', family='line', method='LC (R / ignore / use / A) and P4REF', guard_rule=None, steger_rule=None, numbering='class readout + numbering'))
    rows.append(dict(section='3 (S3-CODE-07)', family='point', method='ENC / DF (coded), B0 = D0 plain', guard_rule=None, steger_rule=None, numbering='class + epipolar'))
    rows.append(dict(section='5 (S5-main)', family='line', method='P4_C3_cal', guard_rule=None, steger_rule=None, numbering='regular ids'))
    rows.append(dict(section='5 (S5-main)', family='point', method='D0', guard_rule=None, steger_rule=None, numbering='epipolar'))
    I = pd.read_csv(H.S12 / 'inputs.csv', low_memory=False); X = pd.read_csv(H.S12 / 'results' / 'inputs_metrics.csv', low_memory=False)
    D = I.merge(X, on='input_key'); ln = D[(D.family == 'line') & (D.source == 'new')].iloc[0]
    s12 = {}
    raw = H.S12 / 'raw' / f"{ln.decode_key}.json"
    if raw.exists():
        mm = K.load_json(H.reg(raw)); s12 = dict(guard_rule=find(mm, 'guard_rule'), steger_rule=find(mm, 'steger_threshold_rule'), detector=find(mm, 'detector'))
        rows[2].update(guard_rule=s12['guard_rule'], steger_rule=s12['steger_rule'])
    E1 = pd.read_csv(H.reg(H.SR / 's4_align01' / 'diagnostics.csv'), low_memory=False); E1 = E1[E1.table_type == 'E1_frontend']
    bridge = dict(source='outputs/surface_revision/s4_align01/diagnostics.csv (E1_frontend, 54 rows) and report §7.1',
                  summary=['CODE-04 (S1/S2 stored images, 2 phases): DF ~ D0 in 12 of 14 conditions at sigma 0-40; DF gains at sigma >= 60/80 for small sizes; R8x2.5 sigma 40/60 DF losses',
                           'CODE-07 (section 3 plain R5x4 / R6x4 / R8x4): DF = B0 image by image at sigma 0/20/40; DF gains at sigma 80/100; negligible non-zero at sigma 60',
                           'section 4 (S4-REAL-02, 384 dot images at sigma >= 80 and all lower sigma): DF never run; D0 only',
                           'CODE-04 regression: B0 == stored D0 on 1143 inputs (candidates, selection, features, clouds)'],
                  n_E1_rows=int(len(E1)))
    out = dict(versions=rows, s12_line_raw_meta=s12, bridge=bridge, t_local=K.now_local())
    H.run_put('B3', out); H.flush_reads('B3')
    return dict(n=len(rows), s12=s12)


def assemble() -> dict:
    parts = {p.stem: pd.read_csv(p, low_memory=False) for p in sorted(DIAG.glob('*.csv'))}
    diag = [v for k, v in parts.items() if not k.startswith('C1')]
    D = pd.concat(diag, ignore_index=True, sort=False)
    lead = [c for c in ('record_type', 'eval_key', 'key', 'section', 'family', 'surface', 'sigma', 'phase', 'layout') if c in D.columns]
    D = D[lead + [c for c in D.columns if c not in lead]]
    H.save_csv(D, H.OUT / 'diagnostics.csv')
    G = pd.concat([v for k, v in parts.items() if k.startswith('C1')], ignore_index=True, sort=False) if any(k.startswith('C1') for k in parts) else pd.DataFrame()
    if len(G):
        H.save_csv(G, H.OUT / 'geometry_scan.csv')
    return dict(diagnostics_rows=len(D), geometry_rows=len(G), record_types=D.record_type.value_counts().to_dict())


def deletion_sensitivity() -> dict:

    import r2_eval as E
    import r2_run as R
    import rdy_platform as P
    from surface_eval_r3 import surface_eval_r3_dev
    t0 = time.time(); M = R.manifest(); C = pd.read_csv(H.reg(K.OUT / 'case_metrics.csv'), low_memory=False)
    G = pd.read_csv(H.reg(DIAG / 'A2_source_grading.csv'), low_memory=False)
    G = G[(G.surface == 'M10_curv') & (G.sigma == 0.0) & (G.phase == 'H0')]
    assert (G.other_classes == 0).all() and (G.n_accepted_unconf == G.n_clipped_source).all()
    dom = E.domain('omega1'); sid = K.RK.SURF_MAP['M10_curv']; dl = P.dom_local('M10_curv')
    Dg = dict(x=dom['x'], y=dom['y'], z_ref=np.asarray(dom[f'z_{sid}'], np.float64), area=np.full(len(dom['x']), dom['step'] ** 2), hash='omega1')
    Dl = dict(x=dl['x'], y=dl['y'], z_ref=np.asarray(dl['z_ref'], np.float64), area=np.asarray(dl['area'], np.float64), hash='local')
    rows = []
    for ek in G.eval_key:
        z = K.load_npz(H.reg(K.EVL / f'{ek}.npz')); valid = z['valid'].astype(bool); pts = z['pts'].astype(np.float64)
        anom = valid & z['accepted'].astype(bool) & z['point_unconf'].astype(bool); st = C[C.eval_key == ek].iloc[0]
        rec = dict(record_type='A1_deletion_sensitivity', eval_key=ek, surface='M10_curv', sigma=0.0, phase='H0', strategy=st.strategy, block=st.block if isinstance(st.block, str) else '',
                   layout=st.layout, n_valid=int(valid.sum()), n_removed=int(anom.sum()))
        for tag, Pv in (('official', pts[valid]), ('without_clipped', pts[valid & ~anom])):
            for dname, Dd in (('glob', Dg), ('loc', Dl)):
                ev = surface_eval_r3_dev(Pv, None, Dd, [K.D_SUPPORT], [K.TAU], support_rule='exact'); g = ev['subdomains']['global']; bd = g['by_d'][str(float(K.D_SUPPORT))]
                rec[f'{dname}_Ctau_{tag}'] = bd['C_tau'][str(float(K.TAU))]; rec[f'{dname}_Cvalid_{tag}'] = bd['C_valid']
        rec['glob_Ctau_stored'] = float(st.glob_Ctau); rec['loc_Ctau_stored'] = float(st.loc_Ctau)
        rows.append(rec)
    D = pd.DataFrame(rows)
    for d in ('glob', 'loc'):
        D[f'{d}_dCtau_pp'] = 100 * (D[f'{d}_Ctau_without_clipped'] - D[f'{d}_Ctau_official'])
    rep = float(max((D.glob_Ctau_official - D.glob_Ctau_stored).abs().max(), (D.loc_Ctau_official - D.loc_Ctau_stored).abs().max()))
    s0 = D[D.strategy == 'S0'].iloc[0]; pairs = []
    for s in ('S1', 'S2', 'S3'):
        for x in D[D.strategy == s].itertuples():
            for d in ('glob', 'loc'):
                a = getattr(x, f'{d}_Ctau_official') - s0[f'{d}_Ctau_official']; b = getattr(x, f'{d}_Ctau_without_clipped') - s0[f'{d}_Ctau_without_clipped']
                pairs.append(dict(contrast=f'{s}-S0', block=x.block, domain=d, official_pp=100 * a, without_pp=100 * b, change_pp=100 * (b - a)))
    Pp = pd.DataFrame(pairs); part('A1_deletion_sensitivity', D)
    out = dict(scope='L6.3, M10, H0, sigma 0; 19 distinct clouds (S0 1, S1 8, S2 2, S3 8); removed points = accepted unconfirmable = clipped-line source (A2)',
               n_clouds=len(D), removed_points_range=[int(D.n_removed.min()), int(D.n_removed.max())], official_reproduction_max_abs=rep,
               dCtau_pp_range={d: [float(D[f'{d}_dCtau_pp'].min()), float(D[f'{d}_dCtau_pp'].max())] for d in ('glob', 'loc')},
               paired_change_pp_range={d: [float(Pp[Pp.domain == d].change_pp.min()), float(Pp[Pp.domain == d].change_pp.max())] for d in ('glob', 'loc')},
               paired_median_official_vs_without={f'{c}_{d}': [float(g.official_pp.median()), float(g.without_pp.median())] for (c, d), g in Pp.groupby(['contrast', 'domain'])},
               note='diagnostic only; the official values and the main table are unchanged; one condition (largest clipped-line load), not an estimate for other conditions',
               seconds=round(time.time() - t0, 1), t_local=K.now_local())
    H.run_put('A1_deletion', out); H.flush_reads('A1_deletion')
    return out


def b2_output_status() -> dict:

    I = pd.read_csv(H.reg(H.S12 / 'inputs.csv'), low_memory=False); X = pd.read_csv(H.reg(H.S12 / 'results' / 'inputs_metrics.csv'), low_memory=False)
    D = I[['input_key', 'sections', 'family', 'surface_class', 'surface', 'layout', 'sigma']].merge(X[['input_key', 'n_accepted', 'id_unconf']], on='input_key', how='left')
    D = D[(D.family == 'line') & (D.surface_class == 'main')].copy()
    S = pd.read_csv(H.reg(H.S345 / 'results' / 'S5-main_cases.csv'), low_memory=False); S = S[S.family == 'line'].rename(columns={'layout_id': 'layout'})
    rows = []
    for sec, d in (('S1', D[D.sections.apply(lambda s: 'S1' in _s12_sections(s))]), ('S2', D[D.sections.apply(lambda s: 'S2' in _s12_sections(s))]), ('S5', S)):
        zero = d.n_accepted.fillna(0) == 0; allu = (d.n_accepted > 0) & (d.id_unconf >= d.n_accepted)
        for (lay, srf, sg), g in d[allu].groupby(['layout', 'surface', 'sigma']):
            rows.append(dict(record_type='B2_output_all_unconf', section=sec, family='line', layout=lay, surface=srf, sigma=float(sg), n_inputs=len(g)))
        rows.append(dict(record_type='B2_output_status', section=sec, family='line', n_inputs=len(d), n_zero_output=int(zero.sum()), n_output_all_unconf=int(allu.sum())))
    T = pd.DataFrame(rows); part('B2_output_status', T)
    out = dict(rows=T[T.record_type == 'B2_output_status'].drop(columns=['record_type']).to_dict('records'),
               note='"has output" counts include inputs whose whole accepted output is unconfirmable; stored fields only', t_local=K.now_local())
    H.run_put('B2_output_status', out); H.flush_reads('B2_output_status')
    return out


def c2_pair_mask() -> list:

    reg = K.load_json(H.reg(K.OUT / 'layouts.json')); r = H.run_load(); beta = 1.15
    out = []
    for g in r['C2']['groups']:
        B = reg['baselines'][g['baseline']]; L = B['layouts']; s1, s3 = g['rms_near_pair']; lim = beta * B['h0_mm']
        blk = s1.split('_')[0]; s3_of_blk = [v['layout'] for v in B['views'] if v['block'] == blk and v['strategy'] == 'S3']
        out.append(dict(baseline=g['baseline'], s1_member=s1, s3_member=s3, h_s1_mm=L[s1]['h_mm'], h_s3_mm=L[s3]['h_mm'], beta_h0_mm=lim,
                        s1_member_satisfies_S3_mask=bool(L[s1]['h_mm'] <= lim), s3_choice_in_s1_block=s3_of_blk[0] if s3_of_blk else None,
                        contrast_valid=bool(L[s1]['h_mm'] > lim and L[s3]['h_mm'] <= lim)))
    return out


def final() -> dict:
    H.run_put('C2_pair_mask_check', dict(note='post-hoc reading of the registered pair rule; the rule is not changed and no new pair is chosen', pairs=c2_pair_mask(), t_local=K.now_local()))
    docs = [H.TASK, H.DRAFTS / 'S4-REAL-03_review.md', H.DRAFTS / 'combined_plan_review.md',
            H.R3 / 'figures' / 'D1_Randlinie_Diagnose.png', H.R3 / 'figures' / 'D2_Punkte_Ausnahme_sigma80.png',
            H.SR / 's4_align01' / 'report_zh.md', H.SR / 's4_align01' / 'geometry_scan.csv']
    H.run_put('documents_read', dict(nature='documents read for this report and the old D2 scan cited by path; hashed at final, not a pre-run snapshot',
                                     files={str(p.relative_to(H.DRAFTS.parent)).replace('\\', '/'): K.sha_file(p) for p in docs}, t_local=K.now_local()))
    r = H.run_load(); bad = [p for p, h in r.get('inputs_read', {}).items() if K.sha_file(K.ROOT / p) != h]
    a02 = {f: K.sha_file(K.OUT / f) == h for f, h in H.reference_R02.items()}
    a03 = {f: K.sha_file(K.ROOT / f) == h for f, h in H.reference_R03.items()}
    code = {K.rel(p): K.sha_file(p) for p in sorted(H.HERE.glob('*.py'))}
    outs = {K.rel(p): K.sha_file(p) for p in sorted(H.OUT.rglob('*')) if p.is_file() and p.name != 'run.json' and '_parts' not in p.parts}
    res = dict(inputs_unchanged=not bad, changed=bad[:10], n_inputs=len(r.get('inputs_read', {})), s4_real02_unchanged=all(a02.values()), s4_real03_unchanged=all(a03.values()),
               final_code=code, outputs=outs, imaging=0, full_redecoding=0, new_detection=0, t_local=K.now_local())
    H.run_put('final', res)
    return {k: v for k, v in res.items() if k not in ('outputs',)}


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd in ('c1', 'c2', 'c3'):
        import s4r4_scan as S
        print(getattr(S, cmd)())
    elif cmd == 'figs':
        import s4r4_figs as F
        print(F.make_all())
    else:
        f = dict(check=check, a=work_a, a1d=deletion_sensitivity, b1=b1, b2o=b2_output_status, b2=b2, b3=b3, assemble=assemble, final=final).get(cmd)
        print(f() if f else __doc__)
