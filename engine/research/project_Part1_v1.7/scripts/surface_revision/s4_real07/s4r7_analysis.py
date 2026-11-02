# Compare held-out layouts, clipping arms and block-level effects.

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import s4r7_common as H

K = H.K
TOL = 1e-12
COLS = ('n_detected', 'n_accepted', 'n_valid', 'c', 'w', 'u', 'n_gross', 'pt_rmse', 'surf_rmse_interpolable', 'sup_rmse', 'Cvalid', 'Ctau', 'n_track_ambiguous', 'n_status_segment_unknown',
        'n_status_11')


def _f(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def domain_rows(c: dict, ev: dict) -> list:
    m = ev['metrics']; nodes = ev['nodes']; si = ev.get('source_identity') or {}
    base = dict(record_type='cond', set=c.get('set'), family=c['family'], baseline=c['baseline'], layout=c['layout'], block=c.get('block'), phase=c['phase'], surface=c['surface'],
                sigma=float(c['sigma']), method=c['method'], image_key=c['image_key'], image_source=c.get('image_source'), decode_key=ev.get('decode_key'), eval_key=ev.get('identity'),
                result_source=ev.get('result_source'), band=bool(c.get('clipped_band_lit') in (True, 'True')), state=m.get('state'),
                n_detected=(m.get('n_detected') if c['family'] == 'line' else m.get('n_features_detected')), n_accepted=m.get('n_accepted'), status_closed=m.get('status_closed'),
                n_status_11=m.get('n_status_clipped_source_abstain'), n_status_segment_unknown=m.get('n_status_segment_unknown'), n_status_no_candidate=m.get('n_status_no_candidate'),
                n_status_guard_chain_short=m.get('n_status_guard_chain_short'), n_track_ambiguous=m.get('n_track_ambiguous'), n_track_duplicate=m.get('n_track_duplicate'),
                n_track_matched=m.get('n_track_matched'), src_correct=si.get('correct'), src_wrong=si.get('wrong'), src_undeterminable=si.get('undeterminable'))
    out = []
    for d in ('glob', 'loc'):
        cc, w, u = int(m[f'{d}_id_correct']), int(m[f'{d}_id_wrong']), int(m[f'{d}_id_unconf']); nv = int(m[f'{d}_n_valid']); g = int(m[f'{d}_n_gross']); nd = nodes[d]
        out.append(dict(base, domain=d, n_valid=nv, c=cc, w=w, u=u, w_share=(w / (cc + w) if cc + w else None), u_share=(u / (cc + w + u) if cc + w + u else None), n_gross=g,
                        gross_rate=(g / nv if nv else None), pt_rmse=_f(m.get(f'{d}_pt_rmse')), pt_p95=_f(m.get(f'{d}_pt_p95')), surf_rmse_interpolable=_f(m.get(f'{d}_surf_rmse')),
                        sup_rmse=_f(nd.get('sup_surf_rmse')), n_supported=nd.get('n_supported'), n_interpolable=nd.get('n_interpolable'), n_nodes=nd.get('n_nodes'),
                        Cvalid=float(m.get(f'{d}_Cvalid') or 0.0), Ctau=float(m.get(f'{d}_Ctau') or 0.0)))
    return out


def default_ctx() -> dict:
    return dict(manifest=H.OUT / 'manifest.csv', plan=H.OUT / '_parts' / 'results_plan.csv', views=H.OUT / '_parts' / 'views.csv', rasters=H.OUT / '_parts' / 'rasters.csv', eval_dir=H.EVL7,
                out=H.OUT)


def _load(ctx: dict):
    import s4r7_run as R
    R.set_read_only(True)
    M = R.read_manifest(ctx['manifest']); evs = {}
    for c in M.to_dict('records'):
        for m in (['D0'] if c['family'] == 'point' else ['P4TABLE', 'CLIP']):
            evs[(c['image_key'], m)] = R.evaluate(c, m, store=ctx['eval_dir'], plan=ctx['plan'])
    return M, evs


def _clip_pairs(M: pd.DataFrame, C: pd.DataFrame, evs: dict, ctx: dict) -> pd.DataFrame:
    rows = []
    for c in M[M.family == 'line'].to_dict('records'):
        k = c['image_key']; eo, en = evs[(k, 'P4TABLE')], evs[(k, 'CLIP')]
        Ao = K.load_npz(ctx['eval_dir'] / f"{eo['identity']}.npz"); An = K.load_npz(ctx['eval_dir'] / f"{en['identity']}.npz")
        eo_ = np.abs(np.asarray(Ao['pts'], np.float64)[:, 2] - np.asarray(Ao['zr'], np.float64)); en_ = np.abs(np.asarray(An['pts'], np.float64)[:, 2] - np.asarray(An['zr'], np.float64))
        rej = ~np.asarray(An['accepted'], bool); acc_o = np.asarray(Ao['accepted'], bool); ab = np.asarray(An['status']) == 11
        for d, dm in (('glob', 'in_global'), ('loc', 'in_local')):
            io = np.asarray(Ao[dm], bool) & np.asarray(Ao['valid'], bool); inn = np.asarray(An[dm], bool) & np.asarray(An['valid'], bool)
            un = io & np.asarray(Ao['point_unconf'], bool); gr = io & (eo_ > H.GROSS_MM); cor = io & np.asarray(Ao['point_correct'], bool)
            o = C[(C.image_key == k) & (C.method == 'P4TABLE') & (C.domain == d)].iloc[0]; n = C[(C.image_key == k) & (C.method == 'CLIP') & (C.domain == d)].iloc[0]
            rec = dict(record_type='clip', set=c['set'], image_key=k, family='line', layout=c['layout'], block=c['block'], phase=c['phase'], surface=c['surface'], sigma=float(c['sigma']),
                       band=bool(c['clipped_band_lit'] in (True, 'True')), domain=d, unconf_rejected=int((un & rej).sum()), gross_rejected=int((gr & rej).sum()),
                       both_rejected=int((un & gr & rej).sum()), union_rejected=int(((un | gr) & rej).sum()), correct_lost=int((cor & rej).sum()), correct_lost_to_abstain=int((cor & ab).sum()),
                       new_wrong=int((inn & np.asarray(An['point_wrong'], bool) & ~(io & np.asarray(Ao['point_wrong'], bool))).sum()),
                       new_unconf=int((inn & np.asarray(An['point_unconf'], bool) & ~un).sum()), new_gross=int((inn & (en_ > H.GROSS_MM) & ~acc_o).sum()), n_abstain=int(ab.sum()),
                       old_state=o.state, new_state=n.state)
            for f in ('c', 'w', 'u', 'n_gross', 'Ctau', 'Cvalid', 'pt_rmse', 'surf_rmse_interpolable', 'sup_rmse'):
                rec[f'old_{f}'] = o[f]; rec[f'new_{f}'] = n[f]
            for f in ('c', 'u', 'n_gross', 'Ctau', 'Cvalid'):
                rec[f'd_{f}'] = float(n[f]) - float(o[f])
            rows.append(rec)
    return pd.DataFrame(rows)


def _views(ctx: dict, C: pd.DataFrame, M: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    VV = pd.read_csv(H.reg(ctx['views']), low_memory=False, keep_default_na=False)
    Rr = pd.read_csv(H.reg(ctx['rasters']), low_memory=False).set_index(['baseline', 'layout', 'phase']) if ctx.get('rasters') else None
    idx = C.set_index(['baseline', 'layout', 'phase', 'surface', 'sigma', 'method', 'domain'])
    rows = []
    for v in VV[VV.status == 'selected'].itertuples():
        fam = 'point' if v.baseline.startswith('R') else 'line'
        for s in H.SURF:
            for sg in (0.0, 40.0):
                for m in (['D0'] if fam == 'point' else ['CLIP', 'P4TABLE']):
                    for d in ('glob', 'loc'):
                        key = (v.baseline, v.layout, v.phase, s, sg, m, d); key0 = (v.baseline, 'regular', v.phase, s, sg, m, d)
                        if key not in idx.index or key0 not in idx.index:
                            raise H.Stop('analysis', f'missing record for view {v.baseline} {v.block} {v.arm} {v.phase} {s} {sg} {m} {d}')
                        a, b = idx.loc[key], idx.loc[key0]
                        rec = dict(record_type='view', set=('anchor' if v.phase == 'H0' else 'main'), family=fam, baseline=v.baseline, block=v.block, arm=v.arm, layout=v.layout, phase=v.phase,
                                   surface=s, sigma=sg, method=m, domain=d, image_key=a.image_key, s0_image_key=b.image_key)
                        if Rr is not None and (v.baseline, v.layout, v.phase) in Rr.index:
                            g = Rr.loc[(v.baseline, v.layout, v.phase)]
                            rec.update(g_min_mm=g.g_min_mm, h_mm=g.h_mm, rms_p0=g.rms_p0, g_min_raster_mm=g.g_min_raster_mm, h_raster_mm=g.h_raster_mm, lit_px=g.lit_px, peak=g.peak)
                        for f in COLS:
                            va, vb = a[f], b[f]
                            rec[f] = va; rec[f'S0_{f}'] = vb
                            rec[f'd_{f}'] = (float(va) - float(vb)) if (va is not None and vb is not None and pd.notna(va) and pd.notna(vb)) else None
                        rows.append(rec)
    Vw = pd.DataFrame(rows)
    con = []
    pairs = (('S2*', 'S1'), ('S3', 'S1'), ('S4*', 'S2*'), ('S4*', 'S3'))
    g = Vw.set_index(['baseline', 'block', 'arm', 'phase', 'surface', 'sigma', 'method', 'domain'])
    for (b, blk, ph, s, sg, m, d), grp in Vw.groupby(['baseline', 'block', 'phase', 'surface', 'sigma', 'method', 'domain']):
        arms = set(grp.arm)
        for a1, a2 in pairs:
            if a1 in arms and a2 in arms:
                x, y = g.loc[(b, blk, a1, ph, s, sg, m, d)], g.loc[(b, blk, a2, ph, s, sg, m, d)]
                con.append(dict(record_type='contrast', set=('anchor' if ph == 'H0' else 'main'), family=x.family, baseline=b, block=blk, contrast=f'{a1}-{a2}', phase=ph, surface=s, sigma=sg,
                                method=m, domain=d, same_layout=bool(x.layout == y.layout), d_Ctau=float(x.Ctau) - float(y.Ctau), d_Cvalid=float(x.Cvalid) - float(y.Cvalid), d_c=float(x.c) - float(y.c),
                                d_sup_rmse=(float(x.sup_rmse) - float(y.sup_rmse) if pd.notna(x.sup_rmse) and pd.notna(y.sup_rmse) else None)))
    return Vw, pd.DataFrame(con)


def _blocks(Vw: pd.DataFrame, Cn: pd.DataFrame, VV: pd.DataFrame | None = None) -> pd.DataFrame:

    rows = []
    base_of = {'point': 'R8x4', 'line': 'L6.3'}
    for src, df, comp_col in (('vs_S0', Vw.assign(comparison=Vw.arm + '-S0', d_Ctau=Vw.d_Ctau, d_Cvalid=Vw.d_Cvalid), 'comparison'), ('within_block', Cn.assign(comparison=Cn.contrast), 'comparison')):
        for keys, g in df.groupby(['family', 'set', 'surface', 'sigma', 'method', 'domain', comp_col]):
            bm = g.assign(d_sup_rmse=pd.to_numeric(g.d_sup_rmse, errors='coerce')).groupby('block')[['d_Ctau', 'd_Cvalid', 'd_sup_rmse']].mean()
            reg_blocks = fail = None
            if VV is not None:
                vv = VV[(VV.baseline == base_of[keys[0]]) & ((VV.phase == 'H0') if keys[1] == 'anchor' else (VV.phase != 'H0'))]
                arms = keys[6].split('-')[:1] if src == 'vs_S0' else keys[6].split('-')
                reg_blocks = int(vv.block.nunique()); fail = int(vv[vv.arm.isin(arms) & (vv.status != 'selected')].block.nunique())
            for metric in ('d_Ctau', 'd_Cvalid', 'd_sup_rmse'):
                sc, unit = (1.0, 'mm') if metric == 'd_sup_rmse' else (100.0, 'pp')
                v = bm[metric].astype(float); n_undef = int(v.isna().sum()); v = v.dropna()
                rows.append(dict(record_type='block', source=src, family=keys[0], set=keys[1], surface=keys[2], sigma=keys[3], method=keys[4], domain=keys[5], comparison=keys[6], metric=metric,
                                 unit=unit, n_blocks=int(len(v)), n_blocks_undefined=n_undef, n_blocks_registered=reg_blocks, n_blocks_generation_failure=fail, n_phases=int(g.phase.nunique()),
                                 median=(float(sc * v.median()) if len(v) else None), min=(float(sc * v.min()) if len(v) else None), max=(float(sc * v.max()) if len(v) else None),
                                 n_pos=int((v > TOL).sum()), n_zero=int((v.abs() <= TOL).sum()), n_neg=int((v < -TOL).sum()),
                                 blocks=';'.join(f'{b}:{sc * x:.4f}' for b, x in v.items())))
    return pd.DataFrame(rows)


def analyse(ctx: dict | None = None, write: bool = True) -> dict:
    ctx = ctx or default_ctx()
    M, evs = _load(ctx)
    rows = []
    for (k, m), ev in evs.items():
        c = M[M.image_key == k].iloc[0].to_dict(); rows += domain_rows(dict(c, method=m), ev)
    C = pd.DataFrame(rows)
    P = _clip_pairs(M, C, evs, ctx)
    Vw, Cn = _views(ctx, C, M)
    VV = pd.read_csv(ctx['views'], low_memory=False, keep_default_na=False)
    Bk = _blocks(Vw, Cn, VV)
    out_df = pd.concat([C, P, Vw, Cn, Bk], ignore_index=True, sort=False); out_df = out_df[['record_type'] + [c for c in out_df.columns if c != 'record_type']]
    summ = dict(n_rows=len(out_df), by_type=out_df.record_type.value_counts().to_dict(), unique_images=int(M.image_key.nunique()), unique_layouts=M.groupby('family').layout.nunique().to_dict(),
                n_views=int(len(VV)), views_selected=int((VV.status == 'selected').sum()), states=C[C.domain == 'glob'].groupby(['family', 'method', 'state']).size().to_dict(),
                status_closed_all=bool(C[C.family == 'line'].status_closed.all()) if (C.family == 'line').any() else None)
    if write:
        H.save_csv(out_df, ctx['out'] / 'results.csv')
        summ.update(examples=examples(C, P, Vw))
        H.run_put('analysis', summ); H.flush_reads('analysis')
    return dict(summ, frames=dict(C=C, P=P, V=Vw, Cn=Cn, B=Bk)) if not write else summ


def examples(C: pd.DataFrame, P: pd.DataFrame, Vw: pd.DataFrame) -> dict:
    v = Vw[(Vw.set == 'main') & (Vw.family == 'point') & (Vw.surface == 'M10_curv') & (Vw.domain == 'glob')].copy()
    v = v.sort_values(['d_Ctau', 'image_key']).iloc[0]
    g = P[P.domain == 'glob'].copy(); g['gred'] = g.old_n_gross - g.new_n_gross
    imp = g.sort_values(['gred', 'image_key'], ascending=[False, True]).iloc[0]
    cost = g.sort_values(['d_Ctau', 'image_key']).iloc[0]
    return dict(dot_m10=dict(image_key=v.image_key, s0_image_key=v.s0_image_key, layout=v.layout, arm=v.arm, block=v.block, phase=v.phase, sigma=v.sigma, d_Ctau=float(v.d_Ctau),
                             label=('largest C_tau drop' if v.d_Ctau < -TOL else 'smallest change (no drop observed)')),
                line_improvement=dict(image_key=imp.image_key, set=imp.set, layout=imp.layout, phase=imp.phase, surface=imp.surface, sigma=imp.sigma, gross_reduction=int(imp.gred),
                                      label=('largest gross reduction' if imp.gred > 0 else 'largest change (no gross reduction observed)')),
                line_cost=dict(image_key=cost.image_key, set=cost.set, layout=cost.layout, phase=cost.phase, surface=cost.surface, sigma=cost.sigma, d_Ctau=float(cost.d_Ctau),
                               label=('largest C_tau drop' if cost.d_Ctau < -TOL else 'smallest C_tau change (no drop observed)')))


def selfcheck() -> dict:
    import r2_eval as E
    import rdy_platform as P
    import s4r6_analysis as S6
    import s4r7_run as R
    R.set_read_only(True)
    Rs = pd.read_csv(H.OUT / 'results.csv', low_memory=False, float_precision='round_trip'); C = Rs[Rs.record_type == 'cond']; Pc = Rs[(Rs.record_type == 'clip') & (Rs.domain == 'glob')]
    V = Rs[(Rs.record_type == 'view') & (Rs.domain == 'glob')]; ex = H.run_load()['analysis']['examples']
    picks = [('dot M10 strategy', ex['dot_m10']['image_key'], 'D0', True), ('dot M10 S0', ex['dot_m10']['s0_image_key'], 'D0', True)]
    d = C[(C.family == 'point') & (C.domain == 'glob') & (C.set == 'main')].sort_values('image_key')
    picks += [('dot B sigma40', d[(d.surface == 'B_curv') & (d.sigma == 40.0)].iloc[0].image_key, 'D0', False),
              ('dot M24 sigma0', d[(d.surface == 'M24_curv') & (d.sigma == 0.0)].iloc[0].image_key, 'D0', False)]
    a = Pc[Pc.set == 'anchor']
    gain = a.assign(r=a.old_n_gross - a.new_n_gross).sort_values(['r', 'image_key'], ascending=[False, True]).iloc[0].image_key
    loss = a.sort_values(['d_Ctau', 'image_key']).iloc[0].image_key
    unch = Pc[(Pc.set == 'main') & (Pc.surface == 'M10_curv') & (Pc.sigma == 40.0)].sort_values('image_key').iloc[0].image_key
    picks += [('line anchor gain (CLIP)', gain, 'CLIP', True), ('line anchor gain (P4)', gain, 'P4TABLE', True), ('line anchor loss (CLIP)', loss, 'CLIP', True),
              ('line main unchanged (CLIP)', unch, 'CLIP', True)]
    out = []
    for lab, key, m, surf in picks:
        c = dict(R.cond(key), method=m); ev = R.evaluate(c, m); A = K.load_npz(H.EVL7 / f"{ev['identity']}.npz")
        pts = np.asarray(A['pts'], np.float64); val = np.asarray(A['valid'], bool); e = pts[:, 2] - np.asarray(A['zr'], np.float64)
        rec = dict(case=lab, image_key=key, method=m, surface=c['surface'], sigma=float(c['sigma']))
        for dd, dm in (('glob', 'in_global'), ('loc', 'in_local')):
            ind = np.asarray(A[dm], bool) & val; row = C[(C.image_key == key) & (C.method == m) & (C.domain == dd)].iloc[0]
            ee = e[ind]; rm = float(np.sqrt(np.mean(ee ** 2))) if len(ee) else None
            rec[f'{dd}_c_equal'] = int((ind & np.asarray(A['point_correct'], bool)).sum()) == int(row.c) if 'point_correct' in A else None
            rec[f'{dd}_gross_equal'] = int((np.abs(ee) > H.GROSS_MM).sum()) == int(row.n_gross)
            rec[f'{dd}_pt_rmse_diff'] = (abs(rm - row.pt_rmse) if rm is not None and pd.notna(row.pt_rmse) else (0.0 if rm is None and pd.isna(row.pt_rmse) else 1.0))
        if surf:
            dom = E.domain('omega1'); sid = K.RK.SURF_MAP[c['surface']]
            gl = S6._surface_independent(pts[val], dom['x'], dom['y'], np.asarray(dom[f'z_{sid}'], np.float64), np.full(len(dom['x']), dom['step'] ** 2))
            dl = P.dom_local(c['surface']); lo = S6._surface_independent(pts[val], dl['x'], dl['y'], np.asarray(dl['z_ref'], np.float64), np.asarray(dl['area'], np.float64))
            for dd, s in (('glob', gl), ('loc', lo)):
                row = C[(C.image_key == key) & (C.method == m) & (C.domain == dd)].iloc[0]
                rec[f'{dd}_Ctau_diff'] = abs(s['Ctau'] - row.Ctau); rec[f'{dd}_Cvalid_diff'] = abs(s['Cvalid'] - row.Cvalid)
                rec[f'{dd}_sup_rmse_diff'] = (abs(s['sup_rmse'] - row.sup_rmse) if s['sup_rmse'] is not None and pd.notna(row.sup_rmse) else (0.0 if s['sup_rmse'] is None and pd.isna(row.sup_rmse) else 1.0))
        out.append(rec)

    ok_pairs = all(abs((C[(C.image_key == r.image_key) & (C.method == 'CLIP') & (C.domain == 'glob')].Ctau.iloc[0] - C[(C.image_key == r.image_key) & (C.method == 'P4TABLE') & (C.domain == 'glob')].Ctau.iloc[0]) - r.d_Ctau) < 1e-12 for r in Pc.itertuples())
    ok_views = all(abs((C[(C.image_key == r.image_key) & (C.method == r.method) & (C.domain == 'glob')].Ctau.iloc[0] - C[(C.image_key == r.s0_image_key) & (C.method == r.method) & (C.domain == 'glob')].Ctau.iloc[0]) - r.d_Ctau) < 1e-12 for r in V.itertuples())
    num = [v for x in out for k, v in x.items() if k.endswith('_diff')]; eqs = [v for x in out for k, v in x.items() if k.endswith('_equal') and v is not None]
    res = dict(cases=out, n_cases=len(out), max_numeric_diff=float(max(num)), all_counts_equal=bool(all(eqs)), clip_pair_diffs_consistent=bool(ok_pairs), view_diffs_consistent=bool(ok_views),
               note='Numerical check using a second computation path.', t_local=K.now_local())
    res['ok'] = bool(res['all_counts_equal'] and ok_pairs and ok_views and res['max_numeric_diff'] < 1e-6)
    H.run_put('selfcheck', res)
    return res
