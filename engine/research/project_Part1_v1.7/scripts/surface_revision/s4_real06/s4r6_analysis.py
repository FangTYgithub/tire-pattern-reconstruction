# Aggregate layout and clipping results and select representative cases.

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import s4r6_common as H

K = H.K
CATS = ('acc_correct', 'acc_wrong', 'acc_unconf', 'acc_invalid', 'no_candidate', 'segment_unknown', 'inconsistent_with_segment', 'triangulation_invalid', 'guard_chain_short',
        'clipped_source_abstain', 'other')
STATUS_NAME = {0: 'accepted', 1: 'no_candidate', 2: 'segment_unknown', 3: 'inconsistent_with_segment', 4: 'triangulation_invalid', 8: 'guard_chain_short', 11: 'clipped_source_abstain'}


def _f(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def domain_rows(c: dict, ev: dict) -> list:
    m = ev['metrics']; nodes = ev['nodes']; out = []
    base = dict(record_type='cond', set=c.get('set'), family=c['family'], baseline=c['baseline'], surface=c['surface'], strategy_view=c.get('strategy_view'), layout=c['layout'], j=c.get('j'),
                block=c.get('block'), phase=c['phase'], sigma=float(c['sigma']), noise_seed=c.get('noise_seed'), method=c['method'], image_key=c['image_key'], image_source=c.get('image_source'),
                decode_key=ev.get('decode_key'), eval_key=ev.get('identity'), result_source=ev.get('result_source'), band=bool(c.get('clipped_band_lit') in (True, 'True')),
                state=m.get('state'), n_detected=(m.get('n_detected') if c['family'] == 'line' else m.get('n_features_detected')), n_accepted=m.get('n_accepted'),
                n_valid_total=m.get('n_valid'), status_closed=m.get('status_closed'), n_status_11=m.get('n_status_clipped_source_abstain'))
    if c['family'] == 'line':
        base.update({f'n_status_{nm}': m.get(f'n_status_{nm}') for nm in ('accepted', 'no_candidate', 'segment_unknown', 'inconsistent_with_segment', 'triangulation_invalid', 'guard_chain_short')})
    si = ev.get('source_identity') or {}
    if c['family'] == 'point':
        base.update(src_correct=si.get('correct'), src_wrong=si.get('wrong'), src_undeterminable=si.get('undeterminable'), src_n_reliable=si.get('n_reliable'))
    for d in ('glob', 'loc'):
        cc, w, u = int(m[f'{d}_id_correct']), int(m[f'{d}_id_wrong']), int(m[f'{d}_id_unconf']); nv = int(m[f'{d}_n_valid']); g = int(m[f'{d}_n_gross'])
        nd = nodes['glob' if d == 'glob' else 'loc']
        out.append(dict(base, domain=d, n_valid=nv, c=cc, w=w, u=u, w_share=(w / (cc + w) if cc + w else None), u_share=(u / (cc + w + u) if cc + w + u else None), n_gross=g,
                        gross_rate=(g / nv if nv else None), pt_rmse=_f(m.get(f'{d}_pt_rmse')), pt_p95=_f(m.get(f'{d}_pt_p95')), surf_rmse_interpolable=_f(m.get(f'{d}_surf_rmse')),
                        sup_rmse=_f(nd.get('sup_surf_rmse')), n_supported=nd.get('n_supported'), n_interpolable=nd.get('n_interpolable'), n_nodes=nd.get('n_nodes'),
                        Cvalid=float(m.get(f'{d}_Cvalid') or 0.0), Ctau=float(m.get(f'{d}_Ctau') or 0.0)))
    return out


def _cats(A: dict) -> np.ndarray:
    st = np.asarray(A['status'], np.int64); cc = np.full(len(st), CATS.index('other'), np.int8)
    for k, nm in STATUS_NAME.items():
        if nm in CATS and nm != 'accepted':
            cc[st == k] = CATS.index(nm)
    acc = np.asarray(A['accepted'], bool); val = np.asarray(A['valid'], bool)
    cc[acc & ~val] = CATS.index('acc_invalid')
    for nm, key in (('acc_unconf', 'point_unconf'), ('acc_wrong', 'point_wrong'), ('acc_correct', 'point_correct')):
        cc[acc & val & np.asarray(A[key], bool)] = CATS.index(nm)
    return cc


def _load_all():
    import s4r6_run as R
    R.set_read_only(True)
    M = R.manifest(); evs = {}; conds = {}
    for c in M.to_dict('records'):
        conds[c['image_key']] = c
        for m in (['D0'] if c['family'] == 'point' else ['P4TABLE', 'CLIP']):
            evs[(c['image_key'], m)] = R.evaluate(c, m)
    return M, conds, evs


# Aggregate layout and clipping comparisons and select representative cases.
def analyse() -> dict:
    M, conds, evs = _load_all()
    rows = []
    for (k, m), ev in evs.items():
        rows += domain_rows(dict(conds[k], method=m), ev)
    C = pd.DataFrame(rows)

    pairs = []; npz = {}
    for k, c in conds.items():
        if c['family'] != 'line':
            continue
        eo, en = evs[(k, 'P4TABLE')], evs[(k, 'CLIP')]
        Ao = K.load_npz(H.EVL6 / f"{eo['identity']}.npz"); An = K.load_npz(H.EVL6 / f"{en['identity']}.npz")
        co, cn = _cats(Ao), _cats(An); T = np.zeros((len(CATS), len(CATS)), np.int64); np.add.at(T, (co, cn), 1); npz[f'{k}__transition'] = T
        e_old = np.abs(np.asarray(Ao['pts'], np.float64)[:, 2] - np.asarray(Ao['zr'], np.float64)); gross_old = np.asarray(Ao['valid'], bool) & (e_old > H.GROSS_MM)
        e_new = np.abs(np.asarray(An['pts'], np.float64)[:, 2] - np.asarray(An['zr'], np.float64)); gross_new = np.asarray(An['valid'], bool) & (e_new > H.GROSS_MM)
        ab = np.asarray(An['status']) == 11; acc_n = np.asarray(An['accepted'], bool); acc_o = np.asarray(Ao['accepted'], bool)
        for d, dm in (('glob', 'in_global'), ('loc', 'in_local')):
            o = C[(C.image_key == k) & (C.method == 'P4TABLE') & (C.domain == d)].iloc[0]; n = C[(C.image_key == k) & (C.method == 'CLIP') & (C.domain == d)].iloc[0]
            ind_o, ind_n = np.asarray(Ao[dm], bool), np.asarray(An[dm], bool)
            corr_o = np.asarray(Ao['point_correct'], bool) & ind_o
            rec = dict(record_type='clip', image_key=k, family='line', surface=c['surface'], strategy_view=c['strategy_view'], layout=c['layout'], j=c['j'], phase=c['phase'],
                       sigma=float(c['sigma']), set=c['set'], band=bool(c['clipped_band_lit'] in (True, 'True')), domain=d, image_source=c['image_source'],
                       p4_source=eo['result_source'], clip_source=en['result_source'], old_state=o.state, new_state=n.state,
                       old_correct_lost=int((corr_o & ~acc_n).sum()), old_correct_lost_to_abstain=int((corr_o & ab).sum()),
                       old_correct_lost_to_unknown=int((corr_o & (np.asarray(An['status']) == 2)).sum()),
                       old_unconf_rejected=int((np.asarray(Ao['point_unconf'], bool) & ind_o & ~acc_n).sum()), old_wrong_rejected=int((np.asarray(Ao['point_wrong'], bool) & ind_o & ~acc_n).sum()),
                       old_gross_rejected=int((gross_old & ind_o & ~acc_n).sum()), new_gross_added=int((gross_new & ind_n & ~acc_o).sum()),
                       new_wrong_added=int((np.asarray(An['point_wrong'], bool) & ind_n & ~(np.asarray(Ao['point_wrong'], bool) & ind_o)).sum()),
                       new_unconf_added=int((np.asarray(An['point_unconf'], bool) & ind_n & ~(np.asarray(Ao['point_unconf'], bool) & ind_o)).sum()),
                       accepted_intersection=int((acc_o & acc_n).sum()), new_accepted_not_old=int((acc_n & ~acc_o).sum()), n_abstain=int(ab.sum()))
            for f in ('c', 'w', 'u', 'n_valid', 'n_gross', 'Ctau', 'Cvalid', 'pt_rmse', 'pt_p95', 'surf_rmse_interpolable', 'sup_rmse', 'w_share', 'u_share'):
                rec[f'old_{f}'] = o[f]; rec[f'new_{f}'] = n[f]
            for f in ('c', 'w', 'u', 'n_gross', 'Ctau', 'Cvalid'):
                rec[f'd_{f}'] = float(n[f]) - float(o[f])
            pairs.append(rec)
    P = pd.DataFrame(pairs)

    keyc = ['family', 'surface', 'phase', 'sigma', 'method', 'domain']; cols = ['c', 'w', 'u', 'n_valid', 'n_gross', 'Ctau', 'Cvalid', 'sup_rmse', 'pt_rmse', 'n_accepted', 'n_detected']
    S0 = C[C.strategy_view == 'S0'].set_index(keyc); S3 = C[(C.strategy_view == 'S3')].set_index(keyc); comp = []
    for r in C[C.set == 'pilot'].itertuples():
        k = tuple(getattr(r, x) for x in keyc)
        for lab, ref in (('s0', S0),) + ((('rms', S3),) if r.strategy_view == 'S1-RMS' else ()):
            b = ref.loc[k]
            rec = dict(record_type=lab, image_key=r.image_key, ref_image_key=b.image_key, family=r.family, surface=r.surface, phase=r.phase, sigma=r.sigma, method=r.method, domain=r.domain,
                       strategy_view=r.strategy_view, layout=r.layout, j=r.j, ref_layout=b.layout, band=r.band)
            for f in cols:
                rec[f'{f}'] = getattr(r, f); rec[f'ref_{f}'] = b[f]
                rec[f'd_{f}'] = (float(getattr(r, f)) - float(b[f])) if (getattr(r, f) is not None and b[f] is not None and pd.notna(getattr(r, f)) and pd.notna(b[f])) else None
            comp.append(rec)
    S = pd.DataFrame(comp)

    BR = pd.read_csv(H.reg(H.R5 / '_parts' / 'B_rasters.csv'), low_memory=False).set_index(['baseline', 'layout', 'phase'])
    gen = []
    for r in C[C.domain == 'glob'].itertuples():
        g = BR.loc[(r.baseline, r.layout, r.phase)]
        gen.append(dict(record_type='gen', image_key=r.image_key, family=r.family, baseline=r.baseline, layout=r.layout, strategy_view=r.strategy_view, j=r.j, phase=r.phase, surface=r.surface,
                        sigma=r.sigma, method=r.method, g_min_mm=g.g_min_mm, h_mm=g.h_mm, rms_p0=g.rms_p0, g_min_raster_mm=g.g_min_raster_mm, h_raster_mm=g.h_raster_mm,
                        edge_gap_raster_mm=g.edge_gap_raster_mm, lit_px=g.lit_px, peak=g.peak, lit_minus_S0=g.lit_minus_S0, n_detected=r.n_detected, n_accepted=r.n_accepted,
                        c=r.c, u=r.u, w=r.w, n_valid=r.n_valid, Cvalid=r.Cvalid, Ctau=r.Ctau, sup_rmse=r.sup_rmse))
    G = pd.DataFrame(gen)
    Rall = pd.concat([C, P, S, G], ignore_index=True, sort=False)
    Rall = Rall[['record_type'] + [x for x in Rall.columns if x != 'record_type']]
    H.save_csv(Rall, H.OUT / 'results.csv'); h = H.save_npz(H.OUT / 'transitions.npz', categories=np.array(CATS), **npz)
    summ = summarise(C, P, S)
    summ.update(n_rows=len(Rall), by_type=Rall.record_type.value_counts().to_dict(), transitions_sha256=h, examples=examples(P), t_local=K.now_local())
    H.run_put('analysis', summ); H.flush_reads('analysis')
    return {k: v for k, v in summ.items() if k not in ('clip_strata', 's0_strata')}


def summarise(C: pd.DataFrame, P: pd.DataFrame, S: pd.DataFrame) -> dict:
    out = dict(n_cond_rows=len(C), states=C[C.domain == 'glob'].groupby(['family', 'method', 'state']).size().to_dict(),
               status_closed_all=bool(C[C.family == 'line'].status_closed.all()), status11_total=int(C[(C.domain == 'glob') & (C.method == 'CLIP')].n_status_11.sum()))
    g = P[P.domain == 'glob']
    st = g.groupby(['band', 'surface', 'sigma']).agg(n=('image_key', 'size'), old_u=('old_u', 'sum'), new_u=('new_u', 'sum'), old_w=('old_w', 'sum'), new_w=('new_w', 'sum'),
                                                     old_gross=('old_n_gross', 'sum'), new_gross=('new_n_gross', 'sum'), correct_lost=('old_correct_lost', 'sum'),
                                                     abstain=('n_abstain', 'sum'), dCtau_min=('d_Ctau', 'min'), dCtau_max=('d_Ctau', 'max'), dCvalid_min=('d_Cvalid', 'min'),
                                                     dCvalid_max=('d_Cvalid', 'max')).reset_index()
    out['clip_strata'] = st.to_dict('records')
    loc = P[P.domain == 'loc']
    out['clip_local'] = dict(any_change=int((loc[['d_c', 'd_w', 'd_u', 'd_Ctau', 'd_Cvalid']].abs().sum(axis=1) > 0).sum()), correct_lost=int(loc.old_correct_lost.sum()))
    out['clip_nonband_changes'] = int((g[~g.band][['d_c', 'd_w', 'd_u', 'd_n_gross', 'd_Ctau', 'd_Cvalid']].abs().sum(axis=1) > 0).sum())
    out['clip_new_wrong'] = int(g.new_wrong_added.sum()); out['clip_new_unconf'] = int(g.new_unconf_added.sum()); out['clip_new_gross'] = int(g.new_gross_added.sum())
    out['clip_state_changes'] = g[g.old_state != g.new_state][['image_key', 'surface', 'phase', 'sigma', 'layout', 'old_state', 'new_state']].to_dict('records')
    out['clip_correct_loss_cases'] = g[g.old_correct_lost > 0][['image_key', 'surface', 'phase', 'sigma', 'layout', 'old_correct_lost', 'd_Ctau']].to_dict('records')
    s = S[S.domain == 'glob']
    out['s0_strata'] = s.groupby(['record_type', 'family', 'strategy_view', 'method', 'sigma']).agg(n=('image_key', 'size'), dCtau_med=('d_Ctau', 'median'), dCtau_min=('d_Ctau', 'min'),
                                                                                                   dCtau_max=('d_Ctau', 'max'), dCvalid_med=('d_Cvalid', 'median'),
                                                                                                   dc_med=('d_c', 'median')).reset_index().to_dict('records')
    return out


def examples(P: pd.DataFrame) -> dict:
    g = P[P.domain == 'glob'].copy(); g['gross_red'] = g.old_n_gross - g.new_n_gross; g['ctau_drop'] = -g.d_Ctau
    imp = g.sort_values(['gross_red', 'image_key'], ascending=[False, True]).iloc[0]
    cost = g.sort_values(['ctau_drop', 'image_key'], ascending=[False, True]).iloc[0]
    ctl = g[(g.layout == imp.layout) & (g.surface == imp.surface) & (g.sigma == imp.sigma) & (g.phase == 'H1')].sort_values('image_key').iloc[0]
    return dict(improvement=dict(image_key=imp.image_key, layout=imp.layout, surface=imp.surface, phase=imp.phase, sigma=imp.sigma, gross_reduction=int(imp.gross_red)),
                cost=dict(image_key=cost.image_key, layout=cost.layout, surface=cost.surface, phase=cost.phase, sigma=cost.sigma, dCtau=float(cost.d_Ctau)),
                control=dict(image_key=ctl.image_key, layout=ctl.layout, surface=ctl.surface, phase=ctl.phase, sigma=ctl.sigma))


def _rho_closed_form(V: np.ndarray) -> np.ndarray:

    rho = np.empty(len(V))
    for t, (a, b, c) in enumerate(V):
        L = np.array([np.linalg.norm(b - c), np.linalg.norm(a - c), np.linalg.norm(a - b)])
        i = int(np.argmax(L)); P = [a, b, c]; C_ = P[i]; A_, B_ = [P[j] for j in range(3) if j != i]
        ab = B_ - A_; ca = C_ - A_; cb = C_ - B_

        def f(x):
            return min(np.linalg.norm(x - a), np.linalg.norm(x - b), np.linalg.norm(x - c))
        area2 = abs(ab[0] * ca[1] - ab[1] * ca[0]); cand = []
        if area2 > 0 and (L ** 2).sum() - 2 * L[i] ** 2 >= 0:
            cand.append(L[0] * L[1] * L[2] / (2 * area2))
        else:
            for P0, dv, w in ((A_, ab, ca), (B_, -ab, cb)):
                den = 2 * np.dot(dv, w)
                if den > 0:
                    tt = np.dot(w, w) / den
                    if 0.0 <= tt <= 1.0:
                        cand.append(f(P0 + tt * dv))
            cand.append(f(A_ + 0.5 * ab))
        rho[t] = max(cand)
    return rho


def _surface_independent(pts: np.ndarray, x, y, z, area, d=6.0, tau=0.15) -> dict:
    from scipy.spatial import Delaunay
    P = pts[np.isfinite(pts).all(1)]
    _, first = np.unique(np.round(P[:, :2] / 1e-6).astype(np.int64), axis=0, return_index=True); P = P[np.sort(first)]
    if len(P) < 3:
        return dict(Cvalid=0.0, Ctau=0.0, sup_rmse=None, interp_rmse=None)
    tri = Delaunay(P[:, :2]); q = np.column_stack([x, y]); s = tri.find_simplex(q); ins = s >= 0
    V = P[tri.simplices][:, :, :2]; rho = _rho_closed_form(V)
    zest = np.full(len(q), np.nan)
    for idx in np.flatnonzero(ins):
        a, b, c = P[tri.simplices[s[idx]]]
        M = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]]); l1, l2 = np.linalg.solve(M, q[idx] - a[:2])
        zest[idx] = a[2] + l1 * (b[2] - a[2]) + l2 * (c[2] - a[2])
    e = zest - z; interp = ins & np.isfinite(e); sup = interp.copy(); sup[ins] &= rho[s[ins]] <= d
    A = area.sum()
    return dict(Cvalid=float(area[sup].sum() / A), Ctau=float(area[sup & (np.abs(e) <= tau)].sum() / A), sup_rmse=(float(np.sqrt(np.mean(e[sup] ** 2))) if sup.any() else None),
                interp_rmse=(float(np.sqrt(np.mean(e[interp] ** 2))) if interp.any() else None))


def selfcheck() -> dict:
    import r2_eval as E
    import rdy_platform as P
    import s4r6_run as R
    R.set_read_only(True)
    Rs = pd.read_csv(H.OUT / 'results.csv', low_memory=False, float_precision='round_trip'); C = Rs[Rs.record_type == 'cond']; Pc = Rs[(Rs.record_type == 'clip') & (Rs.domain == 'glob')]
    ex = H.run_load()['analysis']['examples']
    pick = []
    d0 = C[(C.family == 'point') & (C.domain == 'glob') & (C.set == 'pilot')].sort_values('image_key')
    pick.append(('dot sigma0', d0[(d0.sigma == 0.0) & (d0.surface == 'B_curv')].iloc[0].image_key, 'D0'))
    pick.append(('dot sigma40', d0[(d0.sigma == 40.0) & (d0.surface == 'M24_curv')].iloc[0].image_key, 'D0'))
    pick += [('line improvement', ex['improvement']['image_key'], 'CLIP'), ('line cost', ex['cost']['image_key'], 'CLIP'), ('line control H1', ex['control']['image_key'], 'CLIP')]
    pk = Pc[(Pc.surface == 'M10_curv') & (Pc.sigma == 40.0)].sort_values('image_key').iloc[0].image_key
    pick.append(('line P4 M10 sigma40', pk, 'P4TABLE'))
    out = []
    for lab, key, m in pick:
        c = dict(R.cond(key), method=m); ev = R.evaluate(c, m); A = K.load_npz(H.EVL6 / f"{ev['identity']}.npz")
        pts = np.asarray(A['pts'], np.float64); val = np.asarray(A['valid'], bool); e = pts[:, 2] - np.asarray(A['zr'], np.float64)
        rec = dict(case=lab, image_key=key, method=m)
        for d, dm in (('glob', 'in_global'), ('loc', 'in_local')):
            ind = np.asarray(A[dm], bool) & val; row = C[(C.image_key == key) & (C.method == m) & (C.domain == d)].iloc[0]
            cc = int((ind & np.asarray(A['point_correct'], bool)).sum()) if 'point_correct' in A else None
            ee = e[ind]; rm = float(np.sqrt(np.mean(ee ** 2))) if len(ee) else None; p95 = float(np.percentile(np.abs(ee), 95)) if len(ee) else None
            rec[f'{d}_c_equal'] = (cc == int(row.c)) if cc is not None else None
            rec[f'{d}_gross_equal'] = int((np.abs(ee) > H.GROSS_MM).sum()) == int(row.n_gross)
            rec[f'{d}_pt_rmse_diff'] = (abs(rm - row.pt_rmse) if rm is not None and pd.notna(row.pt_rmse) else (0.0 if rm is None and pd.isna(row.pt_rmse) else None))
            rec[f'{d}_pt_p95_diff'] = (abs(p95 - row.pt_p95) if p95 is not None and pd.notna(row.pt_p95) else (0.0 if p95 is None and pd.isna(row.pt_p95) else None))
        if lab in ('line improvement', 'line cost', 'line control H1', 'dot sigma40'):
            dom = E.domain('omega1'); sid = K.RK.SURF_MAP[c['surface']]
            gl = _surface_independent(pts[val], dom['x'], dom['y'], np.asarray(dom[f'z_{sid}'], np.float64), np.full(len(dom['x']), dom['step'] ** 2))
            dl = P.dom_local(c['surface']); lo = _surface_independent(pts[val], dl['x'], dl['y'], np.asarray(dl['z_ref'], np.float64), np.asarray(dl['area'], np.float64))
            for d, s in (('glob', gl), ('loc', lo)):
                row = C[(C.image_key == key) & (C.method == m) & (C.domain == d)].iloc[0]
                rec[f'{d}_Ctau_diff'] = abs(s['Ctau'] - row.Ctau); rec[f'{d}_Cvalid_diff'] = abs(s['Cvalid'] - row.Cvalid)
                rec[f'{d}_sup_rmse_diff'] = (abs(s['sup_rmse'] - row.sup_rmse) if s['sup_rmse'] is not None and pd.notna(row.sup_rmse) else (0.0 if s['sup_rmse'] is None and pd.isna(row.sup_rmse) else None))
                rec[f'{d}_interp_rmse_diff'] = (abs(s['interp_rmse'] - row.surf_rmse_interpolable) if s['interp_rmse'] is not None and pd.notna(row.surf_rmse_interpolable) else None)
        out.append(rec)

    pd_ok = []
    for r in Pc.itertuples():
        o = C[(C.image_key == r.image_key) & (C.method == 'P4TABLE') & (C.domain == 'glob')].iloc[0]; n = C[(C.image_key == r.image_key) & (C.method == 'CLIP') & (C.domain == 'glob')].iloc[0]
        pd_ok.append(abs((n.Ctau - o.Ctau) - r.d_Ctau) < 1e-12 and (n.c - o.c) == r.d_c and (n.n_gross - o.n_gross) == r.d_n_gross)
    num = [v for x in out for k, v in x.items() if k.endswith('_diff') and v is not None]
    eqs = [v for x in out for k, v in x.items() if k.endswith('_equal') and v is not None]
    res = dict(cases=out, max_numeric_diff=float(max(num)) if num else None, all_counts_equal=bool(all(eqs)), pair_diffs_consistent=bool(all(pd_ok)), n_pairs_checked=len(pd_ok),
               note='Stored-array check using a closed-form support radius and barycentric interpolation.',
               t_local=K.now_local())
    res['ok'] = bool(res['all_counts_equal'] and res['pair_diffs_consistent'] and (res['max_numeric_diff'] is None or res['max_numeric_diff'] < 1e-6))
    H.run_put('selfcheck', res)
    return res
