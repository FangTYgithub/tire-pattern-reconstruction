# Link reconstruction outcomes to ideal sampling and independent interpolation checks.

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import tc_common as H

TOL = 1e-12
DOMS = ('glob', 'loc')


def _split(prefix: str, x: dict) -> dict:
    ci, cv, ct = x[f'{prefix}C_int'], x[f'{prefix}C_valid_d6'], x[f'{prefix}C_tau_d6']
    a, b, c = 1.0 - ci, ci - cv, cv - ct
    return dict(share_not_interpolable=a, share_not_supported=b, share_error_above_tau=c, split_sum_minus_1mCtau=(a + b + c) - (1.0 - ct), split_nonneg=bool(min(a, b, c) >= -1e-15))


def _ref_records(batch: str) -> pd.DataFrame:
    RF = pd.read_csv(H.PARTS / f'{batch}_refs.csv', low_memory=False, float_precision='round_trip'); rows = []
    for r in RF.to_dict('records'):
        for dn in DOMS:
            x = {k[len(dn) + 1:]: v for k, v in r.items() if k.startswith(dn + '_')}
            rows.append(dict(record_type='ref', batch=batch, ref_key=r['ref_key'], platform_surface=r['platform_surface'], domain=dn, **x, **_split('', x)))
    return pd.DataFrame(rows)


def _s12_views() -> pd.DataFrame:
    V = pd.read_csv(H.reg(H.S12 / 'results' / 'views.csv'), low_memory=False, float_precision='round_trip')
    return V[(V.surface_class == 'main') & (V.regime == 'resource_cap_11695')].copy()


# Join actual reconstruction outcomes with the corresponding ideal-sampling references.
def link() -> dict:
    t0 = time.time()
    REF = pd.concat([_ref_records('S12'), _ref_records('R7')], ignore_index=True)
    bad = REF[(REF.split_sum_minus_1mCtau.abs() > 1e-12) | ~REF.split_nonneg]
    if len(bad):
        raise H.Stop('check', f'{len(bad)} reference splits not closed / negative')
    rk = REF.set_index(['ref_key', 'domain'])

    I = pd.read_csv(H.PARTS / 'S12_inputs.csv', low_memory=False, float_precision='round_trip')
    rmap = {(a, b): k for a, b, k in zip(I.platform_surface, I.raster_key, I.ref_key)}
    V = _s12_views(); V['ref_key'] = [rmap[(a, b)] for a, b in zip(V.platform_surface, V.raster_key)]
    act0 = I.set_index('input_key')
    links = []
    for v in V.to_dict('records'):
        for dn in DOMS:
            rf = rk.loc[(v['ref_key'], dn)]
            act = dict(C_tau=float(v[f'{dn}_Ctau'] or 0.0) if pd.notna(v[f'{dn}_Ctau']) else 0.0, C_valid=float(v[f'{dn}_Cvalid']) if pd.notna(v[f'{dn}_Cvalid']) else 0.0,
                       C_int=float(v[f'{dn}_cov_interp']) if pd.notna(v[f'{dn}_cov_interp']) else 0.0, int_rmse=v[f'{dn}_surf_rmse'], pt_rmse=v[f'{dn}_pt_rmse'])
            rec = dict(record_type='link', batch='S12', section=v['section'], family=v['family'], layout=v['layout'], size_mm=v['size_mm'], surface=v['surface'], phase=v['phase_label'],
                       sigma=float(v['sigma']), rep=int(v['rep']), view_id=v['view_id'], input_key=v['input_key'], ref_key=v['ref_key'], domain=dn, state=v['state'],
                       **{f'act_{k}': val for k, val in act.items()}, ref_C_int=rf.C_int, ref_C_valid=rf.C_valid_d6, ref_C_tau=rf.C_tau_d6, ref_sup_rmse=rf.sup_rmse_d6, ref_int_rmse=rf.int_rmse,
                       ref_C_tau_d5=rf.C_tau_d5, ref_C_tau_d7=rf.C_tau_d7, ref_C_valid_d5=rf.C_valid_d5, ref_C_valid_d7=rf.C_valid_d7)
            if float(v['sigma']) == 0.0:
                a = act0.loc[v['input_key']]
                rec.update(act_sup_rmse=a[f'act_{dn}_sup_rmse_d6'], act_C_tau_d5=a[f'act_{dn}_C_tau_d5'], act_C_tau_d7=a[f'act_{dn}_C_tau_d7'],
                           act_C_valid_d5=a[f'act_{dn}_C_valid_d5'], act_C_valid_d7=a[f'act_{dn}_C_valid_d7'])
            rec.update(gap_C_tau=rec['act_C_tau'] - rec['ref_C_tau'], gap_C_valid=rec['act_C_valid'] - rec['ref_C_valid'], gap_C_int=rec['act_C_int'] - rec['ref_C_int'])
            links.append(rec)
    L12 = pd.DataFrame(links)

    RC = pd.read_csv(H.PARTS / 'R7_conditions.csv', low_memory=False, float_precision='round_trip')
    r7map = {(bb, a, b, c): k for bb, a, b, c, k in zip(RC.baseline, RC.surface, RC.layout, RC.phase, RC.ref_key)}
    S7 = pd.read_csv(H.reg(H.R7 / 'results.csv'), low_memory=False, float_precision='round_trip'); C7 = S7[S7.record_type == 'cond']
    a7 = RC.set_index(['image_key', 'method'])
    l7 = []
    for c in C7.to_dict('records'):
        dn = c['domain']; key = r7map[(c['baseline'], c['surface'], c['layout'], c['phase'])]; rf = rk.loc[(key, dn)]
        rec = dict(record_type='link', batch='R7', set=c['set'], family=c['family'], baseline=c['baseline'], layout=c['layout'], block=c['block'], surface=c['surface'], phase=c['phase'],
                   sigma=float(c['sigma']), method=c['method'], image_key=c['image_key'], ref_key=key, domain=dn, state=c['state'], act_C_tau=float(c['Ctau']), act_C_valid=float(c['Cvalid']),
                   act_int_rmse=c['surf_rmse_interpolable'], act_sup_rmse=c['sup_rmse'], act_pt_rmse=c['pt_rmse'], ref_C_int=rf.C_int, ref_C_valid=rf.C_valid_d6, ref_C_tau=rf.C_tau_d6,
                   ref_sup_rmse=rf.sup_rmse_d6, ref_int_rmse=rf.int_rmse, ref_C_tau_d5=rf.C_tau_d5, ref_C_tau_d7=rf.C_tau_d7)
        if float(c['sigma']) == 0.0:
            a = a7.loc[(c['image_key'], c['method'])]; rec['act_C_int'] = a[f'act_{dn}_C_int']
        rec.update(gap_C_tau=rec['act_C_tau'] - rec['ref_C_tau'], gap_C_valid=rec['act_C_valid'] - rec['ref_C_valid'])
        l7.append(rec)
    L7 = pd.DataFrame(l7)
    own = {(i, m): k for i, m, k in zip(RC.image_key, RC.method, RC.ref_key)}
    chk = [own[(r['image_key'], r['method'])] == r['ref_key'] for r in l7 if r['sigma'] == 0.0]
    if not all(chk):
        raise H.Stop('check', f'{chk.count(False)} sigma-0 R7 links point to a foreign reference')

    ri = [dict(record_type='ref_input', batch='S12', section=r['sections'], family=r['family'], layout=r['layout'], size_mm=r['size_mm'], surface=r['surface'], phase=r['phase_label'],
               input_key=r['input_key'], ref_key=r['ref_key'], n_ref=r['ref_n_ref'], n_out_of_window=r['ref_n_out_of_window'], source=r['source']) for r in I.to_dict('records')]
    for r in RC[RC.method != 'CLIP'].to_dict('records'):
        ri.append(dict(record_type='ref_input', batch='R7', set=r['set'], family=r['family'], baseline=r['baseline'], layout=r['layout'], block=r['block'], surface=r['surface'], phase=r['phase'],
                       image_key=r['image_key'], ref_key=r['ref_key'], n_ref=r['ref_n_ref'], n_out_of_window=r['ref_n_out_of_window']))
    RI = pd.DataFrame(ri)
    SUM = pd.concat([_summ_s12(L12), _summ_r7(L7)], ignore_index=True)
    SH = share_s1(I)
    out_df = pd.concat([REF, RI, L12, L7, SUM, SH], ignore_index=True, sort=False)
    out_df = out_df[['record_type'] + [c for c in out_df.columns if c != 'record_type']]
    H.part('B_results', out_df)
    dd = dict(S12=dict(source_inputs=int((RI.batch == 'S12').sum()), views_sigma0=int(((L12.sigma == 0) & (L12.domain == 'glob')).sum()), unique_references=int(I.ref_key.nunique()),
                       by_group=RI[RI.batch == 'S12'].groupby(['section', 'family']).agg(inputs=('input_key', 'size'), refs=('ref_key', 'nunique')).reset_index().to_dict('records'),
                       views_all_sigma=int((L12.domain == 'glob').sum())),
              R7=dict(source_inputs=int((RI.batch == 'R7').sum()), unique_references=int(RC.ref_key.nunique()), conditions_all_sigma=int(C7[C7.domain == 'glob'].image_key.nunique()),
                      method_results_all_sigma=int((L7.domain == 'glob').sum())),
              total_source_inputs=int(len(RI)), total_unique_references=int(I.ref_key.nunique() + RC.ref_key.nunique()), actual_above_reference_glob_records=dict(
                  S12=int(((L12.gap_C_tau > 0) & (L12.domain == 'glob')).sum()), R7=int(((L7.gap_C_tau > 0) & (L7.domain == 'glob')).sum())),
              splits_closed=True, seconds=round(time.time() - t0, 1), t_local=H.now_local())
    H.run_put('B_link', dd); H.flush_reads('B_link')
    return dd


def _summ_s12(L: pd.DataFrame) -> pd.DataFrame:
    keys = ['section', 'family', 'layout', 'size_mm', 'surface', 'sigma', 'domain']
    g1 = L.groupby(keys + ['phase'], dropna=False).agg(act=('act_C_tau', 'mean'), ref=('ref_C_tau', 'first'), refv=('ref_C_valid', 'first'), refi=('ref_C_int', 'first'),
                                                      actv=('act_C_valid', 'mean'), acti=('act_C_int', 'mean'), no_out=('state', lambda s: int((s == 'no_output').sum())), n=('view_id', 'size')).reset_index()
    rows = []
    for k, g in g1.groupby(keys, dropna=False):
        gap = g.act - g.ref
        rows.append(dict(record_type='summary', batch='S12', **dict(zip(keys, k)), n_phases=int(len(g)), n_views=int(g.n.sum()), n_no_output=int(g.no_out.sum()),
                         act_C_tau_mean=float(g.act.mean()), act_C_tau_min=float(g.act.min()), act_C_tau_max=float(g.act.max()), ref_C_tau_mean=float(g.ref.mean()), ref_C_tau_min=float(g.ref.min()),
                         ref_C_tau_max=float(g.ref.max()), gap_C_tau_mean=float(gap.mean()), gap_C_tau_min=float(gap.min()), gap_C_tau_max=float(gap.max()), n_phase_act_above_ref=int((gap > TOL).sum()),
                         ref_C_valid_mean=float(g.refv.mean()), ref_C_int_mean=float(g.refi.mean()), act_C_valid_mean=float(g.actv.mean()), act_C_int_mean=float(g.acti.mean())))
    return pd.DataFrame(rows)


def _summ_r7(L: pd.DataFrame) -> pd.DataFrame:
    V = pd.read_csv(H.reg(H.R7 / '_parts' / 'views.csv'), low_memory=False, keep_default_na=False)
    rows = []
    idx = L.set_index(['baseline', 'layout', 'phase', 'surface', 'sigma', 'method', 'domain'])
    for (base, st), vv in V[V.status == 'selected'].groupby(['baseline', 'arm']):
        fam = 'point' if base == 'R8x4' else 'line'
        for setn, phs in (('main', ('R0', 'R1')), ('anchor', ('H0',))):
            w = vv[vv.phase.isin(phs)]
            if not len(w):
                continue
            for s in H.MAIN:
                for sg in (0.0, 40.0):
                    for m in (['D0'] if fam == 'point' else ['P4TABLE', 'CLIP']):
                        for dn in DOMS:
                            per = []
                            for v in w.itertuples():
                                a = idx.loc[(base, v.layout, v.phase, s, sg, m, dn)]; b = idx.loc[(base, 'regular', v.phase, s, sg, m, dn)]
                                per.append(dict(block=v.block, phase=v.phase, d_act=float(a.act_C_tau - b.act_C_tau), d_ref=float(a.ref_C_tau - b.ref_C_tau),
                                                gap=float(a.act_C_tau - a.ref_C_tau), gap0=float(b.act_C_tau - b.ref_C_tau)))
                            P = pd.DataFrame(per).groupby('block')[['d_act', 'd_ref', 'gap', 'gap0']].mean()
                            rows.append(dict(record_type='summary', batch='R7', set=setn, family=fam, baseline=base, arm=st, surface=s, sigma=sg, method=m, domain=dn, n_blocks=int(len(P)),
                                             **{f'{c}_{f}': float(getattr(P[c], f)()) for c in ('d_act', 'd_ref', 'gap') for f in ('median', 'min', 'max')},
                                             gap_S0_median=float(P.gap0.median()),
                                             **{f'{c}_n_pos': int((P[c] > TOL).sum()) for c in ('d_act', 'd_ref')}, **{f'{c}_n_neg': int((P[c] < -TOL).sum()) for c in ('d_act', 'd_ref')},
                                             blocks_d_act=';'.join(f'{b}:{100 * x:.3f}' for b, x in P.d_act.items()), blocks_d_ref=';'.join(f'{b}:{100 * x:.3f}' for b, x in P.d_ref.items())))
    return pd.DataFrame(rows)


def share_s1(I: pd.DataFrame) -> pd.DataFrame:

    rows = []
    s1 = I[I.sections.str.contains('S1')]
    for (fam, sid, ph), g in s1.groupby(['family', 'platform_surface', 'phase_label']):
        g = g.sort_values('size_mm'); keys = list(g.ref_key)
        rec = dict(record_type='share', batch='S12', section='S1', family=fam, platform_surface=sid, phase=ph, sizes=';'.join(f'{x:g}' for x in g.size_mm),
                   n_sizes=int(len(g)), n_unique_refs=int(len(set(keys))), groups=';'.join(f'{s:g}:{k[-6:]}' for s, k in zip(g.size_mm, keys)))
        if fam == 'line':
            arrs = {}
            for r in g.itertuples():
                with np.load((H.ROOT / r.eval_json).with_suffix('.npz')) as A:
                    arrs[r.size_mm] = {(int(k), int(v)): xyz for k, v, xyz in zip(A['ref_k'], A['ref_v'], np.asarray(A['ref_xyz'], np.float64))}
            base = arrs[1.0] if 1.0 in arrs else next(iter(arrs.values()))
            for sz, a in arrs.items():
                com = set(a) & set(base)
                sh = [float(np.hypot(*(a[c][:2] - base[c][:2]))) for c in com]
                rec[f'w{sz:g}_rows'] = int(len(a)); rec[f'w{sz:g}_common_rows_vs_w1'] = int(len(com)); rec[f'w{sz:g}_max_xy_shift_vs_w1_mm'] = (max(sh) if sh else None)
                rec[f'w{sz:g}_median_xy_shift_vs_w1_mm'] = (float(np.median(sh)) if sh else None)
        rows.append(rec)
    return pd.DataFrame(rows)


def _indep(P: np.ndarray, D: dict, d_list=(5.0, 6.0, 7.0), tau=H.TAU) -> dict:

    from scipy.spatial import Delaunay
    import s4r6_analysis as S6
    x, y, z, area = (np.asarray(D[k], np.float64) for k in ('x', 'y', 'z_ref', 'area'))
    Q = P[np.isfinite(P).all(1)]
    _, first = np.unique(np.round(Q[:, :2] / 1e-9).astype(np.int64), axis=0, return_index=True); Q = Q[np.sort(first)]
    tri = Delaunay(Q[:, :2]); q = np.column_stack([x, y]); s = tri.find_simplex(q); ins = s >= 0
    rho = S6._rho_closed_form(Q[tri.simplices][:, :, :2])
    zest = np.full(len(q), np.nan)
    for i in np.flatnonzero(ins):
        a, b, c = Q[tri.simplices[s[i]]]
        M = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]]); l1, l2 = np.linalg.solve(M, q[i] - a[:2])
        zest[i] = a[2] + l1 * (b[2] - a[2]) + l2 * (c[2] - a[2])
    e = zest - z; it = ins & np.isfinite(e); A = area.sum()
    out = dict(n_used=int(len(Q)), n_int=int(it.sum()), C_int=float(area[it].sum() / A), int_rmse=float(np.sqrt(np.mean(e[it] ** 2))) if it.any() else None)
    for d in d_list:
        sp = it.copy(); sp[ins] &= rho[s[ins]] <= d
        out.update({f'n_sup_d{d:g}': int(sp.sum()), f'C_valid_d{d:g}': float(area[sp].sum() / A), f'C_tau_d{d:g}': float(area[sp & (np.abs(e) <= tau)].sum() / A),
                    f'sup_rmse_d{d:g}': (float(np.sqrt(np.mean(e[sp] ** 2))) if sp.any() else None), f'sup_p95_d{d:g}': (float(np.percentile(np.abs(e[sp]), 95)) if sp.any() else None)})
    return out


PICKS = [
    ('S1 dot (shared by d 1 / 2 / 3 mm), B', 'S12', dict(sections='S1', family='point', platform_surface='B_curv')),
    ('S1 line w 0.5 mm, M10', 'S12', dict(sections='S1', family='line', platform_surface='A_curv', size_mm=0.5)),
    ('S2 T8x4, M24', 'S12', dict(sections='S2', family='point', platform_surface='C_curv', layout='T8x4')),
    ('S2 R12x4, M10 (sparse, rho near d)', 'S12', dict(sections='S2', family='point', platform_surface='A_curv', layout='R12x4')),
    ('S2 line 12 mm 0 deg, B (sparse)', 'S12', dict(sections='S2', family='line', platform_surface='B_curv', layout='s12_phi0')),
    ('S2 R8x8, M24 (z-dependent support)', 'S12', dict(sections='S2', family='point', platform_surface='C_curv', layout='R8x8')),
    ('S2 line 4.45 mm 45 deg, M10 (dense)', 'S12', dict(sections='S2', family='line', platform_surface='A_curv', layout='s4.45_phi45')),
    ('R7 dot S1 B024 R1, M10 (perturbed, F4 example)', 'R7', dict(image_key='R2I-0bab6034e6ca0bb58693a12d')),
    ('R7 line perturbed j0.30_B025_c0 H0, M24 (band, anchor)', 'R7', dict(family='line', surface='M24_curv', layout='j0.30_B025_c0', phase='H0')),
    ('R7 line S0 R0, M24 (new phase, no band)', 'R7', dict(family='line', surface='M24_curv', layout='regular', phase='R0')),
]


def selfcheck() -> dict:
    import tc_ref as R
    from pathlib import Path
    I = pd.read_csv(H.PARTS / 'S12_inputs.csv', low_memory=False, float_precision='round_trip'); RC = pd.read_csv(H.PARTS / 'R7_conditions.csv', low_memory=False, float_precision='round_trip')
    RF = {b: pd.read_csv(H.PARTS / f'{b}_refs.csv', low_memory=False, float_precision='round_trip').set_index('ref_key') for b in ('S12', 'R7')}
    out = []
    for lab, batch, sel in PICKS:
        T = I if batch == 'S12' else RC[RC.method != 'CLIP']
        m = np.ones(len(T), bool)
        for k, v in sel.items():
            m &= (T[k].astype(str).str.contains(v) if k == 'sections' else (T[k] == v)).values
        r = T[m].sort_values('input_key' if batch == 'S12' else 'image_key').iloc[0]
        sid = r['platform_surface'] if batch == 'S12' else H.PLATFORM[r['surface']]
        zp = (H.ROOT / r['eval_json']).with_suffix('.npz') if batch == 'S12' else Path(R.H.R7 / 'evals' / f"{r['eval_key']}.npz")
        P, _ = R.ref_from_eval(zp, r['family'], sid)
        if R.ref_key(batch, sid, P) != r['ref_key']:
            raise H.Stop('check', f'{lab}: reference key not reproduced')
        ref = RF[batch].loc[r['ref_key']]
        rec = dict(record_type='selfcheck', case=lab, batch=batch, ref_key=r['ref_key'], n_points=int(len(P)))
        for dn in DOMS:
            ind = _indep(P, R.domain(batch, sid, dn))
            for k in ('C_int', 'C_valid_d5', 'C_valid_d6', 'C_valid_d7', 'C_tau_d5', 'C_tau_d6', 'C_tau_d7'):
                rec[f'{dn}_{k}_diff'] = abs(ind[k] - float(ref[f'{dn}_{k}']))
            for k in ('int_rmse', 'sup_rmse_d6', 'sup_p95_d6'):
                a, b = ind[k], ref[f'{dn}_{k}']
                rec[f'{dn}_{k}_diff'] = (abs(a - float(b)) if (a is not None and pd.notna(b)) else (0.0 if (a is None and pd.isna(b)) else 1.0))
            rec[f'{dn}_n_sup_d6_equal'] = bool(ind['n_sup_d6'] == int(ref[f'{dn}_n_sup_d6']))
            rec[f'{dn}_C_tau_d6'] = float(ref[f'{dn}_C_tau_d6'])
        out.append(rec)
    D = pd.DataFrame(out)
    cdiff = float(D[[c for c in D.columns if c.endswith('_diff') and ('_C_' in c)]].max().max())
    rdiff = float(D[[c for c in D.columns if c.endswith('_diff') and ('rmse' in c or 'p95' in c)]].max().max())
    res = dict(n_cases=len(D), max_C_diff=cdiff, max_rmse_p95_diff_mm=rdiff, counts_equal=bool(D[[c for c in D.columns if c.endswith('_equal')]].all().all()),
               ok=bool(cdiff <= 1e-9 and rdiff <= 1e-6 and D[[c for c in D.columns if c.endswith('_equal')]].all().all()),
               note='Stored-array check using Delaunay triangulation, barycentric interpolation and closed-form support radius.', t_local=H.now_local())
    H.part('selfcheck', D); H.run_put('B_selfcheck', dict(res, cases=D.to_dict('records')))
    return res
