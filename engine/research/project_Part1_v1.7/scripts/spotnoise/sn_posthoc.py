# Analyze stored dot detection, recovery and tolerance results.

import sys, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sn_lib as L

A_IN = L.OUT / 'analysis'
P_OUT = A_IN / 'posthoc'
GATES_XY = (1.0, 1.5, 2.0)
GATES_Z = (0.5, 1.0, 2.0)
BASE = (1.5, 1.0)
RECOVERY_MIN = 0.9
CRIT = dict(occ_dent_min=0.02, dent_abs_max_pct=10.0, gross_max=0.01, rmse_max_sigma0_mm=0.06, rmse_max_noisy_mm=0.15)
THR_RMSE, THR_DENT, THR_RATIO = 0.005, 1.0, 0.05
TOP = L.SIGMAS[-1]


def gname(gxy, gz):
    return f'{gxy:g}_{gz:g}'


def recovery(pts, amb, er, exp, gxy, gz):

    n_exp = len(exp)
    if n_exp == 0:
        return np.nan, np.nan, 0
    fin = np.isfinite(pts).all(axis=1)
    q = fin & ~amb & np.isfinite(er) & (np.abs(np.nan_to_num(er, nan=1e9)) <= gz)
    P = pts[q][:, :2].astype(np.float64)
    if len(P) == 0:
        return 0.0, 0.0, 0
    D = np.sqrt(((exp[:, None, :].astype(np.float64) - P[None, :, :]) ** 2).sum(axis=2))
    allowed = D <= gxy
    r, c = linear_sum_assignment(np.where(allowed, D, 1e6))
    o2o = int(allowed[r, c].sum()) / n_exp
    j = np.argmin(D, axis=1); dmin = D[np.arange(n_exp), j]
    nn = len(set(j[dmin <= gxy].tolist())) / n_exp
    return o2o, nn, int(len(P))


def per_case(df):
    rows, spacing, posck, exp_hash = [], {}, [], {}
    for row in df.itertuples():
        z = np.load(L.OUT / 'main' / 'cases' / (row.key + '.npz'))
        pts, amb, er = z['pts'], z['amb'].astype(bool), z['err_ref']
        exy, ind, hit = z['exp_xy'].astype(np.float64), z['exp_in_disc'].astype(bool), np.isfinite(z['exp_xy']).all(axis=1)
        exp = exy[ind]
        rec = dict(key=row.key, n_exp_disc_npz=int(len(exp)))
        for gxy in GATES_XY:
            for gz in GATES_Z:
                o, n, nq = recovery(pts, amb, er, exp, gxy, gz)
                rec[f'rec_o2o_{gname(gxy, gz)}'] = o; rec[f'rec_nn_{gname(gxy, gz)}'] = n
                if (gxy, gz) == BASE:
                    rec['n_qualified_base'] = nq
        rows.append(rec)
        sp = (row.surface, int(row.phase))
        h = hashlib.sha256(np.ascontiguousarray(exy[hit].round(6)).tobytes()).hexdigest()
        exp_hash.setdefault(sp, set()).add(h)
        if sp not in spacing:
            from scipy.spatial import cKDTree
            allp = exy[hit]; dsc = exp
            d_all, _ = cKDTree(allp).query(allp, k=2); d_dsc, _ = cKDTree(dsc).query(dsc, k=2)
            spacing[sp] = dict(surface=sp[0], phase=sp[1], n_expected_hit=int(hit.sum()), n_expected_disc=int(len(dsc)),
                               min_spacing_disc_mm=float(d_dsc[:, 1].min()), median_nn_disc_mm=float(np.median(d_dsc[:, 1])),
                               min_spacing_all_mm=float(d_all[:, 1].min()), p1_nn_all_mm=float(np.percentile(d_all[:, 1], 1)),
                               nominal_row_pitch_mm=float(L.LATTICES['S1'][1]))
        if row.sigma == 0:
            from scipy.spatial import cKDTree
            fin = np.isfinite(pts).all(axis=1)
            if fin.any() and len(exp):
                d, _ = cKDTree(pts[fin][:, :2].astype(np.float64)).query(exp)
                posck.append(pd.DataFrame(dict(key=row.key, surface=row.surface, bright=row.bright, r_mm=row.r_mm, phase=row.phase, dist_mm=d)))
    sp = pd.DataFrame(spacing.values()).sort_values(['surface', 'phase'])
    for g in GATES_XY:
        sp[f'disjoint_gates_{g:g}mm'] = sp.min_spacing_disc_mm > 2 * g
    sp['expected_xy_identical_within_surface_phase'] = [len(exp_hash[(s, p)]) == 1 for s, p in zip(sp.surface, sp.phase)]
    return pd.DataFrame(rows), sp, pd.concat(posck, ignore_index=True)


def layers(df):
    lim = np.where(np.isclose(df.sigma, 0), CRIT['rmse_max_sigma0_mm'], CRIT['rmse_max_noisy_mm'])
    out = pd.DataFrame(dict(key=df.key))
    out['L1_status'] = df.status
    out['L1_status_ok'] = df.status.eq('ok')
    out['L2_gross_le_1pct'] = df.gross_rate_rel.le(CRIT['gross_max']).fillna(False)
    out['L2_recovery_ge_0p9'] = df[f'rec_o2o_{gname(*BASE)}'].ge(RECOVERY_MIN).fillna(False)
    out['L2_geometric'] = out.L2_gross_le_1pct & out.L2_recovery_ge_0p9
    c = dict(status_ok=df.status.eq('ok'), occ_dent=df.occ_dent_rel.ge(CRIT['occ_dent_min']).fillna(False),
             dent=df.abs_dent_rel_pct.le(CRIT['dent_abs_max_pct']).fillna(False), gross=out.L2_gross_le_1pct,
             rmse=pd.Series(df.ref_rmse_rel_mm.values <= lim, index=df.index).where(df.ref_rmse_rel_mm.notna(), False))
    for k, v in c.items():
        out[f'L3_sub_{k}'] = v.astype(bool)
    out['L3_research_criterion'] = np.logical_and.reduce([out[f'L3_sub_{k}'] for k in c])
    out['L3_rmse_limit_mm'] = lim
    out['P_output_ratio_ge_0p9'] = df.completeness.ge(0.9).fillna(False)
    out['P_dent_le_10pct'] = df.abs_dent_rel_pct.le(10.0).fillna(False)
    return out


CRITERIA = {'L1_status_ok': 'Algorithm status: all normal', 'L2_gross_le_1pct': 'Geometry: gross-error rate <= 1 %', 'L2_recovery_ge_0p9': 'Geometry: position recovery >= 0.9',
            'L2_geometric': 'Geometry: both criteria', 'L3_research_criterion': 'Research criterion: PM-015 combination, post hoc',
            'P_output_ratio_ge_0p9': 'Registered: output count ratio >= 0.9', 'P_dent_le_10pct': 'Registered: absolute dent error <= 10 %'}


def tolerance(df, lay):
    d = df[['key', 'surface', 'bright', 'r_mm', 'phase', 'sigma']].merge(lay, on='key')
    rows = []
    for (surf, b, r, ph), sub in d.groupby(['surface', 'bright', 'r_mm', 'phase']):
        for crit in CRITERIA:
            kn, passall = [], []
            for sg in L.SIGMAS:
                cs = sub[np.isclose(sub.sigma, sg)]
                n = int(len(cs)); k = int(cs[crit].sum())
                kn.append(f'{k}/{n}'); passall.append(n > 0 and k == n)
            if not passall[0]:
                bnd, label, first_fail = None, 'Baseline failed', 0.0
            else:
                i = 0
                while i + 1 < len(passall) and passall[i + 1]:
                    i += 1
                bnd = L.SIGMAS[i]; first_fail = L.SIGMAS[i + 1] if i + 1 < len(passall) else None
                label = f'≥{TOP:g} (right-censored)' if first_fail is None else ('0 (only the noiseless level passed)' if bnd == 0 else f'{bnd:g}')
            start = L.SIGMAS.index(first_fail) + 1 if first_fail is not None else len(L.SIGMAS)
            repass = [L.SIGMAS[j] for j in range(start, len(L.SIGMAS)) if passall[j]]
            mixed = [L.SIGMAS[j] for j in range(len(L.SIGMAS)) if kn[j].split('/')[0] not in ('0', kn[j].split('/')[1])]
            rows.append(dict(surface=surf, bright=b, r_mm=r, d_mm=2 * r, phase=ph, criterion=crit, criterion_zh=CRITERIA[crit],
                             **{f'k_n_sigma{int(s)}': x for s, x in zip(L.SIGMAS, kn)}, boundary_sigma=bnd, first_fail_sigma=first_fail,
                             label=label, repass_levels=' '.join(f'{x:g}' for x in repass), mixed_levels=' '.join(f'{x:g}' for x in mixed)))
    return pd.DataFrame(rows)


def compact(t):

    def s(row):
        if row.label == 'Baseline failed':
            v = '×'
        elif row.first_fail_sigma is None or (isinstance(row.first_fail_sigma, float) and np.isnan(row.first_fail_sigma)):
            v = f'≥{TOP:g}'
        else:
            v = f'{row.boundary_sigma:g}'
        return v + ('*' if row.repass_levels else '')
    t = t.copy(); t['cell'] = t.apply(s, axis=1)
    piv = t.pivot_table(index=['surface', 'bright', 'r_mm'], columns='criterion', values='cell', aggfunc=lambda x: ' / '.join(x))
    return piv[list(CRITERIA)].reset_index()


def noise_free_size(df):
    z = df[np.isclose(df.sigma, 0)]
    rows = []
    for (surf, b, ph), sub in z.groupby(['surface', 'bright', 'phase']):
        sub = sub.sort_values('r_mm'); rec = dict(surface=surf, bright=b, phase=ph)
        for m, thr in (('ref_rmse_rel_mm', THR_RMSE), ('abs_dent_rel_pct', THR_DENT), ('dent_rel_pct', None), ('completeness', THR_RATIO),
                       (f'rec_o2o_{gname(*BASE)}', THR_RATIO)):
            v = sub[m].values.astype(float)
            rng = float(np.nanmax(v) - np.nanmin(v)); end = float(v[-1] - v[0])
            rec[f'{m}_range'] = rng; rec[f'{m}_end_r15_minus_r05'] = end
            rec[f'{m}_argmax_r'] = float(sub.r_mm.values[int(np.nanargmax(v))]); rec[f'{m}_argmin_r'] = float(sub.r_mm.values[int(np.nanargmin(v))])
            rec[f'{m}_signs'] = ''.join('+' if x > 0 else ('-' if x < 0 else '0') for x in np.diff(v))
            if thr is not None:
                rec[f'{m}_range_gt_thr'] = rng > thr; rec[f'{m}_end_gt_thr'] = abs(end) > thr
        rows.append(rec)
    return pd.DataFrame(rows)


def classify(vals, thr):
    if len(vals) < 6 or any(not np.isfinite(v) for v in vals.values()):
        return 'insufficient (undefined values)'
    sg = {k: (0 if abs(v) < thr else int(np.sign(v))) for k, v in vals.items()}
    nz = [s for s in sg.values() if s != 0]
    if not nz:
        return 'negligible'
    if all(s == nz[0] for s in nz):
        return 'stable' if len(nz) == 6 else 'consistent, partly negligible'
    within = all(len({sg[(surf, ph)] for ph in L.PHASES} - {0}) <= 1 for surf in L.SURFACES)
    return 'surface-dependent' if within else 'phase-dependent'


def recovery_effects(df):
    m = f'rec_o2o_{gname(*BASE)}'
    pm = df.groupby(['surface', 'bright', 'r_mm', 'sigma', 'phase'])[m].mean().reset_index()
    cells = pm.groupby(['surface', 'bright', 'r_mm', 'sigma'])[m].agg(['mean', 'min', 'max']).reset_index()
    cells.columns = ['surface', 'bright', 'r_mm', 'sigma', 'recovery_mean', 'recovery_min', 'recovery_max']
    size, bright = [], []
    for (b, sg), sub in pm.groupby(['bright', 'sigma']):
        vals = {}
        for surf in L.SURFACES:
            for ph in L.PHASES:
                q = sub[(sub.surface == surf) & (sub.phase == ph)].set_index('r_mm')[m]
                vals[(surf, ph)] = float(q.get(L.R_MAX, np.nan) - q.get(L.R_MIN, np.nan))
        arr = np.array([v for v in vals.values() if np.isfinite(v)])
        size.append(dict(bright=b, sigma=sg, delta_mean=arr.mean() if len(arr) else np.nan, delta_min=arr.min() if len(arr) else np.nan,
                         delta_max=arr.max() if len(arr) else np.nan, **{f'delta_{s}': np.nanmean([vals[(s, p)] for p in L.PHASES]) for s in L.SURFACES},
                         stability_posthoc=classify(vals, THR_RATIO)))
    d = df[['surface', 'phase', 'r_mm', 'sigma', 'seed', 'bright', m]]
    j = d[d.bright == 'A'].drop(columns='bright').merge(d[d.bright == 'B'].drop(columns='bright'), on=['surface', 'phase', 'r_mm', 'sigma', 'seed'], suffixes=('_A', '_B'))
    j['d'] = j[m + '_B'] - j[m + '_A']
    pmd = j.groupby(['surface', 'phase', 'r_mm', 'sigma']).d.mean().reset_index()
    for (r, sg), sub in pmd.groupby(['r_mm', 'sigma']):
        vals = {(row.surface, row.phase): float(row.d) for row in sub.itertuples()}
        arr = np.array(list(vals.values()))
        bright.append(dict(r_mm=r, sigma=sg, delta_B_minus_A_mean=arr.mean(), delta_min=arr.min(), delta_max=arr.max(),
                           **{f'delta_{s}': np.nanmean([vals.get((s, p), np.nan) for p in L.PHASES]) for s in L.SURFACES},
                           stability_posthoc=classify(vals, THR_RATIO)))
    return cells, pd.DataFrame(size), pd.DataFrame(bright)


def sensitivity(df):
    base = f'rec_o2o_{gname(*BASE)}'
    cells = df.groupby(['surface', 'bright', 'r_mm', 'sigma'])[[f'rec_o2o_{gname(a, b)}' for a in GATES_XY for b in GATES_Z]].mean()
    rows = []
    for a in GATES_XY:
        for b in GATES_Z:
            c = f'rec_o2o_{gname(a, b)}'; nn = f'rec_nn_{gname(a, b)}'
            dev = (cells[c] - cells[base]).abs(); sound = cells[base] >= RECOVERY_MIN
            diff = (df[c] - df[nn]).abs()
            rows.append(dict(gate_xy_mm=a, gate_z_mm=b, max_abs_dev_cells=float(dev.max()), max_abs_dev_sound_cells=float(dev[sound].max()),
                             n_cells_dev_gt_0p05=int((dev > 0.05).sum()), n_cases_zero_qualified_or_zero=int((df[c] == 0).sum()),
                             n_cases_undefined=int(df[c].isna().sum()), max_abs_o2o_minus_nn=float(diff.max()), n_cases_o2o_ne_nn=int((diff > 1e-12).sum()),
                             n_cases_nn_nan_with_exp=int((df[nn].isna() & (df.n_exp_disc_npz > 0)).sum())))
    return pd.DataFrame(rows)


def to_md(t, path):
    lines = ['# SpotNoise tolerance summary (post hoc)', '',
             'Cells show passed / total cases (one per phase at sigma 0; three at sigma > 0). The continuous boundary is the highest level with all preceding levels passing. '
             'Baseline failed means failure at sigma 0; >=100 (right-censored) means the top level passed. Re-entry lists passing levels after the first failure. '
             'The PM-015 criterion is applied post hoc: RMSE limits are 0.06 mm at sigma 0 and 0.15 mm at sigma > 0; these are study thresholds.', '']
    for crit, zh in CRITERIA.items():
        lines += [f'## {zh} (`{crit}`)', '', '| Surface | Brightness | r [mm] | Phase | sigma0 | sigma20 | sigma40 | sigma60 | sigma80 | sigma100 | Continuous boundary | Re-entry |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for row in t[t.criterion == crit].sort_values(['surface', 'bright', 'r_mm', 'phase']).itertuples():
            ks = [getattr(row, f'k_n_sigma{int(s)}') for s in L.SIGMAS]
            lines.append(f'| {row.surface} | {row.bright} | {row.r_mm:g} | {row.phase} | ' + ' | '.join(ks) + f' | {row.label} | {row.repass_levels or "—"} |')
        lines.append('')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    P_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(A_IN / 'cases.csv')
    rec, spacing, posck = per_case(df)
    df = df.merge(rec, on='key')
    lay = layers(df)
    tol = tolerance(df, lay)
    comp = compact(tol)
    nfs = noise_free_size(df)
    cells_rec, size_rec, bright_rec = recovery_effects(df)
    sens = sensitivity(df)
    pos = posck.groupby(['surface', 'bright']).dist_mm.agg(['count', 'median', lambda x: float(np.percentile(x, 95)), 'max']).reset_index()
    pos.columns = ['surface', 'bright', 'n_dots', 'median_mm', 'p95_mm', 'max_mm']
    rec.to_csv(P_OUT / 'geometric_recovery.csv', index=False)
    spacing.to_csv(P_OUT / 'expected_spacing.csv', index=False)
    lay.to_csv(P_OUT / 'layers.csv', index=False)
    tol.to_csv(P_OUT / 'tolerance_full.csv', index=False)
    comp.to_csv(P_OUT / 'tolerance_compact.csv', index=False)
    to_md(tol, P_OUT / 'tolerance_full.md')
    nfs.to_csv(P_OUT / 'noise_free_size.csv', index=False)
    cells_rec.to_csv(P_OUT / 'cells_recovery.csv', index=False)
    size_rec.to_csv(P_OUT / 'size_effect_recovery.csv', index=False)
    bright_rec.to_csv(P_OUT / 'brightness_effect_recovery.csv', index=False)
    sens.to_csv(P_OUT / 'gate_sensitivity.csv', index=False)
    pos.to_csv(P_OUT / 'expected_position_check_sigma0.csv', index=False)
    zero_q = int(((rec.n_exp_disc_npz > 0) & (rec.n_qualified_base == 0)).sum())
    summ = dict(generated=L.now(), kind='post hoc (not pre-registered)', n_cases=int(len(rec)), n_exp_disc_range=[int(rec.n_exp_disc_npz.min()), int(rec.n_exp_disc_npz.max())], n_exp_disc_npz_equals_cases_csv=bool((df.n_exp_disc_npz == df.n_exp_disc).all()),
                n_cases_undefined_recovery=int(rec[f'rec_o2o_{gname(*BASE)}'].isna().sum()), n_cases_exp_but_no_qualified_output_base=zero_q,
                gate_sensitivity=sens.to_dict('records'), expected_spacing=spacing.to_dict('records'), expected_position_check_sigma0=pos.to_dict('records'),
                research_criterion=dict(source='PM-015 frozen combined criterion (research convention, not a validated tyre requirement)', borrowed='post hoc for this batch', **CRIT,
                                        note='RMSE limits 0.06 mm at sigma 0 and 0.15 mm at sigma > 0 are research conventions, not validated tyre requirements'),
                noise_free_rmse_range_gt_0p005=int(nfs['ref_rmse_rel_mm_range_gt_thr'].sum()), noise_free_rmse_end_gt_0p005=int(nfs['ref_rmse_rel_mm_end_gt_thr'].sum()),
                noise_free_groups=int(len(nfs)))
    L.save(P_OUT / 'posthoc_summary.json', summ)
    print(json.dumps({k: summ[k] for k in ('n_cases', 'n_exp_disc_range', 'n_cases_undefined_recovery', 'n_cases_exp_but_no_qualified_output_base',
                                           'noise_free_rmse_range_gt_0p005', 'noise_free_rmse_end_gt_0p005', 'noise_free_groups')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
