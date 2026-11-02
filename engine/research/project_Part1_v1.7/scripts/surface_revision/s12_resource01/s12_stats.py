# Compute paired size-spacing effects and aggregate phase results.

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_common as K

COV = ['glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_cov_interp', 'loc_cov_interp', 'glob_recovery', 'loc_recovery', 'img_recovery']
ERR = ['glob_pt_rmse', 'glob_pt_mae', 'glob_pt_p95', 'glob_pt_bias', 'glob_gross_rate', 'glob_surf_rmse', 'glob_surf_mae', 'glob_surf_p95', 'loc_pt_rmse', 'loc_pt_mae', 'loc_pt_p95',
       'loc_gross_rate', 'loc_surf_rmse', 'loc_surf_p95', 'wrong_share', 'unconf_share', 'mismatch_rate']
CNT = ['n_valid', 'n_accepted', 'glob_n_valid', 'loc_n_valid', 'end_grey_0', 'end_grey_255']
PHASE_LIST = ['glob_Ctau', 'loc_Ctau', 'glob_recovery', 'glob_pt_rmse', 'glob_pt_p95', 'glob_surf_rmse', 'wrong_share', 'unconf_share']
CMETRICS = ['glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_recovery', 'glob_pt_rmse', 'glob_pt_p95', 'glob_surf_rmse', 'wrong_share', 'unconf_share']
KEYS = ['section', 'surface', 'surface_class', 'family', 'regime', 'size_mm', 'layout', 'sigma']
S2_P_PAR = ['R6x4', 'R7.35x4', 'R8x4', 'R10x4', 'R12x4']
S2_P_PERP = ['R8x2.5', 'R8x3', 'R8x4', 'R8x6', 'R8x8']


def _one(r: dict) -> dict:
    import s12_reuse as RU
    from types import SimpleNamespace
    ns = SimpleNamespace(**r)
    ep = RU.eval_record_path(ns); ip = RU.image_record_path(ns)
    rec = K.load_json(ep); m = rec['metrics']; info = rec.get('info') or {}
    out = dict(input_key=r['input_key'], eval_record=K.rel(ep), image_record=K.rel(ip))
    for k, v in m.items():
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
    for blk, pre in (('component_reasons', 'comp_'), ('selection_codes', 'sel_')):
        for k, v in (info.get(blk) or {}).items():
            out[pre + k] = v
    det = info.get('det_info') or info.get('detect_info') or {}
    for k in ('otsu_threshold', 'n_filtered_noise', 'n_border_excluded', 'n_area_rejected', 'n_elong_rejected', 'blob_area_median_px', 'steger_threshold', 'noise_sigma_hat', 'n_ridge_px',
              'n_ridge_components', 'n_vlines'):
        if k in det and not isinstance(det[k], (dict, list)):
            out['det_' + k] = det[k]
    ir = K.load_json(ip)
    eg = ir.get('end_grey')
    if eg is None:
        with np.load(ip.with_suffix('.npz')) as z:
            img = np.asarray(z['image'])
        eg = dict(frac_0=float((img == 0).mean()), frac_255=float((img == 255).mean()))
    out['end_grey_0'] = eg['frac_0']; out['end_grey_255'] = eg['frac_255']
    return out


def inputs_metrics(U: pd.DataFrame, workers=8) -> pd.DataFrame:
    rows = U.to_dict('records')
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_one, rows, chunksize=64))
    M = pd.DataFrame(out)
    K.save_csv(M, K.RESULTS / 'inputs_metrics.csv')
    return M


def views(V: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in M.columns if c not in V.columns or c == 'input_key']
    X = V.merge(M[keep], on='input_key', how='left', validate='many_to_one')
    no = X.state == 'no_output'
    for c in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_cov_interp', 'loc_cov_interp', 'glob_recovery', 'loc_recovery', 'img_recovery'):
        X.loc[no, c] = 0.0
    den = pd.to_numeric(X.id_denominator, errors='coerce')
    X['wrong_share'] = np.where(den > 0, X.id_wrong / den.where(den > 0), np.nan)
    X['unconf_share'] = np.where(den > 0, X.id_unconf / den.where(den > 0), np.nan)
    X['zero_correct_recovery'] = (pd.to_numeric(X.glob_n_recovered, errors='coerce').fillna(0) == 0)
    for c in COV + ERR + CNT:
        if c in X:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    return X


def phase_table(X: pd.DataFrame, metrics=None) -> pd.DataFrame:

    metrics = metrics or (COV + ERR + CNT)
    g = X.groupby(KEYS + ['phase_label'], dropna=False)
    P = g[[m for m in metrics if m in X]].agg(lambda s: s.mean() if s.notna().all() else np.nan).reset_index()
    P['n_reps'] = g.size().values
    P['inputs'] = g.input_key.agg(lambda s: ';'.join(s)).values
    return P


def conditions(X: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    out = []
    Pg = {k: g for k, g in P.groupby(KEYS, dropna=False)}
    for k, g in X.groupby(KEYS, dropna=False):
        ph = Pg[k].sort_values('phase_label')
        rec = dict(zip(KEYS, k))
        rec.update(n_views=int(len(g)), n_phases=int(g.phase_label.nunique()), reps=','.join(str(x) for x in sorted(g.rep.unique())), n_no_output=int((g.state == 'no_output').sum()),
                   n_zero_correct_recovery=int(g.zero_correct_recovery.sum()), n_unique_inputs=int(g.input_key.nunique()),
                   n_inputs_shared_other_regime=int(g.same_input_as_other_regime.sum()), n_inputs_shared_S1_S2=int(g.shared_S1_S2.sum()),
                   Q0_min=float(g.Q0.min()), Q0_max=float(g.Q0.max()), peak_f32_min=float(g.peak_f32.min()), peak_f32_max=float(g.peak_f32.max()), Q_actual_min=float(g.Q_actual.min()),
                   Q_actual_max=float(g.Q_actual.max()), budget_usage_min=(float(g.budget_usage.min()) if g.budget_usage.notna().any() else None),
                   budget_usage_max=(float(g.budget_usage.max()) if g.budget_usage.notna().any() else None), active_limit=';'.join(sorted(set(g.active_limit))),
                   p_parallel_mm=g.p_parallel_mm.iloc[0], p_perp_mm=g.p_perp_mm.iloc[0], line_pitch_mm=g.line_pitch_mm.iloc[0], orientation_deg=g.orientation_deg.iloc[0],
                   sum_id_correct=int(g.id_correct.fillna(0).sum()), sum_id_wrong=int(g.id_wrong.fillna(0).sum()), sum_id_unconf=int(g.id_unconf.fillna(0).sum()),
                   sum_id_denominator=int(g.id_denominator.fillna(0).sum()), sum_glob_n_expected=int(g.glob_n_expected.fillna(0).sum()), sum_glob_n_recovered=int(g.glob_n_recovered.fillna(0).sum()))
        for m in COV + ERR + CNT:
            if m not in ph:
                continue
            v = ph[m].to_numpy(float); d = v[~np.isnan(v)]
            rec[f'{m}_mean'] = float(d.mean()) if len(d) else None
            rec[f'{m}_n_phases_def'] = int(len(d))
            if m in PHASE_LIST or m in ('glob_Cvalid',):
                rec[f'{m}_median'] = float(np.median(d)) if len(d) else None
                rec[f'{m}_min'] = float(d.min()) if len(d) else None
                rec[f'{m}_max'] = float(d.max()) if len(d) else None
                rec[f'{m}_phases'] = json.dumps([None if np.isnan(x) else round(float(x), 6) for x in v])
            if m in ERR:
                cv = g[m].dropna()
                rec[f'{m}_case_mean_def'] = float(cv.mean()) if len(cv) else None
                rec[f'{m}_n_cases_def'] = int(len(cv))
        out.append(rec)
    return pd.DataFrame(out)


_IDX = {}


def boot_idx(n):
    if n not in _IDX:
        _IDX[n] = np.random.Generator(np.random.PCG64(K.BOOT_SEED)).integers(0, n, size=(K.BOOT_N, n))
    return _IDX[n]


def _pairs(C: pd.DataFrame) -> list[tuple]:

    out = []
    s1 = C[C.section == 'S1']
    for (surf, fam, reg, sig), g in s1.groupby(['surface', 'family', 'regime', 'sigma']):
        sizes = sorted(g.size_mm.unique())
        for a, b in zip(sizes[:-1], sizes[1:]):
            out.append(('S1_size_adjacent', dict(section='S1', surface=surf, family=fam, regime=reg, sigma=sig, size_mm=a), dict(size_mm=b)))
        out.append(('S1_size_min_max', dict(section='S1', surface=surf, family=fam, regime=reg, sigma=sig, size_mm=sizes[0]), dict(size_mm=sizes[-1])))
    for (surf, fam, size, sig), g in s1.groupby(['surface', 'family', 'size_mm', 'sigma']):
        regs = set(g.regime)
        if {'resource_cap_11695', 'fixed_peak_1'} <= regs:
            out.append(('S1_regime_cap_vs_fixed', dict(section='S1', surface=surf, family=fam, size_mm=size, sigma=sig, regime='fixed_peak_1'), dict(regime='resource_cap_11695')))
        if {'resource_cap_11695', 'equal_integral_5850'} <= regs:
            out.append(('S1_regime_cap_vs_5850', dict(section='S1', surface=surf, family=fam, size_mm=size, sigma=sig, regime='equal_integral_5850'), dict(regime='resource_cap_11695')))
    s2 = C[C.section == 'S2']
    for (surf, sig), g in s2.groupby(['surface', 'sigma']):
        for name, ser in (('S2_dots_p_parallel', S2_P_PAR), ('S2_dots_p_perp', S2_P_PERP)):
            for a, b in zip(ser[:-1], ser[1:]):
                out.append((name, dict(section='S2', surface=surf, family='point', sigma=sig, layout=a), dict(layout=b)))
        out.append(('S2_dots_T_vs_R', dict(section='S2', surface=surf, family='point', sigma=sig, layout='R8x4'), dict(layout='T8x4')))
        for phi in (0.0, 45.0):
            lays = [f's{p:g}_phi{phi:g}' for p in K.S2_PITCHES]
            for a, b in zip(lays[:-1], lays[1:]):
                out.append((f'S2_lines_pitch_phi{phi:g}', dict(section='S2', surface=surf, family='line', sigma=sig, layout=a), dict(layout=b)))
        for p in K.S2_PITCHES:
            out.append(('S2_lines_phi45_vs_0', dict(section='S2', surface=surf, family='line', sigma=sig, layout=f's{p:g}_phi0'), dict(layout=f's{p:g}_phi45')))
    for k, g in C.groupby(['section', 'surface', 'family', 'regime', 'size_mm', 'layout']):
        sig = sorted(g.sigma.unique())
        for a, b in zip(sig[:-1], sig[1:]):
            out.append(('sigma_step', dict(zip(['section', 'surface', 'family', 'regime', 'size_mm', 'layout'], k), sigma=a), dict(sigma=b)))
    return out


def contrasts(P: pd.DataFrame, C: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    Pi = P.set_index(KEYS + ['phase_label']).sort_index()
    inputs_of = X.groupby(KEYS + ['phase_label', 'rep']).input_key.first()
    rows = []
    for name, base, other in _pairs(C):
        sel_a = C.copy()
        for k, v in base.items():
            sel_a = sel_a[sel_a[k] == v]
        if len(sel_a) != 1:
            continue
        ka = tuple(sel_a.iloc[0][KEYS]); kb_d = dict(zip(KEYS, ka)); kb_d.update(other)
        sel_b = C.copy()
        for k, v in kb_d.items():
            sel_b = sel_b[sel_b[k] == v] if not (isinstance(v, float) and np.isnan(v)) else sel_b[sel_b[k].isna()]
        if len(sel_b) != 1:
            continue
        kb = tuple(sel_b.iloc[0][KEYS])
        A = Pi.loc[ka]; B = Pi.loc[kb]
        phases = sorted(set(A.index) & set(B.index))
        if len(phases) != len(A) or len(phases) != len(B):
            continue
        same = bool((inputs_of.loc[ka].values == inputs_of.loc[kb].values).all()) if len(inputs_of.loc[ka]) == len(inputs_of.loc[kb]) else False
        for m in CMETRICS:
            a = A.loc[phases, m].to_numpy(float); b = B.loc[phases, m].to_numpy(float); e = b - a
            rec = dict(contrast=name, **{f'a_{k}': v for k, v in zip(KEYS, ka)}, b_regime=kb[KEYS.index('regime')], b_size_mm=kb[KEYS.index('size_mm')], b_layout=kb[KEYS.index('layout')],
                       b_sigma=kb[KEYS.index('sigma')], metric=m, n_phases=len(phases), identical_inputs=same, direction='b - a')
            if np.isnan(e).any():
                rec.update(state='undefined', n_phases_def=int((~np.isnan(e)).sum()), reason='error metric undefined in a needed phase (no output / no point in the domain)')
            else:
                rec.update(state='ok', mean=float(e.mean()), median=float(np.median(e)), min=float(e.min()), max=float(e.max()), n_pos=int((e > 0).sum()), n_neg=int((e < 0).sum()),
                           n_zero=int((e == 0).sum()), per_phase=json.dumps([round(float(x), 7) for x in e]), a_mean=float(a.mean()), b_mean=float(b.mean()))
                if len(e) >= 8:
                    bs = e[boot_idx(len(e))].mean(axis=1); rec.update(res_lo_5=float(np.percentile(bs, 5)), res_hi_95=float(np.percentile(bs, 95)))
                else:
                    rec.update(interval_note='3 phases: min-max only, no resampling interval')
            rows.append(rec)
    return pd.DataFrame(rows)


def rankings(C: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (sec, surf, fam, reg), g in C.groupby(['section', 'surface', 'family', 'regime']):
        g = g.copy(); g['config'] = np.where(g.section == 'S1', g.size_mm.map(lambda x: f'{x:g}'), g.layout)
        base = None
        for sig, h in g.groupby('sigma'):
            h = h.sort_values(['glob_Ctau_mean', 'config'], ascending=[False, True]).reset_index(drop=True)
            h['rank'] = np.arange(1, len(h) + 1)
            if sig == 0.0:
                base = dict(zip(h.config, h['rank']))
            rho = None
            if base is not None and len(h) > 2:
                r0 = np.array([base[c] for c in h.config], float); r1 = h['rank'].to_numpy(float)
                rho = float(np.corrcoef(r0, r1)[0, 1])
            for r in h.itertuples():
                out.append(dict(section=sec, surface=surf, family=fam, regime=reg, sigma=sig, rank=r.rank, config=r.config, glob_Ctau_mean=r.glob_Ctau_mean, loc_Ctau_mean=r.loc_Ctau_mean,
                                glob_pt_p95_mean=r.glob_pt_p95_mean, glob_pt_p95_n_phases_def=r.glob_pt_p95_n_phases_def, wrong_share_case_mean=r.wrong_share_case_mean_def,
                                unconf_share_case_mean=r.unconf_share_case_mean_def, n_no_output=r.n_no_output, n_zero_correct_recovery=r.n_zero_correct_recovery, n_views=r.n_views,
                                rank_sigma0=(base or {}).get(r.config), spearman_vs_sigma0=rho))
    return pd.DataFrame(out)


def diagnostics(X: pd.DataFrame) -> dict:
    cols = ['view_id', 'input_key', 'section', 'surface', 'family', 'regime', 'size_mm', 'layout', 'phase_label', 'sigma', 'rep', 'peak_f32', 'state', 'n_components', 'n_candidates', 'n_selected',
            'n_matched', 'n_ambiguous', 'n_duplicate', 'n_detected', 'n_accepted', 'n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'id_denominator', 'glob_n_expected', 'glob_n_recovered',
            'glob_recovery', 'loc_recovery', 'glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'glob_gross_rate', 'glob_surf_rmse', 'loc_pt_rmse', 'loc_pt_p95',
            'loc_surf_rmse', 'end_grey_0', 'end_grey_255'] + sorted(c for c in X.columns if c.startswith(('comp_', 'sel_', 'n_status_', 'det_')))
    cols = [c for c in cols if c in X.columns]
    a = X[(X.section == 'S1') & (X.family == 'point') & (X.size_mm == 0.5) & (X.surface_class == 'main')][cols].sort_values(['surface', 'regime', 'sigma', 'phase_label', 'rep'])
    b = X[(X.family == 'line') & ((X.state == 'no_output') | ((X.section == 'S2') & X.layout.isin(['s4.45_phi0', 's5_phi0'])))][cols].sort_values(['section', 'surface', 'layout', 'sigma', 'phase_label', 'rep'])
    c = X[(X.section == 'S2') & (X.glob_pt_rmse > 1.0)][cols].sort_values(['surface', 'family', 'layout', 'sigma', 'phase_label', 'rep'])
    return dict(diag_M24_small_dots=a, diag_line_no_output=b, diag_S2_tails=c)


def run(workers=8, scope='all') -> dict:
    V = pd.read_csv(K.MANIFEST, keep_default_na=False, float_precision='round_trip')
    U = pd.read_csv(K.INPUTS, keep_default_na=False, float_precision='round_trip')
    if scope == 'main':
        V = V[V.surface_class == 'main']; U = U[U.surface_class == 'main']
    for c in ('p_parallel_mm', 'p_perp_mm', 'line_pitch_mm', 'orientation_deg', 'budget_usage', 'Q_budget', 'diameter_mm', 'phase_frac'):
        V[c] = pd.to_numeric(V[c], errors='coerce')
    M = inputs_metrics(U, workers)
    X = views(V, M); K.save_csv(X, K.RESULTS / 'views.csv')
    P = phase_table(X); K.save_csv(P, K.RESULTS / 'phase_values.csv')
    C = conditions(X, P); K.save_csv(C, K.RESULTS / 'conditions.csv')
    D = contrasts(P, C, X); K.save_csv(D, K.RESULTS / 'contrasts.csv')
    R = rankings(C); K.save_csv(R, K.RESULTS / 'rankings.csv')
    dg = diagnostics(X)
    for k, v in dg.items():
        K.save_csv(v, K.RESULTS / f'{k}.csv')
    res = dict(scope=scope, inputs=len(M), views=len(X), conditions=len(C), contrasts=len(D), rankings=len(R), states=X.drop_duplicates('input_key').state.value_counts().to_dict(),
               diagnostics={k: len(v) for k, v in dg.items()}, t_local=K.now_local())
    K.save_json(K.CHECKS / 'stats_run.json', res)
    return res


if __name__ == '__main__':
    print(json.dumps(K.clean(run(scope=(sys.argv[1] if len(sys.argv) > 1 else 'all'))), ensure_ascii=False, indent=1))
