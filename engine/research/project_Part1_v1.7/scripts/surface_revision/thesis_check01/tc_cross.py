# Compare identity, depth-window, resource and surface effects across studies.

from __future__ import annotations

import numpy as np
import pandas as pd

import tc_common as H


def _views() -> pd.DataFrame:
    return pd.read_csv(H.reg(H.S12 / 'results' / 'views.csv'), low_memory=False, float_precision='round_trip')


def identity() -> pd.DataFrame:
    V = _views(); v = V[(V.surface_class == 'main') & (V.regime == 'resource_cap_11695')].copy(); rows = []
    v['grp'] = np.where(v.section == 'S1', 'S1', 'S2')
    for (sec, fam, sg), g in v.groupby(['grp', 'family', 'sigma']):
        c, w, u = g.id_correct.sum(), g.id_wrong.sum(), g.id_unconf.sum(); n = c + w + u
        fw = g.id_wrong / (g.id_correct + g.id_wrong).replace(0, np.nan)
        rows.append(dict(record_type='identity', batch='S12', section=sec, family=fam, sigma=sg, n_views=len(g), n_no_output=int((g.state == 'no_output').sum()), c=int(c), w=int(w), u=int(u),
                         f_wrong=(w / (c + w) if c + w else None), u_share=(u / n if n else None), risk_low=(w / n if n else None), risk_high=((w + u) / n if n else None),
                         views_w_pos=int((g.id_wrong > 0).sum()), views_fwrong_above_0p1pct=int((fw > 0.001).sum()), max_w_per_view=int(g.id_wrong.max()),
                         median_u_share_views=float((g.id_unconf / g.id_denominator.replace(0, np.nan)).median()) if (g.id_denominator > 0).any() else None,
                         views_zero_correct=int(((g.id_correct == 0) & (g.state == 'ok')).sum()), median_accepted=float(g.id_denominator.median())))
    uq = v.drop_duplicates('input_key')
    for (fam, sg), g in list(uq.groupby(['family', 'sigma'])) + [((f, 'all'), uq[uq.family == f]) for f in ('point', 'line')] + [(('all', 'all'), uq)]:
        c, w, u = g.id_correct.sum(), g.id_wrong.sum(), g.id_unconf.sum(); n = c + w + u
        rows.append(dict(record_type='identity', batch='S12', section='S1+S2 unique inputs', family=fam, sigma=sg, n_views=len(g), c=int(c), w=int(w), u=int(u), f_wrong=(w / (c + w) if c + w else None),
                         u_share=(u / n if n else None), risk_low=(w / n if n else None), risk_high=((w + u) / n if n else None), views_w_pos=int((g.id_wrong > 0).sum()),
                         views_fwrong_above_0p1pct=int(((g.id_wrong / (g.id_correct + g.id_wrong).replace(0, np.nan)) > 0.001).sum()), max_w_per_view=int(g.id_wrong.max())))
    R7 = pd.read_csv(H.reg(H.R7 / 'results.csv'), low_memory=False, float_precision='round_trip'); C = R7[(R7.record_type == 'cond') & (R7.domain == 'glob')]
    for (st, fam, m, sg), g in C.groupby(['set', 'family', 'method', 'sigma']):
        c, w, u = g.c.sum(), g.w.sum(), g.u.sum(); n = c + w + u
        rows.append(dict(record_type='identity', batch='R7', set=st, family=fam, method=m, sigma=sg, n_views=len(g), c=int(c), w=int(w), u=int(u), f_wrong=(w / (c + w) if c + w else None),
                         u_share=(u / n if n else None), risk_low=(w / n if n else None), risk_high=((w + u) / n if n else None), views_w_pos=int((g.w > 0).sum()),
                         views_fwrong_above_0p1pct=int(((g.w / (g.c + g.w).replace(0, np.nan)) > 0.001).sum()), max_w_per_view=int(g.w.max()), domain='glob (Omega1)',
                         n_gross=int(g.n_gross.sum()), views_gross=int((g.n_gross > 0).sum())))
    return pd.DataFrame(rows)


def unconf_split() -> pd.DataFrame:

    V = _views(); v = V[(V.surface_class == 'main') & (V.regime == 'resource_cap_11695')].drop_duplicates('input_key')
    acc = {}
    S = H.s2(); mmpp = S.SL.mmpp()
    for r in v.itertuples():
        if not r.id_unconf:
            continue
        p = H.ROOT / r.eval_record.replace('.json', '.npz')
        with np.load(p) as z:
            un = z['point_unconf'].astype(bool) & z['accepted'].astype(bool)
            if r.family == 'point':
                pf = z['point_feature'].astype(np.int64); dmin = np.asarray(z['feat_truth_dist_px'], np.float64)[pf[un]]; rp = float(r.size_mm) / mmpp + 1.0
                cls = dict(nonfinite=int((~np.isfinite(dmin)).sum()), near_dot_but_projector_out=int((np.isfinite(dmin) & (dmin <= rp)).sum()),
                           beyond_r_plus_1px=int((np.isfinite(dmin) & (dmin > rp)).sum()))
                acc.setdefault(('dmin',) + ('S1' if r.section == 'S1' else 'S2', r.family, float(r.sigma)), []).extend(dmin[np.isfinite(dmin)].tolist())
            else:
                tk = np.asarray(z['truth_k'], np.int64)[un]
                cls = dict(no_registered_line=int((tk <= -999999).sum()), other=int((tk > -999999).sum()))
        key = ('S1' if r.section == 'S1' else 'S2', r.family, float(r.sigma))
        a = acc.setdefault(key, {}); [a.__setitem__(k, a.get(k, 0) + (vv or 0)) for k, vv in cls.items()]
        a['n_unconf'] = a.get('n_unconf', 0) + int(un.sum())
    rows = []
    for k, a in sorted((k, a) for k, a in acc.items() if k[0] != 'dmin'):
        dm = np.asarray(acc.get(('dmin',) + k, []), np.float64)
        rows.append(dict(record_type='unconf_split', batch='S12', section=k[0], family=k[1], sigma=k[2], **a,
                         **({f'dmin_px_p{q}': float(np.percentile(dm, q)) for q in (10, 50, 90)} if dm.size else {})))
    return pd.DataFrame(rows)


def tail() -> pd.DataFrame:
    V = _views(); v = V[(V.surface_class == 'main') & (V.regime == 'resource_cap_11695') & (V.layout == 's6.3_phi0')]
    pick = v[((v.surface == 'M10_curv') & (v.phase_label == 'ph1') & (v.sigma <= 40)) | ((v.phase_label == 'ph1') & (v.sigma == 0)) | ((v.surface == 'M10_curv') & (v.phase_label == 'ph0') & (v.sigma == 0))]
    pick = pick.drop_duplicates('input_key').sort_values(['surface', 'phase_label', 'sigma', 'rep']).head(8); rows = []
    for r in pick.itertuples():
        with np.load(H.reg(H.ROOT / r.eval_record.replace('.json', '.npz'))) as z:
            val = z['valid'].astype(bool); pts = z['pts']; e = pts[:, 2] - z['zr']; gross = val & (np.abs(np.nan_to_num(e)) > 1.0); un = z['point_unconf'].astype(bool) & val
            refk = np.unique(z['ref_k']); lid = z['line_id'][gross]; tk = z['truth_k'][gross]; ch = z['chain'][gross]
            rows.append(dict(record_type='tail_check', batch='S12', surface=r.surface, phase=r.phase_label, sigma=r.sigma, rep=r.rep, eval_record=r.eval_record, n_valid=int(val.sum()), n_unconf=int(un.sum()),
                             n_gross=int(gross.sum()), n_wrong=int((z['point_wrong'].astype(bool) & val).sum()), registered_ids=f'{refk.min()}..{refk.max()}', gross_line_ids=';'.join(map(str, sorted(set(lid.tolist())))),
                             gross_truth_k=';'.join(map(str, sorted(set(tk.tolist())))), gross_chains=';'.join(map(str, sorted(set(ch.tolist())))),
                             gross_abs_err_median_mm=(float(np.median(np.abs(e[gross]))) if gross.any() else None), gross_y_min=(float(pts[gross, 1].min()) if gross.any() else None),
                             gross_y_max=(float(pts[gross, 1].max()) if gross.any() else None), gross_in_local=int((z['in_local'].astype(bool) & gross).sum()), unconf_equals_gross=bool(np.array_equal(un, gross)),
                             glob_pt_rmse=r.glob_pt_rmse))
    return pd.DataFrame(rows)


def window() -> pd.DataFrame:
    import tc_ref as R
    S = H.s2(); rows = []
    for surf in H.MAIN:
        sid = H.PLATFORM[surf]; dl = S.roi_domain(R.mesh(sid))
        doms = [('global Omega0 (S1-S3, S12)', R.domain('S12', sid, 'glob')['z_ref']), ('global Omega1 (S4-REAL-05..07)', R.domain('R7', sid, 'glob')['z_ref']),
                ('local r <= 15 mm', np.asarray(dl['z_ref'], np.float64))]
        for name, z in doms:
            lo, hi = float(z.min()), float(z.max())
            rows.append(dict(record_type='window', surface=surf, domain=name, n_nodes=int(len(z)), z_min=lo, z_max=hi, span=hi - lo, window_lo=H.WINDOW[0], window_hi=H.WINDOW[1],
                             margin_below=lo - H.WINDOW[0], margin_above=H.WINDOW[1] - hi, window_share_spanned=(hi - lo) / (H.WINDOW[1] - H.WINDOW[0]),
                             nodes_inside_window=float(np.mean((z >= H.WINDOW[0]) & (z <= H.WINDOW[1])))))
        fz = dl['z_range_fov']
        rows.append(dict(record_type='window', surface=surf, domain='FOV square |x|,|y| <= FOV_HALF (grid truth)', z_min=float(fz[0]), z_max=float(fz[1]), span=float(fz[1] - fz[0]),
                         window_lo=H.WINDOW[0], window_hi=H.WINDOW[1], margin_below=float(fz[0]) - H.WINDOW[0], margin_above=H.WINDOW[1] - float(fz[1]),
                         window_share_spanned=float(fz[1] - fz[0]) / 20.0, nodes_inside_window=float(dl['window_cov_fov'])))
    D = pd.DataFrame(rows)
    allg = D[D.domain.str.startswith('global Omega0')]
    D = pd.concat([D, pd.DataFrame([dict(record_type='window', surface='all three', domain='global Omega0 union', z_min=float(allg.z_min.min()), z_max=float(allg.z_max.max()),
                                         span=float(allg.z_max.max() - allg.z_min.min()), window_lo=-16.0, window_hi=4.0, margin_below=float(allg.z_min.min()) + 16.0, margin_above=4.0 - float(allg.z_max.max()))])],
                  ignore_index=True)
    return D


def resource() -> pd.DataFrame:
    T = pd.read_csv(H.reg(H.S12 / 'results' / 'resource_table.csv'), low_memory=False, float_precision='round_trip'); rows = []
    for (sec, fam, lay, sz, reg), g in T.groupby(['section', 'family', 'layout', 'size_mm', 'regime'], dropna=False):
        rows.append(dict(record_type='resource', section=sec, family=fam, layout=lay, size_mm=sz, regime=reg, n_phases=len(g), Q0_mean=float(g.Q0.mean()), Q0_min=float(g.Q0.min()), Q0_max=float(g.Q0.max()),
                         peak_mean=float(g.peak_f32.mean()), peak_min=float(g.peak_f32.min()), peak_max=float(g.peak_f32.max()), Q_actual_min=float(g.Q_actual.min()), Q_actual_max=float(g.Q_actual.max()),
                         budget_usage_min=(float(g.budget_usage.min()) if g.budget_usage.notna().any() else None), budget_usage_max=(float(g.budget_usage.max()) if g.budget_usage.notna().any() else None),
                         active_limit=';'.join(sorted(set(g.active_limit))), **{f'snr_proxy_s{s}': float(255.0 * g.peak_f32.mean() / s) for s in (20, 40, 60, 80, 100)}))
    return pd.DataFrame(rows)


def regimes() -> pd.DataFrame:
    V = _views(); v = V[(V.surface_class == 'main') & (V.section == 'S1')].copy(); rows = []

    k = ['surface', 'family', 'size_mm', 'phase_label', 'sigma', 'rep']
    chk = v.groupby(k).agg(n_reg=('regime', 'nunique'), n_raster=('raster_key', 'nunique'), n_seed=('noise_seed', lambda s: s.fillna(-1).nunique())).reset_index()
    for (fam, sz, reg, sg), g in v.groupby(['family', 'size_mm', 'regime', 'sigma']):
        ph = g.groupby(['surface', 'phase_label']).glob_Ctau.mean().groupby('surface').mean()
        rows.append(dict(record_type='regime', section='S1', family=fam, size_mm=sz, regime=reg, sigma=sg, Ctau_glob_3surf_mean=float(ph.mean()), **{f'Ctau_glob_{H.SHORT[s]}': float(ph.get(s, np.nan)) for s in H.MAIN},
                         peak_mean=float(g.peak_f32.mean()), n_views=len(g)))
    D = pd.DataFrame(rows)
    D['same_raster_seed_across_regimes'] = bool((chk.n_raster == 1).all() and (chk.n_seed == 1).all())
    return D


def gauss() -> pd.DataFrame:
    V = _views(); v = V[(V.surface_class == 'supplement') & (V.regime == 'resource_cap_11695')]; rows = []
    for (sec, fam, lay, sz, sg), g in v.groupby(['section', 'family', 'layout', 'size_mm', 'sigma']):
        a = pd.to_numeric(g.amp_err_peak_mm, errors='coerce')
        rows.append(dict(record_type='gauss', section=sec, family=fam, layout=lay, size_mm=sz, sigma=sg, n_views=len(g), n_no_output=int((g.state == 'no_output').sum()),
                         amp_err_peak_median_mm=(float(a.median()) if a.notna().any() else None), amp_err_peak_min_mm=(float(a.min()) if a.notna().any() else None),
                         amp_err_peak_max_mm=(float(a.max()) if a.notna().any() else None), loc_Ctau_mean=float(g.loc_Ctau.fillna(0).mean()), glob_Ctau_mean=float(g.glob_Ctau.fillna(0).mean())))
    return pd.DataFrame(rows)


def surf02() -> pd.DataFrame:
    c = H.load_json(H.reg(H.SR / 'surf02' / 'checks' / 'surf02_checks.json')); rows = []
    t = c['c1_triangulation_consistency']; rows.append(dict(record_type='surf02', check='projection-triangulation self-consistency (known correspondence, 4 surfaces x 240 points)',
                                                            value=max(v['max_abs_mm'] for v in t['per_surface'].values()), unit='mm (max |dev|)', tolerance=t['tolerance_mm'], passed=t['passed']))
    t = c['c2_ray_hits_vs_analytic']; rows.append(dict(record_type='surf02', check='projector / camera ray hits vs analytic surface',
                                                        value=max(max(v['max_abs_vs_analytic_mm'], v.get('cam_max_abs_vs_analytic_mm', 0.0)) for v in t['per_surface'].values()), unit='mm (max |dev|)',
                                                        tolerance=t['tolerance_mm'], passed=t['passed']))
    t = c['c3_old_default_path_regression']; rows.append(dict(record_type='surf02', check='old default path regression (stored images re-rendered)', value=0.0, unit='differing pixels', passed=t.get('passed')))
    t = c['c4_metric_recompute']; rows.append(dict(record_type='surf02', check='independent M5 metric recomputation (32 cases)', value=t['worst_abs_diff'], unit='worst abs diff', tolerance=t['tolerance'], passed=t['passed']))
    t = c['c5_grid_resolution']; rows.append(dict(record_type='surf02', check='grid resolution 0.5 x 0.25 vs 0.25 x 0.125 mm (point RMSE)', value=t['max_abs_diff_pts_rmse_mm'], unit='mm', tolerance=0.01, passed=t['passed']))
    rows.append(dict(record_type='surf02', check='all checks passed', value=None, unit='', passed=c['all_passed']))
    return pd.DataFrame(rows)


def run() -> dict:
    parts = dict(C_identity=identity(), C_unconf_split=unconf_split(), C_tail=tail(), C_window=window(), C_resource=resource(), C_regimes=regimes(), C_gauss=gauss(), C_surf02=surf02())
    for k, d in parts.items():
        H.part(k, d)
    out = {k: len(d) for k, d in parts.items()}
    H.run_put('C_cross', dict(rows=out, t_local=H.now_local())); H.flush_reads('C_cross')
    return out
