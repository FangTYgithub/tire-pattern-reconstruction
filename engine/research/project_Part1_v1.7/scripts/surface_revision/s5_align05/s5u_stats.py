# Calculate joint-study condition summaries and paired front-end effects.

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import s5u_common as K
import s5r_bind as B

_A2 = K.ROOT / 'scripts' / 'surface_revision' / 's5_align02'
if str(_A2) not in sys.path:
    sys.path.append(str(_A2))
import s5c_stats as ST

KEY = ['combo', 'surface', 'phase_label', 'sigma', 'rep']
PH = [f'ph{i}' for i in range(8)]
COV = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_cov_interp', 'loc_cov_interp', 'glob_recovery')
ERR = ('glob_pt_rmse', 'glob_pt_p95', 'glob_pt_max', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'loc_surf_rmse', 'sup_surf_rmse')
EFFECT_METRICS = ST.K.METRICS
CASE_COLS = ['manifest_key', 'image_key', 'decode_key', 'eval_record', 'source', 'surface', 'family', 'combo', 'radius_mm', 'diameter_mm', 'line_width_mm', 'p_parallel_mm',
             'p_perp_mm', 'line_pitch_mm', 'orientation_deg', 'phase_label', 'phase_offset_par_mm', 'phase_offset_perp_mm', 'sigma', 'rep', 'noise_seed', 'front_end', 'peak',
             'integral', 'state', 'n_out', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'id_denominator', 'risk_wrong',
             'unconf_share', 'glob_id_correct', 'glob_id_wrong', 'glob_id_unconf', 'loc_id_correct', 'loc_id_wrong', 'loc_id_unconf', *COV, *ERR,
             'glob_n_supported_nodes', 'image_identity', 'decode_identity', 'eval_identity', 'content_sha']


def case_table(old: pd.DataFrame, new: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    o = old.copy(); o['source'] = np.where(o.method == 'DF', 'S5-ALIGN-02 DF shard (S5-ALIGN-03 binding)', 's345_ready record (S5-ALIGN-03 binding)')
    o['eval_record'] = np.where(o.method == 'DF', 'outputs/surface_revision/s5_align02/df/df_' + o.combo + '.npz', 'outputs/surface_revision/s345_ready/run/eval/' + o.decode_key + '.npz')
    n = new.copy(); n['source'] = 'S5-ALIGN-04/05 formal store'; n['decode_key'] = n.image_key + '__' + n.method
    if len(n):
        n['eval_record'] = n.eval_record.str.replace(r'\.json$', '.npz', regex=True)
    C = pd.concat([o, n], ignore_index=True, sort=False)
    C['front_end'] = C.method; C['sigma'] = C.sigma.astype(float); C['rep'] = C.rep.astype(int)
    man = M.rename(columns={'manifest_key': 'manifest_key'})[['combo', 'surface', 'phase_label', 'sigma', 'rep', 'manifest_key', 'family', 'radius_mm', 'diameter_mm', 'line_width_mm',
                                                               'p_parallel_mm', 'p_perp_mm', 'line_pitch_mm', 'orientation_deg', 'phase_offset_par_mm', 'phase_offset_perp_mm', 'noise_seed',
                                                               'peak', 'integral']]
    C = C.drop(columns=[c for c in ('family', 'peak') if c in C.columns]).merge(man, on=KEY, how='left', validate='many_to_one')
    if C.manifest_key.isna().any():
        raise K.Stop('refused', 'case rows without a manifest row')
    c, w, u = (pd.to_numeric(C[x], errors='coerce') for x in ('id_correct', 'id_wrong', 'id_unconf'))
    C['risk_wrong'] = np.where(c + w > 0, w / (c + w).where(c + w > 0), np.nan); C['unconf_share'] = np.where(c + w + u > 0, u / (c + w + u).where(c + w + u > 0), np.nan)
    for col in CASE_COLS:
        if col not in C.columns:
            C[col] = np.nan
    return C[CASE_COLS].sort_values(['family', 'combo', 'surface', 'sigma', 'phase_label', 'rep', 'front_end']).reset_index(drop=True)


def _stat_frame(C: pd.DataFrame) -> pd.DataFrame:
    X = C.rename(columns={'combo': 'pair_role'}).copy(); X['set_name'] = 'S5-main'
    for m in set(EFFECT_METRICS) | {'glob_Ctau', 'loc_Ctau'}:
        X[m] = pd.to_numeric(X[m], errors='coerce')
    return X


def cross_noise(T: pd.DataFrame, idx, a='40.0', b='0.0') -> pd.DataFrame:

    add = []; grp = ['set_name', 'surface', 'family', 'set', 'effect', 'metric', 'front_end']
    for k, g in T[T.state == 'ok'].groupby(grp):
        if not {a, b} <= set(g.sigma):
            continue
        e = g[g.sigma == a].iloc[0][PH].to_numpy(float) - g[g.sigma == b].iloc[0][PH].to_numpy(float)
        add.append(dict(record_type='interaction', **dict(zip(grp, k)), sigma=f'{a[:-2]}-{b[:-2]}', **ST.summarise(e, idx)))
    return pd.DataFrame(add)


def effects(C: pd.DataFrame) -> pd.DataFrame:
    idx = ST.resample_idx(); X = _stat_frame(C)
    T = ST.interactions(X, ST.K.FOUR, True, idx, metrics=EFFECT_METRICS, extra_keys=('front_end',))
    T = T[T.sigma != '40-0']
    T = pd.concat([T, cross_noise(T, idx)], ignore_index=True); T['analysis'] = 'registered'
    H = ST.interactions(X[X.family == 'line'], ST.K.FOUR_POSTHOC, False, idx, metrics=EFFECT_METRICS, extra_keys=('front_end',))
    H = H[H.sigma != '40-0']; H = pd.concat([H, cross_noise(H, idx)], ignore_index=True); H['analysis'] = 'posthoc (reference S5-ALIGN-01 review 3.2)'
    F = pd.concat([T, H], ignore_index=True)
    diff = []
    for k, g in F[(F.state == 'ok') & (F.family == 'point')].groupby(['surface', 'sigma', 'set', 'effect', 'metric', 'analysis']):
        if set(g.front_end) != {'D0', 'DF'}:
            continue
        e = g[g.front_end == 'DF'].iloc[0][PH].to_numpy(float) - g[g.front_end == 'D0'].iloc[0][PH].to_numpy(float)
        diff.append(dict(record_type='interaction_df_minus_d0', set_name='S5-main', family='point', **dict(zip(['surface', 'sigma', 'set', 'effect', 'metric', 'analysis'], k)),
                         front_end='DF-D0', **ST.summarise(e, idx)))
    return pd.concat([F, pd.DataFrame(diff)], ignore_index=True)


def _phase_mean(g: pd.DataFrame, m: str):
    v = pd.to_numeric(g[m], errors='coerce'); ph = v.groupby(g.phase_label).mean()
    return float(ph.mean()) if len(ph) else None


def condition_summary(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in C.groupby(['family', 'combo', 'surface', 'sigma', 'front_end']):
        r = dict(zip(['family', 'combo', 'surface', 'sigma', 'front_end'], k), n_cases=len(g), n_phases=int(g.phase_label.nunique()), n_reps=int(g.rep.nunique()),
                 n_no_output=int((g.state == 'no_output').sum()), n_ok=int((g.state == 'ok').sum()), radius_mm=g.radius_mm.iloc[0], line_width_mm=g.line_width_mm.iloc[0],
                 p_parallel_mm=g.p_parallel_mm.iloc[0], p_perp_mm=g.p_perp_mm.iloc[0], line_pitch_mm=g.line_pitch_mm.iloc[0], peak=float(g.peak.iloc[0]))
        for m in COV:
            r[f'{m}_mean8'] = _phase_mean(g, m)
        for m in ERR:
            v = pd.to_numeric(g[m], errors='coerce').dropna()
            r[f'{m}_n_defined'] = int(len(v)); r[f'{m}_median'] = float(v.median()) if len(v) else None; r[f'{m}_max'] = float(v.max()) if len(v) else None
        cc, ww, uu = (int(pd.to_numeric(g[x], errors='coerce').sum()) for x in ('id_correct', 'id_wrong', 'id_unconf'))
        r.update(pooled_c=cc, pooled_w=ww, pooled_u=uu, risk_wrong_pooled=(ww / (cc + ww) if cc + ww else None), unconf_share_pooled=(uu / (cc + ww + uu) if cc + ww + uu else None),
                 cases_with_w=int((pd.to_numeric(g.id_wrong) > 0).sum()), cases_with_u=int((pd.to_numeric(g.id_unconf) > 0).sum()),
                 n_valid_mean8=_phase_mean(g, 'n_valid'), risk_definition='point-pooled over all accepted points of the condition')
        rows.append(r)
    return pd.DataFrame(rows)


def frontend_conditions(C: pd.DataFrame) -> pd.DataFrame:

    d = C[C.family == 'point']; a = d[d.front_end == 'D0']; b = d[d.front_end == 'DF']
    P = a.merge(b, on=KEY, suffixes=('_D0', '_DF'), validate='one_to_one')
    if (P.image_key_D0 != P.image_key_DF).any():
        raise K.Stop('refused', 'D0 / DF pair on different images')
    rows = []
    for k, g in P.groupby(['combo', 'surface', 'sigma']):
        r = dict(record_type='frontend_condition', combo=k[0], surface=k[1], sigma=k[2], n_pairs=len(g))
        for m in ('glob_Ctau', 'loc_Ctau', 'n_valid'):
            dd = pd.to_numeric(g[f'{m}_DF'], errors='coerce') - pd.to_numeric(g[f'{m}_D0'], errors='coerce')
            r[f'd_{m}_mean8'] = float(dd.groupby(g.phase_label).mean().mean())
        for m in ('glob_pt_rmse', 'sup_surf_rmse'):
            dd = (pd.to_numeric(g[f'{m}_DF'], errors='coerce') - pd.to_numeric(g[f'{m}_D0'], errors='coerce')).dropna()
            r[f'd_{m}_median_defined_pairs'] = float(dd.median()) if len(dd) else None; r[f'{m}_n_defined_pairs'] = int(len(dd))
        for fe in ('D0', 'DF'):
            cc, ww, uu = (int(pd.to_numeric(g[f'{x}_{fe}'], errors='coerce').sum()) for x in ('id_correct', 'id_wrong', 'id_unconf'))
            r[f'risk_wrong_pooled_{fe}'] = ww / (cc + ww) if cc + ww else None; r[f'unconf_share_pooled_{fe}'] = uu / (cc + ww + uu) if cc + ww + uu else None
            r[f'pooled_w_{fe}'] = ww; r[f'pooled_u_{fe}'] = uu
        rows.append(r)
    return pd.DataFrame(rows)


def regression_old_slice(C: pd.DataFrame) -> dict:

    O = C[C.sigma.isin([0.0, 40.0]) & ~C.source.str.startswith('S5-ALIGN-04/05')]
    E = effects(O)
    key = ['set_name', 'surface', 'family', 'sigma', 'set', 'effect', 'metric']
    reg = K.read_csv(K.K3.reg(K.SR / 's345_ready' / 'results' / 'S5_interactions.csv')); reg = reg[reg.set_name == 'S5-main']
    main = E[(E.analysis == 'registered') & E.front_end.isin(['D0', 'P4']) & (E.record_type == 'interaction')].drop(columns=['front_end'])
    r1 = B._cmp(main, reg, key)
    fe_ref = K.read_csv(K.K3.reg(K.SR / 's5_align02' / '_parts' / 'C_df_interactions.csv'))
    fe = E[(E.analysis == 'registered') & (E.family == 'point') & E.set.str.startswith('dots') & E.metric.isin(['glob_Ctau', 'loc_Ctau', 'glob_pt_rmse'])].drop(columns=['analysis'])
    r2 = B._cmp(fe, fe_ref, key + ['front_end'])
    ph_ref = K.read_csv(K.K3.reg(K.SR / 's5_align02' / '_parts' / 'A_interactions_posthoc_63.csv')); ph_ref = ph_ref[ph_ref.set_name == 'S5-main']
    ph = E[E.analysis.str.startswith('posthoc')].drop(columns=['front_end'])
    r3 = B._cmp(ph, ph_ref, key)
    return dict(registered_main=r1, front_end=r2, posthoc_main=r3, ok=bool(r1['ok'] and r2['ok'] and r3['ok']))


def selfcheck(C: pd.DataFrame) -> dict:

    out = []
    for s in K.K3.MAIN:
        for fam in ('point', 'line'):
            for sg in (20.0, 100.0):
                g = C[(C.surface == s) & (C.family == fam) & (C.sigma == sg)].sort_values(['combo', 'phase_label', 'rep', 'front_end'])
                r = g.iloc[0]; z = K.load_npz(K.ROOT / r.eval_record)
                acc, val = z['accepted'].astype(bool), z['valid'].astype(bool); ing, inl = z['in_global'].astype(bool), z['in_local'].astype(bool)
                pc, pw, pu = (z[a].astype(bool) for a in ('point_correct', 'point_wrong', 'point_unconf'))
                e = z['pts'][:, 2] - z['zr'].astype(np.float64)
                rec = dict(record=r.eval_record, combo=r.combo, surface=s, sigma=sg, front_end=r.front_end, state=r.state)
                chk = dict(n_out=int(len(z['pts'])) == int(r.n_out), n_accepted=int(acc.sum()) == int(r.n_accepted), n_valid=int(val.sum()) == int(r.n_valid),
                           glob_n_valid=int(ing.sum()) == int(r.glob_n_valid), loc_n_valid=int(inl.sum()) == int(r.loc_n_valid),
                           ids=(int((pc & acc).sum()), int((pw & acc).sum()), int((pu & acc).sum())) == (int(r.id_correct), int(r.id_wrong), int(r.id_unconf)))
                dmax = 0.0
                for d, msk in (('glob', ing), ('loc', inl)):
                    ee = e[msk]; ee = ee[np.isfinite(ee)]
                    rm = float(np.sqrt(np.mean(ee * ee))) if ee.size else None; p9 = float(np.percentile(np.abs(ee), 95)) if ee.size else None
                    for got, tab in ((rm, r[f'{d}_pt_rmse']), (p9, r[f'{d}_pt_p95'])):
                        tv = K.K3.num(tab)
                        dmax = max(dmax, 0.0 if (got is None and tv is None) else (abs(got - tv) if (got is not None and tv is not None) else float('inf')))
                ns = K.K3.num(r.glob_n_supported_nodes)
                rec.update(counts_equal=all(chk.values()), pt_max_abs_diff_mm=dmax,
                           cvalid_from_nodes=(abs(ns / 30947 - float(r.glob_Cvalid)) if ns is not None else None),
                           note='C_valid compared with the saved supported-node count / 30947 nodes (equal node areas); no new interpolation')
                out.append(rec)
    return dict(records=out, ok=bool(all(x['counts_equal'] and x['pt_max_abs_diff_mm'] <= 1e-6 for x in out)))
