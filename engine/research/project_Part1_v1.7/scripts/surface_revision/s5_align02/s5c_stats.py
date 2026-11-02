# Compute phase-block size-spacing effects and four-cell interactions.

from __future__ import annotations

import numpy as np
import pandas as pd

import s5c_common as K

reference_63 = {('B_curv', 0.0): (-6.48, -7.30, -0.82, -1.94, 0.29), ('M10_curv', 0.0): (33.04, 37.30, 4.26, 1.73, 6.91), ('M24_curv', 0.0): (-11.93, -11.19, 0.74, -0.24, 1.80),
            ('B_curv', 40.0): (10.36, 14.48, 4.12, -2.21, 10.55), ('M10_curv', 40.0): (35.64, 28.76, -6.88, -13.13, -0.50), ('M24_curv', 40.0): (6.73, 11.05, 4.32, -6.07, 15.11)}


def resample_idx() -> np.ndarray:
    return np.random.Generator(np.random.PCG64(K.SEED)).integers(0, 8, size=(K.N_RES, 8))


def phase_values(g: pd.DataFrame, cells: dict, metric: str, role_col='pair_role'):

    out, why = {}, None
    sg = float(g.sigma.iloc[0]); exp = 1 if sg == 0.0 else 2
    for ph in K.PHASES:
        row = {}
        for c in cells:
            v = g[(g[role_col] == c) & (g.phase_label == ph)]
            if len(v) != exp or v.rep.nunique() != exp:
                why = why or f'{ph}: cell {c} has {len(v)} rows / {v.rep.nunique()} reps of {exp}'; row = None; break
            x = pd.to_numeric(v[metric], errors='coerce')
            if x.isna().any():
                why = why or f'{ph}: {metric} undefined in {c} (no output)'; row = None; break
            row[c] = float(x.mean())
        if row is not None:
            out[ph] = sum(w * row[c] for c, w in cells.items())
    return out, why


def summarise(e: np.ndarray, idx: np.ndarray) -> dict:
    bs = e[idx].mean(axis=1)
    return dict(state='ok', mean=float(e.mean()), median=float(np.median(e)), n_pos=int((e > 0).sum()), n_neg=int((e < 0).sum()), n_zero=int((e == 0).sum()),
                min=float(e.min()), max=float(e.max()), res_lo_5=float(np.percentile(bs, 5)), res_hi_95=float(np.percentile(bs, 95)), **dict(zip(K.PHASES, e)))


def effect_defs(fam: str, four: dict, with_width: bool) -> list:
    defs = [(k, dict(E_small={v['ss']: 1, v['sd']: -1}, E_large={v['ls']: 1, v['ld']: -1}, I={v['ls']: 1, v['ld']: -1, v['ss']: -1, v['sd']: 1})) for k, v in four.items()
            if (k.startswith('dots') and fam == 'point') or (k.startswith('lines') and fam == 'line')]
    if with_width and fam == 'line':
        defs.append(('line_D_width', {f'D_width_s{s}': {b: 1, a: -1} for s, (a, b) in K.D_WIDTH.items()}))
    return defs


def interactions(C: pd.DataFrame, four: dict, with_width: bool, idx: np.ndarray, metrics=K.METRICS, extra_keys=(), role_col='pair_role') -> pd.DataFrame:
    rows = []
    for key, g in C.groupby(['set_name', 'surface', 'family', 'sigma', *extra_keys]):
        sn, surf, fam, sg = key[:4]; ex = dict(zip(extra_keys, key[4:]))
        for setn, comps in effect_defs(fam, four, with_width):
            for eff, w in comps.items():
                for m in metrics:
                    vals, why = phase_values(g, w, m, role_col)
                    rec = dict(record_type='interaction', set_name=sn, surface=surf, family=fam, sigma=str(sg), set=setn, effect=eff, metric=m, **ex)
                    rec.update(summarise(np.array([vals[p] for p in K.PHASES]), idx) if len(vals) == 8 else dict(state='undefined', n_phases_complete=len(vals), reason=why))
                    rows.append(rec)
    T = pd.DataFrame(rows)
    add = []
    grp = ['set_name', 'surface', 'family', 'set', 'effect', 'metric', *extra_keys]
    for k, g in T[T.state == 'ok'].groupby(grp):
        if set(g.sigma) != {'0.0', '40.0'}:
            continue
        e = g[g.sigma == '40.0'].iloc[0][K.PHASES].to_numpy(float) - g[g.sigma == '0.0'].iloc[0][K.PHASES].to_numpy(float)
        add.append(dict(record_type='interaction', **dict(zip(grp, k)), sigma='40-0', **summarise(e, idx)))
    return pd.concat([T, pd.DataFrame(add)], ignore_index=True)


def errata(C: pd.DataFrame) -> pd.DataFrame:
    R = []
    for surf, combo, sg, ph, rep in (('B_curv', 'L_w0.5_s6.3', 0.0, 'ph1', 0), ('B_curv', 'L_w1.5_s4.45', 0.0, 'ph0', 0)):
        r = C[(C.set_name == 'S5-main') & (C.surface == surf) & (C.pair_role == combo) & (C.sigma == sg) & (C.phase_label == ph) & (C.rep == rep)]
        if len(r) != 1:
            raise K.Stop('scope', f'erratum case {surf} {combo} {ph}: {len(r)} rows')
        r = r.iloc[0]; m = K.load_json(K.reg(K.S345 / 'run' / 'eval' / f'{r.decode_key}.json'))['metrics']
        R.append(dict(record_type='erratum_value', item='tail case: RMSE and maximum error separated', surface=surf, combo=combo, sigma=sg, phase=ph, rep=rep, decode_key=r.decode_key,
                      glob_pt_rmse_mm=m['glob_pt_rmse'], glob_pt_max_mm=m['glob_pt_max'], id_unconf=m['id_unconf'], loc_id_unconf=m['loc_id_unconf'], state=m['state']))
    P = K.plan_s5(); s = P[P.set_formal == 'S5-main'].groupby('family').seed_rule.unique()
    for fam, rules in s.items():
        R.append(dict(record_type='erratum_value', item='seed rule per family (S5-ALIGN-01 compat row named only the line rule)', family=fam, seed_rule=';'.join(rules),
                      seeds_per_surface_phase_rep=int(P[(P.set_formal == 'S5-main') & (P.family == fam) & (P.sigma > 0)].groupby(['surface', 'phase_label', 'rep']).noise_seed.nunique().max()),
                      seed_surface_independent=bool(P[(P.set_formal == 'S5-main') & (P.family == fam) & (P.sigma > 0)].groupby(['phase_label', 'rep']).noise_seed.nunique().max() == 1)))
    return pd.DataFrame(R)


def stats() -> dict:
    C = K.cases(); idx = resample_idx()
    T = interactions(C, K.FOUR, True, idx)
    O = pd.read_csv(K.reg(K.S345 / 'results' / 'S5_interactions.csv'), low_memory=False); O['sigma'] = O.sigma.astype(str)
    key = ['set_name', 'surface', 'family', 'sigma', 'set', 'effect', 'metric']
    M = T.merge(O, on=key, how='outer', suffixes=('', '_orig'), indicator=True); both = M[M._merge == 'both']
    cols = ['mean', 'median', 'min', 'max', 'res_lo_5', 'res_hi_95'] + K.PHASES
    dmax = {c: float((pd.to_numeric(both[c], errors='coerce') - pd.to_numeric(both[f'{c}_orig'], errors='coerce')).abs().max()) for c in cols}
    rep693 = dict(n_recomputed=len(T), n_original=len(O), only_recomputed=int((M._merge == 'left_only').sum()), only_original=int((M._merge == 'right_only').sum()),
                  state_equal=bool((both.state == both.state_orig).all()), sign_counts_equal=bool(((both.state != 'ok') | ((both.n_pos == both.n_pos_orig) & (both.n_neg == both.n_neg_orig))).all()),
                  max_abs_diff=dmax, undefined_rows=int((T.state != 'ok').sum()))
    rep693['ok'] = bool(rep693['only_recomputed'] == 0 and rep693['only_original'] == 0 and rep693['state_equal'] and rep693['sign_counts_equal'] and max(dmax.values()) <= 1e-12)

    H = interactions(C[C.family == 'line'], K.FOUR_POSTHOC, False, idx).assign(analysis='posthoc_reference_review_3.2')
    cmp = []
    for (s, sg), (es, el, i, lo, hi) in reference_63.items():
        g = H[(H.set_name == 'S5-main') & (H.surface == s) & (H.sigma == str(sg)) & (H.metric == 'glob_Ctau')].set_index('effect')
        mine = (100 * g.loc['E_small', 'mean'], 100 * g.loc['E_large', 'mean'], 100 * g.loc['I', 'mean'], 100 * g.loc['I', 'res_lo_5'], 100 * g.loc['I', 'res_hi_95'])
        cmp.append(dict(record_type='reference_63_check', surface=s, sigma=sg, **{f'{n}_mine': v for n, v in zip(('E_small', 'E_large', 'I', 'lo', 'hi'), mine)},
                        **{f'{n}_reference': v for n, v in zip(('E_small', 'E_large', 'I', 'lo', 'hi'), (es, el, i, lo, hi))},
                        max_abs_diff_pp=float(max(abs(a - b) for a, b in zip(mine, (es, el, i, lo, hi)))), n_pos=int(g.loc['I', 'n_pos']), n_neg=int(g.loc['I', 'n_neg'])))
    CM = pd.DataFrame(cmp)
    E = errata(C)
    K.part('A_interactions_registered', T); K.part('A_interactions_posthoc_63', H); K.part('A_reference63_check', CM); K.part('A_errata', E)
    out = dict(reproduction_693=rep693, posthoc_rows=len(H), posthoc_undefined=int((H.state != 'ok').sum()), reference_63_max_abs_diff_pp=float(CM.max_abs_diff_pp.max()),
               reference_63_ok=bool(CM.max_abs_diff_pp.max() <= 0.006), errata=E.drop(columns=['record_type']).to_dict('records'), t_local=K.now())
    K.run_put('A_stats', out); K.flush('stats')
    if not rep693['ok']:
        raise K.Stop('check', f'693-row reproduction failed: {rep693}')
    return out
