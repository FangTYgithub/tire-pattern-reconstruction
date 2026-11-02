# Select admissible and near-optimal configurations under registered metric thresholds.

from __future__ import annotations

import numpy as np
import pandas as pd

import r_common as K

REF = dict(point='D_r1_p8x6', line='L_w0.5_s8')
TIE = 0.02
EPS = 1e-12
RMSE_MAX, U_MAX = 0.30, 0.05


def _levels(g: pd.DataFrame, col: str) -> float:
    b = g.groupby('phase_block')[col].mean()
    return float(b.mean())


def check_complete(df: pd.DataFrame, expected: pd.DataFrame) -> None:
    key = ['family', 'env', 'sigma', 'config', 'surface', 'phase_block', 'rep']
    d = df.duplicated(key, keep=False)
    if d.any():
        raise K.Stop('integrity', f'duplicate image keys: {df.loc[d, key].head(3).values.tolist()}')
    m = expected[key].merge(df[key], on=key, how='outer', indicator=True)
    miss = m[m._merge == 'left_only']; extra = m[m._merge == 'right_only']
    if len(miss) or len(extra):
        raise K.Stop('integrity', f'image set differs from the source manifest: {len(miss)} missing, {len(extra)} unexpected (e.g. {miss[key].head(2).values.tolist()})')


def select(df: pd.DataFrame, expected: pd.DataFrame, floor: float = 0.50, gcol: str = 'glob_Ctau', lcol: str = 'loc_Ctau') -> tuple[pd.DataFrame, pd.DataFrame]:

    check_complete(df, expected)
    surf_rows, picks = [], []
    for (fam, env, sg), g in df.groupby(['family', 'env', 'sigma'], sort=True):
        comp = []
        for cfg, gc in g.groupby('config', sort=True):
            per = {}
            for s in K.SURFS:
                gs = gc[gc.surface == s]
                gl, ll = _levels(gs, gcol), _levels(gs, lcol)
                rm_ok = bool(gs.glob_pt_rmse.notna().all() and (gs.glob_pt_rmse <= RMSE_MAX).all())
                w_ok = bool((gs.w.fillna(1) == 0).all())
                u_ok = bool(gs.u_share.notna().all() and (gs.u_share <= U_MAX).all() and ((gs.c.fillna(0) + gs.w.fillna(0)) > 0).all())
                ok = bool(gl >= floor and ll >= floor and rm_ok and w_ok and u_ok)
                per[s] = dict(glob=gl, loc=ll, feasible=ok)
                surf_rows.append(dict(family=fam, env=env, sigma=sg, config=cfg, surface=s, glob_level=gl, loc_level=ll, ctau_ok=bool(gl >= floor and ll >= floor), rmse_ok=rm_ok,
                                      w_ok=w_ok, u_ok=u_ok, feasible=ok, max_img_rmse=(float(gs.glob_pt_rmse.max()) if gs.glob_pt_rmse.notna().any() else None),
                                      n_no_output=int((gs.state == 'no_output').sum())))
            qualified = set(per) == set(K.SURFS) and all(per[s]['feasible'] for s in K.SURFS)
            comp.append(dict(config=cfg, qualified=qualified, score=min(per[s]['glob'] for s in K.SURFS), worst_local=min(per[s]['loc'] for s in K.SURFS),
                             max_img_rmse=(float(gc.glob_pt_rmse.max()) if gc.glob_pt_rmse.notna().any() else np.inf)))
        C = pd.DataFrame(comp); Q = C[C.qualified]
        rec = dict(family=fam, env=env, sigma=sg, floor=floor, n_configs=len(C), qualified=';'.join(sorted(Q.config)))
        if len(Q) == 0:
            bf = C.sort_values(['score', 'config'], ascending=[False, True]).iloc[0]
            rec.update(preferred=None, decision='kein qualifizierter Kandidat', failing_reference=bf.config, failing_reference_score=float(bf.score), tie_band='')
        else:
            best = float(Q.score.max()); tie = Q[Q.score >= best - TIE - EPS]
            rec['tie_band'] = ';'.join(sorted(tie.config)); rec['best_score'] = best
            ref = REF[fam]
            if ref in set(tie.config):
                pref, why = ref, 'Referenz im Gleichstandsband'
            else:
                t = tie.copy()
                t = t.sort_values(['worst_local', 'max_img_rmse', 'config'], ascending=[False, True, True])
                pref = t.iloc[0].config
                if len(t) == 1:
                    why = 'einziger Kandidat im Gleichstandsband'
                elif abs(t.iloc[0].worst_local - t.iloc[1].worst_local) > EPS:
                    why = 'höhere schlechteste lokale C_tau'
                elif abs(t.iloc[0].max_img_rmse - t.iloc[1].max_img_rmse) > EPS:
                    why = 'kleinere maximale globale Punkt-RMSE'
                else:
                    why = 'Konfigurations-ID (lexikographisch)'
            p = Q[Q.config == pref].iloc[0]
            rec.update(preferred=pref, decision=why, pref_score=float(p.score), pref_worst_local=float(p.worst_local), pref_max_img_rmse=float(p.max_img_rmse))
        rec['components'] = C.to_dict('records')
        picks.append(rec)
    return pd.DataFrame(surf_rows), pd.DataFrame(picks)


def _synthetic(configs: dict, fam: str = 'line') -> tuple[pd.DataFrame, pd.DataFrame]:

    rows = []
    for cfg, p in configs.items():
        for s in K.SURFS:
            for b in K.BLOCKS:
                for r in K.REPS:
                    no = p.get('no_output_surface') == s
                    rows.append(dict(family=fam, env='E', sigma=20.0, config=cfg, surface=s, phase_block=b, rep=r, state=('no_output' if no else 'ok'),
                                     glob_Ctau=(0.0 if no else p['glob'] + p.get('surf_glob', {}).get(s, 0.0)), loc_Ctau=(0.0 if no else p['loc']),
                                     glob_pt_rmse=(np.nan if no else p.get('rmse', 0.1) + (p.get('rmse_peak', 0.0) if (s == 'B_curv' and b == 0 and r == 1) else 0.0)),
                                     w=(0 if no else p.get('w', 0)), u_share=(np.nan if no else p.get('u', 0.0)), c=(0 if no else 100)))
    df = pd.DataFrame(rows)
    return df, df[['family', 'env', 'sigma', 'config', 'surface', 'phase_block', 'rep']].copy()


def tests() -> dict:
    out = {}

    def expect_stop(name, fn):
        try:
            fn(); out[name] = 'NOT REFUSED'
        except K.Stop as e:
            out[name] = f'refused ({e.code})'

    base = {'L_a': dict(glob=0.70, loc=0.75), 'L_b': dict(glob=0.60, loc=0.70)}
    df, ex = _synthetic(base)
    expect_stop('missing_surface', lambda: select(df[~((df.config == 'L_a') & (df.surface == 'M24_curv'))], ex))
    expect_stop('missing_rep', lambda: select(df[~((df.config == 'L_a') & (df.surface == 'B_curv') & (df.phase_block == 1) & (df.rep == 2))], ex))
    expect_stop('duplicate_key', lambda: select(pd.concat([df, df.iloc[[0]]], ignore_index=True), ex))
    d2, e2 = _synthetic({'L_a': dict(glob=0.70, loc=0.75, no_output_surface='M10_curv'), 'L_b': dict(glob=0.60, loc=0.70)})
    _, p = select(d2, e2); out['legal_no_output'] = 'ok' if p.iloc[0].preferred == 'L_b' else f'WRONG {p.iloc[0].preferred}'
    d3, e3 = _synthetic({'L_a': dict(glob=0.40, loc=0.75), 'L_b': dict(glob=0.60, loc=0.30)})
    _, p = select(d3, e3); out['all_fail'] = 'ok' if (p.iloc[0].preferred is None and p.iloc[0].decision == 'kein qualifizierter Kandidat' and p.iloc[0].failing_reference == 'L_b') else 'WRONG'
    d4, e4 = _synthetic({'L_w0.5_s8': dict(glob=0.685, loc=0.70), 'L_top': dict(glob=0.70, loc=0.90)})
    _, p = select(d4, e4); out['reference_in_tie_band'] = 'ok' if p.iloc[0].preferred == 'L_w0.5_s8' else f'WRONG {p.iloc[0].preferred}'
    d5, e5 = _synthetic({'L_x': dict(glob=0.70, loc=0.80, rmse=0.10, rmse_peak=0.15), 'L_y': dict(glob=0.70, loc=0.80, rmse=0.10, rmse_peak=0.05)})
    _, p = select(d5, e5); out['local_tie_rmse_decides'] = 'ok' if (p.iloc[0].preferred == 'L_y' and p.iloc[0].decision == 'kleinere maximale globale Punkt-RMSE') else f'WRONG {p.iloc[0].preferred}'
    d6, e6 = _synthetic({'L_q': dict(glob=0.70, loc=0.80), 'L_p': dict(glob=0.70, loc=0.80)})
    _, p = select(d6, e6); out['final_id_tiebreak'] = 'ok' if (p.iloc[0].preferred == 'L_p' and p.iloc[0].decision == 'Konfigurations-ID (lexikographisch)') else f'WRONG {p.iloc[0].preferred}'
    d7, e7 = _synthetic({'L_m': dict(glob=0.70, loc=0.80), 'L_n': dict(glob=0.71, loc=0.60), 'L_o': dict(glob=0.55, loc=0.95)})
    _, pa = select(d7, e7); _, pb = select(d7.sample(frac=1.0, random_state=7).reset_index(drop=True), e7.sample(frac=1.0, random_state=3))
    out['shuffle_invariant'] = 'ok' if (pa.preferred.tolist() == pb.preferred.tolist() and pa.decision.tolist() == pb.decision.tolist()) else 'WRONG'
    out['shuffle_case_pick'] = pa.iloc[0].preferred
    d8, e8 = _synthetic({'L_a': dict(glob=0.70, loc=0.75, w=1), 'L_b': dict(glob=0.60, loc=0.70)})
    _, p = select(d8, e8); out['wrong_id_excludes'] = 'ok' if p.iloc[0].preferred == 'L_b' else 'WRONG'
    d9, e9 = _synthetic({'L_a': dict(glob=0.70, loc=0.75, u=0.06), 'L_b': dict(glob=0.60, loc=0.70)})
    _, p = select(d9, e9); out['u_share_excludes'] = 'ok' if p.iloc[0].preferred == 'L_b' else 'WRONG'
    ok = all(v.startswith('refused') for k, v in out.items() if k in ('missing_surface', 'missing_rep', 'duplicate_key'))
    ok = ok and all(v == 'ok' for k, v in out.items() if k not in ('missing_surface', 'missing_rep', 'duplicate_key', 'shuffle_case_pick'))
    out['ok'] = bool(ok)
    return out


def image_table(T: pd.DataFrame | None = None, tau: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    R = pd.read_csv(K.SRC5 / 'results.csv', float_precision='round_trip', low_memory=False)
    R = R[(((R.family == 'point') & (R.method == 'DF')) | ((R.family == 'line') & (R.method == 'P4'))) & (R.sigma > 0)].copy()
    M = pd.read_csv(K.SRC5 / 'formal_manifest.csv', low_memory=False)
    ex = M[M.sigma > 0][['family', 'env', 'sigma', 'config', 'surface', 'phase_block', 'rep']].copy()
    df = R[['family', 'env', 'sigma', 'config', 'surface', 'phase_block', 'rep', 'state', 'glob_Ctau', 'loc_Ctau', 'glob_pt_rmse', 'w', 'u_share', 'c', 'case_key']].copy()
    if T is not None:
        q = T[(T.d_mm == 6.0) & (T.tau_mm == tau)].pivot_table(index='case_key', columns='domain', values='Ctau', aggfunc='first')
        df['glob_Ctau'] = df.case_key.map(q['glob']); df['loc_Ctau'] = df.case_key.map(q['loc'])
        if df[['glob_Ctau', 'loc_Ctau']].isna().any().any():
            raise K.Stop('integrity', f'tau {tau}: C_tau missing for some images')
    return df, ex


def run(T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df, ex = image_table(None)
    Sm, Pm = select(df, ex, 0.50)
    old = pd.read_csv(K.SRC6 / '_work' / 'screening_mp5_pick.csv')
    J = Pm.merge(old[['family', 'env', 'sigma', 'preferred']], on=['family', 'env', 'sigma'], how='outer', suffixes=('', '_06A'), indicator=True)
    same = [(r['preferred'] == r['preferred_06A']) or (pd.isna(r['preferred']) and pd.isna(r['preferred_06A'])) for r in J.to_dict('records')]
    reg = dict(n_conditions=len(Pm), n_06A=len(old), all_matched=bool((J._merge == 'both').all()), identical_picks=bool(all(same)),
               differences=J[~np.array(same)][['family', 'env', 'sigma', 'preferred', 'preferred_06A']].to_dict('records'))
    allS, allP = [Sm.assign(floor=0.50, tau_mm=0.15, version='main')], [Pm.assign(tau_mm=0.15, version='main')]
    for floor in (0.50, 0.70):
        for tau in K.TAUS:
            d, e = image_table(T, tau)
            S2, P2 = select(d, e, floor)
            v = 'main (A1 table)' if (floor == 0.50 and tau == 0.15) else 'sensitivity'
            allS.append(S2.assign(floor=floor, tau_mm=tau, version=v)); allP.append(P2.assign(tau_mm=tau, version=v))
            if v.startswith('main'):
                reg['main_via_A1_table_identical'] = bool(P2.preferred.fillna('-').tolist() == Pm.preferred.fillna('-').tolist())
    return pd.concat(allS, ignore_index=True), pd.concat(allP, ignore_index=True), reg
