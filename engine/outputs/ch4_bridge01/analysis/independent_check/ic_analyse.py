# Compare reconstruction mechanisms, ranking stability and interactions from stored results.

import sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts/ch4_bridge01'))
import cb_analyse as AN
G = AN.G; C = G.C
HERE = Path(__file__).resolve().parent
A = G.OUT / 'analysis'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)


def mechanism():
    R, S, X, *_ = AN.assemble(); X = X.set_index('case_key', drop=False)
    import cb_adapter as AD
    AD.platform()
    from surface_eval_r3 import surface_eval_r3_dev
    import rdy_platform as RP
    dom = RP.dom_global()
    cfgs = ['D_r1_p8x4', 'D_r1_T8x4', 'D_r1_p8x6', 'D_r0.75_p8x4', 'D_r0.75_p8x6', 'L_w0.5_s8', 'L_w1_s8']
    rows = []
    sel = R[R.config.isin(cfgs) & R.env.isin(['NOMINAL', 'BLUR2', 'G05']) & R.sigma.isin([0, 20, 40])]
    for r in sel.to_dict('records'):
        x = G.rowdict(X.loc[r['case_key']].to_dict()); p = AN.pack_path(x); m = x['method']
        with np.load(p, allow_pickle=False) as z:
            pts = np.asarray(z[m + '__ev__pts'], float); v = np.asarray(z[m + '__ev__valid'], bool); zr = np.asarray(z[m + '__ev__zr'], float)
        sid = {'B_curv': 'B_curv', 'M10_curv': 'A_curv', 'M24_curv': 'C_curv'}[r['surface']]
        D = dict(x=dom['x'], y=dom['y'], z_ref=np.asarray(dom[f'z_{sid}'], float), area=np.full(len(dom['x']), dom['step'] ** 2), hash=dom['hash'])
        pv = pts[v]; ideal = pv.copy(); ideal[:, 2] = zr[v]
        out = dict(case_key=r['case_key'], config=r['config'], env=r['env'], sigma=r['sigma'], surface=r['surface'], phase_block=r['phase_block'], n_valid=int(v.sum()),
                   Ctau=r['glob_Ctau'], Cvalid=r['glob_Cvalid'], pt_rmse=r['glob_pt_rmse'])
        if len(pv) >= 3:
            g = surface_eval_r3_dev(ideal, None, D, [6.0], [.15], support_rule='exact')['subdomains']['global']['by_d']['6.0']
            out.update(C_ideal=g['C_tau']['0.15'] or 0.0, Cvalid_ideal=g['C_valid'] or 0.0)
        else:
            out.update(C_ideal=0.0, Cvalid_ideal=0.0)
        rows.append(out)
    T = pd.DataFrame(rows); T.to_csv(HERE / 'mechanism_cases.csv', index=False)
    s = T.groupby(['env', 'sigma', 'config']).agg(n_valid=('n_valid', 'mean'), pt_rmse=('pt_rmse', 'mean'), Cvalid=('Cvalid', 'mean'), C_ideal=('C_ideal', 'mean'),
                                                   Ctau=('Ctau', 'mean')).reset_index()
    s['height_share'] = s.Ctau / s.C_ideal
    print('== A mechanism (means over 3 surfaces x 4 blocks x reps)'); print(s.round(4).to_string(index=False))
    s.to_csv(HERE / 'mechanism_summary.csv', index=False)


def stability():
    J = C.csv(A / 'selection_changes.csv')
    rows = []
    for (fam, e, s), t in J.groupby(['family', 'env', 'sigma']):
        reps = t.preferred_new.dropna(); old = t.preferred_old.dropna()
        rows.append(dict(family=fam, env=e, sigma=s, variants_with_solution_new=len(reps), variants_with_solution_old=len(old),
                         reps_new=';'.join(f'{k}:{v}' for k, v in reps.value_counts().items()), reps_old=';'.join(f'{k}:{v}' for k, v in old.value_counts().items())))
    T = pd.DataFrame(rows); T.to_csv(HERE / 'requirement_stability.csv', index=False)
    print('== B representatives over 16 requirement variants (count per config)')
    print(T[(T.family == 'point') & (T.sigma <= 60)].to_string(index=False))


def knife_edges():
    J = C.csv(A / 'selection_changes.csv'); m = J[(J.tau_mm == .15)]
    k = m[m.preferred_new.notna() & ((m.pref_score_new - m.floor).abs() < .02)][['family', 'env', 'sigma', 'd_mm', 'floor', 'preferred_new', 'pref_score_new', 'near_optimal_new']]
    print('== C representatives within 2 pp of their floor (tau 0.15)'); print(k.to_string(index=False))
    mult = m[m.near_optimal_new.fillna('').str.count(';') >= 1][['family', 'env', 'sigma', 'd_mm', 'floor', 'near_optimal_new', 'preferred_new', 'best_score_new']]
    print('== C near-optimal sets with >= 2 configurations (tau 0.15)'); print(mult.to_string(index=False))
    k.to_csv(HERE / 'knife_edges.csv', index=False); mult.to_csv(HERE / 'near_sets_multi.csv', index=False)


def cross_family():
    J = C.csv(A / 'selection_changes.csv'); rows = []
    for (e, s, d, tau, fl), t in J.groupby(['env', 'sigma', 'd_mm', 'tau_mm', 'floor']):
        p = t[t.family == 'point'].iloc[0]; l = t[t.family == 'line'].iloc[0]
        bp, bl = p.best_score_new, l.best_score_new
        rows.append(dict(env=e, sigma=s, d=d, tau=tau, floor=fl, dot=bp, line=bl, winner=('dot' if (pd.notna(bp) and (pd.isna(bl) or bp > bl)) else ('line' if pd.notna(bl) else 'none')),
                         old_winner=('dot' if (pd.notna(p.best_score_old) and (pd.isna(l.best_score_old) or p.best_score_old > l.best_score_old)) else ('line' if pd.notna(l.best_score_old) else 'none'))))
    T = pd.DataFrame(rows); T.to_csv(HERE / 'cross_family.csv', index=False)
    print('== D winners where any family is admissible (counts)')
    print(pd.crosstab([T.sigma], [T.old_winner, T.winner]).to_string())
    print(T[(T.winner == 'line')].to_string(index=False))


if __name__ == '__main__':
    C.install_guard()
    for f in sys.argv[1:]:
        dict(mechanism=mechanism, stability=stability, knife=knife_edges, cross=cross_family)[f]()
