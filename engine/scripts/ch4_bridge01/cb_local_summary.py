# Summarize local reconstruction metrics for the candidate comparisons.

from __future__ import annotations
import numpy as np
import pandas as pd
import cb_common as G

C = G.C
AN = G.OUT / 'analysis'


def main():
    L = pd.read_csv(AN / 'local_all.csv.gz'); R = pd.read_csv(AN / 'results_all.csv.gz', usecols=['case_key', 'config', 'env', 'sigma', 'surface', 'phase_block', 'rep', 'glob_Ctau'])
    J = C.csv(AN / 'selection_changes.csv')
    X = L.merge(R, on='case_key', validate='many_to_one')
    surf = X.groupby(['config', 'env', 'sigma', 'surface', 'd_mm', 'tau_mm']).apply(
        lambda g: pd.Series(dict(loc_Ctau=g.groupby('phase_block').loc_Ctau.mean().mean(), loc_Cvalid=g.groupby('phase_block').loc_Cvalid.mean().mean(),
                                 loc_sup_rmse=g.loc_sup_rmse.mean(), loc_unsupported=1 - g.groupby('phase_block').loc_Cvalid.mean().mean())), include_groups=False).reset_index()
    sc = surf.groupby(['config', 'env', 'sigma', 'd_mm', 'tau_mm']).agg(loc_J=('loc_Ctau', 'min'), loc_Cvalid_min=('loc_Cvalid', 'min'),
                                                                         loc_unsupported_max=('loc_unsupported', 'max'), loc_sup_rmse_max=('loc_sup_rmse', 'max')).reset_index()
    rows = []
    for r in J[(J.d_mm == 6) & (J.tau_mm == .15) & (J.floor == .5)].to_dict('records'):
        for tag in ('old', 'new'):
            cfg = r[f'preferred_{tag}']
            if not isinstance(cfg, str):
                continue
            q = sc[(sc.config == cfg) & (sc.env == r['env']) & (sc.sigma == r['sigma']) & (sc.d_mm == 6) & (sc.tau_mm == .15)].iloc[0]
            rows.append(dict(family=r['family'], env=r['env'], sigma=r['sigma'], set=tag, config=cfg, global_J=r[f'pref_score_{tag}'], local_J=q.loc_J,
                             local_Cvalid_min=q.loc_Cvalid_min, local_unsupported_max=q.loc_unsupported_max, local_support_rmse_max=q.loc_sup_rmse_max))
    T = pd.DataFrame(rows)
    C.atomic(AN / 'local_surfaces.csv.gz', lambda p: surf.to_csv(p, index=False, compression='gzip'))
    C.save_csv(AN / 'local_representatives.csv', T)
    piv = T.pivot_table(index=['family', 'env', 'sigma'], columns='set', values=['global_J', 'local_J'])
    print((100 * piv).round(1).to_string())
    print(T.groupby(['family', 'set'])[['global_J', 'local_J', 'local_unsupported_max', 'local_support_rmse_max']].mean().round(4).to_string())


if __name__ == '__main__':
    C.install_guard(); main()
