# Build repeat-aware encoding tables and paired effects.

from __future__ import annotations

import numpy as np
import pandas as pd

import c5_tables as T5

ADD = [f'{d}_{f}' for d in ('glob', 'loc') for f in ('n_valid', 'n_interp', 'n_nodes', 'cov_interp', 'surf_rmse', 'surf_p95')]
COND = set(T5.COND) | {'glob_surf_rmse', 'glob_surf_p95', 'loc_surf_rmse', 'loc_surf_p95', 'loc_pt_rmse', 'loc_pt_p95'}
CASE_VALUES = ['glob_Ctau', 'glob_Cvalid', 'glob_gap', 'loc_Ctau', 'loc_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'glob_surf_p95',
               'loc_surf_rmse', 'loc_surf_p95']
PAIR_VALUES = ['d_glob_Ctau', 'd_glob_Cvalid', 'd_glob_gap', 'd_loc_Ctau', 'a_glob_Ctau', 'b_glob_Ctau', 'd_glob_pt_rmse', 'd_w', 'd_u']


def case_table(T: pd.DataFrame, M: pd.DataFrame, cfg5: dict, derived: str) -> pd.DataFrame:
    C = T5.case_table(T, M, cfg5)
    if derived in ('compute', 'row'):
        E = T[(T.layer == 'eval') & T.complete].set_index('key')
        for f in ADD + ['glob_surf_rmse']:
            C[f] = pd.to_numeric(C.eval_key.map(E[f'x_{f}']), errors='coerce')
        C['n_accepted_derived'] = pd.to_numeric(C.eval_key.map(E['x_n_accepted']), errors='coerce')
        C['state_derived'] = C.eval_key.map(E['x_state'])
    C['accepted_but_no_valid'] = (C.n_accepted > 0) & (C.n_valid == 0)
    return C


def summaries(C: pd.DataFrame, P: pd.DataFrame, case_keys: list, pair_keys: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    vals = [v for v in CASE_VALUES if v in C.columns]
    SC = T5.phase_summary(C, case_keys, vals, COND)
    SP = T5.phase_summary(P, pair_keys, PAIR_VALUES, {'d_glob_pt_rmse'}) if len(P) else pd.DataFrame()
    return SC, SP


def pair_counts(P: pd.DataFrame) -> pd.DataFrame:

    if not len(P):
        return pd.DataFrame()
    a = P.groupby(['effect', 'family', 'layout', 'sigma', 'pattern']).size().rename('pairs_per_level').reset_index()
    t = P.groupby(['effect', 'family', 'layout', 'sigma']).size().rename('pairs_all_levels').reset_index()
    return a.merge(t, on=['effect', 'family', 'layout', 'sigma'])
