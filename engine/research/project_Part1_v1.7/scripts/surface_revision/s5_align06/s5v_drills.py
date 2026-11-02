# Test joint-study table consistency and rejection rules.

from __future__ import annotations

import io

import numpy as np
import pandas as pd

import s5v_common as K
import s5v_effects as EF
import s5v_figs as FIG


def _expect_stop(fn, *a, **kw) -> dict:
    try:
        r = fn(*a, **kw)
    except K.Stop as e:
        return dict(refused=True, code=e.code, msg=e.msg[:240])
    return dict(refused=False, result=str(r)[:200])


def _csv(E: pd.DataFrame, FC: pd.DataFrame) -> io.StringIO:
    b = io.StringIO(); F = FC.copy(); F['sigma'] = F.sigma.map(lambda v: v if isinstance(v, str) else f'{float(v):.1f}')
    pd.concat([E, F], ignore_index=True, sort=False).to_csv(b, index=False); b.seek(0)
    return b


def run_all(C: pd.DataFrame, E: pd.DataFrame, FC: pd.DataFrame, E5: pd.DataFrame, FC5: pd.DataFrame) -> dict:
    res = {}

    i = C.index[(C.surface == 'B_curv') & (C.combo == 'D_r1_p8x4') & (C.sigma == 60.0) & (C.phase_label == 'ph2') & (C.rep == 1) & (C.front_end == 'D0')][0]
    res['record_missing'] = _expect_stop(lambda: EF.validate(EF.build(C.drop(index=i))))
    res['record_duplicated'] = _expect_stop(lambda: EF.validate(EF.build(pd.concat([C, C.loc[[i]]], ignore_index=True))))

    und = E.index[E.state != 'ok'][0]
    res['effect_key_duplicated'] = _expect_stop(EF.validate, pd.concat([E, E.iloc[[0]]], ignore_index=True))
    res['legal_undefined_row_dropped'] = _expect_stop(EF.validate, E.drop(index=und))
    x = E.iloc[[0]].copy(); x['sigma'] = '50.0'
    res['effect_key_extra'] = _expect_stop(EF.validate, pd.concat([E, x], ignore_index=True))

    res['csv_fc_key_missing'] = _expect_stop(EF.load_paired, _csv(E, FC.iloc[1:]))
    res['csv_fc_key_duplicated'] = _expect_stop(EF.load_paired, _csv(E, pd.concat([FC, FC.iloc[[0]]], ignore_index=True)))
    for lab in ('40-0', '40', '45.0'):
        F = FC.copy(); F['sigma'] = F.sigma.map(lambda v: f'{float(v):.1f}'); F.iloc[0, F.columns.get_loc('sigma')] = lab
        res[f'csv_fc_sigma_{lab}'] = _expect_stop(EF.load_paired, _csv(E, F))
    x = E.copy(); x.iloc[0, x.columns.get_loc('sigma')] = '40'
    res['csv_effect_sigma_40'] = _expect_stop(EF.load_paired, _csv(x, FC))

    x = E.copy(); j = x.index[(x.state == 'ok')][0]; x.at[j, 'mean'] = float(x.at[j, 'mean']) + 1e-9
    res['old_value_changed'] = _expect_stop(EF.compare_old, x, FC, E5, FC5)

    Fs = FC.copy(); Fs['sigma'] = Fs.sigma.map(lambda v: f'{float(v):.1f}')
    res['f4_text_sigma_05_path'] = _expect_stop(FIG.f4, Fs, C, save=False)
    Fn = FC.copy(); Fn.iloc[0, Fn.columns.get_loc('d_glob_Ctau_mean8')] = np.nan
    res['f4_one_value_missing'] = _expect_stop(FIG.f4, Fn, C, save=False)
    base = FIG.f4b(FC, save=False)
    Fu = FC.copy(); k = Fu.index[pd.to_numeric(Fu.d_glob_pt_rmse_median_defined_pairs, errors='coerce').notna()][0]; Fu.at[k, 'd_glob_pt_rmse_median_defined_pairs'] = np.nan
    gap = FIG.f4b(Fu, save=False)
    res['f4b_undefined_kept_as_gap'] = dict(refused=False, base=base, with_gap=gap, ok=bool(gap['n_plotted'] == base['n_plotted'] - 1 == gap['n_defined_in_table']))
    want_refused = [k for k in res if k != 'f4b_undefined_kept_as_gap']
    summary = {k: bool(res[k]['refused']) for k in want_refused}
    summary['f4b_undefined_kept_as_gap'] = res['f4b_undefined_kept_as_gap']['ok']
    return dict(results=res, summary=summary, all_ok=all(summary.values()), t=K.now())
