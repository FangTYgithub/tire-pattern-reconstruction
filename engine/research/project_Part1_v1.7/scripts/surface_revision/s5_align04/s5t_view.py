# Build validated joint-study views and same-image front-end pairs.

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import s5t_common as K
import s5r_gates as G
import s5r_store as S

KEYS = ['combo', 'surface', 'phase_label', 'sigma', 'rep', 'method']
COUNTS = ['n_out', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'id_denominator', 'glob_id_correct', 'glob_id_wrong',
          'glob_id_unconf', 'loc_id_correct', 'loc_id_wrong', 'loc_id_unconf']
CANON = KEYS + ['image_key', 'content_sha', 'record_complete', 'state'] + COUNTS + ['glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'loc_Ctau', 'loc_Cvalid', 'loc_cov_interp',
                                                                                     'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'loc_surf_rmse', 'sup_surf_rmse']
MET = ['n_out', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'id_denominator', 'glob_id_correct', 'glob_id_wrong', 'glob_id_unconf',
       'loc_id_correct', 'loc_id_wrong', 'loc_id_unconf', 'glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'loc_Ctau', 'loc_Cvalid', 'loc_cov_interp', 'glob_pt_rmse', 'glob_pt_p95',
       'loc_pt_rmse', 'loc_pt_p95', 'glob_pt_max', 'glob_surf_rmse', 'glob_surf_p95', 'loc_surf_rmse', 'glob_n_interp', 'loc_n_interp', 'glob_recovery', 'loc_recovery',
       'glob_n_expected', 'glob_n_recovered', 'n_candidates', 'n_selected', 'n_matched', 'n_ambiguous', 'n_detected', 'n_status_segment_unknown']


def _isint(x) -> bool:
    if isinstance(x, (bool, np.bool_)) or x is None or isinstance(x, str):
        return False
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f) and f == int(f))


def strict_gate(R: pd.DataFrame, expected: set, mode: str, view: str) -> dict:
    reasons = []

    def bad(check, msg, **kw):
        reasons.append(dict(check=check, msg=msg, **K.clean(kw)))
    miss = [c for c in CANON if c not in R.columns]
    if miss:
        bad('columns', 'canonical columns missing (a missing column is not a legal undefined value)', missing=miss)
        return dict(ok=False, n_reasons=len(reasons), by_check=dict(Counter(r['check'] for r in reasons)), reasons=reasons, view=view, mode=mode, n_rows=len(R))
    for i, x in zip(R.index, R.rep):
        if not _isint(x):
            bad('types', 'rep must be an integral number (1.5, NaN, text refused before any conversion)', row=int(i), value=str(x))
    for i, x in zip(R.index, R.sigma):
        if K.K3.num(x) is None:
            bad('types', 'sigma not a finite number', row=int(i), value=str(x))
    for c in COUNTS:
        for i, x in zip(R.index, R[c]):
            if not _isint(x) or float(x) < 0:
                bad('types', f'{c} must be a finite non-negative integer', row=int(i), value=str(x))
    if reasons:
        return dict(ok=False, n_reasons=len(reasons), by_check=dict(Counter(r['check'] for r in reasons)), reasons=reasons[:40], view=view, mode=mode, n_rows=len(R))
    T = R.copy(); T['rep'] = T.rep.astype(float).astype(int); T['sigma'] = T.sigma.astype(float)
    g = G.gate_results(T, expected, mode, view)
    return g


def store_rows(store, rows: list, pr: dict, methods_of) -> tuple[list, list]:

    out, probs = [], []
    for row in rows:
        k = row['key']; iid = S.image_identity(row['f'], pr); st, why = store.status('image', k, iid)
        if st != 'reuse':
            probs.append(dict(key=k, stage='image', status=st, why=why)); continue
        isha = store.content('image', k)
        for method in methods_of(row):
            dk = f'{k}__{method}'
            did = S.decode_identity(iid, isha, method, S.method_params(method, row['f']['peak_f32'], row.get('radius_mm')))
            st, why = store.status('decode', dk, did)
            if st != 'reuse':
                probs.append(dict(key=dk, stage='decode', status=st, why=why)); continue
            eid = S.eval_identity(store.content('decode', dk), pr); st, why = store.status('eval', dk, eid)
            if st != 'reuse':
                probs.append(dict(key=dk, stage='eval', status=st, why=why)); continue
            J = K.load_json(store._p('eval', dk)); m = J['meta'].get('metrics') or {}; sup = J['meta'].get('support') or {}
            r = dict(combo=row['combo'], surface=row['surface'], phase_label=row['phase_label'], sigma=row['sigma'], rep=row['rep'], method=method, image_key=row['manifest_key'],
                     image_content=isha, image_identity=iid, decode_identity=did, eval_identity=eid, content_sha=J['npz_sha256'], record_complete=bool(J.get('complete') is True),
                     state=m.get('state'), sup_surf_rmse=sup.get('sup_surf_rmse'), glob_n_supported_nodes=sup.get('glob_n_supported_nodes'),
                     eval_record=K.rel(store._p('eval', dk)), decode_record=K.rel(store._p('decode', dk)), image_record=K.rel(store._p('image', k)),
                     **{c: m.get(c) for c in MET})
            out.append(r)
    return out, probs


def risks(df: pd.DataFrame) -> pd.DataFrame:
    c, w, u = (pd.to_numeric(df[x], errors='coerce') for x in ('id_correct', 'id_wrong', 'id_unconf'))
    df = df.copy()
    df['risk_wrong_den'] = c + w; df['risk_wrong'] = np.where(c + w > 0, w / (c + w).where(c + w > 0), np.nan)
    df['unconf_den'] = c + w + u; df['unconf_share'] = np.where(c + w + u > 0, u / (c + w + u).where(c + w + u > 0), np.nan)
    return df


def pairs(df: pd.DataFrame) -> dict:

    key = ['combo', 'surface', 'phase_label', 'sigma', 'rep']
    d0 = df[df.method == 'D0']; dfm = df[df.method == 'DF']
    c0, c1 = Counter(map(tuple, d0[key].values.tolist())), Counter(map(tuple, dfm[key].values.tolist()))
    if any(v > 1 for v in c0.values()) or any(v > 1 for v in c1.values()):
        raise K.Stop('refused', 'more than one D0 or DF record per observation key - pairing refused')
    M = d0.merge(dfm, on=key, suffixes=('_D0', '_DF'), how='inner', validate='one_to_one')
    for c in ('image_key', 'image_content'):
        if f'{c}_D0' in M and (M[f'{c}_D0'] != M[f'{c}_DF']).any():
            raise K.Stop('refused', f'D0 and DF of one observation consumed different images ({c})')
    rows = []
    for r in M.to_dict('records'):
        e = {k: r[k] for k in key}
        for m in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'n_accepted', 'n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'glob_pt_rmse', 'glob_pt_p95', 'sup_surf_rmse'):
            a, b = K.K3.num(r.get(f'{m}_D0')), K.K3.num(r.get(f'{m}_DF'))
            e[f'{m}_D0'], e[f'{m}_DF'] = a, b; e[f'd_{m}'] = (b - a) if (a is not None and b is not None) else None
        rows.append(e)
    return dict(n_pairs=len(M), n_d0=len(d0), n_df=len(dfm), unpaired_d0=len(d0) - len(M), unpaired_df=len(dfm) - len(M), rows=rows)


def expected_methods(keys) -> set:
    return {(c, s, p, float(sg), int(r), m) for (c, s, p, sg, r) in keys for m in (['D0', 'DF'] if c in K.K3.DOT9 else ['P4'])}
