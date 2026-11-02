# Exercise layout scheduling, integrity checks and failure handling.

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

import r2_common as K

TMP = K.RUNDIR / 'drill_tmp'


def _M():
    return pd.read_csv(K.OUT / 'manifest.csv', keep_default_na=False, low_memory=False, float_precision='round_trip')


def missing_column():
    import r2_manifest as RM
    RM.validate(_M().drop(columns=['sigma0_ref_key']))


def duplicate():
    import r2_manifest as RM
    M = _M(); RM.validate(pd.concat([M, M.iloc[[5]]], ignore_index=True))


def wrong_key():
    import r2_eval as E
    import r2_manifest as RM
    M = _M(); args = (K.run_load()['bind']['m3_digest'], E.domain('omega1')['hash'], E.domain('omega_star')['hash'])
    assert RM.rekey(M, *args)['ok'], 'the unaltered manifest must pass'
    i = int(np.flatnonzero((M.sigma == 40.0).to_numpy())[3]); j = int(np.flatnonzero((M.sigma == 60.0).to_numpy())[3])
    M.loc[i, 'image_key'] = M.loc[j, 'image_key']
    try:
        RM.rekey(M, *args)
    except K.Stop as e:
        assert ' 1 rows ' in f' {e.msg} ', f'expected exactly one refused row: {e.msg}'
        raise


def version_change():
    import r2_main as RMAIN
    fp = dict(K.run_load()['bind']['fingerprint_digests']); fp['decode'] = 'f' * 32
    RMAIN.preflight(current=fp)


def _fake_eval(dirp, key, state, n_valid, ctau, rmse, family='point'):
    m = dict(state=state, n_valid=n_valid, n_accepted=n_valid, glob_Ctau=ctau, loc_Ctau=ctau, glob_Cvalid=ctau, loc_Cvalid=ctau, glob_cov_interp=ctau, loc_cov_interp=ctau,
             glob_pt_rmse=rmse, glob_surf_rmse=rmse, loc_pt_rmse=rmse, loc_surf_rmse=rmse, glob_n_valid=n_valid, loc_n_valid=n_valid, id_correct=n_valid, id_wrong=0, id_unconf=0,
             id_denominator=n_valid)
    h = K.save_npz(dirp / f'{key}.npz', pts=np.zeros((n_valid, 3), np.float32), valid=np.ones(n_valid, bool))
    ex = dict(source_identity=dict(correct=n_valid, wrong=0, undeterminable=0, n_reliable=n_valid)) if family == 'point' else {}
    K.save_json(dirp / f'{key}.json', dict(identity=key, metrics=m, extras=ex, complete=True, npz_sha256=h))


def legal_no_output():
    import r2_analysis as A
    M = _M(); reg = K.load_json(K.OUT / 'layouts.json')
    S = M[(M.baseline == 'R8x4') & (M.surface == 'B_curv') & (M.sigma == 100.0) & (M.phase == 'H0') & M.strategy.isin(['S0', 'S1'])].head(2)
    d = TMP / 'eval'; shutil.rmtree(TMP, ignore_errors=True); d.mkdir(parents=True)
    _fake_eval(d, S.iloc[0].eval_key, 'no_output', 0, 0.0, None); _fake_eval(d, S.iloc[1].eval_key, 'ok', 120, 0.31, 0.05)
    C = A.case_rows(S, reg, eval_dir=d)
    r0 = C.iloc[0]
    assert r0.state == 'no_output' and r0.glob_Ctau == 0.0 and pd.isna(r0.glob_pt_rmse) and pd.isna(r0.glob_surf_rmse) and pd.isna(r0.wrong_rate), 'no-output semantics'
    assert C.iloc[1].glob_Ctau == 0.31 and C.iloc[1].wrong_rate == 0.0
    shutil.rmtree(TMP, ignore_errors=True)
    return dict(no_output_row=dict(Ctau=float(r0.glob_Ctau), pt_rmse=None, wrong_rate=None))


def partial_batch():

    import r2_analysis as A
    M = _M(); reg = K.load_json(K.OUT / 'layouts.json')
    d = TMP / 'eval_partial'; shutil.rmtree(TMP, ignore_errors=True); d.mkdir(parents=True)
    try:
        for ek in M.eval_key.unique()[:2]:
            for suf in ('.json', '.npz'):
                if (K.EVL / f'{ek}{suf}').exists():
                    shutil.copy2(K.EVL / f'{ek}{suf}', d / f'{ek}{suf}')
        A.case_rows(M, reg, eval_dir=d, require_complete=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def pairing():
    import r2_analysis as A
    M = _M(); C = M.copy(); C['record'] = 'complete'; C['state'] = 'ok'
    rng = {k: (int(k[-6:], 16) % 1000) / 1000.0 for k in C.image_key.unique()}
    for m in A.PAIR_METRICS:
        C[m] = C.image_key.map(rng)
    P = A.paired(C); Pp = P[P.table_type == 'paired']
    assert (Pp.n_pairs == 32).all(), 'every contrast: 8 blocks x 4 phases'
    s43 = Pp[(Pp.contrast == 'S4-S3') & (Pp.metric == 'glob_Ctau')]
    same = M[M.strategy.isin(['S3', 'S4'])].pivot_table(index=['baseline', 'surface', 'sigma', 'block', 'phase'], columns='strategy', values='image_key', aggfunc='first')
    n_same = int((same.S3 == same.S4).sum())
    assert int(s43.n_same_image_pairs.sum()) == n_same, 'shared images counted as identical pairs'
    assert (s43[s43.n_same_image_pairs == 32]['main'] == 0.0).all(), 'identical layouts give zero'
    a0 = P[(P.table_type == 'absolute') & (P.contrast == 'S0') & (P.metric == 'glob_Ctau')]
    assert (a0.n_views == 4).all(), 'S0 = one shared control per phase (4 views), never 8 x 4'
    s10 = Pp[(Pp.contrast == 'S1-S0') & (Pp.metric == 'glob_Ctau')].iloc[0]
    return dict(n_paired_rows=int(len(Pp)), pairs_per_contrast=32, n_same_S3_S4=n_same, s0_views_per_arm=4, example_S1_S0_main=float(s10['main']))


def interrupt():
    shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
    key = 'R2E-drill'; p = TMP / f'{key}.json'
    res = {}
    res['missing'] = K.complete(p, key)
    K.save_json(p, dict(identity=key, complete=True, npz_sha256='0' * 64)); res['json_without_npz'] = K.complete(p, key)
    h = K.save_npz(p.with_suffix('.npz'), a=np.arange(3)); K.save_json(p, dict(identity=key, complete=False, npz_sha256=h)); res['flag_false'] = K.complete(p, key)
    K.save_json(p, dict(identity='R2E-other', complete=True, npz_sha256=h)); res['identity_mismatch'] = K.complete(p, key)
    K.save_json(p, dict(identity=key, complete=True, npz_sha256=h)); res['valid'] = K.complete(p, key)
    np.savez_compressed(open(p.with_suffix('.npz'), 'wb'), a=np.arange(4)); res['npz_tampered'] = K.complete(p, key)
    (TMP / f'{key}.json.tmp999').write_text('{', encoding='utf-8'); res['tmp_leftover_ignored'] = not K.complete(TMP / f'{key}.json.tmp999', key)
    B = K.Budget(); B.f = TMP / 'budget.json'; B.a = TMP / 'attempts.jsonl'
    n1 = B.reserve('k1', 'drill'); n2 = B.reserve('k2', 'drill'); B2 = K.Budget(); B2.f, B2.a = B.f, B.a; n3 = B2.reserve('k3', 'drill')
    res['budget_sequence'] = [n1, n2, n3]
    ok = (res == dict(missing=False, json_without_npz=False, flag_false=False, identity_mismatch=False, valid=True, npz_tampered=False, tmp_leftover_ignored=True, budget_sequence=[1, 2, 3]))
    shutil.rmtree(TMP, ignore_errors=True)
    assert ok, f'completeness / ledger semantics {res}'
    return res


def budget_cap():
    shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
    B = K.Budget(); B.f = TMP / 'budget.json'; B.a = TMP / 'attempts.jsonl'
    K.save_json(B.f, dict(cap=K.CAP_RENDER, used=K.CAP_RENDER, by_purpose={}))
    try:
        B.reserve('k', 'drill')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def crash():
    raise RuntimeError('deliberate uncaught exception (drill)')


DRILLS = dict(missing_column=(missing_column, 2), duplicate=(duplicate, 2), wrong_key=(wrong_key, 2), version_change=(version_change, 2), legal_no_output=(legal_no_output, 0),
              partial_batch=(partial_batch, 2), pairing=(pairing, 0), interrupt=(interrupt, 0), budget_cap=(budget_cap, 2), crash=(crash, 1))
