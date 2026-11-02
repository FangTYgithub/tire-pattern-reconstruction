# Apply DF to size-spacing images and compare results with the D0 records.

from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

import fe_common as K

STAGE = 'A_s12'
S12 = K.ROOT / 'outputs' / 'surface_revision' / 's12_resource01'
C4 = K.ROOT / 'outputs' / 'surface_revision' / 's3_code04' / 'results'
CACHE = K.OUT / 's12' / 'cache'
REGIMES = ('resource_cap_11695', 'fixed_peak_1')
_S: dict = {}


def _setup():
    if _S:
        return _S
    K.paths_s12()
    import c4_common as K4
    import c4_s12 as C4S
    import rdy_platform as P
    import s12_case as SC12
    import s3c_dev as DV
    _S.update(K4=K4, C4S=C4S, P=P, SC12=SC12, DV=DV)
    return _S


def plan() -> pd.DataFrame:

    V = pd.read_csv(S12 / 'results' / 'views.csv', low_memory=False)
    V = V[(V.surface_class == 'main') & (V.family == 'point') & V.regime.isin(REGIMES)]
    U = pd.read_csv(S12 / 'inputs.csv', low_memory=False).set_index('input_key')
    g = V.groupby('input_key')
    rows = []
    for ik, grp in g:
        u = U.loc[ik].to_dict(); r0 = grp.iloc[0]
        rows.append(dict(input_key=ik, image_record=r0.image_record, eval_record=r0.eval_record, views=';'.join(sorted(set(grp.view_id.astype(str)))),
                         view_regimes=';'.join(sorted(set(grp.regime))), sections=';'.join(sorted(set(grp.section))), surface=u['surface'], platform_surface=u['platform_surface'],
                         layout=u['layout'], size_mm=float(u['size_mm']), phase_label=u['phase_label'], sigma=float(u['sigma']), rep=int(u['rep']), peak_target=float(u['peak_target']),
                         peak_f32=u['peak_f32'], raster_key=u['raster_key'], raster_source=u['raster_source'], template_cand_id=u['template_cand_id'], regimes=u['regimes'],
                         family=u['family'], decode_key=u['decode_key'], eval_key=u['eval_key']))
    D = pd.DataFrame(rows).sort_values('input_key').reset_index(drop=True)
    have = set()
    p = C4 / 'C1_dots_cases.csv'
    if p.exists():
        C = pd.read_csv(p, low_memory=False); have = set(C[C.rule == 'DF'].input_key)
    D['df_in_s3_code04'] = D.input_key.isin(have)
    return D


def plan_file() -> Path:
    return K.OUT / 's12' / 'plan_A.csv'


def write_plan() -> dict:
    D = plan(); K.save_csv(D, plan_file())
    s = dict(stage=STAGE, inputs=len(D), with_old_df=int(D.df_in_s3_code04.sum()), by_layout=D.groupby('layout').size().to_dict(), by_sigma=D.groupby('sigma').size().to_dict(),
             by_view_regimes=D.groupby('view_regimes').size().to_dict(), digest=K.digest(sorted(D.input_key), 32), t=K.now())
    K.save_json(K.OUT / 's12' / 'plan_A.json', s); return s


def work(row: dict) -> dict:
    S = _setup(); K4, C4S, P, SC12, DV = S['K4'], S['C4S'], S['P'], S['SC12'], S['DV']
    t0 = time.monotonic(); ik = row['input_key']
    jp, zp = CACHE / f'{ik}.json', CACHE / f'{ik}.npz'
    if jp.exists() and zp.exists():
        rec = K.load_json(jp)
        if rec.get('status') == 'ok' and rec.get('npz_sha256') == K.sha_file(zp):
            return rec
    u = C4S._u(dict(row))
    R = SC12.raster_of(u); binary = np.asarray(R['binary'], np.float32); ctx = SC12.context(u, R, u['peak_target'])
    img = K.load_npz(K.ROOT / str(u['image_record']).replace('.json', '.npz'))['image']
    Z = K.load_npz(K.ROOT / str(u['eval_record']).replace('.json', '.npz')); J = K.load_json(K.ROOT / u['eval_record']); m0 = J['metrics']
    mesh = P.mesh(u['surface']); sid = K4.RK.SURF_MAP[u['surface']]
    base = {k: u[k] for k in ('input_key', 'sections', 'view_regimes', 'surface', 'layout', 'size_mm', 'phase_label', 'sigma', 'rep', 'peak_target')}

    K.ledger(STAGE, 'decode_attempt', ik, 'start', rule='B0')
    try:
        D0 = DV.dot_decode(img, binary, 1, [u['size_mm']], 'B0', 'binding')
    except Exception as e:
        K.ledger(STAGE, 'decode_attempt', ik, 'failed', rule='B0', error=repr(e)[:200]); raise
    K.ledger(STAGE, 'decode_attempt', ik, 'ok', rule='B0')
    reg = C4S._regress(D0, Z)
    if not reg['regression_ok']:
        K.save_json(jp, dict(input_key=ik, status='technical_regression_failed', regression=reg, t=K.now()))
        raise K.Stop('technical', f'{ik}: original D0 not reproduced {reg}')
    mE, _ = DV.evaluate(sid, 'point', DV.dot_dec(D0, 'single'), ctx, mesh)
    diff = K.metric_diffs(mE, m0, K.CHECK)
    if diff:
        K.save_json(jp, dict(input_key=ik, status='technical_eval_inconsistent', diffs=diff, t=K.now()))
        raise K.Stop('technical', f'{ik}: B0 re-evaluation differs from the stored metrics {diff}')

    K.ledger(STAGE, 'decode_attempt', ik, 'start', rule='DF')
    try:
        D = DV.dot_decode(img, binary, 1, [u['size_mm']], K.FLOOR_RULE, 'binding')
    except Exception as e:
        K.ledger(STAGE, 'decode_attempt', ik, 'failed', rule='DF', error=repr(e)[:200]); raise
    K.ledger(STAGE, 'decode_attempt', ik, 'ok', rule='DF')
    m, arr = DV.evaluate(sid, 'point', DV.dot_dec(D, 'single'), ctx, mesh)
    r_d0 = dict(base, rule='D0', source='stored record (B0 regressed, re-evaluation equal)', **{k: m0.get(k) for k in K.METRICS})
    r_d0.update(n_candidates=len(D0['cd']['cands']), n_floor=0, a_floor_px2=None)
    r_df = dict(base, rule='DF', source='new decode + evaluation (S3-FE-01)', **{k: m.get(k) for k in K.METRICS})
    r_df.update(n_candidates=len(D['cd']['cands']), n_floor=int(D['cd'].get('n_floor', 0)), a_floor_px2=D['cd'].get('a_floor'))
    rows = [r_d0, r_df]
    A = dict(pts=np.asarray(arr['pts'], np.float32), accepted=np.asarray(arr['accepted'], bool), valid=np.asarray(arr['valid'], bool), zr=np.asarray(arr['zr'], np.float32),
             in_global=np.asarray(arr['in_global'], bool), in_local=np.asarray(arr['in_local'], bool), point_correct=np.asarray(arr['point_correct'], bool),
             point_wrong=np.asarray(arr['point_wrong'], bool), point_unconf=np.asarray(arr['point_unconf'], bool), feats=np.asarray(D['feats'], np.float32).reshape(-1, 2))
    h = K.save_npz(zp, **A)
    rec = dict(input_key=ik, status='ok', regression=reg, eval_consistency_ok=True, rows=rows, npz_sha256=h, attempts=2, seconds=round(time.monotonic() - t0, 2),
               image_record=u['image_record'], eval_record=u['eval_record'], t=K.now())
    K.save_json(jp, rec); return rec


def _worker(row):
    K.install_guard()
    try:
        return work(row)
    except Exception as e:
        raise RuntimeError(f'{type(e).__name__}: {e}')


def done_keys() -> set:
    if not CACHE.exists():
        return set()
    return {p.stem for p in CACHE.glob('*.json') if K.load_json(p).get('status') == 'ok'}


def run(workers: int = 14, limit: int | None = None, keys: list | None = None) -> dict:
    t0 = time.time(); D = pd.read_csv(plan_file(), low_memory=False)
    if keys:
        D = D[D.input_key.isin(keys)]
    done = done_keys(); need = D[~D.input_key.isin(done)]
    if limit:
        need = need.head(limit)
    rows = need.to_dict('records'); fails, ok = [], 0; it = iter(rows); inflight = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        def submit_next():
            r = next(it, None)
            if r is not None:
                inflight[ex.submit(_worker, r)] = r['input_key']
            return r is not None
        for _ in range(workers * 2):
            if not submit_next():
                break
        while inflight:
            dn, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for f in dn:
                key = inflight.pop(f)
                try:
                    f.result(); ok += 1
                except Exception as e:
                    fails.append(dict(key=key, error=traceback.format_exception_only(type(e), e)[-1][:400]))
            if fails:
                for f in list(inflight):
                    if f.cancel():
                        inflight.pop(f)
                continue
            while len(inflight) < workers * 2 and submit_next():
                pass
            if (ok + len(fails)) % 100 == 0:
                K.save_json(K.OUT / 's12' / 'progress_A.json', dict(planned=len(rows), ok=ok, failures=len(fails), seconds=round(time.time() - t0, 1), t=K.now()))
    st = dict(stage=STAGE, planned=len(rows), ok=ok, failures=fails, done_total=len(done_keys()), plan_total=len(D), ledger=K.ledger_counts(STAGE), seconds=round(time.time() - t0, 1), t=K.now())
    K.save_json(K.OUT / 's12' / 'state_A.json', st)
    if fails:
        raise K.Stop('technical', f'{len(fails)} technical failures; stopped submitting: {fails[:2]}')
    return st


# Aggregate paired front-end results and compare parameter rankings by condition.
def aggregate() -> dict:
    D = pd.read_csv(plan_file(), low_memory=False)
    recs = [K.load_json(CACHE / f'{ik}.json') for ik in D.input_key if (CACHE / f'{ik}.json').exists()]
    okr = [r for r in recs if r.get('status') == 'ok']
    rows = [dict(x, dataset='S12') for r in okr for x in r['rows']]
    R = pd.DataFrame(rows)

    V = pd.read_csv(S12 / 'results' / 'views.csv', low_memory=False)
    V = V[(V.surface_class == 'main') & (V.family == 'point') & V.regime.isin(REGIMES)][['view_id', 'input_key', 'section', 'regime', 'surface', 'layout', 'size_mm', 'phase_label', 'sigma', 'rep']]
    X = V.merge(R.drop(columns=['sections', 'view_regimes', 'surface', 'layout', 'size_mm', 'phase_label', 'sigma', 'rep']), on='input_key', how='left')
    K.save_csv(R, K.RESULTS / 'A_s12_cases.csv'); K.save_csv(X, K.RESULTS / 'A_s12_views.csv')

    PM = ('glob_Ctau', 'glob_Cvalid', 'glob_recovery', 'glob_pt_rmse', 'glob_pt_p95', 'id_correct', 'id_wrong', 'id_unconf', 'n_accepted', 'n_candidates')
    out = []
    for ik, g in R.groupby('input_key'):
        a = g[g.rule == 'DF']; b = g[g.rule == 'D0']
        if len(a) != 1 or len(b) != 1:
            continue
        a = a.iloc[0]; b = b.iloc[0]
        e = {k: b[k] for k in ('input_key', 'surface', 'layout', 'size_mm', 'phase_label', 'sigma', 'rep', 'view_regimes')}
        for m in PM:
            x, y = pd.to_numeric(a[m], errors='coerce'), pd.to_numeric(b[m], errors='coerce')
            e[f'd_{m}'] = (float(x) - float(y)) if (pd.notna(x) and pd.notna(y)) else np.nan
            e[f'{m}_DF'] = x; e[f'{m}_D0'] = y
        out.append(e)
    Pd = pd.DataFrame(out); K.save_csv(Pd, K.RESULTS / 'A_s12_paired.csv')

    cells = []
    for k, g in X.groupby(['section', 'regime', 'surface', 'layout', 'size_mm', 'sigma', 'rule']):
        ph = g.groupby('phase_label'); c = dict(zip(['section', 'regime', 'surface', 'layout', 'size_mm', 'sigma', 'rule'], k), n_images=len(g), n_phases=int(g.phase_label.nunique()))
        for m in ('glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'glob_recovery', 'glob_pt_rmse', 'glob_pt_p95', 'glob_surf_rmse', 'loc_Ctau'):
            pv = ph[m].apply(lambda s: pd.to_numeric(s, errors='coerce').mean())
            c[f'{m}_mean'] = float(pv.mean()) if pv.notna().any() else np.nan; c[f'{m}_min'] = float(pv.min()) if pv.notna().any() else np.nan
            c[f'{m}_max'] = float(pv.max()) if pv.notna().any() else np.nan; c[f'{m}_phase'] = ';'.join('nan' if pd.isna(x) else f'{x:.5f}' for x in pv.values)
        c.update(id_correct=int(pd.to_numeric(g.id_correct, errors='coerce').sum()), id_wrong=int(pd.to_numeric(g.id_wrong, errors='coerce').sum()),
                 id_unconf=int(pd.to_numeric(g.id_unconf, errors='coerce').sum()), n_no_output=int((pd.to_numeric(g.n_accepted, errors='coerce') == 0).sum()),
                 n_wrong_images=int((pd.to_numeric(g.id_wrong, errors='coerce') > 0).sum()))
        cells.append(c)
    Cc = pd.DataFrame(cells); K.save_csv(Cc, K.RESULTS / 'A_s12_cells.csv')

    xc = dict(overlap=0, equal_Ctau=0, max_abs_diff=None)
    p = C4 / 'C1_dots_cases.csv'
    if p.exists():
        C = pd.read_csv(p, low_memory=False); C = C[C.rule == 'DF'].drop_duplicates('input_key').set_index('input_key')
        M = R[R.rule == 'DF'].set_index('input_key').join(C[['glob_Ctau', 'id_correct', 'id_wrong', 'id_unconf', 'n_candidates']], rsuffix='_c4', how='inner')
        if len(M):
            d = (pd.to_numeric(M.glob_Ctau, errors='coerce') - pd.to_numeric(M.glob_Ctau_c4, errors='coerce')).abs()
            xc = dict(overlap=int(len(M)), equal_Ctau=int((d <= 1e-9).sum()), max_abs_diff=float(d.max()), equal_candidates=int((M.n_candidates == M.n_candidates_c4).sum()))
    s = dict(stage=STAGE, records=len(recs), ok=len(okr), not_ok=[r['input_key'] for r in recs if r.get('status') != 'ok'][:20], rows=len(R), views=len(X), paired=len(Pd), cells=len(Cc),
             regression_ok=int(sum(1 for r in okr if r['regression']['regression_ok'])), crosscheck_s3_code04=xc, t=K.now())
    K.save_json(K.RESULTS / 'A_s12_summary.json', s); return s
