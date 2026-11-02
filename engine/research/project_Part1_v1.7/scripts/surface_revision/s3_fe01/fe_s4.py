# Apply DF to stored layout images and compare layout rankings.

from __future__ import annotations

import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

import fe_common as K

STAGE = 'C_s4'
CACHE = K.OUT / 's4' / 'cache'
R2OUT = K.ROOT / 'outputs' / 'surface_revision' / 's4_real02'
R7OUT = K.ROOT / 'outputs' / 'surface_revision' / 's4_real07'
_S: dict = {}
_TR = ('matched', 'ambiguous', 'duplicate', 'best_idx', 'point_idx')


def _setup():
    if _S:
        return _S
    K.paths_s12(); K.paths_s4()
    import r2_common as K2
    import r2_eval as E2
    import r2_run as R2
    import rdy_decode as DC
    import s3c_dev as DV
    import s4r7_common as H7
    import s4r7_run as R7
    R7._C['ro'] = True
    _S.update(K2=K2, E2=E2, R2=R2, DC=DC, DV=DV, H7=H7, R7=R7)
    return _S


def plan() -> pd.DataFrame:
    m2 = pd.read_csv(R2OUT / 'manifest.csv', low_memory=False)
    p2 = m2[m2.family == 'point'].drop_duplicates('image_key').copy()
    p2['dataset'] = 'S4-REAL-02'; p2['group'] = 'six_noise'
    p2 = p2[['dataset', 'group', 'image_key', 'decode_key', 'eval_key', 'baseline', 'surface', 'platform_surface', 'strategy', 'block', 'layout', 'layout_id', 'phase', 'sigma', 'rep', 'raster_key', 'peak_f32', 'domain_main', 'boundary']]
    m7 = pd.read_csv(R7OUT / 'manifest.csv', low_memory=False); d7 = m7[m7.family == 'point'].copy()
    r7 = pd.read_csv(R7OUT / 'results.csv', low_memory=False)
    c7 = r7[(r7.family == 'point') & (r7.method == 'D0') & (r7.record_type == 'cond')].drop_duplicates('image_key')[['image_key', 'decode_key', 'eval_key', 'result_source']]
    d7 = d7.merge(c7, on='image_key', how='left')
    d7['dataset'] = 'S4-REAL-07'; d7['group'] = 'new_layout'; d7['strategy'] = d7.layout.map(lambda s: 'S0' if s == 'regular' else None)

    view = r7[(r7.family == 'point') & (r7.method == 'D0') & (r7.record_type == 'view')].drop_duplicates('image_key').set_index('image_key')
    if 'arm' in view.columns:
        d7['strategy'] = d7.image_key.map(view['arm']).fillna(d7['strategy'])
    d7 = d7[['dataset', 'group', 'image_key', 'decode_key', 'eval_key', 'baseline', 'surface', 'platform_surface', 'strategy', 'block', 'layout', 'j', 'cand', 'phase', 'sigma', 'raster_key', 'peak_f32', 'boundary', 'set', 'result_source']]
    d7['rep'] = 0
    D = pd.concat([p2, d7], ignore_index=True, sort=False).sort_values(['dataset', 'image_key']).reset_index(drop=True)
    return D


def plan_file() -> Path:
    return K.OUT / 's4' / 'plan_C.csv'


def write_plan() -> dict:
    D = plan(); K.save_csv(D, plan_file())
    s = dict(stage=STAGE, inputs=len(D), by_dataset=D.groupby('dataset').size().to_dict(), by_strategy=D.groupby(['dataset', 'strategy'], dropna=False).size().to_dict(),
             missing_decode_key=int(D.decode_key.isna().sum()), digest=K.digest(sorted(D.image_key), 32), t=K.now())
    s['by_strategy'] = {f'{a}|{b}': v for (a, b), v in s['by_strategy'].items()}
    K.save_json(K.OUT / 's4' / 'plan_C.json', s); return s


def _regress_raw(D0: dict, raw: dict, DC) -> dict:
    cands = np.array(D0['cd']['cands'], np.float64).reshape(-1, 7) if len(D0['cd']['cands']) else np.zeros((0, 7))
    pts, info, tr = D0['arms']['single']; p = np.asarray(pts, np.float64).reshape(-1, 3)
    out = dict(n_cand_new=int(len(cands)), n_cand_stored=int(len(raw['cand_table'])))
    out['cand_equal'] = K.eq_arr(cands, raw['cand_table'])
    out['feats_equal'] = K.eq_arr(np.asarray(D0['feats'], np.float32).reshape(-1, 2), raw['feats'])
    out['select_equal'] = bool(K.eq_arr(np.asarray(D0['code']), raw['sel_code']) and K.eq_arr(np.asarray(D0['fidx']), raw['sel_fidx']))
    out['pts_equal'] = K.eq_arr(p, raw['a0_pts'])
    out['accepted_equal'] = K.eq_arr(DC._accept(pts, tr), raw['a0_accepted'])
    out['trace_equal'] = bool(all(K.eq_arr(np.asarray(tr[k]), raw[f'a0_tr_{k}']) for k in _TR if f'a0_tr_{k}' in raw))
    out['regression_ok'] = bool(out['cand_equal'] and out['feats_equal'] and out['select_equal'] and out['pts_equal'] and out['accepted_equal'] and out['trace_equal'])
    return out


def _load(row: dict) -> dict:
    S = _setup(); K2, R2, R7, H7 = S['K2'], S['R2'], S['R7'], S['H7']
    ik = row['image_key']
    if row['dataset'] == 'S4-REAL-02':
        if 'm2' not in _S:
            _S['m2'] = R2.manifest()
        m2 = _S['m2']; r = m2[(m2.image_key == ik) & (m2.family == 'point')].iloc[0].to_dict()
        inp = R2.inputs_of(r); img = K.load_npz(K2.IMG / f'{ik}.npz')['image']
        raw = K.load_npz(K2.RAW / f"{row['decode_key']}.npz"); m0 = K.load_json(K2.EVL / f"{row['eval_key']}.json")['metrics']
    else:
        c = R7.cond(ik); inp = R7.inputs(c); img, _sha = R7.image(c)
        raw = K.load_npz(H7.DEC7 / f"{row['decode_key']}.npz"); m0 = K.load_json(H7.EVL7 / f"{row['eval_key']}.json")['metrics']
    return dict(inp=inp, img=np.asarray(img, np.uint8), raw=raw, m0=m0)


def work(row: dict) -> dict:
    S = _setup(); K2, E2, DC, DV = S['K2'], S['E2'], S['DC'], S['DV']
    t0 = time.monotonic(); ik = row['image_key']; jp, zp = CACHE / f'{ik}.json', CACHE / f'{ik}.npz'
    if jp.exists() and zp.exists():
        rec = K.load_json(jp)
        if rec.get('status') == 'ok' and rec.get('npz_sha256') == K.sha_file(zp):
            return rec
    L = _load(row); inp, img, raw, m0 = L['inp'], L['img'], L['raw'], L['m0']
    import sp2_common as S2
    rad = float(S2.R_MM); dom = E2.domain('omega1'); surface = row['surface']
    base = {k: row.get(k) for k in ('dataset', 'group', 'image_key', 'surface', 'strategy', 'block', 'layout', 'phase', 'sigma', 'rep', 'j', 'set')}
    K.ledger(STAGE, 'decode_attempt', ik, 'start', rule='B0')
    try:
        D0 = DV.dot_decode(img, np.asarray(inp['binary'], np.float32), 1, [rad], 'B0', 'binding')
    except Exception as e:
        K.ledger(STAGE, 'decode_attempt', ik, 'failed', rule='B0', error=repr(e)[:200]); raise
    K.ledger(STAGE, 'decode_attempt', ik, 'ok', rule='B0')
    reg = _regress_raw(D0, raw, DC)
    if not reg['regression_ok']:
        K.save_json(jp, dict(image_key=ik, status='technical_regression_failed', regression=reg, t=K.now()))
        raise K.Stop('technical', f'{ik}: original D0 not reproduced {reg}')
    mE, _a, _i = E2.evaluate('point', surface, DV.dot_dec(D0, 'single'), inp, K2.BOUNDARY, dom)
    diff = K.metric_diffs(mE, m0, K.CHECK)
    if diff:
        K.save_json(jp, dict(image_key=ik, status='technical_eval_inconsistent', diffs=diff, t=K.now()))
        raise K.Stop('technical', f'{ik}: B0 re-evaluation differs from the stored metrics {diff}')
    K.ledger(STAGE, 'decode_attempt', ik, 'start', rule='DF')
    try:
        D = DV.dot_decode(img, np.asarray(inp['binary'], np.float32), 1, [rad], K.FLOOR_RULE, 'binding')
    except Exception as e:
        K.ledger(STAGE, 'decode_attempt', ik, 'failed', rule='DF', error=repr(e)[:200]); raise
    K.ledger(STAGE, 'decode_attempt', ik, 'ok', rule='DF')
    m, arr, _info = E2.evaluate('point', surface, DV.dot_dec(D, 'single'), inp, K2.BOUNDARY, dom)
    r_d0 = dict(base, rule='D0', source='stored record (B0 regressed, re-evaluation equal)', **{k: m0.get(k) for k in K.METRICS}); r_d0.update(n_candidates=len(D0['cd']['cands']), n_floor=0, a_floor_px2=None)
    r_df = dict(base, rule='DF', source='new decode + evaluation (S3-FE-01)', **{k: m.get(k) for k in K.METRICS}); r_df.update(n_candidates=len(D['cd']['cands']), n_floor=int(D['cd'].get('n_floor', 0)), a_floor_px2=D['cd'].get('a_floor'))
    A = dict(pts=np.asarray(arr['pts'], np.float32), accepted=np.asarray(arr['accepted'], bool), valid=np.asarray(arr['valid'], bool), zr=np.asarray(arr['zr'], np.float32),
             in_global=np.asarray(arr['in_global'], bool), point_correct=np.asarray(arr['point_correct'], bool), point_wrong=np.asarray(arr['point_wrong'], bool),
             point_unconf=np.asarray(arr['point_unconf'], bool), feats=np.asarray(D['feats'], np.float32).reshape(-1, 2))
    h = K.save_npz(zp, **A)
    rec = dict(image_key=ik, dataset=row['dataset'], status='ok', regression=reg, eval_consistency_ok=True, rows=[r_d0, r_df], npz_sha256=h, attempts=2, seconds=round(time.monotonic() - t0, 2), t=K.now())
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


def run(workers: int = 6, limit: int | None = None, keys: list | None = None) -> dict:
    t0 = time.time(); D = pd.read_csv(plan_file(), low_memory=False)
    if keys:
        D = D[D.image_key.isin(keys)]
    need = D[~D.image_key.isin(done_keys())]
    if limit:
        need = need.head(limit)
    rows = need.to_dict('records'); fails, ok = [], 0; it = iter(rows); inflight = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        def submit_next():
            r = next(it, None)
            if r is not None:
                inflight[ex.submit(_worker, r)] = r['image_key']
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
    st = dict(stage=STAGE, planned=len(rows), ok=ok, failures=fails, done_total=len(done_keys()), plan_total=len(D), ledger=K.ledger_counts(STAGE), seconds=round(time.time() - t0, 1), t=K.now())
    K.save_json(K.OUT / 's4' / 'state_C.json', st)
    if fails:
        raise K.Stop('technical', f'{len(fails)} technical failures; stopped submitting: {fails[:2]}')
    return st


def aggregate() -> dict:
    D = pd.read_csv(plan_file(), low_memory=False)
    recs = [K.load_json(CACHE / f'{ik}.json') for ik in D.image_key if (CACHE / f'{ik}.json').exists()]
    okr = [r for r in recs if r.get('status') == 'ok']
    R = pd.DataFrame([x for r in okr for x in r['rows']]); K.save_csv(R, K.RESULTS / 'C_s4_cases.csv')
    PM = ('glob_Ctau', 'glob_Cvalid', 'glob_recovery', 'glob_pt_rmse', 'glob_pt_p95', 'id_correct', 'id_wrong', 'id_unconf', 'n_accepted', 'n_candidates', 'n_ambiguous')
    out = []
    for ik, g in R.groupby('image_key'):
        a = g[g.rule == 'DF']; b = g[g.rule == 'D0']
        if len(a) != 1 or len(b) != 1:
            continue
        a = a.iloc[0]; b = b.iloc[0]; e = {k: b[k] for k in ('dataset', 'group', 'image_key', 'surface', 'strategy', 'block', 'layout', 'phase', 'sigma', 'rep')}
        for mname in PM:
            x, y = pd.to_numeric(a[mname], errors='coerce'), pd.to_numeric(b[mname], errors='coerce')
            e[f'd_{mname}'] = (float(x) - float(y)) if (pd.notna(x) and pd.notna(y)) else np.nan; e[f'{mname}_DF'] = x; e[f'{mname}_D0'] = y
        out.append(e)
    Pd = pd.DataFrame(out); K.save_csv(Pd, K.RESULTS / 'C_s4_paired.csv')
    cells = []
    for k, g in R.groupby(['dataset', 'strategy', 'surface', 'sigma', 'rule'], dropna=False):
        c = dict(zip(['dataset', 'strategy', 'surface', 'sigma', 'rule'], k), n_images=len(g), n_blocks=int(g.block.nunique(dropna=True)), n_phases=int(g.phase.nunique()))
        for mname in ('glob_Ctau', 'glob_Cvalid', 'glob_recovery', 'glob_pt_rmse', 'glob_pt_p95', 'glob_surf_rmse'):
            v = pd.to_numeric(g[mname], errors='coerce'); c[f'{mname}_mean'] = float(v.mean()) if v.notna().any() else np.nan; c[f'{mname}_median'] = float(v.median()) if v.notna().any() else np.nan
        c.update(id_correct=int(pd.to_numeric(g.id_correct, errors='coerce').sum()), id_wrong=int(pd.to_numeric(g.id_wrong, errors='coerce').sum()),
                 id_unconf=int(pd.to_numeric(g.id_unconf, errors='coerce').sum()), n_no_output=int((pd.to_numeric(g.n_accepted, errors='coerce') == 0).sum()))
        cells.append(c)
    Cc = pd.DataFrame(cells); K.save_csv(Cc, K.RESULTS / 'C_s4_cells.csv')
    s = dict(stage=STAGE, records=len(recs), ok=len(okr), not_ok=[r.get('image_key') for r in recs if r.get('status') != 'ok'][:20], rows=len(R), paired=len(Pd), cells=len(Cc),
             regression_ok=int(sum(1 for r in okr if r['regression']['regression_ok'])), by_dataset=R[R.rule == 'DF'].groupby('dataset').size().to_dict(), t=K.now())
    K.save_json(K.RESULTS / 'C_s4_summary.json', s); return s
