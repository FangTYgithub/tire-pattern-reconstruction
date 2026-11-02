# Apply DF to the additional dot-radius cases and compare size rankings.

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

import fe_common as K

STAGE = 'B_size02'
S02 = K.HOME / 'outputs' / 's1_size02'
CACHE = K.OUT / 'size02' / 'cache'
_S: dict = {}


def _setup():
    if _S:
        return _S
    K.paths_s12()
    import c4_s12 as C4S
    import rdy_platform as RP
    import s1r_common as S1
    import s1r_eval as E
    import s3c_dev as DV
    _S.update(C4S=C4S, RP=RP, S1=S1, E=E, DV=DV)
    return _S


def readpack(p: Path) -> tuple[dict, dict]:
    with np.load(p, allow_pickle=False) as z:
        return json.loads(str(z['_meta'])), {k: z[k] for k in z.files if k != '_meta'}


def plan() -> pd.DataFrame:
    m = pd.read_csv(S02 / 'manifest.csv', low_memory=False)
    m['pack'] = [str(S02 / 'cases' / f'{ik}.npz') for ik in m.input_key]
    m['pack_exists'] = [Path(p).exists() for p in m.pack]
    return m.sort_values('input_key').reset_index(drop=True)


def plan_file() -> Path:
    return K.OUT / 'size02' / 'plan_B.csv'


def write_plan() -> dict:
    D = plan(); K.save_csv(D, plan_file())
    s = dict(stage=STAGE, inputs=len(D), packs_present=int(D.pack_exists.sum()), by_regime_sigma=D.groupby(['regime', 'sigma']).size().to_dict(),
             by_size=D.groupby('size_mm').size().to_dict(), digest=K.digest(sorted(D.input_key), 32), t=K.now())
    s['by_regime_sigma'] = {f'{a}|{int(b)}': v for (a, b), v in s['by_regime_sigma'].items()}
    K.save_json(K.OUT / 'size02' / 'plan_B.json', s); return s


def context(r: dict, S1) -> dict:
    p = S02 / 'rasters' / f"{r['raster_key']}.npz"
    if p.exists():
        R = K.load_npz(p)
    else:
        R = S1.load_raster(r['raster_key'])
    b = np.asarray(R['binary'], np.float32); pk = float(r['peak_target'])
    return dict(binary=b, centres=R['centres'], pattern=b * np.float32(pk), peak=pk, size_mm=float(r['size_mm']), meta_full=None, row={'radius_mm': float(r['size_mm'])})


def work(row: dict) -> dict:
    S = _setup(); C4S, RP, S1, E, DV = S['C4S'], S['RP'], S['S1'], S['E'], S['DV']
    t0 = time.monotonic(); ik = row['input_key']; jp, zp = CACHE / f'{ik}.json', CACHE / f'{ik}.npz'
    if jp.exists() and zp.exists():
        rec = K.load_json(jp)
        if rec.get('status') == 'ok' and rec.get('npz_sha256') == K.sha_file(zp):
            return rec
    meta, A = readpack(Path(row['pack']))
    if not meta.get('complete') or 'image' not in A or 'D0' not in meta['methods']:
        raise K.Stop('input', f'{ik}: pack incomplete')
    img = np.asarray(A['image'], np.uint8); ctx = context(row, S1); sf = row['surface']; sid = row['platform_surface']; rad = float(row['size_mm'])
    m0 = meta['methods']['D0']['metrics']
    Z = {k[4:]: v for k, v in A.items() if k.startswith('D0__')}
    base = {k: row[k] for k in ('input_key', 'surface', 'regime', 'size_mm', 'phase_label', 'sigma', 'rep', 'peak_target')}
    K.ledger(STAGE, 'decode_attempt', ik, 'start', rule='B0')
    try:
        D0 = DV.dot_decode(img, ctx['binary'], 1, [rad], 'B0', 'binding')
    except Exception as e:
        K.ledger(STAGE, 'decode_attempt', ik, 'failed', rule='B0', error=repr(e)[:200]); raise
    K.ledger(STAGE, 'decode_attempt', ik, 'ok', rule='B0')
    reg = C4S._regress(D0, Z)
    if not reg['regression_ok']:
        K.save_json(jp, dict(input_key=ik, status='technical_regression_failed', regression=reg, t=K.now()))
        raise K.Stop('technical', f'{ik}: original D0 not reproduced {reg}')
    mE, _a, _i, _n = E.evaluate_case(sid, 'point', DV.dot_dec(D0, 'single'), ctx, RP.mesh(sf), RP.dom_local(sf), RP.dom_global(), False)
    diff = K.metric_diffs(mE, m0, K.CHECK)
    if diff:
        K.save_json(jp, dict(input_key=ik, status='technical_eval_inconsistent', diffs=diff, t=K.now()))
        raise K.Stop('technical', f'{ik}: B0 re-evaluation differs from the stored metrics {diff}')
    K.ledger(STAGE, 'decode_attempt', ik, 'start', rule='DF')
    try:
        D = DV.dot_decode(img, ctx['binary'], 1, [rad], K.FLOOR_RULE, 'binding')
    except Exception as e:
        K.ledger(STAGE, 'decode_attempt', ik, 'failed', rule='DF', error=repr(e)[:200]); raise
    K.ledger(STAGE, 'decode_attempt', ik, 'ok', rule='DF')
    m, arr, _info, _node = E.evaluate_case(sid, 'point', DV.dot_dec(D, 'single'), ctx, RP.mesh(sf), RP.dom_local(sf), RP.dom_global(), False)
    r_d0 = dict(base, rule='D0', source='stored pack (B0 regressed, re-evaluation equal)', **{k: m0.get(k) for k in K.METRICS}); r_d0.update(n_candidates=len(D0['cd']['cands']), n_floor=0, a_floor_px2=None)
    r_df = dict(base, rule='DF', source='new decode + evaluation (S3-FE-01)', **{k: m.get(k) for k in K.METRICS}); r_df.update(n_candidates=len(D['cd']['cands']), n_floor=int(D['cd'].get('n_floor', 0)), a_floor_px2=D['cd'].get('a_floor'))
    old_df = None
    if 'DF' in meta['methods']:
        mo = meta['methods']['DF']['metrics']; old_df = dict(diffs=K.metric_diffs(m, mo, K.CHECK), equal=not K.metric_diffs(m, mo, K.CHECK))
    Aout = dict(pts=np.asarray(arr['pts'], np.float32), accepted=np.asarray(arr['accepted'], bool), valid=np.asarray(arr['valid'], bool), zr=np.asarray(arr['zr'], np.float32),
                in_global=np.asarray(arr['in_global'], bool), in_local=np.asarray(arr['in_local'], bool), point_correct=np.asarray(arr['point_correct'], bool),
                point_wrong=np.asarray(arr['point_wrong'], bool), point_unconf=np.asarray(arr['point_unconf'], bool), feats=np.asarray(D['feats'], np.float32).reshape(-1, 2))
    h = K.save_npz(zp, **Aout)
    rec = dict(input_key=ik, status='ok', regression=reg, eval_consistency_ok=True, old_df_crosscheck=old_df, rows=[r_d0, r_df], npz_sha256=h, attempts=2, seconds=round(time.monotonic() - t0, 2), t=K.now())
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
        D = D[D.input_key.isin(keys)]
    need = D[~D.input_key.isin(done_keys())]
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
    st = dict(stage=STAGE, planned=len(rows), ok=ok, failures=fails, done_total=len(done_keys()), plan_total=len(D), ledger=K.ledger_counts(STAGE), seconds=round(time.time() - t0, 1), t=K.now())
    K.save_json(K.OUT / 'size02' / 'state_B.json', st)
    if fails:
        raise K.Stop('technical', f'{len(fails)} technical failures; stopped submitting: {fails[:2]}')
    return st


def aggregate() -> dict:
    D = pd.read_csv(plan_file(), low_memory=False)
    recs = [K.load_json(CACHE / f'{ik}.json') for ik in D.input_key if (CACHE / f'{ik}.json').exists()]
    okr = [r for r in recs if r.get('status') == 'ok']
    R = pd.DataFrame([dict(x, dataset='S1-SIZE-02') for r in okr for x in r['rows']]); K.save_csv(R, K.RESULTS / 'B_size02_cases.csv')
    PM = ('glob_Ctau', 'glob_Cvalid', 'glob_recovery', 'glob_pt_rmse', 'glob_pt_p95', 'glob_surf_rmse', 'id_correct', 'id_wrong', 'id_unconf', 'n_accepted', 'n_candidates')
    out = []
    for ik, g in R.groupby('input_key'):
        a = g[g.rule == 'DF']; b = g[g.rule == 'D0']
        if len(a) != 1 or len(b) != 1:
            continue
        a = a.iloc[0]; b = b.iloc[0]; e = {k: b[k] for k in ('input_key', 'surface', 'regime', 'size_mm', 'phase_label', 'sigma', 'rep')}
        for mname in PM:
            x, y = pd.to_numeric(a[mname], errors='coerce'), pd.to_numeric(b[mname], errors='coerce')
            e[f'd_{mname}'] = (float(x) - float(y)) if (pd.notna(x) and pd.notna(y)) else np.nan; e[f'{mname}_DF'] = x; e[f'{mname}_D0'] = y
        out.append(e)
    Pd = pd.DataFrame(out); K.save_csv(Pd, K.RESULTS / 'B_size02_paired.csv')
    cells = []
    for k, g in R.groupby(['regime', 'surface', 'size_mm', 'sigma', 'rule']):
        ph = g.groupby('phase_label'); c = dict(zip(['regime', 'surface', 'size_mm', 'sigma', 'rule'], k), n_images=len(g), n_phases=int(g.phase_label.nunique()))
        for mname in ('glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'glob_recovery', 'glob_pt_rmse', 'glob_pt_p95', 'glob_surf_rmse'):
            pv = ph[mname].apply(lambda s: pd.to_numeric(s, errors='coerce').mean())
            c[f'{mname}_mean'] = float(pv.mean()) if pv.notna().any() else np.nan; c[f'{mname}_min'] = float(pv.min()) if pv.notna().any() else np.nan
            c[f'{mname}_max'] = float(pv.max()) if pv.notna().any() else np.nan
        c.update(id_correct=int(pd.to_numeric(g.id_correct, errors='coerce').sum()), id_wrong=int(pd.to_numeric(g.id_wrong, errors='coerce').sum()),
                 id_unconf=int(pd.to_numeric(g.id_unconf, errors='coerce').sum()), n_no_output=int((pd.to_numeric(g.n_accepted, errors='coerce') == 0).sum()))
        cells.append(c)
    Cc = pd.DataFrame(cells); K.save_csv(Cc, K.RESULTS / 'B_size02_cells.csv')
    xc = [r['old_df_crosscheck'] for r in okr if r.get('old_df_crosscheck')]
    s = dict(stage=STAGE, records=len(recs), ok=len(okr), not_ok=[r['input_key'] for r in recs if r.get('status') != 'ok'][:20], rows=len(R), paired=len(Pd), cells=len(Cc),
             regression_ok=int(sum(1 for r in okr if r['regression']['regression_ok'])), old_df_overlap=len(xc), old_df_equal=int(sum(1 for x in xc if x['equal'])), t=K.now())
    K.save_json(K.RESULTS / 'B_size02_summary.json', s); return s
