# Compute derived encoding metrics from validated case records.

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import c6_common as K

DERIV_VERSION = 'S3-CODE-06 derived evaluation v1'
EVAL_FILES = ['scripts/surface_revision/sections_adapter/s1r_eval.py', 'scripts/surface_revision/sections_adapter/s1r_common.py', 'src/surface_eval_r3.py',
              'scripts/surface_revision/s3_code01/s3c_dev.py', 'scripts/surface_revision/s3_code02/c2_decode.py', 'scripts/surface_revision/s3_code03/c3_store.py',
              'scripts/surface_revision/s3_code03/c3_eval.py', 'scripts/surface_revision/s3_code04/c4_store.py', 'scripts/surface_revision/s3_code04/c4_parity.py',
              'scripts/surface_revision/surf02/s2_common.py', 'scripts/surface_revision/s345_ready/rdy_platform.py']
DOMS = ('glob', 'loc')
FIELDS = ['state', 'n_accepted', 'n_valid'] + [f'{d}_{f}' for d in DOMS for f in ('n_valid', 'pt_rmse', 'pt_p95', 'n_nodes', 'n_interp', 'cov_interp', 'surf_rmse', 'surf_p95', 'Cvalid', 'Ctau')]
SAME_AS_RECORD = ['n_accepted', 'n_valid', 'glob_Cvalid', 'loc_Cvalid', 'glob_Ctau', 'loc_Ctau', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse']


def definition() -> dict:
    import rdy_platform as P
    rr = K.K4.K3.RELEASE / 'release_surf02.json'
    return dict(version=DERIV_VERSION, files={f: K.sha_file(K.ROOT / f) for f in EVAL_FILES}, release=K.sha_file(rr), dom_global=P.dom_global()['hash'],
                d_support=float(K.K4.RK.D_SUPPORT), tau=float(K.K4.RK.TAU))


def decode_source(store, decode_key: str, hashes: K.Hashes) -> dict:

    st, rec = store.inspect('decode', decode_key)
    if st != 'ok':
        raise K.Stop('records', f'decode record {decode_key}: {st}')
    if rec.get('kind') == 'ref':
        npz = K.ROOT / rec['source_npz']; js = K.ROOT / rec['source_json']
        return dict(npz=str(npz), meta_file=str(js), meta_from='source_json', npz_sha=hashes.sha(npz), meta_sha=hashes.sha(js))
    p = store.path('decode', decode_key)
    return dict(npz=str(p.with_suffix('.npz')), meta_file=str(p), meta_from='record', npz_sha=hashes.sha(p.with_suffix('.npz')), meta_sha=hashes.sha(p))


def _evaluate(job: dict) -> dict:

    import c2_decode as CD
    import c3_eval as EV
    import c3_store as ST
    import c4_store as S4
    import rdy_platform as P
    import s3c_dev as DV
    row = job['row']; inp = S4.inputs_of(row); ctx = EV.ctx_of(row, inp); mesh = P.mesh(row['surface']); sid = K.K4.RK.SURF_MAP[row['surface']]; out = []
    for r in job['recs']:
        A = K.load_npz(r['npz']); meta = K.load_json(r['meta_file'])['meta']
        D = ST.unpack(row['family'], A, meta); dec = CD.dot_dec(D, r['arm']) if row['family'] == 'dots' else CD.line_dec(D)
        m, _ = DV.evaluate(sid, 'point' if row['family'] == 'dots' else 'line', dec, ctx, mesh)
        out.append(dict(r, fields={f: m.get(f) for f in FIELDS}))
    return dict(image_key=row['image_key'], out=out)


def derive(T: pd.DataFrame, M: pd.DataFrame, store, out_dir, hashes: K.Hashes, workers: int = 14, tag: str = '') -> tuple[pd.DataFrame, dict]:

    out_dir = K.guard(out_dir); defin = dict(definition(), tag=tag); dkey = K.digest(defin, 24)
    E = T[(T.layer == 'eval') & T.complete]; Mi = M.drop_duplicates('image_key').set_index('image_key')
    rows, todo = [], {}
    for e in E.itertuples():
        src = decode_source(store, e.decode_key, hashes)
        kin = dict(definition=dkey, npz=src['npz_sha'], meta=src['meta_sha'], arm=e.arm, rule=e.rule, family=e.family, surface=str(Mi.loc[e.image_key, 'surface']),
                   raster=str(Mi.loc[e.image_key, 'raster_key']))
        key = 'C6X-' + K.digest(kin, 24); p = out_dir / f'{key}.json'
        if p.exists():
            try:
                j = K.load_json(p)
                if j.get('key_inputs') == kin and all(f in j.get('fields', {}) for f in FIELDS):
                    rows.append(dict(key=e.key, x_key=key, action='reused', **{f'x_{f}': j['fields'][f] for f in FIELDS})); continue
            except Exception:
                pass
        todo.setdefault(e.image_key, []).append(dict(key=e.key, x_key=key, kin=kin, arm=e.arm, npz=src['npz'], meta_file=src['meta_file']))
    jobs = [dict(row=dict(Mi.loc[ik].to_dict(), image_key=ik), recs=v) for ik, v in todo.items()]
    if jobs:
        res = []
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(_evaluate, jobs, chunksize=2))
        else:
            res = [_evaluate(j) for j in jobs]
        for r in res:
            for x in r['out']:
                K.save_json(out_dir / f"{x['x_key']}.json", dict(identity=x['x_key'], eval_key=x['key'], key_inputs=x['kin'], fields=x['fields'], definition=defin['version'], t_local=K.now_local()))
                rows.append(dict(key=x['key'], x_key=x['x_key'], action='computed', **{f'x_{f}': x['fields'][f] for f in FIELDS}))
    X = pd.DataFrame(rows)
    info = dict(definition_key=dkey, n=len(X), computed=int((X.action == 'computed').sum()) if len(X) else 0, reused=int((X.action == 'reused').sum()) if len(X) else 0,
                dir=K.rel(out_dir))
    return X, info


def consistency(T: pd.DataFrame, tol: float = K.TOL) -> list:

    E = T[(T.layer == 'eval') & T.complete]; probs = []
    for f in SAME_AS_RECORD:
        if f'r_{f}' not in E.columns or f'x_{f}' not in E.columns:
            continue
        a, b = pd.to_numeric(E[f'r_{f}'], errors='coerce'), pd.to_numeric(E[f'x_{f}'], errors='coerce')
        bad = ((a - b).abs() > tol) | (a.isna() != b.isna())
        if bad.any():
            probs.append(dict(category='semantics', check=f'derived evaluation differs from the record: {f}', n=int(bad.sum()), detail=float((a - b).abs().max()), keys=list(E.key[bad].head(5))))
    return probs
