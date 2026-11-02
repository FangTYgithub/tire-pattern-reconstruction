# Re-decode stored dot images with DF and compare the saved reconstruction arrays.

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd

import tr_common as K

KEEP = ('state', 'n_out', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'glob_n_nodes', 'loc_n_nodes', 'glob_Cvalid', 'loc_Cvalid', 'glob_Ctau', 'loc_Ctau',
        'glob_cov_interp', 'loc_cov_interp', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'glob_surf_p95', 'loc_surf_rmse', 'loc_surf_p95',
        'glob_recovery', 'loc_recovery', 'id_correct', 'id_wrong', 'id_unconf', 'id_denominator', 'glob_id_correct', 'glob_id_wrong', 'glob_id_unconf',
        'loc_id_correct', 'loc_id_wrong', 'loc_id_unconf', 'glob_gross_rate', 'n_out_of_window', 'img_n_expected', 'img_n_recovered')
ARR = ('pts', 'accepted', 'valid', 'zr', 'in_global', 'in_local', 'point_correct', 'point_wrong', 'point_unconf', 'point_feature', 'feat_truth_dot', 'feat_confirmable', 'exp_recovered')
SHARD_N = 32


def plan() -> pd.DataFrame:
    import tr_plan as PL
    r = K.run_load(); keys = pd.DataFrame(r['plan']['a3_keys'])
    A = PL.a3_views().set_index('input_key').loc[keys.input_key].reset_index()
    A['reuse_class'] = keys.reuse_class.values; A['replay'] = keys.replay.values
    return A


def _inputs(u: dict) -> tuple:
    import rdy_platform as P
    import s12_case as SC12
    sh = K.run_load()['plan']['inputs_sha256']
    for f in (u['image_record'].replace('.json', '.npz'), u['eval_record'], u['eval_record'].replace('.json', '.npz')):
        if K.sha(K.ROOT / f) != sh[f]:
            raise K.Stop('technical', f"{u['input_key']}: input file changed since registration: {f}")
    R = SC12.raster_of(u); binary = np.asarray(R['binary'], np.float32); ctx = SC12.context(u, R, u['peak_target'])
    img = K.load_npz(K.ROOT / u['image_record'].replace('.json', '.npz'))['image']
    return R, binary, ctx, img, P.mesh(u['surface']), K.RK.SURF_MAP[u['surface']]


def _u(row: dict) -> dict:
    u = dict(row)
    for k in ('size_mm', 'peak_target', 'sigma'):
        u[k] = float(u[k])
    return u


def _arrays(arr: dict) -> dict:
    return {k: np.asarray(arr[k]) for k in ARR if k in arr}


def work_df(row: dict) -> dict:
    import c2_decode as CD
    import s3c_dev as DV
    u = _u(row); t0 = time.monotonic()
    R, binary, ctx, img, mesh, sid = _inputs(u)
    D = DV.dot_decode(img, binary, 1, [u['size_mm']], 'DF', 'binding')
    m, arr = DV.evaluate(sid, 'point', CD.dot_dec(D, 'single'), ctx, mesh)
    return dict(input_key=u['input_key'], metrics={k: m.get(k) for k in KEEP}, arrays=_arrays(arr), n_candidates=len(D['cd']['cands']), n_floor=int(D['cd'].get('n_floor', 0)),
                a_floor=D['cd'].get('a_floor'), seconds=round(time.monotonic() - t0, 2))


def _regress(D: dict, Z: dict) -> dict:

    cands = np.array(D['cd']['cands'], np.float64).reshape(-1, 7) if len(D['cd']['cands']) else np.zeros((0, 7))
    out = dict(n_cand_new=len(cands), n_cand_stored=int(len(Z['cand_uv'])))
    same_n = len(cands) == len(Z['cand_uv'])
    out['cand_equal'] = bool(same_n and np.array_equal(cands[:, :2].astype(Z['cand_uv'].dtype), Z['cand_uv']) and np.array_equal(cands[:, 2].astype(Z['cand_area'].dtype), Z['cand_area'])
                             and np.array_equal(cands[:, 3].astype(Z['cand_elong'].dtype), Z['cand_elong']) and np.array_equal(cands[:, 5].astype(Z['cand_label'].dtype), Z['cand_label']))
    out['select_equal'] = bool(same_n and np.array_equal(np.asarray(D['code']).astype(Z['cand_code'].dtype), Z['cand_code']) and np.array_equal(np.asarray(D['fidx']).astype(Z['cand_fidx'].dtype), Z['cand_fidx']))
    f = np.asarray(D['feats'], np.float32).reshape(-1, 2); out['feats_equal'] = bool(f.shape == Z['feats'].shape and np.array_equal(f, Z['feats']))
    pts, info, tr = D['arms']['single']; p = np.asarray(pts, np.float64).reshape(-1, 3); ps = np.asarray(Z['pts'], np.float64)
    out['pts_equal'] = bool(p.shape == ps.shape and np.array_equal(np.nan_to_num(p, nan=1e30), np.nan_to_num(ps, nan=1e30)))
    out['trace_equal'] = bool(all(np.array_equal(np.asarray(tr[k]).astype(Z[f'feat_{k}'].dtype), Z[f'feat_{k}']) for k in ('matched', 'ambiguous', 'duplicate', 'best_idx', 'point_idx')))
    out['regression_ok'] = bool(out['cand_equal'] and out['select_equal'] and out['feats_equal'] and out['pts_equal'] and out['trace_equal'])
    return out


def _same(a, b) -> bool:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or (isinstance(b, float) and np.isnan(b))
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        return float(a) == float(b) or abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)))
    return str(a) == str(b)


def replay() -> dict:
    import c2_decode as CD
    import s3c_dev as DV
    A = plan(); C4 = K.read_csv(K.C4 / 'results' / 'C1_dots_cases.csv'); C4 = C4[C4.rule == 'DF'].set_index('input_key')
    out = []
    for r in A[A.replay.notna()].to_dict('records'):
        u = _u(r); ik = u['input_key']; R, binary, ctx, img, mesh, sid = _inputs(u)
        if u['replay'] == 'DF_vs_S3-CODE-04':
            K.reserve('decode_replay_DF', ik, 'decode', K.CAPS['decode_total'])
            try:
                D = DV.dot_decode(img, binary, 1, [u['size_mm']], 'DF', 'binding')
            except Exception as e:
                K.ledger('decode_replay_DF', ik, 'failed', error=repr(e)[:200]); raise
            K.ledger('decode_replay_DF', ik, 'ok')
            m, arr = DV.evaluate(sid, 'point', CD.dot_dec(D, 'single'), ctx, mesh)
            Z = K.load_npz(K.C4 / 'cache' / 'C1' / f'{ik}.npz'); ref = C4.loc[ik]
            pts = np.asarray(arr['pts'], np.float32)
            chk = dict(pts_equal=bool(pts.shape == Z['DF_pts'].shape and np.array_equal(np.nan_to_num(pts, nan=1e30), np.nan_to_num(Z['DF_pts'], nan=1e30))),
                       acc_equal=bool(np.array_equal(np.asarray(arr['accepted'], bool), Z['DF_acc'])),
                       zr_equal=bool(np.array_equal(np.asarray(arr['zr'], np.float32), Z['DF_zr'])))
            diffs = {k: (ref.get(k), m.get(k)) for k in KEEP if k in ref.index and not _same(ref.get(k), m.get(k))}
            chk.update(metric_diffs=K.clean(diffs), ok=bool(chk['pts_equal'] and chk['acc_equal'] and chk['zr_equal'] and not diffs))
        else:
            K.reserve('decode_replay_D0', ik, 'decode', K.CAPS['decode_total'])
            try:
                D = DV.dot_decode(img, binary, 1, [u['size_mm']], 'B0', 'binding')
            except Exception as e:
                K.ledger('decode_replay_D0', ik, 'failed', error=repr(e)[:200]); raise
            K.ledger('decode_replay_D0', ik, 'ok')
            Z = K.load_npz(K.ROOT / u['eval_record'].replace('.json', '.npz')); chk = _regress(D, Z); chk['ok'] = chk['regression_ok']
            J = K.load_json(K.ROOT / u['eval_record']); m, _ = DV.evaluate(sid, 'point', CD.dot_dec(D, 'single'), ctx, mesh)
            diffs = {k: (J['metrics'].get(k), m.get(k)) for k in KEEP if k in J['metrics'] and not _same(J['metrics'].get(k), m.get(k))}
            chk.update(metric_diffs=K.clean(diffs)); chk['ok'] = bool(chk['ok'] and not diffs)
        out.append(dict(input_key=ik, surface=u['surface'], part=u['part'], replay=u['replay'], **chk))
    res = dict(n=len(out), n_ok=sum(1 for x in out if x['ok']), cases=out, t=K.now())
    K.run_put('a3_replay', res)
    if res['n_ok'] != res['n']:
        raise K.Stop('technical', f"replay: {res['n'] - res['n_ok']} of {res['n']} not reproduced - A3 DF batch not started")
    return {k: v for k, v in res.items() if k != 'cases'}


def done_keys() -> dict:
    out = {}
    for p in sorted(K.SHARDS.glob('a3_df_*.npz')):
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                if k.endswith('|metrics'):
                    out[k.split('|')[0]] = p.name
    return out


def _write_shard(results: list) -> str:
    n = len(list(K.SHARDS.glob('a3_df_*.npz'))) if K.SHARDS.exists() else 0
    arrays = {}
    for r in results:
        ik = r['input_key']
        for name, a in r['arrays'].items():
            arrays[f'{ik}|{name}'] = a
        meta = dict(metrics=r['metrics'], n_candidates=r['n_candidates'], n_floor=r['n_floor'], a_floor=r['a_floor'], seconds=r['seconds'])
        arrays[f'{ik}|metrics'] = np.array(json.dumps(K.clean(meta), default=str))
    p = K.SHARDS / f'a3_df_{n:03d}.npz'
    return K.save_npz(p, **arrays)


def run(workers: int = 12) -> dict:
    rp = K.run_load().get('a3_replay') or {}
    if not rp or rp.get('n_ok') != rp.get('n'):
        raise K.Stop('refused', 'replay check missing or failed: run `tr_a3.py replay` first')
    A = plan(); todo = A[A.reuse_class == 'new_DF']; done = done_keys()
    need = todo[~todo.input_key.isin(done)]
    used = K.count('decode')
    if used + len(need) > K.CAPS['decode_total']:
        raise K.Stop('budget', f'decodes {used} reserved + {len(need)} planned > {K.CAPS["decode_total"]}')
    rows = need.to_dict('records'); it = iter(rows); inflight = {}; buf = []; fails = []; ok = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        def submit_next():
            r = next(it, None)
            if r is not None:
                K.ledger('decode_DF', r['input_key'], 'reserved')
                inflight[ex.submit(work_df, r)] = r['input_key']
            return r is not None
        for _ in range(workers * 2):
            if not submit_next():
                break
        while inflight:
            dn, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for f in dn:
                key = inflight.pop(f)
                try:
                    res = f.result(); K.ledger('decode_DF', key, 'ok', seconds=res['seconds']); buf.append(res); ok += 1
                except Exception as e:
                    K.ledger('decode_DF', key, 'failed', error=traceback.format_exception_only(type(e), e)[-1][:300]); fails.append(key)
                if len(buf) >= SHARD_N:
                    _write_shard(buf); buf = []
                    print(f'[{K.now()}] {ok}/{len(rows)} DF done, {round((time.time() - t0) / 60, 1)} min', flush=True)
            if fails:
                for f in list(inflight):
                    if f.cancel():
                        K.ledger('decode_DF', inflight.pop(f), 'cancelled_not_started')
                continue
            while len(inflight) < workers * 2 and submit_next():
                pass
    if buf:
        _write_shard(buf)
    st = dict(planned=len(rows), ok=ok, failed=fails, decodes_reserved_total=K.count('decode'), minutes=round((time.time() - t0) / 60, 1), t=K.now())
    K.run_put('a3_run', st)
    if fails:
        raise K.Stop('technical', f'{len(fails)} DF failures; submission stopped', detail=fails[:5])
    return st


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['replay', 'run']); ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    try:
        r = replay() if a.cmd == 'replay' else run(a.workers)
        print(json.dumps(K.clean(r), ensure_ascii=False, default=str)[:3000])
    except K.Stop as e:
        print(json.dumps(dict(code=e.code, msg=e.msg, detail=K.clean(e.detail)), ensure_ascii=False, default=str)[:3000]); raise SystemExit(K.EXIT.get(e.code, 2))


if __name__ == '__main__':
    main()
