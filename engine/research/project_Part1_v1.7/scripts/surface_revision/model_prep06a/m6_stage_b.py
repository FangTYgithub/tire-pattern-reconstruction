# Compare one-ray and sub-ray images under the same imaging conditions.

from __future__ import annotations

import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import m6_common as K

K5 = K.K5
STORE = K.OUT / 'data' / 'b'


def stage_gate() -> dict:
    import m6_stage_a as A
    p = K.protocol(); probs = []
    if 'rules' not in p:
        probs.append('rules not registered')
    if not (p.get('A0') or {}).get('ok'):
        probs.append('A0 drills / regression not passed')
    if not (p.get('A1') or {}).get('reproduced'):
        probs.append('A1 reproduction not exact')
    if 'A3' not in p:
        probs.append('A3 not completed')
    b = p.get('binding') or {}
    code_now = A.code_hashes(); drift = [f for f, h in (b.get('code') or {}).items() if code_now.get(f) != h]
    inp_now = {f: K.sha_file(K.ROOT / f) for f in K.INPUT_FILES}; inp_drift = [f for f, h in (b.get('inputs') or {}).items() if inp_now.get(f) != h]
    chk = A.mp5_check()
    if drift:
        probs.append(f'code drift {drift}')
    if inp_drift:
        probs.append(f'input drift {inp_drift}')
    if chk['outputs_changed'] or chk['protected_changed'] or chk['freeze_problems']:
        probs.append(f'MODEL-PREP-05 changed {chk}')
    T = pd.read_csv(K.OUT / '_work' / 'a3_arms.csv')
    if T.duplicated(['case_key', 'arm']).any():
        probs.append('A3 duplicate (image, arm) keys')
    L = K.Ledger()
    if L.count('regress_P4', 'decode_arm') > K.CAPS['decode_A']:
        probs.append('A decode cap exceeded')
    rec = dict(t=K.now(), passed=not probs, problems=probs, code_drift=drift, input_drift=inp_drift, mp5_check=chk, ledger_decode_A=L.count('regress_P4', 'decode_arm'))
    K.protocol_put('gate_B', rec)
    return rec


def _parent_rows() -> pd.DataFrame:
    M = pd.read_csv(K.SRC5 / 'formal_manifest.csv', float_precision='round_trip', low_memory=False)
    c = K.SUBRAY
    sel = M[M.config.isin(c['configs']) & M.env.isin([e for e, _ in c['envs']]) & M.sigma.isin(c['sigmas']) & M.surface.isin(c['surfaces']) &
            M.phase_block.isin(c['blocks']) & (M.rep == c['rep'])].copy()
    if len(sel) != 96 or sel.case_key.duplicated().any():
        raise K.Stop('identity', f'sub-ray parents: {len(sel)} rows (96 expected)')
    return sel.sort_values(['config', 'env', 'sigma', 'surface', 'phase_block']).reset_index(drop=True)


def _compat_rows() -> pd.DataFrame:
    M = pd.read_csv(K.SRC5 / 'formal_manifest.csv', float_precision='round_trip', low_memory=False)
    c = K.SUBRAY['compat']
    sel = M[M.config.isin(c['configs']) & (M.env == c['env']) & M.sigma.isin(c['sigmas']) & (M.surface == c['surface']) & (M.phase_block == c['block']) &
            (((M.sigma == 0) & (M.rep == 0)) | ((M.sigma > 0) & (M.rep == 1)))]
    if len(sel) != 6:
        raise K.Stop('identity', f'compat rows {len(sel)} (6 expected)')
    return sel


def _render(pr: dict, rays: int):
    import m6_platform as PL
    import mp3_forward as F
    spec = dict(pr['spec'])
    if rays != 1:
        spec['rays_per_pixel'] = int(rays)
    mesh = PL.platform()['RP'].s2().build_mesh(pr['sid'])
    out = F.render(mesh, pr['pattern'], spec)
    img = out['image']
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.shape != (512, 512):
        raise K.Stop('integrity', f"{pr['case_id']}: render returned {type(img).__name__}")
    if int(out['diag'].get('rays_per_pixel', 0)) != rays:
        raise K.Stop('integrity', f"{pr['case_id']}: rays {out['diag'].get('rays_per_pixel')} != {rays}")
    return img, out['diag'], K5.digest(dict(pr['forward_record'], spec=spec))


def _work3(row: dict) -> dict:
    import m6_platform as PL
    t0 = time.time()
    try:
        pr = PL.prep(row)
        img, diag, fk3 = _render(pr, 3)
        raw, dmeta, thr = PL.p4_decode(img, pr)
        ev, rec, fi = PL.evaluate(pr, 'P4', raw, dmeta)
        return dict(ok=True, case_key=row['case_key'], img=img, diag=K.clean(diag), fk3=fk3, raw=raw, ev=ev, rec=rec, thr=thr, seconds=round(time.time() - t0, 1))
    except BaseException as e:
        import traceback
        return dict(ok=False, case_key=row['case_key'], error=repr(e)[:400], trace=traceback.format_exc()[-1500:])


# Run and pair the one-ray and sub-ray rendering conditions.
def stage_b(workers: int = 8) -> dict:
    import m6_platform as PL
    g = K.protocol().get('gate_B') or {}
    if not g.get('passed'):
        raise K.Stop('refused', 'stage B: technical gate not passed')
    L = K.Ledger(); out = dict(t0=K.now())

    comp = {}
    for r in _compat_rows().to_dict('records'):
        ck = r['case_key']
        rid = L.reserve('render_compat', ck, K.CAP_B_RENDER)
        try:
            pr = PL.prep(r); img, diag, fk = _render(pr, 1); L.end(rid, 'ok')
        except BaseException as e:
            L.end(rid, 'failed', error=repr(e)[:300]); raise
        meta, A = K.read_pack(ck, prefixes=('image',))
        same = bool(np.array_equal(img, A['image'])); fk_ok = fk == pr['forward_key']
        comp[ck] = dict(pixel_identical=same, n_diff_px=int((img != A['image']).sum()), forward_key_equal=fk_ok, sigma=float(r['sigma']))
        if not same:
            K.save_npz(STORE / 'compat_mismatch' / f"{ck.replace('|', '__')}.npz", dict(case_key=ck), dict(image=img, stored=A['image']))
    out['compat'] = comp; out['compat_ok'] = all(v['pixel_identical'] and v['forward_key_equal'] for v in comp.values())
    K.protocol_put('B', dict(out, status='compat done'))
    if not out['compat_ok']:
        K.protocol_put('B', dict(out, status='technical failure: 1x1 not equivalent'))
        raise K.Stop('technical', '1x1 compatibility failed: 3x3 not started')

    rows = _parent_rows().to_dict('records'); res = {}
    def run_one(r):
        rid = L.reserve('render_3x3', r['case_key'], K.CAP_B_RENDER); did = L.reserve('decode_3x3', r['case_key'], K.CAP_B_DECODE)
        return rid, did
    rid, did = run_one(rows[0]); x = _work3(rows[0])
    L.end(rid, 'ok' if x['ok'] else 'failed', error=x.get('error')); L.end(did, 'ok' if x['ok'] else 'failed')
    if not x['ok']:
        K.protocol_put('B', dict(out, status='technical failure in the first 3x3 case', error=x['error'], trace=x['trace']))
        raise K.Stop('technical', f"first 3x3 case failed: {x['error']}")
    res[x['case_key']] = x
    e2e = _save_and_pair({x['case_key']: x}, rows[:1], check_only=False)
    out['first_case_end_to_end'] = e2e
    K.protocol_put('B', dict(out, status='first 3x3 case end to end done'))
    ids = {}
    for r in rows[1:]:
        ids[r['case_key']] = run_one(r)
    t0 = time.time(); failed = None
    with Pool(workers) as pool:
        for i, y in enumerate(pool.imap_unordered(_work3, rows[1:])):
            rid, did = ids[y['case_key']]
            L.end(rid, 'ok' if y['ok'] else 'failed', error=y.get('error')); L.end(did, 'ok' if y['ok'] else 'failed')
            if not y['ok']:
                failed = y; pool.terminate(); break
            res[y['case_key']] = y
            if (i + 1) % 12 == 0:
                print(f'B 3x3 {i + 2}/96 {round(time.time() - t0)} s', flush=True)
    if failed:
        for ck, (rid, did) in ids.items():
            if ck not in res and ck != failed['case_key']:
                L.end(rid, 'aborted'); L.end(did, 'aborted')
    P = _save_and_pair(res, rows, check_only=False)
    out.update(n_3x3=len(res), failed=(None if not failed else dict(case_key=failed['case_key'], error=failed['error'])), seconds=round(time.time() - t0, 1),
               ledger=dict(render_compat=L.count('render_compat'), render_3x3=L.count('render_3x3'), decode_3x3=L.count('decode_3x3')),
               status=('complete' if (not failed and len(res) == 96) else 'technical failure (partial results kept)'))
    K.protocol_put('B', out)
    return dict(out, pairs=P)


def _save_and_pair(res: dict, rows: list, check_only: bool) -> dict:

    import m6_platform as PL
    by = {}
    for r in rows:
        if r['case_key'] in res:
            by.setdefault((r['config'], r['env'], float(r['sigma'])), []).append(r)
    recs = []
    for (cfg, env, sg), rr in by.items():
        pf = STORE / f'{cfg}__{env}__s{int(sg)}.npz'
        arrays, meta = {}, dict(condition=dict(config=cfg, env=env, sigma=sg), images={})
        if pf.exists():
            with np.load(pf, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files if k != '_meta'}; meta = json.loads(str(z['_meta']))
        for r in rr:
            x = res[r['case_key']]; tag = f"{r['surface']}__b{int(r['phase_block'])}"
            arrays[f'{tag}__image'] = x['img']
            arrays.update({f'{tag}__raw__{k}': v for k, v in x['raw'].items()}); arrays.update({f'{tag}__ev__{k}': v for k, v in x['ev'].items()})
            meta['images'][tag] = dict(case_key_parent=r['case_key'], forward_key_3x3=x['fk3'], image_sha=K5.content_sha(x['img']), diag=x['diag'], metrics=x['rec']['metrics'], support=x['rec'].get('support'),
                                       full_identity=x['rec']['full_identity'], state=x['rec']['state'], thr=x['thr'], raw_digest=K5.arrays_digest(x['raw']),
                                       eval_digest=K5.arrays_digest(x['ev']), seconds=x['seconds'])
        if not check_only:
            K.save_npz(pf, meta, arrays)

        for r in rr:
            tag = f"{r['surface']}__b{int(r['phase_block'])}"
            if not check_only:
                with np.load(pf, allow_pickle=False) as z:
                    mm = json.loads(str(z['_meta']))['images'][tag]; ev_rb = {k[len(f'{tag}__ev__'):]: z[k] for k in z.files if k.startswith(f'{tag}__ev__')}
                if K5.arrays_digest(ev_rb) != mm['eval_digest']:
                    raise K.Stop('integrity', f'{pf.name} {tag}: read-back eval digest differs')
            m3 = meta['images'][tag]['metrics']; fi3 = (meta['images'][tag]['full_identity'].get('counts') or {}).get('all') or {}
            pm, _ = K.read_pack(r['case_key'], prefixes=('image',))
            old = pm['results']['P4']; m1 = old['metrics']; fi1 = ((old.get('full_identity') or {}).get('counts') or {}).get('all') or {}
            rec = dict(case_key=r['case_key'], config=r['config'], env=r['env'], psf_true_px=float(r['psf_true_px']), sigma=float(r['sigma']), surface=r['surface'],
                       phase_block=int(r['phase_block']), state_1=old['state'], state_3=meta['images'][tag]['state'])
            for k in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'n_valid', 'glob_n_valid', 'n_accepted'):
                rec[f'{k}_1'] = m1.get(k); rec[f'{k}_3'] = m3.get(k)
            sup1 = old.get('support') or {}
            rec['sup_surf_rmse_1'] = sup1.get('sup_surf_rmse'); rec['sup_surf_rmse_3'] = (meta['images'][tag].get('support') or {}).get('sup_surf_rmse')
            rec['c_1'], rec['w_1'], rec['u_1'] = fi1.get('c'), fi1.get('w'), fi1.get('u'); rec['c_3'], rec['w_3'], rec['u_3'] = fi3.get('c'), fi3.get('w'), fi3.get('u')
            recs.append(rec)
    return dict(n_pairs=len(recs), rows=recs if len(recs) <= 2 else None, table=(K.rel(K.save_csv(pd.DataFrame(recs), K.OUT / '_work' / 'b_pairs.csv')) if not check_only else None))
