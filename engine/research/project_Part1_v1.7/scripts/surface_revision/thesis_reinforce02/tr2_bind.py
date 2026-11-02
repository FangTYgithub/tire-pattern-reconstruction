# Check result reuse by rendering, decoding and evaluation identity layers.

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import tr2_common as K

ARR = ('pts', 'accepted', 'valid', 'zr', 'in_global', 'in_local', 'point_correct', 'point_wrong', 'point_unconf', 'point_feature', 'feat_truth_dot',
       'feat_confirmable', 'exp_recovered')
PER_POINT = ('accepted', 'valid', 'zr', 'in_global', 'in_local', 'point_correct', 'point_wrong', 'point_unconf', 'point_feature')
REQ_METRICS = ('state', 'n_accepted', 'n_valid', 'glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_pt_rmse', 'id_correct', 'id_wrong', 'id_unconf')


_CHILD = r'''
import ast, importlib, json, sys
root, entries, paths = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
sys.dont_write_bytecode = True
for p in paths:
    if p not in sys.path: sys.path.insert(0, p)
import os
seen, queue, failed = set(), list(entries), {}
def names(f):
    out = set()
    try:
        t = ast.parse(open(f, encoding='utf-8').read())
    except Exception:
        return out
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            out.add(n.module)
    return out
try:
    import tr_common  # noqa: F401  (runtime path setup of the THESIS-REINFORCE layers)
except BaseException:
    pass
def collect():
    for mm in list(sys.modules.values()):
        g = getattr(mm, '__file__', None)
        if g and os.path.abspath(g).startswith(root) and g.endswith('.py') and os.path.abspath(g) not in seen:
            queue.append(os.path.abspath(g))
for _pass in range(3):
    while queue:
        f = queue.pop()
        if f in seen: continue
        seen.add(f)
        for m in sorted(names(f)):
            try:
                importlib.import_module(m); failed.pop(m, None)
            except BaseException as e:
                failed[m] = repr(e)[:160]; continue
            collect()
    for m in list(failed):
        try:
            importlib.import_module(m); failed.pop(m, None); collect()
        except BaseException:
            pass
print(json.dumps(dict(files=sorted(seen), failed=failed)))
'''
def module_closure(entry_files: list, extra_paths: list) -> dict:
    root = str(K.ROOT.resolve())
    r = subprocess.run([sys.executable, '-B', '-c', _CHILD, root, json.dumps([str(p) for p in entry_files]), json.dumps([str(p) for p in extra_paths])],
                       capture_output=True, text=True, encoding='utf-8', cwd=str(K.HERE), timeout=900)
    if r.returncode != 0:
        raise K.Stop('technical', f'module closure failed: {r.stderr[-800:]}')
    j = json.loads(r.stdout.strip().splitlines()[-1])
    files = {K.rel(p): K.sha(p) for p in j['files']}
    return dict(files=files, n=len(files), import_failures=j['failed'])


def frozen_lists() -> dict:
    R = K.ROOT / 'outputs' / 'surface_revision'; out = {}
    for name, p, path in (('SURF-02 v1.1 dependencies', R / 'surf02/release/dependencies_surf02_v1_1.json', ('files',)),
                          ('S1-REAL-01 dependencies', R / 's1_real01/config/s1_dependencies.json', ('files',)),
                          ('S2 formal batch freeze', R / 's2_prepare/run/freeze_formal_batch.json', ('files',)),
                          ('S345 ready freeze', R / 's345_ready/freeze_s345_ready.json', ('files',)),
                          ('S3-CODE-04 formal package closure', R / 's3_code04/formal/frozen.json', ('closure', 'files'))):
        j = K.load_json(p)
        for k in path:
            j = j[k]
        out[name] = dict(file=K.rel(p), sha256=K.sha(p), files={str(k).replace('\\', '/'): v for k, v in j.items()})
    return out


def compare_with_frozen(files: dict, frozen: dict) -> pd.DataFrame:
    rows = []
    for f, h in sorted(files.items()):
        rec = dict(record_type='bind_code', file=f, sha256_now=h)
        hit = [(n, fl['files'][f]) for n, fl in frozen.items() if f in fl['files']]
        rec['frozen_in'] = ';'.join(n for n, _ in hit)
        rec['equal_to_frozen'] = (all(v == h for _, v in hit) if hit else None)
        rec['differs_from'] = ';'.join(n for n, v in hit if v != h)
        rows.append(rec)
    return pd.DataFrame(rows)


def data_deps_tr01() -> dict:

    import tr_a2 as A2
    files = set()
    R = K.ROOT / 'outputs' / 'surface_revision'
    for p in (R / 's12_resource01/results/views.csv', A2.S3_MANIFEST, A2.S3_KEYS, R / 's3_code07/analysis/formal/current/cases.csv', K.T1.TC / 'results.csv',
              R / 'sections_adapter/candidates_s1s5.csv', R / 'sections_adapter/binding.json', R / 'surf02/release/release_surf02.json',
              R / 'surf02/release/dependencies_surf02_v1_1.json'):
        files.add(p)
    d12, l12, d3, l3 = A2.plan()
    for r in d12 + l12:
        files.add(K.ROOT / r['eval_record'].replace('.json', '.npz')); files.add(K.ROOT / r['eval_record'])
    for r in d3 + l3:
        for _arm, ek, dk in r['arms']:
            for kind, k in (('eval', ek), ('dec', dk)):
                files.add(A2.S3_REC / kind / f'{k}.npz')
    G = K.read_csv(K.OUT1 / 'identity_geometry.csv'); V = G[G.record_type == 'A4_view']
    E = K.read_csv(A2.S3_KEYS); E = E[E.layer == 'eval']
    s3v = V[V.source == 'S3']
    for ik, arm in set(zip(s3v.input_key, s3v.arm.astype(str))):
        for k in E[(E.image_key == ik) & (E.arm.astype(str) == arm)].key:
            files.add(A2.S3_REC / 'eval' / f'{k}.npz')
    S12V = K.read_csv(K.T1.S12 / 'results' / 'views.csv'); S12V = S12V[S12V.surface_class == 'main'].drop_duplicates('input_key').set_index('input_key')
    for ik in V[V.source == 'S12'].input_key.unique():
        if ik in S12V.index:
            files.add(K.ROOT / S12V.loc[ik, 'eval_record'].replace('.json', '.npz'))
    import s1r_common as S1K
    used_rk = set(S12V.loc[S12V.index.isin(set(V[V.source == 'S12'].input_key) | {r['input_key'] for r in d12 + l12})].raster_key)
    for rk in used_rk:
        for suf in ('.json', '.npz'):
            q = S1K.RASTER_DIR / f'{rk}{suf}'
            if q.exists():
                files.add(q)
    sel = K.load_json(K.OUT1 / 'run.json')['a1_selection']
    for c in sel['cases']:
        if c.get('eval_record'):
            files.add(K.ROOT / c['eval_record'].replace('.json', '.npz')); files.add(K.ROOT / c['eval_record'])
    files = sorted({p.resolve() for p in files})
    miss = [K.rel(p) for p in files if not p.exists()]
    hashes = {K.rel(p): K.sha(p) for p in files if p.exists()}
    import rdy_platform as P
    dg = P.dom_global(); dom = dict(global_hash=dg.get('hash'))
    for s in ('B_curv', 'M10_curv', 'M24_curv'):
        dom[f'local_{s}_hash'] = P.dom_local(s).get('hash')
    return dict(n_files=len(hashes), missing=miss, sha256=hashes, domains=dom)


def recheck_tr01_inputs() -> dict:
    h = K.load_json(K.OUT1 / 'run.json')['plan']['inputs_sha256']
    changed = [f for f, v in h.items() if not (K.ROOT / f).exists() or K.sha(K.ROOT / f) != v]
    return dict(n=len(h), changed=changed)


class Refused(Exception):
    pass


def identity_layers(image_sha: str, decode_id: str, eval_id: str) -> dict:
    return dict(forward=image_sha, decode=decode_id, eval=eval_id)


def check_entry(key: str, arrays: dict, meta: dict, have: dict, want: dict) -> dict:

    if have.get('forward') != want.get('forward'):
        raise Refused(f'{key}: forward image identity changed - nothing reusable')
    if have.get('decode') != want.get('decode'):
        return dict(key=key, reuse=('image',), redo=('decode', 'evaluation'), reason='decode method identity changed')
    miss = [a for a in ARR if a not in arrays]
    if miss:
        raise Refused(f'{key}: missing arrays {miss}')
    n = len(arrays['pts'])
    if arrays['pts'].ndim != 2 or arrays['pts'].shape[1] != 3:
        raise Refused(f'{key}: pts shape {arrays["pts"].shape}')
    bad = [a for a in PER_POINT if len(arrays[a]) != n]
    if bad:
        raise Refused(f'{key}: per-point arrays of wrong length {bad} (n = {n})')
    if len(arrays['feat_truth_dot']) != len(arrays['feat_confirmable']):
        raise Refused(f'{key}: feature arrays of unequal length')
    if not isinstance(meta, dict) or 'metrics' not in meta:
        raise Refused(f'{key}: metrics record missing')
    m = meta['metrics']; mm = [f for f in REQ_METRICS if f not in m]
    if mm:
        raise Refused(f'{key}: metrics fields missing {mm}')
    acc = np.asarray(arrays['accepted'], bool); val = np.asarray(arrays['valid'], bool)
    c, w, u = (int((acc & np.asarray(arrays[a], bool)).sum()) for a in ('point_correct', 'point_wrong', 'point_unconf'))
    if int(m['n_accepted']) != int(acc.sum()) or int(m['n_valid']) != int(val.sum()) or (c, w, u) != (int(m['id_correct']), int(m['id_wrong']), int(m['id_unconf'])):
        raise Refused(f'{key}: metrics do not close against the arrays')
    if m['state'] not in ('ok', 'no_output'):
        raise Refused(f'{key}: illegal state {m["state"]}')
    legal_empty = int(acc.sum()) == 0
    if legal_empty:
        if float(m['glob_Ctau'] or 0.0) != 0.0 or float(m['loc_Ctau'] or 0.0) != 0.0 or not (m['glob_pt_rmse'] is None or (isinstance(m['glob_pt_rmse'], float) and np.isnan(m['glob_pt_rmse']))):
            raise Refused(f'{key}: no output but coverage > 0 or a defined error')
    if have.get('eval') != want.get('eval'):
        return dict(key=key, reuse=('image', 'point_cloud'), redo=('evaluation',), reason='evaluation identity changed', legal_no_output=legal_empty)
    return dict(key=key, reuse=('image', 'point_cloud', 'metrics'), redo=(), legal_no_output=legal_empty,
                undefined=dict(pt_rmse=legal_empty, risk_wrong=(c + w == 0), unconf_share=(c + w + u == 0)))


def read_shard(path) -> dict:

    try:
        with np.load(path, allow_pickle=False) as z:
            names = list(z.files); data = {k: np.asarray(z[k]) for k in names}
    except Exception as e:
        raise Refused(f'{K.rel(path) if str(path).startswith(str(K.ROOT)) else path}: unreadable ({type(e).__name__})')
    out = {}
    for k, v in data.items():
        ik, a = k.split('|', 1)
        if a == 'metrics':
            out.setdefault(ik, [{}, None])[1] = json.loads(str(v))
        else:
            out.setdefault(ik, [{}, None])[0][a] = v
    return out


def check_shards(paths, have_of, want_of) -> dict:
    seen, res, refused = {}, [], []
    for p in paths:
        try:
            S = read_shard(p)
        except Refused as e:
            refused.append(str(e)); continue
        for ik, (arr, meta) in S.items():
            if ik in seen:
                refused.append(f'{ik}: duplicate key in {seen[ik]} and {p.name}'); continue
            seen[ik] = p.name
            try:
                res.append(check_entry(ik, arr, meta, have_of(ik), want_of(ik)))
            except Refused as e:
                refused.append(str(e))
    return dict(n_entries=len(seen), n_checked=len(res), refused=refused, full_reuse=sum(1 for r in res if r['redo'] == ()),
                legal_no_output=sum(1 for r in res if r.get('legal_no_output')), partial=[r for r in res if r['redo']])


def a3_identities() -> dict:

    r1 = K.load_json(K.OUT1 / 'run.json'); h = r1['plan']['inputs_sha256']
    import tr_a3 as A3
    A = A3.plan(); new = A[A.reuse_class == 'new_DF']
    img = {}
    for row in new.to_dict('records'):
        p = K.rel(K.ROOT / row['image_record'].replace('.json', '.npz')) if 'image_record' in row else None
        img[row['input_key']] = dict(path=p, sha_registered=h.get(p), sha_now=(K.sha(K.ROOT / p) if p and (K.ROOT / p).exists() else None))
    return dict(images=img)


# Test reuse decisions for rendering, decoding and evaluation records.
def selftests(real_shard) -> dict:

    tmp = K.guard(K.OUT / '_selftest_tmp' / 'x').parent
    res = {}
    try:
        S = read_shard(real_shard); ik, (arr, meta) = next(iter(S.items()))
        ident = identity_layers('img-A', 'DF|binding|code-1', 'eval-1')
        res['real_entry_full_reuse'] = check_entry(ik, arr, meta, ident, ident)['redo'] == ()
        res['decode_rule_change_reuses_image_only'] = check_entry(ik, arr, meta, ident, identity_layers('img-A', 'DS|binding|code-1', 'eval-1'))['reuse'] == ('image',)
        res['eval_change_keeps_point_cloud'] = check_entry(ik, arr, meta, ident, identity_layers('img-A', 'DF|binding|code-1', 'eval-2'))['reuse'] == ('image', 'point_cloud')
        for name, mutate, want in (
                ('forward_change_refused', lambda a, m: (a, m), identity_layers('img-B', 'DF|binding|code-1', 'eval-1')),
                ('missing_array_refused', lambda a, m: ({k: v for k, v in a.items() if k != 'zr'}, m), ident),
                ('wrong_length_refused', lambda a, m: ({**a, 'valid': a['valid'][:-1]}, m), ident),
                ('missing_metrics_field_refused', lambda a, m: (a, {**m, 'metrics': {k: v for k, v in m['metrics'].items() if k != 'glob_Ctau'}}), ident),
                ('metrics_only_entry_refused', lambda a, m: ({}, m), ident),
                ('metrics_not_closing_refused', lambda a, m: (a, {**m, 'metrics': {**m['metrics'], 'n_accepted': int(m['metrics']['n_accepted']) + 1}}), ident)):
            a2, m2 = mutate(arr, meta)
            try:
                check_entry(ik, a2, m2, ident, want); res[name] = False
            except Refused:
                res[name] = True

        n = len(arr['pts']); empty = {**arr, 'accepted': np.zeros(n, bool), 'valid': np.zeros(n, bool)}
        mE = {**meta, 'metrics': {**meta['metrics'], 'state': 'no_output', 'n_accepted': 0, 'n_valid': 0, 'glob_Ctau': 0.0, 'loc_Ctau': 0.0, 'glob_Cvalid': 0.0,
                                   'loc_Cvalid': 0.0, 'glob_pt_rmse': None, 'id_correct': 0, 'id_wrong': 0, 'id_unconf': 0}}
        r = check_entry(ik, empty, mE, ident, ident)
        res['legal_no_output_kept'] = r['legal_no_output'] and r['undefined']['pt_rmse'] and r['undefined']['risk_wrong'] and r['undefined']['unconf_share']
        bad = {**mE, 'metrics': {**mE['metrics'], 'glob_Ctau': 0.1}}
        try:
            check_entry(ik, empty, bad, ident, ident); res['empty_with_coverage_refused'] = False
        except Refused:
            res['empty_with_coverage_refused'] = True

        f1 = tmp / 'a.npz'; f2 = tmp / 'b.npz'; f3 = tmp / 'c.npz'
        flat = {f'{ik}|{k}': v for k, v in arr.items()}; flat[f'{ik}|metrics'] = np.array(json.dumps(meta, default=str))
        np.savez_compressed(open(K.guard(f1), 'wb'), **flat); np.savez_compressed(open(K.guard(f2), 'wb'), **flat)
        raw = open(f1, 'rb').read(); open(K.guard(f3), 'wb').write(raw[: len(raw) // 2])
        r = check_shards([f1, f2], lambda k: ident, lambda k: ident)
        res['duplicate_key_across_files_refused'] = any('duplicate key' in x for x in r['refused']) and r['n_checked'] == 1
        r = check_shards([f3], lambda k: ident, lambda k: ident)
        res['corrupted_file_refused'] = len(r['refused']) == 1 and r['n_checked'] == 0

        np.savez_compressed(open(K.guard(tmp / 'm.npz'), 'wb'), **{f'{ik}|metrics': np.array(json.dumps(meta, default=str))})
        old_rule_done = any(k.endswith('|metrics') for k in np.load(tmp / 'm.npz').files)
        r = check_shards([tmp / 'm.npz'], lambda k: ident, lambda k: ident)
        res['old_resume_rule_would_accept_metrics_only'] = bool(old_rule_done)
        res['new_guard_refuses_metrics_only'] = r['n_checked'] == 0 and len(r['refused']) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    passed = all(v for k, v in res.items())
    return dict(passed=passed, checks=res, temp_dir_removed=not tmp.exists())
