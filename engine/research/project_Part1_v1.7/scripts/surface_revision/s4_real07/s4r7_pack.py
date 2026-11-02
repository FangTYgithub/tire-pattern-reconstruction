# Assemble the held-out layout results and report tables.

from __future__ import annotations

import json

import pandas as pd

import s4r7_common as H

K = H.K


def _flat(d: dict) -> dict:
    return {k: (json.dumps(K.clean(v), ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in d.items()}


def assemble() -> dict:
    r = H.run_load(); rows = []
    p = H.OUT / '_parts' / 'A1_union.csv'
    parts = [pd.read_csv(p, low_memory=False)] if p.exists() else []
    for k, v in (r.get('A2', {}).get('tests') or {}).items():
        rows.append(dict(record_type='A2_isolation', test=k, ok=bool(v)))
    for c in r.get('A3', {}).get('cases', []):
        rows.append(dict(record_type='A3_regression', **_flat(c)))
    for k, v in (r.get('drill', {}).get('checks') or {}).items():
        rows.append(dict(record_type='drill', test=k, ok=bool(v)))
    for lab in ('run_subset', 'run_all'):
        for c in (r.get(lab, {}).get('pair_checks') or {}).get('cases', []):
            rows.append(dict(record_type=f'pair_invariant_{lab[4:]}', **_flat(c)))
    for c in r.get('selfcheck', {}).get('cases', []):
        rows.append(dict(record_type='selfcheck', **_flat(c)))
    for c in r.get('freeze_checks', []):
        rows.append(dict(record_type='freeze_check', test=c['stage'], ok=c['ok'], detail=json.dumps(c['differences'])))
    parts.append(pd.DataFrame(rows))
    D = pd.concat(parts, ignore_index=True, sort=False); D = D[['record_type'] + [c for c in D.columns if c != 'record_type']]
    H.save_csv(D, H.OUT / 'checks.csv')
    return dict(rows=len(D), record_types=D.record_type.value_counts().to_dict())


def final() -> dict:
    import s4r7_bind as B
    fc = H.freeze_check('end', data_digests=B.data_digests())
    r = H.run_load(); bad = [p for p, h in r.get('inputs_read', {}).items() if K.sha_file(K.ROOT / p) != h]
    r6 = json.loads((H.R6 / 'run.json').read_text(encoding='utf-8'))['final']
    r6_ok = all(K.sha_file(K.ROOT / k) == v for k, v in list(r6['outputs'].items()) + list(r6['final_code'].items()))
    rb, db = H.render_budget().state(), H.decode_budget().state()
    outs = {K.rel(p): K.sha_file(p) for p in sorted(H.OUT.rglob('*')) if p.is_file() and p.name != 'run.json'
            and not any(x in p.parts for x in ('_parts', 'decodes', 'evals', 'images', '_entry', '_isolation', 'run')) and '.tmp' not in p.name}
    cnt = dict(images_new=len(list(H.IMG7.glob('*.json'))), decodes_new=len(list(H.DEC7.glob('*.json'))), evals=len(list(H.EVL7.glob('*.json'))),
               regression_decodes=len(list((H.ENTRY / 'decodes').glob('*.json'))), regression_evals=len(list((H.ENTRY / 'evals').glob('*.json'))))
    res = dict(freeze_end=fc, inputs_unchanged=not bad, changed=bad[:10], n_inputs=len(r.get('inputs_read', {})), s4_real06_final_unchanged=bool(r6_ok),
               render_budget=rb, decode_budget=db, counts=cnt, final_code=H.own_code(), outputs=outs, t_local=K.now_local())
    H.run_put('final', res)
    return {k: v for k, v in res.items() if k not in ('outputs', 'final_code')}
