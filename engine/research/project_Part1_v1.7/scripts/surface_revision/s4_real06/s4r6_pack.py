# Assemble layout summary tables and report artifacts.

from __future__ import annotations

import json

import pandas as pd

import s4r6_common as H

K = H.K


def assemble() -> dict:
    parts = []
    for name in ('A1', 'A2', 'gate_raster'):
        p = H.OUT / '_parts' / f'{name}.csv'
        if p.exists():
            d = pd.read_csv(p, low_memory=False)
            if 'record_type' not in d.columns:
                d['record_type'] = name
            parts.append(d)
    r = H.run_load(); rows = []
    for c in r.get('gate_entry', {}).get('cases', []):
        rows.append(dict(record_type='gate_entry', **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in c.items()}))
    for k, v in (r.get('A4', {}).get('tests') or {}).items():
        rows.append(dict(record_type='A4_isolation', test=k, ok=v['ok'], detail=json.dumps({a: b for a, b in v.items() if a != 'ok'}, default=str)))
    for lab in ('run_subset', 'run_all'):
        for c in (r.get(lab, {}).get('pair_checks') or {}).get('cases', []):
            rows.append(dict(record_type=f'pair_invariant_{lab[4:]}', **c))
    for c in r.get('selfcheck', {}).get('cases', []):
        rows.append(dict(record_type='selfcheck', **c))
    parts.append(pd.DataFrame(rows))
    D = pd.concat(parts, ignore_index=True, sort=False); D = D[['record_type'] + [c for c in D.columns if c != 'record_type']]
    H.save_csv(D, H.OUT / 'checks.csv')
    return dict(rows=len(D), record_types=D.record_type.value_counts().to_dict())


def final() -> dict:
    r = H.run_load(); bad = [p for p, h in r.get('inputs_read', {}).items() if K.sha_file(K.ROOT / p) != h]
    a05 = {p: K.sha_file(K.ROOT / p) == h for p, h in H.reference_R05.items()}; a02 = {p: K.sha_file(K.ROOT / p) == h for p, h in H.reference_R02.items()}
    rb, db = H.render_budget().state(), H.decode_budget().state()
    outs = {K.rel(p): K.sha_file(p) for p in sorted(H.OUT.rglob('*')) if p.is_file() and p.name != 'run.json' and not any(x in p.parts for x in ('_parts', 'decodes', 'evals', 'images', '_entry', '_isolation'))}
    cnt = dict(images_new=len(list(H.IMG6.glob('*.json'))), decodes_new=len(list(H.DEC6.glob('*.json'))), evals=len(list(H.EVL6.glob('*.json'))),
               entry_decodes=len(list((H.OUT / '_entry' / 'decodes').glob('*.json'))))
    res = dict(inputs_unchanged=not bad, changed=bad[:10], n_inputs=len(r.get('inputs_read', {})), s4_real05_unchanged=all(a05.values()), s4_real02_anchors_unchanged=all(a02.values()),
               frozen_sections=H.run_verify_frozen()['sections'], render_budget=rb, decode_budget=db, counts=cnt, final_code=H.own_code(), outputs=outs, t_local=K.now_local())
    H.run_put('final', res)
    return {k: v for k, v in res.items() if k not in ('outputs', 'final_code')}
