# Provide configuration, dependency and budget helpers for clipping comparisons.

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / 'scripts' / 'surface_revision' / 's4_real02', ROOT / 'scripts' / 'surface_revision' / 's4_real04', HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import r2_common as K

SR = ROOT / 'outputs' / 'surface_revision'
OUT = SR / 's4_real05'
RUN = OUT / 'run.json'
FIGS = OUT / 'figures'
RUNDIR = OUT / 'run'
DEC = OUT / 'decodes'
R2, R4 = SR / 's4_real02', SR / 's4_real04'
S345 = SR / 's345_ready'
SPAT03 = ROOT / 'outputs' / 'spatial_layout' / 'spat03'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK = DRAFTS / 'S4-REAL-05_protocol.md'
REVIEW = DRAFTS / 'S4-REAL-04_review.md'
reference_R04 = {'outputs/surface_revision/s4_real04/report_zh.md': '46833945144ed03f48e3dc657ccd8374beddb9b13aab673884e45eda3e800625',
             'outputs/surface_revision/s4_real04/run.json': 'b45f4d88da5073346cd6389a28169cb4d4998d974d29ed291b7f12abd0da2585',
             'outputs/surface_revision/s4_real04/diagnostics.csv': '65f17b012a7a46a2b90218073f13847493cfac67158727503fa8a9f65951def0',
             'outputs/surface_revision/s4_real04/geometry_scan.csv': '5357d8f89b8e5183f92ec1963e67f0376c33464b9ba7d594d8cb78018d65d021'}
reference_R02 = {'outputs/surface_revision/s4_real02/run.json': 'd9020731a1dabc847df284906f6ae528ef9180ed11ba85efec73c18db8c06e38',
             'outputs/surface_revision/s4_real02/manifest.csv': 'c9689e279f089ba7f42e337a1300394131fee7a3add8115a9392e7d9956980a1',
             'outputs/surface_revision/s4_real02/case_metrics.csv': '9c10c7fcc96649b59fd97463e1017f26ec3026385264c9481ab0a924e78e0bf6',
             'outputs/surface_revision/s4_real02/layouts.json': '20f28d4bc275c615224a9ae89135028b9815d96f3444947ee085a40a58728f16'}
CAP_DECODE = 96
Q = 11695.0

READ: dict = {}


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}'); self.code = code; self.msg = msg


def reg(p) -> Path:

    p = Path(p); r = K.rel(p)
    if r not in READ and p.exists():
        READ[r] = K.sha_file(p)
    return p


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in q.parents and q != OUT.resolve():
        raise Stop('guard', f'write outside {OUT} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def _atomic(p, write) -> Path:
    q = guard(p); t = q.with_name(q.name + f'.tmp{os.getpid()}'); write(t)
    for _ in range(100):
        try:
            os.replace(t, q); return q
        except PermissionError:
            time.sleep(0.05)
    raise Stop('technical', f'atomic replace failed for {q}')


def save_json(p, obj) -> Path:
    return _atomic(p, lambda t: t.write_text(json.dumps(K.clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'))


def save_csv(df: pd.DataFrame, p) -> Path:
    return _atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


def save_npz(p, **arrays) -> str:
    _atomic(p, lambda t: np.savez_compressed(open(t, 'wb'), **arrays))
    return K.sha_file(p)


def read_csv(p, **kw) -> pd.DataFrame:
    return pd.read_csv(reg(p), low_memory=False, float_precision='round_trip', **kw)


def run_load() -> dict:
    return json.loads(RUN.read_text(encoding='utf-8')) if RUN.exists() else dict(task='S4-REAL-05', task_book=K.rel(TASK),  acceptance='reference (independent)',
                                                                                created=K.now_local())


def run_put(section: str, obj, freeze: bool = False) -> dict:
    r = run_load(); fz = r.get('frozen', {})
    if section in fz:
        raise Stop('freeze', f'run.json section {section} is frozen; never rewritten')
    if freeze:
        r.setdefault('frozen', {})[section] = dict(digest=K.digest(obj, 32), t_local=K.now_local())
    r[section] = obj; r['updated'] = K.now_local(); save_json(RUN, r)
    return r


def run_verify_frozen() -> dict:
    r = run_load(); bad = [s for s, v in r.get('frozen', {}).items() if K.digest(r.get(s), 32) != v['digest']]
    if bad:
        raise Stop('freeze', f'frozen run.json sections changed: {bad}')
    return dict(ok=True, sections=sorted(r.get('frozen', {})))


def flush_reads(section: str) -> None:
    r = run_load(); ir = r.setdefault('inputs_read', {})
    for k, v in READ.items():
        if k in ir and ir[k] != v:
            raise Stop('input', f'{k} changed during the run ({section})')
        ir.setdefault(k, v)
    r.setdefault('inputs_read_by_section', {})[section] = sorted(READ)
    r['updated'] = K.now_local(); save_json(RUN, r)


def part(name: str, df: pd.DataFrame) -> None:
    save_csv(df, OUT / '_parts' / f'{name}.csv')


class Budget:


    def __init__(self):
        self.f = RUNDIR / 'budget.json'; self.a = RUNDIR / 'attempts.jsonl'

    def state(self) -> dict:
        return json.loads(self.f.read_text(encoding='utf-8')) if self.f.exists() else dict(cap=CAP_DECODE, used=0, by_purpose={})

    def reserve(self, key: str, purpose: str, limit: int | None = None) -> int:
        b = self.state()
        if int(b['cap']) != CAP_DECODE:
            raise Stop('budget', f"budget cap {b['cap']} != {CAP_DECODE}")
        if b['used'] >= b['cap']:
            raise Stop('budget', f"decode cap exhausted ({b['used']} / {b['cap']})")
        if limit is not None and b['by_purpose'].get(purpose, 0) >= limit:
            raise Stop('budget', f"purpose {purpose} cap {limit} exhausted")
        b['used'] += 1; b['by_purpose'][purpose] = b['by_purpose'].get(purpose, 0) + 1
        save_json(self.f, b); self.log(dict(n=b['used'], key=key, purpose=purpose, event='reserve'))
        return b['used']

    def log(self, rec: dict) -> None:
        guard(self.a)
        with open(self.a, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(K.clean(dict(rec, t=K.now_local())), ensure_ascii=False) + '\n')


def code_hashes(paths) -> dict:
    return {K.rel(p): K.sha_file(p) for p in paths}


def own_code() -> dict:
    return code_hashes(sorted(HERE.glob('*.py')))
