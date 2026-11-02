# Provide guarded storage and attempt records for supplementary evaluations.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'scripts' / 'surface_revision'
for _p in (str(SR / 's3_code03'),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c3_common as K3

OUT = ROOT / 'outputs' / 'surface_revision' / 'thesis_reinforce01'
SANDBOX = OUT / 'sandbox'
for _k, _sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
    os.environ[_k] = str(SANDBOX / _sub)
os.environ['LINE01RB_RENDER_CAP'] = '0'
for _p in (str(HERE),) + tuple(str(SR / q) for q in ('s12_resource01', 'sections_adapter', 's2_prepare', 's345_ready', 's3_code01', 's3_code02')):
    if _p not in sys.path:
        sys.path.append(_p)

RUN = OUT / 'run.json'
SHARDS = OUT / 'shards'
FIGS = OUT / 'figures'
LEDGER = OUT / 'ledger.jsonl'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK = DRAFTS / 'THESIS-REINFORCE-01_protocol.md'
MODEL_PREP_TASK = DRAFTS / 'MODEL-PREP-01_protocol.md'
MODEL_PREP_OUT = ROOT / 'outputs' / 'surface_revision' / 'model_prep01'
MODEL_PREP_SCRIPTS = SR / 'model_prep01'
S12 = ROOT / 'outputs' / 'surface_revision' / 's12_resource01'
C4 = ROOT / 'outputs' / 'surface_revision' / 's3_code04'
TC = ROOT / 'outputs' / 'surface_revision' / 'thesis_check01'
RK = K3.RK
CAPS = dict(new_imaging=0, decode_total=880, decode_replay=12, eval_calls_A1=900)
Stop = K3.Stop
EXIT = dict(K3.EXIT, budget=3)
clean, digest, rel, load_json, load_npz = K3.clean, K3.digest, K3.rel, K3.load_json, K3.load_npz


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() != q and OUT.resolve() not in q.parents:
        raise Stop('guard', f'write outside {rel(OUT)} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def _atomic(p, write) -> Path:
    q = guard(p); t = q.with_name(q.name + f'.tmp{os.getpid()}'); write(t)
    for _ in range(100):
        try:
            os.replace(t, q); return q
        except PermissionError:
            time.sleep(0.05)
    raise Stop('technical', f'cannot replace {q}')


def save_json(p, obj) -> Path:
    return _atomic(p, lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, default=str), encoding='utf-8'))


def save_csv(df, p) -> Path:
    return _atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


def save_npz(p, **arrays) -> str:
    q = _atomic(p, lambda t: np.savez_compressed(open(t, 'wb'), **arrays))
    return sha(q)


def read_csv(p, **kw):
    import pandas as pd
    return pd.read_csv(p, low_memory=False, float_precision='round_trip', **kw)


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='THESIS-REINFORCE-01', task_book=str(TASK),   created=now())


def run_put(section: str, obj) -> dict:
    r = run_load(); r[section] = clean(obj); r['updated'] = now(); save_json(RUN, r)
    return r


def ledger(kind: str, key: str, status: str, **info) -> None:
    q = guard(LEDGER)
    with open(q, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(dict(t=now(), pid=os.getpid(), kind=kind, key=key, status=status, **info)), ensure_ascii=False, default=str) + chr(10))


def ledger_rows() -> list:
    if not LEDGER.exists():
        return []
    return [json.loads(x) for x in LEDGER.read_text(encoding='utf-8').splitlines() if x.strip()]


def count(kind_prefix: str, status: str = 'reserved') -> int:
    return sum(1 for r in ledger_rows() if r['kind'].startswith(kind_prefix) and r['status'] == status)


def reserve(kind: str, key: str, cap_kind: str, cap: int, **info) -> None:

    used = count(cap_kind)
    if used + 1 > cap:
        raise Stop('budget', f'{cap_kind}: {used} reserved calls, cap {cap}')
    ledger(kind, key, 'reserved', **info)
