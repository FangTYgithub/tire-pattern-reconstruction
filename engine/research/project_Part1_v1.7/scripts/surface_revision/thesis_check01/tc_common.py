# Provide input identities and output helpers for cross-study diagnostics.

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'outputs' / 'surface_revision'
OUT = SR / 'thesis_check01'
RUN = OUT / 'run.json'
FIGS = OUT / 'figures'
PARTS = OUT / '_parts'
CACHE = OUT / 'cache'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK = DRAFTS / 'THESIS-CHECK-01_protocol.md'
S12, R7, R4, R2 = SR / 's12_resource01', SR / 's4_real07', SR / 's4_real04', SR / 's4_real02'
for _p in (ROOT / 'src', ROOT / 'scripts' / 'surface_revision' / 'surf02', ROOT / 'scripts' / 'surface_revision' / 'sections_adapter', ROOT / 'scripts' / 'surface_revision' / 's2_prepare',
           ROOT / 'scripts' / 'surface_revision' / 's12_resource01', ROOT / 'scripts' / 'surface_revision' / 's4_real02', ROOT / 'scripts' / 'surface_revision' / 's345_ready',
           ROOT / 'scripts' / 'surface_revision' / 's4_real06', HERE):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

MAIN = ('B_curv', 'M10_curv', 'M24_curv')
SHORT = {'B_curv': 'B', 'M10_curv': 'M10', 'M24_curv': 'M24'}
PLATFORM = {'M10_curv': 'A_curv', 'M24_curv': 'C_curv', 'B_curv': 'B_curv'}
D_MAIN, TAU = 6.0, 0.15
D_SENS = (5.0, 6.0, 7.0)
WINDOW = (-16.0, 4.0)
READ: dict = {}


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}'); self.code = code; self.msg = msg


def now_local() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(p).resolve())


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def digest(obj, n: int = 24) -> str:
    return sha_bytes(json.dumps(clean(obj), sort_keys=True, default=str).encode())[:n]


def reg(p) -> Path:

    p = Path(p)
    if not p.exists():
        raise Stop('missing', f'source missing: {rel(p)}')
    r = rel(p)
    if r not in READ:
        READ[r] = sha_file(p)
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
    return _atomic(p, lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'))


def save_csv(df: pd.DataFrame, p) -> Path:
    return _atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


def save_npz(p, **arrays) -> str:
    _atomic(p, lambda t: np.savez_compressed(open(t, 'wb'), **arrays))
    return sha_file(p)


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='THESIS-CHECK-01', task_book=rel(TASK),  acceptance='reference (independent)', created=now_local())


def run_put(section: str, obj) -> dict:
    r = run_load()
    if section in r.get('frozen', []):
        raise Stop('freeze', f'run.json section {section} is fixed (written before the computation); never rewritten')
    r[section] = obj; r['updated'] = now_local(); save_json(RUN, r)
    return r


def run_fix(section: str, obj) -> dict:

    r = run_load()
    if section in r:
        if digest(r[section]) != digest(obj):
            raise Stop('freeze', f'{section} already written with a different content')
        return r
    r[section] = obj; r.setdefault('frozen', []).append(section); r['updated'] = now_local(); save_json(RUN, r)
    return r


def flush_reads(section: str) -> None:
    r = run_load(); ir = r.setdefault('sources_read', {})
    for k, v in READ.items():
        if k in ir and ir[k] != v:
            raise Stop('input', f'{k} changed during the work ({section})')
        ir.setdefault(k, v)
    r['updated'] = now_local(); save_json(RUN, r)


def part(name: str, df: pd.DataFrame) -> Path:
    return save_csv(df, PARTS / f'{name}.csv')


def s2():
    import s2_common as S
    return S
