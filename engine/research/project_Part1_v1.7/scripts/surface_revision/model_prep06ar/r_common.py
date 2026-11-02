# Define candidate extensions and shared environment-study settings.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'scripts' / 'surface_revision'
M6A_DIR = SR / 'model_prep06a'
for _q in (M6A_DIR,):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))
import m6_common as K6

K5 = K6.K5
OUT = ROOT / 'outputs' / 'surface_revision' / 'model_prep06ar'
LOGS = OUT / 'logs'
SRC6 = ROOT / 'outputs' / 'surface_revision' / 'model_prep06a'
SRC5 = ROOT / 'outputs' / 'surface_revision' / 'model_prep05'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK_BOOK = DRAFTS / 'MODEL-PREP-06A-R_protocol.md'

TASK6 = DRAFTS / 'MODEL-PREP-06A_protocol.md'

TASK_ID = 'MODEL-PREP-06A-R'
FREEZE = OUT / 'freeze.json'
PROTOCOL = OUT / 'protocol.json'
REDIRECT_ENV = 'M6R_FREEZE_REDIRECT'
SURFS = ('B_curv', 'M10_curv', 'M24_curv')
BLOCKS, REPS = (0, 1, 2, 3), (1, 2)
TAUS = (0.10, 0.15, 0.20, 0.30)
DS = (5.0, 6.0)
TOL = 1e-10


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f'{code}: {msg}'); self.code = code; self.msg = msg


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).resolve().as_posix()


def absp(r: str) -> Path:
    p = Path(r)
    return p if p.is_absolute() else ROOT / r


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def undef(x) -> bool:
    return x is None or (isinstance(x, (float, np.floating)) and math.isnan(float(x)))


def num(x):
    return None if undef(x) else float(x)


clean = K6.clean
digest = K6.digest


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() != q and OUT.resolve() not in q.parents:
        raise Stop('refused', f'write outside {rel(OUT)} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def atomic(p, write) -> Path:
    q = guard(p); t = q.with_name(q.name + f'.tmp{os.getpid()}')
    write(t)
    for _ in range(200):
        try:
            os.replace(t, q)
            return q
        except PermissionError:
            time.sleep(0.05)
    raise Stop('technical', f'cannot replace {q}')


def save_json(p, obj) -> Path:
    return atomic(p, lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'))


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save_csv(df, p) -> Path:
    return atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


def protocol() -> dict:
    return load_json(PROTOCOL) if PROTOCOL.exists() else dict(task=TASK_ID, task_book=str(TASK_BOOK),   created=now())


def protocol_put(section: str, obj) -> dict:
    p = protocol(); p[section] = clean(obj); p['updated'] = now(); save_json(PROTOCOL, p)
    return p


def _redirect() -> dict:
    s = os.environ.get(REDIRECT_ENV)
    return json.loads(s) if s else {}


def hash_set(paths) -> dict:
    red = _redirect(); out = {}
    for r in paths:
        p = Path(red[r]) if r in red else absp(r)
        out[r] = sha_file(p) if p.exists() else None
    return out


def verify_freeze(stage: str) -> dict:

    if not FREEZE.exists():
        raise Stop('freeze', f'{stage}: no freeze.json (run the stage freeze after the isolation tests)')
    fz = load_json(FREEZE)
    cur = hash_set(list(fz['files']))
    bad = sorted(r for r, h in fz['files'].items() if cur.get(r) != h)
    if bad:
        raise Stop('freeze', f'{stage}: {len(bad)} frozen file(s) differ from freeze.json version {fz["version"]}: {bad[:6]}')
    if _redirect() and stage != 'verify_only':
        raise Stop('refused', f'{stage}: freeze redirect set (isolation test mode); production stages do not run')
    return dict(version=fz['version'], n_files=len(fz['files']), freeze_sha256=sha_file(FREEZE), t=now())
