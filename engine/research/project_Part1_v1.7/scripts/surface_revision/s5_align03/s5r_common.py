# Define joint-study result keys and shared storage utilities.

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
S345 = SR / 's345_ready'
A2 = SR / 's5_align02'
C5 = SR / 's3_code05'
OUT = SR / 's5_align03'
RUN = OUT / 'run.json'
FIGS = OUT / 'figures'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK = DRAFTS / 'S5-ALIGN-03_protocol.md'
REVIEW = DRAFTS / 'S5-ALIGN-02_review.md'
MANIFEST = A2 / 'manifest_next.csv'


DOT9 = {'D_r1_p6x4': (1.0, 6.0, 4.0), 'D_r1_p10x4': (1.0, 10.0, 4.0), 'D_r1_p8x4': (1.0, 8.0, 4.0), 'D_r1_p8x6': (1.0, 8.0, 6.0), 'D_r1.5_p6x4': (1.5, 6.0, 4.0),
        'D_r1.5_p10x4': (1.5, 10.0, 4.0), 'D_r1.5_p8x4': (1.5, 8.0, 4.0), 'D_r1.5_p8x6': (1.5, 8.0, 6.0), 'D_r0.75_p8x4': (0.75, 8.0, 4.0)}
LINE6 = {'L_w0.5_s4.45': (0.5, 4.45), 'L_w0.5_s6.3': (0.5, 6.3), 'L_w0.5_s8': (0.5, 8.0), 'L_w1.5_s4.45': (1.5, 4.45), 'L_w1.5_s6.3': (1.5, 6.3), 'L_w1.5_s8': (1.5, 8.0)}
COMBOS = tuple(DOT9) + tuple(LINE6)
MAIN = ('B_curv', 'M10_curv', 'M24_curv')
PLATFORM = {'M10_curv': 'A_curv', 'M24_curv': 'C_curv', 'B_curv': 'B_curv', 'M10_curv_gauss': 'A_curv_gauss'}
PHASES = tuple(f'ph{i}' for i in range(8))
SIGMAS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
REPS = {0.0: (0,), 20.0: (1, 2), 40.0: (1, 2), 60.0: (1, 2), 80.0: (1, 2), 100.0: (1, 2)}
EXISTING_SIGMAS = (0.0, 40.0)
PLATFORM_VERSION = 'SURF-02 v1.1 (2026-10-12)'
PSF_PX = 1.0
Q = 11695.0
INTEGRAL_TOL = Q * 2.0 ** -24 + 1e-9
MAIN_METHOD = {'point': 'D0', 'line': 'P4'}
NEXT8 = (('D_r0.75_p8x4', 'B_curv', 'ph0'), ('D_r1.5_p6x4', 'M24_curv', 'ph1'), ('L_w0.5_s6.3', 'M10_curv', 'ph1'), ('L_w1.5_s4.45', 'B_curv', 'ph0'))
HEAVY = ('m3_virtual_camera', 'r5_detect', 'm4_reconstruction', 'm4_lines', 'surface_eval_r3', 's3c_dev', 'adp_surf02', 's2_common')
READ: dict = {}


def family(combo: str) -> str:
    return 'point' if combo.startswith('D_') else 'line'


def expected_keys(combos=COMBOS, surfaces=MAIN, phases=PHASES, sigmas=SIGMAS) -> set:

    return {(c, s, p, float(sg), int(r)) for c in combos for s in surfaces for p in phases for sg in sigmas for r in REPS[float(sg)]}


class Stop(Exception):
    def __init__(self, code: str, msg: str, detail=None):
        super().__init__(f'[{code}] {msg}'); self.code = code; self.msg = msg; self.detail = detail


EXIT = dict(ok=0, refused=3, technical=2, mismatch=4, corrupt=5, lock=6, forbidden=7, version=8, missing=9, scope=10, permit=11, guard=12, input=13)


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha(p) -> str:
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
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, Path):
        return rel(o)
    return o


def digest(obj, n: int = 24) -> str:
    return sha_bytes(json.dumps(clean(obj), sort_keys=True, default=str).encode())[:n]


def reg(p) -> Path:
    p = Path(p)
    if not p.exists():
        raise Stop('missing', f'source missing: {rel(p)}')
    r = rel(p)
    if r not in READ:
        READ[r] = sha(p)
    return p


def guard(p, base: Path = OUT) -> Path:
    q = Path(p).resolve()
    if base.resolve() not in q.parents and q != base.resolve():
        raise Stop('guard', f'write outside {rel(base)} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def atomic(p, write, base: Path = OUT) -> Path:
    q = guard(p, base); t = q.with_name(q.name + f'.tmp{os.getpid()}'); write(t)
    for _ in range(100):
        try:
            os.replace(t, q); return q
        except PermissionError:
            time.sleep(0.05)
    raise Stop('technical', f'atomic replace failed for {q}')


def save_json(p, obj, base: Path = OUT) -> Path:
    return atomic(p, lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'), base)


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def load_npz(p) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='S5-ALIGN-03', task_book=rel(TASK),  acceptance='reference (independent)', created=now())


def run_put(section: str, obj) -> dict:
    r = run_load(); r[section] = obj; r['updated'] = now(); save_json(RUN, r)
    return r


def flush(section: str) -> None:
    r = run_load(); ir = r.setdefault('sources_read', {})
    for k, v in READ.items():
        if k in ir and ir[k] != v:
            raise Stop('input', f'{k} changed during the work ({section})')
        ir.setdefault(k, v)
    r['updated'] = now(); save_json(RUN, r)


def heavy_loaded() -> list:

    return sorted(m for m in HEAVY if m in sys.modules)


def num(x):

    if x is None or isinstance(x, (bool, np.bool_, str)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def frame(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows)


def read_csv(p, **kw) -> pd.DataFrame:

    return pd.read_csv(p, low_memory=False, float_precision='round_trip', **kw)
