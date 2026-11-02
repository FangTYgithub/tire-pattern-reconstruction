# Provide shared settings for the complete encoding experiment.

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(ROOT / 'scripts' / 'surface_revision' / 's3_code06'), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c6_common as K6

K5, K4 = K6.K5, K6.K4
OUT = ROOT / 'outputs' / 'surface_revision' / 's3_code07'
SANDBOX = OUT / 'sandbox'
for _k, _sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
    os.environ[_k] = str(SANDBOX / _sub)
os.environ['LINE01RB_RENDER_CAP'] = '0'
FORMAL, PLAN, RESULTS, FIGS = OUT / 'formal', OUT / 'plan', OUT / 'results', OUT / 'figures'
CHECKS_FILE = OUT / 'checks.json'
CONFIGS = HERE / 'configs'
TASK_ID = 'S3-CODE-07'
TASK_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-07_protocol.md'
REVIEW_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-06_review.md'
O4, O6 = K4.OUT, K6.OUT
PERMIT_SCOPE = 'S3-CODE-07 formal section 3 (6,336 images, N0-N7)'

Stop, now_local, sha_file, clean, digest, rel = K4.Stop, K4.now_local, K4.sha_file, K4.clean, K4.digest, K4.rel
load_json, load_npz, norm_seed, is_null, Hashes = K4.load_json, K4.load_npz, K5.norm_seed, K5.is_null, K6.Hashes
EXIT = dict(K6.EXIT, interrupted=9, products=19, disk=20)
TOL = 1e-12
PARITY_OF_PHASE = dict(K4.PARITY_OF_PHASE)


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in (q, *q.parents):
        raise Stop('technical', f'write outside {rel(OUT)} refused: {q}')
    return q


def _atomic(p: Path, write) -> Path:
    p = guard(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp'); write(tmp)
    for _ in range(50):
        try:
            os.replace(tmp, p); return p
        except PermissionError:
            time.sleep(0.1)
    raise Stop('technical', f'atomic replace failed for {p}')


def save_json(p, obj) -> Path:
    return _atomic(Path(p), lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'))


def save_csv(df, p) -> Path:
    return _atomic(Path(p), lambda t: df.to_csv(t, index=False, encoding='utf-8'))


def save_npz(p, **arrays) -> str:
    def w(t):
        with open(t, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
    return sha_file(_atomic(Path(p), w))


def append_line(p, obj) -> None:
    p = guard(p); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(obj), ensure_ascii=False) + '\n')


def checks_add(name: str, obj: dict) -> None:

    c = load_json(CHECKS_FILE) if CHECKS_FILE.exists() else dict(task=TASK_ID, created=now_local(), note='Numerical consistency checks.')
    if name in c:
        raise Stop('check', f'checks entry {name} exists; never overwritten')
    c[name] = dict(obj, t_local=now_local()); save_json(CHECKS_FILE, c)


def ref(path) -> dict:
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    return dict(file=rel(p), sha256=sha_file(p))


def abspath(p) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p
