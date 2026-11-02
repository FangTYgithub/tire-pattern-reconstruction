# Provide storage and identity helpers for encoding-result checks.

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S4 = ROOT / 'scripts' / 'surface_revision' / 's3_code04'
for _p in (str(S4), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c4_common as K4

OUT = ROOT / 'outputs' / 'surface_revision' / 's3_code05'
SANDBOX = OUT / 'sandbox'
BATCHES = OUT / 'batches'
FORMAL = OUT / 'formal'
CHECKS_FILE = OUT / 'checks.json'
CONFIGS = HERE / 'configs'
TASK_ID = 'S3-CODE-05'
TASK_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-05_protocol.md'
REVIEW_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-04_review.md'
O3, O4 = K4.O3, K4.OUT

Stop, now_local, sha_file, clean, digest, rel = K4.Stop, K4.now_local, K4.sha_file, K4.clean, K4.digest, K4.rel
load_json, load_npz = K4.load_json, K4.load_npz
EXIT = dict(K4.EXIT, input=6, structure=7, records=8, columns=10, semantics=11, pairing=12, config=17)
TOL = 1e-12
PARITY_OF_PHASE = K4.PARITY_OF_PHASE


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


def checks_add(name: str, obj: dict) -> None:

    c = load_json(CHECKS_FILE) if CHECKS_FILE.exists() else dict(task=TASK_ID, created=now_local())
    if name in c:
        raise Stop('check', f'checks entry {name} exists; never overwritten')
    c[name] = dict(obj, t_local=now_local()); save_json(CHECKS_FILE, c)


def ref(path) -> dict:
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    return dict(file=rel(p), sha256=sha_file(p))


def is_null(x) -> bool:

    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == '' or x.strip().lower() in ('nan', 'none')
    try:
        return bool(math.isnan(float(x)))
    except (TypeError, ValueError):
        return False


def norm_seed(x):

    if is_null(x):
        return None
    f = float(x)
    if not f.is_integer():
        raise Stop('check', f'seed {x!r} is not an integer')
    return int(f)


def dir_digest(paths) -> str:

    items = []
    for d in paths:
        d = Path(d)
        if d.exists():
            for f in sorted(d.rglob('*')):
                if f.is_file():
                    st = f.stat(); items.append((rel(f), st.st_size, st.st_mtime_ns))
    return digest(items, 24)
