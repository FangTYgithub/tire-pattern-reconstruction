# Provide result paths and dependency checks for encoding diagnostics.

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(ROOT / 'scripts' / 'surface_revision' / 's3_code07'), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c7_common as K7

K4 = K7.K4
OUT = ROOT / 'outputs' / 'surface_revision' / 's3_code08'
for _k, _sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
    os.environ[_k] = str(OUT / 'sandbox' / _sub)
os.environ['LINE01RB_RENDER_CAP'] = '0'
FIGS = OUT / 'figures'
CHECKS_FILE = OUT / 'checks.json'
C7 = K7.OUT
C7_CUR = C7 / 'analysis' / 'formal' / 'current'
C7_REC = C7 / 'formal_run' / 'records'
C7_RUN = C7 / 'formal_run' / 'run'
TASK_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-08_protocol.md'
REVIEW_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-07_review.md'
START = '2026-10-16T09:00:00+02:00'

Stop, now_local, sha_file, clean, digest, rel = K4.Stop, K4.now_local, K4.sha_file, K4.clean, K4.digest, K4.rel
load_json, load_npz = K4.load_json, K4.load_npz


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


def checks_add(name: str, obj: dict) -> None:
    c = load_json(CHECKS_FILE) if CHECKS_FILE.exists() else dict(task='S3-CODE-08', created=now_local(), note='Numerical consistency checks.')
    if name in c:
        raise Stop('check', f'checks entry {name} exists; never overwritten')
    c[name] = dict(obj, t_local=now_local()); save_json(CHECKS_FILE, c)


def tables() -> dict:

    import pandas as pd
    T = {}
    for k, f in dict(C='cases.csv', P='pairs.csv', SC='case_phase_summary.csv', SP='pair_phase_summary.csv', B='bootstrap.csv', R='risk_summary.csv', RO='readout_summary.csv',
                     DS='diag_summary.csv', CM='condition_map.csv').items():
        X = pd.read_csv(C7_CUR / f, low_memory=False)
        if 'layout' in X.columns:
            X['layout'] = X.layout.astype(str)
        T[k] = X
    return T
