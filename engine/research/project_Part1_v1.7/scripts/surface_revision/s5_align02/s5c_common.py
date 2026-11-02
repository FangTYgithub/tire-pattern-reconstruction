# Provide result identities and storage for joint size-spacing experiments.

from __future__ import annotations

import datetime
import hashlib
import json
import os

import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'outputs' / 'surface_revision'
S345 = SR / 's345_ready'
S12 = SR / 's12_resource01'
TC = SR / 'thesis_check01'
A1 = SR / 's5_align01'
C5 = SR / 's3_code05'
OUT = SR / 's5_align02'
RUN = OUT / 'run.json'
PARTS = OUT / '_parts'
DFDIR = OUT / 'df'
LEDGER_DIR = OUT / 'ledger'
FIGS = OUT / 'figures'
SANDBOX = OUT / 'sandbox'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK = DRAFTS / 'S5-ALIGN-02_protocol.md'
REVIEW = DRAFTS / 'S5-ALIGN-01_review.md'

PHASES = [f'ph{i}' for i in range(8)]
MAIN = ('B_curv', 'M10_curv', 'M24_curv')
GAUSS = 'M10_curv_gauss'
SHORT = {'M10_curv': 'M10', 'M24_curv': 'M24', 'B_curv': 'B', 'M10_curv_gauss': 'M10+Gauß'}
PLATFORM = {'M10_curv': 'A_curv', 'M24_curv': 'C_curv', 'B_curv': 'B_curv', 'M10_curv_gauss': 'A_curv_gauss'}
FOUR = {'dots_parallel': dict(sd='D_r1_p6x4', ss='D_r1_p10x4', ld='D_r1.5_p6x4', ls='D_r1.5_p10x4'),
        'dots_perp': dict(sd='D_r1_p8x4', ss='D_r1_p8x6', ld='D_r1.5_p8x4', ls='D_r1.5_p8x6'),
        'lines': dict(sd='L_w0.5_s4.45', ss='L_w0.5_s8', ld='L_w1.5_s4.45', ls='L_w1.5_s8')}
FOUR_POSTHOC = {'lines_6.3_8': dict(sd='L_w0.5_s6.3', ss='L_w0.5_s8', ld='L_w1.5_s6.3', ls='L_w1.5_s8')}
D_WIDTH = {s: (f'L_w0.5_s{s}', f'L_w1.5_s{s}') for s in ('4.45', '6.3', '8')}
METRICS = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'sup_surf_rmse')
SEED, N_RES = 2026101204, 10000
DOT9 = ('D_r1_p6x4', 'D_r1_p10x4', 'D_r1_p8x4', 'D_r1_p8x6', 'D_r1.5_p6x4', 'D_r1.5_p10x4', 'D_r1.5_p8x4', 'D_r1.5_p8x6', 'D_r0.75_p8x4')
LINE6 = ('L_w0.5_s4.45', 'L_w0.5_s6.3', 'L_w0.5_s8', 'L_w1.5_s4.45', 'L_w1.5_s6.3', 'L_w1.5_s8')
SIGMAS6 = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
CAPS = dict(new_imaging=0, decode_attempts=1296, df_images=648)
TOL = dict(C=1e-9, err_mm=1e-6)
READ: dict = {}


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}'); self.code = code; self.msg = msg


EXIT = dict(technical=2, budget=3, check=4, scope=5, version=6, guard=7, freeze=8, missing=9, gate=10)


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


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in q.parents and q != OUT.resolve():
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
    raise Stop('technical', f'atomic replace failed for {q}')


def save_json(p, obj) -> Path:
    return _atomic(p, lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'))


def save_csv(df: pd.DataFrame, p) -> Path:
    return _atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


def save_npz(p, **arrays) -> str:
    def w(t):
        with open(t, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
    _atomic(p, w)
    return sha(p)


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def load_npz(p) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='S5-ALIGN-02', task_book=rel(TASK),  acceptance='reference (independent)', created=now())


def run_put(section: str, obj) -> dict:
    r = run_load()
    if section in r.get('frozen', []):
        raise Stop('freeze', f'run.json section {section} was fixed before the computation; never rewritten')
    r[section] = obj; r['updated'] = now(); save_json(RUN, r)
    return r


def run_fix(section: str, obj) -> dict:

    r = run_load()
    if section in r:
        if digest(r[section]) != digest(obj):
            raise Stop('freeze', f'{section} already fixed with a different content')
        return r
    r[section] = obj; r.setdefault('frozen', []).append(section); r.setdefault('fixed_at', {})[section] = now(); r['updated'] = now(); save_json(RUN, r)
    return r


def flush(section: str) -> None:
    r = run_load(); ir = r.setdefault('sources_read', {})
    for k, v in READ.items():
        if k in ir and ir[k] != v:
            raise Stop('input', f'{k} changed during the work ({section})')
        ir.setdefault(k, v)
    r['updated'] = now(); save_json(RUN, r)


def part(name: str, df: pd.DataFrame) -> Path:
    return save_csv(df, PARTS / f'{name}.csv')


def read_part(name: str) -> pd.DataFrame:
    return pd.read_csv(PARTS / f'{name}.csv', low_memory=False)


def ledger(kind: str, key: str, status: str, **kw) -> None:
    p = guard(LEDGER_DIR / f'ledger_{os.getpid()}.jsonl')
    with open(p, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(dict(kind=kind, key=key, status=status, t=now(), **kw)), ensure_ascii=False) + '\n')


def ledger_rows() -> list:
    out = []
    if LEDGER_DIR.exists():
        for p in sorted(LEDGER_DIR.glob('ledger_*.jsonl')):
            out += [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    return out


def decode_attempts() -> int:
    return sum(1 for r in ledger_rows() if r['kind'] == 'decode_attempt' and r['status'] == 'start')


def plan_s5() -> pd.DataFrame:
    P = pd.read_csv(reg(S345 / 'plan_s345.csv'), low_memory=False)
    return P[P.set_formal.isin(['S5-main', 'S5-gauss'])].copy()


def cases() -> pd.DataFrame:
    A = pd.read_csv(reg(S345 / 'results' / 'S5-main_cases.csv'), low_memory=False).assign(set_name='S5-main')
    B = pd.read_csv(reg(S345 / 'results' / 'S5-gauss_cases.csv'), low_memory=False).assign(set_name='S5-gauss')
    C = pd.concat([A, B], ignore_index=True)
    S = pd.read_csv(reg(S345 / 'results' / 'S5_cases_supplement.csv'), low_memory=False)
    C = C.merge(S[['decode_key', 'geometry_id', 'phase_offset_par_mm', 'phase_offset_perp_mm', 'peak_actual', 'lit_px', 'integral_actual', 'sup_surf_rmse', 'n_detected']],
                on='decode_key', how='left', validate='one_to_one')
    C['sup_surf_rmse'] = pd.to_numeric(C.sup_surf_rmse, errors='coerce')
    if C.decode_key.duplicated().any() or len(C) != 1536:
        raise Stop('scope', f'S5 cases: {len(C)} rows, duplicated decode keys {int(C.decode_key.duplicated().sum())}')
    return C


def ratio(a, b):
    return (float(a) / float(b)) if b else None
