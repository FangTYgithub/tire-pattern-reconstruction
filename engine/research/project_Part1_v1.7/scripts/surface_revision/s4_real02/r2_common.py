# Define layout cases, boundary rules and evaluation thresholds.

from __future__ import annotations

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
sys.path.insert(0, str(ROOT / 'scripts' / 'surface_revision' / 's345_ready'))
import rdy_common as RK

OUT = ROOT / 'outputs' / 'surface_revision' / 's4_real02'
for _k, _sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
    os.environ[_k] = str(OUT / 'sandbox' / _sub)
os.environ['LINE01RB_RENDER_CAP'] = '0'
for _p in (str(HERE), str(ROOT / 'scripts' / 'surface_revision' / 'pix_bound01'), str(ROOT / 'scripts' / 'surface_revision' / 's12_resource01')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SR = ROOT / 'outputs' / 'surface_revision'
PIX, S4B, S345 = SR / 'pix_bound01', SR / 's4_align02', SR / 's345_ready'
IMG, RAW, EVL, EQV, FIGS, RUNDIR = OUT / 'images', OUT / 'raw', OUT / 'eval', OUT / 'equiv', OUT / 'figures', OUT / 'run'
RUN = OUT / 'run.json'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK = DRAFTS / 'S4-REAL-02_protocol.md'
REVIEW = DRAFTS / 'PIX-BOUND-01_S4-ALIGN-02_review.md'
START = '2026-10-16T20:05:00+02:00'

BOUNDARY = 'symmetric_footprint'
Q = 11695.0  # Sum of normalized projector pixel intensities; peak is capped separately.
CAP_RENDER = 2560
N_FORMAL, N_VIEWS = 2520, 4752
SIGMAS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)  # Registered noise standard deviations [8-bit gray levels] for this experiment.
PHASES = ('H0', 'H1', 'H2', 'H3')
BLOCKS = (0, 3, 6, 9, 12, 15, 18, 21)
STRATS = ('S0', 'S1', 'S2', 'S3', 'S4')
BASES = (('R8x4', 'point', 'dots'), ('L6.3', 'line', 'lines'))
SURF = ('B_curv', 'M10_curv', 'M24_curv')
SHORT = {'B_curv': 'B', 'M10_curv': 'M10', 'M24_curv': 'M24'}
ANCHORS = {'outputs/surface_revision/s4_align02/s4_manifest_candidates.csv': '57fde588fda35483b39ed0c0ddbc6c71207a779e94af1fde781d6cfe183ffc07',
           'outputs/surface_revision/pix_bound01/domains/omega1.npz': 'cc6dd47dbab7186b87d9fd7f73b01f3007035a2e626b05098da9faa2655c7358',
           'outputs/surface_revision/pix_bound01/domains/omega_star.npz': '0aee8912211cc5979b4aa44885854268019e280ceeb27b9fa1e93cc5ef1d839e'}
D_SUPPORT, TAU = 6.0, 0.15  # Support scale and height tolerance [mm].


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}'); self.code = code; self.msg = msg


def now_local() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def digest(obj, n: int = 24) -> str:
    return sha_bytes(json.dumps(clean(obj), sort_keys=True, default=str).encode())[:n]


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in q.parents and q != OUT.resolve():
        raise Stop('guard', f'write outside {rel(OUT)} refused: {q}')
    if 'sandbox' in q.parts:
        raise Stop('guard', 'the sandbox path is never written')
    return q


def _atomic(p: Path, write) -> Path:
    p = guard(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + f'.tmp{os.getpid()}')
    write(tmp)
    for _ in range(100):
        try:
            os.replace(tmp, p); return p
        except PermissionError:
            time.sleep(0.05)
    raise Stop('technical', f'atomic replace failed for {p}')


def save_json(p, obj) -> Path:
    return _atomic(Path(p), lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'))


def save_csv(df: pd.DataFrame, p) -> Path:
    return _atomic(Path(p), lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


def save_npz(p, **arrays) -> str:
    p = Path(p)
    _atomic(p, lambda t: np.savez_compressed(open(t, 'wb'), **arrays))
    return sha_file(p)


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def load_npz(p) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def complete(p: Path, identity: str) -> bool:

    p = Path(p)
    if not p.exists():
        return False
    try:
        r = load_json(p)
    except Exception:
        return False
    z = p.with_suffix('.npz')
    return r.get('complete') is True and r.get('identity') == identity and z.exists() and sha_file(z) == r.get('npz_sha256')


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='S4-REAL-02', created=now_local(), note='Numerical consistency checks.')


def run_update(section: str, obj, freeze: bool = False) -> dict:
    r = run_load(); fz = r.get('frozen', {})
    if section in fz:
        raise Stop('freeze', f'run.json section {section} is frozen; never rewritten')
    if freeze:
        r.setdefault('frozen', {})[section] = dict(digest=digest(obj, 32), t_local=now_local())
    r[section] = obj; r['updated'] = now_local(); save_json(RUN, r)
    return r


def run_verify_frozen() -> dict:
    r = run_load(); bad = [s for s, v in r.get('frozen', {}).items() if digest(r.get(s), 32) != v['digest']]
    if bad:
        raise Stop('freeze', f'frozen run.json sections changed: {bad}')
    return dict(ok=True, sections=sorted(r.get('frozen', {})))


class Budget:


    def __init__(self):
        self.f = RUNDIR / 'budget.json'; self.a = RUNDIR / 'attempts.jsonl'

    def state(self) -> dict:
        return load_json(self.f) if self.f.exists() else dict(cap=CAP_RENDER, used=0, by_purpose={})

    def reserve(self, key: str, purpose: str) -> int:
        b = self.state()
        if int(b['cap']) != CAP_RENDER:
            raise Stop('budget', f"budget cap {b['cap']} != {CAP_RENDER}")
        if b['used'] >= b['cap']:
            raise Stop('budget', f"render cap exhausted ({b['used']} / {b['cap']})")
        b['used'] += 1; b['by_purpose'][purpose] = b['by_purpose'].get(purpose, 0) + 1
        save_json(self.f, b); self.log(dict(n=b['used'], key=key, purpose=purpose, event='reserve'))
        return b['used']

    def log(self, rec: dict) -> None:
        guard(self.a); self.a.parent.mkdir(parents=True, exist_ok=True)
        with open(self.a, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(clean(dict(rec, t=now_local())), ensure_ascii=False) + '\n')


def code_fingerprint(paths) -> dict:
    out = {}
    for p in paths:
        q = ROOT / p
        out[p] = sha_file(q) if q.exists() else 'MISSING'
    return out
