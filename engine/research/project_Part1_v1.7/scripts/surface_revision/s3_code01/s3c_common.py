# Define encoding-study configurations and shared resource limits.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RDY = ROOT / 'scripts' / 'surface_revision' / 's345_ready'
if str(RDY) not in sys.path:
    sys.path.insert(0, str(RDY))
import rdy_common as RK

OUT = ROOT / 'outputs' / 'surface_revision' / 's3_code01'
SANDBOX = OUT / 'sandbox'
for _k, _sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
    os.environ[_k] = str(SANDBOX / _sub)
os.environ['LINE01RB_RENDER_CAP'] = '0'
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

S345 = ROOT / 'outputs' / 'surface_revision' / 's345_ready'
S345_RUN = S345 / 'run'
S3_CASES = S345 / 'results' / 'S3-reduced_cases.csv'
S345_PLAN = S345 / 'plan_s345.csv'
S12 = ROOT / 'outputs' / 'surface_revision' / 's12_resource01'
S1OUT = ROOT / 'outputs' / 'surface_revision' / 's1_real01'
SNAPSHOT = ROOT / 'outputs' / 'line01r' / 'night01r2' / 'config' / 'snapshot_line_encoding_dev_v1.json'
R3CFG = ROOT / 'outputs' / 'line01r' / 'night01r3' / 'config' / 'r3_config.json'
TASK_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-01_protocol.md'
TASK_ID = 'S3-CODE-01'

PROTOCOL = OUT / 'protocol.json'
RESULTS, FIG, CHECKS, CACHE = OUT / 'results', OUT / 'figures', OUT / 'checks', OUT / 'cache'
LEDGER = OUT / 'run' / 'ledger.jsonl'
FORMAL = OUT / 'formal'


PLATFORM_VERSION = RK.PLATFORM_VERSION
WINDOW_MM, D_SUPPORT, TAU, GROSS, PSF_PX, Q_BUDGET = RK.WINDOW_MM, RK.D_SUPPORT, RK.TAU, RK.GROSS, RK.PSF_PX, RK.Q
MAIN = ('M10_curv', 'M24_curv', 'B_curv')
SHORT = {'M10_curv': 'M10', 'M24_curv': 'M24', 'B_curv': 'B'}
SIGMAS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)  # Registered noise standard deviations [8-bit gray levels] for this experiment.
CAM = dict(f_mm=25.0, pixel_mm=0.03, wd_mm=150.0, baseline_mm=50.0, size_px=512)
NOMINAL_MM_PER_PX = CAM['pixel_mm'] * CAM['wd_mm'] / CAM['f_mm']

EXIT = dict(ok=0, technical=2, cap=3, check=4, platform=12, permit=14, freeze=15, lock=18)


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}')
        self.code, self.msg = code, msg


def now_local() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


clean = RK.clean
digest = RK.digest
rel = RK.rel


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


def save_text(p, text: str) -> Path:
    return _atomic(Path(p), lambda t: t.write_text(text, encoding='utf-8'))


def save_csv(df, p) -> Path:
    return _atomic(Path(p), lambda t: df.to_csv(t, index=False, encoding='utf-8'))


def save_npz(p, **arrays) -> str:
    def w(t):
        with open(t, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
    q = _atomic(Path(p), w)
    return sha_file(q)


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def load_npz(p) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def ledger(kind: str, key: str, status: str, **info) -> None:

    if kind == 'render':
        raise Stop('cap', 'M3 imaging is not authorised in S3-CODE-01 (render calls = 0)')
    p = guard(LEDGER); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(dict(t=now_local(), kind=kind, key=key, status=status, **info)), ensure_ascii=False) + '\n')


def ledger_counts() -> dict:
    if not LEDGER.exists():
        return {}
    out: dict = {}
    for line in LEDGER.read_text(encoding='utf-8').splitlines():
        r = json.loads(line); k = (r['kind'], r['status'], r.get('stage', ''))
        out['|'.join(k)] = out.get('|'.join(k), 0) + 1
    return out


def register_sources(name: str, paths) -> dict:

    rec = {rel(p): sha_file(p) for p in sorted(set(Path(x) for x in paths))}
    save_json(CHECKS / f'sources_{name}.json', dict(name=name, n=len(rec), t_local=now_local(), files=rec))
    return rec


def verify_sources(name: str) -> dict:
    r = load_json(CHECKS / f'sources_{name}.json'); bad = [k for k, h in r['files'].items() if not (ROOT / k).exists() or sha_file(ROOT / k) != h]
    return dict(name=name, n=r['n'], changed=bad, ok=not bad)
