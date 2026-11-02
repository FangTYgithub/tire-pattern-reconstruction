# Define common surfaces, resource limits and evaluation settings for pattern studies.

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
OUT = ROOT / 'outputs' / 'surface_revision' / 's345_ready'
SANDBOX = OUT / 'sandbox'
RUN = OUT / 'run'
LEDGER = RUN / 'ledger.jsonl'
LOCK = RUN / 'run.lock'
RESULTS = OUT / 'results'
FIG = OUT / 'figures'
PLAN_CSV = OUT / 'plan_s345.csv'
CONFIG = OUT / 'config_s345_ready.json'
FREEZE = OUT / 'freeze_s345_ready.json'
CHECKS = OUT / 'checks.json'
README = OUT / 'README_study.md'

SRC_LIST = ROOT / 'outputs' / 'surface_revision' / 's345_prepare' / 'candidates_s345.csv'
SRC_LIST_ADP = ROOT / 'outputs' / 'surface_revision' / 'sections_adapter' / 'candidates_s1s5.csv'
PREP_CODE = ROOT / 'scripts' / 'surface_revision' / 's345_prepare'
S1OUT = ROOT / 'outputs' / 'surface_revision' / 's1_real01'
S2_FREEZE = ROOT / 'outputs' / 'surface_revision' / 's2_prepare' / 'run' / 'freeze_formal_batch.json'
S2_PROGRESS = ROOT / 'outputs' / 'surface_revision' / 's2_prepare' / 'progress.json'
S2_LEDGER = ROOT / 'outputs' / 'surface_revision' / 's2_prepare' / 'run' / 'ledger.jsonl'
TASK_FILE = ROOT.parents[1] / 'protocols' / 'S345-PREP-03_protocol.md'
TASK_ID = 'S345-PREP-03'


PLATFORM_VERSION = 'SURF-02 v1.1 (2026-10-12)'
WINDOW_MM = (-16.0, 4.0)  # Lower and upper reconstruction depths [mm].
MARGIN_MM = 1e-6
C_REF_MM = 16.645465475819755
D_SUPPORT, TAU, GROSS = 6.0, 0.15, 1.0  # Support scale, height tolerance and gross-error threshold [mm].
PSF_PX = 1.0  # Fixed Gaussian PSF standard deviation [px] for this experiment.
Q = 11695.0  # Sum of normalized projector pixel intensities; peak is capped separately.
N_GLOBAL_NODES, N_LOCAL_NODES = 30947, 22605
GLOBAL_DOMAIN_HASH = 'e240df004a66e58d8dfb12162a8a18ba6452c6a58c7664a860dd352b2b3b0ea3'
SURF_MAP = {'M10_curv': 'A_curv', 'M24_curv': 'C_curv', 'B_curv': 'B_curv', 'M10_curv_gauss': 'A_curv_gauss'}
MAIN = ('M10_curv', 'M24_curv', 'B_curv')
GAUSS = 'M10_curv_gauss'
SHORT = {'M10_curv': 'M10', 'M24_curv': 'M24', 'B_curv': 'B', 'M10_curv_gauss': 'M10+Gauß'}

EXIT = dict(ok=0, technical=2, cap=3, platform=12, permit=14, dependency=15, unexpected=16, check=17, lock=18)

ENV = {'LINE01RB_OUT_DIR': SANDBOX / 'line_out', 'LINE01RB_CASES_DIR': SANDBOX / 'line_cases', 'LINE01RB_LEDGER_DIR': SANDBOX / 'line_ledger', 'LINE01RB_RENDER_CAP': '0'}
_PATHS = ['src', 'scripts/spotnoise', 'scripts/spotnoise_round3', 'scripts/spotnoise_round5', 'scripts/spotnoise_suppA', 'scripts/spacing_layout', 'scripts/pm013', 'scripts/pm014',
          'scripts/pm015', 'scripts/line01', 'scripts/line01r', 'scripts/encoding_prep', 'scripts/spatial_layout', 'scripts/spatial_layout/spat01r', 'scripts/spatial_layout/spat02pr2',
          'scripts/spatial_layout/spat03', 'scripts/surface_revision/surf01', 'scripts/surface_revision/surf02', 'scripts/surface_revision/sections_adapter', '']
_DONE = False


class Stop(Exception):


    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}')
        self.code = code
        self.msg = msg


def setup() -> dict:

    global _DONE
    for k, v in ENV.items():
        os.environ[k] = str(v)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    for p in _PATHS:
        q = str(ROOT / p)
        if q not in sys.path:
            sys.path.append(q)
    _DONE = True
    return {k: os.environ.get(k) for k in list(ENV) + ['LINE01RB_PATHS_JSON', 'LINE01RB_FREEZE_JSON']}


setup()


def now_local() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b: bytes, n: int | None = None) -> str:
    h = hashlib.sha256(b).hexdigest()
    return h[:n] if n else h


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
        return str(o)
    return o


def digest(obj, n: int | None = None) -> str:
    h = hashlib.sha256(json.dumps(clean(obj), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return h[:n] if n else h


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in (q, *q.parents):
        raise Stop('technical', f'write outside {rel(OUT)} refused: {q}')
    return q


def _replace(tmp: Path, p: Path) -> None:
    for _ in range(50):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(0.1)
    raise Stop('technical', f'atomic replace failed for {p}')


def save_json(p, obj) -> Path:
    p = guard(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8')
    _replace(tmp, p)
    return p


def save_text(p, text: str) -> Path:
    p = guard(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(text, encoding='utf-8')
    _replace(tmp, p)
    return p


def save_npz(p, **arrays) -> str:
    p = guard(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    _replace(tmp, p)
    return sha_file(p)


def save_csv(df, p) -> Path:
    p = guard(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8')
    _replace(tmp, p)
    return p


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def complete(p, identity: str | None = None) -> bool:

    p = Path(p)
    if not p.exists():
        return False
    try:
        r = load_json(p)
    except Exception:
        return False
    z = p.with_suffix('.npz')
    ok = r.get('complete') is True and z.exists() and sha_file(z) == r.get('npz_sha256')
    return bool(ok and (identity is None or r.get('identity') == identity))


def norm_seed(x):

    if x is None:
        return None
    s = str(x).strip()
    if s in ('', 'nan', 'None', 'NaN'):
        return None
    return int(round(float(s)))
