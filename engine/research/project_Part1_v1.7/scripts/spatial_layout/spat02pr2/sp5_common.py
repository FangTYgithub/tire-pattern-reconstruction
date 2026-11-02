# Provide result identities, atomic writes and execution ledgers for layout cases.

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPL = HERE.parent
ROOT = SPL.parents[1]
for _p in (SPL, SPL / 'spat01r', SPL / 'spat02', HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(HERE) in sys.path:
    sys.path.remove(str(HERE)); sys.path.insert(0, str(HERE))
import sp2_common as S2

OUT = ROOT / 'outputs' / 'spatial_layout' / 'spat02pr2'
SANDBOX = OUT / 'sandbox'
OUT_SCI = ROOT / 'outputs' / 'spatial_layout' / 'spat02'
OUT_V4 = ROOT / 'outputs' / 'spatial_layout' / 'spat02pr'
IN02P = ROOT / 'outputs' / 'spatial_layout' / 'spat02p'
IN1R = S2.OUT
DOCS = S2.DOCS
MASTER = ROOT.parents[1]
TASK_FILE = MASTER / 'protocols' / 'SPAT-02PR2_protocol.md'

TASK_ID = 'SPAT-02PR2'
PACKAGE_VERSION = 'spat02_pilot_v5'
FREEZE_TAG = 'pilot_v5'

MAIN_CSV = IN02P / 'manifest_main_canonical_v3.csv'
VIEWS_CSV = IN02P / 'manifest_strategy_views_v3.csv'
REF_CSV = IN02P / 'manifest_equivalence_reference_v3.csv'
IMAGES_CSV = IN1R / 'pilot_image_tasks_v2.csv'
LAYOUTS_JSON = IN1R / 'pilot_layouts_v2.json'

STATUS_TERMINAL = ('ok', 'no_output', 'technical_error', 'missing_file', 'untested')
STATUS_SCIENTIFIC = ('ok', 'no_output')
STAGES = ('imaging', 'decode_main', 'decode_ref')
BUDGET = dict(imaging_planned=528, imaging_retry=53, decode_main_planned=528, decode_main_retry=53, decode_ref_planned=64, decode_ref_retry=7)
MAX_ATTEMPTS_PER_KEY_STAGE = 2
RETRYABLE = ('transient_io', 'product_missing', 'product_corrupt')
EXIT = dict(ok=0, freeze_mismatch=2, budget_exceeded=3, duplicate_key=4, equivalence_failed=5, backend_error=6, manifest_error=7, invalid_record=8,
            version_conflict=9, stale_downstream=10, context_mix=11, s0_dependency_unverifiable=12, incomplete=13, interrupted=130)
IMG_SHAPE = (512, 512)
PEAK_ATOL = 1e-12

VALIDITY = ('valid', 'missing', 'corrupt', 'unavailable', 'not_reverifiable', 'stale', 'conflict')

M3_RUNTIME_KEYS = ('time', 'timestamp', 'seconds', 'elapsed', 'started', 'finished', 'started_local', 'finished_local', 'wall_time_s')
M3_DOWNSTREAM_KEYS = ('n_valid_pixels',)


class StopRun(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg); self.code = code


def now_local() -> str:
    return S2.now_local()


def sha_file(p) -> str:
    return S2.sha_file(p)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def digest(o) -> str:
    return S2.digest(o)


def array_digest(a) -> str:

    import numpy as np
    a = np.ascontiguousarray(a)
    return sha_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())[:32]


def clean(o):
    return S2.clean(o)


def canon_json(o) -> str:

    return json.dumps(clean(o), sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def stored_m3(m3_info) -> dict:

    m = json.loads(canon_json(m3_info if isinstance(m3_info, dict) else {}))
    return {k: v for k, v in m.items() if k not in M3_RUNTIME_KEYS}


def product_components(image, pattern_bin, peak, m3_info) -> dict:

    import numpy as np
    out = {}
    for name, a in (('image', image), ('pattern_bin', pattern_bin)):
        a = np.ascontiguousarray(a)
        out[name] = sha_bytes(f'{name}|{a.dtype.str}|{tuple(a.shape)}|'.encode() + a.tobytes())[:32]
    p = float(peak)
    out['peak'] = sha_bytes(b'peak|' + (p.hex() if math.isfinite(p) else 'nonfinite').encode())[:32]
    out['m3_info'] = sha_bytes(b'm3_info|' + canon_json(stored_m3(m3_info)).encode())[:32]
    return out


def product_digest_of(components: dict) -> str:
    return sha_bytes(canon_json({k: components[k] for k in ('image', 'pattern_bin', 'peak', 'm3_info')}).encode())[:32]


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    return S2.save_json(p, obj)


def load_json(p):
    return S2.load_json(p)


def atomic_write_json(p: Path, obj) -> None:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, p)


def log(msg: str, which: str = 'main', out: Path | None = None) -> None:
    out = Path(out or OUT); out.mkdir(parents=True, exist_ok=True)
    line = f'[{now_local()}] {msg}'
    with open(out / f'log_{which}.txt', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


def rel(p) -> str:
    return S2.rel(p)


class Ledger:


    def __init__(self, path: Path, kind: str):
        self.path = Path(path); self.kind = kind
        if self.path.exists():
            self.L = load_json(self.path)
            if self.L.get('kind') != kind:
                raise StopRun(EXIT['context_mix'], f"ledger kind {self.L.get('kind')} != {kind} in {self.path.parent}")
        else:
            self.L = dict(kind=kind, package_version=PACKAGE_VERSION, created_local=now_local(), counts={s: 0 for s in STAGES}, per_key={}, events=[])

    def attempt(self, stage: str, key: str, why: str) -> int:
        k = f'{stage}|{key}'; n = self.L['per_key'].get(k, 0) + 1
        self.L['per_key'][k] = n; self.L['counts'][stage] = self.L['counts'].get(stage, 0) + 1
        self.L['events'].append(dict(time=now_local(), stage=stage, key=key, phase='attempt', n=n, why=why)); self._save(); return n

    def note(self, stage: str, key: str, **info):
        self.L['events'].append(dict(time=now_local(), stage=stage, key=key, phase='note', **info)); self._save()

    def attempts(self, stage: str, key: str) -> int:
        return int(self.L['per_key'].get(f'{stage}|{key}', 0))

    def count(self, stage: str) -> int:
        return int(self.L['counts'].get(stage, 0))

    def _save(self):
        self.L['updated_local'] = now_local(); atomic_write_json(self.path, self.L)
