# Define layout blocks, phase offsets, noise seeds and resource limits.

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
for _p in (SPL, SPL / 'spat01r', SPL / 'spat02', SPL / 'spat02pr2', HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(HERE) in sys.path:
    sys.path.remove(str(HERE)); sys.path.insert(0, str(HERE))
import sp2_common as S2
import sp5_common as C5

OUT = ROOT / 'outputs' / 'spatial_layout' / 'spat03'
SANDBOX = OUT / 'sandbox'
RUN_DIR = OUT / 'formal_run1'
REPLAY_DIR = OUT / 'replay_v5'
DOCS = S2.DOCS
MASTER = ROOT.parents[1]
TASK_FILE = MASTER / 'protocols' / 'SPAT-03_protocol.md'

PILOT_RUN = ROOT / 'outputs' / 'spatial_layout' / 'spat02' / 'pilot_v5_run1'
FREEZE_V5 = ROOT / 'outputs' / 'spatial_layout' / 'spat02pr2' / 'freeze_pilot_v5.json'
FREEZE_V5_SHA = 'b6fd3826d1b725f523a8e577a8a0d12db416aa1ac757bd086b9501daa312a44c'
TASK_ID = 'SPAT-03'
PACKAGE_VERSION = 'spat03_formal_v1'
FREEZE_TAG = 'spat03_v1'


GEN_LEDGER = OUT / 'generation_ledger.jsonl'
GEN_SUMMARY = OUT / 'generation_summary.json'
LAYOUTS_JSON = OUT / 'formal_layouts.json'
PRECHECK = OUT / 'phase_precheck_H.jsonl'
IMAGES_CSV = OUT / 'manifest_images.csv'
MAIN_CSV = OUT / 'manifest_main_canonical.csv'
VIEWS_CSV = OUT / 'manifest_strategy_views.csv'
REF_CSV = OUT / 'manifest_equivalence_reference.csv'
GENFAIL_CSV = OUT / 'manifest_generation_failures.csv'
COUNTS_JSON = OUT / 'manifest_counts.json'


N_BLOCKS = 24
BLOCKS = tuple(range(N_BLOCKS))
N_CAND = 1000
J_WEAK, J_STRONG = 0.10, 0.20
CORE = ('R5x4', 'L4.45'); COST = ('R8x4', 'L6.3')
STRATEGIES = ('S1_free', 'S2_gmin_0.70', 'S3_hmax_1.15', 'S4_joint_0.70_1.15')
PHASES_ROLE = 'holdout'
PREFIX_BUDGETS = (100, 250, 500, 1000)
SEED_BASE_S3 = 3_000_000_000
NOISE_SEED_BASE_S3 = 3_100_000_000


def block_id(b: int) -> str:
    return f'B{int(b):03d}'


def seed_s3(baseline_index: int, block: int, cand: int) -> int:
    if not (0 <= block < 100 and 0 <= cand < 1000):
        raise ValueError('block / candidate outside the injective range')
    return int(SEED_BASE_S3 + 100000 * int(baseline_index) + 1000 * int(block) + int(cand))


def holdout_phases(family: str) -> list[dict]:
    return [p for p in S2.registered_phases(family) if p['role'] == PHASES_ROLE]


def noise_seed_s3(family: str, surface: str, phase_id: str, sigma: float) -> int:
    phases = [p['phase_id'] for p in holdout_phases(family)]
    idx = ((list(S2.SURFACES).index(surface) * 10 + phases.index(phase_id)) * len(S2.SIGMAS) + list(S2.SIGMAS).index(float(sigma)))
    return int(NOISE_SEED_BASE_S3 + (0 if family == 'dots' else 1000) + idx)


def budget_for(n_images: int, n_main: int, n_ref: int = 64) -> dict:

    ri, rm = int(math.ceil(0.10 * n_images)), int(math.ceil(0.10 * n_main))
    ri = min(ri, 5140 - n_images); rm = min(rm, 5140 - n_main)
    return dict(imaging_planned=int(n_images), imaging_retry=int(ri), decode_main_planned=int(n_main), decode_main_retry=int(rm), decode_ref_planned=int(n_ref), decode_ref_retry=7)


STATUS_TERMINAL = C5.STATUS_TERMINAL
STATUS_SCIENTIFIC = C5.STATUS_SCIENTIFIC
STAGES = C5.STAGES
MAX_ATTEMPTS_PER_KEY_STAGE = C5.MAX_ATTEMPTS_PER_KEY_STAGE
RETRYABLE = C5.RETRYABLE
EXIT = dict(C5.EXIT, time_limit=14)
IMG_SHAPE = C5.IMG_SHAPE
PEAK_ATOL = C5.PEAK_ATOL
VALIDITY = C5.VALIDITY
StopRun = C5.StopRun
now_local, sha_file, sha_bytes, digest, array_digest, clean = C5.now_local, C5.sha_file, C5.sha_bytes, C5.digest, C5.array_digest, C5.clean
product_components, product_digest_of, rel = C5.product_components, C5.product_digest_of, C5.rel


def atomic_write_json(p: Path, obj) -> None:

    import time as _t
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8')
    for i in range(50):
        try:
            os.replace(tmp, p); return
        except PermissionError:
            if i == 49:
                raise
            _t.sleep(0.1)


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    return S2.save_json(p, obj)


def load_json(p):
    return S2.load_json(p)


def log(msg: str, which: str = 'main', out: Path | None = None) -> None:
    out = Path(out or OUT); out.mkdir(parents=True, exist_ok=True)
    line = f'[{now_local()}] {msg}'
    with open(out / f'log_{which}.txt', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


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
