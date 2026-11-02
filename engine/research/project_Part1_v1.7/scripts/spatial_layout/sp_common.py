# Provide geometry units, seeds and input checks for spatial-layout experiments.

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

for _p in (ROOT / 'scripts' / 'encoding_prep', ROOT / 'scripts' / 'pm013', ROOT / 'scripts' / 'pm014', ROOT / 'scripts' / 'pm015', ROOT / 'scripts' / 'spotnoise_suppA',
           ROOT / 'scripts' / 'spotnoise_round3', ROOT / 'scripts' / 'spotnoise_round4', ROOT / 'scripts' / 'spotnoise_round5', ROOT / 'scripts' / 'spotnoise',
           ROOT / 'scripts' / 'spacing_layout', ROOT / 'scripts' / 'line01', ROOT / 'scripts' / 'line01r', ROOT / 'src', ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import l01rb_common as B

OUT = ROOT / 'outputs' / 'spatial_layout' / 'spat01'
FIG = OUT / 'figures'
DOCS = ROOT / 'docs' / 'spatial_layout'
TASK_FILE = ROOT.parents[1] / 'protocols' / 'SPAT-01_protocol.md'
TASK_ID = 'SPAT-01'

IMG = 512
Q_BUDGET = float(B.Q_BUDGET)
PEAK_CAP = float(B.PEAK_CAP)
R_MM = 1.0
W_MM = 1.0
PHI_DEG = 0.0
DOT_BASELINES = {'R5x4': (5.0, 4.0), 'R8x4': (8.0, 4.0)}
LINE_BASELINES = {'L4.45': 4.45, 'L6.3': 6.3}
LINE_REFERENCE_ONLY = {'L8.0': 8.0}
J_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.30)
N_BLOCKS, N_PER_BLOCK = 10, 100
GMIN_FRACS = (0.70, 0.85)
HMAX_FRACS = (1.05, 1.15)
SEED_BASE = 10800000
REF_PHASE_DOT = (0.5, 0.5)
REF_PHASE_LINE = 0.0
DOT_WINDOW_MM = (0.0, 20.0)
LINE_WINDOW_MM = (float(B.Z0_MM), float(B.Z1_MM))
STRATEGIES = ('S1_free',) + tuple(f'S2_gmin_{g:.2f}' for g in GMIN_FRACS) + tuple(f'S3_hmax_{h:.2f}' for h in HMAX_FRACS) + tuple(f'S4_joint_{g:.2f}_{h:.2f}' for g in GMIN_FRACS for h in HMAX_FRACS)

_MMPP = None


def mmpp() -> float:
    global _MMPP
    if _MMPP is None:
        from m2_pattern import _compute_fov_mm, FOCAL_LENGTH_MM, PIXEL_PITCH_MM, WORKING_DIST_MM
        _MMPP = float(_compute_fov_mm(IMG, FOCAL_LENGTH_MM, PIXEL_PITCH_MM, WORKING_DIST_MM)[2])
    return _MMPP


def now_local() -> str:
    return B.now_local()


def sha_file(p) -> str:
    return B.sha_file(Path(p))


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_array(a: np.ndarray) -> str:
    return B.sha_array(np.asarray(a))


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(B.clean(obj), sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    return B.save_json(Path(p), obj)


def load_json(p):
    return B.load_json(Path(p))


def clean(o):
    return B.clean(o)


def ensure_dirs():
    for d in (OUT, FIG, DOCS):
        d.mkdir(parents=True, exist_ok=True)


def log(msg: str, which: str = 'main') -> None:
    ensure_dirs()
    line = f'[{now_local()}] {msg}'
    with open(OUT / f'log_{which}.txt', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


def status(**kw) -> dict:
    p = OUT / 'status.json'
    st = load_json(p) if p.exists() else {}
    st.update(kw); st['updated_local'] = now_local(); save_json(p, st)
    return st


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace('\\', '/')
    except ValueError:
        return str(p)


INPUT_FILES = [
    'src/m1_tire_surface.py', 'src/m2_pattern.py', 'src/m3_virtual_camera.py', 'src/m4_reconstruction.py', 'src/m4_lines.py', 'src/m5_evaluation.py',
    'src/surface_eval_r3.py', 'src/pattern_predictor.py',
    'scripts/line01/l01_common.py', 'scripts/line01/l01_truth.py', 'scripts/line01r/l01r_common.py', 'scripts/line01r/l01rb_common.py', 'scripts/line01r/l01rb_run.py',
    'scripts/line01r/lc_pipeline.py', 'scripts/line01r/lc_run.py', 'scripts/line01r/l01rc1_matrix.py',
    'scripts/spacing_layout/sl_lib.py', 'scripts/spacing_layout/sl_common.py', 'scripts/spotnoise/sn_lib.py', 'scripts/spotnoise_round5/r5_detect.py',
    'scripts/spotnoise_round4/r4_wp2.py', 'scripts/spotnoise/sn_posthoc.py', 'scripts/spotnoise_suppA/sa_defs.py', 'scripts/pm014/pm014_surface.py', 'scripts/pm015/pm015_lib.py',
    'scripts/encoding_prep/ep_patterns.py',
    'outputs/line01r/c3/config/paths_c3_validation.json', 'outputs/line01r/c3/night/formal_manifest_4928_c3.csv',
    'outputs/line01r/night01r2/dot/dot_common_results_v3r1.csv', 'outputs/line01r/night01r2/dot/summary_v3r1.json', 'outputs/line01r/c3/dot_reuse/dot_reuse_manifest.csv',
    'outputs/spacing_layout/design/budget.json',
]


def register_inputs(extra_files=(), name='inputs_start.json') -> dict:

    files = list(INPUT_FILES) + [rel(p) for p in extra_files]
    rec = {}
    for f in files:
        p = ROOT / f
        rec[f] = dict(sha256=sha_file(p), bytes=int(p.stat().st_size)) if p.exists() else dict(missing=True)
    out = dict(task=TASK_ID, task_file=str(TASK_FILE), task_sha256=sha_file(TASK_FILE) if TASK_FILE.exists() else None, generated_local=now_local(), n_files=len(rec), files=rec)
    save_json(OUT / name, out)
    return out


def check_inputs(start_name='inputs_start.json', end_name='inputs_end_check.json') -> dict:
    start = load_json(OUT / start_name)
    changed = []; missing = []
    for f, r in start['files'].items():
        p = ROOT / f
        if not p.exists():
            missing.append(f); continue
        if 'sha256' in r and sha_file(p) != r['sha256']:
            changed.append(f)
    out = dict(checked_local=now_local(), n_files=len(start['files']), changed=changed, missing=missing, ok=(not changed and not missing))
    save_json(OUT / end_name, out)
    return out


def px(mm: float) -> float:
    return float(mm) / mmpp()


def mm(px_: float) -> float:
    return float(px_) * mmpp()


def seed_for(base_index: int, j_index: int, block: int, i: int) -> int:
    return int(SEED_BASE + 100000 * base_index + 10000 * j_index + 100 * block + i)
