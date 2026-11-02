# Define spatial-layout phases, seeds and input identities.

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SP1 = HERE.parent
ROOT = SP1.parents[1]
for _p in (SP1, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sp_common as S1

OUT = ROOT / 'outputs' / 'spatial_layout' / 'spat01r'
OUT1 = S1.OUT
DOCS = S1.DOCS
TASK_FILE = ROOT.parents[1] / 'protocols' / 'SPAT-01R_protocol.md'
TASK_ID = 'SPAT-01R'
GENERATOR_VERSION = 'geometry_v2_spat01r'
ADAPTER_VERSION = 'sp2_lines_table_v2'

IMG = S1.IMG
Q_BUDGET = S1.Q_BUDGET
R_MM, W_MM = S1.R_MM, S1.W_MM
DOT_BASELINES = dict(S1.DOT_BASELINES)
LINE_BASELINES = dict(S1.LINE_BASELINES)
BASELINES = [dict(index=0, family='dots', name='R5x4', p0_mm=5.0), dict(index=1, family='dots', name='R8x4', p0_mm=8.0),
             dict(index=10, family='lines', name='L4.45', p0_mm=4.45), dict(index=11, family='lines', name='L6.3', p0_mm=6.3)]
CORE = ('R5x4', 'L4.45'); COST_SIDE = ('R8x4', 'L6.3')
J_LEVELS = (0.10, 0.20)
J_MAX = 0.20
N_BLOCKS, N_PER_BLOCK = 10, 100
PILOT_BLOCKS = (0, 1, 2)
GMIN_FRAC = 0.70
HMAX_FRAC = 1.15
STRATEGIES = ('S1_free', 'S2_gmin_0.70', 'S3_hmax_1.15', 'S4_joint_0.70_1.15')
S4_J = (0.20,)
SEED_BASE_V2 = 10850000
NOISE_SEED_BASE_V2 = 10900000
SURFACES = ('baseline', 'curv_x_strong')
SIGMAS = (0.0, 20.0)
REF_PHASE_DOT = S1.REF_PHASE_DOT
REF_PHASE_LINE = S1.REF_PHASE_LINE
LINE_PHASES_PILOT_64 = (-25, -9, 7, 23)
LINE_PHASES_HOLDOUT_64 = (-21, 15, -3, 29)


def _positions_dots() -> dict:
    p = json.loads((OUT1 / 'positions_used_dots.json').read_text(encoding='utf-8'))
    hold = [tuple(float(v) for v in x) for x in p['holdout32']]
    return dict(pilot=hold[16:20], holdout=hold[20:24], source='outputs/spatial_layout/spat01/positions_used_dots.json: holdout32[16:24] (scrambled Sobol seed 20261007, sl_design.offsets)')


def registered_phases(family: str) -> list[dict]:

    out = []
    if family == 'dots':
        pd_ = _positions_dots()
        out.append(dict(phase_id='ref', role='reference', u=float(REF_PHASE_DOT[0]), v=float(REF_PHASE_DOT[1])))
        for i, (u, v) in enumerate(pd_['pilot']):
            out.append(dict(phase_id=f'P{i}', role='pilot', u=u, v=v, source_index=16 + i))
        for i, (u, v) in enumerate(pd_['holdout']):
            out.append(dict(phase_id=f'H{i}', role='holdout', u=u, v=v, source_index=20 + i))
    else:
        out.append(dict(phase_id='ref', role='reference', phase_frac=float(REF_PHASE_LINE), k64=0))
        for i, k in enumerate(LINE_PHASES_PILOT_64):
            out.append(dict(phase_id=f'P{i}', role='pilot', phase_frac=k / 64.0, k64=k))
        for i, k in enumerate(LINE_PHASES_HOLDOUT_64):
            out.append(dict(phase_id=f'H{i}', role='holdout', phase_frac=k / 64.0, k64=k))
    return out


def seed_v2(baseline_index: int, block: int, proposal: int) -> int:
    return int(SEED_BASE_V2 + 100000 * baseline_index + 100 * block + proposal)


def mmpp() -> float:
    return S1.mmpp()


def px(mm: float) -> float:
    return S1.px(mm)


def now_local() -> str:
    return S1.now_local()


def sha_file(p) -> str:
    return S1.sha_file(p)


def sha_array(a) -> str:
    return S1.sha_array(a)


def digest(o) -> str:
    return S1.digest(o)


def clean(o):
    return S1.clean(o)


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    return S1.save_json(p, obj)


def load_json(p):
    return S1.load_json(p)


def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'rasters').mkdir(exist_ok=True)


def log(msg: str, which: str = 'main') -> None:
    ensure_dirs()
    line = f'[{now_local()}] {msg}'
    with open(OUT / f'log_{which}.txt', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


def rel(p) -> str:
    return S1.rel(p)


INPUT_FILES = list(S1.INPUT_FILES) + ['scripts/spatial_layout/sp_common.py', 'scripts/spatial_layout/sp_lines_table.py', 'scripts/spatial_layout/sp_dots_columns.py', 'scripts/spatial_layout/sp_geometry.py',
                                      'outputs/spatial_layout/spat01/positions_used_dots.json', 'outputs/spatial_layout/spat01/geometry_regular_baselines.json', 'outputs/spatial_layout/spat01/regression_cases_preregistered.json',
                                      'outputs/spatial_layout/spat01/regression_run_raw.json']


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
    start = load_json(OUT / start_name); changed = []; missing = []
    for f, r in start['files'].items():
        p = ROOT / f
        if not p.exists():
            missing.append(f); continue
        if 'sha256' in r and sha_file(p) != r['sha256']:
            changed.append(f)
    out = dict(checked_local=now_local(), n_files=len(start['files']), changed=changed, missing=missing, ok=(not changed and not missing))
    save_json(OUT / end_name, out)
    return out


def ledger(**kw) -> dict:

    p = OUT / 'ledger.json'; L = load_json(p) if p.exists() else dict(task=TASK_ID, historical_decodes_spat01=98, this_round_decode_attempts=0, this_round_decodes_succeeded=0, geometry_rebuilds=0, new_research_images=0, events=[])
    for k, v in kw.items():
        if k == 'event':
            L['events'].append(dict(time=now_local(), **v))
        else:
            L[k] = v
    L['total_decodes_cumulative'] = L['historical_decodes_spat01'] + L['this_round_decode_attempts']; L['updated_local'] = now_local()
    save_json(p, L); return L
