# Track line-experiment dependencies, case identities and execution counts.

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
L01 = ROOT / 'scripts' / 'line01'
OUT = ROOT / 'outputs' / 'line01r' / 'a'
OLD = ROOT / 'outputs' / 'line01'
OLD_CASES = OLD / 'cases'
LEDGER_DIR = OUT / 'ledger'
TASK_FILE = Path(r'C:\Users\16944\Documents\reference\2026-09-21\dang\outputs\research_runs\tasks\LINE-01R-A_protocol.md')
PLAN_FILE = Path(r'C:\Users\16944\Documents\reference\2026-09-21\dang\outputs\research_runs\tasks\LINE-01R_protocol.md')
AUDIT_FILE = Path(r'C:\Users\16944\Documents\reference\2026-09-21\dang\outputs\research_runs\audits\LINE-01_review.md')
AUDIT_JSON = AUDIT_FILE.parent / 'LINE-01_independent_numbers_2026-10-09.json'
TASK_ID = 'LINE-01R-A'
RENDER_CAP = 64

for _p in (str(L01), str(ROOT / 'src'), str(ROOT / 'scripts' / 'spacing_layout'), str(ROOT / 'scripts' / 'pm014'), str(ROOT / 'scripts' / 'pm015'), str(ROOT / 'scripts' / 'spotnoise'), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import l01_common as C0

now_local = C0.now_local
now_utc = C0.now_utc
sha_file = C0.sha_file
sha_bytes = C0.sha_bytes
sha_array = C0.sha_array
clean = C0.clean
save_json = C0.save_json
load_json = C0.load_json
save_text = C0.save_text
save_csv = C0.save_csv
rel = C0.rel
surface = C0.surface
omega = C0.omega
geom = C0.geom
grid_truth = C0.grid_truth
design = C0.design
m4_cfg = C0.m4_cfg
m3_cfg = C0.m3_cfg

D_LIST = [2.0, 3.0, 4.0, 4.5, 6.0, 8.0]
TAU_LIST = [0.05, 0.1, 0.15, 0.3, 0.5]
DETECTORS_OLD = ['C1_rowcentroid', 'C2_sg_h6', 'C2_sg_h11', 'C3_steger']


PROGRESS = OUT / 'progress.json'


def progress(**kw) -> dict:
    if PROGRESS.exists():
        pr = load_json(PROGRESS)
    else:
        pr = dict(task_id=TASK_ID, scope='A: definition correction, re-evaluation and re-decoding of the LINE-01 images, window interface, B input table, D design card, rehearsal; research imaging 0',
                  state='setup', started_local=now_local(), started_utc=now_utc(), stages={}, counts={}, blockers=[], pending_decisions=[], executor='execution (local session)',
                  selfcheck_status='not run', independent_review_status='not started (reference)')
    pr.update(kw); pr['updated_local'] = now_local(); pr['updated_utc'] = now_utc()
    save_json(PROGRESS, pr)
    return pr


def stage(name: str, **kw) -> dict:
    pr = progress(); st = dict(pr.get('stages', {}))
    st[name] = {**st.get(name, {}), **clean(kw), 'updated_local': now_local()}
    return progress(stages=st)


def log(msg: str) -> None:
    line = f'[{now_local()}] {msg}'
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'run.log', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def src_versions() -> dict:
    return {f'src/{n}': sha_file(ROOT / 'src' / n) for n in ('m2_pattern.py', 'm3_virtual_camera.py', 'm4_reconstruction.py', 'm4_lines.py', 'surface_eval_r3.py', 'm5_evaluation.py', 'm1_tire_surface.py')}


def image_key(spec: dict, image_sha: str) -> str:

    s = {k: spec[k] for k in ('surface', 'phi_deg', 'w_mm', 's_mm', 'peak', 'sigma', 'phase_frac', 'noise_seed', 'psf_px')}
    return hashlib.sha256((json.dumps(s, sort_keys=True) + image_sha).encode()).hexdigest()[:16]


def decode_key(path_params: dict) -> str:

    return hashlib.sha256((sha_file(ROOT / 'src' / 'm4_lines.py') + sha_file(ROOT / 'src' / 'm4_reconstruction.py') + json.dumps(clean(path_params), sort_keys=True)).encode()).hexdigest()[:16]


def eval_key(d_list=D_LIST, tau_list=TAU_LIST, support_rule='exact') -> str:
    return hashlib.sha256((sha_file(ROOT / 'src' / 'surface_eval_r3.py') + json.dumps(dict(d=d_list, tau=tau_list, rule=support_rule))).encode()).hexdigest()[:16]


def ledger_add(rec: dict) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_DIR / f'ledger_{os.getpid()}.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(rec), ensure_ascii=False) + '\n')


def ledger_records() -> list[dict]:
    recs = []
    if LEDGER_DIR.exists():
        for p in sorted(LEDGER_DIR.glob('ledger_*.jsonl')):
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    recs.append(json.loads(line))
    return sorted(recs, key=lambda r: r.get('t_local', ''))


def ledger_counts() -> dict:
    recs = ledger_records()
    by = {}
    for r in recs:
        b = by.setdefault(r.get('purpose', '?'), dict(attempts=0, ok=0, failed=0))
        b['attempts'] += 1; b['ok' if r.get('ok') else 'failed'] += 1
    return dict(total_attempts=len(recs), total_ok=sum(1 for r in recs if r.get('ok')), research_imaging=0, cap_tests_regression=RENDER_CAP, by_purpose=by)


def budget_check(n: int) -> None:
    c = ledger_counts()
    if c['total_attempts'] + n > RENDER_CAP:
        raise RuntimeError(f'render cap of this round exhausted: {c["total_attempts"]} + {n} > {RENDER_CAP}')


def old_case_ids(split: str | None = None) -> list[str]:
    specs = design()
    return [s['case_id'] for s in specs if split is None or s['split'] == split]


def old_spec(case_id: str) -> dict:
    for s in design():
        if s['case_id'] == case_id:
            return s
    raise KeyError(case_id)


def load_old_case(case_id: str, arrays: bool = True):
    rec = load_json(OLD_CASES / f'{case_id}.json')
    z = np.load(OLD_CASES / f'{case_id}.npz') if arrays else None
    return rec, z


def env_record() -> dict:
    return C0.env_record()
