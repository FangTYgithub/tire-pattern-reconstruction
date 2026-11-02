# Provide source identities and storage for supplementary input checks.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'scripts' / 'surface_revision'
TR1 = SR / 'thesis_reinforce01'
if str(TR1) not in sys.path:
    sys.path.insert(0, str(TR1))
import tr_common as T1

OUT = ROOT / 'outputs' / 'surface_revision' / 'thesis_reinforce02'
RUN = OUT / 'run.json'
FIGS = OUT / 'figures'
OUT1 = T1.OUT
DRAFTS = T1.DRAFTS
TASK = DRAFTS / 'THESIS-REINFORCE-02_protocol.md'

MODEL_PREP_TASK = T1.MODEL_PREP_TASK
MODEL_PREP_OUT, MODEL_PREP_SCRIPTS = T1.MODEL_PREP_OUT, T1.MODEL_PREP_SCRIPTS
CAPS = dict(new_imaging=0, decode=0, eval_domain_calls=48, eval_combos=6)
Stop = T1.Stop
clean, rel, load_json, load_npz, read_csv, sha = T1.clean, T1.rel, T1.load_json, T1.load_npz, T1.read_csv, T1.sha
CODE_FILES = ('tr2_common.py', 'tr2_states.py', 'tr2_a2.py', 'tr2_a1.py', 'tr2_bind.py', 'tr2_plan.py', 'tr2_main.py', 'tr2_figs.py')
COMPUTE_FILES = ('tr2_common.py', 'tr2_states.py', 'tr2_a2.py', 'tr2_a1.py', 'tr2_bind.py', 'tr2_plan.py', 'tr2_main.py')


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() != q and OUT.resolve() not in q.parents:
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
    raise Stop('technical', f'cannot replace {q}')


def save_json(p, obj) -> Path:
    return _atomic(p, lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, default=str), encoding='utf-8'))


def save_csv(df, p) -> Path:
    return _atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='THESIS-REINFORCE-02', task_book=str(TASK),   created=now())


def run_put(section: str, obj) -> dict:
    r = run_load(); r[section] = clean(obj); r['updated'] = now(); save_json(RUN, r)
    return r


def code_identity(files=COMPUTE_FILES) -> dict:
    return {f: sha(HERE / f) for f in files if (HERE / f).exists()}


def require_registered(section: str) -> dict:

    reg = (run_load().get('registration') or {}).get('history') or []
    if not reg:
        raise Stop('refused', f'{section}: no code registration in run.json (run `tr2_main.py register` first)')
    cur = code_identity(); last = reg[-1]
    diff = sorted(f for f in cur if last['compute_code'].get(f) != cur[f])
    if diff:
        raise Stop('refused', f'{section}: computation code changed after the last registration: {diff}')
    return dict(registration_t=last['t'], compute_code=cur)


def model_prep_status() -> dict:
    files = []
    for d in (MODEL_PREP_OUT, MODEL_PREP_SCRIPTS):
        if d.exists():
            files += [dict(path=rel(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(d.rglob('*')) if p.is_file()]
    return dict(checked=now(), outputs_dir=rel(MODEL_PREP_OUT), scripts_dir=rel(MODEL_PREP_SCRIPTS), outputs_exist=MODEL_PREP_OUT.exists(),
                scripts_exist=MODEL_PREP_SCRIPTS.exists(), n_files=len(files), files=files[:200],
                task_book_sha256=sha(MODEL_PREP_TASK) if MODEL_PREP_TASK.exists() else None,
                finding=('delivery files present - status must be read from its own report (delivered pending review / accepted / missing)' if files else
                         'no MODEL-PREP-01 delivery found under the registered paths at check time (not a judgement on whether that task runs or failed)'))
