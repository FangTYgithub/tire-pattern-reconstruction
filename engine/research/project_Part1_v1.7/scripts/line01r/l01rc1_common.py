# Provide line-case evaluation and dependency helpers for calibration checks.

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs' / 'line01r' / 'c1'
LEDGER_DIR = OUT / 'ledger'
TASK_FILE = Path(r'C:\Users\16944\Documents\reference\2026-09-21\dang\outputs\research_runs\tasks\LINE-01R-C1_protocol.md')
AUDIT_FILE = Path(r'C:\Users\16944\Documents\reference\2026-09-21\dang\outputs\research_runs\audits\LINE-01R-B_review.md')
AUDIT_JSON = AUDIT_FILE.parent / 'LINE-01R-B_review.json'
TASK_ID = 'LINE-01R-C1'
RENDER_CAP_RESEARCH = 0
RENDER_CAP_TESTS = 64

sys.path.insert(0, str(HERE))
import l01rb_common as B

now_local, now_utc, sha_file, sha_bytes, sha_array, clean, save_json, load_json, save_text, save_csv, rel = (B.now_local, B.now_utc, B.sha_file, B.sha_bytes, B.sha_array, B.clean, B.save_json, B.load_json, B.save_text, B.save_csv, B.rel)
PROGRESS = OUT / 'progress_c1.json'


def progress(**kw) -> dict:
    if PROGRESS.exists():
        pr = load_json(PROGRESS)
    else:
        pr = dict(task_id=TASK_ID, state='setup', started_local=now_local(), started_utc=now_utc(), stages={}, counts={}, blockers=[], pending_decisions=[], executor='execution (local session)',
                  selfcheck_status='not run', independent_review_status='not started (reference)', research_imaging_cap=RENDER_CAP_RESEARCH)
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
    with open(OUT / 'run_c1.log', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def ledger_add(rec: dict) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_DIR / f'tests_{os.getpid()}.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(rec), ensure_ascii=False) + '\n')


def ledger_counts() -> dict:
    recs = []
    if LEDGER_DIR.exists():
        for p in sorted(LEDGER_DIR.glob('tests_*.jsonl')):
            recs += [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    by = {}
    for r in recs:
        b = by.setdefault(r.get('purpose', '?'), dict(attempts=0, ok=0, failed=0)); b['attempts'] += 1; b['ok' if r.get('ok') else 'failed'] += 1
    return dict(research=dict(attempts=0, cap=RENDER_CAP_RESEARCH), tests=dict(attempts=len(recs), ok=sum(1 for r in recs if r.get('ok')), failed=sum(1 for r in recs if not r.get('ok')), cap=RENDER_CAP_TESTS, by_purpose=by))


def wrapped_render_tests(purpose: str):
    import m3_virtual_camera as M3
    orig = M3.synthesise_camera_image

    def f(*a, **k):
        if ledger_counts()['tests']['attempts'] + 1 > RENDER_CAP_TESTS:
            raise RuntimeError('C1 test-render cap (64) exhausted')
        rec = dict(t_local=now_local(), purpose=purpose, ok=False); t0 = time.monotonic()
        try:
            out = orig(*a, **k)
        except Exception as e:
            rec.update(error=repr(e)[:200], seconds=round(time.monotonic() - t0, 3)); ledger_add(rec); raise
        rec.update(ok=True, seconds=round(time.monotonic() - t0, 3), image_sha256=sha_bytes(out[0].tobytes())); ledger_add(rec)
        return out
    return orig, f


def b_metrics():
    import pandas as pd
    return pd.read_csv(B.OUT / 'metrics_b.csv', low_memory=False)


def b_case(case_id_prefix: str):

    for p in B.CASES.glob('*.json'):
        if p.stem.startswith(case_id_prefix) and not p.name.endswith('.error.json'):
            return p.stem
    raise KeyError(case_id_prefix)


def stats_meta(df, metric_cols, unit='case', phases=2, noise_realisations=1):

    out = dict(unit=unit, n_cases=int(len(df)), n_phases=phases, n_noise_realisations_per_cell=noise_realisations, aggregation=['mean', 'median', 'min', 'max', 'p95'], metrics={})
    for m in metric_cols:
        s = df[m]
        v = s.dropna()
        out['metrics'][m] = dict(n_valid=int(len(v)), n_missing=int(s.isna().sum()), mean=(float(v.mean()) if len(v) else None), median=(float(v.median()) if len(v) else None), min=(float(v.min()) if len(v) else None),
                                 max=(float(v.max()) if len(v) else None), p95=(float(v.quantile(0.95)) if len(v) else None))
    return out


def src_versions() -> dict:
    v = B.src_versions()
    v.update({f'scripts/line01r/{n}': sha_file(HERE / n) for n in ('l01rb_common.py', 'l01rb_run.py', 'l01rb_analysis.py', 'l01r_common.py') if (HERE / n).exists()})
    v['scripts/line01/l01_common.py'] = sha_file(ROOT / 'scripts' / 'line01' / 'l01_common.py')
    return v
