# Provide source registration and guarded output for joint-study analysis.

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_A5 = HERE.parent / 's5_align05'
if str(_A5) not in sys.path:
    sys.path.insert(0, str(_A5))
import s5u_common as K5

K3 = K5.K3
ROOT, SR = K5.ROOT, K5.SR
A5 = K5.OUT
A5RUN = A5 / 'run.json'
OUT = SR / 's5_align06'
FIGS = OUT / 'figures'
RUN = OUT / 'run.json'
TASK = K3.DRAFTS / 'S5-ALIGN-06_protocol.md'
REVIEW = K3.DRAFTS / 'S5-ALIGN-05_review.md'
Stop = K3.Stop
EXIT = dict(K5.EXIT, code_changed=19)
now, sha, digest, rel, clean, load_json, load_npz, read_csv = K3.now, K3.sha, K3.digest, K3.rel, K3.clean, K3.load_json, K3.load_npz, K3.read_csv
SURF = ('B_curv', 'M10_curv', 'M24_curv')
SIG = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
SIG_LABELS = tuple(f'{s:.1f}' for s in SIG)
CROSS = '40-0'
POSTHOC = 'posthoc (reference S5-ALIGN-01 review 3.2)'
TABLE_E, TABLE_FC = 'four_cell_and_frontend_effects', 'frontend_condition'
REGISTERED_MESH = {'B_curv': '0a1ac8130478b682', 'M10_curv': '80db5f55eca4e989', 'M24_curv': '4c503bf09d6454cd'}


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() != q and OUT.resolve() not in q.parents:
        raise Stop('guard', f'write outside {rel(OUT)} refused: {q}')
    return q


def save_json(p, obj) -> Path:
    return K3.save_json(guard(p), obj, OUT)


def save_csv(df, p) -> Path:
    return K3.atomic(guard(p), lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'), OUT)


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='S5-ALIGN-06', task_book=str(TASK), review=str(REVIEW), 
                                                    acceptance='reference (independent, pending)', created=now())


def run_put(section: str, obj) -> dict:
    r = run_load(); r[section] = clean(obj); r['updated'] = now(); save_json(RUN, r)
    return r


def project_files_loaded() -> dict:
    return {f: sha(ROOT / f) for f in K5.loaded_project_files()}


def register_code() -> dict:

    files = project_files_loaded()
    out = dict(files=files, digest=digest(files), n_files=len(files), t=now(),
               note='all project modules imported by s5v_entry (06 scripts, frozen 05 / 02 / 03 / 04 modules, platform modules for the mesh loader)')
    run_put('code_binding', out)
    return out


def check_code() -> dict:
    reg = (run_load().get('code_binding') or {}).get('files')
    if not reg:
        raise Stop('freeze', 'no code binding: run `s5v_entry.py register` first')
    now_files = project_files_loaded()
    changed = sorted(f for f, h in now_files.items() if reg.get(f) != h)
    if changed:
        raise Stop('code_changed', f'{len(changed)} loaded project files differ from the registration (re-register and rerun the affected checks)', detail=changed[:10])
    return dict(ok=True, n_files=len(now_files), digest=digest(reg))
