# Collect joint-study dependencies and provide output helpers.

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_A4 = HERE.parent / 's5_align04'
_A2 = HERE.parent / 's5_align02'
for _p in (_A4,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import s5t_common as K4

K3 = K4.K3
ROOT, SR = K4.ROOT, K4.SR
OUT = SR / 's5_align05'
FIGS = OUT / 'figures'
ISO = OUT / '_isolation'
RUN = OUT / 'run.json'
if os.environ.get('S5U_RUN'):
    _r = Path(os.environ['S5U_RUN']).resolve()
    if (OUT / '_isolation').resolve() not in _r.parents:
        raise SystemExit('S5U_RUN must point below outputs/surface_revision/s5_align05/_isolation')
    RUN = _r
FORMAL_ROOT, FORMAL_STORE, LEDGER = K4.FORMAL_ROOT, K4.FORMAL_STORE, K4.LEDGER
A3RUN, A4RUN = K4.A3RUN, K4.OUT / 'run.json'
TASK = K3.DRAFTS / 'S5-ALIGN-05_protocol.md'
REVIEW = K3.DRAFTS / 'S5-ALIGN-04_review.md'
ALLOWED = (OUT, FORMAL_ROOT)
CAPS = dict(render=2872, decode_D0=1724, decode_DF=1724, decode_P4=1148, evaluate=4596)
Stop = K3.Stop
EXIT = dict(K4.EXIT, unbound=18)
now, sha, digest, rel, clean, load_json, load_npz, read_csv = K3.now, K3.sha, K3.digest, K3.rel, K3.clean, K3.load_json, K3.load_npz, K3.read_csv


def root_of(p) -> Path:
    q = Path(p).resolve()
    for r in ALLOWED:
        if r.resolve() == q or r.resolve() in q.parents:
            return r
    raise Stop('guard', f'write outside the authorised roots refused: {q}')


def save_json(p, obj) -> Path:
    return K3.save_json(p, obj, root_of(p))


def save_csv(df, p) -> Path:
    return K3.atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'), root_of(p))


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='S5-ALIGN-05', task_book=str(TASK),  acceptance='reference (independent)', created=now())


def run_put(section: str, obj) -> dict:
    r = run_load()
    if section in r.get('fixed', []):
        raise Stop('freeze', f'run.json section {section} is fixed')
    r[section] = obj; r['updated'] = now(); save_json(RUN, r)
    return r


def run_fix(section: str, obj) -> dict:
    r = run_load()
    if section in r:
        if digest(r[section]) != digest(obj):
            raise Stop('freeze', f'{section} already fixed with a different content')
        return r
    r[section] = obj; r.setdefault('fixed', []).append(section); r.setdefault('fixed_at', {})[section] = now(); r['updated'] = now(); save_json(RUN, r)
    return r


def run_append(section: str, item: dict, keep: int = 120) -> None:
    r = run_load(); lst = r.get(section, []); lst.append(clean(item)); r[section] = lst[-keep:]; r['updated'] = now(); save_json(RUN, r)


_INDEX: dict = {}


def _index() -> dict:
    if not _INDEX:
        for f in list((ROOT / 'scripts').rglob('*.py')) + list((ROOT / 'src').glob('*.py')) + list(ROOT.glob('*.py')):
            if 'sandbox' in f.parts or '__pycache__' in f.parts:
                continue
            _INDEX.setdefault(f.stem, []).append(f.resolve())
    return _INDEX


def _spec_files(name: str) -> list:
    try:
        sp = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError, ModuleNotFoundError):
        sp = None
    if sp is not None and sp.origin and sp.origin.endswith('.py'):
        f = Path(sp.origin).resolve()
        return [f] if ROOT.resolve() in f.parents else []
    if sp is None and '.' not in name and name in _index():
        return _index()[name]
    return []


def closure(entries) -> list:

    todo = list(entries); seen = set(); files = set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for f in _spec_files(name):
            files.add(rel(f))
            tree = ast.parse(Path(f).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    todo += [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    todo += [node.module] + [f'{node.module}.{a.name}' for a in node.names]
    return sorted(files)


def loaded_project_files() -> list:
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if f and str(f).endswith('.py'):
            q = Path(f).resolve()
            if ROOT.resolve() in q.parents:
                out.add(rel(q))
    return sorted(out)


def json_lines(p: Path) -> list:
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []
