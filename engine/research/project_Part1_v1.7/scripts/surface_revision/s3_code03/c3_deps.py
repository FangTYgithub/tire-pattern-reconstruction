# Collect imported source and data dependencies of the encoding pipeline.

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import c3_common as K

ENTRIES = dict(input=['c3_raster', 's3c_formal'], decode=['c3_store', 'c3_raster', 's3c_formal'], eval=['c3_eval', 'c3_store', 'c3_raster', 's3c_formal'],
               diag=['c3_diag', 'c3_eval', 'c3_store', 'c3_raster', 's3c_formal'], render=['rdy_platform', 'adp_binding', 'adp_surf02', 'm3_virtual_camera'])
PROJECT = K.ROOT.resolve()


_INDEX: dict | None = None
AMBIGUOUS: dict = {}


def _index() -> dict:

    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for f in list((K.ROOT / 'scripts').rglob('*.py')) + list((K.ROOT / 'src').glob('*.py')) + list(K.ROOT.glob('*.py')):
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
        return [f] if PROJECT in f.parents else []
    if sp is None and '.' not in name and name in _index():
        c = _index()[name]
        if len(c) > 1:
            AMBIGUOUS[name] = [K.rel(x) for x in c]
        return c
    return []


def _imports_of(f: Path) -> set:
    tree = ast.parse(f.read_text(encoding='utf-8'), filename=str(f)); names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module); names |= {f'{node.module}.{a.name}' for a in node.names}
    return names


def closure(entries) -> dict:

    todo = [e for e in entries]; seen_mod = set(); files = {}
    while todo:
        name = todo.pop()
        if name in seen_mod:
            continue
        seen_mod.add(name)
        for f in _spec_files(name):
            r = K.rel(f); files.setdefault(r, []).append(name)
            for n in _imports_of(f):
                if n not in seen_mod:
                    todo.append(n)
    return files


def capture() -> dict:
    layers = {}
    for layer, ents in ENTRIES.items():
        cl = closure(ents); layers[layer] = dict(entries=ents, files=sorted(cl), n=len(cl))

    man = set(K.load_json(K.RELEASE / 'dependencies_surf02_v1_1.json')['files'])
    union = sorted(set().union(*[set(v['files']) for v in layers.values()]))
    c2 = K.load_json(K.O2 / 'protocol.json')['freeze_C']['closure']['files']
    out = dict(method='static AST closure (top-level + function-level imports), resolved with the runtime sys.path, fallback: project module-name index '
                      '(names whose directory is added to sys.path only at run time; several candidates -> all kept); project files only', layers=layers,
               ambiguous_names=AMBIGUOUS,
               union=union, n_union=len(union), not_in_surf02_manifest=sorted(set(union) - man), not_in_s3code02_freeze=sorted(set(union) - set(c2)),
               s3code02_freeze_not_in_union=sorted(set(c2) - set(union) - {K.rel(c) for c in K.CONFIGS}), sys_path=[K.rel(p) if str(p).startswith(str(K.ROOT)) else p for p in sys.path[:12]],
               t_local=K.now_local())
    K.save_json(K.DEPS_FILE, out)
    return out


def loaded_project_files() -> list:
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if f and str(f).endswith('.py'):
            q = Path(f).resolve()
            if PROJECT in q.parents:
                out.add(K.rel(q))
    return sorted(out)


def check(loaded: list, layers=('decode', 'eval', 'diag')) -> dict:
    d = K.load_json(K.DEPS_FILE); allowed = set().union(*[set(d['layers'][l]['files']) for l in layers])
    infra = {K.rel(K.HERE / m) for m in ('c3_common.py', 'c3_deps.py', 'c3_run.py', 'c3_collect.py', 'c3_import.py', 'c3_drill.py', 'c3_diagrun.py', 'c3_risk.py',
                                           'c3_figs.py', 'c3_plan.py', 'c3_analysis.py', 'c3_protocol.py')}
    miss = sorted(set(loaded) - allowed - infra)
    return dict(n_loaded=len(loaded), missing_from_closure=miss, ok=not miss)


if __name__ == '__main__':
    if sys.argv[1] == 'capture':
        r = capture(); print(json.dumps({k: (v if k != 'layers' else {l: x['n'] for l, x in v.items()}) for k, v in r.items() if k not in ('union',)}, indent=1, ensure_ascii=False)[:4000])
