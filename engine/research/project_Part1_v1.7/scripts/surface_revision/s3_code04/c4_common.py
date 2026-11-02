# Define encoding data paths and shared experiment settings.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S3C3 = ROOT / 'scripts' / 'surface_revision' / 's3_code03'
for _p in (str(S3C3),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c3_common as K3

OUT = ROOT / 'outputs' / 'surface_revision' / 's3_code04'
SANDBOX = OUT / 'sandbox'
for _k, _sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
    os.environ[_k] = str(SANDBOX / _sub)
os.environ['LINE01RB_RENDER_CAP'] = '0'
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for _p in ('s12_resource01', 'sections_adapter', 's2_prepare'):
    _q = str(ROOT / 'scripts' / 'surface_revision' / _p)
    if _q not in sys.path:
        sys.path.append(_q)

TASK_ID = 'S3-CODE-04'
TASK_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-04_protocol.md'
REVIEW_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-03_review.md'
PROTOCOL = OUT / 'protocol.json'
CHECKS_FILE = OUT / 'checks.json'
RESULTS, FIG, FORMAL, RUN, CACHE = OUT / 'results', OUT / 'figures', OUT / 'formal', OUT / 'run', OUT / 'cache'
O3, O2, O1 = K3.OUT, K3.O2, K3.O1
RK = K3.RK
MAIN, SHORT, SIGMAS = K3.MAIN, K3.SHORT, K3.SIGMAS
GATE, GATE_SENS, RESID_LIMIT, Q_BUDGET = K3.GATE, K3.GATE_SENS, K3.RESID_LIMIT, K3.Q_BUDGET
Stop, now_local, sha_file, clean, digest, rel = K3.Stop, K3.now_local, K3.sha_file, K3.clean, K3.digest, K3.rel
load_json, load_npz, norm_seed, render_seed = K3.load_json, K3.load_npz, K3.norm_seed, K3.render_seed
EXIT = dict(K3.EXIT, incomplete=5, budget=3)
CAPS = dict(decode_total=3480, decode_C=3456, decode_B=24, unique_old_dot_inputs=1152, new_imaging=0)
PARITY_OF_PHASE = {f'N{i}': i % 2 for i in range(8)}


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in (q, *q.parents):
        raise Stop('technical', f'write outside {rel(OUT)} refused: {q}')
    return q


def _atomic(p: Path, write) -> Path:
    p = guard(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp'); write(tmp)
    for _ in range(50):
        try:
            os.replace(tmp, p); return p
        except PermissionError:
            time.sleep(0.1)
    raise Stop('technical', f'atomic replace failed for {p}')


def save_json(p, obj) -> Path:
    return _atomic(Path(p), lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'))


def save_text(p, text: str) -> Path:
    return _atomic(Path(p), lambda t: t.write_text(text, encoding='utf-8'))


def save_csv(df, p) -> Path:
    return _atomic(Path(p), lambda t: df.to_csv(t, index=False, encoding='utf-8'))


def save_npz(p, **arrays) -> str:
    def w(t):
        with open(t, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
    return sha_file(_atomic(Path(p), w))


def complete(p, identity: str) -> bool:

    p = Path(p)
    if not p.exists():
        return False
    try:
        r = load_json(p)
    except Exception:
        return False
    if r.get('complete') is not True or r.get('identity') != identity:
        return False
    try:
        if r.get('kind') == 'ref':
            src = ROOT / r['source_npz']
            if not (src.exists() and sha_file(src) == r['source_sha256']):
                return False
            if r.get('source_json'):
                sj = ROOT / r['source_json']
                return bool(sj.exists() and r.get('source_json_sha256') and sha_file(sj) == r['source_json_sha256'])
            return True
        z = p.with_suffix('.npz')
        return bool(z.exists() and sha_file(z) == r.get('npz_sha256'))
    except (OSError, KeyError):
        return False


def ledger(path: Path, kind: str, key: str, status: str, **info) -> None:
    p = guard(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(dict(t=now_local(), kind=kind, key=key, status=status, **info)), ensure_ascii=False) + '\n')


def ledger_rows(root: Path) -> list:
    rows = []
    for p in sorted(Path(root).rglob('ledger*.jsonl')):
        rows += [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    return rows


def decode_attempts(package: str) -> int:

    d = {'C': RUN / 'C', 'B': RUN / 'B'}[package]
    return sum(1 for r in ledger_rows(d) if r['kind'] == 'decode_attempt' and r['status'] == 'start') if d.exists() else 0


DEPS_FILE = FORMAL / 'deps_layers.json'
CONFIGS = K3.CONFIGS
PLATFORM_CONFIGS = K3.PLATFORM_CONFIGS
ATTRIBUTION_VERSION = K3.ATTRIBUTION_VERSION
PACKAGE_MODULES = ('c4_common.py', 'c4_parity.py', 'c4_store.py', 'c4_run.py', 'c4_collect.py', 'c4_analysis.py', 'c4_figs.py', 'c4_plan.py', 'c4_deps.py', 'c4_protocol.py')
files_digest, eval_constants = K3.files_digest, K3.eval_constants


def layer_files(layer: str, deps: dict | None = None) -> list:
    d = deps or load_json(DEPS_FILE)
    return sorted(d['layers'][layer]['files'])


def fingerprint(layer: str, root: Path = ROOT, deps: dict | None = None) -> str:
    files = layer_files(layer, deps)
    body = dict(layer=layer, code=files_digest(files, root), platform=files_digest([rel(c) for c in PLATFORM_CONFIGS], root))
    if layer == 'decode':
        body['configs'] = files_digest([rel(c) for c in CONFIGS], root)
        body['class_map'] = files_digest([rel(FORMAL / 'class_map.csv')], root)
    if layer in ('eval', 'diag'):
        body['constants'] = eval_constants()
    if layer == 'diag':
        body['attribution'] = ATTRIBUTION_VERSION
    return digest(body, 20)


image_key = K3.image_key


def decode_key(img_key: str, family: str, method: str, rule: str, params: dict, fp: str) -> str:
    return 'C4D-' + digest(dict(img=img_key, family=family, method=method, rule=rule, params=params, fp=fp), 24)


def eval_key(dec_key: str, arm: str, fp: str) -> str:
    return 'C4E-' + digest(dict(dec=dec_key, arm=arm, fp=fp), 24)


def diag_key(dec_key: str, arm: str, ref_type: str, fp: str) -> str:
    return 'C4G-' + digest(dict(dec=dec_key, arm=arm, ref=ref_type, attribution=ATTRIBUTION_VERSION, fp=fp), 24)


def record_npz(p) -> Path:
    r = load_json(p)
    return (ROOT / r['source_npz']) if r.get('kind') == 'ref' else Path(p).with_suffix('.npz')


def package_files(deps: dict | None = None) -> list:
    d = deps or load_json(DEPS_FILE)
    fs = set()
    for layer in d['layers'].values():
        fs |= set(layer['files'])
    fs |= {rel(c) for c in CONFIGS} | {rel(c) for c in PLATFORM_CONFIGS}
    fs |= {rel(HERE / m) for m in PACKAGE_MODULES if (HERE / m).exists()}
    return sorted(fs)


def freeze_closure(extra=(), root: Path = ROOT, deps: dict | None = None) -> dict:
    files = package_files(deps) + [rel(x) if Path(x).is_absolute() else str(x) for x in extra]
    rr, rm = K3.RELEASE / 'release_surf02.json', K3.RELEASE / 'dependencies_surf02_v1_1.json'
    return dict(files=files_digest(sorted(set(files)), root), surf02_release=dict(record=rel(rr), record_sha256=sha_file(rr), manifest=rel(rm), manifest_sha256=sha_file(rm)))


verify_closure = K3.verify_closure
PERMIT = FORMAL / 'permit.json'


def require_permit(permit_path: Path | None = None) -> dict:
    p = Path(permit_path or PERMIT)
    if not p.exists():
        raise Stop('no_imaging', f'no formal permit ({rel(p) if p.is_absolute() else p}); real imaging refused before any M3 call')
    pm = load_json(p)
    need = ('task_file', 'task_sha256', 'released_by', 'manifest_sha256', 'closure_sha256')
    if not all(pm.get(k) for k in need):
        raise Stop('permit', f'permit incomplete: needs {need}')
    if not (ROOT.parents[1] / pm['task_file']).exists() or sha_file(ROOT.parents[1] / pm['task_file']) != pm['task_sha256']:
        raise Stop('permit', 'permit task file missing or changed')
    return pm
