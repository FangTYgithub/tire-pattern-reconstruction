# Provide common configuration and storage helpers for coded-pattern cases.

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
S3C2 = ROOT / 'scripts' / 'surface_revision' / 's3_code02'
S3C1 = ROOT / 'scripts' / 'surface_revision' / 's3_code01'
for _p in (str(S3C2), str(S3C1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c2_common as K2

OUT = ROOT / 'outputs' / 'surface_revision' / 's3_code03'
SANDBOX = OUT / 'sandbox'
for _k, _sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
    os.environ[_k] = str(SANDBOX / _sub)
os.environ['LINE01RB_RENDER_CAP'] = '0'
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TASK_ID = 'S3-CODE-03'
TASK_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-03_protocol.md'

PROTOCOL = OUT / 'protocol.json'
RESULTS, CHECKS, FIG, MAPS, RUN = OUT / 'results', OUT / 'checks', OUT / 'figures', OUT / 'maps', OUT / 'run'
REFS, EVA, DIAG, DEC = OUT / 'refs', OUT / 'eval', OUT / 'diag', OUT / 'dec'
FORMAL = OUT / 'formal'
O2, O1 = K2.OUT, K2.O1
RK = K2.RK
MAIN, SHORT, SIGMAS = K2.MAIN, K2.SHORT, K2.SIGMAS
GATE, GATE_SENS = K2.GATE, K2.GATE_SENS
RESID_LIMIT = K2.RESID_LIMIT
Q_BUDGET = K2.Q_BUDGET
Stop, now_local, sha_file, clean, digest, rel = K2.Stop, K2.now_local, K2.sha_file, K2.clean, K2.digest, K2.rel
load_json, load_npz, norm_seed, render_seed = K2.load_json, K2.load_npz, K2.norm_seed, K2.render_seed
EXIT = dict(ok=0, technical=2, cap=3, check=4, interrupted=9, stage=13, permit=14, freeze=15, no_imaging=16, lock=18)
CAPS = dict(new_imaging=0, drill_images=12, drill_method_exec=48)
C2_C_START = '2026-10-15T08:32:31+02:00'


def guard(p, root: Path | None = None) -> Path:
    q = Path(p).resolve(); base = (root or OUT).resolve()
    if OUT.resolve() not in (q, *q.parents) or base not in (q, *q.parents):
        raise Stop('technical', f'write outside {rel(base)} refused: {q}')
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


DEPS_FILE = CHECKS / 'deps_layers.json'
CONFIGS = [ROOT / 'outputs/line01r/night01r2/config/snapshot_line_encoding_dev_v1.json', ROOT / 'outputs/line01r/night01r3/config/r3_config.json',
           O1 / 'results/precheck_levels.json']
RELEASE = ROOT / 'outputs/surface_revision/surf02/release'
PLATFORM_CONFIGS = [ROOT / 'outputs/surface_revision/sections_adapter/binding.json', ROOT / 'outputs/surface_revision/s1_real01/config/s1_config.json',
                    ROOT / 'outputs/surface_revision/s1_real01/config/s1_dependencies.json', RELEASE / 'release_surf02.json', RELEASE / 'dependencies_surf02_v1_1.json']
ATTRIBUTION_VERSION = 'src_v1 (sl_lib.reference_map / classify / output_identity on the actual components; single source >= 50 % of the source support)'


def files_digest(files, root: Path = ROOT) -> dict:
    out = {}
    for f in files:
        k = str(f) if not Path(f).is_absolute() else rel(f)
        p = Path(root) / k
        out[k] = sha_file(p) if p.exists() else None
    return out


def layer_files(layer: str, deps: dict | None = None) -> list:
    d = deps or load_json(DEPS_FILE)
    return sorted(d['layers'][layer]['files'])


def eval_constants() -> dict:
    return dict(domain=RK.GLOBAL_DOMAIN_HASH, d=RK.D_SUPPORT, tau=RK.TAU, gross=RK.GROSS, gate=GATE, gate_sens=list(GATE_SENS))


def fingerprint(layer: str, root: Path = ROOT, deps: dict | None = None) -> str:

    files = layer_files(layer, deps)
    body = dict(layer=layer, code=files_digest(files, root), platform=files_digest([rel(c) for c in PLATFORM_CONFIGS], root))
    if layer == 'decode':
        body['configs'] = files_digest([rel(c) for c in CONFIGS], root)
    if layer in ('eval', 'diag'):
        body['constants'] = eval_constants()
    if layer == 'diag':
        body['attribution'] = ATTRIBUTION_VERSION
    return digest(body, 20)


def image_key(raster_key: str, peak_f32: float, surface: str, sigma: float, seed, m3_digest: str) -> str:
    return K2.image_key(raster_key, peak_f32, surface, sigma, seed, m3_digest)


def decode_key(img_key: str, family: str, method: str, rule: str, params: dict, fp: str) -> str:
    return 'C3D-' + digest(dict(img=img_key, family=family, method=method, rule=rule, params=params, fp=fp), 24)


def eval_key(dec_key: str, arm: str, fp: str) -> str:
    return 'C3E-' + digest(dict(dec=dec_key, arm=arm, fp=fp), 24)


def diag_key(dec_key: str, arm: str, ref_type: str, fp: str) -> str:
    return 'C3G-' + digest(dict(dec=dec_key, arm=arm, ref=ref_type, attribution=ATTRIBUTION_VERSION, fp=fp), 24)


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
            return bool(src.exists() and sha_file(src) == r['source_sha256'])
        z = p.with_suffix('.npz')
        return bool(z.exists() and sha_file(z) == r.get('npz_sha256'))
    except (OSError, KeyError):
        return False


def record_npz(p) -> Path:
    r = load_json(p)
    return (ROOT / r['source_npz']) if r.get('kind') == 'ref' else Path(p).with_suffix('.npz')


PACKAGE_MODULES = ('c3_common.py', 'c3_eval.py', 'c3_diag.py', 'c3_store.py', 'c3_raster.py', 'c3_run.py', 'c3_collect.py', 'c3_plan.py', 'c3_analysis.py', 'c3_risk.py',
                   'c3_figs.py', 'c3_deps.py', 'c3_protocol.py')


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
    rr, rm = RELEASE / 'release_surf02.json', RELEASE / 'dependencies_surf02_v1_1.json'
    return dict(files=files_digest(sorted(set(files)), root), surf02_release=dict(record=rel(rr), record_sha256=sha_file(rr), manifest=rel(rm), manifest_sha256=sha_file(rm)))


def verify_closure(fz: dict, root: Path = ROOT) -> list:
    now = files_digest(list(fz['files']), root)
    bad = [k for k, h in fz['files'].items() if now.get(k) != h]
    s = fz['surf02_release']
    if sha_file(ROOT / s['record']) != s['record_sha256'] or sha_file(ROOT / s['manifest']) != s['manifest_sha256']:
        bad.append('surf02_release')
    return bad


def ledger(path: Path, kind: str, key: str, status: str, **info) -> None:

    p = guard(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(dict(t=now_local(), kind=kind, key=key, status=status, **info)), ensure_ascii=False) + '\n')


def ledger_rows(root: Path) -> list:
    rows = []
    for p in sorted((Path(root) / 'run').glob('ledger*.jsonl')):
        rows += [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    return rows


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
