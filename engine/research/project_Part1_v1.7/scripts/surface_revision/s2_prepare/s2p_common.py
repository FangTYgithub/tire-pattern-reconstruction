# Define spacing-layout settings, resources and platform dependencies.

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
ROOT = HERE.parents[2]
OUT = ROOT / 'outputs' / 'surface_revision' / 's2_prepare'
SANDBOX = OUT / 'sandbox'
ADP = ROOT / 'scripts' / 'surface_revision' / 'sections_adapter'
S1OUT = ROOT / 'outputs' / 'surface_revision' / 's1_real01'
REL = ROOT / 'outputs' / 'surface_revision' / 'surf02' / 'release'
CANDS = ROOT / 'outputs' / 'surface_revision' / 'sections_adapter' / 'candidates_s1s5.csv'
BINDING = ROOT / 'outputs' / 'surface_revision' / 'sections_adapter' / 'binding.json'

Q = 11695.0  # Sum of normalized projector pixel intensities; peak is capped separately.
WINDOW = (-16.0, 4.0)  # Lower and upper reconstruction depths [mm].
C_REF = 16.645465475819755
D_SUPPORT, TAU, GROSS = 6.0, 0.15, 1.0  # Support scale, height tolerance and gross-error threshold [mm].
PSF_PX = 1.0  # Fixed Gaussian PSF standard deviation [px] for this experiment.
MAIN = ('M10_curv', 'M24_curv', 'B_curv')
GAUSS = 'M10_curv_gauss'
SURFACES = MAIN + (GAUSS,)
PLATFORM_ID = {'M10_curv': 'A_curv', 'M24_curv': 'C_curv', 'B_curv': 'B_curv', 'M10_curv_gauss': 'A_curv_gauss'}
ARM = {'point': 'D0', 'line': 'P4_C3_cal'}

STABLE_INPUTS = ['outputs/surface_revision/surf02/release/release_surf02.json', 'outputs/surface_revision/surf02/release/dependencies_surf02_v1_1.json',
                 'outputs/surface_revision/sections_adapter/binding.json', 'outputs/surface_revision/sections_adapter/candidates_s1s5.csv',
                 'outputs/surface_revision/s1_real01/config/s1_config.json', 'outputs/surface_revision/s1_real01/config/s1_dependencies.json',
                 'outputs/surface_revision/s1_real01/domain/global_domain_step0p50.json', 'outputs/surface_revision/s1_real01/domain/global_domain_step0p50.npz',
                 'outputs/surface_revision/s1_real01/manifest/s1_images.csv', 'outputs/surface_revision/s1_real01/manifest/s1_views.csv',
                 'outputs/spacing_layout/design/case_list.csv', 'outputs/spacing_layout/design/budget.json', 'outputs/line01r/c3/night/formal_manifest_4928_c3.csv',
                 'outputs/line01r/c3/config/paths_c3_validation.json',
                 'src/m2_pattern.py', 'src/m3_virtual_camera.py', 'src/m4_reconstruction.py', 'src/m4_lines.py', 'src/surface_eval_r3.py',
                 'scripts/spacing_layout/sl_lib.py', 'scripts/spotnoise/sn_lib.py', 'scripts/line01r/l01rb_common.py', 'scripts/spotnoise_round5/r5_detect.py',
                 'scripts/surface_revision/surf02/s2_common.py', 'scripts/surface_revision/sections_adapter/adp_surf02.py', 'scripts/surface_revision/sections_adapter/adp_binding.py',
                 'scripts/surface_revision/sections_adapter/adp_patterns.py', 'scripts/surface_revision/sections_adapter/s1r_eval.py', 'scripts/surface_revision/sections_adapter/s1r_common.py']
GENERATORS = ['src/m2_pattern.py', 'scripts/spacing_layout/sl_lib.py', 'scripts/spotnoise/sn_lib.py', 'scripts/line01r/l01rb_common.py', 'scripts/surface_revision/sections_adapter/adp_patterns.py']


JOINT10 = dict(new=['C013282', 'C013244', 'C013224', 'C013279', 'C013736', 'C018447', 'C018309', 'C018194'], reuse=['C018107', 'C019194'])
FIRST_REAL = 'C013244'
CAPS_JOINT10 = dict(render=8, decode=8)
RUN_DIR = OUT / 'run'
LEDGER = RUN_DIR / 'ledger.jsonl'
LOCK = RUN_DIR / 'run.lock'
PERMIT = RUN_DIR / 'permit_joint10.json'
FREEZE = RUN_DIR / 'freeze_joint10.json'
RESULTS = OUT / 'results'
S1_DEPS = S1OUT / 'config' / 's1_dependencies.json'


def norm_seed(x):

    if x is None:
        return None
    s = str(x).strip()
    if s in ('', 'nan', 'None', 'NaN'):
        return None
    return int(round(float(s)))


def setup():
    for k, sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
        os.environ.setdefault(k, str(SANDBOX / sub))
    os.environ.setdefault('LINE01RB_RENDER_CAP', '0')
    for p in (ADP, ROOT / 'scripts' / 'surface_revision' / 'surf02'):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import adp_patterns as PT
    PT.setup_paths()


def now_local() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def digest(obj, n=None) -> str:
    h = hashlib.sha256(json.dumps(clean(obj), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return h[:n] if n else h


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in (q, *q.parents):
        raise RuntimeError(f'S2-PREP: write outside {OUT} refused: {q}')
    return q


def save_json(p, obj):
    p = guard(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8')
    for _ in range(50):
        try:
            os.replace(tmp, p); return
        except PermissionError:
            time.sleep(0.1)
    raise RuntimeError(f'atomic replace failed for {p}')


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def input_hashes() -> dict:
    return {f: (sha_file(ROOT / f) if (ROOT / f).exists() else None) for f in STABLE_INPUTS}


def platform_state() -> dict:

    rec = load_json(REL / 'release_surf02.json') if (REL / 'release_surf02.json').exists() else None
    out = dict(released=bool(rec and rec.get('released') is True), version=(rec or {}).get('version'), scope=(rec or {}).get('scope'))
    if rec:
        man = load_json(ROOT / rec['dependency_manifest']); files = man.get('files', {})
        out['dependency_drift'] = [k for k, v in files.items() if not (ROOT / k).exists() or sha_file(ROOT / k) != v]
        out['n_dependencies'] = len(files)
        b = load_json(BINDING)
        out['binding_release_sha_ok'] = (b['platform'].get('release_record_sha256') == sha_file(REL / 'release_surf02.json'))
        out['binding_version'] = b['platform'].get('expected_version'); out['binding_methods'] = b.get('methods')
        basis = rec.get('basis') or {}
        out['replay'] = basis.get('replay') if isinstance(basis, dict) else None
    s1d = S1OUT / 'config' / 's1_dependencies.json'
    out['s1_dependencies_sha256'] = sha_file(s1d) if s1d.exists() else None
    g = S1OUT / 'domain' / 'global_domain_step0p50.json'
    out['global_domain'] = dict(file=rel(g), hash=(load_json(g)['hash'] if g.exists() else None), n_nodes=(load_json(g)['n_nodes'] if g.exists() else None))
    log = S1OUT / 'run.log'
    out['s1_run_log_tail'] = log.read_text(encoding='utf-8', errors='replace').strip().splitlines()[-2:] if log.exists() else None
    out['bound_ok'] = bool(out['released'] and out.get('dependency_drift') == [] and out.get('binding_release_sha_ok') and set((out.get('binding_methods') or {})) >= {'D0', 'P4_C3_cal'})
    return out
