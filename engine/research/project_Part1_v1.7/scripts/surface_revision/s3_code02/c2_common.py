# Define decoder calibration and configuration paths for coded patterns.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S3C1 = ROOT / 'scripts' / 'surface_revision' / 's3_code01'
for _p in (str(S3C1),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import s3c_common as K1

OUT = ROOT / 'outputs' / 'surface_revision' / 's3_code02'
SANDBOX = OUT / 'sandbox'
for _k, _sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
    os.environ[_k] = str(SANDBOX / _sub)
os.environ['LINE01RB_RENDER_CAP'] = '0'
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

RK = K1.RK
O1 = K1.OUT
TASK_FILE = ROOT.parents[1] / 'protocols' / 'S3-CODE-02_protocol.md'
TASK_ID = 'S3-CODE-02'
PROTOCOL = OUT / 'protocol.json'
RESULTS, FIG, CHECKS, CACHE, RUN = OUT / 'results', OUT / 'figures', OUT / 'checks', OUT / 'cache', OUT / 'run'
IMG, DEC, EVA = OUT / 'images', OUT / 'dec', OUT / 'eval'
LEDGER = RUN / 'ledger.jsonl'

PLATFORM_VERSION = RK.PLATFORM_VERSION
MAIN = ('M10_curv', 'M24_curv', 'B_curv')
SHORT = {'M10_curv': 'M10', 'M24_curv': 'M24', 'B_curv': 'B'}
SIGMAS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
Q_BUDGET = RK.Q
NOMINAL_MM_PER_PX = K1.NOMINAL_MM_PER_PX
CAPS = dict(render_planned=1224, render_attempts=1248, old_inputs=960)
GATE, GATE_SENS = 0.001, (0.0005, 0.005)
DCT_FLAG, DCT_FLAG_SENS = 0.01, (0.005, 0.02)
RESID_LIMIT = dict(dots=0.01, lines=0.05)

EXIT = dict(ok=0, technical=2, cap=3, check=4, interrupted=9, stage=13, permit=14, freeze=15, lock=18)


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}')
        self.code, self.msg = code, msg


def now_local() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


clean, digest, rel = RK.clean, RK.digest, RK.rel


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


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def load_npz(p) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def norm_seed(x):

    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if s in ('', 'nan', 'NaN', 'None'):
        return None
    return int(round(float(s)))


def render_seed(sigma: float, seed) -> int:

    s = norm_seed(seed)
    if float(sigma) == 0.0:
        if s is not None:
            raise Stop('check', 'sigma 0 carries a seed (sigma-0 images are noise-free and never repeated by seed label)')
        return 0
    if s is None:
        raise Stop('check', f'sigma {sigma} without a registered seed')
    return s


DECODE_FILES = [S3C1 / 's3c_dev.py', S3C1 / 's3c_precheck.py', HERE / 'c2_decode.py',
                ROOT / 'scripts/surface_revision/s345_ready/rdy_decode.py', ROOT / 'scripts/surface_revision/s345_ready/rdy_inputs.py',
                ROOT / 'scripts/line01r/lc_pipeline.py', ROOT / 'scripts/line01r/lc_readout.py', ROOT / 'scripts/line01r/lc_r4_runtime.py',
                ROOT / 'scripts/line01r/lc_r3_reliability.py', ROOT / 'scripts/line01r/lc_pattern.py', ROOT / 'scripts/line01r/lc_r4_common.py',
                ROOT / 'scripts/encoding_prep/ep_detect.py', ROOT / 'scripts/encoding_prep/ep_patterns.py', ROOT / 'scripts/spotnoise_round5/r5_detect.py',
                ROOT / 'src/m2_pattern.py', ROOT / 'src/m4_reconstruction.py', ROOT / 'src/m4_lines.py']
DECODE_CONFIGS = [ROOT / 'outputs/line01r/night01r2/config/snapshot_line_encoding_dev_v1.json', ROOT / 'outputs/line01r/night01r3/config/r3_config.json',
                  O1 / 'results/precheck_levels.json']
EVAL_FILES = [ROOT / 'scripts/surface_revision/sections_adapter/s1r_eval.py', ROOT / 'scripts/surface_revision/sections_adapter/s1r_common.py',
              HERE / 'c2_identity.py', ROOT / 'scripts/spacing_layout/sl_lib.py', ROOT / 'scripts/line01r/lc_r4_rules.py']
RELEASE = ROOT / 'outputs/surface_revision/surf02/release'


def _files_digest(files) -> dict:
    return {rel(f): (sha_file(f) if Path(f).exists() else None) for f in files}


def decode_fingerprint(extra_configs=()) -> str:
    return digest(dict(code=_files_digest(DECODE_FILES), configs=_files_digest(list(DECODE_CONFIGS) + list(extra_configs))), 20)


def eval_fingerprint() -> str:
    return digest(dict(code=_files_digest(EVAL_FILES), domain=RK.GLOBAL_DOMAIN_HASH, d=RK.D_SUPPORT, tau=RK.TAU, gross=RK.GROSS), 20)


def image_key(raster_key: str, peak_f32: float, surface: str, sigma: float, seed, m3_digest: str) -> str:
    return 'C2I-' + digest(dict(raster=raster_key, peak_f32=float(np.float32(peak_f32)), surface=RK.SURF_MAP[surface], psf=RK.PSF_PX, sigma=float(sigma),
                                seed=norm_seed(seed), m3=m3_digest, platform=PLATFORM_VERSION), 24)


def decode_key(img_key: str, family: str, method: str, rule: str, params: dict, fp: str) -> str:
    return 'C2D-' + digest(dict(img=img_key, family=family, method=method, rule=rule, params=params, fp=fp), 24)


def eval_key(dec_key: str, arm: str, fp: str) -> str:
    return 'C2E-' + digest(dict(dec=dec_key, arm=arm, fp=fp), 24)


def complete(p, identity: str) -> bool:

    p = Path(p)
    if not p.exists():
        return False
    try:
        r = load_json(p)
    except Exception:
        return False
    z = p.with_suffix('.npz')
    try:
        return bool(r.get('complete') is True and r.get('identity') == identity and z.exists() and sha_file(z) == r.get('npz_sha256'))
    except OSError:
        return False


def freeze_closure(extra=()) -> dict:
    files = list(DECODE_FILES) + list(DECODE_CONFIGS) + list(EVAL_FILES) + [HERE / f for f in ('c2_common.py', 'c2_run.py', 'c2_report.py', 'c2_calib.py', 'c2_plan.py')] + list(extra)
    rel_rec = RELEASE / 'release_surf02.json'; rel_man = RELEASE / 'dependencies_surf02_v1_1.json'
    return dict(files=_files_digest(files), surf02_release=dict(record=rel(rel_rec), record_sha256=sha_file(rel_rec), manifest=rel(rel_man), manifest_sha256=sha_file(rel_man)))


def verify_closure(fz: dict) -> list:
    now = _files_digest([ROOT / k for k in fz['files']])
    bad = [k for k, h in fz['files'].items() if now.get(k) != h]
    s = fz['surf02_release']
    if sha_file(ROOT / s['record']) != s['record_sha256'] or sha_file(ROOT / s['manifest']) != s['manifest_sha256']:
        bad.append('surf02_release')
    return bad


def ledger(kind: str, key: str, status: str, **info) -> None:

    p = guard(LEDGER); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(dict(t=now_local(), kind=kind, key=key, status=status, **info)), ensure_ascii=False) + '\n')


def ledger_rows(path=None) -> list:
    p = Path(path or LEDGER)
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []


def render_attempts(path=None) -> int:
    return sum(1 for r in ledger_rows(path) if r['kind'] == 'render_attempt')


def register_sources(name: str, paths) -> dict:
    rec = {rel(p): sha_file(p) for p in sorted(set(Path(x) for x in paths))}
    save_json(CHECKS / f'sources_{name}.json', dict(name=name, n=len(rec), t_local=now_local(), files=rec))
    return rec


def verify_sources(name: str) -> dict:
    r = load_json(CHECKS / f'sources_{name}.json'); bad = [k for k, h in r['files'].items() if not (ROOT / k).exists() or sha_file(ROOT / k) != h]
    return dict(name=name, n=r['n'], changed=bad, ok=not bad)
