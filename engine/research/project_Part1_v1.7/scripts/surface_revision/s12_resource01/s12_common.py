# Define the size-spacing experiment matrix, resource limits and evaluation settings.

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
OUT = ROOT / 'outputs' / 'surface_revision' / 's12_resource01'
SANDBOX = OUT / 'sandbox'
for _k, _sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
    os.environ.setdefault(_k, str(SANDBOX / _sub))
os.environ.setdefault('LINE01RB_RENDER_CAP', '0')
ADP = ROOT / 'scripts' / 'surface_revision' / 'sections_adapter'
S2P = ROOT / 'scripts' / 'surface_revision' / 's2_prepare'
for _p in (ADP, S2P, ROOT / 'scripts' / 'surface_revision' / 'surf02', HERE):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

S1OUT = ROOT / 'outputs' / 'surface_revision' / 's1_real01'
S2OUT = ROOT / 'outputs' / 'surface_revision' / 's2_prepare'
REL = ROOT / 'outputs' / 'surface_revision' / 'surf02' / 'release'
NOISE_EXAMPLE = ROOT / 'outputs' / 'surface_revision' / 's1_line_noise_example'
TASK_FILE = ROOT.parents[1] / 'protocols' / 'S12-RESOURCE-01_protocol.md'
TASK_ID = 'S12-RESOURCE-01'

MANIFEST = OUT / 'manifest.csv'
INPUTS = OUT / 'inputs.csv'
PROTOCOL = OUT / 'protocol.json'
LEDGER = OUT / 'run' / 'ledger.jsonl'
LOCK = OUT / 'run' / 'run.lock'
IMG_DIR, RAW_DIR, EVAL_DIR = OUT / 'images', OUT / 'raw', OUT / 'eval'
RESULTS, FIG, CHECKS = OUT / 'results', OUT / 'figures', OUT / 'checks'

Q_BUDGET, V_MAX, Q_LOW = 11695.0, 1.0, 5850.0  # Main integral cap, peak ceiling and lower-budget control [normalized gray units].
REGIMES = ('resource_cap_11695', 'fixed_peak_1', 'equal_integral_5850')
REGIME_DE = {'resource_cap_11695': 'gemeinsame Obergrenze (Integral 11695, Peak 1)', 'fixed_peak_1': 'fester Peak 1 (Mechanismuskontrolle)',
             'equal_integral_5850': 'gleiches Integral 5850 (Punkte, Niedrigbudget)'}
SIGMAS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)  # Registered noise standard deviations [8-bit gray levels] for this experiment.
DOT_RADII = (0.5, 1.0, 1.5)  # Registered dot radii [mm]; additional size-study radii are supplied by SIZE02.
LINE_WIDTHS = (0.5, 0.75, 1.0, 1.25, 1.5)  # Registered line widths [mm].
S2_LATTICES = ('R6x4', 'R7.35x4', 'R8x4', 'R10x4', 'R12x4', 'R8x2.5', 'R8x3', 'R8x6', 'R8x8', 'T8x4')
S2_PITCHES = (4.45, 5.0, 5.6, 6.3, 8.0, 10.0, 12.0)  # Line pitches tested in the spacing study [mm].
S2_PHIS = (0.0, 45.0)  # Line orientations tested in the spacing study [deg].
MAIN = ('M10_curv', 'M24_curv', 'B_curv')
GAUSS = 'M10_curv_gauss'
SURFACES = MAIN + (GAUSS,)
SURF_INDEX = {'M10_curv': 0, 'M24_curv': 1, 'B_curv': 2, 'M10_curv_gauss': 3}
PLATFORM_ID = {'M10_curv': 'A_curv', 'M24_curv': 'C_curv', 'B_curv': 'B_curv', 'M10_curv_gauss': 'A_curv_gauss'}
SHORT = {'M10_curv': 'M10', 'M24_curv': 'M24', 'B_curv': 'B', 'M10_curv_gauss': 'M10+Gauß'}
ARM = {'point': 'D0', 'line': 'P4_C3_cal'}
PSF_PX = 1.0  # Fixed Gaussian PSF standard deviation [px] for this experiment.
D_SUPPORT, TAU, GROSS = 6.0, 0.15, 1.0  # Support scale, height tolerance and gross-error threshold [mm].
SENS_D, SENS_TAU = (5.0, 6.0, 7.0), (0.10, 0.15, 0.20)  # Support-distance and height-tolerance sensitivity levels [mm].
WINDOW = (-16.0, 4.0)  # Lower and upper reconstruction depths [mm].
C_REF_MM = 16.645465475819755
BOOT_SEED, BOOT_N = 2026101401, 10000  # Fixed resampling seed and number of bootstrap draws.
CAPS = dict(render=8192 + 16, decode=8192 + 16, replay_decode=12)
FIRST_MAX = 16

STOP_EXIT = dict(technical_stop=11, platform_mismatch=12, matrix_drift=13, cap=14, usage=2, reuse_refused=16, lock=18)


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg); self.code = code; self.exit_code = STOP_EXIT.get(code, 11)


for _d in (OUT, SANDBOX, OUT / 'run', IMG_DIR, RAW_DIR, EVAL_DIR, RESULTS, FIG, CHECKS):
    _d.mkdir(parents=True, exist_ok=True)


def now_local() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b: bytes, n: int | None = None) -> str:
    h = hashlib.sha256(b).hexdigest()
    return h[:n] if n else h


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


def digest(obj, n: int | None = None) -> str:
    h = hashlib.sha256(json.dumps(clean(obj), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return h[:n] if n else h


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in (q, *q.parents):
        raise Stop('technical_stop', f'write outside {rel(OUT)} refused: {q}')
    return q


def save_json(p, obj) -> Path:
    p = guard(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8')
    for _ in range(50):
        try:
            os.replace(tmp, p); return p
        except PermissionError:
            time.sleep(0.1)
    raise Stop('technical_stop', f'atomic replace failed for {p}')


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save_npz(p, **arrays) -> str:
    p = guard(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    for _ in range(50):
        try:
            os.replace(tmp, p); break
        except PermissionError:
            time.sleep(0.1)
    return sha_file(p)


def save_csv(df, p) -> Path:
    p = guard(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, p)
    return p


def complete(p, identity: str | None = None, id_field: str = 'identity') -> bool:

    p = Path(p)
    if not p.exists():
        return False
    try:
        r = load_json(p)
    except Exception:
        return False
    z = p.with_suffix('.npz')
    ok = r.get('complete') is True and z.exists() and sha_file(z) == r.get('npz_sha256')
    return bool(ok and (identity is None or r.get(id_field) == identity))


def seed_s1_dot(phase_index: int, rep: int) -> int:
    return 9205000 + 10 * int(phase_index) + int(rep)


def seed_s2_dot(position_index: int, rep: int) -> int:
    return 9420000 + 10 * int(position_index) + int(rep)


def seed_line(surface: str, phase_index: int, sigma: float, rep: int) -> int | None:
    s = float(sigma); si = SURF_INDEX[surface]; pi = int(phase_index)
    if s == 0.0:
        return None
    if s in (20.0, 40.0, 60.0):
        return 10300000 + ((si * 8 + pi) * 4 + (0.0, 20.0, 40.0, 60.0).index(s)) * 2 + int(rep)
    if s in (80.0, 100.0):
        return 10600000 + ((si * 8 + pi) * 2 + (80.0, 100.0).index(s)) * 2 + int(rep)
    raise Stop('matrix_drift', f'no registered line seed for sigma {sigma}')


SEED_RULES = dict(s1_dot='9205000 + 10 phase_index + rep (sn_lib, surface independent; same base realisation for every sigma)',
                  s2_dot='9420000 + 10 position_index + rep (sl_lib, position index from the old list; surface independent; same base realisation for every sigma)',
                  line_20_40_60='10300000 + ((surface_index * 8 + phase_index) * 4 + sigma_index) * 2 + rep, sigma_index from [0, 20, 40, 60] (unchanged old rule)',
                  line_80_100='10600000 + ((surface_index * 8 + phase_index) * 2 + high_sigma_index) * 2 + rep, high_sigma_index from [80, 100] (new block)',
                  surface_index='M10 0, M24 1, B 2, M10+Gauss 3', sigma0='no effective seed (deterministic image; one image per phase)')


def phase_index(label: str) -> int:
    return int(str(label).replace('sn_phase', '').replace('ph', '').replace('pos', ''))


def resource(binary: np.ndarray, regime: str) -> dict:

    b = np.asarray(binary) > 0
    Q0 = float(b.sum(dtype=np.float64))
    if Q0 <= 0:
        raise Stop('technical_stop', 'empty raster')
    if regime == 'resource_cap_11695':
        target = min(V_MAX, Q_BUDGET / Q0); budget = Q_BUDGET
        lim = 'peak_limited' if Q0 < Q_BUDGET else ('integral_limited' if Q0 > Q_BUDGET else 'both_limits')
    elif regime == 'fixed_peak_1':
        target = 1.0; budget = None; lim = 'fixed_peak'
    elif regime == 'equal_integral_5850':
        target = Q_LOW / Q0; budget = Q_LOW; lim = 'equal_integral_low'
        if target > 1.0 + 1e-12:
            raise Stop('technical_stop', f'equal integral 5850 infeasible (Q0 {Q0})')
    else:
        raise Stop('usage', f'unknown regime {regime}')
    pf = np.float32(target)
    q_act = float((b.astype(np.float32) * pf).sum(dtype=np.float64))
    return dict(Q0=Q0, Q_budget=budget, peak_target=float(target), peak_f32=float(pf), Q_actual=q_act, budget_usage=(q_act / budget if budget else None), active_limit=lim)


def resource_check(r: dict, regime: str) -> list[str]:

    bad = []
    if r['peak_f32'] > 1.0:
        bad.append(f"peak {r['peak_f32']} > 1")
    if r['Q_budget'] is not None and r['Q_actual'] > r['Q_budget'] * (1 + 1e-6):
        bad.append(f"Q_actual {r['Q_actual']} > budget {r['Q_budget']}")
    if r['active_limit'] in ('integral_limited', 'equal_integral_low') and abs(r['Q_actual'] - r['Q_budget']) / r['Q_budget'] > 1e-6:
        bad.append(f"integral-limited Q_actual {r['Q_actual']} vs {r['Q_budget']}")
    if r['active_limit'] in ('peak_limited', 'fixed_peak') and abs(r['Q_actual'] - r['Q0']) > 1e-6 * r['Q0']:
        bad.append(f"peak-limited Q_actual {r['Q_actual']} vs Q0 {r['Q0']}")
    return bad


_PL: dict = {}


def s2():
    import s1r_common as S1K
    return S1K.s2()


def bound():
    import s1r_common as S1K
    return S1K.bound()


def platform_ids() -> dict:

    if not _PL:
        import s1r_common as S1K
        S = s2(); bd = bound()
        m3 = {k: v for k, v in S.m3_cfg(0.0, 0).items() if k not in ('noise_sigma', 'mottle_seed')}
        gd = load_json(S1OUT / 'domain' / 'global_domain_step0p50.json')
        _PL.update(version=bd.version, m3=digest(m3), m3_cfg=m3, window=[float(v) for v in bd.window()],
                   dot_cfg=digest({k: v for k, v in S.dot_cfg().items() if isinstance(v, (int, float, str, bool, list, tuple)) or v is None}),
                   p4=digest(S.P4), s1r_eval=sha_file(ADP / 's1r_eval.py'), global_domain=gd['hash'], global_nodes=int(gd['n_nodes']),
                   sensor=digest(S.geometry_record()), s1k=S1K)
    return _PL


def input_key(raster_key: str, peak_f32: float, platform_surface: str, sigma: float, seed) -> str:
    P = platform_ids()
    return 'S12I-' + digest(dict(raster=raster_key, peak_f32=float(np.float32(peak_f32)), surface=platform_surface, psf=PSF_PX, sigma=float(sigma),
                                 seed=(None if float(sigma) == 0 else int(seed)), m3=P['m3'], platform=P['version']), 24)


def decode_key(ikey: str, family: str, peak_f32: float) -> str:
    P = platform_ids()
    return 'S12D-' + digest(dict(img=ikey, arm=ARM[family], params=(P['dot_cfg'] if family == 'point' else P['p4']),
                                 pattern_peak_f32=(float(np.float32(peak_f32)) if family == 'line' else None), window=P['window']), 24)


def eval_key(dkey: str) -> str:
    P = platform_ids()
    return 'S12E-' + digest(dict(dec=dkey, s1r_eval=P['s1r_eval'], global_domain=P['global_domain'], d=D_SUPPORT, tau=TAU, gross=GROSS), 24)
