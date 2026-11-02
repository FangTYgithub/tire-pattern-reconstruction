# Build size-study rasters and load evaluation domains and platform settings.

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import adp_common as AC
import adp_patterns as PT

PT.setup_paths()
ROOT = AC.ROOT
OUT = ROOT / 'outputs' / 'surface_revision' / 's1_real01'
REL_DIR = ROOT / 'outputs' / 'surface_revision' / 'surf02' / 'release'
RELEASE = REL_DIR / 'release_surf02.json'
PLATFORM_DEPS = REL_DIR / 'dependencies_surf02_v1_1.json'
CONFIG_JSON = OUT / 'config' / 's1_config.json'
S1_DEPS = OUT / 'config' / 's1_dependencies.json'
VIEWS_CSV = OUT / 'manifest' / 's1_views.csv'
IMAGES_CSV = OUT / 'manifest' / 's1_images.csv'
RASTER_DIR = OUT / 'rasters'
IMG_DIR = OUT / 'images'
DEC_DIR = OUT / 'decodes'
DOMAIN_DIR = OUT / 'domain'
LEDGER_DIR = OUT
TASK_FILE = ROOT.parents[1] / 'protocols' / 'S1-REAL-01_protocol.md'
TASK_ID = 'S1-REAL-01'

Q_DOT, Q_LINE = 5850.0, 11695.0
DOT_RADII = (0.5, 1.0, 1.5)
LINE_WIDTHS = (0.5, 0.75, 1.0, 1.25, 1.5)
DOT_SIGMAS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
LINE_SIGMAS = (0.0, 20.0, 40.0)
REGIMES = ('fixed_peak', 'equal_integral')
SURFACES = AC.SURFACE_IDS
MAIN = ('M10_curv', 'M24_curv', 'B_curv')
GAUSS = 'M10_curv_gauss'
ARM = {'point': 'D0', 'line': 'P4_C3_cal'}
GLOBAL_STEP, GLOBAL_STEP_FINE = 0.5, 0.25
GLOBAL_HALF_MM = 46.0
VIS_TOL_MM = 1e-4
D_SUPPORT, TAU = 6.0, 0.15
PSF_PX = 1.0
DE_SURF = {k: AC.SURFACES[k]['de'] for k in SURFACES}
SHORT = {'M10_curv': 'M10', 'M24_curv': 'M24', 'B_curv': 'B', 'M10_curv_gauss': 'M10+Gauß'}
FAMILY_DE = {'point': 'Punkte', 'line': 'Linien'}
REGIME_DE = {'fixed_peak': 'fester Peak 1', 'equal_integral': 'gleiches Integral'}
for d in (OUT / 'config', OUT / 'manifest', RASTER_DIR, IMG_DIR, DEC_DIR, DOMAIN_DIR, OUT / 'checks', OUT / 'tables', OUT / 'figures'):
    d.mkdir(parents=True, exist_ok=True)

clean, digest, save_json, load_json, sha_file, rel, now_local = AC.clean, AC.digest, AC.save_json, AC.load_json, AC.sha_file, AC.rel, AC.now_local
Stop = AC.AdapterStop


def sha_bytes(b: bytes, n: int | None = None) -> str:
    h = hashlib.sha256(b).hexdigest()
    return h[:n] if n else h


def assert_out(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in (q, *q.parents) and REL_DIR.resolve() not in (q, *q.parents):
        raise Stop('technical_stop', f'write outside {rel(OUT)} / {rel(REL_DIR)} refused: {q}')
    return q


def save_npz(p: Path, **arrays) -> str:
    p = assert_out(p); tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    for _ in range(50):
        try:
            os.replace(tmp, p); break
        except PermissionError:
            time.sleep(0.1)
    return sha_file(p)


def complete(json_path: Path) -> bool:

    if not json_path.exists():
        return False
    try:
        r = load_json(json_path)
    except Exception:
        return False
    npz = json_path.with_suffix('.npz')
    return r.get('complete') is True and npz.exists() and sha_file(npz) == r.get('npz_sha256')


_S2 = None
_BD = None


def s2():
    global _S2
    if _S2 is None:
        import adp_surf02 as SA
        _S2 = SA.s2()
    return _S2


def bound():

    global _BD
    if _BD is None:
        import adp_binding as BD
        _BD = BD.bind(required_methods=sorted(set(ARM.values())))
    return _BD


LAZY_MODULES = ('m2_pattern', 'm3_virtual_camera', 'm4_reconstruction', 'm4_lines', 'surface_eval_r3', 'cv2', 'scipy.interpolate', 'scipy.spatial', 'scipy.ndimage')


def collect_dependencies(adapter: bool) -> dict:

    import importlib
    s2()
    for m in LAZY_MODULES:
        importlib.import_module(m)
    if adapter:
        for m in ('adp_binding', 'adp_surf02', 's1r_eval'):
            importlib.import_module(m)
    files = {}
    sa = (ROOT / 'scripts' / 'surface_revision' / 'sections_adapter').resolve()
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if not f or not str(f).endswith('.py'):
            continue
        q = Path(f).resolve()
        if ROOT.resolve() not in q.parents:
            continue
        if not adapter and sa in q.parents:
            continue
        if adapter and sa in q.parents and q.name not in ('adp_common.py', 'adp_binding.py', 'adp_surf02.py', 'adp_patterns.py', 's1r_common.py', 's1r_eval.py'):
            continue
        files[rel(q)] = sha_file(q)
    import m1_tire_surface as M1
    for v in ('A', 'B', 'C'):
        q = ROOT / 'data' / 'raw' / M1._CSV_NAMES[v]
        files[rel(q)] = sha_file(q)
    files[rel(s2().P4_FILE)] = sha_file(s2().P4_FILE)
    return dict(sorted(files.items()))


def drift(manifest: dict) -> list[str]:
    bad = []
    for rp, h in manifest['files'].items():
        q = AC.absp(rp)
        if not q.exists():
            bad.append(f'{rp}: missing')
        elif sha_file(q) != h:
            bad.append(f'{rp}: sha256 differs')
    return bad


def check_s1_dependencies() -> dict:

    if not S1_DEPS.exists():
        raise Stop('platform_mismatch', f'{rel(S1_DEPS)} missing: run `s1r_run.py release` first')
    m = load_json(S1_DEPS); bad = drift(m)
    if bad:
        raise Stop('platform_mismatch', f'S1 numerical dependencies changed ({len(bad)}): {bad[:6]} -> stored results may not be reused')
    return m


def raster_key(binary: np.ndarray) -> str:
    b = np.ascontiguousarray(np.asarray(binary) > 0, dtype=np.uint8)
    return 'R-' + sha_bytes(str(b.shape).encode() + b.tobytes(), 20)


def build_raster(family: str, size_mm: float, phase_label: str, phase_frac=None) -> dict:

    if family == 'point':
        row = dict(generator='sn_lib.S1_lattice_8x4', phase_label=phase_label, radius_mm=size_mm, brightness_rule='v=1', cand_id='raster', section='S1')
    else:
        row = dict(generator='m2.olines', line_pitch_mm=8.0, line_width_mm=size_mm, orientation_deg=0.0, phase_frac=float(phase_frac), brightness_rule='peak=1.0', cand_id='raster',
                   section='S1', raster_sha256_registered='')
    b = PT.build(row)
    binary = (np.asarray(b['binary']) > 0)
    out = dict(binary=binary, centres=b['centres'], meta_full=b['meta_full'], key=raster_key(binary), integral=float(binary.sum()),
               n_features=(int(len(b['centres'])) if family == 'point' else int(b['meta_full']['olines']['n_lines'])))
    return out


def save_raster(r: dict, family: str, size_mm: float, phase_label: str, phase_frac) -> Path:
    p = RASTER_DIR / f"{r['key']}.json"
    if complete(p):
        return p
    arrays = dict(binary=r['binary'].astype(np.uint8))
    if r['centres'] is not None:
        arrays['centres'] = np.asarray(r['centres'], np.float64)
    h = save_npz(p.with_suffix('.npz'), **arrays)
    save_json(assert_out(p), dict(raster_key=r['key'], family=family, size_mm=size_mm, phase_label=phase_label, phase_frac=phase_frac, integral_I0=r['integral'], n_features=r['n_features'],
                                  generator=('sn_lib.centres(S1, phase) + sn_lib.binary_pattern(c, r)' if family == 'point' else 'l01rb_common.pattern_for (m2_pattern olines)'),
                                  meta_olines=(clean(r['meta_full']['olines']) if r['meta_full'] else None), npz_sha256=h, complete=True))
    return p


_RCACHE: dict = {}


def load_raster(key: str) -> dict:

    if key in _RCACHE:
        return _RCACHE[key]
    p = RASTER_DIR / f'{key}.json'
    if not complete(p):
        raise Stop('technical_stop', f'raster {key} missing or corrupt')
    rec = load_json(p)
    with np.load(p.with_suffix('.npz')) as z:
        binary = np.asarray(z['binary']) > 0
        centres = np.asarray(z['centres'], np.float64) if 'centres' in z.files else None
    if raster_key(binary) != key:
        raise Stop('technical_stop', f'raster {key}: content hash differs')
    meta_full = None
    if rec['family'] == 'line':
        r = build_raster('line', rec['size_mm'], rec['phase_label'], rec['phase_frac'])
        if r['key'] != key:
            raise Stop('technical_stop', f'line raster {key}: regenerated raster differs')
        meta_full = r['meta_full']
    out = dict(rec=rec, binary=binary.astype(np.float32), centres=centres, meta_full=meta_full)
    if len(_RCACHE) > 64:
        _RCACHE.clear()
    _RCACHE[key] = out
    return out


def pattern_of(binary_f32: np.ndarray, peak: float) -> np.ndarray:
    return (np.asarray(binary_f32, np.float32) * np.float32(peak)).astype(np.float32)


def lattice(step: float):
    n = int(round(2 * GLOBAL_HALF_MM / step))
    x = -GLOBAL_HALF_MM + step * np.arange(n + 1)
    return x, x.copy()


_DOM: dict = {}


def domain_path(step: float) -> Path:
    return DOMAIN_DIR / ('global_domain_step' + f'{step:.2f}'.replace('.', 'p') + '.json')


def global_domain(step: float = GLOBAL_STEP) -> dict:

    key = float(step)
    if key not in _DOM:
        p = domain_path(step)
        if not complete(p):
            raise Stop('technical_stop', f'global domain {rel(p)} missing: run `s1r_run.py domain`')
        rec = load_json(p)
        with np.load(p.with_suffix('.npz')) as z:
            d = {k: np.asarray(z[k]) for k in z.files}
        d.update(rec=rec, step=float(rec['step_mm']), x0=float(rec['x0_mm']), y0=float(rec['y0_mm']), nx=int(rec['nx']), ny=int(rec['ny']), hash=rec['hash'])
        _DOM[key] = d
    return _DOM[key]


def in_global(xy: np.ndarray, dom: dict) -> np.ndarray:

    xy = np.asarray(xy, np.float64).reshape(-1, 2); out = np.zeros(len(xy), bool)
    fin = np.isfinite(xy).all(axis=1)
    if fin.any():
        ix = np.rint((xy[fin, 0] - dom['x0']) / dom['step']).astype(np.int64); iy = np.rint((xy[fin, 1] - dom['y0']) / dom['step']).astype(np.int64)
        ok = (ix >= 0) & (ix < dom['nx']) & (iy >= 0) & (iy < dom['ny'])
        res = np.zeros(int(fin.sum()), bool)
        res[ok] = dom['mask'][iy[ok], ix[ok]]
        out[fin] = res
    return out


def in_local(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    with np.errstate(invalid='ignore'):
        return np.isfinite(xy).all(axis=1) & (np.hypot(xy[:, 0], xy[:, 1]) <= float(s2().ROI_R_MM))


def phase_index(label: str) -> int:
    return int(str(label).replace('sn_phase', '').replace('ph', ''))
