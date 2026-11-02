# Provide common paths, surface identifiers and serialization for study adapters.

from __future__ import annotations

import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MASTER = ROOT.parents[1]
OUT = ROOT / 'outputs' / 'surface_revision' / 'sections_adapter'
BINDING_JSON = OUT / 'binding.json'
CANDIDATES_CSV = OUT / 'candidates_s1s5.csv'
SUMMARY_JSON = OUT / 'matrix_summary.json'
CHECKS_JSON = OUT / 'checks' / 'checks_latest.json'
CASES_ROOT = OUT / 'cases'
SURF01_REGISTRY = ROOT / 'outputs' / 'surface_revision' / 'surf01' / 'surface_registry.json'


EXIT = dict(ok=0, error=2, usage=3, platform_unbound=10, platform_unreleased=11, platform_mismatch=12, capability_missing=13,
            run_not_authorised=14, matrix_drift=15, technical_stop=16, check_failed=17, legacy_geometry_refused=18)


class AdapterStop(RuntimeError):


    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}')
        self.code = code
        self.exit_code = EXIT[code]


SURFACES = {

    'M10_curv':       dict(order=0, de='M10 + Zusatzkrümmung', zh='M10＋修正附加弯曲', surf01='A_curv', gauss=False, report='main'),
    'M24_curv':       dict(order=1, de='M24 + Zusatzkrümmung', zh='M24＋修正附加弯曲', surf01='C_curv', gauss=False, report='main'),
    'B_curv':         dict(order=2, de='Ebene Fahrbahn B (Reifenkontur) + Zusatzkrümmung', zh='平地B＋修正附加弯曲', surf01='B_curv', gauss=False, report='main'),
    'M10_curv_gauss': dict(order=3, de='M10 + Zusatzkrümmung + Gauß-Erhebung (5 mm)', zh='M10＋附加弯曲＋向轮内5 mm高斯隆起', surf01='A_curv_gauss', gauss=True,
                           report='mechanism (local reference bulge, d 5 mm, FWHM 20 mm)'),
}
SURFACE_IDS = tuple(sorted(SURFACES, key=lambda k: SURFACES[k]['order']))
LEGACY_SURFACES = ('baseline', 'curv_x_strong', 'curv_x', 'curv_y', 'flat', 'A', 'B', 'C')


EXPECTED_CONVENTIONS = dict(
    mapping='x = -x_p, y = y_p, z = c - H (H positive toward the tyre interior)',
    c_reference='c = H_B,base(0, 0) of flat-ground tyre B, read once by the platform, shared by all four surfaces',
    c_approx_mm=16.64546548,
    camera_C=(0.0, 25.0, -150.0), projector_C=(0.0, -25.0, -150.0),
    working_distance_note='150 mm is the nominal project parameter, not a verified Mappes installation distance',
    per_case_zmin_shift=False,
    search_window_candidate_mm=(-16.0, 4.0),
    negative_z_valid=True,
)


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
    if isinstance(o, (list, tuple)):
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


def save_json(p, obj) -> None:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
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


def absp(p) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def assert_inside_out(p) -> Path:

    q = Path(p).resolve()
    if OUT.resolve() not in (q, *q.parents):
        raise AdapterStop('technical_stop', f'write outside the adapter output directory refused: {q}')
    return q
