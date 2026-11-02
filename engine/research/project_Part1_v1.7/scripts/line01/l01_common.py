# Define continuous-line cases, imaging settings and result storage.

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs' / 'line01'
DOCS = ROOT / 'docs' / 'line01'
CASES = OUT / 'cases'
LEDGER_DIR = OUT / 'ledger'
TASK_ID = 'LINE-01'
TASK_FILE = Path(r'C:\Users\16944\Documents\reference\2026-09-21\dang\outputs\research_runs\tasks\LINE-01_protocol.md')

ADVICE_FILE = ROOT.parents[1] / 'protocols' / 'line_grid_plan.md'

for _p in (ROOT / 'src', ROOT / 'scripts' / 'spacing_layout', ROOT / 'scripts' / 'pm014', ROOT / 'scripts' / 'pm015',
           ROOT / 'scripts' / 'spotnoise', ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

IMG = 512


SURFACES = ('plane', 'tilt5', 'baseline', 'curv_x_strong')
PHIS = (0.0, 45.0)
WIDTHS = (0.5, 1.0, 1.5)
SIGMAS = (0.0, 20.0, 40.0)
PHASES = {'dev': (-3 / 8, -1 / 8, 1 / 8, 3 / 8), 'sealed': (-7 / 16, -3 / 16, 1 / 16, 5 / 16)}
S_MM = 8.0
PEAK = 0.5
PSF_PX = 1.0
VOLUME_MM = 20.0
NOISE_BASE = 10090000
TILT_SLOPE = math.tan(math.radians(5.0))
DENT_R_MM = 15.0
FOV_HALF_MM = 46.08
KEY_FIELDS = ('v', 'split', 'surface', 'phi_deg', 'w_mm', 's_mm', 'peak', 'sigma', 'phase_index', 'phase_frac', 'noise_seed', 'psf_px')
CAPS = {'main': 576, 'analytic': 32, 'debug': 32}
PURPOSE_GROUP = {'main_dev': 'main', 'main_sealed': 'main', 'smoke': 'analytic', 'analytic': 'analytic', 'throughput': 'analytic',
                 'regression_before': 'analytic', 'regression_after': 'analytic', 'debug': 'debug'}


def design() -> list[dict]:

    specs, i = [], 0
    for split in ('dev', 'sealed'):
        for surf in SURFACES:
            for phi in PHIS:
                for w in WIDTHS:
                    for sig in SIGMAS:
                        for pi, ph in enumerate(PHASES[split]):
                            specs.append(dict(v=1, split=split, surface=surf, phi_deg=float(phi), w_mm=float(w), s_mm=S_MM, peak=PEAK,
                                              sigma=float(sig), phase_index=pi, phase_frac=float(ph), noise_seed=NOISE_BASE + i, psf_px=PSF_PX,
                                              case_index=i))
                            i += 1
    for s in specs:
        s['case_id'] = key(s)
    return specs


def key(s: dict) -> str:
    return hashlib.sha256(json.dumps({k: s[k] for k in KEY_FIELDS}, sort_keys=True).encode('utf-8')).hexdigest()


def now_local() -> str:
    return datetime.datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def rel(p: Path) -> str:
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def clean(o):

    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _reject_constant(c):
    raise ValueError(f'non-finite JSON constant {c!r} is not allowed')


def save_json(p: Path, obj) -> Path:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')
    tmp.replace(p)
    return p


def load_json(p: Path):
    return json.loads(Path(p).read_text(encoding='utf-8'), parse_constant=_reject_constant)


def save_text(p: Path, text: str) -> Path:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8', newline='\n')
    return p


def save_csv(df, p: Path) -> Path:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding='utf-8', lineterminator='\n')
    return p


def f_or_none(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


PROGRESS = OUT / 'progress.json'


def progress(**kw) -> dict:
    if PROGRESS.exists():
        pr = load_json(PROGRESS)
    else:
        pr = dict(task_id=TASK_ID, scope='W0-W6 basic implementation and validation of oriented continuous lines (no formal L1/L2 scans)',
                  state='setup', started_local=now_local(), started_utc=now_utc(), stages={}, counts={}, blockers=[], pending_decisions=[],
                  executor='execution (local session)', selfcheck_status='not run', independent_review_status='not started (reference)')
    pr.update(kw)
    pr['updated_local'] = now_local(); pr['updated_utc'] = now_utc()
    save_json(PROGRESS, pr)
    return pr


def stage(name: str, **kw) -> dict:
    pr = progress()
    st = dict(pr.get('stages', {}))
    st[name] = {**st.get(name, {}), **clean(kw), 'updated_local': now_local()}
    return progress(stages=st)


def log(msg: str) -> None:
    line = f'[{now_local()}] {msg}'
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'run.log', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def ledger_add(rec: dict) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_DIR / f'ledger_{os.getpid()}.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(rec), ensure_ascii=False) + '\n')


def ledger_records() -> list[dict]:
    recs = []
    if LEDGER_DIR.exists():
        for p in sorted(LEDGER_DIR.glob('ledger_*.jsonl')):
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    recs.append(json.loads(line))
    return sorted(recs, key=lambda r: r.get('t_local', ''))


def ledger_counts() -> dict:
    recs = ledger_records()
    out = {g: dict(attempts=0, ok=0, failed=0, unique_images=0, cap=CAPS[g]) for g in CAPS}
    seen = {g: set() for g in CAPS}
    by_purpose: dict = {}
    for r in recs:
        g = PURPOSE_GROUP.get(r.get('purpose'), 'debug')
        out[g]['attempts'] += 1
        out[g]['ok' if r.get('ok') else 'failed'] += 1
        if r.get('ok') and r.get('image_sha256'):
            seen[g].add(r['image_sha256'])
        bp = by_purpose.setdefault(r.get('purpose'), dict(attempts=0, ok=0))
        bp['attempts'] += 1; bp['ok'] += int(bool(r.get('ok')))
    for g in CAPS:
        out[g]['unique_images'] = len(seen[g])
    return dict(groups=out, by_purpose=by_purpose, total_attempts=len(recs), total_ok=sum(1 for r in recs if r.get('ok')))


def budget_check(purpose: str, n: int = 1) -> None:
    g = PURPOSE_GROUP[purpose]
    c = ledger_counts()['groups'][g]
    if c['attempts'] + n > CAPS[g]:
        raise RuntimeError(f'imaging budget of group {g} exhausted: {c["attempts"]} attempts + {n} > cap {CAPS[g]} (purpose {purpose})')


def render(mesh: dict, pattern: np.ndarray, m3_cfg: dict, purpose: str, case_id: str, extra: dict | None = None):

    from m3_virtual_camera import synthesise_camera_image
    budget_check(purpose)
    rec = dict(t_local=now_local(), purpose=purpose, case_id=case_id, ok=False, seconds=None, image_sha256=None, error=None, pid=os.getpid(),
               **(extra or {}))
    t0 = time.monotonic()
    try:
        img, hit, pts3d, info = synthesise_camera_image(mesh_dict=mesh, pattern=pattern, cfg=m3_cfg)
    except Exception as e:
        rec.update(seconds=round(time.monotonic() - t0, 3), error=repr(e)[:300])
        ledger_add(rec)
        raise
    rec.update(ok=True, seconds=round(time.monotonic() - t0, 3), image_sha256=sha_bytes(img.tobytes()), n_rays=int(info['n_rays_cast']),
               n_hits=int(info['n_hits']), n_valid_pixels=int(info['n_valid_pixels']))
    ledger_add(rec)
    return img, info


def m3_cfg(sigma: float, seed: int) -> dict:

    return {'batch_size': 512, 'noise_sigma': float(sigma), 'psf_sigma_px': PSF_PX, 'albedo': 1.0, 'albedo_exposure': 1.0, 'specular_k': 0.0,
            'mottle_amp': 0.0, 'outlier_frac': 0.0, 'mottle_seed': int(seed)}


def m4_cfg() -> dict:
    import m4_reconstruction as M4
    return M4._resolve_sensor_cfg({'measurement_volume_mm': VOLUME_MM})


_SURF: dict = {}


def surface(name: str):

    if name in _SURF:
        return _SURF[name]
    import sl_lib as L
    import pm014_surface as S14
    if name in ('baseline', 'curv_x_strong'):
        mesh, cfg, ref = L.surface(name)
        zmin = float(np.asarray(mesh['Z']).min())

        def ref_fn(x, y, _ref=ref, _cfg=cfg, _z=zmin):
            return np.asarray(S14.z_ref_at(_ref, _cfg, x, y), np.float64) - _z
        desc = dict(kind='pm014_fixed_reference', name=name, fwhm_mm=20.0, depth_mm=5.0, curv_x_1_per_mm=float(cfg.get('curv_x_1_per_mm', 0.0) or 0.0),
                    z_shift_m3_mm=-zmin, nodes_sha256=cfg['pm014_reference']['nodes_sha256'])
    elif name in ('plane', 'tilt5'):
        mesh0, cfg0, _ = L.surface('baseline')
        X = np.asarray(mesh0['X'], np.float32); Y = np.asarray(mesh0['Y'], np.float32)
        slope = TILT_SLOPE if name == 'tilt5' else 0.0
        Z = (slope * Y.astype(np.float64)).astype(np.float32)
        mesh = {'X': X, 'Y': Y, 'Z': Z, 'Z_base': Z.copy()}
        zmin = float(Z.min())
        cfg = dict(fov_half_width_mm=float(cfg0['fov_half_width_mm']), grid_step_x_mm=float(cfg0['grid_step_x_mm']), grid_step_y_mm=float(cfg0['grid_step_y_mm']),
                   grid_shape_ny_nx=list(Z.shape), has_indentation=False, indentation=None, indentation2=None, double_indentation=None, has_curvature=False,
                   curvature=None, variant='LINE-01 synthetic plane', line01_surface=name, slope_dz_dy=slope, z_shift_m3_mm=-zmin,
                   depth_max_mm=float(Z.max() - zmin))

        def ref_fn(x, y, _s=slope, _z=zmin):
            return _s * np.asarray(y, np.float64) - _z
        desc = dict(kind='synthetic_plane', name=name, slope_dz_dy=slope, tilt_deg=5.0 if slope else 0.0, z_shift_m3_mm=-zmin,
                    note='z = slope * y on the PM-014 grid; M3 aligns the grid minimum to z = 0 (grid_to_triangles / heightfield_intersect rule)')
    else:
        raise ValueError(name)
    _SURF[name] = (mesh, cfg, ref_fn, desc)
    return _SURF[name]


def grid_truth(mesh: dict):

    from scipy.interpolate import RegularGridInterpolator
    Z = np.asarray(mesh['Z'], np.float64)
    f = RegularGridInterpolator((np.asarray(mesh['Y'][:, 0], np.float64), np.asarray(mesh['X'][0, :], np.float64)), Z - Z.min(), method='linear',
                                bounds_error=False, fill_value=np.nan)
    return f, float(Z.min())


_GEOM: dict = {}


def geom() -> dict:

    if not _GEOM:
        from m3_virtual_camera import SENSOR_DEFAULTS, build_intrinsics, build_extrinsics
        from m2_pattern import _compute_fov_mm, FOCAL_LENGTH_MM, PIXEL_PITCH_MM, WORKING_DIST_MM
        cfg = dict(SENSOR_DEFAULTS)
        K = build_intrinsics(cfg); proj, cam = build_extrinsics(cfg)
        fov_deg, fov_mm, mmpp = _compute_fov_mm(IMG, FOCAL_LENGTH_MM, PIXEL_PITCH_MM, WORKING_DIST_MM)
        theta = 2.0 * math.atan(cfg['baseline_mm'] / 2.0 / cfg['working_dist_mm'])
        _GEOM.update(cfg=cfg, K=K, proj=proj, cam=cam, mm_per_px=float(mmpp), fov_mm=float(fov_mm), fov_deg=float(fov_deg), theta_rad=float(theta),
                     theta_deg=float(math.degrees(theta)), f_px=float(K[0, 0]))
    return _GEOM


def rays(uv: np.ndarray, ext: dict):
    g = geom(); K = g['K']
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    d = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))], axis=1)
    return np.tile(ext['center'], (len(uv), 1)), (ext['R_inv'] @ d.T).T


def cast(uv: np.ndarray, ext: dict, mesh: dict):

    from m3_virtual_camera import heightfield_intersect
    o, d = rays(uv, ext)
    if len(o) == 0:
        return np.zeros(0, bool), np.zeros((0, 3))
    hit, pts, _, _ = heightfield_intersect(o, d, mesh)
    return np.asarray(hit, bool), np.asarray(pts, np.float64)


def project(points: np.ndarray, ext: dict):
    from m3_virtual_camera import project_to_camera
    g = geom()
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    uv, inside = project_to_camera(pts, g['K'], ext, IMG, IMG)
    return np.asarray(uv, np.float64), np.asarray(inside, bool)


_OMEGA: dict = {}


def omega(name: str) -> dict:

    if name in _OMEGA:
        return _OMEGA[name]
    mesh, cfg, ref_fn, desc = surface(name)
    g = geom()
    X = np.asarray(mesh['X'], np.float64); Y = np.asarray(mesh['Y'], np.float64); Z = np.asarray(mesh['Z'], np.float64); Za = Z - Z.min()

    pre = (np.abs(X) <= 70.0) & (np.abs(Y) <= 70.0)
    idx = np.flatnonzero(pre.ravel())
    P = np.column_stack([X.ravel()[idx], Y.ravel()[idx], Za.ravel()[idx]])
    uv_p, in_p = project(P, g['proj'])
    uv_c, in_c = project(P, g['cam'])
    keep = in_p & in_c
    sel = idx[keep]
    x, y, z = X.ravel()[sel], Y.ravel()[sel], Za.ravel()[sel]
    dx = float(cfg['grid_step_x_mm']); dy = float(cfg['grid_step_y_mm'])
    out = dict(name=name, idx=sel, x=x, y=y, z_ref=z, area=np.full(len(sel), dx * dy), dx=dx, dy=dy,
               in_dent=(x ** 2 + y ** 2) <= DENT_R_MM ** 2, in_fov_legacy=(np.abs(x) <= FOV_HALF_MM) & (np.abs(y) <= FOV_HALF_MM),
               uv_proj=uv_p[keep], uv_cam=uv_c[keep], n_nodes=int(len(sel)), area_total_mm2=float(len(sel) * dx * dy),
               x_range=(float(x.min()), float(x.max())), y_range=(float(y.min()), float(y.max())), z_range=(float(z.min()), float(z.max())))
    out['hash'] = sha_bytes(sel.astype(np.int64).tobytes() + z.astype(np.float64).tobytes())
    _OMEGA[name] = out
    return out


def source_versions(paths: list[Path]) -> dict:
    return {rel(p): dict(sha256=sha_file(p), bytes=p.stat().st_size, mtime_local=datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().strftime('%Y-%m-%dT%H:%M:%S%z'))
            for p in paths if p.exists()}


def env_record() -> dict:
    import platform
    import numpy, scipy, cv2, pandas, matplotlib
    return dict(python=sys.version.split()[0], platform=platform.platform(), numpy=numpy.__version__, scipy=scipy.__version__, opencv=cv2.__version__,
                pandas=pandas.__version__, matplotlib=matplotlib.__version__, cpu_count=os.cpu_count())
