# Construct a continuous reference profile and sample it on configurable surface grids.

import sys, hashlib, json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline, RegularGridInterpolator
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import m1_tire_surface as M1

H_REF = 0.0625
L_DEFAULT = 20.0
POLY = 3
BIN_MM = 1.0
_REF = {}
_TEMPLATE = {}


def reference_profile(variant='A', L_mm=L_DEFAULT, h_ref=H_REF):

    k = (variant, float(L_mm), float(h_ref))
    if k in _REF:
        return _REF[k]
    csv = ROOT / 'data' / 'raw' / M1._CSV_NAMES[variant]
    raw = M1.load_raw_csv(csv)
    env = M1.extract_upper_envelope(raw, BIN_MM)
    y_src, z_src = env['y_mm'].to_numpy(), env['z_upper_raw_mm'].to_numpy()
    y_ref = np.arange(y_src.min(), y_src.max() + h_ref * 0.5, h_ref)
    z_int = PchipInterpolator(y_src, z_src)(y_ref)
    w = int(round(L_mm / h_ref)) + 1
    if w % 2 == 0:
        w += 1
    z_s = savgol_filter(z_int, window_length=w, polyorder=POLY)
    ref = dict(variant=variant, L_mm=float(L_mm), h_ref=float(h_ref), window_points=int(w), span_mm=float((w - 1) * h_ref), polyorder=POLY,
               y_ref=y_ref, z_ref=z_s, spline=CubicSpline(y_ref, z_s), y_range=(float(y_src.min()), float(y_src.max())),
               envelope_points=int(len(y_src)), nodes_sha256=hashlib.sha256(np.ascontiguousarray(z_s).tobytes()).hexdigest())
    _REF[k] = ref
    return ref


def dip_analytic(x, y, depth, fwhm, center=(0.0, 0.0), aspect=1.0, asym=1.0, asym_angle_deg=0.0):

    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    sx = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sy = aspect * sx
    phi = np.radians(asym_angle_deg)
    u = (x - center[0]) * np.cos(phi) + (y - center[1]) * np.sin(phi)
    v = -(x - center[0]) * np.sin(phi) + (y - center[1]) * np.cos(phi)
    su = np.where(u >= 0, asym * sx, sx)
    r2 = (u / su) ** 2 + (v / sy) ** 2
    return depth * np.exp(-0.5 * r2)


def template(variant, surf_params, grid_xy, base_params):

    k = (variant, json.dumps(surf_params, sort_keys=True), tuple(grid_xy), json.dumps(base_params, sort_keys=True))
    if k not in _TEMPLATE:
        while len(_TEMPLATE) >= 2:
            _TEMPLATE.pop(next(iter(_TEMPLATE)))
        _TEMPLATE[k] = M1.build_surface(variant, raw_data_dir=ROOT / 'data' / 'raw', grid_step_x_mm=grid_xy[0], grid_step_y_mm=grid_xy[1],
                                        **base_params, **surf_params)
    return _TEMPLATE[k]


def z_ref_at(ref, params, x, y):

    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    kx = float(params.get('curv_x_1_per_mm', 0.0) or 0.0); ky = float(params.get('curv_y_extra_1_per_mm', 0.0) or 0.0)
    x0 = float(params.get('curv_ref_x_mm', 0.0) or 0.0); y0 = float(params.get('curv_ref_y_mm', 0.0) or 0.0)
    base = ref['spline'](y) + 0.5 * kx * (x - x0) ** 2 + 0.5 * ky * (y - y0) ** 2
    d = float(params.get('indent_depth_mm', 0.0) or 0.0)
    if d > 0:
        base = base - dip_analytic(x, y, d, float(params['indent_width_mm']), (float(params.get('indent_center_x_mm', 0.0)), float(params.get('indent_center_y_mm', 0.0))),
                                   float(params.get('indent_aspect', 1.0)), float(params.get('indent_asym', 1.0)), float(params.get('indent_asym_angle_deg', 0.0)))
    return base


def build_fixed_surface(variant, surf_params, grid_xy, base_params, L_mm=L_DEFAULT):

    mesh0, _, cfg0 = template(variant, surf_params, grid_xy, base_params)
    ref = reference_profile(variant, L_mm)
    X, Y = np.asarray(mesh0['X'], np.float64), np.asarray(mesh0['Y'], np.float64)
    params = {**cfg0}
    kx = float(params.get('curv_x_1_per_mm', 0.0) or 0.0); ky = float(params.get('curv_y_extra_1_per_mm', 0.0) or 0.0)
    x0 = float(params.get('curv_ref_x_mm', 0.0) or 0.0); y0 = float(params.get('curv_ref_y_mm', 0.0) or 0.0)
    Zb = ref['spline'](Y[:, 0])[:, None] + np.zeros_like(X) + 0.5 * kx * (X - x0) ** 2 + 0.5 * ky * (Y - y0) ** 2
    d = float(params.get('indent_depth_mm', 0.0) or 0.0)
    gt, mask = None, None
    Z = Zb.copy()
    if d > 0:
        dip, gt = M1.make_indentation(X.astype(np.float32), Y.astype(np.float32), d, params['indent_width_mm'],
                                      center_xy_mm=(params['indent_center_x_mm'], params['indent_center_y_mm']), aspect=params.get('indent_aspect', 1.0),
                                      shape=params.get('indent_shape', 'gaussian'), order=params.get('indent_order', 2.0),
                                      area_threshold_frac=params.get('indent_area_threshold_frac', 0.10), grid_step_x_mm=grid_xy[0], grid_step_y_mm=grid_xy[1],
                                      fov_half_width_mm=cfg0['fov_half_width_mm'], asym=params.get('indent_asym', 1.0), asym_angle_deg=params.get('indent_asym_angle_deg', 0.0))
        mask = gt.pop('contour_mask')
        Z = Zb - dip.astype(np.float64)
    Z32 = Z.astype(np.float32); Zb32 = Zb.astype(np.float32)
    Zr = Z32 - Z32.min()
    mesh = {'X': np.asarray(mesh0['X'], np.float32), 'Y': np.asarray(mesh0['Y'], np.float32), 'Z': Z32, 'Z_base': Zb32}
    if mask is not None:
        mesh['indent_mask'] = mask
    cfg = dict(cfg0)
    cfg.update(z_abs_min_mm=float(Z32.min()), z_abs_max_mm=float(Z32.max()), depth_max_mm=float(Zr.max()),
               fov_z_range_mm=M1.fov_z_range_mm(mesh['Y'], Zr, cfg0['fov_half_width_mm']), indentation=gt, has_indentation=gt is not None,
               pm014_reference=dict(model='fixed', L_mm=ref['L_mm'], h_ref=ref['h_ref'], window_points=ref['window_points'], span_mm=ref['span_mm'],
                                    polyorder=ref['polyorder'], nodes_sha256=ref['nodes_sha256'], envelope_points=ref['envelope_points'],
                                    note='production contact_features in this config come from the grid-dependent M1 profile and are not used'))
    return mesh, cfg, ref


def grid_interp(mesh):

    Z = np.asarray(mesh['Z'], np.float64); y = np.asarray(mesh['Y'][:, 0], np.float64); x = np.asarray(mesh['X'][0, :], np.float64)
    D = Z - Z.min()
    return RegularGridInterpolator((y, x), D, method='linear', bounds_error=False, fill_value=np.nan), float(Z.min())


def probe_points(n=4000, half=40.0, seed=20261003):
    r = np.random.default_rng(seed).random((n, 2)) * 2 * half - half
    return r[:, 0], r[:, 1]
