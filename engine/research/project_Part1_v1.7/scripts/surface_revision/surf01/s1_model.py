# Build the physical tire surfaces and map them into the imaging coordinate frame.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts' / 'pm014'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import config as CFG
import m1_tire_surface as M1
import pm014_surface as S14

KAPPA_H = +0.0025
R_EQ_MM = 1.0 / KAPPA_H
WD = float(CFG.M3_WORKING_DIST_MM)
BASELINE = float(CFG.M3_BASELINE_MM)
FOV_HALF = float(M1.sensor_fov_half_width_mm())
VARIANTS = {'A': dict(label='M10 (Guthrie 2017, Fig. 14)', de='A: M10-Mutter (Fig. 14)'), 'B': dict(label='flat ground (Fig. 9)', de='B: ebene Fahrbahn (Fig. 9)'),
            'C': dict(label='M24 (Fig. 13)', de='C: M24-Mutter (Fig. 13)')}
GAUSS = dict(d_mm=5.0, fwhm_mm=20.0, center=(0.0, 0.0), aspect=1.0)
R_P_TO_C = np.diag([-1.0, 1.0, -1.0])


def source_profile(variant: str) -> dict:

    ref = S14.reference_profile(variant)
    return dict(variant=variant, csv=M1._CSV_NAMES[variant], spline=ref['spline'], y_range=tuple(ref['y_range']), nodes_sha256=ref['nodes_sha256'],
                envelope_points=ref['envelope_points'], L_mm=ref['L_mm'], h_ref=ref['h_ref'], window_points=ref['window_points'],
                preprocessing='upper envelope (max H per 1 mm bin of the raw digitised points, paper orientation) -> PCHIP 0.0625 mm -> Savitzky-Golay span 20 mm, order 3 -> cubic spline')


def g_of(prof: dict, y_p):
    return prof['spline'](np.asarray(y_p, np.float64))


def h_curv(x_p, kappa_H=KAPPA_H, x0=0.0):
    return 0.5 * kappa_H * (np.asarray(x_p, np.float64) - x0) ** 2


def h_gauss(x_p, y_p, d_mm=GAUSS['d_mm'], fwhm_mm=GAUSS['fwhm_mm'], center=GAUSS['center'], aspect=GAUSS['aspect']):

    x = np.asarray(x_p, np.float64); y = np.asarray(y_p, np.float64)
    sx = fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0))); sy = aspect * sx
    return d_mm * np.exp(-0.5 * (((x - center[0]) / sx) ** 2 + ((y - center[1]) / sy) ** 2))


def h_base(prof: dict, x_p, y_p, kappa_H=KAPPA_H, x0=0.0):
    x = np.asarray(x_p, np.float64); y = np.asarray(y_p, np.float64)
    return g_of(prof, y) + h_curv(x, kappa_H, x0) + 0.0 * x


# Combine the measured profile with the configured curvature and optional Gaussian deformation.
def h_total(prof: dict, x_p, y_p, kappa_H=KAPPA_H, gauss: bool = False, x0=0.0, **gk):
    H = h_base(prof, x_p, y_p, kappa_H, x0)
    return H + (h_gauss(x_p, y_p, **gk) if gauss else 0.0)


# Transform physical surface coordinates into the imaging coordinate frame.
def p_to_c(P, c=0.0):

    P = np.asarray(P, np.float64).reshape(-1, 3)
    return P @ R_P_TO_C.T + np.array([0.0, 0.0, c])


# Transform imaging coordinates back into the physical surface frame.
def c_to_p(Cpts, c=0.0):
    Cpts = np.asarray(Cpts, np.float64).reshape(-1, 3)
    return (Cpts - np.array([0.0, 0.0, c])) @ R_P_TO_C


def sensors_in_P(c=0.0):
    cam_C = np.array([0.0, +BASELINE / 2, -WD]); prj_C = np.array([0.0, -BASELINE / 2, -WD])
    return dict(camera_C=cam_C, projector_C=prj_C, camera_P=c_to_p(cam_C, c)[0], projector_P=c_to_p(prj_C, c)[0],
                view_direction_C=np.array([0.0, 0.0, 1.0]), view_direction_P=(R_P_TO_C.T @ np.array([0.0, 0.0, 1.0])))


GRIDS = {'coarse': (2.0, 1.0), 'fine': (0.5, 0.25)}


def grid(prof: dict, name: str, x_half=100.0):
    dx, dy = GRIDS[name]
    x = np.arange(-x_half, x_half + dx * 0.5, dx)
    y0, y1 = prof['y_range']
    y = np.arange(y0, y1 + dy * 0.5, dy)
    X, Y = np.meshgrid(x, y)
    return X, Y


# Construct the physical surface and transform its grid into the imaging frame.
def build(prof: dict, grid_name: str, kappa_H: float, gauss: bool) -> dict:
    X, Y = grid(prof, grid_name)
    Hg = g_of(prof, Y); Hc = h_curv(X, kappa_H); Hd = h_gauss(X, Y) if gauss else np.zeros_like(X)
    H = Hg + Hc + Hd
    Xc, Yc, Zc = -X, Y, -H
    zmin = float(Zc.min())
    return dict(X_p=X, Y_p=Y, H_src=Hg, H_curv=Hc, H_gauss=Hd, H=H, x_c=Xc, y_c=Yc, z_c=Zc, c=0.0, m3_shift=-zmin,
                meta=dict(variant=prof['variant'], grid=grid_name, kappa_H=kappa_H, gauss=gauss, gauss_params=(GAUSS if gauss else None)))


def m3_alignment(S: dict, fov_half=FOV_HALF) -> dict:

    H = S['H']; X = S['X_p']; Y = S['Y_p']
    iy = np.argmin(np.abs(Y[:, 0])); ix = np.argmin(np.abs(X[0, :])); H0 = float(H[iy, ix])
    out = {}
    for name, m in (('full_patch', np.ones_like(H, bool)), ('fov_square', (np.abs(X) <= fov_half) & (np.abs(Y) <= fov_half))):
        Hmax = float(H[m].max()); Hmin = float(H[m].min())
        out[name] = dict(H_max=Hmax, H_min=Hmin, H_axis=H0, sensor_height_H=Hmax + WD, axis_depth_mm=WD + Hmax - H0, depth_range_mm=Hmax - Hmin,
                         axis_depth_minus_WD=Hmax - H0)
    m = (np.abs(X) <= fov_half) & (np.abs(Y) <= fov_half)
    out['fov_depth_range_mm'] = float(H[m].max() - H[m].min())
    out['volume_20mm_covers_fov_range'] = bool(out['fov_depth_range_mm'] <= 20.0)
    return out
