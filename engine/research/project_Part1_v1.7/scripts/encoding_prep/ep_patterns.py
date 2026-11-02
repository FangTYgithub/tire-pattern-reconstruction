# Build coded dot rasters and calculate spacing, resource and ideal-sampling descriptors.

import math
import numpy as np

import ep_common as C
import sl_lib as L
import sn_lib as SL
from m2_pattern import _lattice_centres, _code_lattice, _draw_dots, coded_dot_radii

IMG = SL.IMG
R_L = 1.0
BUDGET_LIT_PX = float(C.load_json(L.OUT / 'design' / 'budget.json')['budget_lit_px'])
LAYOUTS = {'R8x4': (8.0, 4.0), 'R6x4': (6.0, 4.0), 'R5x4': (5.0, 4.0), 'R4x4': (4.0, 4.0)}
PHASES4 = [L.latin_indices()[i] for i in (0, 2, 4, 6)]


def mmpp():
    return SL.mmpp()


def rotated_offset(u, v, p_par_px, p_perp_px, rotation_deg):
    ox, oy = u * p_par_px, v * p_perp_px
    if rotation_deg:
        c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
        ox, oy = ox * c - oy * s, ox * s + oy * c
    return ox, oy


def lattice_centres(p_par, p_perp, u, v, arrangement='R', rotation_deg=0.0, k=1, code_shift=1):

    kk = mmpp(); sp, pp = p_par / kk, p_perp / kk
    ox, oy = rotated_offset(u, v, sp, pp, rotation_deg)
    if k > 1:
        cx, cy, cls = _code_lattice(IMG, sp, pp, int(code_shift), int(k), ox, oy, float(rotation_deg))
    else:
        cx, cy, _, _ = _lattice_centres(IMG, sp, pp, ox, oy, stagger_frac=0.5 if arrangement == 'T' else 0.0, rotation_deg=float(rotation_deg))
        cls = np.zeros(np.shape(cx), np.int64)
    cx, cy, cls = np.ravel(cx), np.ravel(cy), np.ravel(cls); m = SL.edge_margin_px()
    keep = (cx >= m) & (cx <= IMG - 1 - m) & (cy >= m) & (cy <= IMG - 1 - m)
    c = np.column_stack([cx[keep], cy[keep]]).astype(np.float64); cls = cls[keep].astype(np.int64)
    order = np.lexsort((c[:, 0], c[:, 1]))
    return c[order], cls[order]


def radii_mm(k, ratio):
    return [R_L] if int(k) == 1 else [float(r) for r in coded_dot_radii(R_L, int(k), float(ratio))]


def render(c, cls, radii):

    pat = np.zeros((IMG, IMG), np.float32)
    for j, r in enumerate(radii):
        if (cls == j).any():
            pat = np.maximum(pat, _draw_dots(IMG, c[cls == j], r / mmpp()))
    return pat.astype(np.float32)


def arm(layout, kind, ratio=0.6, rotation_deg=0.0, k=2):

    p_par, p_perp = LAYOUTS[layout] if isinstance(layout, str) else layout
    name = layout if isinstance(layout, str) else f'R{p_par:g}x{p_perp:g}'
    if kind == 'plain':
        return dict(layout=name, arm='plain', p_par=p_par, p_perp=p_perp, k=1, ratio=1.0, rotation_deg=0.0, code_shift=1, radii_mm=[R_L])
    if kind == 'coded':
        return dict(layout=name, arm=f'coded{k}', p_par=p_par, p_perp=p_perp, k=int(k), ratio=float(ratio), rotation_deg=0.0, code_shift=1, radii_mm=radii_mm(k, ratio))
    if kind == 'rot':
        return dict(layout=name, arm=f'rot{rotation_deg:g}', p_par=p_par, p_perp=p_perp, k=1, ratio=1.0, rotation_deg=float(rotation_deg), code_shift=1, radii_mm=[R_L])
    raise ValueError(kind)


def build(a, u, v):
    c, cls = lattice_centres(a['p_par'], a['p_perp'], u, v, 'R', a['rotation_deg'], a['k'], a['code_shift'])
    return c, cls, render(c, cls, a['radii_mm'])


def gaps(c, cls, radii):

    from scipy.spatial import cKDTree
    if len(c) < 2:
        return dict(centre_nn_min_mm=None, edge_gap_min_mm=None, edge_gap_median_mm=None)
    d, j = cKDTree(c).query(c, k=2)
    r = np.asarray(radii)[cls] / mmpp(); rj = np.asarray(radii)[cls[j[:, 1]]] / mmpp()
    gap = (d[:, 1] - r - rj) * mmpp()
    return dict(centre_nn_min_mm=float(d[:, 1].min() * mmpp()), edge_gap_min_mm=float(gap.min()), edge_gap_median_mm=float(np.median(gap)))


def resources(c, cls, radii, binpat, budget=BUDGET_LIT_PX):
    lit_cls = [float(_draw_dots(IMG, c[cls == j], r / mmpp()).sum()) if (cls == j).any() else 0.0 for j, r in enumerate(radii)]
    lit = float(binpat.sum()); v_b = budget / lit if lit else float('nan')
    return dict(n_dots=int(len(c)), n_per_class=[int((cls == j).sum()) for j in range(len(radii))], lit_px=lit, lit_px_per_class=lit_cls,
                lit_frac=float((binpat > 0).mean()), v_fixed_peak=1.0, integral_fixed_peak=lit, budget_lit_px=float(budget), v_equal_integral=float(v_b),
                integral_equal_integral=float(budget), equal_integral_feasible=bool(v_b <= 1.0 + 1e-9), sat_px_at_v1=0)


def expected(c, surf, fwhm=20.0):
    cx = L.ctx(surf, fwhm)
    return cx, SL.expected_dots(cx['mesh'], c)


def ideal_sampling(c, surf, fwhm=20.0):

    cx, exp = expected(c, surf, fwhm)
    a = L.stage_a(cx, exp)
    a['n_visible'] = int(np.isfinite(exp['uv']).all(axis=1).sum()); a['n_cam_inner'] = int(exp['cam_inner'].sum())
    return a
