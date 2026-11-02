# Generate ordinary and size-coded pattern rasters for registered cases.

from __future__ import annotations

import numpy as np

import c2_common as K

DOT_LAYOUTS = {'R5x4': (5.0, 4.0), 'R6x4': (6.0, 4.0), 'R8x4': (8.0, 4.0)}
LINE_LAYOUTS = ('4.45', '5.6', '8')


def levels() -> dict:
    return K.load_json(K.O1 / 'results' / 'precheck_levels.json')['levels']


def dot_raster(layout: str, u: float, v: float, radii) -> dict:
    import ep_patterns as EP
    p_par, p_perp = DOT_LAYOUTS[layout]
    c, cls = EP.lattice_centres(p_par, p_perp, u, v, 'R', 0.0, 2, 1)
    if len(radii) == 1:
        cls = np.zeros(len(c), np.int64)
    return dict(c=c, cls=cls, pat=EP.render(c, cls, list(radii)))


def build(fam: str, layout: str, pattern: str, ph: dict, lev: dict) -> dict:
    import s3c_precheck as PC
    if fam == 'dots':
        radii = [PC.R0] if pattern == 'plain' else list(lev['dots']['radii'][pattern])
        R = dot_raster(layout, ph['u'], ph['v'], radii)
        return dict(binary=(np.asarray(R['pat']) > 0).astype(np.float32), centres=R['c'], classes=R['cls'], radii=radii, meta_full=None)
    wn, ww = (None, None) if pattern == 'plain' else lev['lines']['widths'][pattern]
    R = PC.line_raster(float(layout), ph['phase_frac'], wn, ww)
    return dict(binary=(np.asarray(R['pat']) > 0).astype(np.float32), centres=None, classes=None, widths=(None if wn is None else [wn, ww]), meta_full=R['meta'])


def raster_key(binary) -> str:
    import s3c_formal as F
    return F.raster_key(binary)
