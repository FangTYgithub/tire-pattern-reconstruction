# Move dot columns while retaining the registered lattice membership.

from __future__ import annotations

import numpy as np

import sp_common as S
import sn_lib as SL
from m2_pattern import _lattice_centres, _draw_dots

GENERATOR = 'sp_dots_columns_v1'


def base_lattice(p_par_mm: float, p_perp_mm: float, u: float, v: float, image_size_px: int = S.IMG):

    k = SL.mmpp(); sp, pp = p_par_mm / k, p_perp_mm / k
    cx, cy, IX, IY = _lattice_centres(image_size_px, sp, pp, u * sp, v * pp)
    return np.ravel(cx).astype(np.float64), np.ravel(cy).astype(np.float64), np.ravel(IX).astype(np.int64), np.ravel(IY).astype(np.int64), sp, pp


def movable_columns(p_par_mm: float, a_mm: float, image_size_px: int = S.IMG) -> dict:

    k = SL.mmpp(); sp = p_par_mm / k; a_px = a_mm / k; m = SL.edge_margin_px(); W = image_size_px
    centre_x = (W - 1) / 2.0
    n_x = int(np.ceil(W / sp)) + 2
    ix = np.arange(-n_x, n_x + 1)
    lo = centre_x + ix * sp; hi = centre_x + (ix + 1) * sp
    mov = (lo >= m + a_px) & (hi <= W - 1 - m - a_px)
    return dict(ix_movable=[int(i) for i in ix[mov]], n_movable=int(mov.sum()), margin_px=float(m), a_px=float(a_px), pitch_px=float(sp), rule='for all u in [0,1): m + a <= cx(ix, u) <= W-1-m-a')


def layout(p_par_mm: float, p_perp_mm: float, u: float, v: float, disp_px_by_ix: dict | None = None, axis: str = 'u', image_size_px: int = S.IMG) -> dict:

    cx, cy, IX, IY, sp, pp = base_lattice(p_par_mm, p_perp_mm, u, v, image_size_px)
    d = np.zeros(len(cx))
    if disp_px_by_ix:
        dd = {int(k): float(val) for k, val in disp_px_by_ix.items()}
        d = np.array([dd.get(int(i), 0.0) for i in IX])
    if axis == 'u':
        cx = cx + d
    elif axis == 'v':
        cy = cy + d
    else:
        raise ValueError(axis)
    m = SL.edge_margin_px(); W = image_size_px
    keep = (cx >= m) & (cx <= W - 1 - m) & (cy >= m) & (cy <= W - 1 - m)
    keep0 = None
    if disp_px_by_ix:
        cx0, cy0, _, _, _, _ = base_lattice(p_par_mm, p_perp_mm, u, v, image_size_px)
        keep0 = (cx0 >= m) & (cx0 <= W - 1 - m) & (cy0 >= m) & (cy0 <= W - 1 - m)
    c = np.column_stack([cx[keep], cy[keep]]).astype(np.float64)
    order = np.lexsort((c[:, 0], c[:, 1]))
    ix_present = sorted(set(int(i) for i in IX[keep]))
    out = dict(c=c[order], ix=IX[keep][order], iy=IY[keep][order], disp_px=d[keep][order], n=int(keep.sum()), ix_present=ix_present, n_columns=len(ix_present),
               rows_present=sorted(set(int(i) for i in IY[keep])), pitch_px=(sp, pp), axis=axis)
    if keep0 is not None:
        out['n_regular_same_phase'] = int(keep0.sum()); out['boundary_in'] = int((keep & ~keep0).sum()); out['boundary_out'] = int((~keep & keep0).sum())
    return out


def raster(c: np.ndarray, r_mm: float = S.R_MM, image_size_px: int = S.IMG) -> np.ndarray:
    return np.asarray(_draw_dots(image_size_px, np.asarray(c, np.float64), r_mm / SL.mmpp()), np.float32)


def column_positions(lay: dict) -> tuple[np.ndarray, np.ndarray]:

    ixs = np.array(lay['ix_present']); pos = np.array([float(np.mean(lay['c'][lay['ix'] == i, 0])) for i in ixs])
    o = np.argsort(pos)
    return ixs[o], pos[o]
