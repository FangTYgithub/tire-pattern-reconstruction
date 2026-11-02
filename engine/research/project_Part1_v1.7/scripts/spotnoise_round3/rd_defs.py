# Define size-brightness comparisons and condition-level summary rules.

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[0] / 'spotnoise_suppA'), str(HERE.parents[0] / 'spotnoise')]
import sa_defs as D

BASE_LABEL = 'Baseline failed'
TOP_PREFIX = 'Top level'
GAP_GRID_MM = 0.5
DISC_R_MM = 15.0


def observed(row):

    lab = str(row['label']).replace('\u57fa\u51c6\u6863\u5373\u4e0d\u5408\u683c', BASE_LABEL).replace('\u9876\u6863', TOP_PREFIX)
    if lab == BASE_LABEL:
        return dict(kind='base_fail', last_pass=None, first_fail=0.0)
    ff = row.get('first_fail_sigma')
    if lab.startswith(TOP_PREFIX) or lab.startswith('≥') or ff is None or (isinstance(ff, float) and np.isnan(ff)):
        return dict(kind='censored', last_pass=float(row['boundary_sigma']) if row.get('boundary_sigma') is not None else None, first_fail=None)
    return dict(kind='numeric', last_pass=float(row['boundary_sigma']), first_fail=float(ff))


def obs_text(o, top):
    if o['kind'] == 'base_fail':
        return 'Baseline failed'
    if o['kind'] == 'censored':
        return f'≥{top:g} (right-censored)'
    return f"{o['last_pass']:g} (first failure {o['first_fail']:g})"


def compare(a, b, top):

    if a['kind'] == 'base_fail' or b['kind'] == 'base_fail':
        return dict(kind='base_fail', value=None)
    if a['kind'] == 'censored' and b['kind'] == 'censored':
        return dict(kind='undetermined', value=None)
    if a['kind'] == 'censored':
        return dict(kind='lower_bound', value=float(top - b['last_pass']))
    if b['kind'] == 'censored':
        return dict(kind='upper_bound', value=float(a['last_pass'] - top))
    return dict(kind='difference', value=float(a['last_pass'] - b['last_pass']))


def order_rank(o, levels):

    if o['kind'] == 'base_fail':
        return -1
    if o['kind'] == 'censored':
        return len(levels)
    return list(levels).index(o['last_pass'])


_GRID = None


def _grid():
    global _GRID
    if _GRID is None:
        g = np.arange(-DISC_R_MM, DISC_R_MM + 1e-9, GAP_GRID_MM)
        gx, gy = np.meshgrid(g, g)
        m = gx ** 2 + gy ** 2 <= DISC_R_MM ** 2
        _GRID = np.column_stack([gx[m], gy[m]])
    return _GRID


def spatial_gap(pts, amb, err_ref, gate_z=1.0):

    q = np.isfinite(pts).all(axis=1) & ~np.asarray(amb, bool) & np.isfinite(err_ref) & (np.abs(np.nan_to_num(err_ref, nan=1e9)) <= gate_z)
    if not q.any():
        return dict(gap_max_mm=None, gap_p95_mm=None, n_qualified=0)
    d, _ = cKDTree(pts[q][:, :2].astype(np.float64)).query(_grid())
    return dict(gap_max_mm=float(d.max()), gap_p95_mm=float(np.percentile(d, 95)), n_qualified=int(q.sum()))
