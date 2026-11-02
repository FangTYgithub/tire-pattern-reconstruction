# Interpolate reconstructed points onto fixed evaluation nodes and measure support and height
# accuracy. Support distance and height tolerances are caller-supplied values in millimetres.

from __future__ import annotations

import math

import numpy as np

EVAL_VERSION = "surface_eval_r3_dev_2"
EVAL_VERSION_V1 = "surface_eval_r3_dev_1"
SUPPORT_RULES = ("exact", "v1_enclosing_circle")
DEDUP_TOL_MM = 1e-6  # Planar tolerance for merging duplicate samples [mm].
_AREA_EPS = 1e-12  # Numerical triangle-area tolerance [mm^2].


# Merge accepted samples at coincident planar locations within the given tolerance.
def dedup_xy(points: np.ndarray, accept: np.ndarray | None = None, tol: float = DEDUP_TOL_MM) -> tuple[np.ndarray, int]:

    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    m = np.isfinite(P).all(axis=1)
    if accept is not None and len(accept) == len(P):
        m &= np.asarray(accept, dtype=bool)
    P = P[m]
    if len(P) == 0:
        return P, 0
    key = np.round(P[:, :2] / tol).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    return P[first], int(len(P) - len(first))


# Build the Delaunay triangulation of the planar sample positions.
def triangulate_xy(P: np.ndarray):

    from scipy.spatial import Delaunay
    try:
        from scipy.spatial import QhullError
    except ImportError:
        from scipy.spatial.qhull import QhullError
    if len(P) < 3:
        return None, "too_few_points"
    c = P[:, :2] - P[:, :2].mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    if s[1] <= 1e-9 * max(float(s[0]), 1.0):
        return None, "collinear"
    try:
        return Delaunay(P[:, :2]), "ok"
    except QhullError as e:
        return None, "qhull_error: " + str(e).splitlines()[0][:80]


def circumradius(P2: np.ndarray, simplices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    a, b, c = P2[simplices[:, 0]], P2[simplices[:, 1]], P2[simplices[:, 2]]
    la = np.linalg.norm(b - c, axis=1); lb = np.linalg.norm(a - c, axis=1); lc = np.linalg.norm(a - b, axis=1)
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = la * lb * lc / (2.0 * area2)
    R = np.where(np.isfinite(R), R, np.inf)
    return R, np.maximum.reduce([la, lb, lc])


def enclosing_circle_radius_v1(P2: np.ndarray, simplices: np.ndarray) -> np.ndarray:

    a, b, c = P2[simplices[:, 0]], P2[simplices[:, 1]], P2[simplices[:, 2]]
    e = np.sort(np.column_stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(a - c, axis=1), np.linalg.norm(a - b, axis=1)]), axis=1)
    R, _ = circumradius(P2, simplices)
    obtuse = e[:, 2] ** 2 >= e[:, 0] ** 2 + e[:, 1] ** 2
    rho = np.where(obtuse, 0.5 * e[:, 2], R)
    return np.where(np.isfinite(rho), rho, np.inf)


# Compute the largest nearest-vertex distance within each triangle.
def covering_radius_exact(P2: np.ndarray, simplices: np.ndarray, return_argmax: bool = False):

    P2 = np.asarray(P2, np.float64)
    V = P2[np.asarray(simplices)]
    T = len(V)
    if T == 0:
        return (np.zeros(0), np.zeros((0, 2))) if return_argmax else np.zeros(0)
    cands = np.full((T, 10, 2), np.nan)

    a, b, c = V[:, 0], V[:, 1], V[:, 2]
    d = 2.0 * ((a[:, 0]) * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    ok = np.abs(d) > _AREA_EPS * (1.0 + np.abs(V).max(axis=(1, 2)) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        a2, b2, c2 = (a ** 2).sum(1), (b ** 2).sum(1), (c ** 2).sum(1)
        ux = (a2 * (b[:, 1] - c[:, 1]) + b2 * (c[:, 1] - a[:, 1]) + c2 * (a[:, 1] - b[:, 1])) / d
        uy = (a2 * (c[:, 0] - b[:, 0]) + b2 * (a[:, 0] - c[:, 0]) + c2 * (b[:, 0] - a[:, 0])) / d
    cc = np.column_stack([ux, uy])

    v0 = b - a; v1 = c - a; v2 = cc - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    inside = ok & np.isfinite(l1) & np.isfinite(l2) & (l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1.0 + 1e-9)
    cands[:, 0, :] = np.where(inside[:, None], cc, np.nan)

    k = 1
    for (i, j) in ((0, 1), (1, 2), (0, 2)):
        n = V[:, j] - V[:, i]
        cst = 0.5 * ((V[:, j] ** 2).sum(1) - (V[:, i] ** 2).sum(1))
        for (p, q) in ((0, 1), (1, 2), (0, 2)):
            e = V[:, q] - V[:, p]
            den_e = (n * e).sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (cst - (n * V[:, p]).sum(1)) / den_e
            valid = np.isfinite(t) & (np.abs(den_e) > 1e-15) & (t >= -1e-9) & (t <= 1.0 + 1e-9)
            pt = V[:, p] + np.clip(t, 0.0, 1.0)[:, None] * e
            cands[:, k, :] = np.where(valid[:, None], pt, np.nan)
            k += 1

    diff = cands[:, :, None, :] - V[:, None, :, :]
    f = np.sqrt((diff ** 2).sum(-1)).min(-1)
    f = np.where(np.isfinite(f), f, -np.inf)
    rho = f.max(axis=1)
    rho = np.where(np.isfinite(rho), rho, np.inf)
    if return_argmax:
        idx = f.argmax(axis=1)
        return rho, cands[np.arange(T), idx]
    return rho


def covering_radius_reference(tri_vertices: np.ndarray, n: int = 400) -> tuple[float, float]:

    V = np.asarray(tri_vertices, np.float64).reshape(3, 2)
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    m = (i + j) <= n
    l1 = i[m] / n; l2 = j[m] / n; l0 = 1.0 - l1 - l2
    X = l0[:, None] * V[0] + l1[:, None] * V[1] + l2[:, None] * V[2]
    f = np.sqrt(((X[:, None, :] - V[None, :, :]) ** 2).sum(-1)).min(1)
    h = float(max(np.linalg.norm(V[1] - V[0]), np.linalg.norm(V[2] - V[1]), np.linalg.norm(V[2] - V[0]))) / n
    return float(f.max()), float(f.max() + h)


# Interpolate sample heights onto fixed query nodes using barycentric coordinates.
def interpolate_on_nodes(P: np.ndarray, tri, xq: np.ndarray, yq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    q = np.column_stack([np.asarray(xq, np.float64), np.asarray(yq, np.float64)])
    s = tri.find_simplex(q)
    z = np.full(len(q), np.nan)
    ok = s >= 0
    if ok.any():
        T = tri.transform[s[ok]]
        r = q[ok] - T[:, 2, :]
        bc = np.einsum("nij,nj->ni", T[:, :2, :], r)
        w = np.column_stack([bc, 1.0 - bc.sum(axis=1)])
        z[ok] = (w * P[tri.simplices[s[ok]], 2]).sum(axis=1)
    return z, s


# Measure each evaluation node's distance to the nearest reconstructed sample.
def nearest_sample_distance(P: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    if len(P) == 0:
        return np.full(len(xq), np.inf)
    d, _ = cKDTree(P[:, :2]).query(np.column_stack([xq, yq]))
    return d


def _stats(e: np.ndarray) -> dict:
    if len(e) == 0:
        return dict(rmse=None, p95=None, bias=None, max_abs=None)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), p95=float(np.percentile(np.abs(e), 95)), bias=float(np.mean(e)), max_abs=float(np.abs(e).max()))


# Evaluate support and height accuracy on fixed nodes for each supplied distance and tolerance pair.
def surface_eval_r3_dev(points: np.ndarray, accept: np.ndarray | None, domain: dict, d_list, tau_list,
                        masks: dict | None = None, return_fields: bool = False, support_rule: str = "exact") -> dict:

    if support_rule not in SUPPORT_RULES:
        raise ValueError(f"support_rule must be one of {SUPPORT_RULES}")
    x = np.asarray(domain["x"], np.float64); y = np.asarray(domain["y"], np.float64)
    z_ref = np.asarray(domain["z_ref"], np.float64); area = np.asarray(domain["area"], np.float64)
    n_nodes = len(x)
    P_in = np.asarray(points, np.float64).reshape(-1, 3)
    n_input = int(len(P_in))
    n_accepted = int(np.asarray(accept, bool).sum()) if (accept is not None and len(accept) == n_input) else int(np.isfinite(P_in).all(axis=1).sum())
    P, n_dup = dedup_xy(P_in, accept)
    tri, reason = triangulate_xy(P)
    z_est = np.full(n_nodes, np.nan); simplex = np.full(n_nodes, -1, np.int64); R_node = np.full(n_nodes, np.inf)
    n_tri = 0; n_degenerate = 0
    if tri is not None:
        R = covering_radius_exact(P[:, :2], tri.simplices) if support_rule == "exact" else enclosing_circle_radius_v1(P[:, :2], tri.simplices)
        n_tri = int(len(tri.simplices)); n_degenerate = int((~np.isfinite(R)).sum())
        z_est, simplex = interpolate_on_nodes(P, tri, x, y)
        inside = simplex >= 0
        R_node[inside] = R[simplex[inside]]
    nn = nearest_sample_distance(P, x, y)
    err = z_est - z_ref
    interpolable = (simplex >= 0) & np.isfinite(err)
    subdomains = {"global": np.ones(n_nodes, bool)}
    for k, m in (masks or {}).items():
        subdomains[k] = np.asarray(m, bool)
    out = dict(version=(EVAL_VERSION if support_rule == "exact" else EVAL_VERSION_V1), support_rule=support_rule, n_input=n_input, n_accepted=n_accepted,
               n_used=int(len(P)), n_duplicates_removed=n_dup, triangulation=reason, triangulated=bool(tri is not None), n_triangles=n_tri,
               n_degenerate_triangles=n_degenerate, domain_hash=domain.get("hash"), d_list=[float(d) for d in d_list], tau_list=[float(t) for t in tau_list], subdomains={})
    for name, m in subdomains.items():
        A = float(area[m].sum())
        A_int = float(area[m & interpolable].sum())
        sub = dict(n_nodes=int(m.sum()), area_mm2=A, n_interpolable=int((m & interpolable).sum()), C_interpolable=(A_int / A) if A > 0 else None, by_d={},
                   nn_dist=dict(max=float(nn[m].max()) if m.any() and np.isfinite(nn[m]).all() else None,
                                p95=float(np.percentile(nn[m], 95)) if m.any() and np.isfinite(nn[m]).all() else None,
                                median=float(np.median(nn[m])) if m.any() and np.isfinite(nn[m]).all() else None))
        for d in d_list:
            sup = m & interpolable & (R_node <= float(d))
            A_sup = float(area[sup].sum())
            e = err[sup]
            row = dict(C_valid=(A_sup / A) if A > 0 else None, missing=(1.0 - A_sup / A) if A > 0 else None, area_supported_mm2=A_sup, area_total_mm2=A,
                       n_supported=int(sup.sum()), **_stats(e), C_tau={}, area_tau_mm2={})
            for tau in tau_list:
                acc = sup & (np.abs(err) <= float(tau))
                A_acc = float(area[acc].sum())
                row["C_tau"][str(float(tau))] = (A_acc / A) if A > 0 else None
                row["area_tau_mm2"][str(float(tau))] = A_acc
            sub["by_d"][str(float(d))] = row
        out["subdomains"][name] = sub
    if return_fields:
        out["fields"] = dict(z_est=z_est.astype(np.float32), err=err.astype(np.float32), simplex=simplex, support_measure=R_node.astype(np.float32), nn_dist=nn.astype(np.float32))
    return out


DEFORMATION_QUANTILES = (0.95, 0.99)
DEFORMATION_VERSION = "deformation_quantiles_v1"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float):

    v = np.asarray(values, np.float64).reshape(-1); w = np.asarray(weights, np.float64).reshape(-1)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return None
    o = np.argsort(v[m]); v = v[m][o]; w = w[m][o]
    cw = np.cumsum(w); cw = (cw - 0.5 * w) / cw[-1]
    return float(np.interp(float(q), cw, v))


# Calculate weighted deformation-height quantiles on the selected domain mask.
def deformation_quantiles(fields: dict, domain: dict, mask: np.ndarray, dip_true: np.ndarray, base_rel: np.ndarray, d_list, q_list=DEFORMATION_QUANTILES) -> dict:

    z_est = np.asarray(fields["z_est"], np.float64); R = np.asarray(fields["support_measure"], np.float64); simplex = np.asarray(fields["simplex"])
    area = np.asarray(domain["area"], np.float64); m = np.asarray(mask, bool)
    dip_true = np.asarray(dip_true, np.float64); base_rel = np.asarray(base_rel, np.float64)
    dip_rec = base_rel - z_est
    A = float(area[m].sum())
    out = dict(version=DEFORMATION_VERSION, q_list=[float(q) for q in q_list], sign_rule="dip = base_rel - z (positive into the surface, pre-registered)", n_nodes=int(m.sum()), area_mm2=A,
               true_all={f"D{int(round(100 * q))}": weighted_quantile(dip_true[m], area[m], q) for q in q_list}, true_max=float(dip_true[m].max()) if m.any() else None, by_d={})
    interp = (simplex >= 0) & np.isfinite(z_est)
    for d in d_list:
        sup = m & interp & (R <= float(d))
        A_sup = float(area[sup].sum())
        row = dict(n_supported=int(sup.sum()), C_valid=(A_sup / A) if A > 0 else None, rec={}, true_supported_conditional={}, abs_diff_mm={}, rel_diff={})
        for q in q_list:
            key = f"D{int(round(100 * q))}"
            rec = weighted_quantile(dip_rec[sup], area[sup], q) if sup.any() else None
            tsup = weighted_quantile(dip_true[sup], area[sup], q) if sup.any() else None
            tall = out["true_all"][key]
            row["rec"][key] = rec; row["true_supported_conditional"][key] = tsup
            row["abs_diff_mm"][key] = (rec - tall) if (rec is not None and tall is not None) else None
            row["rel_diff"][key] = ((rec - tall) / tall) if (rec is not None and tall is not None and abs(tall) >= 1e-6) else None
        out["by_d"][str(float(d))] = row
    return out


# Calculate errors and gross-error counts at the accepted reconstructed points.
def point_metrics(points: np.ndarray, accept: np.ndarray | None, z_ref_at_points: np.ndarray, gross_mm: float = 1.0, prefix: str = "pts") -> dict:

    P = np.asarray(points, np.float64).reshape(-1, 3)
    m = np.isfinite(P).all(axis=1)
    if accept is not None and len(accept) == len(P):
        m &= np.asarray(accept, bool)
    zr = np.asarray(z_ref_at_points, np.float64).reshape(-1)
    m &= np.isfinite(zr)
    e = P[m, 2] - zr[m]
    st = _stats(e)
    return {f"{prefix}_n": int(m.sum()), f"{prefix}_rmse_mm": st["rmse"], f"{prefix}_p95_mm": st["p95"], f"{prefix}_bias_mm": st["bias"], f"{prefix}_max_abs_mm": st["max_abs"],
            f"{prefix}_gross_rate": (float((np.abs(e) > gross_mm).mean()) if len(e) else None), f"{prefix}_n_gross": int((np.abs(e) > gross_mm).sum())}
