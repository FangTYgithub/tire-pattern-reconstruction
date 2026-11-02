# Calculate geometric and signal-based descriptors of a projector pattern.

from __future__ import annotations

import math

import numpy as np

from m4_reconstruction import (pattern_features, projector_tracks, search_window, _resolve_sensor_cfg, projector_size_classes,
                               projector_bright_classes, projector_dash_classes, MEASUREMENT_VOLUME_MM, BORDER_MARGIN_PX,
                               VLINE_TYPES, DOT_TYPES, ANCHORED_TYPES, PROP_Z_JUMP_MM, ANCHOR_TOL_PX, SIZE_UNCERTAIN_FRAC)
from pattern_predictor import predict_ambiguity, predict_drift_outcomes

MM_PER_PX = 0.18
VOLUMES = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
DRIFTS_PX = (1.0, 2.0, 3.0, 4.0, 5.8, 8.0, 10.0, 14.0)
DRIFT_REF_PX = 5.8
A_MARGIN_THR = 0.05
DENT_SLOPE, DENT_OFFSET = 52.0, -3.0


NOISE_GAP_NODES = ((1.5, 0.0), (2.2, 20.0), (3.3, 55.0), (4.4, 70.0), (5.6, 90.0), (8.3, 100.0))
NOISE_FILL_NODES = ((0.010, 0.0), (0.025, 40.0), (0.046, 60.0), (0.064, 80.0), (0.079, 100.0))
NOISE_SMALL_DOT_PX, NOISE_SMALL_DOT_CAP = 4.0, 50.0
NOISE_COORD_PENALTY_PX = 1.0


def noise_limit(gap_px: float, fill: float, r_px: float, coordination: int = 2) -> float:

    if not (np.isfinite(gap_px) and np.isfinite(fill)):
        return float("nan")
    g_eff = gap_px - (NOISE_COORD_PENALTY_PX if coordination >= 4 else 0.0)
    s_gap = float(np.interp(g_eff, [x for x, _ in NOISE_GAP_NODES], [y for _, y in NOISE_GAP_NODES]))
    s_fill = float(np.interp(fill, [x for x, _ in NOISE_FILL_NODES], [y for _, y in NOISE_FILL_NODES]))
    s = min(s_gap, s_fill)
    if np.isfinite(r_px) and r_px < NOISE_SMALL_DOT_PX:
        s = min(s, NOISE_SMALL_DOT_CAP)
    return s


def features_and_classes(pattern: np.ndarray, pattern_type: str, cfg: dict):

    if pattern_type in ("coded_dots", *ANCHORED_TYPES):
        pts, ids, areas = pattern_features(pattern, pattern_type, return_areas=True)
        return pts, np.asarray(ids), np.asarray(projector_size_classes(areas, cfg)), areas
    if pattern_type == "bright_dots":
        pts, ids, inten = pattern_features(pattern, pattern_type, return_intensity=True)
        _, _, areas = pattern_features((pattern > 0.005).astype(np.float32), "dots", return_areas=True)
        return pts, np.asarray(ids), np.asarray(projector_bright_classes(inten, cfg)), areas
    if pattern_type == "dashed_vlines" and int(cfg.get("dash_levels", 1) or 1) > 1:
        pts, ids = pattern_features(pattern, pattern_type)
        return pts, np.asarray(ids), np.asarray(projector_dash_classes(pts, ids, pattern, cfg)), None
    if pattern_type in DOT_TYPES:
        pts, ids, areas = pattern_features(pattern, pattern_type, return_areas=True)
        return pts, np.asarray(ids), None, areas
    pts, ids = pattern_features(pattern, pattern_type)
    return pts, np.asarray(ids), None, None


def visible_count(pts: np.ndarray, cfg: dict, depth_mm: float) -> int:

    if len(pts) == 0:
        return 0
    A, B = projector_tracks(pts, cfg, *search_window(cfg, depth_mm))
    mid = 0.5 * (A + B)
    W = int(cfg["image_size_px"]); m = BORDER_MARGIN_PX
    inside = (mid[:, 0] >= m) & (mid[:, 0] <= W - 1 - m) & (mid[:, 1] >= m) & (mid[:, 1] <= W - 1 - m)
    return int(inside.sum())


def pitches_mm(pts: np.ndarray, k: int = 6, k_fallback: int = 12) -> tuple[float, float]:

    from scipy.spatial import cKDTree
    if len(pts) < k + 1:
        return float("nan"), float("nan")
    kk = min(k_fallback, len(pts) - 1)
    d, j = cKDTree(pts).query(pts, k=kk + 1)
    du = np.abs(pts[j[:, 1:], 0] - pts[:, None, 0]); dv = np.abs(pts[j[:, 1:], 1] - pts[:, None, 1])
    dist = d[:, 1:]
    near = np.arange(kk)[None, :] < k
    along = np.where((du >= dv) & near, dist, np.inf).min(axis=1)
    perp = np.where((dv > du) & near, dist, np.inf).min(axis=1)
    along = np.where(np.isfinite(along), along, np.where(du >= dv, dist, np.inf).min(axis=1))
    perp = np.where(np.isfinite(perp), perp, np.where(dv > du, dist, np.inf).min(axis=1))
    p_par = float(np.median(along[np.isfinite(along)])) * MM_PER_PX if np.isfinite(along).any() else float("nan")
    p_perp = float(np.median(perp[np.isfinite(perp)])) * MM_PER_PX if np.isfinite(perp).any() else float("nan")
    return p_par, p_perp


def coordination_number(pts: np.ndarray) -> float:

    from scipy.spatial import cKDTree
    if len(pts) < 8:
        return float("nan")
    d, _ = cKDTree(pts).query(pts, k=min(9, len(pts)))
    dn = d[:, 1:]
    return float(np.median((dn <= dn[:, :1] + 0.5).sum(axis=1)))


def blur_limit_px(pattern: np.ndarray, pattern_type: str, pts: np.ndarray, areas) -> tuple[float, float]:

    if pattern_type in VLINE_TYPES or pattern_type == "lines":
        pat = np.asarray(pattern) > 0.5
        profiles = pat if pattern_type in VLINE_TYPES else pat.T
        best = None
        for row in profiles:
            on = np.flatnonzero(row)
            if len(on) < 2:
                continue
            breaks = np.flatnonzero(np.diff(on) > 1)
            if len(breaks) < 2:
                continue
            starts = np.r_[on[0], on[breaks + 1]]; ends = np.r_[on[breaks], on[-1]]
            gaps = starts[1:] - ends[:-1] - 1
            if best is None or len(gaps) > best[0]:
                best = (len(gaps), float(np.median(gaps)))
        if best is None:
            return float("nan"), float("nan")
        gap_mm = best[1] * MM_PER_PX
        return gap_mm / (3.0 * MM_PER_PX), gap_mm
    from scipy.spatial import cKDTree
    if len(pts) < 3 or areas is None:
        return float("nan"), float("nan")
    r = np.sqrt(np.asarray(areas, dtype=np.float64) / math.pi)
    W = pattern.shape[1]
    inner = (pts.min(axis=1) >= 2 * r.max()) & (pts.max(axis=1) <= W - 1 - 2 * r.max())
    d, j = cKDTree(pts).query(pts, k=min(7, len(pts)))


    dn, jn = d[:, 1:], j[:, 1:]
    gap_px = np.where(dn <= dn[:, :1] + 0.5, dn - r[:, None] - r[jn], np.inf).min(axis=1)
    g = gap_px[inner] if inner.sum() >= 10 else gap_px
    gap_mm = float(np.percentile(g, 10)) * MM_PER_PX
    return gap_mm / (3.0 * MM_PER_PX), gap_mm


def _first_crossing(x: np.ndarray, y: np.ndarray, thr: float) -> float:
    for i in range(len(x)):
        if y[i] > thr:
            if i == 0:
                return float(x[0])
            return float(x[i - 1] + (thr - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])) if y[i] != y[i - 1] else float(x[i])
    return float("inf")


# Calculate pattern geometry, signal and predicted ambiguity descriptors.
def evaluate_pattern(
    pattern:        np.ndarray,
    pattern_type:   str,
    sensor_cfg:     dict | None = None,
    depth_range_mm: float | None = None,
    volumes:        tuple = VOLUMES,
    drifts_px:      tuple = DRIFTS_PX,
    fwhm_mm:        float = 20.0,
    z_samples:      np.ndarray | None = None,
    n_t:            int = 21,
    fast:           bool = True,
) -> dict:

    cfg = _resolve_sensor_cfg(sensor_cfg)
    depth = float(depth_range_mm or cfg.get("measurement_volume_mm", MEASUREMENT_VOLUME_MM))
    anchored = pattern_type in ANCHORED_TYPES
    lines = pattern_type in VLINE_TYPES or pattern_type == "lines"
    pts, ids, classes, areas = features_and_classes(pattern, pattern_type, cfg)
    fov_mm = pattern.shape[1] * MM_PER_PX
    n_feat = int(len(np.unique(ids))) if lines else int(len(pts))
    out: dict = {"pattern_type": pattern_type, "n_features": n_feat, "n_row_features": int(len(pts)),
                 "C_D": (len(pts) if not lines else n_feat * pattern.shape[0]) / (fov_mm * fov_mm),
                 "n_visible": visible_count(pts, cfg, depth), "design_volume_mm": depth}

    a_curve, a_prop = {}, {}
    for vol in volumes:
        res = predict_ambiguity(pattern, pattern_type, sensor_cfg={**cfg, "measurement_volume_mm": float(vol)},
                                depth_range_mm=float(vol), z_samples=z_samples, n_t=n_t,
                                prop_jump_mm=PROP_Z_JUMP_MM if anchored else None, fast=fast)
        a_curve[float(vol)] = float(res["A_pred"])
        if anchored:
            a_prop[float(vol)] = float(res["A_prop"])
    vx = np.array(sorted(a_curve)); vy = np.array([a_curve[v] for v in vx])
    out.update(A_curve=a_curve, A_design=float(np.interp(depth, vx, vy)), vol_margin_mm=_first_crossing(vx, vy, A_MARGIN_THR))
    if anchored:
        px = np.array([a_prop[v] for v in vx])
        out.update(A_prop_curve=a_prop, A_prop_design=float(np.interp(depth, vx, px)),
                   vol_margin_prop_mm=_first_crossing(vx, px, A_MARGIN_THR))

    if lines:
        out.update(drift_mode="immune", drift_silent_px=float("inf"), drift_loud_px=float("inf"), dz_silent_mm=0.0,
                   drift_curve={})
    else:
        curve = {}
        for dv in drifts_px:
            o = [predict_drift_outcomes(pattern, pattern_type, s * dv, sensor_cfg={**cfg, "measurement_volume_mm": depth},
                                        depth_range_mm=depth, z_samples=z_samples, n_t=max(7, n_t // 2), fast=fast) for s in (-1.0, 1.0)]
            k = int(np.argmax([x["f_silent"] for x in o]))
            curve[float(dv)] = dict(silent=o[k]["f_silent"], loud=o[k]["f_rejected"] + o[k]["f_ambiguous"],
                                    dz_mm=o[k]["dz_silent_median_mm"])
        dx = np.array(sorted(curve)); sil = np.array([curve[d]["silent"] for d in dx]); loud = np.array([curve[d]["loud"] for d in dx])
        s_on, l_on = _first_crossing(dx, sil, 0.01), _first_crossing(dx, loud, 0.10)
        ref = curve.get(DRIFT_REF_PX) or curve[min(curve, key=lambda d: abs(d - DRIFT_REF_PX))]
        out.update(drift_curve=curve, drift_silent_px=s_on, drift_loud_px=l_on,
                   drift_mode="silent" if s_on <= l_on + 1e-9 else "loud", dz_silent_mm=ref["dz_mm"], silent_at_ref=ref["silent"])
        if anchored:
            out.update(drift_mode="corrected", drift_corrected_to_px=float(cfg.get("anchor_tol_px", ANCHOR_TOL_PX)),
                       drift_mode_without_anchors="silent" if s_on <= l_on + 1e-9 else "loud")

    blur, gap_mm = blur_limit_px(pattern, pattern_type, pts, areas)
    out.update(blur_px=blur, gap_mm=gap_mm)

    if lines:
        u = np.sort(np.unique(np.round(pts[:, 0])))
        p_par = float(np.median(np.diff(u))) * MM_PER_PX if len(u) > 1 else float("nan")
        p_perp = MM_PER_PX
    else:
        p_par, p_perp = pitches_mm(pts)
    rho = 0.5 * math.hypot(p_par / fwhm_mm, p_perp / fwhm_mm) if np.isfinite(p_par) and np.isfinite(p_perp) else float("nan")
    out.update(p_par_mm=p_par, p_perp_mm=p_perp, rho=rho, dent_pct=max(0.0, DENT_SLOPE * rho + DENT_OFFSET) if np.isfinite(rho) else float("nan"))

    span = float("nan")
    if pattern_type == "coded_dots" or anchored:
        rr = cfg.get("size_radii_px")
        span = 2.0 * math.log(max(rr) / min(rr)) if rr else float("nan")
    elif pattern_type == "bright_dots":
        lv = cfg.get("bright_levels")
        span = math.log(max(lv) / min(lv)) if lv else float("nan")
    elif pattern_type == "dashed_vlines" and int(cfg.get("dash_levels", 1) or 1) > 1:
        span = (int(cfg["dash_levels"]) - 1) * math.log(1.0 / float(cfg.get("dash_ratio", 0.5)))
    out.update(code_span_log=span, code_uncertain_band_log=SIZE_UNCERTAIN_FRAC * span if np.isfinite(span) else float("nan"),
               min_radius_px=float(np.sqrt(np.min(areas) / math.pi)) if areas is not None and len(areas) else float("nan"))

    if not lines and areas is not None and len(areas):
        fill = float(np.sum(areas)) / float(pattern.shape[0] * pattern.shape[1])
        z = coordination_number(pts)
        out.update(fill_frac=fill, coordination=z,
                   noise_sigma_max=noise_limit(gap_mm / MM_PER_PX, fill, float(np.sqrt(np.median(areas) / math.pi)), int(z) if np.isfinite(z) else 2))
    else:
        out.update(fill_frac=float("nan"), coordination=float("nan"), noise_sigma_max=float("nan"))
    return out
