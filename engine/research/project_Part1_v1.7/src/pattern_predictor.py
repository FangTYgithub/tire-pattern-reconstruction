# Predict candidate ambiguity and drift sensitivity from projector-camera geometry.

from __future__ import annotations

import math

import numpy as np

from m4_reconstruction import (_resolve_sensor_cfg, pattern_features, projector_tracks, search_window,
                               projector_size_classes, VLINE_TYPES,
                               EPI_AMBIG_TOL_PX, EPI_T_MARGIN, MEASUREMENT_VOLUME_MM)


def disparity_scale_px_per_mm(cfg: dict | None = None) -> float:

    c = _resolve_sensor_cfg(cfg)
    mm_per_px = c["pixel_pitch_mm"] / c["focal_length_mm"] * c["working_dist_mm"]
    theta = 2.0 * math.atan(c["baseline_mm"] / 2.0 / c["working_dist_mm"])
    return math.tan(theta) / mm_per_px


def depth_weights(depth_mm: float, z_samples: np.ndarray | None, n_t: int) -> tuple[np.ndarray, np.ndarray]:

    ts = (np.arange(n_t) + 0.5) / n_t
    if z_samples is None:
        return ts, np.full(n_t, 1.0 / n_t)
    z = np.asarray(z_samples, dtype=np.float64)
    z = z[np.isfinite(z)]
    hist, _ = np.histogram(np.clip(z / depth_mm, 0.0, 1.0 - 1e-9), bins=n_t, range=(0.0, 1.0))
    w = hist.astype(np.float64)
    return ts, (w / w.sum() if w.sum() > 0 else np.full(n_t, 1.0 / n_t))


def _neighbour_table(A: np.ndarray, B: np.ndarray, t_margin: float, tol_px: float, extra_px: float = 0.0) -> np.ndarray:

    from scipy.spatial import cKDTree
    mid = 0.5 * (A + B)
    reach = float(np.linalg.norm(B - A, axis=1).max()) * (1.0 + t_margin) + tol_px + extra_px + 1.0
    lists = cKDTree(mid).query_ball_point(mid, reach)
    K = max(1, max(len(l) for l in lists))
    J = np.full((len(A), K), -1, dtype=np.int64)
    for i, l in enumerate(lists):
        J[i, :len(l)] = l
    return J


def _seg_dist_table(P: np.ndarray, J: np.ndarray, A: np.ndarray, AB: np.ndarray, L2: np.ndarray, t_margin: float):

    valid = J >= 0
    Jc = np.where(valid, J, 0)
    d = P[:, None, :] - A[Jc]
    t = np.clip((d * AB[Jc]).sum(axis=-1) / L2[Jc], -t_margin, 1.0 + t_margin)
    dist = np.linalg.norm(P[:, None, :] - (A[Jc] + t[..., None] * AB[Jc]), axis=-1)
    return np.where(valid, dist, np.inf), t


def _min_dist_to_other_tracks(P: np.ndarray, A: np.ndarray, B: np.ndarray, ids: np.ndarray,
                              t_margin: float, chunk: int = 256,
                              classes: np.ndarray | None = None, return_t: bool = False,
                              table: np.ndarray | None = None):

    AB = B - A
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)
    if table is not None:
        dist, t = _seg_dist_table(P, table, A, AB, L2, t_margin)
        Jc = np.where(table >= 0, table, 0)
        excl = (table < 0) | (ids[Jc] == ids[:, None])
        if classes is not None:
            excl = excl | (classes[Jc] != classes[:, None])
        dm = np.where(excl, np.inf, dist)
        j = dm.argmin(axis=1)
        rows = np.arange(len(P))
        return (dm[rows, j], t[rows, j]) if return_t else dm[rows, j]
    out = np.empty(len(P))
    t_out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk]
        d = p[:, None, :] - A[None, :, :]
        t = np.clip((d * AB[None, :, :]).sum(axis=-1) / L2[None, :], -t_margin, 1.0 + t_margin)
        closest = A[None, :, :] + t[..., None] * AB[None, :, :]
        dist = np.linalg.norm(p[:, None, :] - closest, axis=-1)
        excl = ids[None, :] == ids[s:s + chunk][:, None]
        if classes is not None:
            excl = excl | (classes[None, :] != classes[s:s + chunk][:, None])
        dm = np.where(excl, np.inf, dist)
        j = dm.argmin(axis=1)
        out[s:s + chunk] = dm[np.arange(len(p)), j]
        t_out[s:s + chunk] = t[np.arange(len(p)), j]
    return (out, t_out) if return_t else out


# Estimate competing projector tracks across the configured depth window.
def predict_ambiguity(
    pattern:        np.ndarray,
    pattern_type:   str,
    sensor_cfg:     dict | None = None,
    depth_range_mm: float | None = None,
    z_samples:      np.ndarray | None = None,
    n_t:            int = 41,
    tol_px:         float | None = None,
    prop_jump_mm:   float | None = None,
    fast:           bool = True,
) -> dict:

    cfg = _resolve_sensor_cfg(sensor_cfg)
    depth = float(depth_range_mm or cfg.get("measurement_volume_mm", MEASUREMENT_VOLUME_MM))
    tol = float(tol_px if tol_px is not None else cfg.get("epi_ambig_tol_px", EPI_AMBIG_TOL_PX))
    t_margin = float(cfg.get("epi_t_margin", EPI_T_MARGIN))
    classes = None
    if pattern_type in ("coded_dots", "anchored_dots"):
        proj_pts, ids, areas = pattern_features(pattern, pattern_type, return_areas=True)
        classes = projector_size_classes(areas, cfg)
    elif pattern_type == "bright_dots":
        from m4_reconstruction import projector_bright_classes
        proj_pts, ids, inten = pattern_features(pattern, pattern_type, return_intensity=True)
        classes = projector_bright_classes(inten, cfg)
    elif pattern_type == "dashed_vlines" and int(cfg.get("dash_levels", 1) or 1) > 1:
        from m4_reconstruction import projector_dash_classes
        proj_pts, ids = pattern_features(pattern, pattern_type)
        classes = projector_dash_classes(proj_pts, ids, pattern, cfg)
    else:
        proj_pts, ids = pattern_features(pattern, pattern_type)
    if (pattern_type == "lines" or pattern_type in VLINE_TYPES) and len(proj_pts) > 1500:


        rows = np.round(proj_pts[:, 1]).astype(int)
        stride = int(math.ceil(len(proj_pts) / 1500))
        keep = (rows % stride) == 0
        proj_pts, ids = proj_pts[keep], np.asarray(ids)[keep]
        if classes is not None:
            classes = np.asarray(classes)[keep]
    N = len(proj_pts)
    empty = dict(A_pred=float("nan"), A_uniform=float("nan"), per_feature=np.zeros(0), n_features=0,
                 depth_mm=depth, tol_px=tol, disparity_px_per_mm=disparity_scale_px_per_mm(cfg))
    if N < 2:
        return empty
    A, B = projector_tracks(proj_pts, cfg, *search_window(cfg, depth))
    ids = np.asarray(ids)
    classes = None if classes is None else np.asarray(classes)
    table = _neighbour_table(A, B, t_margin, tol) if fast else None
    ts, w = depth_weights(depth, z_samples, n_t)
    amb = np.zeros((n_t, N), dtype=bool)
    amb_prop = np.zeros((n_t, N), dtype=bool)
    for k, t in enumerate(ts):
        P = A + t * (B - A)
        if prop_jump_mm:
            dmin, t_other = _min_dist_to_other_tracks(P, A, B, ids, t_margin, classes=classes, return_t=True, table=table)
            amb[k] = dmin <= tol
            amb_prop[k] = amb[k] & (np.abs(t_other - t) * depth <= 0.5 * float(prop_jump_mm))
        else:
            amb[k] = _min_dist_to_other_tracks(P, A, B, ids, t_margin, classes=classes, table=table) <= tol
    per_feature = amb.mean(axis=0)
    a_uniform = float(amb.mean())
    a_pred = float((w[:, None] * amb).sum() / N)
    out = dict(A_pred=a_pred, A_uniform=a_uniform, per_feature=per_feature, n_features=int(N),
               depth_mm=depth, tol_px=tol, disparity_px_per_mm=disparity_scale_px_per_mm(cfg))
    if prop_jump_mm:
        out.update(A_prop=float((w[:, None] * amb_prop).sum() / N), A_prop_uniform=float(amb_prop.mean()))
    return out


# Predict correspondence changes under the supplied image-position drift.
def predict_drift_outcomes(
    pattern:        np.ndarray,
    pattern_type:   str,
    dv_px:          float,
    du_px:          float = 0.0,
    sensor_cfg:     dict | None = None,
    depth_range_mm: float | None = None,
    z_samples:      np.ndarray | None = None,
    n_t:            int = 21,
    tol_px:         float | None = None,
    chunk:          int = 256,
    fast:           bool = True,
) -> dict:

    cfg = _resolve_sensor_cfg(sensor_cfg)
    depth = float(depth_range_mm or cfg.get("measurement_volume_mm", MEASUREMENT_VOLUME_MM))
    tol = float(tol_px if tol_px is not None else cfg.get("epi_ambig_tol_px", EPI_AMBIG_TOL_PX))
    t_margin = float(cfg.get("epi_t_margin", EPI_T_MARGIN))
    classes = None
    if pattern_type in ("coded_dots", "anchored_dots"):
        proj_pts, ids, areas = pattern_features(pattern, pattern_type, return_areas=True)
        classes = np.asarray(projector_size_classes(areas, cfg))
    elif pattern_type == "bright_dots":
        from m4_reconstruction import projector_bright_classes
        proj_pts, ids, inten = pattern_features(pattern, pattern_type, return_intensity=True)
        classes = np.asarray(projector_bright_classes(inten, cfg))
    else:
        proj_pts, ids = pattern_features(pattern, pattern_type)
    ids = np.asarray(ids)
    N = len(proj_pts)
    if N < 2:
        return dict(f_correct=np.nan, f_ambiguous=np.nan, f_rejected=np.nan, f_silent=np.nan, n_features=N)
    A, B = projector_tracks(proj_pts, cfg, *search_window(cfg, depth))
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)
    ts, w = depth_weights(depth, z_samples, n_t)
    frac = np.zeros(4)
    dz_sum = dz_abs_sum = w_silent = 0.0
    dz_all: list = []; w_all: list = []
    shift = np.array([float(du_px), float(dv_px)])
    table = _neighbour_table(A, B, t_margin, tol, extra_px=float(np.hypot(*shift))) if fast else None
    for k, t in enumerate(ts):
        P = A + t * AB + shift
        cnt = np.zeros(4)
        if table is not None:
            dist, tt = _seg_dist_table(P, table, A, AB, L2, t_margin)
            Jc = np.where(table >= 0, table, 0)
            if classes is not None:
                dist = np.where(classes[Jc] != classes[:, None], np.inf, dist)
            best = dist.argmin(axis=1)
            rows = np.arange(N)
            bd = dist[rows, best]
            jb = np.where(np.isfinite(bd), Jc[rows, best], rows)
            d2 = np.where(ids[Jc] == ids[jb][:, None], np.inf, dist).min(axis=1)
            rej = bd > tol
            amb = ~rej & (d2 <= tol)
            own = ids[jb] == ids
            silent = ~rej & ~amb & ~own
            cnt += [np.sum(~rej & ~amb & own), np.sum(amb), np.sum(rej), np.sum(silent)]
            if silent.any():
                dz = (tt[rows, best][silent] - t) * depth
                dz_sum += w[k] * dz.sum() / N; dz_abs_sum += w[k] * np.abs(dz).sum() / N
                w_silent += w[k] * silent.sum() / N
                dz_all.append(dz); w_all.append(np.full(len(dz), w[k] / N))
            frac += w[k] * cnt / N
            continue
        for s in range(0, N, chunk):
            p = P[s:s + chunk]
            d = p[:, None, :] - A[None, :, :]
            tt = np.clip((d * AB[None, :, :]).sum(axis=-1) / L2[None, :], -t_margin, 1.0 + t_margin)
            dist = np.linalg.norm(p[:, None, :] - (A[None, :, :] + tt[..., None] * AB[None, :, :]), axis=-1)
            if classes is not None:
                dist = np.where(classes[None, :] != classes[s:s + chunk][:, None], np.inf, dist)
            best = dist.argmin(axis=1)
            rows = np.arange(len(p))
            bd = dist[rows, best]
            d2 = np.where(ids[None, :] == ids[best][:, None], np.inf, dist).min(axis=1)
            rej = bd > tol
            amb = ~rej & (d2 <= tol)
            own = ids[best] == ids[s:s + chunk]
            silent = ~rej & ~amb & ~own
            cnt += [np.sum(~rej & ~amb & own), np.sum(amb), np.sum(rej), np.sum(silent)]
            if silent.any():
                dz = (tt[rows, best][silent] - t) * depth
                dz_sum += w[k] * dz.sum() / N; dz_abs_sum += w[k] * np.abs(dz).sum() / N
                w_silent += w[k] * silent.sum() / N
                dz_all.append(dz); w_all.append(np.full(len(dz), w[k] / N))
        frac += w[k] * cnt / N
    dz_med = float("nan")
    if dz_all:
        dz_c = np.concatenate(dz_all); w_c = np.concatenate(w_all)
        o = np.argsort(dz_c); cw = np.cumsum(w_c[o])
        dz_med = float(dz_c[o][np.searchsorted(cw, 0.5 * cw[-1])])
    return dict(f_correct=float(frac[0]), f_ambiguous=float(frac[1]), f_rejected=float(frac[2]),
                f_silent=float(frac[3]), n_features=int(N), dv_px=float(dv_px), du_px=float(du_px),
                dz_silent_mm=float(dz_sum / w_silent) if w_silent > 0 else float("nan"),
                dz_silent_abs_mm=float(dz_abs_sum / w_silent) if w_silent > 0 else float("nan"),
                dz_silent_median_mm=dz_med)


def regular_lattice_ambiguity(spacing_mm: float, depth_range_mm: float, sensor_cfg: dict | None = None,
                              z_samples: np.ndarray | None = None) -> float:

    c = _resolve_sensor_cfg(sensor_cfg)
    mm_per_px = c["pixel_pitch_mm"] / c["focal_length_mm"] * c["working_dist_mm"]
    theta = 2.0 * math.atan(c["baseline_mm"] / 2.0 / c["working_dist_mm"])
    z_alias = spacing_mm / math.tan(theta)
    if z_samples is None:
        return float(min(1.0, 2.0 * max(0.0, 1.0 - z_alias / depth_range_mm)))
    z = np.asarray(z_samples, dtype=np.float64)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return float("nan")
    p_minus = float(np.mean(z <= depth_range_mm - z_alias))
    p_plus = float(np.mean(z >= z_alias))
    return float(min(1.0, p_minus + p_plus))
