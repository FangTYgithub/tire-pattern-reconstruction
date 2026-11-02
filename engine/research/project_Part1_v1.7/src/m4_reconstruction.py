# Detect image features, assign projector correspondences and triangulate 3D points. Study adapters
# override these generic defaults and select the point or continuous-line front end.

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


BINARY_THRESHOLD  = 0  # Fixed gray threshold; zero selects Otsu thresholding.
MIN_AREA_PX       = 4  # Minimum connected-component area [pixels].


MAX_AREA_PX       = 5000  # Maximum connected-component area for generic dot detection [pixels].
MAX_LINE_AREA_PX  = 40000  # Maximum connected-component area for generic line detection [pixels].


LINE_MIN_AREA_FRAC = 0.1  # Minimum line-component area relative to the median line area.

MIN_INTENSITY     = 10  # Minimum gray value used in generic feature extraction.


BLOB_AREA_CHECK   = True  # Enable the relative-area check in the generic dot front end.
BLOB_AREA_LO      = 0.5  # Lower accepted blob area / reference-area ratio.
BLOB_AREA_HI      = 1.5  # Upper accepted blob area / reference-area ratio.
BLOB_ELONG_MAX    = 2.5  # Maximum blob elongation ratio.
BLOB_AREA_MIN_N   = 8  # Minimum component count required by the relative-area check.


EXCLUDE_BORDER_BLOBS = True  # Reject components touching the selected image boundary.
BORDER_MARGIN_PX     = 3  # Width of the boundary exclusion band [px].


CORRESPONDENCE_METHOD = "epipolar"
EPI_TOL_PX        = 3.0  # Distance tolerance from a feature to a projector track [px]; fixed study value: 3.


EPI_AMBIG_TOL_PX  = 3.0  # Distance used to flag competing projector tracks [px]; fixed study value: 3.
EPI_T_MARGIN      = 0.02  # Allowed extension beyond either track endpoint, as a fraction of track length.
EPI_CHUNK         = 512  # Feature count per vectorized matching batch.
EPI_ONE_TO_ONE    = True  # Require unique projector assignments in the generic matching path.


SEARCH_MODE            = "volume"  # Use a common depth volume or the alternative configured search rule.
MEASUREMENT_VOLUME_MM  = 15.0  # Generic default depth span [mm]; study adapters use [-16, 4] mm.


IMAGE_SIZE_PX     = 512  # Nominal square image size [px].
FOCAL_LENGTH_MM   = 25.0  # Equivalent pinhole focal length [mm].
PIXEL_PITCH_MM    = 0.030  # Pixel pitch [mm/px].
BASELINE_MM       = 50.0  # Projector-camera baseline [mm].
WORKING_DIST_MM   = 150.0  # Nominal working distance [mm].


FX_PX = FOCAL_LENGTH_MM / PIXEL_PITCH_MM
CX_PX = (IMAGE_SIZE_PX - 1) / 2.0
CY_PX = (IMAGE_SIZE_PX - 1) / 2.0


PatternType = Literal["lines", "dots", "vlines", "random_dots", "graded_dots", "coded_dots", "dashed_vlines", "bright_dots",
                      "epi_random_dots", "olines"]
DOT_TYPES   = ("dots", "random_dots", "graded_dots", "coded_dots", "bright_dots", "epi_random_dots", "anchored_dots")
CODED_TYPES = ("coded_dots", "bright_dots")
ANCHORED_TYPES = ("anchored_dots",)
ANCHOR_TOL_PX     = 15.0
ANCHOR_MIN_USED   = 4
PROP_NEIGHBOURS   = 8
PROP_RADIUS_FACTOR = 4.0
PROP_Z_JUMP_MM    = 6.0
PROP_MAX_ITER     = 30
VLINE_TYPES = ("vlines", "dashed_vlines")


SIZE_UNCERTAIN_FRAC = 0.15  # Relative uncertainty band used when assigning size classes.


SIZE_DPRIME_MIN     = 3.0  # Minimum class-separation statistic for the generic code-readability check.
SIZE_SPAN_MIN_FRAC  = 0.5  # Minimum measured / designed log-area-span ratio.
SIZE_SPAN_MAX_FRAC  = 2.0  # Maximum measured / designed log-area-span ratio.


SIZE_BALANCE_MIN    = 0.1  # Minimum fraction represented by the smaller size class.


CODE_REF            = "global"
CODE_REF_NEIGHBOURS = 8  # Neighbour count for local class normalization.


# Detect camera features, match them to projector features and triangulate valid pairs.
def reconstruct(
    camera_image: np.ndarray,
    pattern:      np.ndarray,
    pattern_type: PatternType = "lines",
    sensor_cfg:   dict | None = None,
    method:       str = CORRESPONDENCE_METHOD,
) -> tuple[np.ndarray, dict]:

    if pattern_type == "olines":
        from m4_lines import reconstruct_olines
        return reconstruct_olines(camera_image, pattern, sensor_cfg=sensor_cfg, method=method)

    cfg    = _resolve_sensor_cfg(sensor_cfg)
    method = (method or CORRESPONDENCE_METHOD).lower()
    if method not in ("epipolar", "nn"):
        raise ValueError(f"unknown correspondence method '{method}' (use 'epipolar' or 'nn')")


    anchored = pattern_type in ANCHORED_TYPES
    dash_coded = pattern_type == "dashed_vlines" and int(cfg.get("dash_levels", 1) or 1) > 1
    size_levels = int(cfg.get("size_levels", 1) or 1) if (pattern_type in CODED_TYPES or anchored) else 1
    radii = cfg.get("size_radii_px") if (pattern_type == "coded_dots" or anchored) else None
    size_log_span = 2.0 * math.log(max(radii) / min(radii)) if radii and min(radii) > 0 else None
    if pattern_type == "bright_dots":
        lv = cfg.get("bright_levels")
        size_log_span = math.log(max(lv) / min(lv)) if lv and min(lv) > 0 else None
    if dash_coded:
        size_levels = int(cfg["dash_levels"])
        size_log_span = (size_levels - 1) * math.log(1.0 / float(cfg.get("dash_ratio", 0.5)))
    design_ratio = balance_min = None
    if anchored and radii:
        design_ratio = (max(radii) / min(radii)) ** 2
        every = cfg.get("anchor_every", (2, 4))
        balance_min = float(cfg.get("anchor_balance_min", 0.4 / max(1, int(every[0]) * int(every[1]) - 1)))
    cam_features, det_info = detect_features(
        camera_image, pattern_type, size_levels=size_levels, size_log_span=size_log_span,
        code_ref=str(cfg.get("code_ref", CODE_REF)), design_ratio=design_ratio, balance_min=balance_min,
    )
    n_detected = int(len(cam_features))

    cam_allowed = det_info.pop("feature_allowed", None)
    det_info.pop("feature_classes", None)
    det_info.pop("feature_areas_px", None)

    base_info = {**det_info, "correspondence_method": method, "n_detected": n_detected}

    if n_detected == 0:
        return np.empty((0, 3), dtype=np.float32), {
            **base_info, "n_points_3d": 0, "triangulation": "no features",
            "n_matched": 0, "n_rejected": 0, "n_ambiguous": 0,
            "ambiguity_ratio": float("nan"), "rejection_ratio": float("nan"),
            "corr_failure_ratio": float("nan"),
            "ambiguous_mask": np.zeros(0, dtype=bool),
        }


    if method == "epipolar" and anchored:
        proj_features, epi = _correspond_anchored(cam_features, cam_allowed, pattern, pattern_type, cfg)
        ambiguous_all = epi["ambiguous"]
    elif method == "epipolar":
        proj_features, epi = _correspond_epipolar(cam_features, pattern, pattern_type, cfg,
                                                  cam_allowed=cam_allowed)
        ambiguous_all = epi["ambiguous"]
    else:
        proj_features = find_correspondences(cam_features, pattern, pattern_type, cfg)
        epi = {}
        ambiguous_all = np.zeros(n_detected, dtype=bool)


    valid = ~np.any(np.isnan(proj_features), axis=1)
    cam_features  = cam_features[valid]
    proj_features = proj_features[valid]
    ambiguous     = ambiguous_all[valid]

    n_matched   = int(valid.sum())
    n_rejected  = n_detected - n_matched
    n_ambiguous = int(ambiguous.sum())
    corr_stats = {
        "n_matched":          n_matched,
        "n_rejected":         n_rejected,
        "n_ambiguous":        n_ambiguous,
        "ambiguity_ratio":    n_ambiguous / n_detected,
        "rejection_ratio":    n_rejected / n_detected,
        "corr_failure_ratio": (n_ambiguous + n_rejected) / n_detected,
    }
    epi_stats = {k: v for k, v in epi.items()
                 if k not in ("ambiguous", "matched", "best_idx", "best_dist",
                              "second_dist", "second_id", "t_best",
                              "ambiguous_second", "duplicate", "proj_classes", "propagated")}

    if n_matched == 0:
        return np.empty((0, 3), dtype=np.float32), {
            **base_info, **corr_stats, **epi_stats,
            "n_points_3d": 0, "triangulation": "no correspondences",
            "ambiguous_mask": np.zeros(0, dtype=bool),
        }


    points_3d = triangulate(cam_features, proj_features, cfg)
    finite    = ~np.isnan(points_3d).any(axis=1)

    info = {
        **base_info,
        **corr_stats,
        **epi_stats,
        "n_correspondences": n_matched,
        "n_points_3d":       len(points_3d),
        "n_points_finite":   int(finite.sum()),
        "depth_min_mm":      float(points_3d[finite, 2].min()) if finite.any() else float("nan"),
        "depth_max_mm":      float(points_3d[finite, 2].max()) if finite.any() else float("nan"),
        "depth_mean_mm":     float(points_3d[finite, 2].mean()) if finite.any() else float("nan"),
        "ambiguous_mask":    ambiguous,
    }
    if "t_best" in epi:
        info["epi_t_best"] = epi["t_best"][valid]
    if "duplicate" in epi:
        info["duplicate_mask"] = epi["duplicate"][valid]
    if "propagated" in epi:
        info["propagated_mask"] = epi["propagated"][valid]

    return points_3d.astype(np.float32), info


# Extract centres and optional class labels for the requested pattern family.
def detect_features(
    camera_image: np.ndarray,
    pattern_type: PatternType = "lines",
    size_levels:  int = 1,
    size_log_span: float | None = None,
    code_ref:     str = CODE_REF,
    design_ratio: float | None = None,
    balance_min:  float | None = None,
) -> tuple[np.ndarray, dict]:

    if camera_image.dtype != np.uint8:
        raise TypeError(
            f"camera_image must be uint8, got {camera_image.dtype}"
        )


    thresh_val = BINARY_THRESHOLD
    if thresh_val == 0:

        thresh_val, binary = cv2.threshold(
            camera_image, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            camera_image, thresh_val, 255, cv2.THRESH_BINARY
        )


    if pattern_type == "lines":
        features, branch_info = _detect_line_centres(camera_image, binary)
    elif pattern_type in DOT_TYPES:
        k = int(size_levels) if (pattern_type in CODED_TYPES or pattern_type in ANCHORED_TYPES) else 1
        features, branch_info = _detect_dot_centroids(camera_image, binary, size_levels=k,
                                                      size_log_span=size_log_span,
                                                      code_kind="brightness" if pattern_type == "bright_dots" else "size",
                                                      code_ref=code_ref, design_ratio=design_ratio, balance_min=balance_min)
    elif pattern_type in VLINE_TYPES:
        kd = int(size_levels) if pattern_type == "dashed_vlines" else 1
        features, branch_info = _detect_vertical_line_centres(camera_image, binary, dash_levels=kd,
                                                              dash_log_span=size_log_span if kd > 1 else None)
    else:
        raise ValueError(
            f"Unsupported pattern_type '{pattern_type}'. "
            "Use 'lines', 'dots', 'vlines', 'random_dots', 'graded_dots', 'coded_dots' or 'dashed_vlines'."
        )

    info = {
        "pattern_type":    pattern_type,
        "otsu_threshold":  float(thresh_val),
        "n_features":      len(features),
        **branch_info,
    }

    return features.astype(np.float32), info


def _area_weighted_median(areas) -> float:

    a = np.sort(np.asarray(areas, dtype=np.float64))
    if a.size == 0:
        return float("nan")
    c = np.cumsum(a)
    return float(a[int(np.searchsorted(c, 0.5 * c[-1]))])


def _line_labels(label_map: np.ndarray, n_labels: int) -> tuple[set, int]:

    areas = np.bincount(label_map.ravel(), minlength=n_labels)
    cand = [lbl for lbl in range(1, n_labels) if MIN_AREA_PX <= areas[lbl] <= MAX_LINE_AREA_PX]
    if not cand:
        return set(), 0
    ref = _area_weighted_median(areas[cand])
    valid = {lbl for lbl in cand if areas[lbl] >= LINE_MIN_AREA_FRAC * ref}
    return valid, len(cand) - len(valid)


# Detect horizontal line centres by column-wise intensity-weighted centroid.
def _detect_line_centres(
    camera_image: np.ndarray,
    binary:       np.ndarray,
) -> tuple[np.ndarray, dict]:

    H, W = camera_image.shape
    features = []


    n_labels, label_map = cv2.connectedComponents(binary)


    valid_labels, n_specks = _line_labels(label_map, n_labels)

    if not valid_labels:
        return np.empty((0, 2), dtype=np.float32), {"n_lines": 0, "n_line_specks": n_specks}


    clean_binary = np.where(np.isin(label_map, list(valid_labels)), 255, 0).astype(binary.dtype)


    for u in range(W):
        col_binary = clean_binary[:, u]
        col_image  = camera_image[:, u].astype(np.float64)


        bright_rows = np.where(col_binary > 0)[0]
        if len(bright_rows) == 0:
            continue


        groups = _group_consecutive(bright_rows)

        for group in groups:

            if EXCLUDE_BORDER_BLOBS and (group[0] <= BORDER_MARGIN_PX or group[-1] >= H - 1 - BORDER_MARGIN_PX):
                continue
            weights = col_image[group]
            if weights.sum() < 1e-6:
                continue
            v_centre = float(np.average(group, weights=weights))
            features.append([float(u), v_centre])

    arr = np.array(features, dtype=np.float32) if features else \
          np.empty((0, 2), dtype=np.float32)

    return arr, {"n_lines": len(valid_labels)}


def _touches_border(stat_row: np.ndarray, W: int, H: int, sides: str = "any") -> bool:

    left, top, w, h = (int(stat_row[cv2.CC_STAT_LEFT]), int(stat_row[cv2.CC_STAT_TOP]),
                       int(stat_row[cv2.CC_STAT_WIDTH]), int(stat_row[cv2.CC_STAT_HEIGHT]))
    m  = BORDER_MARGIN_PX
    lr = left <= m or left + w >= W - m
    tb = top <= m or top + h >= H - m
    if sides == "lr":
        return lr
    if sides == "tb":
        return tb
    return lr or tb


def _elongation(M: dict) -> float:

    m00 = M["m00"]
    if m00 < 1e-9:
        return 1.0
    mu20, mu02, mu11 = M["mu20"] / m00, M["mu02"] / m00, M["mu11"] / m00
    tr, det = mu20 + mu02, mu20 * mu02 - mu11 * mu11
    disc = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    return float(l1 / l2) if l2 > 1e-9 else float("inf")


def _kmeans_1d(x: np.ndarray, k: int, n_iter: int = 50) -> np.ndarray:

    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.size == 0:
        return np.full(k, np.nan)
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(n_iter):
        lab = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
        new = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return np.sort(c)


def _size_classes(log_area: np.ndarray, centres: np.ndarray,
                  frac: float = SIZE_UNCERTAIN_FRAC) -> tuple[np.ndarray, np.ndarray]:

    la = np.asarray(log_area, dtype=np.float64)
    k = len(centres)
    cls = np.abs(la[:, None] - centres[None, :]).argmin(axis=1)
    allowed = np.zeros((len(la), k), dtype=bool)
    allowed[np.arange(len(la)), cls] = True
    for j in range(k - 1):
        b, gap = 0.5 * (centres[j] + centres[j + 1]), centres[j + 1] - centres[j]
        near = np.abs(la - b) <= frac * gap
        allowed[near, j] = True
        allowed[near, j + 1] = True
    return cls, allowed


# Filter connected components and compute intensity-weighted dot centres.
def _detect_dot_centroids(
    camera_image: np.ndarray,
    binary:       np.ndarray,
    size_levels:  int = 1,
    size_log_span: float | None = None,
    code_kind:    str = "size",
    code_ref:     str = CODE_REF,
    design_ratio: float | None = None,
    balance_min:  float | None = None,
) -> tuple[np.ndarray, dict]:

    n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(binary)
    H, W = binary.shape

    cands: list[tuple[float, float, int, float, float]] = []
    n_filtered = 0
    n_border   = 0

    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])


        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            n_filtered += 1
            continue


        if EXCLUDE_BORDER_BLOBS and _touches_border(stats[lbl], W, H, "any"):
            n_border += 1
            continue


        mask = (label_map == lbl).astype(np.uint8)
        mean_intensity = float(cv2.mean(camera_image, mask=mask)[0])
        if mean_intensity < MIN_INTENSITY:
            n_filtered += 1
            continue


        vals = camera_image[mask.astype(bool)]
        core_intensity = float(vals[vals >= np.median(vals)].mean()) if vals.size else mean_intensity


        masked = camera_image.astype(np.float32) * mask
        M = cv2.moments(masked)
        if M["m00"] < 1e-6:
            continue

        u_centre = M["m10"] / M["m00"]
        v_centre = M["m01"] / M["m00"]
        cands.append((u_centre, v_centre, area, _elongation(M), core_intensity))


    features: list[list[float]] = []
    n_area = n_elong = 0
    area_median = float("nan")
    k = max(1, int(size_levels))
    if cands and k > 1:
        if code_kind == "brightness":
            return _bright_dot_features(cands, k, n_filtered, n_border, size_log_span, code_ref)
        return _coded_dot_features(cands, k, n_filtered, n_border, size_log_span, code_ref, design_ratio, balance_min)
    if cands:
        area_median = _area_weighted_median([c[2] for c in cands])
        check = BLOB_AREA_CHECK and len(cands) >= BLOB_AREA_MIN_N
        for u_c, v_c, area, elong, _ in cands:
            if check and not (BLOB_AREA_LO * area_median <= area <= BLOB_AREA_HI * area_median):
                n_area += 1
                continue
            if check and area >= 4 and elong > BLOB_ELONG_MAX:
                n_elong += 1
                continue
            features.append([u_c, v_c])

    arr = np.array(features, dtype=np.float32) if features else \
          np.empty((0, 2), dtype=np.float32)

    info = {
        "n_dots":              len(features),
        "n_filtered_noise":    n_filtered,
        "n_border_excluded":   n_border,
        "n_area_rejected":     n_area,
        "n_elong_rejected":    n_elong,
        "blob_area_median_px": round(area_median, 2) if np.isfinite(area_median) else float("nan"),
    }
    if k > 1:
        info.update({"feature_classes": np.zeros(0, np.int64), "feature_allowed": np.zeros((0, k), bool),
                     "feature_areas_px": np.zeros(0)})
    return arr, info


def _local_code_split(xy: np.ndarray, lf: np.ndarray, n_neigh: int = CODE_REF_NEIGHBOURS):

    from scipy.spatial import cKDTree
    n = len(lf)
    kq = min(n_neigh + 1, n)
    _, j = cKDTree(xy).query(xy, k=kq)
    j = np.atleast_2d(j)
    if kq > 4:
        nb = np.sort(lf[j[:, 1:]], axis=1)
        ref = 0.5 * (nb[:, :2].mean(axis=1) + nb[:, -2:].mean(axis=1))
    elif kq > 1:
        ref = np.median(lf[j[:, 1:]], axis=1)
    else:
        ref = lf
    c = lf - ref
    lo, hi = c[c <= 0], c[c > 0]
    centres = np.array([lo.mean() if lo.size else -1e-6, hi.mean() if hi.size else 1e-6])
    cls, allowed = _size_classes(c, centres)
    return c, centres, cls, allowed


# Classify dot areas and apply class-readability and shape checks.
def _coded_dot_features(cands: list, k: int, n_filtered: int, n_border: int,
                        size_log_span: float | None = None, code_ref: str = CODE_REF,
                        design_ratio: float | None = None, balance_min: float | None = None) -> tuple[np.ndarray, dict]:

    areas = np.array([c[2] for c in cands], dtype=np.float64)
    la = np.log(areas)
    local = code_ref == "local" and k == 2 and len(cands) >= 3
    if local:
        xy = np.array([[c[0], c[1]] for c in cands], dtype=np.float64)
        la_c, centres, cls, allowed = _local_code_split(xy, la)
    elif design_ratio and k == 2:
        ref = _area_weighted_median(areas)
        centres = np.array([math.log(ref), math.log(ref * float(design_ratio))])
        cls, allowed = _size_classes(la, centres)
        la_c = la
    else:
        centres = _kmeans_1d(la, k)
        cls, allowed = _size_classes(la, centres)
        la_c = la
    refs = np.array([_area_weighted_median(areas[cls == j]) if (cls == j).any() else math.exp(centres[j])
                     for j in range(k)])
    check = BLOB_AREA_CHECK and len(cands) >= BLOB_AREA_MIN_N
    keep = np.ones(len(cands), dtype=bool)
    n_area = n_elong = 0
    for i, (_, _, area, elong, *_rest) in enumerate(cands):
        if check and (area < BLOB_AREA_LO * refs[0] or area > BLOB_AREA_HI * refs[-1]):
            keep[i] = False; n_area += 1; continue
        if check and area >= 4 and elong > BLOB_ELONG_MAX:
            keep[i] = False; n_elong += 1; continue
    feats = np.array([[c[0], c[1]] for c in cands], dtype=np.float32)[keep]
    la_k, cls_k = la_c[keep], cls[keep]

    lo, hi = la_k[cls_k == 0], la_k[cls_k == k - 1]
    pooled = math.sqrt(0.5 * (lo.var() + hi.var())) if (lo.size > 1 and hi.size > 1) else float("nan")
    dprime = float((hi.mean() - lo.mean()) / pooled) if pooled and pooled > 0 else float("nan")

    counts = np.bincount(cls_k, minlength=k)
    balance = float(counts.min() / counts.max()) if counts.max() > 0 else 0.0
    span = float(centres[-1] - centres[0])
    readable = bool(np.isfinite(dprime) and dprime >= SIZE_DPRIME_MIN
                    and balance >= (float(balance_min) if balance_min is not None else SIZE_BALANCE_MIN)
                    and (not size_log_span or SIZE_SPAN_MIN_FRAC * float(size_log_span) <= span
                         <= SIZE_SPAN_MAX_FRAC * float(size_log_span)))
    allowed_k = allowed[keep]
    if not readable:
        allowed_k = np.ones_like(allowed_k)
    return feats.reshape(-1, 2), {
        "n_dots":              int(keep.sum()),
        "n_filtered_noise":    n_filtered,
        "n_border_excluded":   n_border,
        "n_area_rejected":     n_area,
        "n_elong_rejected":    n_elong,
        "blob_area_median_px": round(float(refs[-1]), 2),
        "size_class_area_px":  " / ".join(f"{math.exp(c):.1f}" for c in (centres if not local else np.log(refs))),
        "size_class_ratio":    round(float(math.exp(centres[-1] - centres[0])), 4),
        "size_class_dprime":   round(dprime, 3) if np.isfinite(dprime) else float("nan"),
        "size_class_balance":  round(balance, 3),
        "size_code_readable":  int(readable),
        "n_size_uncertain":    int((allowed_k.sum(axis=1) > 1).sum()),
        "code_ref":            "local" if local else "global",
        "feature_classes":     cls_k.astype(np.int64),
        "feature_allowed":     allowed_k,
        "feature_areas_px":    areas[keep],
    }


# Classify dot intensities and reject unreadable class splits.
def _bright_dot_features(cands: list, k: int, n_filtered: int, n_border: int,
                         log_span: float | None = None, code_ref: str = CODE_REF) -> tuple[np.ndarray, dict]:

    areas = np.array([c[2] for c in cands], dtype=np.float64)
    inten = np.array([c[4] for c in cands], dtype=np.float64)
    li = np.log(np.maximum(inten, 1.0))
    local = code_ref == "local" and k == 2 and len(cands) >= 3
    if local:
        xy = np.array([[c[0], c[1]] for c in cands], dtype=np.float64)
        li_c, centres, cls, allowed = _local_code_split(xy, li)
    else:
        centres = _kmeans_1d(li, k)
        cls, allowed = _size_classes(li, centres)
        li_c = li
    refs = np.array([_area_weighted_median(areas[cls == j]) if (cls == j).any() else float(np.median(areas))
                     for j in range(k)])
    check = BLOB_AREA_CHECK and len(cands) >= BLOB_AREA_MIN_N
    keep = np.ones(len(cands), dtype=bool)
    n_area = n_elong = 0
    for i, (_, _, area, elong, _i) in enumerate(cands):
        ref = refs[cls[i]]
        if check and not (BLOB_AREA_LO * ref <= area <= BLOB_AREA_HI * ref):
            keep[i] = False; n_area += 1; continue
        if check and area >= 4 and elong > BLOB_ELONG_MAX:
            keep[i] = False; n_elong += 1; continue
    feats = np.array([[c[0], c[1]] for c in cands], dtype=np.float32)[keep]
    li_k, cls_k = li_c[keep], cls[keep]
    lo, hi = li_k[cls_k == 0], li_k[cls_k == k - 1]
    pooled = math.sqrt(0.5 * (lo.var() + hi.var())) if (lo.size > 1 and hi.size > 1) else float("nan")
    dprime = float((hi.mean() - lo.mean()) / pooled) if pooled and pooled > 0 else float("nan")
    counts = np.bincount(cls_k, minlength=k)
    balance = float(counts.min() / counts.max()) if counts.max() > 0 else 0.0
    span = float(centres[-1] - centres[0])
    readable = bool(np.isfinite(dprime) and dprime >= SIZE_DPRIME_MIN and balance >= SIZE_BALANCE_MIN
                    and (not log_span or SIZE_SPAN_MIN_FRAC * float(log_span) <= span <= SIZE_SPAN_MAX_FRAC * float(log_span)))
    allowed_k = allowed[keep]
    if not readable:
        allowed_k = np.ones_like(allowed_k)
    return feats.reshape(-1, 2), {
        "n_dots":              int(keep.sum()),
        "n_filtered_noise":    n_filtered,
        "n_border_excluded":   n_border,
        "n_area_rejected":     n_area,
        "n_elong_rejected":    n_elong,
        "blob_area_median_px": round(float(refs[-1]), 2),
        "size_class_area_px":  " / ".join(f"{math.exp(c):.1f}" for c in centres),
        "size_class_ratio":    round(float(math.exp(centres[-1] - centres[0])), 4),
        "size_class_dprime":   round(dprime, 3) if np.isfinite(dprime) else float("nan"),
        "size_class_balance":  round(balance, 3),
        "size_code_readable":  int(readable),
        "n_size_uncertain":    int((allowed_k.sum(axis=1) > 1).sum()),
        "code_kind":           "brightness",
        "code_ref":            "local" if local else "global",
        "feature_classes":     cls_k.astype(np.int64),
        "feature_allowed":     allowed_k,
        "feature_areas_px":    areas[keep],
    }


def _group_consecutive(rows: np.ndarray) -> list[np.ndarray]:

    if len(rows) == 0:
        return []
    groups = []
    start  = 0
    for i in range(1, len(rows)):
        if rows[i] != rows[i - 1] + 1:
            groups.append(rows[start:i])
            start = i
    groups.append(rows[start:])
    return groups


# Extract row-wise centres and segment descriptors from vertical line components.
def _detect_vertical_line_centres(
    camera_image: np.ndarray,
    binary:       np.ndarray,
    dash_levels:  int = 1,
    dash_log_span: float | None = None,
) -> tuple[np.ndarray, dict]:

    H, W = camera_image.shape
    features = []
    feat_lbl: list[int] = []
    n_border = 0

    n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(binary)


    valid_labels, n_specks = _line_labels(label_map, n_labels)

    if not valid_labels:
        return np.empty((0, 2), dtype=np.float32), {"n_vlines": 0, "n_border_excluded": 0, "n_line_specks": n_specks}

    clean_binary = np.where(np.isin(label_map, list(valid_labels)), 255, 0).astype(binary.dtype)


    end_rows: set = set()
    if int(dash_levels) > 1:
        for lbl in valid_labels:
            top, h = int(stats[lbl, cv2.CC_STAT_TOP]), int(stats[lbl, cv2.CC_STAT_HEIGHT])
            end_rows.add((lbl, top)); end_rows.add((lbl, top + h - 1))

    for v in range(H):
        row_binary = clean_binary[v, :]
        row_image  = camera_image[v, :].astype(np.float64)

        bright_cols = np.where(row_binary > 0)[0]
        if len(bright_cols) == 0:
            continue

        groups = _group_consecutive(bright_cols)

        for group in groups:

            if EXCLUDE_BORDER_BLOBS and (group[0] <= BORDER_MARGIN_PX or group[-1] >= W - 1 - BORDER_MARGIN_PX):
                n_border += 1
                continue
            lbl = int(label_map[v, int(group[0])])
            if end_rows and (lbl, v) in end_rows:
                continue
            weights = row_image[group]
            if weights.sum() < 1e-6:
                continue
            u_centre = float(np.average(group, weights=weights))
            features.append([u_centre, float(v)])
            feat_lbl.append(lbl)

    arr = np.array(features, dtype=np.float32) if features else \
          np.empty((0, 2), dtype=np.float32)

    info = {"n_vlines": len(valid_labels), "n_border_excluded": n_border, "n_line_specks": n_specks}
    k = int(dash_levels)
    if k > 1 and len(features):
        lbls = np.asarray(feat_lbl)
        comps = np.array(sorted(set(lbls.tolist())))
        heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
        tops = stats[:, cv2.CC_STAT_TOP].astype(np.int64)
        truncated = (tops <= BORDER_MARGIN_PX) | (tops + heights.astype(np.int64) >= H - 1 - BORDER_MARGIN_PX)
        full = comps[~truncated[comps]]
        lh = np.log(np.maximum(heights, 1.0))
        if len(full) >= 2 * k:
            centres = _kmeans_1d(lh[full], k)
            c_cls, c_allowed = _size_classes(lh[full], centres)
            lo, hi = lh[full][c_cls == 0], lh[full][c_cls == k - 1]
            pooled = math.sqrt(0.5 * (lo.var() + hi.var())) if (lo.size > 1 and hi.size > 1) else float("nan")
            dprime = float((hi.mean() - lo.mean()) / pooled) if pooled and pooled > 0 else float("nan")
            counts = np.bincount(c_cls, minlength=k)
            balance = float(counts.min() / counts.max()) if counts.max() > 0 else 0.0
            span = float(centres[-1] - centres[0])
            readable = bool(np.isfinite(dprime) and dprime >= SIZE_DPRIME_MIN and balance >= SIZE_BALANCE_MIN
                            and (not dash_log_span or SIZE_SPAN_MIN_FRAC * float(dash_log_span) <= span
                                 <= SIZE_SPAN_MAX_FRAC * float(dash_log_span)))
        else:
            centres = np.full(k, np.nan); dprime = float("nan"); balance = 0.0; span = float("nan"); readable = False
        cls_map = {int(c): int(cc) for c, cc in zip(full, c_cls)} if readable else {}
        allowed_map = {int(c): a for c, a in zip(full, c_allowed)} if readable else {}
        n_trunc_resolved = 0
        if readable:


            lefts = stats[:, cv2.CC_STAT_LEFT].astype(np.float64); widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
            cxs = lefts + 0.5 * widths
            bounds = 0.5 * (centres[1:] + centres[:-1]) + SIZE_UNCERTAIN_FRAC * np.diff(centres)
            for c in comps[truncated[comps]]:
                above = int(np.searchsorted(bounds, lh[c]))
                if above >= 1 and k == 2:
                    cls_map[int(c)] = k - 1; allowed_map[int(c)] = np.eye(k, dtype=bool)[k - 1]; n_trunc_resolved += 1
                    continue
                near = [f for f in full if abs(cxs[f] - cxs[c]) <= max(widths[c], widths[f]) + 4
                        and (tops[f] - (tops[c] + heights[c]) <= 15 or tops[c] - (tops[f] + heights[f]) <= 15)
                        and (tops[f] >= tops[c] + heights[c] - 1 or tops[f] + heights[f] <= tops[c] + 1)]
                if near:
                    votes = np.bincount([cls_map[int(f)] for f in near], minlength=k)
                    cc = int(votes.argmax())
                    cls_map[int(c)] = cc; allowed_map[int(c)] = np.eye(k, dtype=bool)[cc]; n_trunc_resolved += 1
        f_cls = np.array([cls_map.get(int(l), 0) for l in lbls], dtype=np.int64)
        f_allowed = np.array([allowed_map.get(int(l), np.ones(k, dtype=bool)) for l in lbls], dtype=bool)
        info.update({
            "size_class_area_px":  " / ".join(f"{math.exp(c):.1f}" for c in centres),
            "size_class_ratio":    round(float(math.exp(span)), 4) if np.isfinite(span) else float("nan"),
            "size_class_dprime":   round(dprime, 3) if np.isfinite(dprime) else float("nan"),
            "size_class_balance":  round(balance, 3),
            "size_code_readable":  int(readable),
            "n_size_uncertain":    int((f_allowed.sum(axis=1) > 1).sum()),
            "n_dash_components":   int(len(comps)),
            "n_dash_truncated":    int(truncated[comps].sum()),
            "n_dash_trunc_resolved": int(n_trunc_resolved),
            "code_kind":           "dash_length",
            "feature_classes":     f_cls,
            "feature_allowed":     f_allowed,
        })
    return arr, info


def _correspond_vlines(
    cam_features: np.ndarray,
    pattern:      np.ndarray,
    cfg:          dict,
) -> np.ndarray:

    H, W = pattern.shape
    N    = len(cam_features)

    pattern_binary = (pattern > 0.5).astype(np.uint8) * 255
    _, pat_binary  = cv2.threshold(pattern_binary, 127, 255, cv2.THRESH_BINARY)
    n_lbl, lbl_map = cv2.connectedComponents(pat_binary)

    line_centres_u = []
    for lbl in range(1, n_lbl):
        cols = np.where((lbl_map == lbl).any(axis=0))[0]
        if len(cols) > 0:
            line_centres_u.append(float(cols.mean()))
    line_centres_u = np.sort(line_centres_u)

    if len(line_centres_u) == 0:
        return np.full((N, 2), np.nan, dtype=np.float32)

    proj_features = np.full((N, 2), np.nan, dtype=np.float32)

    if len(line_centres_u) > 1:
        spacing_est = float(np.median(np.diff(line_centres_u)))
    else:
        spacing_est = float(line_centres_u[0]) if line_centres_u[0] > 0 else 64.0

    for i, (u_c, v_c) in enumerate(cam_features):
        dists = np.abs(line_centres_u - u_c)
        nearest_idx = int(np.argmin(dists))

        if dists[nearest_idx] > spacing_est * 0.75:
            continue

        u_p = line_centres_u[nearest_idx]
        v_p = v_c

        proj_features[i] = [float(u_p), float(v_p)]

    return proj_features


# Dispatch the pattern-specific camera-to-projector matching method.
def find_correspondences(
    cam_features: np.ndarray,
    pattern:      np.ndarray,
    pattern_type: PatternType,
    cfg:          dict,
) -> np.ndarray:

    N = len(cam_features)
    proj_features = np.full((N, 2), np.nan, dtype=np.float32)

    if pattern_type == "lines":
        proj_features = _correspond_lines(cam_features, pattern, cfg)
    elif pattern_type in DOT_TYPES:
        proj_features = _correspond_dots(cam_features, pattern, cfg)
    elif pattern_type == "vlines":
        proj_features = _correspond_vlines(cam_features, pattern, cfg)

    return proj_features


def _correspond_lines(
    cam_features: np.ndarray,
    pattern:      np.ndarray,
    cfg:          dict,
) -> np.ndarray:

    H, W = pattern.shape
    N    = len(cam_features)


    pattern_binary = (pattern > 0.5).astype(np.uint8) * 255
    _, pat_binary  = cv2.threshold(pattern_binary, 127, 255, cv2.THRESH_BINARY)
    n_lbl, lbl_map = cv2.connectedComponents(pat_binary)

    line_centres_v = []
    for lbl in range(1, n_lbl):
        rows = np.where((lbl_map == lbl).any(axis=1))[0]
        if len(rows) > 0:
            line_centres_v.append(float(rows.mean()))
    line_centres_v = np.sort(line_centres_v)

    if len(line_centres_v) == 0:
        return np.full((N, 2), np.nan, dtype=np.float32)

    proj_features = np.full((N, 2), np.nan, dtype=np.float32)


    if len(line_centres_v) > 1:
        spacing_est = float(np.median(np.diff(line_centres_v)))
    else:
        spacing_est = float(line_centres_v[0]) if line_centres_v else 64.0

    for i, (u_c, v_c) in enumerate(cam_features):


        dists = np.abs(line_centres_v - v_c)
        nearest_idx = int(np.argmin(dists))


        if dists[nearest_idx] > spacing_est * 0.75:
            continue

        v_p = line_centres_v[nearest_idx]


        u_p = u_c

        proj_features[i] = [float(u_p), float(v_p)]

    return proj_features


# Assign generic dot correspondences using the extracted projector lattice.
def _correspond_dots(
    cam_features: np.ndarray,
    pattern:      np.ndarray,
    cfg:          dict,
) -> np.ndarray:

    H, W = pattern.shape
    N    = len(cam_features)


    pat_u8 = (pattern * 255).astype(np.uint8)
    _, pat_bin = cv2.threshold(pat_u8, 127, 255, cv2.THRESH_BINARY)
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(pat_bin)

    dot_centres = []
    for lbl in range(1, n_lbl):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 2:
            continue
        mask = (lbl_map == lbl).astype(np.uint8)
        M = cv2.moments(mask)
        if M["m00"] < 1e-6:
            continue
        u_dot = M["m10"] / M["m00"]
        v_dot = M["m01"] / M["m00"]
        dot_centres.append([u_dot, v_dot])

    if not dot_centres:
        return np.full((N, 2), np.nan, dtype=np.float32)

    dot_centres_arr = np.array(dot_centres, dtype=np.float32)


    fx            = cfg["focal_length_mm"] / cfg["pixel_pitch_mm"]
    working_dist  = cfg["working_dist_mm"]


    if len(dot_centres_arr) > 1:
        all_u_sorted = np.sort(np.unique(np.round(dot_centres_arr[:, 0], 0)))
        all_v_sorted = np.sort(np.unique(np.round(dot_centres_arr[:, 1], 0)))
        if len(all_u_sorted) > 1:
            spacing_est = float(np.median(np.diff(all_u_sorted)))
        elif len(all_v_sorted) > 1:
            spacing_est = float(np.median(np.diff(all_v_sorted)))
        else:
            spacing_est = 30.0
    else:
        spacing_est = 30.0

    proj_features = np.full((N, 2), np.nan, dtype=np.float32)

    for i, (u_c, v_c) in enumerate(cam_features):


        dists = np.linalg.norm(dot_centres_arr - np.array([u_c, v_c]), axis=1)
        idx   = int(np.argmin(dists))


        if dists[idx] > spacing_est * 0.75:
            continue

        proj_features[i] = dot_centres_arr[idx]

    return proj_features


def _m3_cfg(cfg: dict) -> dict:

    return {
        "image_width_px":  cfg["image_size_px"],
        "image_height_px": cfg["image_size_px"],
        "focal_length_mm": cfg["focal_length_mm"],
        "pixel_pitch_mm":  cfg["pixel_pitch_mm"],
        "baseline_mm":     cfg["baseline_mm"],
        "working_dist_mm": cfg["working_dist_mm"],
        "noise_sigma": 0.0, "brightness": 255.0,
        "batch_size": 256, "intensity_thresh": 0.01,
    }


# Extract projector feature locations and optional area or intensity descriptors.
def pattern_features(
    pattern:      np.ndarray,
    pattern_type: PatternType,
    return_areas: bool = False,
    return_intensity: bool = False,
):

    pat = np.asarray(pattern, dtype=np.float64)
    if pat.max() <= 1.0:
        pat = pat * 255.0
    thresh = 2 if pattern_type == "bright_dots" else 127
    _, pat_bin = cv2.threshold(pat.astype(np.uint8), thresh, 255, cv2.THRESH_BINARY)
    n_lbl, lbl_map, stats, cents = cv2.connectedComponentsWithStats(pat_bin)

    pts: list[list[float]] = []
    ids: list[int] = []
    if pattern_type in DOT_TYPES:
        for lbl in range(1, n_lbl):
            if stats[lbl, cv2.CC_STAT_AREA] < 1:
                continue
            pts.append([float(cents[lbl, 0]), float(cents[lbl, 1])])
            ids.append(lbl)
    elif pattern_type in VLINE_TYPES:
        for lbl in range(1, n_lbl):
            ys, xs = np.nonzero(lbl_map == lbl)
            for v in np.unique(ys):
                pts.append([float(xs[ys == v].mean()), float(v)])
                ids.append(lbl)
    elif pattern_type == "lines":
        for lbl in range(1, n_lbl):
            ys, xs = np.nonzero(lbl_map == lbl)
            for u in np.unique(xs):
                pts.append([float(u), float(ys[xs == u].mean())])
                ids.append(lbl)
    else:
        raise ValueError(f"unsupported pattern_type '{pattern_type}'")

    pts_a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ids_a = np.asarray(ids, dtype=np.int64)
    if return_intensity:
        sums = np.bincount(lbl_map.ravel(), weights=pat.ravel(), minlength=n_lbl)
        cnts = np.bincount(lbl_map.ravel(), minlength=n_lbl)
        inten = (sums[ids_a] / np.maximum(cnts[ids_a], 1) / 255.0) if len(ids_a) else np.zeros(0)
        return pts_a, ids_a, inten
    if return_areas:
        areas = stats[ids_a, cv2.CC_STAT_AREA].astype(np.float64) if len(ids_a) else np.zeros(0)
        return pts_a, ids_a, areas
    return pts_a, ids_a


# Assign projector size classes using the registered pattern settings.
def projector_size_classes(proj_areas: np.ndarray, cfg: dict) -> np.ndarray:

    la = np.log(np.maximum(np.asarray(proj_areas, dtype=np.float64), 1.0))
    radii = cfg.get("size_radii_px")
    if radii:
        centres = np.log(np.pi * np.asarray(sorted(radii), dtype=np.float64) ** 2)
    else:
        centres = _kmeans_1d(la, int(cfg.get("size_levels", 2) or 2))
    return np.abs(la[:, None] - centres[None, :]).argmin(axis=1).astype(np.int64)


# Assign intensity classes to projector dots.
def projector_bright_classes(proj_inten: np.ndarray, cfg: dict) -> np.ndarray:

    li = np.log(np.maximum(np.asarray(proj_inten, dtype=np.float64), 1e-3))
    levels = cfg.get("bright_levels")
    if levels:
        centres = np.log(np.asarray(sorted(levels), dtype=np.float64))
    else:
        centres = _kmeans_1d(li, int(cfg.get("size_levels", 2) or 2))
    return np.abs(li[:, None] - centres[None, :]).argmin(axis=1).astype(np.int64)


# Assign length classes to projector dash segments.
def projector_dash_classes(proj_pts: np.ndarray, ids: np.ndarray, pattern: np.ndarray, cfg: dict) -> np.ndarray:

    pat = np.asarray(pattern, dtype=np.float64)
    if pat.max() <= 1.0:
        pat = pat * 255.0
    _, pat_bin = cv2.threshold(pat.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    n_lbl, _, stats, cents = cv2.connectedComponentsWithStats(pat_bin)
    H = pat.shape[0]
    lengths = np.asarray(cfg.get("dash_lengths_px") or [], dtype=np.float64)
    k = int(cfg.get("dash_levels", len(lengths) or 2))
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    tops = stats[:, cv2.CC_STAT_TOP].astype(np.int64)
    truncated = (tops <= 0) | (tops + heights.astype(np.int64) >= H)
    if len(lengths) == k and k > 1:                        # Class 0: shortest segment.
        cl = np.abs(np.log(np.maximum(heights, 1.0))[:, None] - np.log(np.sort(lengths))[None, :]).argmin(axis=1)
    else:
        cl = np.abs(np.log(np.maximum(heights, 1.0))[:, None] - _kmeans_1d(np.log(np.maximum(heights[1:], 1.0)), k)[None, :]).argmin(axis=1)
    col = np.round(cents[:, 0]).astype(int)
    line_cls: dict = {}
    for lbl in range(1, n_lbl):
        if not truncated[lbl]:
            line_cls.setdefault(col[lbl], []).append(int(cl[lbl]))
    for lbl in range(1, n_lbl):
        if truncated[lbl]:
            votes = line_cls.get(col[lbl]) or [v for c in range(col[lbl] - 2, col[lbl] + 3) for v in line_cls.get(c, [])]
            if votes:
                cl[lbl] = int(np.bincount(votes).argmax())
    return cl[np.asarray(ids, dtype=np.int64)].astype(np.int64)


def _estimate_spacing_px(pts: np.ndarray, ids: np.ndarray, pattern_type: PatternType) -> float:

    if len(pts) < 2:
        return 30.0
    if pattern_type in DOT_TYPES:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(pts).query(pts, k=2)
        return float(np.median(d[:, 1]))
    axis = 0 if pattern_type in VLINE_TYPES else 1
    uniq = np.unique(ids)
    centres = np.sort([pts[ids == i, axis].mean() for i in uniq])
    if pattern_type == "dashed_vlines":
        centres = np.unique(np.round(centres, 1))
    return float(np.median(np.diff(centres))) if len(centres) > 1 else 30.0


# Resolve the lower and upper search depths from the effective configuration.
def search_window(cfg: dict, depth_mm: float) -> tuple[float, float]:

    z0 = float(cfg.get("search_z_min_mm", 0.0) or 0.0)
    return z0, z0 + float(depth_mm)


# Project feature rays into the camera at the two search-depth limits.
def projector_tracks(
    proj_pts: np.ndarray,
    cfg:      dict,
    z_min:    float = 0.0,
    z_max:    float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:

    from m3_virtual_camera import build_intrinsics, build_extrinsics, project_to_camera
    m3  = _m3_cfg(cfg)
    K   = build_intrinsics(m3)
    proj_ext, cam_ext = build_extrinsics(m3)
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]

    u, v   = proj_pts[:, 0], proj_pts[:, 1]
    d_sens = np.stack([(u - cx) / fx, (v - cy) / fx, np.ones_like(u)], axis=1)
    d_w    = (proj_ext["R_inv"] @ d_sens.T).T
    o      = proj_ext["center"]

    def _at_plane(z: float) -> np.ndarray:
        t = (z - o[2]) / d_w[:, 2]
        return o + t[:, None] * d_w

    W, H = m3["image_width_px"], m3["image_height_px"]
    A, _ = project_to_camera(_at_plane(z_min), K, cam_ext, W, H)
    B, _ = project_to_camera(_at_plane(z_max), K, cam_ext, W, H)
    return A, B


# Match camera centres to projector tracks within the distance tolerance. Competing tracks and
# optional class constraints determine acceptance.
def match_epipolar(
    cam_feats:    np.ndarray,
    A:            np.ndarray,
    B:            np.ndarray,
    ids:          np.ndarray,
    tol_px:       float = EPI_TOL_PX,
    ambig_tol_px: float = EPI_AMBIG_TOL_PX,
    t_margin:     float = EPI_T_MARGIN,
    chunk:        int   = EPI_CHUNK,
    allowed:      np.ndarray | None = None,
    proj_classes: np.ndarray | None = None,
) -> dict:

    cam = np.asarray(cam_feats, dtype=np.float64)
    N, M = len(cam), len(A)
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)
    coded = allowed is not None and proj_classes is not None

    best_idx    = np.full(N, -1, dtype=np.int64)
    best_dist   = np.full(N, np.inf)
    second_dist = np.full(N, np.inf)
    second_id   = np.full(N, -1, dtype=np.int64)
    t_best      = np.full(N, np.nan)

    for s in range(0, N, chunk):
        c   = cam[s:s + chunk]
        d   = c[:, None, :] - A[None, :, :]
        t   = (d * AB[None, :, :]).sum(axis=-1) / L2[None, :]
        tc  = np.clip(t, -t_margin, 1.0 + t_margin)
        closest = A[None, :, :] + tc[..., None] * AB[None, :, :]
        dist = np.linalg.norm(c[:, None, :] - closest, axis=-1)
        if coded:
            dist = np.where(allowed[s:s + chunk][:, proj_classes], dist, np.inf)

        bi = dist.argmin(axis=1)
        rows = np.arange(len(c))
        bd = dist[rows, bi]
        same = ids[None, :] == ids[bi][:, None]
        dist_other = np.where(same, np.inf, dist)
        si = dist_other.argmin(axis=1)
        sd = dist_other[rows, si]

        best_idx[s:s + chunk]    = bi
        best_dist[s:s + chunk]   = bd
        second_dist[s:s + chunk] = sd
        second_id[s:s + chunk]   = np.where(np.isfinite(sd), ids[si], -1)
        t_best[s:s + chunk]      = tc[rows, bi]

    matched   = best_dist <= tol_px
    ambiguous = matched & (second_dist <= ambig_tol_px)
    return {
        "best_idx": best_idx, "best_dist": best_dist,
        "second_dist": second_dist, "second_id": second_id,
        "t_best": t_best, "matched": matched, "ambiguous": ambiguous,
    }


# Build projector tracks, apply class constraints and return accepted correspondences.
def _correspond_epipolar(
    cam_features: np.ndarray,
    pattern:      np.ndarray,
    pattern_type: PatternType,
    cfg:          dict,
    cam_allowed:  np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:

    N = len(cam_features)
    dash_coded = pattern_type == "dashed_vlines" and int(cfg.get("dash_levels", 1) or 1) > 1
    coded = (pattern_type in CODED_TYPES or dash_coded) and cam_allowed is not None and len(cam_allowed) == N
    if coded and pattern_type == "bright_dots":
        proj_pts, ids, proj_inten = pattern_features(pattern, pattern_type, return_intensity=True)
        proj_cls = projector_bright_classes(proj_inten, cfg)
    elif coded and dash_coded:
        proj_pts, ids = pattern_features(pattern, pattern_type)
        proj_cls = projector_dash_classes(proj_pts, ids, pattern, cfg)
    elif coded:
        proj_pts, ids, proj_areas = pattern_features(pattern, pattern_type, return_areas=True)
        proj_cls = projector_size_classes(proj_areas, cfg)
        if cam_allowed.shape[1] <= int(proj_cls.max(initial=0)):
            pad = np.zeros((N, int(proj_cls.max()) + 1 - cam_allowed.shape[1]), dtype=bool)
            cam_allowed = np.concatenate([cam_allowed, pad], axis=1)
    else:
        proj_pts, ids = pattern_features(pattern, pattern_type)
        proj_cls = None
    nan_out = np.full((N, 2), np.nan, dtype=np.float32)
    if len(proj_pts) == 0:
        return nan_out, {"matched": np.zeros(N, bool), "ambiguous": np.zeros(N, bool),
                         "search_depth_mm": float("nan"), "spacing_est_mm": float("nan")}


    tol_px       = float(cfg.get("epi_tol_px", EPI_TOL_PX))
    ambig_tol_px = float(cfg.get("epi_ambig_tol_px", EPI_AMBIG_TOL_PX))
    mm_per_px  = cfg["pixel_pitch_mm"] / cfg["focal_length_mm"] * cfg["working_dist_mm"]
    theta      = 2.0 * math.atan(cfg["baseline_mm"] / 2.0 / cfg["working_dist_mm"])
    spacing_px = _estimate_spacing_px(proj_pts, ids, pattern_type)
    spacing_mm = spacing_px * mm_per_px
    guard_px   = 2.0 * max(tol_px, ambig_tol_px)
    dz_design  = spacing_mm / math.tan(theta)
    mode = str(cfg.get("search_mode", SEARCH_MODE)).lower()
    if cfg.get("search_depth_mm"):
        search_depth = float(cfg["search_depth_mm"]); dz_source = "override"
    elif mode == "volume":
        search_depth = float(cfg.get("measurement_volume_mm", MEASUREMENT_VOLUME_MM))
        dz_source = "measurement_volume"
    elif mode == "period":
        search_depth = max(spacing_px - guard_px, 0.25 * spacing_px) * mm_per_px / math.tan(theta)
        dz_source = "spacing_over_tan_theta_minus_guard"
    else:
        raise ValueError(f"unknown search_mode '{mode}' (use 'volume' or 'period')")

    A, B = projector_tracks(proj_pts, cfg, *search_window(cfg, search_depth))
    m = match_epipolar(
        cam_features, A, B, ids,
        tol_px       = tol_px,
        ambig_tol_px = ambig_tol_px,
        t_margin     = float(cfg.get("epi_t_margin", EPI_T_MARGIN)),
        allowed      = cam_allowed if coded else None,
        proj_classes = proj_cls if coded else None,
    )


    one_to_one = bool(cfg.get("epi_one_to_one", EPI_ONE_TO_ONE))
    mt         = m["matched"]
    amb_second = m["ambiguous"].copy()
    dup        = np.zeros(N, dtype=bool)
    n_dup_feat = 0
    if one_to_one and mt.any():
        claimed = m["best_idx"][mt]
        counts  = np.bincount(claimed, minlength=len(proj_pts))
        dup[mt] = counts[claimed] >= 2
        n_dup_feat = int((counts >= 2).sum())
    m["ambiguous"]        = amb_second | dup
    m["ambiguous_second"] = amb_second
    m["duplicate"]        = dup

    proj_features = nan_out.copy()
    proj_features[mt] = proj_pts[m["best_idx"][mt]].astype(np.float32)

    bd = m["best_dist"][mt]
    sd = m["second_dist"][mt]
    info = {
        **m,
        "epi_one_to_one":           one_to_one,
        "n_ambiguous_second":       int(amb_second.sum()),
        "n_duplicate_claims":       int(dup.sum()),
        "n_duplicate_features":     n_dup_feat,
        "search_depth_mm":          round(float(search_depth), 4),
        "search_depth_source":      dz_source,
        "search_mode":              mode,
        "design_unique_depth_mm":   round(float(dz_design), 4),
        "guard_band_px":            float(guard_px),
        "spacing_est_mm":           round(float(spacing_mm), 4),
        "spacing_est_px":           round(float(spacing_px), 3),
        "epi_tol_px":               tol_px,
        "epi_n_proj_features":      int(len(proj_pts)),
        "epi_n_proj_ids":           int(len(np.unique(ids))),
        "epi_best_dist_mean_px":    round(float(bd.mean()), 4) if mt.any() else float("nan"),
        "epi_best_dist_p95_px":     round(float(np.percentile(bd, 95)), 4) if mt.any() else float("nan"),
        "epi_second_dist_median_px": round(float(np.median(sd[np.isfinite(sd)])), 4)
                                     if np.isfinite(sd).any() else float("nan"),
        "epi_t_mean":               round(float(np.nanmean(m["t_best"][mt])), 4) if mt.any() else float("nan"),
        "epi_t_max":                round(float(np.nanmax(m["t_best"][mt])), 4) if mt.any() else float("nan"),
    }
    if coded:
        info["proj_classes"] = proj_cls
        info["size_code_n_classes"] = int(proj_cls.max(initial=0)) + 1
        info["size_code_proj_counts"] = " / ".join(str(int((proj_cls == c).sum()))
                                                   for c in range(int(proj_cls.max(initial=0)) + 1))
    return proj_features, info


def _tracks_within(cam: np.ndarray, A: np.ndarray, B: np.ndarray, tol_px: float, t_margin: float,
                   allowed: np.ndarray | None = None, proj_classes: np.ndarray | None = None,
                   chunk: int = EPI_CHUNK) -> list:

    AB = B - A
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)
    out: list = []
    for s in range(0, len(cam), chunk):
        c = cam[s:s + chunk]
        d = c[:, None, :] - A[None, :, :]
        t = (d * AB[None, :, :]).sum(axis=-1) / L2[None, :]
        tc = np.clip(t, -t_margin, 1.0 + t_margin)
        dist = np.linalg.norm(c[:, None, :] - (A[None, :, :] + tc[..., None] * AB[None, :, :]), axis=-1)
        if allowed is not None and proj_classes is not None:
            dist = np.where(allowed[s:s + chunk][:, proj_classes], dist, np.inf)
        within = dist <= tol_px
        for i in range(len(c)):
            js = np.nonzero(within[i])[0]
            out.append((js, tc[i, js]))
    return out


# Propagate accepted depth estimates to nearby ambiguous features under the registered neighbourhood
# rules.
def _propagate_depth(cam: np.ndarray, best_idx: np.ndarray, t_best: np.ndarray, matched: np.ndarray,
                     ambiguous: np.ndarray, cands: list, depth_mm: float, pitch_px: float) -> np.ndarray:

    from scipy.spatial import cKDTree
    resolved = matched & ~ambiguous
    propagated = np.zeros(len(cam), dtype=bool)
    todo = np.nonzero(matched & ambiguous)[0]
    if len(todo) == 0 or resolved.sum() < 3:
        return propagated
    tree = cKDTree(cam)
    radius = PROP_RADIUS_FACTOR * float(pitch_px)
    for _ in range(PROP_MAX_ITER):
        changed = False
        for q in todo:
            if not ambiguous[q]:
                continue
            js, ts = cands[q]
            if len(js) == 0:
                continue
            nb = [j for j in tree.query_ball_point(cam[q], radius) if j != q and resolved[j]]
            if len(nb) < 3:
                continue
            nb = sorted(nb, key=lambda j: float(np.hypot(*(cam[j] - cam[q]))))[:PROP_NEIGHBOURS]
            z_ref = float(np.median(t_best[nb])) * depth_mm
            zc = ts * depth_mm
            k = int(np.argmin(np.abs(zc - z_ref)))
            gap = float(np.min(np.diff(np.sort(zc)))) if len(zc) > 1 else np.inf
            if abs(zc[k] - z_ref) <= min(0.5 * gap, PROP_Z_JUMP_MM):
                best_idx[q], t_best[q] = js[k], ts[k]
                ambiguous[q] = False; resolved[q] = True; propagated[q] = True
                changed = True
        if not changed:
            break
    return propagated


# Use anchor correspondences to guide matching of the surrounding dot lattice.
def _correspond_anchored(
    cam_features: np.ndarray,
    cam_allowed:  np.ndarray | None,
    pattern:      np.ndarray,
    pattern_type: PatternType,
    cfg:          dict,
) -> tuple[np.ndarray, dict]:

    N = len(cam_features)
    nan_out = np.full((N, 2), np.nan, dtype=np.float32)
    proj_pts, ids, proj_areas = pattern_features(pattern, pattern_type, return_areas=True)
    proj_cls = projector_size_classes(proj_areas, cfg)
    empty = {"matched": np.zeros(N, bool), "ambiguous": np.zeros(N, bool), "search_depth_mm": float("nan"),
             "spacing_est_mm": float("nan"), "drift_est_px": float("nan"), "n_anchor_used": 0, "anchor_readable": 0}
    if len(proj_pts) == 0 or N == 0:
        return nan_out, empty
    readable = cam_allowed is not None and len(cam_allowed) == N and cam_allowed.shape[1] >= 2 and bool((cam_allowed.sum(axis=1) < 2).any())
    tol_px       = float(cfg.get("epi_tol_px", EPI_TOL_PX))
    ambig_tol_px = float(cfg.get("epi_ambig_tol_px", EPI_AMBIG_TOL_PX))
    t_margin     = float(cfg.get("epi_t_margin", EPI_T_MARGIN))
    anchor_tol   = float(cfg.get("anchor_tol_px", ANCHOR_TOL_PX))
    propagate    = bool(cfg.get("propagate", True))
    mm_per_px  = cfg["pixel_pitch_mm"] / cfg["focal_length_mm"] * cfg["working_dist_mm"]
    theta      = 2.0 * math.atan(cfg["baseline_mm"] / 2.0 / cfg["working_dist_mm"])
    spacing_px = _estimate_spacing_px(proj_pts, ids, pattern_type)
    spacing_mm = spacing_px * mm_per_px
    if cfg.get("search_depth_mm"):
        search_depth = float(cfg["search_depth_mm"]); dz_source = "override"
    else:
        search_depth = float(cfg.get("measurement_volume_mm", MEASUREMENT_VOLUME_MM)); dz_source = "measurement_volume"
    A, B = projector_tracks(proj_pts, cfg, *search_window(cfg, search_depth))
    AB = B - A
    nrm = np.stack([-AB[:, 1], AB[:, 0]], axis=1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    nrm[nrm[:, 1] < 0] *= -1.0                                       # Normals point to +v (image rows).
    cam = np.asarray(cam_features, dtype=np.float64)


    drift_est = drift_mad = float("nan")
    n_anchor_cam = n_anchor_matched = n_used = 0
    is_anchor_track = proj_cls == 1
    if readable and is_anchor_track.any():
        conf_anchor = cam_allowed[:, 1] & ~cam_allowed[:, 0]
        n_anchor_cam = int(conf_anchor.sum())
        if n_anchor_cam:
            ia = np.nonzero(is_anchor_track)[0]
            ma = match_epipolar(cam[conf_anchor], A[ia], B[ia], ids[ia], tol_px=anchor_tol,
                                ambig_tol_px=anchor_tol, t_margin=t_margin)
            used = ma["matched"] & ~ma["ambiguous"]
            n_anchor_matched = int(ma["matched"].sum()); n_used = int(used.sum())
            if n_used >= ANCHOR_MIN_USED:
                bi = ia[ma["best_idx"][used]]
                closest = A[bi] + ma["t_best"][used][:, None] * AB[bi]
                res = ((cam[conf_anchor][used] - closest) * nrm[bi]).sum(axis=1)
                drift_est = float(np.median(res))
                drift_mad = float(np.median(np.abs(res - drift_est)))
    cam_corr = cam.copy()
    if np.isfinite(drift_est):
        n_mean = nrm[is_anchor_track].mean(axis=0) if is_anchor_track.any() else nrm.mean(axis=0)
        n_mean /= max(np.linalg.norm(n_mean), 1e-12)
        cam_corr = cam - drift_est * n_mean[None, :]


    frame_valid = not (readable and is_anchor_track.any() and n_used < ANCHOR_MIN_USED and bool(cfg.get("anchor_require", True)))
    if not frame_valid:
        z = np.zeros(N, dtype=bool)
        return nan_out, {**empty, "anchor_readable": 1, "n_anchor_cam": n_anchor_cam, "n_anchor_matched": n_anchor_matched,
                         "n_anchor_used": n_used, "anchor_frame_valid": 0, "matched": z, "ambiguous": z,
                         "n_ambiguous_second": 0, "n_duplicate_claims": 0, "search_depth_mm": round(float(search_depth), 4),
                         "spacing_est_mm": round(float(spacing_mm), 4), "epi_tol_px": tol_px}


    allowed = cam_allowed if readable else None
    m = match_epipolar(cam_corr, A, B, ids, tol_px=tol_px, ambig_tol_px=ambig_tol_px, t_margin=t_margin,
                       allowed=allowed, proj_classes=proj_cls if readable else None)
    propagated = np.zeros(N, dtype=bool)
    if propagate and (m["matched"] & m["ambiguous"]).any():
        cands = _tracks_within(cam_corr, A, B, ambig_tol_px, t_margin, allowed, proj_cls if readable else None)

        pitch_cam = float(spacing_px) * float(np.median(np.linalg.norm(AB, axis=1))) / max(float(search_depth), 1e-9) / 1.6
        propagated = _propagate_depth(cam_corr, m["best_idx"], m["t_best"], m["matched"], m["ambiguous"], cands,
                                      float(search_depth), max(pitch_cam, 8.0))


    one_to_one = bool(cfg.get("epi_one_to_one", EPI_ONE_TO_ONE))
    mt = m["matched"]
    amb_second = m["ambiguous"].copy()
    dup = np.zeros(N, dtype=bool)
    n_dup_feat = 0
    if one_to_one and mt.any():
        claimed = m["best_idx"][mt]
        counts = np.bincount(claimed, minlength=len(proj_pts))
        dup[mt] = counts[claimed] >= 2
        n_dup_feat = int((counts >= 2).sum())
    m["ambiguous"] = amb_second | dup
    m["ambiguous_second"] = amb_second
    m["duplicate"] = dup
    m["propagated"] = propagated

    proj_features = nan_out.copy()
    proj_features[mt] = proj_pts[m["best_idx"][mt]].astype(np.float32)
    bd = m["best_dist"][mt]; sd = m["second_dist"][mt]
    info = {
        **m,
        "epi_one_to_one":           one_to_one,
        "n_ambiguous_second":       int(amb_second.sum()),
        "n_duplicate_claims":       int(dup.sum()),
        "n_duplicate_features":     n_dup_feat,
        "search_depth_mm":          round(float(search_depth), 4),
        "search_depth_source":      dz_source,
        "search_mode":              "volume",
        "design_unique_depth_mm":   round(float(spacing_mm / math.tan(theta)), 4),
        "guard_band_px":            2.0 * max(tol_px, ambig_tol_px),
        "spacing_est_mm":           round(float(spacing_mm), 4),
        "spacing_est_px":           round(float(spacing_px), 3),
        "epi_tol_px":               tol_px,
        "epi_n_proj_features":      int(len(proj_pts)),
        "epi_n_proj_ids":           int(len(np.unique(ids))),
        "epi_best_dist_mean_px":    round(float(bd.mean()), 4) if mt.any() else float("nan"),
        "epi_best_dist_p95_px":     round(float(np.percentile(bd, 95)), 4) if mt.any() else float("nan"),
        "epi_second_dist_median_px": round(float(np.median(sd[np.isfinite(sd)])), 4) if np.isfinite(sd).any() else float("nan"),
        "epi_t_mean":               round(float(np.nanmean(m["t_best"][mt])), 4) if mt.any() else float("nan"),
        "epi_t_max":                round(float(np.nanmax(m["t_best"][mt])), 4) if mt.any() else float("nan"),
        "proj_classes":             proj_cls,
        "size_code_n_classes":      2,
        "size_code_proj_counts":    " / ".join(str(int((proj_cls == c).sum())) for c in range(2)),

        "anchor_readable":          int(readable),
        "anchor_tol_px":            anchor_tol,
        "n_anchor_cam":             n_anchor_cam,
        "n_anchor_matched":         n_anchor_matched,
        "n_anchor_used":            n_used,
        "drift_est_px":             round(drift_est, 4) if np.isfinite(drift_est) else float("nan"),
        "drift_mad_px":             round(drift_mad, 4) if np.isfinite(drift_mad) else float("nan"),
        "n_propagated":             int(propagated.sum()),
        "n_unresolved":             int((mt & m["ambiguous"]).sum()),
        "propagate":                int(propagate),
        "anchor_frame_valid":       1,
    }
    return proj_features, info


# Recover 3D positions from matched camera and projector image coordinates.
def triangulate(
    cam_features:  np.ndarray,
    proj_features: np.ndarray,
    cfg:           dict,
) -> np.ndarray:

    fx = cfg["focal_length_mm"] / cfg["pixel_pitch_mm"]
    cx = (cfg["image_size_px"] - 1) / 2.0
    cy = (cfg["image_size_px"] - 1) / 2.0
    b  = cfg["baseline_mm"]
    z0 = cfg["working_dist_mm"]

    N = len(cam_features)


    O_cam  = np.array([0.0,  b / 2.0, -z0])
    O_proj = np.array([0.0, -b / 2.0, -z0])


    from m3_virtual_camera import build_extrinsics as _build_extrinsics
    _m3_cfg = {
        "image_width_px":  cfg["image_size_px"],
        "image_height_px": cfg["image_size_px"],
        "focal_length_mm": cfg["focal_length_mm"],
        "pixel_pitch_mm":  cfg["pixel_pitch_mm"],
        "baseline_mm":     cfg["baseline_mm"],
        "working_dist_mm": cfg["working_dist_mm"],
        "noise_sigma": 0.0, "brightness": 255.0,
        "batch_size": 256, "intensity_thresh": 0.01,
    }
    _proj_ext, _cam_ext = _build_extrinsics(_m3_cfg)


    R_cam  = _cam_ext["R"]
    R_proj = _proj_ext["R"]
    t_cam  = _cam_ext["t"]
    t_proj = _proj_ext["t"]


    O_cam  = -R_cam.T  @ t_cam
    O_proj = -R_proj.T @ t_proj

    def _rays(features: np.ndarray, R: np.ndarray) -> np.ndarray:

        u = features[:, 0]
        v = features[:, 1]
        # Sensor frame: z is the optical axis.
        dirs_sensor = np.stack([
            (u - cx) / fx,
            (v - cy) / fx,
            np.ones(len(u))
        ], axis=1)

        dirs_world = dirs_sensor @ R
        norms = np.linalg.norm(dirs_world, axis=1, keepdims=True)
        return dirs_world / norms

    d_cam  = _rays(cam_features,  R_cam)
    d_proj = _rays(proj_features, R_proj)


    w  = O_cam - O_proj

    a  = np.einsum("ni,ni->n", d_cam, d_proj)
    b1 = np.einsum("ni,i->n",  d_cam,  w)
    b2 = np.einsum("ni,i->n",  d_proj, w)

    denom = a * a - 1.0


    valid = np.abs(denom) > 1e-10
    s = np.where(valid, (a * b1 - b2) / np.where(valid, denom, 1.0), 0.0)
    t = np.where(valid, a * s - b1, 0.0)


    valid &= (t > 0) & (s > 0)

    P = O_cam  + t[:, None] * d_cam
    Q = O_proj + s[:, None] * d_proj

    points_3d = ((P + Q) / 2.0).astype(np.float32)


    points_3d[~valid] = np.nan

    return points_3d


# Merge supplied sensor settings with module defaults.
def _resolve_sensor_cfg(sensor_cfg: dict | None) -> dict:

    defaults = {
        "focal_length_mm": FOCAL_LENGTH_MM,
        "pixel_pitch_mm":  PIXEL_PITCH_MM,
        "baseline_mm":     BASELINE_MM,
        "working_dist_mm": WORKING_DIST_MM,
        "image_size_px":   IMAGE_SIZE_PX,
    }
    if sensor_cfg:
        defaults.update(sensor_cfg)
    return defaults


# Plot camera detections and the accepted 3D reconstruction.
def visualise_reconstruction(
    camera_image: np.ndarray,
    features:     np.ndarray,
    points_3d:    np.ndarray,
    ground_truth: np.ndarray | None = None,
    save_dir:     Path | None = None,
    show:         bool = False,
) -> None:

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("M4 Reconstruction", fontsize=12)


    ax = axes[0]
    ax.imshow(camera_image, cmap="gray", origin="upper")
    if len(features) > 0:
        ax.scatter(
            features[:, 0], features[:, 1],
            s=1, c="red", alpha=0.4, linewidths=0
        )
    ax.set_title(f"Camera image + features (N={len(features)})")
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")


    ax = axes[1]
    if len(points_3d) > 0:
        valid = ~np.isnan(points_3d).any(axis=1)
        pts   = points_3d[valid]
        if len(pts) > 0:
            ax.scatter(pts[:, 1], pts[:, 2], s=1, c="steelblue",
                       alpha=0.5, label="reconstructed")
        if ground_truth is not None:
            ax.scatter(ground_truth[:, 1], ground_truth[:, 2],
                       s=1, c="orange", alpha=0.3, label="ground truth (M1)")
            ax.legend(markerscale=5, fontsize=8)
    ax.set_title("YZ cross-section")
    ax.set_xlabel("Y [mm]")
    ax.set_ylabel("Z [mm]")
    ax.invert_yaxis()


    ax = axes[2]
    if len(points_3d) > 0:
        valid = ~np.isnan(points_3d).any(axis=1)
        pts   = points_3d[valid]
        if len(pts) > 0:
            sc = ax.scatter(
                pts[:, 0], pts[:, 1], c=pts[:, 2],
                s=2, cmap="plasma", alpha=0.6
            )
            plt.colorbar(sc, ax=ax, label="Z [mm]")
    ax.set_title("XY plane (colour = depth Z)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / "m4_reconstruction.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from m1_tire_surface    import build_surface
    from m2_pattern         import generate_pattern
    from m3_virtual_camera  import synthesise_camera_image

    ROOT    = Path(__file__).resolve().parents[1]
    OUT_FIG = ROOT / "outputs" / "figures"
    RAW_DIR = ROOT / "data" / "raw"

    print("=" * 60)
    print("M4 Reconstruction — Self-test")
    print("=" * 60)


    print("\n[M1] Building tire surface ...")
    mesh, depth_map, m1_cfg = build_surface(
        "A", raw_data_dir=RAW_DIR,
        grid_step_x_mm=5.0, grid_step_y_mm=2.0,
    )
    print(f"     grid: {m1_cfg['grid_shape_ny_nx']}")


    print("\n[M2] Generating line pattern (spacing=2.0 mm) ...")
    pattern, meta = generate_pattern("lines", spacing_mm=2.0, feature_size_mm=0.3)
    print(f"     density={meta['density']*100:.1f}%  "
          f"features={meta['feature_count']}")


    print("\n[M3] Synthesising camera image ...")
    cam_img, hit_map, hit_pts, m3_cfg = synthesise_camera_image(
        mesh_dict=mesh,
        pattern=pattern,
        cfg={"batch_size": 512},
    )
    print(f"     hits={m3_cfg['n_hits']}  "
          f"valid_px={m3_cfg['n_valid_pixels']}  "
          f"max_I={cam_img.max()}")


    print("\n[M4-Step1] Detecting features ...")
    features, det_info = detect_features(cam_img, "lines")
    print(f"     n_features  = {det_info['n_features']}")
    print(f"     n_lines     = {det_info.get('n_lines', '—')}")
    print(f"     otsu_thresh = {det_info['otsu_threshold']:.1f}")
    assert len(features) > 0, "ERROR: no features detected"
    assert features.dtype == np.float32, "ERROR: wrong dtype"
    assert features.shape[1] == 2, "ERROR: wrong shape"
    print("     ✅ feature detection passed")


    print("\n[M4-Step2] Finding correspondences ...")
    cfg = _resolve_sensor_cfg(None)
    proj_feats = find_correspondences(features, pattern, "lines", cfg)
    valid_corr = ~np.any(np.isnan(proj_feats), axis=1)
    print(f"     correspondences: {valid_corr.sum()} / {len(features)}")
    assert valid_corr.sum() > 0, "ERROR: no correspondences found"
    print("     ✅ correspondence passed")


    print("\n[M4-Step3] Triangulating ...")
    points_3d = triangulate(
        features[valid_corr], proj_feats[valid_corr], cfg
    )
    valid_pts = ~np.isnan(points_3d).any(axis=1)
    pts = points_3d[valid_pts]
    print(f"     3D points: {len(pts)}")
    if len(pts) > 0:
        print(f"     Z range:   {pts[:,2].min():.2f} ~ {pts[:,2].max():.2f} mm")
        print(f"     Y range:   {pts[:,1].min():.2f} ~ {pts[:,1].max():.2f} mm")
    assert len(pts) > 0, "ERROR: no valid 3D points"
    print("     ✅ triangulation passed")


    print("\n[M4-Full] Running reconstruct() ...")
    pts_full, info = reconstruct(cam_img, pattern, "lines")
    print(f"     n_points_3d = {info['n_points_3d']}")
    print(f"     depth range = {info['depth_min_mm']:.2f} ~ "
          f"{info['depth_max_mm']:.2f} mm")
    print("     ✅ full pipeline passed")


    print("\n[Visualise] Saving figure ...")
    visualise_reconstruction(
        cam_img, features, pts_full,
        save_dir=OUT_FIG,
    )


    print("\n[M4-Dots] Testing dot detection ...")
    pat_dots, _ = generate_pattern("dots", spacing_mm=2.0, feature_size_mm=0.15)

    dot_cam = (pat_dots * 200).astype(np.uint8)
    feats_d, info_d = detect_features(dot_cam, "dots")
    print(f"     detected dots = {info_d['n_dots']}")
    assert info_d["n_dots"] > 0, "ERROR: no dots detected"
    print("     ✅ dot detection passed")

    print("\n" + "=" * 60)
    print("M4 ✅ All tests passed")
    print("=" * 60)
