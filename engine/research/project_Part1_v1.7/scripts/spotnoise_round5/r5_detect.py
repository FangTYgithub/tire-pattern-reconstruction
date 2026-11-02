# Expose dot-detection stages and reconstruct from selected candidates.

import math
import numpy as np
import cv2

import r5_common as C
import m4_reconstruction as M4

CFG = M4._resolve_sensor_cfg({'measurement_volume_mm': C.SL.VOLUME_MM})
REASON = {0: 'candidate', 1: 'abs_area_small', 2: 'abs_area_large', 3: 'border', 4: 'intensity', 5: 'm00'}
SEL = {0: 'accepted', 1: 'rel_area_small', 2: 'rel_area_large', 3: 'elongation', 4: 'group_unavailable'}


def candidates(camera_image, thresh=None):

    if camera_image.dtype != np.uint8:
        raise TypeError('camera_image must be uint8')
    if thresh is None:
        thresh_val, binary = cv2.threshold(camera_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresh_val, binary = cv2.threshold(camera_image, float(thresh), 255, cv2.THRESH_BINARY)
    n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(binary)
    H, W = binary.shape
    reason = np.zeros(n_labels, np.int8); reason[0] = -1
    cands = []
    n_filtered = 0
    n_border = 0
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < M4.MIN_AREA_PX or area > M4.MAX_AREA_PX:
            n_filtered += 1
            reason[lbl] = 1 if area < M4.MIN_AREA_PX else 2
            continue
        if M4.EXCLUDE_BORDER_BLOBS and M4._touches_border(stats[lbl], W, H, "any"):
            n_border += 1
            reason[lbl] = 3
            continue
        mask = (label_map == lbl).astype(np.uint8)
        mean_intensity = float(cv2.mean(camera_image, mask=mask)[0])
        if mean_intensity < M4.MIN_INTENSITY:
            n_filtered += 1
            reason[lbl] = 4
            continue
        vals = camera_image[mask.astype(bool)]
        core_intensity = float(vals[vals >= np.median(vals)].mean()) if vals.size else mean_intensity
        masked = camera_image.astype(np.float32) * mask
        M = cv2.moments(masked)
        if M["m00"] < 1e-6:
            reason[lbl] = 5
            continue
        u_centre = M["m10"] / M["m00"]
        v_centre = M["m01"] / M["m00"]
        cands.append((u_centre, v_centre, area, M4._elongation(M), core_intensity, lbl, mean_intensity))
    return dict(thresh=float(thresh_val), binary=binary, label_map=label_map, stats=stats, n_labels=int(n_labels), reason=reason,
                cands=cands, n_filtered=n_filtered, n_border=n_border)


def select(cd, group, a_ref=None):

    cands = cd['cands']
    nc = len(cands)
    code = np.zeros(nc, np.int8); fidx = np.full(nc, -1, np.int64)
    features = []
    n_area = n_elong = 0
    area_median = float("nan")
    ref_used = float('nan')
    needs_ref = M4.BLOB_AREA_CHECK and nc >= M4.BLOB_AREA_MIN_N
    if group == 'D1' and needs_ref and (a_ref is None or not np.isfinite(a_ref)):
        code[:] = 4
        return None, None, code, fidx, ref_used
    if cands:
        area_median = M4._area_weighted_median([c[2] for c in cands])
        check = M4.BLOB_AREA_CHECK and len(cands) >= M4.BLOB_AREA_MIN_N
        ref = area_median if group == 'D0' else ((float(a_ref) if a_ref is not None else float('nan')) if group == 'D1' else None)
        ref_used = float(ref) if ref is not None else float('nan')
        for j, (u_c, v_c, area, elong, _, _, _) in enumerate(cands):
            if group != 'D2' and check and not (M4.BLOB_AREA_LO * ref <= area <= M4.BLOB_AREA_HI * ref):
                n_area += 1
                code[j] = 1 if area < M4.BLOB_AREA_LO * ref else 2
                continue
            if check and area >= 4 and elong > M4.BLOB_ELONG_MAX:
                n_elong += 1
                code[j] = 3
                continue
            fidx[j] = len(features)
            features.append([u_c, v_c])
    arr = np.array(features, dtype=np.float32) if features else np.empty((0, 2), dtype=np.float32)
    branch = {"n_dots": len(features), "n_filtered_noise": cd['n_filtered'], "n_border_excluded": cd['n_border'], "n_area_rejected": n_area,
              "n_elong_rejected": n_elong, "blob_area_median_px": round(area_median, 2) if np.isfinite(area_median) else float("nan")}
    det_info = {"pattern_type": 'dots', "otsu_threshold": float(cd['thresh']), "n_features": len(arr), **branch}
    return arr.astype(np.float32), det_info, code, fidx, ref_used


# Match the supplied dot detections to projector features and triangulate accepted pairs.
def reconstruct_from(cam_features, det_info, pattern, cfg=CFG):

    method = 'epipolar'
    pattern_type = 'dots'
    n_detected = int(len(cam_features))
    det_info = dict(det_info)
    cam_allowed = det_info.pop("feature_allowed", None)
    det_info.pop("feature_classes", None)
    det_info.pop("feature_areas_px", None)
    base_info = {**det_info, "correspondence_method": method, "n_detected": n_detected}
    tr = dict(matched=np.zeros(n_detected, bool), ambiguous=np.zeros(n_detected, bool), duplicate=np.zeros(n_detected, bool),
              best_idx=np.full(n_detected, -1, np.int64), point_idx=np.full(n_detected, -1, np.int64))
    if n_detected == 0:
        return np.empty((0, 3), dtype=np.float32), {
            **base_info, "n_points_3d": 0, "triangulation": "no features",
            "n_matched": 0, "n_rejected": 0, "n_ambiguous": 0,
            "ambiguity_ratio": float("nan"), "rejection_ratio": float("nan"),
            "corr_failure_ratio": float("nan"),
            "ambiguous_mask": np.zeros(0, dtype=bool),
        }, tr
    proj_features, epi = M4._correspond_epipolar(cam_features, pattern, pattern_type, cfg, cam_allowed=cam_allowed)
    ambiguous_all = epi["ambiguous"]
    valid = ~np.any(np.isnan(proj_features), axis=1)
    tr['matched'] = valid.copy(); tr['ambiguous'] = np.asarray(ambiguous_all, bool) & valid
    if 'duplicate' in epi:
        tr['duplicate'] = np.asarray(epi['duplicate'], bool) & valid
    if 'best_idx' in epi:
        tr['best_idx'] = np.where(valid, np.asarray(epi['best_idx'], np.int64), -1)
    tr['point_idx'][valid] = np.arange(int(valid.sum()))
    cam_features = cam_features[valid]
    proj_features = proj_features[valid]
    ambiguous = ambiguous_all[valid]
    n_matched = int(valid.sum())
    n_rejected = n_detected - n_matched
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
        }, tr
    points_3d = M4.triangulate(cam_features, proj_features, cfg)
    finite = ~np.isnan(points_3d).any(axis=1)
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
    return points_3d.astype(np.float32), info, tr


def projector_dot_index(pattern, centres):

    pts, _ids = M4.pattern_features(pattern, 'dots')
    if len(pts) == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    d = np.hypot(pts[:, None, 0] - centres[None, :, 0], pts[:, None, 1] - centres[None, :, 1])
    j = d.argmin(axis=1)
    return j.astype(np.int64), d[np.arange(len(pts)), j]
