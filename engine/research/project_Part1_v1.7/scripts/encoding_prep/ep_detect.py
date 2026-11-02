# Inspect dot candidates and compare geometric matching with class-aware matching.

import math
import numpy as np

import m4_reconstruction as M4
import r5_detect as RD

CFG = RD.CFG
SEL = {0: 'accepted', 1: 'rel_area_small', 2: 'rel_area_large', 3: 'elongation'}


def coded_cfg(radii_mm, k, mm_per_px):
    return {**CFG, 'size_levels': int(k), 'size_radii_px': [float(r) / mm_per_px for r in radii_mm]}


# Split candidate dots into area classes and check class readability before matching.
def select_coded(cd, k, radii_px, code_ref=M4.CODE_REF):

    cands = cd['cands']
    size_log_span = 2.0 * math.log(max(radii_px) / min(radii_px)) if radii_px and min(radii_px) > 0 else None
    nc = len(cands)
    code = np.zeros(nc, np.int8); fidx = np.full(nc, -1, np.int64)
    if nc == 0:
        info = {"pattern_type": 'coded_dots', "otsu_threshold": float(cd['thresh']), "n_features": 0, "n_dots": 0, "n_filtered_noise": cd['n_filtered'],
                "n_border_excluded": cd['n_border'], "n_area_rejected": 0, "n_elong_rejected": 0, "blob_area_median_px": float("nan"),
                "feature_classes": np.zeros(0, np.int64), "feature_allowed": np.zeros((0, k), bool), "feature_areas_px": np.zeros(0)}
        return np.empty((0, 2), np.float32), info, code, fidx, dict(cls_all=np.zeros(0, np.int64), allowed_all=np.zeros((0, k), bool), readable=False)
    areas = np.array([c[2] for c in cands], dtype=np.float64)
    la = np.log(areas)
    local = code_ref == "local" and k == 2 and len(cands) >= 3
    if local:
        xy = np.array([[c[0], c[1]] for c in cands], dtype=np.float64)
        la_c, centres, cls, allowed = M4._local_code_split(xy, la)
    else:
        centres = M4._kmeans_1d(la, k)
        cls, allowed = M4._size_classes(la, centres)
        la_c = la
    refs = np.array([M4._area_weighted_median(areas[cls == j]) if (cls == j).any() else math.exp(centres[j]) for j in range(k)])
    check = M4.BLOB_AREA_CHECK and len(cands) >= M4.BLOB_AREA_MIN_N
    keep = np.ones(len(cands), dtype=bool)
    n_area = n_elong = 0
    for i, (_, _, area, elong, *_rest) in enumerate(cands):
        if check and (area < M4.BLOB_AREA_LO * refs[0] or area > M4.BLOB_AREA_HI * refs[-1]):
            keep[i] = False; n_area += 1; code[i] = 1 if area < M4.BLOB_AREA_LO * refs[0] else 2; continue
        if check and area >= 4 and elong > M4.BLOB_ELONG_MAX:
            keep[i] = False; n_elong += 1; code[i] = 3; continue
    feats = np.array([[c[0], c[1]] for c in cands], dtype=np.float32)[keep]
    la_k, cls_k = la_c[keep], cls[keep]
    lo, hi = la_k[cls_k == 0], la_k[cls_k == k - 1]
    pooled = math.sqrt(0.5 * (lo.var() + hi.var())) if (lo.size > 1 and hi.size > 1) else float("nan")
    dprime = float((hi.mean() - lo.mean()) / pooled) if pooled and pooled > 0 else float("nan")
    counts = np.bincount(cls_k, minlength=k)
    balance = float(counts.min() / counts.max()) if counts.max() > 0 else 0.0
    span = float(centres[-1] - centres[0])
    readable = bool(np.isfinite(dprime) and dprime >= M4.SIZE_DPRIME_MIN and balance >= M4.SIZE_BALANCE_MIN
                    and (not size_log_span or M4.SIZE_SPAN_MIN_FRAC * float(size_log_span) <= span <= M4.SIZE_SPAN_MAX_FRAC * float(size_log_span)))
    allowed_k = allowed[keep]
    if not readable:
        allowed_k = np.ones_like(allowed_k)
    fidx[keep] = np.arange(int(keep.sum()))

    adj = []
    for j in range(k - 1):
        a, b = la_k[cls_k == j], la_k[cls_k == j + 1]
        p = math.sqrt(0.5 * (a.var() + b.var())) if (a.size > 1 and b.size > 1) else float("nan")
        adj.append(float((b.mean() - a.mean()) / p) if p and p > 0 else float("nan"))
    info = {"pattern_type": 'coded_dots', "otsu_threshold": float(cd['thresh']), "n_features": int(keep.sum()),
            "n_dots": int(keep.sum()), "n_filtered_noise": cd['n_filtered'], "n_border_excluded": cd['n_border'], "n_area_rejected": n_area,
            "n_elong_rejected": n_elong, "blob_area_median_px": round(float(refs[-1]), 2),
            "size_class_area_px": " / ".join(f"{math.exp(c):.1f}" for c in (centres if not local else np.log(refs))),
            "size_class_ratio": round(float(math.exp(centres[-1] - centres[0])), 4),
            "size_class_dprime": round(dprime, 3) if np.isfinite(dprime) else float("nan"),
            "size_class_balance": round(balance, 3), "size_code_readable": int(readable),
            "n_size_uncertain": int((allowed_k.sum(axis=1) > 1).sum()), "code_ref": "local" if local else "global",
            "size_class_dprime_adjacent": adj, "size_class_counts": counts.tolist(), "size_log_span_design": size_log_span, "size_log_span_measured": span,
            "feature_classes": cls_k.astype(np.int64), "feature_allowed": allowed_k, "feature_areas_px": areas[keep]}
    cinfo = dict(cls_all=cls.astype(np.int64), allowed_all=allowed, centres=centres, refs=refs, readable=readable, dprime=dprime, balance=balance, span=span)
    return feats.reshape(-1, 2).astype(np.float32), info, code, fidx, cinfo


def reconstruct_from(cam_features, det_info, pattern, cfg, pattern_type, cam_allowed):

    method = 'epipolar'
    n_detected = int(len(cam_features))
    det_info = dict(det_info)
    for kk in ("feature_allowed", "feature_classes", "feature_areas_px"):
        det_info.pop(kk, None)
    base_info = {**det_info, "correspondence_method": method, "n_detected": n_detected}
    tr = dict(matched=np.zeros(n_detected, bool), ambiguous=np.zeros(n_detected, bool), duplicate=np.zeros(n_detected, bool),
              best_idx=np.full(n_detected, -1, np.int64), point_idx=np.full(n_detected, -1, np.int64))
    if n_detected == 0:
        return np.empty((0, 3), dtype=np.float32), {**base_info, "n_points_3d": 0, "triangulation": "no features", "n_matched": 0, "n_rejected": 0,
                                                    "n_ambiguous": 0, "ambiguity_ratio": float("nan"), "rejection_ratio": float("nan"),
                                                    "corr_failure_ratio": float("nan"), "ambiguous_mask": np.zeros(0, dtype=bool)}, tr
    proj_features, epi = M4._correspond_epipolar(cam_features, pattern, pattern_type, cfg, cam_allowed=cam_allowed)
    ambiguous_all = epi["ambiguous"]
    valid = ~np.any(np.isnan(proj_features), axis=1)
    tr['matched'] = valid.copy(); tr['ambiguous'] = np.asarray(ambiguous_all, bool) & valid
    if 'duplicate' in epi:
        tr['duplicate'] = np.asarray(epi['duplicate'], bool) & valid
    if 'best_idx' in epi:
        tr['best_idx'] = np.where(valid, np.asarray(epi['best_idx'], np.int64), -1)
    tr['point_idx'][valid] = np.arange(int(valid.sum()))
    cam_features = cam_features[valid]; proj_features = proj_features[valid]; ambiguous = ambiguous_all[valid]
    n_matched = int(valid.sum()); n_rejected = n_detected - n_matched; n_ambiguous = int(ambiguous.sum())
    corr_stats = {"n_matched": n_matched, "n_rejected": n_rejected, "n_ambiguous": n_ambiguous, "ambiguity_ratio": n_ambiguous / n_detected,
                  "rejection_ratio": n_rejected / n_detected, "corr_failure_ratio": (n_ambiguous + n_rejected) / n_detected}
    epi_stats = {k: v for k, v in epi.items() if k not in ("ambiguous", "matched", "best_idx", "best_dist", "second_dist", "second_id", "t_best",
                                                            "ambiguous_second", "duplicate", "proj_classes", "propagated")}
    if 'proj_classes' in epi:
        tr['proj_classes'] = np.asarray(epi['proj_classes'], np.int64)
    if n_matched == 0:
        return np.empty((0, 3), dtype=np.float32), {**base_info, **corr_stats, **epi_stats, "n_points_3d": 0, "triangulation": "no correspondences",
                                                    "ambiguous_mask": np.zeros(0, dtype=bool)}, tr
    points_3d = M4.triangulate(cam_features, proj_features, cfg)
    finite = ~np.isnan(points_3d).any(axis=1)
    info = {**base_info, **corr_stats, **epi_stats, "n_correspondences": n_matched, "n_points_3d": len(points_3d), "n_points_finite": int(finite.sum()),
            "depth_min_mm": float(points_3d[finite, 2].min()) if finite.any() else float("nan"),
            "depth_max_mm": float(points_3d[finite, 2].max()) if finite.any() else float("nan"),
            "depth_mean_mm": float(points_3d[finite, 2].mean()) if finite.any() else float("nan"), "ambiguous_mask": ambiguous}
    if "t_best" in epi:
        info["epi_t_best"] = epi["t_best"][valid]
    if "duplicate" in epi:
        info["duplicate_mask"] = epi["duplicate"][valid]
    return points_3d.astype(np.float32), info, tr


def search_depth(cfg):
    if cfg.get("search_depth_mm"):
        return float(cfg["search_depth_mm"])
    return float(cfg.get("measurement_volume_mm", M4.MEASUREMENT_VOLUME_MM))


def projector_side(pattern, cfg, pattern_type):

    if pattern_type == 'coded_dots':
        pts, ids, areas = M4.pattern_features(pattern, pattern_type, return_areas=True)
        cls = M4.projector_size_classes(areas, cfg)
    else:
        pts, ids = M4.pattern_features(pattern, pattern_type); cls = np.zeros(len(pts), np.int64)
    A, B = M4.projector_tracks(pts, cfg, 0.0, search_depth(cfg))
    return dict(pts=pts, ids=np.asarray(ids), cls=np.asarray(cls, np.int64), A=A, B=B)


def track_candidates(cam_features, proj, cfg, allowed, chunk=256):

    cam = np.asarray(cam_features, np.float64); A, B, ids, pcls = proj['A'], proj['B'], proj['ids'], proj['cls']
    tol = float(cfg.get("epi_tol_px", M4.EPI_TOL_PX)); t_margin = float(cfg.get("epi_t_margin", M4.EPI_T_MARGIN))
    AB = B - A; L2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)
    N = len(cam); n_geom = np.zeros(N, np.int64); n_class = np.zeros(N, np.int64); geom_sets = [None] * N; class_sets = [None] * N
    for s in range(0, N, chunk):
        c = cam[s:s + chunk]
        d = c[:, None, :] - A[None, :, :]
        t = (d * AB[None, :, :]).sum(axis=-1) / L2[None, :]
        tc = np.clip(t, -t_margin, 1.0 + t_margin)
        closest = A[None, :, :] + tc[..., None] * AB[None, :, :]
        dist = np.linalg.norm(c[:, None, :] - closest, axis=-1)
        within = dist <= tol
        adm = within & allowed[s:s + chunk][:, pcls] if allowed is not None else within
        for i in range(len(c)):
            g = np.unique(ids[within[i]]); a = np.unique(ids[adm[i]])
            n_geom[s + i] = len(g); n_class[s + i] = len(a); geom_sets[s + i] = g; class_sets[s + i] = a
    return dict(n_geom=n_geom, n_class=n_class, geom_ids=geom_sets, class_ids=class_sets)
