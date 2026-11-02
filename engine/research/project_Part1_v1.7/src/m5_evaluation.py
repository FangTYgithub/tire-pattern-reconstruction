# Compute point errors, profile metrics and sensitivity summaries. Thesis surface coverage is
# evaluated by surface_eval_r3.py and the study evaluation adapters.

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator


UNIQUENESS_WINDOW_PX  = 32  # Side length of the local image patch [px].
UNIQUENESS_STRIDE_PX  = 16  # Stride between sampled patches [px].
UNIQUENESS_N_SAMPLES  = 200  # Maximum sampled patch count for the uniqueness diagnostic.
UNIQUENESS_BLANK_REL  = 0.02  # Relative threshold for excluding blank patches.


UNIQUENESS_BLANK_STD  = 1e-6  # Minimum patch standard deviation used by normalization.


INTERP_METHOD         = "linear"  # Interpolation used to evaluate reference heights at sample positions.


SWEEP_SPACING_MM      = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]  # Standalone sweep pitches [mm]; formal cases use their registered manifests.
SWEEP_PATTERN_TYPES   = ["lines", "dots"]  # Pattern families in the standalone sweep.


SWEEP_GRID_STEP_X_MM  = 5.0  # Standalone longitudinal surface-grid step [mm].
SWEEP_GRID_STEP_Y_MM  = 2.0  # Standalone lateral surface-grid step [mm].
SWEEP_BATCH_SIZE      = 512  # Ray batch size for the standalone sweep.


CENTRAL_ZONE_Y_MM     = 80.0  # Lateral half-width of the profile diagnostic region [mm].

COVERAGE_Y_TOL_MM     = 20.0  # Profile-extremum location tolerance [mm]; not the surface support scale.


CE_LOG2_1_PLUS_SNR_BIT = 6.0
CE_Q_SIMULTANEOUS_IMG  = 1  # Simultaneous image count in the channel diagnostic; not the gray budget.


CE_F_FRAMES            = 1  # Temporal frame count in the channel diagnostic.


CE_DELTA_Z_MODE          = "design_unique_depth"
CE_NN_DECODER_RANGE_FACTOR = 0.5
CE_THETA_DEG_FALLBACK    = 18.9


def _rmse_empty(
    n_negative: int = 0,
    depth_recon: np.ndarray | None = None,
    depth_gt:    np.ndarray | None = None,
) -> dict:

    nan = float("nan")
    return {
        "rmse_mm": nan, "mae_mm": nan, "max_err_mm": nan, "n_evaluated": 0,
        "rmse_all_mm": nan, "mae_all_mm": nan, "max_err_all_mm": nan,
        "n_evaluated_all": 0,
        "n_negative_depth": n_negative,
        "neg_depth_err_mean": nan, "neg_depth_err_max": nan,
        "coverage": 0.0, "fov_coverage": nan,
        "depth_recon":     (depth_recon if depth_recon is not None else np.array([])).astype(np.float32),
        "depth_gt_interp": (depth_gt    if depth_gt    is not None else np.array([])).astype(np.float32),
    }


# Compare reconstructed depths with the interpolated surface at the same planar positions. Use the
# mesh alignment convention when forming the reference depths.
def compute_rmse(
    points_3d:  np.ndarray,
    mesh:       dict,
) -> dict:


    valid = ~np.isnan(points_3d).any(axis=1)
    pts   = points_3d[valid]

    if len(pts) == 0:
        return _rmse_empty()


    depth_recon = pts[:, 2].astype(np.float64)


    Z_grid  = mesh["Z"].astype(np.float64)
    z_min   = Z_grid.min()
    depth_gt_grid = Z_grid - z_min


    # X: columns; Y: rows.
    X_grid = mesh["X"].astype(np.float64)
    Y_grid = mesh["Y"].astype(np.float64)

    y_vals = Y_grid[:, 0]
    x_vals = X_grid[0, :]


    if y_vals[0] > y_vals[-1]:
        y_vals          = y_vals[::-1]
        depth_gt_grid   = depth_gt_grid[::-1, :]
    if x_vals[0] > x_vals[-1]:
        x_vals          = x_vals[::-1]
        depth_gt_grid   = depth_gt_grid[:, ::-1]

    interp = RegularGridInterpolator(
        (y_vals, x_vals),
        depth_gt_grid,
        method      = INTERP_METHOD,
        bounds_error= False,
        fill_value  = np.nan,
    )


    xy_recon = np.column_stack([pts[:, 1], pts[:, 0]])
    depth_gt_interp = interp(xy_recon).astype(np.float64)


    in_range = ~np.isnan(depth_gt_interp)
    d_recon  = depth_recon[in_range]
    d_gt     = depth_gt_interp[in_range]

    if len(d_recon) == 0:
        return _rmse_empty(0, depth_recon, depth_gt_interp)


    errors_all  = d_recon - d_gt
    rmse_all    = float(np.sqrt(np.mean(errors_all ** 2)))
    mae_all     = float(np.mean(np.abs(errors_all)))
    max_err_all = float(np.max(np.abs(errors_all)))
    n_all       = int(len(d_recon))


    n_negative         = int((d_recon < 0).sum())
    physical_mask      = d_recon >= 0
    d_recon_clean      = d_recon[physical_mask]
    d_gt_clean         = d_gt[physical_mask]

    if n_negative > 0:
        neg_abs_err        = np.abs(errors_all[~physical_mask])
        neg_depth_err_mean = float(neg_abs_err.mean())
        neg_depth_err_max  = float(neg_abs_err.max())
    else:
        neg_depth_err_mean = float("nan")
        neg_depth_err_max  = float("nan")

    if len(d_recon_clean) == 0:
        out = _rmse_empty(n_negative, depth_recon, depth_gt_interp)
        out.update({
            "rmse_all_mm": rmse_all, "mae_all_mm": mae_all,
            "max_err_all_mm": max_err_all, "n_evaluated_all": n_all,
            "neg_depth_err_mean": neg_depth_err_mean,
            "neg_depth_err_max":  neg_depth_err_max,
        })
        return out


    errors  = d_recon_clean - d_gt_clean
    rmse    = float(np.sqrt(np.mean(errors ** 2)))
    mae     = float(np.mean(np.abs(errors)))
    max_err = float(np.max(np.abs(errors)))


    grid_total   = len(y_vals) * len(x_vals)
    coverage     = len(d_recon_clean) / grid_total


    pts_in_range = pts[in_range]
    if len(pts_in_range) > 1:
        y_fov_min = pts_in_range[:, 1].min()
        y_fov_max = pts_in_range[:, 1].max()
        x_fov_min = pts_in_range[:, 0].min()
        x_fov_max = pts_in_range[:, 0].max()

        y_step = float(y_vals[1] - y_vals[0]) if len(y_vals) > 1 else 1.0
        x_step = float(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 1.0
        n_fov_y = max(1, int(round((y_fov_max - y_fov_min) / abs(y_step))) + 1)
        n_fov_x = max(1, int(round((x_fov_max - x_fov_min) / abs(x_step))) + 1)
        fov_cells    = n_fov_y * n_fov_x
        fov_coverage = len(d_recon_clean) / fov_cells
    else:
        fov_coverage = float("nan")

    return {

        "rmse_mm":          rmse,
        "mae_mm":           mae,
        "max_err_mm":       max_err,
        "n_evaluated":      len(d_recon_clean),

        "rmse_all_mm":      rmse_all,
        "mae_all_mm":       mae_all,
        "max_err_all_mm":   max_err_all,
        "n_evaluated_all":  n_all,

        "n_negative_depth": n_negative,
        "neg_depth_err_mean": neg_depth_err_mean,
        "neg_depth_err_max":  neg_depth_err_max,
        "coverage":         round(coverage, 6),
        "fov_coverage":     round(fov_coverage, 4) if not np.isnan(fov_coverage) else float("nan"),
        "depth_recon":      depth_recon.astype(np.float32),
        "depth_gt_interp":  depth_gt_interp.astype(np.float32),
    }


# Compare reconstructed profile extrema and coverage with the surface descriptors.
def compute_contact_feature_errors(
    points_3d:        np.ndarray,
    m1_config:        dict,
    central_zone_y_mm: float = CENTRAL_ZONE_Y_MM,
    coverage_y_tol_mm: float = COVERAGE_Y_TOL_MM,
) -> dict:

    gt_cf = m1_config.get("contact_features", {})


    valid = ~np.isnan(points_3d).any(axis=1)
    pts   = points_3d[valid]

    if len(pts) == 0 or not gt_cf:
        return {
            "gt_central_z_std_mm":      gt_cf.get("central_z_std_mm",  float("nan")),
            "gt_central_z_range_mm":    gt_cf.get("central_z_range_mm", float("nan")),
            "gt_indentation_depth_mm":  gt_cf.get("indentation_depth_mm", float("nan")),
            "recon_central_z_std_mm":   float("nan"),
            "recon_central_z_range_mm": float("nan"),
            "recon_indentation_depth_mm": float("nan"),
            "error_central_z_std_mm":   float("nan"),
            "error_central_z_range_mm": float("nan"),
            "error_indentation_depth_mm": float("nan"),
            "indentation_center_covered": False,
            "n_central_points":          0,
            "note": "no valid points",
        }


    central_mask = np.abs(pts[:, 1]) < central_zone_y_mm   # Y: lateral direction.
    pts_central  = pts[central_mask]
    n_central    = int(central_mask.sum())

    if n_central >= 3:
        z_central        = pts_central[:, 2]
        recon_z_std      = float(z_central.std())
        recon_z_range    = float(z_central.max() - z_central.min())
        recon_z_max      = float(z_central.max())
    else:
        recon_z_std = recon_z_range = recon_z_max = float("nan")


    side_mask = np.abs(pts[:, 1]) > 90.0
    if side_mask.sum() >= 3:
        z_ref_recon      = float(pts[side_mask, 2].mean())
        z_min_recon      = float(pts[:, 2].min())
        recon_depth      = z_ref_recon - z_min_recon
    else:
        recon_depth = float("nan")


    gt_center_y   = gt_cf.get("indentation_center_y_mm", 0.0)
    near_centre   = np.abs(pts[:, 1] - gt_center_y) < coverage_y_tol_mm
    centre_covered = bool(near_centre.any())


    gt_z_std    = gt_cf.get("central_z_std_mm",     float("nan"))
    gt_z_range  = gt_cf.get("central_z_range_mm",   float("nan"))
    gt_depth    = gt_cf.get("indentation_depth_mm",  float("nan"))

    def _err(recon, gt):
        if np.isnan(recon) or np.isnan(gt):
            return float("nan")
        return round(abs(recon - gt), 4)

    return {

        "gt_central_z_std_mm":        round(gt_z_std, 4),
        "gt_central_z_range_mm":      round(gt_z_range, 4),
        "gt_indentation_depth_mm":    round(gt_depth, 4),

        "recon_central_z_std_mm":     round(recon_z_std, 4)  if not np.isnan(recon_z_std) else float("nan"),
        "recon_central_z_range_mm":   round(recon_z_range, 4) if not np.isnan(recon_z_range) else float("nan"),
        "recon_indentation_depth_mm": round(recon_depth, 4)   if not np.isnan(recon_depth) else float("nan"),

        "error_central_z_std_mm":        _err(recon_z_std,   gt_z_std),
        "error_central_z_range_mm":      _err(recon_z_range, gt_z_range),
        "error_indentation_depth_mm":    _err(recon_depth,   gt_depth),

        "indentation_center_covered":    centre_covered,
        "n_central_points":              n_central,
        "coverage_y_tol_mm":             coverage_y_tol_mm,
    }


# Measure local image-patch similarity with normalized cross-correlation.
def compute_uniqueness(
    camera_image: np.ndarray,
    window_px:    int   = UNIQUENESS_WINDOW_PX,
    stride_px:    int   = UNIQUENESS_STRIDE_PX,
    n_samples:    int   = UNIQUENESS_N_SAMPLES,
) -> dict:

    H, W = camera_image.shape
    img  = camera_image.astype(np.float32)
    nan  = float("nan")


    half = window_px // 2
    row_centres = np.arange(half, H - half, stride_px)
    col_centres = np.arange(half, W - half, stride_px)
    positions   = [(r, c) for r in row_centres for c in col_centres]
    n_positions = len(positions)

    empty = {
        "uniqueness_mean": nan, "uniqueness_median": nan, "uniqueness_min": nan,
        "uniqueness_scores": np.array([], dtype=np.float32),
        "n_windows": 0, "n_positions": n_positions, "n_blank": 0, "blank_ratio": nan,
    }
    if n_positions == 0:
        return empty


    stds = np.array(
        [img[r - half : r + half, c - half : c + half].std() for (r, c) in positions],
        dtype=np.float64,
    )
    blank_thr    = max(UNIQUENESS_BLANK_STD, UNIQUENESS_BLANK_REL * float(img.max()))
    blank_mask   = stds < blank_thr
    n_blank      = int(blank_mask.sum())
    blank_ratio  = n_blank / n_positions
    nonblank_idx = np.flatnonzero(~blank_mask)

    if len(nonblank_idx) < 2:

        empty.update({"n_blank": n_blank, "blank_ratio": round(blank_ratio, 4)})
        return empty


    rng = np.random.default_rng(seed=42)
    if len(nonblank_idx) > n_samples:
        nonblank_idx = rng.choice(nonblank_idx, size=n_samples, replace=False)


    patches = []
    for i in nonblank_idx:
        r, c  = positions[i]
        patch = img[r - half : r + half, c - half : c + half]
        patch = (patch - patch.mean()) / patch.std()
        patches.append(patch.ravel())

    patches = np.stack(patches, axis=0)
    M, D    = patches.shape


    # Normalized patches: mean 0, standard deviation 1.
    ncc_matrix = (patches @ patches.T) / D


    np.fill_diagonal(ncc_matrix, -np.inf)
    max_ncc         = ncc_matrix.max(axis=1)
    uniqueness      = (1.0 - max_ncc).clip(0.0, 1.0)

    return {
        "uniqueness_mean":   float(uniqueness.mean()),
        "uniqueness_median": float(np.median(uniqueness)),
        "uniqueness_min":    float(uniqueness.min()),
        "uniqueness_scores": uniqueness.astype(np.float32),
        "n_windows":         int(M),
        "n_positions":       int(n_positions),
        "n_blank":           n_blank,
        "blank_ratio":       round(blank_ratio, 4),
    }


def _default_theta_deg() -> float:

    try:
        import config as _cfg
        return float(_cfg.M3_EFFECTIVE_THETA_DEG)
    except Exception:
        return CE_THETA_DEG_FALLBACK


# Estimate the depth interval implied by feature spacing and the specified triangulation angle.
def unique_measurement_depth_mm(
    spacing_mm: float,
    theta_deg:  float | None = None,
    decoder:    str = "design",
) -> float:

    theta = _default_theta_deg() if theta_deg is None else float(theta_deg)
    dz = float(spacing_mm) / math.tan(math.radians(theta))
    if decoder == "design":
        return dz
    if decoder == "nn":
        return dz * CE_NN_DECODER_RANGE_FACTOR
    raise ValueError(f"unknown decoder '{decoder}' (use 'design' or 'nn')")


# Calculate the channel-efficiency diagnostic from sample count, depth range and error.
def compute_channel_efficiency(
    camera_image:    np.ndarray,
    n_points_3d:      int,
    measurement_depth_mm: float,
    rmse_mm:          float,
    log2_1_plus_snr_bit: float = CE_LOG2_1_PLUS_SNR_BIT,
    Q:                int   = CE_Q_SIMULTANEOUS_IMG,
    F:                int   = CE_F_FRAMES,
) -> dict:

    H, W = camera_image.shape
    Nx_Ny = H * W

    rho_3d = n_points_3d / Nx_Ny


    delta_z_mm = max(rmse_mm, 1e-6)
    dynamic_range = measurement_depth_mm / delta_z_mm

    M2D_bit = Nx_Ny * log2_1_plus_snr_bit
    M3D_bit = n_points_3d * math.log2(1.0 + dynamic_range)

    channel_efficiency = M3D_bit / (Q * F * M2D_bit) if M2D_bit > 0 else float("nan")


    return {
        "rho_3d":             round(rho_3d, 6),
        "dynamic_range":       round(dynamic_range, 2),
        "delta_z_mm":          round(delta_z_mm, 5),
        "M2D_bit":             round(M2D_bit, 2),
        "M3D_bit":             round(M3D_bit, 2),
        "channel_efficiency":  round(channel_efficiency, 7),
        "note": (
            "delta_z is a proxy (= RMSE from Layer A), not a calibrated noise "
            "floor — see NOTE_DYNAMIC_RANGE. n_points_3d is used directly as "
            "N3D without independence verification — see NOTE_CHANNEL_EFFICIENCY. "
            "Q=1: single-camera setup, projector not counted as image source."
        ),
    }


# Calculate the channel-efficiency reference using the valid camera-pixel count.
def compute_theoretical_max_eta(
    camera_image:         np.ndarray,
    n_valid_pixels_m3:    int,
    measurement_depth_mm: float,
    rmse_mm:              float,
    log2_1_plus_snr_bit:  float = CE_LOG2_1_PLUS_SNR_BIT,
    Q:                    int   = CE_Q_SIMULTANEOUS_IMG,
    F:                    int   = CE_F_FRAMES,
) -> dict:

    H, W  = camera_image.shape
    Nx_Ny = H * W
    delta_z_mm    = max(rmse_mm, 1e-6)
    dynamic_range = measurement_depth_mm / delta_z_mm

    M2D_bit       = Nx_Ny * log2_1_plus_snr_bit
    M3D_max_bit   = n_valid_pixels_m3 * math.log2(1.0 + dynamic_range)
    eta_max       = M3D_max_bit / (Q * F * M2D_bit) if M2D_bit > 0 else float("nan")
    rho_max       = n_valid_pixels_m3 / Nx_Ny

    return {
        "rho_3d_theoretical":  round(rho_max, 6),
        "eta_theoretical":     round(eta_max, 7),
        "n_valid_pixels_m3":   n_valid_pixels_m3,
        "note": (
            "Theoretical upper bound: assumes all M3 valid pixels are "
            "successfully reconstructed by M4 (perfect correspondence). "
            "Gap to actual eta reflects M4 algorithm efficiency."
        ),
    }


_EVAL_UNIQ_WINDOW_PX = 32
_EVAL_UNIQ_STRIDE_PX = 32
_EVAL_UNIQ_N_SAMPLES = 200

# Compare recovered deformation descriptors with the analytic reference.
def compute_indentation_errors(
    points_3d: np.ndarray,
    m1_config: dict,
) -> dict:

    valid = ~np.isnan(points_3d).any(axis=1)
    pts   = points_3d[valid]

    nan_result = {
        "recon_depth_max":  float("nan"),
        "gt_depth_max":     float("nan"),
        "depth_max_error":  float("nan"),
        "recon_depth_mean": float("nan"),
        "recon_area_mm2":   float("nan"),
        "gt_area_mm2":      float("nan"),
        "area_error_pct":   float("nan"),
        "n_points_used":    0,
        "gt_area_source":   "none",
    }

    if len(pts) == 0:
        return nan_result

    recon_depth_max  = float(pts[:, 2].max())
    recon_depth_mean = float(pts[:, 2].mean())

    gt_cf = m1_config.get("contact_features", {})
    gt_depth_max = gt_cf.get("indentation_depth_mm",
                             m1_config.get("depth_max_mm", float("nan")))
    depth_max_error = abs(recon_depth_max - gt_depth_max) if not (
        np.isnan(recon_depth_max) or np.isnan(gt_depth_max)) else float("nan")

    recon_area_mm2 = float("nan")
    if len(pts) >= 3:
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts[:, :2])
            recon_area_mm2 = float(hull.volume)
        except Exception:
            pass

    gt_area_source = "none"
    gt_area_mm2 = float("nan")
    if "contact_patch_area_mm2" in gt_cf:
        gt_area_mm2 = float(gt_cf["contact_patch_area_mm2"])
        gt_area_source = "contact_features"
    else:
        mesh_x = m1_config.get("x_range_mm", None)
        mesh_y = m1_config.get("y_range_mm", None)
        if mesh_x is not None and mesh_y is not None:
            width_x = abs(mesh_x[1] - mesh_x[0]) if isinstance(mesh_x, (tuple, list)) else float(mesh_x)
            width_y = abs(mesh_y[1] - mesh_y[0]) if isinstance(mesh_y, (tuple, list)) else float(mesh_y)
            gt_area_mm2 = float(width_x * width_y)
            gt_area_source = "mesh_bbox"
        elif "grid_shape_ny_nx" in m1_config:
            ny, nx = m1_config["grid_shape_ny_nx"]
            step_x = m1_config.get("grid_step_x_mm", 2.0)
            step_y = m1_config.get("grid_step_y_mm", 1.0)
            gt_area_mm2 = float((nx - 1) * step_x * (ny - 1) * step_y)
            gt_area_source = "mesh_bbox"

    area_error_pct = float("nan")
    if not (np.isnan(recon_area_mm2) or np.isnan(gt_area_mm2)) and gt_area_mm2 > 0:
        area_error_pct = abs(recon_area_mm2 - gt_area_mm2) / gt_area_mm2 * 100.0

    return {
        "recon_depth_max":  round(recon_depth_max, 4),
        "gt_depth_max":     round(gt_depth_max, 4) if not np.isnan(gt_depth_max) else float("nan"),
        "depth_max_error":  round(depth_max_error, 4) if not np.isnan(depth_max_error) else float("nan"),
        "recon_depth_mean": round(recon_depth_mean, 4),
        "recon_area_mm2":   round(recon_area_mm2, 2) if not np.isnan(recon_area_mm2) else float("nan"),
        "gt_area_mm2":      round(gt_area_mm2, 2) if not np.isnan(gt_area_mm2) else float("nan"),
        "area_error_pct":   round(area_error_pct, 2) if not np.isnan(area_error_pct) else float("nan"),
        "n_points_used":    len(pts),
        "gt_area_source":   gt_area_source,
    }


# Summarize correspondence ambiguity and accepted reconstruction counts.
def compute_ambiguity_metrics(
    recon_info:   dict | None,
    camera_image: np.ndarray,
    points_3d:    np.ndarray,
) -> dict:

    ri   = recon_info or {}
    H, W = camera_image.shape
    n_pix = float(H * W)
    n_det = int(ri.get("n_detected", 0) or 0)
    n_amb = int(ri.get("n_ambiguous", 0) or 0)
    A = n_amb / n_det if n_det > 0 else float("nan")

    finite = ~np.isnan(points_3d).any(axis=1) if len(points_3d) else np.zeros(0, bool)
    amb_mask = ri.get("ambiguous_mask")
    if amb_mask is None or len(amb_mask) != len(finite):
        amb_mask = np.zeros(len(finite), dtype=bool)
    n_reliable = int((finite & ~amb_mask).sum())
    D = n_reliable / n_pix
    Q = D * (1.0 - A) if np.isfinite(A) else D
    return {
        "A_ambiguity":        round(A, 5) if np.isfinite(A) else float("nan"),
        "D_density":          round(D, 7),
        "Q_reliable_density": round(Q, 7),
        "n_reliable":         n_reliable,
    }


IND_PEAK_FRAC_FOR_CENTRE = 0.5


# Interpolate the recovered deformation and compare its shape with the reference field.
def compute_indentation_recovery(
    points_3d:  np.ndarray,
    mesh:       dict,
    m1_config:  dict,
    ambiguous_mask: np.ndarray | None = None,
) -> dict:

    nan = float("nan")
    out = {
        "ind_has_gt": False, "ind_depth_gt_mm": nan, "ind_depth_rec_mm": nan,
        "ind_depth_rec_pts_mm": nan, "ind_depth_err_mm": nan, "ind_center_err_mm": nan,
        "ind_area_gt_mm2": nan, "ind_area_rec_mm2": nan, "ind_area_err_pct": nan,
        "ind_iou": nan, "ind_coverage_frac": nan, "ind_n_pts_in_contour": 0,
        "ind_slope_gt_deg": nan, "ind_slope_rec_deg": nan, "ind_slope_err_pct": nan,
        "ind_grad_rms_err": nan, "ind_grad_rms_rel": nan, "ind_n_slope_cells": 0,
        "ind_surround_offset_mm": nan, "ind_depth_err_rel_mm": nan,
        "ind_single_dent_valid": True, "ind_dip_rms_mm": nan,
        "ind_dent1_depth_err_mm": nan, "ind_dent2_depth_err_mm": nan,
    }
    ind = m1_config.get("indentation") if m1_config else None
    if not ind or "Z_base" not in mesh or "indent_mask" not in mesh:
        return out
    out["ind_has_gt"] = True
    double = m1_config.get("double_indentation") if m1_config else None
    out["ind_single_dent_valid"] = not bool(double)

    X = np.asarray(mesh["X"], dtype=np.float64)
    Y = np.asarray(mesh["Y"], dtype=np.float64)
    Z = np.asarray(mesh["Z"], dtype=np.float64)
    Zb = np.asarray(mesh["Z_base"], dtype=np.float64)
    mask_gt = np.asarray(mesh["indent_mask"], dtype=bool)
    d_gt  = float(ind["depth_mm"])
    cx, cy = (float(v) for v in ind["center_xy_mm"])
    frac = float(ind.get("area_threshold_frac", 0.10))
    half = float(m1_config.get("fov_half_width_mm", 46.0))
    step_x = float(m1_config.get("grid_step_x_mm", np.abs(np.diff(X[0, :])).mean()))
    step_y = float(m1_config.get("grid_step_y_mm", np.abs(np.diff(Y[:, 0])).mean()))
    cell = step_x * step_y

    dip_gt   = Zb - Z
    base_rel = Zb - Z.min()
    fov = (np.abs(X) <= half) & (np.abs(Y) <= half)

    if not double:
        out["ind_depth_gt_mm"] = round(d_gt, 4)
        out["ind_area_gt_mm2"] = round(float((mask_gt & fov).sum()) * cell, 3)


    finite = ~np.isnan(points_3d).any(axis=1) if len(points_3d) else np.zeros(0, bool)
    if ambiguous_mask is not None and len(ambiguous_mask) == len(finite):
        finite = finite & ~np.asarray(ambiguous_mask, dtype=bool)
    pts = points_3d[finite].astype(np.float64)
    if len(pts) < 4:
        return out

    from scipy.interpolate import RegularGridInterpolator, griddata
    y_vals, x_vals = Y[:, 0], X[0, :]
    base_interp = RegularGridInterpolator((y_vals, x_vals), base_rel, method="linear",
                                          bounds_error=False, fill_value=np.nan)
    base_at_pts = base_interp(np.column_stack([pts[:, 1], pts[:, 0]]))
    dip_pts = base_at_pts - pts[:, 2]
    ok = np.isfinite(dip_pts)
    pts, dip_pts = pts[ok], dip_pts[ok]
    if len(pts) < 4:
        return out

    if not double:
        out["ind_depth_rec_pts_mm"] = round(float(dip_pts.max()), 4)


    mask_interp = RegularGridInterpolator((y_vals, x_vals), mask_gt.astype(np.float64),
                                          method="nearest", bounds_error=False, fill_value=0.0)
    out["ind_n_pts_in_contour"] = int((mask_interp(np.column_stack([pts[:, 1], pts[:, 0]])) > 0.5).sum())


    try:
        dip_grid = griddata(pts[:, :2], dip_pts, (X, Y), method="linear")
    except Exception:
        return out
    valid = np.isfinite(dip_grid) & fov
    if valid.sum() < 4:
        return out

    gt_cells = mask_gt & fov
    out["ind_coverage_frac"] = round(float((gt_cells & valid).sum() / max(1, gt_cells.sum())), 4)

    dg = np.where(valid, dip_grid, np.nan)
    peak = float(np.nanmax(dg))
    out["ind_depth_rec_mm"] = round(peak, 4)
    out["ind_depth_err_mm"] = round(peak - d_gt, 4)


    surround = valid & ~gt_cells
    if surround.sum() >= 20:
        offset = float(np.nanmedian(dg[surround]))
        out["ind_surround_offset_mm"] = round(offset, 4)
        out["ind_depth_err_rel_mm"] = round(peak - offset - d_gt, 4)


    core = valid & (dg >= IND_PEAK_FRAC_FOR_CENTRE * peak)
    if core.any() and peak > 0:
        w = np.where(core, dg, 0.0)
        cx_r = float((w * X).sum() / w.sum()); cy_r = float((w * Y).sum() / w.sum())
        if float(ind.get("asym", 1.0)) != 1.0:

            gt_core = valid & (dip_gt >= IND_PEAK_FRAC_FOR_CENTRE * float(dip_gt[valid].max()))
            wg = np.where(gt_core, dip_gt, 0.0)
            if wg.sum() > 0:
                cx = float((wg * X).sum() / wg.sum())
                cy = float((wg * Y).sum() / wg.sum())
        out["ind_center_err_mm"] = round(float(np.hypot(cx_r - cx, cy_r - cy)), 4)


    rec_cells = valid & (dg >= frac * d_gt)
    area_rec  = float(rec_cells.sum()) * cell
    out["ind_area_rec_mm2"] = round(area_rec, 3)
    if out["ind_area_gt_mm2"] > 0:
        out["ind_area_err_pct"] = round((area_rec - out["ind_area_gt_mm2"]) / out["ind_area_gt_mm2"] * 100.0, 3)
    union = (rec_cells | gt_cells).sum()
    out["ind_iou"] = round(float((rec_cells & gt_cells).sum() / union), 4) if union > 0 else nan


    gy_r, gx_r = np.gradient(np.where(valid, dg, np.nan), step_y, step_x)
    gy_g, gx_g = np.gradient(dip_gt, step_y, step_x)
    slope_r = np.hypot(gx_r, gy_r)
    slope_g = np.hypot(gx_g, gy_g)
    cells = gt_cells & valid & np.isfinite(slope_r)
    out["ind_n_slope_cells"] = int(cells.sum())
    if cells.sum() >= 4:
        s_gt  = float(slope_g[cells].max())
        s_rec = float(slope_r[cells].max())
        out["ind_slope_gt_deg"]  = round(float(np.degrees(np.arctan(s_gt))), 3)
        out["ind_slope_rec_deg"] = round(float(np.degrees(np.arctan(s_rec))), 3)
        if s_gt > 0:
            out["ind_slope_err_pct"] = round(100.0 * (s_rec - s_gt) / s_gt, 3)
            rms = float(np.sqrt(np.mean((slope_r[cells] - slope_g[cells]) ** 2)))
            out["ind_grad_rms_err"] = round(rms, 5)
            out["ind_grad_rms_rel"] = round(rms / s_gt, 4)


    if double:
        for k in ("ind_depth_gt_mm", "ind_depth_rec_mm", "ind_depth_rec_pts_mm", "ind_depth_err_mm", "ind_center_err_mm",
                  "ind_area_gt_mm2", "ind_area_rec_mm2", "ind_area_err_pct", "ind_iou", "ind_slope_gt_deg", "ind_slope_rec_deg",
                  "ind_slope_err_pct", "ind_grad_rms_err", "ind_grad_rms_rel", "ind_surround_offset_mm", "ind_depth_err_rel_mm"):
            out[k] = nan
        out["ind_single_dent_valid"] = False
        cells_f = gt_cells & valid
        if cells_f.sum() >= 4:
            out["ind_dip_rms_mm"] = round(float(np.sqrt(np.mean((dg[cells_f] - dip_gt[cells_f]) ** 2))), 4)
        rec_interp = RegularGridInterpolator((y_vals, x_vals), dg, method="linear", bounds_error=False, fill_value=np.nan)
        ind2 = m1_config.get("indentation2") or {}
        for tag, c, d_true in (("1", ind["center_xy_mm"], double.get("depth_at_center1_mm")),
                               ("2", ind2.get("center_xy_mm"), double.get("depth_at_center2_mm"))):
            if c is None or d_true is None:
                continue
            v = float(rec_interp([[float(c[1]), float(c[0])]])[0])
            if np.isfinite(v):
                out[f"ind_dent{tag}_depth_err_mm"] = round(v - float(d_true), 4)
    return out


OUTLIER_ERR_MM = 1.0


def _outlier_stats(rm: dict, tag: str, err_mm: float = OUTLIER_ERR_MM) -> dict:

    nan = float("nan")
    empty = {f"outlier_rate_{tag}": nan, f"p95_err_{tag}_mm": nan,
             f"median_err_{tag}_mm": nan, f"n_outliers_{tag}": 0}
    dr, dg = rm.get("depth_recon"), rm.get("depth_gt_interp")
    if dr is None or dg is None or len(dr) == 0 or len(dr) != len(dg):
        return empty
    e = np.abs(np.asarray(dr, dtype=np.float64) - np.asarray(dg, dtype=np.float64))
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return empty
    n_out = int((e > err_mm).sum())
    return {f"outlier_rate_{tag}":  round(n_out / len(e), 5),
            f"p95_err_{tag}_mm":    round(float(np.percentile(e, 95)), 5),
            f"median_err_{tag}_mm": round(float(np.median(e)), 5),
            f"n_outliers_{tag}":    n_out}


# Combine point, profile, ambiguity and image diagnostics for one reconstruction.
def evaluate_all(
    points_3d:  np.ndarray,
    mesh:       dict,
    camera_image: np.ndarray,
    m1_config:  dict,
    m3_info:    dict,
    uniq_window_px: int   = _EVAL_UNIQ_WINDOW_PX,
    uniq_stride_px: int   = _EVAL_UNIQ_STRIDE_PX,
    uniq_n_samples: int   = _EVAL_UNIQ_N_SAMPLES,
    spacing_mm:     float | None = None,
    theta_deg:      float | None = None,
    recon_info:     dict | None = None,
) -> dict:


    valid_mask   = ~np.isnan(points_3d).any(axis=1)
    n_pts_total  = int(valid_mask.sum())


    rmse_m = compute_rmse(points_3d, mesh)


    n_neg     = rmse_m.get("n_negative_depth", 0)
    neg_ratio = n_neg / n_pts_total if n_pts_total > 0 else 1.0
    data_valid = neg_ratio <= 0.30


    cf_err = compute_contact_feature_errors(points_3d, m1_config)


    uniq_m = compute_uniqueness(
        camera_image,
        window_px = uniq_window_px,
        stride_px = uniq_stride_px,
        n_samples = uniq_n_samples,
    )


    if spacing_mm is not None and spacing_mm > 0:
        dz_design   = unique_measurement_depth_mm(spacing_mm, theta_deg, "design")
        dz_realized = unique_measurement_depth_mm(spacing_mm, theta_deg, "nn")
        dz_source   = "design_unique_depth"
    else:
        dz_design   = float(m1_config.get("depth_max_mm", 40.0))
        dz_realized = float("nan")
        dz_source   = "depth_max_legacy"

    delta_z_noise = rmse_m.get("rmse_all_mm", float("nan"))
    if delta_z_noise is None or np.isnan(delta_z_noise):
        delta_z_noise = rmse_m.get("rmse_mm", float("nan"))


    ce = compute_channel_efficiency(
        camera_image         = camera_image,
        n_points_3d          = n_pts_total,
        measurement_depth_mm = dz_design,
        rmse_mm              = delta_z_noise,
    )


    ct = compute_theoretical_max_eta(
        camera_image         = camera_image,
        n_valid_pixels_m3    = m3_info.get("n_valid_pixels", 0),
        measurement_depth_mm = dz_design,
        rmse_mm              = delta_z_noise,
    )


    if np.isfinite(dz_realized):
        ce_r = compute_channel_efficiency(
            camera_image         = camera_image,
            n_points_3d          = n_pts_total,
            measurement_depth_mm = dz_realized,
            rmse_mm              = delta_z_noise,
        )
        eta_realized      = ce_r.get("channel_efficiency", float("nan"))
        decoder_range_eff = dz_realized / dz_design if dz_design > 0 else float("nan")
    else:
        eta_realized      = float("nan")
        decoder_range_eff = float("nan")


    metrics: dict = {

        "data_valid":           data_valid,
        "n_pts_total":          n_pts_total,
        "neg_depth_ratio":      round(neg_ratio, 4),


        "rmse_mm":              rmse_m.get("rmse_mm",          float("nan")),
        "mae_mm":               rmse_m.get("mae_mm",           float("nan")),
        "max_err_mm":           rmse_m.get("max_err_mm",       float("nan")),
        "n_evaluated":          rmse_m.get("n_evaluated",      0),

        "rmse_all_mm":          rmse_m.get("rmse_all_mm",      float("nan")),
        "mae_all_mm":           rmse_m.get("mae_all_mm",       float("nan")),
        "max_err_all_mm":       rmse_m.get("max_err_all_mm",   float("nan")),
        "n_evaluated_all":      rmse_m.get("n_evaluated_all",  0),
        "neg_depth_err_mean":   rmse_m.get("neg_depth_err_mean", float("nan")),
        "neg_depth_err_max":    rmse_m.get("neg_depth_err_max",  float("nan")),
        "n_negative_depth":     rmse_m.get("n_negative_depth", 0),
        "coverage":             rmse_m.get("coverage",         float("nan")),
        "fov_coverage":         rmse_m.get("fov_coverage",     float("nan")),


        "cf_gt_z_std_mm":       cf_err.get("gt_central_z_std_mm",        float("nan")),
        "cf_gt_z_range_mm":     cf_err.get("gt_central_z_range_mm",      float("nan")),
        "cf_gt_depth_mm":       cf_err.get("gt_indentation_depth_mm",    float("nan")),
        "cf_err_z_std_mm":      cf_err.get("error_central_z_std_mm",     float("nan")),
        "cf_err_z_range_mm":    cf_err.get("error_central_z_range_mm",   float("nan")),
        "cf_err_depth_mm":      cf_err.get("error_indentation_depth_mm", float("nan")),
        "cf_center_covered":    cf_err.get("indentation_center_covered",  False),
        "cf_n_central_pts":     cf_err.get("n_central_points",            0),


        "uniq_mean":            uniq_m.get("uniqueness_mean",   float("nan")),
        "uniq_median":          uniq_m.get("uniqueness_median", float("nan")),
        "uniq_min":             uniq_m.get("uniqueness_min",    float("nan")),
        "uniq_n_windows":       uniq_m.get("n_windows",         0),
        "uniq_n_positions":     uniq_m.get("n_positions",       0),
        "uniq_n_blank":         uniq_m.get("n_blank",           0),
        "uniq_blank_ratio":     uniq_m.get("blank_ratio",       float("nan")),


        "eta":                  ce.get("channel_efficiency",  float("nan")),
        "rho_3d":               ce.get("rho_3d",              float("nan")),
        "dynamic_range":        ce.get("dynamic_range",       float("nan")),


        "eta_theoretical":      ct.get("eta_theoretical",      float("nan")),
        "rho_3d_theoretical":   ct.get("rho_3d_theoretical",   float("nan")),


        "eta_dz_mm":            round(dz_design, 4),
        "eta_dz_source":        dz_source,
        "eta_dz_realized_mm":   round(dz_realized, 4) if np.isfinite(dz_realized) else float("nan"),
        "decoder_range_eff":    round(decoder_range_eff, 4) if np.isfinite(decoder_range_eff) else float("nan"),
        "eta_realized":         eta_realized,
        "eta_noise_mm":         round(float(delta_z_noise), 6) if np.isfinite(delta_z_noise) else float("nan"),
    }


    indent_err = compute_indentation_errors(points_3d, m1_config)
    metrics.update(indent_err)


    ri = recon_info or {}
    nan = float("nan")
    metrics.update({
        "decoder":            ri.get("correspondence_method", ""),
        "n_detected":         ri.get("n_detected", 0),
        "n_matched":          ri.get("n_matched", 0),
        "n_rejected":         ri.get("n_rejected", 0),
        "n_ambiguous":        ri.get("n_ambiguous", 0),
        "ambiguity_ratio":    ri.get("ambiguity_ratio", nan),
        "rejection_ratio":    ri.get("rejection_ratio", nan),
        "corr_failure_ratio": ri.get("corr_failure_ratio", nan),
        "search_depth_mm":    ri.get("search_depth_mm", nan),
        "search_mode":        ri.get("search_mode", ""),
        "epi_best_dist_mean_px": ri.get("epi_best_dist_mean_px", nan),
        "epi_t_max":          ri.get("epi_t_max", nan),


        "n_ambiguous_second": ri.get("n_ambiguous_second", ri.get("n_ambiguous", 0)),
        "n_duplicate_claims": ri.get("n_duplicate_claims", 0),
        "n_area_rejected":    ri.get("n_area_rejected", 0),
        "n_elong_rejected":   ri.get("n_elong_rejected", 0),

        "anchor_readable":    ri.get("anchor_readable", ""),
        "n_anchor_used":      ri.get("n_anchor_used", ""),
        "drift_est_px":       ri.get("drift_est_px", nan),
        "drift_mad_px":       ri.get("drift_mad_px", nan),
        "n_propagated":       ri.get("n_propagated", ""),
        "n_unresolved":       ri.get("n_unresolved", ""),
        "anchor_frame_valid": ri.get("anchor_frame_valid", ""),
    })


    metrics.update(compute_ambiguity_metrics(ri, camera_image, points_3d))


    amb_mask = ri.get("ambiguous_mask")
    if amb_mask is None or len(amb_mask) != len(points_3d):
        amb_mask = np.zeros(len(points_3d), dtype=bool)
    amb_mask = np.asarray(amb_mask, dtype=bool)
    rel = compute_rmse(points_3d[~amb_mask], mesh) if amb_mask.any() else rmse_m
    metrics["rmse_reliable_mm"]     = rel.get("rmse_all_mm", nan)
    metrics["max_err_reliable_mm"]  = rel.get("max_err_all_mm", nan)
    metrics["n_reliable_evaluated"] = rel.get("n_evaluated_all", 0)


    metrics.update(_outlier_stats(rel, "reliable"))
    metrics.update(_outlier_stats(rmse_m, "all"))


    metrics.update(compute_indentation_recovery(points_3d, mesh, m1_config,
                                                ri.get("ambiguous_mask")))

    return metrics

# Run the standalone pattern-spacing sweep using the supplied grid and batch settings.
def run_sensitivity_sweep(
    raw_data_dir:   Path,
    pattern_types:  list[str]  = SWEEP_PATTERN_TYPES,
    spacing_values: list[float] = SWEEP_SPACING_MM,
    grid_step_x_mm: float      = SWEEP_GRID_STEP_X_MM,
    grid_step_y_mm: float      = SWEEP_GRID_STEP_Y_MM,
    batch_size:     int        = SWEEP_BATCH_SIZE,
    verbose:        bool       = True,
) -> dict:

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m1_tire_surface   import build_surface
    from m2_pattern        import generate_pattern
    from m3_virtual_camera import synthesise_camera_image
    from m4_reconstruction import reconstruct

    results = {pt: [] for pt in pattern_types}


    if verbose:
        print("[M1] Building tire surface ...")
    mesh, depth_map, m1_cfg = build_surface(
        "A",
        raw_data_dir   = raw_data_dir,
        grid_step_x_mm = grid_step_x_mm,
        grid_step_y_mm = grid_step_y_mm,
    )
    if verbose:
        print(f"     grid: {m1_cfg['grid_shape_ny_nx']}, "
              f"depth max: {m1_cfg['depth_max_mm']:.2f} mm")

    total = len(pattern_types) * len(spacing_values)
    run   = 0

    for pt in pattern_types:
        for sp in spacing_values:
            run += 1
            if verbose:
                print(f"\n[{run}/{total}] pattern={pt}, spacing={sp} mm")


            try:
                pattern, meta = generate_pattern(pt, spacing_mm=sp)
            except ValueError as e:
                if verbose:
                    print(f"     SKIP (M2 error): {e}")
                continue


            cam_img, _, _, m3_info = synthesise_camera_image(
                mesh_dict = mesh,
                pattern   = pattern,
                cfg       = {"batch_size": batch_size},
            )
            if verbose:
                print(f"     M3: hits={m3_info['n_hits']}, "
                      f"valid_px={m3_info['n_valid_pixels']}")


            pts_3d, recon_info = reconstruct(cam_img, pattern, pt)
            if verbose:
                print(f"     M4: n_points={recon_info['n_points_3d']}")


            rmse_metrics = compute_rmse(pts_3d, mesh)


            uniq_metrics = compute_uniqueness(cam_img)

            if verbose:
                print(f"     RMSE={rmse_metrics['rmse_mm']:.4f} mm, "
                      f"coverage={rmse_metrics['coverage']*100:.1f}%, "
                      f"uniqueness={uniq_metrics['uniqueness_mean']:.4f}")

            results[pt].append({
                "spacing_mm":      sp,
                "spatial_freq":    1.0 / sp,
                "density":         meta["density"],
                "rmse_mm":         rmse_metrics["rmse_mm"],
                "mae_mm":          rmse_metrics["mae_mm"],
                "max_err_mm":      rmse_metrics["max_err_mm"],
                "coverage":        rmse_metrics["coverage"],
                "n_evaluated":     rmse_metrics["n_evaluated"],
                "uniqueness_mean": uniq_metrics["uniqueness_mean"],
                "uniqueness_min":  uniq_metrics["uniqueness_min"],
                "n_points_3d":     recon_info["n_points_3d"],
                "metadata":        meta,
            })

    return results


# Plot point-height errors, their distribution and the recovered profile.
def visualise_rmse(
    points_3d:        np.ndarray,
    mesh:             dict,
    rmse_metrics:     dict,
    save_dir:         Path | None = None,
    show:             bool = False,
) -> None:

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rmse = rmse_metrics.get("rmse_mm", float("nan"))
    fig.suptitle(
        f"M5 RMSE Evaluation — RMSE = {rmse:.4f} mm  "
        f"(N = {rmse_metrics.get('n_evaluated', 0)})",
        fontsize=12,
    )

    d_recon = rmse_metrics.get("depth_recon",      np.array([]))
    d_gt    = rmse_metrics.get("depth_gt_interp",  np.array([]))


    ax = axes[0]
    if len(d_recon) > 0 and len(d_gt) > 0:
        in_range = ~np.isnan(d_gt)
        ax.scatter(d_gt[in_range], d_recon[in_range],
                   s=1, alpha=0.3, c="steelblue")
        lim = max(d_gt[in_range].max(), d_recon[in_range].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="ideal")
        ax.legend(fontsize=8)
    ax.set_xlabel("Ground truth depth [mm]")
    ax.set_ylabel("Reconstructed depth [mm]")
    ax.set_title("Recon vs GT")


    ax = axes[1]
    valid = ~np.isnan(points_3d).any(axis=1)
    pts   = points_3d[valid]
    if len(pts) > 0 and len(d_recon) > 0 and len(d_gt) > 0:
        in_range = ~np.isnan(d_recon[:len(pts)]) & ~np.isnan(d_gt[:len(pts)])
        errors   = np.abs(d_recon[:len(pts)] - d_gt[:len(pts)])
        sc = ax.scatter(
            pts[in_range, 0], pts[in_range, 1],
            c=errors[in_range], cmap="hot_r", s=2, alpha=0.6,
            vmin=0, vmax=max(errors[in_range].max(), 0.01),
        )
        plt.colorbar(sc, ax=ax, label="|error| [mm]")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_title("Depth error spatial map")


    ax = axes[2]
    if len(d_recon) > 0 and len(d_gt) > 0:
        in_range = ~np.isnan(d_gt)
        errors   = d_recon[in_range] - d_gt[in_range]
        ax.hist(errors, bins=40, color="steelblue", edgecolor="white", lw=0.3)
        ax.axvline(0, color="red", lw=1, ls="--")
        ax.set_xlabel("Depth error [mm]")
        ax.set_ylabel("Count")
        ax.set_title(f"Error distribution  (MAE={rmse_metrics.get('mae_mm',float('nan')):.4f} mm)")

    plt.tight_layout()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / "m5_rmse_evaluation.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
    if show:
        plt.show()
    plt.close(fig)


# Plot error and coverage against the supplied pattern-spacing sweep.
def visualise_sensitivity_curves(
    results:  dict,
    save_dir: Path | None = None,
    show:     bool = False,
) -> None:

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("M5 Sensitivity Analysis", fontsize=13)

    colors = {"lines": "steelblue", "dots": "darkorange"}
    markers = {"lines": "o", "dots": "s"}

    for pt, records in results.items():
        if not records:
            continue
        freqs  = [r["spatial_freq"]    for r in records]
        rmses  = [r["rmse_mm"]         for r in records]
        uniqs  = [r["uniqueness_mean"] for r in records]
        color  = colors.get(pt, "gray")
        marker = markers.get(pt, "^")


        axes[0].plot(freqs, rmses, marker=marker, color=color,
                     label=pt, lw=1.5, ms=6)

        for f, r, sp in zip(freqs, rmses, [rec["spacing_mm"] for rec in records]):
            if not np.isnan(r):
                axes[0].annotate(
                    f"{sp}mm", (f, r),
                    textcoords="offset points", xytext=(4, 4), fontsize=7,
                    color=color,
                )


        axes[1].plot(freqs, uniqs, marker=marker, color=color,
                     label=pt, lw=1.5, ms=6)

    axes[0].set_xlabel("Spatial frequency [lp/mm]")
    axes[0].set_ylabel("RMSE [mm]")
    axes[0].set_title("Reconstruction RMSE vs Pattern Frequency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)

    axes[1].set_xlabel("Spatial frequency [lp/mm]")
    axes[1].set_ylabel("Uniqueness score (NCC)")
    axes[1].set_title("Pattern Uniqueness vs Pattern Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / "m5_sensitivity_curves.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
    if show:
        plt.show()
    plt.close(fig)


# Write scalar sweep results while excluding array-valued outputs.
def save_results_csv(
    results:  dict,
    save_dir: Path,
) -> None:

    import csv

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / "m5_sensitivity_results.csv"

    fieldnames = [
        "pattern_type", "spacing_mm", "spatial_freq", "density",
        "rmse_mm", "mae_mm", "max_err_mm", "coverage",
        "uniqueness_mean", "uniqueness_min", "n_points_3d", "n_evaluated",
    ]

    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pt, records in results.items():
            for r in records:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"  Saved: {fname}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    ROOT    = Path(__file__).resolve().parents[1]
    OUT_FIG = ROOT / "outputs" / "figures"
    OUT_DAT = ROOT / "outputs" / "data"
    RAW_DIR = ROOT / "data" / "raw"

    print("=" * 60)
    print("M5 Evaluation — Self-test")
    print("=" * 60)


    from m1_tire_surface   import build_surface
    from m2_pattern        import generate_pattern
    from m3_virtual_camera import synthesise_camera_image
    from m4_reconstruction import reconstruct


    print("\n[M1] Building tire surface (coarse grid) ...")
    mesh, depth_map, m1_cfg = build_surface(
        "A", raw_data_dir=RAW_DIR,
        grid_step_x_mm=5.0, grid_step_y_mm=2.0,
    )
    print(f"     grid: {m1_cfg['grid_shape_ny_nx']}, "
          f"depth max: {m1_cfg['depth_max_mm']:.2f} mm")


    print("\n[Single run] lines, spacing=2.0 mm")
    pattern, meta = generate_pattern("lines", spacing_mm=2.0, feature_size_mm=0.3)
    cam_img, _, _, _ = synthesise_camera_image(mesh, pattern, {"batch_size": 512})
    pts_3d, recon_info = reconstruct(cam_img, pattern, "lines")
    print(f"     n_points_3d: {recon_info['n_points_3d']}")


    print("\n[Layer A] Computing RMSE ...")
    rmse_m = compute_rmse(pts_3d, mesh)
    print(f"     RMSE       = {rmse_m['rmse_mm']:.4f} mm")
    print(f"     MAE        = {rmse_m['mae_mm']:.4f} mm")
    print(f"     max error  = {rmse_m['max_err_mm']:.4f} mm")
    print(f"     n_evaluated= {rmse_m['n_evaluated']}")
    print(f"     coverage   = {rmse_m['coverage']*100:.1f}%")
    assert not np.isnan(rmse_m["rmse_mm"]) or rmse_m["n_evaluated"] == 0, \
        "ERROR: RMSE is NaN but n_evaluated > 0"
    print("     ✅ Layer A passed")


    print("\n[Layer B] Computing uniqueness ...")
    uniq_m = compute_uniqueness(cam_img)
    print(f"     uniqueness_mean   = {uniq_m['uniqueness_mean']:.4f}")
    print(f"     uniqueness_median = {uniq_m['uniqueness_median']:.4f}")
    print(f"     uniqueness_min    = {uniq_m['uniqueness_min']:.4f}")
    print(f"     n_windows         = {uniq_m['n_windows']}")
    assert uniq_m["n_windows"] > 0, "ERROR: no windows evaluated"
    assert 0 <= uniq_m["uniqueness_mean"] <= 1, "ERROR: uniqueness out of range"
    print("     ✅ Layer B passed")


    print("\n[Visualise] RMSE figure ...")
    visualise_rmse(pts_3d, mesh, rmse_m, save_dir=OUT_FIG)


    print("\n[Layer C] Mini sensitivity sweep (lines only, 3 spacings) ...")
    sweep_results = run_sensitivity_sweep(
        raw_data_dir   = RAW_DIR,
        pattern_types  = ["lines"],
        spacing_values = [1.5, 2.0, 3.0],
        verbose        = True,
    )

    records = sweep_results["lines"]
    assert len(records) > 0, "ERROR: no sweep records"
    print(f"\n  Sweep summary:")
    print(f"  {'spacing':>8}  {'freq':>8}  {'RMSE':>8}  {'uniq':>8}  {'cover':>8}")
    for r in records:
        print(f"  {r['spacing_mm']:>8.1f}  "
              f"{r['spatial_freq']:>8.3f}  "
              f"{r['rmse_mm']:>8.4f}  "
              f"{r['uniqueness_mean']:>8.4f}  "
              f"{r['coverage']*100:>7.1f}%")
    print("  ✅ Layer C passed")


    print("\n[Visualise] Sensitivity curves ...")
    visualise_sensitivity_curves(sweep_results, save_dir=OUT_FIG)


    print("\n[Save] Results CSV ...")
    save_results_csv(sweep_results, save_dir=OUT_DAT)

    print("\n" + "=" * 60)
    print("M5 ✅ All tests passed")
    print("=" * 60)
