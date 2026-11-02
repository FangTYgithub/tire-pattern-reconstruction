# Run standalone size, spacing and noise sweeps and summarize repeated phases.

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


ROOT    = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
RAW_DIR = ROOT / "data" / "raw"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_DAT = ROOT / "outputs" / "data"

sys.path.insert(0, str(SRC_DIR))


DEFAULT_PATTERN_TYPES  = ["dots"]
DEFAULT_SPACINGS_MM    = [5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0]
DEFAULT_VARIANT        = "A"
RANDOM_TYPES           = ("random_dots", "graded_dots", "epi_random_dots")
DEFAULT_FEATURE_SIZE   = {
    "dots":   1.0,
    "random_dots": 1.0,
    "graded_dots": 1.0,
    "coded_dots": 1.0,
    "bright_dots": 1.0,
    "epi_random_dots": 1.0,
    "anchored_dots": 1.0,
    "dashed_vlines": 0.5,
    "lines":  2.0,
    "vlines": 1.0,
}

GRID_STEP_X_MM = 5.0
GRID_STEP_Y_MM = 2.0
BATCH_SIZE     = 512
NOISE_SIGMA    = 0.0

UNIQ_WINDOW_PX = 32
UNIQ_STRIDE_PX = 16
UNIQ_N_SAMPLES = 200


# Render, reconstruct and evaluate one pattern with the supplied model settings.
def _single_run(
    mesh:            dict,
    m1_config:       dict,
    pattern_type:    str,
    spacing_mm:      float,
    feature_size_mm: float,
    noise_sigma:     float = 0.0,
    phase_offset_x_mm: float = 0.0,
    phase_offset_y_mm: float = 0.0,
    phase_idx:       int   = 0,
    decoder:         str   = "epipolar",
    renderer:        str   = "psf",
    pattern_seed:    int   = 0,
    psf_sigma_px:    float | None = None,
    min_sep_frac:    float = 0.6,
    measurement_volume_mm: float | None = None,
    spacing_perp_mm: float | None = None,
    stagger_frac:    float = 0.0,
    jitter_frac:     float = 0.0,
    rotation_deg:    float = 0.0,
    shear_frac:      float = 0.0,
    baseline_mm:     float | None = None,
    albedo:          float = 1.0,
    albedo_exposure: float = 1.0,
    specular_k:      float = 0.0,
    mottle_amp:      float = 0.0,
    outlier_frac:    float = 0.0,
    cam_rot_deg:     tuple = (0.0, 0.0, 0.0),
    cam_trans_mm:    tuple = (0.0, 0.0, 0.0),
    rays_per_pixel:  int   = 1,
    size_levels:     int   = 2,
    size_ratio:      float = 0.6,
    code_shift:      int   = 1,
    bright_ratio:    float = 0.5,
    code_ref:        str   = "global",
    epi_along_mm:    float = 6.5,
    epi_across_mm:   float = 1.08,
    dash_length_mm:  float = 5.0,
    dash_gap_mm:     float = 1.0,
    dash_phases:     int   = 2,
    epi_tol_px:      float | None = None,
    anchor_ratio:    float = 1.5,
    anchor_every:    tuple = (2, 4),
    anchor_tol_px:   float | None = None,
    propagate:       bool = True,
    dash_levels:     int   = 1,
    dash_ratio:      float = 0.5,
    verbose:         bool = False,
) -> dict | None:

    from m2_pattern        import generate_pattern
    from m3_virtual_camera import synthesise_camera_image
    from m4_reconstruction import reconstruct
    from m5_evaluation     import evaluate_all


    actual_dz = m1_config.get("fov_z_range_mm", None)
    if actual_dz is None or (isinstance(actual_dz, float) and np.isnan(actual_dz)):
        actual_dz = m1_config.get("contact_features", {}).get("central_z_range_mm", None)

    try:
        pattern, meta = generate_pattern(
            pattern_type,
            spacing_mm      = spacing_mm,
            feature_size_mm = feature_size_mm,
            depth_range_mm  = actual_dz,
            phase_offset_x_mm = phase_offset_x_mm,
            phase_offset_y_mm = phase_offset_y_mm,
            seed            = pattern_seed,
            min_sep_frac    = min_sep_frac,
            spacing_perp_mm = spacing_perp_mm,
            stagger_frac    = stagger_frac,
            jitter_frac     = jitter_frac,
            rotation_deg    = rotation_deg,
            shear_frac      = shear_frac,
            size_levels     = size_levels,
            size_ratio      = size_ratio,
            code_shift      = code_shift,
            bright_ratio    = bright_ratio,
            epi_along_mm    = epi_along_mm,
            epi_across_mm   = epi_across_mm,
            dash_length_mm  = dash_length_mm,
            dash_gap_mm     = dash_gap_mm,
            dash_phases     = dash_phases,
            anchor_ratio    = anchor_ratio,
            anchor_every    = tuple(int(x) for x in anchor_every),
            dash_levels     = int(dash_levels),
            dash_ratio      = float(dash_ratio),
        )
    except ValueError as e:
        if verbose:
            print(f"     [M2 skip] {e}")
        return None


    m3_cfg = {"batch_size": BATCH_SIZE, "noise_sigma": noise_sigma}
    if renderer == "legacy":
        m3_cfg.update({"deposit": "max", "psf_sigma_px": 0.0})
    elif psf_sigma_px is not None:
        m3_cfg["psf_sigma_px"] = float(psf_sigma_px)
    if baseline_mm:
        m3_cfg["baseline_mm"] = float(baseline_mm)

    m3_cfg.update({"albedo": float(albedo), "albedo_exposure": float(albedo_exposure),
                   "specular_k": float(specular_k), "mottle_amp": float(mottle_amp),
                   "outlier_frac": float(outlier_frac), "mottle_seed": int(pattern_seed)})
    if any(cam_rot_deg) or any(cam_trans_mm):
        m3_cfg.update({"cam_rot_deg": tuple(float(a) for a in cam_rot_deg),
                       "cam_trans_mm": tuple(float(a) for a in cam_trans_mm)})
    if rays_per_pixel and int(rays_per_pixel) > 1:
        m3_cfg["rays_per_pixel"] = int(rays_per_pixel)
    cam_img, _, _, m3_info = synthesise_camera_image(
        mesh_dict = mesh,
        pattern   = pattern,
        cfg       = m3_cfg,
    )


    sensor_cfg: dict = {}
    if measurement_volume_mm:
        sensor_cfg["measurement_volume_mm"] = float(measurement_volume_mm)
    if baseline_mm:
        sensor_cfg["baseline_mm"] = float(baseline_mm)
    if pattern_type == "coded_dots":
        from m2_pattern import coded_dot_radii
        sensor_cfg["size_levels"] = int(meta.get("size_levels", size_levels))
        sensor_cfg["size_radii_px"] = [r / float(meta["mm_per_px"])
                                       for r in coded_dot_radii(feature_size_mm, size_levels, size_ratio)]
    if pattern_type == "bright_dots":
        from m2_pattern import bright_levels
        sensor_cfg["size_levels"] = int(meta.get("size_levels", size_levels))
        sensor_cfg["bright_levels"] = bright_levels(size_levels, bright_ratio)
    if pattern_type in ("coded_dots", "bright_dots") and code_ref and code_ref != "global":
        sensor_cfg["code_ref"] = str(code_ref)
    if pattern_type == "anchored_dots":
        sensor_cfg.update({"size_levels": 2, "size_radii_px": [feature_size_mm / float(meta["mm_per_px"]),
                                                              feature_size_mm * float(anchor_ratio) / float(meta["mm_per_px"])],
                           "anchor_every": tuple(int(x) for x in anchor_every), "propagate": bool(propagate)})
        if anchor_tol_px is not None:
            sensor_cfg["anchor_tol_px"] = float(anchor_tol_px)
    if pattern_type == "dashed_vlines" and int(dash_levels) > 1:
        from m2_pattern import dash_lengths
        sensor_cfg.update({"dash_levels": int(dash_levels), "dash_ratio": float(dash_ratio),
                           "dash_lengths_px": [x / float(meta["mm_per_px"])
                                               for x in dash_lengths(dash_length_mm, int(dash_levels), dash_ratio)]})
    if epi_tol_px is not None:
        sensor_cfg["epi_tol_px"] = float(epi_tol_px)
        sensor_cfg["epi_ambig_tol_px"] = float(epi_tol_px)
    pts_3d, recon_info = reconstruct(cam_img, pattern, pattern_type, method=decoder,
                                     sensor_cfg=sensor_cfg or None)
    theta_deg = (float(np.degrees(2.0 * np.arctan(float(baseline_mm) / 2.0 / m3_info["working_dist_mm"])))
                 if baseline_mm else None)


    metrics = evaluate_all(
        points_3d    = pts_3d,
        mesh         = mesh,
        camera_image = cam_img,
        m1_config    = m1_config,
        m3_info      = m3_info,
        uniq_window_px = UNIQ_WINDOW_PX,
        uniq_stride_px = UNIQ_STRIDE_PX,
        uniq_n_samples = UNIQ_N_SAMPLES,
        spacing_mm     = spacing_mm,
        theta_deg      = theta_deg,
        recon_info     = recon_info,
    )

    return {

        "pattern_type":    pattern_type,
        "spacing_mm":      spacing_mm,
        "feature_size_mm": feature_size_mm,
        "noise_sigma":     noise_sigma,
        "phase_idx":       phase_idx,
        "phase_offset_x_mm": round(phase_offset_x_mm, 4),
        "phase_offset_y_mm": round(phase_offset_y_mm, 4),
        "indent_depth_mm": float(m1_config.get("indent_depth_mm", 0.0) or 0.0),
        "indent_width_mm": float(m1_config.get("indent_width_mm", 0.0) or 0.0),
        "renderer":        renderer,
        "variant":         str(m1_config.get("variant", "")),
        "psf_sigma_px":    float(m3_info.get("psf_sigma_px", float("nan"))),
        "min_sep_frac":    float(min_sep_frac) if pattern_type in RANDOM_TYPES else "",
        "measurement_volume_mm": float(measurement_volume_mm) if measurement_volume_mm else "",
        "spacing_perp_mm": meta.get("spacing_perp_mm", ""),
        "stagger_frac":    meta.get("stagger_frac", ""),
        "jitter_frac":     meta.get("jitter_frac", ""),
        "lattice":         meta.get("lattice", ""),
        "rotation_deg":    meta.get("rotation_deg", ""),
        "shear_frac":      meta.get("shear_frac", ""),
        "baseline_mm":     float(m3_info.get("baseline_mm", float("nan"))),
        "indent_shape":    str(m1_config.get("indent_shape", "gaussian")),
        "indent_order":    float(m1_config.get("indent_order", 2.0) or 2.0),
        "indent_center_x_mm": float(m1_config.get("indent_center_x_mm", 0.0) or 0.0),
        "indent_aspect":   float(m1_config.get("indent_aspect", 1.0) or 1.0),
        "curv_x_1_per_mm": float(m1_config.get("curv_x_1_per_mm", 0.0) or 0.0),
        "curv_y_extra_1_per_mm": float(m1_config.get("curv_y_extra_1_per_mm", 0.0) or 0.0),
        "indent_asym":     float(m1_config.get("indent_asym", 1.0) or 1.0),
        "indent_asym_angle_deg": float(m1_config.get("indent_asym_angle_deg", 0.0) or 0.0),
        "indent2_depth_mm": float(m1_config.get("indent2_depth_mm", 0.0) or 0.0),
        "indent2_width_mm": float(m1_config.get("indent2_width_mm", 0.0) or 0.0),
        "indent2_center_x_mm": float(m1_config.get("indent2_center_x_mm", 0.0) or 0.0),
        "indent2_center_y_mm": float(m1_config.get("indent2_center_y_mm", 0.0) or 0.0),
        "albedo":          float(m3_info.get("albedo", 1.0)),
        "albedo_exposure": float(m3_info.get("albedo_exposure", 1.0)),
        "specular_k":      float(m3_info.get("specular_k", 0.0)),
        "mottle_amp":      float(m3_info.get("mottle_amp", 0.0)),
        "outlier_frac":    float(m3_info.get("outlier_frac", 0.0)),
        "splat":           str(m3_info.get("splat", "bilinear")),
        "cam_pitch_deg":   float(m3_info.get("cam_rot_deg", (0, 0, 0))[0]),
        "cam_yaw_deg":     float(m3_info.get("cam_rot_deg", (0, 0, 0))[1]),
        "cam_roll_deg":    float(m3_info.get("cam_rot_deg", (0, 0, 0))[2]),
        "cam_shift_along_mm":  float(m3_info.get("cam_trans_mm", (0, 0, 0))[0]),
        "cam_shift_across_mm": float(m3_info.get("cam_trans_mm", (0, 0, 0))[1]),
        "cam_shift_axial_mm":  float(m3_info.get("cam_trans_mm", (0, 0, 0))[2]),
        "rays_per_pixel":  int(m3_info.get("rays_per_pixel", 1) or 1),
        "size_levels":     int(meta["size_levels"]) if "size_levels" in meta else "",
        "size_ratio":      float(meta["size_ratio"]) if "size_ratio" in meta else "",
        "code_shift":      int(meta["code_shift"]) if "code_shift" in meta else "",
        "bright_ratio":    float(meta["bright_ratio"]) if "bright_ratio" in meta else "",
        "code_ref":        str(recon_info.get("code_ref", "")) if pattern_type in ("coded_dots", "bright_dots") else "",
        "epi_along_mm":    meta.get("epi_along_mm", ""),
        "epi_across_mm":   meta.get("epi_across_mm", ""),
        "epi_saturated":   meta.get("epi_saturated", ""),
        "n_placed":        meta.get("n_placed", ""),
        "same_class_pitch_mm": float(meta["same_class_pitch_mm"]) if "same_class_pitch_mm" in meta else "",
        "dash_length_mm":  float(meta["dash_length_mm"]) if "dash_length_mm" in meta else "",
        "dash_gap_mm":     float(meta["dash_gap_mm"]) if "dash_gap_mm" in meta else "",
        "dash_phases":     int(meta["dash_phases"]) if "dash_phases" in meta else "",
        "dash_levels":     int(meta["dash_levels"]) if "dash_levels" in meta else "",
        "dash_ratio":      float(meta["dash_ratio"]) if "dash_ratio" in meta else "",
        "size_class_ratio":  recon_info.get("size_class_ratio", ""),
        "size_class_dprime": recon_info.get("size_class_dprime", ""),
        "n_size_uncertain":  recon_info.get("n_size_uncertain", ""),
        "size_code_readable": recon_info.get("size_code_readable", ""),
        "size_class_balance": recon_info.get("size_class_balance", ""),
        "anchor_ratio":    float(meta["anchor_ratio"]) if "anchor_ratio" in meta else "",
        "anchor_every":    meta.get("anchor_every", ""),
        "anchor_tol_px":   recon_info.get("anchor_tol_px", ""),
        "propagate":       recon_info.get("propagate", ""),
        "epi_tol_px":      float(epi_tol_px) if epi_tol_px is not None else "",
        "pattern_seed":    int(pattern_seed) if (pattern_type in RANDOM_TYPES or jitter_frac > 0) else "",
        "regularity_index": meta.get("regularity_index", float("nan")),
        "nn_dist_mean_mm": meta.get("nn_dist_mean_mm", float("nan")),
        "spatial_freq":    1.0 / spacing_mm,


        "density":           meta["density"],
        "feature_count":     meta["feature_count"],
        "uniqueness_ratio":  meta.get("uniqueness_ratio",    float("nan")),
        "aliasing_risk":     meta.get("aliasing_risk",       None),
        "depth_range_used":  meta.get("depth_range_used_mm", float("nan")),


        **metrics,


        "n_points_3d": recon_info["n_points_3d"],
        "n_valid_px":  m3_info["n_valid_pixels"],


        "_pattern":   pattern,
        "_cam_img":   cam_img,
        "_pts_3d":    pts_3d,
    }


def _row_key(rec: dict) -> tuple:

    return (
        str(rec["pattern_type"]),
        f"{float(rec['spacing_mm']):.4f}",
        f"{float(rec['feature_size_mm']):.4f}",
        f"{float(rec.get('noise_sigma', 0.0)):.4f}",
        str(int(float(rec.get("phase_idx", 0)))),
    )


def _typed(row: dict) -> dict:

    out: dict = {}
    for k, v in row.items():
        s = "" if v is None else str(v).strip()
        if s.lower() == "true":
            out[k] = True
        elif s.lower() == "false":
            out[k] = False
        elif s == "":
            out[k] = ""
        else:
            try:
                out[k] = float(s)
            except ValueError:
                out[k] = s
    return out


def load_partial(path: Path) -> list[dict]:

    if not path.exists():
        return []

    with open(path, newline="", encoding="utf-8-sig") as f:
        return [_typed(r) for r in csv.DictReader(f)]


def append_partial(path: Path, rec: dict) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({k: rec.get(k, "") for k in CSV_FIELDS})
        f.flush()


# Run the selected sweep and save per-case results and resumable checkpoints.
def run_preliminary(
    pattern_types:   list[str]   = DEFAULT_PATTERN_TYPES,
    spacings_mm:     list[float] = DEFAULT_SPACINGS_MM,
    feature_sizes_mm: list[float] | None = None,
    noise_sigmas:    list[float] | None = None,
    variant:          str = DEFAULT_VARIANT,
    n_phases:         int = 1,
    phase_seed:       int = 42,
    checkpoint_path:  Path | None = None,
    resume:           bool = False,
    indent_depth_mm:  float = 0.0,
    indent_width_mm:  float | None = None,
    decoder:          str = "epipolar",
    renderer:         str = "psf",
    psf_sigma_px:     float | None = None,
    min_sep_frac:     float = 0.6,
    measurement_volume_mm: float | None = None,
    spacing_perp_mm:  float | None = None,
    stagger_frac:     float = 0.0,
    jitter_frac:      float = 0.0,
    rotation_deg:     float = 0.0,
    shear_frac:       float = 0.0,
    baseline_mm:      float | None = None,
    indent_shape:     str = "gaussian",
    indent_order:     float = 2.0,
    indent_center_x_mm: float = 0.0,
    indent_aspect:    float = 1.0,
    albedo:           float = 1.0,
    albedo_exposure:  float = 1.0,
    specular_k:       float = 0.0,
    mottle_amp:       float = 0.0,
    outlier_frac:     float = 0.0,
    cam_rot_deg:      tuple = (0.0, 0.0, 0.0),
    cam_trans_mm:     tuple = (0.0, 0.0, 0.0),
    rays_per_pixel:   int = 1,
    size_levels:      int = 2,
    size_ratio:       float = 0.6,
    code_shift:       int = 1,
    bright_ratio:     float = 0.5,
    code_ref:         str = "global",
    epi_along_mm:     float = 6.5,
    epi_across_mm:    float = 1.08,
    dash_length_mm:   float = 5.0,
    dash_gap_mm:      float = 1.0,
    dash_phases:      int = 2,
    epi_tol_px:       float | None = None,
    anchor_ratio:     float = 1.5,
    anchor_every:     tuple = (2, 4),
    anchor_tol_px:    float | None = None,
    propagate:        bool = True,
    dash_levels:      int = 1,
    dash_ratio:       float = 0.5,
    m1_extra:         dict | None = None,
    verbose:          bool = True,
) -> dict[str, list[dict]]:

    from m1_tire_surface import build_surface

    results: dict[str, list[dict]] = {pt: [] for pt in pattern_types}


    done_keys: set[tuple] = set()
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)

        import json
        from m1_tire_surface import DEFAULTS as surface_defaults
        geometry = dict(variant=variant, indent_depth_mm=indent_depth_mm,
                        indent_width_mm=indent_width_mm, indent_shape=indent_shape,
                        indent_order=indent_order, indent_center_x_mm=indent_center_x_mm,
                        indent_aspect=indent_aspect,
                        extra={k: v for k, v in (m1_extra or {}).items() if v != surface_defaults.get(k)})
        signature = json.dumps(geometry, sort_keys=True)
        meta_path = checkpoint_path.with_suffix(".geometry.json")
        if resume and checkpoint_path.exists():
            if meta_path.exists():
                if json.loads(meta_path.read_text(encoding="utf-8")) != json.loads(signature):
                    raise ValueError("Checkpoint surface geometry differs; use a new out_name or disable resume")
            elif geometry["extra"]:
                raise ValueError("Legacy checkpoint has no geometry signature; cannot resume an extended surface")
        else:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(signature, encoding="utf-8")
        if resume:
            prev = load_partial(checkpoint_path)
            for rec in prev:
                pt_prev = str(rec.get("pattern_type", ""))
                if pt_prev in results:
                    results[pt_prev].append(rec)
                    done_keys.add(_row_key(rec))
            if verbose:
                print(f"[Resume] {len(done_keys)} finished run(s) loaded from "
                      f"{checkpoint_path.name}; they will be skipped.")
        elif checkpoint_path.exists():
            checkpoint_path.unlink()
            if verbose:
                print(f"[Checkpoint] discarded stale {checkpoint_path.name} (no --resume).")


    if verbose:
        print("[M1] Building tire surface (shared across all runs) ...")
    m1_kwargs = dict(raw_data_dir=RAW_DIR,
                     grid_step_x_mm=GRID_STEP_X_MM, grid_step_y_mm=GRID_STEP_Y_MM)
    if indent_depth_mm and indent_depth_mm > 0:
        m1_kwargs["indent_depth_mm"] = float(indent_depth_mm)
        if indent_width_mm:
            m1_kwargs["indent_width_mm"] = float(indent_width_mm)
        m1_kwargs.update({"indent_shape": str(indent_shape), "indent_order": float(indent_order),
                          "indent_center_x_mm": float(indent_center_x_mm),
                          "indent_aspect": float(indent_aspect)})
    if m1_extra:
        m1_kwargs.update(m1_extra)
    mesh, _, m1_cfg = build_surface(variant, **m1_kwargs)
    if verbose:
        print(f"     grid={m1_cfg['grid_shape_ny_nx']}  "
              f"depth_max={m1_cfg['depth_max_mm']:.2f} mm  "
              f"fov_z_range={m1_cfg.get('fov_z_range_mm', float('nan')):.2f} mm")
        if m1_cfg.get("has_indentation"):
            ind = m1_cfg["indentation"]
            print(f"     indentation: depth={ind['depth_mm']:.1f} mm  FWHM={ind['width_fwhm_mm']:.1f} mm  "
                  f"area={ind['area_mm2']:.0f} mm²  max_slope={ind['max_slope_deg']:.1f}°  in_fov={ind['in_fov']}")
        if m1_cfg.get("has_curvature"):
            cv = m1_cfg["curvature"]
            print(f"     curvature: curv_x={cv['curv_x_1_per_mm']} /mm  curv_y_extra={cv['curv_y_extra_1_per_mm']} /mm  "
                  f"z_add range {cv['z_add_min_mm']:.2f}…{cv['z_add_max_mm']:.2f} mm")
        if m1_cfg.get("double_indentation"):
            dd = m1_cfg["double_indentation"]
            print(f"     second dent: separation {dd['separation_mm']:.1f} mm  peak_total {dd['peak_total_grid_mm']:.2f} mm  "
                  f"local maxima {dd['n_local_maxima_grid']}  (single-dent metrics reported NaN)")

    ns_list = noise_sigmas if noise_sigmas is not None else [NOISE_SIGMA]

    for pt in pattern_types:

        default_fs = DEFAULT_FEATURE_SIZE.get(pt, 0.15)
        fs_list    = feature_sizes_mm if feature_sizes_mm is not None else [default_fs]
        n_ph  = max(1, int(n_phases))
        quads = [(sp, fs, ns, k) for sp in spacings_mm
                 for fs in fs_list for ns in ns_list for k in range(n_ph)]

        total = len(quads)
        t_all = time.time()

        for run_idx, (sp, fs, ns, k) in enumerate(quads, 1):
            t0 = time.time()


            if done_keys and _row_key({"pattern_type": pt, "spacing_mm": sp,
                                       "feature_size_mm": fs, "noise_sigma": ns,
                                       "phase_idx": k}) in done_keys:
                if verbose:
                    print(f"[{run_idx}/{total}] {pt}  spacing={sp} mm  "
                          f"feature_size={fs} mm  phase={k}  — resumed, skipped")
                continue


            pseed = int(phase_seed) * 1000 + int(k)
            if k == 0 or pt in RANDOM_TYPES:
                ox = oy = 0.0
            else:
                rng = np.random.default_rng(
                    [int(phase_seed), int(round(sp * 100)),
                     int(round(fs * 100)), int(round(ns * 10)), int(k)]
                )
                ox, oy = (float(v) for v in rng.uniform(0.0, sp, size=2))

            if verbose:
                ns_tag = f"  noise={ns}" if ns > 0 else ""
                if pt in RANDOM_TYPES:
                    ph_tag = f"  seed={pseed}" if n_ph > 1 else ""
                else:
                    ph_tag = (f"  phase={k}/{n_ph-1} (dx={ox:.2f}, dy={oy:.2f} mm)"
                              if n_ph > 1 else "")
                print(f"\n[{run_idx}/{total}] {pt}  spacing={sp} mm  "
                      f"feature_size={fs} mm{ns_tag}{ph_tag} ...")

            rec = _single_run(mesh, m1_cfg, pt, sp, fs,
                              noise_sigma=ns,
                              phase_offset_x_mm=ox, phase_offset_y_mm=oy,
                              phase_idx=k, decoder=decoder, renderer=renderer,
                              pattern_seed=pseed, psf_sigma_px=psf_sigma_px,
                              min_sep_frac=min_sep_frac,
                              measurement_volume_mm=measurement_volume_mm,
                              spacing_perp_mm=spacing_perp_mm, stagger_frac=stagger_frac,
                              jitter_frac=jitter_frac, rotation_deg=rotation_deg, shear_frac=shear_frac,
                              baseline_mm=baseline_mm,
                              albedo=albedo, albedo_exposure=albedo_exposure, specular_k=specular_k,
                              mottle_amp=mottle_amp, outlier_frac=outlier_frac,
                              cam_rot_deg=cam_rot_deg, cam_trans_mm=cam_trans_mm, rays_per_pixel=rays_per_pixel,
                              size_levels=size_levels, size_ratio=size_ratio, code_shift=code_shift,
                              bright_ratio=bright_ratio, code_ref=code_ref,
                              epi_along_mm=epi_along_mm, epi_across_mm=epi_across_mm,
                              dash_length_mm=dash_length_mm, dash_gap_mm=dash_gap_mm,
                              dash_phases=dash_phases, epi_tol_px=epi_tol_px,
                              anchor_ratio=anchor_ratio, anchor_every=anchor_every, anchor_tol_px=anchor_tol_px,
                              propagate=propagate, dash_levels=dash_levels, dash_ratio=dash_ratio, verbose=verbose)

            if rec is None:
                if verbose:
                    print("     Skipped.")
                continue

            results[pt].append(rec)
            if checkpoint_path is not None:
                append_partial(checkpoint_path, rec)

            if verbose:
                valid_mark = '✅' if rec.get('data_valid', True) else '❌'
                amb = rec.get('ambiguity_ratio', float('nan'))
                rej = rec.get('rejection_ratio', float('nan'))
                print(f"     {valid_mark} RMSE={rec['rmse_mm']:.4f}/{rec.get('rmse_all_mm', float('nan')):.4f} mm  "
                      f"uniq={rec.get('uniq_mean',float('nan')):.3f}  "
                      f"amb={amb:.3f} rej={rej:.3f}  N={rec.get('n_points_3d', 0)}  "
                      f"({time.time()-t0:.1f}s)")

    if verbose:
        print(f"\n  Total elapsed: {time.time()-t_all:.1f} s")

    return results


# Plot reconstruction metrics against spacing for each pattern family.
def plot_sensitivity_curves(
    results:  dict[str, list[dict]],
    save_dir: Path | None = None,
    show:     bool = False,
    tag:      str | None = None,
    subtitle: str | None = None,
) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#f8f8f8")
    fig.suptitle(
        "Preliminary Sensitivity Analysis\n"
        + (subtitle if subtitle else "Tire Inner Surface — Dot/Line Pattern Sweep"),
        fontsize=12, fontweight="bold", y=1.01,
    )

    COLORS  = {"dots": "#2176AE", "lines": "#E07A5F", "vlines": "#E07A5F",
               "random_dots": "#57A773", "graded_dots": "#8E5EA2", "default": "#555555"}
    MARKERS = {"dots": "o", "lines": "s", "vlines": "s",
               "random_dots": "D", "graded_dots": "v", "default": "^"}

    LINESTYLES = ["-", "--", "-.", ":"]

    def _agg(vals: list[float]) -> tuple[float, float]:
        arr = np.array(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return float("nan"), float("nan")
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return float(arr.mean()), std

    for pt, records in results.items():
        if not records:
            continue
        color  = COLORS.get(pt,  COLORS["default"])
        marker = MARKERS.get(pt, MARKERS["default"])


        groups: dict[float, dict[float, list[dict]]] = {}
        for r in records:
            groups.setdefault(float(r["feature_size_mm"]), {}) \
                  .setdefault(float(r["spacing_mm"]), []).append(r)

        for fs_i, (fs, by_sp) in enumerate(sorted(groups.items())):
            ls   = LINESTYLES[fs_i % len(LINESTYLES)]
            sps  = sorted(by_sp.keys())
            freqs = [1.0 / sp for sp in sps]
            rm    = [_agg([r["rmse_mm"]   for r in by_sp[sp]]) for sp in sps]
            un    = [_agg([r["uniq_mean"] for r in by_sp[sp]]) for sp in sps]
            cov   = [_agg([r["coverage"]  for r in by_sp[sp]])[0] for sp in sps]
            n_rep = max(len(v) for v in by_sp.values())
            label = f"{pt} fs={fs:g}" + (f" (n={n_rep})" if n_rep > 1 else "")


            ax = axes[0]
            ax.errorbar(
                freqs, [m for m, _ in rm], yerr=[s for _, s in rm],
                marker=marker, color=color, ls=ls, label=label,
                lw=2, ms=7, capsize=3, zorder=3,
            )
            if fs_i == 0:
                for f, (m, _), sp, c in zip(freqs, rm, sps, cov):
                    if np.isnan(m):
                        continue
                    ax.annotate(
                        f"{sp:g} mm\n(cov {c*100:.0f}%)",
                        xy=(f, m), xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
                    )


            ax = axes[1]
            ax.errorbar(
                freqs, [m for m, _ in un], yerr=[s for _, s in un],
                marker=marker, color=color, ls=ls, label=label,
                lw=2, ms=7, capsize=3, zorder=3,
            )


    ax = axes[0]
    ax.set_xlabel("Spatial frequency [lp/mm]", fontsize=10)
    ax.set_ylabel("RMSE [mm]", fontsize=10)
    ax.set_title("Reconstruction Accuracy vs Pattern Frequency", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls="--")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)


    ax.axhline(
        0.5, color="gray", lw=1, ls=":", alpha=0.6,
        label="0.5 mm reference"
    )
    ax.text(
        ax.get_xlim()[1] * 0.98, 0.5,
        "0.5 mm ref.", va="bottom", ha="right",
        fontsize=7, color="gray",
    )


    ax = axes[1]
    ax.set_xlabel("Spatial frequency [lp/mm]", fontsize=10)
    ax.set_ylabel("Uniqueness score (NCC)", fontsize=10)
    ax.set_title("Pattern Uniqueness vs Pattern Frequency", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls="--")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=8)

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / (f"prelim_sensitivity_curves_{tag}.png" if tag
                            else "prelim_sensitivity_curves.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {fname}")
    if show:
        plt.show()
    plt.close(fig)


# Arrange camera images and reconstructed samples for visual comparison.
def plot_image_grid(
    results:  dict[str, list[dict]],
    save_dir: Path | None = None,
    show:     bool = False,
    tag:      str | None = None,
) -> None:

    for pt, all_records in results.items():


        records = [r for r in all_records
                   if r.get("phase_idx", 0) == 0 and "_pattern" in r]
        if not records:
            continue

        n_sp = len(records)
        fig, axes = plt.subplots(
            n_sp, 3,
            figsize=(12, 3.5 * n_sp),
            squeeze=False,
        )
        fig.patch.set_facecolor("#f8f8f8")
        fig.suptitle(
            f"Per-Spacing Overview — {pt.capitalize()} Pattern",
            fontsize=12, fontweight="bold",
        )

        for row, rec in enumerate(records):
            sp    = rec["spacing_mm"]
            pat   = rec["_pattern"]
            cimg  = rec["_cam_img"]
            pts   = rec["_pts_3d"]


            ax = axes[row, 0]
            ax.imshow(pat, cmap="gray", origin="upper", vmin=0, vmax=1)
            ax.set_title(f"Pattern  spacing={sp} mm", fontsize=9)
            ax.set_xlabel("u [px]", fontsize=7)
            ax.set_ylabel("v [px]", fontsize=7)
            ax.tick_params(labelsize=6)


            ax = axes[row, 1]
            ax.imshow(cimg, cmap="gray", origin="upper", vmin=0, vmax=255)
            ax.set_title(
                f"Camera image  "
                f"(density {rec['density']*100:.1f}%)",
                fontsize=9,
            )
            ax.set_xlabel("u [px]", fontsize=7)
            ax.set_ylabel("v [px]", fontsize=7)
            ax.tick_params(labelsize=6)


            ax = axes[row, 2]
            valid  = ~np.isnan(pts).any(axis=1)
            pts_v  = pts[valid]
            rmse_val = rec.get("rmse_mm", float("nan"))

            if len(pts_v) > 0:
                sc = ax.scatter(
                    pts_v[:, 0], pts_v[:, 1],
                    c=pts_v[:, 2], cmap="viridis", s=15, linewidths=0,
                )
                plt.colorbar(sc, ax=ax, label="Z [mm]", pad=0.02)
            ax.set_title(
                f"Point cloud  RMSE={rmse_val:.3f} mm\n"
                f"uniq={rec.get('uniq_mean', float('nan')):.3f}  "
                f"fov_cov={rec.get('fov_coverage', 0)*100:.1f}%",
                fontsize=9,
            )
            ax.set_xlabel("X [mm]", fontsize=7)
            ax.set_ylabel("Y [mm]", fontsize=7)
            ax.tick_params(labelsize=6)


            ax = axes[row, 0]
            ax.imshow(pat, cmap="gray", origin="upper", vmin=0, vmax=1)
            ax.set_title(f"Pattern  spacing={sp} mm", fontsize=9)
            ax.set_xlabel("u [px]", fontsize=7)
            ax.set_ylabel("v [px]", fontsize=7)
            ax.tick_params(labelsize=6)


            ax = axes[row, 1]
            ax.imshow(cimg, cmap="gray", origin="upper", vmin=0, vmax=255)
            ax.set_title(
                f"Camera image  "
                f"(density {rec['density']*100:.1f}%)",
                fontsize=9,
            )
            ax.set_xlabel("u [px]", fontsize=7)
            ax.set_ylabel("v [px]", fontsize=7)
            ax.tick_params(labelsize=6)

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = save_dir / (f"prelim_grid_{pt}_{tag}.png" if tag
                                else f"prelim_grid_{pt}.png")
            fig.savefig(fname, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Saved: {fname}")
        if show:
            plt.show()
        plt.close(fig)


CSV_FIELDS = [

    "pattern_type", "spacing_mm", "feature_size_mm", "noise_sigma",
    "phase_idx", "phase_offset_x_mm", "phase_offset_y_mm",
    "indent_depth_mm", "indent_width_mm", "renderer",
    "variant", "psf_sigma_px", "min_sep_frac", "measurement_volume_mm",
    "spacing_perp_mm", "stagger_frac", "jitter_frac", "lattice", "rotation_deg", "shear_frac", "baseline_mm",
    "indent_shape", "indent_order", "indent_center_x_mm", "indent_aspect",
    "curv_x_1_per_mm", "curv_y_extra_1_per_mm", "indent_asym", "indent_asym_angle_deg",
    "indent2_depth_mm", "indent2_width_mm", "indent2_center_x_mm", "indent2_center_y_mm",
    "albedo", "albedo_exposure", "specular_k", "mottle_amp", "outlier_frac", "splat",
    "cam_pitch_deg", "cam_yaw_deg", "cam_roll_deg",
    "cam_shift_along_mm", "cam_shift_across_mm", "cam_shift_axial_mm", "rays_per_pixel",
    "size_levels", "size_ratio", "code_shift", "same_class_pitch_mm", "bright_ratio", "code_ref",
    "epi_along_mm", "epi_across_mm", "epi_saturated", "n_placed",
    "anchor_ratio", "anchor_every", "anchor_tol_px", "propagate",
    "anchor_readable", "n_anchor_used", "drift_est_px", "drift_mad_px", "n_propagated", "n_unresolved", "anchor_frame_valid",
    "dash_length_mm", "dash_gap_mm", "dash_phases", "dash_levels", "dash_ratio",
    "size_class_ratio", "size_class_dprime", "n_size_uncertain", "size_code_readable", "size_class_balance",
    "epi_tol_px",
    "pattern_seed", "regularity_index", "nn_dist_mean_mm",
    "spatial_freq",

    "density", "feature_count", "uniqueness_ratio", "aliasing_risk",
    "depth_range_used",

    "data_valid", "n_pts_total", "neg_depth_ratio",

    "decoder", "n_detected", "n_matched", "n_rejected", "n_ambiguous",
    "ambiguity_ratio", "rejection_ratio", "corr_failure_ratio",
    "search_depth_mm", "search_mode", "epi_best_dist_mean_px", "epi_t_max",

    "n_ambiguous_second", "n_duplicate_claims", "n_area_rejected", "n_elong_rejected",

    "A_ambiguity", "D_density", "Q_reliable_density", "n_reliable",
    "rmse_reliable_mm", "max_err_reliable_mm", "n_reliable_evaluated",
    "outlier_rate_reliable", "p95_err_reliable_mm", "median_err_reliable_mm", "n_outliers_reliable",
    "outlier_rate_all",

    "ind_has_gt", "ind_depth_gt_mm", "ind_depth_rec_mm", "ind_depth_rec_pts_mm",
    "ind_depth_err_mm", "ind_center_err_mm", "ind_area_gt_mm2", "ind_area_rec_mm2",
    "ind_area_err_pct", "ind_iou", "ind_coverage_frac", "ind_n_pts_in_contour",
    "ind_slope_gt_deg", "ind_slope_rec_deg", "ind_slope_err_pct",
    "ind_grad_rms_err", "ind_grad_rms_rel", "ind_n_slope_cells",
    "ind_surround_offset_mm", "ind_depth_err_rel_mm",
    "ind_single_dent_valid", "ind_dip_rms_mm", "ind_dent1_depth_err_mm", "ind_dent2_depth_err_mm",

    "rmse_mm", "mae_mm", "max_err_mm", "n_evaluated",
    "rmse_all_mm", "mae_all_mm", "max_err_all_mm", "n_evaluated_all",
    "n_negative_depth", "neg_depth_err_mean", "neg_depth_err_max",
    "coverage", "fov_coverage",

    "cf_gt_z_std_mm", "cf_gt_z_range_mm", "cf_gt_depth_mm",
    "cf_err_z_std_mm", "cf_err_z_range_mm", "cf_err_depth_mm",
    "cf_center_covered", "cf_n_central_pts",

    "uniq_mean", "uniq_median", "uniq_min", "uniq_n_windows",
    "uniq_n_positions", "uniq_n_blank", "uniq_blank_ratio",

    "eta", "rho_3d", "dynamic_range",
    "eta_theoretical", "rho_3d_theoretical",
    "eta_dz_mm", "eta_dz_source", "eta_dz_realized_mm",
    "decoder_range_eff", "eta_realized", "eta_noise_mm",

    "n_points_3d", "n_valid_px",

    "recon_depth_max", "gt_depth_max", "depth_max_error",
    "recon_depth_mean", "recon_area_mm2", "gt_area_mm2",
    "area_error_pct", "n_points_used", "gt_area_source",
]


SUMMARY_KEYS    = ["pattern_type", "spacing_mm", "feature_size_mm", "noise_sigma"]
SUMMARY_METRICS = [
    "rmse_mm", "rmse_all_mm", "mae_mm", "max_err_mm",
    "neg_depth_ratio", "neg_depth_err_mean",
    "ambiguity_ratio", "rejection_ratio", "corr_failure_ratio",
    "Q_reliable_density", "rmse_reliable_mm", "max_err_reliable_mm",
    "outlier_rate_reliable", "p95_err_reliable_mm",
    "n_duplicate_claims", "n_area_rejected",
    "ind_depth_err_mm", "ind_center_err_mm", "ind_area_err_pct", "ind_iou",
    "ind_slope_err_pct", "ind_grad_rms_rel",
    "n_evaluated", "n_points_3d",
    "uniq_mean", "uniq_blank_ratio",
    "fov_coverage", "eta", "eta_realized",
    "cf_err_z_std_mm", "cf_err_z_range_mm", "depth_max_error",
]


# Save per-run sweep metrics to CSV (excludes array fields).
def save_csv(
    results:  dict[str, list[dict]],
    save_dir: Path,
    out_name: str = "prelim_results",
) -> Path:

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / f"{out_name}.csv"

    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for pt, records in results.items():
            for rec in records:
                row = {k: rec.get(k, "") for k in CSV_FIELDS}


                try:
                    row["phase_idx"] = int(float(row.get("phase_idx", 0) or 0))
                except (TypeError, ValueError):
                    pass
                writer.writerow(row)

    print(f"  Saved: {fname}")
    return fname


def aggregate_phases(results: dict[str, list[dict]]) -> list[dict]:

    groups: dict[tuple, list[dict]] = {}
    for pt, records in results.items():
        for rec in records:
            key = tuple(rec.get(k) for k in SUMMARY_KEYS)
            groups.setdefault(key, []).append(rec)

    rows = []
    for key, recs in groups.items():
        row = dict(zip(SUMMARY_KEYS, key))
        row["n_runs"]  = len(recs)
        row["n_valid"] = sum(1 for r in recs if r.get("data_valid", False))
        for m in SUMMARY_METRICS:
            vals = []
            for r in recs:
                try:
                    vals.append(float(r.get(m, float("nan"))))
                except (TypeError, ValueError):
                    vals.append(float("nan"))
            arr = np.array(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr):
                row[f"{m}_mean"] = float(arr.mean())
                row[f"{m}_std"]  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                row[f"{m}_min"]  = float(arr.min())
                row[f"{m}_max"]  = float(arr.max())
            else:
                for s in ("mean", "std", "min", "max"):
                    row[f"{m}_{s}"] = float("nan")
        rows.append(row)

    rows.sort(key=lambda r: (str(r["pattern_type"]), float(r["spacing_mm"]),
                             float(r["feature_size_mm"]), float(r["noise_sigma"])))
    return rows


def save_summary_csv(
    results:  dict[str, list[dict]],
    save_dir: Path,
    out_name: str = "prelim_results",
) -> Path:

    rows = aggregate_phases(results)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / f"{out_name}_summary.csv"
    if not rows:
        print("  (no rows to summarise)")
        return fname
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {fname}")
    return fname


def print_summary(results: dict[str, list[dict]]) -> None:

    has_noise = any(
        rec.get("noise_sigma", 0) > 0
        for recs in results.values() for rec in recs
    )
    noise_col = f"  {'noise':>5}" if has_noise else ""

    header = (
        f"  {'pattern':>6}  {'sp mm':>6}  {'r mm':>5}"
        f"{noise_col}  "
        f"{'valid':>5}  {'RMSE':>7}  {'fov_cov':>7}  "
        f"{'cf_zstd':>7}  {'covered':>7}  {'uniq':>6}  {'N pts':>6}"
    )
    print("\n" + "=" * len(header))
    print("  PRELIMINARY SENSITIVITY ANALYSIS — SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for pt, records in results.items():
        for rec in records:
            ns_str = f"  {rec.get('noise_sigma', 0):>5.1f}" if has_noise else ""
            print(
                f"  {rec.get('pattern_type','?'):>6}  "
                f"{rec.get('spacing_mm', float('nan')):>6.1f}  "
                f"{rec.get('feature_size_mm', float('nan')):>5.2f}"
                f"{ns_str}  "
                f"{'✅' if rec.get('data_valid', True) else '❌':>5}  "
                f"{rec.get('rmse_mm', float('nan')):>7.3f}  "
                f"{rec.get('fov_coverage', float('nan'))*100:>6.1f}%  "
                f"{rec.get('cf_err_z_std_mm', float('nan'))!s:>7}  "
                f"{str(rec.get('cf_center_covered','?')):>7}  "
                f"{rec.get('uniq_mean', float('nan')):>6.4f}  "
                f"{rec.get('n_points_3d', 0):>6}"
            )

    print("=" * len(header))


def print_phase_summary(results: dict[str, list[dict]]) -> None:

    rows = aggregate_phases(results)
    if not rows:
        return

    def ms(row, m, fmt="{:.3f}"):
        mu, sd = row.get(f"{m}_mean", float("nan")), row.get(f"{m}_std", float("nan"))
        if np.isnan(mu):
            return "   —   "
        return (fmt + "±" + fmt).format(mu, sd)

    header = (
        f"  {'pattern':>6}  {'sp mm':>6}  {'r mm':>5}  {'n':>2}  {'valid':>5}  "
        f"{'RMSE filt':>13}  {'RMSE all':>13}  {'neg%':>10}  "
        f"{'uniq':>13}  {'blank%':>10}  {'N pts':>9}"
    )
    print("\n" + "=" * len(header))
    print("  PHASE-AGGREGATED SUMMARY  (mean ± std over phase repeats)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"  {str(row['pattern_type']):>6}  "
            f"{float(row['spacing_mm']):>6.1f}  "
            f"{float(row['feature_size_mm']):>5.2f}  "
            f"{row['n_runs']:>2}  "
            f"{row['n_valid']:>3}/{row['n_runs']:<1}  "
            f"{ms(row, 'rmse_mm'):>13}  "
            f"{ms(row, 'rmse_all_mm'):>13}  "
            f"{ms(row, 'neg_depth_ratio', '{:.2f}'):>10}  "
            f"{ms(row, 'uniq_mean'):>13}  "
            f"{ms(row, 'uniq_blank_ratio', '{:.2f}'):>10}  "
            f"{ms(row, 'n_points_3d', '{:.0f}'):>9}"
        )
    print("=" * len(header))


# Parse standalone sweep options, run the selected cases and export summaries.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preliminary sensitivity analysis across spacing and feature_size values."
    )
    parser.add_argument(
        "--patterns", nargs="+",
        default=DEFAULT_PATTERN_TYPES,
        choices=["dots", "lines", "vlines", "random_dots", "graded_dots", "coded_dots", "dashed_vlines", "bright_dots",
                 "epi_random_dots", "anchored_dots"],
        help="Pattern types to sweep (default: dots). random_dots / graded_dots use "
             "seeds as the repeat axis (--n_phases N → N seeds).",
    )
    parser.add_argument(
        "--spacings", nargs="+", type=float,
        default=DEFAULT_SPACINGS_MM,
        help=(
            "spacing_mm values (default: 5 8 10 12 15 18 20). "
            "Valid range with pixel=30µm: spacing >= 5mm (smaller → aliasing). "
            "Recommended full sweep: 5 6 8 10 12 15 18 20"
        ),
    )
    parser.add_argument(
        "--feature_sizes", nargs="+", type=float,
        default=None,
        help=(
            "feature_size_mm values to sweep (dot radius or line width). "
            "If omitted, uses the default per pattern type "
            "(dots=0.15mm, lines=0.30mm). "
            "If provided, runs cartesian product: spacings × feature_sizes. "
            "Example: --feature_sizes 0.05 0.10 0.15 0.20"
        ),
    )
    parser.add_argument(
        "--noise_sigma", nargs="+", type=float,
        default=None,
        help=(
            "Gaussian noise sigma values to sweep. "
            "If omitted, uses noise_sigma=0 (ideal sensor). "
            "If provided, runs cartesian product: spacings × feature_sizes × noise_sigmas. "
            "Example: --noise_sigma 0 5 10 20"
        ),
    )
    parser.add_argument(
        "--variant", default=DEFAULT_VARIANT,
        choices=["A", "B", "C"],
        help="M1 surface variant: A=M10 Fig14, B=flat Fig9, C=M24 Fig13 (default: A)",
    )
    parser.add_argument(
        "--n_phases", type=int, default=1,
        help=(
            "Repeats per configuration with random pattern phase offsets "
            "(default 1 = nominal pattern only). Phase 0 is always the nominal "
            "un-shifted pattern; phases 1..N-1 shift the lattice by a random "
            "(dx, dy) in [0, spacing). With N>1 a <out_name>_summary.csv with "
            "mean/std over phases is written in addition to the per-run CSV."
        ),
    )
    parser.add_argument(
        "--phase_seed", type=int, default=42,
        help="Base seed for the phase offsets (default 42).",
    )
    parser.add_argument(
        "--out_name", default="prelim_results",
        help=(
            "Base name of the CSV outputs in outputs/data/: <out_name>.csv "
            "(per run) and <out_name>_summary.csv (phase-aggregated). "
            "Default: prelim_results"
        ),
    )
    parser.add_argument(
        "--indent_depth", type=float, default=0.0,
        help=(
            "Block B (v1.17): depth [mm] of a Gaussian indentation placed on the "
            "optical axis (0 = none, legacy surface). Model family: 2 / 5 / 10."
        ),
    )
    parser.add_argument(
        "--indent_width", type=float, default=None,
        help="Indentation full width at half depth [mm] (default: M1 DEFAULTS, 20 mm).",
    )
    parser.add_argument(
        "--renderer", default="psf", choices=["psf", "legacy"],
        help=(
            "M3 image formation (v1.18): 'psf' = summed ray deposition + camera PSF "
            "(default); 'legacy' = max deposition without PSF (fragments dots, ISSUE-016)."
        ),
    )
    parser.add_argument(
        "--decoder", default="epipolar", choices=["epipolar", "nn"],
        help=(
            "M4 correspondence decoder (v1.18): 'epipolar' = geometry-exact track "
            "search with ambiguity flags (default); 'nn' = legacy nearest neighbour."
        ),
    )
    parser.add_argument(
        "--psf_sigma", type=float, default=None,
        help=(
            "Block G blur (defocus) axis: camera PSF sigma [px] for the 'psf' "
            "renderer (default: M3 SENSOR_DEFAULTS, 1.0 px)."
        ),
    )
    parser.add_argument(
        "--min_sep_frac", type=float, default=0.6,
        help=(
            "Minimum centre separation of random / graded dots as a fraction of "
            "the (local) spacing (default 0.6; block G uses 0.8, the spacing constraint)."
        ),
    )
    parser.add_argument(
        "--measurement_volume", type=float, default=None,
        help=(
            "Depth range [mm] searched by the epipolar decoder (M4 search_mode "
            "'volume'; default 15 mm). Block G uses the application volume 20 mm "
            "(Variant C 13.2 mm + 10 mm indentation)."
        ),
    )
    parser.add_argument(
        "--spacing_perp", type=float, default=None,
        help="Block I-2: row pitch ACROSS the epipolar direction for dots [mm] "
             "(default: = --spacings → square grid). --spacings is the pitch ALONG it.",
    )
    parser.add_argument(
        "--stagger", type=float, default=0.0,
        help="Block I-2: shift of every second dot row along the epipolar direction as a "
             "fraction of the spacing (0.5 = hexagonal-like).",
    )
    parser.add_argument(
        "--jitter", type=float, default=0.0,
        help="Block I-2: uniform random displacement of every dot, ±jitter × spacing "
             "(regular → random axis; seeded per repeat).",
    )
    parser.add_argument(
        "--rotation", type=float, default=0.0,
        help="Block J-6: rotation of the dot lattice about the optical axis [°] (rows no longer parallel "
             "to the epipolar line; dots / coded_dots only).",
    )
    parser.add_argument(
        "--shear", type=float, default=0.0,
        help="Block J-9: progressive shift of dot row n by n·shear·spacing along the row (mod one pitch; "
             "dots only). With --spacings (row pitch), --spacing_perp (row spacing) and --rotation this "
             "describes any 2-D lattice (0.5 = stagger 0.5).",
    )
    parser.add_argument(
        "--baseline", type=float, default=None,
        help="Block I-2: projector–camera baseline [mm] for M3 and M4 (default 50 mm, "
             "θ ≈ 18.9°); sets the triangulation-angle axis.",
    )
    parser.add_argument(
        "--indent_shape", default="gaussian", choices=["gaussian", "plateau"],
        help="Block I-4: dent shape — 'plateau' = super-Gaussian flat top (flat-road contact patch).",
    )
    parser.add_argument("--indent_order", type=float, default=2.0,
                        help="Super-Gaussian order for --indent_shape plateau (default 12 when plateau).")
    parser.add_argument("--indent_center_x", type=float, default=0.0,
                        help="Dent / plateau centre along X [mm] (offset puts the plateau edge in the FOV).")
    parser.add_argument("--indent_aspect", type=float, default=1.0,
                        help="Block J-1: dent aspect σ_y/σ_x (Y = epipolar direction; 1 = circular). "
                             "--indent_width stays the FWHM along X.")

    parser.add_argument("--curv_x", type=float, default=0.0,
                        help="Task 1: additive longitudinal curvature [1/mm] (z_add = ½·κ·x²; tyre liner ≈ −0.0025).")
    parser.add_argument("--curv_y_extra", type=float, default=0.0,
                        help="Task 1: EXTRA lateral curvature [1/mm] added on top of the digitised profile.")
    parser.add_argument("--indent_asym", type=float, default=1.0,
                        help="Task 1: dent width ratio wide side / narrow side along --indent_asym_angle (1 = symmetric).")
    parser.add_argument("--indent_asym_angle", type=float, default=0.0,
                        help="Task 1: direction of the wide side [°, counter-clockwise from +x].")
    parser.add_argument("--indent2_depth", type=float, default=0.0,
                        help="Task 1: depth [mm] of a second Gaussian dent added to the first (0 = none).")
    parser.add_argument("--indent2_width", type=float, default=20.0, help="Task 1: FWHM [mm] of the second dent.")
    parser.add_argument("--indent2_center_x", type=float, default=0.0, help="Task 1: second dent centre x [mm].")
    parser.add_argument("--indent2_center_y", type=float, default=0.0, help="Task 1: second dent centre y [mm].")
    parser.add_argument("--cam_pitch", type=float, default=0.0,
                        help="Block J-2: rendering-camera rotation [°] about its column axis (∥ baseline) — the image "
                             "shifts ACROSS the epipolar line by ≈ 14.5 px/° (rejection beyond the tolerance). M4 stays nominal.")
    parser.add_argument("--cam_yaw", type=float, default=0.0,
                        help="Block J-2: rotation [°] about the row axis — shift ALONG the epipolar line (≈ 14.5 px/°, "
                             "silent depth bias ≈ 9 mm/°).")
    parser.add_argument("--cam_roll", type=float, default=0.0, help="Block J-2: rotation [°] about the optical axis.")
    parser.add_argument("--cam_shift_along", type=float, default=0.0,
                        help="Block J-2: camera centre displaced along the baseline [mm] (≈ 5.5 px/mm along the epipolar "
                             "line → depth bias ≈ 3 mm/mm).")
    parser.add_argument("--cam_shift_across", type=float, default=0.0,
                        help="Block J-2: displacement along the image rows [mm] (≈ 5.5 px/mm across the epipolar line).")
    parser.add_argument("--cam_shift_axial", type=float, default=0.0,
                        help="Block J-2: displacement along the optical axis [mm] (magnification change).")
    parser.add_argument("--rays_per_pixel", type=int, default=1,
                        help="Block J-7: n × n sub-rays per projector pixel (1 = the v1.18–v1.36 renderer; 3 removes the "
                             "resampling moiré that makes equally bright dots differ by ±15 %%).")
    parser.add_argument("--albedo", type=float, default=1.0, help="Block I-4: diffuse albedo (liner ≈ 0.03).")
    parser.add_argument("--albedo_exposure", type=float, default=1.0,
                        help="Albedo the exposure is set for (signal ∝ albedo/albedo_exposure).")
    parser.add_argument("--specular", type=float, default=0.0, help="Block I-4: Phong specular weight k_s.")
    parser.add_argument("--mottle", type=float, default=0.0, help="Block I-4: albedo mottling amplitude (0–1).")
    parser.add_argument("--outliers", type=float, default=0.0, help="Block I-4: hot/dead pixel fraction.")
    parser.add_argument("--size_levels", type=int, default=2,
                        help="Block I-6: number of code classes of coded_dots / bright_dots (2 or 3).")
    parser.add_argument("--bright_ratio", type=float, default=0.5,
                        help="Block J-7: intensity ratio between neighbouring classes of bright_dots (dimmest / brightest "
                             "for 2 classes; levels ratio^(k-1-c)).")
    parser.add_argument("--epi_along", type=float, default=6.5,
                        help="Block J-8: along-epipolar half-width of the exclusion window of epi_random_dots [mm] "
                             "(design: (L(ΔZ) + tol)·0.18 = 6.3 mm for ΔZ 20 mm, tol 3 px; default 6.5).")
    parser.add_argument("--epi_across", type=float, default=1.08,
                        help="Block J-8: cross-epipolar half-width of the exclusion window [mm] (tol·0.18 = 0.54 densest, "
                             "2·tol·0.18 = 1.08 keeps a drift margin of one tolerance).")
    parser.add_argument("--code_ref", default="global", choices=["global", "local"],
                        help="Block J-7: class reference of a 2-class code — 'global' k-means over the frame (v1.29) or "
                             "'local' contrast to the 8 nearest neighbours (removes shading / mottle trends).")
    parser.add_argument("--size_ratio", type=float, default=0.6,
                        help="Block I-6: smallest / largest dot radius of coded_dots (feature size = largest).")
    parser.add_argument("--code_shift", type=int, default=1,
                        help="Block I-6: class shift from one dot row to the next (1: checkerboard for 2 classes).")
    parser.add_argument("--dash_length", type=float, default=5.0, help="Block I-6: dash length of dashed_vlines [mm].")
    parser.add_argument("--dash_gap", type=float, default=1.0,
                        help="Block I-6: dark guard band between the dashes of neighbouring phases [mm].")
    parser.add_argument("--dash_phases", type=int, default=2,
                        help="Block I-6: dashed_vlines phases - only every k-th line is lit in a given row.")
    parser.add_argument("--dash_levels", type=int, default=1,
                        help="Block C: length code of dashed_vlines — line n carries class n mod k with dash length "
                             "L·ratio^c (k > 1 replaces the phase code; same-class pitch k × spacing).")
    parser.add_argument("--dash_ratio", type=float, default=0.5,
                        help="Block C: dash-length ratio between neighbouring classes (default 0.5).")
    parser.add_argument("--epi_tol", type=float, default=None,
                        help="Block I-5: epipolar match / ambiguity tolerance [px] (default M4: 3 px).")
    parser.add_argument("--anchor_ratio", type=float, default=1.5,
                        help="Block D-2: anchor radius / lattice-dot radius of anchored_dots (default 1.5).")
    parser.add_argument("--anchor_every", type=int, nargs=2, default=[2, 4], metavar=("COLS", "ROWS"),
                        help="Block D-2: every COLS-th column of every ROWS-th row is an anchor (default 2 4).")
    parser.add_argument("--anchor_tol", type=float, default=None,
                        help="Block D-2: match tolerance of the anchors [px] (default M4: 15 px).")
    parser.add_argument("--no_propagate", action="store_true",
                        help="Block D-2: disable the depth propagation of the anchored decoder (ablation).")
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume an interrupted sweep: load outputs/data/<out_name>.partial.csv "
            "and skip the runs it already contains. Every finished run is always "
            "appended to that file while the sweep runs (v1.15); it is removed "
            "once the final <out_name>.csv has been written."
        ),
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show figures interactively",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Project Part 1 — Preliminary Sensitivity Analysis")
    print("=" * 60)
    print(f"  Pattern types  : {args.patterns}")
    print(f"  Spacings [mm]  : {args.spacings}")
    n_fs = 1
    if args.feature_sizes:
        print(f"  Feature sizes  : {args.feature_sizes}")
        n_fs = len(args.feature_sizes)
    else:
        print(f"  Feature sizes  : default per pattern type")
    n_ns = 1
    if args.noise_sigma:
        print(f"  Noise sigmas   : {args.noise_sigma}")
        n_ns = len(args.noise_sigma)
    else:
        print(f"  Noise sigma    : 0.0 (ideal)")
    n_ph = max(1, args.n_phases)
    if n_ph > 1:
        print(f"  Phase repeats  : {n_ph} per configuration (seed {args.phase_seed})")
    print(f"  Variant        : {args.variant}")
    print(f"  Decoder        : {args.decoder}")
    print(f"  Renderer       : {args.renderer}"
          + (f"  (PSF sigma {args.psf_sigma} px)" if args.psf_sigma is not None else ""))
    if args.measurement_volume:
        print(f"  Search volume  : {args.measurement_volume} mm (M4 'volume' mode)")
    if args.spacing_perp or args.stagger or args.jitter or args.rotation or args.shear:
        print(f"  Dot lattice    : row pitch {args.spacing_perp or 'same'} mm, stagger {args.stagger}, "
              f"jitter {args.jitter} (block I-2), rotation {args.rotation}° (block J-6), shear {args.shear} (block J-9)")
    if args.baseline:
        print(f"  Baseline       : {args.baseline} mm (M3 + M4 + M5 θ)")
    if args.indent_shape != "gaussian" or args.indent_center_x:
        print(f"  Dent shape     : {args.indent_shape} (order {args.indent_order}), centre x {args.indent_center_x} mm")
    if any((args.cam_pitch, args.cam_yaw, args.cam_roll, args.cam_shift_along, args.cam_shift_across, args.cam_shift_axial)):
        print(f"  Camera drift   : pitch {args.cam_pitch}°, yaw {args.cam_yaw}°, roll {args.cam_roll}°; shift along "
              f"{args.cam_shift_along} / across {args.cam_shift_across} / axial {args.cam_shift_axial} mm "
              f"(rendering only; decoder nominal — block J-2)")
    if args.albedo != 1.0 or args.specular or args.mottle or args.outliers:
        print(f"  Reflectance    : albedo {args.albedo}/{args.albedo_exposure}, specular {args.specular}, "
              f"mottle {args.mottle}, outliers {args.outliers}")
    if any(p in RANDOM_TYPES for p in args.patterns):
        print(f"  Min separation : {args.min_sep_frac} x spacing (random / graded dots)")
    if "epi_random_dots" in args.patterns:
        print(f"  Epipolar window: along ±{args.epi_along} mm, across ±{args.epi_across} mm (block J-8)")
    if "bright_dots" in args.patterns:
        print(f"  Brightness code: {args.size_levels} classes, intensity ratio {args.bright_ratio}, "
              f"shift {args.code_shift} (block J-7)")
    if "coded_dots" in args.patterns:
        print(f"  Size code      : {args.size_levels} classes, radius ratio {args.size_ratio}, "
              f"shift {args.code_shift} (block I-6)")
    if "dashed_vlines" in args.patterns:
        print(f"  Dashes         : length {args.dash_length} mm, guard gap {args.dash_gap} mm, "
              f"{args.dash_phases} phases (block I-6)")
    if args.epi_tol is not None:
        print(f"  Epipolar tol   : {args.epi_tol} px (block I-5)")
    if args.indent_depth and args.indent_depth > 0:
        print(f"  Indentation    : depth {args.indent_depth} mm, "
              f"FWHM {args.indent_width or 'default'} mm, aspect {args.indent_aspect}, on optical axis")
    m1_extra = {}
    if args.curv_x:
        m1_extra["curv_x_1_per_mm"] = args.curv_x
    if args.curv_y_extra:
        m1_extra["curv_y_extra_1_per_mm"] = args.curv_y_extra
    if args.indent_asym != 1.0 or args.indent_asym_angle:
        m1_extra.update({"indent_asym": args.indent_asym, "indent_asym_angle_deg": args.indent_asym_angle})
    if args.indent2_depth:
        m1_extra.update({"indent2_depth_mm": args.indent2_depth, "indent2_width_mm": args.indent2_width,
                         "indent2_center_x_mm": args.indent2_center_x, "indent2_center_y_mm": args.indent2_center_y})
    if m1_extra:
        print(f"  Surface (task 1): {m1_extra}")
    print(f"  Output name    : {args.out_name}")
    n_runs = len(args.patterns) * len(args.spacings) * n_fs * n_ns * n_ph
    print(f"  Total runs     : {n_runs}  (~{n_runs * 50 // 60} min estimated)")
    partial_path = OUT_DAT / f"{args.out_name}.partial.csv"
    print(f"  Checkpoint     : {partial_path.name}"
          + ("  (resuming)" if args.resume else ""))
    print()


    results = run_preliminary(
        pattern_types    = args.patterns,
        spacings_mm      = args.spacings,
        feature_sizes_mm = args.feature_sizes,
        noise_sigmas     = args.noise_sigma,
        variant          = args.variant,
        n_phases         = n_ph,
        phase_seed       = args.phase_seed,
        checkpoint_path  = partial_path,
        resume           = args.resume,
        indent_depth_mm  = args.indent_depth,
        indent_width_mm  = args.indent_width,
        decoder          = args.decoder,
        renderer         = args.renderer,
        psf_sigma_px     = args.psf_sigma,
        min_sep_frac     = args.min_sep_frac,
        measurement_volume_mm = args.measurement_volume,
        spacing_perp_mm  = args.spacing_perp,
        stagger_frac     = args.stagger,
        jitter_frac      = args.jitter,
        rotation_deg     = args.rotation,
        shear_frac       = args.shear,
        baseline_mm      = args.baseline,
        indent_shape     = args.indent_shape,
        indent_order     = args.indent_order,
        indent_center_x_mm = args.indent_center_x,
        indent_aspect    = args.indent_aspect,
        albedo           = args.albedo,
        albedo_exposure  = args.albedo_exposure,
        specular_k       = args.specular,
        mottle_amp       = args.mottle,
        outlier_frac     = args.outliers,
        cam_rot_deg      = (args.cam_pitch, args.cam_yaw, args.cam_roll),
        cam_trans_mm     = (args.cam_shift_along, args.cam_shift_across, args.cam_shift_axial),
        rays_per_pixel   = args.rays_per_pixel,
        size_levels      = args.size_levels,
        size_ratio       = args.size_ratio,
        code_shift       = args.code_shift,
        bright_ratio     = args.bright_ratio,
        code_ref         = args.code_ref,
        epi_along_mm     = args.epi_along,
        epi_across_mm    = args.epi_across,
        anchor_ratio     = args.anchor_ratio,
        anchor_every     = tuple(args.anchor_every),
        anchor_tol_px    = args.anchor_tol,
        propagate        = not args.no_propagate,
        dash_levels      = args.dash_levels,
        dash_ratio       = args.dash_ratio,
        dash_length_mm   = args.dash_length,
        dash_gap_mm      = args.dash_gap,
        dash_phases      = args.dash_phases,
        epi_tol_px       = args.epi_tol,
        m1_extra         = m1_extra or None,
        verbose          = True,
    )


    print_summary(results)
    if n_ph > 1:
        print_phase_summary(results)


    print("\n[Save] CSV results ...")
    final_csv = save_csv(results, save_dir=OUT_DAT, out_name=args.out_name)
    if n_ph > 1:
        save_summary_csv(results, save_dir=OUT_DAT, out_name=args.out_name)

    if final_csv.exists() and partial_path.exists():
        partial_path.unlink()
        print(f"  Removed checkpoint {partial_path.name} (consolidated into {final_csv.name})")


    fig_tag = None if args.out_name == "prelim_results" else args.out_name

    variant_names = {"A": "Variant A — M10 nut (Fig.14)",
                     "B": "Variant B — flat road reference (Fig.9)",
                     "C": "Variant C — M24 nut (Fig.13)"}
    subtitle = (f"{variant_names.get(args.variant, 'Variant ' + args.variant)} — "
                f"Tire Inner Surface — Pattern Sweep"
                + (f" ({n_ph} phase repeats)" if n_ph > 1 else ""))

    print("\n[Figure] Sensitivity curves ...")
    plot_sensitivity_curves(results, save_dir=OUT_FIG, show=args.show,
                            tag=fig_tag, subtitle=subtitle)

    print("\n[Figure] Per-spacing image grids ...")
    plot_image_grid(results, save_dir=OUT_FIG, show=args.show, tag=fig_tag)

    print("\n" + "=" * 60)
    print("Preliminary analysis complete.")
    print(f"  Figures : {OUT_FIG}")
    print(f"  Data    : {OUT_DAT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
