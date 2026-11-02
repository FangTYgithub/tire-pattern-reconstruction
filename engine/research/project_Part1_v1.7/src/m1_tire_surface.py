# Build height-field meshes from digitized tire cross-sections. These standalone defaults are
# overridden by the surface configuration used by study adapters.

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


DEFAULTS: dict = {

    "bin_width_mm":         1.0,  # Envelope bin width [mm].


    "grid_step_y_mm":       1.0,  # Standalone lateral grid step [mm]; study grids are supplied by adapters.


    "interp_kind":          "pchip",  # Profile interpolation: pchip or linear.


    "smoothing_window":     21,  # Preferred Savitzky-Golay window [samples]; adjusted to an odd valid length.
    "smoothing_polyorder":  3,  # Savitzky-Golay polynomial order; below the effective window length.


    "surface_length_x_mm":  200.0,  # Longitudinal extent of the standalone mesh [mm].
    "grid_step_x_mm":         2.0,  # Standalone longitudinal grid step [mm].


    "indent_depth_mm":          0.0,  # Optional deformation amplitude [mm]; zero disables this component.
    "indent_width_mm":         20.0,  # Deformation full width at half maximum [mm].
    "indent_center_x_mm":       0.0,  # Longitudinal deformation centre [mm].
    "indent_center_y_mm":       0.0,  # Lateral deformation centre [mm].
    "indent_aspect":            1.0,  # Ratio of deformation widths in the two planar directions.
    "indent_shape":        "gaussian",  # Analytic deformation profile; gaussian is the default.


    "indent_order":             2.0,
    "indent_area_threshold_frac": 0.10,


    "curv_x_1_per_mm":          0.0,  # Additional longitudinal curvature [1/mm]; study surfaces use their own mapping.
    "curv_y_extra_1_per_mm":    0.0,  # Additional lateral curvature [1/mm].
    "curv_ref_x_mm":            0.0,
    "curv_ref_y_mm":            0.0,


    "indent_asym":              1.0,
    "indent_asym_angle_deg":    0.0,


    "indent2_depth_mm":         0.0,  # Second deformation amplitude [mm]; zero disables it.
    "indent2_width_mm":        20.0,  # Second deformation full width at half maximum [mm].
    "indent2_center_x_mm":      0.0,
    "indent2_center_y_mm":      0.0,
    "indent2_aspect":           1.0,


    "measurement_width_y_mm":   200.0,
    "measurement_length_x_mm":  280.0,
    "measurement_arc_deg":        40.0,
    "tyre_radius_estimate_mm":   280.0 / np.deg2rad(40.0),
    "contact_centre_x_mm":       150.0,
    "m10_obstacle_height_mm":     10.0,
    "m10_max_deformation_mm":     16.0,
}


_CSV_NAMES: dict[str, str] = {
    "A": "1405Fig14M10_Default_Dataset.csv",
    "B": "Fig9_Flat_Default_Dataset.csv",
    "C": "Fig13_M24_Default_Dataset.csv",


}

_VARIANT_LABELS: dict[str, str] = {
    "A": "M10_deformed_Fig14",
    "B": "flat_ground_Fig9",
    "C": "M24_deformed_Fig13",
}


_VARIANT_REFERENCE: dict[str, dict] = {
    "A": {
        "obstacle":                "M10 Nyloc nut",
        "obstacle_height_mm":      10.0,
        "max_deformation_paper_mm":16.0,
        "note": "Digitised cross-section passes near nut centre.",
    },
    "B": {
        "obstacle":                "none (flat ground)",
        "obstacle_height_mm":       0.0,
        "max_deformation_paper_mm": 0.0,
        "note": (
            "Flat-ground contact. Absolute Z coordinates are NOT aligned "
            "with Variant A/C because Guthrie et al. report that "
            "cross-run coordinate alignment is unreliable (stabiliser arm "
            "flex). Use z_relative for all comparisons."
        ),
    },
    "C": {
        "obstacle":                "M24 Nyloc nut",
        "obstacle_height_mm":      23.1,
        "max_deformation_paper_mm":30.0,
        "note": (
            "Digitised cross-section may not pass through nut centre "
            "(nut appears laterally offset in Fig.13). Local depth in "
            "digitisation is conservative (~21 mm vs paper 30 mm)."
        ),
    },
}


# Read the digitized lateral position and height columns in millimetres.
def load_raw_csv(csv_path: Path) -> pd.DataFrame:

    df = pd.read_csv(csv_path, header=None, names=["y_mm", "z_mm"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError(f"No valid numeric rows found in {csv_path}")
    return df.reset_index(drop=True)


# Reduce multiple heights in each lateral bin to its maximum height.
def extract_upper_envelope(
    raw: pd.DataFrame,
    bin_width_mm: float,
) -> pd.DataFrame:

    raw = raw.copy()
    raw["y_bin_mm"] = (raw["y_mm"] / bin_width_mm).round() * bin_width_mm

    envelope = (
        raw.groupby("y_bin_mm", as_index=False)
        .agg(
            z_upper_raw_mm=("z_mm", "max"),
            point_count=("z_mm", "size"),
        )
        .rename(columns={"y_bin_mm": "y_mm"})
        .sort_values("y_mm", kind="mergesort")
        .reset_index(drop=True)
    )
    return envelope


# Interpolate the envelope onto an equally spaced lateral grid.
def interpolate_to_uniform_grid(
    envelope: pd.DataFrame,
    grid_step_mm: float,
    kind: str,
) -> pd.DataFrame:

    y_src  = envelope["y_mm"].to_numpy()
    z_src  = envelope["z_upper_raw_mm"].to_numpy()
    y_grid = np.arange(y_src.min(), y_src.max() + grid_step_mm * 0.5, grid_step_mm)

    if kind == "pchip":
        z_interp = PchipInterpolator(y_src, z_src)(y_grid)
    elif kind == "linear":
        z_interp = np.interp(y_grid, y_src, z_src)
    else:
        raise ValueError(f"interp_kind must be 'pchip' or 'linear', got {kind!r}")

    return pd.DataFrame({"y_mm": y_grid, "z_interp_mm": z_interp})


def _valid_sg_window(n_points: int, preferred: int) -> int:

    if n_points < 5:
        raise ValueError(f"Too few points for smoothing ({n_points}); need ≥ 5.")
    window = min(preferred, n_points)
    if window % 2 == 0:
        window -= 1
    return max(window, 5)


# Apply Savitzky-Golay smoothing with a window fitted to the available samples.
def smooth_profile(
    interpolated: pd.DataFrame,
    window: int,
    polyorder: int,
) -> pd.DataFrame:

    z_in  = interpolated["z_interp_mm"].to_numpy()
    n     = len(z_in)
    w     = _valid_sg_window(n, window)
    p     = min(polyorder, w - 1)

    z_smooth   = savgol_filter(z_in, window_length=w, polyorder=p)
    z_relative = z_smooth - z_smooth.min()

    return pd.DataFrame({
        "y_mm":          interpolated["y_mm"].to_numpy(),
        "z_interp_mm":   z_in,
        "z_smooth_mm":   z_smooth,
        "z_relative_mm": z_relative,
    })


# Repeat the lateral profile along the longitudinal axis to create the surface mesh.
def extrude_profile(
    profile: pd.DataFrame,
    surface_length_x_mm: float,
    grid_step_x_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    half  = surface_length_x_mm / 2.0
    x_1d  = np.arange(-half, half + grid_step_x_mm * 0.5, grid_step_x_mm)
    y_1d  = profile["y_mm"].to_numpy()

    X, Y = np.meshgrid(x_1d, y_1d)
    Z    = np.tile(profile["z_smooth_mm"].to_numpy()[:, None],   (1, len(x_1d)))
    Zr   = np.tile(profile["z_relative_mm"].to_numpy()[:, None], (1, len(x_1d)))

    return (
        X.astype(np.float32), Y.astype(np.float32),
        Z.astype(np.float32), Zr.astype(np.float32),
    )


# Extract profile heights and characteristic contact-region positions.
def extract_contact_features(
    profile:           pd.DataFrame,
    sidewall_y_min_mm: float = 90.0,
) -> dict:

    y  = profile["y_mm"].to_numpy()
    zr = profile["z_relative_mm"].to_numpy()


    side_mask = np.abs(y) > sidewall_y_min_mm
    ref_z = float(zr[side_mask].mean()) if side_mask.sum() > 3 else float(zr.max())


    z_min_idx = int(np.argmin(zr))
    z_min     = float(zr[z_min_idx])
    y_center  = float(y[z_min_idx])


    central_mask = np.abs(y) < 80.0
    zr_central   = zr[central_mask]

    if len(zr_central) == 0:
        c_std = c_max = c_min = float("nan")
    else:
        c_std = float(zr_central.std())
        c_max = float(zr_central.max())
        c_min = float(zr_central.min())

    return {

        "ref_z_sidewall_mm":       round(ref_z, 3),

        "z_min_mm":                round(z_min, 3),
        "indentation_depth_mm":    round(ref_z - z_min, 3),
        "indentation_center_y_mm": round(y_center, 1),

        "central_z_std_mm":        round(c_std, 3),
        "central_z_max_mm":        round(c_max, 3),
        "central_z_min_mm":        round(c_min, 3),
        "central_z_range_mm":      round(c_max - c_min, 3),
    }


# Construct the optional deformation field and its reference descriptors. Width is the full width at
# half maximum in millimetres.
def make_indentation(
    X: np.ndarray,
    Y: np.ndarray,
    depth_mm: float,
    width_fwhm_mm: float,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    aspect: float = 1.0,
    shape: str = "gaussian",
    area_threshold_frac: float = 0.10,
    grid_step_x_mm: float | None = None,
    grid_step_y_mm: float | None = None,
    fov_half_width_mm: float | None = None,
    order: float = 2.0,
    asym: float = 1.0,
    asym_angle_deg: float = 0.0,
) -> tuple[np.ndarray, dict]:

    if shape not in ("gaussian", "plateau"):
        raise ValueError(f"indent_shape '{shape}' not implemented ('gaussian' | 'plateau')")
    if not np.isfinite([depth_mm, width_fwhm_mm, *center_xy_mm]).all() or depth_mm <= 0 or width_fwhm_mm <= 0:
        raise ValueError("depth_mm and width_fwhm_mm must be > 0")
    n = 2.0 if shape == "gaussian" else (float(order) if order and float(order) > 2.0 else 12.0)
    asym = float(asym); phi_deg = float(asym_angle_deg)
    if not (np.isfinite(asym) and asym > 0.0) or not np.isfinite(float(aspect)) or float(aspect) <= 0.0:
        raise ValueError(f"asym and aspect must be finite and > 0 (got asym {asym}, aspect {aspect})")
    if not np.isfinite(phi_deg):
        raise ValueError("asym_angle_deg must be finite")

    cx, cy   = float(center_xy_mm[0]), float(center_xy_mm[1])
    k_half   = (2.0 * np.log(2.0)) ** (1.0 / n)
    sigma_x  = float(width_fwhm_mm) / (2.0 * k_half)
    sigma_y  = sigma_x * float(aspect)
    dxm = np.asarray(X, dtype=np.float64) - cx
    dym = np.asarray(Y, dtype=np.float64) - cy
    if asym == 1.0 and phi_deg == 0.0:
        r = np.sqrt((dxm / sigma_x) ** 2 + (dym / sigma_y) ** 2)
    else:
        c, s = np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))
        u = dxm * c + dym * s
        v = -dxm * s + dym * c
        sig_u = np.where(u >= 0.0, sigma_x * asym, sigma_x)
        r = np.sqrt((u / sig_u) ** 2 + (v / sigma_y) ** 2)
    dip = float(depth_mm) * np.exp(-0.5 * r ** n)

    frac  = float(area_threshold_frac)
    k     = (-2.0 * np.log(frac)) ** (1.0 / n)
    rx, ry = k * sigma_x, k * sigma_y
    rx_wide = k * sigma_x * asym
    mask  = dip >= frac * depth_mm


    if grid_step_x_mm is None:
        grid_step_x_mm = float(np.abs(np.diff(np.asarray(X)[0, :])).mean()) if X.shape[1] > 1 else 1.0
    if grid_step_y_mm is None:
        grid_step_y_mm = float(np.abs(np.diff(np.asarray(Y)[:, 0])).mean()) if Y.shape[0] > 1 else 1.0
    cell_area = float(grid_step_x_mm) * float(grid_step_y_mm)


    sig_min = min(sigma_x, sigma_x * asym, sigma_y)
    gy, gx = np.gradient(dip, float(grid_step_y_mm), float(grid_step_x_mm))
    max_slope_grid = float(np.degrees(np.arctan(np.hypot(gx, gy).max())))
    if n == 2.0:
        max_slope_an = float(np.degrees(np.arctan(depth_mm / (sig_min * np.sqrt(np.e)))))
        volume = float(2.0 * np.pi * sigma_x * sigma_y * depth_mm) * (1.0 + asym) / 2.0
    else:
        max_slope_an = max_slope_grid
        volume = float(dip.sum()) * cell_area

    in_fov = None
    if fov_half_width_mm is not None:


        ext_y = ry if (asym == 1.0 and phi_deg == 0.0) else max(rx_wide, rx, ry)
        in_fov = bool(abs(cy) + ext_y <= fov_half_width_mm)

    gt = {
        "shape":               shape,
        "depth_mm":            float(depth_mm),
        "depth_grid_mm":       float(dip.max()),
        "center_xy_mm":        (cx, cy),
        "width_fwhm_mm":       float(width_fwhm_mm),
        "sigma_x_mm":          round(sigma_x, 4),
        "sigma_y_mm":          round(sigma_y, 4),
        "aspect":              float(aspect),
        "area_threshold_frac": frac,
        "r_threshold_x_mm":    round(rx, 4),
        "r_threshold_y_mm":    round(ry, 4),
        "r_threshold_x_wide_mm": round(rx_wide, 4),
        "asym":                asym,
        "asym_angle_deg":      phi_deg,
        "area_mm2":            round(float(np.pi * rx * ry) * (1.0 + asym) / 2.0, 3),
        "area_grid_mm2":       round(float(mask.sum()) * cell_area, 3),
        "n_contour_cells":     int(mask.sum()),
        "contour_mask":        mask,
        "max_slope_deg":       round(max_slope_an, 3),
        "max_slope_grid_deg":  round(max_slope_grid, 3),
        "volume_mm3":          round(volume, 3),
        "order":               float(n),
        "in_fov":              in_fov,
    }
    return dip.astype(np.float32), gt


# Load, smooth and extrude the selected profile, then apply supplied geometry overrides. Returns the
# mesh, relative-depth map, features and effective configuration.
def build_surface(
    variant: str = "A",
    csv_path: Path | None = None,
    raw_data_dir: Path | None = None,
    deformation_scale: float = 1.0,
    **param_overrides,
) -> tuple[dict, np.ndarray, dict]:

    params = {**DEFAULTS, **param_overrides}


    if csv_path is None:
        if variant not in _CSV_NAMES:
            registered = list(_CSV_NAMES.keys())
            raise ValueError(
                f"Unknown variant {variant!r}. "
                f"Registered variants: {registered}. "
                f"To add a new variant see the module docstring."
            )
        base_dir  = raw_data_dir or (Path(__file__).resolve().parent / "data" / "raw")
        csv_path  = base_dir / _CSV_NAMES[variant]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {csv_path}\n"
            f"Expected two-column WebPlotDigitizer export (no header: y_mm, z_mm)."
        )

    variant_label = _VARIANT_LABELS.get(variant, csv_path.stem)


    raw        = load_raw_csv(csv_path)
    envelope   = extract_upper_envelope(raw, params["bin_width_mm"])
    interp     = interpolate_to_uniform_grid(
                     envelope, params["grid_step_y_mm"], params["interp_kind"])
    profile    = smooth_profile(
                     interp, params["smoothing_window"], params["smoothing_polyorder"])
    X, Y, Z, Zr = extrude_profile(
                     profile,
                     params["surface_length_x_mm"],
                     params["grid_step_x_mm"])


    if deformation_scale != 1.0:
        Z_min = Z.min()
        Zr_scaled = Zr * deformation_scale
        Z = Z_min + Zr_scaled
        Zr = Zr_scaled


    contact_features = extract_contact_features(profile)

    fov_half = sensor_fov_half_width_mm()


    curvature = None
    kx = float(params.get("curv_x_1_per_mm", 0.0) or 0.0)
    ky = float(params.get("curv_y_extra_1_per_mm", 0.0) or 0.0)
    if kx != 0.0 or ky != 0.0:
        if not (np.isfinite(kx) and np.isfinite(ky)) or max(abs(kx), abs(ky)) > CURV_MAX_1_PER_MM:
            raise ValueError(f"curvature must be finite and |κ| ≤ {CURV_MAX_1_PER_MM} /mm (bending radius ≥ "
                             f"{1.0 / CURV_MAX_1_PER_MM:.0f} mm): got curv_x {kx}, curv_y_extra {ky}")
        x0 = float(params.get("curv_ref_x_mm", 0.0) or 0.0); y0 = float(params.get("curv_ref_y_mm", 0.0) or 0.0)
        z_add = 0.5 * kx * (X.astype(np.float64) - x0) ** 2 + 0.5 * ky * (Y.astype(np.float64) - y0) ** 2
        Z  = (Z.astype(np.float64) + z_add).astype(np.float32)
        Zr = (Z - Z.min()).astype(np.float32)
        curvature = {"curv_x_1_per_mm": kx, "curv_y_extra_1_per_mm": ky, "ref_xy_mm": (x0, y0),
                     "form": "z_add = ½·curv_x·(x − x_ref)² + ½·curv_y_extra·(y − y_ref)², added to the extruded profile "
                             "(the profile's own lateral curvature is untouched)",
                     "z_add_min_mm": float(z_add.min()), "z_add_max_mm": float(z_add.max()),
                     "z_add_at_fov_edge_x_mm": float(0.5 * kx * fov_half ** 2)}


    Z_base      = Z.copy()
    indentation = None
    indent_mask = None
    indentation2 = None
    double_ind  = None
    d_indent    = float(params.get("indent_depth_mm", 0.0) or 0.0)
    d_indent2   = float(params.get("indent2_depth_mm", 0.0) or 0.0)
    if d_indent2 < 0.0 or not np.isfinite(d_indent2):
        raise ValueError(f"indent2_depth_mm must be ≥ 0 (got {d_indent2})")
    if d_indent2 > 0.0 and d_indent <= 0.0:
        raise ValueError("a second dent (indent2_depth_mm > 0) needs a first dent (indent_depth_mm > 0)")
    if d_indent > 0.0:
        dip, indentation = make_indentation(
            X, Y, d_indent, params["indent_width_mm"],
            center_xy_mm        = (params["indent_center_x_mm"], params["indent_center_y_mm"]),
            aspect              = params.get("indent_aspect", 1.0),
            shape               = params.get("indent_shape", "gaussian"),
            order               = params.get("indent_order", 2.0),
            area_threshold_frac = params.get("indent_area_threshold_frac", 0.10),
            grid_step_x_mm      = params["grid_step_x_mm"],
            grid_step_y_mm      = params["grid_step_y_mm"],
            fov_half_width_mm   = fov_half,
            asym                = params.get("indent_asym", 1.0),
            asym_angle_deg      = params.get("indent_asym_angle_deg", 0.0),
        )
        indent_mask = indentation.pop("contour_mask")
        if d_indent2 > 0.0:
            dip2, indentation2 = make_indentation(
                X, Y, d_indent2, params.get("indent2_width_mm", 20.0),
                center_xy_mm        = (params.get("indent2_center_x_mm", 0.0), params.get("indent2_center_y_mm", 0.0)),
                aspect              = params.get("indent2_aspect", 1.0),
                shape               = "gaussian",
                area_threshold_frac = params.get("indent_area_threshold_frac", 0.10),
                grid_step_x_mm      = params["grid_step_x_mm"],
                grid_step_y_mm      = params["grid_step_y_mm"],
                fov_half_width_mm   = fov_half,
            )
            mask2 = indentation2.pop("contour_mask")
            double_ind = double_indentation_truth(dip, dip2, indentation, indentation2, X, Y, indent_mask, mask2,
                                                  params.get("indent_area_threshold_frac", 0.10))
            indentation["single_dent_metrics_valid"] = False
            indent_mask = indent_mask | mask2
            dip = dip + dip2
        Z  = (Z - dip).astype(np.float32)
        Zr = (Z - Z.min()).astype(np.float32)

    mesh      = {"X": X, "Y": Y, "Z": Z, "Z_base": Z_base}
    if indent_mask is not None:
        mesh["indent_mask"] = indent_mask
    depth_map = Zr


    fov_dz    = fov_z_range_mm(Y, Zr, fov_half)

    config = {
        **params,
        "variant":             variant,
        "variant_label":       variant_label,
        "variant_reference":   _VARIANT_REFERENCE.get(variant, {}),
        "csv_path":            str(csv_path),
        "deformation_scale":   deformation_scale,
        "raw_point_count":     len(raw),
        "envelope_bin_count":  len(envelope),
        "profile_point_count": len(profile),
        "grid_shape_ny_nx":    Z.shape,
        "y_range_mm":          (float(Y[:, 0].min()), float(Y[:, 0].max())),
        "x_range_mm":          (float(X[0, :].min()), float(X[0, :].max())),
        "z_abs_min_mm":        float(Z.min()),
        "z_abs_max_mm":        float(Z.max()),
        "depth_max_mm":        float(Zr.max()),
        "fov_half_width_mm":   fov_half,
        "fov_z_range_mm":      fov_dz,
        "contact_features":    contact_features,
        "has_indentation":     indentation is not None,
        "indentation":         indentation,

        "has_curvature":       curvature is not None,
        "curvature":           curvature,
        "indentation2":        indentation2,
        "double_indentation":  double_ind,
    }

    return mesh, depth_map, config


CURV_MAX_1_PER_MM = 0.02


# Evaluate the analytic deformation at one planar position in millimetres.
def dent_value_at(px: float, py: float, gt: dict) -> float:

    cx, cy = gt["center_xy_mm"]
    n, asym = float(gt["order"]), float(gt.get("asym", 1.0))
    sx = float(gt["width_fwhm_mm"]) / (2.0 * (2.0 * np.log(2.0)) ** (1.0 / n))
    sy = sx * float(gt["aspect"])
    phi = np.radians(float(gt.get("asym_angle_deg", 0.0)))
    dxm, dym = float(px) - float(cx), float(py) - float(cy)
    u = dxm * np.cos(phi) + dym * np.sin(phi); v = -dxm * np.sin(phi) + dym * np.cos(phi)
    su = sx * asym if u >= 0.0 else sx
    r = np.sqrt((u / su) ** 2 + (v / sy) ** 2)
    return float(gt["depth_mm"] * np.exp(-0.5 * r ** n))


# Combine two deformation fields and calculate their shared reference descriptors.
def double_indentation_truth(dip1: np.ndarray, dip2: np.ndarray, gt1: dict, gt2: dict, X: np.ndarray, Y: np.ndarray,
                             mask1: np.ndarray, mask2: np.ndarray, frac: float) -> dict:

    from scipy.ndimage import maximum_filter
    total = np.asarray(dip1, dtype=np.float64) + np.asarray(dip2, dtype=np.float64)
    c1 = tuple(float(v) for v in gt1["center_xy_mm"]); c2 = tuple(float(v) for v in gt2["center_xy_mm"])
    i_max = int(np.argmax(total)); iy, ix = np.unravel_index(i_max, total.shape)
    thr = float(frac) * max(float(gt1["depth_mm"]), float(gt2["depth_mm"]))
    local_max = (total == maximum_filter(total, size=3, mode="nearest")) & (total > thr)

    from scipy.ndimage import label
    n_max = int(label(local_max)[1])
    union = mask1 | mask2
    return {
        "rule":                     "additive: dip_total = dip1 + dip2",
        "separation_mm":            float(np.hypot(c2[0] - c1[0], c2[1] - c1[1])),
        "depth_at_center1_mm":      float(gt1["depth_mm"]) + dent_value_at(c1[0], c1[1], gt2),
        "depth_at_center2_mm":      float(gt2["depth_mm"]) + dent_value_at(c2[0], c2[1], gt1),
        "peak_total_grid_mm":       float(total.max()),
        "peak_total_xy_mm":         (float(X[iy, ix]), float(Y[iy, ix])),
        "n_local_maxima_grid":      n_max,
        "overlap_frac_contour":     float((mask1 & mask2).sum() / union.sum()) if union.sum() else 0.0,
        "single_dent_metrics_valid": False,
    }


_FOV_HALF_WIDTH_FALLBACK_MM = 46.1


def sensor_fov_half_width_mm() -> float:

    try:
        import config as _cfg
        sensor_half = _cfg.M3_IMAGE_WIDTH_PX * _cfg.M3_PIXEL_PITCH_MM / 2.0
        return float(sensor_half / _cfg.M3_FOCAL_LENGTH_MM * _cfg.M3_WORKING_DIST_MM)
    except Exception:
        return _FOV_HALF_WIDTH_FALLBACK_MM


# Return the depth range inside the requested lateral field of view.
def fov_z_range_mm(Y: np.ndarray, Zr: np.ndarray, half_width_mm: float) -> float:

    mask = np.abs(np.asarray(Y)[:, 0]) <= half_width_mm
    if not mask.any():
        return float("nan")
    z = np.asarray(Zr)[mask, :]
    return float(z.max() - z.min())


def plot_envelope_check(
    raw: pd.DataFrame,
    envelope: pd.DataFrame,
    profile: pd.DataFrame,
    config: dict,
    ax=None,
) -> plt.Axes:

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.plot(raw["y_mm"], raw["z_mm"],
            "k.", markersize=2, alpha=0.3, label="raw extracted points")
    ax.plot(envelope["y_mm"], envelope["z_upper_raw_mm"],
            "bo", markersize=3, alpha=0.7,
            label=f"upper envelope ({config['bin_width_mm']:g} mm bins)")
    ax.plot(profile["y_mm"], profile["z_smooth_mm"],
            "r-", linewidth=1.8, label="smoothed profile")
    ax.set_xlabel("Lateral y [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(f"Envelope check — {config['variant_label']}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    return ax


def plot_profile(mesh: dict, config: dict, ax=None) -> plt.Axes:

    mid = mesh["Z"].shape[1] // 2
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mesh["Y"][:, mid], mesh["Z"][:, mid],
            color="steelblue", linewidth=1.8)
    ax.set_xlabel("Lateral y [mm]")
    ax.set_ylabel("z_smooth [mm]")
    ax.set_title(f"Lateral cross-section — {config['variant_label']}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4)
    return ax


def plot_depth_map(depth_map: np.ndarray, config: dict, ax=None) -> plt.Axes:

    xmin, xmax = config["x_range_mm"]
    ymin, ymax = config["y_range_mm"]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        depth_map,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower", aspect="auto", cmap="plasma",
    )
    plt.colorbar(im, ax=ax, label="Depth z_relative [mm]")
    ax.set_xlabel("Longitudinal x [mm]")
    ax.set_ylabel("Lateral y [mm]")
    ax.set_title(f"Depth map — {config['variant_label']}")
    return ax


def plot_surface_3d(mesh: dict, config: dict, ax=None) -> plt.Axes:

    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        mesh["X"], mesh["Y"], mesh["Z"],
        cmap="viridis", linewidth=0, antialiased=True, alpha=0.9,
    )
    ax.set_xlabel("Longitudinal x [mm]")
    ax.set_ylabel("Lateral y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(f"Tire inner surface — {config['variant_label']}")
    ax.view_init(elev=25, azim=-135)
    ax.set_box_aspect((
        np.ptp(mesh["X"]), np.ptp(mesh["Y"]), max(np.ptp(mesh["Z"]), 1.0)
    ))
    return ax


# Build one surface variant and save its grids and diagnostic plots.
def _run_and_save(
    variant: Literal["A", "B"],
    csv_path: Path,
    out_dir: Path,
) -> tuple[dict, np.ndarray, dict]:

    params = {**DEFAULTS}


    raw      = load_raw_csv(csv_path)
    envelope = extract_upper_envelope(raw, params["bin_width_mm"])
    interp   = interpolate_to_uniform_grid(
                   envelope, params["grid_step_y_mm"], params["interp_kind"])
    profile  = smooth_profile(
                   interp, params["smoothing_window"], params["smoothing_polyorder"])
    X, Y, Z, Zr = extrude_profile(
                   profile, params["surface_length_x_mm"], params["grid_step_x_mm"])

    mesh      = {"X": X, "Y": Y, "Z": Z}
    depth_map = Zr
    config    = {
        **params,
        "variant":             variant,
        "variant_label":       _VARIANT_LABELS.get(variant, csv_path.stem),
        "csv_path":            str(csv_path),
        "raw_point_count":     len(raw),
        "envelope_bin_count":  len(envelope),
        "profile_point_count": len(profile),
        "grid_shape_ny_nx":    Z.shape,
        "y_range_mm":          (float(Y[:, 0].min()), float(Y[:, 0].max())),
        "x_range_mm":          (float(X[0, :].min()), float(X[0, :].max())),
        "z_abs_min_mm":        float(Z.min()),
        "z_abs_max_mm":        float(Z.max()),
        "depth_max_mm":        float(Zr.max()),
        "fov_half_width_mm":   sensor_fov_half_width_mm(),
        "fov_z_range_mm":      fov_z_range_mm(Y, Zr, sensor_fov_half_width_mm()),
    }


    tag = f"variant_{variant}"
    print(f"\n{'='*55}")
    print(f"  Variant        : {config['variant_label']}")
    print(f"  Raw points     : {config['raw_point_count']}")
    print(f"  Envelope bins  : {config['envelope_bin_count']}")
    print(f"  Profile points : {config['profile_point_count']}")
    print(f"  Grid shape     : {config['grid_shape_ny_nx']}  (Ny × Nx)")
    print(f"  y range        : {config['y_range_mm'][0]:.1f} … {config['y_range_mm'][1]:.1f} mm")
    print(f"  x range        : {config['x_range_mm'][0]:.1f} … {config['x_range_mm'][1]:.1f} mm")
    print(f"  z abs range    : {config['z_abs_min_mm']:.2f} … {config['z_abs_max_mm']:.2f} mm")
    print(f"  Max depth      : {config['depth_max_mm']:.2f} mm")
    print(f"{'='*55}")

    out_dir.mkdir(parents=True, exist_ok=True)


    fig, ax = plt.subplots(figsize=(9, 4))
    plot_envelope_check(raw, envelope, profile, config, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"m1_{tag}_envelope_check.png", dpi=200)
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(9, 4))
    plot_profile(mesh, config, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"m1_{tag}_profile.png", dpi=200)
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(9, 5))
    plot_depth_map(depth_map, config, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"m1_{tag}_depth_map.png", dpi=200)
    plt.close(fig)


    fig = plt.figure(figsize=(10, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    plot_surface_3d(mesh, config, ax=ax3d)
    fig.tight_layout()
    fig.savefig(out_dir / f"m1_{tag}_surface_3d.png", dpi=200)
    plt.close(fig)


    np.savez_compressed(
        out_dir / f"m1_{tag}_grid.npz",
        X=mesh["X"], Y=mesh["Y"], Z=mesh["Z"], depth_map=depth_map,
    )


    profile.to_csv(out_dir / f"m1_{tag}_profile.csv", index=False)

    print(f"  Outputs → {out_dir}")
    return mesh, depth_map, config


def main() -> None:

    CSV_DIR = Path(__file__).resolve().parent / "data" / "raw"
    OUT_DIR = Path(__file__).resolve().parent / "outputs" / "m1"

    variants_to_run = {
        "A": CSV_DIR / _CSV_NAMES["A"],

    }

    for variant, csv_path in variants_to_run.items():
        if not csv_path.exists():
            print(f"[SKIP] Variant {variant}: CSV not found at {csv_path}")
            continue
        _run_and_save(variant, csv_path, OUT_DIR)

    print("\nM1 complete.")


if __name__ == "__main__":
    main()
