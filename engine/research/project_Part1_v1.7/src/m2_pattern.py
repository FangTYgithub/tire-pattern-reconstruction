# Rasterize point and line patterns and return their geometry metadata. Formal cases pass size,
# spacing and phase settings through experiment inputs.

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


SENSOR_THETA_DEG      = 18.9  # Angle for the geometric uniqueness estimate [deg].
SENSOR_DEPTH_RANGE_MM = 40.0  # Standalone depth-range estimate [mm]; not the study search window.


IMAGE_SIZE_PX      = 512  # Square projector raster size [px]; nominal study setting: 512.
FOCAL_LENGTH_MM    = 25.0  # Equivalent pinhole focal length [mm].
PIXEL_PITCH_MM     = 0.030  # Equivalent sensor/projector pixel pitch [mm/px].

WORKING_DIST_MM    = 150.0  # Nominal working distance [mm].


LINE_SPACING_MM    = 2.0  # Direct-call default line pitch [mm]; experiment rows override it.
LINE_WIDTH_MM      = 0.3  # Direct-call default line width [mm]; tested size-study widths: 0.5-1.5 mm.


DOT_SPACING_MM     = 2.0  # Direct-call default dot pitch [mm]; experiment rows specify both axes.
DOT_RADIUS_MM      = 0.15  # Direct-call default radius [mm]; size-study radii: 0.5, 0.75, 1, 1.25, 1.5 mm.


UNIQUENESS_LAMBDA      = 0.5  # Multiplier in the geometric uniqueness estimate.
UNIQUENESS_WINDOW_MM   = 5.0  # Local uniqueness window size [mm].


def _compute_fov_mm(
    image_size_px: int   = IMAGE_SIZE_PX,
    focal_length_mm: float = FOCAL_LENGTH_MM,
    pixel_pitch_mm: float  = PIXEL_PITCH_MM,
    working_dist_mm: float = WORKING_DIST_MM,
) -> tuple[float, float, float]:

    sensor_size_mm = image_size_px * pixel_pitch_mm
    fov_deg  = 2.0 * math.degrees(
        math.atan(sensor_size_mm / 2.0 / focal_length_mm)
    )
    fov_mm   = 2.0 * working_dist_mm * math.tan(
        math.radians(fov_deg / 2.0)
    )
    mm_per_px = fov_mm / image_size_px
    return fov_deg, fov_mm, mm_per_px


PatternType = Literal["lines", "dots", "vlines", "random_dots", "graded_dots", "coded_dots", "dashed_vlines", "bright_dots",
                      "epi_random_dots", "anchored_dots", "olines"]
DOT_TYPES   = ("dots", "random_dots", "graded_dots", "coded_dots", "bright_dots", "epi_random_dots", "anchored_dots")
RANDOM_TYPES = ("random_dots", "graded_dots", "epi_random_dots")
LINE_TYPES  = ("lines", "vlines", "dashed_vlines", "olines")


def bright_levels(size_levels: int, bright_ratio: float) -> list[float]:

    k = int(size_levels)
    if k < 2 or not (0.0 < bright_ratio < 1.0):
        raise ValueError("bright_dots needs size_levels ≥ 2 and 0 < bright_ratio < 1")
    return [float(bright_ratio ** (k - 1 - c)) for c in range(k)]


# Compute the shortest centre-to-centre spacing of the specified staggered lattice.
def dot_lattice_min_pitch(spacing: float, spacing_perp: float, stagger_frac: float = 0.0,
                          shear_frac: float = 0.0, n_rows: int = 6) -> float:

    off = lambda j: ((j % 2) * stagger_frac + j * shear_frac)
    best = float(spacing)
    for i in (0, 1):
        for n in range(1, n_rows + 1):
            d_row = off(i + n) - off(i)
            m = np.arange(-n_rows - 2, n_rows + 3)
            du = (m + d_row) * spacing
            best = min(best, float(np.min(np.hypot(du, n * spacing_perp))))
    return best


# Return the dot-radius levels defined by the size ratio and number of classes.
def coded_dot_radii(r_large: float, size_levels: int, size_ratio: float) -> list[float]:

    k = int(size_levels)
    if k < 2:
        return [float(r_large)]
    return [float(r_large) * float(size_ratio) ** (1.0 - i / (k - 1)) for i in range(k)]


# Create a projector raster and geometry metadata for the requested pattern family. Feature size means
# dot radius or line width, in millimetres.
def generate_pattern(
    pattern_type:    PatternType = "lines",
    spacing_mm:      float | None = None,
    feature_size_mm: float | None = None,
    image_size_px:   int   = IMAGE_SIZE_PX,
    focal_length_mm: float = FOCAL_LENGTH_MM,
    pixel_pitch_mm:  float = PIXEL_PITCH_MM,
    working_dist_mm: float = WORKING_DIST_MM,
    depth_range_mm:  float | None = None,
    phase_offset_x_mm: float = 0.0,
    phase_offset_y_mm: float = 0.0,
    seed:            int   = 0,
    min_sep_frac:    float = 0.6,
    graded_edge_spacing_mm: float | None = None,
    graded_radius_mm:       float | None = None,
    spacing_perp_mm: float | None = None,
    stagger_frac:    float = 0.0,
    jitter_frac:     float = 0.0,
    size_levels:     int   = 2,
    size_ratio:      float = 0.6,
    code_shift:      int   = 1,
    dash_length_mm:  float = 5.0,
    dash_gap_mm:     float = 1.0,
    dash_phases:     int   = 2,
    rotation_deg:    float = 0.0,
    bright_ratio:    float = 0.5,
    epi_along_mm:    float = 6.5,
    epi_across_mm:   float = 1.08,
    shear_frac:      float = 0.0,
    anchor_ratio:    float = 1.5,
    anchor_every:    tuple[int, int] = (2, 4),
    dash_levels:     int = 1,
    dash_ratio:      float = 0.5,
    line_angle_deg:  float = 0.0,
    line_phase_frac: float = 0.0,
) -> tuple[np.ndarray, dict]:


    if pattern_type == "lines":
        spacing_mm      = spacing_mm      or LINE_SPACING_MM
        feature_size_mm = feature_size_mm or LINE_WIDTH_MM
    elif pattern_type == "dots":
        spacing_mm      = spacing_mm      or DOT_SPACING_MM
        feature_size_mm = feature_size_mm or DOT_RADIUS_MM
    elif pattern_type in ("vlines", "dashed_vlines", "olines"):
        spacing_mm      = spacing_mm      or LINE_SPACING_MM
        feature_size_mm = feature_size_mm or LINE_WIDTH_MM
    elif pattern_type in RANDOM_TYPES or pattern_type in ("coded_dots", "bright_dots", "anchored_dots"):
        spacing_mm      = spacing_mm      or DOT_SPACING_MM
        feature_size_mm = feature_size_mm or DOT_RADIUS_MM
    else:
        raise ValueError(
            f"Unsupported pattern_type '{pattern_type}'. "
            "Supported: 'lines', 'dots', 'vlines', 'random_dots', 'graded_dots', 'coded_dots', 'dashed_vlines', 'bright_dots'."
        )


    fov_deg, fov_mm, mm_per_px = _compute_fov_mm(
        image_size_px, focal_length_mm, pixel_pitch_mm, working_dist_mm
    )


    spacing_px      = spacing_mm      / mm_per_px
    feature_size_px = feature_size_mm / mm_per_px
    offset_x_px     = phase_offset_x_mm / mm_per_px
    offset_y_px     = phase_offset_y_mm / mm_per_px


    if pattern_type in LINE_TYPES:
        gap_mm = spacing_mm - feature_size_mm
    elif pattern_type == "coded_dots":


        radii = coded_dot_radii(feature_size_mm, size_levels, size_ratio)
        pitch_min = min(spacing_mm, spacing_perp_mm) if spacing_perp_mm else spacing_mm
        pair = radii[-1] + (radii[-2] if len(radii) > 1 else radii[-1])
        diag = math.hypot(spacing_mm, spacing_perp_mm or spacing_mm) - 2.0 * radii[-1]
        gap_mm = min(pitch_min - pair, diag)
    elif pattern_type == "anchored_dots":

        gap_mm = dot_lattice_min_pitch(spacing_mm, spacing_perp_mm or spacing_mm, stagger_frac,
                                       shear_frac) - (1.0 + float(anchor_ratio)) * feature_size_mm
    elif pattern_type == "dots" and (stagger_frac or shear_frac):


        gap_mm = dot_lattice_min_pitch(spacing_mm, spacing_perp_mm or spacing_mm, stagger_frac,
                                       shear_frac) - 2.0 * feature_size_mm
    else:
        pitch_min = min(spacing_mm, spacing_perp_mm) if spacing_perp_mm else spacing_mm
        gap_mm = pitch_min - 2.0 * feature_size_mm  # feature_size is the radius.
    if gap_mm <= 0:
        raise ValueError(
            f"Feature too large for given spacing: "
            f"gap = {gap_mm:.3f} mm ≤ 0. "
            f"Reduce feature_size_mm or increase spacing_mm."
        )
    if pattern_type == "dashed_vlines" and (dash_length_mm <= 0 or int(dash_phases) < 1 or dash_gap_mm < 0):
        raise ValueError(f"dashed_vlines needs dash_length_mm > 0, dash_gap_mm ≥ 0, dash_phases ≥ 1 "
                         f"(got {dash_length_mm}, {dash_gap_mm}, {dash_phases})")


    if rotation_deg and pattern_type not in ("dots", "coded_dots", "bright_dots", "anchored_dots"):
        raise ValueError("rotation_deg is only defined for dot lattices ('dots', 'coded_dots', 'bright_dots', 'anchored_dots')")
    if shear_frac and pattern_type not in ("dots", "anchored_dots"):
        raise ValueError("shear_frac is only defined for plain and anchored dot lattices")
    if pattern_type == "anchored_dots" and (float(anchor_ratio) <= 1.0 or min(int(anchor_every[0]), int(anchor_every[1])) < 1):
        raise ValueError("anchored_dots needs anchor_ratio > 1 and anchor_every ≥ (1, 1)")


    dot_centres = None
    extra_meta: dict = {}
    if pattern_type == "lines":
        pattern = _generate_lines(
            image_size_px, spacing_px, feature_size_px,
            offset_px = offset_y_px,
        )
    elif pattern_type == "vlines":
        pattern = _generate_vertical_lines(
            image_size_px, spacing_px, feature_size_px,
            offset_px = offset_x_px,
        )
    elif pattern_type == "olines":
        pattern, line_info = _generate_oriented_lines(
            image_size_px, spacing_px, feature_size_px,
            angle_deg = float(line_angle_deg), phase_frac = float(line_phase_frac),
        )
        extra_meta = {"line_angle_deg": float(line_angle_deg), "line_phase_frac": float(line_phase_frac), "olines": line_info}
    elif pattern_type == "dashed_vlines":
        kl = max(1, int(dash_levels))
        pattern = _generate_dashed_vlines(
            image_size_px, spacing_px, feature_size_px,
            dash_px = dash_length_mm / mm_per_px, gap_px = dash_gap_mm / mm_per_px,
            n_phases = int(dash_phases), offset_x_px = offset_x_px, offset_y_px = offset_y_px,
            dash_levels = kl, dash_ratio = float(dash_ratio),
        )
        if kl > 1:
            lengths = dash_lengths(dash_length_mm, kl, dash_ratio)
            extra_meta = {"dash_length_mm": float(dash_length_mm), "dash_gap_mm": float(dash_gap_mm),
                          "dash_phases": 1, "dash_levels": kl, "dash_ratio": float(dash_ratio),
                          "dash_lengths_mm": " / ".join(f"{x:.3f}" for x in lengths),
                          "same_class_pitch_mm": float(kl * spacing_mm), "row_pitch_lit_mm": float(spacing_mm),
                          "lit_row_fraction": float(np.mean([x / (x + dash_gap_mm) for x in lengths]))}
        else:
            k = int(dash_phases)
            extra_meta = {"dash_length_mm": float(dash_length_mm), "dash_gap_mm": float(dash_gap_mm),
                          "dash_phases": k, "row_pitch_lit_mm": float(k * spacing_mm),
                          "lit_row_fraction": float(dash_length_mm / (dash_length_mm + dash_gap_mm))}
    elif pattern_type == "coded_dots":
        perp_mm = spacing_perp_mm if spacing_perp_mm else spacing_mm
        radii_mm = coded_dot_radii(feature_size_mm, size_levels, size_ratio)
        pattern, dot_centres, classes = _generate_coded_dots(
            image_size_px, spacing_px, [r / mm_per_px for r in radii_mm],
            spacing_perp_px = perp_mm / mm_per_px, code_shift = int(code_shift),
            offset_x_px = offset_x_px, offset_y_px = offset_y_px,
            rotation_deg = float(rotation_deg),
        )
        k = len(radii_mm)
        extra_meta = {
            "spacing_perp_mm": float(perp_mm), "stagger_frac": 0.0, "jitter_frac": 0.0,
            "rotation_deg": float(rotation_deg),
            "lattice": "coded", "size_levels": k, "size_ratio": float(size_ratio),
            "code_shift": int(code_shift),
            "size_radii_mm": " / ".join(f"{r:.3f}" for r in radii_mm),
            "same_class_pitch_mm": float(k * spacing_mm),
            "class_counts": " / ".join(str(int((classes == c).sum())) for c in range(k)),
        }
    elif pattern_type == "bright_dots":
        perp_mm = spacing_perp_mm if spacing_perp_mm else spacing_mm
        levels = bright_levels(size_levels, bright_ratio)
        pattern, dot_centres, classes = _generate_bright_dots(
            image_size_px, spacing_px, feature_size_px, levels,
            spacing_perp_px = perp_mm / mm_per_px, code_shift = int(code_shift),
            offset_x_px = offset_x_px, offset_y_px = offset_y_px, rotation_deg = float(rotation_deg),
        )
        k = len(levels)
        extra_meta = {
            "spacing_perp_mm": float(perp_mm), "stagger_frac": 0.0, "jitter_frac": 0.0,
            "rotation_deg": float(rotation_deg),
            "lattice": "bright", "size_levels": k, "bright_ratio": float(bright_ratio),
            "code_shift": int(code_shift),
            "bright_levels": " / ".join(f"{x:.3f}" for x in levels),
            "same_class_pitch_mm": float(k * spacing_mm),
            "class_counts": " / ".join(str(int((classes == c).sum())) for c in range(k)),
        }
    elif pattern_type == "anchored_dots":
        perp_mm = spacing_perp_mm if spacing_perp_mm else spacing_mm
        r_anchor_px = feature_size_px * float(anchor_ratio)
        every = (int(anchor_every[0]), int(anchor_every[1]))
        pattern, dot_centres, classes = _generate_anchored_dots(
            image_size_px, spacing_px, feature_size_px, r_anchor_px, every,
            spacing_perp_px = perp_mm / mm_per_px, stagger_frac = stagger_frac, shear_frac = float(shear_frac),
            rotation_deg = float(rotation_deg), offset_x_px = offset_x_px, offset_y_px = offset_y_px,
        )
        extra_meta = {
            "spacing_perp_mm": float(perp_mm), "stagger_frac": float(stagger_frac), "jitter_frac": 0.0,
            "rotation_deg": float(rotation_deg), "shear_frac": float(shear_frac),
            "lattice": "anchored", "size_levels": 2, "anchor_ratio": float(anchor_ratio),
            "anchor_every": f"{every[0]} / {every[1]}",
            "size_radii_mm": f"{feature_size_mm:.3f} / {feature_size_mm * float(anchor_ratio):.3f}",
            "anchor_pitch_mm": f"{every[0] * spacing_mm:.2f} / {every[1] * perp_mm:.2f}",
            "n_anchors": int((classes == 1).sum()),
            "class_counts": " / ".join(str(int((classes == c).sum())) for c in range(2)),
        }
    elif pattern_type == "epi_random_dots":
        pattern, dot_centres, extra_meta = _generate_epi_random_dots(
            image_size_px, spacing_px, feature_size_px,
            seed = seed, min_sep_frac = min_sep_frac,
            win_u_px = float(epi_along_mm) / mm_per_px, win_v_px = float(epi_across_mm) / mm_per_px,
        )
        extra_meta.update({"epi_along_mm": float(epi_along_mm), "epi_across_mm": float(epi_across_mm)})
    elif pattern_type == "random_dots":
        pattern, dot_centres, extra_meta = _generate_random_dots(
            image_size_px, spacing_px, feature_size_px,
            seed = seed, min_sep_frac = min_sep_frac,
        )
    elif pattern_type == "graded_dots":
        edge_mm   = graded_edge_spacing_mm if graded_edge_spacing_mm else 2.0 * spacing_mm
        radius_mm = graded_radius_mm if graded_radius_mm else fov_mm / 2.0
        pattern, dot_centres, extra_meta = _generate_graded_dots(
            image_size_px, spacing_px, edge_mm / mm_per_px, feature_size_px,
            radius_px = radius_mm / mm_per_px, seed = seed, min_sep_frac = min_sep_frac,
        )
        extra_meta.update({"graded_edge_spacing_mm": edge_mm, "graded_radius_mm": radius_mm})
    else:
        perp_mm = spacing_perp_mm if spacing_perp_mm else spacing_mm
        pattern, dot_centres = _generate_dots(
            image_size_px, spacing_px, feature_size_px,
            offset_x_px = offset_x_px, offset_y_px = offset_y_px,
            spacing_perp_px = perp_mm / mm_per_px,
            stagger_frac = stagger_frac, jitter_frac = jitter_frac, seed = seed,
            rotation_deg = float(rotation_deg), shear_frac = float(shear_frac),
        )
        extra_meta = {
            "spacing_perp_mm": float(perp_mm),
            "stagger_frac":    float(stagger_frac),
            "jitter_frac":     float(jitter_frac),
            "rotation_deg":    float(rotation_deg),
            "shear_frac":      float(shear_frac),
            "lattice": ("jittered" if jitter_frac > 0 else
                        "sheared" if shear_frac else
                        "staggered" if stagger_frac > 0 else
                        "rect" if abs(perp_mm - spacing_mm) > 1e-9 else "square"),
        }


    meta = _build_metadata(
        pattern_type    = pattern_type,
        pattern         = pattern,
        spacing_mm      = spacing_mm,
        feature_size_mm = feature_size_mm,
        spacing_px      = spacing_px,
        feature_size_px = feature_size_px,
        image_size_px   = image_size_px,
        fov_deg         = fov_deg,
        fov_mm          = fov_mm,
        mm_per_px       = mm_per_px,
        working_dist_mm = working_dist_mm,
        depth_range_mm  = depth_range_mm,
        phase_offset_x_mm = phase_offset_x_mm,
        phase_offset_y_mm = phase_offset_y_mm,
        seed            = seed if (pattern_type in RANDOM_TYPES or jitter_frac > 0) else None,
        dot_centres     = dot_centres,
        extra_meta      = extra_meta,
    )

    return pattern, meta


# Draw equally spaced horizontal bands on the projector raster.
def _generate_lines(
    image_size_px: int,
    spacing_px:    float,
    line_width_px: float,
    offset_px:     float = 0.0,
) -> np.ndarray:

    H = W = image_size_px
    pattern = np.zeros((H, W), dtype=np.float32)

    centre = (H - 1) / 2.0 + offset_px


    k = 0
    while True:
        placed = False
        for sign in ([0] if k == 0 else [+1, -1]):
            row_centre = centre + sign * k * spacing_px
            r_min = int(math.floor(row_centre - line_width_px / 2.0))
            r_max = int(math.ceil( row_centre + line_width_px / 2.0))
            r_min = max(r_min, 0)
            r_max = min(r_max, H - 1)
            if r_min <= H - 1 and r_max >= 0:
                pattern[r_min:r_max + 1, :] = 1.0
                placed = True
        if k > 0 and not placed:
            break
        k += 1

    return pattern


# Draw a dot lattice with the supplied offsets, staggering, rotation and jitter.
def _generate_dots(
    image_size_px:   int,
    spacing_px:      float,
    dot_radius_px:   float,
    offset_x_px:     float = 0.0,
    offset_y_px:     float = 0.0,
    spacing_perp_px: float | None = None,
    stagger_frac:    float = 0.0,
    jitter_frac:     float = 0.0,
    seed:            int   = 0,
    rotation_deg:    float = 0.0,
    shear_frac:      float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:

    H = W = image_size_px
    cx, cy, _, _ = _lattice_centres(image_size_px, spacing_px, spacing_perp_px, offset_x_px, offset_y_px,
                                    stagger_frac, shear_frac, rotation_deg, jitter_frac, seed)
    r = dot_radius_px
    drawn  = (cx >= -r) & (cx <= W - 1 + r) & (cy >= -r) & (cy <= H - 1 + r)
    inside = (cx >= 0) & (cx <= W - 1) & (cy >= 0) & (cy <= H - 1)
    pattern = _draw_dots(image_size_px, np.stack([cx[drawn], cy[drawn]], axis=1), dot_radius_px)
    centres = np.stack([cx[inside], cy[inside]], axis=1).astype(np.float64)
    return pattern, centres


# Calculate dot centres and their lattice indices before drawing the raster.
def _lattice_centres(image_size_px: int, spacing_px: float, spacing_perp_px: float | None, offset_x_px: float,
                     offset_y_px: float, stagger_frac: float = 0.0, shear_frac: float = 0.0, rotation_deg: float = 0.0,
                     jitter_frac: float = 0.0, seed: int = 0):

    H = W = image_size_px
    sp_y = float(spacing_perp_px) if spacing_perp_px else float(spacing_px)
    centre_y = (H - 1) / 2.0 + offset_y_px
    centre_x = (W - 1) / 2.0 + offset_x_px

    n_x = int(math.ceil(W / spacing_px)) + 2
    n_y = int(math.ceil(H / sp_y)) + 2
    if rotation_deg:
        n_x = n_y = int(math.ceil(1.5 * W / min(spacing_px, sp_y))) + 2
    IY, IX = np.meshgrid(np.arange(-n_y, n_y + 1), np.arange(-n_x, n_x + 1), indexing="ij")
    cy = centre_y + IY * sp_y
    cx = centre_x + IX * spacing_px + (IY % 2) * float(stagger_frac) * spacing_px
    if shear_frac:
        cx = cx + np.mod(IY * float(shear_frac), 1.0) * spacing_px
    if rotation_deg:
        c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
        xr, yr = cx - centre_x, cy - centre_y
        cx, cy = centre_x + xr * c - yr * s, centre_y + xr * s + yr * c
    if jitter_frac > 0:
        rng = np.random.default_rng(int(seed))
        cx = cx + rng.uniform(-1.0, 1.0, cx.shape) * float(jitter_frac) * spacing_px
        cy = cy + rng.uniform(-1.0, 1.0, cy.shape) * float(jitter_frac) * spacing_px
    return cx, cy, IX, IY


# Draw a dot lattice with larger anchor dots at the selected index intervals.
def _generate_anchored_dots(
    image_size_px:   int,
    spacing_px:      float,
    dot_radius_px:   float,
    anchor_radius_px: float,
    anchor_every:    tuple[int, int] = (2, 4),
    spacing_perp_px: float | None = None,
    stagger_frac:    float = 0.0,
    shear_frac:      float = 0.0,
    rotation_deg:    float = 0.0,
    offset_x_px:     float = 0.0,
    offset_y_px:     float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    H = W = image_size_px
    cx, cy, IX, IY = _lattice_centres(image_size_px, spacing_px, spacing_perp_px, offset_x_px, offset_y_px,
                                      stagger_frac, shear_frac, rotation_deg)
    m, n = max(1, int(anchor_every[0])), max(1, int(anchor_every[1]))
    cls = ((np.mod(IX, m) == 0) & (np.mod(IY, n) == 0)).astype(np.int64)
    pattern = np.zeros((H, W), dtype=np.float32)
    for c, r in enumerate((dot_radius_px, anchor_radius_px)):
        drawn = (cls == c) & (cx >= -r) & (cx <= W - 1 + r) & (cy >= -r) & (cy <= H - 1 + r)
        pattern = np.maximum(pattern, _draw_dots(image_size_px, np.stack([cx[drawn], cy[drawn]], axis=1), r))
    inside = (cx >= 0) & (cx <= W - 1) & (cy >= 0) & (cy <= H - 1)
    centres = np.stack([cx[inside], cy[inside]], axis=1).astype(np.float64)
    return pattern, centres, cls[inside]


# Assign radius classes to lattice positions and draw the coded dots.
def _generate_coded_dots(
    image_size_px:   int,
    spacing_px:      float,
    radii_px:        list[float],
    spacing_perp_px: float | None = None,
    code_shift:      int = 1,
    offset_x_px:     float = 0.0,
    offset_y_px:     float = 0.0,
    rotation_deg:    float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    H = W = image_size_px
    k = len(radii_px)
    cx, cy, cls = _code_lattice(image_size_px, spacing_px, spacing_perp_px, code_shift, k,
                                offset_x_px, offset_y_px, rotation_deg)
    pattern = np.zeros((H, W), dtype=np.float32)
    for c, r in enumerate(radii_px):
        drawn = (cls == c) & (cx >= -r) & (cx <= W - 1 + r) & (cy >= -r) & (cy <= H - 1 + r)
        pattern = np.maximum(pattern, _draw_dots(image_size_px, np.stack([cx[drawn], cy[drawn]], axis=1), r))
    inside = (cx >= 0) & (cx <= W - 1) & (cy >= 0) & (cy <= H - 1)
    centres = np.stack([cx[inside], cy[inside]], axis=1).astype(np.float64)
    return pattern, centres, cls[inside].astype(np.int64)


# Calculate dot centres and periodic class labels from the lattice indices.
def _code_lattice(image_size_px: int, spacing_px: float, spacing_perp_px: float | None, code_shift: int, k: int,
                  offset_x_px: float, offset_y_px: float, rotation_deg: float = 0.0):

    H = W = image_size_px
    sp_y = float(spacing_perp_px) if spacing_perp_px else float(spacing_px)
    centre_y = (H - 1) / 2.0 + offset_y_px
    centre_x = (W - 1) / 2.0 + offset_x_px
    n_x = int(math.ceil(W / spacing_px)) + 2
    n_y = int(math.ceil(H / sp_y)) + 2
    if rotation_deg:
        n_x = n_y = int(math.ceil(1.5 * W / min(spacing_px, sp_y))) + 2
    IY, IX = np.meshgrid(np.arange(-n_y, n_y + 1), np.arange(-n_x, n_x + 1), indexing="ij")
    cy = centre_y + IY * sp_y
    cx = centre_x + IX * spacing_px
    if rotation_deg:
        c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
        xr, yr = cx - centre_x, cy - centre_y
        cx, cy = centre_x + xr * c - yr * s, centre_y + xr * s + yr * c
    cls = np.mod(IX + int(code_shift) * IY, k)
    return cx, cy, cls


# Draw equal-radius dots with class-dependent intensities.
def _generate_bright_dots(
    image_size_px:   int,
    spacing_px:      float,
    radius_px:       float,
    levels:          list[float],
    spacing_perp_px: float | None = None,
    code_shift:      int = 1,
    offset_x_px:     float = 0.0,
    offset_y_px:     float = 0.0,
    rotation_deg:    float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    H = W = image_size_px
    k = len(levels)
    cx, cy, cls = _code_lattice(image_size_px, spacing_px, spacing_perp_px, code_shift, k,
                                offset_x_px, offset_y_px, rotation_deg)
    r = radius_px
    pattern = np.zeros((H, W), dtype=np.float32)
    for c, lev in enumerate(levels):
        drawn = (cls == c) & (cx >= -r) & (cx <= W - 1 + r) & (cy >= -r) & (cy <= H - 1 + r)
        pattern = np.maximum(pattern, float(lev) * _draw_dots(image_size_px, np.stack([cx[drawn], cy[drawn]], axis=1), r))
    inside = (cx >= 0) & (cx <= W - 1) & (cy >= 0) & (cy <= H - 1)
    centres = np.stack([cx[inside], cy[inside]], axis=1).astype(np.float64)
    return pattern, centres, cls[inside].astype(np.int64)


# Draw vertical dashed lines with the selected lengths and phase offsets.
def _generate_dashed_vlines(
    image_size_px: int,
    spacing_px:    float,
    line_width_px: float,
    dash_px:       float,
    gap_px:        float,
    n_phases:      int = 2,
    offset_x_px:   float = 0.0,
    offset_y_px:   float = 0.0,
    dash_levels:   int = 1,
    dash_ratio:    float = 0.5,
) -> np.ndarray:

    H = W = image_size_px
    pattern = np.zeros((H, W), dtype=np.float32)
    kl = max(1, int(dash_levels))
    k = max(1, int(n_phases)) if kl == 1 else 1
    step = dash_px + gap_px
    period = k * step
    centre = (W - 1) / 2.0 + offset_x_px
    v0 = (H - 1) / 2.0 + offset_y_px
    rows = np.arange(H, dtype=np.float64)
    n_max = int(math.ceil(W / spacing_px)) + 2
    for n in range(-n_max, n_max + 1):
        col_centre = centre + n * spacing_px
        c_min = int(math.floor(col_centre - line_width_px / 2.0))
        c_max = int(math.ceil(col_centre + line_width_px / 2.0))
        c_min, c_max = max(c_min, 0), min(c_max, W - 1)
        if c_min > W - 1 or c_max < 0 or c_min > c_max:
            continue
        if kl > 1:
            d_c = dash_px * float(dash_ratio) ** (n % kl)
            lit = np.mod(rows - v0, d_c + gap_px) < d_c
        else:
            lit = np.mod(rows - v0 - (n % k) * step, period) < dash_px
        pattern[np.ix_(lit, np.arange(c_min, c_max + 1))] = 1.0
    return pattern


def dash_lengths(dash_length_mm: float, dash_levels: int, dash_ratio: float) -> list[float]:

    return [float(dash_length_mm) * float(dash_ratio) ** c for c in range(max(1, int(dash_levels)))]


# Draw equally spaced vertical bands on the projector raster.
def _generate_vertical_lines(
    image_size_px: int,
    spacing_px:    float,
    line_width_px: float,
    offset_px:     float = 0.0,
) -> np.ndarray:

    H = W = image_size_px
    pattern = np.zeros((H, W), dtype=np.float32)

    centre = (W - 1) / 2.0 + offset_px

    k = 0
    while True:
        placed = False
        for sign in ([0] if k == 0 else [+1, -1]):
            col_centre = centre + sign * k * spacing_px
            c_min = int(math.floor(col_centre - line_width_px / 2.0))
            c_max = int(math.ceil( col_centre + line_width_px / 2.0))
            c_min = max(c_min, 0)
            c_max = min(c_max, W - 1)
            if c_min <= W - 1 and c_max >= 0:
                pattern[:, c_min:c_max + 1] = 1.0
                placed = True
        if k > 0 and not placed:
            break
        k += 1

    return pattern


# Draw parallel continuous lines at the supplied angle and fractional phase.
def _generate_oriented_lines(
    image_size_px: int,
    spacing_px:    float,
    line_width_px: float,
    angle_deg:     float = 0.0,
    phase_frac:    float = 0.0,
) -> tuple[np.ndarray, dict]:

    H = W = image_size_px
    cu, cv = (W - 1) / 2.0, (H - 1) / 2.0
    phi = math.radians(angle_deg)
    nx, ny = math.cos(phi), math.sin(phi)
    tx, ty = -math.sin(phi), math.cos(phi)
    vv, uu = np.mgrid[0:H, 0:W]
    q = (uu - cu) * nx + (vv - cv) * ny                  # Signed normal coordinate, pixel centres.
    r = q / spacing_px - phase_frac
    k = np.round(r)
    dist = np.abs(r - k) * spacing_px
    lit = dist <= line_width_px / 2.0
    pattern = lit.astype(np.float32)
    ids, counts = np.unique(k[lit].astype(np.int64), return_counts=True)
    segments: dict = {}
    clipped: dict = {}
    for kk, cnt in zip(ids.tolist(), counts.tolist()):
        qk = (kk + phase_frac) * spacing_px
        p0 = np.array([cu + qk * nx, cv + qk * ny]); t = np.array([tx, ty])
        lo, hi = -math.inf, math.inf
        for axis, (a, b) in enumerate(((-0.5, W - 0.5), (-0.5, H - 0.5))):
            if abs(t[axis]) < 1e-12:
                if not (a <= p0[axis] <= b):
                    lo, hi = math.inf, -math.inf
            else:
                l1, l2 = (a - p0[axis]) / t[axis], (b - p0[axis]) / t[axis]
                lo, hi = max(lo, min(l1, l2)), min(hi, max(l1, l2))
        length = max(hi - lo, 0.0) if hi > lo else 0.0
        if length > 0:
            segments[str(kk)] = {"q_px": float(qk), "start_uv": (p0 + lo * t).tolist(), "end_uv": (p0 + hi * t).tolist(),
                                 "length_px": float(length), "lit_px": int(cnt), "mean_lit_width_px": float(cnt / length)}
        else:
            clipped[str(kk)] = {"q_px": float(qk), "lit_px": int(cnt)}
    ids_in = [int(x) for x in ids if str(int(x)) in segments]
    info = {"angle_deg": float(angle_deg), "phase_frac": float(phase_frac), "spacing_px": float(spacing_px), "width_px": float(line_width_px),
            "centre_uv": (cu, cv), "normal_uv": (nx, ny), "direction_uv": (tx, ty), "ids": ids_in, "n_lines": int(len(ids_in)),
            "lit_px_total": int(lit.sum()), "segments": segments, "clipped_lines": clipped,
            "rule": "pixel lit when |n.(p - c) - (k + phase_frac) * spacing_px| <= line_width_px / 2; ids = lines whose centre segment lies inside the image (fully clipped bands stay lit and are listed under clipped_lines)"}
    return pattern, info


def _draw_dots(image_size_px: int, centres: np.ndarray, dot_radius_px: float) -> np.ndarray:

    H = W = image_size_px
    pattern = np.zeros((H, W), dtype=np.float32)
    r  = int(math.ceil(dot_radius_px)) + 1
    r2 = dot_radius_px ** 2
    for cx, cy in centres:
        x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 2)
        y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        pattern[y0:y1, x0:x1][(xs - cx) ** 2 + (ys - cy) ** 2 <= r2] = 1.0
    return pattern


# Sample dot centres subject to a position-dependent minimum separation.
def _dart_throw(
    rng:        np.random.Generator,
    n_target:   int,
    W:          int,
    H:          int,
    margin:     float,
    min_sep_fn, accept_prob_fn=None,
    max_tries:  int | None = None,
) -> np.ndarray:

    max_tries = max_tries or 60 * max(n_target, 1)
    pts: list[list[float]] = []
    arr = np.empty((0, 2))
    tries = 0
    while len(pts) < n_target and tries < max_tries:
        tries += 1
        x = rng.uniform(margin, W - 1 - margin)
        y = rng.uniform(margin, H - 1 - margin)
        if accept_prob_fn is not None and rng.uniform() > accept_prob_fn(x, y):
            continue
        if len(pts) and np.min(np.hypot(arr[:, 0] - x, arr[:, 1] - y)) < min_sep_fn(x, y):
            continue
        pts.append([x, y])
        arr = np.asarray(pts, dtype=np.float64)
    return arr


# Generate randomly placed dots with a minimum centre separation.
def _generate_random_dots(
    image_size_px: int,
    spacing_px:    float,
    dot_radius_px: float,
    seed:          int = 0,
    min_sep_frac:  float = 0.6,
) -> tuple[np.ndarray, np.ndarray, dict]:

    rng      = np.random.default_rng(int(seed))
    n_target = int(round((image_size_px / spacing_px) ** 2))
    min_sep  = max(min_sep_frac * spacing_px, 2.0 * dot_radius_px + 2.0)
    centres  = _dart_throw(rng, n_target, image_size_px, image_size_px,
                           margin=dot_radius_px + 1.0, min_sep_fn=lambda x, y: min_sep)
    pattern  = _draw_dots(image_size_px, centres, dot_radius_px)
    return pattern, centres, {"n_target": n_target, "min_sep_px": round(min_sep, 3),
                              "n_placed": int(len(centres))}


# Generate random dots subject to separation in an epipolar neighbourhood.
def _generate_epi_random_dots(
    image_size_px: int,
    spacing_px:    float,
    dot_radius_px: float,
    seed:          int = 0,
    min_sep_frac:  float = 0.9,
    win_u_px:      float = 36.0,
    win_v_px:      float = 6.0,
) -> tuple[np.ndarray, np.ndarray, dict]:

    rng      = np.random.default_rng(int(seed))
    n_target = int(round((image_size_px / spacing_px) ** 2))
    min_sep  = max(min_sep_frac * spacing_px, 2.0 * dot_radius_px + 2.0)
    margin   = dot_radius_px + 1.0
    W = H = image_size_px
    pts: list[list[float]] = []
    arr = np.empty((0, 2))
    tries, max_tries = 0, 200 * max(n_target, 1)
    while len(pts) < n_target and tries < max_tries:
        tries += 1
        x = rng.uniform(margin, W - 1 - margin)
        y = rng.uniform(margin, H - 1 - margin)
        if len(pts):
            du = np.abs(arr[:, 0] - x); dv = np.abs(arr[:, 1] - y)
            if np.min(np.hypot(du, dv)) < min_sep:
                continue
            if np.any((du < win_u_px) & (dv < win_v_px)):
                continue
        pts.append([x, y])
        arr = np.asarray(pts, dtype=np.float64)
    pattern = _draw_dots(image_size_px, arr, dot_radius_px)
    return pattern, arr, {"n_target": n_target, "min_sep_px": round(min_sep, 3), "n_placed": int(len(arr)),
                          "epi_window_u_px": round(float(win_u_px), 2), "epi_window_v_px": round(float(win_v_px), 2),
                          "epi_saturated": bool(len(arr) < n_target), "epi_tries": int(tries)}


# Vary dot density between the central and outer regions.
def _generate_graded_dots(
    image_size_px:     int,
    spacing_center_px: float,
    spacing_edge_px:   float,
    dot_radius_px:     float,
    radius_px:         float,
    seed:              int = 0,
    min_sep_frac:      float = 0.6,
) -> tuple[np.ndarray, np.ndarray, dict]:

    rng = np.random.default_rng(int(seed))
    c   = (image_size_px - 1) / 2.0

    def s_of(x: float, y: float) -> float:
        r = math.hypot(x - c, y - c)
        t = min(r / radius_px, 1.0) if radius_px > 0 else 1.0
        return spacing_center_px + (spacing_edge_px - spacing_center_px) * t


    g = np.linspace(0, image_size_px - 1, 65)
    xs, ys = np.meshgrid(g, g)
    n_target = int(round(np.sum([1.0 / s_of(x, y) ** 2 for x, y in zip(xs.ravel(), ys.ravel())])
                         * (image_size_px / 65.0) ** 2))
    sep_floor = 2.0 * dot_radius_px + 2.0
    centres = _dart_throw(
        rng, n_target, image_size_px, image_size_px, margin=dot_radius_px + 1.0,
        min_sep_fn     = lambda x, y: max(min_sep_frac * s_of(x, y), sep_floor),
        accept_prob_fn = lambda x, y: (spacing_center_px / s_of(x, y)) ** 2,
        max_tries      = 120 * max(n_target, 1),
    )
    pattern = _draw_dots(image_size_px, centres, dot_radius_px)
    return pattern, centres, {"n_target": n_target, "n_placed": int(len(centres)),
                              "spacing_center_px": round(spacing_center_px, 3),
                              "spacing_edge_px": round(spacing_edge_px, 3)}


# Collect the geometric dimensions, feature locations and resource descriptors of the raster.
def _build_metadata(
    pattern_type:    str,
    pattern:         np.ndarray,
    spacing_mm:      float,
    feature_size_mm: float,
    spacing_px:      float,
    feature_size_px: float,
    image_size_px:   int,
    fov_deg:         float,
    fov_mm:          float,
    mm_per_px:       float,
    working_dist_mm: float,
    depth_range_mm:  float | None = None,
    phase_offset_x_mm: float = 0.0,
    phase_offset_y_mm: float = 0.0,
    seed:            int | None = None,
    dot_centres:     np.ndarray | None = None,
    extra_meta:      dict | None = None,
) -> dict:

    H, W = pattern.shape


    density            = float(pattern.mean())
    spatial_freq_lpmm  = 1.0 / spacing_mm

    regularity_index = 1.0
    nn_mean_mm = nn_min_mm = nn_std_mm = float("nan")


    em = extra_meta or {}
    n_code = int(em.get("size_levels", em.get("dash_levels", em.get("dash_phases", 1))) or 1)
    if pattern_type in LINE_TYPES:
        feature_count = int(round(fov_mm / spacing_mm))
        n_epipolar    = max(1, int(round(feature_count / n_code)))
    else:
        n_per_axis    = fov_mm / spacing_mm
        feature_count = int(round(n_per_axis ** 2))


        n_epipolar    = max(1, int(round(n_per_axis / n_code)))
        if dot_centres is not None and len(dot_centres) > 1:
            from scipy.spatial import cKDTree
            feature_count = int(len(dot_centres))
            d, _ = cKDTree(dot_centres).query(dot_centres, k=2)
            nn = d[:, 1] * mm_per_px
            nn_mean_mm, nn_min_mm, nn_std_mm = float(nn.mean()), float(nn.min()), float(nn.std())
            regularity_index = float(max(0.0, 1.0 - nn_std_mm / nn_mean_mm)) if nn_mean_mm > 0 else 0.0


    spacing_deg = float(spacing_px * (fov_deg / image_size_px))


    window_size_mm   = UNIQUENESS_WINDOW_MM
    if pattern_type in LINE_TYPES:
        n_in_window = window_size_mm / spacing_mm
    else:
        n_in_window = (window_size_mm / spacing_mm) ** 2
    uniqueness_estimate = float(
        1.0 - math.exp(-UNIQUENESS_LAMBDA * n_in_window)
    )


    theta_rad       = math.radians(SENSOR_THETA_DEG)
    delta_Z         = depth_range_mm if depth_range_mm is not None else SENSOR_DEPTH_RANGE_MM
    N_L             = n_epipolar if n_epipolar > 0 else 1
    uniqueness_ratio = fov_mm / (N_L * delta_Z * math.tan(theta_rad))
    aliasing_risk    = uniqueness_ratio <= 1.0


    meta: dict = {

        "pattern_type":       pattern_type,
        "image_size_px":      image_size_px,
        "spacing_mm":         spacing_mm,
        "feature_size_mm":    feature_size_mm,
        "phase_offset_x_mm":  round(phase_offset_x_mm, 4),
        "phase_offset_y_mm":  round(phase_offset_y_mm, 4),
    }


    if pattern_type == "lines":
        meta["line_width_mm"]  = feature_size_mm
        meta["dot_radius_mm"]  = None
        meta["orientation"]    = "horizontal"
    elif pattern_type in ("vlines", "dashed_vlines"):
        meta["line_width_mm"]  = feature_size_mm
        meta["dot_radius_mm"]  = None
        meta["orientation"]    = "vertical"
    elif pattern_type == "olines":
        meta["line_width_mm"]  = feature_size_mm
        meta["dot_radius_mm"]  = None
        meta["orientation"]    = "oriented"
    else:
        meta["line_width_mm"]  = None
        meta["dot_radius_mm"]  = feature_size_mm
        meta["grid_type"]      = {"random_dots": "random", "graded_dots": "graded", "epi_random_dots": "epi_random",
                                  "coded_dots": "coded", "bright_dots": "bright", "anchored_dots": "anchored"}.get(pattern_type, "square")
        if seed is not None:
            meta["seed"] = int(seed)
    if extra_meta:
        meta.update(extra_meta)

    meta.update({

        "density":             round(density, 4),
        "spatial_freq_lpmm":   round(spatial_freq_lpmm, 4),
        "feature_count":       feature_count,
        "n_features_epipolar": n_epipolar,


        "mm_per_px":           round(mm_per_px, 5),
        "fov_mm":              round(fov_mm, 3),
        "fov_deg":             round(fov_deg, 4),
        "working_dist_mm":     working_dist_mm,
        "spacing_px":          round(spacing_px, 2),
        "feature_size_px":     round(feature_size_px, 2),
        "spacing_deg":         round(spacing_deg, 5),


        "regularity_index":    round(regularity_index, 4),
        "nn_dist_mean_mm":     round(nn_mean_mm, 4) if nn_mean_mm == nn_mean_mm else float("nan"),
        "nn_dist_min_mm":      round(nn_min_mm, 4)  if nn_min_mm == nn_min_mm else float("nan"),
        "nn_dist_std_mm":      round(nn_std_mm, 4)  if nn_std_mm == nn_std_mm else float("nan"),


        "uniqueness_ratio":    round(uniqueness_ratio, 4),
        "aliasing_risk":       aliasing_risk,
        "depth_range_used_mm": round(delta_Z, 3),
        "aliasing_note": (
            "Willomitzer Eq.(4.4): ΔX/(N_L·ΔZ·tanθ)={:.4f}  ΔZ={:.2f}mm ({}). ".format(
                uniqueness_ratio, delta_Z,
                "actual FOV" if depth_range_mm is not None else "full cross-section (pessimistic)"
            )
            + ("RISK: N_L·ΔZ·tanθ > ΔX — aliasing expected at this spacing."
               if aliasing_risk else
               "OK: uniqueness condition satisfied.")
        ),


        "uniqueness_estimate": round(uniqueness_estimate, 4),
        "uniqueness_estimate_note": (
            "preliminary — 1-exp(-{:.1f} * n_features_in_window), "
            "pending M5 validation".format(UNIQUENESS_LAMBDA)
        ),
    })

    return meta


# Plot the raster together with its geometric metadata.
def visualise_pattern(
    pattern:  np.ndarray,
    metadata: dict,
    save_dir: Path | None = None,
    show:     bool = False,
) -> None:

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"M2 Pattern — {metadata['pattern_type']}  "
        f"spacing={metadata['spacing_mm']} mm  "
        f"density={metadata['density']*100:.1f}%",
        fontsize=12,
    )


    ax = axes[0]
    ax.imshow(pattern, cmap="gray", vmin=0, vmax=1, origin="upper")
    ax.set_title("Pattern (512×512 px)")
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")


    ax = axes[1]
    ax.axis("off")
    lines_text = [
        ("pattern_type",       metadata["pattern_type"]),
        ("spacing_mm",         f"{metadata['spacing_mm']} mm"),
        ("spacing_px",         f"{metadata['spacing_px']:.1f} px"),
        ("spacing_deg",        f"{metadata['spacing_deg']:.4f}°"),
        ("feature_size_mm",    f"{metadata['feature_size_mm']} mm"),
        ("density",            f"{metadata['density']*100:.2f} %"),
        ("spatial_freq_lpmm",  f"{metadata['spatial_freq_lpmm']:.3f} lp/mm"),
        ("feature_count",      str(metadata["feature_count"])),
        ("fov_mm",             f"{metadata['fov_mm']:.2f} mm"),
        ("mm_per_px",          f"{metadata['mm_per_px']:.4f} mm/px"),
        ("working_dist_mm",    f"{metadata['working_dist_mm']} mm"),
        ("regularity_index",   f"{metadata['regularity_index']:.2f}"),
        ("uniqueness_est.",    f"{metadata['uniqueness_estimate']:.4f}"),
    ]
    col_x = [0.02, 0.55]
    for i, (key, val) in enumerate(lines_text):
        y = 0.97 - i * 0.073
        ax.text(col_x[0], y, key,    transform=ax.transAxes, fontsize=9,
                va="top", color="#444444", fontfamily="monospace")
        ax.text(col_x[1], y, val,    transform=ax.transAxes, fontsize=9,
                va="top", color="#000000", fontfamily="monospace")
    ax.set_title("Metadata")

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        pt   = metadata["pattern_type"]
        smm  = metadata["spacing_mm"]
        fname = save_dir / f"m2_{pt}_spacing{smm}mm.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path

    OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"

    print("=" * 60)
    print("M2 Pattern Generator — Self-test")
    print("=" * 60)


    print("\n[1] Equispaced lines  (spacing=2.0 mm, width=0.3 mm)")
    pat_l, meta_l = generate_pattern("lines", spacing_mm=2.0, feature_size_mm=0.3)
    assert pat_l.dtype == np.float32,          "ERROR: dtype"
    assert pat_l.shape == (512, 512),          "ERROR: shape"
    assert pat_l.min() >= 0.0,                 "ERROR: min"
    assert pat_l.max() <= 1.0,                 "ERROR: max"
    assert meta_l["pattern_type"] == "lines",  "ERROR: pattern_type"
    assert meta_l["regularity_index"] == 1.0,  "ERROR: regularity"
    assert meta_l["feature_count"] > 0,        "ERROR: feature_count"
    print(f"  shape          : {pat_l.shape}")
    print(f"  density        : {meta_l['density']*100:.2f} %")
    print(f"  spatial_freq   : {meta_l['spatial_freq_lpmm']:.3f} lp/mm")
    print(f"  feature_count  : {meta_l['feature_count']}")
    print(f"  spacing_px     : {meta_l['spacing_px']:.1f} px")
    print(f"  spacing_deg    : {meta_l['spacing_deg']:.4f} °")
    print(f"  fov_mm         : {meta_l['fov_mm']:.2f} mm")
    print(f"  uniqueness_est : {meta_l['uniqueness_estimate']:.4f}")
    visualise_pattern(pat_l, meta_l, save_dir=OUT)
    print("  ✅ lines passed")


    print("\n[2] Equispaced dots  (spacing=2.0 mm, radius=0.15 mm)")
    pat_d, meta_d = generate_pattern("dots", spacing_mm=2.0, feature_size_mm=0.15)
    assert pat_d.dtype == np.float32,         "ERROR: dtype"
    assert pat_d.shape == (512, 512),         "ERROR: shape"
    assert meta_d["pattern_type"] == "dots",  "ERROR: pattern_type"
    print(f"  shape          : {pat_d.shape}")
    print(f"  density        : {meta_d['density']*100:.2f} %")
    print(f"  spatial_freq   : {meta_d['spatial_freq_lpmm']:.3f} lp/mm")
    print(f"  feature_count  : {meta_d['feature_count']}")
    print(f"  uniqueness_est : {meta_d['uniqueness_estimate']:.4f}")
    visualise_pattern(pat_d, meta_d, save_dir=OUT)
    print("  ✅ dots passed")


    print("\n[3] Spacing sweep (lines): 1.0 ~ 5.0 mm")
    print(f"  {'spacing_mm':>10}  {'density%':>8}  {'freq lp/mm':>10}  "
          f"{'feat_count':>10}  {'uniqueness':>10}")
    for sp in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        _, m = generate_pattern("lines", spacing_mm=sp, feature_size_mm=0.3)
        print(f"  {sp:>10.1f}  {m['density']*100:>8.2f}  "
              f"{m['spatial_freq_lpmm']:>10.3f}  "
              f"{m['feature_count']:>10}  "
              f"{m['uniqueness_estimate']:>10.4f}")
    print("  ✅ sweep passed")


    print("\n[4] Error handling")
    try:
        generate_pattern("lines", spacing_mm=0.5, feature_size_mm=0.6)
        print("  ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"  ✅ ValueError correctly raised: {e}")

    print("\n" + "=" * 60)
    print("M2 ✅ All tests passed")
    print("=" * 60)
