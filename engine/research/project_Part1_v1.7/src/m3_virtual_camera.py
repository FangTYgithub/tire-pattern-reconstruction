# Project patterns onto a surface and synthesize camera images with shading, blur and noise. Callers
# supply the effective geometry and imaging configuration.

from __future__ import annotations
from pathlib import Path

import numpy as np


SENSOR_DEFAULTS: dict = {

    "image_width_px":    512,  # Camera image width [px]; nominal study setting: 512.
    "image_height_px":   512,  # Camera image height [px]; nominal study setting: 512.


    "focal_length_mm":   25.0,  # Equivalent pinhole focal length [mm].
    "pixel_pitch_mm":    0.030,  # Pixel pitch [mm/px].


    "baseline_mm":       50.0,  # Projector-camera baseline [mm]; nominal study setting: 50.
    "working_dist_mm":   150.0,  # Nominal working distance [mm]; study setting: 150.


    "cam_rot_deg":       (0.0, 0.0, 0.0),  # Camera-frame rotation offsets (x, y, z) [deg].
    "cam_trans_mm":      (0.0, 0.0, 0.0),  # Camera-frame translation offsets (x, y, z) [mm].


    "noise_sigma":       0.0,  # Noise standard deviation [8-bit gray levels]; study levels: 0, 20, 40, 60, 80, 100.
    "brightness":        255.0,  # Output gray-value scale; nominal 8-bit maximum: 255.


    "ray_epsilon":       1e-7,  # Ray-triangle numerical tolerance; not an experimental noise parameter.
    "batch_size":        256,  # Number of rays processed per triangle-intersection batch.
    "intensity_thresh":  0.01,  # Normalized projector intensity cutoff for tracing rays.


    "intersector":       "heightfield",  # Ray-surface solver; study cases use heightfield.


    "splat":             "bilinear",  # Pixel deposition kernel; study cases use bilinear.


    "albedo":            1.0,  # Diffuse reflectance multiplier; nominal study setting: 1.
    "albedo_exposure":   1.0,  # Exposure normalization for the diffuse reflectance multiplier.
    "specular_k":        0.0,  # Specular contribution; zero in the main study configuration.
    "specular_n":        20.0,  # Specular-lobe exponent; used only when specular_k is nonzero.
    "mottle_amp":        0.0,  # Reflectance-texture amplitude; zero in the main study configuration.
    "mottle_scale_mm":   15.0,  # Reflectance-texture spatial scale [mm].
    "mottle_seed":       0,  # Random seed used by the configured image-noise and texture path.
    "outlier_frac":      0.0,  # Fraction of injected impulse outliers; zero in the main study configuration.


    "deposit":           "sum",  # Combine coincident ray contributions; sum conserves deposited intensity.
    "psf_sigma_px":      1.0,  # Gaussian PSF standard deviation [px]; Chapter 4 levels: 1, 1.5, 2, 2.5, 3.


    "rays_per_pixel":    1,  # Samples per pixel axis: 1 normally, 3 in the 3x3 sub-ray checks.
}


# Construct the pinhole intrinsic matrix from focal length, pixel pitch and image size.
def build_intrinsics(cfg: dict) -> np.ndarray:

    f_px = cfg["focal_length_mm"] / cfg["pixel_pitch_mm"]
    cx   = (cfg["image_width_px"]  - 1) / 2.0
    cy   = (cfg["image_height_px"] - 1) / 2.0
    return np.array([
        [f_px,    0,  cx],
        [   0, f_px,  cy],
        [   0,    0,   1],
    ], dtype=np.float64)


def _look_at_rotation(position: np.ndarray,
                      target:   np.ndarray) -> np.ndarray:

    z_cam = target - position
    z_cam /= np.linalg.norm(z_cam)

    up = np.array([1.0, 0.0, 0.0])
    x_cam = np.cross(z_cam, up)
    if np.linalg.norm(x_cam) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        x_cam = np.cross(z_cam, up)
    x_cam /= np.linalg.norm(x_cam)

    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)

    return np.stack([x_cam, y_cam, z_cam], axis=0)


# Place the projector and camera using the specified baseline and working distance.
def build_extrinsics(cfg: dict) -> tuple[dict, dict]:

    half_b = cfg["baseline_mm"] / 2.0
    wd     = cfg["working_dist_mm"]
    origin = np.zeros(3)

    def _ext(center: np.ndarray) -> dict:
        R = _look_at_rotation(center, origin)
        return {"center": center.copy(),
                "R": R, "t": -R @ center, "R_inv": R.T}

    cam = _ext(np.array([0.0, half_b, -wd]))
    rot = tuple(float(a) for a in cfg.get("cam_rot_deg", (0.0, 0.0, 0.0)))
    trn = tuple(float(a) for a in cfg.get("cam_trans_mm", (0.0, 0.0, 0.0)))
    if any(rot) or any(trn):
        cam = perturb_pose(cam, rot, trn)
    return (_ext(np.array([0.0, -half_b, -wd])),
            cam)


def _sensor_frame_rotation(rot_deg: tuple) -> np.ndarray:

    rx, ry, rz = np.radians(np.asarray(rot_deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# Apply sensor-frame rotations in degrees and translations in millimetres.
def perturb_pose(ext: dict, rot_deg: tuple, trans_mm: tuple) -> dict:

    R_new  = _sensor_frame_rotation(rot_deg) @ ext["R"]
    center = ext["center"] + ext["R_inv"] @ np.asarray(trans_mm, dtype=np.float64)
    return {"center": center, "R": R_new, "t": -R_new @ center, "R_inv": R_new.T}


Z_ALIGNMENT_MODES = ("min", "absolute")


def _aligned_z(mesh_dict: dict, Z: np.ndarray) -> np.ndarray:

    mode = str(mesh_dict.get("z_alignment", "min")) if hasattr(mesh_dict, "get") else "min"
    if mode == "min":
        return Z - Z.min()
    if mode == "absolute":
        return Z
    raise ValueError(f"unknown z_alignment '{mode}' (use one of {Z_ALIGNMENT_MODES})")


# Convert the height-field grid into vertices and triangular faces.
def grid_to_triangles(
    mesh_dict: dict,
) -> tuple[np.ndarray, np.ndarray]:

    X = mesh_dict["X"].astype(np.float64)
    Y = mesh_dict["Y"].astype(np.float64)
    Z = mesh_dict["Z"].astype(np.float64)


    Z = _aligned_z(mesh_dict, Z)

    Ny, Nx = Z.shape
    vertices = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    rows, cols = np.mgrid[0:Ny-1, 0:Nx-1]
    r, c = rows.ravel(), cols.ravel()

    v00 =  r      * Nx + c
    v10 = (r + 1) * Nx + c
    v01 =  r      * Nx + (c + 1)
    v11 = (r + 1) * Nx + (c + 1)

    tri0  = np.stack([v00, v10, v01], axis=1)
    tri1  = np.stack([v10, v11, v01], axis=1)
    faces = np.concatenate([tri0, tri1], axis=0).astype(np.int32)

    return vertices, faces


# Intersect batches of rays with the triangular mesh and retain the nearest valid hit.
def moller_trumbore_batch(
    ray_origins:    np.ndarray,
    ray_directions: np.ndarray,
    vertices:       np.ndarray,
    faces:          np.ndarray,
    epsilon:        float = 1e-7,
    batch_size:     int   = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    N = ray_origins.shape[0]
    hit_mask    = np.zeros(N, dtype=bool)
    hit_points  = np.zeros((N, 3), dtype=np.float64)
    hit_normals = np.zeros((N, 3), dtype=np.float64)
    hit_faces   = np.full(N, -1,   dtype=np.int32)
    hit_t       = np.full(N, np.inf)


    v0 = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - v0
    e2 = vertices[faces[:, 2]] - v0

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        orig = ray_origins[start:end]
        dire = ray_directions[start:end]
        B    = orig.shape[0]


        d = dire[:, np.newaxis, :]
        e2e = e2[np.newaxis, :, :]
        e1e = e1[np.newaxis, :, :]
        v0e = v0[np.newaxis, :, :]

        h = np.cross(d, e2e)
        a = np.sum(e1e * h, axis=2)

        valid = np.abs(a) > epsilon
        f_inv = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)

        s = orig[:, np.newaxis, :] - v0e
        u = f_inv * np.sum(s * h, axis=2)
        valid &= (u >= 0.0) & (u <= 1.0)

        q = np.cross(s, e1e)
        v = f_inv * np.sum(d * q, axis=2)
        valid &= (v >= 0.0) & (u + v <= 1.0)

        t_val = f_inv * np.sum(e2e * q, axis=2)
        valid &= (t_val > epsilon)

        t_inf    = np.where(valid, t_val, np.inf)
        nearest  = np.argmin(t_inf, axis=1)
        t_near   = t_inf[np.arange(B), nearest]

        ray_hit = t_near < np.inf
        g_idx   = np.arange(start, end)
        update  = ray_hit & (t_near < hit_t[g_idx])

        if update.any():
            lu = np.where(update)[0]
            gi = g_idx[update]
            fi = nearest[lu]
            tv = t_near[lu]

            hit_mask[gi]   = True
            hit_t[gi]      = tv
            hit_points[gi] = orig[lu] + dire[lu] * tv[:, np.newaxis]
            hit_faces[gi]  = fi


    valid_h = hit_mask & (hit_faces >= 0)
    if valid_h.any():
        fi  = hit_faces[valid_h]
        n   = np.cross(
            vertices[faces[fi, 1]] - vertices[faces[fi, 0]],
            vertices[faces[fi, 2]] - vertices[faces[fi, 0]],
        )
        nl  = np.linalg.norm(n, axis=1, keepdims=True)
        nl  = np.where(nl < 1e-12, 1.0, nl)
        hit_normals[valid_h] = n / nl

    return hit_mask, hit_points, hit_normals, hit_faces


# Intersect rays with the interpolated height field using a bounded iterative search.
def heightfield_intersect(
    ray_origins:    np.ndarray,
    ray_directions: np.ndarray,
    mesh_dict:      dict,
    n_iter:         int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    X = mesh_dict["X"].astype(np.float64)
    Y = mesh_dict["Y"].astype(np.float64)
    Z = mesh_dict["Z"].astype(np.float64)
    Z = _aligned_z(mesh_dict, Z)
    xv, yv = X[0, :].copy(), Y[:, 0].copy()
    if xv[0] > xv[-1]:
        xv, Z = xv[::-1], Z[:, ::-1]
    if yv[0] > yv[-1]:
        yv, Z = yv[::-1], Z[::-1, :]
    Nx, Ny = len(xv), len(yv)
    x0, x1, y0, y1 = xv[0], xv[-1], yv[0], yv[-1]
    z_lo, z_hi = float(Z.min()) - 1.0, float(Z.max()) + 1.0

    def _cell(x, y):
        xc = np.clip(x, x0, x1)
        yc = np.clip(y, y0, y1)
        j = np.clip(np.searchsorted(xv, xc, side="right") - 1, 0, Nx - 2)
        i = np.clip(np.searchsorted(yv, yc, side="right") - 1, 0, Ny - 2)
        fx = (xc - xv[j]) / (xv[j + 1] - xv[j])
        fy = (yc - yv[i]) / (yv[i + 1] - yv[i])
        return i, j, fx, fy

    def _f(x, y):
        i, j, fx, fy = _cell(x, y)
        return ((1.0 - fy) * ((1.0 - fx) * Z[i, j] + fx * Z[i, j + 1])
                + fy * ((1.0 - fx) * Z[i + 1, j] + fx * Z[i + 1, j + 1]))

    o = np.asarray(ray_origins, dtype=np.float64)
    d = np.asarray(ray_directions, dtype=np.float64)
    N = o.shape[0]
    ok = d[:, 2] > 1e-12
    dz = np.where(ok, d[:, 2], 1.0)
    t_lo = (z_lo - o[:, 2]) / dz
    t_hi = (z_hi - o[:, 2]) / dz
    for _ in range(int(n_iter)):
        t_mid = 0.5 * (t_lo + t_hi)
        p = o + t_mid[:, None] * d
        above = (p[:, 2] - _f(p[:, 0], p[:, 1])) > 0.0
        t_hi = np.where(above, t_mid, t_hi)
        t_lo = np.where(above, t_lo, t_mid)
    t = 0.5 * (t_lo + t_hi)
    p = o + t[:, None] * d
    hit = (ok & (t > 0.0)
           & (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1))

    i, j, fx, fy = _cell(p[:, 0], p[:, 1])
    dfdx = ((1.0 - fy) * (Z[i, j + 1] - Z[i, j]) + fy * (Z[i + 1, j + 1] - Z[i + 1, j])) / (xv[j + 1] - xv[j])
    dfdy = ((1.0 - fx) * (Z[i + 1, j] - Z[i, j]) + fx * (Z[i + 1, j + 1] - Z[i, j + 1])) / (yv[i + 1] - yv[i])
    n = np.stack([dfdx, dfdy, -np.ones(N)], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    hit_points  = np.where(hit[:, None], p, 0.0)
    hit_normals = np.where(hit[:, None], n, 0.0)
    hit_cells   = np.where(hit, i * (Nx - 1) + j, -1).astype(np.int32)
    return hit, hit_points, hit_normals, hit_cells


# Project world points onto camera pixels and return the visible-point mask.
def project_to_camera(
    points_world: np.ndarray,
    K:            np.ndarray,
    ext:          dict,
    image_width:  int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:

    pts_cam = (ext["R"] @ points_world.T).T + ext["t"]
    in_front = pts_cam[:, 2] > 1e-6

    safe_z   = np.where(in_front, pts_cam[:, 2], 1.0)
    u = K[0, 0] * pts_cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / safe_z + K[1, 2]

    in_fov = (
        in_front &
        (u >= 0) & (u < image_width) &
        (v >= 0) & (v < image_height)
    )
    return np.stack([u, v], axis=1), in_fov


# Weight projector intensity by the surface incidence cosine.
def lambertian_intensity(
    hit_normals:    np.ndarray,
    ray_directions: np.ndarray,
    pattern_values: np.ndarray,
) -> np.ndarray:

    norms  = np.linalg.norm(ray_directions, axis=1, keepdims=True)
    norms  = np.where(norms < 1e-12, 1.0, norms)
    d_unit = ray_directions / norms
    cos_a  = np.abs(np.einsum('ni,ni->n', hit_normals, -d_unit))
    return pattern_values * cos_a


# Apply diffuse reflectance and the optional specular and mottled components.
def shade_intensity(
    hit_normals:    np.ndarray,
    ray_directions: np.ndarray,
    pattern_values: np.ndarray,
    hit_points:     np.ndarray,   # World coordinates [mm], (N, 3).
    cam_center:     np.ndarray,
    cfg:            dict,
) -> np.ndarray:

    norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
    d_unit = ray_directions / np.where(norms < 1e-12, 1.0, norms)
    n_unit = hit_normals / np.maximum(np.linalg.norm(hit_normals, axis=1, keepdims=True), 1e-12)
    cos_a = np.einsum("ni,ni->n", n_unit, -d_unit)
    # Orient normals toward incident light.
    n_unit = np.where(cos_a[:, None] < 0, -n_unit, n_unit)
    cos_a = np.abs(cos_a)

    albedo = float(cfg.get("albedo", 1.0)) / max(float(cfg.get("albedo_exposure", 1.0)), 1e-9)
    amp = float(cfg.get("mottle_amp", 0.0) or 0.0)
    if amp > 0.0 and len(hit_points) > 0:
        from scipy.ndimage import gaussian_filter
        from scipy.interpolate import RegularGridInterpolator
        scale = float(cfg.get("mottle_scale_mm", 15.0))
        rng = np.random.default_rng(int(cfg.get("mottle_seed", 0)))
        x0, x1 = float(hit_points[:, 0].min()) - scale, float(hit_points[:, 0].max()) + scale
        y0, y1 = float(hit_points[:, 1].min()) - scale, float(hit_points[:, 1].max()) + scale
        step = max(scale / 4.0, 0.5)
        xs = np.arange(x0, x1 + step, step); ys = np.arange(y0, y1 + step, step)
        field = gaussian_filter(rng.normal(size=(len(ys), len(xs))), sigma=scale / step, mode="wrap")
        field /= max(float(np.abs(field).max()), 1e-12)
        f = RegularGridInterpolator((ys, xs), field, bounds_error=False, fill_value=0.0)
        local = albedo * (1.0 - amp * (0.5 + 0.5 * f(np.column_stack([hit_points[:, 1], hit_points[:, 0]]))))
    else:
        local = np.full(len(cos_a), albedo)

    intensity = pattern_values * local * cos_a
    k_s = float(cfg.get("specular_k", 0.0) or 0.0)
    if k_s > 0.0 and len(hit_points) > 0:
        r = d_unit - 2.0 * np.einsum("ni,ni->n", d_unit, n_unit)[:, None] * n_unit
        v = np.asarray(cam_center, dtype=np.float64)[None, :] - hit_points
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        spec = np.clip(np.einsum("ni,ni->n", r, v), 0.0, 1.0) ** float(cfg.get("specular_n", 20.0))
        intensity = intensity + pattern_values * k_s * spec
    return intensity


# Trace projector rays to the surface, deposit their signal in the camera and apply blur and noise.
# Return the image and rendering diagnostics.
def synthesise_camera_image(
    mesh_dict: dict,
    pattern:   np.ndarray,
    cfg:       dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:

    cfg = {**SENSOR_DEFAULTS, **(cfg or {})}
    W, H = cfg["image_width_px"], cfg["image_height_px"]


    K = build_intrinsics(cfg)
    proj_ext, cam_ext = build_extrinsics(cfg)


    vertices, faces = grid_to_triangles(mesh_dict)


    pat = pattern.astype(np.float64)
    if pat.max() > 1.0:
        pat = pat / 255.0


    bright_r, bright_c = np.where(pat > cfg["intensity_thresh"])
    N_bright = bright_r.size


    camera_image  = np.zeros((H, W), dtype=np.float64)
    hit_map       = np.zeros((H, W), dtype=bool)
    hit_points_3d = np.zeros((0, 3), dtype=np.float32)

    if N_bright == 0:
        out_cfg = {**cfg, "n_rays_cast": 0, "n_hits": 0,
                   "n_valid_pixels": 0, "mesh_triangles": int(faces.shape[0]),
                   "z_alignment": str(mesh_dict.get("z_alignment", "min")) if hasattr(mesh_dict, "get") else "min"}
        return np.zeros((H, W), dtype=np.uint8), hit_map, hit_points_3d, out_cfg

    pat_vals = pat[bright_r, bright_c]


    n_ss = int(cfg.get("rays_per_pixel", 1) or 1)
    if n_ss > 1:
        offs = (np.arange(n_ss) + 0.5) / n_ss - 0.5
        oy, ox = np.meshgrid(offs, offs, indexing="ij")
        bright_r = (bright_r[:, None] + oy.ravel()[None, :]).ravel()
        bright_c = (bright_c[:, None] + ox.ravel()[None, :]).ravel()
        pat_vals = np.repeat(pat_vals, n_ss * n_ss) / float(n_ss * n_ss)
        N_bright = bright_r.size


    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_n = (bright_c - cx) / fx
    y_n = (bright_r - cy) / fy

    # Projector frame: (x_n, y_n, 1).
    dirs_cam   = np.stack([x_n, y_n, np.ones(N_bright)], axis=1)
    dirs_world = (proj_ext["R_inv"] @ dirs_cam.T).T
    origins    = np.tile(proj_ext["center"], (N_bright, 1))


    intersector = str(cfg.get("intersector", "heightfield")).lower()
    if intersector == "triangles":
        hit_mask, hit_pts, hit_norms, _ = moller_trumbore_batch(
            origins, dirs_world, vertices, faces,
            epsilon=cfg["ray_epsilon"], batch_size=cfg["batch_size"],
        )
    elif intersector == "heightfield":
        hit_mask, hit_pts, hit_norms, _ = heightfield_intersect(origins, dirs_world, mesh_dict)
    else:
        raise ValueError(f"unknown intersector '{intersector}' (use 'heightfield' or 'triangles')")
    n_hits = int(hit_mask.sum())


    if n_hits > 0:
        pts_h   = hit_pts[hit_mask]
        dirs_h  = dirs_world[hit_mask]
        norms_h = hit_norms[hit_mask]
        pvals_h = pat_vals[hit_mask]

        intensity      = shade_intensity(norms_h, dirs_h, pvals_h, pts_h, cam_ext["center"], cfg)
        uv_px, in_fov  = project_to_camera(pts_h, K, cam_ext, W, H)

        if in_fov.any():
            uv = uv_px[in_fov].astype(np.float64)
            I  = intensity[in_fov]
            deposit = str(cfg.get("deposit", "sum")).lower()
            splat   = str(cfg.get("splat", "bilinear")).lower()
            if deposit == "max" or splat == "nearest":


                u_px = np.round(uv[:, 0]).astype(int)
                v_px = np.round(uv[:, 1]).astype(int)
                inside = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)
                u_px, v_px, Ii = u_px[inside], v_px[inside], I[inside]
                if deposit == "max":
                    np.maximum.at(camera_image, (v_px, u_px), Ii)
                else:
                    np.add.at(camera_image, (v_px, u_px), Ii)
                hit_map[v_px, u_px] = True
            else:


                u, v = uv[:, 0], uv[:, 1]
                u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
                fu, fv = u - u0, v - v0
                for du, dv, w in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)),
                                  (0, 1, (1 - fu) * fv),       (1, 1, fu * fv)):
                    uu, vv = u0 + du, v0 + dv
                    ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (w > 0)
                    np.add.at(camera_image, (vv[ok], uu[ok]), I[ok] * w[ok])
                un = np.round(u).astype(int); vn = np.round(v).astype(int)
                ok = (un >= 0) & (un < W) & (vn >= 0) & (vn < H)
                hit_map[vn[ok], un[ok]] = True

            hit_points_3d = pts_h[in_fov].astype(np.float32)


    psf = float(cfg.get("psf_sigma_px", 0.0) or 0.0)
    if psf > 0.0 and n_hits > 0:
        from scipy.ndimage import gaussian_filter
        camera_image = gaussian_filter(camera_image, sigma=psf, mode="constant")
    camera_image = np.clip(camera_image, 0.0, 1.0)      # Saturation before sensor noise.


    sigma = cfg["noise_sigma"]
    if sigma > 0.0:


        rng_n        = np.random.default_rng(int(cfg.get("mottle_seed", 0)) + 104729)
        noise        = rng_n.normal(0.0, sigma / 255.0, (H, W))
        camera_image = camera_image + noise


    out_frac = float(cfg.get("outlier_frac", 0.0) or 0.0)
    if out_frac > 0.0:
        rng_o = np.random.default_rng(int(cfg.get("mottle_seed", 0)) + 7919)
        n_out = int(round(out_frac * H * W))
        if n_out > 0:
            idx = rng_o.choice(H * W, size=n_out, replace=False)
            vals = rng_o.random(n_out) < 0.5
            flat = camera_image.reshape(-1)
            flat[idx] = np.where(vals, 1.0, 0.0)
            camera_image = flat.reshape(H, W)

    camera_image_u8 = np.clip(
        camera_image * cfg["brightness"], 0, 255
    ).astype(np.uint8)

    out_config = {
        **cfg,
        "K":                   K.tolist(),
        "projector_center_mm": proj_ext["center"].tolist(),
        "camera_center_mm":    cam_ext["center"].tolist(),
        "effective_theta_deg": float(np.degrees(
            2 * np.arctan(cfg["baseline_mm"] / 2 / cfg["working_dist_mm"])
        )),
        "mesh_triangles":      int(faces.shape[0]),
        "intersector":         intersector,
        "z_alignment":         str(mesh_dict.get("z_alignment", "min")) if hasattr(mesh_dict, "get") else "min",
        "n_rays_cast":         int(N_bright),
        "n_hits":              n_hits,
        "n_valid_pixels":      int(hit_map.sum()),
        "deposit":             str(cfg.get("deposit", "sum")),
        "psf_sigma_px":        float(cfg.get("psf_sigma_px", 0.0) or 0.0),
        "surface_z_range_mm":  (float(vertices[:, 2].min()),
                                float(vertices[:, 2].max())),
    }
    return camera_image_u8, hit_map, hit_points_3d, out_config


def plot_camera_image(camera_image: np.ndarray, config: dict, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(camera_image, cmap="gray", vmin=0, vmax=255, origin="upper")
    ax.set_title(
        f"Synthetic camera image\n"
        f"hits={config['n_hits']},  "
        f"valid px={config['n_valid_pixels']},  "
        f"θ={config['effective_theta_deg']:.1f}°"
    )
    ax.set_xlabel("u [px]");  ax.set_ylabel("v [px]")
    return ax


def plot_hit_map(hit_map: np.ndarray, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(hit_map, cmap="hot", origin="upper")
    ax.set_title("Hit map (valid projections)")
    ax.set_xlabel("u [px]");  ax.set_ylabel("v [px]")
    return ax


# Render standalone point and line examples from the selected surface profile.
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    OUT_DIR  = Path("/mnt/user-data/outputs/m3_outputs")
    CSV_PATH = Path("/mnt/user-data/uploads/fig14_curve_processed.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)


    import pandas as pd
    from scipy.interpolate import PchipInterpolator

    df  = pd.read_csv(CSV_PATH).sort_values("y_mm").drop_duplicates("y_mm")
    y   = df["y_mm"].to_numpy()
    z   = df["z_smooth_mm"].to_numpy()
    yg  = np.arange(y.min(), y.max() + 0.5, 1.0)
    zg  = PchipInterpolator(y, z)(yg)
    x1d = np.arange(-100, 101, 2.0)
    X, Y_grid = np.meshgrid(x1d, yg)
    Z_grid    = np.tile(zg[:, None], (1, len(x1d)))

    mesh = {
        "X": X.astype(np.float32),
        "Y": Y_grid.astype(np.float32),
        "Z": Z_grid.astype(np.float32),
    }
    print(f"Surface: Y={Y_grid.min():.0f}~{Y_grid.max():.0f} mm, "
          f"Z={Z_grid.min():.1f}~{Z_grid.max():.1f} mm, "
          f"shape={Z_grid.shape}")


    patterns = {
        "horizontal_lines": _make_hlines(512, spacing=20, width=3),
        "dot_grid":         _make_dots(512, spacing=30, radius=3),
    }

    for name, pattern in patterns.items():
        print(f"\n--- Pattern: {name} ---")
        print(f"  Density: {(pattern > 0).mean()*100:.1f}%")

        cam_img, hit_map, hit_pts, cfg = synthesise_camera_image(
            mesh_dict=mesh, pattern=pattern
        )

        print(f"  Rays cast    : {cfg['n_rays_cast']}")
        print(f"  Surface hits : {cfg['n_hits']}")
        print(f"  Valid pixels : {cfg['n_valid_pixels']}")
        print(f"  Surface Z    : {cfg['surface_z_range_mm']} mm")
        print(f"  Projector    : {cfg['projector_center_mm']} mm")
        print(f"  Camera       : {cfg['camera_center_mm']} mm")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(pattern, cmap="gray", origin="upper")
        axes[0].set_title(f"Pattern: {name}\ndensity={pattern.mean()*100:.0f}%")
        plot_camera_image(cam_img, cfg, ax=axes[1])
        plot_hit_map(hit_map, ax=axes[2])
        fig.suptitle(f"M3 — {name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"m3_{name}.png", dpi=200)
        plt.close(fig)
        print(f"  Saved: m3_{name}.png")

    print("\nM3 complete.")


def _make_hlines(size: int, spacing: int = 20, width: int = 3) -> np.ndarray:
    p = np.zeros((size, size), dtype=np.float32)
    for r in range(0, size, spacing):
        p[r:r+width, :] = 1.0
    return p


def _make_dots(size: int, spacing: int = 30, radius: int = 3) -> np.ndarray:
    p = np.zeros((size, size), dtype=np.float32)
    for r in range(spacing//2, size, spacing):
        for c in range(spacing//2, size, spacing):
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    if dr*dr + dc*dc <= radius*radius:
                        rr, cc = r+dr, c+dc
                        if 0 <= rr < size and 0 <= cc < size:
                            p[rr, cc] = 1.0
    return p


if __name__ == "__main__":
    main()
