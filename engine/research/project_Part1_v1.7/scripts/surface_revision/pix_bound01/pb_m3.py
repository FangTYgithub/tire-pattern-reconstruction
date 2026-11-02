# Render with selectable pixel-boundary conventions for paired boundary checks.

from __future__ import annotations

import difflib
import inspect

import numpy as np

import m3_virtual_camera as M3S
from m3_virtual_camera import (SENSOR_DEFAULTS, build_extrinsics, build_intrinsics, grid_to_triangles, heightfield_intersect,
                               moller_trumbore_batch, shade_intensity)

BOUNDARIES = ('legacy', 'symmetric_footprint')
SNAPSHOT_VERSION = 'pb_m3_v1'


def in_footprint(u: np.ndarray, v: np.ndarray, W: int, H: int, boundary: str) -> np.ndarray:

    if boundary == 'legacy':
        return (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if boundary == 'symmetric_footprint':
        return (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    raise ValueError(f'unknown boundary version {boundary!r}')


def project_to_camera(points_world: np.ndarray, K: np.ndarray, ext: dict, image_width: int, image_height: int, boundary: str = 'legacy'):

    pts_cam = (ext["R"] @ points_world.T).T + ext["t"]
    in_front = pts_cam[:, 2] > 1e-6

    safe_z   = np.where(in_front, pts_cam[:, 2], 1.0)
    u = K[0, 0] * pts_cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / safe_z + K[1, 2]

    in_fov = in_front & in_footprint(u, v, image_width, image_height, boundary)
    return np.stack([u, v], axis=1), in_fov


# Render the pattern with the selected pixel-boundary convention.
def synthesise_camera_image(mesh_dict: dict, pattern: np.ndarray, cfg: dict | None = None, boundary: str = 'legacy'):

    if boundary not in BOUNDARIES:
        raise ValueError(f'unknown boundary version {boundary!r}')
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
        uv_px, in_fov  = project_to_camera(pts_h, K, cam_ext, W, H, boundary)

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
    camera_image = np.clip(camera_image, 0.0, 1.0)


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
        "pix_bound_boundary":  boundary,
        "pix_bound_snapshot":  SNAPSHOT_VERSION,
    }
    return camera_image_u8, hit_map, hit_points_3d, out_config


def render(sid: str, pattern, peak, sigma, seed, psf_px, boundary: str):

    import adp_surf02 as SA
    S = SA.s2()
    cfg = S.m3_cfg(float(sigma), int(seed))
    if abs(float(cfg.get('psf_sigma_px', psf_px)) - float(psf_px)) > 1e-12:
        raise RuntimeError(f"platform PSF {cfg.get('psf_sigma_px')} != candidate PSF {psf_px}")
    img, hit, pts3d, m3 = synthesise_camera_image(S.build_mesh(sid), np.asarray(pattern, np.float32), cfg, boundary)
    if m3.get('z_alignment') != 'absolute':
        raise RuntimeError(f"M3 alignment {m3.get('z_alignment')} (expected 'absolute', no Z - min Z)")
    return dict(image=np.asarray(img, np.uint8), hit_map=hit, n_points_3d=int(len(pts3d)),
                meta=dict(z_alignment=m3.get('z_alignment'), n_hits=int(m3['n_hits']), noise_sigma=float(m3['noise_sigma']), mottle_seed=int(m3['mottle_seed']),
                          n_valid_pixels=int(m3['n_valid_pixels']), boundary=boundary, snapshot=SNAPSHOT_VERSION))


def snapshot_diff() -> dict:

    def body(f):
        src = inspect.getsource(f).splitlines()
        out = []; in_doc = False
        for ln in src:
            s = ln.strip()
            if s.startswith('"""') or s.startswith("'''"):
                if not (len(s) > 3 and s.endswith(s[:3]) and len(s) >= 6):
                    in_doc = not in_doc
                continue
            if in_doc or not s or s.startswith('#'):
                continue
            out.append(ln.split('#')[0].rstrip() if '#' in ln and not ('"' in ln.split('#')[0] or "'" in ln.split('#')[0]) else ln.rstrip())
        return out
    res = {}
    for name in ('project_to_camera', 'synthesise_camera_image'):
        a, b = body(getattr(M3S, name)), body(globals()[name])
        d = [l for l in difflib.unified_diff(a, b, lineterm='', n=0) if l[:1] in '+-' and not l.startswith(('+++', '---'))]
        res[name] = d
    return res
