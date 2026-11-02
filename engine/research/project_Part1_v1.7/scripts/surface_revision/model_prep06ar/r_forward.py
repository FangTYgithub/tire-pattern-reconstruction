# Render the reversed-gradient field with the existing imaging sequence.

from __future__ import annotations

import math

import numpy as np

import r_fields as RF
from mp3_forward import K, M3, noise_array


# Trace the pattern and apply the selected signed gain field and image conditions.
def render(mesh: dict, pattern, spec: dict, keep: bool = False) -> dict:
    m3 = M3()
    cfg = {**m3.SENSOR_DEFAULTS, **spec['base_cfg']}
    cfg['rays_per_pixel'] = int(spec['rays_per_pixel'])
    W, H = cfg['image_width_px'], cfg['image_height_px']
    K3 = m3.build_intrinsics(cfg)
    proj_ext, cam_ext = m3.build_extrinsics(cfg)
    vertices, faces = m3.grid_to_triangles(mesh)
    pat = np.asarray(pattern).astype(np.float64)
    if pat.max() > 1.0:
        pat = pat / 255.0
    bright_r, bright_c = np.where(pat > cfg['intensity_thresh'])
    N_bright = bright_r.size
    camera_image = np.zeros((H, W), dtype=np.float64)
    hit_map = np.zeros((H, W), dtype=bool)
    n_hits = 0; g_hits = np.zeros(0); xy_hits = np.zeros((0, 2)); n_fov = 0
    light = dict(fov_shaded_ungained=0.0, fov_shaded_gained=0.0)
    diag = dict(z_alignment=str(mesh.get('z_alignment', 'min')), psf_true_px=float(spec['psf_true_px']), field_id=spec['field_id'], field=spec['field'],
                noise=spec['noise'], rays_per_pixel=int(spec['rays_per_pixel']))
    if N_bright == 0:
        img = np.zeros((H, W), dtype=np.uint8)
        diag.update(n_rays_cast=0, n_hits=0, n_valid_pixels=0, gain_stage='no rays', image_sha=K.content_sha(img))
        return dict(image=img, diag=diag, arrays={})
    pat_vals = pat[bright_r, bright_c]
    n_ss = int(cfg.get('rays_per_pixel', 1) or 1)
    if n_ss > 1:
        offs = (np.arange(n_ss) + 0.5) / n_ss - 0.5
        oy, ox = np.meshgrid(offs, offs, indexing='ij')
        bright_r = (bright_r[:, None] + oy.ravel()[None, :]).ravel()
        bright_c = (bright_c[:, None] + ox.ravel()[None, :]).ravel()
        pat_vals = np.repeat(pat_vals, n_ss * n_ss) / float(n_ss * n_ss)
        N_bright = bright_r.size
    fx, fy = K3[0, 0], K3[1, 1]; cx, cy = K3[0, 2], K3[1, 2]
    x_n = (bright_c - cx) / fx; y_n = (bright_r - cy) / fy
    dirs_cam = np.stack([x_n, y_n, np.ones(N_bright)], axis=1)
    dirs_world = (proj_ext['R_inv'] @ dirs_cam.T).T
    origins = np.tile(proj_ext['center'], (N_bright, 1))
    if str(cfg.get('intersector', 'heightfield')).lower() != 'heightfield':
        raise K.Stop('preflight', 'only the heightfield intersector (M3 default) is registered')
    hit_mask, hit_pts, hit_norms, _ = m3.heightfield_intersect(origins, dirs_world, mesh)
    n_hits = int(hit_mask.sum())
    if n_hits > 0:
        pts_h = hit_pts[hit_mask]; dirs_h = dirs_world[hit_mask]; norms_h = hit_norms[hit_mask]; pvals_h = pat_vals[hit_mask]
        intensity = m3.shade_intensity(norms_h, dirs_h, pvals_h, pts_h, cam_ext['center'], cfg)
        g_hits = RF.gain(spec['field'], pts_h[:, 0], pts_h[:, 1])
        xy_hits = pts_h[:, :2].copy()
        shaded_ungained = intensity
        intensity = intensity * g_hits
        uv_px, in_fov = m3.project_to_camera(pts_h, K3, cam_ext, W, H)
        n_fov = int(in_fov.sum())
        light = dict(fov_shaded_ungained=float(shaded_ungained[in_fov].sum()), fov_shaded_gained=float(intensity[in_fov].sum()))
        if in_fov.any():
            uv = uv_px[in_fov].astype(np.float64)
            I = intensity[in_fov]
            deposit = str(cfg.get('deposit', 'sum')).lower(); splat = str(cfg.get('splat', 'bilinear')).lower()
            if deposit != 'sum' or splat != 'bilinear':
                raise K.Stop('preflight', f'deposit {deposit} / splat {splat}: only the M3 defaults sum / bilinear are registered')
            u, v = uv[:, 0], uv[:, 1]
            u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
            fu, fv = u - u0, v - v0
            for du, dv, w in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
                uu, vv = u0 + du, v0 + dv
                ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (w > 0)
                np.add.at(camera_image, (vv[ok], uu[ok]), I[ok] * w[ok])
            un = np.round(u).astype(int); vn = np.round(v).astype(int)
            ok = (un >= 0) & (un < W) & (vn >= 0) & (vn < H)
            hit_map[vn[ok], un[ok]] = True
    E_pre = camera_image.copy() if keep else None
    h_pre = K.content_sha(camera_image)
    pre_sum = float(camera_image.sum()); pre_max = float(camera_image.max())
    psf = float(spec['psf_true_px'])
    if psf > 0.0 and n_hits > 0:
        from scipy.ndimage import gaussian_filter
        camera_image = gaussian_filter(camera_image, sigma=psf, mode='constant')
    h_post = K.content_sha(camera_image)
    E_post = camera_image.copy() if keep else None
    sat_frac = float((camera_image > 1.0).mean()); post_max = float(camera_image.max())
    post_stats = dict(post_sum=float(camera_image.sum()), post_p99_lit=(float(np.percentile(camera_image[camera_image > 0], 99)) if (camera_image > 0).any() else 0.0),
                      n_px_post_gt_10grey=int((camera_image * 255.0 > 10.0).sum()), n_px_post_gt_50grey=int((camera_image * 255.0 > 50.0).sum()))
    camera_image = np.clip(camera_image, 0.0, 1.0)
    noise = noise_array(spec['noise'], (H, W))
    if noise is not None:
        camera_image = camera_image + noise
    if float(cfg.get('outlier_frac', 0.0) or 0.0) > 0.0:
        raise K.Stop('preflight', 'outlier pixels are off in this round')
    img = np.clip(camera_image * cfg['brightness'], 0, 255).astype(np.uint8)
    diag.update(n_rays_cast=int(N_bright), n_hits=n_hits, n_in_fov=n_fov, n_valid_pixels=int(hit_map.sum()), mesh_triangles=int(faces.shape[0]),
                surface_z_range_mm=[float(vertices[:, 2].min()), float(vertices[:, 2].max())], gain_stage='applied to every hit ray (no bypass)',
                gain_n=int(len(g_hits)), gain_min=(float(g_hits.min()) if len(g_hits) else None), gain_max=(float(g_hits.max()) if len(g_hits) else None),
                gain_mean_at_hits=(float(g_hits.mean()) if len(g_hits) else None), gain_sha=K.content_sha(g_hits), hits_xy_sha=K.content_sha(xy_hits),
                irr_pre_psf_sha=h_pre, irr_post_psf_sha=h_post, irr_post_psf_max=post_max, frac_pre_clip_above_1=sat_frac, noise_sha=(K.content_sha(noise) if noise is not None else None),
                noise_std_preclip_grey=(float(noise.std() * 255.0) if noise is not None else 0.0), image_sha=K.content_sha(img),
                image_frac_0=float((img == 0).mean()), image_frac_255=float((img == 255).mean()), grey_sum=int(img.astype(np.int64).sum()))
    diag.update(light, received_light_ratio=(light['fov_shaded_gained'] / light['fov_shaded_ungained'] if light['fov_shaded_ungained'] > 0 else None),
                gain_hit_weighted_mean=(light['fov_shaded_gained'] / light['fov_shaded_ungained'] if light['fov_shaded_ungained'] > 0 else None),
                irr_pre_psf_sum=pre_sum, irr_pre_psf_max=pre_max, signal_peak_noise_free_grey=255.0 * post_max, image_mean_grey=float(img.mean()), **post_stats)
    arrays = {}
    if keep:
        arrays = dict(irr_pre_psf=E_pre, irr_post_psf=E_post, gains=g_hits, hits_xy=xy_hits)
    return dict(image=img, diag=diag, arrays=arrays)
