# Render images with uniform or x-gradient illumination and controlled noise.

from __future__ import annotations

import math

import numpy as np

import mp2_common as K

NOISE_TAG = 0x4E4F4953
FIELD_TAG = 0x4649454C
OUTLIER_TAG = 0x4F55544C
COMPAT_NOISE_OFFSET = 104729
NOISE_MODES = ('compat', 'independent')
FIELD_KINDS = ('const', 'gradient_x')
FIELD_KEYS = dict(const=('kind', 'value'), gradient_x=('kind', 'mean', 'amp', 'x_half_mm', 'y_half_mm'))
OFF_TERMS = dict(mottle_amp=0.0, specular_k=0.0, outlier_frac=0.0, albedo=1.0, albedo_exposure=1.0)


def M3():
    import m3_virtual_camera as m
    return m


def field_const(value: float) -> dict:
    return validate_field(dict(kind='const', value=float(value)))


def field_gradient_x(mean: float, amp: float, half_mm: float) -> dict:

    return validate_field(dict(kind='gradient_x', mean=float(mean), amp=float(amp), x_half_mm=float(half_mm), y_half_mm=float(half_mm)))


def validate_field(spec: dict) -> dict:
    kind = spec.get('kind')
    if kind not in FIELD_KINDS:
        raise K.Stop('preflight', f'field kind {kind!r} not registered ({FIELD_KINDS})')
    extra = sorted(set(spec) - set(FIELD_KEYS[kind]))
    if extra or any(k not in spec for k in FIELD_KEYS[kind]):
        raise K.Stop('preflight', f'field {kind}: keys {sorted(spec)} differ from the registered {FIELD_KEYS[kind]} (no data-derived entries allowed)')
    vals = {k: spec[k] for k in FIELD_KEYS[kind] if k != 'kind'}
    for k, v in vals.items():
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.floating, np.integer)) or not math.isfinite(float(v)):
            raise K.Stop('preflight', f'field {kind}: {k} = {v!r} is not a finite number')
    if kind == 'const':
        lo = hi = float(spec['value'])
    else:
        if float(spec['x_half_mm']) <= 0 or float(spec['y_half_mm']) <= 0 or float(spec['amp']) < 0:
            raise K.Stop('preflight', f'field {kind}: non-positive domain or negative amplitude')
        lo, hi = float(spec['mean']) - float(spec['amp']), float(spec['mean']) + float(spec['amp'])
    if lo < 0.0 or hi > 1.0:
        raise K.Stop('preflight', f'field {kind}: gain range [{lo}, {hi}] outside [0, 1] (attenuation only; no brightening, no renormalisation)')
    return {k: (spec[k] if k == 'kind' else float(spec[k])) for k in FIELD_KEYS[kind]}


def field_id(spec: dict) -> str:
    return K.digest(dict(field=validate_field(spec), frame='SURF-02 compute frame x = -x_p, y = y_p [mm]', evaluation='analytic at the ray-surface intersection',
                         out_of_domain='clamp (nearest domain value)'))


def gain(spec: dict, x, y) -> np.ndarray:

    s = validate_field(spec)
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.shape != y.shape:
        raise K.Stop('technical', f'gain: x {x.shape} and y {y.shape} differ')
    if s['kind'] == 'const':
        return np.full(x.shape, s['value'], np.float64)
    t = np.clip(x / s['x_half_mm'], -1.0, 1.0)
    return s['mean'] + s['amp'] * t


def field_domain_mean(spec: dict, n: int = 2001) -> dict:

    s = validate_field(spec)
    if s['kind'] == 'const':
        return dict(analytic=s['value'], numeric=s['value'], domain='everywhere', n_grid=0)
    h = s['x_half_mm']; e = (np.arange(n) + 0.5) / n * 2 * h - h
    X, Y = np.meshgrid(e, (np.arange(n) + 0.5) / n * 2 * s['y_half_mm'] - s['y_half_mm'])
    g = gain(s, X, Y)
    return dict(analytic=s['mean'], numeric=float(g.mean()), domain=f"|x| <= {h} mm, |y| <= {s['y_half_mm']} mm (compute frame)", n_grid=n, g_min=float(g.min()), g_max=float(g.max()))


def noise_spec(sigma, seed, mode: str = 'compat') -> dict:
    if isinstance(sigma, (bool, np.bool_)) or sigma is None:
        raise K.Stop('preflight', f'noise sigma {sigma!r} is not a number')
    try:
        s = float(sigma)
    except (TypeError, ValueError):
        raise K.Stop('preflight', f'noise sigma {sigma!r} is not a number') from None
    if not math.isfinite(s) or s < 0:
        raise K.Stop('preflight', f'noise sigma {sigma!r} must be finite and >= 0')
    if mode not in NOISE_MODES:
        raise K.Stop('preflight', f'noise mode {mode!r} not registered ({NOISE_MODES})')
    if s == 0.0:
        empty = seed is None or (isinstance(seed, float) and math.isnan(seed)) or (not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and int(seed) == 0)
        if not empty:
            raise K.Stop('preflight', f'sigma 0 with an unexpected seed {seed!r} (registered: empty / NaN / 0)')
        return dict(sigma=0.0, seed=None, mode=mode, m3_seed=0, stream='none (sigma 0: no draw)')
    sd = K.finite_int(seed, 'noise seed')
    stream = (f'default_rng({sd} + {COMPAT_NOISE_OFFSET}).normal(0, sigma/255)' if mode == 'compat'
              else f'default_rng(SeedSequence({sd}, spawn_key=({NOISE_TAG},))).normal(0, sigma/255)')
    return dict(sigma=s, seed=sd, mode=mode, m3_seed=sd, stream=stream)


def noise_rng(ns: dict):
    if ns['sigma'] == 0.0:
        return None
    if ns['mode'] == 'compat':
        return np.random.default_rng(int(ns['seed']) + COMPAT_NOISE_OFFSET)
    return np.random.default_rng(np.random.SeedSequence(int(ns['seed']), spawn_key=(NOISE_TAG,)))


def noise_array(ns: dict, shape=(512, 512)):

    rng = noise_rng(ns)
    return None if rng is None else rng.normal(0.0, ns['sigma'] / 255.0, shape)


def forward_spec(base_cfg: dict, psf_true_px: float, field: dict, noise: dict, rays_per_pixel: int = 1) -> dict:

    psf = float(psf_true_px)
    if not math.isfinite(psf) or psf < 0:
        raise K.Stop('preflight', f'psf_true_px {psf_true_px!r} must be finite and >= 0')
    n = K.finite_int(rays_per_pixel, 'rays_per_pixel')
    if n != 1:
        raise K.Stop('preflight', f'rays_per_pixel {n}: this round keeps the source cases (1 ray per projector pixel)')
    for k, v in OFF_TERMS.items():
        if float(base_cfg.get(k, v)) != v:
            raise K.Stop('preflight', f'base cfg {k} = {base_cfg.get(k)} (must be {v}: no M3 mottle / specular / outlier / albedo term in this adapter)')
    if 'rays_per_pixel' in base_cfg:
        raise K.Stop('preflight', 'base cfg already sets rays_per_pixel')
    if float(base_cfg['noise_sigma']) != float(noise['sigma']) or int(base_cfg['mottle_seed']) != int(noise['m3_seed']):
        raise K.Stop('preflight', f"base cfg sigma / seed {base_cfg['noise_sigma']} / {base_cfg['mottle_seed']} differ from the noise identity {noise}")
    return dict(base_cfg=dict(base_cfg), psf_true_px=psf, field=validate_field(field), field_id=field_id(field), noise=dict(noise), rays_per_pixel=n,
                boundary='legacy M3 (0 <= u < W before the bilinear splat)', splat='bilinear', deposit='sum', intersector='heightfield',
                order='shade -> gain -> project -> deposit -> PSF -> clip[0,1] -> +noise -> clip(255 x)/uint8 truncation')


# Trace the pattern and apply the gain field, blur, clipping and noise settings.
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
        g_hits = gain(spec['field'], pts_h[:, 0], pts_h[:, 1])
        xy_hits = pts_h[:, :2].copy()
        intensity = intensity * g_hits
        uv_px, in_fov = m3.project_to_camera(pts_h, K3, cam_ext, W, H)
        n_fov = int(in_fov.sum())
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
    psf = float(spec['psf_true_px'])
    if psf > 0.0 and n_hits > 0:
        from scipy.ndimage import gaussian_filter
        camera_image = gaussian_filter(camera_image, sigma=psf, mode='constant')
    h_post = K.content_sha(camera_image)
    E_post = camera_image.copy() if keep else None
    sat_frac = float((camera_image > 1.0).mean()); post_max = float(camera_image.max())
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
    arrays = {}
    if keep:
        arrays = dict(irr_pre_psf=E_pre, irr_post_psf=E_post, gains=g_hits, hits_xy=xy_hits)
    return dict(image=img, diag=diag, arrays=arrays)
