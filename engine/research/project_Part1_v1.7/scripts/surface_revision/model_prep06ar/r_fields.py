# Define and validate illumination fields, including a reversed y gradient.

from __future__ import annotations

import math

import numpy as np

import r_common as K

KIND_REV = 'gradient_y_reversed'
KEYS_REV = ('kind', 'mean', 'amp', 'x_half_mm', 'y_half_mm')


def _F():
    import mp3_forward as F
    return F


def validate(spec: dict) -> dict:
    if spec.get('kind') != KIND_REV:
        return _F().validate_field(spec)
    extra = sorted(set(spec) - set(KEYS_REV))
    if extra or any(k not in spec for k in KEYS_REV):
        raise K.Stop('preflight', f'field {KIND_REV}: keys {sorted(spec)} differ from {KEYS_REV}')
    for k in KEYS_REV[1:]:
        v = spec[k]
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.floating, np.integer)) or not math.isfinite(float(v)):
            raise K.Stop('preflight', f'field {KIND_REV}: {k} = {v!r} is not a finite number')
    if float(spec['x_half_mm']) <= 0 or float(spec['y_half_mm']) <= 0 or float(spec['amp']) < 0:
        raise K.Stop('preflight', f'field {KIND_REV}: non-positive domain or negative amplitude')
    lo, hi = float(spec['mean']) - float(spec['amp']), float(spec['mean']) + float(spec['amp'])
    if lo < 0.0 or hi > 1.0:
        raise K.Stop('preflight', f'field {KIND_REV}: gain range [{lo}, {hi}] outside [0, 1]')
    return {k: (spec[k] if k == 'kind' else float(spec[k])) for k in KEYS_REV}


def reversed_y(mean: float, amp: float, half_mm: float) -> dict:
    return validate(dict(kind=KIND_REV, mean=float(mean), amp=float(amp), x_half_mm=float(half_mm), y_half_mm=float(half_mm)))


def gain(spec: dict, x, y) -> np.ndarray:
    if spec.get('kind') != KIND_REV:
        return _F().gain(spec, x, y)
    s = validate(spec)
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.shape != y.shape:
        raise K.Stop('technical', f'gain: x {x.shape} and y {y.shape} differ')
    return s['mean'] - s['amp'] * np.clip(y / s['y_half_mm'], -1.0, 1.0)


def field_id(spec: dict) -> str:
    if spec.get('kind') != KIND_REV:
        return _F().field_id(spec)
    return K5digest(dict(field=validate(spec), frame='SURF-02 compute frame x = -x_p, y = y_p [mm]', evaluation='analytic at the ray-surface intersection',
                         out_of_domain='clamp (nearest domain value)', definition='g_rev(x, y) = g_gradient_y(x, -y) on the same symmetric domain'))


def K5digest(o) -> str:
    return K.K5.digest(o)


def forward_spec(base_cfg: dict, psf_true_px: float, field: dict, noise: dict) -> dict:
    F = _F()
    if field.get('kind') != KIND_REV:
        return F.forward_spec(base_cfg, psf_true_px, field, noise)
    psf = float(psf_true_px)
    if not math.isfinite(psf) or psf < 0:
        raise K.Stop('preflight', f'psf_true_px {psf_true_px!r} must be finite and >= 0')
    for k, v in F.OFF_TERMS.items():
        if float(base_cfg.get(k, v)) != v:
            raise K.Stop('preflight', f'base cfg {k} = {base_cfg.get(k)} (must be {v})')
    if 'rays_per_pixel' in base_cfg:
        raise K.Stop('preflight', 'base cfg already sets rays_per_pixel')
    if float(base_cfg['noise_sigma']) != float(noise['sigma']) or int(base_cfg['mottle_seed']) != int(noise['m3_seed']):
        raise K.Stop('preflight', f"base cfg sigma / seed {base_cfg['noise_sigma']} / {base_cfg['mottle_seed']} differ from the noise identity {noise}")
    return dict(base_cfg=dict(base_cfg), psf_true_px=psf, field=validate(field), field_id=field_id(field), noise=dict(noise), rays_per_pixel=1,
                boundary='legacy M3 (0 <= u < W before the bilinear splat)', splat='bilinear', deposit='sum', intersector='heightfield',
                order='shade -> gain -> project -> deposit -> PSF -> clip[0,1] -> +noise -> clip(255 x)/uint8 truncation')


def tests() -> dict:

    F = _F(); out = {}
    rng = np.random.default_rng(20261021)
    x = rng.uniform(-80, 80, 20000); y = rng.uniform(-80, 80, 20000)
    for spec in (F.field_const(1.0), F.field_const(0.75), F.field_const(0.5), F.field_gradient('x', 0.75, 0.25, 46.08), F.field_gradient('y', 0.75, 0.25, 46.08)):
        out[f"old_{spec['kind']}_{spec.get('value', spec.get('mean'))}"] = dict(gain=bool(np.array_equal(gain(spec, x, y), F.gain(spec, x, y))), field_id=field_id(spec) == F.field_id(spec),
                                                                             validate=validate(spec) == F.validate_field(spec))
    g0 = F.field_gradient('y', 0.75, 0.25, 46.08); r = reversed_y(0.75, 0.25, 46.08)
    half = 46.08
    out['domain_symmetric'] = bool(g0['y_half_mm'] == half and -half == -g0['y_half_mm'])
    out['mirror'] = bool(np.array_equal(gain(r, x, y), F.gain(g0, x, -y)))
    out['end_points'] = dict(y_minus=float(gain(r, np.zeros(1), np.array([-half]))[0]), y_plus=float(gain(r, np.zeros(1), np.array([half]))[0]),
                             y0=float(gain(r, np.zeros(1), np.zeros(1))[0]), beyond=float(gain(r, np.zeros(1), np.array([3 * half]))[0]))
    yy = np.linspace(-half, half, 101); gg = gain(r, np.zeros_like(yy), yy)
    out['monotone_decreasing_in_y'] = bool(np.all(np.diff(gg) < 0))
    out['range'] = [float(gg.min()), float(gg.max())]
    out['field_id_distinct'] = field_id(r) != F.field_id(g0)
    ref = {}
    for name, bad in (('extra_key', dict(r, scale=2.0)), ('missing_key', {k: v for k, v in r.items() if k != 'amp'}), ('negative_amp', dict(r, amp=-0.25)),
                      ('range', dict(r, mean=0.9)), ('nan', dict(r, mean=float('nan'))), ('bool', dict(r, amp=True))):
        try:
            validate(bad); ref[name] = 'NOT REFUSED'
        except K.Stop:
            ref[name] = 'refused'
    out['refusals'] = ref
    ok = all(all(v.values()) for k, v in out.items() if k.startswith('old_')) and out['mirror'] and out['domain_symmetric'] and out['monotone_decreasing_in_y']
    ok = ok and out['end_points'] == dict(y_minus=1.0, y_plus=0.5, y0=0.75, beyond=0.5) and out['range'] == [0.5, 1.0] and out['field_id_distinct'] and all(v == 'refused' for v in ref.values())
    out['ok'] = bool(ok)
    return out
