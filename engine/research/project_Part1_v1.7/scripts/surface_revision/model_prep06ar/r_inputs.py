# Build candidate templates and reproducible forward-model records.

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import r_common as K

K5 = K.K5
CANDS = 'outputs/surface_revision/sections_adapter/candidates_s1s5.csv'
NEW_LINES = {'L_w0.75_s8': 0.75, 'L_w1_s8': 1.0, 'L_w1.25_s8': 1.25}
NEW_DOTS = {'D_r1_p8x2.5': (1.0, 2.5, 'S2'), 'D_r1_p8x3': (1.0, 3.0, 'S2'), 'D_r1_p8x8': (1.0, 8.0, 'S2'), 'D_r1.5_p8x3': (1.5, 3.0, 'S5')}
EQUIV_ROWS = {'L_w0.5_s8': ('S1', dict(line_width_mm=0.5)), 'L_w1.5_s8': ('S1', dict(line_width_mm=1.5)), 'D_r1_p8x4': ('S2', dict(radius_mm=1.0, p_perp_mm=4.0)),
              'D_r1_p8x6': ('S2', dict(radius_mm=1.0, p_perp_mm=6.0))}
NUMCOLS = ('line_width_mm', 'line_pitch_mm', 'orientation_deg', 'radius_mm', 'p_parallel_mm', 'p_perp_mm')
_C: dict = {}


def cands() -> pd.DataFrame:
    if 'C' not in _C:
        C = pd.read_csv(K.absp(CANDS), low_memory=False)
        for c in NUMCOLS:
            C[f'_{c}'] = pd.to_numeric(C[c], errors='coerce')
        _C['C'] = C
    return _C['C']


def _row(q: pd.DataFrame) -> dict:
    if len(q) == 0:
        raise K.Stop('preflight', 'template filter matched no registered row')
    q = q.sort_values('cand_id')
    return {k: v for k, v in q.iloc[0].to_dict().items() if not str(k).startswith('_')}


def template_rows(M5: pd.DataFrame) -> dict:
    C = cands(); T = {}
    for cfg in sorted(M5.config.unique()):
        cid = M5[M5.config == cfg].template_cand_id.iloc[0]
        T[cfg] = dict(source='MODEL-PREP-05 frozen S5 template', row=_row(C[C.cand_id == cid]))
    for cfg, w in NEW_LINES.items():
        T[cfg] = dict(source='S1 equal-integral row (orientation 0, pitch 8)', row=_row(C[(C.section == 'S1') & (C.family == 'line') & (C._line_width_mm == w) & (C._line_pitch_mm == 8.0)
                                                                                     & (C._orientation_deg == 0) & (C.regime == 'equal_integral')]))
    for cfg, (rad, pp, sec) in NEW_DOTS.items():
        T[cfg] = dict(source=f'{sec} equal-integral row (R lattice)', row=_row(C[(C.section == sec) & (C.family == 'point') & (C._radius_mm == rad) & (C._p_parallel_mm == 8.0)
                                                                              & (C._p_perp_mm == pp) & (C.regime == 'equal_integral') & (C.arrangement == 'R')]))
    for v in T.values():
        r = v['row']
        if r['family'] == 'point' and str(r.get('arrangement')) != 'R':
            raise K.Stop('preflight', f"{r['cand_id']}: dot template lattice {r.get('arrangement')} (R required)")
        v['cand_id'] = r['cand_id']
    return T


def equivalence_rows() -> dict:
    C = cands(); out = {}
    for cfg, (sec, filt) in EQUIV_ROWS.items():
        fam = 'point' if cfg.startswith('D_') else 'line'
        q = C[(C.section == sec) & (C.regime == 'equal_integral') & (C.family == fam)]
        q = q[(q._line_pitch_mm == 8.0) & (q._orientation_deg == 0)] if fam == 'line' else q[(q._p_parallel_mm == 8.0) & (q.arrangement == 'R')]
        for k, v in filt.items():
            q = q[q[f'_{k}'] == v]
        out[cfg] = _row(q)
    return out


def with_phase(src: dict, family: str, phase: dict) -> dict:
    r = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in src.items()}
    r['raster_sha256_registered'] = ''
    if family == 'point':
        r['phase_u'], r['phase_v'] = float(phase['u']), float(phase['v'])
    else:
        r['phase_frac'] = float(phase['phi'])
    return r


_IN: dict = {}


# Build the phase-specific raster, apply its resource limit and return the geometry metadata.
def inputs(tpl_row: dict, family: str, phase: dict) -> dict:
    import mp5_inputs as IN
    key = K.digest(dict(cand=tpl_row['cand_id'], family=family, phase=phase))
    if key not in _IN:
        _IN[key] = IN.build_inputs(with_phase(tpl_row, family, phase))
    return _IN[key]


def norm_seed(sigma: float, seed):

    s = float(sigma)
    empty = seed is None or (isinstance(seed, float) and math.isnan(seed)) or (isinstance(seed, str) and seed.strip() in ('', 'nan', 'None'))
    if s == 0.0:
        if not empty:
            raise K.Stop('preflight', f'sigma 0 with seed {seed!r}')
        return None
    if empty or isinstance(seed, (bool, np.bool_)):
        raise K.Stop('preflight', f'sigma {s} without a valid seed ({seed!r})')
    f = float(seed)
    if not math.isfinite(f) or f != int(f):
        raise K.Stop('preflight', f'sigma {s}: seed {seed!r} is not a finite integer')
    return int(f)


_MESH: dict = {}


def mesh_sha(sid: str) -> str:
    import rdy_platform as RP
    if sid not in _MESH:
        S = RP.s2(); _MESH[sid] = S.mesh_sha(S.build_mesh(sid))
    return _MESH[sid]


# Assemble the numerical imaging specification and its content identity without rendering.
def forward_record(row: dict, inp: dict) -> tuple[dict, dict, str]:

    import m3_virtual_camera as M3
    import mp3_forward as F
    import rdy_platform as RP
    import r_fields as RF
    S = RP.s2()
    seed = norm_seed(row['sigma'], row.get('seed'))
    ns = F.noise_spec(float(row['sigma']), seed, 'compat'); base = dict(S.m3_cfg(float(row['sigma']), int(ns['m3_seed'])))
    field = json.loads(row['field_json']) if isinstance(row.get('field_json'), str) else row['field']
    spec = RF.forward_spec(base, float(row['psf_true_px']), field, ns)
    route = 'r_forward' if field.get('kind') == RF.KIND_REV else 'mp3_forward'
    adapter = K5.sha(K.SR / 'model_prep06ar' / 'r_forward.py') if route == 'r_forward' else K5.sha(K.SR / 'model_prep03' / 'mp3_forward.py')
    pat = np.asarray(inp['pattern'], np.float32)
    rec = dict(sid=row['platform_surface'], mesh_sha16=mesh_sha(row['platform_surface']), raster_key=inp['raster_key'], pattern=K5.content_sha(pat), peak_f32=float(np.float32(inp['peak'])),
               integral=float(pat.sum(dtype=np.float64)), spec=spec, platform=S.VERSION, m3_file=K5.sha(M3.__file__), adapter_file=adapter)
    return spec, rec, route


# Select DF or the calibrated P4 settings for the candidate geometry.
def decoder_binding(family: str, tpl_row: dict, inp: dict) -> dict:
    import r_common as _K
    if family == 'point':
        return dict(method='DF', path="s3c_dev.dot_decode(img, binary, 1, [radius], 'DF', 'binding') + s3c_dev.dot_dec(D, 'single')", radius_mm=float(tpl_row['radius_mm']))
    w_px = float(inp['meta_full']['olines']['width_px'])
    return dict(method='P4', path='m4_lines.run_olines(img, olines, s2.line_cfg(peak), pattern, **P4_C3_cal) via adp_surf02 / rdy_decode', width_px=w_px,
                steger_sigma_px=max(1.0, w_px / (2.0 * math.sqrt(3.0))))
