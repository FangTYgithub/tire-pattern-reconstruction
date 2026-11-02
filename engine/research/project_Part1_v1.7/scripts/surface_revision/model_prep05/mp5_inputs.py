# Build phase-specific rasters and bind imaging and decoding inputs.

from __future__ import annotations

import json
import math
import os

import numpy as np

import mp5_common as K

Q_CAP = 11695.0
INPUT_FIELDS = ('binary', 'peak', 'pattern', 'raster_key', 'lit_px', 'centres', 'classes', 'meta_full', 'brightness', 'registered_hash_ok', 'table', 'enc_spec', 'ordinary')


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x.strip() in ('', 'nan', 'None'))


def template_source(cand_id: str) -> dict:
    import rdy_inputs as RI
    rows = RI.adp_rows()
    if cand_id not in rows:
        raise K.Stop('preflight', f'template candidate {cand_id} not registered')
    return dict(rows[cand_id])


def phase_of(row: dict) -> dict:
    if row['family'] == 'point':
        return dict(u=float(row['phase_u']), v=float(row['phase_v']))
    return dict(phi=float(row['phase_frac']))


def source_with_phase(cand_id: str, family: str, phase: dict) -> dict:
    src = template_source(cand_id)
    if str(src.get('section')) != 'S5':
        raise K.Stop('preflight', f'{cand_id}: template is not an S5 candidate')
    same = ((float(src['phase_u']), float(src['phase_v'])) == (float(phase['u']), float(phase['v']))) if family == 'point' else (float(src['phase_frac']) == float(phase['phi']))
    if not same:
        src['raster_sha256_registered'] = ''
    if family == 'point':
        src['phase_u'], src['phase_v'] = float(phase['u']), float(phase['v'])
    else:
        src['phase_frac'] = float(phase['phi'])
    return src


# Rasterize the supplied candidate and calculate its effective peak and integral.
def build_inputs(src: dict) -> dict:

    import adp_patterns as PT
    import rdy_common as RC
    import rdy_inputs as RI
    b = PT.build(src)
    binary = (np.asarray(b['binary']) > 0).astype(np.float32)
    peak = float(b['peak'])
    out = dict(binary=binary, peak=peak, pattern=(binary * np.float32(peak)).astype(np.float32), raster_key=RI.raster_key(binary), lit_px=float(binary.sum(dtype=np.float64)),
               centres=b['centres'], classes=b['classes'], meta_full=b['meta_full'], brightness=b['brightness'], registered_hash_ok=b['registered_hash_ok'], table=None, enc_spec=None, ordinary=None)
    if peak > 1.0 + 1e-9:
        raise K.Stop('technical', f'peak {peak} > 1 (budget not realisable)')
    req = RC.Q / out['lit_px']
    if abs(req - peak) > 1e-9 * max(1.0, req):
        raise K.Stop('technical', f'peak {peak!r} != Q / lit {req!r}')
    return out


_CACHE: dict = {}


def inputs_for(cand_id: str, family: str, phase: dict) -> dict:
    key = K.digest(dict(cand=cand_id, family=family, phase=phase))
    if key not in _CACHE:
        _CACHE[key] = build_inputs(source_with_phase(cand_id, family, phase))
    return _CACHE[key]


def input_key(cand_id: str, family: str, phase: dict, inp: dict) -> str:
    return K.digest(dict(template=cand_id, family=family, phase=phase, raster_key=inp['raster_key'], pattern=K.content_sha(np.asarray(inp['pattern'], np.float32)),
                         peak_f32=float(np.float32(inp['peak']))))


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        A, B = np.asarray(a), np.asarray(b)
        return A.shape == B.shape and A.dtype == B.dtype and bool(np.array_equal(A, B))
    return K.canon(K.clean(a)) == K.canon(K.clean(b))


def equivalence_16() -> dict:

    import mp3_chain as CH3
    M = K.read_csv(K.MP3_OUT / 'manifest.csv'); N = M[M.record_type == 'new'].drop_duplicates(['combo', 'phase_label'])
    q = CH3.realq(); res = {}
    for r in N.to_dict('records'):
        src = template_source(r['cand_id'])
        fam = r['family']
        ph = dict(u=float(src['phase_u']), v=float(src['phase_v'])) if fam == 'point' else dict(phi=float(src['phase_frac']))
        a = build_inputs(source_with_phase(r['cand_id'], fam, ph)); b = q.inputs(dict(cand_id_sigma0=r['cand_id'], family=fam))
        res[f"{r['combo']}|{r['phase_label']}"] = {f: _same(a.get(f), b.get(f)) for f in INPUT_FIELDS} | dict(raster_key=a['raster_key'], mp3_raster_key=r['raster_key'])
    ok = all(all(v for k, v in d.items() if k in INPUT_FIELDS) and d['raster_key'] == d['mp3_raster_key'] for d in res.values())
    return dict(ok=ok, n=len(res), cases=res)


_Q: dict = {}


def adapter():

    if 'q' in _Q:
        return _Q['q']
    import s5t_platform as PF

    class AdapterQ(PF.RealQ):
        name = 'adapter'

        def __init__(self):
            super().__init__()
            self._reg = {}

        def register(self, key: str, inp: dict) -> None:
            if key in self._reg and self._reg[key]['raster_key'] != inp['raster_key']:
                raise K.Stop('technical', f'input key {key} re-registered with another raster')
            self._reg[key] = inp

        def inputs(self, row) -> dict:
            if 'cand_id_sigma0' in row or 'cand_id' in row:
                raise K.Stop('refused', 'adapter: candidate-id input lookup refused (new-phase inputs only via the registered input key)')
            k = row.get('input_key')
            if k not in self._reg:
                raise K.Stop('technical', f'adapter: unregistered input key {k!r}')
            return self._reg[k]

        def render(self, row):
            raise K.Stop('refused', 'adapter: imaging goes through mp3_forward.render with the frozen forward spec, not RealQ.render')

    q = AdapterQ()
    sb = K.P0.sandbox
    for k, sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
        os.environ[k] = str(sb / sub)
    os.environ['LINE01RB_RENDER_CAP'] = '0'
    _Q['q'] = q
    return q


_MESH: dict = {}


def mesh_sha(sid: str) -> str:
    import rdy_platform as RP
    if sid not in _MESH:
        S = RP.s2(); _MESH[sid] = S.mesh_sha(S.build_mesh(sid))
    return _MESH[sid]


# Rebuild the raster and verify its resource and forward-model identity before execution.
def prepare(row: dict) -> dict:

    import m3_virtual_camera as M3
    import mp3_forward as F
    import rdy_platform as RP
    S = RP.s2(); fam = row['family']; ph = phase_of(row)
    inp = inputs_for(row['template_cand_id'], fam, ph)
    pat = np.asarray(inp['pattern'], np.float32)
    chk = dict(raster=inp['raster_key'] == row['raster_key'], lit_px=float(inp['lit_px']) == float(row['lit_px']), peak_f32=float(np.float32(inp['peak'])) == float(row['peak_f32']),
               integral=abs(float(pat.sum(dtype=np.float64)) - float(row['integral'])) <= 1e-6, peak_rule=abs(float(inp['peak']) - min(1.0, Q_CAP / float(inp['lit_px']))) <= 1e-9,
               integral_cap=float(pat.sum(dtype=np.float64)) <= Q_CAP * (1.0 + 2.0 ** -23))
    if not all(chk.values()):
        raise K.Stop('preflight', f"{row['case_key']}: adapter input differs from the frozen manifest {chk}")
    seed = None if float(row['sigma']) == 0.0 else K.K3.finite_int(row['seed'], f"{row['case_key']} seed")
    if float(row['sigma']) == 0.0 and not _is_missing(row.get('seed')):
        raise K.Stop('preflight', f"{row['case_key']}: sigma 0 with a seed")
    ns = F.noise_spec(float(row['sigma']), seed, 'compat'); base = dict(S.m3_cfg(float(row['sigma']), int(ns['m3_seed'])))
    field = json.loads(row['field_json'])
    if F.field_id(field) != row['field_id']:
        raise K.Stop('preflight', f"{row['case_key']}: field id differs")
    spec = F.forward_spec(base, float(row['psf_true_px']), field, ns)
    fwd = dict(sid=row['platform_surface'], mesh_sha16=mesh_sha(row['platform_surface']), raster_key=inp['raster_key'], pattern=K.content_sha(pat), peak_f32=float(np.float32(inp['peak'])),
               integral=float(pat.sum(dtype=np.float64)), spec=spec, platform=S.VERSION, m3_file=K.sha(M3.__file__), adapter_file=K.sha(K.SR / 'model_prep03' / 'mp3_forward.py'))
    fk = K.digest(fwd)
    if fk != row['forward_key']:
        raise K.Stop('preflight', f"{row['case_key']}: recomputed forward key differs from the frozen manifest")
    ik = input_key(row['template_cand_id'], fam, ph, inp)
    adapter().register(ik, inp)
    radius = float(row['radius_mm']) if fam == 'point' else float('nan')
    return dict(row=row, case_id=row['case_key'], family=fam, sid=row['platform_surface'], surface=row['surface'], pattern=pat, spec=spec, noise=ns, forward_key=fk, forward_record=fwd,
                input_key=ik, input_record=dict(template_cand_id=row['template_cand_id'], phase=ph, raster_key=inp['raster_key'], lit_px=inp['lit_px'], peak=inp['peak'],
                                                integral=float(pat.sum(dtype=np.float64))),
                decoder_row=dict(input_key=ik, radius_mm=radius, family=fam), ctx=dict(meta_full=inp['meta_full'], binary=inp['binary']), resource=chk)
