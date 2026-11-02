# Build projector rasters from candidate rows with the existing pattern generators.

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache

import numpy as np

import adp_common as C

Q_BUDGET = 11695.0
_SANDBOX = C.OUT / 'legacy_sandbox'
_PATHS = ['src', 'scripts/spotnoise', 'scripts/spotnoise_round3', 'scripts/spotnoise_round5', 'scripts/spotnoise_suppA', 'scripts/spacing_layout', 'scripts/pm014',
          'scripts/pm015', 'scripts/line01', 'scripts/line01r', 'scripts/encoding_prep', 'scripts/spatial_layout', 'scripts/spatial_layout/spat01r', 'scripts/spatial_layout/spat03', '']


def setup_paths():
    os.environ.setdefault('LINE01RB_OUT_DIR', str(_SANDBOX / 'line_out'))
    os.environ.setdefault('LINE01RB_CASES_DIR', str(_SANDBOX / 'line_cases'))
    os.environ.setdefault('LINE01RB_LEDGER_DIR', str(_SANDBOX / 'line_ledger'))
    os.environ.setdefault('LINE01RB_RENDER_CAP', '0')
    for p in _PATHS:
        q = str(C.ROOT / p)
        if q not in sys.path:
            sys.path.append(q)


def _flt(v):
    return float(v) if str(v) not in ('', 'nan', 'None') else None


def _bright(row, binpat_or_integral) -> tuple[float, str]:

    br = str(row['brightness_rule'])
    integral = float(binpat_or_integral if np.isscalar(binpat_or_integral) else np.asarray(binpat_or_integral, np.float64).sum())
    if br.startswith('v='):
        return float(br.split()[0][2:]), 'fixed dot value'
    if br.startswith('peak='):
        return float(br.split()[0][5:]), 'fixed peak'
    if br.startswith('Q='):
        q = float(br.split()[0][2:])
        return q / integral, f'equal integral Q {q:g} / integral {integral:g}'
    raise C.AdapterStop('technical_stop', f'unknown brightness rule {br!r}')


@lru_cache(maxsize=1)
def _spat_layouts():
    return C.load_json(C.ROOT / 'outputs' / 'spatial_layout' / 'spat03' / 'formal_layouts.json')['baselines']


# Dispatch the registered pattern generator and apply the requested peak intensity.
def build(row) -> dict:

    setup_paths()
    gen = str(row['generator']).split(' ')[0]
    meta = {}; meta_full = None
    c = cls = None
    if gen == 'sn_lib.S1_lattice_8x4':
        import sn_lib as SN
        c = SN.centres('S1', int(str(row['phase_label']).replace('sn_phase', '')))
        binpat = np.asarray(SN.binary_pattern(c, float(row['radius_mm'])), np.float32)
        reg_hash, reg_fn = None, None
    elif gen == 'sl_lib.lattice':
        import sl_lib as SLL
        c = SLL.centres(str(row['layout_id']) if row['layout_id'] else f"R{float(row['p_parallel_mm']):g}x{float(row['p_perp_mm']):g}", float(row['phase_u']), float(row['phase_v']))
        binpat = np.asarray(SLL.SL.binary_pattern(c, float(row['radius_mm'])), np.float32)
        reg_hash = row.get('raster_sha256_registered') or None; reg_fn = 'f32'
    elif gen == 'm2.olines':
        import l01rb_common as B
        spec = dict(s_mm=float(row['line_pitch_mm']), w_mm=float(row['line_width_mm']), phi_deg=float(row['orientation_deg'] or 0.0), phase_frac=float(row['phase_frac']))
        binpat, m = B.pattern_for(spec); meta_full = m; meta = dict(n_lines=int(m['olines']['n_lines']), width_px=float(m['olines']['width_px']), spacing_px=float(m['olines']['spacing_px']))
        reg_hash = row.get('raster_sha256_registered') or None
        reg_fn = 'f32' if str(row['section']) == 'S5' else 'line_sha_array'
    elif gen == 'ep_lib.coded_lattice':
        import ep_patterns as EP
        code = str(row['code']).split()
        a = dict(layout=row['layout_id'], arm=code[0], p_par=float(row['p_parallel_mm']), p_perp=float(row['p_perp_mm']), k=int(code[1][1:]), ratio=float(code[2][5:]),
                 rotation_deg=float(code[3][3:]), code_shift=int(code[4][5:]), radii_mm=[float(x) / 2.0 for x in str(row['diameter_mm']).split(';')])
        c, cls, binpat = EP.build(a, float(row['phase_u']), float(row['phase_v']))
        binpat = np.asarray(binpat, np.float32); meta = dict(n_classes=int(len(set(np.asarray(cls).tolist()))))
        reg_hash, reg_fn = None, None
    elif gen == 'lc_pattern':
        import lc_pattern as LC
        spec = dict(pattern_kind=row['code'], s_mm=float(row['line_pitch_mm']), phi_deg=float(row['orientation_deg']), phase_frac=float(row['phase_frac']),
                    w_mm=(_flt(row['line_width_mm']) if row['code'] == 'uncoded' else 1.0))
        binpat, m = LC.pattern_for_spec(spec); binpat = np.asarray(binpat, np.float32); meta_full = m
        reg_hash = row.get('raster_sha256_registered') or None; reg_fn = 'line_sha_array'
    elif gen == 'spat03':
        import sp2_common as S2
        b, pid = str(row['layout_id']).split(':', 1)
        fam = 'dots' if row['family'] == 'point' else 'lines'
        L = next(x for x in _spat_layouts()[b]['layouts'] if str(x['proposal_id']) == pid)
        disp = {int(k): float(v) for k, v in L['disp_px'].items()}
        ph = next(p for p in S2.registered_phases(fam) if p['phase_id'] == row['phase_label'])
        if fam == 'dots':
            import sp_dots_columns as DC
            lay = DC.layout(*S2.DOT_BASELINES[b], ph['u'], ph['v'], disp or None, 'u'); pat = DC.raster(lay['c']); c = lay['c']
        else:
            import sp2_lines_table as LT2
            pat, _m = LT2.table_spec(S2.LINE_BASELINES[b], S2.W_MM, ph['phase_frac'], disp or None)
        binpat = (np.asarray(pat) > 0).astype(np.float32)
        reg_hash = row.get('raster_sha256_registered') or None; reg_fn = 'spat_sha_array'
    else:
        raise C.AdapterStop('technical_stop', f'no raster adapter for generator {row["generator"]!r}')
    peak, rule = _bright(row, binpat)
    if peak > 1.0 + 1e-9:
        raise C.AdapterStop('technical_stop', f'{row["cand_id"]}: required peak {peak:.4f} > 1 (budget not realisable)')
    pattern = (binpat * np.float32(peak)).astype(np.float32)
    h = dict(f32=C_hash_f32(binpat))
    ok = None
    if reg_hash and ';' not in str(reg_hash):
        if reg_fn == 'line_sha_array':
            import l01rb_common as B
            h['line_sha_array'] = B.sha_array(binpat)
        elif reg_fn == 'spat_sha_array':
            import sp2_common as S2
            h['spat_sha_array'] = S2.sha_array(binpat.astype(np.uint8))
        ok = (h.get(reg_fn) == reg_hash)
    return dict(pattern=pattern, binary=binpat, peak=float(peak), brightness=rule, lit_px=float((binpat > 0).sum()), raster_f32_sha256=h['f32'], registered_hash=reg_hash,
                registered_hash_fn=reg_fn, registered_hash_ok=ok, meta=meta, meta_full=meta_full, centres=(None if c is None else np.asarray(c, np.float64)),
                classes=(None if cls is None else np.asarray(cls)))


def C_hash_f32(a) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv(C.CANDIDATES_CSV, keep_default_na=False)
    for cid in sys.argv[1:]:
        r = df[df.cand_id == cid].iloc[0].to_dict()
        b = build(r)
        print(json.dumps({k: v for k, v in b.items() if k not in ('pattern', 'binary')}, ensure_ascii=False))
