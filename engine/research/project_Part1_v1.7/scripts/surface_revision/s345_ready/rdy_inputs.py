# Build and cache raster inputs for registered pattern candidates.

from __future__ import annotations

from functools import lru_cache

import numpy as np

import rdy_common as K


@lru_cache(maxsize=1)
def adp_rows():
    import pandas as pd
    d = pd.read_csv(K.SRC_LIST_ADP, keep_default_na=False, low_memory=False)
    d = d[d.section.isin(['S3', 'S4', 'S5'])]
    return {r['cand_id']: r for r in d.to_dict('records')}


def raster_key(binary) -> str:
    b = np.ascontiguousarray(np.asarray(binary) > 0, dtype=np.uint8)
    return 'R-' + K.sha_bytes(str(b.shape).encode() + b.tobytes(), 20)


def _spat_table(row: dict) -> dict:

    import sp2_common as S2
    import adp_patterns as PT
    b, pid = str(row['layout_id']).split(':', 1)
    fam = 'dots' if row['family'] == 'point' else 'lines'
    L = next(x for x in PT._spat_layouts()[b]['layouts'] if str(x['proposal_id']) == pid)
    disp = {int(k): float(v) for k, v in L['disp_px'].items()}
    ph = next(p for p in S2.registered_phases(fam) if p['phase_id'] == row['phase_label'])
    out = dict(baseline=b, proposal_id=pid, phase_id=row['phase_label'], disp=disp, j=float(L['j']), arms=L.get('arms'), family=fam, phase=ph)
    if fam == 'dots':
        import sp_dots_columns as DC
        lay = DC.layout(*S2.DOT_BASELINES[b], ph['u'], ph['v'], disp or None, 'u'); lay0 = DC.layout(*S2.DOT_BASELINES[b], ph['u'], ph['v'], None, 'u')
        out.update(pat=np.asarray(DC.raster(lay['c'])), centres=np.asarray(lay['c'], np.float64), centres_regular=np.asarray(lay0['c'], np.float64), ix=np.asarray(lay['ix']), spec=None)
    else:
        import sp2_lines_table as LT2
        pat, meta = LT2.table_spec(S2.LINE_BASELINES[b], S2.W_MM, ph['phase_frac'], disp or None)
        out.update(pat=np.asarray(pat), centres=None, spec=meta['olines'], meta=meta)
    return out


def _ordinary_s0(row: dict, T: dict) -> dict:

    import sp2_common as S2
    ph = T['phase']
    if T['family'] == 'dots':
        import sl_lib as SLL
        c = SLL.centres(T['baseline'], ph['u'], ph['v']); pat = np.asarray(SLL.SL.binary_pattern(c, S2.R_MM), np.float32)
        return dict(binary=(pat > 0).astype(np.float32), centres=np.asarray(c, np.float64), meta_full=None)
    import l01rb_common as B
    pat, meta = B.pattern_for(dict(s_mm=S2.LINE_BASELINES[T['baseline']], w_mm=S2.W_MM, phi_deg=0.0, phase_frac=ph['phase_frac']))
    return dict(binary=(np.asarray(pat) > 0).astype(np.float32), centres=None, meta_full=meta)


_CACHE: dict = {}


def build(row: dict) -> dict:

    key = row['cand_id']
    if key in _CACHE:
        return _CACHE[key]
    import adp_patterns as PT
    src = adp_rows()[row['cand_id']]
    b = PT.build(src)
    if b['registered_hash'] and ';' not in str(b['registered_hash']) and b['registered_hash_ok'] is False:
        raise K.Stop('technical', f"{row['cand_id']}: raster does not reproduce its registered hash ({b['registered_hash_fn']})")
    binary = (np.asarray(b['binary']) > 0).astype(np.float32)
    peak = float(b['peak'])
    out = dict(binary=binary, peak=peak, pattern=(binary * np.float32(peak)).astype(np.float32), raster_key=raster_key(binary), lit_px=float(binary.sum(dtype=np.float64)),
               centres=b['centres'], classes=b['classes'], meta_full=b['meta_full'], brightness=b['brightness'], registered_hash_ok=b['registered_hash_ok'], table=None, enc_spec=None, ordinary=None)
    if peak > 1.0 + 1e-9:
        raise K.Stop('technical', f"{row['cand_id']}: peak {peak} > 1 (budget not realisable; not clipped)")
    sec = row['section']
    if sec in ('S4', 'S5'):
        req = K.Q / out['lit_px']
        if abs(req - peak) > 1e-9 * max(1.0, req):
            raise K.Stop('technical', f"{row['cand_id']}: registered peak {peak!r} != Q / lit {req!r}")
    if sec == 'S3' and row['family'] == 'point':
        code = str(src['code']).split()
        out['enc_spec'] = dict(k=int(code[1][1:]), radii_mm=[float(x) / 2.0 for x in str(src['diameter_mm']).split(';')], code_ref='global', arm=code[0], layout=src['layout_id'])
    if sec == 'S4':
        T = _spat_table(src)
        if raster_key((np.asarray(T['pat']) > 0).astype(np.float32)) != out['raster_key']:
            raise K.Stop('technical', f"{row['cand_id']}: table raster differs from the adapter raster")
        if T['family'] == 'dots' and (T['centres'].shape != np.asarray(out['centres']).shape or not np.array_equal(T['centres'], np.asarray(out['centres'], np.float64))):
            raise K.Stop('technical', f"{row['cand_id']}: table centres differ from the adapter centres")
        out['table'] = T
        if str(row.get('strategy', '')).startswith('S0') or not T['disp']:
            o = _ordinary_s0(src, T)
            out['ordinary'] = dict(o, raster_equal=bool(np.array_equal(o['binary'] > 0, binary > 0)), raster_key=raster_key(o['binary']))
    _CACHE[key] = out
    if len(_CACHE) > 256:
        _CACHE.clear()
    return out


def m3_digest() -> str:
    import rdy_platform as P
    return K.digest({k: v for k, v in P.s2().m3_cfg(0.0, 0).items() if k not in ('noise_sigma', 'mottle_seed')})


def image_identity(row: dict, inp: dict, parts: dict) -> dict:
    sigma = float(row['sigma']); seed = K.norm_seed(row['noise_seed']) if sigma > 0 else None
    if sigma > 0 and seed is None:
        raise K.Stop('technical', f"{row['cand_id']}: sigma {sigma} without a registered seed")
    ident = dict(raster=inp['raster_key'], peak_f32=float(np.float32(inp['peak'])), surface=K.SURF_MAP[row['surface']], psf=K.PSF_PX, sigma=sigma, seed=seed, m3=m3_digest(),
                 platform=K.PLATFORM_VERSION, image_part=parts['image'])
    return dict(identity=K.digest(ident, 24), seed=seed, fields=ident)
