# Build new-phase cases, resource records and raster inputs.

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

import tr5_common as K

SURF_SID = {'B_curv': 'B_curv', 'M10_curv': 'A_curv', 'M24_curv': 'C_curv'}
DOT_CONFIGS = {'r0.5/R8x4': ('R8x4', 0.5), 'r1.0/R8x4': ('R8x4', 1.0), 'r1.0/T8x4': ('T8x4', 1.0), 'r1.0/R8x6': ('R8x6', 1.0)}
LINE_CONFIGS = {'s8': 8.0, 's10': 10.0}
SIGMAS = dict(point=(60.0, 80.0), line=(40.0, 60.0))
N_BLOCKS = dict(point=8, line=4)
SEED_BASE = dict(point=9_830_000, line=9_840_000)
Q_CAP = 11695.0
METHODS = dict(point=('D0', 'DF'), line=('P4',))
CONTRASTS = {'C1': ('point', 'r1.0/R8x4', 'r0.5/R8x4'), 'C2': ('point', 'r1.0/T8x4', 'r1.0/R8x4'), 'C3': ('point', 'r1.0/R8x6', 'r1.0/R8x4'),
             'C4': ('point', 'r1.0/T8x4', 'r1.0/R8x6'), 'C5': ('line', 's10', 's8')}
MANIFEST_COLS = ('forward_key', 'family', 'config', 'layout', 'size_mm', 'line_pitch_mm', 'orientation_deg', 'surface', 'platform_surface', 'block', 'sigma', 'noise_k',
                 'seed', 'u', 'v', 'phase_frac', 'seq_k', 'raster_key', 'lit_px', 'peak', 'peak_f32', 'integral_actual', 'active_limit', 'n_features', 'rays_per_pixel',
                 'methods', 'lit_unregistered_lines', 'contrasts')


def _paths():
    import tr_common
    for q in ('s3_code01', 's3_code02', 's3_code03', 's345_ready', 'sections_adapter', 'surf02', 's2_prepare', 's12_resource01', 'thesis_reinforce03'):
        p = str(K.SR / q)
        if p not in sys.path:
            sys.path.append(p)


_paths()


def s2():
    import rdy_platform as P
    return P.s2()


def _int_exact(x, name) -> int:
    try:
        f = float(x)
    except (TypeError, ValueError):
        raise K.Stop('preflight', f'{name} {x!r} is not a number') from None
    if not math.isfinite(f) or f != round(f):
        raise K.Stop('preflight', f'{name} {x!r} is not a finite integer')
    return int(round(f))


def core_rows() -> pd.DataFrame:
    p = K.OUT3 / 'plan.csv'
    if K.sha(p) != K.TR03_PLAN_SHA:
        raise K.Stop('preflight', f'TR03 plan.csv sha256 {K.sha(p)} != registered {K.TR03_PLAN_SHA}')
    P = K.read_csv(p); B = P[(P.record_type == 'B2_image') & (P.subset.isin(['dots_core', 'lines_core']))].copy()
    if len(B) != 480 or B.forward_key.nunique() != 480 or B.raster_key.nunique() != 40:
        raise K.Stop('preflight', f'core rows {len(B)}, unique forward keys {B.forward_key.nunique()}, rasters {B.raster_key.nunique()} (expected 480 / 480 / 40)')
    if (P[(P.record_type == 'B2_image')].subset.str.contains('optional')).sum() != 108:
        raise K.Stop('preflight', 'optional subset size differs from the registered 108 (not used, only checked)')
    return B


def raster_row(fam: str, cfg: str, r: dict) -> dict:

    if fam == 'point':
        lay, rad = DOT_CONFIGS[cfg]; arr, pu, pv = lay[0], float(lay[1:].split('x')[0]), float(lay.split('x')[1])
        return dict(generator='sl_lib.lattice', layout_id=f'{arr}{pu:g}x{pv:g}', phase_u=float(r['u']), phase_v=float(r['v']), radius_mm=rad, p_parallel_mm=pu, p_perp_mm=pv,
                    line_pitch_mm=None, line_width_mm=1.0, orientation_deg=0.0, phase_frac=None, section='S2', cand_id='B2PLAN', brightness_rule='peak=1.0',
                    raster_sha256_registered='', code='', diameter_mm='')
    return dict(generator='m2.olines', layout_id=f's{LINE_CONFIGS[cfg]:g}_phi0', phase_u=None, phase_v=None, radius_mm=None, p_parallel_mm=None, p_perp_mm=None,
                line_pitch_mm=LINE_CONFIGS[cfg], line_width_mm=1.0, orientation_deg=0.0, phase_frac=float(r['phase_frac']), section='S2', cand_id='B2PLAN',
                brightness_rule='peak=1.0', raster_sha256_registered='', code='', diameter_mm='')


_RAST: dict = {}


def raster(fam: str, cfg: str, r: dict) -> dict:

    import adp_patterns as PT
    import s1r_common as S1K
    rk = r['raster_key']
    if rk in _RAST:
        return _RAST[rk]
    b = PT.build(raster_row(fam, cfg, r))
    binary = (np.asarray(b['binary']) > 0)
    key = S1K.raster_key(binary)
    if key != rk:
        raise K.Stop('preflight', f'raster rebuilt as {key}, registered {rk} ({fam} {cfg} block {r["block"]})')
    out = dict(raster_key=key, binary=binary.astype(np.float32), centres=b['centres'], meta_full=b['meta_full'], lit_px=int(binary.sum()),
               n_features=(len(b['centres']) if fam == 'point' else len((b['meta_full']['olines'] or {}).get('ids', []))))
    if fam == 'line':
        import tr_a4 as A4
        sc = A4.scan_raster(out['binary'], b['meta_full']['olines'])
        out.update(lit_lines=sorted(sc['lines']), registered_ids=sc['ids'], lit_unregistered=sc['extra'], registered_without_lit=sc['registered_without_lit'],
                   olines=dict(spacing_px=float(b['meta_full']['olines']['spacing_px']), width_px=float(b['meta_full']['olines']['width_px']),
                               phase_frac=float(b['meta_full']['olines']['phase_frac']), n_lines=int(b['meta_full']['olines']['n_lines'])))
    _RAST[rk] = out
    return out


def resource(lit: int) -> dict:
    peak = min(1.0, Q_CAP / float(lit)); f32 = float(np.float32(peak))
    return dict(peak=peak, peak_f32=f32, integral_actual=lit * f32, active_limit=('peak_limited' if peak >= 1.0 else 'integral_limited'))


# Assemble experiment rows with their configuration and input identities.
def build_manifest() -> tuple[pd.DataFrame, dict]:
    import s12_common as S12
    B = core_rows(); rows = []; rasters = {}
    for r in B.to_dict('records'):
        fam = r['family']; cfg = r['config']; surf = r['surface']
        if fam not in METHODS or surf not in SURF_SID or (fam == 'point' and cfg not in DOT_CONFIGS) or (fam == 'line' and cfg not in LINE_CONFIGS):
            raise K.Stop('preflight', f"unregistered family / surface / configuration: {fam} {surf} {cfg}")
        block = _int_exact(r['block'], 'block'); k = _int_exact(r['noise_k'], 'noise_k'); seed = _int_exact(r['seed'], 'seed'); sigma = float(r['sigma'])
        if block not in range(N_BLOCKS[fam]) or k not in (0, 1) or sigma not in SIGMAS[fam]:
            raise K.Stop('preflight', f'{r["forward_key"]}: block {block} / k {k} / sigma {sigma} outside the registered design')
        if seed != SEED_BASE[fam] + 10 * block + k or str(r['base_field']) != f'{SEED_BASE[fam]}+10*{block}+{k}':
            raise K.Stop('preflight', f'{r["forward_key"]}: seed {seed} / base field {r["base_field"]} differ from the registered rule')
        if _int_exact(r['rays_per_pixel'], 'rays_per_pixel') != 1 or str(r['front_ends']) != ';'.join(METHODS[fam]):
            raise K.Stop('preflight', f'{r["forward_key"]}: rays / front ends differ')
        R = raster(fam, cfg, r); rasters[R['raster_key']] = R
        res = resource(R['lit_px'])
        if R['lit_px'] != _int_exact(r['lit_px'], 'lit_px') or abs(res['peak'] - float(r['peak_cap'])) > 1e-12 or abs(res['integral_actual'] - float(r['Q_cap'])) > 1e-6 * Q_CAP \
                or res['active_limit'] != r['active_limit'] or R['n_features'] != _int_exact(r['n_features'], 'n_features'):
            raise K.Stop('preflight', f'{r["forward_key"]}: resource / feature count differs from the registration ({R["lit_px"]}, {res}, {R["n_features"]})')
        if fam == 'line' and len(R['lit_unregistered']) != _int_exact(r['lit_unregistered_lines'], 'lit_unregistered_lines'):
            raise K.Stop('preflight', f'{r["forward_key"]}: lit unregistered lines {R["lit_unregistered"]} vs registered {r["lit_unregistered_lines"]}')
        sid = SURF_SID[surf]
        fk = S12.input_key(R['raster_key'], res['peak_f32'], sid, sigma, seed)
        if fk != r['forward_key']:
            raise K.Stop('preflight', f'forward key {fk} != registered {r["forward_key"]}')
        con = [c for c, (f, a, b) in CONTRASTS.items() if f == fam and cfg in (a, b)]
        rows.append(dict(forward_key=fk, family=fam, config=cfg, layout=(DOT_CONFIGS[cfg][0] if fam == 'point' else f's{LINE_CONFIGS[cfg]:g}'),
                         size_mm=(DOT_CONFIGS[cfg][1] if fam == 'point' else 1.0), line_pitch_mm=(LINE_CONFIGS[cfg] if fam == 'line' else None),
                         orientation_deg=(0.0 if fam == 'line' else None), surface=surf, platform_surface=sid, block=block, sigma=sigma, noise_k=k, seed=seed,
                         u=(float(r['u']) if fam == 'point' else None), v=(float(r['v']) if fam == 'point' else None),
                         phase_frac=(float(r['phase_frac']) if fam == 'line' else None), seq_k=(_int_exact(r['seq_k'], 'seq_k') if fam == 'line' else None),
                         raster_key=R['raster_key'], lit_px=R['lit_px'], peak=res['peak'], peak_f32=res['peak_f32'], integral_actual=res['integral_actual'],
                         active_limit=res['active_limit'], n_features=R['n_features'], rays_per_pixel=1, methods=';'.join(METHODS[fam]),
                         lit_unregistered_lines=(';'.join(map(str, R['lit_unregistered'])) if fam == 'line' else None), contrasts=';'.join(con)))
    M = pd.DataFrame(rows)[list(MANIFEST_COLS)].sort_values(['family', 'config', 'surface', 'block', 'sigma', 'noise_k'], kind='mergesort').reset_index(drop=True)
    return M, rasters


def coverage_check(M: pd.DataFrame) -> dict:

    probs = []
    for fam, cfgs in (('point', DOT_CONFIGS), ('line', LINE_CONFIGS)):
        F = M[M.family == fam]
        for cfg in cfgs:
            for s in SURF_SID:
                for b in range(N_BLOCKS[fam]):
                    for sg in SIGMAS[fam]:
                        for k in (0, 1):
                            n = int(((F.config == cfg) & (F.surface == s) & (F.block == b) & (F.sigma == sg) & (F.noise_k == k)).sum())
                            if n != 1:
                                probs.append(f'{fam} {cfg} {s} b{b} s{sg} k{k}: {n}')

        G = F.groupby(['surface', 'block', 'sigma', 'noise_k']).seed.nunique()
        if (G != 1).any():
            probs.append(f'{fam}: seed differs between configurations of one key')
    n_methods = int(sum(len(METHODS[f]) for f in M.family))
    by_method = {m: int(sum(1 for f in M.family if m in METHODS[f])) for m in ('D0', 'DF', 'P4')}
    return dict(passed=not probs and len(M) == 480 and n_methods == 864 and M.forward_key.is_unique and M.raster_key.nunique() == 40 and by_method == dict(D0=384, DF=384, P4=96),
                problems=probs[:20], n_images=len(M), n_method_outputs=n_methods, by_method=by_method, n_rasters=int(M.raster_key.nunique()),
                by_family=M.family.value_counts().to_dict(), optional_included=0)


def already_run(M: pd.DataFrame) -> dict:

    R = K.ROOT / 'outputs' / 'surface_revision'; keys = set(M.forward_key); hits = {}; sources = []
    V = K.read_csv(R / 's12_resource01' / 'results' / 'views.csv'); sources.append('s12_resource01/results/views.csv')
    h = keys & set(V.input_key.dropna())
    if h:
        hits['s12 views'] = sorted(h)
    files = sorted(R.glob('*/manifest*.csv')) + sorted(R.glob('*/*/manifest*.csv')) + sorted(R.glob('*/*/*/manifest*.csv')) + [R / 's12_resource01' / 'inputs.csv']
    files += [p for p in (R / 'thesis_reinforce04' / 'results.csv',) if p.exists()]
    for p in files:
        if not p.exists():
            continue
        T = K.read_csv(p); sources.append(K.rel(p))
        for c in ('input_key', 'forward_key', 'image_key'):
            if c in T:
                h = keys & set(T[c].dropna().astype(str))
                if h:
                    hits[f'{K.rel(p)}:{c}'] = sorted(h)
    img_dirs = [d for d in R.glob('*/images') if d.is_dir()] + [d for d in R.glob('*/*/images') if d.is_dir()]
    for d in img_dirs:
        h = {p.stem for p in d.glob('S12I-*.json')} & keys
        if h:
            hits[K.rel(d)] = sorted(h)
    if K.DATA.exists():
        own = {p.stem for p in (K.DATA).glob('S12I-*.npz')}
        if own:
            sources.append('own data/ (resume, not a conflict)')
    return dict(passed=not hits, conflicts=hits, n_sources=len(sources), n_image_dirs=len(img_dirs),
                note='the TR03 plan rows themselves are unrun proposals and are not counted as runs')


def context(row: dict) -> dict:

    import s1r_common as S1K
    fam = row['family']; R = _RAST.get(row['raster_key'])
    if R is None:
        R = raster(fam, row['config'], dict(row, phase_frac=row.get('phase_frac'), u=row.get('u'), v=row.get('v')))
    pattern = S1K.pattern_of(R['binary'], float(row['peak']))
    if float(pattern.max()) != float(np.float32(row['peak'])) or abs(float(pattern.sum(dtype=np.float64)) - float(row['integral_actual'])) > 1e-6 * Q_CAP:
        raise K.Stop('preflight', f"{row['forward_key']}: pattern scaling check failed")
    size = float(row['size_mm'])
    return dict(binary=R['binary'], centres=R['centres'], pattern=pattern, peak=float(row['peak']), size_mm=size, meta_full=R['meta_full'], row=dict(radius_mm=size))


def manifest_digest(M: pd.DataFrame) -> str:
    return K.digest(M[list(MANIFEST_COLS)].to_dict('records'))
