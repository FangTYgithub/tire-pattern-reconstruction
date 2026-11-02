# Select new phases and check periodic distances from existing layouts.

from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

import tr3_common as K

GOLD = 0.6180339887498949
Q_CAP = 11695.0
DOT_CONFIGS = (('r0.5/R8x4', 'R8x4', 0.5), ('r1.0/R8x4', 'R8x4', 1.0), ('r1.0/T8x4', 'T8x4', 1.0), ('r1.0/R8x6', 'R8x6', 1.0))
SEED_RULE = dict(noise_base_dots=9_830_000, noise_base_lines=9_840_000, rule='seed = noise_base + 10 * block + k (k = 0 / 1); one base field per block and k, '
                 'scaled for every sigma (the S12 convention): sigma 60 and 80 of one block / k are correlated, not independent repetitions')
MMPP_NOMINAL = 0.18


def lattice_of(name: str):
    m = re.search(r'([RT])(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)', str(name))
    return (m.group(1), float(m.group(2)), float(m.group(3))) if m else None


def translations(arr: str):
    if arr == 'R':
        return [(a, b) for a in range(-2, 3) for b in range(-2, 3)]
    return [(a + b / 2.0, b) for a in range(-2, 3) for b in range(-4, 5)]


def lattice_distance_mm(du: float, dv: float, name: str) -> float:
    arr, pu, pv = lattice_of(name)
    return float(min(np.hypot((du - tu) * pu, (dv - tv) * pv) for tu, tv in translations(arr)))


def build(fam, layout, size_mm, u=None, v=None, phase_frac=None) -> dict:
    import adp_patterns as PT
    import s1r_common as S1K
    if fam == 'point':
        arr, pu, pv = lattice_of(layout)
        row = dict(generator='sl_lib.lattice', layout_id=f'{arr}{pu:g}x{pv:g}', phase_u=u, phase_v=v, radius_mm=size_mm, p_parallel_mm=pu, p_perp_mm=pv, line_pitch_mm=None,
                   line_width_mm=1.0, orientation_deg=0.0, phase_frac=None, section='S2', cand_id='B2PLAN', brightness_rule='peak=1.0', raster_sha256_registered='', code='', diameter_mm='')
    else:
        row = dict(generator='m2.olines', layout_id=f's{float(layout):g}_phi0', phase_u=None, phase_v=None, radius_mm=None, p_parallel_mm=None, p_perp_mm=None,
                   line_pitch_mm=float(layout), line_width_mm=float(size_mm), orientation_deg=0.0, phase_frac=phase_frac, section='S2', cand_id='B2PLAN',
                   brightness_rule='peak=1.0', raster_sha256_registered='', code='', diameter_mm='')
    b = PT.build(row); K.CALLS['pattern_raster'] += 1
    binary = np.asarray(b['binary']) > 0; lit = float(binary.sum(dtype=np.float64)); peak = min(1.0, Q_CAP / lit)
    out = dict(raster_key=S1K.raster_key(binary), lit_px=lit, peak_cap=peak, Q_cap=lit * peak, active_limit=('peak_limited' if peak >= 1.0 else 'integral_limited'),
               n_features=(len(b['centres']) if fam == 'point' else len((b['meta_full']['olines'] or {}).get('ids', []))))
    if fam != 'point':
        import tr_a4 as A4
        sc = A4.scan_raster(binary.astype(np.float32), b['meta_full']['olines'])
        out.update(lit_unregistered_lines=len(sc['extra']), registered_without_lit=len(sc['registered_without_lit']))
    else:
        out['centres'] = np.asarray(b['centres'], float)
    return out


def period_checks() -> dict:
    res = {}
    a, b = build('point', 'T8x4', 1.0, 0.1, 0.2), build('point', 'T8x4', 1.0, 0.6, 0.2)
    res['T8x4_(0.1,0.2)_vs_(0.6,0.2)'] = dict(distance_mm=lattice_distance_mm(0.5, 0.0, 'T8x4'), raster_keys=[a['raster_key'], b['raster_key']], same_raster=a['raster_key'] == b['raster_key'],
                                              expected='different layouts (reference counterexample)')
    for nm, (lay, p, q) in dict(T_half_pitch_plus_row=('T8x4', (0.1, 0.2), (0.6, 1.2)), T_two_rows=('T8x4', (0.1, 0.2), (0.1, 2.2)), T_one_column=('T8x4', (0.1, 0.2), (1.1, 0.2)),
                                R_one_column=('R8x4', (0.1, 0.2), (1.1, 0.2)), R_one_row=('R8x4', (0.1, 0.2), (0.1, 1.2)), R_half_pitch=('R8x4', (0.1, 0.2), (0.6, 0.2)),
                                T_one_row_only=('T8x4', (0.1, 0.2), (0.1, 1.2))).items():
        x, y = build('point', lay, 1.0, *p), build('point', lay, 1.0, *q)
        res[nm] = dict(layout=lay, a=p, b=q, distance_mm=lattice_distance_mm(q[0] - p[0], q[1] - p[1], lay), infinite_equivalent=lattice_distance_mm(q[0] - p[0], q[1] - p[1], lay) < 1e-9,
                       finite_same_raster=x['raster_key'] == y['raster_key'], finite_same_centres=bool(len(x['centres']) == len(y['centres']) and np.allclose(x['centres'][np.lexsort(x['centres'].T[::-1])], y['centres'][np.lexsort(y['centres'].T[::-1])])))
    ok = (not res['T8x4_(0.1,0.2)_vs_(0.6,0.2)']['same_raster'] and res['T8x4_(0.1,0.2)_vs_(0.6,0.2)']['distance_mm'] > 0
          and all(r['infinite_equivalent'] == r['finite_same_raster'] for k, r in res.items() if k != 'T8x4_(0.1,0.2)_vs_(0.6,0.2)'))
    return dict(passed=bool(ok), cases=res)


def registry() -> dict:
    R = K.ROOT / 'outputs' / 'surface_revision'; pos, ph, seeds, src = [], [], set(), []
    U = K.read_csv(K.ROOT / 'outputs/size_spacing/s5_03/usage/used_phases_seeds.csv'); S = K.read_csv(K.ROOT / 'outputs/size_spacing/s5_04/usage/used_phases_seeds_supplement.csv')
    d = U[U.off_u.notna()]; pos += [dict(lattice=str(l), size=r, u=a, v=b, source='S5-03 registry') for l, r, a, b in zip(d.lattice, d.r_mm, d.off_u, d.off_v)]
    d = S[S.off_u.notna()]; pos += [dict(lattice='unknown', size=r, u=a, v=b, source='S5-04 supplement') for r, a, b in zip(d.r_mm, d.off_u, d.off_v)]
    d = U[U.phase_frac.notna()]; ph += [dict(pitch=s, width=w, orientation=None, phase=f, source='S5-03 registry') for s, w, f in zip(d.s_mm, d.w_mm, d.phase_frac)]
    for X in (U, S):
        for c in ('noise_seed', 'seed'):
            if c in X:
                seeds |= set(int(x) for x in X[c].dropna())
    src += [dict(source='S5-03 used_phases_seeds.csv', rows=len(U), covers='until 2026-10-11 (27 folders)'), dict(source='S5-04 supplement', rows=len(S), covers='26 further folders')]
    C = K.read_csv(R / 'sections_adapter/candidates_s1s5.csv')
    d = C[C.phase_u.notna()]; pos += [dict(lattice=str(l).split(':')[0], size=r, u=a, v=b, source='candidates_s1s5.csv') for l, r, a, b in zip(d.layout_id, d.radius_mm, d.phase_u, d.phase_v)]
    d = C[C.phase_frac.notna()]; ph += [dict(pitch=s, width=w, orientation=o, phase=f, source='candidates_s1s5.csv') for s, w, o, f in zip(d.line_pitch_mm, d.line_width_mm, d.orientation_deg, d.phase_frac)]
    seeds |= set(int(x) for x in C.noise_seed.dropna()); src.append(dict(source='candidates_s1s5.csv', rows=len(C), covers='sections S1-S5'))
    known_rk = set(K.read_csv(K.T1.S12 / 'results' / 'views.csv').raster_key.dropna())
    for p in sorted(R.glob('*/manifest*.csv')) + sorted(R.glob('*/*/manifest*.csv')) + sorted(R.glob('*/*/*/manifest*.csv')):
        M = K.read_csv(p); src.append(dict(source=K.rel(p), rows=len(M)))
        uc = 'u' if 'u' in M else ('phase_u' if 'phase_u' in M else None); vc = 'v' if 'v' in M else ('phase_v' if 'phase_v' in M else None)
        lay = M['layout'].astype(str) if 'layout' in M else pd.Series(['unknown'] * len(M))
        if uc and vc:
            m = M[uc].notna() & M[vc].notna()
            sz = M.get('sizes_mm', M.get('size_mm', pd.Series([None] * len(M))))
            pos += [dict(lattice=l.split(':')[0], size=s, u=a, v=b, source=K.rel(p)) for l, s, a, b in zip(lay[m], sz[m], M[uc][m], M[vc][m])]
        if 'phase_frac' in M:
            m = M.phase_frac.notna()
            for l, f in zip(lay[m], M.phase_frac[m]):
                mm = re.search(r'(?:^|[sL_])(\d+(?:\.\d+)?)', l); ph.append(dict(pitch=(float(mm.group(1)) if mm else None), width=None, orientation=None, phase=f, source=K.rel(p)))
        for c in ('seed', 'noise_seed'):
            if c in M:
                seeds |= set(int(x) for x in pd.to_numeric(M[c], errors='coerce').dropna())
        if 'raster_key' in M:
            known_rk |= set(M.raster_key.dropna())
    J = K.load_json(K.ROOT / 'outputs/size_spacing/s5_04/usage/unseen_scope_check.json')
    for f in J.get('spat_line_phases', {}):
        for s in (J.get('spat_line_pitches') or {}).values():
            ph.append(dict(pitch=float(s), width=J.get('spat_line_width_mm'), orientation=None, phase=float(f), source='SPAT phases (S5-04 scope check)'))
    P = pd.DataFrame(pos).drop_duplicates(['lattice', 'size', 'u', 'v']); F = pd.DataFrame(ph).drop_duplicates(['pitch', 'width', 'orientation', 'phase'])
    return dict(positions=P, phases=F, seeds=seeds, raster_keys=known_rk, sources=src,
                not_covered='technical renders without a manifest (deleted development renders) are not listed; MODEL-PREP-01 has no delivery; '
                            'S1 point phases are PM-013 quantiles (in the S5-03 registry, lattice R8x4 / generic)')


def dot_positions(reg: dict) -> list:
    pos = K.load_json(K.OUT2 / 'run.json')['plans']['b2']['positions']; P = reg['positions']; out = []
    for q in pos:
        rec = dict(block=q['block'], u=q['u'], v=q['v'], old_min_dist_mm=q['min_dist_used_mm'])
        for nm, lay, r in DOT_CONFIGS:
            same = P[P.lattice == lay]
            d_same = [lattice_distance_mm(q['u'] - a, q['v'] - b, lay) for a, b in zip(same.u, same.v)]
            exact = [i for i, d in enumerate(d_same) if d < 1e-9]
            other = P[(P.lattice != lay)]
            rec[nm] = dict(nearest_same_lattice_mm=(float(min(d_same)) if d_same else None), n_same_lattice_registered=len(same),
                           position_used_same_lattice=bool(exact), nearest_other_layout_normalised=(float(min(np.hypot(((q['u'] - other.u + 0.5) % 1 - 0.5), ((q['v'] - other.v + 0.5) % 1 - 0.5)))) if len(other) else None))
        out.append(rec)
    return out


def _phase_used(F: pd.DataFrame, pitch: float, phi: float) -> tuple[bool, float]:
    m = F.pitch.isna() | (pd.to_numeric(F.pitch, errors='coerce').sub(pitch).abs() < 1e-6)
    m &= F.orientation.isna() | (pd.to_numeric(F.orientation, errors='coerce').abs() < 1e-9)
    used = pd.to_numeric(F.phase[m], errors='coerce').dropna().values
    if not len(used):
        return False, 1.0
    d = np.abs(((phi - used) + 0.5) % 1.0 - 0.5)
    return bool((d < 1e-9).any()), float(d.min())


def common_phases(reg: dict, pitches: tuple, n: int, require_edge: bool, kmax: int = 4000) -> dict:
    F = reg['phases']; known = reg['raster_keys']; acc, rejected = [], {}
    for k in range(1, kmax + 1):
        phi = float(((k * GOLD) % 1.0) - 0.5); info = {}; ok = True
        for p in pitches:
            used, dmin = _phase_used(F, p, phi)
            g = build('line', p, 1.0, phase_frac=phi)
            reasons = []
            if used:
                reasons.append('phase used')
            if g['raster_key'] in known:
                reasons.append('raster used')
            if g['registered_without_lit']:
                reasons.append('registered line not lit')
            if require_edge and g['lit_unregistered_lines'] == 0:
                reasons.append('no lit unregistered edge line (diagnostic precondition)')
            info[p] = dict(g, min_dist_used_phase=dmin, min_dist_used_mm=dmin * p, min_dist_used_px_nominal=dmin * p / MMPP_NOMINAL)
            if reasons:
                ok = False; rejected.setdefault(';'.join(reasons), 0); rejected[';'.join(reasons)] += 1
        if ok:
            acc.append(dict(block=len(acc), k=k, phase_frac=phi, per_pitch=info))
            if len(acc) == n:
                break
    return dict(accepted=acc, n_requested=n, found=len(acc) == n, rejected_counts=rejected, last_k=k)


def plan(reg: dict, dots: list, core: dict, diag: dict) -> pd.DataFrame:
    import s12_common as S12
    import rdy_platform as P
    surf = ('B_curv', 'M10_curv', 'M24_curv'); rows = []
    geo = {}
    for q in dots:
        for nm, lay, r in DOT_CONFIGS:
            geo[(nm, q['block'])] = {k: v for k, v in build('point', lay, r, q['u'], q['v']).items() if k != 'centres'}
    for nm, p in (('s8', 8.0), ('s10', 10.0)):
        for q in core['accepted']:
            geo[(nm, q['block'])] = {k: v for k, v in q['per_pitch'][p].items()}
    for nm, p in (('s5', 5.0), ('s6.3', 6.3)):
        for q in diag[p]['accepted']:
            geo[(nm, q['block'])] = {k: v for k, v in q['per_pitch'][p].items()}

    def add(subset, fam, nm, block, sigma, k, extra):
        g = geo[(nm, block)]; base = SEED_RULE['noise_base_dots'] if fam == 'point' else SEED_RULE['noise_base_lines']
        seed = (base + 10 * block + k) if sigma > 0 else None
        for s in surf:
            sid = {'B_curv': 'B_curv', 'M10_curv': 'A_curv', 'M24_curv': 'C_curv'}[s]
            fk = S12.input_key(g['raster_key'], float(np.float32(g['peak_cap'])), sid, float(sigma), seed)
            rows.append(dict(record_type='B2_image', status='proposal (not authorised)', subset=subset, family=fam, config=nm, surface=s, block=block, sigma=float(sigma), noise_k=k,
                             seed=seed, base_field=(f'{base}+10*{block}+{k}' if seed else None), forward_key=fk, rays_per_pixel=1,
                             front_ends=('D0;DF' if fam == 'point' else 'P4'), **extra,
                             **{kk: g[kk] for kk in ('raster_key', 'lit_px', 'peak_cap', 'Q_cap', 'active_limit', 'n_features') if kk in g},
                             **({'lit_unregistered_lines': g.get('lit_unregistered_lines'), 'nearest_used_phase_mm': g.get('min_dist_used_mm'),
                                 'nearest_used_phase_px_nominal': g.get('min_dist_used_px_nominal')} if fam == 'line' else {})))
    for q in dots:
        for nm, lay, r in DOT_CONFIGS:
            _, pu, pv = lattice_of(lay)
            ex = dict(u=q['u'], v=q['v'], offset_par_mm=round(q['u'] * pu, 4), offset_perp_mm=round(q['v'] * pv, 4), nearest_same_lattice_mm=q[nm]['nearest_same_lattice_mm'],
                      position_used_same_lattice=q[nm]['position_used_same_lattice'])
            for sg in (60.0, 80.0):
                for k in (0, 1):
                    add('dots_core', 'point', nm, q['block'], sg, k, ex)
            add('dots_bridge_sigma40_optional', 'point', nm, q['block'], 40.0, 0, ex)
    for q in core['accepted']:
        for nm in ('s8', 's10'):
            for sg in (40.0, 60.0):
                for k in (0, 1):
                    add('lines_core', 'line', nm, q['block'], sg, k, dict(phase_frac=q['phase_frac'], seq_k=q['k'], width_mm=1.0, orientation_deg=0.0))
    for nm, p in (('s5', 5.0), ('s6.3', 6.3)):
        for q in diag[p]['accepted']:
            add('lines_edge_diagnostic_optional', 'line', nm, q['block'], 0.0, 0, dict(phase_frac=q['phase_frac'], seq_k=q['k'], width_mm=1.0, orientation_deg=0.0))
    T = pd.DataFrame(rows)
    T.attrs['platform'] = P.s2().VERSION
    return T


def plan_checks(T: pd.DataFrame, reg: dict) -> dict:
    V = K.read_csv(K.T1.S12 / 'results' / 'views.csv')
    seeds = set(int(x) for x in T.seed.dropna())
    return dict(n_images=len(T), by_subset=T.groupby('subset').size().to_dict(), forward_keys_unique=bool(T.forward_key.is_unique),
                forward_key_collisions_with_S12=sorted(set(T.forward_key) & set(V.input_key)), raster_key_collisions_with_registry=sorted(set(T.raster_key) & reg['raster_keys']),
                seed_collisions=sorted(seeds & reg['seeds']), peak_le_1=bool((T.peak_cap <= 1 + 1e-12).all()), n_registry_positions=len(reg['positions']),
                n_registry_phases=len(reg['phases']), n_registry_seeds=len(reg['seeds']), n_known_raster_keys=len(reg['raster_keys']))


STAT_CONTRACT = dict(
    unit='block (registered position / phase) x surface x noise realisation x sigma; D0 and DF are two processings of one image',
    order='average the noise realisations inside a block, then weight blocks equally; every surface separately; no cross-surface mean that hides a reversal',
    resampling='descriptive block bootstrap keeping all arms and realisations of a block together; two noise realisations are two, not many independent trials',
    metrics='global / local C_tau and C_valid (d 6 mm, tau 0.15 mm), point RMSE / P95 with defined counts, c / w / u with the A4-corrected identity version, '
            'w / (c + w) and u / (c + w + u) separately, absolute w; no output -> coverage 0 and conditional errors undefined',
    frozen='D0, DF, P4, evaluation domains, thresholds and identity version bound by the R2 fingerprints at authorisation; no change after results',
    decision='support: same direction on all three surfaces, >= 6 of 8 (dots) / 3 of 4 (lines) blocks with the same sign, block-bootstrap interval excluding 0 on >= 2 '
             'surfaces; narrow: only some surfaces / front-ends or intervals across 0 ("cannot show stable superiority"); insufficient / negative: opposite '
             'direction on >= 2 surfaces -> the specific superiority claim is withdrawn; no sample extension after a non-reproduced ordering',
    scope='new phases inside the studied position domain of the same three surfaces, the same device geometry and the same noise model; no claim for new '
          'surfaces, positions outside the domain or other devices')
