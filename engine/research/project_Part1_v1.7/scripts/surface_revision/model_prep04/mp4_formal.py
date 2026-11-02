# Define formal environment conditions, phases and noise seeds.

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import mp4_common as K

RHO = 1.324717957244746
SQ2M1 = math.sqrt(2.0) - 1.0
Q_CAP = 11695.0
CONFIGS = ('D_r1_p8x4', 'D_r1_p8x6', 'D_r1.5_p8x4', 'D_r1.5_p8x6', 'L_w0.5_s6.3', 'L_w0.5_s8', 'L_w1.5_s6.3', 'L_w1.5_s8')
DOTS, LINES = CONFIGS[:4], CONFIGS[4:]
TRANSITION_LINES = ('L_w0.5_s8', 'L_w1.5_s8')
SURFACES = ('B_curv', 'M10_curv', 'M24_curv')
SURF_SID = {'B_curv': 'B_curv', 'M10_curv': 'A_curv', 'M24_curv': 'C_curv'}
SURF_IDX = {'B_curv': 0, 'M10_curv': 1, 'M24_curv': 2}
SIGMAS, REPS, N_BLOCKS, DIAG_BLOCK, S0_SURFACE = (20.0, 40.0), (1, 2), 4, 0, 'M10_curv'
SEED_BASE = dict(point=9_870_000, line=9_880_000)
ENVS = dict(NOMINAL=dict(psf=1.0, field=('const', 1.0)), BLUR2=dict(psf=2.0, field=('const', 1.0)), BLUR3=dict(psf=3.0, field=('const', 1.0)),
            UNIFORM=dict(psf=1.0, field=('const', 0.75)), GRAD_X=dict(psf=1.0, field=('gradient_x', 0.75, 0.25)), GRAD_Y=dict(psf=1.0, field=('gradient_y', 0.75, 0.25)))
TRANSITION_ENV = dict(BLUR1_5=dict(psf=1.5, field=('const', 1.0)))
METHODS = dict(point=('D0', 'DF'), line=('P4',))
MIN_DIST = dict(point=0.05, line=0.015)
RULE_HISTORY = [dict(t='2026-10-19 (first registration)', line_min_dist=0.03, outcome='infeasible: 0 accepted phases for k <= 5000 (19924 rejections, all "near used phase"); '
                     'the registry already uses 85 (31 distinct) phases at 6.3 mm and 173 (51 distinct) at 8 mm incl. pitch-unknown rows; the best common phase reaches '
                     '0.0234 cycles (geometry of the registry only, no performance read)'),
                dict(t='2026-10-19 (re-registration)', line_min_dist=0.015, change='single alternative: line threshold 0.015 cycles (6.3 mm: 0.095 mm / 0.53 projector px; '
                     '8 mm: 0.12 mm / 0.67 px); new phases additionally mutually >= the family threshold; sequence, registry and all other rules unchanged')]
EXPECTED = dict(core=2304, sigma0_diag=48, transition=96, transition_sigma0=2, forward_keys=2450, method_results=3626)
HALF_MM = 46.08


def rules() -> dict:
    return dict(phase=dict(dots='(u_k, v_k) = (frac(0.5 + k / rho), frac(0.5 + k / rho^2)), rho = 1.324717957244746 (plastic number), k = 1, 2, ...',
                           lines='phi_k = frac(0.5 + k (sqrt(2) - 1)) - 0.5, k = 1, 2, ...', common='one sequence per family, common normalised coordinates for its 4 configurations',
                           accept='raster builds (registered template row, phase fields replaced); every registered line lit (lines); raster key unused; normalised distance to the '
                                  'registry >= threshold for every configuration', min_dist=MIN_DIST, dot_metric='torus distance in cell units vs R / unknown lattices, any size',
                           line_metric='circular distance vs same pitch (or pitch unknown), orientation 0 / unknown, any width', n_blocks=N_BLOCKS, diag_block=DIAG_BLOCK,
                           registry='thesis_reinforce03/tr3_b2.registry()', never='performance results are never read; clipped lit lines are recorded, not selected on',
                           mutual='accepted phases of a family are mutually >= the family threshold (same metric)', history=RULE_HISTORY),
                noise=dict(rule='seed = base + 100 * surface_index + 10 * block + rep', base=SEED_BASE, surface_index=SURF_IDX,
                           shared='inside a (family, surface, block, rep) block: across sigma (scaled base field), environments and the four configurations',
                           independent='between blocks', sigma0='rep 0, no seed, one realisation'),
                design=dict(configs=CONFIGS, surfaces=SURFACES, sigmas=SIGMAS, reps=REPS, envs={k: v for k, v in ENVS.items()}, transition=dict(lines=TRANSITION_LINES, env=TRANSITION_ENV),
                            sigma0=dict(surface=S0_SURFACE, block=DIAG_BLOCK), methods=METHODS, expected=EXPECTED, q_cap=Q_CAP))


def templates() -> dict:

    M = K.read_csv(K.MP3_OUT / 'manifest.csv'); C = K.read_csv(K.OUTSR / 'sections_adapter' / 'candidates_s1s5.csv')
    N = M[M.record_type == 'new']; out = {}
    for cfg in CONFIGS:
        q = N[N.combo == cfg].drop_duplicates('phase_label')
        refs = {}
        for r in q.to_dict('records'):
            c = C[C.cand_id == r['cand_id']]
            if len(c) != 1:
                raise K.Stop('preflight', f'{cfg}: candidate {r["cand_id"]} rows {len(c)}')
            refs[r['phase_label']] = dict(cand_id=r['cand_id'], raster_key=r['raster_key'], row=c.iloc[0].to_dict())
        out[cfg] = refs
    return out


def _row_with_phase(tpl: dict, fam: str, ph: dict) -> dict:
    r = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in tpl.items()}
    r['raster_sha256_registered'] = ''
    if fam == 'point':
        r['phase_u'], r['phase_v'] = float(ph['u']), float(ph['v'])
    else:
        r['phase_frac'] = float(ph['phi'])
    return r


_RAST: dict = {}


def build(tpl: dict, fam: str, ph: dict) -> dict:

    import adp_patterns as PT
    import rdy_inputs as RI
    row = _row_with_phase(tpl, fam, ph); key = K.digest(row)
    if key in _RAST:
        return _RAST[key]
    b = PT.build(row); K.AUX['pattern_raster'] += 1
    binary = (np.asarray(b['binary']) > 0).astype(np.float32); peak = float(b['peak'])
    pattern = (binary * np.float32(peak)).astype(np.float32); lit = float(binary.sum(dtype=np.float64))
    out = dict(raster_key=RI.raster_key(binary), lit_px=lit, peak=peak, peak_f32=float(np.float32(peak)), integral=float(pattern.sum(dtype=np.float64)),
               pattern_sha=K.content_sha(pattern), meta_full=b['meta_full'], n_centres=(len(b['centres']) if b.get('centres') is not None else None), binary=binary, pattern=pattern)
    out['resource_ok'] = bool(peak <= 1.0 + 1e-12 and abs(peak - min(1.0, Q_CAP / lit)) <= 1e-9 * max(1.0, min(1.0, Q_CAP / lit)) and out['integral'] <= Q_CAP * (1.0 + 2.0 ** -23))
    if fam == 'line':
        import tr_a4 as A4
        sc = A4.scan_raster(binary, b['meta_full']['olines'])
        out.update(lit_unregistered=sc['extra'], registered_without_lit=sc['registered_without_lit'])
    _RAST[key] = out
    return out


def builder_consistency(T: dict) -> dict:

    import mp3_chain as CH
    q = CH.realq(); res = {}
    for cfg, refs in T.items():
        fam = 'point' if cfg in DOTS else 'line'
        for ph, ref in refs.items():
            r = ref['row']
            phd = dict(u=r['phase_u'], v=r['phase_v']) if fam == 'point' else dict(phi=r['phase_frac'])
            b = build(r, fam, phd); inp = q.inputs(dict(cand_id_sigma0=ref['cand_id'], family=fam))
            res[f'{cfg}|{ph}'] = dict(raster=b['raster_key'] == ref['raster_key'] == inp['raster_key'], pattern=b['pattern_sha'] == K.content_sha(np.asarray(inp['pattern'], np.float32)),
                                      peak=b['peak'] == float(inp['peak']), lit=b['lit_px'] == float(inp['lit_px']))
    return dict(ok=all(all(v.values()) for v in res.values()), cases=res)


def _torus(du, dv):
    a = (np.asarray(du) + 0.5) % 1.0 - 0.5; b = (np.asarray(dv) + 0.5) % 1.0 - 0.5
    return np.hypot(a, b)


# Select phase offsets separated from the existing phase registry.
def select_phases(T: dict, reg: dict, kmax: int = 5000) -> dict:
    Pz = reg['positions']; F = reg['phases']; known = reg['raster_keys']
    lat = Pz.lattice.astype(str)
    Rpos = Pz[lat.str.startswith('R') | lat.isin(['unknown', 'nan', 'generic', 'None'])]
    out = {}
    for fam, cfgs in (('point', DOTS), ('line', LINES)):
        acc, rejected = [], {}
        for k in range(1, kmax + 1):
            ph = (dict(u=float((0.5 + k / RHO) % 1.0), v=float((0.5 + k / RHO ** 2) % 1.0)) if fam == 'point' else dict(phi=float(((0.5 + k * SQ2M1) % 1.0) - 0.5)))
            reasons, info = [], {}
            if fam == 'point':
                d = _torus(ph['u'] - pd.to_numeric(Rpos.u).values, ph['v'] - pd.to_numeric(Rpos.v).values)
                dmin = float(d.min()) if len(d) else 1.0
                dall = float(_torus(ph['u'] - pd.to_numeric(Pz.u).values, ph['v'] - pd.to_numeric(Pz.v).values).min())
                if dmin < MIN_DIST['point']:
                    reasons.append('near used position')
                info['min_dist_R_unknown'] = dmin; info['min_dist_any_lattice'] = dall
            else:
                for cfg in cfgs:
                    pitch = float(T[cfg]['ph0']['row']['line_pitch_mm'])
                    m = F.pitch.isna() | (pd.to_numeric(F.pitch, errors='coerce').sub(pitch).abs() < 1e-6)
                    m &= F.orientation.isna() | (pd.to_numeric(F.orientation, errors='coerce').abs() < 1e-9)
                    used = pd.to_numeric(F.phase[m], errors='coerce').dropna().values
                    dmin = float(np.abs(((ph['phi'] - used) + 0.5) % 1.0 - 0.5).min()) if len(used) else 1.0
                    info[f'min_dist_{cfg}'] = dmin
                    if dmin < MIN_DIST['line']:
                        reasons.append(f'near used phase ({cfg})')
            if not reasons and acc:
                dm = (min(float(_torus(ph['u'] - q['u'], ph['v'] - q['v'])) for q in acc) if fam == 'point'
                      else min(abs(((ph['phi'] - q['phi']) + 0.5) % 1.0 - 0.5) for q in acc))
                info['min_dist_to_accepted'] = dm
                if dm < MIN_DIST[fam]:
                    reasons.append('near an accepted new phase')
            if not reasons:
                for cfg in cfgs:
                    b = build(T[cfg]['ph0']['row'], fam, ph)
                    if b['raster_key'] in known:
                        reasons.append(f'raster used ({cfg})')
                    if not b['resource_ok']:
                        reasons.append(f'resource rule ({cfg})')
                    if fam == 'line' and b['registered_without_lit']:
                        reasons.append(f'registered line not lit ({cfg})')
                    info[cfg] = dict(raster_key=b['raster_key'], lit_px=b['lit_px'], peak=b['peak'], lit_unregistered=b.get('lit_unregistered'))
            if reasons:
                for x in reasons:
                    rejected[x.split(' (')[0]] = rejected.get(x.split(' (')[0], 0) + 1
                continue
            acc.append(dict(block=len(acc), k=k, **ph, info=info))
            if len(acc) == N_BLOCKS:
                break
        out[fam] = dict(accepted=acc, found=len(acc) == N_BLOCKS, rejected=rejected, last_k=k)
    return out


def _seed(fam, surface, block, rep):
    return int(SEED_BASE[fam] + 100 * SURF_IDX[surface] + 10 * block + rep)


# Enumerate candidate, surface, phase, noise and environment combinations.
def manifest(T: dict, phases: dict) -> pd.DataFrame:
    import mp3_forward as F
    import rdy_platform as RP
    import m3_virtual_camera as M3
    S = RP.s2(); mesh_sha = {s: S.mesh_sha(S.build_mesh(SURF_SID[s])) for s in SURFACES}
    fwd_fp = K.digest({f: K.sha(K.ROOT / f) for f in __import__('mp3_chain').FORWARD_FILES})

    def field(env, envs):
        f = envs[env]['field']
        return F.field_const(f[1]) if f[0] == 'const' else F.field_gradient(f[0].split('_')[1], f[1], f[2], HALF_MM)

    rows = []

    def add(subset, cfg, surface, ph, sigma, rep, env, envs):
        fam = 'point' if cfg in DOTS else 'line'
        tpl = T[cfg]['ph0']['row']; b = build(tpl, fam, ph)
        seed = None if sigma == 0.0 else _seed(fam, surface, ph['block'], rep)
        ns = F.noise_spec(sigma, seed, 'compat'); base = dict(S.m3_cfg(float(sigma), int(ns['m3_seed'])))
        fld = field(env, envs); spec = F.forward_spec(base, float(envs[env]['psf']), fld, ns)
        rec = dict(sid=SURF_SID[surface], mesh_sha16=mesh_sha[surface], raster_key=b['raster_key'], pattern=b['pattern_sha'], peak_f32=b['peak_f32'], integral=b['integral'], spec=spec,
                   platform=S.VERSION, m3_file=K.sha(M3.__file__), adapter_file=K.sha(K.SR / 'model_prep03' / 'mp3_forward.py'))
        r = dict(subset=subset, family=fam, config=cfg, template_cand_id=T[cfg]['ph0']['cand_id'], radius_mm=tpl.get('radius_mm'), line_width_mm=tpl.get('line_width_mm'),
                 p_parallel_mm=tpl.get('p_parallel_mm'), p_perp_mm=tpl.get('p_perp_mm'), line_pitch_mm=tpl.get('line_pitch_mm'), surface=surface, platform_surface=SURF_SID[surface],
                 phase_block=ph['block'], phase_seq_k=ph['k'], phase_u=ph.get('u'), phase_v=ph.get('v'), phase_frac=ph.get('phi'),
                 phase_offset_par_mm=(ph['u'] * float(tpl['p_parallel_mm']) if fam == 'point' else ph['phi'] * float(tpl['line_pitch_mm'])),
                 phase_offset_perp_mm=(ph['v'] * float(tpl['p_perp_mm']) if fam == 'point' else None),
                 sigma=float(sigma), rep=int(rep), seed=seed, noise_block=(None if seed is None else f'{fam}|{surface}|b{ph["block"]}|r{rep}'), env=env, psf_true_px=float(envs[env]['psf']),
                 psf_nominal_px=1.0, field_kind=fld['kind'], field_json=K.canon(fld), field_id=F.field_id(fld), raster_key=b['raster_key'], lit_px=b['lit_px'], peak=b['peak'],
                 peak_f32=b['peak_f32'], integral=b['integral'], resource_ok=b['resource_ok'], lit_unregistered_lines=(json.dumps(b.get('lit_unregistered')) if fam == 'line' else None),
                 registered_without_lit=(json.dumps(b.get('registered_without_lit')) if fam == 'line' else None),
                 methods=';'.join(METHODS[fam]), n_methods=len(METHODS[fam]), forward_key=K.digest(rec), forward_fingerprint=fwd_fp)
        r['case_key'] = f"{cfg}|{surface}|b{ph['block']}|s{int(sigma)}|r{rep}|{env}"
        r['parent_case_key'] = f"{cfg}|{surface}|b{ph['block']}|s{int(sigma)}|r{rep}|NOMINAL"
        rows.append(r)

    ph_of = {fam: {q['block']: dict(block=q['block'], k=q['k'], **({'u': q['u'], 'v': q['v']} if fam == 'point' else {'phi': q['phi']})) for q in phases[fam]['accepted']}
             for fam in ('point', 'line')}
    for cfg in CONFIGS:
        fam = 'point' if cfg in DOTS else 'line'
        for s in SURFACES:
            for bl in range(N_BLOCKS):
                for sg in SIGMAS:
                    for rp in REPS:
                        for env in ENVS:
                            add('core', cfg, s, ph_of[fam][bl], sg, rp, env, ENVS)
        for env in ENVS:
            add('sigma0_diag', cfg, S0_SURFACE, ph_of[fam][DIAG_BLOCK], 0.0, 0, env, ENVS)
    envs_t = dict(ENVS, **TRANSITION_ENV)
    for cfg in TRANSITION_LINES:
        for s in SURFACES:
            for bl in range(N_BLOCKS):
                for sg in SIGMAS:
                    for rp in REPS:
                        add('transition', cfg, s, ph_of['line'][bl], sg, rp, 'BLUR1_5', envs_t)
        add('transition_sigma0', cfg, S0_SURFACE, ph_of['line'][DIAG_BLOCK], 0.0, 0, 'BLUR1_5', envs_t)
    M = pd.DataFrame(rows)
    fk = M.set_index('case_key')['forward_key']
    M['parent_forward_key'] = [None if e == 'NOMINAL' else fk.get(p) for e, p in zip(M.env, M.parent_case_key)]
    M.loc[M.env == 'NOMINAL', 'parent_case_key'] = None
    M['seed'] = pd.Series([None if x is None or (isinstance(x, float) and math.isnan(x)) else int(x) for x in M['seed']], dtype=object)
    return M


def checks(M: pd.DataFrame, phases: dict, cons: dict, reg: dict, registered_rules_t: str | None, manifest_t: str) -> list:
    rows = []

    def add(cid, ok, desc, **detail):
        rows.append(dict(check_id=cid, passed=bool(ok), description=desc, detail=json.dumps(K.clean(detail), ensure_ascii=False, default=str)[:3500]))

    n = M.subset.value_counts().to_dict()
    add('D_counts', len(M) == EXPECTED['forward_keys'] and M.forward_key.is_unique and M.case_key.is_unique and all(n.get(k, 0) == v for k, v in EXPECTED.items() if k in n or k in ('core', 'sigma0_diag', 'transition', 'transition_sigma0'))
        and int(M.n_methods.sum()) == EXPECTED['method_results'], 'subsets 2304 / 48 / 96 / 2 = 2450 unique forward keys; 3626 method results (dots D0 + DF, lines P4)',
        subsets=n, n=len(M), unique_forward=int(M.forward_key.nunique()), methods=int(M.n_methods.sum()))
    add('D_phase_rule_first', bool(registered_rules_t) and registered_rules_t < manifest_t, 'phase / noise / analysis rules registered before the phases were generated',
        rules_t=registered_rules_t, manifest_t=manifest_t)
    ok = phases['point']['found'] and phases['line']['found']
    dmins = {fam: [min(v for kk, v in q['info'].items() if kk.startswith('min_dist') and kk != 'min_dist_any_lattice') for q in phases[fam]['accepted']] for fam in ('point', 'line')}
    ok &= all(d >= MIN_DIST[f] for f in dmins for d in dmins[f])
    old = set(K.read_csv(K.MP3_OUT / 'manifest.csv').raster_key.dropna())
    ok &= not (set(M.raster_key) & (reg['raster_keys'] | old))
    add('D_phases_unseen', ok, 'four new phases per family by the registered rule; normalised distance to the registry >= threshold; raster keys never used (registry + MODEL-PREP-03)',
        phases={f: [{k: v for k, v in q.items() if k != 'info'} for q in phases[f]['accepted']] for f in phases}, min_dist=dmins, rejected={f: phases[f]['rejected'] for f in phases},
        last_k={f: phases[f]['last_k'] for f in phases})
    add('D_builder_consistency', cons['ok'], 'the template builder reproduces raster key, pattern, peak and lit pixels of all 16 MODEL-PREP-03 rasters of the registered route (RealQ.inputs)', cases=cons['cases'])
    add('D_resource', bool(M.resource_ok.all()), 'peak = min(1, 11695 / lit px) and integral <= 11695 (1 + 2^-23) for every raster', n_bad=int((~M.resource_ok).sum()),
        peak_range=[float(M.peak.min()), float(M.peak.max())])
    L = M[M.family == 'line']; LU = L.drop_duplicates('raster_key')
    unlit = LU[LU.registered_without_lit.map(lambda s: json.loads(s) != [])]
    add('D_lines_lit', len(LU) == 16 and len(unlit) == 0, 'every registered line id is lit in all 16 line rasters (selection rule, re-checked on the manifest); clipped lit unregistered lines recorded (not selected on)',
        unlit=unlit[['config', 'phase_block', 'registered_without_lit']].to_dict('records'), clipped=LU[['config', 'phase_block', 'lit_unregistered_lines']].to_dict('records'))
    s = M[M.sigma > 0]; blk = s.groupby('noise_block').seed.nunique(); sb = s.groupby('seed').noise_block.nunique()
    s0 = M[M.sigma == 0]
    ok = bool((blk == 1).all() and (sb == 1).all() and s0.seed.map(lambda x: x is None).all() and set(s.rep) == {1, 2} and set(s0.rep) == {0}
              and not (set(int(x) for x in s.seed) & reg['seeds']))
    add('D_noise_blocks', ok, 'one seed per (family, surface, block, rep) block and vice versa; blocks independent; sigma 0 without seed (rep 0); no seed used before',
        n_blocks=int(len(blk)), seeds_used_before=sorted(set(int(x) for x in s.seed) & reg['seeds'])[:5])
    E = M[M.env != 'NOMINAL']; nom = M[M.env == 'NOMINAL'].set_index('forward_key')
    par = E.parent_forward_key
    ok = bool(par.notna().all() and par.isin(nom.index).all())
    if ok:
        P = nom.loc[par.values]
        same = ((P.config.values == E.config.values) & (P.surface.values == E.surface.values) & (P.phase_block.values == E.phase_block.values) & (P.sigma.values == E.sigma.values)
                & (P.rep.values == E.rep.values) & np.array([(a == b) for a, b in zip(P.seed.values, E.seed.values)]))
        ok = bool(same.all())
        per_parent = E.groupby('parent_forward_key').env.apply(lambda x: x.is_unique).all()
        ok &= bool(per_parent)
    add('D_pairing', ok, 'every environment case has exactly one new nominal parent with the same configuration, surface, block, sigma, rep and seed; one case per environment and parent',
        n_env_cases=len(E), n_parents=int(nom.shape[0]))
    add('D_new_nominals', bool((M.env == 'NOMINAL').sum() == 8 * 3 * 4 * 2 * 2 + 8), 'new phases get their own nominal images (384 core + 8 sigma 0); no old nominal replaces a new-phase control',
        n_nominal=int((M.env == 'NOMINAL').sum()))
    add('D_no_scientific_calls', K.CALLS == dict(render=0, decode=0, evaluation=0), 'no imaging / decoding / evaluation while planning (pattern rasters only)', calls=K.CALLS, aux=K.AUX)
    return rows


def time_estimate(M: pd.DataFrame) -> dict:
    R = K.load_json(K.MP3_OUT / 'run.json')['runs']
    tim = {}
    for run in R:
        for cid, t in (run.get('timing') or {}).items():
            tim[cid] = t
    rows = []
    for cid, t in tim.items():
        fam = 'line' if cid.startswith('L_') else 'point'
        for step, v in t.items():
            rows.append(dict(family=fam, step=step, s=float(v)))
    D = pd.DataFrame(rows); mean = D.groupby(['family', 'step']).s.mean().to_dict()
    nimg = M.family.value_counts().to_dict()
    est = dict(render=sum(nimg.get(f, 0) * mean.get((f, 'render'), 0.0) for f in nimg),
               decode=nimg.get('point', 0) * (mean.get(('point', 'decode:D0'), 0) + mean.get(('point', 'decode:DF'), 0)) + nimg.get('line', 0) * mean.get(('line', 'decode:P4'), 0),
               evaluation=nimg.get('point', 0) * (mean.get(('point', 'eval:D0'), 0) + mean.get(('point', 'eval:DF'), 0)) + nimg.get('line', 0) * mean.get(('line', 'eval:P4'), 0))
    mp3_cases = len(tim)
    return dict(source=f'MODEL-PREP-03 run.json timing of {mp3_cases} real cases (single process; includes the sigma-0 and sigma-40 mix of that pilot)',
                mean_s={f'{a}|{b}': round(v, 3) for (a, b), v in mean.items()}, images=nimg,
                estimate_min={k: round(v / 60.0, 1) for k, v in est.items()}, total_compute_min=round(sum(est.values()) / 60.0, 1),
                mp3_wall_min_for_512=41.7, scaled_wall_min=round(41.7 * len(M) / 512.0, 1),
                analysis_figures='not timed in MODEL-PREP-03 (stage durations were not recorded); expected minutes, not hours - to be measured at the entry check',
                note='estimate only; the real entry check must measure it and does not depend on this number')


def analysis_rules() -> dict:
    return dict(
        units='reports per surface x sigma x environment x configuration; the three fixed surfaces are not random repetitions',
        contrasts=dict(F1='environment - own new nominal parent per configuration (same surface, block, sigma, rep, seed)',
                       F2='size (r1.5 - r1 at 8x4 / 8x6; w1.5 - w0.5 at 6.3 / 8) and spacing (8x6 - 8x4 at r1 / r1.5; 8 - 6.3 at w0.5 / w1.5) inside each environment, paired by surface, block, sigma, rep',
                       F3='difference in differences: F2 under environment - F2 under nominal', F4='GRAD_X / GRAD_Y - UNIFORM (same block)', F5='DF - D0 on the same image (bridge only)',
                       F6='PSF 1.5 transition: L_w1.5_s8 - L_w0.5_s8 at PSF 1 / 1.5 / 2 / 3 (same block); no claim of a continuous critical value'),
        aggregation='per phase block: mean of the two noise repetitions first; then the 4-block mean, range, per-block sign and leave-one-block-out (4 means); no bootstrap / significance tests; '
                    'points along lines are never independent samples; merges strictly one-to-one on keys incl. block and rep (no cartesian products)',
        metrics=('glob_Ctau', 'loc_Ctau', 'glob_pt_rmse', 'glob_pt_p95', 'n_valid', 'n_accepted', 'c', 'w', 'u', 'w_share', 'u_share', 'glob_Cvalid', 'sup_surf_rmse'),
        definitions='zero denominators undefined; legal no-output: Ctau 0, errors undefined, kept in every denominator; complete line identity (A4: clipped lit lines included)',
        risk='wrong ids per image and per true line / segment against the own parent; activation (parent w = 0 -> w > 0) listed per image; batch totals never replace the per-image list',
        reproduction='an ordering / sign is "reproduced" only if the 4-block mean has the pilot sign AND at least 3 of 4 blocks agree AND leave-one-block-out keeps the sign; '
                     'otherwise "not consistently reproduced / limited evidence"; near-zero differences (|mean| < the block range) are reported as such; losers and counterexamples listed',
        thresholds='existing engineering thresholds (tau 0.15 mm, d 6 mm, gross 1 mm, vote 0.9, window [-16, 4] mm) unchanged; no threshold is tuned to pass',
        diagnostics='received light, noise-free signal peak, decoder thresholds / rejection / ambiguity counts reused from the existing fields; fixed-domain mean != received light',
        pilot_use='the 520 pilot images and the old nominal controls are development material (explanation, figures); never pooled into the validation statistics')


def figure_plan() -> list:
    return [dict(no=1, name='Ranking / performance under blur (PSF 1 / 1.5 / 2 / 3)', content='global Ctau per configuration, per surface x sigma, 4 block means + range; lines incl. PSF 1.5'),
            dict(no=2, name='Dot size-spacing conditional curves', content='F2 / F3 for dots per environment; per-block markers'),
            dict(no=3, name='Uniform vs gradients (paired losses)', content='F1 (UNIFORM - nominal) and F4 (GRAD - UNIFORM), global and local, with received-light ratio as covariate panel'),
            dict(no=4, name='Risk vs coverage', content='w / (c + w) and absolute w per image vs Cvalid; activation cases marked; dots and lines side by side'),
            dict(no=5, name='Typical topography (pre-selected)', content='rule: M10, block 0, sigma 40, rep 1, D_r1_p8x4 and L_w1.5_s8, nominal / BLUR3 / UNIFORM / GRAD_Y; chosen before the run'),
            dict(no=6, name='Failure case (rule-based, post hoc label)', content='rule: the line image with the largest w increase over its own parent (ties: smallest sha256 of the case key); '
                                                                              'labelled "post hoc failure diagnosis"; no frequency claim')]


def future_entry_checks() -> list:
    return ['the strict gate mp4_gate.final_status (mode run) on the raw ledger; failed / interrupted / aborted / unclosed refuse completion (C drills of this round)',
            'decoder inputs for unregistered new-phase rasters: an input override of the registered executor (RealQ / rdy_inputs route) that is proven equivalent on the 16 '
            'MODEL-PREP-03 rasters (raster, pattern, peak, centres / line family) and on stored MODEL-PREP-03 images (identical decode / evaluation arrays) BEFORE any new image',
            'recompute every planned forward key from the frozen manifest and compare; any difference stops the run',
            'freeze of forward / decode / evaluation layers equal to the MODEL-PREP-03 freeze (reused fingerprints); new runner layers frozen separately',
            'noise: regenerated noise array of the first images equals default_rng(seed + 104729) for the block seed; sigma 20 / 40 of one block share the base field',
            'true PSF / field / noise / truth never reach the decoder (whitelisted design row only)',
            'real entry: first images of each family / environment (incl. PSF 1.5 and sigma 0) checked like MODEL-PREP-03 C2 before the remaining images; counted in the budget',
            'caps: 2450 forward calls, 3626 decodes (D0 1176, DF 1176, P4 1274), 3626 evaluations; failures counted, no unbounded retries',
            'analysis: one-to-one merges incl. block and rep; completeness 3626 rows; reproduction rule as frozen; figures by the frozen selection rules']
