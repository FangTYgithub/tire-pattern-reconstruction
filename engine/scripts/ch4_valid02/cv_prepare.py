# Select validation phases and register the complete candidate matrix.

from __future__ import annotations
import sys, json, time, glob, itertools, hashlib, sqlite3, subprocess, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import cv_common as V

G, C = V.G, V.C
import cb_adapter as A
REG = V.OUT / 'registration.json'
MAN = V.OUT / 'manifest.csv'
KNOWN = V.OUT / '_known_keys.csv.gz'
AN_OLD = V.BRIDGE / 'analysis'
MAIN = dict(d_mm=6.0, tau_mm=0.15, floor=0.5)
LABELS = {
    'kept': 'Original recommendation and near-optimal range reproduced: original representative admissible and within 2 pp of the best admissible same-family candidate',
    'usable_reorder': 'Usability reproduced, ranking changed: original representative admissible but more than 2 pp behind',
    'not_confirmed': 'Recommendation not confirmed: original representative not admissible; failing gate and margin listed',
    'fail_repeated': 'Failure region reproduced: no admissible candidate originally or in the new phases',
    'boundary_changed': 'Admissibility boundary changed: no admissible candidate originally, admissible candidate(s) in the new phases',
}
FIGURE_RULES = dict(block='V1', rep=1, surfaces=['M10_curv', 'M24_curv'], panels={
    'F6': dict(env=['NOMINAL'], sigma=[20], configs=['D_r1_T8x4', 'D_r1_p8x4', 'D_r1_p8x6', 'L_w0.5_s8']),
    'F7': dict(env=['BLUR2'], sigma=[40], configs=['D_r1_T8x4', 'D_r0.75_p8x4', 'D_r0.75_p8x6', 'D_r1_p8x6']),
    'F8': dict(env=['NOMINAL'], sigma=[60, 80], configs=['D_r0.75_p8x4', 'D_r0.75_p8x6', 'D_r1_p8x6']),
    'F9': dict(env=['BLUR2_5', 'BLUR3'], sigma=[20], configs=['L_w0.5_s8', 'L_w1_s8', 'L_w1.25_s8', 'L_w1.5_s8'])},
    note='fixed before any new image; no-output panels are shown as they are; no replacement by successful cases; '
         'at most one additional diagnostic board, chosen mechanically by the largest gate deviation, if a new failure needs to be shown')
KEY_PAIRS = [('D_r1_p8x4', 'D_r1_T8x4'), ('D_r1_p8x6', 'D_r1_T8x4'), ('D_r1_p8x4', 'D_r0.75_p8x4'), ('D_r0.75_p8x4', 'D_r0.75_p8x6'),
             ('D_r1_p8x4', 'D_r1_p8x6'), ('D_r1.5_p8x4', 'D_r1.5_p8x6'), ('L_w0.5_s8', 'L_w1.25_s8'), ('L_w0.5_s8', 'L_w1.5_s8'),
             ('L_w0.5_s8', 'L_w0.5_s10'), ('L_w1_s8', 'L_w1_s10')]
FOURCELLS = {'dot_r1_r0.75': dict(a='D_r1_p8x4', b='D_r1_p8x6', c='D_r0.75_p8x4', d='D_r0.75_p8x6'),
             'dot_r1_r1.5': dict(a='D_r1_p8x4', b='D_r1_p8x6', c='D_r1.5_p8x4', d='D_r1.5_p8x6'),
             'line_w0.5_w1': dict(a='L_w0.5_s8', b='L_w0.5_s10', c='L_w1_s8', d='L_w1_s10')}
PAIR_KEYS = ['surface', 'env', 'sigma', 'phase_block', 'rep', 'seed', 'method']
NEAR_FLOOR_PP = 2.0


def sha(p):
    return C.sha(p)


def recs(df):
    return [G.rowdict(r) for r in df.to_dict('records')]


# Record candidate recommendations and expected comparisons before validation.
def predictions():
    sys.path.insert(0, str(V.HOME / 'scripts/ch4_bridge01'))
    import cb_analyse as CA
    ch = C.csv(AN_OLD / 'choices_all.csv'); sc = pd.read_csv(AN_OLD / 'config_scores.csv.gz')
    R = pd.read_csv(AN_OLD / 'results_all.csv.gz', low_memory=False)
    G.require(len(ch) == 1920 and len(R) == 22440 and R.config.nunique() == 17, 'Old prediction tables incomplete')
    src = {str(p.resolve()): sha(p) for p in [AN_OLD / 'choices_all.csv', AN_OLD / 'config_scores.csv.gz', AN_OLD / 'paired_effects.csv', AN_OLD / 'fourcell.csv',
                                               AN_OLD / 'results_all.csv.gz', AN_OLD / 'sensitivity_all.csv.gz', AN_OLD / 'surfaces_all.csv.gz', V.BRIDGE / 'protocol.md',
                                               V.BRIDGE / 'freeze.json', V.BRIDGE / 'holdout_registration.json']}
    old_choices = recs(ch[['family', 'env', 'sigma', 'd_mm', 'tau_mm', 'floor', 'qualified', 'near_optimal', 'preferred', 'best_score', 'pref_score', 'status']])
    main = ch[(ch.d_mm == 6) & (ch.tau_mm == .15) & (ch.floor == .5)]
    scm = sc[(sc.d_mm == 6) & (sc.tau_mm == .15)]
    eff = pd.concat([CA.paired_effect(R, a, b) for a, b in KEY_PAIRS], ignore_index=True)
    effm = eff.groupby(['left', 'right', 'env', 'sigma']).agg(mean_delta=('mean_delta', 'mean'), surf_min=('mean_delta', 'min'), surf_max=('mean_delta', 'max'),
                                                             blocks_pos=('blocks_pos', 'sum'), blocks=('blocks', 'sum')).reset_index()
    def J(cfgs, s=None):
        q = scm[scm.config.isin(cfgs)]
        return recs((q if s is None else q[q.sigma.isin(s)])[['config', 'env', 'sigma', 'J', 'safe']])
    c1 = dict(statement='T8x4 has a higher main-requirement J than R8x4 and R8x6 at low noise (sigma 0 / 20) across the 10 environments',
              J=J(['D_r1_T8x4', 'D_r1_p8x4', 'D_r1_p8x6']),
              paired=recs(effm[effm.right == 'D_r1_T8x4']))
    c2 = dict(statement='sigma 40-60: admissible set, near-optimal set and representative among small dots, sparse spacing and the staggered lattice',
              choices=recs(ch[(ch.family == 'point') & ch.sigma.isin([40, 60]) & (ch.d_mm == 6) & (ch.tau_mm == .15)]),
              J=J(G.DOTS, [40, 60]))
    c3 = dict(statement='lines: wider lines are selected as blur increases; line width / pitch choice under illumination loss',
              choices=recs(main[main.family == 'line']), J=J(G.LINES),
              paired=recs(effm[effm.left.str.startswith('L_')]))
    fam = []
    for (e, s, d, tau, fl), t in ch.groupby(['env', 'sigma', 'd_mm', 'tau_mm', 'floor']):
        p = t[t.family == 'point'].iloc[0]; l = t[t.family == 'line'].iloc[0]
        bp = p.best_score if pd.notna(p.best_score) else None; bl = l.best_score if pd.notna(l.best_score) else None
        win = 'none' if bp is None and bl is None else ('point' if bl is None or (bp is not None and bp > bl) else 'line')
        fam.append(dict(env=e, sigma=s, d_mm=d, tau_mm=tau, floor=fl, point_best=bp, line_best=bl, winner=win, point_pref=p.preferred, line_pref=l.preferred))
    c4 = dict(statement='relaxing tau from 0.15 to 0.30 mm moves the family choice from dots to lines at low noise', family_best=[G.rowdict(x) for x in fam])
    four = {}
    for name, cells in FOURCELLS.items():
        f = CA.fourcell(R, cells, G.fam_of(cells['a']), name)
        four[name] = recs(f.groupby(['env', 'sigma']).agg(DiD=('DiD', 'mean'), DiD_surf_min=('DiD', 'min'), DiD_surf_max=('DiD', 'max'),
                                                         size_low=('size_or_width_at_low', 'mean'), size_high=('size_or_width_at_high', 'mean')).reset_index())
    inp = A.inputs(A.platform()['T']['D_r0.75_p8x6']['row'], 'point', A.phases()['point'][0])
    c5 = dict(statement='small-dot sparsification gain limited by the peak cap; size-spacing interaction differs for small and large dots; line interaction changes sign with environment',
              fourcell=four, N2_peak_cap=dict(lit_px=float(inp['lit_px']), peak=float(inp['peak']), integral_share_of_Q=float(np.asarray(inp['pattern'], np.float64).sum()) / G.Q))
    c6 = dict(statement='main requirement: no admissible candidate at sigma 80 and 100', choices=recs(main[main.sigma.isin([80, 100])][['family', 'env', 'sigma', 'status', 'preferred']]))
    near = []
    for r in ch.to_dict('records'):
        r = G.rowdict(r); band = NEAR_FLOOR_PP / 100
        if r['preferred'] is not None and r['pref_score'] - r['floor'] <= band:
            near.append(dict(r, kind='admissible_near_floor', margin_pp=100 * (r['pref_score'] - r['floor'])))
        q = sc[(sc.family == r['family']) & (sc.env == r['env']) & (sc.sigma == r['sigma']) & (sc.d_mm == r['d_mm']) & (sc.tau_mm == r['tau_mm']) & sc.safe & (sc.J < r['floor'])]
        if len(q) and r['floor'] - q.J.max() <= band:
            b = q.sort_values(['J', 'config'], ascending=[False, True]).iloc[0]
            near.append(dict(r, kind='safe_below_floor', config=b.config, J=float(b.J), margin_pp=100 * (float(b.J) - r['floor'])))
    c7 = dict(statement=f'conditions within {NEAR_FLOOR_PP:g} pp of the 50 % / 70 % floor (admissible representative just above, or a gate-safe candidate just below)', cases=near)
    claims = {'1_T8x4_low_noise': c1, '2_dot_sigma40_60': c2, '3_lines_blur_lighting': c3, '4_tau_family_switch': c4, '5_peak_cap_interactions': c5,
              '6_sigma80_100_fail': c6, '7_near_floor': c7}
    return src, old_choices, json.loads(C.canon(claims)), json.loads(C.canon(recs(effm)))


def used_material():
    import tr3_b2 as REGB
    import sl_lib as SLL
    reg = REGB.registry(); ps = reg['positions']; ls = reg['phases']
    pts = [(float(a), float(b)) for a, b in zip(pd.to_numeric(ps.u, errors='coerce'), pd.to_numeric(ps.v, errors='coerce')) if np.isfinite(a) and np.isfinite(b)]
    lns = list(pd.to_numeric(ls.phase, errors='coerce').dropna().astype(float))
    seeds = set(int(x) for x in reg['seeds']); rasters = set(map(str, reg['raster_keys'])); fks = set(); sources = [dict(path='tr3_b2.registry()', rows=len(ps) + len(ls))]
    grid = SLL.positions()
    files = sorted(set(glob.glob(str(G.PROJECT / 'outputs/**/*manifest*.csv'), recursive=True) + glob.glob(str(V.HOME / 'outputs/**/*manifest*.csv'), recursive=True)
                       + [str(G.BASE / 'sections_adapter/candidates_s1s5.csv')]))
    for f in files:
        p = Path(f)
        if p.resolve().is_relative_to(V.OUT.resolve()):
            continue
        hdr = list(pd.read_csv(p, nrows=0, encoding='utf-8-sig').columns)
        use = [c for c in hdr if c in ('phase_u', 'phase_v', 'u', 'v', 'phase_frac', 'phase_index', 'phase_label', 'seed', 'noise_seed', 'raster_key', 'forward_key')]
        if not use:
            continue
        t = pd.read_csv(p, usecols=use, low_memory=False, encoding='utf-8-sig'); n0 = (len(pts), len(lns))
        for uc, vc in (('phase_u', 'phase_v'), ('u', 'v')):
            if uc in t and vc in t:
                q = t[[uc, vc]].apply(pd.to_numeric, errors='coerce').dropna(); pts += list(q.itertuples(index=False, name=None))
        if 'phase_index' in t and 'phase_label' in t:
            lab = t.phase_label.astype(str); idx = pd.to_numeric(t.phase_index, errors='coerce')
            m = lab.str.match(r'^pos\d+$') & idx.notna()
            G.require((lab[m].str[3:].astype(int) == idx[m].astype(int)).all(), f'Position label / index mismatch in {p}')
            pts += [grid[i] for i in sorted(set(idx[m].astype(int)))]
        if 'phase_frac' in t:
            lns += list(pd.to_numeric(t.phase_frac, errors='coerce').dropna().astype(float))
        for col in ('seed', 'noise_seed'):
            if col in t:
                seeds |= set(int(x) for x in pd.to_numeric(t[col], errors='coerce').dropna())
        if 'raster_key' in t:
            rasters |= set(t.raster_key.dropna().astype(str))
        if 'forward_key' in t:
            fks |= set(t.forward_key.dropna().astype(str))
        new_p = np.asarray(pts[n0[0]:], float).reshape(-1, 2); new_l = np.asarray(lns[n0[1]:], float)
        sources.append(dict(path=str(p), sha256=sha(p), rows=len(t), points_added=len(new_p), line_phases_added=len(new_l),
                            point_range=[float(new_p.min()), float(new_p.max())] if len(new_p) else None, line_range=[float(new_l.min()), float(new_l.max())] if len(new_l) else None))
    P = np.unique(np.round(np.asarray(pts, float), 12), axis=0); Lp = np.unique(np.round(np.asarray(lns, float), 12))
    return P, Lp, seeds, rasters, fks, sources


def r_distance(c, used):

    d = c[:, None, :] - used[None, :, :]; d = (d + .5) % 1 - .5
    return np.hypot(d[..., 0], d[..., 1])


def t_distance(c, used):

    d = c[:, None, :] - used[None, :, :]; l0 = np.round(d[..., 1]); best = np.full(d.shape[:2], np.inf)
    for k in (-1, 0, 1):
        l = l0 + k; du = (d[..., 0] - .5 * l + .5) % 1 - .5; dv = d[..., 1] - l
        best = np.minimum(best, np.hypot(du, dv))
    return best


def pd_distance(c, used):

    c = np.atleast_2d(np.asarray(c, float)); used = np.atleast_2d(np.asarray(used, float))
    out = np.empty(len(c))
    for i in range(0, len(c), 256):
        out[i:i + 256] = np.minimum(r_distance(c[i:i + 256], used), t_distance(c[i:i + 256], used)).min(axis=1)
    return out


def l_distance(c, used):
    c = np.asarray(c, float).reshape(-1); used = np.asarray(used, float).reshape(-1)
    return np.abs((c[:, None] - used[None, :] + .5) % 1 - .5).min(axis=1)


# Select phase offsets and check their separation from existing cases.
def choose_phases(P, Lp, rasters):
    import mp4_formal as MF
    S = A.platform(); out = {}; audit = {}; ks = np.arange(1, 20001)
    for fam, cfgs in (('point', G.DOTS), ('line', G.LINES)):
        if fam == 'point':
            cand = np.column_stack([(.5 + ks / MF.RHO) % 1, (.5 + ks / MF.RHO ** 2) % 1]); minimum = .05
            dist = pd_distance(cand, P); dist0 = dist.copy()
            maximin = int((dist >= minimum).sum()) < 50
            rule = ('fallback before imaging: geometry maximin over the 20000-point golden sequence (0.05-cell rule infeasible)' if maximin
                    else 'legacy rule: first golden-sequence points (k ascending) at >= 0.05 cell from every used phase and from the accepted blocks')
        else:
            cand = ((.5 + ks * MF.SQ2M1) % 1 - .5); minimum = 1e-9; maximin = True
            dist = l_distance(cand, Lp); dist0 = dist.copy()
            rule = 'geometry maximin over the 20000-point sqrt(2)-1 sequence (periodic phase-fraction distance, all pitches)'
        order = list(np.argsort(-dist, kind='stable')) if maximin else [i for i in range(len(ks)) if dist[i] >= minimum]
        acc, reject = [], {}
        while order and len(acc) < V.N_PHASES:
            i = int(order.pop(0))
            if fam == 'point':
                ph = dict(u=float(cand[i, 0]), v=float(cand[i, 1]))
                used = np.vstack([P, [[a['u'], a['v']] for a in acc]]) if acc else P
                near = float(pd_distance([[ph['u'], ph['v']]], used)[0])
            else:
                ph = dict(phi=float(cand[i]))
                near = float(l_distance([ph['phi']], np.r_[Lp, [a['phi'] for a in acc]])[0])
            if near < minimum:
                reject['distance'] = reject.get('distance', 0) + 1; continue
            why = []; keys = {}
            for cfg in cfgs:
                inp = A.inputs(S['T'][cfg]['row'], fam, ph); keys[cfg] = inp['raster_key']
                if inp['raster_key'] in rasters:
                    why.append('seen_raster')
                if fam == 'line':
                    b = MF.build(S['T'][cfg]['row'], fam, ph)
                    if b['registered_without_lit']:
                        why.append('registered_without_lit')
                    MF._RAST.clear()
            if why:
                for w in set(why):
                    reject[w] = reject.get(w, 0) + 1
                continue
            acc.append(dict(block=len(acc), label=f'V{len(acc)}', k=int(ks[i]), **ph, min_distance_to_used_and_accepted=near, raster_keys=keys))
            if maximin:
                if fam == 'point':
                    dist = np.minimum(dist, pd_distance(cand, [[ph['u'], ph['v']]]))
                else:
                    dist = np.minimum(dist, l_distance(cand, [ph['phi']]))
                remaining = set(order); order = [int(j) for j in np.argsort(-dist, kind='stable') if int(j) in remaining]
        G.require(len(acc) == V.N_PHASES, f'Not enough novel {fam} phases; stop before imaging')
        out[fam] = acc
        audit[fam] = dict(rule=rule, metric=('min(R-torus, T-torus) distance in cell units' if fam == 'point' else 'periodic phase-fraction distance'), min_distance=minimum,
                          rejected=reject, qualifying_sequence_points=int((dist0 >= minimum).sum()),
                          configs=cfgs, used_count=int(len(P) if fam == 'point' else len(Lp)))

    pa = np.array([[a['u'], a['v']] for a in out['point']]); la = np.array([a['phi'] for a in out['line']])
    audit['point']['between_blocks_min'] = float(min(pd_distance(pa[i:i + 1], np.delete(pa, i, 0))[0] for i in range(len(pa))))
    audit['line']['between_blocks_min'] = float(min(l_distance(la[i:i + 1], np.delete(la, i))[0] for i in range(len(la))))
    return out, audit


def seeds_free(used):
    base = dict(V.SEED_BASE); log = {}
    for fam in base:
        while {base[fam] + 100 * s + 10 * b + r for s in range(3) for b in range(4) for r in (1, 2)} & used:
            base[fam] += 100000
        log[fam] = dict(base=base[fam], bumped=base[fam] != V.SEED_BASE[fam], seeds=sorted(base[fam] + 100 * s + 10 * b + r for s in range(3) for b in range(4) for r in (1, 2)))
    return base, log


def bridge_holdout_recheck(P):

    h = C.load(V.BRIDGE / 'holdout_registration.json')['phase_registry']['phases'].get('point', {}).get('phases', [])
    hp = np.array([[a['u'], a['v']] for a in h]) if h else np.zeros((0, 2))
    return [dict(block=a['block'], u=a['u'], v=a['v'], registered_min_distance=a['min_distance'],
                 r_torus_min=float(r_distance(hp[i:i + 1], P[~np.all(np.isclose(P, hp[i], atol=1e-12), axis=1)]).min()),
                 t_torus_min=float(t_distance(hp[i:i + 1], P[~np.all(np.isclose(P, hp[i], atol=1e-12), axis=1)]).min())) for i, a in enumerate(h)]


def register():
    C.install_guard(); t = time.time()
    G.require(not REG.exists(), 'registration.json already exists; predictions and phases are frozen once')
    A.platform()
    src, old_choices, claims, old_pairs = predictions()
    P, Lp, seeds, rasters, fks, sources = used_material()
    phases, audit = choose_phases(P, Lp, rasters)
    base, seedlog = seeds_free(seeds)
    rec = dict(task='CH4-VALID-02', created=time.time(), prediction_sources=src, old_choices=old_choices, old_key_pairs=old_pairs, claims=claims, labels=LABELS,
               analysis=dict(policy=C.POLICY, J='repetition means within phase -> 4 equal phase blocks -> per-surface mean; J = minimum over B_curv / M10_curv / M24_curv',
                             near_band_pp=2, near_floor_pp=NEAR_FLOOR_PP, floors=[.5, .7], d_mm=[5., 6.], tau_mm=[.1, .15, .2, .3], main=MAIN,
                             reference=dict(point='D_r1_p8x6', line='L_w0.5_s8'), selection='mp06d analysis.tables (frozen), applied to the 22440 new images alone',
                             gates='per image: point RMSE <= 0.30 mm, w = 0, u / (c + w + u) <= 5 %, defined metrics and exposure; no output = C_tau 0 and stays in the denominator',
                             key_pairs=KEY_PAIRS, fourcells=FOURCELLS, pair_keys=PAIR_KEYS, cross_family='condition-matched means only, no image pairing',
                             intervals='phase-block range (min / max of the four block means); no resampling of points or sigma levels'),
               phases=phases, phase_registry=dict(audit=audit, sources=sources, known_point_positions=int(len(P)), known_line_phases=int(len(Lp)),
                                                  known_seeds=len(seeds), known_rasters=len(rasters), known_forward_keys=len(fks),
                                                  bridge_holdout_point_recheck=bridge_holdout_recheck(P)),
               seed_base=base, seed_log=seedlog, seed_rule='seed = base[family] + 100 * surface index + 10 * block + rep; sigma 0 without noise draw',
               figure_rules=FIGURE_RULES, seconds=round(time.time() - t, 1))
    C.save(REG, rec)
    K = pd.concat([pd.DataFrame(dict(key=sorted(rasters), kind='raster')), pd.DataFrame(dict(key=sorted(fks), kind='forward'))], ignore_index=True)
    C.atomic(KNOWN, lambda q: K.to_csv(q, index=False, compression='gzip'))
    print(C.canon(dict(phases={f: [(round(a.get('u', a.get('phi')), 4), round(a.get('v', 0.0), 4), round(a['min_distance_to_used_and_accepted'], 4)) for a in v] for f, v in phases.items()},
                       rules={f: audit[f]['rule'][:40] for f in audit}, seeds=base, known=(len(P), len(Lp), len(seeds), len(rasters), len(fks)), seconds=rec['seconds'])))


def known_keys():
    k = pd.read_csv(KNOWN, dtype=str)
    return set(k[k.kind == 'raster'].key), set(k[k.kind == 'forward'].key)


def entry_rows(M):
    pick = []
    def key(cfg, sf, b, s, rep, env):
        return V.case_key(cfg, sf, b, s, rep if s > 0 else 0, env)
    for i, cfg in enumerate(G.ALL):
        pick.append(key(cfg, G.SURFS[i % 3], i % 4, G.SIGMAS[i % 6], 1, G.ENVS[i % 10]))
        pick.append(key(cfg, G.SURFS[(i + 1) % 3], (i + 2) % 4, G.SIGMAS[(i + 3) % 6], 2, G.ENVS[(i + 3) % 10]))
    for cfg in ('D_r1_p8x4', 'D_r1_T8x4'):
        for rep in (1, 2):
            pick.append(key(cfg, 'M24_curv', 1, 40, rep, 'NOMINAL'))
    k = 0
    while len(dict.fromkeys(pick)) < 48:
        pick.append(key(G.ALL[(5 * k + 3) % 17], G.SURFS[(k + 2) % 3], (k + 1) % 4, G.SIGMAS[(k + 5) % 6], 1 + k % 2, G.ENVS[(3 * k + 7) % 10])); k += 1
    pick = list(dict.fromkeys(pick))[:48]
    E = M.set_index('case_key', drop=False).loc[pick]
    cov = dict(configs=int(E.config.nunique()), surfaces=int(E.surface.nunique()), sigmas=sorted(int(x) for x in E.sigma.unique()), routes=sorted(E.forward_route.unique().tolist()),
               peak_capped=int((E.peak >= 1).sum()), integral_limited=int((E.peak < 1).sum()), envs=int(E.env.nunique()), families=E.family.value_counts().to_dict())
    G.require(len(E) == 48 and cov['configs'] == 17 and cov['surfaces'] == 3 and len(cov['sigmas']) == 6 and len(cov['routes']) == 2 and cov['peak_capped'] > 0
              and cov['integral_limited'] > 0, f'Entry coverage incomplete {cov}')
    return set(pick), cov


def gate(M, reg, entry=None):
    need = set(C.IDENTITY + ['action', 'source_pack', 'entry_subset', 'integral', 'peak', 'lit_px', 'rays_per_pixel', 'field_json', 'global_domain_sha256',
                             'evaluation_source_sha256', 'forward_route', 'raster_key'])
    G.require(need.issubset(M.columns), 'Manifest missing columns: ' + str(need - set(M.columns)))
    want = {V.case_key(c, sf, b, s, r, e) for c, e, sf, b, s in itertools.product(G.ALL, G.ENVS, G.SURFS, range(4), G.SIGMAS) for r in ([0] if s == 0 else [1, 2])}
    G.require(len(M) == V.CAP and set(M.case_key) == want, 'Validation matrix not closed (17 x 10 x 3 x 4 x 11)')
    G.require(not M.case_key.duplicated().any() and not M.forward_key.duplicated().any(), 'Duplicate case / forward keys')
    G.require((M.family == 'point').sum() == 13200 and (M.family == 'line').sum() == 9240 and (M.sigma == 0).sum() == 2040
              and M.groupby('sigma').size().drop(0).eq(4080).all() and M.config.value_counts().eq(1320).all() and M.env.value_counts().eq(2244).all(), 'Count closure')
    G.require(M.action.eq('new').all(), 'Every validation input is new')
    base = reg['seed_base']; ph = reg['phases']
    for r in M.to_dict('records'):
        r = G.rowdict(r); fam = G.fam_of(r['config']); b = int(str(r['phase_block'])[1:])
        G.require(r['family'] == fam and r['method'] == G.method_of(r['config']) and r['rays_per_pixel'] == 1, 'Family / method / sampling')
        G.require(r['psf_true_px'] == G.PSF[r['env']] and json.loads(r['field_json']) == G.FIELDS[r['env']], 'Environment binding')
        G.require(r['rep'] in ([0] if r['sigma'] == 0 else [1, 2]), 'Repetition label')
        G.require(r['seed'] == (None if r['sigma'] == 0 else G.seed_of(fam, r['surface'], b, r['rep'], base)), 'Seed rule')
        G.require(abs(r['peak'] - min(1.0, G.Q / r['lit_px'])) <= 1e-9 and 0 < r['peak'] <= 1 and r['integral'] <= G.Q * (1 + 2 ** -23), 'Resource rule')
        G.require((r['forward_route'] == 'r_forward') == (r['env'] == 'GRAD_Y_REV'), 'Forward route')
        G.require(r['case_key'] == V.case_key(r['config'], r['surface'], b, r['sigma'], r['rep'], r['env']), 'Case key / fields')
        p = ph[fam][b]
        G.require((fam == 'point' and r['phase_u'] == p['u'] and r['phase_v'] == p['v']) or (fam == 'line' and r['phase_frac'] == p['phi']), 'Phase binding')
        G.require(r['raster_key'] == p['raster_keys'][r['config']], 'Raster binding')
    seen_r, seen_f = known_keys()
    G.require(not M.raster_key.isin(seen_r).any(), 'A validation raster was seen before')
    G.require(not M.forward_key.isin(seen_f).any(), 'A validation input coincides with an executed input')
    used = {int(s) for s in pd.to_numeric(M.seed, errors='coerce').dropna()}
    G.require(used == {s for f in base for s in reg['seed_log'][f]['seeds']}, 'Seed set differs from the registration')
    if entry is not None:
        G.require(set(M[M.entry_subset].case_key) == entry and int(M.entry_subset.sum()) == 48, 'Entry subset mismatch')


def manifest():
    C.install_guard(); t = time.time(); reg = C.load(REG); A.platform()
    sys.path.insert(0, str(V.HOME / 'scripts/ch4_bridge01'))
    import cb_prepare as P
    tmpl = C.csv(V.BRIDGE / 'manifest_main.csv').iloc[0]
    rows = []
    for cfg, env, sf, b, s in itertools.product(G.ALL, G.ENVS, G.SURFS, range(4), G.SIGMAS):
        fam = G.fam_of(cfg); p = reg['phases'][fam][b]
        ph = dict(block=b, k=p['k'], **({'u': p['u'], 'v': p['v']} if fam == 'point' else {'phi': p['phi']}))
        for rep in ([0] if s == 0 else [1, 2]):
            rows.append(A.plan_row(cfg, sf, b, s, rep, env, part='CV02_VALIDATION', ph=ph, seed_base=reg['seed_base'], block_label=f'V{b}'))
    M = pd.DataFrame(rows)
    M['action'] = 'new'; M['source_pack'] = ''; M['source_pack_sha256'] = ''; M['source_batch'] = ''
    M['method_bundle_sha256'] = P.bundle(); M['global_domain_sha256'] = tmpl.global_domain_sha256; M['evaluation_source_sha256'] = tmpl.evaluation_source_sha256
    M['module'] = np.where(M.env.isin(['BLUR1_5', 'BLUR2', 'BLUR2_5', 'BLUR3']), 'blur', np.where(M.env == 'NOMINAL', 'shared_nominal', 'lighting'))
    M['execution_authorised'] = True; M['entry_subset'] = False
    E, cov = entry_rows(M); M['entry_subset'] = M.case_key.isin(E)
    gate(M, reg, E)
    M = M.sort_values(['config', 'env', 'surface', 'phase_block', 'sigma', 'rep']).reset_index(drop=True)
    C.save_csv(MAN, M)
    gate(C.csv(MAN), reg, E)
    rec = dict(sha256=sha(MAN), rows=len(M), entry=cov, entry_keys=sorted(E), registration_sha256=sha(REG), bundle=P.bundle(),
               by_config=M.config.value_counts().to_dict(), routes=M.forward_route.value_counts().to_dict(), peak_capped=int((M.peak >= 1).sum()),
               seconds=round(time.time() - t, 1))
    C.save(V.OUT / 'registration_manifest.json', rec)
    print(C.canon({k: rec[k] for k in ('rows', 'entry', 'routes', 'peak_capped', 'seconds')}))


def replay():
    C.install_guard(); S = A.platform(); PL = S['p']['PL']
    sys.path.insert(0, str(V.HOME / 'scripts/ch4_bridge01'))
    import cb_prepare as P
    old = P.old_manifest(); bm = C.csv(V.BRIDGE / 'manifest_main.csv')
    def first(df, sigma, **kw):
        for s2 in [sigma] + [x for x in G.SIGMAS if x != sigma]:
            q = df[df.sigma == s2]
            for k, v in kw.items():
                q = q[q[k] == v]
            if len(q):
                return G.rowdict(q.sort_values('case_key').iloc[0].to_dict())
        raise C.Stop(f'No stored case {kw}')
    newold = old[old.action == 'new']
    picks = [('old', first(newold, 60.0, family='point', env='BLUR2', module='blur')),
             ('old', first(newold, 20.0, family='line', env='GRAD_Y_REV', module='lighting')),
             ('old', first(newold, 100.0, family='point', env='UNIFORM', module='lighting')),
             ('old', first(newold, 0.0, family='line', env='BLUR3', module='blur')),
             ('bridge', first(bm, 0.0, config='D_r0.75_p8x6', env='NOMINAL')),
             ('bridge', first(bm, 80.0, config='D_r1_T8x4', env='GRAD_Y_REV')),
             ('bridge', first(bm, 40.0, config='L_w0.5_s10', env='BLUR2')),
             ('bridge', first(bm, 100.0, config='L_w1_s10', env='NOMINAL'))]
    cases = []
    for src, r in picks:
        pr = A.prep(r)
        if src == 'old':
            p = P.stored_pack(r)
        else:
            p = V.BRIDGE / 'main' / 'data' / (hashlib.sha256(r['case_key'].encode()).hexdigest()[:24] + '.npz')
            with sqlite3.connect(V.BRIDGE / 'main' / 'store.sqlite') as con:
                h = con.execute("SELECT sha FROM cases WHERE ck=? AND status='complete'", (r['case_key'],)).fetchone()
            G.require(h is not None and sha(p) == h[0], 'Bridge pack checksum')
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z['_meta'])); arr = {k: z[k] for k in z.files if k != '_meta'}
        m = r['method']; raw = {k[len(m) + 7:]: v for k, v in arr.items() if k.startswith(m + '__raw__')}
        evs = {k[len(m) + 6:]: v for k, v in arr.items() if k.startswith(m + '__ev__')}
        ev, rec, _ = PL.evaluate(pr, m, raw, meta['decode_meta'])
        diff_ev = sorted(k for k in evs if not (k in ev and np.asarray(ev[k]).shape == evs[k].shape and np.array_equal(np.asarray(ev[k]), evs[k], equal_nan=evs[k].dtype.kind == 'f')))
        sens = G.B.multi_eval({k: np.asarray(v) for k, v in ev.items()}, r['surface'])
        sd = max(abs(a['Ctau'] - b['Ctau']) + abs(a['Cvalid'] - b['Cvalid']) for a, b in zip(sens, meta['sensitivity']))
        c = dict(source=src, case_key=r['case_key'], config=r['config'], env=r['env'], sigma=r['sigma'], peak=r['peak'], route=r['forward_route'],
                 inputs_equal=C.canon(pr['inputs']) == C.canon(meta['inputs']), input_key_equal=pr['input_key'] == meta['input_key'],
                 forward_key_equal=pr['forward_key'] == meta['forward_key'], ev_arrays_differing=diff_ev, metrics_equal=C.canon(rec['metrics']) == C.canon(meta['metrics']),
                 sensitivity_maxdiff=sd, state=rec['state'])
        c['ok'] = bool(c['inputs_equal'] and c['input_key_equal'] and c['forward_key_equal'] and not diff_ev and c['metrics_equal'] and sd < 1e-12); cases.append(c)
    C.save(V.OUT / 'prep_replay.json', dict(cases=cases, ok=all(c['ok'] for c in cases), note='stored decoder arrays re-evaluated; no render, no decode'))
    print(C.canon([(c['config'], c['env'], c['sigma'], c['route'], c['ok']) for c in cases]))
    G.require(all(c['ok'] for c in cases), 'Replay of stored results failed')


# Test paired statistics, input checks and interrupted result storage.
def drills():
    import cv_run as RUN
    import run as ENGINE
    C.install_guard(); res = []; M = C.csv(MAN); reg = C.load(REG)
    r0 = G.rowdict(M[(M.sigma == 20) & (M.family == 'point')].iloc[0]); r00 = G.rowdict(M[(M.sigma == 0) & (M.family == 'point')].iloc[0])
    def good(r, n=40):
        return dict({k: r[k] for k in C.IDENTITY}, state='ok', glob_Ctau=.5, glob_Cvalid=.6, glob_pt_rmse=.05, glob_pt_p95=.1, glob_n_valid=n, n_valid=n, n_accepted=n, c=n, w=0, u=0)
    def expect_stop(name, fn):
        try:
            fn()
        except (C.Stop, ValueError, KeyError) as e:
            res.append(dict(drill=name, refused=True, reason=str(e)[:160])); return
        res.append(dict(drill=name, refused=False)); raise C.Stop('Drill accepted corrupted input: ' + name)
    def expect_ok(name, fn):
        out = fn(); res.append(dict(drill=name, ok=True)); return out
    exp = pd.DataFrame([r0, r00])
    expect_ok('legal no-output row (C_tau 0, errors undefined) accepted', lambda: C.metric_gate(pd.DataFrame([
        dict(good(r0, 0), glob_Ctau=0.0, glob_Cvalid=0.0, glob_pt_rmse=None, glob_pt_p95=None, state='no_output'), good(r00)]), exp))
    expect_stop('missing row', lambda: C.metric_gate(pd.DataFrame([good(r0)]), exp))
    expect_stop('duplicate rep', lambda: C.metric_gate(pd.DataFrame([good(r0), good(r0), good(r00)]), exp))
    expect_stop('missing identity count', lambda: C.metric_gate(pd.DataFrame([dict(good(r0), c=None), good(r00)]), exp))
    expect_stop('missing field (column)', lambda: C.metric_gate(pd.DataFrame([good(r0), good(r00)]).drop(columns='w'), exp))
    expect_stop('no output with positive coverage', lambda: C.metric_gate(pd.DataFrame([dict(good(r0, 0), glob_pt_rmse=None, glob_pt_p95=None, state='no_output'), good(r00)]), exp))
    expect_stop('error filled with 0 for no output', lambda: C.metric_gate(pd.DataFrame([dict(good(r0, 0), glob_Ctau=0.0, glob_Cvalid=0.0, glob_pt_rmse=0.0, glob_pt_p95=0.0, state='no_output'), good(r00)]), exp))
    expect_stop('technical state as completion', lambda: C.metric_gate(pd.DataFrame([dict(good(r0), state='technical_stop'), good(r00)]), exp))
    expect_stop('sigma 0 duplicated as two reps', lambda: RUN.result_gate(pd.DataFrame([good(r00), dict(good(r00), rep=1, case_key=r00['case_key'].replace('|r0|', '|r1|'))]), exp.iloc[[1]]))
    q = M[(M.surface == 'M24_curv') & (M.env == 'NOMINAL') & (M.sigma == 40) & (M.phase_block == 'V1') & M.config.isin(['D_r1_p8x4', 'D_r1_T8x4'])]
    Rq = pd.DataFrame([good(G.rowdict(x)) for x in q.to_dict('records')])
    pr = RUN.pair(Rq, Rq, 'D_r1_p8x4', 'D_r1_T8x4')
    G.require(len(pr) == 2 and (pr.seed_left if 'seed_left' in pr else pr.seed).notna().all(), 'Pairing drill must give exactly two pairs')
    res.append(dict(drill='2 candidates x 2 reps -> exactly 2 pairs', ok=True, pairs=len(pr)))
    dup = pd.concat([Rq, Rq[Rq.config == 'D_r1_p8x4'].iloc[:1]])
    expect_stop('duplicated pairing key (cartesian product)', lambda: RUN.pair(dup, dup, 'D_r1_p8x4', 'D_r1_T8x4'))
    expect_stop('cross-family pairing refused (unequal sets)', lambda: RUN.pair(Rq, Rq.assign(config='L_w0.5_s8').iloc[:1], 'D_r1_p8x4', 'L_w0.5_s8'))

    seen = []
    def after(r, o):
        seen.append(o['ok'])
    expect_stop('worker exception stops dispatch', lambda: ENGINE.bounded_dispatch([dict(case_key=f'D{i}', bad=(i == 1)) for i in range(4)], RUN._drill_job,
                                                                                lambda r: None, after, lambda r: None, 1))
    G.require(seen == [True, False], f'Dispatch continued after the failure {seen}')

    cmd = [sys.executable, '-X', 'utf8', '-B', str(V.HERE / 'cv_run.py'), 'entry', '--workers', '1']
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    G.require(p.returncode != 0 and not (V.OUT / 'main' / 'store.sqlite').exists(), f'Entry command must fail before the freeze (rc {p.returncode})')
    res.append(dict(drill='entry exits non-zero on a technical error, nothing rendered', ok=True, returncode=p.returncode, tail=p.stderr.strip().splitlines()[-1][:160]))

    sys.path.insert(0, str(V.HOME / 'scripts/ch4_bridge01'))
    import cb_prepare as PB
    root = V.OUT / 'drill'; st = C.Store(root); fid = 'DRILL'
    with st.connect() as con:
        con.execute('DELETE FROM cases'); con.execute('DELETE FROM events')
    for f in (root / 'data').glob('*'):
        f.unlink()
    rr = dict(r0, case_key='DRILL|' + r0['case_key']); arrays = PB.synthetic_arrays(rr)
    meta = dict(case_key=rr['case_key'], forward_key=rr['forward_key'], method=rr['method'], result=dict(good(rr, 3), case_key=rr['case_key'], glob_pt_p95=.05))
    st.reserve(rr, fid); st.commit(rr, fid, meta, arrays)
    G.require(st.resume(rr, fid) is not None, 'Resume failed'); res.append(dict(drill='resume committed pack without re-render', ok=True))
    expect_stop('second reservation of a committed case', lambda: st.reserve(rr, fid))
    expect_stop('stale freeze binding on resume', lambda: st.resume(rr, 'OTHER'))
    expect_stop('stale row binding on resume (seed changed)', lambda: st.resume(dict(rr, seed=rr['seed'] + 1), fid))
    pk = st.path(rr['case_key']); b = pk.read_bytes(); pk.write_bytes(b[:-7] + b'corrupt')
    expect_stop('corrupted pack', lambda: st.resume(rr, fid))
    rr2 = dict(rr, case_key=rr['case_key'] + '|E'); st.reserve(rr2, fid); st.fail(rr2['case_key'], 'drill: worker exception')
    expect_stop('technical stop is not completion', lambda: st.resume(rr2, fid))
    rr3 = dict(rr, case_key=rr['case_key'] + '|K'); st.reserve(rr3, fid)
    expect_stop('interrupted reservation without pack is not completion', lambda: st.resume(rr3, fid))
    bad = M.copy(); bad.loc[bad.index[5], 'seed'] = 1
    expect_stop('manifest drift (seed)', lambda: gate(bad, reg))
    bad = M.copy(); bad.loc[bad.index[9], 'phase_u'] = 0.123
    expect_stop('manifest drift (phase)', lambda: gate(bad, reg))
    expect_stop('manifest missing row', lambda: gate(M.drop(M.index[7]), reg))
    expect_stop('manifest duplicated row', lambda: gate(pd.concat([M, M.iloc[[3]]]), reg))
    C.save(V.OUT / 'prep_drills.json', dict(synthetic=True, drills=res, n=len(res), all_ok=all(x.get('refused') or x.get('ok') for x in res)))
    print(C.canon(dict(drills=len(res), refused=sum(bool(x.get('refused')) for x in res), ok=sum(bool(x.get('ok')) for x in res))))


def memory():

    import ctypes
    class MS(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys, m.ullAvailPhys


def freeze():
    import cv_run as RUN
    import shutil
    C.install_guard(); reg = C.load(REG); M = C.csv(MAN)
    E, _ = entry_rows(M); gate(M, reg, E)
    for f in ('prep_replay.json', 'prep_drills.json', 'registration_manifest.json'):
        G.require((V.OUT / f).exists(), 'Preparation step missing: ' + f)
    G.require(C.load(V.OUT / 'prep_replay.json')['ok'] and C.load(V.OUT / 'prep_drills.json')['all_ok'], 'Replay / drills failed')
    rm = C.load(V.OUT / 'registration_manifest.json')
    G.require(rm['sha256'] == sha(MAN) and rm['registration_sha256'] == sha(REG), 'Manifest / registration changed after registration')
    old = C.load(V.BRIDGE / 'freeze.json'); C.verify_files(old['files']); files = dict(old['files'])
    A.platform(); RUN.touch_runtime(M)
    for p in C.READS:
        if Path(p).is_file() and not str(p).endswith('.npz'):
            files[p] = sha(p)
    extra = (list(V.HERE.glob('cv_*.py')) + list((V.HOME / 'scripts/ch4_bridge01').glob('cb_*.py')) + list((V.HOME / 'scripts/mp06d_blur').glob('*.py'))
             + list((V.HOME / 'scripts/mp06f_lighting').glob('*.py')) + [MAN, REG, V.OUT / 'registration_manifest.json', KNOWN])
    for p in extra:
        files[str(Path(p).resolve())] = sha(p)
    files.update(reg['prediction_sources'])
    C.verify_files(files)
    free = shutil.disk_usage(V.OUT).free / 2 ** 30; need = V.CAP * 0.30 / 1024 + 1.0
    mem = memory()
    G.require(free - need >= 5.0, f'Disk: {free:.1f} GB free, {need:.1f} GB to write')
    import importlib.metadata
    f = dict(files=files, freeze_id=C.digest(files), manifest_sha256=sha(MAN), registration_sha256=sha(REG), policy=C.POLICY, cap=V.CAP, retry_cap=V.RETRY_CAP,
             resources=dict(disk_free_gb=round(free, 2), planned_write_gb=round(need, 2), remaining_after_gb=round(free - need, 2), ram_total_gb=round(mem[0] / 2 ** 30, 1),
                            ram_available_gb=round(mem[1] / 2 ** 30, 1)),
             packages={k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'matplotlib']}, python=sys.version, created=time.time())
    C.save(V.OUT / 'freeze.json', f); print(C.canon(dict(freeze_id=f['freeze_id'], files=len(files), resources=f['resources'])))


if __name__ == '__main__':
    try:
        dict(register=register, manifest=manifest, replay=replay, drills=drills, freeze=freeze)[sys.argv[1]]()
    except Exception:
        traceback.print_exc(); sys.exit(2)
