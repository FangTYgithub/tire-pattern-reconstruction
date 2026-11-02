# Prepare sub-ray comparisons and new-phase checks for candidate recommendations.

from __future__ import annotations
import sys, json, time, itertools, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import cb_common as G
import cb_adapter as A

C = G.C

SUBRAY = dict(points=dict(configs=['D_r0.75_p8x4', 'D_r0.75_p8x6'], envs=['NOMINAL', 'BLUR2'], sigmas=[20, 40]),
              lines=dict(configs=['L_w1_s10', 'L_w0.5_s10'], envs=['BLUR2', 'G05'], sigmas=[20]), block=1, rep=1, rays_per_pixel=3,
              sensitivity_rule='flag "numerically sensitive" if |dC_tau| > 2 pp, or a two-candidate difference changes sign with |.| > 2 pp on either sampling, '
                               'or the 50 % / 70 % surface-level status changes; 2 pp is the near-optimal engineering band, not an error bound')


def bound_freeze():
    C.install_guard(); f = C.load(G.OUT / 'freeze.json'); C.verify_files(f['files'])
    G.require(C.digest(f['files']) == f['freeze_id'], 'Freeze digest'); return f


def subray():
    import cb_run as RUN
    f = bound_freeze(); M = C.csv(G.OUT / 'manifest_main.csv').set_index('case_key', drop=False); rows = []
    for part in ('points', 'lines'):
        s = SUBRAY[part]
        for cfg, env, sf, sg in itertools.product(s['configs'], s['envs'], G.SURFS, s['sigmas']):
            ck = G.case_key(cfg, sf, SUBRAY['block'], sg, SUBRAY['rep'], env); par = G.rowdict(M.loc[ck].to_dict())
            pr = A.prep(par); fk3 = RUN.subray_key(pr, SUBRAY['rays_per_pixel'])
            r = dict(par, part='CB01_SUBRAY_3x3', case_key=ck + '|3x3', forward_key=fk3, parent_case_key=ck, parent_forward_key=par['forward_key'],
                     rays_per_pixel=SUBRAY['rays_per_pixel'], entry_subset=False, action='new')
            rows.append(r)
    S = pd.DataFrame(rows)
    G.require(len(S) == 36 and not S.case_key.duplicated().any() and not S.forward_key.duplicated().any() and not S.forward_key.isin(M.forward_key).any(), 'Subray manifest')
    C.save_csv(G.OUT / 'manifest_subray.csv', S)
    C.save(G.OUT / 'subray_registration.json', dict(freeze_id=f['freeze_id'], manifest_sha256=C.sha(G.OUT / 'manifest_subray.csv'), n=len(S), rule=SUBRAY,
                                                    note='same frozen forward record and noise identity as the 1x1 parent; only rays_per_pixel changes (MODEL-PREP-06A convention); '
                                                         'not a continuum limit'))
    print(C.canon(dict(subray=len(S), by_config=S.config.value_counts().to_dict())))


HOLD_SEEDS = dict(point=23910000, line=23920000)
MAIN = dict(d_mm=6.0, tau_mm=0.15, floor=0.5)
BLUR_ENVS = ['NOMINAL', 'BLUR1_5', 'BLUR2', 'BLUR2_5', 'BLUR3']
GRAD_ENVS = ['GRAD_X', 'GRAD_Y', 'GRAD_Y_REV']
HOLD_RULES = {
    'H1': 'required: NOMINAL sigma 20, D_r1_T8x4 vs L_w0.5_s8 (cross-family, condition-matched only)',
    'H2': 'dot blur module (PSF 1-3, sigma > 0): condition where a new candidate changes representative or admissibility, largest J gain; vs old-12 representative',
    'H3': 'remaining conditions with a new representative: closest to the 50 % or 70 % floor, d 5 first then d 6; vs old-12 representative of the same requirement',
    'H4': 'required: G05 sigma 20, best admissible dot of the merged set vs old D_r1_p8x6; if identical, highest-scoring new dot as challenger',
    'H5': 'gradients (x, y, -y), sigma > 0: largest positive J change of a new dot vs old representative; none positive -> highest-scoring new-dot challenger',
    'H6': 'lines, blur or lighting, sigma > 0: largest J change of N4 / N5 vs old line representative (kept even without a gain)',
    'order': 'environment order of section 3 (blur then lighting), sigma ascending, configuration ID as deterministic tie rule; duplicates removed; at most 6 x 48 images',
    'confirmation': 'per group: recommended still admissible on the 4 new blocks (phase-mean per surface >= floor + every-image gates); registered direction kept '
                    '(same family: paired surface-block differences; cross-family: condition-matched means); both within 2 pp or not',
}


def _old_rep(J, fam, env, s, d, floor):
    q = J[(J.family == fam) & (J.env == env) & (J.sigma == s) & (J.d_mm == d) & (J.tau_mm == MAIN['tau_mm']) & (J.floor == floor)].iloc[0]
    return q


def _cfg_score(sc, cfg, env, s, d):
    q = sc[(sc.config == cfg) & (sc.env == env) & (sc.sigma == s) & (sc.d_mm == d) & (sc.tau_mm == MAIN['tau_mm'])]
    return float(q.J.iloc[0]) if len(q) else np.nan


def _old_best_or_rep(J, sc, fam, env, s, d, floor):

    q = _old_rep(J, fam, env, s, d, floor)
    if isinstance(q.preferred_old, str) and q.preferred_old:
        return q.preferred_old, True
    old = G.OLD_DOTS if fam == 'point' else G.OLD_LINES
    t = sc[(sc.config.isin(old)) & (sc.env == env) & (sc.sigma == s) & (sc.d_mm == d) & (sc.tau_mm == MAIN['tau_mm'])].sort_values(['J', 'config'], ascending=[False, True])
    return t.config.iloc[0], False


# Select candidate comparisons and boundary cases for new-phase verification.
def holdout_groups():
    AN = G.OUT / 'analysis'
    J = C.csv(AN / 'selection_changes.csv'); sc = pd.read_csv(AN / 'config_scores.csv.gz')
    envs = G.ENVS
    def order(df):
        df = df.assign(_e=df.env.map(envs.index)); return df.sort_values(['_e', 'sigma']).drop(columns='_e')
    groups = []; used = set()
    def add(gid, fam, env, s, d, floor, rec, ref, why, rec_ok=True, ref_ok=True):
        key = (env, s, tuple(sorted((rec, ref))))
        if key in used or rec == ref:
            return False
        used.add(key); groups.append(dict(group=gid, family=fam, env=env, sigma=int(s), d_mm=d, floor=floor, recommended=rec, reference=ref, reason=why,
                                         recommended_admissible_dev=rec_ok, reference_admissible_dev=ref_ok,
                                         J_rec_dev=_cfg_score(sc, rec, env, s, d), J_ref_dev=_cfg_score(sc, ref, env, s, d)))
        return True
    m = J[(J.d_mm == MAIN['d_mm']) & (J.tau_mm == MAIN['tau_mm']) & (J.floor == MAIN['floor'])]

    q = m[(m.family == 'point') & (m.env == 'NOMINAL') & (m.sigma == 20)].iloc[0]
    add('H1', 'cross', 'NOMINAL', 20, 6.0, .5, 'D_r1_T8x4', 'L_w0.5_s8', 'required: clear low-noise cross-family adjustment',
        'D_r1_T8x4' in str(q.qualified_new).split(';'))

    c = m[(m.family == 'point') & m.env.isin(BLUR_ENVS) & (m.sigma > 0) & m.preferred_new.isin(G.NEW_CFGS) & (m.representative_changed | (m.status_old != m.status_new))]
    if len(c):
        c = order(c.assign(g=c.best_gain_pp)).sort_values('g', ascending=False, kind='stable').iloc[0]
        ref, ok = _old_best_or_rep(J, sc, 'point', c.env, c.sigma, 6.0, .5)
        add('H2', 'point', c.env, c.sigma, 6.0, .5, c.preferred_new, ref, f'largest J gain {c.best_gain_pp:.2f} pp in the dot blur module', True, ok)

    for d in (5.0, 6.0):
        c = J[(J.d_mm == d) & (J.tau_mm == MAIN['tau_mm']) & (J.sigma > 0) & J.preferred_new.isin(G.NEW_CFGS) & J.representative_changed].copy()
        if not len(c):
            continue
        c['margin'] = (c.pref_score_new.astype(float) - c.floor).abs()
        c = order(c).sort_values('margin', kind='stable')
        done = False
        for x in c.itertuples():
            ref, ok = _old_best_or_rep(J, sc, x.family, x.env, x.sigma, d, x.floor)
            if add('H3', x.family, x.env, x.sigma, d, x.floor, x.preferred_new, ref, f'closest to the {int(100 * x.floor)} % floor (margin {100 * x.margin:.2f} pp), d {d:g}', True, ok):
                done = True; break
        if done:
            break

    q = m[(m.family == 'point') & (m.env == 'G05') & (m.sigma == 20)].iloc[0]
    best = q.preferred_new if isinstance(q.preferred_new, str) else None
    if best in (None, 'D_r1_p8x6'):
        t = sc[sc.config.isin([x for x in G.NEW_CFGS if x.startswith('D_')]) & (sc.env == 'G05') & (sc.sigma == 20) & (sc.d_mm == 6.0) & (sc.tau_mm == .15)].sort_values(['J', 'config'], ascending=[False, True])
        best = t.config.iloc[0]
    add('H4', 'point', 'G05', 20, 6.0, .5, best, 'D_r1_p8x6', 'required: attenuation g = 0.50, sigma 20', best in str(q.qualified_new).split(';'),
        'D_r1_p8x6' in str(q.qualified_old).split(';'))

    rows = []
    for x in m[(m.family == 'point') & m.env.isin(GRAD_ENVS) & (m.sigma > 0)].itertuples():
        ref, ok = _old_best_or_rep(J, sc, 'point', x.env, x.sigma, 6.0, .5)
        for cfg in [c for c in G.NEW_CFGS if c.startswith('D_')]:
            rows.append(dict(env=x.env, sigma=x.sigma, cfg=cfg, ref=ref, ok=ok, gain=_cfg_score(sc, cfg, x.env, x.sigma, 6.0) - _cfg_score(sc, ref, x.env, x.sigma, 6.0),
                             adm=cfg in str(x.qualified_new).split(';')))
    t = order(pd.DataFrame(rows)); pos = t[t.gain > 0]
    pick = (pos if len(pos) else t).sort_values(['gain', 'cfg'], ascending=[False, True], kind='stable').iloc[0]
    add('H5', 'point', pick.env, pick.sigma, 6.0, .5, pick.cfg, pick.ref, f'gradient: J change {100 * pick.gain:+.2f} pp vs old representative' + ('' if len(pos) else ' (no positive change)'), bool(pick.adm), bool(pick.ok))

    rows = []
    for x in m[(m.family == 'line') & (m.sigma > 0)].itertuples():
        ref, ok = _old_best_or_rep(J, sc, 'line', x.env, x.sigma, 6.0, .5)
        for cfg in ['L_w1_s10', 'L_w0.5_s10']:
            rows.append(dict(env=x.env, sigma=x.sigma, cfg=cfg, ref=ref, ok=ok, gain=_cfg_score(sc, cfg, x.env, x.sigma, 6.0) - _cfg_score(sc, ref, x.env, x.sigma, 6.0),
                             adm=cfg in str(x.qualified_new).split(';')))
    t = order(pd.DataFrame(rows)).sort_values(['gain', 'cfg'], ascending=[False, True], kind='stable').iloc[0]
    add('H6', 'line', t.env, t.sigma, 6.0, .5, t.cfg, t.ref, f'10 mm pitch: J change {100 * t.gain:+.2f} pp vs old line representative', bool(t.adm), bool(t.ok))
    G.require(len(groups) <= 6, 'More than six holdout groups')
    return groups


def holdout_registry():

    import tr3_b2 as REG
    reg = REG.registry(); ps = reg['positions']; ls = reg['phases']
    pts = [(float(a), float(b)) for a, b in zip(pd.to_numeric(ps.u, errors='coerce'), pd.to_numeric(ps.v, errors='coerce')) if np.isfinite(a) and np.isfinite(b)]
    lns = list(pd.to_numeric(ls.phase, errors='coerce').dropna().astype(float))
    seeds = set(int(x) for x in reg['seeds']); rasters = set(reg['raster_keys']); sources = []
    files = [G.BASE / 'model_prep05/formal_manifest.csv', G.BASE / 'model_prep06c/holdout_manifest.csv', G.BASE / 's12_resource01/manifest.csv'] + list((G.HOME / 'outputs').glob('*/*manifest*.csv'))
    for p in sorted(set(files)):
        if p.name == 'manifest_holdout.csv' or not p.exists():
            continue
        t = C.csv(p); sources.append(dict(path=str(p), sha256=C.sha(p), rows=len(t)))
        if {'phase_u', 'phase_v'}.issubset(t):
            pts += list(t[['phase_u', 'phase_v']].dropna().astype(float).itertuples(index=False, name=None))
        if 'phase_frac' in t:
            lns += list(pd.to_numeric(t.phase_frac, errors='coerce').dropna())
        for col in ('seed', 'noise_seed'):
            if col in t:
                seeds |= set(int(v) for v in pd.to_numeric(t[col], errors='coerce').dropna())
        if 'raster_key' in t:
            rasters |= set(t.raster_key.dropna())
    return np.unique(np.asarray(pts, float), axis=0), np.unique(np.asarray(lns, float)), seeds, rasters, sources


def t_distance(p, used):

    d = np.asarray(p, float)[None, :] - np.atleast_2d(used)
    best = np.full(len(d), np.inf)
    for l in (-2, -1, 0, 1, 2):
        e = d.copy(); e[:, 0] -= .5 * l; e[:, 1] -= l
        e[:, 0] = (e[:, 0] + .5) % 1 - .5
        best = np.minimum(best, np.hypot(e[:, 0], e[:, 1]))
    return best


# Select new phase offsets separated from the registered point and line layouts.
def choose_holdout_phases(groups):
    import cb_adapter as A
    import mp4_formal as MF
    pts, lns, seeds, rasters, sources = holdout_registry()
    S = A.platform(); out = {}; audit = {}
    cfgs = {'point': sorted({c for g in groups for c in (g['recommended'], g['reference']) if c.startswith('D_')}),
            'line': sorted({c for g in groups for c in (g['recommended'], g['reference']) if c.startswith('L_')})}
    for fam in ('point', 'line'):
        if not cfgs[fam]:
            continue
        acc = []; reject = {}; minimum = .05 if fam == 'point' else 1e-9; ks = np.arange(1, 20001)
        if fam == 'line':
            vals = (.5 + ks * MF.SQ2M1) % 1 - .5
            dist = np.min(np.abs((vals[:, None] - lns[None, :] + .5) % 1 - .5), axis=1); order = list(ks[np.argsort(-dist, kind='stable')])
        else:
            order = list(ks)
        while order:
            k = int(order.pop(0))
            ph = {'u': float((.5 + k / MF.RHO) % 1), 'v': float((.5 + k / MF.RHO ** 2) % 1)} if fam == 'point' else {'phi': float((.5 + k * MF.SQ2M1) % 1 - .5)}
            if fam == 'point':
                used = np.vstack([pts, np.array([[a['u'], a['v']] for a in acc])]) if acc else pts
                near = float(t_distance([ph['u'], ph['v']], used).min())
            else:
                used = np.r_[lns, [a['phi'] for a in acc]]; near = float(np.min(np.abs((ph['phi'] - used + .5) % 1 - .5)))
            if near < minimum:
                reject['distance'] = reject.get('distance', 0) + 1; continue
            why = []; keys = {}
            for cfg in cfgs[fam]:
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
            acc.append(dict(block=len(acc), k=k, **ph, min_distance=near, raster_keys=keys))
            if len(acc) == 4:
                break
            if fam == 'line':
                dist = np.minimum(dist, np.abs((vals - ph['phi'] + .5) % 1 - .5)); remaining = set(order)
                order = [int(x) for x in ks[np.argsort(-dist, kind='stable')] if int(x) in remaining]
        G.require(len(acc) == 4, f'Not enough novel {fam} phases; stop before imaging')
        out[fam] = acc
        audit[fam] = dict(min_distance=minimum, metric='staggered-lattice torus (conservative for R and T)' if fam == 'point' else 'phase fraction, all pitches',
                          rejected=reject, phases=acc, configs=cfgs[fam])
    new = {base + 100 * s + 10 * b + r for base in HOLD_SEEDS.values() for s, b, r in itertools.product(range(3), range(4), [1, 2])}
    G.require(not new & seeds, 'Holdout noise seeds previously used')
    return out, dict(phases=audit, known_positions=len(pts), known_line_phases=len(lns), known_seeds=len(seeds), sources=sources,
                     qualification='same position domain, new phases and new noise realisations; not out-of-domain generalisation')


def holdout():
    import cb_adapter as A
    f = bound_freeze(); G.require((G.OUT / 'analysis/tables_check.json').exists(), 'Selection not frozen yet')
    G.require(not (G.OUT / 'holdout_registration.json').exists(), 'Holdout already registered; never re-register after imaging')
    sel = dict(choices=C.sha(G.OUT / 'analysis/choices_all.csv'), changes=C.sha(G.OUT / 'analysis/selection_changes.csv'))
    A.platform()
    groups = holdout_groups(); phases, audit = choose_holdout_phases(groups); rows = []
    for g in groups:
        for cfg in (g['recommended'], g['reference']):
            fam = G.fam_of(cfg)
            for sf, b in itertools.product(G.SURFS, range(4)):
                for rep in ([0] if g['sigma'] == 0 else [1, 2]):
                    ph = dict(phases[fam][b]); ph = dict(block=b, k=ph['k'], **({'u': ph['u'], 'v': ph['v']} if fam == 'point' else {'phi': ph['phi']}))
                    r = A.plan_row(cfg, sf, b, g['sigma'], rep, g['env'], part='CB01_HOLDOUT', ph=ph, seed_base=HOLD_SEEDS, block_label=f'X{b}')
                    r.update(group=g['group'], action='new', source_pack='', source_pack_sha256='', entry_subset=False,
                             method_bundle_sha256=P_bundle(), global_domain_sha256=C.csv(G.OUT / 'manifest_main.csv').global_domain_sha256.iloc[0],
                             evaluation_source_sha256=C.csv(G.OUT / 'manifest_main.csv').evaluation_source_sha256.iloc[0])
                    rows.append(r)
    H = pd.DataFrame(rows).drop_duplicates('forward_key')
    G.require(len(H) <= G.CAPS['holdout'] and not H.case_key.duplicated().any(), 'Holdout manifest over cap / duplicated')
    main = C.csv(G.OUT / 'manifest_main.csv')
    G.require(not H.forward_key.isin(main.forward_key).any() and not H.raster_key.isin(main.raster_key).any(), 'Holdout input seen in the main matrix')
    C.save_csv(G.OUT / 'manifest_holdout.csv', H)
    rec = dict(freeze_id=f['freeze_id'], manifest_sha256=C.sha(G.OUT / 'manifest_holdout.csv'), n=len(H), groups=groups, rules=HOLD_RULES, phase_registry=audit,
               seeds=HOLD_SEEDS, selection_files=sel, main_requirement=MAIN, created=time.time())
    C.save(G.OUT / 'holdout_registration.json', rec)
    print(C.canon(dict(n=len(H), groups=[(g['group'], g['env'], g['sigma'], g['d_mm'], g['recommended'], g['reference']) for g in groups])))


def P_bundle():
    import cb_prepare as P
    return P.bundle()


if __name__ == '__main__':
    try:
        dict(subray=subray, holdout=holdout)[sys.argv[1]]()
    except Exception:
        traceback.print_exc(); sys.exit(2)
