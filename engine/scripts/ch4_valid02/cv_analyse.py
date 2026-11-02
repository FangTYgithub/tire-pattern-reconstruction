# Compare candidate selection and paired effects on the new-phase validation data.

from __future__ import annotations
import sys, json, time, itertools, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import cv_common as V

G, C = V.G, V.C
import cv_run as RUN
import cv_prepare as P
import analysis as A0
AN = V.OUT / 'analysis'
KEYS = ['family', 'env', 'sigma', 'd_mm', 'tau_mm', 'floor']
SUR = ['B_curv', 'M10_curv', 'M24_curv']
RES_EXTRA = ['raster_key', 'peak', 'integral', 'lit_px', 'forward_route', 'module', 'phase_seq_k', 'entry_subset']


def _collect_init(fid):
    C.install_guard()


def _collect_one(args):
    r, fid = args
    meta = RUN.Store().resume(r, fid)
    G.require(meta is not None and meta.get('scientific') is True and meta['case_key'] == r['case_key'], 'Missing / foreign pack: ' + r['case_key'])
    return meta['result'], [dict(case_key=r['case_key'], **v) for v in meta['sensitivity']]


def assemble(workers=8):
    from concurrent.futures import ProcessPoolExecutor
    f, M = RUN.verify(); fid = f['freeze_id']; t0 = time.time()
    jobs = [(G.rowdict(r), fid) for r in M.to_dict('records')]; rows, sens = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_collect_init, initargs=(fid,)) as ex:
        for i, (res, s) in enumerate(ex.map(_collect_one, jobs, chunksize=64)):
            rows.append(res); sens.extend(s)
            if (i + 1) % 5000 == 0:
                print(f'collect {i + 1}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    R = pd.DataFrame(rows); S = pd.DataFrame(sens)
    RUN.result_gate(R, M); A0.sensitivity_gate(S, R)
    X = M[C.IDENTITY + RES_EXTRA]
    R = R.merge(M[['case_key'] + RES_EXTRA], on='case_key', validate='one_to_one')
    return R, S, X, M, f, round(time.time() - t0, 1)


def select(R, S, X):
    long, surf, ch = A0.tables(R, S, X[C.IDENTITY])
    sc = surf.groupby(['family', 'config', 'env', 'sigma', 'd_mm', 'tau_mm']).agg(J=('Ctau', 'min'), safe=('safe', 'all'), n_surf=('surface', 'nunique'),
                                                                                phase_min=('phase_min', 'min'), rmse_max=('rmse_max', 'max')).reset_index()
    return long, surf, ch, sc


def block_J(long):

    b = long.groupby(['config', 'env', 'sigma', 'd_mm', 'tau_mm', 'surface', 'phase_block']).Ctau.mean().reset_index()
    j = b.groupby(['config', 'env', 'sigma', 'd_mm', 'tau_mm', 'phase_block']).Ctau.min().reset_index()
    return j.groupby(['config', 'env', 'sigma', 'd_mm', 'tau_mm']).Ctau.agg(J_block_min='min', J_block_max='max').reset_index()


def _set(x):
    return set() if x is None or (isinstance(x, float) and np.isnan(x)) or x == '' else set(str(x).split(';'))


def _fail_reason(q, floor):

    out = []
    for r in q.sort_values('surface').to_dict('records'):
        s = r['surface'].split('_')[0]; why = []
        if r['Ctau'] < floor:
            why.append(f"Cτ {r['Ctau']:.3f} < {floor:.2f}")
        if r['n_no_output'] > 0:
            why.append(f"{int(r['n_no_output'])} Bilder ohne Ausgabe")
        if r['rmse_n'] < r['n_images'] - r['n_no_output']:
            why.append(f"{int(r['n_images'] - r['n_no_output'] - r['rmse_n'])} Bilder ohne gültige Punkte")
        if r['min_exposure'] == 0 and r['n_no_output'] == 0:
            why.append('Exposition 0')
        if r['rmse_n'] == r['n_images'] and r['rmse_max'] > .3:
            why.append(f"RMSE max {r['rmse_max']:.3f}")
        if r['wrong'] > 0:
            why.append(f"w = {int(r['wrong'])}")
        if r['u_n'] == r['n_images'] and r['u_max'] > .05:
            why.append(f"u-Anteil max {100 * r['u_max']:.1f} %")
        if r['n_phases'] != 4:
            why.append('Phasen fehlen')
        if why:
            out.append(s + ': ' + ', '.join(why))
    return '; '.join(out)


# Compare candidate feasibility, rankings and paired effects on new-phase data.
def validate(old, ch, surf, sc, sc_old, bj):

    o = old.rename(columns={c: c + '_old' for c in ['qualified', 'near_optimal', 'preferred', 'best_score', 'pref_score', 'status']})
    n = ch.rename(columns={c: c + '_new' for c in ['qualified', 'near_optimal', 'preferred', 'best_score', 'pref_score', 'status', 'n_configs']})
    T = o.merge(n, on=KEYS, validate='one_to_one', how='outer', indicator=True)
    G.require(T._merge.eq('both').all() and len(T) == 1920, 'Validation table not closed (1920 conditions)')
    T = T.drop(columns='_merge')
    S = surf.set_index(['config', 'env', 'sigma', 'd_mm', 'tau_mm']).sort_index(); scn = sc.set_index(['config', 'env', 'sigma', 'd_mm', 'tau_mm'])
    sco = sc_old.set_index(['config', 'env', 'sigma', 'd_mm', 'tau_mm']); bj = bj.set_index(['config', 'env', 'sigma', 'd_mm', 'tau_mm'])
    rows = []
    for r in T.to_dict('records'):
        r = G.rowdict(r); p = r['preferred_old']; k = None if p is None else (p, r['env'], r['sigma'], r['d_mm'], r['tau_mm'])
        add = dict(J_oldpref_old=None, J_oldpref_new=None, safe_oldpref_new=None, qualified_oldpref_new=None, margin_pp=None, gap_to_best_pp=None,
                   oldpref_in_new_near=None, J_oldpref_block_min=None, J_oldpref_block_max=None, dJ_oldpref_pp=None, fail_reason='')
        if p is not None:
            q = S.loc[k].reset_index(); G.require(len(q) == 3, 'Surface rows of the old representative')
            J = float(q.Ctau.min()); qual = bool(q.safe.all() and q.Ctau.ge(r['floor']).all())
            G.require(qual == (p in _set(r['qualified_new'])), 'Qualification differs from the frozen selection code')
            add.update(J_oldpref_old=float(sco.loc[k].J), J_oldpref_new=J, safe_oldpref_new=bool(q.safe.all()), qualified_oldpref_new=qual, margin_pp=100 * (J - r['floor']),
                       oldpref_in_new_near=p in _set(r['near_optimal_new']), J_oldpref_block_min=float(bj.loc[k].J_block_min), J_oldpref_block_max=float(bj.loc[k].J_block_max))
            add['dJ_oldpref_pp'] = 100 * (J - add['J_oldpref_old'])
            if r['best_score_new'] is not None:
                add['gap_to_best_pp'] = 100 * (r['best_score_new'] - J)
            if not qual:
                add['fail_reason'] = _fail_reason(q, r['floor'])
            label = ('kept' if add['oldpref_in_new_near'] else 'usable_reorder') if qual else 'not_confirmed'
        else:
            label = 'fail_repeated' if r['preferred_new'] is None else 'boundary_changed'

            g = sc[(sc.family == r['family']) & (sc.env == r['env']) & (sc.sigma == r['sigma']) & (sc.d_mm == r['d_mm']) & (sc.tau_mm == r['tau_mm'])]
            gs = g[g.safe].sort_values(['J', 'config'], ascending=[False, True])
            add['fail_reason'] = (f"bester torsicherer Kandidat {gs.config.iloc[0]} J = {gs.J.iloc[0]:.3f}" if len(gs) else 'kein Kandidat erfüllt die Bildbedingungen')
        qo, qn = _set(r['qualified_old']), _set(r['qualified_new']); no, nn = _set(r['near_optimal_old']), _set(r['near_optimal_new'])
        rows.append(dict(r, **add, label=label, representative_changed=(r['preferred_old'] or '-') != (r['preferred_new'] or '-'),
                         qualified_added=';'.join(sorted(qn - qo)), qualified_removed=';'.join(sorted(qo - qn)),
                         near_added=';'.join(sorted(nn - no)), near_removed=';'.join(sorted(no - nn)),
                         best_dJ_pp=(100 * (r['best_score_new'] - r['best_score_old']) if r['best_score_new'] is not None and r['best_score_old'] is not None else None)))
    T = pd.DataFrame(rows)

    ck = ['env', 'sigma', 'd_mm', 'tau_mm', 'floor']
    pt = T[T.family == 'point'][ck + ['best_score_old', 'best_score_new']]; ln = T[T.family == 'line'][ck + ['best_score_old', 'best_score_new']]
    fam = pt.merge(ln, on=ck, suffixes=('_point', '_line'), validate='one_to_one')
    G.require(len(fam) == 960, 'Family comparison not closed')
    def win(a, b):
        if pd.isna(a) and pd.isna(b):
            return 'none'
        return 'point' if pd.isna(b) or (pd.notna(a) and a > b) else 'line'
    W = fam[ck].assign(family_winner_old=[win(a, b) for a, b in zip(fam.best_score_old_point, fam.best_score_old_line)],
                       family_winner_new=[win(a, b) for a, b in zip(fam.best_score_new_point, fam.best_score_new_line)],
                       point_best_new=fam.best_score_new_point.values, line_best_new=fam.best_score_new_line.values,
                       point_best_old=fam.best_score_old_point.values, line_best_old=fam.best_score_old_line.values)
    T = T.merge(W, on=ck, validate='many_to_one'); G.require(len(T) == 1920, 'Validation table lost rows')
    order = ['family', 'env', 'sigma', 'd_mm', 'tau_mm', 'floor', 'label', 'preferred_old', 'preferred_new', 'representative_changed', 'J_oldpref_old', 'J_oldpref_new',
             'dJ_oldpref_pp', 'J_oldpref_block_min', 'J_oldpref_block_max', 'qualified_oldpref_new', 'margin_pp', 'best_score_old', 'best_score_new', 'best_dJ_pp',
             'gap_to_best_pp', 'oldpref_in_new_near', 'near_optimal_old', 'near_optimal_new', 'near_added', 'near_removed', 'qualified_old', 'qualified_new',
             'qualified_added', 'qualified_removed', 'status_old', 'status_new', 'pref_score_old', 'pref_score_new', 'fail_reason', 'family_winner_old', 'family_winner_new',
             'point_best_old', 'point_best_new', 'line_best_old', 'line_best_new', 'safe_oldpref_new', 'n_configs_new']
    env_order = {e: i for i, e in enumerate(G.ENVS)}
    T = T.assign(_e=T.env.map(env_order)).sort_values(['family', 'd_mm', 'tau_mm', 'floor', '_e', 'sigma']).drop(columns='_e')
    return T[order + [c for c in T.columns if c not in order]]


def _agg(D, col, kind, name, extra):

    blk = D.groupby(['env', 'sigma', 'phase_block'])[col].mean().reset_index()
    srf = D.groupby(['env', 'sigma', 'surface'])[col].mean().reset_index()
    out = srf.groupby(['env', 'sigma'])[col].agg(effect='mean', surf_min='min', surf_max='max').reset_index()
    br = blk.groupby(['env', 'sigma'])[col].agg(block_min='min', block_max='max', blocks_pos=lambda x: int((x > 0).sum()), blocks='size').reset_index()
    out = out.merge(br, on=['env', 'sigma'], validate='one_to_one')
    for s in SUR:
        q = srf[srf.surface == s].set_index(['env', 'sigma'])[col]
        out[f'effect_{s.split("_")[0]}'] = [q.get((e, g)) for e, g in zip(out.env, out.sigma)]
    out.insert(0, 'kind', kind); out.insert(1, 'name', name)
    for k, v in extra.items():
        out[k] = v
    return out


def key_effects(R):
    rows = []
    for left, right in P.KEY_PAIRS:
        q = RUN.pair(R, R, left, right, P.PAIR_KEYS)
        D = q.groupby(['surface', 'env', 'sigma', 'phase_block']).delta_Ctau.mean().reset_index()
        rows.append(_agg(D, 'delta_Ctau', 'pair', f'{right} − {left}', dict(left=left, right=right, n_pairs=len(q))))
    for name, cells in P.FOURCELLS.items():
        m = {k: R[R.config == v].groupby(['surface', 'env', 'sigma', 'phase_block']).glob_Ctau.mean() for k, v in cells.items()}
        idx = m['a'].index
        G.require(all(set(x.index) == set(idx) for x in m.values()), f'Four-cell blocks incomplete {name}')
        D = pd.DataFrame({k: v.reindex(idx) for k, v in m.items()})
        D['DiD'] = (D.d - D.c) - (D.b - D.a); D['spacing_at_ref'] = D.b - D.a; D['spacing_at_alt'] = D.d - D.c
        D = D.reset_index()
        rows.append(_agg(D, 'DiD', 'fourcell', name, dict(left=cells['a'], right=cells['d'], cells=json.dumps(cells), n_pairs=None)))
        for col, lab in (('spacing_at_ref', 'spacing effect at reference size/width'), ('spacing_at_alt', 'spacing effect at alternative size/width')):
            rows.append(_agg(D, col, 'fourcell_simple', f'{name}: {lab}', dict(left=cells['a'] if col == 'spacing_at_ref' else cells['c'],
                                                                               right=cells['b'] if col == 'spacing_at_ref' else cells['d'], cells=json.dumps(cells), n_pairs=None)))
    return pd.concat(rows, ignore_index=True)


def compare_effects(new, old, sc_new, sc_old):
    k = ['kind', 'name', 'env', 'sigma']
    o = old[k + ['effect', 'surf_min', 'surf_max', 'block_min', 'block_max', 'blocks_pos']].rename(columns=lambda c: c if c in k else c + '_old')
    T = new.merge(o, on=k, validate='one_to_one', how='left')
    G.require(T.effect_old.notna().all(), 'Old effect missing')
    main_new = sc_new[(sc_new.d_mm == 6) & (sc_new.tau_mm == .15)].set_index(['config', 'env', 'sigma'])
    main_old = sc_old[(sc_old.d_mm == 6) & (sc_old.tau_mm == .15)].set_index(['config', 'env', 'sigma'])
    def adm(t, cfg, e, s):
        x = t.loc[(cfg, e, s)]; return bool(x.safe and x.J >= .5)
    for side in ('left', 'right'):
        T[f'{side}_admissible_old'] = [adm(main_old, c, e, s) for c, e, s in zip(T[side], T.env, T.sigma)]
        T[f'{side}_admissible_new'] = [adm(main_new, c, e, s) for c, e, s in zip(T[side], T.env, T.sigma)]
    T['same_sign'] = np.sign(T.effect.round(12)) == np.sign(T.effect_old.round(12))
    T['both_within_2pp'] = (T.effect.abs() <= .02) & (T.effect_old.abs() <= .02)
    T['change_pp'] = 100 * (T.effect - T.effect_old)
    T['old_inside_new_block_range'] = (T.effect_old >= T.block_min - 1e-12) & (T.effect_old <= T.block_max + 1e-12)
    return T


def claims(VT, KE, ch_new, sc_new, sc_old):
    m = VT[(VT.d_mm == 6) & (VT.tau_mm == .15) & (VT.floor == .5)]
    out = {}
    lab = m.groupby(['family', 'label']).size().unstack(fill_value=0)
    out['labels_main'] = {f: lab.loc[f].to_dict() for f in lab.index}
    out['labels_main_by_sigma'] = {f'{f}|{int(s)}': g.label.value_counts().to_dict() for (f, s), g in m.groupby(['family', 'sigma'])}
    out['labels_all'] = VT.groupby(['family', 'd_mm', 'tau_mm', 'floor', 'label']).size().rename('n').reset_index().to_dict('records')

    t = KE[(KE.kind == 'pair') & (KE.right == 'D_r1_T8x4')]
    out['1_T8x4'] = {}
    for (left, s), g in t.groupby(['left', 'sigma']):
        out['1_T8x4'][f'{left}|{int(s)}'] = dict(envs=len(g), envs_positive_new=int(g.effect.gt(0).sum()), envs_positive_old=int(g.effect_old.gt(0).sum()),
                                                 mean_new_pp=float(100 * g.effect.mean()), mean_old_pp=float(100 * g.effect_old.mean()),
                                                 min_new_pp=float(100 * g.effect.min()), max_new_pp=float(100 * g.effect.max()))
    tq = sc_new[(sc_new.config == 'D_r1_T8x4') & (sc_new.d_mm == 6) & (sc_new.tau_mm == .15)]
    out['1_T8x4_admissible_new'] = {int(s): int((g.safe & (g.J >= .5)).sum()) for s, g in tq.groupby('sigma')}
    tq = sc_old[(sc_old.config == 'D_r1_T8x4') & (sc_old.d_mm == 6) & (sc_old.tau_mm == .15)]
    out['1_T8x4_admissible_old'] = {int(s): int((g.safe & (g.J >= .5)).sum()) for s, g in tq.groupby('sigma')}

    q = VT[(VT.family == 'point') & VT.sigma.isin([40, 60]) & (VT.d_mm == 6) & (VT.tau_mm == .15)]
    out['2_dots_40_60'] = q[['env', 'sigma', 'floor', 'label', 'preferred_old', 'preferred_new', 'near_optimal_old', 'near_optimal_new', 'qualified_old', 'qualified_new',
                             'J_oldpref_new', 'margin_pp', 'gap_to_best_pp']].to_dict('records')

    q = m[m.family == 'line']
    out['3_lines'] = q[['env', 'sigma', 'label', 'preferred_old', 'preferred_new', 'near_optimal_old', 'near_optimal_new', 'J_oldpref_new', 'gap_to_best_pp']].to_dict('records')

    q = VT[(VT.family == 'point') & (VT.d_mm == 6) & (VT.floor == .5)]
    out['4_family'] = {f'{t:g}': dict(same=int((g.family_winner_old == g.family_winner_new).sum()), n=len(g),
                                      old=g.family_winner_old.value_counts().to_dict(), new=g.family_winner_new.value_counts().to_dict(),
                                      low_noise_old=g[g.sigma <= 20].family_winner_old.value_counts().to_dict(), low_noise_new=g[g.sigma <= 20].family_winner_new.value_counts().to_dict())
                       for t, g in q.groupby('tau_mm')}

    f = KE[KE.kind == 'fourcell']
    out['5_fourcell'] = {n: dict(same_sign=int(g.same_sign.sum()), n=len(g), mean_new_pp=float(100 * g.effect.mean()), mean_old_pp=float(100 * g.effect_old.mean()),
                                 pos_new=int((g.effect > 0).sum()), pos_old=int((g.effect_old > 0).sum()))
                         for n, g in f.groupby('name')}

    q = m[m.sigma.isin([80, 100])]
    out['6_high_noise'] = q.groupby(['family', 'sigma']).label.value_counts().rename('n').reset_index().to_dict('records')
    out['6_high_noise_admissible'] = q[q.status_new == 'qualified'][['family', 'env', 'sigma', 'preferred_new', 'best_score_new']].to_dict('records')

    return out


def near_floor(VT, reg):
    rows = []
    idx = VT.set_index(KEYS)
    for c in reg['claims']['7_near_floor']['cases']:
        k = tuple(c[x] for x in KEYS); r = G.rowdict(idx.loc[k].to_dict())
        rows.append(dict({x: c[x] for x in KEYS}, kind=c['kind'], config=c.get('config') or c['preferred'], margin_old_pp=c['margin_pp'], label=r['label'],
                         preferred_new=r['preferred_new'], J_oldpref_new=r['J_oldpref_new'], margin_new_pp=r['margin_pp'], best_new=r['best_score_new'],
                         status_new=r['status_new']))
    return pd.DataFrame(rows)


def old_matrix():

    reg = C.load(P.REG)
    for p, h in reg['prediction_sources'].items():
        G.require(C.sha(p) == h, 'Prediction source changed: ' + p)
    R = pd.read_csv(P.AN_OLD / 'results_all.csv.gz', low_memory=False, float_precision='round_trip')
    S = pd.read_csv(P.AN_OLD / 'sensitivity_all.csv.gz', low_memory=False, float_precision='round_trip')
    return R, S, reg


def selftest():

    C.install_guard(); t0 = time.time(); R, S, reg = old_matrix(); X = R[C.IDENTITY]
    long, surf, ch, sc = select(R, S, X); bj = block_J(long)
    old = pd.DataFrame(reg['old_choices'])
    T = validate(old, ch, surf, sc, sc, bj)
    G.require(set(T.label) <= {'kept', 'fail_repeated'} and T.representative_changed.eq(False).all() and T.dJ_oldpref_pp.fillna(0).abs().max() < 1e-12, 'Self-test labels')
    KE = key_effects(R); CE = compare_effects(KE, KE, sc, sc)
    G.require(CE.change_pp.abs().max() < 1e-12, 'Self-test effects')
    reg_pairs = pd.DataFrame(reg['old_key_pairs'])
    j = KE[KE.kind == 'pair'].merge(reg_pairs, on=['left', 'right', 'env', 'sigma'], validate='one_to_one')
    G.require(len(j) == 600 and np.allclose(j.effect, j.mean_delta, atol=1e-12, rtol=0), 'Pair aggregation differs from the registered old values')
    f = KE[KE.kind == 'fourcell']
    for name, g in f.groupby('name'):
        o = pd.DataFrame(reg['claims']['5_peak_cap_interactions']['fourcell'][name]).merge(g, on=['env', 'sigma'], validate='one_to_one')
        G.require(np.allclose(o.DiD, o.effect, atol=1e-12, rtol=0), 'Four-cell aggregation differs from the registered old values ' + name)
    rec = dict(ok=True, conditions=len(T), labels=T.label.value_counts().to_dict(), effects=len(KE), seconds=round(time.time() - t0, 1))
    C.save(AN / 'selftest_old_vs_old.json', rec); print(C.canon(rec))


def tables():
    C.install_guard(); t0 = time.time()
    R, S, X, M, f, sec = assemble()
    Ro, So, reg = old_matrix()
    long, surf, ch, sc = select(R, S, X); bj = block_J(long)
    _, _, ch_o, sc_o = select(Ro, So, Ro[C.IDENTITY])
    old = pd.DataFrame(reg['old_choices'])
    j = ch_o.merge(old, on=KEYS, suffixes=('', '_reg'), validate='one_to_one')
    G.require(len(j) == 1920 and all((j[c].fillna('').astype(str) == j[c + '_reg'].fillna('').astype(str)).all() for c in ['qualified', 'near_optimal', 'preferred', 'status']),
              'Old selection not reproduced from the old records')
    VT = validate(old, ch, surf, sc, sc_o, bj)
    KE = key_effects(R); KE_o = key_effects(Ro); CE = compare_effects(KE, KE_o, sc, sc_o)
    NF = near_floor(VT, reg); CL = claims(VT, CE, ch, sc, sc_o)
    CL['7_near_floor'] = dict(n=len(NF), labels=NF.label.value_counts().to_dict(), by_kind={k: g.label.value_counts().to_dict() for k, g in NF.groupby('kind')})

    keep = [c for c in R.columns]
    C.atomic(V.OUT / 'results.csv.gz', lambda q: R[keep].sort_values('case_key').to_csv(q, index=False, compression='gzip'))
    Sx = S.merge(R[['case_key'] + [c for c in C.IDENTITY if c != 'case_key']], on='case_key', validate='many_to_one')
    C.atomic(V.OUT / 'sensitivity.csv.gz', lambda q: Sx.sort_values(['case_key', 'd_mm', 'tau_mm']).to_csv(q, index=False, compression='gzip'))
    C.save_csv(V.OUT / 'validation_table.csv', VT)
    C.save_csv(V.OUT / 'key_effects.csv', CE)
    AN.mkdir(parents=True, exist_ok=True)
    for name, t in [('surfaces.csv.gz', surf), ('config_scores.csv.gz', sc), ('config_scores_old.csv.gz', sc_o), ('block_J.csv.gz', bj), ('near_floor.csv', NF), ('choices_new.csv', ch)]:
        if name.endswith('.gz'):
            C.atomic(AN / name, lambda q, tab=t: tab.to_csv(q, index=False, compression='gzip'))
        else:
            C.save_csv(AN / name, t)

    st = RUN.Store()
    with st.connect() as con:
        led = dict(con.execute('SELECT status,count(*) FROM cases GROUP BY status').fetchall())
        ev = dict(con.execute("SELECT stage||':'||status,count(*) FROM events GROUP BY stage,status").fetchall())
    fin = R[['glob_Ctau', 'glob_Cvalid']].to_numpy(); err = R[['glob_pt_rmse', 'glob_pt_p95', 'glob_surf_rmse']]
    chk = dict(freeze_id=f['freeze_id'], images=len(R), unique_case_keys=int(R.case_key.nunique()), unique_forward_keys=int(R.forward_key.nunique()),
               families=R.family.value_counts().to_dict(), sigmas={str(int(k)): int(v) for k, v in R.sigma.value_counts().sort_index().items()},
               configs=int(R.config.nunique()), per_config=sorted(set(R.config.value_counts())), per_env=sorted(set(R.env.value_counts())),
               sensitivity_rows=len(S), sensitivity_per_image=sorted(set(S.groupby('case_key').size())), states=R.state.value_counts().to_dict(),
               no_output_rmse_undefined=bool(R[R.state == 'no_output'].glob_pt_rmse.isna().all()), coverage_finite=bool(np.isfinite(fin).all()),
               errors_finite_where_defined=bool(np.isfinite(err.to_numpy(dtype=float)[~np.isnan(err.to_numpy(dtype=float))]).all()),
               entry_included=int(R.entry_subset.sum()), ledger=led, events=ev, retries=int(sum(RUN.Store().retries().values())),
               packs_verified=len(R), pack_gate='Store.resume -> pack_gate on every pack (checksum, binding, arrays vs summary)', collect_seconds=sec,
               old_selection_reproduced=True, prediction_sources_unchanged=True, bridge_freeze_unchanged=True,
               selection_conditions=len(VT), key_effect_rows=len(CE), pairs_one_to_one=True,
               analysis_code={p.name: C.sha(p) for p in sorted(V.HERE.glob('cv_*.py'))}, claims=CL, seconds=round(time.time() - t0, 1))
    C.verify_files(C.load(V.BRIDGE / 'freeze.json')['files'])
    C.save(V.OUT / 'checks.json', chk)
    print(C.canon(dict(images=len(R), states=chk['states'], labels=CL['labels_main'], seconds=chk['seconds'])))


if __name__ == '__main__':
    try:
        dict(tables=tables, selftest=selftest)[sys.argv[1]]()
    except Exception:
        traceback.print_exc(); sys.exit(2)
