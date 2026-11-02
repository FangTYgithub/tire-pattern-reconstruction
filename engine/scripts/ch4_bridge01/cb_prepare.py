# Check candidate inputs and construct the Chapter 4 execution manifest.

from __future__ import annotations
import sys, json, time, itertools, sqlite3, traceback, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cb_common as G
import cb_adapter as A

C = G.C
PREP = G.OUT / 'prep'
MANIFEST = G.OUT / 'manifest_main.csv'
ENTRY_PLAN = [('B_curv', 'NOMINAL'), ('M10_curv', 'BLUR2'), ('M24_curv', 'GRAD_Y_REV')]
CHECK_COLS = ['forward_key', 'raster_key', 'pattern_sha', 'peak_f32', 'integral', 'lit_px', 'seed', 'm3_seed', 'field_json', 'field_id', 'forward_route',
              'method_binding', 'template_cand_id', 'phase_u', 'phase_v', 'phase_frac', 'rays_per_pixel', 'psf_true_px', 'method']


def old_manifest() -> pd.DataFrame:
    b = C.csv(G.OLD['blur_manifest']); l = C.csv(G.OLD['light_manifest'])
    nb, nl = b[b.env == 'NOMINAL'].set_index('case_key'), l[l.env == 'NOMINAL'].set_index('case_key')
    G.require(set(nb.index) == set(nl.index) and (nb.forward_key == nl.forward_key.reindex(nb.index)).all(), 'Shared nominal cases differ between modules')
    m = pd.concat([b.assign(module='blur'), l[l.env != 'NOMINAL'].assign(module='lighting')], ignore_index=True)
    G.require(len(m) == 15840 and not m.case_key.duplicated().any() and not m.forward_key.duplicated().any(), 'Old main matrix is not 15840 unique cases')
    return m


def entry_cases() -> set:
    out = set()
    for i, cfg in enumerate(G.NEW_CFGS):
        for j, (sf, env) in enumerate(ENTRY_PLAN):
            out.add(G.case_key(cfg, sf, (i + j) % 4, 0, 0, env))
            out.add(G.case_key(cfg, sf, (i + j + 2) % 4, 20 if (i + j) % 2 == 0 else 40, 1 if i % 2 == 0 else 2, env))
    G.require(len(out) == 30, 'Entry plan must have 30 distinct cases')
    return out


def bundle() -> str:
    files = {p.name: C.sha(p) for p in sorted(G.HERE.glob('cb_*.py')) if p.name in ('cb_common.py', 'cb_adapter.py')}
    return C.digest(dict(old_method_bundle='a0d7b2a9a95e5f30240cb4c148db272b8032374323ce46c9271ec9a7c838d89c', adapter=files))


def gate_main(M: pd.DataFrame, entry: set | None = None):
    need = set(C.IDENTITY + ['action', 'source_pack', 'source_pack_sha256', 'entry_subset', 'integral', 'peak', 'lit_px', 'rays_per_pixel', 'field_json',
                             'global_domain_sha256', 'evaluation_source_sha256', 'forward_route'])
    G.require(need.issubset(M.columns), 'Manifest missing columns: ' + str(need - set(M.columns)))
    G.require(set(M.case_key) == G.expected_keys(G.NEW_CFGS) and len(M) == 6600, 'Main matrix not closed (5 x 10 x 132)')
    G.require(not M.case_key.duplicated().any() and not M.forward_key.duplicated().any(), 'Duplicate case / forward keys')
    for r in M.to_dict('records'):
        r = G.rowdict(r); cfg = r['config']; fam = G.fam_of(cfg); s = r['sigma']
        G.require(r['family'] == fam and r['method'] == G.method_of(cfg) and r['rays_per_pixel'] == 1, 'Family / method / sampling mismatch')
        G.require(r['psf_true_px'] == G.PSF[r['env']] and json.loads(r['field_json']) == G.FIELDS[r['env']], 'Environment mismatch')
        G.require(r['rep'] in ([0] if s == 0 else [1, 2]), 'Wrong repetition label')
        G.require(r['seed'] == (None if s == 0 else G.seed_of(fam, r['surface'], r['phase_block'], r['rep'])), 'Wrong seed / zero-noise seed')
        G.require(r['case_key'] == G.case_key(cfg, r['surface'], r['phase_block'], s, r['rep'], r['env']), 'Case key / fields mismatch')
        G.require(abs(r['peak'] - min(1.0, G.Q / r['lit_px'])) <= 1e-9 and 0 < r['peak'] <= 1 and r['integral'] <= G.Q * (1 + 2 ** -23), 'Resource rule A = min(1, Q/S0) violated')
        G.require((r['forward_route'] == 'r_forward') == (r['env'] == 'GRAD_Y_REV'), 'Forward route mismatch')
        G.require(r['action'] in ('new', 'reuse'), 'Unknown action')
    for fam, g in M.groupby('family'):
        cols = ['phase_u', 'phase_v'] if fam == 'point' else ['phase_frac']
        G.require(g.groupby('phase_block')[cols].nunique().eq(1).all().all(), 'Normalised phase block aliases two phases')
    if entry is not None:
        G.require(set(M[M.entry_subset].case_key) == entry and M[M.entry_subset].action.eq('new').all(), 'Entry subset mismatch')


def manifest():
    C.install_guard(); t = time.time(); old = old_manifest()
    oldfk = set(old.forward_key); tmpl = old.iloc[0]
    rows = []
    for cfg, env, sf, b, s in itertools.product(G.NEW_CFGS, G.ENVS, G.SURFS, range(4), G.SIGMAS):
        for rep in ([0] if s == 0 else [1, 2]):
            rows.append(A.plan_row(cfg, sf, b, s, rep, env))
    M = pd.DataFrame(rows)
    M['action'] = np.where(M.forward_key.isin(oldfk), 'reuse', 'new')
    M['source_pack'] = ''; M['source_pack_sha256'] = ''; M['source_batch'] = ''
    M['method_bundle_sha256'] = bundle(); M['global_domain_sha256'] = tmpl.global_domain_sha256; M['evaluation_source_sha256'] = tmpl.evaluation_source_sha256
    M['execution_authorised'] = True
    E = entry_cases(); M['entry_subset'] = M.case_key.isin(E)
    M['module'] = np.where(M.env.isin(['BLUR1_5', 'BLUR2', 'BLUR2_5', 'BLUR3']), 'blur', np.where(M.env == 'NOMINAL', 'shared_nominal', 'lighting'))
    gate_main(M, E)
    G.require((M.action == 'new').all(), 'A new-candidate input coincides with an old case; reuse would need explicit binding')
    C.save_csv(MANIFEST, M.sort_values(['config', 'env', 'surface', 'phase_block', 'sigma', 'rep']))

    R = []
    for cfg in G.ALL:
        for b in range(4):
            ph = A.phases()[G.fam_of(cfg)][b]
            info = A.raster_info(cfg, ph); info.update(phase_block=b, phase_u=ph.get('u'), phase_v=ph.get('v'), phase_frac=ph.get('phi'))
            if G.fam_of(cfg) == 'point':
                tpl = A.platform()['T'][cfg]['row']
                info.update(offset_par_mm=ph['u'] * float(tpl['p_parallel_mm']), offset_perp_mm=ph['v'] * float(tpl['p_perp_mm']), arrangement=tpl['arrangement'],
                            radius_mm=float(tpl['radius_mm']), p_par_mm=float(tpl['p_parallel_mm']), p_perp_mm=float(tpl['p_perp_mm']))
            else:
                tpl = A.platform()['T'][cfg]['row']
                info.update(offset_mm=ph['phi'] * float(tpl['line_pitch_mm']), width_mm=float(tpl['line_width_mm']), pitch_mm=float(tpl['line_pitch_mm']))
            info['template'] = A.platform()['T'][cfg]['source']
            R.append(info)
    C.save_csv(PREP / 'rasters.csv', pd.DataFrame(R))
    T = {cfg: dict(source=v['source'], base_cand_id=v.get('base_cand_id'), change=v.get('change'), row={k: G.rowdict({k: x})[k] for k, x in v['row'].items()})
         for cfg, v in A.platform()['T'].items() if cfg in G.NEW_CFGS}
    C.save(PREP / 'new_templates.json', T)
    summ = dict(rows=len(M), new=int((M.action == 'new').sum()), reuse=int((M.action == 'reuse').sum()), entry=int(M.entry_subset.sum()),
                by_config=M.config.value_counts().to_dict(), by_env=M.env.value_counts().to_dict(), routes=M.forward_route.value_counts().to_dict(),
                manifest_sha256=C.sha(MANIFEST), bundle=bundle(), seconds=round(time.time() - t, 1))
    C.save(PREP / 'manifest_summary.json', summ); print(C.canon(summ))


def stored_pack(r: dict):

    store = G.OLD['blur_store'] if r['module'] == 'blur' else G.OLD['light_store']
    p = store.parent / 'data' / (hashlib.sha256(r['case_key'].encode()).hexdigest()[:24] + '.npz')
    with sqlite3.connect(store) as con:
        row = con.execute('SELECT sha,status FROM cases WHERE ck=?', (r['case_key'],)).fetchone()
    G.require(row is not None and row[1] == 'complete' and C.sha(p) == row[0], 'Old pack missing / changed: ' + r['case_key'])
    return p


# Check candidate raster equivalence and reproduce selected stored evaluations.
def equivalence():
    C.install_guard(); t = time.time(); old = old_manifest(); out = {}

    diffs = []
    for r in old.to_dict('records'):
        r = G.rowdict(r); q = G.rowdict(A.plan_row(r['config'], r['surface'], int(r['phase_block']), r['sigma'], int(r['rep']), r['env']))
        d = [k for k in CHECK_COLS if q[k] != r[k]]
        if d:
            diffs.append(dict(case_key=r['case_key'], fields=d))
    out['planner'] = dict(rows=len(old), differing=len(diffs), examples=diffs[:5], fields=CHECK_COLS)
    G.require(not diffs, 'Adapter planner differs from frozen old manifests')

    want = [('point', 'BLUR2', 40, 'blur'), ('point', 'G05', 60, 'lighting'), ('line', 'BLUR2_5', 20, 'blur'), ('line', 'GRAD_Y_REV', 40, 'lighting'),
            ('point', 'GRAD_Y_REV', 0, 'lighting'), ('line', 'UNIFORM', 100, 'lighting')]
    newold = old[(old.action == 'new')]
    picks = []
    for k, (fam, env, s, mod) in enumerate(want):
        g = pd.DataFrame()
        for s2 in [s] + [x for x in G.SIGMAS if x != s]:
            g = newold[(newold.family == fam) & (newold.env == env) & (newold.sigma == s2) & (newold.module == mod)].sort_values('case_key')
            if len(g):
                break
        G.require(len(g) > 0, f'No stored old case for {fam} {env}')
        picks.append(G.rowdict(g.iloc[(7 * k) % len(g)].to_dict()))
    S = A.platform(); PL = S['p']['PL']; cases = []
    for r in picks:
        pr = A.prep(r); p = stored_pack(r)
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z['_meta'])); arr = {k: z[k] for k in z.files if k != '_meta'}
        m = r['method']; raw = {k[len(m) + 7:]: v for k, v in arr.items() if k.startswith(m + '__raw__')}
        evs = {k[len(m) + 6:]: v for k, v in arr.items() if k.startswith(m + '__ev__')}
        ev, rec, _ = PL.evaluate(pr, m, raw, meta['decode_meta'])
        same_ev = sorted(k for k in evs if not (k in ev and np.asarray(ev[k]).shape == evs[k].shape and np.array_equal(np.asarray(ev[k]), evs[k], equal_nan=evs[k].dtype.kind == 'f')))
        sens = G.B.multi_eval({k: np.asarray(v) for k, v in ev.items()}, r['surface'])
        sd = max(abs(a['Ctau'] - b['Ctau']) + abs(a['Cvalid'] - b['Cvalid']) for a, b in zip(sens, meta['sensitivity']))
        c = dict(case_key=r['case_key'], route=r['forward_route'], input_key_equal=pr['input_key'] == meta['input_key'], inputs_equal=C.canon(pr['inputs']) == C.canon(meta['inputs']),
                 forward_key_equal=pr['forward_key'] == meta['forward_key'], ev_arrays_differing=same_ev, metrics_equal=C.canon(rec['metrics']) == C.canon(meta['metrics']),
                 sensitivity_maxdiff=sd, state=rec['state'])
        c['ok'] = bool(c['input_key_equal'] and c['inputs_equal'] and c['forward_key_equal'] and not same_ev and c['metrics_equal'] and sd < 1e-12)
        cases.append(c)
    out['stored_old_cases'] = cases
    G.require(all(c['ok'] for c in cases), 'Stored old cases not reproduced: ' + str([c for c in cases if not c['ok']]))

    s12 = C.csv(G.BASE / 's12_resource01/manifest.csv')
    import sl_lib as SLL
    pos = SLL.positions(); rasters = []
    for cfg, flt in [('D_r1_T8x4', (s12.family == 'point') & (s12.layout == 'T8x4') & (s12.size_mm == 1.0)),
                     ('D_r1_p8x4', (s12.family == 'point') & (s12.layout == 'R8x4') & (s12.size_mm == 1.0) & (s12.section == 'S2')),
                     ('L_w1_s10', (s12.family == 'line') & (s12.layout == 's10_phi0') & (s12.size_mm == 1.0))]:
        g = s12[flt].drop_duplicates(['phase_label', 'raster_key'])
        G.require(len(g) > 0 and not g.phase_label.duplicated().any(), f'S12 raster rows for {cfg}')
        tpl = S['T'][cfg]['row'] if cfg != 'D_r1_p8x4' else S['T']['D_r1_p8x4']['row']
        for q in g.to_dict('records'):
            if G.fam_of(cfg) == 'point':
                u, v = pos[int(q['phase_index'])]; ph = dict(u=float(u), v=float(v))
            else:
                ph = dict(phi=float(q['phase_frac']))
            inp = A.inputs(tpl, G.fam_of(cfg), ph)
            rasters.append(dict(config=cfg, phase=q['phase_label'], s12_raster_key=q['raster_key'], adapter_raster_key=inp['raster_key'], equal=inp['raster_key'] == q['raster_key'],
                                s12_peak=q['peak_target'], adapter_peak=float(inp['peak'])))
    out['s12_rasters'] = dict(n=len(rasters), equal=int(sum(x['equal'] for x in rasters)), cases=rasters)
    G.require(all(x['equal'] for x in rasters), 'Adapter rasters differ from S12 stored rasters')

    mmpp = float(SLL.SL.mmpp()); an = []
    for b in range(4):
        ph = A.phases()['point'][b]
        i1 = A.inputs(S['T']['D_r0.75_p8x4']['row'], 'point', ph); i2 = A.inputs(S['T']['D_r0.75_p8x6']['row'], 'point', ph)
        c2 = np.asarray(i2['centres']); xs = np.unique(np.round(c2[:, 0], 6)); ys = np.unique(np.round(c2[:, 1], 6))
        dpar = np.diff(xs).min() * mmpp; dperp = np.diff(ys).min() * mmpp
        an.append(dict(config='D_r0.75_p8x6', block=b, n_dots=len(c2), min_col_step_mm=dpar, min_row_step_mm=dperp, lit_per_dot=i2['lit_px'] / len(c2),
                       lit_per_dot_N1=i1['lit_px'] / len(i1['centres']), peak=i2['peak'], S0=i2['lit_px'], offset_rows_mm=float(ys.min() * mmpp)))
        lp = A.phases()['line'][b]
        l4 = A.inputs(S['T']['L_w1_s10']['row'], 'line', lp); l5 = A.inputs(S['T']['L_w0.5_s10']['row'], 'line', lp)
        o4, o5 = l4['meta_full']['olines'], l5['meta_full']['olines']
        an.append(dict(config='L_w0.5_s10', block=b, n_lines=int(o5['n_lines']), width_mm=float(o5['width_px']) * mmpp, spacing_mm=float(o5['spacing_px']) * mmpp,
                       n_lines_w1=int(o4['n_lines']), spacing_w1_mm=float(o4['spacing_px']) * mmpp, lit=l5['lit_px'], lit_w1=l4['lit_px'], peak=l5['peak']))
    out['analytic'] = an
    for a in an:
        if a['config'] == 'D_r0.75_p8x6':
            G.require(abs(a['min_col_step_mm'] - 8) < .05 and abs(a['min_row_step_mm'] - 6) < .05 and abs(a['lit_per_dot'] / a['lit_per_dot_N1'] - 1) < .03 and a['peak'] == 1.0, 'N2 analytic check')
        else:
            G.require(abs(a['spacing_mm'] - 10) < 1e-9 and abs(a['width_mm'] - .5) < 1e-9 and a['n_lines'] == a['n_lines_w1'] and abs(a['spacing_w1_mm'] - 10) < 1e-9, 'N5 analytic check')
    out['seconds'] = round(time.time() - t, 1)
    C.save(PREP / 'equivalence.json', out)
    print(C.canon(dict(planner=out['planner']['differing'], stored=[c['ok'] for c in cases], s12_rasters=f"{out['s12_rasters']['equal']}/{out['s12_rasters']['n']}",
                       analytic=len(an), seconds=out['seconds'])))


def judge(df, keys, ds=6.0, tau=.15, floor=.5):

    out = []
    for k, g in df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        s = g.groupby(['surface', 'phase']).Ctau.mean().groupby('surface').mean()
        if set(s.index) != set(G.SURFS):
            continue
        out.append(dict(zip(keys, k), J=float(s.min()), **{f'C_{x}': float(s[x]) for x in G.SURFS}, n_img=len(g), n_phase=int(g.phase.nunique()), bad_img=int((~g.safe).sum()),
                        qualified=bool(s.min() >= floor and g.safe.all())))
    return pd.DataFrame(out)


def bridge():

    C.install_guard(); t = time.time(); A.platform()
    rows = []
    a = pd.read_csv(G.BASE / 's3_fe01/results/A_s12_cases.csv', low_memory=False)
    a = a[(a.rule == 'DF') & a.layout.isin(['R8x4', 'T8x4', 'R8x6']) & (a.size_mm == 1.0) & a.view_regimes.astype(str).str.contains('resource_cap_11695') & a.sigma.isin([0, 20, 40, 60])]
    G.require(not a.input_key.duplicated().any(), 'Duplicate S12 DF inputs')
    for r in a.to_dict('records'):
        with np.load(G.BASE / 's3_fe01/s12/cache' / (r['input_key'] + '.npz'), allow_pickle=False) as z:
            ev = {k: z[k] for k in ('pts', 'valid')}
        sens = G.B.multi_eval(ev, r['surface'])
        rows += [dict(batch='S12 (S3-FE-01 DF)', config={'R8x4': 'D_r1_p8x4', 'T8x4': 'D_r1_T8x4', 'R8x6': 'D_r1_p8x6'}[r['layout']], surface=r['surface'], phase=r['phase_label'],
                      sigma=r['sigma'], rep=r['rep'], stored_main=r['glob_Ctau'], rmse=r['glob_pt_rmse'], c=r['id_correct'], w=r['id_wrong'], u=r['id_unconf'], state=r['state'], **x) for x in sens]
    con = sqlite3.connect(G.HOME / 'PatternLab/data/index.sqlite')
    s5 = pd.read_sql("select config, surface, phase, rep, sigma, state, glob_Ctau, glob_pt_rmse, c, w, u, eval_path, eval_prefix, eval_format, image_key, extra from records "
                     "where dataset='s5_six' and family='point' and view='DF' and config in ('D_r0.75_p8x4','D_r1_p8x6','D_r1_p8x4') and sigma in (0,20,40,60)", con)
    con.close()
    s5['mk'] = s5.extra.map(lambda x: json.loads(x)['manifest_key'])
    G.require(not s5.mk.duplicated().any(), 'Duplicate S5 manifest keys')
    shards = {}
    for r in s5.to_dict('records'):
        p = G.PROJECT / str(r['eval_path'])[2:]
        if p not in shards:
            shards[p] = np.load(p, allow_pickle=False)
        z = shards[p]; pre = r['eval_prefix'] if r['eval_format'] == 'shard' else ''
        G.require(r['eval_format'] in ('shard', 'std'), 'Unknown stored evaluation format')
        ev = {k: z[pre + k] for k in ('pts', 'valid')}
        sens = G.B.multi_eval(ev, r['surface'])
        rows += [dict(batch='S5 six-level (DF)', config=r['config'], surface=r['surface'], phase=r['phase'], sigma=r['sigma'], rep=r['rep'], stored_main=r['glob_Ctau'],
                      rmse=r['glob_pt_rmse'], c=r['c'], w=r['w'], u=r['u'], state=r['state'], **x) for x in sens]
    for z in shards.values():
        z.close()
    T = pd.DataFrame(rows)
    main = T[(T.d_mm == 6) & (T.tau_mm == .15)]
    diff = float((main.Ctau - main.stored_main).abs().max())
    tot = T.c.fillna(0) + T.w.fillna(0) + T.u.fillna(0)
    T['safe'] = (T.state != 'no_output') & T.rmse.notna() & (T.rmse <= .3) & (T.w.fillna(0) == 0) & (T.c.fillna(0) > 0) & ((T.u.fillna(0) / tot.where(tot > 0)).fillna(1) <= .05)
    C.atomic(PREP / 'bridge_cases.csv.gz', lambda p: T.to_csv(p, index=False, compression='gzip'))
    J = []
    for (bt, d, tau), g in T.groupby(['batch', 'd_mm', 'tau_mm']):
        for floor in (.5, .7):
            j = judge(g, ['config', 'sigma'], d, tau, floor); j.insert(0, 'batch', bt); j.insert(1, 'd_mm', d); j.insert(2, 'tau_mm', tau); j.insert(3, 'floor', floor); J.append(j)
    J = pd.concat(J, ignore_index=True); C.save_csv(PREP / 'bridge_selection.csv', J)
    rec = dict(cases=int(len(T) / 8), main_maxdiff_vs_stored=diff, seconds=round(time.time() - t, 1),
               note='Condition-matched chapter-3 summaries (8 phases), not chapter-4 paired cases; stored main C_tau reproduced by the chapter-4 evaluator')
    C.save(PREP / 'bridge.json', rec)
    G.require(diff < 1e-9, f'Chapter-4 evaluator does not reproduce stored chapter-3 main coverage ({diff})')
    print(C.canon(rec))
    print(J[(J.floor == .5) & (J.tau_mm.isin([.1, .15]))].pivot_table(index=['batch', 'config', 'd_mm', 'tau_mm'], columns='sigma', values='J').round(3).to_string())


def drills():

    import cb_run as RUN
    C.install_guard(); res = []
    M = C.csv(MANIFEST); r0 = G.rowdict(M[(M.sigma == 20) & (M.family == 'point')].iloc[0]); r00 = G.rowdict(M[(M.sigma == 0) & (M.family == 'point')].iloc[0])
    def good(r, n=40):
        return dict({k: r[k] for k in C.IDENTITY}, state='ok', glob_Ctau=.5, glob_Cvalid=.6, glob_pt_rmse=.05, glob_pt_p95=.1, glob_n_valid=n, n_valid=n, n_accepted=n, c=n, w=0, u=0)
    def expect_stop(name, fn):
        try:
            fn()
        except (C.Stop, ValueError, KeyError) as e:
            res.append(dict(drill=name, refused=True, reason=str(e)[:160])); return
        res.append(dict(drill=name, refused=False)); raise C.Stop('Drill accepted corrupted input: ' + name)
    exp = pd.DataFrame([r0, r00])
    expect_stop('missing row', lambda: C.metric_gate(pd.DataFrame([good(r0)]), exp))
    expect_stop('duplicate rep', lambda: C.metric_gate(pd.DataFrame([good(r0), good(r0), good(r00)]), exp))
    expect_stop('identity missing', lambda: C.metric_gate(pd.DataFrame([dict(good(r0), c=None), good(r00)]), exp))
    expect_stop('zero points with positive coverage', lambda: C.metric_gate(pd.DataFrame([dict(good(r0, 0), glob_pt_rmse=None, glob_pt_p95=None, state='no_output'), good(r00)]), exp))
    expect_stop('valid points without RMSE', lambda: C.metric_gate(pd.DataFrame([dict(good(r0), glob_pt_rmse=None), good(r00)]), exp))
    expect_stop('wrong seed / rep identity', lambda: C.metric_gate(pd.DataFrame([dict(good(r0), seed=r0['seed'] + 1), good(r00)]), exp))
    expect_stop('sigma0 duplicated as two reps', lambda: RUN.result_gate(pd.DataFrame([good(r00), dict(good(r00), rep=1, case_key=r00['case_key'].replace('r0', 'r1'))]), exp.iloc[[1]]))
    expect_stop('cartesian pairing', lambda: RUN.pair(pd.DataFrame([good(r0), good(r0)]).assign(config='a'), pd.DataFrame([good(r0)]).assign(config='b'), 'a', 'b'))

    root = G.OUT / 'drill'
    st = C.Store(root); fid = 'DRILL'
    rr = dict(r0, case_key='DRILL|' + r0['case_key']); arrays = synthetic_arrays(rr)
    meta = dict(case_key=rr['case_key'], forward_key=rr['forward_key'], method=rr['method'], result=dict(good(rr, 3), case_key=rr['case_key'], glob_pt_p95=.05))
    with st.connect() as con:
        con.execute('DELETE FROM cases'); con.execute('DELETE FROM events')
    for p in (root / 'data').glob('*'):
        p.unlink()
    st.reserve(rr, fid); st.commit(rr, fid, meta, arrays)
    G.require(st.resume(rr, fid) is not None, 'Resume of committed pack failed'); res.append(dict(drill='resume committed pack without rerender', refused=False, ok=True))
    expect_stop('second reservation of a committed case', lambda: st.reserve(rr, fid))
    expect_stop('stale freeze binding on resume', lambda: st.resume(rr, 'OTHER'))
    p = st.path(rr['case_key']); b = p.read_bytes(); p.write_bytes(b[:-7] + b'corrupt')
    expect_stop('corrupted pack', lambda: st.resume(rr, fid))
    rr2 = dict(rr, case_key=rr['case_key'] + '|E'); st.reserve(rr2, fid); st.fail(rr2['case_key'], 'drill: worker exception')
    expect_stop('technical stop is not completion', lambda: st.resume(rr2, fid))
    expect_stop('rule change refused', lambda: RUN.check_policy(dict(C.POLICY, tie_band=.03)))
    C.save(PREP / 'drills.json', dict(synthetic=True, drills=res, n=len(res), all_ok=all(x.get('refused') or x.get('ok') for x in res)))
    print(C.canon(dict(drills=len(res), refused=sum(bool(x.get('refused')) for x in res))))


def synthetic_arrays(r):
    n = 3; m = r['method']; ev = dict(pts=np.array([[0, 0, -1.], [1, 0, -1.], [0, 1, -1.]]), accepted=np.ones(n, bool), valid=np.ones(n, bool), in_global=np.ones(n, bool),
                                       zr=np.array([-1.05, -1.05, -1.05]), point_correct=np.ones(n, bool), point_wrong=np.zeros(n, bool), point_unconf=np.zeros(n, bool))
    a = dict(image=np.zeros((512, 512), np.uint8)); a[m + '__raw__pts'] = ev['pts']
    a.update({m + '__ev__' + k: v for k, v in ev.items()})
    return a


def freeze():
    import cb_run as RUN
    C.install_guard(); M = C.csv(MANIFEST); gate_main(M, entry_cases())
    G.require(M.method_bundle_sha256.eq(bundle()).all(), 'Manifest bound to another adapter version')
    for f in ('equivalence.json', 'bridge.json', 'drills.json'):
        G.require((PREP / f).exists(), 'Preparation step missing: ' + f)
    G.require(C.load(PREP / 'drills.json')['all_ok'], 'Drills failed')
    old = C.load(G.OLD['light_freeze']); C.verify_files(old['files'])
    files = dict(old['files'])
    A.platform(); RUN.touch_runtime()
    for p in C.READS:
        if Path(p).is_file() and not str(p).endswith('.npz'):
            files[p] = C.sha(p)
    for p in list(G.HERE.glob('*.py')) + [MANIFEST, PREP / 'new_templates.json', PREP / 'rasters.csv', G.OLD['blur_manifest'], G.OLD['light_manifest'],
                                         G.OLD['blur_results'], G.OLD['blur_sens'], G.OLD['light_results'], G.OLD['light_sens']]:
        files[str(Path(p).resolve())] = C.sha(p)
    import importlib.metadata
    f = dict(files=files, freeze_id=C.digest(files), manifest_sha256=C.sha(MANIFEST), policy=C.POLICY, bundle=bundle(), caps=G.CAPS,
             packages={k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'matplotlib']}, python=sys.version, created=time.time())
    C.save(G.OUT / 'freeze.json', f)
    print(C.canon(dict(freeze_id=f['freeze_id'], files=len(files), manifest=f['manifest_sha256'][:16])))


if __name__ == '__main__':
    stage = sys.argv[1]
    try:
        dict(manifest=manifest, equivalence=equivalence, bridge=bridge, drills=drills, freeze=freeze)[stage]()
    except Exception:
        traceback.print_exc(); sys.exit(2)
