# Bind joint-study results and compare their identities with registered inputs.

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import s5r_common as K
import s5r_gates as G
import s5r_store as S

_A2CODE = K.ROOT / 'scripts' / 'surface_revision' / 's5_align02'
DF_COMBOS = tuple(K.DOT9)
KEEP_ARR_DEC = ('pts', 'accepted', 'feats')
KEEP_ARR_EVAL = ('valid', 'zr', 'in_local', 'in_global', 'point_correct', 'point_wrong', 'point_unconf')


def _cases() -> pd.DataFrame:
    A = K.read_csv(K.reg(K.S345 / 'results' / 'S5-main_cases.csv')).assign(set_name='S5-main')
    B = K.read_csv(K.reg(K.S345 / 'results' / 'S5-gauss_cases.csv')).assign(set_name='S5-gauss')
    C = pd.concat([A, B], ignore_index=True)
    Sp = K.read_csv(K.reg(K.S345 / 'results' / 'S5_cases_supplement.csv'))
    C = C.merge(Sp[['decode_key', 'geometry_id', 'phase_offset_par_mm', 'phase_offset_perp_mm', 'peak_actual', 'lit_px', 'integral_actual', 'sup_surf_rmse', 'n_detected',
                    'glob_n_supported_nodes']], on='decode_key', how='left', validate='one_to_one')
    C['sup_surf_rmse'] = pd.to_numeric(C.sup_surf_rmse, errors='coerce')
    if len(C) != 1536 or C.decode_key.duplicated().any():
        raise K.Stop('scope', f'S5 cases {len(C)} rows / duplicated decode keys')
    return C


def dependency_state() -> dict:

    fz = K.load_json(K.reg(K.S345 / 'freeze_s345_ready.json'))['files']
    c5 = K.load_json(K.reg(K.C5 / 'formal' / 'frozen.json'))['closure']['files']
    a2 = K.load_json(K.reg(K.A2 / 'run.json'))['final']
    files = sorted(set(sum(S.DEC_DEPS.values(), [])) | set(S.EVAL_DEPS) | set(S.RENDER_DEPS))
    rows = []
    for f in files:
        cur = S.fhash(f); cov = []
        for name, ref in (('s345_ready freeze', fz), ('S3-CODE-05 closure', c5)):
            if f in ref:
                cov.append(dict(freeze=name, equal=bool(ref[f] == cur)))
        rows.append(dict(file=f, sha256=cur, covered_by=cov, ok=bool(cov) and all(c['equal'] for c in cov)))
    a2_out = {k: v for k, v in a2['outputs'].items() if k.startswith('outputs/surface_revision/s5_align02/df/') or k.endswith('manifest_next.csv')}
    a2_bad = [k for k, v in a2_out.items() if K.sha(K.ROOT / k) != v]
    a2_code_bad = [k for k, v in a2['code'].items() if K.sha(K.ROOT / k) != v]
    s345_bad = [k for k, v in fz.items() if not (K.ROOT / k).exists() or K.sha(K.ROOT / k) != v]
    man = K.load_json(K.reg(K.SR / 'surf02' / 'release' / 'dependencies_surf02_v1_1.json'))['files']
    s2_bad = [k for k, h in man.items() if K.sha(K.ROOT / k) != h]
    return dict(files=rows, all_deps_ok=all(r['ok'] for r in rows), s5_align02_df_and_manifest_changed=a2_bad, s5_align02_code_changed=a2_code_bad,
                s345_freeze_changed=s345_bad, surf02_changed=s2_bad,
                ok=bool(all(r['ok'] for r in rows) and not a2_bad and not a2_code_bad and not s345_bad and not s2_bad))


def _df_shard(combo: str) -> tuple[dict, dict]:
    j = K.load_json(K.reg(K.A2 / 'df' / f'df_{combo}.json')); z = K.load_npz(K.reg(K.A2 / 'df' / f'df_{combo}.npz'))
    return j, z


def compat(write: bool = True) -> dict:
    dep = dependency_state()
    if not dep['ok']:
        raise K.Stop('version', 'a dependency of the existing records changed; reuse stopped', detail=dep)
    pr = S.platform_refs(); C = _cases(); P = K.read_csv(K.reg(K.S345 / 'plan_s345.csv'))
    P = P[P.set_formal.isin(['S5-main', 'S5-gauss'])].set_index('decode_key')
    rec, bad = {}, []
    for r in C.itertuples():
        ij = K.S345 / 'run' / 'images' / f'{r.image_key}.json'; I = K.load_json(K.reg(ij))
        meth = 'D0' if r.family == 'point' else 'P4'
        rj = K.S345 / 'run' / 'raw' / f'{r.image_key}__{meth}.json'; Rw = K.load_json(K.reg(rj))
        ej = K.S345 / 'run' / 'eval' / f'{r.decode_key}.json'; E = K.load_json(K.reg(ej))
        shas = {p: K.sha(K.reg(p.with_suffix('.npz'))) for p in (ij, rj, ej)}
        for p, rr in ((ij, I), (rj, Rw), (ej, E)):
            if rr.get('complete') is not True or shas[p] != rr['npz_sha256']:
                bad.append(K.rel(p))
        f = {k: v for k, v in I['identity_fields'].items() if k != 'image_part'}
        iid = S.image_identity(f, pr); did = S.decode_identity(iid, shas[ij], meth, S.method_params(meth, f['peak_f32'], P.loc[r.decode_key, 'radius_mm']))
        eid = S.eval_identity(shas[rj], pr)
        rec[r.decode_key] = dict(key=[r.pair_role, r.surface, r.phase_label, float(r.sigma), int(r.rep)], set=r.set_name, method=meth, image_key=r.image_key,
                                 image=dict(json=K.rel(ij), npz_sha256=shas[ij], stored_identity=I['identity'], identity=iid),
                                 decode=dict(json=K.rel(rj), npz_sha256=shas[rj], stored_identity=Rw['identity'], identity=did),
                                 eval=dict(json=K.rel(ej), npz_sha256=shas[ej], stored_identity=E['identity'], identity=eid))
    df = {}
    a2o = K.load_json(K.A2 / 'run.json')['final']['outputs']
    for combo in DF_COMBOS:
        j, z = _df_shard(combo)
        nz, nj = K.rel(K.A2 / 'df' / f'df_{combo}.npz'), K.rel(K.A2 / 'df' / f'df_{combo}.json')
        if K.sha(K.ROOT / nz) != a2o[nz] or K.sha(K.ROOT / nj) != a2o[nj] or K.sha(K.ROOT / nz) != j['npz_sha256'] or j.get('complete') is not True:
            bad.append(nz)
        rows = {x['image_key']: x for x in j['rows'] if x['front_end'] == 'DF'}
        if sorted(rows) != sorted(j['image_keys']) or len(rows) != 72:
            bad.append(f'{nz}: declared image set differs from its rows')
        ent = {}
        for ik, x in rows.items():
            main = rec[x['decode_key']]
            arr = {a: z[f'{ik}__{a}'] for a in KEEP_ARR_DEC + KEEP_ARR_EVAL}
            dc = S.content_digest({a: arr[a] for a in KEEP_ARR_DEC}); ec = S.content_digest({a: arr[a] for a in KEEP_ARR_EVAL})
            f = K.load_json(K.ROOT / main['image']['json'])['identity_fields']; f = {k: v for k, v in f.items() if k != 'image_part'}
            did = S.decode_identity(main['image']['identity'], main['image']['npz_sha256'], 'DF', S.method_params('DF', f['peak_f32'], x['radius_mm']))
            ent[ik] = dict(key=main['key'], decode_key=x['decode_key'], image_identity=main['image']['identity'], decode_identity=did, decode_content=dc,
                           eval_identity=S.eval_identity(dc, pr), eval_content=ec)
        df[combo] = dict(npz=nz, npz_sha256=K.sha(K.ROOT / nz), json=nj, json_sha256=K.sha(K.ROOT / nj), image_keys_digest=K.digest(sorted(j['image_keys']), 32), n_images=len(ent), images=ent)
    out = dict(note='registered post hoc in S5-ALIGN-03 (Nachregistrierung); NOT a freeze made before the original computation', platform=pr,
               dependencies=dep, n_records=len(rec), n_df_images=sum(v['n_images'] for v in df.values()), mismatches=bad, records=rec, df=df, t=K.now())
    if bad:
        raise K.Stop('corrupt', f'{len(bad)} stored files / shards differ from their records; reuse stopped', detail=bad[:10])
    if write:
        K.run_put('compat', out); K.flush('compat')
    return out


def verify_binding(cp: dict) -> dict:

    fresh = compat(write=False); bad = []
    for dk, r in cp['records'].items():
        f = fresh['records'].get(dk)
        if f is None or any(f[s][x] != r[s][x] for s in ('image', 'decode', 'eval') for x in ('npz_sha256', 'identity')):
            bad.append(dk)
    for combo, d in cp['df'].items():
        f = fresh['df'][combo]
        if f['npz_sha256'] != d['npz_sha256'] or f['image_keys_digest'] != d['image_keys_digest'] or any(f['images'][ik] != v for ik, v in d['images'].items()):
            bad.append(combo)
    return dict(ok=not bad, mismatches=bad[:10], n_checked=len(cp['records']) + len(cp['df']))


def views(cp: dict) -> dict:
    C = _cases().set_index('decode_key'); rows_main, rows_fail, rows_gauss, rows_df, arr_cons = [], [], [], [], []
    for dk, r in cp['records'].items():
        E = K.load_json(K.ROOT / r['eval']['json']); m = E['metrics']
        row = dict(combo=r['key'][0], surface=r['key'][1], phase_label=r['key'][2], sigma=r['key'][3], rep=r['key'][4], method=r['method'], image_key=r['image_key'], decode_key=dk,
                   record_complete=E.get('complete') is True, content_sha=r['eval']['npz_sha256'], state=m['state'], sup_surf_rmse=C.loc[dk, 'sup_surf_rmse'],
                   **{k: m.get(k) for k in ('n_out', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'id_denominator',
                                            'glob_id_correct', 'glob_id_wrong', 'glob_id_unconf', 'loc_id_correct', 'loc_id_wrong', 'loc_id_unconf', 'glob_Ctau', 'glob_Cvalid',
                                            'glob_cov_interp', 'loc_Ctau', 'loc_Cvalid', 'loc_cov_interp', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95',
                                            'glob_surf_rmse', 'loc_surf_rmse', 'glob_n_interp', 'loc_n_interp', 'glob_recovery', 'loc_recovery', 'glob_pt_max')})
        if r['set'] == 'S5-gauss':
            rows_gauss.append(row)
        elif r['key'][0] in K.COMBOS:
            rows_main.append(row)
        else:
            rows_fail.append(row)
    for combo, d in cp['df'].items():
        j, z = _df_shard(combo); rows = {x['image_key']: x for x in j['rows'] if x['front_end'] == 'DF'}
        for ik, e in d['images'].items():
            x = rows[ik]; acc = z[f'{ik}__accepted'].astype(bool); val = z[f'{ik}__valid'].astype(bool); ing = z[f'{ik}__in_global'].astype(bool); inl = z[f'{ik}__in_local'].astype(bool)
            pc, pw, pu = (z[f'{ik}__{a}'].astype(bool) for a in ('point_correct', 'point_wrong', 'point_unconf'))
            pts = z[f'{ik}__pts']; e_pts = pts[:, 2] - z[f'{ik}__zr'].astype(np.float64)
            view = dict(n_out=int(len(pts)), glob_id_correct=int((pc & ing).sum()), glob_id_wrong=int((pw & ing).sum()), glob_id_unconf=int((pu & ing).sum()),
                        loc_id_correct=int((pc & inl).sum()), loc_id_wrong=int((pw & inl).sum()), loc_id_unconf=int((pu & inl).sum()))
            cons = dict(n_accepted=int(acc.sum()) == x['n_accepted'], n_valid=int(val.sum()) == x['n_valid'], glob_n_valid=int(ing.sum()) == x['glob_n_valid'],
                        loc_n_valid=int(inl.sum()) == x['loc_n_valid'], ids=(int((pc & acc).sum()), int((pw & acc).sum()), int((pu & acc).sum())) == (x['id_correct'], x['id_wrong'], x['id_unconf']))
            dmax = 0.0
            for dname, msk in (('glob', ing), ('loc', inl)):
                ee = e_pts[msk]; ee = ee[np.isfinite(ee)]
                rm = float(np.sqrt(np.mean(ee * ee))) if ee.size else None
                st = x.get(f'{dname}_pt_rmse')
                dmax = max(dmax, abs(rm - st) if (rm is not None and st is not None) else (0.0 if rm is None and st is None else float('inf')))
            arr_cons.append(dict(all_equal=all(cons.values()), pt_rmse_max_abs_diff=dmax))
            rows_df.append(dict(combo=e['key'][0], surface=e['key'][1], phase_label=e['key'][2], sigma=e['key'][3], rep=e['key'][4], method='DF', image_key=ik, decode_key=e['decode_key'],
                                record_complete=True, content_sha=e['eval_content'], **view, **{k: x.get(k) for k in ('state', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid',
                                'id_correct', 'id_wrong', 'id_unconf', 'id_denominator', 'glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'loc_Ctau', 'loc_Cvalid', 'loc_cov_interp',
                                'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'loc_surf_rmse', 'sup_surf_rmse', 'glob_recovery', 'loc_recovery', 'glob_pt_max')}))
    AC = pd.DataFrame(arr_cons)
    return dict(main=pd.DataFrame(rows_main + rows_df), failure=pd.DataFrame(rows_fail), gauss=pd.DataFrame(rows_gauss),
                df_array_view=dict(n=len(AC), all_counts_equal=int(AC.all_equal.sum()), pt_rmse_max_abs_diff=float(AC.pt_rmse_max_abs_diff.replace(np.inf, np.nan).max()),
                                   n_inf=int(np.isinf(AC.pt_rmse_max_abs_diff).sum())))


def expected_sets() -> dict:
    main = {(c, s, p, sg, r, K.MAIN_METHOD[K.family(c)]) for (c, s, p, sg, r) in K.expected_keys(sigmas=K.EXISTING_SIGMAS)}
    df = {(c, s, p, sg, r, 'DF') for (c, s, p, sg, r, _m) in main if c in K.DOT9}
    fail = {(c, s, p, sg, r, 'D0') for (c, s, p, sg, r) in K.expected_keys(combos=('D_r1.5_p8x3',), sigmas=K.EXISTING_SIGMAS)}
    gcombos = tuple(K.COMBOS) + ('D_r1.5_p8x3',)
    gauss = {(c, s, p, sg, r, K.MAIN_METHOD[K.family(c)]) for (c, s, p, sg, r) in K.expected_keys(combos=gcombos, surfaces=('M10_curv_gauss',), sigmas=K.EXISTING_SIGMAS)}
    return dict(main=main | df, failure=fail, gauss=gauss)


def _st():
    if str(_A2CODE) not in sys.path:
        sys.path.append(str(_A2CODE))
    import s5c_stats as ST
    return ST


_REASON = __import__('re').compile(r'(ph\d): (\S+) undefined in (?:cell )?(\S+)')


def _norm(d: pd.DataFrame) -> pd.DataFrame:

    d = d.copy(); d['sigma'] = d.sigma.astype(str)
    if 'n_phases_complete' not in d:
        d['n_phases_complete'] = np.nan
    d['n_phases_complete'] = pd.to_numeric(d.n_phases_complete, errors='coerce').where(d.state != 'ok', 8)
    if 'reason' in d:
        d['reason'] = [('|'.join(_REASON.search(str(x)).groups()) if isinstance(x, str) and _REASON.search(str(x)) else ('' if not isinstance(x, str) else x)) for x in d.reason]
    d['n_phases_planned_'] = 8
    d['n_reps_planned_'] = d.sigma.map({'0.0': 1.0, '40.0': 2.0}).astype(float)
    return d


def _cmp(new: pd.DataFrame, ref: pd.DataFrame, key: list) -> dict:
    den = None
    if 'n_phases_planned' in ref:
        r2 = _norm(ref)
        den = bool((r2.n_phases_planned == r2.n_phases_planned_).all() and (pd.to_numeric(r2.n_reps_planned, errors='coerce').fillna(-1) == r2.n_reps_planned_.fillna(-1)).all())
    ref = _norm(ref); new = _norm(new)
    M = new.merge(ref, on=key, how='outer', suffixes=('', '_ref'), indicator=True); both = M[M._merge == 'both']
    num = [c for c in ('mean', 'median', 'min', 'max', 'res_lo_5', 'res_hi_95', 'ph0', 'ph1', 'ph2', 'ph3', 'ph4', 'ph5', 'ph6', 'ph7') if c in new and f'{c}_ref' in M]
    dmax = max(float((pd.to_numeric(both[c], errors='coerce') - pd.to_numeric(both[f'{c}_ref'], errors='coerce')).abs().max()) for c in num) if len(both) else None
    nan_eq = all(bool((pd.to_numeric(both[c], errors='coerce').isna() == pd.to_numeric(both[f'{c}_ref'], errors='coerce').isna()).all()) for c in num)
    cnt_eq = all(bool((pd.to_numeric(both[c], errors='coerce').fillna(-1) == pd.to_numeric(both[f'{c}_ref'], errors='coerce').fillna(-1)).all())
                 for c in ('n_pos', 'n_neg', 'n_zero', 'n_phases_complete') if c in new and f'{c}_ref' in M)
    st_eq = bool((both.state == both.state_ref).all())
    reason_eq = bool((both.reason.fillna('') == both.reason_ref.fillna('')).all()) if 'reason' in new and 'reason_ref' in M else True
    return dict(n_new=len(new), n_ref=len(ref), only_new=int((M._merge == 'left_only').sum()), only_ref=int((M._merge == 'right_only').sum()), max_abs_diff=dmax,
                undefined_pattern_equal=nan_eq, counts_equal=cnt_eq, state_equal=st_eq, reason_equal=reason_eq, n_undefined=int((new.state != 'ok').sum()),
                stored_denominators_match_design=den,
                ok=bool((M._merge == 'both').all() and dmax is not None and dmax <= 1e-12 and nan_eq and cnt_eq and st_eq and reason_eq and den is not False))


def replay(V: dict) -> dict:
    ST = _st(); idx = ST.resample_idx(); C = _cases()

    allv = pd.concat([V['main'][V['main'].method != 'DF'], V['failure'], V['gauss']])
    bound = allv.set_index('decode_key')
    for c in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse'):
        a = pd.to_numeric(C.set_index('decode_key')[c], errors='coerce'); b = pd.to_numeric(bound[c], errors='coerce').reindex(a.index)
        if not ((a - b).abs().fillna(0) <= 0).all() or not (a.isna() == b.isna()).all():
            raise K.Stop('refused', f'case table column {c} differs from the bound eval records')
    T = ST.interactions(C, ST.K.FOUR, True, idx)
    reg = K.read_csv(K.reg(K.S345 / 'results' / 'S5_interactions.csv'))
    key = ['set_name', 'surface', 'family', 'sigma', 'set', 'effect', 'metric']
    r693 = _cmp(T, reg, key)
    H = ST.interactions(C[C.family == 'line'], ST.K.FOUR_POSTHOC, False, idx)
    r180 = _cmp(H, K.read_csv(K.reg(K.A2 / '_parts' / 'A_interactions_posthoc_63.csv')), key)

    dm = V['main']; dots = dm[dm.combo.isin(K.DOT9)].copy()
    R = dots.rename(columns={'combo': 'pair_role'}).assign(set_name='S5-main', family='point', front_end=lambda d: np.where(d.method == 'DF', 'DF', 'D0'))
    for c in ('glob_Ctau', 'loc_Ctau', 'glob_pt_rmse'):
        R[c] = pd.to_numeric(R[c], errors='coerce')
    F = ST.interactions(R, ST.K.FOUR, False, idx, metrics=('glob_Ctau', 'loc_Ctau', 'glob_pt_rmse'), extra_keys=('front_end',))
    diff = []
    for k, g in F[F.state == 'ok'].groupby(['surface', 'sigma', 'set', 'effect', 'metric']):
        if set(g.front_end) != {'D0', 'DF'}:
            continue
        e = g[g.front_end == 'DF'].iloc[0][ST.K.PHASES].to_numpy(float) - g[g.front_end == 'D0'].iloc[0][ST.K.PHASES].to_numpy(float)
        diff.append(dict(record_type='interaction_df_minus_d0', set_name='S5-main', family='point', **dict(zip(['surface', 'sigma', 'set', 'effect', 'metric'], k)), front_end='DF-D0', **ST.summarise(e, idx)))
    F2 = pd.concat([F, pd.DataFrame(diff)], ignore_index=True)
    r486 = _cmp(F2, K.read_csv(K.reg(K.A2 / '_parts' / 'C_df_interactions.csv')), key + ['front_end'])
    m24 = F2[(F2.surface == 'M24_curv') & (F2.set == 'dots_parallel') & (F2.effect == 'I') & (F2.metric == 'glob_Ctau') & (F2.sigma == '40.0') & F2.front_end.isin(['D0', 'DF'])]
    signs = {r.front_end: f'{int(r.n_pos)}/{int(r.n_neg)}' for r in m24.itertuples()}
    return dict(registered_693=r693, posthoc_180=r180, front_end_486=r486, m24_dots_parallel_sigma40_signs=signs, rows=dict(registered=len(T), posthoc=len(H), front_end=len(F2)),
                ok=bool(r693['ok'] and r180['ok'] and r486['ok']))


def errata(V: dict) -> dict:
    dm = V['main']; d0 = dm[(dm.method == 'D0') & (pd.to_numeric(dm.id_wrong, errors='coerce') > 0)]
    pts = []
    for r in d0.itertuples():
        z = K.load_npz(K.reg(K.S345 / 'run' / 'eval' / f'{r.decode_key}.npz'))
        w = z['point_wrong'].astype(bool) & z['accepted'].astype(bool)
        e = np.abs(z['pts'][w, 2] - z['zr'][w].astype(np.float64))
        for i, v in enumerate(e):
            pts.append(dict(decode_key=r.decode_key, surface=r.surface, combo=r.combo, phase=r.phase_label, rep=r.rep, abs_err_mm=float(v),
                            valid=bool(z['valid'][w][i]), in_global=bool(z['in_global'][w][i]), in_local=bool(z['in_local'][w][i])))
    W = pd.DataFrame(pts)
    df = dm[dm.method == 'DF']; d0all = dm[(dm.method == 'D0') & dm.combo.isin(K.DOT9)]
    tot = {fe: dict(accepted=int(pd.to_numeric(x.n_accepted).sum()), correct=int(pd.to_numeric(x.id_correct).sum()), wrong=int(pd.to_numeric(x.id_wrong).sum()))
           for fe, x in (('D0', d0all), ('DF', df))}
    tot40 = {fe: dict(accepted=int(pd.to_numeric(x[x.sigma == 40.0].n_accepted).sum()), correct=int(pd.to_numeric(x[x.sigma == 40.0].id_correct).sum()))
             for fe, x in (('D0', d0all), ('DF', df))}
    tot0 = {fe: int(pd.to_numeric(x[x.sigma == 0.0].id_correct).sum()) for fe, x in (('D0', d0all), ('DF', df))}
    pair = []
    for r in d0.itertuples():
        x = df[df.image_key == r.image_key].iloc[0]
        pair.append(dict(surface=r.surface, phase=r.phase_label, rep=r.rep, d0_glob_pt_rmse=K.num(r.glob_pt_rmse), df_glob_pt_rmse=K.num(x.glob_pt_rmse), d0_wrong=int(r.id_wrong), df_wrong=int(x.id_wrong)))
    per_surf = {}
    for s in K.MAIN:
        a = d0all[(d0all.surface == s) & (d0all.combo == 'D_r1.5_p6x4') & (d0all.sigma == 40.0)]; b = df[(df.surface == s) & (df.combo == 'D_r1.5_p6x4') & (df.sigma == 40.0)]
        per_surf[s] = dict(d0_max_rmse=float(pd.to_numeric(a.glob_pt_rmse).max()), df_max_rmse=float(pd.to_numeric(b.glob_pt_rmse).max()))
    L = K.read_csv(K.reg(K.A2 / '_parts' / 'B_case_link.csv'))
    above = L[L.gap_glob_Ctau > 1e-12]
    CT = K.read_csv(K.reg(K.A2 / '_parts' / 'D_contexts.csv')).set_index(['pair_role', 'phase_label'])
    cent = {}
    for pitch in ('4.45', '6.3', '8'):
        dd = [float(np.hypot(CT.loc[(f'L_w1.5_s{pitch}', p), 'lit_centroid_u_px'] - CT.loc[(f'L_w0.5_s{pitch}', p), 'lit_centroid_u_px'],
                             CT.loc[(f'L_w1.5_s{pitch}', p), 'lit_centroid_v_px'] - CT.loc[(f'L_w0.5_s{pitch}', p), 'lit_centroid_v_px'])) for p in K.PHASES]
        refd = []
        for s in K.MAIN:
            m0 = L[(L.set_name == 'S5-main') & (L.surface == s) & (L.sigma == 0.0)]
            a = m0[m0.pair_role == f'L_w0.5_s{pitch}'].set_index('phase_label').ref_glob_C_tau_d6; b = m0[m0.pair_role == f'L_w1.5_s{pitch}'].set_index('phase_label').ref_glob_C_tau_d6
            refd.append((b - a).reindex(list(K.PHASES)))
        rd = pd.concat(refd)
        cent[pitch] = dict(lit_centroid_shift_px_max=max(dd), n_phase_same_centroid=int(sum(x < 1e-9 for x in dd)), ref_Ctau_diff_mean_pp=[float(100 * x.mean()) for x in refd],
                           ref_Ctau_diff_single_phase_abs_max_pp=float(100 * rd.abs().max()))
    return dict(wrong_points=dict(n=len(W), abs_err_range_mm=[float(W.abs_err_mm.min()), float(W.abs_err_mm.max())], n_valid=int(W.valid.sum()), n_in_global_valid=int((W.valid & W.in_global).sum()),
                                  valid_errors_mm=sorted(float(x) for x in W[W.valid & W.in_global].abs_err_mm), n_in_local=int(W.in_local.sum()), points=W.to_dict('records')),
                accepted_totals_all_648=tot, accepted_totals_sigma40_432=tot40, correct_sigma0=tot0, paired_rmse_wrong_images=pair, per_surface_max_rmse_r15_6x4_s40=per_surf,
                cases_actual_above_reference=int(len(above)), line_lit_centroid_and_reference=cent)


def analyze(mode: str = 'full', subset_keys=None, view_hook=None, write: bool = True) -> dict:
    cp = K.run_load().get('compat')
    if not cp:
        raise K.Stop('refused', 'no compatibility registration: run `compat` first')
    M = K.read_csv(K.reg(K.MANIFEST)); reg = G.registered()
    gm = G.gate_manifest(M, 'full', None, reg)
    if not gm['ok']:
        raise K.Stop('refused', 'manifest gate failed', detail=gm)
    vb = verify_binding(cp)
    if not vb['ok']:
        raise K.Stop('refused', 'source binding of the consumed records failed', detail=vb)
    V = views(cp)
    if view_hook is not None:
        V = view_hook(V)
    ex = expected_sets()
    if mode == 'subset':
        if not subset_keys:
            raise K.Stop('refused', 'subset analysis needs a pre-bound key set')
        sk = {tuple(x) for x in subset_keys}
        ex = {k: {e for e in v if e[:5] in sk} for k, v in ex.items()}
        V = {k: (v[[tuple(x) in sk for x in v[['combo', 'surface', 'phase_label', 'sigma', 'rep']].itertuples(index=False, name=None)]] if isinstance(v, pd.DataFrame) else v) for k, v in V.items()}
    gates = {name: G.gate_results(V[name], ex[name], mode, name) for name in ('main', 'failure', 'gauss')}
    if not all(g['ok'] for g in gates.values()):
        raise K.Stop('refused', 'result gate failed - no statistics, no report', detail={k: dict(ok=g['ok'], by_check=g['by_check'], first=g['reasons'][:3]) for k, g in gates.items()})
    out = dict(mode=mode, manifest_gate=dict(ok=gm['ok'], n_rows=gm['n_rows'], n_expected=gm['n_expected'], integral_range=gm['integral_range']), binding=vb,
               result_gates={k: {x: v for x, v in g.items() if x != 'reasons'} for k, g in gates.items()}, df_array_view=V['df_array_view'])
    if mode == 'full':
        out['replay'] = replay(V); out['errata'] = errata(V)
        cons = [(d, r['eval']['identity'], r['eval']['npz_sha256']) for d, r in cp['records'].items()] + \
               [(e['decode_key'] + '__DF', e['eval_identity'], e['eval_content']) for d in cp['df'].values() for e in d['images'].values()]
        out['analysis_identity'] = S.analysis_identity(cons, ['set_name', 'surface', 'family', 'sigma', 'set', 'effect', 'metric', 'front_end'],
                                                       dict(seed=2026101204, n_res=10000, order='rep mean in phase -> four-cell -> 8-phase mean'),
                                                       {K.rel(p): K.sha(p) for p in sorted(list(_A2CODE.glob('s5c_stats.py')) + list(K.HERE.glob('s5r_*.py')))})
        if not out['replay']['ok']:
            raise K.Stop('refused', 'replay differs from the stored statistics', detail=out['replay'])
    else:
        out['label'] = f'subset technical check - denominators: { {k: len(v) for k, v in ex.items()} } (not a complete delivery)'
    if write:
        K.run_put('analysis_full' if mode == 'full' else 'analysis_subset', out); K.flush('analyze')
    return out
