# Pair environment cases and summarize changes relative to nominal imaging.

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import mp3_common as K
import mp3_design as D

METRICS = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'glob_surf_rmse', 'sup_surf_rmse', 'n_valid', 'glob_n_valid',
           'loc_n_valid', 'c', 'w', 'u', 'w_share', 'u_share')
EFFECT_METRICS = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'sup_surf_rmse', 'n_valid', 'c', 'w', 'u', 'w_share', 'u_share')
REQ_COLUMNS = ('row_key', 'image_key', 'family', 'combo', 'surface', 'phase_label', 'sigma', 'rep', 'seed', 'env', 'method', 'source', 'state') + METRICS
KEYS = ['family', 'combo', 'surface', 'phase_label', 'sigma', 'rep', 'method']
SIZE_PAIRS = dict(point=[('D_r1.5_p8x4', 'D_r1_p8x4', 'size@p8x4'), ('D_r1.5_p8x6', 'D_r1_p8x6', 'size@p8x6')], line=[('L_w1.5_s6.3', 'L_w0.5_s6.3', 'size@s6.3'), ('L_w1.5_s8', 'L_w0.5_s8', 'size@s8')])
SPACING_PAIRS = dict(point=[('D_r1_p8x6', 'D_r1_p8x4', 'spacing@r1'), ('D_r1.5_p8x6', 'D_r1.5_p8x4', 'spacing@r1.5')], line=[('L_w0.5_s8', 'L_w0.5_s6.3', 'spacing@w0.5'), ('L_w1.5_s8', 'L_w1.5_s6.3', 'spacing@w1.5')])
RULES = dict(contrasts=('E1 env_vs_nominal', 'E2 grad_vs_uniform', 'E3 config_effect', 'E4 config_effect_change', 'E5 frontend_bridge'), metrics=EFFECT_METRICS,
             size_pairs=SIZE_PAIRS, spacing_pairs=SPACING_PAIRS, summary='per phase values; mean / min / max over phases; n defined; no bootstrap / tests',
             main_dot_front_end='DF', selfcheck='8 new image-method results: first 8 by sha256(case_id|method) among new results (sorted), lightweight recomputation from the stored arrays')


def undef(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def ratio(a, b):
    return (float(a) / float(b)) if b else None


def _row_common(m: dict, sup: dict | None, c: int, w: int, u: int) -> dict:
    out = {k: m.get(k) for k in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_cov_interp', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'glob_surf_rmse', 'n_valid',
                                 'glob_n_valid', 'loc_n_valid', 'n_accepted', 'id_correct', 'id_wrong', 'id_unconf', 'n_candidates', 'n_selected', 'n_matched', 'n_ambiguous',
                                 'n_detected', 'n_confirmable_features', 'img_n_expected', 'img_n_recovered', 'glob_recovery')}
    out['sup_surf_rmse'] = (sup or {}).get('sup_surf_rmse')
    out.update(c=c, w=w, u=u, w_share=ratio(w, c + w), u_share=ratio(u, c + w + u))
    return out


def new_rows(P: K.Paths, N: pd.DataFrame, ledger_counts: dict | None = None) -> list:
    import mp3_chain as CH
    rows = []
    for r in N.to_dict('records'):
        meta, A, st = CH.pack_read(P, r['case_id'])
        im = meta.get('image') or {}; dg = im.get('diag') or {}
        for m in r['methods'].split(';'):
            res = (meta.get('results') or {}).get(m) or {}
            mt = res.get('metrics') or {}
            fi = (res.get('full_identity') or {}).get('counts', {}).get('all') if r['family'] == 'line' else None
            c, w, u = ((fi['c'], fi['w'], fi['u']) if fi else (mt.get('id_correct'), mt.get('id_wrong'), mt.get('id_unconf')))
            row = dict(row_key=f"{r['case_id']}|{m}", image_key=r['case_id'], family=r['family'], combo=r['combo'], surface=r['surface'], phase_label=r['phase_label'],
                       phase_u=r['phase_u'], phase_v=r['phase_v'], phase_frac=r['phase_frac'], sigma=r['sigma'], rep=r['rep'], seed=r['seed'], env=r['env'], psf_true_px=r['psf_true_px'],
                       psf_nominal_px=r['psf_nominal_px'], field_kind=r['field_kind'], integral=r['integral'], peak=r['peak'], limit_type=r['limit_type'], method=m, source='new',
                       source_record=K.rel(CH.pack_path(P, r['case_id'])), parent_nominal_key=r['parent_nominal_key'], parent_nominal_image_key=r['parent_nominal_image_key'],
                       forward_key=im.get('forward_key'), image_content_sha=im.get('content_sha'), decode_id=res.get('decode_id'), eval_id=res.get('eval_id'), state=res.get('state'),
                       stage=r['stage'], pack_state=st)
            row.update(_row_common(mt, res.get('support'), c, w, u) if mt else {k: None for k in METRICS})
            row.update(n_unconf_evaluator=mt.get('id_unconf'), identity_rule=('complete (A4)' if r['family'] == 'line' else 'evaluator'))
            for k in ('received_light_ratio', 'gain_hit_weighted_mean', 'gain_min', 'gain_max', 'gain_mean_at_hits', 'irr_pre_psf_sum', 'irr_pre_psf_max', 'irr_post_psf_max', 'post_sum',
                      'post_p99_lit', 'signal_peak_noise_free_grey', 'frac_pre_clip_above_1', 'image_mean_grey', 'image_frac_0', 'image_frac_255', 'n_px_post_gt_10grey', 'n_px_post_gt_50grey',
                      'noise_sha', 'n_hits', 'n_in_fov'):
                row[f'diag_{k}'] = dg.get(k)
            nom = res.get('nominal') or {}
            row.update(dec_otsu=nom.get('otsu_threshold'), dec_a_floor_px2=nom.get('a_floor_px2'), dec_steger_thr_design=nom.get('steger_threshold_design'), dec_sigma_hat=nom.get('noise_sigma_hat_image'))
            if ledger_counts is not None:
                row.update(calls_render=ledger_counts.get(('render', r['case_id']), 0), calls_decode=ledger_counts.get(('decode', f"{r['case_id']}|{m}"), 0),
                           calls_eval=ledger_counts.get(('eval', f"{r['case_id']}|{m}"), 0))
            rows.append(row)
    return rows


def nominal_rows(O: pd.DataFrame, S: pd.DataFrame) -> list:

    import mp3_chain as CH
    import tr5_chain as T5
    C = K.read_csv(D.S5_CASES); rows = []
    Ox = O.set_index('nominal_key')
    for s in S.to_dict('records'):
        o = Ox.loc[s['nominal_key']]
        c = C[(C.manifest_key == s['nominal_key']) & (C.front_end == s['method'])].iloc[0].to_dict()
        mt = {k: (None if undef(v) else v) for k, v in c.items()}
        if o['family'] == 'line':
            ev = D.eval_arrays_of(s)
            q = CH.realq(); inp = q.inputs(dict(cand_id_sigma0=o['cand_id'], family=o['family']))
            farr, finfo = T5.full_identity(dict(family='line', surface=o['surface'], forward_key=s['nominal_key']), dict(meta_full=inp['meta_full'], binary=inp['binary']), ev)
            cc = finfo['counts']['all']; cw = (cc['c'], cc['w'], cc['u'])
        else:
            cw = (mt.get('id_correct'), mt.get('id_wrong'), mt.get('id_unconf'))
        row = dict(row_key=f"{s['nominal_key']}|{s['method']}", image_key=s['image_key'], family=o['family'], combo=o['combo'], surface=o['surface'], phase_label=o['phase_label'],
                   sigma=float(o['sigma']), rep=int(o['rep']), seed=o['seed'], env='NOMINAL', psf_true_px=1.0, psf_nominal_px=1.0, field_kind='const(1)', integral=o['integral'],
                   peak=o['peak'], method=s['method'], source='reuse', source_record=s['eval_record'], parent_nominal_key=s['nominal_key'], parent_nominal_image_key=s['image_key'],
                   image_identity_s5=s['image_identity'], decode_id=s['decode_key'], eval_id=s['eval_identity'], state=mt.get('state'), stage='nominal')
        row.update(_row_common(mt, dict(sup_surf_rmse=mt.get('sup_surf_rmse')), int(cw[0]), int(cw[1]), int(cw[2])))
        row.update(n_unconf_evaluator=mt.get('id_unconf'), identity_rule=('complete (A4)' if o['family'] == 'line' else 'evaluator'))
        rows.append(row)
    return rows


def results_table(P: K.Paths, N: pd.DataFrame, O: pd.DataFrame, S: pd.DataFrame, ledger_counts=None) -> pd.DataFrame:
    T = pd.DataFrame(new_rows(P, N, ledger_counts) + nominal_rows(O, S))
    T['seed'] = pd.Series([None if undef(x) else int(x) for x in T['seed']], dtype=object)
    return T


def completeness(T: pd.DataFrame, n_new: int, n_reuse: int) -> dict:
    probs = []
    miss_cols = [c for c in REQ_COLUMNS if c not in T.columns]
    if miss_cols:
        probs.append(f'missing columns {miss_cols}')
    if T.row_key.duplicated().any():
        probs.append(f'duplicate row keys {int(T.row_key.duplicated().sum())}')
    nn, nr = int((T.source == 'new').sum()), int((T.source == 'reuse').sum())
    if nn != n_new or nr != n_reuse:
        probs.append(f'rows new {nn} / reuse {nr}, expected {n_new} / {n_reuse}')
    if not miss_cols:
        bad_state = T[~T.state.isin(['ok', 'no_output'])]
        if len(bad_state):
            probs.append(f'{len(bad_state)} rows without a legal state (incomplete / missing result): {bad_state.row_key.tolist()[:5]}')
        no = T[T.state == 'no_output']
        bad_no = no[(no.glob_Ctau.astype(float) != 0.0) | ~no.glob_pt_rmse.map(undef)]
        if len(bad_no):
            probs.append(f'no-output rows with Ctau != 0 or a defined point error: {bad_no.row_key.tolist()[:5]}')
        ok = T[T.state == 'ok']
        if ok.glob_Ctau.map(undef).any():
            probs.append('ok rows with undefined Ctau')
        pc = T[T.source == 'new']
        if 'pack_state' in pc and (pc.pack_state != 'read').any():
            probs.append(f"{int((pc.pack_state != 'read').sum())} new rows without a readable pack")
    return dict(ok=not probs, problems=probs, n_new=nn, n_reuse=nr, n_no_output=int((T.state == 'no_output').sum()) if 'state' in T else None)


def pair(L: pd.DataFrame, R: pd.DataFrame, keys: list, what: str) -> pd.DataFrame:
    for nm, X in (('left', L), ('right', R)):
        if X.duplicated(keys).any():
            raise K.Stop('technical', f'{what}: duplicate pairing keys on the {nm} side ({int(X.duplicated(keys).sum())})')
    M = L.merge(R, on=keys, how='outer', suffixes=('_l', '_r'), indicator=True, validate='one_to_one')
    if (M._merge != 'both').any():
        raise K.Stop('technical', f"{what}: {int((M._merge != 'both').sum())} unmatched rows (missing group or wrong key)")
    return M.drop(columns='_merge')


def _diff_rows(M: pd.DataFrame, contrast: str, label_cols: dict) -> list:
    out = []
    for r in M.to_dict('records'):
        for mt in EFFECT_METRICS:
            a, b = r.get(f'{mt}_l'), r.get(f'{mt}_r')
            dfd = not (undef(a) or undef(b))
            out.append(dict(record_type='pair', contrast=contrast, **{k: r.get(v) for k, v in label_cols.items()}, metric=mt, left_key=r.get('row_key_l'), right_key=r.get('row_key_r'),
                            left_value=(None if undef(a) else float(a)), right_value=(None if undef(b) else float(b)), value=(float(a) - float(b) if dfd else None), defined=dfd,
                            state_left=r.get('state_l'), state_right=r.get('state_r')))
    return out


def _sel(T, **kw):
    q = T
    for k, v in kw.items():
        q = q[q[k] == v] if not isinstance(v, (list, tuple, set)) else q[q[k].isin(list(v))]
    return q


COLS = ['row_key', 'state', 'seed'] + list(EFFECT_METRICS)


def same_seeds(M: pd.DataFrame) -> bool:

    norm = lambda x: None if undef(x) else int(x)
    return all(norm(a) == norm(b) for a, b in zip(M.seed_l.tolist(), M.seed_r.tolist()))


def effects(T: pd.DataFrame, combos=None) -> pd.DataFrame:

    combos = tuple(D.COMBOS if combos is None else combos)
    rows = []
    base_keys = ['family', 'combo', 'surface', 'phase_label', 'sigma', 'rep', 'method']
    nom = _sel(T, env='NOMINAL')
    for env in D.ENVS:
        e = _sel(T, env=env)
        M = pair(e[base_keys + COLS], nom[base_keys + COLS], base_keys, f'E1 {env} vs nominal')
        if not same_seeds(M):
            raise K.Stop('technical', f'E1 {env}: environment and nominal rows with different noise seeds')
        rows += _diff_rows(M.assign(env=env), 'E1 env_vs_nominal', dict(family='family', config='combo', surface='surface', phase_label='phase_label', sigma='sigma', env='env', method='method'))
    uni = _sel(T, env='UNIFORM')
    for env in ('GRAD_X', 'GRAD_Y'):
        M = pair(_sel(T, env=env)[base_keys + COLS], uni[base_keys + COLS], base_keys, f'E2 {env} vs uniform')
        if not same_seeds(M):
            raise K.Stop('technical', f'E2 {env}: gradient and uniform rows with different noise seeds')
        rows += _diff_rows(M.assign(env=env), 'E2 grad_vs_uniform', dict(family='family', config='combo', surface='surface', phase_label='phase_label', sigma='sigma', env='env', method='method'))
    ck = ['family', 'surface', 'phase_label', 'sigma', 'rep', 'method', 'env']
    cfg_eff = []
    for fam in ('point', 'line'):
        for (a, b, lab) in SIZE_PAIRS[fam] + SPACING_PAIRS[fam]:
            if a not in combos or b not in combos:
                continue
            L = _sel(T, combo=a); R = _sel(T, combo=b)
            M = pair(L[ck + COLS], R[ck + COLS], ck, f'E3 {lab}')
            if not same_seeds(M):
                raise K.Stop('technical', f'E3 {lab}: paired configurations with different noise seeds')
            rr = _diff_rows(M.assign(config=f'{a} - {b}', pair_label=lab), 'E3 config_effect', dict(family='family', config='config', pair_label='pair_label', surface='surface',
                                                                                                    phase_label='phase_label', sigma='sigma', env='env', method='method'))
            rows += rr; cfg_eff += rr
    E3 = pd.DataFrame(cfg_eff)
    if len(E3):
        k4 = ['family', 'config', 'pair_label', 'surface', 'phase_label', 'sigma', 'method', 'metric']
        nomE = E3[E3.env == 'NOMINAL']
        for env in D.ENVS:
            M = pair(E3[E3.env == env][k4 + ['value', 'defined']], nomE[k4 + ['value', 'defined']], k4, f'E4 {env}')
            for r in M.to_dict('records'):
                dfd = bool(r['defined_l'] and r['defined_r'])
                rows.append(dict(record_type='pair', contrast='E4 config_effect_change', **{k: r[k] for k in k4 if k != 'metric'}, env=env, metric=r['metric'],
                                 left_value=r['value_l'], right_value=r['value_r'], value=(r['value_l'] - r['value_r'] if dfd else None), defined=dfd))
    fk = ['family', 'combo', 'surface', 'phase_label', 'sigma', 'rep', 'env', 'image_key']
    pts = T[T.family == 'point']
    M = pair(pts[pts.method == 'DF'][fk + COLS], pts[pts.method == 'D0'][fk + COLS], fk, 'E5 DF vs D0 same image')
    rows += _diff_rows(M.assign(method='DF-D0'), 'E5 frontend_bridge', dict(family='family', config='combo', surface='surface', phase_label='phase_label', sigma='sigma', env='env', method='method'))
    E = pd.DataFrame(rows)
    return pd.concat([E, summaries(E)], ignore_index=True)


def summaries(E: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in ('contrast', 'family', 'config', 'pair_label', 'surface', 'sigma', 'env', 'method', 'metric') if k in E.columns]
    out = []
    for kv, g in E.groupby(keys, dropna=False):
        d = dict(zip(keys, kv)); vals = {r['phase_label']: r['value'] for r in g.to_dict('records')}
        v = [x for x in g.value.tolist() if not undef(x)]
        out.append(dict(record_type='summary', **d, n_pairs=int(len(g)), n_defined=len(v), mean=(float(np.mean(v)) if v else None), min=(float(min(v)) if v else None),
                        max=(float(max(v)) if v else None), ph0=vals.get('ph0'), ph1=vals.get('ph1')))
    return pd.DataFrame(out)


def selfcheck_pick(T: pd.DataFrame, n: int = 8) -> list:
    import hashlib
    keys = sorted(T[T.source == 'new'].row_key.tolist(), key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    return keys[:n]


def selfcheck(P: K.Paths, T: pd.DataFrame, picks: list) -> list:
    import mp3_chain as CH
    out = []
    for rk in picks:
        cid, m = rk.rsplit('|', 1)
        meta, A, _ = CH.pack_read(P, cid)
        ev = {k[len(f'{m}__ev__'):]: v for k, v in A.items() if k.startswith(f'{m}__ev__')}
        pts = np.asarray(ev['pts'], np.float64); zr = np.asarray(ev['zr'], np.float64); val = np.asarray(ev['valid'], bool)
        ig = np.asarray(ev['in_global'], bool); il = np.asarray(ev['in_local'], bool); acc = np.asarray(ev['accepted'], bool)
        e = pts[:, 2] - zr if len(pts) else np.zeros(0)
        rm = lambda s: (math.sqrt(float(np.mean(e[s] ** 2))) if s.any() else None)
        if 'full_point_correct' in ev:
            c, w, u = (int(np.asarray(ev[k], bool).sum()) for k in ('full_point_correct', 'full_point_wrong', 'full_point_unconf'))
        else:
            c, w, u = (int(np.asarray(ev[k], bool).sum()) for k in ('point_correct', 'point_wrong', 'point_unconf'))
        row = T[T.row_key == rk].iloc[0]
        zmax = float(np.nanmax(np.abs(zr))) if len(zr) else 0.0
        tol = 2.0 * float(np.spacing(np.float32(max(zmax, 1.0))))
        got = dict(n_valid=int(val.sum()), glob_n_valid=int(ig.sum()), loc_n_valid=int(il.sum()), glob_pt_rmse=rm(ig), loc_pt_rmse=rm(il), c=c, w=w, u=u, n_accepted=int(acc.sum()))
        diffs = {}
        for k, v in got.items():
            ref = row.get(k) if k != 'n_accepted' else (meta['results'][m]['metrics'] or {}).get('n_accepted')
            if undef(ref) and v is None:
                continue
            if undef(ref) != (v is None):
                diffs[k] = (ref, v); continue
            if k.endswith('rmse'):
                if abs(float(ref) - float(v)) > tol:
                    diffs[k] = (ref, v)
            elif int(ref) != int(v):
                diffs[k] = (ref, v)
        den_ok = (c + w + u) == int(acc.sum())
        out.append(dict(row_key=rk, ok=(not diffs and den_ok), diffs=diffs, denominator_ok=den_ok, tol_mm=tol,
                        max_rmse_diff=max([abs(float(row.get(k)) - float(got[k])) for k in ('glob_pt_rmse', 'loc_pt_rmse') if got[k] is not None and not undef(row.get(k))], default=0.0)))
    return out
