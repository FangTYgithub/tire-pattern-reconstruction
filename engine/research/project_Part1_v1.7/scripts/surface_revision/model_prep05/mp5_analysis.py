# Analyze nominal and disturbed images and summarize paired environment effects.

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

import mp5_common as K
import mp3_analysis as A3

EFFECT_METRICS = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'sup_surf_rmse', 'n_valid', 'n_accepted', 'c', 'w', 'u', 'w_share', 'u_share')
METRIC_COLS = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'glob_surf_rmse', 'sup_surf_rmse', 'n_valid', 'glob_n_valid',
               'loc_n_valid', 'n_accepted', 'c', 'w', 'u', 'w_share', 'u_share')
REQ_COLUMNS = ('row_key', 'case_key', 'subset', 'family', 'config', 'surface', 'phase_block', 'sigma', 'rep', 'seed', 'env', 'method', 'parent_case_key', 'state', 'pack_state') + METRIC_COLS
CORE_ENVS = ('NOMINAL', 'BLUR2', 'BLUR3', 'UNIFORM', 'GRAD_X', 'GRAD_Y')
PSF_ENVS = (('NOMINAL', 1.0), ('BLUR1_5', 1.5), ('BLUR2', 2.0), ('BLUR3', 3.0))
SIZE_PAIRS, SPACING_PAIRS = A3.SIZE_PAIRS, A3.SPACING_PAIRS
BLOCKS, REPS = (0, 1, 2, 3), (1, 2)
PILOT = dict(F1='E1 env_vs_nominal', F2='E3 config_effect', F3='E4 config_effect_change', F4='E2 grad_vs_uniform', F5='E5 frontend_bridge', F6='E3 config_effect')
VERDICTS = ('reproduziert', 'begrenzte Evidenz', 'nicht reproduziert', 'Pilot ohne Richtung', 'kein Pilotvergleich', 'nicht bestimmbar')
BOARD_CASES = tuple(f'{c}|M10_curv|b0|s40|r1|{e}' for c in ('D_r1_p8x4', 'L_w1.5_s8') for e in ('NOMINAL', 'BLUR3', 'UNIFORM', 'GRAD_Y'))
RULES = dict(contrasts=('F1 env_vs_parent', 'F2 config_effect', 'F3 config_effect_change', 'F4 grad_vs_uniform', 'F5 frontend_bridge', 'F6 width_psf_8mm'), metrics=EFFECT_METRICS,
             size_pairs=SIZE_PAIRS, spacing_pairs=SPACING_PAIRS, blocks=BLOCKS, reps=REPS,
             aggregation='pair -> mean of reps {1,2} per block -> 4 blocks: mean, values, min/max, signs, leave-one-block-out; sigma 0 single example; no tests, no near-zero label',
             reproduction='glob_Ctau only, pilot counterpart = MODEL-PREP-03 effects summary; reproduziert = mean sign = pilot sign AND >= 3/4 blocks AND all 4 LOO means; '
                          'begrenzte Evidenz = mean sign only; nicht reproduziert = opposite sign or 0; descriptive, not a test',
             risk='per environment image vs own parent: w increase, activation, true lines, segments, points on lit-but-unregistered lines',
             board_cases=BOARD_CASES, failure_case='line image (P4) with the largest w increase over its own parent; ties: smallest sha256(case_key); none -> no failure board',
             selfcheck='per (method x surface) the first row by sha256(row_key) + the first legal no-output row + the first row with w > 0 (if any): recomputed from stored arrays')


def undef(x) -> bool:
    return A3.undef(x)


def results_table(P, M: pd.DataFrame) -> pd.DataFrame:
    import mp3_chain as CH3
    rows = []
    for r in M.to_dict('records'):
        meta, A, st = CH3.pack_read(P, r['case_key'])
        im = meta.get('image') or {}; dg = im.get('diag') or {}
        extra = json.loads(r['lit_unregistered_lines']) if isinstance(r.get('lit_unregistered_lines'), str) and r['lit_unregistered_lines'].strip() else []
        for m in str(r['methods']).split(';'):
            res = (meta.get('results') or {}).get(m) or {}; mt = res.get('metrics') or {}
            fi = (res.get('full_identity') or {}).get('counts', {}).get('all') if r['family'] == 'line' else None
            c, w, u = ((fi['c'], fi['w'], fi['u']) if fi else (mt.get('id_correct'), mt.get('id_wrong'), mt.get('id_unconf')))
            row = dict(row_key=f"{r['case_key']}|{m}", case_key=r['case_key'], subset=r['subset'], family=r['family'], config=r['config'], surface=r['surface'], phase_block=int(r['phase_block']),
                       phase_u=r.get('phase_u'), phase_v=r.get('phase_v'), phase_frac=r.get('phase_frac'), sigma=float(r['sigma']), rep=int(r['rep']),
                       seed=(None if undef(r.get('seed')) else int(r['seed'])), noise_block=r.get('noise_block'), env=r['env'], psf_true_px=float(r['psf_true_px']), field_kind=r['field_kind'],
                       method=m, parent_case_key=(None if undef(r.get('parent_case_key')) else r['parent_case_key']), forward_key=r['forward_key'], image_sha=im.get('content_sha'),
                       execution_fp=im.get('execution_fp'), input_key=im.get('input_key'), decode_id=res.get('decode_id'), eval_id=res.get('eval_id'), state=res.get('state'), pack_state=st,
                       lit_unregistered_lines=json.dumps(extra))
            if mt:
                row.update(A3._row_common(mt, res.get('support'), c, w, u))
                row['glob_pt_p95'] = mt.get('glob_pt_p95')
            else:
                row.update({k: None for k in METRIC_COLS})
            row.update(n_unconf_evaluator=mt.get('id_unconf'), identity_rule=('complete (A4)' if r['family'] == 'line' else 'evaluator'))
            ev = {k[len(f'{m}__ev__'):]: v for k, v in A.items() if k.startswith(f'{m}__ev__')}
            if r['family'] == 'line' and 'full_point_wrong' in ev:
                wm = np.asarray(ev['full_point_wrong'], bool) & np.asarray(ev['accepted'], bool); tk = np.asarray(ev['full_truth_k'])[wm]
                row.update(w_true_lines=int(len(np.unique(tk))), w_segments=int(len(np.unique(np.asarray(ev['segment_id'])[wm]))), w_on_unregistered_line=int(np.isin(tk, extra).sum()))
            else:
                row.update(w_true_lines=None, w_segments=None, w_on_unregistered_line=None)
            for k in ('received_light_ratio', 'gain_hit_weighted_mean', 'gain_min', 'gain_max', 'signal_peak_noise_free_grey', 'image_mean_grey', 'image_frac_255', 'frac_pre_clip_above_1',
                      'noise_sha', 'psf_true_px'):
                row[f'diag_{k}'] = dg.get(k)
            nom = res.get('nominal') or {}
            row.update(dec_otsu=nom.get('otsu_threshold'), dec_a_floor_px2=nom.get('a_floor_px2'), dec_steger_thr_design=nom.get('steger_threshold_design'), dec_sigma_hat=nom.get('noise_sigma_hat_image'))
            rows.append(row)
    T = pd.DataFrame(rows)
    T['seed'] = pd.Series([None if undef(x) else int(x) for x in T['seed']], dtype=object)
    return T


def completeness(T: pd.DataFrame, n_rows: int, n_images: int) -> dict:
    probs = []
    miss = [c for c in REQ_COLUMNS if c not in T.columns]
    if miss:
        return dict(ok=False, problems=[f'missing columns {miss}'], n_rows=len(T))
    if T.row_key.duplicated().any():
        probs.append(f'duplicate row keys {int(T.row_key.duplicated().sum())}')
    if len(T) != n_rows or T.case_key.nunique() != n_images:
        probs.append(f'rows {len(T)} / images {T.case_key.nunique()}, expected {n_rows} / {n_images}')
    bad = T[~T.state.isin(['ok', 'no_output'])]
    if len(bad):
        probs.append(f'{len(bad)} rows without a legal state: {bad.row_key.tolist()[:4]}')
    no = T[T.state == 'no_output']
    if len(no[(no.glob_Ctau.astype(float) != 0.0) | ~no.glob_pt_rmse.map(undef)]):
        probs.append('no-output rows with Ctau != 0 or a defined point error')
    if T[T.state == 'ok'].glob_Ctau.map(undef).any():
        probs.append('ok rows with undefined Ctau')
    for n_col, e_col in (('glob_n_valid', 'glob_pt_rmse'), ('loc_n_valid', 'loc_pt_rmse')):
        mism = T[(T[n_col].fillna(0).astype(float) > 0) == T[e_col].map(undef)]
        if len(mism):
            probs.append(f'{len(mism)} rows: {n_col} and {e_col} inconsistent (valid points without error or error without valid points): {mism.row_key.tolist()[:3]}')
    if (T.pack_state != 'read').any():
        probs.append(f"{int((T.pack_state != 'read').sum())} rows without a readable pack")
    return dict(ok=not probs, problems=probs, n_rows=len(T), n_images=int(T.case_key.nunique()), n_no_output=int(len(no)))


COLS = ['row_key', 'case_key', 'state', 'seed', 'parent_case_key'] + list(EFFECT_METRICS)


def _diff(M: pd.DataFrame, contrast: str, labels: dict) -> list:
    out = []
    for r in M.to_dict('records'):
        for mt in EFFECT_METRICS:
            a, b = r.get(f'{mt}_l'), r.get(f'{mt}_r')
            dfd = not (undef(a) or undef(b))
            out.append(dict(record_type='pair', contrast=contrast, **{k: (r.get(v) if isinstance(v, str) and (v in r) else v) for k, v in labels.items()}, metric=mt,
                            left_key=r.get('row_key_l'), right_key=r.get('row_key_r'), left_value=(None if undef(a) else float(a)), right_value=(None if undef(b) else float(b)),
                            value=(float(a) - float(b) if dfd else None), defined=dfd, state_left=r.get('state_l'), state_right=r.get('state_r')))
    return out


def _seeds_equal(M) -> bool:
    return A3.same_seeds(M)


BASE = ['surface', 'phase_block', 'sigma', 'rep', 'method']


def _check_cols(T):
    for c in REQ_COLUMNS:
        if c not in T.columns:
            raise K.Stop('technical', f'contrasts: column {c} missing')


def contrast_f1(T: pd.DataFrame) -> list:
    _check_cols(T)
    rows = []; base = BASE
    R0 = T[T.env == 'NOMINAL']
    for env in sorted(set(T.env) - {'NOMINAL'}):
        L = T[T.env == env]; R = R0[R0.config.isin(set(L.config))]
        M = A3.pair(L[['config'] + base + COLS], R[['config'] + base + COLS], ['config'] + base, f'F1 {env}')
        if not (M.parent_case_key_l == M.case_key_r).all() or not _seeds_equal(M):
            raise K.Stop('technical', f'F1 {env}: a pair is not bound to its own parent (parent key / seed)')
        rows += _diff(M, 'F1 env_vs_parent', dict(family='family_x', config='config', surface='surface', phase_block='phase_block', sigma='sigma', rep='rep', env=env, method='method'))
    return rows


def contrasts(T: pd.DataFrame) -> pd.DataFrame:
    _check_cols(T)
    base = BASE
    rows = contrast_f1(T)
    F2 = []
    for fam in ('point', 'line'):
        for (a, b, lab) in SIZE_PAIRS[fam] + SPACING_PAIRS[fam]:
            for env in CORE_ENVS:
                L = T[(T.config == a) & (T.env == env)]; R = T[(T.config == b) & (T.env == env)]
                M = A3.pair(L[base + COLS], R[base + COLS], base, f'F2 {lab} {env}')
                if not _seeds_equal(M):
                    raise K.Stop('technical', f'F2 {lab} {env}: different noise seeds')
                rr = _diff(M, 'F2 config_effect', dict(family=fam, config=f'{a} - {b}', pair_label=lab, surface='surface', phase_block='phase_block', sigma='sigma', rep='rep', env=env, method='method'))
                rows += rr; F2 += rr
    E2 = pd.DataFrame(F2)
    k4 = ['family', 'config', 'pair_label', 'surface', 'phase_block', 'sigma', 'rep', 'method', 'metric']
    nomE = E2[E2.env == 'NOMINAL']
    for env in CORE_ENVS[1:]:
        M = A3.pair(E2[E2.env == env][k4 + ['value', 'defined', 'left_key']], nomE[k4 + ['value', 'defined', 'left_key']], k4, f'F3 {env}')
        for r in M.to_dict('records'):
            dfd = bool(r['defined_l'] and r['defined_r'])
            rows.append(dict(record_type='pair', contrast='F3 config_effect_change', **{k: r[k] for k in k4 if k != 'metric'}, env=env, metric=r['metric'], left_key=r['left_key_l'],
                             right_key=r['left_key_r'], left_value=r['value_l'], right_value=r['value_r'], value=(r['value_l'] - r['value_r'] if dfd else None), defined=dfd))
    U = T[T.env == 'UNIFORM']
    for env in ('GRAD_X', 'GRAD_Y'):
        M = A3.pair(T[T.env == env][['config'] + base + COLS], U[['config'] + base + COLS], ['config'] + base, f'F4 {env}')
        if not _seeds_equal(M):
            raise K.Stop('technical', f'F4 {env}: different noise seeds')
        rows += _diff(M, 'F4 grad_vs_uniform', dict(family='family_x', config='config', surface='surface', phase_block='phase_block', sigma='sigma', rep='rep', env=env, method='method'))
    pts = T[T.family == 'point']
    M = A3.pair(pts[pts.method == 'DF'][['case_key', 'config', 'surface', 'phase_block', 'sigma', 'rep', 'env'] + [c for c in COLS if c != 'case_key']],
                pts[pts.method == 'D0'][['case_key'] + [c for c in COLS if c != 'case_key']], ['case_key'], 'F5 DF-D0')
    rows += _diff(M, 'F5 frontend_bridge', dict(family='point', config='config', surface='surface', phase_block='phase_block', sigma='sigma', rep='rep', env='env', method='DF-D0'))
    for env, _ in PSF_ENVS:
        L = T[(T.config == 'L_w1.5_s8') & (T.env == env)]; R = T[(T.config == 'L_w0.5_s8') & (T.env == env)]
        M = A3.pair(L[base + COLS], R[base + COLS], base, f'F6 {env}')
        if not _seeds_equal(M):
            raise K.Stop('technical', f'F6 {env}: different noise seeds')
        rows += _diff(M, 'F6 width_psf_8mm', dict(family='line', config='L_w1.5_s8 - L_w0.5_s8', pair_label='size@s8', surface='surface', phase_block='phase_block', sigma='sigma', rep='rep',
                                                  env=env, method='method'))
    return frame(rows, T)


def frame(rows: list, T: pd.DataFrame) -> pd.DataFrame:
    E = pd.DataFrame(rows)
    if len(E):
        fam_of = dict(zip(T.config, T.family))
        E['family'] = [f if isinstance(f, str) and f in ('point', 'line') else fam_of.get(c) for f, c in zip(E.family, E.config)]
    return E


def _sgn(x) -> int:
    return 0 if x is None or x == 0 else (1 if x > 0 else -1)


def block_values(g: pd.DataFrame, allow_incomplete: bool = False) -> dict:

    out = {}
    for b, gb in g.groupby('phase_block'):
        reps = sorted(int(x) for x in gb.rep)
        if reps != list(REPS):
            if allow_incomplete and set(reps) <= set(REPS) and len(set(reps)) == len(reps):
                pass
            else:
                raise K.Stop('technical', f'block {b}: repetitions {reps} (exactly one pair per repetition 1 and 2 required)')
        v = [float(x) for x, d in zip(gb.value, gb.defined) if d and not undef(x)]
        out[int(b)] = ((sum(v) / len(v)) if v else None, len(v))
    if not allow_incomplete and sorted(out) != list(BLOCKS):
        raise K.Stop('technical', f'phase blocks {sorted(out)} (all four required)')
    return out


def verdict(mean, vals: list, loo: list, pilot) -> str:
    if pilot is None or undef(pilot):
        return 'kein Pilotvergleich'
    s = _sgn(float(pilot))
    if s == 0:
        return 'Pilot ohne Richtung'
    if mean is None:
        return 'nicht bestimmbar'
    if _sgn(mean) != s:
        return 'nicht reproduziert'
    if sum(_sgn(v) == s for v in vals if v is not None) >= 3 and all(_sgn(x) == s for x in loo if x is not None) and len([x for x in loo if x is not None]) == 4:
        return 'reproduziert'
    return 'begrenzte Evidenz'


def pilot_table() -> dict:
    E = K.read_csv(K.MP3_OUT / 'effects.csv')
    S = E[(E.record_type == 'summary') & (E.metric == 'glob_Ctau')]
    out = {}
    for r in S.to_dict('records'):
        lab = r['pair_label'] if isinstance(r.get('pair_label'), str) else r['config']
        out[(r['contrast'], lab, r['surface'], float(r['sigma']), r['env'], r['method'])] = r['mean']
    return out


def summaries(E: pd.DataFrame, pilot: dict | None = None, allow_incomplete: bool = False) -> pd.DataFrame:
    pilot = pilot_table() if pilot is None else pilot
    P = E[E.record_type == 'pair'].copy()
    P['label'] = [pl if isinstance(pl, str) else c for pl, c in zip(P.get('pair_label', pd.Series([None] * len(P), index=P.index)), P.config)]
    out = []
    keys = ['contrast', 'family', 'label', 'surface', 'sigma', 'env', 'method', 'metric']
    for k, g in P.groupby(keys, dropna=False):
        rec = dict(zip(keys, k)); rec['record_type'] = 'summary'; rec['n_pairs'] = len(g); rec['n_pairs_defined'] = int(g.defined.sum())
        if float(rec['sigma']) == 0.0:
            if len(g) != 1 or int(g.rep.iloc[0]) != 0:
                raise K.Stop('technical', f"sigma-0 group {k}: {len(g)} pairs / rep {sorted(set(g.rep))} (exactly one rep-0 pair)")
            v = g.value.iloc[0]
            rec.update(sigma0_single=True, mean=(None if undef(v) else float(v)), verdict='Einzelbeispiel (sigma 0)')
            out.append(rec); continue
        bv = block_values(g, allow_incomplete)
        vals = [bv.get(b, (None, 0))[0] for b in BLOCKS]
        dv = [v for v in vals if v is not None]
        mean = (sum(dv) / len(dv)) if dv else None
        loo = []
        for i in BLOCKS:
            o = [vals[j] for j in BLOCKS if j != i and vals[j] is not None]
            loo.append((sum(o) / len(o)) if o else None)
        rec.update(sigma0_single=False, mean=mean, min=(min(dv) if dv else None), max=(max(dv) if dv else None), n_blocks_defined=len(dv),
                   n_pos=sum(v > 0 for v in dv), n_neg=sum(v < 0 for v in dv), n_zero=sum(v == 0 for v in dv),
                   **{f'b{b}': vals[b] for b in BLOCKS}, **{f'loo{b}': loo[b] for b in BLOCKS}, **{f'b{b}_n_reps': bv.get(b, (None, 0))[1] for b in BLOCKS})
        if rec['metric'] == 'glob_Ctau':
            c = rec['contrast'].split()[0]
            pk = None
            if c in PILOT and not (c == 'F6' and rec['env'] == 'BLUR1_5'):
                pk = pilot.get((PILOT[c], rec['label'] if c not in ('F6',) else 'size@s8', rec['surface'], float(rec['sigma']), rec['env'], rec['method']))
            rec['pilot_mean'] = pk
            rec['verdict'] = verdict(mean, vals, loo, pk) if not allow_incomplete else 'Vorschau'
        out.append(rec)
    return pd.DataFrame(out)


def levels(T: pd.DataFrame, allow_incomplete: bool = False) -> pd.DataFrame:

    rows = []
    for mt in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'w', 'w_share', 'diag_received_light_ratio'):
        Q = T[['config', 'family', 'surface', 'sigma', 'env', 'method', 'phase_block', 'rep', mt]].rename(columns={mt: 'value'})
        Q['defined'] = ~Q.value.map(undef)
        for k, g in Q.groupby(['config', 'family', 'surface', 'sigma', 'env', 'method']):
            rec = dict(zip(['config', 'family', 'surface', 'sigma', 'env', 'method'], k), metric=mt, record_type='level')
            if float(rec['sigma']) == 0.0:
                rec.update(sigma0_single=True, mean=(None if undef(g.value.iloc[0]) else float(g.value.iloc[0])))
            else:
                bv = block_values(g, allow_incomplete); vals = [bv.get(b, (None, 0))[0] for b in BLOCKS]; dv = [v for v in vals if v is not None]
                rec.update(sigma0_single=False, mean=(sum(dv) / len(dv) if dv else None), min=(min(dv) if dv else None), max=(max(dv) if dv else None), n_blocks_defined=len(dv),
                           **{f'b{b}': vals[b] for b in BLOCKS})
            rows.append(rec)
    return pd.DataFrame(rows)


def risk(E: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    F = E[(E.record_type == 'pair') & (E.contrast == 'F1 env_vs_parent') & (E.metric == 'w')]
    Tx = T.set_index('row_key'); rows = []
    for r in F.to_dict('records'):
        if r['defined'] and float(r['value']) > 0:
            L = Tx.loc[r['left_key']]; R = Tx.loc[r['right_key']]
            rows.append(dict(record_type='risk', row_key=r['left_key'], parent_row_key=r['right_key'], family=L['family'], config=L['config'], surface=L['surface'], phase_block=int(L['phase_block']),
                             sigma=float(L['sigma']), rep=int(L['rep']), env=L['env'], method=L['method'], w=int(L['w']), w_parent=int(R['w']), w_increase=int(L['w']) - int(R['w']),
                             activation=bool(int(R['w']) == 0), c=int(L['c']), w_share=L['w_share'], w_true_lines=L['w_true_lines'], w_segments=L['w_segments'],
                             w_on_unregistered_line=L['w_on_unregistered_line'], lit_unregistered_lines=L['lit_unregistered_lines']))
    return pd.DataFrame(rows)


def select_failure(Rk: pd.DataFrame) -> str | None:
    if Rk is None or not len(Rk):
        return None
    L = Rk[Rk.method == 'P4']
    if not len(L):
        return None
    mx = int(L.w_increase.max()); c = L[L.w_increase == mx]
    keys = sorted(c.row_key.map(lambda k: k.rsplit('|', 1)[0]).tolist(), key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    return keys[0]


def selfcheck_pick(T: pd.DataFrame) -> list:
    h = lambda k: hashlib.sha256(k.encode()).hexdigest()
    picks = []
    for m in ('D0', 'DF', 'P4'):
        for s in ('B_curv', 'M10_curv', 'M24_curv'):
            q = T[(T.method == m) & (T.surface == s)]
            if len(q):
                picks.append(sorted(q.row_key, key=h)[0])
    no = T[T.state == 'no_output']
    if len(no):
        picks.append(sorted(no.row_key, key=h)[0])
    ww = T[T.w.fillna(0).astype(float) > 0]
    if len(ww):
        picks.append(sorted(ww.row_key, key=h)[0])
    return list(dict.fromkeys(picks))
