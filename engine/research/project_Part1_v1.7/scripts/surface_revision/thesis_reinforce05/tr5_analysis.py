# Evaluate new-phase results, paired effects and recommendation stability.

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import tr5_chain as CH
import tr5_common as K
import tr5_manifest as MF

BOOT_N, BOOT_SEED = 10_000, 20261019
SAME_SIGN_MIN = dict(point=6, line=3)
COVER = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_recovery', 'loc_recovery')
ERR = ('glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'loc_surf_rmse')
COUNT = ('n_valid', 'n_accepted', 'full_c', 'full_w', 'full_u', 'glob_n_valid', 'loc_n_valid')
RISK = ('w_share', 'u_share')
CORE = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'n_valid', 'n_accepted', 'full_c', 'full_w', 'full_u', 'state')
SUMMARY_METRICS = COVER + ERR + COUNT + RISK + ('no_output',)
DIRECTION = {m: 'higher' for m in COVER} | {m: 'lower' for m in ERR + RISK}
LABELS_ZH = dict(supported='Replicated', qualified='Conditional on surface, noise or detector', insufficient='Insufficient evidence', reversed='Direction reversed', exploratory='Exploratory comparison (no registered direction)')


def _undef(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _ratio(a, b):
    return (float(a) / float(b)) if b else None


def _missing_dir(e) -> bool:

    if e is None:
        return True
    try:
        return bool(pd.isna(e))
    except (TypeError, ValueError):
        return False


def results(rows: list, root, fps: dict) -> pd.DataFrame:
    comp = CH.completeness(rows, root, fps)
    if not comp['complete']:
        raise K.Stop('result_integrity', f"analysis refused: {comp['n_missing']} missing / invalid results, e.g. {comp['missing'][:4]}")
    out = []
    for row in rows:
        meta, A, _ = CH.pack_read(root, row['forward_key'])
        im = meta['image']
        for m in MF.METHODS[row['family']]:
            r = meta['results'][m]; mt = r['metrics']; fi = r.get('full_identity') or {}
            if row['family'] == 'line':
                fc, fw, fu = (fi['counts']['all'][x] for x in 'cwu'); gfc = fi['counts']['glob']; lfc = fi['counts']['loc']
            else:
                fc, fw, fu = int(mt['id_correct']), int(mt['id_wrong']), int(mt['id_unconf'])
                gfc = dict(c=mt.get('glob_id_correct'), w=mt.get('glob_id_wrong'), u=mt.get('glob_id_unconf'))
                lfc = dict(c=mt.get('loc_id_correct'), w=mt.get('loc_id_wrong'), u=mt.get('loc_id_unconf'))
            rec = {k: row[k] for k in ('forward_key', 'family', 'config', 'layout', 'size_mm', 'line_pitch_mm', 'surface', 'platform_surface', 'block', 'sigma', 'noise_k', 'seed',
                                       'raster_key', 'lit_px', 'peak', 'peak_f32', 'integral_actual', 'active_limit', 'rays_per_pixel')}
            rec.update(method=m, forward_id=im['forward_id'], image_sha=im['content_sha'], decode_id=r['decode_id'], eval_id=r['eval_id'], state=r['state'],
                       n_out=mt.get('n_out'), n_accepted=mt['n_accepted'], n_valid=mt['n_valid'], no_output=int(r['state'] == 'no_output'),
                       accepted_but_all_invalid=int(mt['n_accepted'] > 0 and mt['n_valid'] == 0), no_accepted_output=int(mt['n_accepted'] == 0),
                       n_detected=mt.get('n_selected', mt.get('n_detected')), n_ambiguous=mt.get('n_ambiguous'),
                       id_correct=mt['id_correct'], id_wrong=mt['id_wrong'], id_unconf=mt['id_unconf'], full_c=fc, full_w=fw, full_u=fu,
                       glob_full_c=gfc['c'], glob_full_w=gfc['w'], glob_full_u=gfc['u'], loc_full_c=lfc['c'], loc_full_w=lfc['w'], loc_full_u=lfc['u'],
                       w_share=_ratio(fw, fc + fw), u_share=_ratio(fu, fc + fw + fu), full_new_labels=fi.get('n_new_labels', 0) if fi.get('applied') else None,
                       extra_lines=(';'.join(map(str, fi.get('extra_lines', []))) if fi.get('applied') else None),
                       grey_sum=im.get('grey_sum'), frac_255=im.get('frac_255'), n_hits=(im.get('render_meta') or {}).get('n_hits'))
            for k in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_cov_interp', 'loc_cov_interp', 'glob_recovery', 'loc_recovery', 'glob_pt_rmse', 'glob_pt_p95',
                      'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'loc_surf_rmse', 'glob_n_valid', 'loc_n_valid', 'glob_gross_rate', 'img_n_expected', 'img_n_recovered'):
                rec[k] = mt.get(k)
            out.append(rec)
    R = pd.DataFrame(out)
    core_gate(R)
    return R


def core_gate(R: pd.DataFrame) -> None:
    miss = [c for c in CORE if c not in R.columns]
    if miss:
        raise K.Stop('result_integrity', f'core result columns missing: {miss}')
    bad = [c for c in CORE if R[c].isna().any()]
    if bad:
        raise K.Stop('result_integrity', f'core result values missing (not filled with 0): {bad}')
    for c in ERR:
        v = pd.to_numeric(R[c], errors='coerce'); d = R[c].notna()
        if (d & (~np.isfinite(v) | (v < 0))).any():
            raise K.Stop('result_integrity', f'{c}: defined values must be finite and >= 0')
    if ((R.state == 'no_output') != (R.n_valid == 0)).any():
        raise K.Stop('result_integrity', 'state / n_valid contradiction')
    if (R.full_c + R.full_w + R.full_u != R.n_accepted).any():
        raise K.Stop('result_integrity', 'c + w + u != accepted outputs')


PAIR_KEY = ('surface', 'block', 'sigma', 'noise_k', 'method')


def _diffs(a: dict, b: dict, prefix='d_') -> dict:
    out = {}
    for m in COVER:
        x, y = a.get(m), b.get(m)
        out[prefix + m] = None if (_undef(x) or _undef(y)) else (float(x) - float(y)) * 100.0
    for m in ERR:
        x, y = a.get(m), b.get(m)
        out[prefix + m] = None if (_undef(x) or _undef(y)) else float(x) - float(y)
    for m in COUNT + ('no_output',):
        out[prefix + m] = float(a[m]) - float(b[m])
    for m in RISK:
        x, y = a.get(m), b.get(m)
        out[prefix + m] = None if (_undef(x) or _undef(y)) else (float(x) - float(y)) * 100.0
    return out


def method_pairs(R: pd.DataFrame, contrast: str) -> pd.DataFrame:
    fam, left, right = MF.CONTRASTS[contrast]
    F = R[(R.family == fam) & (R.config.isin([left, right]))]
    if F.empty:
        raise K.Stop('pairing', f'{contrast}: empty core subset')
    if F[list(PAIR_KEY) + ['config', 'forward_key', 'seed']].isna().any().any():
        raise K.Stop('pairing', f'{contrast}: missing configuration / key labels')
    rows = []
    for key, g in F.groupby(list(PAIR_KEY), sort=True):
        L, Rr = g[g.config == left], g[g.config == right]
        if len(L) != 1 or len(Rr) != 1 or len(g) != 2:
            raise K.Stop('pairing', f'{contrast} {dict(zip(PAIR_KEY, key))}: {len(L)} left / {len(Rr)} right arms (one-to-one required)')
        a, b = L.iloc[0].to_dict(), Rr.iloc[0].to_dict()
        if int(a['seed']) != int(b['seed']):
            raise K.Stop('pairing', f'{contrast} {key}: seeds differ ({a["seed"]} / {b["seed"]})')
        if a['forward_key'] == b['forward_key']:
            raise K.Stop('pairing', f'{contrast} {key}: both arms have the same forward key')
        rows.append(dict(record_type='pair_method', contrast=contrast, family=fam, left=left, right=right, **dict(zip(PAIR_KEY, key)), seed=int(a['seed']),
                         forward_key_left=a['forward_key'], forward_key_right=b['forward_key'], state_left=a['state'], state_right=b['state'],
                         glob_Ctau_left=a['glob_Ctau'] * 100, glob_Ctau_right=b['glob_Ctau'] * 100, full_w_left=a['full_w'], full_w_right=b['full_w'],
                         full_u_left=a['full_u'], full_u_right=b['full_u'], **_diffs(a, b)))
    P = pd.DataFrame(rows)

    for (s, sg, m), g in P.groupby(['surface', 'sigma', 'method']):
        if g.groupby(['block', 'noise_k']).size().max() != 1:
            raise K.Stop('pairing', f'{contrast} {s} {sg} {m}: duplicated realisation pairs')
    return P


def frontend_pairs(R: pd.DataFrame) -> pd.DataFrame:
    D = R[R.family == 'point']; rows = []
    if D.empty:
        raise K.Stop('pairing', 'front-end pairs: empty dot subset')
    for fk, g in D.groupby('forward_key', sort=True):
        a, b = g[g.method == 'DF'], g[g.method == 'D0']
        if len(a) != 1 or len(b) != 1:
            raise K.Stop('pairing', f'{fk}: {len(a)} DF / {len(b)} D0 results (one each required)')
        a, b = a.iloc[0].to_dict(), b.iloc[0].to_dict()
        if a['image_sha'] != b['image_sha'] or a['eval_id'] != b['eval_id']:
            raise K.Stop('pairing', f'{fk}: front ends do not share the image content / evaluation version')
        rows.append(dict(record_type='pair_frontend', contrast='DF-D0', family='point', left='DF', right='D0', config=a['config'], surface=a['surface'], block=a['block'],
                         sigma=a['sigma'], noise_k=a['noise_k'], method='DF-D0', seed=int(a['seed']), forward_key_left=fk, forward_key_right=fk,
                         state_left=a['state'], state_right=b['state'], glob_Ctau_left=a['glob_Ctau'] * 100, glob_Ctau_right=b['glob_Ctau'] * 100,
                         full_w_left=a['full_w'], full_w_right=b['full_w'], full_u_left=a['full_u'], full_u_right=b['full_u'], **_diffs(a, b)))
    return pd.DataFrame(rows)


def blocks(P: pd.DataFrame, by: tuple) -> pd.DataFrame:
    rows = []
    for key, g in P.groupby(list(by) + ['block'], sort=True):
        rec = dict(zip(list(by) + ['block'], key)); rec['record_type'] = 'block'; rec['n_realisations'] = len(g)
        for m in SUMMARY_METRICS:
            v = pd.to_numeric(g['d_' + m], errors='coerce').dropna()
            rec['d_' + m] = float(v.mean()) if len(v) else None; rec['n_' + m] = int(len(v))
        rows.append(rec)
    return pd.DataFrame(rows)


def boot_index(family: str, n_blocks: int) -> np.ndarray:
    rng = np.random.default_rng(BOOT_SEED)
    return rng.integers(0, n_blocks, size=(BOOT_N, n_blocks))


def _interval(vals: np.ndarray, idx: np.ndarray) -> tuple:

    S = vals[idx]
    with np.errstate(invalid='ignore'):
        cnt = np.isfinite(S).sum(axis=1); means = np.where(cnt > 0, np.nansum(S, axis=1) / np.maximum(cnt, 1), np.nan)
    ok = np.isfinite(means)
    if not ok.any():
        return None, None, 1.0
    return float(np.percentile(means[ok], 2.5)), float(np.percentile(means[ok], 97.5)), float(1 - ok.mean())


def cells(B: pd.DataFrame, by: tuple, family_of, expected: dict | None = None) -> pd.DataFrame:
    rows = []; idx_cache = {}
    for key, g in B.groupby(list(by), sort=True):
        rec = dict(zip(by, key)); rec['record_type'] = 'cell'; fam = family_of(rec)
        nb_design = MF.N_BLOCKS[fam]; g = g.sort_values('block')
        if fam not in idx_cache:
            idx_cache[fam] = boot_index(fam, nb_design)
        rec['n_blocks'] = int(len(g)); rec['family'] = fam
        exp = None if expected is None else expected.get((rec.get('contrast'), rec.get('method'), float(rec['sigma']), rec['surface']))
        rec['expected_sign'] = exp
        for m in SUMMARY_METRICS:
            v = pd.to_numeric(g['d_' + m], errors='coerce'); d = v.dropna()
            rec['mean_' + m] = float(d.mean()) if len(d) else None
            rec['median_' + m] = float(d.median()) if len(d) else None
            rec['min_' + m] = float(d.min()) if len(d) else None
            rec['max_' + m] = float(d.max()) if len(d) else None
            rec['nblocks_defined_' + m] = int(len(d)); rec['npairs_defined_' + m] = int(g['n_' + m].sum())
            if m in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'loc_pt_rmse', 'glob_surf_rmse', 'w_share', 'u_share', 'full_w', 'full_u', 'no_output'):
                if len(g) == nb_design:
                    vals = np.full(nb_design, np.nan); vals[g.block.astype(int).values] = v.values.astype(float)
                    lo, hi, und = _interval(vals, idx_cache[fam])
                else:
                    lo, hi, und = None, None, None
                rec['lo_' + m], rec['hi_' + m], rec['boot_undefined_share_' + m] = lo, hi, und
        rec['blocks_glob_Ctau'] = ';'.join(f'{int(b)}:{x:.3f}' for b, x in zip(g.block, g['d_glob_Ctau']) if not _undef(x))
        if not _missing_dir(exp):
            s = np.sign(pd.to_numeric(g['d_glob_Ctau'], errors='coerce').values)
            rec['n_blocks_expected_sign'] = int((s == exp).sum()); rec['n_blocks_opposite_sign'] = int((s == -exp).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def _surface_label(c: dict, fam: str) -> str:
    e = c['expected_sign']; m, lo, hi = c['mean_glob_Ctau'], c['lo_glob_Ctau'], c['hi_glob_Ctau']
    if _missing_dir(e) or _undef(m):
        return 'exploratory'
    excl_exp = lo is not None and ((e > 0 and lo > 0) or (e < 0 and hi < 0))
    excl_opp = lo is not None and ((e > 0 and hi < 0) or (e < 0 and lo > 0))
    if excl_opp:
        return 'opposite'
    if np.sign(m) == e and excl_exp and c.get('n_blocks_expected_sign', 0) >= SAME_SIGN_MIN[fam]:
        return 'supports'
    if np.sign(m) == e:
        return 'same_direction_weak'
    return 'opposite_direction_weak'


def decisions(C: pd.DataFrame, registry: list) -> pd.DataFrame:
    rows = []
    for cl in registry:
        g = C[(C.contrast == cl['contrast']) & (C.method == cl['method']) & (C.sigma == cl['sigma'])]
        if g.empty:
            continue
        fam = g.family.iloc[0]; per = {}
        for c in g.to_dict('records'):
            per[c['surface']] = dict(label=_surface_label(c, fam), mean_pp=c['mean_glob_Ctau'], lo=c['lo_glob_Ctau'], hi=c['hi_glob_Ctau'],
                                     n_expected=c.get('n_blocks_expected_sign'), n_blocks=c['n_blocks'], d_w_share_pp=c['mean_w_share'], d_u_share_pp=c['mean_u_share'],
                                     d_full_w=c['mean_full_w'], d_loc_Ctau_pp=c['mean_loc_Ctau'], d_glob_pt_rmse_mm=c['mean_glob_pt_rmse'], d_no_output=c['mean_no_output'])
        exps = [c['expected_sign'] for c in g.to_dict('records')]
        if all(_missing_dir(e) for e in exps):
            label = 'exploratory'
        else:
            labs = [v['label'] for v in per.values()]
            means_ok = all((not _undef(c['mean_glob_Ctau'])) and not _missing_dir(c['expected_sign']) and np.sign(c['mean_glob_Ctau']) == c['expected_sign'] for c in g.to_dict('records'))
            nsame_ok = all((c.get('n_blocks_expected_sign') or 0) >= SAME_SIGN_MIN[fam] for c in g.to_dict('records'))
            n_excl_exp = sum(1 for c in g.to_dict('records') if c['lo_glob_Ctau'] is not None and not _missing_dir(c['expected_sign']) and
                             ((c['expected_sign'] > 0 and c['lo_glob_Ctau'] > 0) or (c['expected_sign'] < 0 and c['hi_glob_Ctau'] < 0)))
            n_opp = labs.count('opposite')
            if means_ok and nsame_ok and n_excl_exp >= 2 and len(g) == 3:
                label = 'supported'
            elif n_opp >= 2:
                label = 'reversed'
            elif n_excl_exp == 0:
                label = 'insufficient'
            else:
                label = 'qualified'
        risk_up = [s for s, v in per.items() if (not _undef(v['mean_pp'])) and (
            (v['mean_pp'] > 0 and ((v['d_w_share_pp'] or 0) > 0 or (v['d_u_share_pp'] or 0) > 0 or (v['d_full_w'] or 0) > 0)) or
            (v['mean_pp'] < 0 and ((v['d_w_share_pp'] or 0) < 0 or (v['d_u_share_pp'] or 0) < 0 or (v['d_full_w'] or 0) < 0)))]
        rows.append(dict(record_type='claim', contrast=cl['contrast'], method=cl['method'], sigma=cl['sigma'], family=fam, kinds=';'.join(cl['kinds']),
                         expected=';'.join(f"{s}:{cl['expected_signs'][s]}" for s in MF.SURF_SID), label=label, label_zh=LABELS_ZH[label],
                         per_surface=K.canon(per), risk_higher_on_favoured_arm=';'.join(risk_up)))
    return pd.DataFrame(rows)


def effects(R: pd.DataFrame, registry_rows: pd.DataFrame, registry: list) -> dict:
    import tr5_claims as CL
    exp = CL.expected(registry_rows)
    P = pd.concat([method_pairs(R, c) for c in MF.CONTRASTS], ignore_index=True)
    FP = frontend_pairs(R)
    BP = blocks(P, ('contrast', 'family', 'method', 'surface', 'sigma'))
    BF = blocks(FP.assign(contrast='DF-D0'), ('contrast', 'config', 'surface', 'sigma'))
    CP = cells(BP, ('contrast', 'method', 'surface', 'sigma'), lambda r: MF.CONTRASTS[r['contrast']][0], exp)
    CF = cells(BF, ('contrast', 'config', 'surface', 'sigma'), lambda r: 'point', None)
    CF['method'] = 'DF-D0'
    D = decisions(CP, registry)
    E = pd.concat([P, FP, BP.assign(record_type='block_method'), BF.assign(record_type='block_frontend'), CP.assign(record_type='cell_method'),
                   CF.assign(record_type='cell_frontend'), D], ignore_index=True, sort=False)
    return dict(effects=E, pairs=P, frontend=FP, cells=CP, cells_frontend=CF, decisions=D)


def select_cases(P: pd.DataFrame, expected: dict, contrast: str, method: str, sigma: float) -> dict:
    g = P[(P.contrast == contrast) & (P.method == method) & (P.sigma == sigma)].copy()
    if g.empty:
        return dict(contrast=contrast, method=method, sigma=sigma, median=None, worst=None, n_candidates=0)
    sg = [expected.get((contrast, method, float(sigma), s)) for s in g.surface]
    note = None
    if all(_missing_dir(x) for x in sg):
        sg = [1] * len(g); note = 'no registered direction: registered C1 / C5 positive direction used'
    g['oriented'] = [(1 if _missing_dir(s) else s) * float(d) for s, d in zip(sg, g.d_glob_Ctau)]
    g = g.sort_values(['surface', 'block', 'noise_k'], kind='mergesort').reset_index(drop=True)
    med = float(np.median(g.oriented))
    order_med = sorted(range(len(g)), key=lambda i: (abs(g.oriented[i] - med), g.surface[i], g.block[i], g.noise_k[i]))
    order_worst = sorted(range(len(g)), key=lambda i: (g.oriented[i], g.surface[i], g.block[i], g.noise_k[i]))
    w = order_worst[0]; m = order_med[0] if order_med[0] != w else order_med[1]

    def rec(i, kind):
        r = g.iloc[i]
        return dict(kind=kind, surface=r.surface, block=int(r.block), noise_k=int(r.noise_k), sigma=float(r.sigma), method=method, forward_key_left=r.forward_key_left,
                    forward_key_right=r.forward_key_right, oriented_diff_pp=float(r.oriented), d_glob_Ctau_pp=float(r.d_glob_Ctau), state_left=r.state_left, state_right=r.state_right)
    return dict(contrast=contrast, method=method, sigma=sigma, median_value=med, n_candidates=len(g), note=note, median=rec(m, 'median'), worst=rec(w, 'worst'))


RULES = dict(
    differences='left - right (C1 r1.0/R8x4 - r0.5/R8x4; C2 r1.0/T8x4 - r1.0/R8x4; C3 r1.0/R8x6 - r1.0/R8x4; C4 r1.0/T8x4 - r1.0/R8x6; C5 s10 - s8, 0 deg, w1); front ends DF - D0',
    pairing='contrast + surface + block + sigma + noise_k + method one-to-one; same registered seed; different forward keys; reject missing / duplicate / unlabeled / empty',
    order='pair -> block mean of 2 realisations -> equal-weight block mean; surfaces separate; errors conditional on both arms defined',
    bootstrap=f'per family {BOOT_N} x n_blocks block indices, numpy default_rng({BOOT_SEED}), shared across contrasts / methods / sigmas / surfaces / metrics; 2.5-97.5 % descriptive',
    decision='supported / reversed / insufficient / qualified / exploratory as in the module docstring (main metric global C_tau; 6 of 8 or 3 of 4 blocks; >= 2 surfaces)',
    risk='w / (c + w), u / (c + w + u), absolute w, no_output count reported per cell; risk on the favoured arm listed next to every claim label',
    cases='F4: C1 sigma 80 D0; F5: C5 sigma 60 P4; median / worst of the oriented realisation differences over all surfaces; ties (surface, block, noise_k)',
    recheck='12 results: one per surface x D0 / DF / P4 (first forward key), then legal no_output, highest u share, highest global point RMSE (first not yet chosen)')


def recheck_selection(R: pd.DataFrame) -> list:
    chosen = []
    for s in MF.SURF_SID:
        for m in ('D0', 'DF', 'P4'):
            g = R[(R.surface == s) & (R.method == m)].sort_values('forward_key')
            if len(g):
                chosen.append((g.forward_key.iloc[0], m))
    extra = [R[R.state == 'no_output'].sort_values('forward_key'), R.sort_values(['u_share', 'forward_key'], ascending=[False, True], na_position='last'),
             R.sort_values(['glob_pt_rmse', 'forward_key'], ascending=[False, True], na_position='last')]
    for g in extra:
        for fk, m in zip(g.forward_key, g.method):
            if (fk, m) not in chosen:
                chosen.append((fk, m)); break

    for g in (extra[2], extra[1]):
        for fk, m in zip(g.forward_key, g.method):
            if len(chosen) >= 12:
                break
            if (fk, m) not in chosen:
                chosen.append((fk, m))
    return chosen[:12]


def recheck(root, picks: list) -> list:

    out = []
    for fk, m in picks:
        meta, A, _ = CH.pack_read(root, fk); mt = meta['results'][m]['metrics']; fi = meta['results'][m].get('full_identity') or {}
        p = lambda k: np.asarray(A[f'{m}__ev__{k}'])
        pts, acc, val, zr = p('pts').astype(np.float64), p('accepted').astype(bool), p('valid').astype(bool), p('zr').astype(np.float64)
        e = pts[:, 2] - zr; rec = dict(forward_key=fk, method=m, state=meta['results'][m]['state'])
        for dn in ('glob', 'loc'):
            sel = val & p('in_global' if dn == 'glob' else 'in_local').astype(bool); x = e[sel]
            rmse = float(np.sqrt(np.mean(x * x))) if x.size else None; p95 = float(np.percentile(np.abs(x), 95)) if x.size else None
            rec[f'{dn}_n'] = int(sel.sum()); rec[f'{dn}_n_stored'] = mt[f'{dn}_n_valid']

            rec[f'{dn}_storage_tol_mm'] = (2.0 * float(np.spacing(np.float32(np.abs(zr[sel]).max()))) if sel.any() else 0.0)
            for nm, v in (('pt_rmse', rmse), ('pt_p95', p95)):
                sv = mt[f'{dn}_{nm}']
                rec[f'{dn}_{nm}_absdiff'] = (None if (v is None and sv is None) else (float('inf') if (v is None or sv is None) else abs(v - sv)))
        cwu = tuple(int((acc & p(k).astype(bool)).sum()) for k in ('point_correct', 'point_wrong', 'point_unconf'))
        rec['cwu_equal'] = cwu == (mt['id_correct'], mt['id_wrong'], mt['id_unconf'])
        if fi.get('applied'):
            fc = tuple(int((acc & p(k).astype(bool)).sum()) for k in ('full_point_correct', 'full_point_wrong', 'full_point_unconf'))
            rec['full_cwu_equal'] = fc == tuple(fi['counts']['all'][x] for x in 'cwu')
        rec['counts_equal'] = int(acc.sum()) == mt['n_accepted'] and int(val.sum()) == mt['n_valid'] and rec['glob_n'] == rec['glob_n_stored'] and rec['loc_n'] == rec['loc_n_stored']
        out.append(rec)
    return out
