# Aggregate environment results and compare paired candidate effects.

from __future__ import annotations

import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import r_common as K

K6 = K.K6
W6 = K.SRC6 / '_work'


def main_rows() -> pd.DataFrame:
    R = pd.read_csv(K.SRC5 / 'results.csv', float_precision='round_trip', low_memory=False)
    m = R[((R.family == 'point') & (R.method == 'DF')) | ((R.family == 'line') & (R.method == 'P4'))].copy()
    if len(m) != 2450 or m.case_key.duplicated().any():
        raise K.Stop('identity', f'main-method rows {len(m)}')
    return m


def _domains(surface: str):
    import rdy_platform as RP
    dl = RP.dom_local(surface); dg = RP.dom_global(); sid = RP.K.SURF_MAP[surface]
    return dict(loc=dict(x=dl['x'], y=dl['y'], z_ref=dl['z_ref'], area=dl['area'], hash=dl['hash']),
                glob=dict(x=dg['x'], y=dg['y'], z_ref=dg[f'z_{sid}'], area=np.full(len(dg['x']), dg['step'] ** 2), hash=dg['hash']))


def _r2_one(args):
    ck, method, surface = args
    from surface_eval_r3 import surface_eval_r3_dev
    meta, A = K6.read_pack(ck, prefixes=(f'{method}__ev__pts', f'{method}__ev__valid'))
    pts = np.asarray(A[f'{method}__ev__pts'], np.float64).reshape(-1, 3); v = np.asarray(A[f'{method}__ev__valid'], bool)
    P = pts[v]; rows = []
    for dom, D in _domains(surface).items():
        g = None
        if len(P) >= 3:
            g = surface_eval_r3_dev(P, None, D, list(K.DS), list(K.TAUS), return_fields=False, support_rule='exact')['subdomains']['global']
        for d in K.DS:
            bd = g['by_d'][str(float(d))] if g else None
            for tau in K.TAUS:
                rows.append(dict(case_key=ck, method=method, domain=dom, d_mm=d, tau_mm=tau, n_valid=int(v.sum()), n_nodes=int(len(D['x'])),
                                 n_interpolable=(int(g['n_interpolable']) if g else 0), Cvalid=(float(bd['C_valid']) if bd else 0.0),
                                 Ctau=(float(bd['C_tau'][str(float(tau))]) if bd else 0.0), sup_n=(int(bd['n_supported']) if bd else 0),
                                 area_supported_mm2=(float(bd['area_supported_mm2']) if bd else 0.0), area_total_mm2=float(np.asarray(D['area']).sum()),
                                 area_tau_mm2=(float(bd['area_tau_mm2'][str(float(tau))]) if bd else 0.0),
                                 sup_rmse=(K.num(bd['rmse']) if bd else None), sup_p95=(K.num(bd['p95']) if bd else None), sup_bias=(K.num(bd['bias']) if bd else None),
                                 sup_max_abs=(K.num(bd['max_abs']) if bd else None), evaluated=bool(g is not None)))
    return rows


def a1_reeval(workers: int = 16) -> tuple[pd.DataFrame, dict]:
    Mn = main_rows(); jobs = list(zip(Mn.case_key, Mn.method, Mn.surface)); rows = []; t0 = time.time()
    with Pool(workers) as pool:
        for rr in pool.imap_unordered(_r2_one, jobs, chunksize=8):
            rows.extend(rr)
    T = pd.DataFrame(rows).merge(Mn[['case_key', 'family', 'config', 'surface', 'phase_block', 'sigma', 'rep', 'env', 'psf_true_px', 'subset', 'state']], on='case_key', how='left', validate='many_to_one')
    T = T.sort_values(['case_key', 'domain', 'd_mm', 'tau_mm']).reset_index(drop=True)
    chk = {}
    if T.duplicated(['case_key', 'domain', 'd_mm', 'tau_mm']).any():
        raise K.Stop('identity', 'duplicate evaluation keys')
    sup = T.sup_n > 0
    chk['rows'] = len(T); chk['supported_rows'] = int(sup.sum())
    chk['supported_with_finite_rmse_p95'] = int((sup & T.sup_rmse.notna() & T.sup_p95.notna() & np.isfinite(T.sup_p95.astype(float))).sum())
    chk['unsupported_with_error_stats'] = int((~sup & (T.sup_rmse.notna() | T.sup_p95.notna())).sum())
    chk['bad_order_0_Ctau_Cvalid_1'] = int(((T.Ctau < 0) | (T.Ctau > T.Cvalid + 1e-15) | (T.Cvalid > 1 + 1e-15)).sum())
    ev = T[T.evaluated]
    chk['Cvalid_vs_area_ratio_maxdiff'] = float(np.abs(ev.Cvalid - ev.area_supported_mm2 / ev.area_total_mm2).max())
    chk['Ctau_vs_area_ratio_maxdiff'] = float(np.abs(ev.Ctau - ev.area_tau_mm2 / ev.area_total_mm2).max())
    gl = ev[ev.domain == 'glob']
    chk['global_Cvalid_vs_node_ratio_maxdiff'] = float(np.abs(gl.Cvalid - gl.sup_n / gl.n_nodes).max())
    chk['p95_le_max'] = int((T.sup_p95 > T.sup_max_abs + 1e-15).sum()) == 0

    O = pd.read_csv(W6 / 'a1_long.csv', float_precision='round_trip', low_memory=False)
    key = ['case_key', 'domain', 'd_mm', 'tau_mm']
    J = T.merge(O, on=key, how='outer', suffixes=('', '_old'), indicator=True, validate='one_to_one')
    reg = dict(rows_new=len(T), rows_old=len(O), unmatched=int((J._merge != 'both').sum()))
    for c in ('Cvalid', 'Ctau', 'sup_n', 'sup_rmse', 'sup_bias', 'n_valid'):
        a = J[c].astype(float); b = J[f'{c}_old'].astype(float)
        reg[f'{c}_nan_pattern_equal'] = bool((a.isna() == b.isna()).all())
        reg[f'{c}_maxdiff'] = float(np.nanmax(np.abs(a - b))) if (a.notna() & b.notna()).any() else 0.0
    reg['old_sup_p95_defined'] = int(O.sup_p95.notna().sum())
    reg['ok'] = bool(reg['unmatched'] == 0 and all(reg[f'{c}_nan_pattern_equal'] for c in ('Cvalid', 'Ctau', 'sup_n', 'sup_rmse', 'sup_bias', 'n_valid'))
                     and all(reg[f'{c}_maxdiff'] <= K.TOL for c in ('Cvalid', 'Ctau', 'sup_n', 'sup_rmse', 'sup_bias', 'n_valid')))
    stored = Mn.set_index('case_key')
    base = T[(T.tau_mm == 0.15) & (T.d_mm == 6.0)].pivot_table(index='case_key', columns='domain', values=['Ctau', 'Cvalid'], aggfunc='first')
    rep = {f'{d}_{m}': float(np.max(np.abs(base[(m, d)].reindex(stored.index).astype(float).to_numpy() - stored[f'{d}_{m}'].astype(float).to_numpy())))
           for d in ('glob', 'loc') for m in ('Ctau', 'Cvalid')}
    a = T[(T.case_key == 'D_r1_p8x6|M10_curv|b0|s20|r1|NOMINAL') & (T.domain == 'glob') & (T.d_mm == 5.0)]
    anchor = dict(case='D_r1_p8x6|M10_curv|b0|s20|r1|NOMINAL', domain='glob', d=5.0, sup_p95=sorted(set(a.sup_p95.round(15))), expected_approx=0.15467083225)
    ok = (chk['supported_rows'] == chk['supported_with_finite_rmse_p95'] and chk['unsupported_with_error_stats'] == 0 and chk['bad_order_0_Ctau_Cvalid_1'] == 0
          and chk['Cvalid_vs_area_ratio_maxdiff'] <= 1e-12 and chk['Ctau_vs_area_ratio_maxdiff'] <= 1e-12 and chk['global_Cvalid_vs_node_ratio_maxdiff'] <= 1e-12
          and chk['p95_le_max'] and reg['ok'] and all(v <= K.TOL for v in rep.values()) and len(anchor['sup_p95']) == 1 and abs(anchor['sup_p95'][0] - 0.15467083225) < 1e-9)
    return T, dict(ok=bool(ok), checks=chk, regression_vs_06A=reg, main_reproduction=rep, anchor=anchor, seconds=round(time.time() - t0, 1), workers=workers,
                   tau_reuse='one surface_eval_r3_dev call per (case, domain) with all d and tau; support / errors computed once')


ENVS7 = ('NOMINAL', 'BLUR1_5', 'BLUR2', 'BLUR3', 'UNIFORM', 'GRAD_X', 'GRAD_Y')


def width_effects(T: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import m6_stats as ST
    L = T[T.config.isin(['L_w0.5_s8', 'L_w1.5_s8']) & T.env.isin(ENVS7) & (T.sigma > 0)]
    keys = ['surface', 'phase_block', 'sigma', 'rep', 'env', 'domain', 'd_mm', 'tau_mm']
    out = []
    for metric in ('Ctau', 'Cvalid'):
        a = L[L.config == 'L_w1.5_s8'][keys + [metric]]; b = L[L.config == 'L_w0.5_s8'][keys + [metric]]
        P = ST.pair(a, b, keys, [metric], suffixes=('_w', '_t')); P['value'] = P[f'{metric}_w'] - P[f'{metric}_t']
        for k, g in P.groupby(['surface', 'sigma', 'env', 'domain', 'd_mm', 'tau_mm']):
            s = ST.blocks(g[['phase_block', 'rep', 'value']])
            out.append(dict(metric=metric, surface=k[0], sigma=k[1], env=k[2], domain=k[3], d_mm=k[4], tau_mm=k[5], n_pairs=len(g), **ST.flat('', s)))
    E = pd.DataFrame(out)
    agg = []
    for k, g in E.groupby(['metric', 'sigma', 'env', 'domain', 'd_mm', 'tau_mm']):
        v = g['mean']
        agg.append(dict(metric=k[0], surface='three surfaces equal weight', sigma=k[1], env=k[2], domain=k[3], d_mm=k[4], tau_mm=k[5], mean=float(v.mean()) if v.notna().all() and len(v) == 3 else None,
                        min=float(v.min()), max=float(v.max()), n_surface_pos=int((v > 0).sum()), n_surface_neg=int((v < 0).sum()),
                        n_block_pos=int(g.n_pos.sum()), n_block_neg=int(g.n_neg.sum()), within_tie_band=bool(abs(v.mean()) < 0.02)))
    E = pd.concat([E, pd.DataFrame(agg)], ignore_index=True)

    O = pd.read_csv(K.SRC6 / 'effects.csv', low_memory=False)
    O = O[(O.analysis == 'A1') & (O.contrast == 'Linienbreite 1,5 - 0,5 mm (8 mm)') & O.surface.isin(K.SURFS)]
    J = O.merge(E[E.surface.isin(K.SURFS)], on=['metric', 'surface', 'sigma', 'env', 'domain', 'd_mm', 'tau_mm'], suffixes=('_old', ''), validate='one_to_one')
    reg = dict(rows_06A=len(O), rows_matched=len(J), rows_06A_Ctau=int((O.metric == 'Ctau').sum()),
               mean_maxdiff=float((J['mean'] - J['mean_old']).abs().max()), blocks_maxdiff=max(float((J[f'b{b}'] - J[f'b{b}_old']).abs().max()) for b in K.BLOCKS))
    reg['ok'] = bool(reg['rows_matched'] == reg['rows_06A'] and reg['mean_maxdiff'] <= 1e-12 and reg['blocks_maxdiff'] <= 1e-12)
    anc = {}
    for env, exp in (('UNIFORM', (-1.82, -0.57, 2.23, 9.81)), ('GRAD_X', (-4.00, -3.20, -0.85, 5.96))):
        q = E[(E.metric == 'Ctau') & (E.surface == 'three surfaces equal weight') & (E.sigma == 40.0) & (E.env == env) & (E.domain == 'glob') & (E.d_mm == 6.0)].set_index('tau_mm')
        anc[env] = {str(t): dict(value_pp=round(100 * float(q.loc[t, 'mean']), 4), expected_pp=e, match_0p01=abs(100 * float(q.loc[t, 'mean']) - e) <= 0.006) for t, e in zip(K.TAUS, exp)}
    reg['anchors'] = anc
    reg['anchors_ok'] = all(v['match_0p01'] for d in anc.values() for v in d.values())
    return E, reg


# Aggregate paired image-sampling differences by surface and environment.
def subray_levels() -> tuple[pd.DataFrame, dict]:
    B = pd.read_csv(W6 / 'b_pairs.csv', float_precision='round_trip', low_memory=False)
    if len(B) != 96 or B.case_key.duplicated().any():
        raise K.Stop('identity', f'b_pairs: {len(B)} rows')

    noise = {}
    for pf in sorted((K.SRC6 / 'data' / 'b').glob('*.npz')):
        with np.load(pf, allow_pickle=False) as z:
            meta = json.loads(str(z['_meta']))
        for tag, im in meta['images'].items():
            pm, _ = K6.read_pack(im['case_key_parent'], prefixes=('image',))
            noise[im['case_key_parent']] = dict(noise_equal=im['diag'].get('noise_sha') == pm['image']['diag'].get('noise_sha'), rays=im['diag'].get('rays_per_pixel'),
                                                rep_label=im['case_key_parent'].split('|')[4])
    if len(noise) != 96 or not all(v['noise_equal'] and v['rays'] == 3 and v['rep_label'] == 'r1' for v in noise.values()):
        raise K.Stop('pairing', 'sub-ray noise / rep binding incomplete')
    B['rep'] = 1
    keys = ['env', 'sigma', 'surface', 'phase_block', 'rep']
    wide = B[B.config == 'L_w1.5_s8']; thin = B[B.config == 'L_w0.5_s8']
    P = wide.merge(thin, on=keys, suffixes=('_w', '_t'), validate='one_to_one')
    if len(P) != 48:
        raise K.Stop('pairing', f'{len(P)} width pairs (48 expected)')
    rows = []
    def flag(old, new):
        tie = abs(old) >= 0.02 and (abs(new) < 0.02 or np.sign(old) != np.sign(new))
        return bool(tie), bool(abs(new - old) >= 0.05)
    for dom in ('glob', 'loc'):
        P[f'old_{dom}'] = P[f'{dom}_Ctau_1_w'] - P[f'{dom}_Ctau_1_t']; P[f'new_{dom}'] = P[f'{dom}_Ctau_3_w'] - P[f'{dom}_Ctau_3_t']
        for r in P.to_dict('records'):
            t, c = flag(r[f'old_{dom}'], r[f'new_{dom}'])
            rows.append(dict(level='pair (surface x phase block)', domain=dom, env=r['env'], sigma=r['sigma'], surface=r['surface'], phase_block=int(r['phase_block']), rep=1,
                             old=r[f'old_{dom}'], new=r[f'new_{dom}'], change=r[f'new_{dom}'] - r[f'old_{dom}'], into_tie_or_flip=t, change_ge_5pp=c,
                             new_wrong_ids=int(((r['w_3_w'] or 0) > 0 and (r['w_1_w'] or 0) == 0) + ((r['w_3_t'] or 0) > 0 and (r['w_1_t'] or 0) == 0))))
        S = P.groupby(['env', 'sigma', 'surface'])[[f'old_{dom}', f'new_{dom}']].mean().reset_index()
        for r in S.to_dict('records'):
            t, c = flag(r[f'old_{dom}'], r[f'new_{dom}'])
            rows.append(dict(level='surface (mean of blocks 0, 2)', domain=dom, env=r['env'], sigma=r['sigma'], surface=r['surface'], old=r[f'old_{dom}'], new=r[f'new_{dom}'],
                             change=r[f'new_{dom}'] - r[f'old_{dom}'], into_tie_or_flip=t, change_ge_5pp=c))
        A = S.groupby(['env', 'sigma'])[[f'old_{dom}', f'new_{dom}']].mean().reset_index()
        for r in A.to_dict('records'):
            t, c = flag(r[f'old_{dom}'], r[f'new_{dom}'])
            rows.append(dict(level='environment x noise (three surfaces equal weight)', domain=dom, env=r['env'], sigma=r['sigma'], surface='alle', old=r[f'old_{dom}'], new=r[f'new_{dom}'],
                             change=r[f'new_{dom}'] - r[f'old_{dom}'], into_tie_or_flip=t, change_ge_5pp=c))
    Lv = pd.DataFrame(rows)
    g = Lv[Lv.domain == 'glob']
    counts = {lv: int((g.level == lv).sum()) for lv in g.level.unique()}
    flags = g[g.into_tie_or_flip | g.change_ge_5pp]
    anc = {}
    for surf, exp in (('B_curv', (-25.8862, -18.3701, 7.5161)), ('M24_curv', (-12.5020, -4.7759, 7.7261))):
        q = g[(g.level.str.startswith('pair')) & (g.env == 'NOMINAL') & (g.sigma == 40.0) & (g.surface == surf) & (g.phase_block == 2)].iloc[0]
        anc[surf] = dict(old_pp=round(100 * q.old, 4), new_pp=round(100 * q.new, 4), change_pp=round(100 * q.change, 4), expected=exp,
                         match=all(abs(a - b) <= 1e-4 for a, b in zip((100 * q.old, 100 * q.new, 100 * q.change), exp)))
    per_image = dict(n_images=96, new_wrong_id_images=int(((B.w_3.fillna(0) > 0) & (B.w_1.fillna(0) == 0)).sum()),
                     max_abs_change_glob_pt_rmse_mm=float((B.glob_pt_rmse_3 - B.glob_pt_rmse_1).abs().max()),
                     max_abs_change_glob_pt_p95_mm=float((B.glob_pt_p95_3 - B.glob_pt_p95_1).abs().max()),
                     state_transitions={f'{a}->{b}': int(n) for (a, b), n in B.groupby(['state_1', 'state_3']).size().items()})
    rec = dict(level_counts_glob=counts, flags_glob=flags.to_dict('records'), flags_loc=Lv[(Lv.domain == 'loc') & (Lv.into_tie_or_flip | Lv.change_ge_5pp)].to_dict('records'),
               anchors=anc, anchors_ok=all(v['match'] for v in anc.values()), per_image=per_image,
               any_tie_or_flip_glob=bool(g.into_tie_or_flip.any()), noise_binding='3x3 noise sha == 1x1 parent noise sha for all 96; rays 3; rep label r1')
    rec['ok'] = bool(counts == {'pair (surface x phase block)': 48, 'surface (mean of blocks 0, 2)': 24, 'environment x noise (three surfaces equal weight)': 8} and rec['anchors_ok'])
    return Lv, rec


def a3_conditions() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    A = pd.read_csv(W6 / 'a3_arms.csv', float_precision='round_trip', low_memory=False)
    if len(A) != 1568 or A.duplicated(['case_key', 'arm']).any() or A.state.isna().any():
        raise K.Stop('identity', 'a3_arms incomplete or duplicated')
    if set(A.state) - {'ok', 'no_output'}:
        raise K.Stop('integrity', f'unexpected states {set(A.state)}')
    nz = A[(A.state == 'no_output') & (A.glob_Ctau.fillna(-1) != 0)]
    if len(nz):
        raise K.Stop('integrity', 'legal no-output rows without C_tau 0')
    A = A[A.sigma > 0].copy()
    rows = []
    for (cfg, env, sg, arm), g in A.groupby(['config', 'env', 'sigma', 'arm']):
        u = g.groupby(['surface', 'phase_block']).glob_pt_rmse.apply(lambda s: int(s.notna().sum()))
        dfu = g[(g.c.fillna(0) + g.w.fillna(0) + g.u.fillna(0)) > 0]
        rows.append(dict(config=cfg, env=env, sigma=sg, arm=arm, alias_of=';'.join(sorted(set(g.alias_of.fillna('')) - {''})), n_img=len(g), n_no_output=int((g.state == 'no_output').sum()),
                         n_rmse_defined=int(g.glob_pt_rmse.notna().sum()), blocks_2_reps_defined=int((u == 2).sum()), blocks_1_rep_defined=int((u == 1).sum()), blocks_0_reps_defined=int((u == 0).sum()),
                         glob_Ctau_mean_all_images=float(g.glob_Ctau.mean()), rmse_mean_over_defined_images=(float(g.glob_pt_rmse.mean()) if g.glob_pt_rmse.notna().any() else None),
                         rmse_max_image=(float(g.glob_pt_rmse.max()) if g.glob_pt_rmse.notna().any() else None),
                         n_identity_defined=len(dfu), u_share_mean_over_defined=(float(dfu.u_share.mean()) if len(dfu) else None),
                         u_share_max_image=(float(dfu.u_share.max()) if len(dfu) else None), w_sum=int(g.w.fillna(0).sum())))
    C = pd.DataFrame(rows)

    piv = {a: A[A.arm == a].set_index('case_key') for a in ('N', 'T', 'S', 'ST')}
    prow = []
    for arm in ('T', 'S', 'ST'):
        X = piv[arm].join(piv['N'][['glob_pt_rmse', 'state', 'glob_Ctau']], rsuffix='_N')
        for (cfg, env, sg), g in X.groupby(['config', 'env', 'sigma']):
            both = g.glob_pt_rmse.notna() & g.glob_pt_rmse_N.notna(); d = (g.glob_pt_rmse - g.glob_pt_rmse_N)[both]
            prow.append(dict(contrast=f'{arm}-N', config=cfg, env=env, sigma=sg, n_img=len(g), n_pairs_used=int(both.sum()), n_excl_N_undefined=int((g.glob_pt_rmse.notna() & g.glob_pt_rmse_N.isna()).sum()),
                             n_excl_arm_undefined=int((g.glob_pt_rmse.isna() & g.glob_pt_rmse_N.notna()).sum()), n_excl_both_undefined=int((g.glob_pt_rmse.isna() & g.glob_pt_rmse_N.isna()).sum()),
                             rmse_diff_mean_mm=(float(d.mean()) if len(d) else None), rmse_diff_min_mm=(float(d.min()) if len(d) else None), rmse_diff_max_mm=(float(d.max()) if len(d) else None),
                             no_output_N_to_ok=int(((g.state_N == 'no_output') & (g.state == 'ok')).sum()), ok_to_no_output=int(((g.state_N == 'ok') & (g.state == 'no_output')).sum()),
                             ctau_diff_image_maxabs_pp=float(100 * (g.glob_Ctau - g.glob_Ctau_N).abs().max())))
    Pd = pd.DataFrame(prow)
    one = A.groupby(['config', 'surface', 'phase_block', 'env', 'sigma', 'arm']).glob_pt_rmse.apply(lambda s: int(s.notna().sum()) == 1)

    L6 = pd.read_csv(W6 / 'a3_levels.csv')
    J = L6.merge(C, on=['config', 'env', 'sigma', 'arm'], how='inner', validate='one_to_one', suffixes=('_06A', ''))
    J['rmse_old_06A'] = J.glob_pt_rmse; J['rmse_new_defined_images'] = J.rmse_mean_over_defined_images
    J['differs'] = (J.rmse_old_06A - J.rmse_new_defined_images).abs() > 1e-12
    changed = J[J.differs | (J.rmse_old_06A.isna() != J.rmse_new_defined_images.isna())][['config', 'env', 'sigma', 'arm', 'n_rmse_defined', 'n_img', 'rmse_old_06A', 'rmse_new_defined_images']]
    tn = Pd[Pd.contrast == 'T-N']
    umax = A.loc[A.u_share.idxmax()]
    rec = dict(groups=len(C), groups_with_one_rep_defined=int(one.sum()), T_minus_N_image_maxabs_pp=float(tn.ctau_diff_image_maxabs_pp.max()),
               u_share_max_image=dict(value=float(umax.u_share), case=umax.case_key, arm=umax.arm, c=umax.c, w=umax.w, u=umax.u),
               u_share_condition_mean_max=float(C.u_share_mean_over_defined.max()), w_total=int(A.w.fillna(0).sum()),
               rmse_definition_changes=changed.to_dict('records'),
               rmse_definition_note='06A a3_levels: per surface rep -> block mean of DEFINED values (NaN skipped), then three surfaces; corrected table: mean over defined images with '
                                    'n_rmse_defined / n_img and the per-block counts of defined reps; paired differences only on images where both arms are defined')
    return C, Pd, rec
