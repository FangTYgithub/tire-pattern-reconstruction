# Analyze support, precision, line response and dot-gap mechanisms.

from __future__ import annotations

import json
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import m6_common as K
import m6_stats as ST

K5 = K.K5


def manifest() -> pd.DataFrame:
    return pd.read_csv(K.SRC5 / 'formal_manifest.csv', float_precision='round_trip', low_memory=False)


def results() -> pd.DataFrame:
    return pd.read_csv(K.SRC5 / 'results.csv', float_precision='round_trip', low_memory=False)


def main_rows(R: pd.DataFrame) -> pd.DataFrame:

    m = R[((R.family == 'point') & (R.method == 'DF')) | ((R.family == 'line') & (R.method == 'P4'))].copy()
    if len(m) != 2450 or m.case_key.duplicated().any():
        raise K.Stop('identity', f'main-method rows {len(m)} (2450 expected) / duplicates {int(m.case_key.duplicated().sum())}')
    return m


def code_hashes() -> dict:
    return {f: K.sha_file(K.ROOT / f) for f in K.CODE_FILES}


def stage_bind() -> dict:
    import m6_platform as PL
    p = K.protocol()
    if 'rules' in p and p.get('first_diagnostic_output'):
        raise K.Stop('refused', 'rules were registered and diagnostic output exists: rules are frozen')
    M = manifest()

    arms = []
    for cfg in K.LINES8:
        r0 = M[(M.config == cfg) & (M.env == 'NOMINAL')].iloc[0].to_dict()
        pr = PL.prep(r0); inp = PL.platform()['q'].inputs(pr['decoder_row']); w_px = float(inp['meta_full']['olines']['width_px'])
        for env, psf in K.PSF_ENVS:
            ss = PL.scale_star(w_px, psf)
            for arm, sc, ps in K.ARMS:
                s = ss['s0'] if sc == 's0' else ss['s_star']
                pp = 1.0 if ps == 'nominal' else psf
                arms.append(dict(config=cfg, env=env, psf_true_px=psf, arm=arm, w_px=w_px, s0=ss['s0'], s_star=ss['s_star'], steger_sigma_px=s, psf_design_px=pp,
                                 coef=PL.coef_at(s), setting_key=K.digest(dict(config=cfg, psf=psf, s=s, psf_design=pp))))
    A = pd.DataFrame(arms)
    first = A.groupby(['config', 'env', 'setting_key']).arm.transform('first')
    A['alias_of'] = np.where(first != A.arm, first, '')
    p4 = K.load_json(K.P4_CFG)['P4_C3_cal']
    ks, cs = PL.p4_nodes()
    rules = dict(selection=K.SELECTION, holdout=K.HOLDOUT, a3_rule=K.A3_RULE, arms=[list(a) for a in K.ARMS], arm_table=A.to_dict('records'),
                 taus=list(K.TAUS), ds=list(K.DS), main=dict(tau=K.TAU0, d=K.D0_SUPPORT), repro_tol=K.REPRO_TOL, representatives=list(K.REPRESENTATIVES),
                 caps=K.CAPS, subray=K.SUBRAY, p4_nodes=ks, p4_coefs=cs, p4_table_id=p4.get('steger_noise_coef_id'),
                 shares=dict(w_share='w/(c+w)', u_share='u/(c+w+u)', zero_denominator='undefined'),
                 layers=dict(feature='expected / detected / candidate rejections by code / accepted / c / w / u (sample unit = feature of one image)',
                             surface='1 - Cvalid (no support), Cvalid - Ctau (supported, error > tau), Ctau (accurate); same node denominator'))
    K.save_json(K.OUT / 'arm_table.json', rules['arm_table'])
    K.protocol_put('rules', rules)
    run = K.load_json(K.SRC5 / 'run.json')
    b = dict(t=K.now(), inputs={f: K.sha_file(K.ROOT / f) for f in K.INPUT_FILES}, code=code_hashes(), task_book_sha256=K.sha_file(K.TASK_BOOK),
             reviews={K.rel(f): K.sha_file(f) for f in K.REVIEWS}, mp5_final_status=run['final']['status'], mp5_outputs_indexed=len(run['final']['outputs']),
             manifest_rows=len(M), p4_table_bound=dict(file=K.rel(K.P4_CFG), nodes=ks, coefs=cs, id=p4.get('steger_noise_coef_id'),
                                                        source_default_table='m4_lines.STEGER_NOISE_COEF (three nodes) NOT used: P4_C3_cal passes the five-node table'),
             s5_align05_status=K.load_json(K.S5A / 'run.json').get('status') or K.load_json(K.S5A / 'run.json').get('final', {}).get('status'))
    b['mp5_check'] = mp5_check()
    K.protocol_put('binding', b)
    return dict(arm_rows=len(A), aliases=int((A.alias_of != '').sum()), mp5_check=b['mp5_check'])


def mp5_check() -> dict:

    import mp5_gate as G
    import mp5_main as MN
    run = K.load_json(K.SRC5 / 'run.json')
    idx = run['final']['outputs']
    changed = [k for k, v in idx.items() if K.sha_file(K.ROOT / k) != v]
    prot = run['protected']['files']
    prot_changed = [k for k, v in prot.items() if K.sha_file((K.ROOT / k).resolve()) != v]
    fz = G.verify_freeze(run.get('freeze'), MN.MANIFEST, MN.rules_all())
    return dict(outputs_indexed=len(idx), outputs_changed=changed, protected=len(prot), protected_changed=prot_changed, freeze_problems=fz)


REGRESS = ('L_w0.5_s8|M10_curv|b0|s0|r0|NOMINAL', 'L_w1.5_s8|M10_curv|b0|s0|r0|BLUR3', 'L_w0.5_s8|B_curv|b1|s20|r1|NOMINAL', 'L_w1.5_s8|M24_curv|b2|s20|r2|BLUR1_5',
           'L_w0.5_s8|M24_curv|b3|s40|r1|BLUR2', 'L_w1.5_s8|B_curv|b0|s40|r2|NOMINAL', 'L_w0.5_s8|M10_curv|b2|s40|r2|BLUR3', 'L_w1.5_s8|M10_curv|b1|s20|r1|BLUR2')


def _expect_stop(fn) -> str:
    try:
        fn()
    except K.Stop as e:
        return f'raised {e.code}'
    except Exception as e:
        return f'raised {type(e).__name__}'
    return 'NOT RAISED'


def drills() -> dict:
    import m6_platform as PL
    M = manifest(); out = {}
    row = M[M.case_key == 'L_w0.5_s8|M10_curv|b0|s20|r1|NOMINAL'].iloc[0].to_dict()
    r1 = dict(row); r1.pop('raster_key')
    out['missing_field'] = _expect_stop(lambda: PL.prep(r1))
    r2 = dict(row); r2['peak_f32'] = float(np.float32(row['peak_f32']) * np.float32(1.001))
    out['config_drift_peak'] = _expect_stop(lambda: PL.prep(r2))
    r3 = dict(row); r3['psf_true_px'] = 2.0
    out['config_drift_forward_key'] = _expect_stop(lambda: PL.prep(r3))
    out['wrong_image_key'] = _expect_stop(lambda: K.read_pack('L_w0.5_s8|M10_curv|b9|s20|r1|NOMINAL'))
    g = pd.DataFrame(dict(phase_block=[0, 0, 0, 1, 1, 2, 2, 3, 3], rep=[1, 1, 2, 1, 2, 1, 2, 1, 2], value=np.arange(9.0)))
    out['duplicate_rep'] = _expect_stop(lambda: ST.blocks(g))
    a = pd.DataFrame(dict(k=[1, 1, 2], v=[0.1, 0.2, 0.3])); b = pd.DataFrame(dict(k=[1, 2], v=[0.0, 0.0]))
    out['many_to_many_pairing'] = _expect_stop(lambda: ST.pair(a, b, ['k'], ['v']))
    g2 = pd.DataFrame(dict(phase_block=[0, 0, 1, 1, 2, 2, 3, 3], rep=[1, 2] * 4, value=[np.nan] * 8))
    s = ST.blocks(g2)
    out['all_undefined_stays_undefined'] = 'ok' if (s['mean'] is None and s['n_blocks_def'] == 0) else f'WRONG {s}'
    out['nonfinite_metric'] = _expect_stop(lambda: _finite_or_stop(np.inf, 'glob_Ctau'))
    R = results(); nooutput = R[(R.method == 'P4') & (R.state == 'no_output')].iloc[0]
    meta, A = K.read_pack(nooutput.case_key, prefixes=('P4__ev__',))
    v = np.asarray(A['P4__ev__valid'], bool)
    out['legal_no_output'] = dict(case=nooutput.case_key, n_valid=int(v.sum()), glob_Ctau=float(nooutput.glob_Ctau), r3_on_empty=(PL.r3_multi(np.zeros((0, 3)), nooutput.surface)['glob'] is None))
    rc = subprocess.run([sys.executable, '-B', str(K.HERE / 'm6_main.py'), 'no_such_stage'], capture_output=True, text=True, cwd=str(K.ROOT))
    out['nonzero_exit_on_bad_stage'] = rc.returncode
    ok = all(str(v).startswith('raised') for k, v in out.items() if k in ('missing_field', 'config_drift_peak', 'config_drift_forward_key', 'wrong_image_key', 'duplicate_rep',
                                                                           'many_to_many_pairing', 'nonfinite_metric'))
    ok = ok and out['all_undefined_stays_undefined'] == 'ok' and out['legal_no_output']['n_valid'] == 0 and out['legal_no_output']['r3_on_empty'] and rc.returncode != 0
    return dict(ok=bool(ok), cases=out)


def _finite_or_stop(x, name):
    if not np.isfinite(x):
        raise K.Stop('integrity', f'{name} non-finite')


def regression(L: K.Ledger) -> dict:
    import m6_platform as PL
    M = manifest().set_index('case_key'); out = {}
    for ck in REGRESS:
        pr = PL.prep(M.loc[ck].to_dict() | dict(case_key=ck))
        meta, A = K.read_pack(ck)
        old = meta['results']['P4']
        rid = L.reserve('regress_P4', ck, K.CAP_A)
        try:
            raw, dmeta, thr = PL.p4_decode(A['image'], pr)
            ev, rec, fi = PL.evaluate(pr, 'P4', raw, dmeta)
            L.end(rid, 'ok')
        except BaseException as e:
            L.end(rid, 'failed', error=repr(e)[:300]); raise
        raw_old = K.sub(A, 'P4__raw__')
        out[ck] = dict(raw_digest=K5.arrays_digest(raw) == old['raw_digest'], raw_arrays_equal=all(np.array_equal(raw[k], raw_old[k], equal_nan=True) for k in raw) and set(raw) == set(raw_old),
                       decode_meta=K5.canon(K5.clean(dmeta)) == K5.canon(old['decode_meta']), eval_digest=K5.arrays_digest(ev) == old['eval_digest'],
                       metrics=K5.canon(K5.clean(rec['metrics'])) == K5.canon(old['metrics']), identity=K5.canon(K5.clean(rec['full_identity'])) == K5.canon(old['full_identity']),
                       state=rec['state'], thr=thr)
    ok = all(all(v for k, v in d.items() if k not in ('state', 'thr')) for d in out.values())
    return dict(ok=ok, n=len(out), cases=out)


def stage_a0() -> dict:
    K.rules_registered()
    L = K.Ledger()
    d = drills(); r = regression(L)
    rec = dict(t=K.now(), drills=d, regression=r, ok=bool(d['ok'] and r['ok']))
    K.protocol_put('A0', rec)
    return dict(ok=rec['ok'], drills=d['ok'], regression=r['ok'], n_regress=r['n'])


def _a1_one(args):
    ck, method, surface = args
    import m6_platform as PL
    meta, A = K.read_pack(ck, prefixes=(f'{method}__ev__pts', f'{method}__ev__valid'))
    pts = np.asarray(A[f'{method}__ev__pts'], np.float64).reshape(-1, 3); v = np.asarray(A[f'{method}__ev__valid'], bool)
    r = PL.r3_multi(pts[v], surface)
    rows = []
    for dom in ('loc', 'glob'):
        g = r[dom]
        for d in K.DS:
            bd = (g or {}).get('by_d', {}).get(str(float(d))) if g else None
            for tau in K.TAUS:
                rows.append(dict(case_key=ck, method=method, domain=dom, d_mm=d, tau_mm=tau, n_valid=int(v.sum()),
                                 Cvalid=(float(bd['C_valid']) if bd else 0.0), Ctau=(float(bd['C_tau'][str(float(tau))]) if bd else 0.0),
                                 sup_n=(int(bd['n_supported']) if bd else 0), sup_rmse=(K.num(bd.get('rmse')) if bd else None), sup_p95=(K.num(bd.get('p95_abs')) if bd else None),
                                 sup_bias=(K.num(bd.get('bias')) if bd else None)))
    return rows


def stage_a1(workers: int = 16) -> dict:
    K.rules_registered()
    R = results(); Mn = main_rows(R)
    K.protocol_put('first_diagnostic_output', dict(t=K.now(), stage='A1'))
    jobs = list(zip(Mn.case_key, Mn.method, Mn.surface))
    t0 = time.time(); rows = []
    with Pool(workers) as pool:
        for i, rr in enumerate(pool.imap_unordered(_a1_one, jobs, chunksize=8)):
            rows.extend(rr)
            if (i + 1) % 250 == 0:
                print(f'A1 {i + 1}/{len(jobs)} {round(time.time() - t0)} s', flush=True)
    T = pd.DataFrame(rows)
    T = T.merge(Mn[['case_key', 'family', 'config', 'surface', 'phase_block', 'sigma', 'rep', 'env', 'psf_true_px', 'subset', 'state', 'glob_pt_rmse', 'glob_pt_p95']], on='case_key', how='left',
                validate='many_to_one')

    base = T[(T.tau_mm == K.TAU0) & (T.d_mm == K.D0_SUPPORT)].pivot_table(index='case_key', columns='domain', values=['Ctau', 'Cvalid'], aggfunc='first')
    base.columns = [f'{d}_{m}' for m, d in base.columns]
    stored = Mn.set_index('case_key')[['glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid']]
    mx = {}
    for dom in ('glob', 'loc'):
        for m in ('Ctau', 'Cvalid'):
            new = base[f'{dom}_{m}'].reindex(stored.index).astype(float).to_numpy()
            old = stored[f'{dom}_{m}'].astype(float).to_numpy()
            if np.isnan(new).any() or np.isnan(old).any():
                raise K.Stop('integrity', f'A1 {dom}_{m}: missing stored or reproduced values')
            mx[f'{dom}_{m}'] = float(np.max(np.abs(old - new)))
    ok = all(v <= K.REPRO_TOL for v in mx.values())
    K.save_csv(T, K.OUT / '_work' / 'a1_long.csv')
    rec = dict(t=K.now(), n_records=len(Mn), n_rows=len(T), max_abs_diff_vs_stored=mx, reproduced=ok, seconds=round(time.time() - t0, 1), workers=workers)
    K.protocol_put('A1', rec)
    return rec


def _a2_one(args):
    ck, method, family = args
    meta, A = K.read_pack(ck)
    r = meta['results'][method]; info = r.get('info') or {}; m = r.get('metrics') or {}
    out = dict(case_key=ck, method=method)
    if family == 'line':
        cnt = info.get('counts') or {}
        out.update(expected=m.get('img_n_expected'), detected=cnt.get('n_detected'), accepted=cnt.get('accepted'), recovered_id=m.get('img_n_recovered'),
                   **{f'st_{k}': cnt.get(k) for k in ('guard_chain_short', 'short_chain', 'no_candidate', 'segment_unknown', 'inconsistent_with_segment', 'border',
                                                      'near_horizontal', 'triangulation_invalid', 'guard_end', 'guard_width')})
        di = info.get('detect_info') or {}
        out.update(steger_threshold=di.get('steger_threshold'), steger_threshold_design=di.get('steger_threshold_design'), sigma_hat=di.get('noise_sigma_hat'),
                   steger_sigma_px=di.get('steger_sigma_px'), coef_eff=di.get('steger_noise_coef_effective'), n_ridge_px=di.get('n_ridge_px'))
    else:
        ev = K.sub(A, f'{method}__ev__')
        exp_in = np.asarray(ev['exp_in_image'], bool); rec_ = np.asarray(ev['exp_recovered'], bool)
        code = np.asarray(ev['cand_code']); cr = info.get('component_reasons') or {}; sc = info.get('selection_codes') or {}
        cuv = np.asarray(ev['cand_uv'], np.float64).reshape(-1, 2); euv = np.asarray(ev['exp_uv'], np.float64).reshape(-1, 2)[exp_in]

        shared = 0; near_d = np.full(len(euv), np.nan)
        if len(cuv) and len(euv):
            dd = np.hypot(euv[:, None, 0] - cuv[None, :, 0], euv[:, None, 1] - cuv[None, :, 1]); j = dd.argmin(axis=1); near_d = dd[np.arange(len(euv)), j]
            u, cts = np.unique(j, return_counts=True); multi = set(u[cts >= 2].tolist())
            shared = int(sum(1 for x in j if x in multi))
        area = np.asarray(ev['cand_area'], np.float64); el = np.asarray(ev['cand_elong'], np.float64)
        out.update(expected=int(exp_in.sum()), recovered_id=int(rec_.sum()), n_components=int(len(code)),
                   **{f'comp_{k}': v for k, v in cr.items()}, **{f'sel_{k}': v for k, v in sc.items()},
                   accepted_features=int(np.asarray(ev['feats']).reshape(-1, 2).shape[0]), points=int(np.asarray(ev['pts']).reshape(-1, 3).shape[0]),
                   exp_sharing_nearest_component=shared, exp_nearest_comp_dist_px_median=(float(np.nanmedian(near_d)) if len(near_d) else None),
                   cand_area_median=(float(np.median(area)) if len(area) else None), cand_area_p90=(float(np.percentile(area, 90)) if len(area) else None),
                   cand_elong_median=(float(np.median(el[np.isfinite(el)])) if np.isfinite(el).any() else None), n_matched=int(np.asarray(ev['feat_matched'], bool).sum()),
                   n_ambiguous=int(np.asarray(ev['feat_ambiguous'], bool).sum()), n_duplicate=int(np.asarray(ev['feat_duplicate'], bool).sum()),
                   area_floor_px2=(r.get('decode_meta') or {}).get('a_floor_px2'), blob_area_median_px=(info.get('det_info') or {}).get('blob_area_median_px'),
                   n_elong_rejected=(info.get('det_info') or {}).get('n_elong_rejected'), n_area_rejected=(info.get('det_info') or {}).get('n_area_rejected'))
    return out


def valley_ratios() -> list:
    """camera-side gap: on the stored sigma-0 images (M10, block 0) the minimum grey value on the straight segment between two expected
    neighbouring dot images (image distance < 1.5 x median nearest-neighbour distance: for 8 x 4 only the cross-epipolar neighbours, for
    8 x 6 both directions) / the mean of the two dot peaks.
    Noise-free single examples only (sigma 0 exists only on M10 block 0); describes the signal distribution, not an algorithmic decision."""
    out = []
    for cfg in K.DOTS5:
        for env in ('NOMINAL', 'BLUR2', 'BLUR3'):
            ck = f'{cfg}|M10_curv|b0|s0|r0|{env}'
            meta, A = K.read_pack(ck, prefixes=('image', 'DF__ev__exp_uv', 'DF__ev__exp_in_image'))
            img = np.asarray(A['image'], np.float64); uv = np.asarray(A['DF__ev__exp_uv'], np.float64).reshape(-1, 2)[np.asarray(A['DF__ev__exp_in_image'], bool)]
            if len(uv) < 2:
                continue
            dd = np.hypot(uv[:, None, 0] - uv[None, :, 0], uv[:, None, 1] - uv[None, :, 1]); np.fill_diagonal(dd, np.inf)
            nn = dd.min(axis=1); lim = 1.5 * float(np.median(nn)); vals = []
            H, W = img.shape
            for i in range(len(uv)):
                for j in np.flatnonzero(dd[i] < lim):
                    if j <= i:
                        continue
                    n = int(max(8, np.ceil(dd[i, j] * 2)))
                    t = np.linspace(0, 1, n); xs = uv[i, 0] + t * (uv[j, 0] - uv[i, 0]); ys = uv[i, 1] + t * (uv[j, 1] - uv[i, 1])
                    xi = np.clip(np.round(xs).astype(int), 0, W - 1); yi = np.clip(np.round(ys).astype(int), 0, H - 1)
                    prof = img[yi, xi]; pk = 0.5 * (prof[:3].max() + prof[-3:].max())
                    if pk > 0:
                        vals.append(float(prof[n // 4: 3 * n // 4 + 1].min() / pk))
            v = np.asarray(vals)
            out.append(dict(config=cfg, env=env, case_key=ck, n_pairs=int(len(v)), valley_ratio_median=(float(np.median(v)) if len(v) else None),
                            valley_ratio_p90=(float(np.percentile(v, 90)) if len(v) else None), frac_pairs_valley_gt_0p5=(float((v > 0.5).mean()) if len(v) else None),
                            nn_dist_px_median=float(np.median(nn))))
    return out


def stage_a2(workers: int = 16) -> dict:
    K.rules_registered()
    R = results(); Mn = main_rows(R)
    jobs = list(zip(Mn.case_key, Mn.method, Mn.family)); t0 = time.time(); rows = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_a2_one, jobs, chunksize=8)):
            rows.append(r)
    F = pd.DataFrame(rows).merge(Mn[['case_key', 'family', 'config', 'surface', 'phase_block', 'sigma', 'rep', 'env', 'psf_true_px', 'subset', 'state', 'glob_Ctau', 'glob_Cvalid',
                                     'loc_Ctau', 'loc_Cvalid', 'c', 'w', 'u', 'w_share', 'u_share', 'n_accepted', 'glob_pt_rmse']], on='case_key', how='left', validate='one_to_one')
    for dom in ('glob', 'loc'):
        F[f'{dom}_no_support'] = 1.0 - F[f'{dom}_Cvalid']; F[f'{dom}_supported_over_tau'] = F[f'{dom}_Cvalid'] - F[f'{dom}_Ctau']
        s = F[f'{dom}_no_support'] + F[f'{dom}_supported_over_tau'] + F[f'{dom}_Ctau']
        if (np.abs(s - 1.0) > 1e-12).any():
            raise K.Stop('integrity', f'A2 surface layer of {dom} does not close')
    K.save_csv(F, K.OUT / '_work' / 'a2_features.csv')
    V = pd.DataFrame(valley_ratios()); K.save_csv(V, K.OUT / '_work' / 'a2_valley_sigma0.csv')
    rec = dict(t=K.now(), n=len(F), seconds=round(time.time() - t0, 1), surface_layer_closes=True, valley_rows=len(V),
               merging_sign_definition='expected visible dots whose nearest candidate component is shared with another expected dot (image-side nearest assignment)',
               valley_definition=valley_ratios.__doc__.strip())
    K.protocol_put('A2', rec)
    return rec


def a3_cases(M: pd.DataFrame) -> pd.DataFrame:
    c = M[M.config.isin(K.LINES8) & M.env.isin([e for e, _ in K.PSF_ENVS])].copy()
    n_noisy = int((c.sigma > 0).sum()); n0 = int((c.sigma == 0).sum())
    if n_noisy != 384 or n0 != 8 or c.case_key.duplicated().any():
        raise K.Stop('identity', f'A3 cases: {n_noisy} noisy (384) / {n0} sigma 0 (8)')
    return c


# Measure line-response margins and dot-profile gaps from stored image cases.
def stage_a3() -> dict:
    import m6_platform as PL
    rules = K.rules_registered()
    AT = pd.DataFrame(rules['arm_table'])
    M = manifest(); C = a3_cases(M); L = K.Ledger()
    store = K.OUT / 'data' / 'a3'
    rows = []; t0 = time.time(); n_dec = 0
    for i, r in enumerate(C.to_dict('records')):
        ck = r['case_key']; at = AT[(AT.config == r['config']) & (AT.env == r['env'])].set_index('arm')
        pf = store / f"{ck.replace('|', '__')}.npz"
        done = {}
        if pf.exists():
            with np.load(pf, allow_pickle=False) as z:
                done = json.loads(str(z['_meta'])).get('arms', {})
        pr = PL.prep(r)
        meta, A = K.read_pack(ck)
        old = meta['results']['P4']; img = A['image']
        inp = PL.platform()['q'].inputs(pr['decoder_row']); w_px = float(inp['meta_full']['olines']['width_px']); peak = float(inp['peak'])
        arrays = {}; arms = dict(done)
        if pf.exists():
            with np.load(pf, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files if k != '_meta'}
        di0 = (old.get('info') or {}).get('detect_info') or {}
        for arm in ('N', 'T', 'S', 'ST'):
            a = at.loc[arm]; s = float(a.steger_sigma_px); pdz = float(a.psf_design_px)
            if arm in arms:
                continue
            if arm == 'N':
                rec = dict(source='MODEL-PREP-05 stored result (reused, no decode)', metrics=old['metrics'], full_identity=old['full_identity'], state=old['state'],
                           thr=dict(steger_sigma_px=di0.get('steger_sigma_px'), psf_sigma_px=1.0, thr_design=di0.get('steger_threshold_design'), thr_final=di0.get('steger_threshold'),
                                    sigma_hat=di0.get('noise_sigma_hat'), coef=di0.get('steger_noise_coef_effective')))
            elif a.alias_of:
                rec = dict(source=f'alias of {a.alias_of} (identical setting {a.setting_key[:12]})', alias_of=a.alias_of)
            else:
                rid = L.reserve('decode_arm', f'{ck}|{arm}', K.CAP_A)
                try:
                    raw, dmeta, thr = PL.p4_decode(img, pr, steger_sigma_px=s, psf_sigma_px=pdz)
                    ev, er, fi = PL.evaluate(pr, 'P4', raw, dmeta)
                    L.end(rid, 'ok'); n_dec += 1
                except BaseException as e:
                    L.end(rid, 'failed', error=repr(e)[:300]); raise
                if abs(thr['steger_sigma_px'] - s) > 0 or abs(thr['psf_sigma_px'] - pdz) > 0 or abs(thr['coef'] - PL.coef_at(s)) > 1e-15:
                    raise K.Stop('identity', f'{ck} {arm}: effective settings {thr} differ from the arm table')
                arrays.update({f'{arm}__raw__{k}': v for k, v in raw.items()}); arrays.update({f'{arm}__ev__{k}': v for k, v in ev.items()})
                rec = dict(source='re-decode of the stored image', metrics=er['metrics'], full_identity=er['full_identity'], state=er['state'], thr=thr, decode_meta=dmeta,
                           raw_digest=K5.arrays_digest(raw), eval_digest=K5.arrays_digest(ev))
            arms[arm] = rec
        K.save_npz(pf, dict(case_key=ck, image_sha=meta['image']['content_sha'], arms=arms, note='image not copied: see MODEL-PREP-05 pack'), arrays)
        for arm in ('N', 'T', 'S', 'ST'):
            a = at.loc[arm]; rec = arms[arm]; src = arms[rec['alias_of']] if rec.get('alias_of') else rec
            m = src['metrics']; fi = ((src.get('full_identity') or {}).get('counts') or {}).get('all') or {}
            c, w, u = fi.get('c'), fi.get('w'), fi.get('u'); th = src.get('thr') or {}
            s = float(a.steger_sigma_px); pdz = float(a.psf_design_px)
            r_true = PL.ridge(w_px, s, r['psf_true_px'], 255.0 * peak)
            rows.append(dict(case_key=ck, arm=arm, alias_of=(rec.get('alias_of') or ''), config=r['config'], surface=r['surface'], phase_block=int(r['phase_block']), sigma=float(r['sigma']),
                             rep=int(r['rep']), env=r['env'], psf_true_px=float(r['psf_true_px']), steger_sigma_px=s, psf_design_px=pdz, state=src.get('state'),
                             glob_Ctau=m.get('glob_Ctau'), loc_Ctau=m.get('loc_Ctau'), glob_Cvalid=m.get('glob_Cvalid'), loc_Cvalid=m.get('loc_Cvalid'), glob_pt_rmse=m.get('glob_pt_rmse'),
                             glob_pt_p95=m.get('glob_pt_p95'), glob_surf_rmse=m.get('glob_surf_rmse'), n_valid=m.get('n_valid'), c=c, w=w, u=u,
                             w_share=((w / (c + w)) if (c is not None and c + w > 0) else None), u_share=((u / (c + w + u)) if (c is not None and c + w + u > 0) else None),
                             thr_design=th.get('thr_design'), thr_final=th.get('thr_final'), sigma_hat=th.get('sigma_hat'), coef=th.get('coef'),
                             thr_noise=((th.get('coef') or 0.0) * (th.get('sigma_hat') or 0.0) if (th.get('sigma_hat') or 0) > 0 else 0.0),
                             ridge_true=r_true, margin=(r_true / th['thr_final'] if th.get('thr_final') else None)))
        if (i + 1) % 40 == 0:
            print(f'A3 {i + 1}/{len(C)} decodes {n_dec} {round(time.time() - t0)} s', flush=True)
    T = pd.DataFrame(rows)
    T['dominant'] = np.where(T.thr_design >= T.thr_noise, 'design', 'noise')
    K.save_csv(T, K.OUT / '_work' / 'a3_arms.csv')
    rec = dict(t=K.now(), images=len(C), decodes_this_run=n_dec, ledger_decode_A=L.count('regress_P4', 'decode_arm'), seconds=round(time.time() - t0, 1))
    K.protocol_put('A3', rec)
    return rec
