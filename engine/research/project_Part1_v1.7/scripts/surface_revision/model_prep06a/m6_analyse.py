# Summarize precision, support and detector-response diagnostics from stored data.

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

import m6_common as K
import m6_stats as ST

W = K.OUT / '_work'
FIG = K.OUT / 'figures'
ENV_DE = {'NOMINAL': 'PSF 1', 'BLUR1_5': 'PSF 1,5', 'BLUR2': 'PSF 2', 'BLUR3': 'PSF 3'}
PSF_OF = dict(K.PSF_ENVS)


def _cell(df: pd.DataFrame, keys: list, value: str) -> list:
    out = []
    for k, g in df.groupby(keys):
        s = ST.blocks(g[['phase_block', 'rep', value]].rename(columns={value: 'value'}))
        out.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), **ST.flat('', s)))
    return out


def _over_surfaces(E: pd.DataFrame, keys: list) -> pd.DataFrame:
    rows = []
    for k, g in E.groupby(keys):
        v = g['mean'].dropna()
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), surface='alle', mean=(float(v.mean()) if len(v) == 3 else None), min=(float(v.min()) if len(v) else None),
                         max=(float(v.max()) if len(v) else None), n_surfaces_def=int(len(v))))
    return pd.DataFrame(rows)


def a1_effects() -> pd.DataFrame:
    T = pd.read_csv(W / 'a1_long.csv', low_memory=False)
    out = []
    L = T[T.config.isin(K.LINES8) & T.env.isin(list(PSF_OF)) & (T.sigma > 0)]
    keys = ['surface', 'phase_block', 'sigma', 'rep', 'env', 'domain', 'd_mm', 'tau_mm']
    for metric in ('Ctau', 'Cvalid'):
        a = L[L.config == 'L_w1.5_s8'][keys + [metric]]; b = L[L.config == 'L_w0.5_s8'][keys + [metric]]
        P = ST.pair(a, b, keys, [metric], suffixes=('_w', '_t'))
        P['value'] = P[f'{metric}_w'] - P[f'{metric}_t']
        for r in _cell(P, ['surface', 'sigma', 'env', 'domain', 'd_mm', 'tau_mm'], 'value'):
            out.append(dict(analysis='A1', contrast='Linienbreite 1,5 - 0,5 mm (8 mm)', metric=metric, **r))
    D = T[(T.family == 'point') & (T.subset == 'core') & T.env.isin(['NOMINAL', 'BLUR2', 'BLUR3']) & (T.sigma > 0) & (T.domain == 'glob')]
    pairs = [('Groesse 3 - 2 mm (8x4)', 'D_r1.5_p8x4', 'D_r1_p8x4'), ('Groesse 3 - 2 mm (8x6)', 'D_r1.5_p8x6', 'D_r1_p8x6'),
             ('Abstand 8x6 - 8x4 (2 mm)', 'D_r1_p8x6', 'D_r1_p8x4'), ('Abstand 8x6 - 8x4 (3 mm)', 'D_r1.5_p8x6', 'D_r1.5_p8x4'),
             ('Konfiguration 3 mm 8x4 - 2 mm 8x6', 'D_r1.5_p8x4', 'D_r1_p8x6')]
    for name, ca, cb in pairs:
        a = D[D.config == ca][keys + ['Ctau']]; b = D[D.config == cb][keys + ['Ctau']]
        P = ST.pair(a, b, keys, ['Ctau'], suffixes=('_a', '_b')); P['value'] = P.Ctau_a - P.Ctau_b
        for r in _cell(P, ['surface', 'sigma', 'env', 'domain', 'd_mm', 'tau_mm'], 'value'):
            out.append(dict(analysis='A1', contrast=name, metric='Ctau', **r))
    E = pd.DataFrame(out)
    S = _over_surfaces(E, ['analysis', 'contrast', 'metric', 'sigma', 'env', 'domain', 'd_mm', 'tau_mm'])
    return pd.concat([E, S], ignore_index=True)


def a1_sigma0() -> pd.DataFrame:
    T = pd.read_csv(W / 'a1_long.csv', low_memory=False)
    L = T[T.config.isin(K.LINES8) & (T.sigma == 0) & (T.domain == 'glob')]
    p = L.pivot_table(index=['env', 'd_mm', 'tau_mm'], columns='config', values='Ctau')
    p['delta_w'] = p['L_w1.5_s8'] - p['L_w0.5_s8']
    return p.reset_index()


def a3_effects() -> pd.DataFrame:
    T = pd.read_csv(W / 'a3_arms.csv', low_memory=False)
    T = T[T.sigma > 0]
    keys = ['config', 'surface', 'phase_block', 'sigma', 'rep', 'env']
    out = []
    piv = {a: T[T.arm == a].set_index(keys) for a in ('N', 'T', 'S', 'ST')}
    for metric in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'u_share', 'w', 'margin'):
        for name, f in (('T-N', lambda d: d['T'] - d['N']), ('S-N', lambda d: d['S'] - d['N']), ('ST-N', lambda d: d['ST'] - d['N']),
                        ('ST-S-T+N', lambda d: d['ST'] - d['S'] - d['T'] + d['N'])):
            d = {a: piv[a][metric].astype(float) for a in piv}
            v = f(d).rename('value').reset_index()
            for r in _cell(v, ['config', 'surface', 'sigma', 'env'], 'value'):
                out.append(dict(analysis='A3', contrast=name, metric=metric, **r))
        for a in ('N', 'ST'):
            x = piv[a][metric].astype(float).rename('value').reset_index()
            w = x[x.config == 'L_w1.5_s8'].drop(columns='config'); t = x[x.config == 'L_w0.5_s8'].drop(columns='config')
            kk = ['surface', 'phase_block', 'sigma', 'rep', 'env']
            P = ST.pair(w, t, kk, ['value'], suffixes=('_w', '_t')); P['value'] = P.value_w - P.value_t
            for r in _cell(P, ['surface', 'sigma', 'env'], 'value'):
                out.append(dict(analysis='A3', contrast=f'Breite 1,5 - 0,5 mm unter Arm {a}', metric=metric, config='L_w1.5_s8 - L_w0.5_s8', **r))
    E = pd.DataFrame(out)
    S = _over_surfaces(E, ['analysis', 'contrast', 'metric', 'config', 'sigma', 'env'])
    return pd.concat([E, S], ignore_index=True)


def a3_levels() -> pd.DataFrame:
    T = pd.read_csv(W / 'a3_arms.csv', low_memory=False)
    out = []
    for (cfg, env, sg, arm), g in T[T.sigma > 0].groupby(['config', 'env', 'sigma', 'arm']):
        rec = dict(config=cfg, env=env, sigma=sg, arm=arm, alias_of=';'.join(sorted(set(g.alias_of.fillna('')))), noise_dominant_share=float((g.dominant == 'noise').mean()),
                   margin_mean=float(g.margin.mean()), steger_sigma_px=float(g.steger_sigma_px.iloc[0]), psf_design_px=float(g.psf_design_px.iloc[0]),
                   no_output=int((g.state == 'no_output').sum()), w_sum=int(g.w.fillna(0).sum()), w_images=int((g.w.fillna(0) > 0).sum()), n_img=len(g))
        for m in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'u_share'):
            sv = []
            for s, gs in g.groupby('surface'):
                sv.append(ST.blocks(gs[['phase_block', 'rep', m]].rename(columns={m: 'value'}))['mean'])
            v = [x for x in sv if x is not None]
            rec[m] = (sum(v) / len(v)) if len(v) == 3 else None; rec[f'{m}_surf_min'] = min(v) if v else None; rec[f'{m}_surf_max'] = max(v) if v else None
        out.append(rec)
    return pd.DataFrame(out)


def b_effects() -> tuple[pd.DataFrame, dict]:
    B = pd.read_csv(W / 'b_pairs.csv', low_memory=False)
    B['d_glob_Ctau'] = B.glob_Ctau_3 - B.glob_Ctau_1
    rows = []
    for metric in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'sup_surf_rmse'):
        d = B[f'{metric}_3'] - B[f'{metric}_1']
        rel = d / B[f'{metric}_1'].where(B[f'{metric}_1'].abs() > 0)
        for (cfg, env, sg), g in B.assign(d=d, rel=rel).groupby(['config', 'env', 'sigma']):
            v = g.d.dropna()
            rows.append(dict(analysis='B', contrast='3x3 - 1x1', metric=metric, config=cfg, env=env, sigma=sg, surface='alle (B/M10/M24 x Bloecke 0, 2)', n_pairs_def=len(v),
                             mean=(float(v.mean()) if len(v) else None), min=(float(v.min()) if len(v) else None), max=(float(v.max()) if len(v) else None),
                             level_1x1=float(g[f'{metric}_1'].mean()) if g[f'{metric}_1'].notna().any() else None,
                             rel_mean=(float(g.rel.dropna().mean()) if g.rel.notna().any() else None)))

    w = B.pivot_table(index=['env', 'sigma', 'surface', 'phase_block'], columns='config', values=['glob_Ctau_1', 'glob_Ctau_3'])
    dw = pd.DataFrame(dict(dw1=w[('glob_Ctau_1', 'L_w1.5_s8')] - w[('glob_Ctau_1', 'L_w0.5_s8')], dw3=w[('glob_Ctau_3', 'L_w1.5_s8')] - w[('glob_Ctau_3', 'L_w0.5_s8')])).reset_index()
    flags = []
    for lvl, keys in (('Zelle (Umgebung x sigma, 3 Flaechen x 2 Bloecke)', ['env', 'sigma']), ('Flaeche (2 Bloecke)', ['env', 'sigma', 'surface'])):
        for k, g in dw.groupby(keys):
            a, b = float(g.dw1.mean()), float(g.dw3.mean())
            band = abs(a) >= 0.02 and (abs(b) < 0.02 or np.sign(a) != np.sign(b))
            chg = abs(b - a) >= 0.05
            flags.append(dict(level=lvl, **dict(zip(keys, k)), dw_1x1=a, dw_3x3=b, change=b - a, into_tie_band_or_sign=bool(band), change_ge_5pp=bool(chg)))
            rows.append(dict(analysis='B', contrast='Breite 1,5 - 0,5 mm: 1x1 vs 3x3', metric='glob_Ctau', config='L_w1.5_s8 - L_w0.5_s8', env=k[0], sigma=k[1],
                             surface=(k[2] if len(k) > 2 else 'alle (B/M10/M24 x Bloecke 0, 2)'), mean=b - a, level_1x1=a, level_3x3=b))
    F = pd.DataFrame(flags)
    new_w = B[(B.w_3.fillna(0) > 0) & (B.w_1.fillna(0) == 0)]
    warn = dict(cell_level=bool((F[F.level.str.startswith('Zelle')][['into_tie_band_or_sign', 'change_ge_5pp']].any(axis=1)).any()),
                surface_level=bool((F[F.level.str.startswith('Flaeche')][['into_tie_band_or_sign', 'change_ge_5pp']].any(axis=1)).any()),
                new_wrong_ids=int(len(new_w)), flagged=F[F[['into_tie_band_or_sign', 'change_ge_5pp']].any(axis=1)].to_dict('records'),
                state_changes=B.groupby(['state_1', 'state_3']).size().reset_index().values.tolist())
    K.save_csv(F, W / 'b_warning_rule.csv')
    return pd.DataFrame(rows), warn


def b_forward_checks() -> dict:

    import glob
    out = []
    for pf in sorted((K.OUT / 'data' / 'b').glob('*.npz')):
        with np.load(pf, allow_pickle=False) as z:
            meta = json.loads(str(z['_meta']))
        for tag, im in meta['images'].items():
            pm, _ = K.read_pack(im['case_key_parent'], prefixes=('image',))
            d1 = pm['image']['diag']; d3 = im['diag']
            out.append(dict(case_key=im['case_key_parent'], rays_3=d3.get('rays_per_pixel'), n_rays_ratio=(d3['n_rays_cast'] / d1['n_rays_cast']) if d1.get('n_rays_cast') else None,
                            noise_sha_equal=d3.get('noise_sha') == d1.get('noise_sha'), pre_psf_sum_ratio=(d3['irr_pre_psf_sum'] / d1['irr_pre_psf_sum']) if d1.get('irr_pre_psf_sum') else None,
                            sat_3=d3.get('frac_pre_clip_above_1')))
    X = pd.DataFrame(out)
    return dict(n=len(X), rays_all_3=bool((X.rays_3 == 3).all()), noise_identical_all=bool(X.noise_sha_equal.all()), n_rays_ratio=[float(X.n_rays_ratio.min()), float(X.n_rays_ratio.max())],
                energy_ratio=[float(X.pre_psf_sum_ratio.min()), float(X.pre_psf_sum_ratio.max())], sat_max=float(X.sat_3.max()))


def a2_summary() -> pd.DataFrame:
    F = pd.read_csv(W / 'a2_features.csv', low_memory=False)
    F = F[(F.sigma > 0) & ((F.subset == 'core') | (F.subset == 'transition'))]
    cols_l = ['expected', 'detected', 'accepted', 'recovered_id', 'st_guard_chain_short', 'st_short_chain', 'st_no_candidate', 'st_segment_unknown', 'st_inconsistent_with_segment',
              'st_border', 'st_near_horizontal', 'st_triangulation_invalid', 'c', 'w', 'u']
    cols_p = ['expected', 'n_components', 'comp_1_abs_area_small', 'comp_2_abs_area_large', 'comp_3_border', 'comp_4_intensity', 'comp_5_m00', 'sel_0_accepted', 'sel_1_rel_area_small',
              'sel_2_rel_area_large', 'sel_3_elongation', 'sel_4_group_unavailable', 'accepted_features', 'n_matched', 'n_ambiguous', 'n_duplicate', 'points', 'recovered_id',
              'exp_sharing_nearest_component', 'c', 'w', 'u']
    out = []
    for (fam, cfg, env, sg), g in F.groupby(['family', 'config', 'env', 'sigma']):
        cols = cols_l if fam == 'line' else cols_p
        rec = dict(family=fam, config=cfg, env=env, sigma=sg, n_img=len(g), no_output=int((g.state == 'no_output').sum()))
        for c in cols:
            if c in g:
                rec[c] = float(pd.to_numeric(g[c], errors='coerce').mean())
        for dom in ('glob', 'loc'):
            for c in (f'{dom}_no_support', f'{dom}_supported_over_tau', f'{dom}_Ctau'):
                rec[c] = float(g[c].mean())
        out.append(rec)
    return pd.DataFrame(out)


def screening_mp5() -> tuple[pd.DataFrame, pd.DataFrame]:
    sel = K.rules_registered()['selection']['feasible']
    R = pd.read_csv(K.SRC5 / 'results.csv', float_precision='round_trip', low_memory=False)
    R = R[(((R.family == 'point') & (R.method == 'DF')) | ((R.family == 'line') & (R.method == 'P4'))) & (R.sigma > 0)]
    rows = []
    for (fam, cfg, env, sg, surf), g in R.groupby(['family', 'config', 'env', 'sigma', 'surface']):
        lv = {m: ST.blocks(g[['phase_block', 'rep', m]].rename(columns={m: 'value'}))['mean'] for m in ('glob_Ctau', 'loc_Ctau')}
        rmse_ok = bool(g.glob_pt_rmse.notna().all() and (g.glob_pt_rmse <= sel['per_image_global_pt_rmse_max_mm']).all())
        w_ok = bool((g.w.fillna(1) <= sel['per_image_w_max']).all())
        u_ok = bool(((g.c.fillna(0) + g.w.fillna(0)) >= sel['per_image_c_plus_w_min']).all() and g.u_share.notna().all() and (g.u_share <= sel['per_image_u_share_max']).all())
        ct_ok = bool(lv['glob_Ctau'] is not None and lv['loc_Ctau'] is not None and lv['glob_Ctau'] >= sel['ctau_global_min'] and lv['loc_Ctau'] >= sel['ctau_local_min'])
        rows.append(dict(family=fam, config=cfg, env=env, sigma=sg, surface=surf, glob_Ctau=lv['glob_Ctau'], loc_Ctau=lv['loc_Ctau'], ctau_ok=ct_ok, rmse_ok=rmse_ok, w_ok=w_ok, u_ok=u_ok,
                         max_img_rmse=(float(g.glob_pt_rmse.max()) if g.glob_pt_rmse.notna().any() else None), feasible=bool(ct_ok and rmse_ok and w_ok and u_ok),
                         n_img=len(g), n_no_output=int((g.state == 'no_output').sum())))
    S = pd.DataFrame(rows)
    pick = []
    for (fam, env, sg), g in S.groupby(['family', 'env', 'sigma']):
        ok = g.groupby('config').feasible.all()
        cand = ok[ok].index.tolist()
        if not cand:
            best = g.groupby('config').glob_Ctau.min().sort_values(ascending=False)
            pick.append(dict(family=fam, env=env, sigma=sg, n_configs=len(ok), qualified=[], preferred=None, note='kein qualifizierter Kandidat',
                             best_failing=best.index[0] if len(best) else None, best_failing_min_Ctau=float(best.iloc[0]) if len(best) else None))
            continue
        sc = g[g.config.isin(cand)].groupby('config').glob_Ctau.min().sort_values(ascending=False)
        top = sc.iloc[0]; tie = sc[sc >= top - 0.02].index.tolist()
        ref = 'D_r1_p8x6' if fam == 'point' else 'L_w0.5_s8'
        if ref in tie:
            pref = ref; why = 'Referenz im Gleichstandsband'
        elif len(tie) == 1:
            pref = tie[0]; why = 'einziger Kandidat im Gleichstandsband'
        else:
            wl = g[g.config.isin(tie)].groupby('config').loc_Ctau.min().sort_values(ascending=False)
            pref = wl.index[0]; why = 'schlechteste lokale C_tau' if len(wl) < 2 or wl.iloc[0] != wl.iloc[1] else 'weitere Regeln'
        pick.append(dict(family=fam, env=env, sigma=sg, n_configs=len(ok), qualified=cand, tie_band=tie, preferred=pref, score_min_Ctau=float(sc[pref]), why=why))
    return S, pd.DataFrame(pick)


def stage_analyse() -> dict:
    E1 = a1_effects(); E3 = a3_effects(); Eb, warn = b_effects(); A2 = a2_summary(); S0 = a1_sigma0()
    L3 = a3_levels(); fwd = b_forward_checks(); Ssc, Spk = screening_mp5()
    A4b = pd.read_csv(W / 'a4_bridge.csv'); A4s = pd.read_csv(W / 'a4_bands.csv')
    A4e = pd.concat([A4b.assign(analysis='A4', contrast='Bruecke MODEL-PREP-05 vs Abschnitt 3.5 (S5-ALIGN-05), aggregiert'),
                     A4s.assign(analysis='A4', contrast='Baender: Gradient - gleichmaessig (gleiche Positionen)')], ignore_index=True)
    E = pd.concat([E1, E3, Eb, A4e], ignore_index=True)
    K.save_csv(E, K.OUT / 'effects.csv')

    T1 = pd.read_csv(W / 'a1_long.csv', low_memory=False).assign(source='A1 Neubewertung gespeicherter Punktwolken', setting='Hauptmethode')
    T3 = pd.read_csv(W / 'a3_arms.csv', low_memory=False).assign(source='A3 Vier-Arm-Diagnose', domain='glob/loc', d_mm=6.0, tau_mm=0.15)
    T3 = T3.rename(columns={'arm': 'setting'})
    Bp = pd.read_csv(W / 'b_pairs.csv', low_memory=False)
    b1 = Bp[[c for c in Bp.columns if not c.endswith('_3')]].rename(columns=lambda c: c[:-2] if c.endswith('_1') else c).assign(setting='1x1 (MODEL-PREP-05)')
    b3 = Bp[[c for c in Bp.columns if not c.endswith('_1')]].rename(columns=lambda c: c[:-2] if c.endswith('_3') else c).assign(setting='3x3')
    TB = pd.concat([b1, b3], ignore_index=True).assign(source='B Teilstrahlpruefung', domain='glob/loc', d_mm=6.0, tau_mm=0.15)
    Mt = pd.concat([T1, T3, TB], ignore_index=True)
    K.save_csv(Mt, K.OUT / 'metrics.csv')
    for name, df in (('a2_summary', A2), ('a1_sigma0', S0), ('a3_levels', L3), ('screening_mp5', Ssc), ('screening_mp5_pick', Spk)):
        K.save_csv(df, W / f'{name}.csv')
    rec = dict(t=K.now(), effects_rows=len(E), metrics_rows=len(Mt), b_warning=warn, b_forward=fwd,
               screening_note='descriptive application of the registered screening rule to MODEL-PREP-05 conditions; not the 06C selection; no holdout slot filled')
    K.protocol_put('analysis', rec)
    return dict(effects=len(E), metrics=len(Mt), b_warning={k: v for k, v in warn.items() if k != 'flagged'}, n_flagged=len(warn['flagged']), b_forward=fwd)


def stage_figs() -> dict:
    import m6_figs as FG
    FIG.mkdir(parents=True, exist_ok=True)
    return FG.all_figures()


def stage_final() -> dict:
    import m6_stage_a as A
    p = K.protocol(); b = p['binding']
    code_now = A.code_hashes(); drift = [f for f, h in b['code'].items() if code_now.get(f) != h]
    inp_now = {f: K.sha_file(K.ROOT / f) for f in K.INPUT_FILES}; inp_drift = [f for f, h in b['inputs'].items() if inp_now.get(f) != h]
    chk = A.mp5_check(); L = K.Ledger()
    du = shutil.disk_usage(str(K.ROOT))
    files = {}
    for q in sorted(K.OUT.rglob('*')):
        if q.is_file() and '_sandbox' not in q.parts and q.name != 'protocol.json':
            files[K.rel(q)] = K.sha_file(q)
    own = {K.rel(q): K.sha_file(q) for q in sorted(K.HERE.glob('*.py'))}
    tot = sum(q.stat().st_size for q in K.OUT.rglob('*') if q.is_file())
    sandbox_files = [K.rel(q) for q in (K.OUT / '_sandbox').rglob('*') if q.is_file()] if (K.OUT / '_sandbox').exists() else []
    rec = dict(t=K.now(), code_drift=drift, input_drift=inp_drift, mp5_check=chk, ledger=dict(regress_P4=L.count('regress_P4'), decode_arm=L.count('decode_arm'),
               render_compat=L.count('render_compat'), render_3x3=L.count('render_3x3'), decode_3x3=L.count('decode_3x3'), decode_total=L.count('regress_P4', 'decode_arm', 'decode_3x3')),
               caps=K.CAPS, disk_free_gb=round(du.free / 1e9, 2), output_bytes=tot, outputs=files, scripts=own, sandbox_files=sandbox_files,
               ok=bool(not drift and not inp_drift and not chk['outputs_changed'] and not chk['protected_changed'] and not chk['freeze_problems']))
    K.protocol_put('final', rec)
    return {k: v for k, v in rec.items() if k not in ('outputs', 'scripts')}
