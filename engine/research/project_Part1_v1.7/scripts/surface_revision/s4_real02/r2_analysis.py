# Analyze layout-strategy effects and phase-block variation.

from __future__ import annotations

import numpy as np
import pandas as pd

import r2_common as K

GM = ('glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'glob_n_interp', 'glob_n_nodes', 'glob_n_valid', 'glob_pt_rmse', 'glob_pt_mae', 'glob_pt_p95', 'glob_surf_rmse', 'glob_surf_p95')
LM = ('loc_Ctau', 'loc_Cvalid', 'loc_cov_interp', 'loc_n_interp', 'loc_n_nodes', 'loc_n_valid', 'loc_pt_rmse', 'loc_pt_mae', 'loc_pt_p95', 'loc_surf_rmse', 'loc_surf_p95')
FM = ('state', 'n_out', 'n_accepted', 'n_valid', 'n_detected', 'n_selected', 'n_matched', 'n_ambiguous', 'n_duplicate', 'n_no_correspondence', 'n_candidates', 'n_confirmable_features',
      'id_correct', 'id_wrong', 'id_unconf', 'img_n_expected', 'img_n_recovered', 'img_recovery')
CONTRASTS = (('S1', 'S0'), ('S2', 'S1'), ('S3', 'S1'), ('S4', 'S3'), ('S4', 'S2'), ('S2', 'S0'), ('S3', 'S0'), ('S4', 'S0'))
PAIR_METRICS = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_pt_rmse', 'glob_pt_mae', 'glob_pt_p95', 'glob_surf_rmse', 'glob_surf_p95', 'glob_sup_surf_rmse', 'loc_pt_rmse',
                'loc_surf_rmse', 'loc_sup_surf_rmse', 'n_accepted', 'n_valid', 'glob_n_supported', 'wrong_rate', 'unconf_share', 'no_output', 'glob_Ctau_ostar')


def _ratio(a, b):
    return (float(a) / float(b)) if (b is not None and b > 0 and a is not None) else None


def case_rows(M: pd.DataFrame, reg: dict, eval_dir=None, require_complete: bool = True) -> pd.DataFrame:

    eval_dir = eval_dir or K.EVL
    cache = {}; rows = []; missing = []
    for r in M.to_dict('records'):
        ek = r['eval_key']
        if ek not in cache:
            p = eval_dir / f'{ek}.json'
            if not K.complete(p, ek):
                missing.append(ek); cache[ek] = None
            else:
                cache[ek] = K.load_json(p)
        e = cache[ek]
        lay = reg['baselines'][r['baseline']]['layouts'][r['layout']]
        base = dict(view_id=r['view_id'], baseline=r['baseline'], family=r['family'], surface=r['surface'], strategy=r['strategy'], block=r['block'], layout=r['layout'], layout_id=r['layout_id'],
                    phase=r['phase'], sigma=float(r['sigma']), noise_seed=r['noise_seed'], image_key=r['image_key'], decode_key=r['decode_key'], eval_key=ek, n_views_of_image=int(r['n_views_of_image']),
                    shared_image=bool(int(r['n_views_of_image']) > 1), entry_subset=bool(str(r['entry_subset']) in ('True', 'true', '1')), boundary=r['boundary'], domain_main='omega1',
                    g_min_mm=lay.get('g_min_mm'), g_max_mm=lay.get('g_max_mm'), h_mm=lay.get('h_mm'), delta_rms_mm=lay.get('delta_rms_mm'), delta_max_mm=lay.get('delta_max_mm'),
                    edge_clearance_min_mm=lay.get('edge_clearance_min_mm'), n_elements_ref=lay.get('n_elements_ref'), n_mask_candidates=r['n_mask_candidates'], n_raster_evaluated=r['n_raster_evaluated'],
                    lit_px=float(r['lit_px']), peak=float(r['peak']), integral_f32=float(r['integral_f32']))
        if e is None:
            rows.append(dict(base, record='missing')); continue
        m = e['metrics']; ms = e.get('metrics_omega_star') or {}; ex = e.get('extras') or {}; nd = ex.get('nodes_omega1') or {}
        rec = dict(base, record='complete', **{k: m.get(k) for k in GM + LM + FM}, **{f'{k}_ostar': ms.get(k) for k in ('glob_Ctau', 'glob_Cvalid', 'glob_surf_rmse', 'glob_pt_rmse', 'glob_n_nodes')},
                   glob_n_supported=(nd.get('glob') or {}).get('n_supported'), loc_n_supported=(nd.get('loc') or {}).get('n_supported'),
                   glob_sup_surf_rmse=(nd.get('glob') or {}).get('sup_surf_rmse'), loc_sup_surf_rmse=(nd.get('loc') or {}).get('sup_surf_rmse'),
                   n_status_accepted=m.get('n_status_accepted'), n_status_segment_unknown=m.get('n_status_segment_unknown'), n_status_no_candidate=m.get('n_status_no_candidate'))
        if r['family'] == 'point':
            si = ex.get('source_identity') or {}
            c, w, u = si.get('correct'), si.get('wrong'), si.get('undeterminable')
            rec.update(identity_main='source identity (sigma-0 labels)', id_main_correct=c, id_main_wrong=w, id_main_unconf=u, id_main_n=si.get('n_reliable'), src_ref_key=si.get('ref_key'))
        else:
            c, w, u = m.get('id_correct'), m.get('id_wrong'), m.get('id_unconf')
            rec.update(identity_main='table truth', id_main_correct=c, id_main_wrong=w, id_main_unconf=u, id_main_n=m.get('id_denominator'))
        rec['wrong_rate'] = _ratio(w, (c or 0) + (w or 0)) if c is not None else None
        rec['unconf_share'] = _ratio(u, (c or 0) + (w or 0) + (u or 0)) if c is not None else None
        rec['no_output'] = float(m.get('state') == 'no_output')
        rows.append(rec)
    if missing and require_complete:
        raise K.Stop('incomplete', f'{len(set(missing))} evaluation records missing / not content-verified: the batch is not complete')
    return pd.DataFrame(rows)


def _block_stat(d: pd.DataFrame, col: str) -> dict:

    x = d.dropna(subset=[col])
    if not len(x):
        return dict(main=None, mean_pairs=None, block_min=None, block_max=None, p10=None, p90=None, n_pairs_defined=0, n_blocks_defined=0)
    bm = x.groupby('block')[col].median()
    return dict(main=float(bm.median()), mean_pairs=float(x[col].mean()), block_min=float(bm.min()), block_max=float(bm.max()), p10=float(x[col].quantile(0.10)), p90=float(x[col].quantile(0.90)),
                n_pairs_defined=int(len(x)), n_blocks_defined=int(bm.size))


def paired(C: pd.DataFrame, partial: bool = False) -> pd.DataFrame:

    out = []
    C = C[C.record == 'complete'].copy()
    for (base, fam, s, sg), g in C.groupby(['baseline', 'family', 'surface', 'sigma']):
        s0 = g[g.strategy == 'S0'].set_index('phase')
        st = {k: g[g.strategy == k].set_index(['block', 'phase']) for k in ('S1', 'S2', 'S3', 'S4')}

        for k in ('S0', 'S1', 'S2', 'S3', 'S4'):
            d = g[g.strategy == k].copy()
            if k == 'S0':
                d['block'] = d['phase']
            for mtr in PAIR_METRICS:
                bsx = _block_stat(d, mtr)
                out.append(dict(table_type='absolute', baseline=base, family=fam, surface=s, sigma=sg, contrast=k, metric=mtr, n_views=len(d), n_no_output=int((d.state == 'no_output').sum()), **bsx))
        for a, b in CONTRASTS:
            A_ = st[a]; Bx = s0 if b == 'S0' else st[b]
            if partial and (not len(A_) or not len(Bx)):
                continue
            rows = []
            for (blk, ph), ra in A_.iterrows():
                key = ph if b == 'S0' else (blk, ph)
                if key not in Bx.index:
                    if partial:
                        continue
                    raise K.Stop('pairing', f'{base} {s} sigma {sg}: {a} {blk} {ph} has no {b} partner')
                rb = Bx.loc[key]
                rows.append(dict(block=blk, phase=ph, same_image=(ra.image_key == rb.image_key), **{m: (None if (pd.isna(ra[m]) or pd.isna(rb[m])) else float(ra[m]) - float(rb[m])) for m in PAIR_METRICS}))
            D = pd.DataFrame(rows)
            for mtr in PAIR_METRICS:
                bsx = _block_stat(D, mtr); ch = D[~D.same_image]
                out.append(dict(table_type='paired', baseline=base, family=fam, surface=s, sigma=sg, contrast=f'{a}-{b}', metric=mtr, n_pairs=len(D), n_same_image_pairs=int(D.same_image.sum()),
                                n_blocks_layout_changed=int(ch.block.nunique()), changed_only_main=_block_stat(ch, mtr)['main'], **bsx))
    return pd.DataFrame(out)


def dense_bridge(C: pd.DataFrame) -> pd.DataFrame:

    import s1r_eval as E
    import r2_eval as RE
    V = pd.read_csv(K.S345 / 'results' / 'S4-reduced_cases.csv', low_memory=False); V = V[V.strategy.astype(str).str.startswith('S0')]
    if len(V) != 48:
        raise K.Stop('input', f'old S0 clouds {len(V)} != 48')
    ds, d0 = RE.domain('omega_star'), RE.domain('omega0'); rows = []
    for r in V.itertuples():
        z = K.load_npz(K.S345 / 'run' / 'eval' / f'{r.decode_key}.npz'); sid = K.RK.SURF_MAP[r.surface]
        Pv = np.asarray(z['pts'], np.float64)[np.asarray(z['valid'], bool)]
        so, _ = E.global_surface(Pv, d0, d0[f'z_{sid}']); ss, _ = E.global_surface(Pv, ds, ds[f'z_{sid}'])
        if abs(so['Ctau'] - float(r.glob_Ctau)) > 1e-12:
            raise K.Stop('check', f'old cloud {r.decode_key}: Omega0 recomputation {so["Ctau"]} != stored {r.glob_Ctau}')
        rows.append(dict(family=r.family, surface=r.surface, sigma=float(r.sigma), phase=r.phase_label, state=r.state, glob_Ctau_ostar=ss['Ctau'], glob_Cvalid_ostar=ss['Cvalid'],
                         glob_surf_rmse_ostar=ss['surf']['rmse'], glob_Ctau_omega0=so['Ctau']))
    O = pd.DataFrame(rows); N = C[(C.record == 'complete') & (C.strategy == 'S0') & C.sigma.isin([0.0, 20.0])]
    out = []
    for (fam, s, sg), g in O.groupby(['family', 'surface', 'sigma']):
        n = N[(N.family == fam) & (N.surface == s) & (N.sigma == sg)]
        for mtr in ('glob_Ctau_ostar', 'glob_Cvalid_ostar', 'glob_surf_rmse_ostar'):
            for who, d, base in (('old', g, 'R5x4' if fam == 'point' else 'L4.45'), ('new', n, 'R8x4' if fam == 'point' else 'L6.3')):
                x = d[mtr].dropna()
                out.append(dict(table_type='dense_bridge_level', batch=who, baseline=base, family=fam, surface=s, sigma=sg, contrast='S0', metric=mtr, domain='omega_star',
                                boundary=('legacy' if who == 'old' else K.BOUNDARY), n_views=len(d), n_no_output=int((d.state == 'no_output').sum()), n_pairs_defined=int(len(x)),
                                main=(float(x.median()) if len(x) else None), mean_pairs=(float(x.mean()) if len(x) else None), block_min=(float(x.min()) if len(x) else None),
                                block_max=(float(x.max()) if len(x) else None)))
    D = pd.read_csv(K.S345 / 'results' / 'S4-reduced_block_differences.csv')
    for (fam, s, sg, st, mtr), g in D[D.metric.isin(['glob_Ctau', 'loc_Ctau', 'glob_surf_rmse', 'loc_surf_rmse', 'wrong_share', 'unconf_share'])].groupby(['family', 'surface', 'sigma', 'strategy', 'metric']):
        x = g.dropna(subset=['block_median']); ph = g[['d_H0', 'd_H1', 'd_H2', 'd_H3']].to_numpy(float).ravel(); ph = ph[np.isfinite(ph)]
        out.append(dict(table_type='old_effect_omega0', batch='old', baseline=('R5x4' if fam == 'point' else 'L4.45'), family=fam, surface=s, sigma=float(sg), contrast=f'{st}-S0', metric=mtr,
                        domain=('omega0 (old global)' if mtr.startswith('glob') else 'local r <= 15 mm (unchanged)'), boundary='legacy', n_pairs=int(4 * len(g)), n_pairs_defined=int(len(ph)),
                        n_blocks_defined=int(len(x)), main=(float(x.block_median.median()) if len(x) else None), mean_pairs=(float(ph.mean()) if len(ph) else None),
                        block_min=(float(x.block_median.min()) if len(x) else None), block_max=(float(x.block_median.max()) if len(x) else None)))
    return pd.DataFrame(out)


def run(M: pd.DataFrame | None = None, require_complete: bool = True, with_bridge: bool = True) -> dict:
    M = pd.read_csv(K.OUT / 'manifest.csv', keep_default_na=False, low_memory=False, float_precision='round_trip') if M is None else M
    reg = K.load_json(K.OUT / 'layouts.json')
    C = case_rows(M, reg, require_complete=require_complete); Pq = paired(C)
    if with_bridge:
        Pq = pd.concat([Pq, dense_bridge(C)], ignore_index=True, sort=False)
    return dict(C=C, P=Pq)
