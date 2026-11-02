# Combine Chapter 4 case results and compute paired and four-cell effects.

from __future__ import annotations
import sys, json, time, hashlib, itertools, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import cb_common as G
import cb_run as RUN
import cb_prepare as P

C = G.C
import analysis as A0
AN = G.OUT / 'analysis'
KEYS = ['family', 'env', 'sigma', 'd_mm', 'tau_mm', 'floor']
RCOLS = None
DOT_CELLS = dict(a='D_r1_p8x4', b='D_r1_p8x6', c='D_r0.75_p8x4', d='D_r0.75_p8x6')
LINE_CELLS = dict(a='L_w0.5_s8', b='L_w0.5_s10', c='L_w1_s8', d='L_w1_s10')
PAIRS = [('D_r1_p8x4', 'D_r1_T8x4'), ('D_r1_p8x6', 'D_r1_T8x4'), ('D_r1_p8x6', 'D_r0.75_p8x4'), ('D_r1_p8x4', 'D_r0.75_p8x4'), ('D_r1_p8x6', 'D_r0.75_p8x6'),
         ('D_r0.75_p8x4', 'D_r0.75_p8x6'), ('L_w1_s8', 'L_w1_s10'), ('L_w0.5_s8', 'L_w0.5_s10'), ('L_w1_s10', 'L_w0.5_s10')]
PAIR_KEYS = ['surface', 'env', 'sigma', 'phase_block', 'rep', 'seed']


def pack_path(r: dict) -> Path:
    if r['config'] in G.NEW_CFGS:
        return RUN.Store('main').path(r['case_key'])
    if r['action'] == 'reuse':
        return Path(r['source_pack'])
    root = G.OLD['blur_store'].parent if r['module'] == 'blur' else G.OLD['light_store'].parent
    return root / 'data' / (hashlib.sha256(r['case_key'].encode()).hexdigest()[:24] + '.npz')


def old_records():
    rb, rl = C.csv(G.OLD['blur_results']), C.csv(G.OLD['light_results'])
    d = rb[rb.env == 'NOMINAL'].set_index('case_key'); e = rl[rl.env == 'NOMINAL'].set_index('case_key').reindex(d.index)
    G.require((d.forward_key == e.forward_key).all() and np.allclose(d.glob_Ctau, e.glob_Ctau, rtol=0, atol=0), 'Shared nominal records differ')
    R = pd.concat([rb, rl[rl.env != 'NOMINAL']], ignore_index=True)
    cols = ['case_key', 'd_mm', 'tau_mm', 'Ctau', 'Cvalid', 'n_nodes', 'sup_n']
    sb, sl = pd.read_csv(G.OLD['blur_sens'], low_memory=False), pd.read_csv(G.OLD['light_sens'], low_memory=False)
    S = pd.concat([sb[cols], sl[cols]], ignore_index=True).drop_duplicates(['case_key', 'd_mm', 'tau_mm'])
    G.require(len(R) == 15840 and not R.case_key.duplicated().any() and len(S) == 15840 * 8, 'Old records not 15840 unique')
    return R, S


def assemble():
    f, M = RUN.verify('main'); E_old = P.old_manifest()
    R_old, S_old = old_records()
    R_new, S_new = RUN.collect('main', M, f); RUN.result_gate(R_new, M); A0.sensitivity_gate(S_new, R_new)
    cols = sorted(set(R_old.columns) & set(R_new.columns))
    R = pd.concat([R_old[cols].assign(batch='old_12'), R_new[cols].assign(batch='new_5')], ignore_index=True)
    S = pd.concat([S_old[['case_key', 'd_mm', 'tau_mm', 'Ctau', 'Cvalid', 'n_nodes', 'sup_n']], S_new[['case_key', 'd_mm', 'tau_mm', 'Ctau', 'Cvalid', 'n_nodes', 'sup_n']]], ignore_index=True)
    ident = C.IDENTITY
    X = pd.concat([E_old[ident + ['action', 'source_pack', 'module', 'raster_key', 'peak', 'integral', 'lit_px']],
                   M[ident + ['action', 'source_pack', 'module', 'raster_key', 'peak', 'integral', 'lit_px']]], ignore_index=True)
    G.require(len(R) == 22440 and not R.case_key.duplicated().any() and set(R.case_key) == set(X.case_key), 'Merged matrix is not 22440 unique cases')
    return R, S, X, R_old, S_old, E_old, f


def published_old_choices():
    a = C.csv(G.HOME / 'outputs/mp06d_blur/analysis/choices.csv'); b = C.csv(G.HOME / 'outputs/mp06f_lighting/choices.csv')
    k = ['family', 'env', 'sigma', 'd_mm', 'tau_mm', 'floor']
    j = a[a.env == 'NOMINAL'].merge(b[b.env == 'NOMINAL'], on=k, suffixes=('_b', '_l'))
    G.require(len(j) == 192 and (j.preferred_b.fillna('') == j.preferred_l.fillna('')).all(), 'Published nominal choices differ between modules')
    return pd.concat([a, b[b.env != 'NOMINAL']], ignore_index=True)


def tables():
    t0 = time.time(); R, S, X, R_old, S_old, E_old, f = assemble()
    long, surf, ch = A0.tables(R, S, X)
    _, surf_o, ch_o = A0.tables(R_old, S_old, E_old[C.IDENTITY])
    pub = published_old_choices()
    cmp = ch_o.merge(pub, on=KEYS, suffixes=('', '_pub'), validate='one_to_one')
    same = all((cmp[c].fillna('').astype(str) == cmp[c + '_pub'].fillna('').astype(str)).all() for c in ['qualified', 'near_optimal', 'preferred', 'status'])
    same &= bool(np.allclose(cmp.best_score.astype(float), cmp.best_score_pub.astype(float), equal_nan=True, atol=1e-12))
    G.require(same and len(ch_o) == 1152 + 960 - 192, 'Old 12-candidate selection not reproduced from the old records')

    J = ch_o.merge(ch, on=KEYS, suffixes=('_old', '_new'), validate='one_to_one')
    J['new_qualified_added'] = [';'.join(sorted(set(str(n).split(';')) - set(str(o).split(';')) - {'', 'nan'})) for o, n in zip(J.qualified_old.fillna(''), J.qualified_new.fillna(''))]
    J['representative_changed'] = J.preferred_old.fillna('-') != J.preferred_new.fillna('-')
    J['best_gain_pp'] = 100 * (J.best_score_new.astype(float) - J.best_score_old.astype(float))
    J['status_change'] = J.status_old + ' -> ' + J.status_new

    sc = surf.groupby(['family', 'config', 'env', 'sigma', 'd_mm', 'tau_mm']).agg(J=('Ctau', 'min'), safe=('safe', 'all'), n_surf=('surface', 'nunique'),
                                                                                    phase_min=('phase_min', 'min'), rmse_max=('rmse_max', 'max')).reset_index()

    main = R.copy()
    four = pd.concat([fourcell(main, DOT_CELLS, 'point', 'r 1 / 0.75 x 8x4 / 8x6'), fourcell(main, LINE_CELLS, 'line', 'w 0.5 / 1 x s 8 / 10')], ignore_index=True)
    eff = pd.concat([paired_effect(main, a, b) for a, b in PAIRS], ignore_index=True)
    AN.mkdir(parents=True, exist_ok=True)
    for name, t in [('results_all.csv.gz', R), ('surfaces_all.csv.gz', surf), ('choices_all.csv', ch), ('choices_old12_reproduced.csv', ch_o), ('selection_changes.csv', J),
                    ('config_scores.csv.gz', sc), ('fourcell.csv', four), ('paired_effects.csv', eff)]:
        p = AN / name
        if name.endswith('.gz'):
            C.atomic(p, lambda q, tab=t: tab.to_csv(q, index=False, compression='gzip'))
        else:
            C.save_csv(p, t)
    C.atomic(AN / 'sensitivity_all.csv.gz', lambda q: S.to_csv(q, index=False, compression='gzip'))
    rec = dict(images=len(R), new=int((R.batch == 'new_5').sum()), old=int((R.batch == 'old_12').sum()), choices=len(ch), old_choices_reproduced=True,
               representative_changes=int(J.representative_changed.sum()), freeze_id=f['freeze_id'], seconds=round(time.time() - t0, 1))
    C.save(AN / 'tables_check.json', rec); print(C.canon(rec))


def _block_means(R, cfg):
    q = R[R.config == cfg]
    return q.groupby(['surface', 'env', 'sigma', 'phase_block']).glob_Ctau.mean()


def fourcell(R, cells, fam, name):

    m = {k: _block_means(R, v) for k, v in cells.items()}
    idx = m['a'].index
    for k in m:
        G.require(set(m[k].index) == set(idx), f'Four-cell blocks incomplete for {cells[k]}')
    D = pd.DataFrame({k: v.reindex(idx) for k, v in m.items()})
    D['size_or_width_at_low'] = D.c - D.a; D['size_or_width_at_high'] = D.d - D.b
    D['spacing_at_ref'] = D.b - D.a; D['spacing_at_alt'] = D.d - D.c
    D['DiD'] = (D.d - D.c) - (D.b - D.a)
    g = D.reset_index().groupby(['surface', 'env', 'sigma'])
    out = g[['a', 'b', 'c', 'd', 'size_or_width_at_low', 'size_or_width_at_high', 'spacing_at_ref', 'spacing_at_alt', 'DiD']].mean()
    out['DiD_blocks_pos'] = g.DiD.apply(lambda x: int((x > 0).sum())); out['DiD_block_min'] = g.DiD.min(); out['DiD_block_max'] = g.DiD.max()
    out = out.reset_index(); out.insert(0, 'family', fam); out.insert(1, 'design', name)
    for k, v in cells.items():
        out[f'cell_{k}'] = v
    return out


def paired_effect(R, left, right):
    q = RUN.pair(R, R, left, right, PAIR_KEYS)
    q['block_delta'] = q.delta_Ctau
    g = q.groupby(['surface', 'env', 'sigma', 'phase_block']).delta_Ctau.mean().reset_index()
    out = g.groupby(['surface', 'env', 'sigma']).agg(mean_delta=('delta_Ctau', 'mean'), block_min=('delta_Ctau', 'min'), block_max=('delta_Ctau', 'max'),
                                                      blocks_pos=('delta_Ctau', lambda x: int((x > 0).sum())), blocks=('delta_Ctau', 'size')).reset_index()
    out.insert(0, 'left', left); out.insert(1, 'right', right)
    return out


_LOC: dict = {}


def _local_init(files):
    C.install_guard()
    import cb_adapter as A
    S = A.platform()
    import rdy_platform as RP
    for sf in G.SURFS:
        _LOC[sf] = RP.dom_local(sf)


def _local_one(r):
    from surface_eval_r3 import surface_eval_r3_dev
    m = r['method']; p = Path(r['pack'])
    with np.load(p, allow_pickle=False) as z:
        pts = z[m + '__ev__pts']; v = z[m + '__ev__valid']
    pv = np.asarray(pts, np.float64)[np.asarray(v, bool)]; dl = _LOC[r['surface']]
    out = []
    if len(pv) < 3:
        for d, t in itertools.product(C.DS, C.TAUS):
            out.append(dict(case_key=r['case_key'], d_mm=d, tau_mm=t, loc_Ctau=0.0, loc_Cvalid=0.0, loc_sup_rmse=None, loc_n_nodes=int(len(dl['x']))))
        return out
    D = dict(x=dl['x'], y=dl['y'], z_ref=dl['z_ref'], area=dl['area'], hash=dl['hash'])
    g = surface_eval_r3_dev(pv, None, D, C.DS, C.TAUS, support_rule='exact')['subdomains']['global']
    for d, t in itertools.product(C.DS, C.TAUS):
        b = g['by_d'][str(d)]
        out.append(dict(case_key=r['case_key'], d_mm=d, tau_mm=t, loc_Ctau=b['C_tau'][str(t)] or 0.0, loc_Cvalid=b['C_valid'] or 0.0, loc_sup_rmse=b['rmse'], loc_n_nodes=g['n_nodes']))
    return out


def local():
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time(); R, S, X, *_ = assemble()
    X = X.set_index('case_key'); jobs = []
    for r in R[['case_key', 'surface', 'method', 'config']].to_dict('records'):
        x = G.rowdict(X.loc[r['case_key']].to_dict() | {'case_key': r['case_key']})
        p = pack_path(x); G.require(p.exists(), 'Missing cloud pack ' + r['case_key']); jobs.append(dict(r, pack=str(p)))
    rows = []
    with ProcessPoolExecutor(max_workers=8, initializer=_local_init, initargs=(None,)) as ex:
        for i, o in enumerate(ex.map(_local_one, jobs, chunksize=40)):
            rows.extend(o)
            if (i + 1) % 2000 == 0:
                print(f'local {i + 1}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    L = pd.DataFrame(rows)
    main = L[(L.d_mm == 6) & (L.tau_mm == .15)].merge(R[['case_key', 'loc_Ctau', 'loc_Cvalid']], on='case_key', suffixes=('', '_stored'), validate='one_to_one')
    diff = float(np.nanmax(np.abs(np.r_[main.loc_Ctau - main.loc_Ctau_stored.fillna(0), main.loc_Cvalid - main.loc_Cvalid_stored.fillna(0)])))
    C.atomic(AN / 'local_all.csv.gz', lambda q: L.to_csv(q, index=False, compression='gzip'))
    dom = {sf: dict(n_nodes=int(len(_dom_local(sf)['x'])), hash=_dom_local(sf)['hash'], r_max_mm=float(np.hypot(_dom_local(sf)['x'], _dom_local(sf)['y']).max())) for sf in G.SURFS}
    rec = dict(cases=len(R), rows=len(L), main_vs_stored_maxdiff=diff, local_domain=dom, seconds=round(time.time() - t0, 1),
               note='Local ROI = registered dom_local (s2.roi_domain, radius 15 mm); functional evidence only, no new admission gate')
    C.save(AN / 'local_check.json', rec); print(C.canon(rec))
    G.require(diff < 1e-10, f'Local re-evaluation differs from stored local metrics ({diff})')


def _dom_local(sf):
    import cb_adapter as A
    S = A.platform()
    import rdy_platform as RP
    return RP.dom_local(sf)


if __name__ == '__main__':
    try:
        C.install_guard()
        dict(tables=tables, local=local)[sys.argv[1]]()
    except Exception:
        traceback.print_exc(); sys.exit(2)
