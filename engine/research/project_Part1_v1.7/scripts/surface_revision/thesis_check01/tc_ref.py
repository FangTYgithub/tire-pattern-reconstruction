# Build truth-sampled point clouds and evaluate them on the study domains.

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import tc_common as H

DEF = dict(
    name='ideal sampling reference; not an information limit, not a theoretical optimum, not a strict upper bound',
    question='given the registered expected feature sampling positions with exactly known true heights: what does the current interpolation with the '
             'current support requirement recover?',
    points='stored exp_xy[exp_in_image] (expected dot-centre hit on the true surface, centre ray inside the camera image; float32 as stored) + true height '
           's2_common.truth_interp(build_mesh(platform surface)) at (x, y) (S4-REAL-04 L1 rule); rows with non-finite height dropped',
    lines='stored registered reference rows ref_xyz (l01_truth / r2_eval_snap line_reference: one true surface point per camera row and registered line, '
          'original row step); the registered set may exclude clipped residual lines - the reference refers to this registered set only',
    filters='none from detection / decoding / matching; no depth-window filter (L1 definition); counts of out-of-window reference points recorded',
    evaluation='surface_eval_r3_dev(points, None, domain, d_list=[5, 6, 7], tau_list=[0.15], support_rule="exact"); main d = 6 mm',
    domains=dict(S12=dict(global_='Omega0 = s1_real01/domain/global_domain_step0p50 (30947 nodes, area 0.25 mm2 each)', local='s2_common.roi_domain (r <= 15 mm, 22605 nodes)'),
                 R7=dict(global_='Omega1 = pix_bound01/domains/omega1 (30986 nodes, area 0.25 mm2 each)', local='s2_common.roi_domain (r <= 15 mm, 22605 nodes)')),
    metrics='n_ref, C_int (interpolable area share), C_valid(d), C_tau(d, 0.15), RMSE / P95 over all interpolable nodes, RMSE / P95 over supported nodes',
    decomposition='1 - C_tau = (1 - C_int) + (C_int - C_valid) + (C_valid - C_tau): not interpolable / interpolable but not supported / supported but '
                  'error > tau; an arithmetic split of coverage shares, NOT a causal split into detection / matching / sampling',
    actual='stored batch metrics (S12 results/views.csv; S4-REAL-07 results.csv); sigma-0 clouds additionally re-evaluated with the same call to prove the '
           'path and to obtain supported RMSE / d-sensitivity; actual above reference is kept (no clipping); no-output: actual C_tau 0, errors undefined',
    linking='S12: every main-regime view (sigma 0-100) -> reference of its (platform surface, raster key) sigma-0 input; R7: every condition -> '
            'reference of its sigma-0 condition with the same (layout, phase, surface). The reference is reused along the noise axis, never counted '
            'as a new sample; phases, repetitions, generation blocks and pairing keys kept',
    support_radius='exact rho_tri of src/surface_eval_r3.covering_radius_exact (largest distance of a triangle point to its nearest vertex); not the '
                   'minimum enclosing circle, not the circumcircle of an obtuse triangle',
    tolerances=dict(C='<= 1e-9', rmse_mm='<= 1e-6'),
    written_before_computation=True)


_C: dict = {}


def _dom_file(batch: str):
    if batch == 'S12':
        return H.SR / 's1_real01' / 'domain' / 'global_domain_step0p50.npz', H.SR / 's1_real01' / 'domain' / 'global_domain_step0p50.json', 30947
    return H.SR / 'pix_bound01' / 'domains' / 'omega1.npz', H.SR / 'pix_bound01' / 'domains' / 'omega1.json', 30986


def domain(batch: str, sid: str, which: str) -> dict:
    key = ('dom', batch, sid, which)
    if key not in _C:
        S = H.s2()
        if which == 'glob':
            z, j, n = _dom_file(batch); rec = H.load_json(H.reg(j)); H.reg(z)
            if H.sha_file(z) != rec['npz_sha256']:
                raise H.Stop('version', f'{H.rel(z)} differs from its record')
            with np.load(z) as a:
                d = {k: np.asarray(a[k]) for k in a.files}
            if len(d['x']) != n or int(rec['n_nodes']) != n:
                raise H.Stop('version', f'{batch} global domain has {len(d["x"])} nodes, expected {n}')
            step = float(rec.get('step_mm', rec.get('step')))
            _C[key] = dict(x=d['x'], y=d['y'], z_ref=np.asarray(d[f'z_{sid}'], np.float64), area=np.full(n, step ** 2), hash=rec['hash'], n=n)
        else:
            dl = S.roi_domain(mesh(sid))
            _C[key] = dict(x=dl['x'], y=dl['y'], z_ref=np.asarray(dl['z_ref'], np.float64), area=np.asarray(dl['area'], np.float64), hash=dl['hash'], n=int(dl['n_nodes']))
    return _C[key]


def mesh(sid: str) -> dict:
    if ('mesh', sid) not in _C:
        _C[('mesh', sid)] = H.s2().build_mesh(sid)
    return _C[('mesh', sid)]


def truth(sid: str):
    if ('tz', sid) not in _C:
        _C[('tz', sid)] = H.s2().truth_interp(mesh(sid))
    return _C[('tz', sid)]


def ref_from_eval(npz_path, family: str, sid: str) -> tuple[np.ndarray, dict]:
    with np.load(npz_path) as A:
        if family == 'point':
            exy = np.asarray(A['exp_xy'], np.float64); vis = np.asarray(A['exp_in_image'], bool)
            xy = exy[vis]; z = truth(sid)(np.column_stack([xy[:, 1], xy[:, 0]])); ok = np.isfinite(z)
            P = np.column_stack([xy[ok], z[ok]]); info = dict(n_expected_design=int(len(exy)), n_visible=int(vis.sum()), n_ref=int(ok.sum()), n_dropped_nonfinite=int((~ok).sum()))
        else:
            R = np.asarray(A['ref_xyz'], np.float64); ok = np.isfinite(R).all(axis=1); P = R[ok]
            info = dict(n_ref_rows=int(len(R)), n_ref=int(ok.sum()), n_dropped_nonfinite=int((~ok).sum()), n_ref_inner=int(np.asarray(A['ref_inner'], bool).sum()),
                        n_lines=int(len(np.unique(np.asarray(A['ref_k'])[ok]))))
    info['n_out_of_window'] = int(((P[:, 2] < H.WINDOW[0]) | (P[:, 2] > H.WINDOW[1])).sum()) if len(P) else 0
    return P, info


def ref_key(batch: str, sid: str, P: np.ndarray) -> str:
    return 'TCR-' + H.sha_bytes(f'{batch}|{sid}|'.encode() + np.ascontiguousarray(P, np.float64).tobytes())[:24]


def eval_cloud(P: np.ndarray, D: dict, d_list=H.D_SENS) -> dict:
    from surface_eval_r3 import surface_eval_r3_dev
    out = dict(n_points=int(len(P)), n_nodes=int(len(D['x'])))
    if len(P) < 3:
        out.update(C_int=0.0, int_rmse=None, int_p95=None, triangulation='fewer than 3 points')
        for d in d_list:
            out.update({f'C_valid_d{d:g}': 0.0, f'C_tau_d{d:g}': 0.0, f'n_sup_d{d:g}': 0, f'sup_rmse_d{d:g}': None, f'sup_p95_d{d:g}': None})
        return out
    ev = surface_eval_r3_dev(P, None, dict(x=D['x'], y=D['y'], z_ref=D['z_ref'], area=D['area'], hash=D['hash']), list(d_list), [H.TAU], return_fields=True, support_rule='exact')
    f = ev.pop('fields'); g = ev['subdomains']['global']; e = np.asarray(f['z_est'], np.float64) - D['z_ref']; ie = e[np.isfinite(e)]
    out.update(C_int=float(g['C_interpolable'] or 0.0), int_rmse=(float(np.sqrt(np.mean(ie ** 2))) if ie.size else None), int_p95=(float(np.percentile(np.abs(ie), 95)) if ie.size else None),
               triangulation=ev['triangulation'], n_used=ev['n_used'], n_dup_removed=ev['n_duplicates_removed'])
    for d in d_list:
        b = g['by_d'][str(float(d))]
        out.update({f'C_valid_d{d:g}': float(b['C_valid'] or 0.0), f'C_tau_d{d:g}': float(b['C_tau'][str(float(H.TAU))] or 0.0), f'n_sup_d{d:g}': int(b['n_supported']),
                    f'sup_rmse_d{d:g}': b['rmse'], f'sup_p95_d{d:g}': b['p95']})
    return out


def s12_eval_path(r) -> Path:
    if r['source'] == 's1':
        return H.SR / 's1_real01' / 'decodes' / f"{r['s1_decode_key']}.json"
    if r['source'] == 's2':
        return H.SR / 's2_prepare' / 'run' / 'eval' / f"{r['s2_decode_key']}.json"
    return H.S12 / 'eval' / f"{r['decode_key']}.json"


def s12_items() -> tuple[pd.DataFrame, pd.DataFrame]:
    M = pd.read_csv(H.reg(H.S12 / 'manifest.csv'), low_memory=False); I = pd.read_csv(H.reg(H.S12 / 'inputs.csv'), low_memory=False)
    v = M[(M.surface_class == 'main') & (M.regime == 'resource_cap_11695') & (M.sigma == 0)].copy()
    u = I[I.input_key.isin(v.input_key)].copy()
    u['eval_path'] = [str(s12_eval_path(r)) for r in u.to_dict('records')]
    return v, u


def _work_s12(r: dict) -> dict:
    t0 = time.time(); sid = r['platform_surface']
    jp = Path(r['eval_path']); zp = jp.with_suffix('.npz')
    rec = json.loads(jp.read_text(encoding='utf-8'))
    if H.sha_file(zp) != rec['npz_sha256']:
        return dict(input_key=r['input_key'], error='npz hash differs from its record')
    P, info = ref_from_eval(zp, r['family'], sid)
    out = dict(input_key=r['input_key'], ref_key=ref_key('S12', sid, P), eval_json=H.rel(jp), eval_npz_sha=rec['npz_sha256'], **{f'ref_{k}': v for k, v in info.items()})
    with np.load(zp) as A:
        Pv = np.asarray(A['pts'], np.float64)[np.asarray(A['valid'], bool)]
    for dn in ('glob', 'loc'):
        a = eval_cloud(Pv, domain('S12', sid, dn))
        out.update({f'act_{dn}_{k}': v for k, v in a.items()})
    out['stored'] = {k: rec['metrics'].get(k) for k in ('glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'glob_surf_rmse', 'loc_Ctau', 'loc_Cvalid', 'loc_cov_interp', 'loc_surf_rmse', 'state', 'n_valid')}
    out['seconds'] = round(time.time() - t0, 2)
    return out


def _work_ref(args) -> dict:
    batch, sid, key, P = args
    out = dict(ref_key=key, batch=batch, platform_surface=sid)
    for dn in ('glob', 'loc'):
        out.update({f'{dn}_{k}': v for k, v in eval_cloud(P, domain(batch, sid, dn)).items()})
    return out


def run_s12(workers: int = 10) -> dict:
    H.run_fix('B_definition', DEF)
    t0 = time.time(); v, u = s12_items()
    if len(v) != 723 or len(u) != 699:
        raise H.Stop('scope', f'S12 sigma-0 main views / inputs {len(v)} / {len(u)} != 723 / 699')
    miss = [p for p in u.eval_path if not Path(p).exists() or not Path(p).with_suffix('.npz').exists()]
    for p in u.eval_path:
        if Path(p).exists():
            H.reg(p); H.reg(Path(p).with_suffix('.npz'))
    rows = u.to_dict('records')
    with Pool(workers) as pool:
        res = pool.map(_work_s12, [r for r in rows if r['eval_path'] not in miss], chunksize=4)
    err = [x for x in res if 'error' in x]
    if err:
        raise H.Stop('version', f'{len(err)} stored S12 evaluation arrays fail their hash: {err[:3]}')
    R = pd.DataFrame(res).merge(u[['input_key', 'platform_surface', 'surface', 'family', 'size_mm', 'layout', 'phase_label', 'sections', 'raster_key', 'source', 'phase_frac']], on='input_key')

    refs = {}
    for r in res:
        sid = u[u.input_key == r['input_key']].platform_surface.iloc[0]
        refs.setdefault(r['ref_key'], (sid, r['input_key']))
    jobs = []
    store = {}
    for k, (sid, ik) in refs.items():
        P, _ = ref_from_eval(Path(u[u.input_key == ik].eval_path.iloc[0]).with_suffix('.npz'), u[u.input_key == ik].family.iloc[0], sid)
        if ref_key('S12', sid, P) != k:
            raise H.Stop('check', f'reference {k} not reproducible')
        jobs.append(('S12', sid, k, P)); store[k] = P
    with Pool(workers) as pool:
        RF = pd.DataFrame(pool.map(_work_ref, jobs, chunksize=2))
    H.save_npz(H.CACHE / 'refs_S12.npz', **store)
    st = pd.DataFrame([dict(input_key=x['input_key'], **{f'stored_{k}': v for k, v in x['stored'].items()}) for x in res])
    H.part('S12_inputs', R.drop(columns=['stored']).merge(st, on='input_key'))
    H.part('S12_refs', RF)
    out = dict(views=len(v), inputs=len(u), inputs_evaluated=len(res), missing_sources=miss, unique_references=len(refs), seconds=round(time.time() - t0, 1), t_local=H.now_local())
    H.run_put('B_S12_run', out); H.flush_reads('B_S12_run')
    return out


def r7_items() -> pd.DataFrame:
    M = pd.read_csv(H.reg(H.R7 / 'manifest.csv'), low_memory=False, keep_default_na=False, float_precision='round_trip')
    Rs = pd.read_csv(H.reg(H.R7 / 'results.csv'), low_memory=False, float_precision='round_trip')
    C = Rs[(Rs.record_type == 'cond') & (Rs.domain == 'glob')][['image_key', 'method', 'eval_key']]
    m = M[M.sigma.astype(float) == 0.0].copy()
    rows = []
    for r in m.to_dict('records'):
        meths = ['D0'] if r['family'] == 'point' else ['P4TABLE', 'CLIP']
        for mt in meths:
            ek = C[(C.image_key == r['image_key']) & (C.method == mt)].eval_key.iloc[0]
            rows.append(dict(r, method=mt, eval_key=ek, eval_path=str(H.R7 / 'evals' / f'{ek}.json')))
    return pd.DataFrame(rows)


def _work_r7(r: dict) -> dict:
    sid = H.PLATFORM[r['surface']]; jp = Path(r['eval_path']); zp = jp.with_suffix('.npz'); rec = json.loads(jp.read_text(encoding='utf-8'))
    if H.sha_file(zp) != rec['npz_sha256']:
        return dict(image_key=r['image_key'], method=r['method'], error='npz hash differs from its record')
    P, info = ref_from_eval(zp, r['family'], sid)
    out = dict(image_key=r['image_key'], method=r['method'], ref_key=ref_key('R7', sid, P), eval_npz_sha=rec['npz_sha256'], **{f'ref_{k}': v for k, v in info.items()})
    with np.load(zp) as A:
        Pv = np.asarray(A['pts'], np.float64)[np.asarray(A['valid'], bool)]
    for dn in ('glob', 'loc'):
        out.update({f'act_{dn}_{k}': v for k, v in eval_cloud(Pv, domain('R7', sid, dn)).items()})
    return out


def run_r7(workers: int = 10) -> dict:
    H.run_fix('B_definition', DEF)
    t0 = time.time(); X = r7_items()
    n_cond = X.image_key.nunique()
    if n_cond != 255:
        raise H.Stop('scope', f'R7 sigma-0 conditions {n_cond} != 255')
    for p in X.eval_path:
        H.reg(p); H.reg(Path(p).with_suffix('.npz'))
    with Pool(workers) as pool:
        res = pool.map(_work_r7, X.to_dict('records'), chunksize=4)
    err = [x for x in res if 'error' in x]
    if err:
        raise H.Stop('version', f'{len(err)} stored R7 evaluation arrays fail their hash: {err[:3]}')
    R = pd.DataFrame(res).merge(X[['image_key', 'method', 'set', 'baseline', 'family', 'layout', 'block', 'phase', 'surface', 'raster_key', 'clipped_band_lit', 'eval_key']], on=['image_key', 'method'])

    ln = R[R.family == 'line'].pivot_table(index='image_key', columns='method', values='ref_key', aggfunc='first')
    same = bool((ln['P4TABLE'] == ln['CLIP']).all())
    if not same:
        raise H.Stop('check', 'P4 / CLIP evaluations of one image carry different reference rows')
    refs = {}; store = {}; jobs = []
    for r in R.drop_duplicates('ref_key').itertuples():
        sid = H.PLATFORM[r.surface]; P, _ = ref_from_eval(Path(X[(X.image_key == r.image_key) & (X.method == r.method)].eval_path.iloc[0]).with_suffix('.npz'), r.family, sid)
        if ref_key('R7', sid, P) != r.ref_key:
            raise H.Stop('check', f'reference {r.ref_key} not reproducible')
        jobs.append(('R7', sid, r.ref_key, P)); store[r.ref_key] = P
    with Pool(workers) as pool:
        RF = pd.DataFrame(pool.map(_work_ref, jobs, chunksize=2))
    H.save_npz(H.CACHE / 'refs_R7.npz', **store)
    H.part('R7_conditions', R); H.part('R7_refs', RF)
    out = dict(conditions=int(n_cond), by_set=X.drop_duplicates('image_key').groupby('set').size().to_dict(), clouds=len(R), unique_references=len(store),
               p4_clip_same_reference=same, seconds=round(time.time() - t0, 1), t_local=H.now_local())
    H.run_put('B_R7_run', out); H.flush_reads('B_R7_run')
    return out


def gate_r4() -> dict:
    from surface_eval_r3 import surface_eval_r3_dev
    D = pd.read_csv(H.reg(H.R4 / '_parts' / 'C2_layers.csv'), low_memory=False, float_precision='round_trip'); L1 = D[D.layer == 'L1_reference']
    rows = []
    for ek, g in L1.groupby('eval_key'):
        fam = g.family.iloc[0]; surface = g.surface.iloc[0]; sid = H.PLATFORM[surface]
        z = H.reg(H.R2 / 'eval' / f'{ek}.npz')
        P, info = ref_from_eval(z, fam, sid)
        for dn, dname in (('glob', 'global'), ('loc', 'local')):
            Dd = domain('R7', sid, dn)
            dl = sorted(g.d_mm.unique().tolist())
            ev = surface_eval_r3_dev(P, None, dict(x=Dd['x'], y=Dd['y'], z_ref=Dd['z_ref'], area=Dd['area'], hash=Dd['hash']), dl, [H.TAU], support_rule='exact')
            s = ev['subdomains']['global']
            for d in dl:
                o = g[(g.domain == dname) & (g.d_mm == d)].iloc[0]; b = s['by_d'][str(float(d))]
                diff = dict(C_int=abs(float(s['C_interpolable']) - float(o.C_int)), C_valid=abs(float(b['C_valid']) - float(o.C_valid)),
                            C_tau=abs(float(b['C_tau'][str(float(H.TAU))]) - float(o.C_tau)),
                            sup_rmse=(abs(float(b['rmse']) - float(o.sup_rmse)) if (b['rmse'] is not None and pd.notna(o.sup_rmse)) else (0.0 if (b['rmse'] is None and pd.isna(o.sup_rmse)) else 1.0)))
                rows.append(dict(record_type='gate_R4', eval_key=ek, baseline=g.baseline.iloc[0], family=fam, surface=surface, strategy=g.strategy.iloc[0], layout=g.layout.iloc[0],
                                 domain=dname, d_mm=d, n_points=int(len(P)), n_points_R4=int(o.n_points), C_tau=float(b['C_tau'][str(float(H.TAU))]), C_tau_R4=float(o.C_tau),
                                 **{f'diff_{k}': v for k, v in diff.items()}))
    G = pd.DataFrame(rows)
    mx = {k: float(G[f'diff_{k}'].max()) for k in ('C_int', 'C_valid', 'C_tau', 'sup_rmse')}
    m24 = G[(G.family == 'line') & (G.surface == 'M24_curv') & (G.strategy == 'S0') & (G.domain == 'global') & (G.d_mm == 6.0)]
    out = dict(ok=bool(all(v <= 1e-12 for v in mx.values()) and (G.n_points == G.n_points_R4).all()), n_clouds=int(G.eval_key.nunique()), n_rows=len(G), max_diff=mx,
               m24_l63_s0_reference=(float(m24.C_tau.iloc[0]) if len(m24) else None), t_local=H.now_local())
    H.part('gate_R4', G); H.run_put('B_gate_R4', out); H.flush_reads('B_gate_R4')
    if not out['ok']:
        raise H.Stop('gate', f'S4-REAL-04 L1 reproduction failed: {mx}')
    return out
