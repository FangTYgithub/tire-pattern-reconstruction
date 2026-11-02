# Check layout result completeness and phase-paired comparisons.

from __future__ import annotations

import numpy as np
import pandas as pd

import r2_common as K

BAND = 8
SCREEN = dict(Ctau_pp=1.0, rmse_mm=0.01, p95_mm=0.02, unconf_pp=1.0)


def _entry(M):
    return M[M.entry_subset.astype(str).isin(['True', 'true', '1'])]


def g1_closure(M: pd.DataFrame) -> dict:
    import r2_manifest as RM
    r = K.run_load(); fpd = r['bind']['fingerprint_digests']; v = RM.validate(M, fpd)
    now = {k: K.digest(x, 32) for k, x in RM.fingerprints().items()}
    E = _entry(M); reg = K.load_json(K.OUT / 'layouts.json'); bad = []
    for x in E.itertuples():
        if x.layout not in reg['baselines'][x.baseline]['layouts']:
            bad.append(f'unknown layout {x.baseline} {x.layout}')
    dots = E[E.family == 'point']
    miss_ref = sorted(set(dots.sigma0_ref_key) - set(E[E.sigma == 0].image_key))
    rec_bad = []
    for x in E.drop_duplicates('image_key').itertuples():
        j = K.load_json(K.IMG / f'{x.image_key}.json')
        if (j['raster_key'] != x.raster_key or float(np.float32(j['peak'])) != float(x.peak_f32) or j['surface'] != x.surface or float(j['sigma']) != float(x.sigma)
                or str(j['noise_seed']) != str(x.noise_seed) or j['boundary'] != K.BOUNDARY or j['image_code'] != x.image_code):
            rec_bad.append(x.image_key)
    for x in E.drop_duplicates('decode_key').itertuples():
        j = K.load_json(K.RAW / f'{x.decode_key}.json')
        if j['image_key'] != x.image_key or j['method'] != x.method or j['arm'] != x.arm or j['decode_code'] != x.decode_code:
            rec_bad.append(x.decode_key)
    for x in E.drop_duplicates('eval_key').itertuples():
        j = K.load_json(K.EVL / f'{x.eval_key}.json')
        if j['decode_key'] != x.decode_key or j['eval_code'] != x.eval_code or j['boundary'] != K.BOUNDARY or j['domain_main']['hash'] != x.domain_main_hash:
            rec_bad.append(x.eval_key)
    ok = (not bad) and (not miss_ref) and (not rec_bad) and now == fpd
    return dict(ok=bool(ok), manifest=v, fingerprints_unchanged=bool(now == fpd), unknown_layouts=bad, entry_sigma0_refs_missing=miss_ref, record_mismatch=rec_bad[:10],
                n_entry_images=int(E.image_key.nunique()), n_entry_views=int(len(E)))


def g2_end_to_end(M: pd.DataFrame, status: dict, ana: dict, figs: dict) -> dict:
    import r2_run as R
    E = _entry(M); U = E.drop_duplicates('image_key')
    n_done = int(sum(R.done(r) for r in U.to_dict('records')))
    C = ana['C']; okC = bool(len(C) == len(E) and (C.record == 'complete').all())
    okF = bool(figs['n_figures'] >= 1 and figs['board_checks'] and all(b['equal'] for b in figs['board_checks']))
    return dict(ok=bool(n_done == 36 and okC and okF and status.get('n_failed') == 0), n_images_complete=n_done, n_views_analysed=int(len(C)), figures=figs['files'],
                board_checks=figs['board_checks'], scheduler=status)


def _line_plane_check(inp: dict) -> dict:
    sp = inp['table']['spec']; pat = np.asarray(inp['binary']) > 0; cu = float(sp['centre_uv'][0])
    ids = sorted(int(k) for k in sp['segments']); q = np.array([float(sp['table']['q_px'][str(k)]) for k in sorted(int(k) for k in sp['table']['q_px'])])
    allid = sorted(int(k) for k in sp['table']['q_px']); uline = cu + q
    yy, xx = np.nonzero(pat); near = np.abs(xx[:, None] - uline[None, :]).argmin(axis=1)
    out = []
    for k in ids:
        i = allid.index(k); sel = near == i
        cen = float(xx[sel].mean()) if sel.any() else None
        out.append(dict(id=k, u_table=float(uline[i]), u_raster_centroid=cen, lit_px=int(sel.sum()), lit_px_table=int(sp['segments'][str(k)]['lit_px'])))
    dev = max(abs(o['u_raster_centroid'] - o['u_table']) for o in out if o['u_raster_centroid'] is not None)
    cnt = all(o['lit_px'] == o['lit_px_table'] for o in out)
    return dict(kind='line planes', n_lines=len(out), max_centroid_dev_px=dev, lit_counts_equal=bool(cnt), ok=bool(dev <= 0.5 and cnt))


def _dot_centre_check(inp: dict) -> dict:
    from scipy import ndimage
    pat = np.asarray(inp['binary']) > 0; c = np.asarray(inp['centres'], np.float64)
    lab, n = ndimage.label(pat); com = np.asarray(ndimage.center_of_mass(pat, lab, range(1, n + 1)), np.float64)[:, ::-1] if n else np.zeros((0, 2))
    inner = (c[:, 0] >= 8) & (c[:, 0] <= 503) & (c[:, 1] >= 8) & (c[:, 1] <= 503)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(com).query(c[inner]) if n else (np.array([np.inf]), None)
    return dict(kind='dot centres', n_components=int(n), n_centres=int(len(c)), n_inner=int(inner.sum()), max_centre_dev_px=float(d.max()), ok=bool(d.max() <= 0.5))


def image_diff(old: np.ndarray, new: np.ndarray, band: int = BAND) -> dict:
    d = new.astype(np.int16) - old.astype(np.int16); ch = d != 0; H, W = d.shape
    yy, xx = np.mgrid[0:H, 0:W]; dist = np.minimum.reduce([yy, xx, H - 1 - yy, W - 1 - xx]); outer = dist < band
    return dict(n_changed=int(ch.sum()), n_changed_outer=int((ch & outer).sum()), n_changed_inner=int((ch & ~outer).sum()), max_abs=int(np.abs(d).max()),
                max_border_dist_changed=(int(dist[ch].max()) if ch.any() else None))


def g3_equivalence(M: pd.DataFrame, legacy: dict) -> dict:
    import r2_run as R
    E = _entry(M); out = dict(s0_equiv=[], planes=[], identity=[], legacy_images=[])
    for x in E[E.layout == 'regular'].drop_duplicates('image_key').itertuples():
        j = K.load_json(K.EQV / f'{x.image_key}.json'); out['s0_equiv'].append(dict(image_key=x.image_key, equal=bool(j['equal']), max_abs_point_diff_mm=j.get('max_abs_point_diff_mm')))
    for x in E.drop_duplicates(['baseline', 'layout', 'phase']).to_dict('records'):
        inp = R.inputs_of(x); chk = _line_plane_check(inp) if x['family'] == 'line' else _dot_centre_check(inp)
        if x['family'] == 'line':
            chk['truth_table_same_spec'] = bool(inp['meta_full']['olines'] is inp['table']['spec'])
            chk['ok'] = bool(chk['ok'] and chk['truth_table_same_spec'])
        out['planes'].append(dict(baseline=x['baseline'], layout=x['layout'], phase=x['phase'], **chk))
    for x in E.drop_duplicates('eval_key').itertuples():
        j = K.load_json(K.EVL / f'{x.eval_key}.json'); m = j['metrics']
        geo_ok = (m['id_correct'] + m['id_wrong'] + m['id_unconf'] == m['id_denominator'] == m['n_accepted'])
        src = (j.get('extras') or {}).get('source_identity')
        src_ok = True if src is None else (src['correct'] + src['wrong'] + src['undeterminable'] == src['n_reliable'])
        out['identity'].append(dict(eval_key=x.eval_key, family=x.family, sigma=x.sigma, geo_closed=bool(geo_ok), src_closed=bool(src_ok), id_unconf=m['id_unconf'],
                                    src_undeterminable=(src or {}).get('undeterminable'), src_role=(src or {}).get('source')))
    for lr in legacy['refs']:
        old = K.load_npz(K.OUT / 'legacy_ref' / f"{lr['legacy_key']}.npz")['image']; new = K.load_npz(K.IMG / f"{lr['formal_image_key']}.npz")['image']
        out['legacy_images'].append(dict(lr, **image_diff(old, new)))
    pix = K.load_json(K.PIX / 'run.json')['final']['code_hashes']
    pb_same = K.sha_file(K.ROOT / 'scripts' / 'surface_revision' / 'pix_bound01' / 'pb_m3.py') == pix['pb_m3.py']
    reg = K.run_load()['bind']['evaluator_regression']['ok']
    n_s0 = int(E[E.layout == 'regular'].image_key.nunique())
    out['n_s0_expected'] = n_s0
    ok = (all(e['equal'] for e in out['s0_equiv']) and len(out['s0_equiv']) == n_s0 == 12 and all(p['ok'] for p in out['planes']) and all(i['geo_closed'] and i['src_closed'] for i in out['identity'])
          and all(li['n_changed_inner'] == 0 for li in out['legacy_images']) and len(out['legacy_images']) == legacy['n'] and pb_same and reg)
    return dict(ok=bool(ok), pb_m3_equals_pix_verified=bool(pb_same), evaluator_regression_ok=bool(reg), **out)


def _semantics(m: dict, nodes: dict | None = None) -> list:
    bad = []
    def n(k):
        v = m.get(k); return None if v is None else float(v)
    for p in ('glob', 'loc'):
        ct, cv, ci = n(f'{p}_Ctau'), n(f'{p}_Cvalid'), n(f'{p}_cov_interp')
        if not (0 <= ct <= cv + 1e-12 <= ci + 2e-12 <= 1 + 3e-12):
            bad.append(f'{p}: C_tau {ct} <= C_valid {cv} <= cov {ci} <= 1 violated')
    if m['state'] == 'no_output':
        for k in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_cov_interp', 'loc_cov_interp'):
            if n(k) != 0.0:
                bad.append(f'no output but {k} = {n(k)}')
        for k in ('glob_pt_rmse', 'glob_pt_p95', 'glob_surf_rmse', 'glob_surf_p95', 'loc_pt_rmse', 'loc_surf_rmse'):
            if m.get(k) is not None:
                bad.append(f'no output but {k} defined')
    if (m['n_valid'] or 0) < 3:
        for k in ('glob_surf_rmse', 'loc_surf_rmse'):
            if m.get(k) is not None:
                bad.append(f'< 3 valid points but {k} defined')
    if m['n_valid'] > m['n_accepted'] or m['glob_n_valid'] > m['n_valid'] or m['loc_n_valid'] > m['n_valid']:
        bad.append('valid counts exceed their population')
    if m['id_correct'] + m['id_wrong'] + m['id_unconf'] != m['n_accepted']:
        bad.append('c + w + u != accepted')
    if nodes:
        for p, g in nodes.items():
            if not (g['n_supported'] <= g['n_interpolable'] <= g['n_nodes']):
                bad.append(f'{p}: supported <= interpolable <= nodes violated')
    return bad


def g4_no_output(M: pd.DataFrame) -> dict:
    E = _entry(M); rows = []
    for x in E.drop_duplicates('eval_key').itertuples():
        j = K.load_json(K.EVL / f'{x.eval_key}.json'); b = _semantics(j['metrics'], (j.get('extras') or {}).get('nodes_omega1'))
        rows.append(dict(eval_key=x.eval_key, family=x.family, surface=x.surface, sigma=x.sigma, strategy=x.strategy, state=j['state'], n_valid=j['metrics']['n_valid'], violations=b))
    syn = synthetic_structures()
    ok = all(not r['violations'] for r in rows) and syn['ok']
    return dict(ok=bool(ok), n_records=len(rows), n_no_output=int(sum(r['state'] == 'no_output' for r in rows)), n_lt3=int(sum((r['n_valid'] or 0) < 3 for r in rows)),
                violations=[r for r in rows if r['violations']], synthetic=syn)


def synthetic_structures() -> dict:

    import r2_eval as E
    import rdy_platform as P
    import s1r_eval as SE
    dom = E.domain('omega1'); sid = K.RK.SURF_MAP['M24_curv']; out = {}
    for n in (0, 2):
        Pv = np.column_stack([np.linspace(-1, 1, n), np.zeros(n), np.full(n, float(np.median(dom[f'z_{sid}'])))]) if n else np.zeros((0, 3))
        sg, z = SE.global_surface(Pv, dom, dom[f'z_{sid}']); nc = E.node_counts(Pv, np.ones(len(Pv), bool), 'M24_curv', dom)
        out[f'{n}_points'] = dict(Ctau=sg['Ctau'], Cvalid=sg['Cvalid'], cov=sg['cov_interp'], surf_rmse=sg['surf']['rmse'], n_interp=sg['n_interp'], z_all_nan=bool(np.isnan(z).all()), nodes=nc)
    ok = all(v['Ctau'] == 0 and v['Cvalid'] == 0 and v['cov'] == 0 and v['surf_rmse'] is None and v['z_all_nan'] and v['nodes']['glob']['n_supported'] == 0 for v in out.values())
    return dict(ok=bool(ok), **out)


def _rho_closed(V: np.ndarray) -> np.ndarray:
    A, B, C = V[:, 0], V[:, 1], V[:, 2]
    a2 = ((B - C) ** 2).sum(1); b2 = ((C - A) ** 2).sum(1); c2 = ((A - B) ** 2).sum(1)
    S = np.stack([a2, b2, c2], 1); o = np.argsort(S, axis=1)
    s_small, s_mid, s_big = np.take_along_axis(S, o, 1).T
    area2 = np.abs((B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1]))
    acute = s_big <= s_small + s_mid
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.sqrt(a2 * b2 * c2) / (2.0 * area2)
        L = np.sqrt(s_big)
        r1 = s_mid * L / (s_mid + s_big - s_small); r2 = s_small * L / (s_small + s_big - s_mid)
    return np.where(acute, R, np.maximum(r1, r2))


def _bilinear(mesh: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gx = np.asarray(mesh['X'][0, :], np.float64); gy = np.asarray(mesh['Y'][:, 0], np.float64); Z = np.asarray(mesh['Z'], np.float64)
    out = np.full(len(x), np.nan); ok = (x >= gx[0]) & (x <= gx[-1]) & (y >= gy[0]) & (y <= gy[-1])
    i = np.clip(np.searchsorted(gx, x[ok]) - 1, 0, len(gx) - 2); jy = np.clip(np.searchsorted(gy, y[ok]) - 1, 0, len(gy) - 2)
    tx = (x[ok] - gx[i]) / (gx[i + 1] - gx[i]); ty = (y[ok] - gy[jy]) / (gy[jy + 1] - gy[jy])
    out[ok] = (Z[jy, i] * (1 - tx) * (1 - ty) + Z[jy, i + 1] * tx * (1 - ty) + Z[jy + 1, i] * (1 - tx) * ty + Z[jy + 1, i + 1] * tx * ty)
    return out


def _indep_surface(P: np.ndarray, x, y, zref, area) -> dict:
    from scipy.spatial import Delaunay
    key = np.round(P[:, :2] / 1e-6).astype(np.int64); _, first = np.unique(key, axis=0, return_index=True); P = P[np.sort(first)]
    tri = Delaunay(P[:, :2]); q = np.column_stack([x, y]); s = tri.find_simplex(q); ins = s >= 0
    V = P[tri.simplices[s[ins]]]
    a, b, c = V[:, 0, :2], V[:, 1, :2], V[:, 2, :2]; qq = q[ins]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    l1 = ((qq[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (qq[:, 1] - a[:, 1])) / det
    l2 = ((b[:, 0] - a[:, 0]) * (qq[:, 1] - a[:, 1]) - (qq[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
    z = np.full(len(x), np.nan); z[ins] = (1 - l1 - l2) * V[:, 0, 2] + l1 * V[:, 1, 2] + l2 * V[:, 2, 2]
    rho = np.full(len(x), np.inf); rho[ins] = _rho_closed(P[tri.simplices[s[ins]]][:, :, :2])
    e = z - zref; interp = ins & np.isfinite(e); sup = interp & (rho <= K.D_SUPPORT); acc = sup & (np.abs(e) <= K.TAU); A = float(area.sum())
    return dict(Ctau=float(area[acc].sum() / A), Cvalid=float(area[sup].sum() / A), cov=float(area[interp].sum() / A), surf_rmse=(float(np.sqrt(np.mean(e[interp] ** 2))) if interp.any() else None),
                n_interp=int(interp.sum()), n_sup=int(sup.sum()), n_acc=int(acc.sum()))


def independent(eval_key: str, surface: str) -> dict:
    import r2_eval as E
    import rdy_platform as P
    j = K.load_json(K.EVL / f'{eval_key}.json'); m = j['metrics']; z = K.load_npz(K.EVL / f'{eval_key}.npz')
    pts = np.asarray(z['pts'], np.float64); acc = np.asarray(z['accepted'], bool); mesh = P.mesh(surface)
    fin = np.isfinite(pts).all(axis=1); zr = np.full(len(pts), np.nan); zr[fin] = _bilinear(mesh, pts[fin, 0], pts[fin, 1])
    valid = acc & fin & np.isfinite(zr) & (pts[:, 2] >= -16.0 - 1e-6) & (pts[:, 2] <= 4.0 + 1e-6)
    dom = E.domain('omega1'); sid = K.RK.SURF_MAP[surface]; dl = P.dom_local(surface)
    pz = np.nan_to_num(pts, nan=1e9)
    ix = np.rint((pz[:, 0] - dom['x0']) / dom['step']).astype(np.int64); iy = np.rint((pz[:, 1] - dom['y0']) / dom['step']).astype(np.int64)
    ing = np.zeros(len(pts), bool); okg = valid & (ix >= 0) & (ix < dom['nx']) & (iy >= 0) & (iy < dom['ny']); ing[okg] = dom['mask'][iy[okg], ix[okg]]
    e = pts[:, 2] - zr; Pv = pts[valid]
    g = _indep_surface(Pv, dom['x'], dom['y'], np.asarray(dom[f'z_{sid}'], np.float64), np.full(len(dom['x']), dom['step'] ** 2))
    lo = _indep_surface(Pv, dl['x'], dl['y'], np.asarray(dl['z_ref'], np.float64), np.asarray(dl['area'], np.float64))
    mine = dict(n_valid=int(valid.sum()), glob_n_valid=int(ing.sum()), glob_pt_rmse=float(np.sqrt(np.mean(e[ing] ** 2))) if ing.any() else None, glob_Ctau=g['Ctau'], glob_Cvalid=g['Cvalid'],
                glob_cov_interp=g['cov'], glob_surf_rmse=g['surf_rmse'], loc_Ctau=lo['Ctau'], loc_Cvalid=lo['Cvalid'], loc_cov_interp=lo['cov'], loc_surf_rmse=lo['surf_rmse'])
    tol = lambda k: 0 if k.endswith('n_valid') else (1e-6 if 'rmse' in k else 1e-12)
    diff = {k: (mine[k], m.get(k)) for k in mine if (mine[k] is None) != (m.get(k) is None) or (mine[k] is not None and abs(float(mine[k]) - float(m[k])) > tol(k))}
    nodes = (j.get('extras') or {}).get('nodes_omega1') or {}
    cnt = dict(glob_interp=(g['n_interp'], nodes.get('glob', {}).get('n_interpolable')), glob_sup=(g['n_sup'], nodes.get('glob', {}).get('n_supported')),
               loc_interp=(lo['n_interp'], nodes.get('loc', {}).get('n_interpolable')), loc_sup=(lo['n_sup'], nodes.get('loc', {}).get('n_supported')))
    cdiff = {k: v for k, v in cnt.items() if v[0] != v[1]}
    return dict(eval_key=eval_key, surface=surface, n_valid=mine['n_valid'], mine=mine, differences=diff, node_count_differences=cdiff, ok=bool(not diff and not cdiff))


def g5_independent(M: pd.DataFrame) -> dict:
    E = _entry(M); U = E.drop_duplicates('eval_key'); rows = []
    for x in U.itertuples():
        j = K.load_json(K.EVL / f'{x.eval_key}.json'); rows.append((x, j['metrics']['n_valid'], j['state']))
    ok_rows = [r for r in rows if r[1] >= 50]
    pick = []
    for fam in ('point', 'line'):
        c = sorted([r for r in ok_rows if r[0].family == fam], key=lambda r: (r[0].surface, r[0].sigma, r[0].strategy))
        if c:
            pick.append(c[0]); pick.append(c[-1])
    res = [independent(r[0].eval_key, r[0].surface) for r in pick]
    no_out = [r for r in rows if r[2] == 'no_output']
    real_no = None
    if no_out:
        r = no_out[0]; j = K.load_json(K.EVL / f'{r[0].eval_key}.json'); real_no = dict(eval_key=r[0].eval_key, violations=_semantics(j['metrics'], (j.get('extras') or {}).get('nodes_omega1')))
    ok = len([x for x in res if x['ok']]) >= 2 and all(x['ok'] for x in res) and (real_no is None or not real_no['violations'])
    return dict(ok=bool(ok), n_checked=len(res), checks=res, real_no_output=real_no, synthetic='see G4 (0 / 2 points)')


def _feat_change(fo: np.ndarray, fn: np.ndarray, band: int = BAND) -> dict:
    from scipy.spatial import cKDTree
    fo = np.asarray(fo, np.float64).reshape(-1, 2); fn = np.asarray(fn, np.float64).reshape(-1, 2)
    edge = lambda f: (np.minimum.reduce([f[:, 0], f[:, 1], 511 - f[:, 0], 511 - f[:, 1]]) < band)
    if not len(fo) or not len(fn):
        return dict(n_old=len(fo), n_new=len(fn), n_appeared_inner=int((~edge(fn)).sum()) if len(fn) else 0, n_lost_inner=int((~edge(fo)).sum()) if len(fo) else 0, n_moved_inner=0)
    d_n, _ = cKDTree(fo).query(fn); d_o, _ = cKDTree(fn).query(fo)
    moved = (d_n > 1e-6) & (d_n <= 0.5); app = d_n > 0.5; lost = d_o > 0.5
    return dict(n_old=len(fo), n_new=len(fn), n_same=int((d_n <= 1e-6).sum()), n_moved=int(moved.sum()), n_appeared=int(app.sum()), n_lost=int(lost.sum()),
                n_moved_inner=int((moved & ~edge(fn)).sum()), n_appeared_inner=int((app & ~edge(fn)).sum()), n_lost_inner=int((lost & ~edge(fo)).sum()))


def legacy_eval(lr: dict, M: pd.DataFrame) -> dict:

    import r2_eval as E
    import r2_run as R
    import rdy_decode as DC
    import rdy_eval as RE
    import rdy_platform as P
    row = M[M.image_key == lr['formal_image_key']].iloc[0].to_dict(); inp = R.inputs_of(row)
    lk = 'R2LE-' + K.digest(dict(legacy=lr['legacy_key'], method=row['method'], arm=row['arm'], boundary='legacy', domain=E.domain('omega1')['hash'], dec=row['decode_code'], ev=row['eval_code']), 24)
    p = K.OUT / 'legacy_ref' / f'{lk}.json'
    if K.complete(p, lk):
        return K.load_json(p)
    img = K.load_npz(K.OUT / 'legacy_ref' / f"{lr['legacy_key']}.npz")['image']
    out = DC.decode(row['method'], img, inp, R._rowlike(row)); dec = out['arms'][row['arm']]
    m, arr, _info = E.evaluate(row['family'], row['surface'], dec, inp, 'legacy', E.domain('omega1')); src = None
    if row['family'] == 'point':
        ref_lr = next(x for x in lr['_all'] if x['family'] == row['family'] and x['surface'] == row['surface'] and x['sigma'] == 0.0)
        ref = img if float(row['sigma']) == 0.0 else K.load_npz(K.OUT / 'legacy_ref' / f"{ref_lr['legacy_key']}.npz")['image']
        src, _c = RE.source_identity(img, ref, inp, dec, P.mesh(row['surface']))
    feats = np.asarray(arr.get('feats', np.zeros((0, 2))), np.float32)
    h = K.save_npz(p.with_suffix('.npz'), pts=np.asarray(arr['pts']), valid=np.asarray(arr['valid']), accepted=np.asarray(arr['accepted']), feats=feats)
    rec = dict(identity=lk, legacy_key=lr['legacy_key'], formal_image_key=lr['formal_image_key'], boundary='legacy', domain='omega1', metrics=m, source_identity=src, npz_sha256=h, complete=True, t_local=K.now_local())
    K.save_json(p, rec)
    return rec


def _find(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                return v
            x = _find(v, key)
            if x is not None:
                return x
    return None


def _threshold_swap(lr: dict, f) -> dict:

    import m4_lines as ML
    import sp_regression as R1
    p = R1.p4_params(); meta = K.load_json(K.RAW / f'{f.decode_key}.json')['meta']
    s_eff, thr_d = _find(meta, 'steger_scale_effective'), _find(meta, 'steger_threshold_design')
    new = K.load_npz(K.IMG / f"{lr['formal_image_key']}.npz")['image']; old = K.load_npz(K.OUT / 'legacy_ref' / f"{lr['legacy_key']}.npz")['image']
    so, sn = ML.estimate_noise_sigma(old)[0], ML.estimate_noise_sigma(new)[0]
    thr = lambda s: max(thr_d, ML.steger_noise_threshold(s, s_eff, p['steger_noise_coef'])) if s else thr_d
    to, tn = thr(so), thr(sn)
    import os
    le = next(x[:-5] for x in os.listdir(K.OUT / 'legacy_ref') if x.startswith('R2LE-') and x.endswith('.json') and K.load_json(K.OUT / 'legacy_ref' / x)['legacy_key'] == lr['legacy_key'])
    F_old, F_new = K.load_npz(K.OUT / 'legacy_ref' / f'{le}.npz')['feats'], K.load_npz(K.EVL / f'{f.eval_key}.npz')['feats']
    fo_o, fn_n = ML.detect_steger(old, s_eff, to)[0], ML.detect_steger(new, s_eff, tn)[0]
    fn_o, fo_n = ML.detect_steger(new, s_eff, to)[0], ML.detect_steger(old, s_eff, tn)[0]
    a, b = _feat_change(fo_o, fn_o), _feat_change(fn_n, fo_n)
    inner = lambda c: c.get('n_moved_inner', 0) + c.get('n_appeared_inner', 0) + c.get('n_lost_inner', 0)
    rep = bool(np.array_equal(np.asarray(fo_o, np.float32), F_old) and np.array_equal(np.asarray(fn_n, np.float32), F_new))
    return dict(sigma_hat_legacy=so, sigma_hat_new=sn, thr_legacy=float(to), thr_new=float(tn), thr_design=float(thr_d), stored_features_reproduced=rep,
                new_image_at_legacy_thr=a, legacy_image_at_new_thr=b, explained=bool(rep and inner(a) == 0 and inner(b) == 0))


def g6_screening(M: pd.DataFrame, legacy: dict) -> dict:
    rows = []; refs = legacy['refs']
    for lr in refs:
        L = legacy_eval(dict(lr, _all=refs), M)
        f = M[(M.image_key == lr['formal_image_key'])].iloc[0]; J = K.load_json(K.EVL / f'{f.eval_key}.json'); N = J['metrics']; O = L['metrics']
        zf = K.load_npz(K.EVL / f'{f.eval_key}.npz'); zl = K.load_npz(K.OUT / 'legacy_ref' / f"{L['identity']}.npz")
        fc = _feat_change(zl['feats'], zf['feats'])
        def dd(k, s=1.0):
            a, b = O.get(k), N.get(k)
            return None if (a is None or b is None) else (float(b) - float(a)) * s
        ex = []
        for k in ('glob_Ctau', 'loc_Ctau'):
            if dd(k, 100) is not None and abs(dd(k, 100)) > SCREEN['Ctau_pp']:
                ex.append(f'{k} {dd(k, 100):+.3f} pp')
        for k in ('glob_pt_rmse', 'glob_surf_rmse', 'loc_pt_rmse', 'loc_surf_rmse'):
            if dd(k) is not None and abs(dd(k)) > SCREEN['rmse_mm']:
                ex.append(f'{k} {dd(k):+.4f} mm')
        for k in ('glob_pt_p95', 'glob_surf_p95'):
            if dd(k) is not None and abs(dd(k)) > SCREEN['p95_mm']:
                ex.append(f'{k} {dd(k):+.4f} mm')
        if dd('unconf_share', 100) is not None and abs(dd('unconf_share', 100)) > SCREEN['unconf_pp']:
            ex.append(f"unconf_share {dd('unconf_share', 100):+.3f} pp")
        if N['id_wrong'] > O['id_wrong']:
            ex.append(f"new wrong identities {O['id_wrong']} -> {N['id_wrong']}")
        if N['state'] != O['state']:
            ex.append(f"state {O['state']} -> {N['state']}")
        if f.family == 'point' and L.get('source_identity') and J['extras'].get('source_identity'):
            if J['extras']['source_identity']['wrong'] > L['source_identity']['wrong']:
                ex.append('new source-identity wrong')
        interior = fc.get('n_moved_inner', 0) + fc.get('n_appeared_inner', 0) + fc.get('n_lost_inner', 0)
        img_inner = image_diff(K.load_npz(K.OUT / 'legacy_ref' / f"{lr['legacy_key']}.npz")['image'], K.load_npz(K.IMG / f"{lr['formal_image_key']}.npz")['image'])['n_changed_inner']
        swap = _threshold_swap(lr, f) if (ex and interior > 0 and f.family == 'line') else None
        if not ex:
            expl, why = None, None
        elif interior == 0:
            expl, why = True, 'only border-band (< 8 px) features changed'
        elif swap is not None and swap['explained'] and img_inner == 0:
            expl, why = True, (f"image-global noise threshold of the registered line detector (design_or_noise): sigma_hat {swap['sigma_hat_legacy']} -> {swap['sigma_hat_new']} "
                               f"(uint8 quantile step from the border-band pixels) moves the Steger threshold {swap['thr_legacy']:.4f} -> {swap['thr_new']:.4f}; with the thresholds swapped "
                               'the interior features of both images are identical and the stored feature sets are reproduced exactly; interior pixels identical')
        else:
            expl, why = False, 'INTERIOR feature changes not explained'
        rows.append(dict(family=f.family, surface=f.surface, sigma=float(f.sigma), legacy_key=lr['legacy_key'], formal_eval_key=f.eval_key, state_old=O['state'], state_new=N['state'],
                         d_glob_Ctau_pp=dd('glob_Ctau', 100), d_loc_Ctau_pp=dd('loc_Ctau', 100), d_glob_surf_rmse_mm=dd('glob_surf_rmse'), d_glob_pt_rmse_mm=dd('glob_pt_rmse'),
                         n_valid_old=O['n_valid'], n_valid_new=N['n_valid'], features=fc, exceedances=ex, interior_feature_changes=int(interior), image_changed_inner=int(img_inner),
                         threshold_swap=swap, explained=expl, explanation=why))
    unexplained = [r for r in rows if r['exceedances'] and not r['explained']]
    return dict(ok=bool(not unexplained), n=len(rows), n_exceeding=int(sum(bool(r['exceedances']) for r in rows)), unexplained=unexplained, pairs=rows, lines=SCREEN)
