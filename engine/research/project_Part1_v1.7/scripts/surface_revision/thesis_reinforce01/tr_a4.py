# Reclassify line identities using every illuminated raster line, including clipped lines.

from __future__ import annotations

import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import tr_common as K

TOL_PX = 1.0


def _family(olines: dict) -> dict:
    o = olines
    return dict(n=np.asarray(o['normal_uv'], np.float64), c=np.asarray(o['centre_uv'], np.float64), s=float(o['spacing_px']), w=float(o['width_px']),
                phase=float(o['phase_frac']), ids=[int(k) for k in o['ids']])


def scan_raster(binary, olines: dict) -> dict:

    B = np.asarray(binary) > 0; fam = _family(olines)
    vv, uu = np.nonzero(B)
    q = (np.column_stack([uu, vv]).astype(np.float64) - fam['c']) @ fam['n']
    r = q / fam['s'] - fam['phase']; k = np.round(r).astype(np.int64); d = np.abs(r - k) * fam['s']
    out = {}
    for kk in np.unique(k):
        s = k == kk
        out[int(kk)] = dict(lit_px=int(s.sum()), u=[int(uu[s].min()), int(uu[s].max())], v=[int(vv[s].min()), int(vv[s].max())], dist_max_px=float(d[s].max()))
    ids = set(fam['ids'])
    extra = sorted(k for k in out if k not in ids)
    return dict(lines=out, ids=sorted(ids), extra=extra, n_lit_px=int(B.sum()), lit_px_extra=int(sum(out[k]['lit_px'] for k in extra)),
                registered_without_lit=sorted(k for k in ids if k not in out), band_px=fam['w'] / 2 + TOL_PX)


def s12_rasters() -> pd.DataFrame:
    V = K.read_csv(K.S12 / 'results' / 'views.csv')
    L = V[(V.family == 'line') & (V.surface_class == 'main')].copy()
    return L


S3_MANIFEST = K.ROOT / 'outputs/surface_revision/s3_code04/formal/manifest.csv'


def s3_manifest() -> pd.DataFrame:
    M = K.read_csv(S3_MANIFEST)
    return M[M.family == 'lines'].copy()


_RAST: dict = {}


def raster_s12(u: dict) -> tuple:
    import s12_case as SC12
    if u['raster_key'] not in _RAST:
        R = SC12.raster_of(u); _RAST[u['raster_key']] = (np.asarray(R['binary'], np.float32), R['meta_full']['olines'])
    return _RAST[u['raster_key']]


def raster_s3(row: dict) -> tuple:
    import sys
    q = str(K.SR / 's3_code04')
    if q not in sys.path:
        sys.path.append(q)
    import c3_raster as CR
    import c4_parity as PA
    if row['raster_key'] not in _RAST:
        R = PA.build('lines', str(row['layout']), row['pattern'], dict(phase_frac=float(row['phase_frac'])), CR.levels(), int(row['code_parity']))
        if CR.raster_key(R['binary']) != row['raster_key']:
            raise K.Stop('technical', f"S3 raster {row['raster_key']} not reproduced")
        _RAST[row['raster_key']] = (np.asarray(R['binary'], np.float32), R['meta_full']['olines'])
    return _RAST[row['raster_key']]


def scan_all() -> pd.DataFrame:
    rows = []
    L = s12_rasters()
    for rk, g in L.groupby('raster_key'):
        u = g.iloc[0].to_dict(); B, o = raster_s12(u); s = scan_raster(B, o)
        rows.append(dict(source='S12', raster_key=rk, section=';'.join(sorted(set(g.section))), layout=u['layout'], size_mm=u['size_mm'], phase_label=u['phase_label'],
                         n_views=len(g), ids=json.dumps(s['ids']), extra=json.dumps(s['extra']), lit_px_extra=s['lit_px_extra'], n_lit_px=s['n_lit_px'],
                         extra_detail=json.dumps({k: s['lines'][k] for k in s['extra']}), registered_without_lit=json.dumps(s['registered_without_lit']), band_px=s['band_px']))
    M = s3_manifest()
    for rk, g in M.groupby('raster_key'):
        r = g.iloc[0].to_dict(); B, o = raster_s3(r); s = scan_raster(B, o)
        rows.append(dict(source='S3', raster_key=rk, section='S3', layout=str(r['layout']), size_mm=r['pattern'], phase_label=r['phase_id'], n_views=len(g),
                         ids=json.dumps(s['ids']), extra=json.dumps(s['extra']), lit_px_extra=s['lit_px_extra'], n_lit_px=s['n_lit_px'],
                         extra_detail=json.dumps({k: s['lines'][k] for k in s['extra']}), registered_without_lit=json.dumps(s['registered_without_lit']), band_px=s['band_px']))
    return pd.DataFrame(rows)


_MESH: dict = {}


def _mesh(surface):
    import rdy_platform as P
    if surface not in _MESH:
        _MESH[surface] = P.mesh(surface)
    return _MESH[surface]


def _geom(mesh, feats, olines):
    import l01_common as C
    fam = _family(olines); g = C.geom()
    feats = np.asarray(feats, np.float64).reshape(-1, 2)
    hit, pts = C.cast(feats, g['cam'], mesh)
    uv, _ = C.project(pts, g['proj'])
    inside = (uv[:, 0] >= -0.5) & (uv[:, 0] <= C.IMG - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= C.IMG - 0.5)
    q = (uv - fam['c']) @ fam['n']; r = q / fam['s'] - fam['phase']; k = np.round(r); dist = np.abs(r - k) * fam['s']
    return dict(hit=hit, in_proj=hit & inside, uv=uv, k=np.where(np.isfinite(k), k, -10 ** 6).astype(np.int64), dist=dist, band=fam['w'] / 2 + TOL_PX, ids=fam['ids'])


def _lit_at(B, uv) -> np.ndarray:

    out = np.zeros(len(uv), bool); fin = np.isfinite(uv).all(axis=1)
    if fin.any():
        u = np.rint(uv[fin, 0]).astype(int); v = np.rint(uv[fin, 1]).astype(int); res = np.zeros(int(fin.sum()), bool)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                uu, vv = u + du, v + dv; ok = (uu >= 0) & (uu < B.shape[1]) & (vv >= 0) & (vv < B.shape[0])
                res[ok] |= B[vv[ok], uu[ok]]
        out[fin] = res
    return out


def _counts(acc, conf, k, lid, sel=None) -> dict:
    s = acc if sel is None else (acc & sel)
    c = int((s & conf & (lid == k)).sum()); w = int((s & conf & (lid != k)).sum()); u = int((s & ~conf).sum())
    return dict(c=c, w=w, u=u)


def reidentify(mesh, feats, olines, extra, B, acc, lid, conf_old, k_old, valid, in_glob, in_loc) -> dict:
    G = _geom(mesh, feats, olines)
    conf_rep = G['in_proj'] & (G['dist'] <= G['band']) & np.isin(G['k'], G['ids'])
    closure = bool(np.array_equal(conf_rep, conf_old) and np.array_equal(np.where(conf_rep, G['k'], 0), np.where(conf_old, k_old, 0)))
    lit = _lit_at(B, G['uv'])
    new_lab = (~conf_old) & G['in_proj'] & (G['dist'] <= G['band']) & np.isin(G['k'], extra) & lit
    conf_new = conf_old | new_lab; k_new = np.where(new_lab, G['k'], k_old)
    out = dict(closure_old_identity=closure, n_features=int(len(feats)), n_new_labels=int(new_lab.sum()), n_new_labels_accepted=int((new_lab & acc).sum()))
    for name, sel in (('all', None), ('glob', valid & in_glob), ('loc', valid & in_loc)):
        o = _counts(acc, conf_old, k_old, lid, sel); n = _counts(acc, conf_new, k_new, lid, sel)
        out.update({f'{name}_old_{x}': o[x] for x in 'cwu'}); out.update({f'{name}_new_{x}': n[x] for x in 'cwu'})
    moved = new_lab & acc
    out['lines_involved'] = json.dumps(sorted(int(x) for x in np.unique(G['k'][moved]))) if moved.any() else '[]'
    out['moved_decoded_ids'] = json.dumps(sorted(int(x) for x in np.unique(lid[moved]))) if moved.any() else '[]'
    out.update(_classes(G, acc & ~conf_new, lit, extra))
    return out


def _classes(G, unc, lit, extra) -> dict:

    h, ip, d, k = G['hit'], G['in_proj'], G['dist'], G['k']
    lit_line = np.isin(k, list(G['ids']) + list(extra))
    a = unc & (~h | ~ip)
    b = unc & ip & (d <= G['band']) & lit_line
    c = unc & ip & ~((d <= G['band']) & lit_line)
    return dict(u_boundary_or_overlap_uncertain=int(a.sum()), u_geometric_unique_not_confirmed=int(b.sum()), u_no_lit_feature=int(c.sum()),
                u_no_lit_feature_but_lit_at_point=int((c & lit).sum()))


def _work_s12(args):
    row, extra = args
    try:
        z = K.load_npz(K.ROOT / row['eval_record'].replace('.json', '.npz'))
        B, o = raster_s12(row)
        acc = z['accepted'].astype(bool); lid = z['line_id'].astype(np.int64)
        st = dict(stored_c=int((acc & z['point_correct']).sum()), stored_w=int((acc & z['point_wrong']).sum()), stored_u=int((acc & z['point_unconf']).sum()))
        out = reidentify(_mesh(row['surface']), z['feats'], o, extra, np.asarray(B) > 0, acc, lid, z['confirmable'].astype(bool), z['truth_k'].astype(np.int64),
                         z['valid'].astype(bool), z['in_global'].astype(bool), z['in_local'].astype(bool))
        out['closure_stored_cwu'] = bool(st['stored_c'] == out['all_old_c'] and st['stored_w'] == out['all_old_w'] and st['stored_u'] == out['all_old_u'])
        return dict(source='S12', view_id=row['view_id'], input_key=row['input_key'], eval_key=row['eval_key'], raster_key=row['raster_key'], section=row['section'], regime=row['regime'],
                    surface=row['surface'], layout=row['layout'], size_mm=row['size_mm'], phase_label=row['phase_label'], sigma=row['sigma'], rep=row['rep'], arm='P4', **out)
    except Exception as e:
        return dict(source='S12', view_id=row['view_id'], error=repr(e)[:300])


def _work_s3(args):
    row, extra = args
    try:
        import s1r_common as S1K
        import rdy_platform as P
        ev = K.ROOT / 'outputs/surface_revision/s3_code07/formal_run/records'
        z = K.load_npz(ev / 'eval' / f"{row['eval_key']}.npz"); d = K.load_npz(ev / 'dec' / f"{row['decode_key']}.npz")
        feats = d['features']; B, o = raster_s3(row)
        if len(feats) != len(z['acc']):
            raise K.Stop('technical', 'feature / point count mismatch')
        acc = z['acc'].astype(bool); lid = z['lid'].astype(np.int64); valid = z['valid'].astype(bool); pts = z['pts'].astype(np.float64)
        mesh = _mesh(row['surface'])
        import l01_truth as T
        ti = T.truth_identity(mesh, feats, o)
        cor, wr, un = (z['bridge'][i].astype(bool) for i in range(3))
        dom = P.dom_global(); in_g = valid & S1K.in_global(pts[:, :2], dom); in_l = valid & S1K.in_local(pts[:, :2])
        out = reidentify(mesh, feats, o, extra, np.asarray(B) > 0, acc, lid, ti['confirmable'], ti['k'], valid, in_g, in_l)
        out['closure_stored_cwu'] = bool(int((acc & cor).sum()) == out['all_old_c'] and int((acc & wr).sum()) == out['all_old_w'] and int((acc & un).sum()) == out['all_old_u'])
        return dict(source='S3', view_id=f"{row['image_key']}|{row['arm']}", input_key=row['image_key'], eval_key=row['eval_key'], raster_key=row['raster_key'], section='S3',
                    regime=row['regime'], surface=row['surface'], layout=str(row['layout']), size_mm=row['pattern'], phase_label=row['phase_id'], sigma=row['sigma'], rep=row['rep'],
                    arm=row['arm'], **out)
    except Exception as e:
        return dict(source='S3', view_id=f"{row.get('image_key')}|{row.get('arm')}", error=repr(e)[:300])


def views_s3(affected: set) -> pd.DataFrame:
    E = K.read_csv(K.ROOT / 'outputs/surface_revision/s3_code07/plan/expected_keys_formal.csv')
    E = E[(E.family == 'lines') & (E.layer == 'eval')].rename(columns={'key': 'eval_key'})
    M = s3_manifest()[['image_key', 'surface', 'layout', 'pattern', 'phase_id', 'phase_frac', 'code_parity', 'sigma', 'rep', 'raster_key', 'regime']]
    X = E.merge(M, on='image_key', how='left', validate='many_to_one')
    if X.raster_key.isna().any():
        raise K.Stop('refused', f'{int(X.raster_key.isna().sum())} S3 line evaluation keys without a manifest row')
    return X[X.raster_key.isin(affected)]


def run(workers: int = 12) -> dict:
    t0 = time.time()
    S = scan_all(); S['affected'] = S.extra.map(lambda x: len(json.loads(x)) > 0)
    aff = S[S.affected]; ex = {r.raster_key: json.loads(r.extra) for r in aff.itertuples()}
    L = s12_rasters(); L = L[L.raster_key.isin(ex)]
    X = views_s3(set(ex))
    jobs12 = [(r, ex[r['raster_key']]) for r in L.to_dict('records')]
    jobs3 = [(r, ex[r['raster_key']]) for r in X.to_dict('records')]
    with Pool(workers) as pool:
        R = pool.map(_work_s12, jobs12, chunksize=8) + pool.map(_work_s3, jobs3, chunksize=8)
    T = pd.DataFrame(R)
    errs = T[T.get('error', pd.Series(dtype=object)).notna()] if 'error' in T else T.iloc[0:0]
    out = dict(rasters_scanned=int(len(S)), rasters_affected=int(len(aff)), by_source=S.groupby('source').affected.agg(['size', 'sum']).to_dict(),
               views_s12=len(jobs12), views_s3=len(jobs3), errors=int(len(errs)), minutes=round((time.time() - t0) / 60, 1), t=K.now())
    if len(errs) == 0:
        out.update(closure_old_identity=bool(T.closure_old_identity.all()), closure_stored_cwu=bool(T.closure_stored_cwu.all()))
    return dict(summary=out, scan=S, views=T)


if __name__ == '__main__':
    import sys
    r = run()
    tmp = sys.argv[1] if len(sys.argv) > 1 else '.'
    r['scan'].to_csv(f'{tmp}/a4_scan.csv', index=False); r['views'].to_csv(f'{tmp}/a4_views.csv', index=False)
    print(json.dumps(K.clean(r['summary']), ensure_ascii=False, default=str))
