# Evaluate layout reconstructions on the selected boundary-domain grid.

from __future__ import annotations

import numpy as np
import pandas as pd

import r2_common as K

_DOM = {}


def domain(name: str) -> dict:

    if name not in _DOM:
        if name == 'omega0':
            import s1r_common as S1K
            _DOM[name] = S1K.global_domain(0.5)
        else:
            z = K.PIX / 'domains' / f'{name}.npz'; j = K.load_json(K.PIX / 'domains' / f'{name}.json')
            if K.sha_file(z) != K.ANCHORS[K.rel(z)] or K.sha_file(z) != j['npz_sha256']:
                raise K.Stop('input', f'domain {name} differs from the task anchor')
            d = K.load_npz(z)
            _DOM[name] = dict(d, step=float(j['step']), x0=float(j['x0']), y0=float(j['y0']), nx=int(j['nx']), ny=int(j['ny']), hash=j['hash'], rec=dict(n_nodes=j['n_nodes']))
    return _DOM[name]


def _ref_fn(boundary):
    import r2_eval_snap as SN

    def f(mesh, spec, pattern):
        return SN.line_reference_table(mesh, spec, 1.0, pattern=np.asarray(pattern, np.float32), boundary=boundary)
    return f


def evaluate(family: str, surface: str, dec: dict, inp: dict, boundary: str, dom_glob: dict):

    import rdy_eval as RE
    import rdy_platform as P
    import r2_eval_snap as SN
    ctx = RE.ctx_eval(dict(family=family, section='S4'), inp); sid = K.RK.SURF_MAP[surface]; mesh = P.mesh(surface); dl = P.dom_local(surface)
    if family == 'line':
        return RE.evaluate_lines_table(sid, dec, ctx, mesh, dl, dom_glob, sid == 'A_curv_gauss', truth_fn=RE._truth_table, ref_fn=_ref_fn(boundary))
    m, arrays, info, _z = SN.evaluate_case(sid, 'point', dec, ctx, mesh, dl, dom_glob, sid == 'A_curv_gauss', boundary=boundary)
    return m, arrays, info


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def regression(n_per_family: int = 6) -> dict:

    import rdy_decode as DC
    import rdy_inputs as RI
    A = K.S345 / 'run'; P_ = pd.read_csv(K.S345 / 'plan_s345.csv', low_memory=False); P_ = P_[P_.set_formal == 'S4-reduced']
    C = pd.read_csv(K.S345 / 'results' / 'S4-reduced_cases.csv', low_memory=False)
    out = []
    for fam in ('point', 'line'):
        c = C[C.family == fam].sort_values('decode_key')
        pick = pd.concat([c[c.state == 'no_output'].head(2), c[c.state != 'no_output'].head(n_per_family - 2)])
        for r in pick.itertuples():
            row = P_[P_.decode_key == r.decode_key].iloc[0].to_dict(); inp = RI.build(row)
            method = 'D0' if fam == 'point' else 'P4TABLE'
            rp = A / 'raw' / f'{r.image_key}__{method}.json'; out_ = DC.unpack(K.load_npz(rp.with_suffix('.npz')), K.load_json(rp)['meta'])
            dec = out_['arms'][row['arm']]
            stored = K.load_json(A / 'eval' / f'{r.decode_key}.json')['metrics']
            m, _a, _i = evaluate(fam, r.surface, dec, inp, 'legacy', domain('omega0'))
            diffs = {k: (stored[k], m.get(k)) for k in stored if (_num(stored[k]) and (not _num(m.get(k)) or abs(float(stored[k]) - float(m[k])) > 1e-12)) or (not _num(stored[k]) and stored[k] != m.get(k) and not isinstance(stored[k], (dict, list)))}
            ms, _a2, _i2 = evaluate(fam, r.surface, dec, inp, K.BOUNDARY, domain('omega0'))
            att = {k: (m.get(k), ms.get(k)) for k in ('id_correct', 'id_wrong', 'id_unconf', 'n_confirmable_features', 'img_n_expected', 'img_n_recovered', 'glob_Ctau', 'loc_Ctau') if m.get(k) != ms.get(k)}
            out.append(dict(family=fam, decode_key=r.decode_key, state=r.state, n_metrics=len(stored), n_diff=len(diffs), diffs=dict(list(diffs.items())[:5]), symmetric_identity_changes=att))
    return dict(cases=out, ok=bool(all(x['n_diff'] == 0 for x in out)), n=len(out))


def identity_tracing() -> dict:

    import l01_common as C0
    import pb_m3 as PM
    import s12_common as S12K
    pix = K.load_json(K.PIX / 'run.json'); g = C0.geom(); S = S12K.s2(); rows = []
    for c in pix['plan']['cases']:
        if c['section'] != 'S12':
            continue
        jp = (K.SR / 's1_real01' / 'decodes' / f"{c['s1_decode_key']}.json" if c['s12_source'] == 's1' else
              (K.SR / 's2_prepare' / 'run' / 'eval' / f"{c['s2_decode_key']}.json" if c['s12_source'] == 's2' else K.SR / 's12_resource01' / 'eval' / f"{c['old_decode_keys'][0]}.json"))
        A = K.load_npz(jp.with_suffix('.npz')); acc = A['accepted'].astype(bool)
        if c['family'] == 'point':
            pf = np.asarray(A['point_feature'], np.int64); feats = np.asarray(A['feats'], np.float64)
            fa = feats[pf[acc & (pf >= 0)]]; mapping = 'point_feature'
        else:
            if len(A['feats']) != len(acc):
                raise K.Stop('check', f"{c['source_key']}: line features / points not aligned")
            fa = np.asarray(A['feats'], np.float64)[acc]; mapping = 'aligned'
        mesh = S.build_mesh(c['input_row']['platform_surface'])
        hit, P3 = C0.cast(fa, g['cam'], mesh)
        uv, in_leg = PM.project_to_camera(np.asarray(P3, np.float64), g['K'], g['proj'], C0.IMG, C0.IMG, 'legacy'); _, in_sym = PM.project_to_camera(np.asarray(P3, np.float64), g['K'], g['proj'], C0.IMG, C0.IMG, K.BOUNDARY)
        strip = lambda a: ((a >= -0.5) & (a < 0)) | ((a >= 511.5) & (a < 512))
        st = hit & (strip(uv[:, 0]) | strip(uv[:, 1]))
        if c['family'] == 'point':
            rule = 'evaluate_case: confirmable = hit & projector-inside & d <= r + 1 px'
            changed = st & (in_leg != in_sym)
            dmin = np.asarray(A['feat_truth_dist_px'], np.float64)[pf[acc & (pf >= 0)]]; r_px = float(c['size']) / S.SL.mmpp() + 1.0
            label_change = int((changed & (dmin <= r_px)).sum())
        else:
            rule = 'l01_truth.truth_identity: projector-inside -0.5 <= u <= W - 0.5 already (closed); unchanged except the endpoint u = 511.5'
            changed = st & (uv[:, 0] == 511.5) | st & (uv[:, 1] == 511.5); label_change = int(changed.sum())
        rows.append(dict(source_key=c['source_key'], pkg=c['pkg'], family=c['family'], mapping=mapping, n_accepted=int(acc.sum()), n_mapped=int(len(fa)), n_strip=int(st.sum()),
                         n_inside_flip=int(changed.sum()), n_label_change=label_change, rule=rule))
    R = pd.DataFrame(rows)
    return dict(n_cases=len(R), n_mapped_all=bool((R.n_mapped == R.n_accepted).all()), n_accepted=int(R.n_accepted.sum()), n_strip=int(R.n_strip.sum()), n_label_change=int(R.n_label_change.sum()),
                per_case=R.to_dict('records'))


def node_counts(pts: np.ndarray, valid: np.ndarray, surface: str, dom_glob: dict) -> dict:

    import rdy_platform as P
    from surface_eval_r3 import surface_eval_r3_dev
    Pv = np.asarray(pts, np.float64)[np.asarray(valid, bool)]; sid = K.RK.SURF_MAP[surface]; dl = P.dom_local(surface)
    doms = dict(glob=dict(x=dom_glob['x'], y=dom_glob['y'], z_ref=np.asarray(dom_glob[f'z_{sid}'], np.float64), area=np.full(len(dom_glob['x']), dom_glob['step'] ** 2), hash=dom_glob['hash']),
                loc=dict(x=dl['x'], y=dl['y'], z_ref=np.asarray(dl['z_ref'], np.float64), area=dl['area'], hash=dl['hash']))
    out = {}
    for nm, D in doms.items():
        n = int(len(D['x']))
        if len(Pv) < 3:
            out[nm] = dict(n_nodes=n, n_interpolable=0, n_supported=0, sup_surf_rmse=None, sup_surf_p95=None, triangulation='fewer than 3 valid points'); continue
        ev = surface_eval_r3_dev(Pv, None, D, [K.D_SUPPORT], [K.TAU], support_rule='exact'); g = ev['subdomains']['global']; bd = g['by_d'][str(float(K.D_SUPPORT))]
        out[nm] = dict(n_nodes=int(g['n_nodes']), n_interpolable=int(g['n_interpolable']), n_supported=int(bd['n_supported']), sup_surf_rmse=bd['rmse'], sup_surf_p95=bd['p95'], triangulation=ev['triangulation'])
    return out
