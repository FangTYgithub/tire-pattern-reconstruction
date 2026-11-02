# Diagnose clipped lines, candidate ambiguity and layout-dependent failures.

"""S4-REAL-03: zero-imaging diagnosis and closure of section 4 (S4-REAL-02 data, read only).

python r3_diag.py check | lines | dots | statements | figs | final | all
  check       S4-REAL-02 anchors (6 files, reference hashes) + effective freeze chain (r2_main.preflight, read only)
  lines       clipped-line diagnosis: registered tables of all L6.3 layouts x phases (no imaging); geometric source of the stored
              unconfirmable line features (camera feature -> true surface -> projector point -> nearest table line / lit raster pixel)
              for all H0 images (the only phase with a clipped line) and every other image with accepted unconfirmable points;
              all other line images from stored metrics; legacy entry references (same geometry on the stored legacy features)
  dots        sigma >= 80 dot images (384 unique): stored detection statistics (Otsu threshold, component reasons, area-weighted median,
              relative-area selection codes, candidate areas) and the comparison block of the exception (same noise field S0, S1 at
              sigma 0 / 60 / 100); 0 detection recomputations
  statements  numbers behind the corrected statements (local positive cells, per-case ranges, error spread, tails)
  figs        two German diagnostic figures
  final       read inputs unchanged (hashes start / end), output hashes
No imaging, no decoding, no re-evaluation: the official identities / metrics are only read; diagnostic labels are separate columns.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'surface_revision' / 's4_real02'))
import r2_common as K

OUT = ROOT / 'outputs' / 'surface_revision' / 's4_real03'
RUN = OUT / 'run.json'
FIGS = OUT / 'figures'
TASK = ROOT.parents[1] / 'protocols' / 'S4-REAL-03_protocol.md'
reference = {'run.json': 'd9020731a1dabc847df284906f6ae528ef9180ed11ba85efec73c18db8c06e38', 'manifest.csv': 'c9689e279f089ba7f42e337a1300394131fee7a3add8115a9392e7d9956980a1',
         'case_metrics.csv': '9c10c7fcc96649b59fd97463e1017f26ec3026385264c9481ab0a924e78e0bf6', 'paired_summary.csv': '667515e83ca18aa7cf142cf855b76bc796947af75861f28eceae49d3ee261618',
         'layouts.json': '20f28d4bc275c615224a9ae89135028b9815d96f3444947ee085a40a58728f16', 'report_zh.md': '00ed41b0ab81923b7ac1c6676ea6e233ce63de867907a645666d43b9c2da364d'}
LINE_ANCHOR = dict(eval_key='R2E-fbaf24dfc297c37eeabea499', decode_key='R2D-681efa20d9b0ad6713b08f15')
DOT_ANCHOR = dict(eval_key='R2E-5c3d44dc96f4b23c6cc5494d', image_key='R2I-51835c82a47cbaa7710e5b57')


def _guard(p: Path) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in q.parents and q != OUT.resolve():
        raise SystemExit(f'write outside {OUT} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def save_json(p, obj):
    q = _guard(p); t = q.with_name(q.name + '.tmp'); t.write_text(json.dumps(K.clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'); os.replace(t, q)


def save_csv(df: pd.DataFrame, p):
    q = _guard(p); t = q.with_name(q.name + '.tmp'); df.to_csv(t, index=False, encoding='utf-8-sig'); os.replace(t, q)


def sha(p) -> str:
    return K.sha_file(p)


def run_load() -> dict:
    return json.loads(RUN.read_text(encoding='utf-8')) if RUN.exists() else dict(task='S4-REAL-03', task_book=str(TASK),  acceptance='reference', created=K.now_local())


def run_put(section: str, obj) -> None:
    r = run_load(); r[section] = obj; r['updated'] = K.now_local(); save_json(RUN, r)


def read_log(paths) -> dict:

    return {K.rel(p): sha(p) for p in paths}


def check() -> dict:
    import r2_main as RM
    S2 = K.OUT
    anchors = {f: dict(expected=h, actual=sha(S2 / f)) for f, h in reference.items()}
    ok = all(v['expected'] == v['actual'] for v in anchors.values())
    pf = RM.preflight()
    rr = K.load_json(K.RUN)
    code = RM.effective_code(rr)
    me = {K.rel(p): sha(p) for p in sorted(HERE.glob('*.py'))}
    out = dict(ok=bool(ok and pf['ok']), anchors=anchors, s4_real02_preflight=pf, s4_real02_effective_code_files=len(code), s4_real02_effective_code_digest=K.digest(code, 32),
               s4_real02_refreezes=pf['refreezes'], s4_real03_code=me, t_local=K.now_local())
    run_put('check', out)
    if not out['ok']:
        raise SystemExit('anchor / freeze check failed')
    return out


_MESH = {}


def _mesh(surface):
    import rdy_platform as P
    if surface not in _MESH:
        _MESH[surface] = P.mesh(surface)
    return _MESH[surface]


def classify_line_features(mesh, feats, spec, binary):

    import sp2_lines_table as LT2
    import sp_lines_table as LT
    feats = np.asarray(feats, np.float64).reshape(-1, 2); n = len(feats)
    ti = LT2.truth_identity_table(mesh, feats, spec); fam = LT.line_family_table(spec)
    uv = ti['proj_uv']; fin = np.isfinite(uv).all(axis=1)
    q = np.where(fin, (np.nan_to_num(uv) - fam['c']) @ fam['n'], np.nan)
    k1, d1, k2, d2 = LT2.nearest_table_line(fam, q)
    band = ti['band_px']; clipped = np.array(sorted(int(k) for k in spec['clipped_lines']), np.int64); ids = np.asarray(spec['ids'], np.int64)
    row_in = fin & (uv[:, 1] >= -0.5) & (uv[:, 1] <= 511.5)
    cls = np.full(n, 'other', object)
    hit = ti['hit']
    cls[~hit] = 'no_hit'
    cls[hit & ~fin] = 'no_projection'
    w1 = hit & fin & (d1 <= band); w2 = hit & fin & (d2 <= band)
    cls[hit & fin & ~w1] = 'between_lines'
    cls[w1 & w2] = 'multi'
    single = w1 & ~w2
    cl = single & np.isin(k1, clipped)
    cls[cl & row_in] = 'clipped_line'; cls[cl & ~row_in] = 'clipped_line_end'
    cp = single & np.isin(k1, ids)
    cls[cp & ~ti['in_proj']] = 'complete_line_out'
    cls[cp & ti['in_proj']] = 'complete_line_in'
    B = np.asarray(binary) > 0
    uu = np.clip(np.rint(np.nan_to_num(uv[:, 0], nan=-9)).astype(int), 0, 511); vv = np.clip(np.rint(np.nan_to_num(uv[:, 1], nan=-9)).astype(int), 0, 511)
    lit = fin & B[vv, uu]
    return dict(cls=cls, k1=k1, d1=d1, d2=d2, proj_uv=uv, xyz=ti['xyz'], lit=lit, band=band, official_confirmable=ti['confirmable'])


def line_image(row: dict) -> dict:
    import r2_run as R
    from m4_lines import STATUS
    inp = R.inputs_of(row); spec = inp['table']['spec']
    z = K.load_npz(K.EVL / f"{row['eval_key']}.npz"); j = K.load_json(K.EVL / f"{row['eval_key']}.json"); m = j['metrics']
    conf = z['confirmable'].astype(bool); acc = z['accepted'].astype(bool); st = z['status'].astype(int)
    pts = z['pts'].astype(np.float64); zr = z['zr'].astype(np.float64); e = pts[:, 2] - zr
    idx = np.flatnonzero(~conf)
    out = dict(eval_key=row['eval_key'], n_features=int(len(conf)), n_unconf_features=int(len(idx)), n_unconf_accepted=int((acc & ~conf).sum()),
               clipped_lines=';'.join(f"{k}@u{255.5 + v['q_px']:.3f}" for k, v in spec['clipped_lines'].items()), geometric_check=True)
    if len(idx) == 0:
        return out
    c = classify_line_features(_mesh(row['surface']), z['feats'][idx], spec, inp['binary'])
    if (c['cls'] == 'complete_line_in').any():
        out['inconsistency'] = f"{int((c['cls'] == 'complete_line_in').sum())} unconfirmable features with a numbered source inside the projector"
    a = acc[idx]
    for name in ('clipped_line', 'clipped_line_end', 'complete_line_out', 'multi', 'between_lines', 'no_hit', 'no_projection', 'other', 'complete_line_in'):
        s = c['cls'] == name
        out[f'feat_{name}'] = int(s.sum()); out[f'acc_{name}'] = int((s & a).sum())
    clp = np.isin(c['cls'], ['clipped_line', 'clipped_line_end'])
    out['clipped_lit_share'] = float(c['lit'][clp & (c['cls'] == 'clipped_line')].mean()) if (clp & (c['cls'] == 'clipped_line')).any() else None
    out['clipped_status'] = ';'.join(f'{STATUS[int(k)]}:{int(n)}' for k, n in zip(*np.unique(st[idx][clp], return_counts=True))) if clp.any() else ''
    sel = idx[clp & a]
    if len(sel):
        out.update(acc_clipped_chains=';'.join(str(int(x)) for x in np.unique(z['chain'][sel])), acc_clipped_decoded_ids=';'.join(str(int(x)) for x in np.unique(z['line_id'][sel])),
                   acc_clipped_err_min=float(e[sel].min()), acc_clipped_err_max=float(e[sel].max()), acc_clipped_err_median=float(np.median(e[sel])),
                   acc_clipped_in_global=int(z['in_global'][sel].sum()), acc_clipped_in_local=int(z['in_local'][sel].sum()),
                   acc_clipped_proj_u_min=float(np.nanmin(c['proj_uv'][clp & a, 0])), acc_clipped_proj_u_max=float(np.nanmax(c['proj_uv'][clp & a, 0])))
    oth = idx[~clp & a]
    if len(oth):
        out.update(acc_other_unconf_abs_err_median=float(np.median(np.abs(e[oth]))), acc_other_unconf_abs_err_max=float(np.abs(e[oth]).max()))
    return out


def _line_worker(row):
    try:
        return line_image(row)
    except Exception as ex:
        import traceback
        return dict(eval_key=row['eval_key'], error=f'{type(ex).__name__}: {ex}', tb=traceback.format_exc()[-1500:])


def tables_all_l63() -> dict:

    import r2_layouts as L
    reg = K.load_json(K.OUT / 'layouts.json'); out = {}
    for lid, lay in reg['baselines']['L6.3']['layouts'].items():
        for ph in K.PHASES:
            T = L.table('L6.3', lay, ph); sp = T['spec']
            out[f'{lid}|{ph}'] = dict(ids_min=int(min(sp['ids'])), ids_max=int(max(sp['ids'])), n_ids=len(sp['ids']), ids_all=[int(x) for x in sp['table']['ids_all']],
                                     clipped={k: dict(u_centre_px=255.5 + v['q_px'], lit_px=int(v['lit_px'])) for k, v in sp['clipped_lines'].items()})
    return out


def anchor_trace() -> dict:

    import l01_common as C0
    import r2_run as R
    M = R.manifest(); row = M[M.eval_key == LINE_ANCHOR['eval_key']].iloc[0].to_dict()
    if row['decode_key'] != LINE_ANCHOR['decode_key']:
        raise SystemExit('line anchor decode key differs')
    inp = R.inputs_of(row); spec = inp['table']['spec']; z = K.load_npz(K.EVL / f"{row['eval_key']}.npz"); j = K.load_json(K.EVL / f"{row['eval_key']}.json")
    acc = z['accepted'].astype(bool); unc = z['point_unconf'].astype(bool); sel = acc & unc
    c = classify_line_features(_mesh(row['surface']), z['feats'][sel], spec, inp['binary'])
    g = C0.geom(); pts = z['pts'].astype(np.float64)[sel]
    uvd, _ = C0.project(pts, g['proj'])
    B = np.asarray(inp['binary']) > 0
    import sp_lines_table as LT
    fam = LT.line_family_table(spec)
    lit_cols = np.flatnonzero(B.any(axis=0)); col_owner = {}
    for u in lit_cols[:8]:
        qq = np.array([u - fam['c'][0]]); k1, d1, _k2, _d2 = __import__('sp2_lines_table').nearest_table_line(fam, qq)
        col_owner[int(u)] = dict(line=int(k1[0]), dist_px=float(d1[0]), lit_rows=int(B[:, u].sum()))
    e = pts[:, 2] - z['zr'].astype(np.float64)[sel]
    other = acc & ~unc
    return dict(eval_key=row['eval_key'], decode_key=row['decode_key'], image_key=row['image_key'], raster_key=row['raster_key'], raster_key_rebuilt=inp['raster_key'],
                lit_px=inp['lit_px'], integral_rel_err=inp['integral_rel_err'], peak_f32=float(row['peak_f32']), ids=[int(x) for x in spec['ids']], ids_all=[int(x) for x in spec['table']['ids_all']],
                clipped_lines=spec['clipped_lines'], phase_frac=spec['phase_frac'], spacing_px=spec['spacing_px'], width_px=spec['width_px'],
                line_centres_u={str(k): 255.5 + fam['q_of'][k] for k in (-7, -6)}, first_lit_columns=col_owner,
                n_accepted=int(acc.sum()), n_unconf=int(sel.sum()), chains=sorted(int(x) for x in np.unique(z['chain'][sel])), decoded_ids=sorted(int(x) for x in np.unique(z['line_id'][sel])),
                classes={k: int(v) for k, v in zip(*np.unique(c['cls'], return_counts=True))}, nearest_line=sorted(int(x) for x in np.unique(c['k1'])),
                d1_px=[float(np.nanmin(c['d1'])), float(np.nanmax(c['d1']))], d2_px_min=float(np.nanmin(c['d2'])), band_px=c['band'], lit_share=float(c['lit'].mean()),
                proj_u_true=[float(np.nanmin(c['proj_uv'][:, 0])), float(np.nanmax(c['proj_uv'][:, 0]))], proj_v_true=[float(np.nanmin(c['proj_uv'][:, 1])), float(np.nanmax(c['proj_uv'][:, 1]))],
                proj_u_decoded=[float(np.nanmin(uvd[:, 0])), float(np.nanmax(uvd[:, 0]))], raster_centre_u_line_m6=float(np.mean(np.flatnonzero(B[256])[(np.flatnonzero(B[256]) > 20) & (np.flatnonzero(B[256]) < 50)])),
                err_z_minus_zr=[float(e.min()), float(e.max())], ray_distance_mm=[float(np.linalg.norm(pts - c['xyz'], axis=1).min()), float(np.linalg.norm(pts - c['xyz'], axis=1).max())],
                y_true=[float(c['xyz'][:, 1].min()), float(c['xyz'][:, 1].max())], y_decoded=[float(pts[:, 1].min()), float(pts[:, 1].max())], x_decoded=[float(pts[:, 0].min()), float(pts[:, 0].max())],
                r_min_decoded=float(np.hypot(pts[:, 0], pts[:, 1]).min()), in_global=int(z['in_global'][sel].sum()), in_local=int(z['in_local'][sel].sum()),
                rmse_other_accepted=float(np.sqrt(np.mean((z['pts'][other, 2].astype(float) - z['zr'][other].astype(float)) ** 2))),
                metrics={k: j['metrics'][k] for k in ('glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'glob_Ctau', 'loc_Ctau', 'id_correct', 'id_wrong', 'id_unconf')},
                official_identity='table truth unchanged (point_unconf stored); diagnostic class is a separate label')


def _n(v) -> int:
    return 0 if v is None or pd.isna(v) else int(v)


def legacy_compare(rows_formal: pd.DataFrame) -> list:

    import r2_run as R
    rr = K.load_json(K.RUN); refs = rr['entry']['legacy_refs']['refs']; M = R.manifest(); out = []
    le = {}
    for f in (K.OUT / 'legacy_ref').glob('R2LE-*.json'):
        x = K.load_json(f); le[x['legacy_key']] = f
    for lr in refs:
        if lr['family'] != 'line':
            continue
        row = M[M.image_key == lr['formal_image_key']].iloc[0].to_dict(); inp = R.inputs_of(row); spec = inp['table']['spec']
        jl = K.load_json(le[lr['legacy_key']]); zl = K.load_npz(le[lr['legacy_key']].with_suffix('.npz'))
        acc = zl['accepted'].astype(bool); feats = zl['feats'].astype(np.float64)
        c = classify_line_features(_mesh(row['surface']), feats[acc], spec, inp['binary']) if acc.any() else None
        n_cl = int(np.isin(c['cls'], ['clipped_line', 'clipped_line_end']).sum()) if c is not None else 0
        fr = rows_formal[rows_formal.eval_key == row['eval_key']].iloc[0].to_dict()
        ml = jl['metrics']; mf = K.load_json(K.EVL / f"{row['eval_key']}.json")['metrics']
        out.append(dict(surface=row['surface'], sigma=float(row['sigma']), legacy_key=lr['legacy_key'], legacy_eval=K.rel(le[lr['legacy_key']]), formal_eval_key=row['eval_key'],
                        legacy_accepted=int(acc.sum()), legacy_acc_clipped_source=n_cl, legacy_id_unconf=ml['id_unconf'], legacy_glob_pt_rmse=ml['glob_pt_rmse'], legacy_glob_pt_p95=ml['glob_pt_p95'],
                        formal_acc_clipped_source=_n(fr.get('acc_clipped_line')) + _n(fr.get('acc_clipped_line_end')), formal_id_unconf=mf['id_unconf'],
                        formal_glob_pt_rmse=mf['glob_pt_rmse'], formal_glob_pt_p95=mf['glob_pt_p95']))
    return out


def lines(workers: int = 10) -> dict:
    import r2_run as R
    t0 = time.time()
    M = R.manifest(); U = M[M.family == 'line'].drop_duplicates('eval_key').copy()
    C = pd.read_csv(K.OUT / 'case_metrics.csv', low_memory=False); C['block'] = C['block'].fillna('')
    views = C.groupby('eval_key').agg(n_views=('view_id', 'size'), strategies=('strategy', lambda s: ';'.join(sorted(set(s)))), blocks=('block', lambda s: ';'.join(sorted(set(x for x in s if x)))))
    L = C[C.family == 'line'].drop_duplicates('eval_key').set_index('eval_key')
    tabs = tables_all_l63()
    with_clipped = sorted({k.split('|')[1] for k, v in tabs.items() if v['clipped']})
    targets = U[(U.phase.isin(with_clipped)) | (U.eval_key.map(L.id_unconf) > 0)]
    rows = [r for r in targets.to_dict('records')]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_line_worker, rows, chunksize=4))
    errs = [x for x in res if 'error' in x]
    if errs:
        raise SystemExit(f'{len(errs)} line diagnosis errors, first {errs[0]}')
    D = pd.DataFrame(res).set_index('eval_key')
    base = U[['eval_key', 'image_key', 'decode_key', 'surface', 'phase', 'sigma', 'layout', 'raster_key']].set_index('eval_key')
    base = base.join(views).join(L[['state', 'n_accepted', 'n_valid', 'id_main_correct', 'id_main_wrong', 'id_main_unconf', 'unconf_share', 'glob_Ctau', 'loc_Ctau', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse']])
    out = base.join(D, how='left').reset_index()
    out['geometric_check'] = out['geometric_check'].fillna(False).astype(bool)
    out['family'] = 'line'
    anchor = anchor_trace(); leg = legacy_compare(out); win = window_explanation()
    read = [K.EVL / f'{k}.npz' for k in targets.eval_key] + [K.EVL / f'{k}.json' for k in U.eval_key]
    summ = summarise_lines(out)
    sec = dict(tables_l63=dict(n_layout_phase=len(tabs), phases_with_clipped=with_clipped, clipped=dict((k, v['clipped']) for k, v in tabs.items() if v['clipped'])),
               selection_rule='geometric check on all images of the phases with a clipped line (all layouts) and on every other image with accepted unconfirmable points; '
                              'only the stored unconfirmable features are traced; all other line images from stored metrics',
               n_unique_line_images=int(len(U)), n_geometric=int(len(targets)), anchor=anchor, legacy=leg, window_explanation=win, summary=summ, seconds=round(time.time() - t0, 1), t_local=K.now_local(),
               inputs_read=read_log(read + [K.OUT / 'case_metrics.csv', K.OUT / 'manifest.csv', K.OUT / 'layouts.json']))
    run_put('lines', sec)
    save_csv(out, OUT / '_lines_part.csv')
    return dict(n_geometric=len(targets), summary=summ['by_condition'][:3], anchor_classes=anchor['classes'])


def window_explanation() -> list:

    import r2_run as R
    from m4_lines import STATUS
    M = R.manifest(); out = []
    for s in K.SURF:
        for sg in (0.0, 40.0):
            r = M[(M.family == 'line') & (M.layout == 'regular') & (M.phase == 'H0') & (M.surface == s) & (M.sigma == sg)].iloc[0].to_dict()
            inp = R.inputs_of(r); z = K.load_npz(K.EVL / f"{r['eval_key']}.npz"); idx = np.flatnonzero(~z['confirmable'].astype(bool))
            c = classify_line_features(_mesh(s), z['feats'][idx], inp['table']['spec'], inp['binary']); cl = np.isin(c['cls'], ['clipped_line', 'clipped_line_end'])
            zt = c['xyz'][cl, 2]; st = z['status'][idx][cl]; acc = z['accepted'][idx][cl].astype(bool); nc = z['n_candidates'][idx][cl]
            e = (z['pts'][idx][cl, 2] - z['zr'][idx][cl])[acc]
            out.append(dict(surface=s, sigma=sg, eval_key=r['eval_key'], n_clipped_features=int(cl.sum()), true_z_mm=[float(zt.min()), float(zt.max())],
                            neighbour_candidate_depth_approx_mm=[float(zt.min()) - 15.5, float(zt.max()) - 15.5], window_mm=list(K.RK.WINDOW_MM),
                            status={STATUS[int(k)]: int(n) for k, n in zip(*np.unique(st, return_counts=True))}, n_candidates={int(k): int(n) for k, n in zip(*np.unique(nc, return_counts=True))},
                            accepted=int(acc.sum()), err_mm=([float(e.min()), float(e.max())] if len(e) else None)))
    return out


def summarise_lines(D: pd.DataFrame) -> dict:
    X = D[D.geometric_check].copy()
    for c in ('acc_clipped_line', 'acc_clipped_line_end', 'feat_clipped_line', 'feat_clipped_line_end', 'acc_between_lines', 'acc_multi', 'acc_complete_line_out', 'feat_complete_line_in'):
        X[c] = X.get(c, 0).fillna(0)
    X['acc_clipped'] = X.acc_clipped_line + X.acc_clipped_line_end; X['feat_clipped'] = X.feat_clipped_line + X.feat_clipped_line_end
    aff = X[X.acc_clipped > 0]
    by = []
    for (s, sg, ph), g in X.groupby(['surface', 'sigma', 'phase']):
        a = g[g.acc_clipped > 0]
        by.append(dict(surface=s, sigma=sg, phase=ph, n_images=len(g), n_views=int(g.n_views.sum()), images_with_clipped_features=int((g.feat_clipped > 0).sum()),
                       images_with_accepted_clipped=int(len(a)), views_with_accepted_clipped=int(a.n_views.sum()), accepted_clipped_points=int(a.acc_clipped.sum()),
                       accepted_between_lines=int(g.acc_between_lines.sum()), accepted_multi=int(g.acc_multi.sum()), accepted_complete_out=int(g.acc_complete_line_out.sum()),
                       err_min=(float(a.acc_clipped_err_min.min()) if len(a) else None), err_max=(float(a.acc_clipped_err_max.max()) if len(a) else None),
                       in_local=int(a.acc_clipped_in_local.fillna(0).sum()), in_global=int(a.acc_clipped_in_global.fillna(0).sum()),
                       glob_pt_p95_median=(float(a.glob_pt_p95.median()) if len(a) else None), unconf_share_max=(float(a.unconf_share.max()) if len(a) else None),
                       clipped_feature_status=';'.join(sorted(set(';'.join(g.clipped_status.dropna()).split(';')) - {''}))))
    return dict(n_images_with_accepted_clipped=int(len(aff)), n_views_with_accepted_clipped=int(aff.n_views.sum()), n_accepted_clipped_points=int(aff.acc_clipped.sum()),
                n_images_with_clipped_features=int((X.feat_clipped > 0).sum()), inconsistencies=int(X.feat_complete_line_in.sum()), by_condition=by)


# Compare dot-front-end outcomes on the stored layout images.
def dots() -> dict:
    import r2_run as R
    t0 = time.time()
    M = R.manifest(); U = M[M.family == 'point'].drop_duplicates('eval_key').copy()
    C = pd.read_csv(K.OUT / 'case_metrics.csv', low_memory=False); C['block'] = C['block'].fillna('')
    views = C.groupby('eval_key').agg(n_views=('view_id', 'size'), strategies=('strategy', lambda s: ';'.join(sorted(set(s)))), blocks=('block', lambda s: ';'.join(sorted(set(x for x in s if x)))))
    P_ = C[C.family == 'point'].drop_duplicates('eval_key').set_index('eval_key')
    sig0 = {}
    rows = []; read = []
    for r in U.to_dict('records'):
        if float(r['sigma']) < 80 and r['eval_key'] != DOT_ANCHOR['eval_key']:
            continue
        rows.append(r)

    s0 = U[U.sigma == 0.0].set_index('image_key')
    out = []
    for r in rows:
        j = K.load_json(K.EVL / f"{r['eval_key']}.json"); z = K.load_npz(K.EVL / f"{r['eval_key']}.npz"); read += [K.EVL / f"{r['eval_key']}.npz"]
        info = j['info']; m = j['metrics']; si = (j.get('extras') or {}).get('source_identity') or {}
        ref_ek = s0.loc[r['sigma0_ref_key'], 'eval_key']
        if ref_ek not in sig0:
            jr = K.load_json(K.EVL / f'{ref_ek}.json'); sig0[ref_ek] = float(jr['info']['det_info'].get('blob_area_median_px', np.nan))
        a0 = sig0[ref_ek]
        area = z['cand_area'].astype(np.float64); code = z['cand_code'].astype(int)
        dot_sized = (area >= 0.5 * a0) & (area <= 1.5 * a0)
        tot = area.sum()
        ref = float(info['det_info'].get('blob_area_median_px', np.nan))
        out.append(dict(eval_key=r['eval_key'], image_key=r['image_key'], surface=r['surface'], phase=r['phase'], sigma=float(r['sigma']), layout=r['layout'], peak_f32=float(r['peak_f32']), lit_px=float(r['lit_px']),
                        integral_f32=float(r['integral_f32']), otsu=float(info['det_info'].get('otsu_threshold', np.nan)), n_components=m.get('n_components'), n_candidates=m.get('n_candidates'),
                        sel_accepted=info['selection_codes']['0_accepted'], sel_rel_small=info['selection_codes']['1_rel_area_small'], sel_rel_large=info['selection_codes']['2_rel_area_large'],
                        sel_elong=info['selection_codes']['3_elongation'], area_ref_px=ref, sigma0_dot_area_px=a0, area_ref_over_sigma0=(ref / a0 if a0 else None),
                        dot_sized_area_share=(float(area[dot_sized].sum() / tot) if tot else None), n_dot_sized_cands=int(dot_sized.sum()),
                        n_dot_sized_accepted=int((dot_sized & (code == 0)).sum()), n_confirmable_selected=int(z['feat_confirmable'].sum()), n_matched=m.get('n_matched'), n_ambiguous=m.get('n_ambiguous'),
                        n_accepted=m['n_accepted'], n_valid=m['n_valid'], src_correct=si.get('correct'), src_wrong=si.get('wrong'), src_undeterminable=si.get('undeterminable'),
                        unconf_share_src=((si.get('undeterminable') or 0) / si['n_reliable'] if si.get('n_reliable') else None), glob_Ctau=m['glob_Ctau'], loc_Ctau=m['loc_Ctau'],
                        glob_pt_rmse=m['glob_pt_rmse'], sigma0_ref_eval=ref_ek))
    D = pd.DataFrame(out).set_index('eval_key').join(views).reset_index(); D['family'] = 'point'
    hi = D[D.sigma >= 80]
    ex = hi[hi.eval_key == DOT_ANCHOR['eval_key']]; rest = hi[hi.eval_key != DOT_ANCHOR['eval_key']]
    stats = dict(n_unique=int(len(hi)), n_glob_positive=int((hi.glob_Ctau > 0).sum()), n_loc_positive=int((hi.loc_Ctau > 0).sum()), rest_glob_max_pct=float(100 * rest.glob_Ctau.max()),
                 rest_loc_max_pct=float(100 * rest.loc_Ctau.max()), exception=ex.iloc[0].to_dict(), n_valid_range=[int(hi.n_valid.min()), int(hi.n_valid.max())],
                 pt_rmse_range=[float(hi.glob_pt_rmse.min()), float(hi.glob_pt_rmse.max())], unconf_share_range=[float(hi.unconf_share_src.min()), float(hi.unconf_share_src.max())],
                 area_ref_over_sigma0=dict(exception=float(ex.area_ref_over_sigma0.iloc[0]), rest_max=float(rest.area_ref_over_sigma0.max()), rest_median=float(rest.area_ref_over_sigma0.median())),
                 dot_sized_area_share=dict(exception=float(ex.dot_sized_area_share.iloc[0]), rest_max=float(rest.dot_sized_area_share.max()), rest_median=float(rest.dot_sized_area_share.median())),
                 n_rest_with_ref_ge_half_sigma0=int((rest.area_ref_over_sigma0 >= 0.5).sum()))
    comp = dot_compare(M, U)
    sec = dict(selection_rule='all sigma >= 80 unique dot images (384) from stored records + the comparison block of the exception', detection_recomputations=0,
               n_rows=len(D), stats=stats, comparison=comp, seconds=round(time.time() - t0, 1), t_local=K.now_local(),
               inputs_read=read_log(read + [K.EVL / f'{k}.json' for k in sig0]))
    run_put('dots', sec)
    save_csv(D, OUT / '_dots_part.csv')
    return dict(stats={k: v for k, v in stats.items() if k != 'exception'})


def dot_compare(M, U) -> list:

    ex = U[U.eval_key == DOT_ANCHOR['eval_key']].iloc[0]
    if ex['image_key'] != DOT_ANCHOR['image_key']:
        raise SystemExit('dot anchor image key differs')
    sel = U[(U.surface == ex['surface']) & (U.phase == ex['phase']) & (((U.layout == 'regular') & (U.sigma == 80.0)) | ((U.layout == ex['layout']) & U.sigma.isin([0.0, 60.0, 80.0, 100.0])))]
    out = []
    for r in sel.sort_values(['layout', 'sigma']).to_dict('records'):
        j = K.load_json(K.EVL / f"{r['eval_key']}.json"); info = j['info']; m = j['metrics']; si = (j.get('extras') or {}).get('source_identity') or {}
        out.append(dict(role=('exception' if r['eval_key'] == DOT_ANCHOR['eval_key'] else ('S0 same noise' if r['layout'] == 'regular' else f"S1 {r['layout']} sigma {r['sigma']:g}")),
                        eval_key=r['eval_key'], image_key=r['image_key'], layout=r['layout'], sigma=float(r['sigma']), noise_seed=r['noise_seed'], peak_f32=float(r['peak_f32']), lit_px=float(r['lit_px']),
                        integral_f32=float(r['integral_f32']), otsu=info['det_info'].get('otsu_threshold'), area_ref_px=info['det_info'].get('blob_area_median_px'),
                        n_components=m.get('n_components'), n_candidates=m.get('n_candidates'), sel=info['selection_codes'], n_selected=m.get('n_selected'), n_confirmable=m.get('n_confirmable_features'),
                        n_matched=m.get('n_matched'), n_ambiguous=m.get('n_ambiguous'), n_accepted=m['n_accepted'], src=dict(c=si.get('correct'), w=si.get('wrong'), u=si.get('undeterminable')),
                        glob_Ctau=m['glob_Ctau'], loc_Ctau=m['loc_Ctau'], glob_pt_rmse=m['glob_pt_rmse']))
    return out


def statements() -> dict:
    P = pd.read_csv(K.OUT / 'paired_summary.csv', low_memory=False); C = pd.read_csv(K.OUT / 'case_metrics.csv', low_memory=False)
    Pp = P[P.table_type == 'paired']
    pos = Pp[(Pp.metric.isin(['glob_Ctau', 'loc_Ctau'])) & Pp.contrast.isin(['S1-S0', 'S2-S0', 'S3-S0', 'S4-S0']) & (Pp['main'] > 1e-12)]
    pos = pos[['family', 'surface', 'sigma', 'contrast', 'metric', 'main', 'block_min', 'block_max', 'mean_pairs', 'p10', 'p90']].sort_values(['metric', 'family', 'surface', 'sigma'])
    A = P[(P.table_type == 'absolute') & (P.metric == 'glob_pt_rmse') & (P.contrast.isin(['S0', 'S1', 'S2', 'S3', 'S4']))]
    spread = A.groupby(['family', 'surface', 'sigma'])['main'].agg(lambda s: float(s.max() - s.min())).reset_index().rename(columns={'main': 'pt_rmse_spread_mm'})
    lines_tail = C[(C.family == 'line')].drop_duplicates('eval_key').sort_values('unconf_share', ascending=False).head(5)[['surface', 'strategy', 'phase', 'sigma', 'unconf_share', 'glob_pt_p95', 'eval_key']]
    out = dict(local_or_global_positive_vs_S0=pos.to_dict('records'),
               glob_S1_S0_max=float(Pp[(Pp.metric == 'glob_Ctau') & (Pp.contrast == 'S1-S0')]['main'].max()), glob_S3_S0_max=float(Pp[(Pp.metric == 'glob_Ctau') & (Pp.contrast == 'S3-S0')]['main'].max()),
               pt_rmse_spread_dots_sigma0_60_max=float(spread[(spread.family == 'point') & (spread.sigma <= 60)].pt_rmse_spread_mm.max()),
               pt_rmse_spread_lines_sigma0_60_max=float(spread[(spread.family == 'line') & (spread.sigma <= 60)].pt_rmse_spread_mm.max()),
               pt_rmse_spread_lines_max_cell=spread[spread.family == 'line'].sort_values('pt_rmse_spread_mm').iloc[-1].to_dict(),
               line_unconf_tail=lines_tail.to_dict('records'),
               abs_glob_Ctau_dots_sigma_ge80_max=float(P[(P.table_type == 'absolute') & (P.family == 'point') & (P.metric == 'glob_Ctau') & (P.sigma >= 80)]['main'].max()),
               abs_loc_Ctau_dots_sigma_ge80_max=float(P[(P.table_type == 'absolute') & (P.family == 'point') & (P.metric == 'loc_Ctau') & (P.sigma >= 80)]['main'].max()),
               s1_s0_glob_by_sigma={f"{f}|{s}": [float(x) for x in Pp[(Pp.metric == 'glob_Ctau') & (Pp.contrast == 'S1-S0') & (Pp.family == f) & (Pp.surface == s)].sort_values('sigma')['main']]
                                    for f in ('point', 'line') for s in K.SURF},
               p10_p90_level='pair values (per phase), block range = block medians (r2_analysis._block_stat)', t_local=K.now_local())
    run_put('statements', out)
    return dict(n_positive_cells=len(pos), positives=pos.to_dict('records'))


def build_diagnostics() -> None:
    a = pd.read_csv(OUT / '_lines_part.csv', low_memory=False); b = pd.read_csv(OUT / '_dots_part.csv', low_memory=False)
    D = pd.concat([a, b], ignore_index=True, sort=False)
    lead = ['family', 'eval_key', 'image_key', 'surface', 'phase', 'sigma', 'layout', 'n_views', 'strategies', 'blocks']
    D = D[lead + [c for c in D.columns if c not in lead]]
    save_csv(D, OUT / 'diagnostics.csv')
    for f in ('_lines_part.csv', '_dots_part.csv'):
        (OUT / f).unlink()


def final() -> dict:
    r = run_load(); bad = []
    for sec in ('lines', 'dots'):
        for p, h in r[sec]['inputs_read'].items():
            if sha(K.ROOT / p) != h:
                bad.append(p)
    anchors = {f: sha(K.OUT / f) == h for f, h in reference.items()}
    outs = {K.rel(p): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'run.json'}
    res = dict(inputs_unchanged=not bad, changed=bad[:10], n_inputs_checked=sum(len(r[s]['inputs_read']) for s in ('lines', 'dots')), s4_real02_anchors_unchanged=all(anchors.values()),
               outputs=outs, imaging=0, formal_decodes=0, detection_recomputations=r['dots']['detection_recomputations'], t_local=K.now_local())
    run_put('final', res)
    return res


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'check':
        print(check())
    elif cmd == 'lines':
        print(lines(int(sys.argv[2]) if len(sys.argv) > 2 else 10))
    elif cmd == 'dots':
        print(dots())
    elif cmd == 'statements':
        print(statements())
    elif cmd == 'merge':
        build_diagnostics(); print('diagnostics.csv written')
    elif cmd == 'figs':
        import r3_figs as F
        print(F.make_all())
    elif cmd == 'final':
        print(final())
    else:
        print(__doc__)
