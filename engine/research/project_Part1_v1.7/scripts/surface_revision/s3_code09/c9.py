# Check encoding tables and update summary figures from saved results.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(ROOT / 'scripts' / 'surface_revision' / 's3_code08'), str(ROOT / 'scripts' / 'surface_revision' / 's3_code07')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c8_common as K

OUT = ROOT / 'outputs' / 'surface_revision' / 's3_code09'
FIGS = OUT / 'figures'
CHECKS = OUT / 'checks.json'
K.OUT, K.FIGS = OUT, FIGS
for _k, _sub in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
    os.environ[_k] = str(OUT / 'sandbox' / _sub)
C8 = ROOT / 'outputs' / 'surface_revision' / 's3_code08'
TASK = ROOT.parents[1] / 'protocols' / 'S3-CODE-09_protocol.md'
REVIEW = ROOT.parents[1] / 'protocols' / 'S3-CODE-08_review.md'
SEL = dict(family='lines', layout='5.6', phase_id='N1', sigma=0.0, rep=0)
RULES = ('R@1.1247', 'ignore')
ANCHOR = {('B_curv', 'mid'): (0.777394, 0.834459), ('B_curv', 'strong'): (0.698064, 0.755065), ('M10_curv', 'mid'): (0.505542, 0.558600),
          ('M10_curv', 'strong'): (0.575080, 0.628041)}


def checks_add(name: str, obj: dict) -> None:
    c = K.load_json(CHECKS) if CHECKS.exists() else dict(task='S3-CODE-09', created=K.now_local(), note='Numerical consistency checks.')
    if name in c:
        raise K.Stop('check', f'checks entry {name} exists; never overwritten')
    c[name] = dict(obj, t_local=K.now_local()); K.save_json(CHECKS, c)


def _selection(C: pd.DataFrame) -> pd.DataFrame:
    m = pd.Series(True, index=C.index)
    for k, v in SEL.items():
        m &= (C[k].astype(str) == v) if k == 'layout' else (C[k] == v)
    X = C[m & C.surface.isin(['B_curv', 'M10_curv']) & C.pattern.isin(['weak', 'mid', 'strong']) & C.rule.isin(RULES)].copy()
    if len(X) != 12 or X.groupby(['surface', 'pattern']).image_key.nunique().max() != 1 or X.groupby(['surface', 'pattern']).size().min() != 2:
        raise K.Stop('check', f'task A selection is not 12 records in 6 same-image R / ignore pairs ({len(X)})')
    return X.sort_values(['surface', 'pattern', 'rule'])


def _records(T) -> list:

    keys = set()
    for r in _selection(T['C']).itertuples():
        keys |= {('dec', r.decode_key), ('eval', r.eval_key)}
    for c in K.load_json(K.C7_CUR / 'case_board.json'):
        if c.get('eval_key'):
            keys |= {('dec', c['decode_key']), ('eval', c['eval_key'])}
    return sorted(K.rel(K.C7_REC / d / f'{k}{ext}') for d, k in keys for ext in ('.json', '.npz'))


def hashes(T) -> dict:
    import c7_freeze as F7
    fz = F7.verify()
    files = [K.rel(K.C7_CUR / f) for f in ('cases.csv', 'pairs.csv', 'case_phase_summary.csv', 'pair_phase_summary.csv', 'risk_summary.csv', 'readout_summary.csv', 'case_board.json', '../status.json')]
    files += [K.rel(K.C7 / 'formal' / 'frozen.json')]
    files += [K.rel(p) for p in sorted(C8.glob('*')) if p.is_file()] + [K.rel(p) for p in sorted((C8 / 'figures').glob('*'))]
    files += [K.rel(p) for p in sorted((ROOT / 'scripts' / 'surface_revision' / 's3_code08').glob('c8_*.py'))]
    files += [K.rel(p) for p in sorted((K.C7_CUR / 'figures').glob('*.png'))]
    rec = _records(T)
    return dict(freeze=dict(ok=fz['ok'], closure_sha256=fz['closure_sha256'], n_files=fz['n_files'], mismatches=fz['mismatches']),
                files={f: K.sha_file(ROOT / f) for f in files}, records={f: K.sha_file(ROOT / f) for f in rec}, n_files=len(files), n_records=len(rec))


def _cloud(r, M):

    import c2_decode as CD
    import c3_store as ST
    import rdy_platform as P
    row = dict(M.loc[r.image_key].to_dict(), image_key=r.image_key)
    dp = K.C7_REC / 'dec' / f'{r.decode_key}.json'; dj = K.load_json(dp)
    if dj.get('kind') == 'ref':
        A = K.load_npz(K.ROOT / dj['source_npz']); dj = dict(dj, meta=K.load_json(K.ROOT / dj['source_json'])['meta'])
    else:
        A = K.load_npz(dp.with_suffix('.npz'))
    D = ST.unpack('lines', A, dj['meta']); dec = CD.line_dec(D); mesh = P.mesh(row['surface'])
    V = P.s2().point_validity(dec['pts'], dec['accepted'], mesh); pts, valid = V['pts'], V['valid']; acc = np.asarray(dec['accepted'], bool)
    E = K.load_npz(K.C7_REC / 'eval' / f'{r.eval_key}.npz'); cor, wr, un = np.asarray(E['bridge'], bool)
    ep = np.asarray(E['pts'])
    align = dict(n_points=len(pts), eval_len=len(ep), eval_pts_dtype=str(ep.dtype),
                 pts_equal=bool(len(ep) == len(pts) and np.array_equal(ep, pts.astype(ep.dtype), equal_nan=True)),
                 pts_max_abs_diff_float64=float(np.nanmax(np.abs(ep.astype(np.float64) - pts))) if len(ep) == len(pts) and np.isfinite(pts).any() else None,
                 acc_equal=bool(np.array_equal(np.asarray(E['acc'], bool), acc)), valid_equal=bool(np.array_equal(np.asarray(E['valid'], bool), valid)),
                 lid_equal=bool(np.array_equal(np.asarray(E['lid']).astype(np.int64), np.asarray(D['line_id']).astype(np.int64))),
                 masks_disjoint=bool(not (cor & wr).any() and not (cor & un).any() and not (wr & un).any()),
                 c=int((acc & cor).sum()), w=int((acc & wr).sum()), u=int((acc & un).sum()), u_valid=int((acc & un & valid).sum()))
    align['counts_equal_table'] = bool(align['c'] == int(r.c) and align['w'] == int(r.w) and align['u'] == int(r.u))
    return row, pts, valid, acc, (acc & un), D, align


def _metrics(pts, valid, keep, sid, dl, dg) -> dict:
    import c6_check6x as X6
    import c7_cases as CS
    from scipy.spatial import Delaunay
    from surface_eval_r3 import DEDUP_TOL_MM
    d, tau = float(K.K4.RK.D_SUPPORT), float(K.K4.RK.TAU)
    Pv = X6._dedup(pts[valid & keep].astype(np.float64), DEDUP_TOL_MM); tri = Delaunay(Pv[:, :2]) if len(Pv) >= 3 else None
    out = dict(n_points_used=int((valid & keep).sum()), n_vertices=len(Pv))
    for nm, x, y, z in (('glob', dg['x'], dg['y'], dg[f'z_{sid}']), ('loc', dl['x'], dl['y'], dl['z_ref'])):
        g = CS._nodes(Pv, tri, x, y, z, d, tau)
        out.update({f'{nm}_Ctau': g['Ctau'], f'{nm}_Cvalid': g['Cvalid'], f'{nm}_n_nodes': g['n_nodes'], f'{nm}_n_interp': g['n_interp'], f'{nm}_n_supported': int((g['state'] > 0).sum()),
                    f'{nm}_n_tau': int((g['state'] == 2).sum()), f'{nm}_surf_rmse': g['surf'][1], f'{nm}_undecided': g['undecided']})
    return out


def task_a(T) -> tuple[pd.DataFrame, dict]:
    import rdy_platform as P
    C, Pp, SP = T['C'], T['P'], T['SP']; X = _selection(C)
    M = pd.read_csv(K.K4.OUT / 'formal' / 'manifest.csv', low_memory=False); M['layout'] = M.layout.astype(str); M = M.drop_duplicates('image_key').set_index('image_key')
    dg = P.dom_global(); per = {}; align = {}
    for r in X.itertuples():
        row, pts, valid, acc, um, D, al = _cloud(r, M)
        sid = K.K4.RK.SURF_MAP[row['surface']]; dl = P.dom_local(row['surface'])
        f0 = _metrics(pts, valid, np.ones(len(pts), bool), sid, dl, dg); f1 = _metrics(pts, valid, ~um, sid, dl, dg)
        per[(r.surface, r.pattern, r.rule)] = dict(r=r, orig=f0, diag=f1, n_accepted=int(acc.sum()), n_valid=int(valid.sum()))
        al['formal_glob_Ctau_diff'] = abs(f0['glob_Ctau'] - float(r.glob_Ctau)); al['formal_loc_Ctau_diff'] = abs(f0['loc_Ctau'] - float(r.loc_Ctau))
        al['formal_n_interp_equal'] = bool(f0['glob_n_interp'] == int(r.glob_n_interp) and f0['loc_n_interp'] == int(r.loc_n_interp))
        align[f'{r.surface}|{r.pattern}|{r.rule}'] = dict(al, eval_key=r.eval_key, decode_key=r.decode_key, image_key=r.image_key)
    rows = []
    E1 = Pp[(Pp.pair_type == 'E1L') & (Pp.a_rule == 'R@1.1247')].set_index(['a_eval_key', 'b_eval_key'])
    for s in ('B_curv', 'M10_curv'):
        for lev in ('weak', 'mid', 'strong'):
            a, b = per[(s, lev, 'R@1.1247')], per[(s, lev, 'ignore')]
            pair_formal = E1.loc[(a['r'].eval_key, b['r'].eval_key)] if (a['r'].eval_key, b['r'].eval_key) in E1.index else None
            sp = SP[(SP.effect == 'E1 R-ignore') & (SP.surface == s) & (SP.layout.astype(str) == '5.6') & (SP.pattern == lev) & (SP.sigma == 0)].iloc[0]
            for dom in ('glob', 'loc'):
                fa, fb = float(getattr(a['r'], f'{dom}_Ctau')), float(getattr(b['r'], f'{dom}_Ctau'))
                ra, rb = a['orig'][f'{dom}_Ctau'], b['orig'][f'{dom}_Ctau']; da, db = a['diag'][f'{dom}_Ctau'], b['diag'][f'{dom}_Ctau']
                d0, d1 = ra - rb, da - db
                rows.append(dict(surface=s, pattern=lev, domain=dom, image_key=a['r'].image_key, R_eval_key=a['r'].eval_key, R_decode_key=a['r'].decode_key, ign_eval_key=b['r'].eval_key,
                                 ign_decode_key=b['r'].decode_key, R_accepted=a['n_accepted'], R_u=int(a['r'].u), ign_accepted=b['n_accepted'], ign_u=int(b['r'].u),
                                 n_nodes=a['orig'][f'{dom}_n_nodes'],
                                 R_interp_orig=a['orig'][f'{dom}_n_interp'], R_supported_orig=a['orig'][f'{dom}_n_supported'], R_interp_diag=a['diag'][f'{dom}_n_interp'], R_supported_diag=a['diag'][f'{dom}_n_supported'],
                                 ign_interp_orig=b['orig'][f'{dom}_n_interp'], ign_supported_orig=b['orig'][f'{dom}_n_supported'], ign_interp_diag=b['diag'][f'{dom}_n_interp'],
                                 ign_supported_diag=b['diag'][f'{dom}_n_supported'],
                                 R_Ctau_formal=fa, R_Ctau_recomputed=ra, R_Ctau_diag=da, ign_Ctau_formal=fb, ign_Ctau_recomputed=rb, ign_Ctau_diag=db,
                                 dCtau_formal=fa - fb, dCtau_pairs_table=(float(pair_formal[f'd_{dom}_Ctau']) if pair_formal is not None else np.nan), dCtau_recomputed=d0, dCtau_diag=d1,
                                 change=d1 - d0, change_phase_equal_contribution=(d1 - d0) / int(sp.n_phases), phase_equal_formal_mean=float(sp[f'd_{dom}_Ctau_mean']), n_phases=int(sp.n_phases),
                                 reps_per_phase=str(sp.reps_per_phase)))
    R = pd.DataFrame(rows)
    chk = dict(selection=dict(records=12, pairs=6, same_image=True), alignment=align,
               max_formal_Ctau_diff=float(max(max(v['formal_glob_Ctau_diff'], v['formal_loc_Ctau_diff']) for v in align.values())),
               all_aligned=bool(all(v['pts_equal'] and v['acc_equal'] and v['valid_equal'] and v['lid_equal'] and v['masks_disjoint'] and v['counts_equal_table'] and v['formal_n_interp_equal']
                                    for v in align.values())),
               pairs_table_equal=bool((R.dCtau_pairs_table - R.dCtau_formal).abs().max() <= 1e-12), undecided_nodes=int(sum(p['orig'][f'{d}_undecided'] + p['diag'][f'{d}_undecided']
                                                                                                                   for p in per.values() for d in ('glob', 'loc'))))
    anc = {}
    for (s, lev), (f0, f1) in ANCHOR.items():
        g = R[(R.surface == s) & (R.pattern == lev) & (R.domain == 'glob')].iloc[0]
        anc[f'{s}|{lev}'] = dict(review_formal=f0, got_formal=round(float(g.dCtau_formal), 6), review_diag=f1, got_diag=round(float(g.dCtau_diag), 6),
                                 abs_diff_formal=abs(float(g.dCtau_formal) - f0), abs_diff_diag=abs(float(g.dCtau_diag) - f1))
    chk['anchors'] = anc; chk['anchors_within_1e-6'] = bool(all(max(v['abs_diff_formal'], v['abs_diff_diag']) <= 1e-6 for v in anc.values()))
    w = R[R.pattern == 'weak']; chk['weak_dCtau_zero'] = bool((w.dCtau_formal.abs() < 1e-15).all() and (w.dCtau_diag.abs() < 1e-15).all())
    chk['ok'] = bool(chk['all_aligned'] and chk['max_formal_Ctau_diff'] <= 1e-12 and chk['pairs_table_equal'] and chk['undecided_nodes'] == 0)
    return R, chk


# Trace selected line outcomes through candidate identities and segment assignments.
def task_b(T) -> dict:
    import c3_eval as EV
    import c3_store as ST
    import c4_store as S4
    import l01_truth as LT
    import rdy_platform as P
    C, Pp = T['C'], T['P']; res = {}
    q = C[(C.family == 'lines') & (C.layout.astype(str) == '8') & (C.rule == 'R@1.1247')]
    e8 = Pp[(Pp.family == 'lines') & (Pp.layout.astype(str) == '8') & (Pp.pair_type == 'E1L')]
    res['B1_8mm_R'] = dict(images=len(q), image_fallback=int((q.image_fallback.astype(str) == 'True').sum()), not_fallback=int((q.image_fallback.astype(str) != 'True').sum()),
                           images_L3_adopted_gt0=int((q.L3_adopted > 0).sum()), L3_adopted_segments=int(q.L3_adopted.sum()), L3_wide_read_narrow=int(q.L3_w2n.sum()),
                           L3_narrow_read_wide=int(q.L3_n2w.sum()), decisions=q.decision.value_counts().to_dict(), E1_pairs_R_and_LB=len(e8), E1_pairs_nonzero=int((e8.d_glob_Ctau != 0).sum()),
                           E1_pairs_loc_nonzero=int((e8.d_loc_Ctau != 0).sum()),
                           denominators='images: all 792 R records; L3 counts: adopted segment classes (segments, not points); E1 pairs: same-image R-ignore and LB-ignore pairs')

    src = (ROOT / 'src' / 'm4_lines.py').read_text(encoding='utf-8').splitlines(); lc = (ROOT / 'scripts' / 'line01r' / 'lc_pipeline.py').read_text(encoding='utf-8').splitlines()
    def find(lines, pat):
        return [i + 1 for i, l in enumerate(lines) if pat in l]
    res['B2_code'] = dict(depth_window_and_projector_range=dict(file='src/m4_lines.py', lines=find(src, 'adm = np.isfinite(mu) & inside') + find(src, 'inside = has_centre &')),
                          segment_vote=dict(file='src/m4_lines.py', lines=find(src, 'if top >= vote_frac * n and second <= (1.0 - vote_frac) * n'), vote_frac=0.9),
                          per_segment_class_filter_no_cross_segment=dict(file='scripts/line01r/lc_pipeline.py', lines=find(lc, 'nothing propagates across segments')),
                          search_window='registered window z0 = -16 mm, volume 20 mm (rdy_decode); margin = 2 % of the volume unless set')
    M = pd.read_csv(K.K4.OUT / 'formal' / 'manifest.csv', low_memory=False); M['layout'] = M.layout.astype(str); M = M.drop_duplicates('image_key').set_index('image_key')
    six = {}
    for r in _selection(C).itertuples():
        row, pts, valid, acc, um, D, al = _cloud(r, M)
        lid = np.asarray(D['line_id']).astype(np.int64); seg = np.asarray(D['segment_id']).astype(np.int64); feat = np.asarray(D['features'], float)
        segs_m8 = sorted(int(x) for x in np.unique(seg[acc & (lid == -8)])); seg_u = sorted(int(x) for x in np.unique(seg[um]))
        inp = S4.inputs_of(dict(row)); ctx = EV.ctx_of(dict(row), inp); fam = LT._family(ctx['meta_full']['olines']); ti = LT.truth_identity(P.mesh(row['surface']), feat, ctx['meta_full']['olines'])
        q_ = (ti['proj_uv'] - fam['c']) @ fam['n']; rr = q_ / fam['s'] - fam['phase']; k = np.round(rr); dist = np.abs(rr - k) * fam['s']
        ext = um & ti['in_proj'] & (dist <= fam['w'] / 2.0 + 1.0) & (k == min(fam['ids']) - 1)
        z = pts[um, 2]
        six[f'{r.surface}|{r.pattern}|{r.rule}'] = dict(u=int(um.sum()), segments_accepted_as_minus8=segs_m8, n_segments_minus8=len(segs_m8), u_segments=seg_u,
                                                       u_truth_k=sorted(int(x) for x in np.unique(k[um][np.isfinite(k[um])])), registered_ids=[int(min(fam['ids'])), int(max(fam['ids']))],
                                                       u_confirmable_if_id_minus9_allowed=int(ext.sum()), u_outside_projector_image=int((um & ~ti['in_proj']).sum()),
                                                       u_hit=int((um & ti['hit']).sum()), u_z_range_mm=[float(np.nanmin(z)), float(np.nanmax(z))])
    res['B2_six_records'] = six
    R6 = {k: v for k, v in six.items() if k.endswith('R@1.1247')}
    res['B2_summary'] = dict(weak_one_segment_minus8=all(v['n_segments_minus8'] == 1 for k, v in R6.items() if '|weak|' in k),
                             mid_strong_two_segments_minus8=all(v['n_segments_minus8'] == 2 for k, v in R6.items() if '|weak|' not in k),
                             ext_counts={k: f"{v['u_confirmable_if_id_minus9_allowed']}/{v['u']}" for k, v in R6.items()})

    X = C[(C.family == 'dots') & (C.rule == 'DF') & (C.arm == 'use') & (C.pattern != 'plain') & (C.sigma == 100)]
    res['B3_sigma100_unconfirmed'] = dict(DF_all=f'{int(X.u.sum())}/{int(X.acc.sum())}', DF_all_share=float(X.u.sum() / X.acc.sum()),
                                          by_layout={lay: dict(u=int(g.u.sum()), acc=int(g.acc.sum()), share=float(g.u.sum() / g.acc.sum())) for lay, g in X.groupby(X.layout.astype(str))})
    s40 = Pp[(Pp.pair_type == 'E1L') & (Pp.a_rule == 'R@1.1247') & (Pp.sigma == 40)]; Ci = C.set_index('eval_key'); pos = s40[s40.d_glob_Ctau > 0]
    res['B3_lines_R_sigma40'] = dict(pairs=len(s40), positive=len(pos), positive_flagged=int((Ci.loc[pos.a_eval_key, 'flag_wrong_all'] == 1).sum()))
    res['ok'] = bool(res['B1_8mm_R']['images'] == 792 and res['B1_8mm_R']['image_fallback'] == 643 and res['B1_8mm_R']['images_L3_adopted_gt0'] == 147
                     and res['B1_8mm_R']['L3_adopted_segments'] == 2364 and res['B1_8mm_R']['L3_wide_read_narrow'] == 263 and res['B1_8mm_R']['E1_pairs_nonzero'] == 0
                     and res['B1_8mm_R']['E1_pairs_R_and_LB'] == 1584 and res['B2_summary']['weak_one_segment_minus8'] and res['B2_summary']['mid_strong_two_segments_minus8'])
    return res


def _case_pair9(fig, gs, c, arr, ident, dg):

    import matplotlib.patches as mp
    from matplotlib.colors import ListedColormap
    import c8_figs as F8
    ax3 = fig.add_subplot(gs[0], projection='3d'); axm = fig.add_subplot(gs[1])
    mask = dg['mask']; lx, ly = dg['lattice_x'], dg['lattice_y']; sid = K.K4.RK.SURF_MAP[c['surface']]
    zt = np.full(mask.shape, np.nan); zt[mask] = dg[f'z_{sid}']; Xg, Yg = np.meshgrid(lx, ly)
    ax3.plot_surface(Xg[::8, ::8], Yg[::8, ::8], zt[::8, ::8], color='0.7', alpha=0.25, linewidth=0, antialiased=False)
    n_acc = int(c['n_accepted'])
    if arr is not None:
        pts, acc = arr['pts'], arr['acc']; ok = acc & np.isfinite(pts).all(axis=1)
        col = np.full(len(pts), '#1f77b4', dtype=object)
        if ident is not None and len(ident) == len(pts):
            col[ident == 'wr'] = '#d62728'; col[ident == 'un'] = '#9467bd'
        if ok.any():
            ax3.scatter(pts[ok, 0], pts[ok, 1], pts[ok, 2], s=0.6 if c['family'] == 'lines' else 2.0, c=list(col[ok]), depthshade=False)
        stm = np.full(mask.shape, np.nan); stm[mask] = arr['state_glob'].astype(float)
        axm.imshow(stm, origin='lower', extent=(lx[0], lx[-1], ly[0], ly[-1]), cmap=ListedColormap(['#d9d9d9', '#ff7f0e', '#2ca02c']), vmin=0, vmax=2)
    axm.add_patch(mp.Circle((0, 0), 15.0, fill=False, ls='--', lw=0.8, color='k'))
    if n_acc == 0:
        ax3.text2D(0.5, 0.55, 'keine akzeptierten Punkte', transform=ax3.transAxes, ha='center', color='#b22222', fontsize=7.5)
    ax3.set_xlim(-46, 46); ax3.set_ylim(-46, 46); ax3.set_zlim(-16, 4); ax3.view_init(elev=28, azim=-60); ax3.set_box_aspect((1, 1, 0.35))
    ax3.set_xticks([-40, 0, 40]); ax3.set_yticks([-40, 0, 40]); ax3.set_zticks([]); ax3.tick_params(labelsize=5.5, pad=-2)
    ax3.set_xlabel('x', fontsize=6.5, labelpad=-9); ax3.set_ylabel('y', fontsize=6.5, labelpad=-9)
    axm.tick_params(labelsize=6.5); axm.set_xticks([-40, 0, 40]); axm.set_yticks([-40, 0, 40]); axm.set_xlabel('x', fontsize=6.5, labelpad=0)
    line2 = f'{n_acc} akzeptierte Punkte' if n_acc > 0 else 'keine akzeptierten Punkte'
    if c.get('group') == 'posthoc':
        lay = F8.LAYDE[str(c['layout'])].split(' (')[0]; ru = 'R' if str(c['rule']).startswith('R@') else str(c['rule'])
        line2 = f"{F8.FAMDE[c['family']]} {lay}, {F8.LDE[c['pattern']]}, {ru}\n" + ('höchster Anteil f' if c['posthoc'] == 'risk' else 'Kollaps σ0 → σ20')
    ax3.set_title(f"{c['surface']} {c['phase_id']} W{c['rep']} σ{int(c['sigma'])}\n{line2}", fontsize=7.5)
    t = f"k {c['c']} · f {c['w']} · u {c['u']}\nCτ {F8._num(c['Ctau'])} · Cvalid {F8._num(c['Cvalid'])}\nRMSE Pkt {F8._num(c.get('rmse'))} · Fl. {F8._num(c.get('surf_rmse'))}"
    if c.get('n_interp') is not None and c.get('surf_rmse') is not None and int(c['n_interp']) < 1000:
        t += f" ({c['n_interp']} Kn.)"
    axm.set_title(t, fontsize=7.5)


def f13_14(T, base):

    import c8_figs as F8
    plt = F8._plt(); SC = T['SC']; logy = base == 'surf_rmse'
    fig, axs = plt.subplots(2, 3, figsize=(F8.W_IN, 5.4), sharex=True)
    for i, fam in enumerate(('dots', 'lines')):
        ru, ar = F8.MAIN[fam]
        for j, lay in enumerate(F8.LAY[fam]):
            ax = axs[i, j]
            for s in F8.SURF:
                q = SC[(SC.family == fam) & (SC.surface == s) & (SC.layout == lay) & (SC.rule == ru) & (SC.arm == ar) & SC.pattern.isin(F8.LEV)]
                for dom, ls, mk in (('glob', '-', 'o'), ('loc', '--', '^')):
                    col = f'{dom}_{base}_mean'; xs, ys, part = [], [], []
                    for sg, g in q.groupby('sigma'):
                        v = g[col].to_numpy(float); fin = np.isfinite(v); full = bool(fin.all()) and (not logy or bool((g[f'{dom}_{base}_n_defined'] >= g.n_rows).all()))
                        xs.append(sg); ys.append(float(v[fin].mean()) if fin.any() else np.nan); part.append(fin.any() and not full)
                    xs, ys, part = np.array(xs), np.array(ys), np.array(part)
                    ax.plot(xs, ys, ls, color=F8.SC_[s], marker=mk, ms=2.8)
                    if part.any():
                        ax.scatter(xs[part], ys[part], s=26, marker=mk, facecolors='white', edgecolors=F8.SC_[s], zorder=3)
            if logy:
                ax.set_yscale('log'); ax.set_ylim(*F8.YLIM_RMSE)
            else:
                ax.set_ylim(-0.03, 1.03)
            F8._de(ax, logy=logy); ax.set_title(f'{F8.FAMDE[fam]} {F8.LAYDE[lay]}'); ax.grid(alpha=0.25); ax.set_xticks(F8.SIG)
            if i == 1:
                ax.set_xlabel('σ (Eingangsrauschen)')
        axs[i, 0].set_ylabel('Flächen-RMSE [mm]' if logy else 'Cτ')
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=F8.SC_[s], lw=2, label=s) for s in F8.SURF] + [ml.Line2D([], [], color='0.3', ls='-', marker='o', ms=3, label='global (30.947 Knoten)'),
                                                                                ml.Line2D([], [], color='0.3', ls='--', marker='^', ms=3, label='lokal (r ≤ 15 mm, 22.605 Knoten)'),
                                                                                ml.Line2D([], [], color='0.3', ls='', marker='o', mfc='white', label='hohl: nicht alle Stufen/Bilder definiert')][:3 if logy else 2]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.075))
    fig.text(0.5, 0.012, 'Überblick: Mittel der drei Codestufen (kein einzelner Code), Phasen gleich gewichtet;\n' + ('Mittel nur über definierte Bilder; gemeinsame y-Skala' if logy else 'gemeinsame y-Skala'),
             ha='center', va='bottom', fontsize=7)
    fig.suptitle(f"Global gegen lokal: {'Flächen-RMSE (bedingt)' if logy else 'Cτ'}, codierte Hauptmethode (Punkte DF, Linien R)", fontsize=8)
    fig.tight_layout(rect=(0, 0.17, 1, 0.985), h_pad=0.8)
    F8._save(fig, 'S3C8_F14_Global_Lokal_Flaechenfehler' if logy else 'S3C8_F13_Global_Lokal_Ctau', f'case_phase_summary.csv (glob/loc_{base}_mean, *_n_defined, n_rows)',
             'layout: averaging / scale note moved below the legend (no overlap); data, scale and symbols as CODE-08', 'formal summary', True)


def task_c(T) -> dict:
    import c8_figs as F8
    orig_save = F8._save

    def save9(fig, name, inputs, change, identity, recommended, cases=''):
        name = name.replace('S3C8_', 'S3C9_')
        if name.startswith(('S3C9_F11', 'S3C9_F12')) and fig._suptitle is not None:
            fig._suptitle.set_text(fig._suptitle.get_text().replace('; RMSE in mm', '\nx, y, z und RMSE in mm (3D: z-Bereich −16 … 4 mm, gleiche Ansicht in allen Fällen)'))
            fig.subplots_adjust(top=1 - 1.0 / fig.get_size_inches()[1])
            change = 'CODE-09: informal normal/degraded labels removed, accepted points shown instead; axis letters and mm note added; same cases, values, colours and view'
        orig_save(fig, name, inputs, change, identity, recommended, cases)
    F8._save = save9; F8._case_pair = _case_pair9
    try:
        F8.INDEX.clear(); board = F8.case_figs(T); f13_14(T, 'Ctau'); f13_14(T, 'surf_rmse')
    finally:
        F8._save = orig_save
    out = dict(figures=[dict(x) for x in F8.INDEX])
    out['board'] = [dict(eval_key=c.get('eval_key'), group=c.get('group'), surface=c['surface'], sigma=c['sigma'], rep=c.get('rep'), n_accepted=c.get('n_accepted'), Ctau=c.get('Ctau'),
                         recompute=c.get('recompute'), max_diff=c.get('max_diff'), state_code07=c.get('state')) for c in board]
    return out


MANUAL = {
    'S3C9_F11a': dict(png=True, pdf=True, ok=True, note='no normal/degraded label; accepted-point line; x/y letters, mm + z-range note; legend in free slot; values = case_board'),
    'S3C9_F11b': dict(png=True, pdf=True, ok=True, note='sigma40 case no longer labelled normal (Ctau 0.228 < sigma20 0.403 shown as numbers only); same cases / view / colours'),
    'S3C9_F12a': dict(png=True, pdf=True, ok=True, note='three surfaces sigma 0/40; labels removed; units note; no overlap'),
    'S3C9_F12b': dict(png=True, pdf=True, ok=True, note='three surfaces sigma 0/40; labels removed; units note; no overlap'),
    'S3C9_F12c': dict(png=True, pdf=None, ok=True, note='post-hoc cases; "keine akzeptierten Punkte" only for the n_accepted = 0 case (B N4 sigma20); appendix, PNG only'),
    'S3C9_F13': dict(png=True, pdf=True, ok=True, note='same layout fix as F14 (title line touched the top-row titles in the PDF); averaging note below the legend'),
    'S3C9_F14': dict(png=True, pdf=True, ok=True, note='no overlap; averaging / scale note below the legend; hollow markers explained'),
    'S3C8_F05a': dict(png=True, pdf=True, ok=True, note='unchanged; defined images / minimum nodes visible; common scale'),
    'S3C8_F05b': dict(png=True, pdf=True, ok=True, note='unchanged; "1/16 · 1 Kn. · 0,000102" visible; N1 note in title'),
    'S3C8_F06a': dict(png=True, pdf=True, ok=True, note='unchanged; all-image denominator, undefined share, u share'),
    'S3C8_F06b': dict(png=True, pdf=True, ok=True, note='unchanged; all-image denominator, undefined share, u share'),
    'S3C8_F07': dict(png=True, pdf=True, ok=True, note='unchanged; pre/post-guard accuracy and abstention with stated denominators'),
    'S3C8_F15': dict(png=True, pdf=True, ok=True, note='unchanged; B shown, M10 checked numerically (diagnostic_cases.csv); post-hoc'),
}


def figure_index(fig_entry: dict) -> pd.DataFrame:

    c7f, c8f, c9f = K.C7_CUR / 'figures', C8 / 'figures', FIGS
    rows = []

    def add(no, png, status, identity, rec, inputs, replaces='', replaced_by='', note=''):
        pdf = png.with_suffix('.pdf')
        rows.append(dict(figure=no, status=status, png=K.rel(png), pdf=(K.rel(pdf) if pdf.exists() else ''), identity=identity, recommended_main_text=rec, inputs=inputs,
                         replaces=replaces, replaced_by=replaced_by, note=note, sha256_png=K.sha_file(png), sha256_pdf=(K.sha_file(pdf) if pdf.exists() else '')))
    cur = {}
    for no, f, ident, rec, inp in (('F01', 'S3C7_F01_E1_Klassenbeitrag_Punkte.png', 'formal summary', True, 'pair_phase_summary (E1 DF use-ignore)'),
                                   ('F02', 'S3C7_F02_E1_Klassenbeitrag_Linien.png', 'formal summary', True, 'pair_phase_summary (E1 R-ignore / LB-ignore)'),
                                   ('F03', 'S3C7_F03_Schema_Ctau_Punkte.png', 'formal summary', True, 'case_phase_summary (plain vs coded C_tau)'),
                                   ('F04', 'S3C7_F04_Schema_Ctau_Linien.png', 'formal summary', True, 'case_phase_summary (P4 vs R C_tau)'),
                                   ('F08', 'S3C7_F08_Frontend_Vergleich.png', 'formal summary', False, 'pair_phase_summary (E3)'),
                                   ('F09', 'S3C7_F09_Bedingungskarte.png', 'formal summary', False, 'condition_map'),
                                   ('F10', 'S3C7_F10_Diagnose_Referenzen.png', 'formal summary', False, 'diag_summary')):
        add(no, c7f / f, 'current (CODE-07, unchanged)', ident, rec, inp); cur[no] = K.rel(c7f / f)
    for no, f, ident, inp in (('F05a', 'S3C8_F05a_Flaechenfehler_Punkte.png', 'formal summary', 'case_phase_summary, cases'), ('F05b', 'S3C8_F05b_Flaechenfehler_Linien.png', 'formal summary', 'case_phase_summary, cases'),
                              ('F06a', 'S3C8_F06a_Risiko_Punkte.png', 'formal summary', 'risk_summary'), ('F06b', 'S3C8_F06b_Risiko_Linien.png', 'formal summary', 'risk_summary'),
                              ('F07', 'S3C8_F07_Lesemechanismus.png', 'formal summary', 'readout_summary'), ('F15', 'S3C8_F15_N1_Randanomalie.png', 'post-hoc diagnosis', 'stored images / records')):
        old = {'F05a': 'S3C7_F05_Flaechenfehler.png', 'F05b': 'S3C7_F05_Flaechenfehler.png', 'F06a': 'S3C7_F06_Identitaetsrisiko.png', 'F06b': 'S3C7_F06_Identitaetsrisiko.png',
               'F07': 'S3C7_F07_Lesemechanismus.png'}.get(no, '')
        add(no, c8f / f, 'current (CODE-08, unchanged)', ident, True, inp, replaces=(K.rel(c7f / old) if old else '')); cur[no] = K.rel(c8f / f)
    for e in fig_entry['figures']:
        p = ROOT / e['figure']; no = p.stem.split('_')[1]; old8 = c8f / p.name.replace('S3C9_', 'S3C8_')
        old7 = {'F11a': 'S3C7_F11_Faelle_Punkte.png', 'F12a': 'S3C7_F11_Faelle_Punkte.png', 'F11b': 'S3C7_F12_Faelle_Linien_PostHoc.png', 'F12b': 'S3C7_F12_Faelle_Linien_PostHoc.png',
                'F12c': 'S3C7_F12_Faelle_Linien_PostHoc.png'}.get(no, '')
        add(no, p, 'current (CODE-09)', e['identity'], bool(e['recommended_main_text']), e['inputs'], replaces=';'.join(x for x in (K.rel(old8), K.rel(c7f / old7) if old7 else '') if x),
            note=e['change']); cur[no] = K.rel(p)
    for f in sorted(c8f.glob('S3C8_*.png')):
        no = f.stem.split('_')[1]
        if cur.get(no) != K.rel(f):
            add(no, f, 'superseded', '', False, '', replaced_by=cur[no])
    for f, nos in (('S3C7_F05_Flaechenfehler.png', 'F05a;F05b'), ('S3C7_F06_Identitaetsrisiko.png', 'F06a;F06b'), ('S3C7_F07_Lesemechanismus.png', 'F07'),
                   ('S3C7_F11_Faelle_Punkte.png', 'F11a;F12a'), ('S3C7_F12_Faelle_Linien_PostHoc.png', 'F11b;F12b;F12c')):
        add(f.split('_')[1], c7f / f, 'superseded', '', False, '', replaced_by=';'.join(cur[n] for n in nos.split(';')))
    return pd.DataFrame(rows)


def final(T) -> dict:
    c = K.load_json(CHECKS); start = c['start']; now = hashes(T)
    changed = sorted(f for f, h in start['files'].items() if now['files'].get(f) != h) + sorted(f for f, h in start['records'].items() if now['records'].get(f) != h)
    B = K.load_json(K.C7_CUR / 'case_board.json'); fig = c.get('figs', {})
    cur = {x['eval_key']: x for x in fig.get('board', [])}
    case_ok = all(abs(cur[b['eval_key']]['Ctau'] - b['Ctau']) <= 1e-12 and cur[b['eval_key']]['n_accepted'] == b['n_accepted'] for b in B if b.get('eval_key'))
    no_out = all((b['state'] == 'no_output') == (int(b['n_accepted']) == 0) for b in B if b.get('eval_key'))
    FI = figure_index(fig); K.save_csv(FI, OUT / 'figure_index.csv')
    cur = FI[FI.status.str.startswith('current')]
    idx = dict(rows=len(FI), current=len(cur), superseded=int((FI.status == 'superseded').sum()), one_current_per_figure=bool(cur.figure.is_unique),
               new_pngs_listed=bool(set(K.rel(p) for p in FIGS.glob('*.png')) <= set(cur.png)), pdf_for_every_recommended_new=bool((cur[cur.png.str.contains('s3_code09') & cur.recommended_main_text].pdf != '').all()))
    outs = {K.rel(p): K.sha_file(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'checks.json'}
    code = {K.rel(p): K.sha_file(p) for p in sorted(HERE.glob('*.py'))}
    res = dict(freeze_end=now['freeze'], inputs_unchanged=not changed, changed=changed, n_files=now['n_files'], n_records=now['n_records'], case_keys_and_values_unchanged=bool(case_ok),
               no_output_label_only_for_zero_accepted=bool(no_out), sandbox_exists=(OUT / 'sandbox').exists(), figure_index=idx, outputs=outs, code=code, manual_figure_checks=MANUAL,
               earlier_entries={k: c[k].get('ok') for k in c if isinstance(c[k], dict) and 'ok' in c[k]})
    res['ok'] = bool(now['freeze']['ok'] and not changed and case_ok and no_out and not res['sandbox_exists'] and MANUAL and all(v.get('ok') for v in MANUAL.values())
                     and all(v for k, v in idx.items() if k not in ('rows', 'current', 'superseded')) and c['diag_2']['ok'] and c['facts']['ok'])
    return res


def main() -> None:
    cmd = sys.argv[1]; name = sys.argv[2] if len(sys.argv) > 2 else cmd; T = K.tables()
    if cmd == 'start':
        r = hashes(T); checks_add(name, r); print(json.dumps(K.clean(dict(freeze=r['freeze'], n_files=r['n_files'], n_records=r['n_records'])), indent=1))
    elif cmd == 'diag':
        R, chk = task_a(T); K.save_csv(R, OUT / 'diagnostic_pairs.csv'); chk['csv_sha256'] = K.sha_file(OUT / 'diagnostic_pairs.csv'); checks_add(name, chk)
        print(R[['surface', 'pattern', 'domain', 'R_Ctau_recomputed', 'ign_Ctau_recomputed', 'dCtau_formal', 'dCtau_diag', 'change', 'change_phase_equal_contribution']].to_string())
        print(json.dumps(K.clean({k: v for k, v in chk.items() if k != 'alignment'}), indent=1)[:4000])
    elif cmd == 'facts':
        r = task_b(T); checks_add(name, r); print(json.dumps(K.clean(r), indent=1, ensure_ascii=False)[:8000])
    elif cmd == 'figs':
        r = task_c(T); checks_add(name, r); print(json.dumps(K.clean(r['figures']), indent=1, ensure_ascii=False)[:3000])
    elif cmd == 'final':
        r = final(T); checks_add(name, r); print(json.dumps(K.clean({k: v for k, v in r.items() if k not in ('outputs',)}), indent=1, ensure_ascii=False)[:5000])


if __name__ == '__main__':
    main()
