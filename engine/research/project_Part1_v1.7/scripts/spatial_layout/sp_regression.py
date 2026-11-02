# Compare regular position-table lines with the continuous-line implementation.

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

import sp_common as S
import sp_dots_columns as DC
import sp_lines_table as LT
import l01rb_common as B
import l01_common as C
import m4_lines as ML
import m4_reconstruction as M4
import sl_lib as L
import sn_lib as SL
import r5_detect as RD
from m2_pattern import generate_pattern

FORMAL = S.ROOT / 'outputs' / 'line01r' / 'formal' / 'cases'
MANIFEST = S.ROOT / 'outputs' / 'line01r' / 'c3' / 'night' / 'formal_manifest_4928_c3.csv'
P4_JSON = S.ROOT / 'outputs' / 'line01r' / 'c3' / 'config' / 'paths_c3_validation.json'
DOT_CSV = S.ROOT / 'outputs' / 'line01r' / 'night01r2' / 'dot' / 'dot_common_results_v3r1.csv'
DOT_REUSE = S.ROOT / 'outputs' / 'line01r' / 'c3' / 'dot_reuse' / 'dot_reuse_manifest.csv'


def p4_params() -> dict:
    pp = dict(json.loads(P4_JSON.read_text(encoding='utf-8'))['P4_C3_cal']); pp['steger_noise_coef'] = {float(k): float(v) for k, v in pp['steger_noise_coef'].items()}
    return pp


def _eq(a, b, f32: bool = False) -> bool:
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == 'f' or b.dtype.kind == 'f':
        if f32:
            a = a.astype(np.float32); b = b.astype(np.float32)
        return bool(np.array_equal(np.nan_to_num(a.astype(np.float64), nan=-9e9), np.nan_to_num(b.astype(np.float64), nan=-9e9)))
    return bool(np.array_equal(a, b))


def _maxdiff(a, b) -> float | None:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return None
    m = np.isfinite(a) & np.isfinite(b)
    if (np.isfinite(a) != np.isfinite(b)).any():
        return float('inf')
    return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0


def regular_equivalence() -> dict:
    rows = []
    for s_mm, ph, phi in ((4.45, 0.0, 0.0), (4.45, -0.46875, 0.0), (6.3, 0.0, 0.0), (6.3, 0.28125, 0.0), (8.0, 0.21875, 0.0), (4.45, 0.15625, 45.0), (6.3, -0.09375, 45.0)):
        pm, mm = generate_pattern('olines', spacing_mm=s_mm, feature_size_mm=S.W_MM, line_angle_deg=phi, line_phase_frac=ph)
        pt, mt = LT.table_spec(s_mm, S.W_MM, ph, None, phi_deg=phi)
        o2, ot = mm['olines'], mt['olines']
        seg_ok = set(o2['segments']) == set(ot['segments']) and all(abs(o2['segments'][k]['q_px'] - ot['segments'][k]['q_px']) < 1e-12 and o2['segments'][k]['lit_px'] == ot['segments'][k]['lit_px'] and abs(o2['segments'][k]['length_px'] - ot['segments'][k]['length_px']) < 1e-9 for k in o2['segments'])
        rows.append(dict(kind='lines_raster', s_mm=s_mm, phase_frac=ph, phi_deg=phi, raster_equal=bool(np.array_equal(np.asarray(pm, np.float32), pt)), xor_px=int((np.asarray(pm) != pt).sum()), ids_equal=(list(o2['ids']) == list(ot['ids'])),
                         segments_equal=bool(seg_ok), clipped_equal=(set(o2['clipped_lines']) == set(ot['clipped_lines'])), n_lines=o2['n_lines']))
        fam_m2 = ML.line_family(o2); fam_t = LT.line_family_table(LT.spec_from_m2(mm))
        rows.append(dict(kind='lines_family_q', s_mm=s_mm, phase_frac=ph, phi_deg=phi, q_equal=bool(np.array_equal(ML.line_q(fam_m2, fam_m2['ids']), fam_t['q'])), ids_equal=bool(np.array_equal(fam_m2['ids'], fam_t['ids']))))
        cl2 = ML.raster_centrelines(np.asarray(pm, np.float32), fam_m2); clt = LT.raster_centrelines_table(pt, fam_t)
        rows.append(dict(kind='lines_raster_centrelines', s_mm=s_mm, phase_frac=ph, phi_deg=phi, dense_equal=_eq(cl2['_u_of_row'], clt['_u_of_row']),
                         rows_equal=all(_eq(cl2[int(k)]['rows'], clt[int(k)]['rows']) and _eq(cl2[int(k)]['u'], clt[int(k)]['u']) for k in fam_m2['ids'])))
    for name in S.DOT_BASELINES:
        for (u, v) in [S.REF_PHASE_DOT] + [L.positions()[i] for i in L.latin_indices()[:4]]:
            c0 = L.centres(name, u, v); lay = DC.layout(*S.DOT_BASELINES[name], u, v, None, 'u')
            b0 = SL.binary_pattern(c0, S.R_MM); bt = DC.raster(lay['c'])
            rows.append(dict(kind='dots_centres_raster', baseline=name, u=u, v=v, centres_equal=bool(np.array_equal(c0, lay['c'])), raster_equal=bool(np.array_equal(b0, bt)), n=int(len(c0))))
    df = pd.DataFrame(rows)
    ok = bool(all(bool(r.get(k, True)) for r in rows for k in ('raster_equal', 'ids_equal', 'segments_equal', 'clipped_equal', 'q_equal', 'dense_equal', 'rows_equal', 'centres_equal')))
    return dict(passed=ok, n_checks=len(rows), rows=rows)


def select_regression_cases() -> list[dict]:

    man = pd.read_csv(MANIFEST, low_memory=False)
    man = man[(man.w_mm == 1.0) & (man.noise_rep == 0)]
    want = []
    for s in (4.45, 6.3):
        for surf in ('baseline', 'curv_x_strong'):
            for reg in ('fixed_peak', 'equal_integral'):
                for sg in (0.0, 40.0):
                    want.append(dict(phi_deg=0.0, s_mm=s, surface=surf, regime=reg, sigma=sg, reason=f'0deg s{s} {surf} {reg} sigma{sg:g}'))
    for reg in ('fixed_peak', 'equal_integral'):
        want.append(dict(phi_deg=0.0, s_mm=8.0, surface='baseline', regime=reg, sigma=0.0, reason='0deg s8 geometric reference, baseline sigma0'))
        want.append(dict(phi_deg=0.0, s_mm=8.0, surface='curv_x_strong', regime=reg, sigma=40.0, reason='0deg s8 geometric reference, curvature sigma40'))
    want += [dict(phi_deg=45.0, s_mm=4.45, surface='baseline', regime='fixed_peak', sigma=0.0, reason='45deg interface compatibility'),
             dict(phi_deg=45.0, s_mm=4.45, surface='curv_x_strong', regime='equal_integral', sigma=40.0, reason='45deg interface compatibility'),
             dict(phi_deg=45.0, s_mm=6.3, surface='baseline', regime='fixed_peak', sigma=40.0, reason='45deg interface compatibility'),
             dict(phi_deg=45.0, s_mm=8.0, surface='curv_x_strong', regime='equal_integral', sigma=0.0, reason='45deg interface compatibility')]
    out = []
    for w in want:
        pi = 0 if w['sigma'] == 0.0 else 1
        sel = man[(man.phi_deg == w['phi_deg']) & (man.s_mm == w['s_mm']) & (man.surface == w['surface']) & (man.regime == w['regime']) & (man.sigma == w['sigma']) & (man.phase_index == pi)]
        if len(sel) == 0:
            out.append(dict(w, case_id=None, missing=True)); continue
        r = sel.iloc[0].to_dict()
        out.append(dict(w, case_id=r['case_id'], phase_index=int(r['phase_index']), phase_frac=float(r['phase_frac']), peak=float(r['peak']), raster_sha256=r['raster_sha256'], json=S.rel(FORMAL / f"{r['case_id']}.json"), npz=S.rel(FORMAL / f"{r['case_id']}.npz")))
    assert len(out) <= 24
    return out


def regression_run(cases: list[dict], max_decodes: int = 48) -> dict:

    import l01_truth as T
    import l01rb_run as RUN
    p4 = p4_params(); rows = []; n_dec = 0; t0 = time.time()
    for c in cases:
        if c.get('missing'):
            rows.append(dict(c, status='missing')); continue
        rec = B.load_json(FORMAL / f"{c['case_id']}.json")
        with np.load(FORMAL / f"{c['case_id']}.npz") as z:
            img = np.asarray(z['image']); stored = {k: np.asarray(z[k]) for k in z.files if k.startswith('P4_C3_cal_') or k.startswith('ref_')}
        spec = rec['spec']
        pat_bin, meta = B.pattern_for(spec); pat = (pat_bin * np.float32(spec['peak'])).astype(np.float32)
        raster_ok = B.sha_array(pat_bin) == spec['raster_sha256']
        pat_t, meta_t = LT.table_spec(spec['s_mm'], spec['w_mm'], spec['phase_frac'], None, phi_deg=spec['phi_deg'])
        gen_ok = bool(np.array_equal(pat_bin.astype(np.float32), pat_t))
        cfg = B.m4_cfg(spec['peak'])
        if n_dec + 2 > max_decodes:
            rows.append(dict(c, status='not_run_budget')); continue
        t1 = time.time(); r_old = ML.run_olines(img, meta['olines'], cfg, pattern=pat, **p4); n_dec += 1; t2 = time.time()
        r_new = LT.run_p4_table(img, LT.spec_from_m2(meta), cfg, pat, p4); n_dec += 1; t3 = time.time()
        cmp = {k: _eq(r_old[k], r_new[k]) for k in ('features', 'comp', 'chain', 'segment_id', 'guard_reason', 'n_candidates', 'admissible', 'candidate_ids', 'line_id', 'status', 'points', 'proj_uv')}
        cmp['counts_equal'] = (r_old['counts'] == r_new['counts']); cmp['plane_check_equal'] = (r_old['checks']['plane_check_max_mm'] == r_new['checks']['plane_check_max_mm'])
        diffs = dict(points_max_abs_mm=_maxdiff(r_old['points'], r_new['points']), proj_uv_max_abs_px=_maxdiff(r_old['proj_uv'], r_new['proj_uv']))
        stored_cmp = dict(status=_eq(stored['P4_C3_cal_status'], r_old['status']), line_id=_eq(stored['P4_C3_cal_line_id'], r_old['line_id']), points_f32=_eq(stored['P4_C3_cal_points'], r_old['points'], f32=True),
                          new_status=_eq(stored['P4_C3_cal_status'], r_new['status']), new_line_id=_eq(stored['P4_C3_cal_line_id'], r_new['line_id']), new_points_f32=_eq(stored['P4_C3_cal_points'], r_new['points'], f32=True))
        Ssurf = RUN.surf(spec['surface'])
        ti_old = T.truth_identity(Ssurf['mesh'], r_old['features'], meta['olines']); ti_new = LT.truth_identity_table(Ssurf['mesh'], r_new['features'], LT.spec_from_m2(meta))
        truth_cmp = dict(k=_eq(ti_old['k'], ti_new['k']), confirmable=_eq(ti_old['confirmable'], ti_new['confirmable']), stored_truth_k=_eq(stored['P4_C3_cal_truth_k'], ti_old['k']), stored_confirmable=_eq(stored['P4_C3_cal_confirmable'], ti_old['confirmable']),
                         n_multi_candidate_new=int(ti_new['n_multi_candidate']))
        ref_old = T.line_reference(Ssurf['mesh'], meta['olines'], 1.0, pattern=pat); ref_new = LT.line_reference_table(Ssurf['mesh'], LT.spec_from_m2(meta), 1.0, pattern=pat)
        ref_cmp = dict(u=_eq(ref_old['rows']['u'], ref_new['rows']['u']), k=_eq(ref_old['rows']['k'], ref_new['rows']['k']), v=_eq(ref_old['rows']['v'], ref_new['rows']['v']), inner=_eq(ref_old['rows']['inner'], ref_new['rows']['inner']), g_xyz=_eq(ref_old['g_xyz'], ref_new['g_xyz']),
                       stored_ref_u_f32=_eq(stored['ref_u'], ref_old['rows']['u'], f32=True), stored_ref_k=_eq(stored['ref_k'], ref_old['rows']['k']))
        E_old = RUN.evaluate(r_old['points'], r_old['accepted'], Ssurf); E_new = RUN.evaluate(r_new['points'], r_new['accepted'], Ssurf)
        def pick(E):
            d = E['r3']['sub']['dent']['by_d']['6.0']; a = E.get('amplitude') or {}
            return dict(ctau15_d6=d['C_tau']['0.15'], cvalid_d6=d['C_valid'], amp_rel_pct=a.get('rel_err_pct'), rmse_mm=E['pts_ref_rmse_mm'], gross_rate=E['pts_ref_gross_rate'], n_accepted=E['n_accepted'])
        eo, en = pick(E_old), pick(E_new)
        sp = rec['paths']['P4_C3_cal']['P']; es = dict(ctau15_d6=sp['r3']['sub']['dent']['by_d']['6.0']['C_tau']['0.15'], cvalid_d6=sp['r3']['sub']['dent']['by_d']['6.0']['C_valid'], amp_rel_pct=(sp.get('amplitude') or {}).get('rel_err_pct'), rmse_mm=sp['pts_ref_rmse_mm'], gross_rate=sp['pts_ref_gross_rate'], n_accepted=sp['n_accepted'])
        eval_cmp = dict(old_vs_new_equal=(eo == en), old_vs_stored_equal=(eo == es), old=eo, new=en, stored=es)
        all_ok = all(cmp.values()) and all(truth_cmp[k] for k in ('k', 'confirmable')) and all(ref_cmp[k] for k in ('u', 'k', 'v', 'inner', 'g_xyz')) and eval_cmp['old_vs_new_equal']
        rows.append(dict(c, status='ok', raster_sha_ok=raster_ok, table_raster_equals_m2=gen_ok, n_features=int(len(r_old['features'])), n_accepted_old=int(r_old['accepted'].sum()), n_accepted_new=int(r_new['accepted'].sum()),
                         arrays_equal=cmp, diffs=diffs, stored_vs_now=stored_cmp, truth=truth_cmp, reference=ref_cmp, evaluation=eval_cmp, all_equal=bool(all_ok), seconds_old=round(t2 - t1, 2), seconds_new=round(t3 - t2, 2),
                         stored_eval_key=rec['eval_key'], current_eval_key=B.eval_key(), stored_src_m4_lines=rec['src_versions']['src/m4_lines.py'][:12], current_src_m4_lines=B.src_versions()['src/m4_lines.py'][:12]))
        S.log(f"regression {c['case_id'][:12]} {c['reason']}: all_equal={all_ok} feat={len(r_old['features'])} acc={int(r_old['accepted'].sum())}/{int(r_new['accepted'].sum())} stored_status={stored_cmp['status']} pts_f32={stored_cmp['points_f32']}", 'regression')
    ok_rows = [r for r in rows if r.get('status') == 'ok']
    return dict(n_cases=len(cases), n_run=len(ok_rows), n_decodes=n_dec, decode_budget=max_decodes, all_equal=bool(ok_rows and all(r['all_equal'] for r in ok_rows)),
                stored_status_equal_all=bool(ok_rows and all(r['stored_vs_now']['status'] for r in ok_rows)), stored_points_f32_equal_all=bool(ok_rows and all(r['stored_vs_now']['points_f32'] for r in ok_rows)),
                table_raster_equals_m2_all=bool(ok_rows and all(r['table_raster_equals_m2'] for r in ok_rows)), seconds=round(time.time() - t0, 1), rows=rows)


def _camera_uv_of_projector_point(Q_uv: np.ndarray, z_mm: float):

    g = C.geom(); o, d = C.rays(np.asarray(Q_uv, np.float64).reshape(-1, 2), g['proj'])
    t = (z_mm - o[:, 2]) / d[:, 2]; X = o + t[:, None] * d
    uv, inside = C.project(X, g['cam'])
    return uv, X, inside


# Compare table-based line geometry with regular-line analytical references.
def analytic_tests() -> dict:
    out = []
    s_mm = 4.45; s_px = S.px(s_mm); w_px = S.px(S.W_MM)
    cfg = B.m4_cfg(1.0)

    disp = {-3: 0.9 * s_px / 2, -2: -0.3 * s_px, 1: 0.25 * s_px, 4: -0.4 * s_px}
    pat, meta = LT.table_spec(s_mm, S.W_MM, 0.0, disp); o = meta['olines']
    fam = LT.line_family_table(o)
    o_perm = dict(o, ids=list(reversed(o['ids'])))
    fam_p = LT.line_family_table(o_perm)

    feats = []; truth_k = []; depths = []
    for k in (-5, -3, -2, 0, 1, 4, 6):
        for z in (0.5, 5.0, 12.0):
            q = fam['q_of'][k]; Q = np.array([[fam['c'][0] + q, 200.0 + 10 * k]])
            uv, X, inside = _camera_uv_of_projector_point(Q, z)
            if inside[0]:
                feats.append(uv[0]); truth_k.append(k); depths.append(z)
    feats = np.array(feats); truth_k = np.array(truth_k)
    cand = LT.line_candidates_table(feats, fam, cfg); cand_p = LT.line_candidates_table(feats, fam_p, cfg)
    col = {int(k): j for j, k in enumerate(cand['ids'])}; col_p = {int(k): j for j, k in enumerate(cand_p['ids'])}
    adm_by_id = {k: cand['admissible'][:, col[k]] for k in col}; adm_by_id_p = {k: cand_p['admissible'][:, col_p[k]] for k in col_p}
    perm_ok = all(np.array_equal(adm_by_id[k], adm_by_id_p[k]) for k in col)
    true_adm = np.array([cand['admissible'][i, col[int(k)]] for i, k in enumerate(truth_k)])
    z_tri = np.array([cand['z'][i, col[int(k)]] for i, k in enumerate(truth_k)])
    out.append(dict(test='T1_id_permutation_and_true_line_admissible', passed=bool(perm_ok and true_adm.all() and np.abs(z_tri - np.array(depths)).max() < 0.05), perm_invariant=bool(perm_ok), true_admissible=int(true_adm.sum()), n=int(len(feats)),
                    max_depth_error_mm=float(np.abs(z_tri - np.array(depths)).max()), n_candidates_hist={str(int(v)): int(c) for v, c in zip(*np.unique(cand['admissible'].sum(axis=1), return_counts=True))}))

    gap = 0.6 * w_px; disp2 = {1: -(s_px - gap) / 2, 2: (s_px - gap) / 2 - s_px + 0.0}

    disp2 = {2: -(s_px - gap)}
    pat2, meta2 = LT.table_spec(s_mm, S.W_MM, 0.0, disp2); o2 = meta2['olines']; fam2 = LT.line_family_table(o2)
    q1, q2 = fam2['q_of'][1], fam2['q_of'][2]
    cl2 = LT.raster_centrelines_table(pat2, fam2)
    qm = 0.5 * (q1 + q2); Qm = np.array([[fam2['c'][0] + qm, 255.0]]); uvm, Xm, _ = _camera_uv_of_projector_point(Qm, 0.0)
    candm = LT.line_candidates_table(uvm, fam2, cfg); colm = {int(k): j for j, k in enumerate(candm['ids'])}
    both = bool(candm['admissible'][0, colm[1]] and candm['admissible'][0, colm[2]])
    mesh_plane = C.surface('plane')[0]
    ti = LT.truth_identity_table(mesh_plane, uvm, o2)
    out.append(dict(test='T2_close_pair', passed=bool(abs((q2 - q1) - gap) < 1e-9 and 1 in cl2 and 2 in cl2 and len(cl2[1]['rows']) > 0 and len(cl2[2]['rows']) > 0 and both and bool(ti['multi_candidate'][0]) and not bool(ti['confirmable'][0])),
                    gap_px=float(q2 - q1), lit_rows_line1=int(len(cl2[1]['rows'])), lit_rows_line2=int(len(cl2[2]['rows'])), both_admissible=both, truth_multi_candidate=bool(ti['multi_candidate'][0]), truth_confirmable=bool(ti['confirmable'][0]), n_within_band=int(ti['n_within_band'][0])))

    base = LT.regular_table(s_mm, 0.0); q_missing = base[3]
    o3 = json.loads(json.dumps(o)); o3['ids'] = [k for k in o3['ids'] if k != 3]; o3['table']['q_px'].pop('3'); o3['table']['ids_all'] = [k for k in o3['table']['ids_all'] if k != 3]; o3['segments'].pop('3', None)
    fam3 = LT.line_family_table(o3)
    Q3 = np.array([[fam3['c'][0] + q_missing, 255.0]]); uv3, _, _ = _camera_uv_of_projector_point(Q3, 2.0)
    cand3 = LT.line_candidates_table(uv3, fam3, cfg)
    out.append(dict(test='T3_missing_segment', passed=bool(3 not in [int(k) for k in cand3['ids']] and not any(int(k) == 3 for k in fam3['ids'])), n_candidates_at_missing_position=int(cand3['admissible'][0].sum()), candidates=[int(k) for k in np.asarray(cand3['ids'])[cand3['admissible'][0]]]))

    ids_reg = [int(k) for k in o['ids']]; k_out = max(ids_reg); q_out = base[k_out]; half = (S.IMG - 1) / 2.0
    disp4 = {k_out: (half + 0.5 + 1.0) - q_out}
    pat4, meta4 = LT.table_spec(s_mm, S.W_MM, 0.0, disp4); o4 = meta4['olines']
    out.append(dict(test='T4_boundary_partially_lit', passed=bool(str(k_out) in o4['clipped_lines'] and k_out not in [int(k) for k in o4['ids']] and o4['clipped_lines'][str(k_out)]['lit_px'] > 0), clipped=o4['clipped_lines'].get(str(k_out)), n_lines=o4['n_lines']))

    try:
        LT.table_spec(s_mm, S.W_MM, 0.0, {2: -s_px}); dup_err = False
    except ValueError:
        dup_err = True
    out.append(dict(test='T5_duplicate_position_rejected', passed=dup_err))

    h_a = LT.table_hash(o); h_b = LT.table_hash(LT.table_spec(s_mm, S.W_MM, 0.0, {**disp, 0: 0.3})[1]['olines']); h_c = LT.table_hash(LT.table_spec(s_mm, S.W_MM, 0.0, dict(disp))[1]['olines'])
    out.append(dict(test='T6_key_changes_with_displacement', passed=bool(h_a != h_b and h_a == h_c), hash_a=h_a, hash_b=h_b))

    Qs = np.array([[fam2['c'][0] + q1, 255.0], [fam2['c'][0] + qm, 255.0], [fam2['c'][0] + q2, 255.0], [fam2['c'][0] + fam2['q_of'][-4], 255.0]]); uv7, _, _ = _camera_uv_of_projector_point(Qs, 0.0)
    ti7 = LT.truth_identity_table(mesh_plane, uv7, o2)
    out.append(dict(test='T7_truth_double_candidate', passed=bool(ti7['multi_candidate'][:3].all() and not ti7['confirmable'][:3].any() and ti7['confirmable'][3] and int(ti7['k'][3]) == -4), multi=[bool(x) for x in ti7['multi_candidate']], confirmable=[bool(x) for x in ti7['confirmable']], k=[int(x) for x in ti7['k']], dist_px=[float(x) for x in ti7['dist_px']]))

    import l01_truth as T
    pm, mm = generate_pattern('olines', spacing_mm=s_mm, feature_size_mm=S.W_MM, line_angle_deg=0.0, line_phase_frac=0.0)
    Qr = np.array([[mm['olines']['centre_uv'][0] + (k + 0.0) * s_px + dq, 100.0 + 20 * i] for i, (k, dq) in enumerate([(-4, 0.0), (-1, 1.2), (0, -2.0), (3, 3.5), (5, 0.4)])]); uvr, _, _ = _camera_uv_of_projector_point(Qr, 1.0)
    a_ = T.truth_identity(mesh_plane, uvr, mm['olines']); b_ = LT.truth_identity_table(mesh_plane, uvr, LT.spec_from_m2(mm))
    out.append(dict(test='T8_regular_truth_equals_frozen', passed=bool(np.array_equal(a_['k'], b_['k']) and np.array_equal(a_['confirmable'], b_['confirmable'])), k_old=[int(x) for x in a_['k']], k_new=[int(x) for x in b_['k']], conf=[bool(x) for x in a_['confirmable']]))
    return dict(passed=bool(all(t['passed'] for t in out)), tests=out)


def eval_binding_check(reg_rows: list[dict]) -> dict:
    import l01rb_run as RUN
    units = []
    csv = pd.read_csv(DOT_CSV, low_memory=False); reuse = pd.read_csv(DOT_REUSE, low_memory=False)
    picks = [('S1', 'D0', 'fixed_peak', 20.0, 'ok'), ('S2', 'D0', 'equal_integral', 40.0, 'ok'), ('S3', 'coded', None, 30.0, 'ok'), ('S1', 'D0', None, None, 'no_output')]
    for sec, method, regime, sigma, st in picks:
        q = csv[(csv.section == sec) & (csv.method == method) & (csv.status == st)]
        if regime is not None:
            q = q[(q.regime_type == regime)]
        if sigma is not None:
            q = q[(q.sigma == sigma)]
        q = q.sort_values('case_key')
        if len(q) == 0:
            units.append(dict(section=sec, method=method, regime=regime, sigma=sigma, status='no_such_unit')); continue
        r = q.iloc[0].to_dict()
        rr = reuse[(reuse.case_key == r['case_key']) & (reuse.study == r['study'])]
        rr = rr[rr.batch == r['batch']] if (rr.batch == r['batch']).any() else rr
        if len(rr) == 0:
            units.append(dict(section=sec, case_key=r['case_key'], status='npz_not_in_reuse_manifest')); continue
        npz = S.ROOT / rr.iloc[0]['npz_path']
        pk, ak = ('pts', 'amb')
        with np.load(npz, allow_pickle=(r['study'] == 'encoding')) as z:
            pts = np.asarray(z[pk], np.float64); amb = np.asarray(z[ak], bool) if ak in z.files else np.zeros(len(pts), bool)
        Ssurf = RUN.surf(r['surface']); fin = np.isfinite(pts).all(axis=1) if len(pts) else np.zeros(0, bool); acc = fin & ~amb
        if acc.sum() == 0:
            units.append(dict(section=sec, method=method, case_key=r['case_key'], status='no_output', stored_status=r['status'], stored_ctau_v3=r.get('Ctau15_d6_v3'), consistent=bool(r['status'] == 'no_output' and float(r.get('Ctau15_d6_v3') or 0.0) == 0.0), npz=S.rel(npz)))
            continue
        E = RUN.evaluate(pts, acc, Ssurf); d = E['r3']['sub']['dent']['by_d']['6.0']; a = E.get('amplitude') or {}
        new = dict(Ctau15_d6=d['C_tau']['0.15'], Cvalid_d6=d['C_valid'], amp_rel_pct=a.get('rel_err_pct'), rmse_mm=E['pts_ref_rmse_mm'], gross_rate=E['pts_ref_gross_rate'], n_accepted=E['n_accepted'])
        old = dict(Ctau15_d6=r['Ctau15_d6'], Cvalid_d6=r['Cvalid_d6'], amp_rel_pct=r['amp_rel_pct'], rmse_mm=r['rmse_mm'], gross_rate=r['gross_rate'], n_accepted=r['n_accepted'])
        diff = {k: (None if (old[k] is None or new[k] is None or (isinstance(old[k], float) and math.isnan(old[k]))) else abs(float(old[k]) - float(new[k]))) for k in new}
        units.append(dict(section=sec, method=method, regime=r['regime_type'], sigma=r['sigma'], case_key=r['case_key'], status='ok', stored=old, recomputed=new, abs_diff=diff, consistent=bool(all((v is None) or v <= 1e-9 for v in diff.values())), npz=S.rel(npz), eval_version_stored=r['eval_version']))
    for rr in [x for x in reg_rows if x.get('status') == 'ok'][:4]:
        e = rr['evaluation']; units.append(dict(section='lines', case_id=rr['case_id'], reason=rr['reason'], status='ok', stored=e['stored'], recomputed=e['old'], consistent=bool(e['old_vs_stored_equal'])))
    return dict(n_units=len(units), all_consistent=bool(all(u.get('consistent', False) for u in units if u.get('status') in ('ok', 'no_output'))), units=units)


def interface_contract(reg: dict, rq: dict, an: dict, ev: dict) -> dict:
    cfg_dot = dict(RD.CFG); z0, z1 = M4.search_window(cfg_dot, float(cfg_dot['measurement_volume_mm']))
    p4 = p4_params()
    return dict(
        task=S.TASK_ID, generated_local=S.now_local(),
        dots=dict(front_end='r5_detect.candidates (Otsu; area 4-5000 px2; border 3 px; mean intensity >= 10) -> r5_detect.select(group=D0: relative area window 0.5-1.5 x area-weighted median when >= 8 candidates, elongation <= 2.5)',
                  correspondence='r5_detect.reconstruct_from -> m4_reconstruction._correspond_epipolar (nearest track, tol 3 px, ambiguity tol 3 px, t margin 0.02, one-to-one) -> M4.triangulate',
                  detection_version='D0 (sections 1 final, 2, 3 plain arms); D1/D2 not used', search_window_mm=[z0, z1], search_mode=cfg_dot.get('search_mode', M4.SEARCH_MODE), effective_cfg={k: cfg_dot[k] for k in ('measurement_volume_mm', 'epi_tol_px', 'epi_ambig_tol_px', 'epi_t_margin', 'epi_one_to_one', 'search_mode') if k in cfg_dot},
                  controlling_keys=dict(measurement_volume_mm='window width', search_z_min_mm='window start (absent -> 0.0)', depth_margin_frac='track overshoot only (EPI_T_MARGIN)'),
                  generator='sp_dots_columns.layout (m2_pattern._lattice_centres + column displacement; edge rule sn_lib.edge_margin_px = r_max 1.5 mm + 2 px); raster m2_pattern._draw_dots',
                  identity='sl_lib.reference_map / classify / output_identity: source labels from the sigma-0 image of the same configuration; output points correct / wrong / undeterminable_(partial|multi|unlabelled|background)',
                  denominators=dict(expected='expected dots in the target disc (sn_lib.expected_dots)', detected='candidates after D0', accepted='matched non-ambiguous output points', identity='per output point'),
                  family_specific=['sl_lib.task_criterion T (gross <= 1 %, recovery >= 0.9, |amp| <= 10 %, RMSE <= 0.06 / 0.15 mm)', 'one-to-one recovery (sn_posthoc.recovery)']),
        lines=dict(path='m4_lines.run_olines with P4_C3_cal (frozen); table adapter sp_lines_table.run_p4_table reproduces it array by array (regression)', p4_params=S.clean(p4), p4_json_sha256=S.sha_file(P4_JSON),
                   search_window_mm=list(S.LINE_WINDOW_MM), window_cfg=B.window_cfg(), controlling_keys=dict(search_z_min_mm='z0', measurement_volume_mm='z1 - z0', search_margin_mm='explicit acceptance tolerance (priority over depth_margin_frac)'),
                   generator='sp_lines_table.table_spec (M2 olines raster rule with nearest-table line)', identity='sp_lines_table.truth_identity_table: nearest table line, band w/2 + 1 px, explicit second-line test (multi_candidate -> unconfirmable)',
                   denominators=dict(reference_rows='l01_truth.line_reference rows (arc-length weighted)', detected='steger features', accepted='status 0', identity='per accepted feature: confirmable correct / wrong / unconfirmable (+ multi_candidate)'),
                   family_specific=['l01_truth.line_metrics', 'C_valid', 'tails p95 / p99']),
        shared_core=dict(function='l01rb_run.evaluate', surface_eval='surface_eval_r3_dev (support exact, v2)', d_list=list(B.D_LIST), tau_list=list(B.TAU_LIST), main=dict(d_mm=6.0, tau_mm=0.15), omega='l01_common.omega (nodes visible in both images), masks dent (r <= 15 mm) / fov_legacy',
                         amplitude='m5_evaluation.compute_indentation_recovery (interpolated peak, FOV window) -> rel_err_pct, pass |rel| <= 10 %', point_metrics='surface_eval_r3.point_metrics: RMSE / p95 / bias, gross |dz| > 1 mm',
                         eval_key=B.eval_key(), no_output_rule='C_valid = C_tau = 0; RMSE / amplitude / D95 undefined (NIGHT-01R v3)', program_error_rule='separate ledger entry, not in the scientific denominator',
                         units=dict(C='area fraction of the mask', amplitude='% of true dent depth 5 mm', rmse='mm at output points', gross='fraction of accepted points')),
        surfaces={s: B.surface_record(s) for s in ('baseline', 'curv_x_strong')}, budget=dict(Q=S.Q_BUDGET, rule='peak = Q / binary raster integral, cap 1.0, per raster', source='outputs/spacing_layout/design/budget.json'),
        src_versions=B.src_versions(), adapter_provenance=LT.PROVENANCE, regular_equivalence=dict(passed=reg['passed'], n_checks=reg['n_checks']), regression=dict(n_run=rq['n_run'], n_decodes=rq['n_decodes'], all_equal=rq['all_equal']),
        analytic_tests=dict(passed=an['passed'], n=len(an['tests'])), evaluation_binding=dict(n_units=ev['n_units'], all_consistent=ev['all_consistent']))


def main():
    S.ensure_dirs(); t0 = time.time()
    cases = select_regression_cases()
    S.save_json(S.OUT / 'regression_cases_preregistered.json', dict(generated_local=S.now_local(), rule='protocol §4', n=len(cases), cases=cases))
    S.register_inputs(extra_files=[S.ROOT / c['json'] for c in cases if c.get('json')] + [S.ROOT / c['npz'] for c in cases if c.get('npz')], name='inputs_start_with_cases.json')
    reg = regular_equivalence(); S.log(f"regular equivalence: passed={reg['passed']} ({reg['n_checks']} checks)", 'regression')
    raw = S.OUT / 'regression_run_raw.json'
    if raw.exists():
        rq = S.load_json(raw); S.log(f"regression: reusing stored run ({rq['n_decodes']} decodes, all_equal={rq['all_equal']})", 'regression')
    else:
        rq = regression_run(cases); S.save_json(raw, rq)
    out = dict(task=S.TASK_ID, generated_local=S.now_local(), new_research_images=0, decodes_total=rq['n_decodes'], regular_equivalence=reg, regression=rq, seconds=None)
    try:
        an = analytic_tests(); S.log(f"analytic tests: passed={an['passed']} " + ', '.join(f"{t['test']}={t['passed']}" for t in an['tests']), 'regression')
    except Exception:
        import traceback
        an = dict(passed=False, tests=[], error=traceback.format_exc()); S.log('analytic tests: PROGRAM ERROR (recorded)', 'regression')
    try:
        ev = eval_binding_check(rq['rows']); S.log(f"evaluation binding: {ev['n_units']} units, all_consistent={ev['all_consistent']}", 'regression')
    except Exception:
        import traceback
        ev = dict(n_units=0, all_consistent=False, units=[], error=traceback.format_exc()); S.log('evaluation binding: PROGRAM ERROR (recorded)', 'regression')
    out.update(analytic_tests=an, evaluation_binding=ev, seconds=round(time.time() - t0, 1))
    S.save_json(S.OUT / 'regression_and_selfcheck.json', out)
    S.save_json(S.OUT / 'interface_eval_contract.json', interface_contract(reg, rq, an, ev))
    print(json.dumps(dict(regular=reg['passed'], regression_all_equal=rq['all_equal'], n_decodes=rq['n_decodes'], analytic=an['passed'], eval_binding=ev['all_consistent'], seconds=out['seconds'])))


if __name__ == '__main__':
    main()
