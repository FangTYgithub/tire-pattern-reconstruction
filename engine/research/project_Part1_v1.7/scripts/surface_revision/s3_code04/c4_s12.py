# Re-evaluate size-spacing cases with alternative point front ends.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

import c4_common as K

C1DIR = K.CACHE / 'C1'
REFDIR = K.CACHE / 'refmaps'
ICODE = {k: i for i, k in enumerate(('correct', 'wrong', 'background', 'undeterminable_partial', 'undeterminable_multi', 'undeterminable_unlabelled', 'none'))}
UNCONF = ('background', 'undeterminable_partial', 'undeterminable_multi', 'undeterminable_unlabelled', 'none')
KEEP = ('state', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'glob_n_nodes', 'loc_n_nodes', 'glob_Cvalid', 'loc_Cvalid', 'glob_Ctau', 'loc_Ctau', 'glob_pt_rmse', 'glob_pt_p95',
        'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'glob_surf_p95', 'loc_surf_rmse', 'loc_surf_p95', 'glob_recovery', 'loc_recovery', 'id_correct', 'id_wrong', 'id_unconf', 'glob_gross_rate')


def _u(row: dict) -> dict:
    u = dict(row)
    for k in ('size_mm', 'peak_target', 'sigma'):
        u[k] = float(u[k])
    return u


def _ref_map(ref_key: str, ref_img, exp_uv, led) -> tuple:
    import sl_lib as L
    p = REFDIR / f'{ref_key}.npz'
    if p.exists():
        z = K.load_npz(p); return z['Sm'], 'cached'
    Sm, info = L.reference_map(np.asarray(ref_img, np.uint8), np.asarray(exp_uv, float)); K.save_npz(p, Sm=np.asarray(Sm))
    led('reference_extraction', 'ok', ref=ref_key)
    return np.asarray(Sm), 'new'


def _sm(cd: dict, Sm) -> dict:
    import sl_lib as L
    cl = L.classify(cd['label_map'], cd['n_labels'], Sm); lab = np.array([c[5] for c in cd['cands']], np.int64)
    return dict(cl=cl, cand_label=lab, cand_src=(cl['cls'][lab].astype(np.int8) if len(lab) else np.zeros(0, np.int8)), cand_dot=(cl['dot'][lab].astype(np.int64) if len(lab) else np.zeros(0, np.int64)))


def _regress(D: dict, Z: dict) -> dict:

    cands = np.array(D['cd']['cands'], np.float64).reshape(-1, 7) if len(D['cd']['cands']) else np.zeros((0, 7))
    out = dict(n_cand_new=len(cands), n_cand_stored=int(len(Z['cand_uv'])))
    same_n = len(cands) == len(Z['cand_uv'])
    out['cand_equal'] = bool(same_n and np.array_equal(cands[:, :2].astype(Z['cand_uv'].dtype), Z['cand_uv']) and np.array_equal(cands[:, 2].astype(Z['cand_area'].dtype), Z['cand_area'])
                             and np.array_equal(cands[:, 3].astype(Z['cand_elong'].dtype), Z['cand_elong']) and np.array_equal(cands[:, 5].astype(Z['cand_label'].dtype), Z['cand_label']))
    out['select_equal'] = bool(same_n and np.array_equal(np.asarray(D['code']).astype(Z['cand_code'].dtype), Z['cand_code']) and np.array_equal(np.asarray(D['fidx']).astype(Z['cand_fidx'].dtype), Z['cand_fidx']))
    f = np.asarray(D['feats'], np.float32).reshape(-1, 2); out['feats_equal'] = bool(f.shape == Z['feats'].shape and np.array_equal(f, Z['feats']))
    pts, info, tr = D['arms']['single']; p = np.asarray(pts, np.float64).reshape(-1, 3); ps = np.asarray(Z['pts'], np.float64)
    out['pts_equal'] = bool(p.shape == ps.shape and np.array_equal(np.nan_to_num(p, nan=1e30), np.nan_to_num(ps, nan=1e30)))
    out['pts_max_abs_diff'] = float(np.nanmax(np.abs(p - ps))) if p.shape == ps.shape and p.size else (0.0 if p.shape == ps.shape else None)
    out['trace_equal'] = bool(all(np.array_equal(np.asarray(tr[k]).astype(Z[f'feat_{k}'].dtype), Z[f'feat_{k}']) for k in ('matched', 'ambiguous', 'duplicate', 'best_idx', 'point_idx')))
    out['regression_ok'] = bool(out['cand_equal'] and out['select_equal'] and out['feats_equal'] and out['pts_equal'] and out['trace_equal'])
    return out


def _counts(ident, acc) -> dict:
    a = np.asarray(acc, bool); i = np.asarray(ident, dtype=object)
    return dict(src_acc=int(a.sum()), src_correct=int((a & (i == 'correct')).sum()), src_wrong=int((a & (i == 'wrong')).sum()), src_unconf=int((a & np.isin(i, UNCONF)).sum()))


# Reprocess one stored size-spacing image with the selected dot front end.
def work_dot(row: dict, led_path: str) -> dict:
    import c2_identity as ID
    import c2_decode as CD
    import rdy_platform as P
    import s12_case as SC12
    import s3c_dev as DV
    import sl_lib as L
    import sn_lib as SL
    t0 = time.monotonic(); u = _u(row); ik = u['input_key']; led_file = Path(led_path)

    def led(kind, status, **kw):
        K.ledger(led_file, kind, ik, status, **kw)
    out_p = C1DIR / f'{ik}.npz'
    if out_p.exists() and (C1DIR / f'{ik}.json').exists():
        return K.load_json(C1DIR / f'{ik}.json')
    R = SC12.raster_of(u); binary = np.asarray(R['binary'], np.float32); ctx = SC12.context(u, R, u['peak_target'])
    sha = K.load_json(K.RESULTS / 'scope_sha_index.json')
    for f in (u['image_record'].replace('.json', '.npz'), u['eval_record'], u['eval_record'].replace('.json', '.npz')):
        if K.sha_file(K.ROOT / f) != sha[f]:
            raise K.Stop('technical', f'{ik}: input file changed since registration: {f}')
    img = K.load_npz(K.ROOT / u['image_record'].replace('.json', '.npz'))['image']
    ref_row = u['ref_record']; ref_img = K.load_npz(K.ROOT / ref_row.replace('.json', '.npz'))['image']
    Z = K.load_npz(K.ROOT / u['eval_record'].replace('.json', '.npz')); J = K.load_json(K.ROOT / u['eval_record'])
    mesh = P.mesh(u['surface']); sid = K.RK.SURF_MAP[u['surface']]; centres = np.asarray(R['centres'], np.float64)
    exp = SL.expected_dots(mesh, centres); Sm, ref_state = _ref_map(u['ref_input_key'], ref_img, exp['uv'], led)
    A = {}; rows = []; base = {k: u[k] for k in ('input_key', 'part', 'section', 'regime', 'source', 'surface', 'layout', 'size_mm', 'phase_label', 'sigma', 'rep')}
    base['parts'] = u.get('parts', u['part'])

    led('decode_attempt', 'start', rule='B0')
    D0 = DV.dot_decode(img, binary, 1, [u['size_mm']], 'B0', 'binding'); led('decode_attempt', 'ok', rule='B0')
    reg = _regress(D0, Z)
    if not reg['regression_ok']:
        K.save_json(C1DIR / f'{ik}.json', dict(input_key=ik, status='technical_regression_failed', regression=reg, rows=[], attempts=1, seconds=round(time.monotonic() - t0, 2)))
        raise K.Stop('technical', f'{ik}: original front end not reproduced {reg}')
    sm0 = _sm(D0['cd'], Sm)
    z = dict(pts=np.asarray(Z['pts'], np.float32), cand_label=Z['cand_label'], cand_fidx=Z['cand_fidx'], feat_matched=Z['feat_matched'].astype(bool), feat_best_idx=Z['feat_best_idx'],
             feat_point_idx=Z['feat_point_idx'], proj_dot=Z['proj_dot'])
    ident0 = np.asarray(L.output_identity(z, sm0['cl']), dtype=object); acc0 = np.asarray(Z['accepted'], bool)
    m0 = J['metrics']

    mE, _ = DV.evaluate(sid, 'point', CD.dot_dec(D0, 'single'), ctx, mesh)
    diff = {k: (m0.get(k), mE.get(k)) for k in KEEP if not ((m0.get(k) is None and mE.get(k) is None) or (isinstance(m0.get(k), float) and isinstance(mE.get(k), float) and abs(m0[k] - mE[k]) <= 1e-9) or m0.get(k) == mE.get(k))}
    rows.append(dict(base, rule='orig_D0', arm='stored', **{k: m0.get(k) for k in KEEP}, **_counts(ident0, acc0), n_candidates=len(D0['cd']['cands']), n_floor=0,
                     n_bg_candidates=int((sm0['cand_src'] == 0).sum()), eval_consistency_diffs=json.dumps(K.clean(diff)), **{f'reg_{k}': v for k, v in reg.items()}))
    A['orig_ident'] = np.array([ICODE.get(str(x), 6) for x in ident0], np.int8); A['orig_cand_src'] = sm0['cand_src']

    for rule in ('DF', 'DS'):
        led('decode_attempt', 'start', rule=rule)
        D = DV.dot_decode(img, binary, 1, [u['size_mm']], rule, 'binding'); led('decode_attempt', 'ok', rule=rule)
        sm = _sm(D['cd'], Sm); m, arr = DV.evaluate(sid, 'point', CD.dot_dec(D, 'single'), ctx, mesh)
        pi = ID.dot_point_identity(sm, D, 'single', binary, centres); acc = np.asarray(arr['accepted'], bool)
        rows.append(dict(base, rule=rule, arm='single', **{k: m.get(k) for k in KEEP}, **_counts(pi['ident'], acc), n_candidates=len(D['cd']['cands']), n_floor=D['cd'].get('n_floor', 0),
                         a_floor=D['cd'].get('a_floor'), n_bg_candidates=int((sm['cand_src'] == 0).sum())))
        A[f'{rule}_pts'] = np.asarray(arr['pts'], np.float32); A[f'{rule}_acc'] = acc; A[f'{rule}_zr'] = np.asarray(arr['zr'], np.float32)
        A[f'{rule}_ident'] = np.array([ICODE.get(str(x), 6) for x in pi['ident']], np.int8); A[f'{rule}_cand_src'] = sm['cand_src']
    h = K.save_npz(out_p, **A)
    rec = dict(input_key=ik, status='ok', ref_map=ref_state, regression=reg, eval_consistency_ok=not diff, rows=rows, npz_sha256=h, attempts=3, seconds=round(time.monotonic() - t0, 2),
               image_record=u['image_record'], note='compact record: DF / DS point clouds + identity codes; original point cloud = stored record (referenced, not copied)')
    K.save_json(C1DIR / f'{ik}.json', rec)
    return rec


def _worker(args):
    row, = args
    led = K.RUN / 'C' / f'ledger_w{os.getpid()}.jsonl'
    return work_dot(row, str(led))


def plan_dots() -> pd.DataFrame:
    S = pd.read_csv(K.RESULTS / 'scope_C1_dots.csv', low_memory=False)
    rec = S.drop_duplicates('input_key').set_index('input_key').image_record
    S['ref_record'] = S.ref_input_key.map(rec)
    parts = S.groupby('input_key').part.apply(lambda x: ';'.join(sorted(set(x))))
    U = S.drop_duplicates('input_key').copy(); U['parts'] = U.input_key.map(parts)
    U['part'] = np.where(U.parts.str.contains('S1_cap'), 'S1_cap', U.part)
    return U


def run_dots(workers: int = 14, limit=None) -> dict:

    U = plan_dots(); ce_ref = set(U[(U.part == 'counterexample_review') & (U.sigma == 0)].input_key)
    todo = U[~U.input_key.isin(ce_ref)]
    if limit:
        todo = todo.head(limit)
    sha = pd.read_csv(K.RESULTS / 'scope_inputs_sha.csv'); K.save_json(K.RESULTS / 'scope_sha_index.json', dict(zip(sha.file, sha.sha256)))
    done = {p.stem for p in C1DIR.glob('*.json') if K.load_json(p).get('status') == 'ok'} if C1DIR.exists() else set()
    need = todo[~todo.input_key.isin(done)]
    used = K.decode_attempts('C')
    if used + 3 * len(need) > K.CAPS['decode_C']:
        raise K.Stop('budget', f'C attempts {used} + {3 * len(need)} planned > {K.CAPS["decode_C"]}')
    rows = need.to_dict('records'); fails, ok = [], 0; it = iter(rows); inflight = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        def submit_next():
            r = next(it, None)
            if r is not None:
                inflight[ex.submit(_worker, (r,))] = r['input_key']
            return r is not None
        for _ in range(workers * 2):
            if not submit_next():
                break
        while inflight:
            dn, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for f in dn:
                key = inflight.pop(f)
                try:
                    f.result(); ok += 1
                except Exception as e:
                    fails.append(dict(key=key, error=traceback.format_exception_only(type(e), e)[-1][:300]))
            if fails:
                for f in list(inflight):
                    if f.cancel():
                        inflight.pop(f)
                continue
            while len(inflight) < workers * 2 and submit_next():
                pass
    not_started = [r['input_key'] for r in rows if not (C1DIR / f"{r['input_key']}.json").exists()]
    st = dict(planned=len(rows), ok=ok, failures=fails, not_started=len(not_started), attempts_total=K.decode_attempts('C'), t_local=K.now_local())
    K.save_json(K.RUN / 'C' / 'state_dots.json', dict(st, not_started_keys=not_started))
    if fails:
        raise K.Stop('technical', f'{len(fails)} technical failures; stopped submitting ({len(not_started)} not started): {fails[:2]}')
    return st


def work_line(row: dict) -> dict:
    import rdy_platform as P
    import s12_case as SC12
    import s1r_eval as E
    u = _u(row); J = K.load_json(K.ROOT / u['eval_record']); Z = K.load_npz(K.ROOT / u['eval_record'].replace('.json', '.npz')); m0 = J['metrics']
    mesh = P.mesh(u['surface'])

    from scipy.interpolate import RegularGridInterpolator
    X, Y, Zg = mesh['X'], mesh['Y'], mesh['Z']; xs, ys = X[0, :] if np.allclose(X[0, :], X[-1, :]) else X[:, 0], Y[:, 0] if np.allclose(Y[:, 0], Y[:, -1]) else Y[0, :]
    if np.allclose(X[0, :], X[-1, :]):
        f = RegularGridInterpolator((Y[:, 0], X[0, :]), Zg, bounds_error=False, fill_value=np.nan); q = lambda p: f(np.column_stack([p[:, 1], p[:, 0]]))
    else:
        f = RegularGridInterpolator((X[:, 0], Y[0, :]), Zg, bounds_error=False, fill_value=np.nan); q = lambda p: f(np.column_stack([p[:, 0], p[:, 1]]))
    pts = np.asarray(Z['pts'], np.float64); acc = np.asarray(Z['accepted'], bool); valid = np.asarray(Z['valid'], bool) if 'valid' in Z else acc
    ing, inl = np.asarray(Z['in_global'], bool), np.asarray(Z['in_local'], bool)
    out = dict(input_key=u['input_key'], part=u['part'], surface=u['surface'], layout=u['layout'], size_mm=u['size_mm'], sigma=u['sigma'], source=u['source'], state=m0.get('state'))
    zt = q(pts) if len(pts) else np.zeros(0); zr = np.asarray(Z['zr'], np.float64)
    out['zr_vs_independent_max_abs'] = float(np.nanmax(np.abs(zt[valid] - zr[valid]))) if valid.any() else None
    for dom, mk in (('glob', ing), ('loc', inl)):
        s = valid & mk; e = pts[s, 2] - zt[s] if s.any() else np.zeros(0)
        rm = float(np.sqrt(np.mean(e ** 2))) if len(e) else None; p95 = float(np.percentile(np.abs(e), 95)) if len(e) else None
        out[f'{dom}_rmse_indep'] = rm; out[f'{dom}_rmse_stored'] = m0.get(f'{dom}_pt_rmse'); out[f'{dom}_p95_indep'] = p95; out[f'{dom}_p95_stored'] = m0.get(f'{dom}_pt_p95')
        out[f'{dom}_n_valid_indep'] = int(s.sum()); out[f'{dom}_n_valid_stored'] = m0.get(f'{dom}_n_valid')
        out[f'{dom}_rmse_absdiff'] = (abs(rm - m0[f'{dom}_pt_rmse']) if (rm is not None and m0.get(f'{dom}_pt_rmse') is not None) else (0.0 if rm is None and m0.get(f'{dom}_pt_rmse') is None else np.inf))
        out[f'{dom}_p95_absdiff'] = (abs(p95 - m0[f'{dom}_pt_p95']) if (p95 is not None and m0.get(f'{dom}_pt_p95') is not None) else (0.0 if p95 is None and m0.get(f'{dom}_pt_p95') is None else np.inf))

    pc, pw, pu = (np.asarray(Z[k], bool) for k in ('point_correct', 'point_wrong', 'point_unconf'))
    out.update(id_correct_mask=int((acc & pc).sum()), id_correct_stored=m0.get('id_correct'), id_wrong_mask=int((acc & pw).sum()), id_wrong_stored=m0.get('id_wrong'),
               id_unconf_mask=int((acc & pu).sum()), id_unconf_stored=m0.get('id_unconf'), n_accepted_mask=int(acc.sum()), n_accepted_stored=m0.get('n_accepted'),
               no_output=bool(acc.sum() == 0), Ctau_stored=m0.get('glob_Ctau'), Cvalid_stored=m0.get('glob_Cvalid'))
    out['counts_equal'] = bool(out['id_correct_mask'] == out['id_correct_stored'] and out['id_wrong_mask'] == out['id_wrong_stored'] and out['id_unconf_mask'] == out['id_unconf_stored']
                               and out['n_accepted_mask'] == out['n_accepted_stored'])
    out['rmse_ok'] = bool(out['glob_rmse_absdiff'] <= 1e-6 and out['loc_rmse_absdiff'] <= 1e-6)
    out['no_output_ok'] = bool((not out['no_output']) or (m0.get('glob_pt_rmse') is None and (m0.get('glob_Ctau') or 0) == 0))
    return out


def run_lines() -> dict:
    S = pd.read_csv(K.RESULTS / 'scope_C2_lines.csv', low_memory=False)
    rows = [work_line(r) for r in S.to_dict('records')]
    D = pd.DataFrame(rows); K.save_csv(D, K.RESULTS / 'C2_lines_consistency.csv')

    s12 = K.load_json(K.ROOT / 'outputs/surface_revision/s12_resource01/protocol.json'); dep = s12.get('dependencies') or {}
    files = {}
    for k in ('release', 'platform_dependency_manifest', 's1_dependencies', 'binding', 'global_domain'):
        if isinstance(dep.get(k), dict) and dep[k].get('file'):
            files[dep[k]['file']] = dep[k]['sha256']
    files.update({f'scripts/surface_revision/s12_resource01/{n}': h for n, h in (dep.get('run_code') or {}).items()})
    files.update(dep.get('reused_modules') or {})
    cmp = {f: (h, K.sha_file(K.ROOT / f) if (K.ROOT / f).exists() else None) for f, h in files.items() if h}
    changed = sorted(f for f, (a, b) in cmp.items() if a != b)
    import rdy_platform as P
    drift = P.release_state().get('drift')
    s = dict(n=len(D), counts_equal=int(D.counts_equal.sum()), rmse_ok=int(D.rmse_ok.sum()), no_output_cases=int(D.no_output.sum()), no_output_ok=int(D.no_output_ok.sum()),
             max_rmse_absdiff=float(np.nanmax(np.r_[D.glob_rmse_absdiff.replace(np.inf, np.nan), D.loc_rmse_absdiff.replace(np.inf, np.nan)])),
             max_p95_absdiff=float(np.nanmax(np.r_[D.glob_p95_absdiff.replace(np.inf, np.nan), D.loc_p95_absdiff.replace(np.inf, np.nan)])),
             zr_vs_independent_max_abs=float(D.zr_vs_independent_max_abs.max()), s12_dependency_files_compared=len(cmp), s12_dependency_files_changed=changed, surf02_drift=drift,
             ok=bool(D.counts_equal.all() and D.rmse_ok.all() and D.no_output_ok.all() and not changed and not drift))
    return s


PAIR_METRICS = ('glob_Ctau', 'glob_Cvalid', 'loc_Ctau', 'glob_pt_rmse', 'glob_pt_p95', 'src_correct', 'src_wrong', 'src_unconf', 'src_acc')


def aggregate() -> dict:
    recs = [K.load_json(p) for p in sorted(C1DIR.glob('*.json'))]
    bad = [r['input_key'] for r in recs if r.get('status') != 'ok']
    rows = [x for r in recs if r.get('status') == 'ok' for x in r['rows']]
    D = pd.DataFrame(rows)
    D['glob_gap'] = pd.to_numeric(D.glob_Cvalid, errors='coerce') - pd.to_numeric(D.glob_Ctau, errors='coerce')
    D['no_output'] = D.src_acc == 0

    V = []
    for _, r in D.iterrows():
        for part in str(r['parts']).split(';'):
            v = r.to_dict(); v['view'] = part; v['regime_view'] = {'S1_cap': 'resource_cap_11695', 'S2_cap': 'resource_cap_11695', 'S1_fixed_peak': 'fixed_peak_1'}.get(part, r['regime'])
            V.append(v)
    V = pd.DataFrame(V); K.save_csv(D, K.RESULTS / 'C1_dots_cases.csv'); K.save_csv(V, K.RESULTS / 'C1_dots_views.csv')

    P = V.pivot_table(index=['input_key', 'view', 'section', 'surface', 'layout', 'size_mm', 'phase_label', 'sigma', 'rep'], columns='rule', values=list(PAIR_METRICS) + ['glob_gap'], aggfunc='first')
    out = []
    for idx, r in P.iterrows():
        d = dict(zip(P.index.names, idx))
        for a, b in (('DF', 'orig_D0'), ('DS', 'orig_D0'), ('DS', 'DF')):
            e = dict(d, contrast=f'{a}-{b}')
            for m in list(PAIR_METRICS) + ['glob_gap']:
                x, y = r.get((m, a)), r.get((m, b))
                e[f'd_{m}'] = (float(x) - float(y)) if (x is not None and y is not None and pd.notna(x) and pd.notna(y)) else np.nan
            e['rmse_undefined_a'] = bool(pd.isna(r.get(('glob_pt_rmse', a)))); e['rmse_undefined_b'] = bool(pd.isna(r.get(('glob_pt_rmse', b))))
            out.append(e)
    Pd = pd.DataFrame(out); K.save_csv(Pd, K.RESULTS / 'C1_dots_paired.csv')

    cells = []
    for k, g in V.groupby(['view', 'surface', 'layout', 'size_mm', 'sigma', 'rule']):
        ph = g.groupby('phase_label')
        c = dict(zip(['view', 'surface', 'layout', 'size_mm', 'sigma', 'rule'], k), n_images=len(g), n_phases=int(g.phase_label.nunique()))
        for m in ('glob_Ctau', 'glob_Cvalid', 'glob_gap', 'glob_pt_rmse', 'glob_pt_p95'):
            pv = ph[m].apply(lambda s: pd.to_numeric(s, errors='coerce').mean())
            c[f'{m}_phase'] = ';'.join('nan' if pd.isna(x) else f'{x:.5f}' for x in pv.values); c[f'{m}_mean'] = float(pv.mean()) if pv.notna().any() else np.nan
            c[f'{m}_min'] = float(pv.min()) if pv.notna().any() else np.nan; c[f'{m}_max'] = float(pv.max()) if pv.notna().any() else np.nan
        c.update(c_=int(g.src_correct.sum()), w=int(g.src_wrong.sum()), u=int(g.src_unconf.sum()), n_no_output=int(g.no_output.sum()), n_unconf=int((g.src_unconf > 0).sum()),
                 n_wrong=int((g.src_wrong > 0).sum()), bg=float(pd.to_numeric(g.n_bg_candidates, errors='coerce').mean()))
        cells.append(c)
    Cc = pd.DataFrame(cells); K.save_csv(Cc, K.RESULTS / 'C1_dots_cells.csv')
    reg = pd.DataFrame([dict(input_key=r['input_key'], **r['regression'], eval_consistency_ok=r['eval_consistency_ok']) for r in recs if r.get('status') == 'ok'])
    K.save_csv(reg, K.RESULTS / 'C1_regression.csv')
    return dict(records=len(recs), not_ok=bad, rows=len(D), views=len(V), paired=len(Pd), cells=len(Cc), regression_ok=int(reg.regression_ok.sum()), eval_consistency_ok=int(reg.eval_consistency_ok.sum()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--limit', type=int); ap.add_argument('--workers', type=int, default=14); a = ap.parse_args()
    try:
        if a.cmd == 'dots':
            print(json.dumps(K.clean(run_dots(a.workers, a.limit)), indent=1)[:3000])
        elif a.cmd == 'lines':
            print(json.dumps(K.clean(run_lines()), indent=1))
        elif a.cmd == 'aggregate':
            print(json.dumps(K.clean(aggregate()), indent=1))
    except K.Stop as e:
        print(e, file=sys.stderr); sys.exit(K.EXIT.get(e.code, 2))
