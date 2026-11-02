# Exercise new-phase scheduling, pairing and result-integrity checks.

from __future__ import annotations

import copy
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import tr5_chain as CH
import tr5_common as K
import tr5_manifest as MF

DRILLS = K.OUT / '_drills'
BASE = {'r0.5/R8x4': 0.30, 'r1.0/R8x4': 0.50, 'r1.0/T8x4': 0.55, 'r1.0/R8x6': 0.60, 's8': 0.40, 's10': 0.45}
SPEC = dict(module='tr5_drills', cls='Stub')


def spec(**mode):
    return dict(SPEC, args=dict(mode=mode))


# Provide deterministic test outputs and injected failures for scheduler checks.
class Stub:
    name = 'stub'

    def __init__(self, mode=None):
        self.mode = dict(mode or {}); self.calls = dict(render=0, decode=0, evaluation=0)

    def render(self, row, ctx):
        self.calls['render'] += 1
        if self.mode.get('render_bad') == row['forward_key']:
            return np.zeros((512, 512), np.float32), {}
        img = ((np.asarray(ctx['pattern']) > 0) * (60 + int(row['seed']) % 50)).astype(np.uint8)
        return img, dict(z_alignment='absolute', n_hits=int((img > 0).sum()), noise_sigma=float(row['sigma']), mottle_seed=int(row['seed']), rays_per_pixel=1)

    def decode(self, method, img, row, ctx):
        self.calls['decode'] += 1; k = self.calls['decode']; fk = row['forward_key']
        if self.mode.get('interrupt_at') == k:
            raise KeyboardInterrupt
        if self.mode.get('error_at') == k:
            raise RuntimeError('stub decode error')
        if self.mode.get('none_key') in (fk, f'{fk}|{method}'):
            return None
        if row['family'] == 'point':
            n = 24; i = np.arange(n)
            feats = np.column_stack([100 + 12.0 * i, 200 + 9.0 * (i % 4)]).astype(np.float32)
            pts = np.column_stack([-20 + 1.7 * i, (i % 5) * 4.0 - 8, 0.1 + 0.01 * (i % 3)]).astype(np.float64)
            cd = dict(cands=[(float(u), float(v), 30, 1.1, 0.0, int(j + 1), 0.0) for j, (u, v) in enumerate(feats)], reason=np.zeros(n, np.int8), n_labels=n + 1,
                      stats=np.zeros((n + 1, 5), np.int32), thresh=50.0, n_filtered=0, n_border=0)
            tr = dict(matched=np.ones(n, bool), ambiguous=np.zeros(n, bool), duplicate=np.zeros(n, bool), best_idx=i.astype(np.int64), point_idx=i.astype(np.int64))
            dec = dict(pts=pts, accepted=np.ones(n, bool), aux=dict(feats=feats, tr=tr, cd=cd, sel_code=np.zeros(n, np.int8), sel_fidx=i.astype(np.int32), det_info={}, info={}))
        else:
            n = 30; i = np.arange(n)
            feats = np.column_stack([150 + 30.0 * (i % 6), 100 + (i // 6) * 40.0]).astype(np.float64)
            pts = np.column_stack([-20 + 1.3 * i, (i % 6) * 5.0 - 12, 0.2 + 0.01 * (i % 4)]).astype(np.float64)
            r = dict(features=feats, line_id=(i % 6).astype(np.int64), status=np.zeros(n, np.int64), chain=np.zeros(n, np.int64), segment_id=np.zeros(n, np.int64),
                     n_candidates=np.ones(n, np.int64), guard_reason=np.zeros(n, np.int64), detect_info={}, counts={}, checks={}, window_z0_mm=-16.0, window_z1_mm=4.0)
            dec = dict(pts=pts, accepted=np.ones(n, bool), aux=dict(r=r))
        if self.mode.get('missing_key') == fk:
            dec.pop('accepted')
        return dec

    def evaluate(self, family, dec, row, ctx):
        self.calls['evaluation'] += 1
        fk = row['forward_key']; pts = np.asarray(dec['pts'], np.float64); n = len(pts); acc = np.asarray(dec['accepted'], bool); i = np.arange(n)
        empty = self.mode.get('empty_key') == fk
        val = acc & (not empty); ig = val.copy(); il = val & (i < 5)
        err = 0.01 * ((i % 3) - 1); zr = (pts[:, 2] - err).astype(np.float32); e = pts[:, 2] - zr.astype(np.float64)
        A = dict(pts=pts, accepted=acc, valid=val, zr=zr, in_global=ig, in_local=il)
        if family == 'point':
            feats = np.asarray(dec['aux']['feats'])
            A.update(feats=feats, point_feature=i.astype(np.int32), feat_truth_dot=i.astype(np.int32), feat_confirmable=np.ones(n, bool), point_correct=acc.copy(),
                     point_wrong=np.zeros(n, bool), point_unconf=np.zeros(n, bool))
        else:
            import tr_a4 as A4
            feats = np.asarray(dec['aux']['r']['features'], np.float64)
            G = A4._geom(A4._mesh(row['surface']), feats, ctx['meta_full']['olines'])
            conf = G['in_proj'] & (G['dist'] <= G['band']) & np.isin(G['k'], G['ids']); k = np.where(conf, G['k'], -1).astype(np.int64)
            lid = np.where(conf, k, 99).astype(np.int64)
            A.update(feats=feats.astype(np.float32), line_id=lid.astype(np.int32), truth_k=k.astype(np.int32), confirmable=conf, point_correct=acc & conf & (lid == k),
                     point_wrong=acc & conf & (lid != k), point_unconf=acc & ~conf)
            dec['aux']['r']['line_id'] = lid
        base = BASE[row['config']] + 0.001 * int(row['block']) + 0.0005 * int(row['noise_k']) + float(row['sigma']) / 10000.0
        st = lambda x: (dict(rmse=float(np.sqrt(np.mean(x * x))), p95=float(np.percentile(np.abs(x), 95))) if x.size else dict(rmse=None, p95=None))
        sg, sl = st(e[ig]), st(e[il])
        c, w, u = (int((acc & A[k]).sum()) for k in ('point_correct', 'point_wrong', 'point_unconf'))
        m = dict(state=('no_output' if empty else 'ok'), n_out=n, n_accepted=int(acc.sum()), n_valid=int(val.sum()), glob_n_valid=int(ig.sum()), loc_n_valid=int(il.sum()),
                 n_selected=n, n_detected=n, n_ambiguous=0, id_correct=c, id_wrong=w, id_unconf=u,
                 glob_id_correct=int((A['point_correct'] & ig).sum()), glob_id_wrong=int((A['point_wrong'] & ig).sum()), glob_id_unconf=int((A['point_unconf'] & ig).sum()),
                 loc_id_correct=int((A['point_correct'] & il).sum()), loc_id_wrong=int((A['point_wrong'] & il).sum()), loc_id_unconf=int((A['point_unconf'] & il).sum()),
                 glob_Ctau=(0.0 if empty else base), loc_Ctau=(0.0 if empty else base + 0.05), glob_Cvalid=(0.0 if empty else 0.8), loc_Cvalid=(0.0 if empty else 0.85),
                 glob_cov_interp=(0.0 if empty else 0.9), loc_cov_interp=(0.0 if empty else 0.95), glob_recovery=(0.0 if empty else 0.9), loc_recovery=(0.0 if empty else 0.9),
                 glob_pt_rmse=sg['rmse'], glob_pt_p95=sg['p95'], loc_pt_rmse=sl['rmse'], loc_pt_p95=sl['p95'], glob_surf_rmse=(None if empty else 0.1),
                 loc_surf_rmse=(None if empty else 0.12), glob_surf_p95=(None if empty else 0.2), loc_surf_p95=(None if empty else 0.22), glob_gross_rate=0.0,
                 img_n_expected=n, img_n_recovered=(0 if empty else n))
        if self.mode.get('nonfinite_key') == fk:
            m['glob_surf_rmse'] = float('inf')
        return m, A, {}


def _root(name):
    p = DRILLS / name
    shutil.rmtree(p, ignore_errors=True)
    return K.Store(p)


def _quiet(_m):
    pass


def _rows(M, surfaces=('M24_curv',), blocks=(0, 1), dots=tuple(MF.DOT_CONFIGS), lines=('s8', 's10')):
    D = M[(M.surface.isin(surfaces)) & (M.block.isin(blocks)) & (((M.family == 'point') & M.config.isin(dots)) | ((M.family == 'line') & M.config.isin(lines)))]
    return D.sort_values(['family', 'config', 'surface', 'block', 'sigma', 'noise_k']).to_dict('records')


def _raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
        return None
    except K.Stop as e:
        return e.msg
    except Exception as e:
        return f'other:{e!r}'


def g1_design(M) -> dict:
    import tr5_claims as CL
    import tr5_main as TM
    res = {}
    cov = MF.coverage_check(M); res['manifest_480_40_864'] = cov['passed']
    res['optional_not_included'] = cov['optional_included'] == 0
    res['integer_seed_from_float_ok'] = MF._int_exact(9830000.0, 'seed') == 9830000
    res['nan_seed_refused'] = _raises(MF._int_exact, float('nan'), 'seed') is not None
    res['fractional_seed_refused'] = _raises(MF._int_exact, 9830000.5, 'seed') is not None
    res['missing_seed_refused'] = _raises(MF._int_exact, None, 'seed') is not None
    n = 0
    for row in M.to_dict('records'):
        CH.forward_id(row, MF.context(row)); n += 1
    res['all_480_contexts_assemble'] = n == 480
    cp = TM.checkpoint_rows(M)
    res['checkpoint_12_registered'] = (len(cp) == 12 and sum(r['family'] == 'point' for r in cp) == 8 and {r['surface'] for r in cp} == {'M24_curv'}
                                      and {r['block'] for r in cp} == {0} and {r['config'] for r in cp if r['family'] == 'point'} == set(MF.DOT_CONFIGS))
    T = CL.registry(); res['claims_54_rows'] = len(T) == 54 and set(T.kind) <= {'text', 'table', 'none'}
    return res


def g2_end_to_end(M, fps) -> tuple[dict, dict]:
    import tr5_analysis as A
    import tr5_claims as CL
    import tr5_fig as F
    res = {}; st = _root('g2'); rows = _rows(M)
    empty = next(r for r in rows if r['family'] == 'point' and r['config'] == 'r1.0/R8x6' and r['block'] == 1 and r['sigma'] == 80.0 and r['noise_k'] == 1)['forward_key']
    out = CH.schedule(rows, st, fps, backend=spec(empty_key=empty), workers=2, log=_quiet)
    res['pool_run_no_failure'] = not out['failures'] and out['n_jobs'] == 48
    comp = CH.completeness(rows, st.root, fps); res['complete_80_results'] = comp['complete'] and comp['n_results'] == 80
    c = st.counts(); res['ledger_closed_ok'] = all(v.get('open', 0) == 0 and v.get('failed', 0) == 0 for v in c.values()) and c['render']['reserved'] == 48 and c['decode']['reserved'] == 80
    R = A.results(rows, st.root, fps)
    res['legal_no_output_kept'] = int((R.state == 'no_output').sum()) == 2 and bool(R[R.state == 'no_output'].glob_Ctau.eq(0).all()) and R[R.state == 'no_output'].glob_pt_rmse.isna().all()
    claims = CL.registry(); csum = CL.claims_summary(claims)
    E = A.effects(R, claims, csum); P = E['pairs']
    per = P.groupby(['contrast', 'surface', 'block', 'sigma', 'method']).size()
    res['two_realisations_give_two_pairs'] = bool((per == 2).all()) and len(P) == 4 * 16 + 8
    ok = P[(P.forward_key_left != empty) & (P.forward_key_right != empty)]
    exact = all(abs(r.d_glob_Ctau - 100 * (BASE[r.left] - BASE[r.right])) < 1e-9 for r in ok.itertuples())
    res['known_differences_recovered_exactly'] = exact and len(ok) == len(P) - 4
    res['frontend_pairs_32'] = len(E['frontend']) == 32
    exp = CL.expected(claims)
    sel = dict(F4=A.select_cases(P, exp, 'C1', 'D0', 80.0), F5=A.select_cases(P, exp, 'C5', 'P4', 60.0))
    res['cases_selected'] = sel['F4']['median'] is not None and sel['F5']['worst'] is not None and sel['F4']['median'] != sel['F4']['worst']
    out_dir = K.guard(DRILLS / 'g2' / 'figs' / 'x').parent
    figs = F.make_all(R, P, E['frontend'], E['cells'], E['cells_frontend'], E['decisions'], sel, st.root, out_dir, technical=True)
    res['five_figures_written'] = len(figs) == 5 and all((K.ROOT / v).exists() for v in figs.values())
    md = F.tables_md(E['cells'], E['decisions']); res['report_tables_generated'] = md.count('\n') > 20
    return res, dict(R=R, rows=rows, root=st.root, P=P, empty=empty)


def g3_pairing(ctx: dict) -> dict:
    import tr5_analysis as A
    R = ctx['R']; res = {}
    i0 = R[(R.config == 'r0.5/R8x4') & (R.method == 'D0')].index[0]
    res['missing_arm_refused'] = 'one-to-one' in (_raises(A.method_pairs, R.drop(index=i0), 'C1') or '')
    R2 = R.copy(); j = R2[(R2.config == 'r0.5/R8x4') & (R2.method == 'D0') & (R2.noise_k == 1)].index[0]; R2.loc[j, 'noise_k'] = 0
    res['duplicate_noise_k_refused'] = 'one-to-one' in (_raises(A.method_pairs, R2, 'C1') or '')
    R3 = R.copy(); R3.loc[i0, 'seed'] = int(R3.loc[i0, 'seed']) + 1
    res['seed_mismatch_refused'] = 'seeds differ' in (_raises(A.method_pairs, R3, 'C1') or '')
    R4 = R.copy(); R4.loc[i0, 'config'] = None
    res['missing_config_label_refused'] = _raises(A.method_pairs, R4, 'C1') is not None
    res['empty_core_subset_refused'] = 'empty' in (_raises(A.method_pairs, R[R.family == 'line'], 'C1') or '')
    R5 = R.drop(index=R[(R.family == 'point') & (R.method == 'DF')].index[0])
    res['frontend_missing_refused'] = 'one each' in (_raises(A.frontend_pairs, R5) or '')
    R6 = R.copy(); k = R6[(R6.family == 'point') & (R6.method == 'DF')].index[0]; R6.loc[k, 'eval_id'] = 'other'
    res['frontend_version_mismatch_refused'] = 'evaluation version' in (_raises(A.frontend_pairs, R6) or '')
    return res


def _synthetic_cells(vals_by_surface: dict, expected_sign: int):
    import tr5_analysis as A
    rows = []
    for s, vals in vals_by_surface.items():
        for b, v in enumerate(vals):
            rec = dict(contrast='C2', family='point', method='D0', surface=s, sigma=60.0, block=b, record_type='block', n_realisations=2)
            for m in A.SUMMARY_METRICS:
                rec['d_' + m] = (v if m == 'glob_Ctau' else 0.0); rec['n_' + m] = 2
            rows.append(rec)
    B = pd.DataFrame(rows)
    exp = {('C2', 'D0', 60.0, s): expected_sign for s in vals_by_surface}
    C = A.cells(B, ('contrast', 'method', 'surface', 'sigma'), lambda r: 'point', exp)
    reg = [dict(contrast='C2', method='D0', sigma=60.0, kinds=['text'], expected_signs={s: expected_sign for s in vals_by_surface})]
    return C, A.decisions(C, reg)


def g4_rules() -> dict:
    import tr5_analysis as A
    res = {}; S = ('B_curv', 'M10_curv', 'M24_curv')
    _, D = _synthetic_cells({s: [1.0 + 0.1 * b for b in range(8)] for s in S}, 1); res['all_positive_supported'] = D.label.iloc[0] == 'supported'
    _, D = _synthetic_cells({s: [-1.0 - 0.1 * b for b in range(8)] for s in S}, 1); res['all_negative_reversed'] = D.label.iloc[0] == 'reversed'
    _, D = _synthetic_cells({s: [(-1) ** b * 0.5 for b in range(8)] for s in S}, 1); res['alternating_insufficient'] = D.label.iloc[0] == 'insufficient'
    _, D = _synthetic_cells({'B_curv': [1.0] * 8, 'M10_curv': [1.0] * 8, 'M24_curv': [-0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.1]}, 1)
    res['mixed_surfaces_qualified'] = D.label.iloc[0] == 'qualified'
    _, D = _synthetic_cells({s: [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0] for s in S}, 1)
    res['five_of_eight_not_supported'] = D.label.iloc[0] != 'supported'
    C1, _ = _synthetic_cells({s: [0.3 * b - 1 for b in range(8)] for s in S}, 1); C2, _ = _synthetic_cells({s: [0.3 * b - 1 for b in range(8)] for s in S}, 1)
    res['bootstrap_deterministic'] = bool(np.allclose(C1.lo_glob_Ctau.astype(float), C2.lo_glob_Ctau.astype(float)))
    res['bootstrap_index_shared'] = bool(np.array_equal(A.boot_index('point', 8), A.boot_index('point', 8)))
    rows = []
    for s, vals in (('B_curv', [0.5, -2.0]), ('M10_curv', [1.0, 3.0]), ('M24_curv', [-2.0, 1.0])):
        for b, v in enumerate(vals):
            rows.append(dict(contrast='C1', method='D0', sigma=80.0, surface=s, block=b, noise_k=0, d_glob_Ctau=v, forward_key_left=f'L{s}{b}', forward_key_right=f'R{s}{b}',
                             state_left='ok', state_right='ok'))
    P = pd.DataFrame(rows); exp = {('C1', 'D0', 80.0, s): 1 for s in ('B_curv', 'M10_curv', 'M24_curv')}
    sel = A.select_cases(P, exp, 'C1', 'D0', 80.0)
    res['worst_is_minimum_with_tie_order'] = sel['worst']['surface'] == 'B_curv' and sel['worst']['block'] == 1
    res['median_is_closest'] = abs(sel['median']['oriented_diff_pp'] - sel['median_value']) == min(abs(v - sel['median_value']) for v in P.d_glob_Ctau)
    P2 = pd.DataFrame(rows[:1]); s2 = A.select_cases(pd.concat([P2, P2.assign(block=1, forward_key_left='x', d_glob_Ctau=0.5)]), exp, 'C1', 'D0', 80.0)
    res['coinciding_median_takes_next'] = s2['median'] != s2['worst']
    return res


def g5_integrity(M, fps) -> dict:
    import tr5_analysis as A
    res = {}; rows = _rows(M, blocks=(0,), dots=('r1.0/R8x4',), lines=())[:4]
    st = _root('g5a'); fk = rows[0]['forward_key']
    out = CH.schedule(rows[:1], st, fps, backend=spec(none_key=f'{fk}|D0'), workers=0, log=_quiet)
    c = st.counts()
    res['none_decoder_is_failure'] = bool(out['failures']) and out['failures'][0]['kind'] == 'result_integrity' and (c.get('decode') or {}).get('failed') == 1
    meta, _, _ = CH.pack_read(st.root, fk); res['no_artefact_for_failed_decode'] = 'D0' not in (meta.get('results') or {})
    res['remaining_steps_aborted_counted'] = (c.get('evaluation') or {}).get('aborted', 0) >= 1 and (c.get('render') or {}).get('ok') == 1
    st = _root('g5b'); out = CH.schedule(rows[1:2], st, fps, backend=spec(missing_key=rows[1]['forward_key']), workers=0, log=_quiet)
    res['missing_core_field_is_failure'] = bool(out['failures']) and out['failures'][0]['kind'] == 'result_integrity'
    st = _root('g5c'); out = CH.schedule(rows[2:3], st, fps, backend=spec(nonfinite_key=rows[2]['forward_key']), workers=0, log=_quiet)
    res['nonfinite_error_is_failure'] = bool(out['failures']) and out['failures'][0]['kind'] == 'integrity' and (st.counts().get('evaluation') or {}).get('failed') == 1
    st = _root('g5d'); out = CH.schedule(rows[3:4], st, fps, backend=spec(render_bad=rows[3]['forward_key']), workers=0, log=_quiet)
    res['bad_render_is_failure'] = bool(out['failures']) and (st.counts().get('render') or {}).get('failed') == 1
    st = _root('g5e'); CH.schedule(rows[:2], st, fps, backend=spec(), workers=0, log=_quiet)
    R = A.results(rows[:2], st.root, fps)
    res['core_gate_nan_refused'] = 'core result values missing' in (_raises(A.core_gate, R.assign(glob_Ctau=[np.nan] + list(R.glob_Ctau[1:]))) or '')
    res['core_gate_inf_refused'] = 'finite' in (_raises(A.core_gate, R.assign(glob_pt_rmse=[float('inf')] + list(R.glob_pt_rmse[1:]))) or '')
    res['core_gate_column_refused'] = 'columns missing' in (_raises(A.core_gate, R.drop(columns=['full_w'])) or '')
    CH.pack_path(st.root, rows[1]['forward_key']).unlink()
    res['missing_pack_refused_by_analysis'] = 'missing' in (_raises(A.results, rows[:2], st.root, fps) or '')
    return res


def g6_resume(M, fps) -> dict:
    res = {}
    rows = [r for r in _rows(M, blocks=(0,), dots=('r1.0/R8x4',), lines=()) if r['family'] == 'point'][:3] + \
        [r for r in _rows(M, blocks=(0,), dots=(), lines=('s8',)) if r['family'] == 'line'][:1]
    st = _root('g6'); CH.schedule(rows, st, fps, backend=spec(), workers=0, log=_quiet)
    c0 = st.counts()['render']['reserved']
    out = CH.schedule(rows, st, fps, backend=spec(), workers=0, log=_quiet); res['intact_not_recomputed'] = out['n_jobs'] == 0 and st.counts()['render']['reserved'] == c0
    f2 = dict(fps, DF='changed-DF'); before = st.counts()
    CH.schedule(rows, st, f2, backend=spec(), workers=0, log=_quiet); after = st.counts()
    nd = sum(1 for r in rows if r['family'] == 'point')
    res['method_change_redecodes_only_that_method'] = (after['render']['reserved'] == before['render']['reserved'] and after['decode_DF']['reserved'] - before['decode_DF']['reserved'] == nd
                                                        and after.get('decode_D0', {}).get('reserved', 0) == before.get('decode_D0', {}).get('reserved', 0))
    f3 = dict(f2, EVAL='changed-EVAL'); before = st.counts()
    CH.schedule(rows, st, f3, backend=spec(), workers=0, log=_quiet); after = st.counts()
    n_res = sum(len(MF.METHODS[r['family']]) for r in rows)
    res['eval_change_redoes_evaluations_only'] = after['decode']['reserved'] == before['decode']['reserved'] and after['evaluation']['reserved'] - before['evaluation']['reserved'] == n_res
    CH.pack_path(st.root, rows[0]['forward_key']).unlink(); before = st.counts()
    CH.schedule(rows, st, f3, backend=spec(), workers=0, log=_quiet); after = st.counts()
    res['deleted_pack_redone_only'] = after['render']['reserved'] - before['render']['reserved'] == 1
    p = CH.pack_path(st.root, rows[1]['forward_key']); b = bytearray(p.read_bytes()); b[len(b) // 2] ^= 0xFF; p.write_bytes(bytes(b))
    need = CH.needs(st.root, rows[1], MF.context(rows[1]), f3); res['corrupted_pack_detected'] = need['render'] is True
    res['ledger_not_proof_of_completion'] = not CH.completeness(rows, st.root, f3)['complete']
    return res


def g7_preflight_budget_exit(M, fps, freeze, permit) -> dict:
    import adp_surf02 as SA
    import tr5_main as TM
    res = {}

    def refused(**kw):
        try:
            TM.preflight(M, kw.pop('permit', permit), kw.pop('freeze', freeze), **kw); return None
        except K.Stop as e:
            return e.code
    res['valid_permit_passes'] = refused() is None
    res['weak_permit_refused'] = refused(permit=dict(authorised=True, task=K.TASK_ID)) == 'refused'
    res['b1_permit_refused'] = refused(permit=(K.load_json(K.OUT4 / 'B1_PERMIT.json') if (K.OUT4 / 'B1_PERMIT.json').exists() else dict(task='THESIS-REINFORCE-04'))) == 'refused'
    pe = dict(permit, budget=dict(permit['budget'], render=600)); pe['permit_id'] = K.digest({a: b for a, b in pe.items() if a != 'permit_id'})
    res['budget_extension_refused'] = refused(permit=pe) == 'refused'
    rel = K.load_json(SA.RELEASE_RECORD); man = K.load_json(K.ROOT / rel['dependency_manifest']); mc = copy.deepcopy(man); mc['files'][sorted(mc['files'])[0]] = '0' * 64
    tmp = K.guard(DRILLS / 'g7' / 'deps_copy.json'); K.save_json(tmp, mc); drift = SA.dependency_drift(tmp)
    res['release_dependency_drift_refused'] = bool(drift) and refused(platform_fn=lambda: dict(passed=False, drift=drift)) == 'preflight'
    for nm, key in (('input_hash_change_refused', 'inputs'), ('closure_change_refused', 'closure')):
        f2 = copy.deepcopy(freeze); f2[key][sorted(f2[key])[0]] = '0' * 64
        p2 = dict(permit, **{('inputs_digest' if key == 'inputs' else 'closure_digest'): K.digest(f2[key])}); p2['permit_id'] = K.digest({a: b for a, b in p2.items() if a != 'permit_id'})
        res[nm] = refused(freeze=f2, permit=p2) == 'preflight'
    f4 = copy.deepcopy(freeze); f4['method_fingerprints']['P4'] = 'x'
    p4 = dict(permit, method_fingerprints=f4['method_fingerprints']); p4['permit_id'] = K.digest({a: b for a, b in p4.items() if a != 'permit_id'})
    res['method_binding_change_refused'] = refused(freeze=f4, permit=p4) == 'preflight'
    rows = _rows(M, blocks=(0,), dots=('r1.0/R8x4',), lines=())[:3]
    _root('g7b'); st = K.Store(DRILLS / 'g7b', caps=dict(K.CAPS, render=2))
    try:
        CH.schedule(rows, st, fps, backend=spec(), workers=0, log=_quiet); res['budget_exhaustion_refused_before_call'] = False
    except K.Stop as e:
        res['budget_exhaustion_refused_before_call'] = e.code == 'budget' and st.counts()['render']['reserved'] == 2 and CH.completeness(rows[:2], st.root, fps)['complete']
    st = _root('g7c')
    try:
        CH.schedule(rows[:1], st, fps, backend=spec(interrupt_at=2), workers=0, log=_quiet); res['interrupt_raised'] = False
    except KeyboardInterrupt:
        res['interrupt_raised'] = True
    c = st.counts(); res['interrupt_counted'] = (c.get('decode') or {}).get('interrupted') == 1 and (c.get('render') or {}).get('ok') == 1
    CH.schedule(rows[:1], st, fps, backend=spec(), workers=0, log=_quiet); c2 = st.counts()
    res['resume_keeps_committed_and_counts'] = c2['render']['reserved'] == 1 and c2['decode']['reserved'] == 3 and CH.completeness(rows[:1], st.root, fps)['complete']
    codes = {}
    for case in ('none', 'interrupt', 'error', 'refused'):
        r = subprocess.run([sys.executable, '-B', str(K.HERE / 'tr5_main.py'), 'drill-exit', '--case', case, '--root', str(DRILLS / f'exit_{case}')], capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=900)
        codes[case] = r.returncode
    codes['usage'] = subprocess.run([sys.executable, '-B', str(K.HERE / 'tr5_main.py'), 'no-such-command'], capture_output=True, text=True, timeout=300).returncode
    res['exit_codes'] = codes
    res['exit_codes_nonzero_registered'] = codes == dict(none=41, interrupt=42, error=40, refused=20, usage=2)
    return res


def run_all(M, fps, freeze, permit) -> dict:
    shutil.rmtree(DRILLS, ignore_errors=True); groups = {}
    try:
        groups['1_design_contexts'] = _safe(lambda: g1_design(M))
        g2 = _safe(lambda: g2_end_to_end(M, fps), pair=True); groups['2_end_to_end_two_realisations'] = g2[0]
        groups['3_pairing_rejections'] = _safe(lambda: g3_pairing(g2[1])) if g2[1] else dict(error='no g2 data')
        groups['4_decision_bootstrap_cases'] = _safe(g4_rules)
        groups['5_integrity_rejections'] = _safe(lambda: g5_integrity(M, fps))
        groups['6_resume_layers'] = _safe(lambda: g6_resume(M, fps))
        groups['7_preflight_budget_interrupt_exit'] = _safe(lambda: g7_preflight_budget_exit(M, fps, freeze, permit))
    finally:
        shutil.rmtree(DRILLS, ignore_errors=True)
    ok = {n: ('error' not in g) and all(v for k, v in g.items() if k != 'exit_codes') for n, g in groups.items()}
    return dict(passed=all(ok.values()), groups_passed=ok, groups=groups, n_checks=sum(len([k for k in g if k != 'exit_codes']) for g in groups.values()),
                isolated_root_removed=not DRILLS.exists(), note='stub backend only; no real renderer, decoder or domain evaluator was called')


def _safe(fn, pair=False):
    import traceback
    try:
        return fn()
    except Exception as e:
        err = dict(error=repr(e)[:400], trace=traceback.format_exc()[-1500:])
        return (err, None) if pair else err


def exit_case(case: str, root: str) -> dict:
    import tr5_main as TM
    from pathlib import Path
    if DRILLS.resolve() in Path(root).resolve().parents:
        shutil.rmtree(root, ignore_errors=True)
    M, _ = MF.build_manifest(); st = K.Store(root)
    if case == 'refused':
        TM.preflight(M, dict(authorised=True, task='x'), dict())
    fps = dict(D0='d', DF='f', P4='p', EVAL='e'); rows = _rows(M, blocks=(0,), dots=('r1.0/R8x4',), lines=())[:1]
    mode = dict(none=dict(none_key=f"{rows[0]['forward_key']}|D0"), interrupt=dict(interrupt_at=1), error=dict(error_at=1))[case]
    out = CH.schedule(rows, st, fps, backend=spec(**mode), workers=0, log=_quiet)
    if out['failures']:
        f = out['failures'][0]
        raise K.Stop('result_integrity' if f.get('kind') in ('result_integrity', 'integrity') else 'technical', f"drill failure: {f.get('error')}")
    return dict(unexpected='no failure')
