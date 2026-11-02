# Summarize encoding effects across surfaces, phases and noise levels.

from __future__ import annotations

import numpy as np
import pandas as pd

import c5_tables as T5
import c6_tables as T6
import c7_common as K

READ_DOTS = ['cls_pre_correct', 'cls_post_correct', 'cls_guard_readable', 'guard_failed', 'small_true', 'large_true', 'src_background', 'src_undeterminable_multi',
             'src_undeterminable_partial', 'src_undeterminable_unlabelled', 'src_recovered_dots']
READ_LINES = ['L1_read', 'L1_n2w', 'L1_w2n', 'L1_no_thr', 'L2_readable', 'L3_adopted', 'L3_n2w', 'L3_w2n', 'seg_known', 'seg_mixed', 'seg_unknown', 'threshold_px']
READ_BOTH = ['err_lower', 'err_upper', 'err_known', 'status_0.0005', 'status_0.005', 'status_bridge', 'b_id_correct', 'b_id_wrong', 'b_id_unconf']
MAN_EXTRA = ['sizes_mm', 'Q0', 'v_scale', 'Q', 'usage', 'limit', 'raster_key', 'class_map_fp', 'stage']
GROUP = ['family', 'surface', 'layout', 'regime', 'pattern', 'sigma', 'rule', 'arm']
MAIN = dict(dots=('DF', 'use'), lines=('R@1.1247', 'R@1.1247'))
E1_MAIN = dict(dots='E1 DF use-ignore', lines='E1 R-ignore')
E2_MAIN = dict(dots='E2 DF coded-plain (scheme difference incl. size-dependent front end)', lines='E3 lines LC(R)-P4 whole scheme')
PLAIN_MAIN = dict(dots=('DF', 'single'), lines=('P4', 'P4'))
DIAG_FIELDS = ['state', 'n_accepted', 'n_valid', 'glob_Ctau', 'glob_Cvalid', 'loc_Ctau', 'loc_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'b_id_correct', 'b_id_wrong', 'b_id_unconf',
               'src_acc', 'src_correct', 'src_wrong', 'src_unconf', 'acc_both', 'acc_diag_only', 'acc_method_only', 'method_arm', 'method_Ctau', 'method_accepted',
               'n_features', 'n_src_confirmable', 'n_background', 'n_partial', 'n_multi', 'n_ref_unlabelled', 'n_no_mapping', 'tc_replaced', 'tc_changed', 'tc_unknown_kept',
               'tc_excluded', 'ti_assigned', 'ti_excluded', 'ti_unmapped_confirmable', 'ti_duplicate_dots', 'ti_duplicate_features_excluded', 'features_used',
               'geom_confirmable', 'class_replaced', 'geom_unconfirmable_kept_all_classes', 'assigned', 'excluded', 'geom_unconfirmable_excluded',
               'confirmable_unmapped_excluded', 'confirmable_features', 'L3_adopted', 'L3_n2w', 'L3_w2n', 'seg_known', 'image_fallback', 'err_lower', 'err_upper']
DIAG_COUNTS = ['n_accepted', 'b_id_correct', 'b_id_wrong', 'b_id_unconf', 'src_acc', 'src_correct', 'src_wrong', 'src_unconf', 'acc_both', 'acc_diag_only', 'acc_method_only',
               'n_features', 'n_src_confirmable', 'n_background', 'n_partial', 'n_multi', 'n_ref_unlabelled', 'n_no_mapping', 'tc_replaced', 'tc_changed', 'tc_unknown_kept',
               'tc_excluded', 'ti_assigned', 'ti_excluded', 'ti_unmapped_confirmable', 'ti_duplicate_features_excluded', 'features_used', 'geom_confirmable', 'class_replaced',
               'geom_unconfirmable_kept_all_classes', 'assigned', 'excluded', 'geom_unconfirmable_excluded', 'confirmable_unmapped_excluded', 'confirmable_features']
DIAG_APPLIES = dict(source=['src_acc', 'src_correct', 'src_wrong', 'src_unconf', 'acc_both', 'acc_diag_only', 'acc_method_only', 'n_features', 'n_src_confirmable', 'n_background',
                            'n_partial', 'n_multi', 'n_ref_unlabelled', 'n_no_mapping', 'method_Ctau'],
                    bridge_geom=['features_used', 'geom_confirmable', 'b_id_correct', 'b_id_wrong', 'b_id_unconf'],
                    lines_v1=['features_used', 'confirmable_features', 'b_id_correct', 'b_id_wrong', 'b_id_unconf', 'L3_adopted', 'seg_known'])


def _num(s):
    return pd.to_numeric(s, errors='coerce')


def case_table(T: pd.DataFrame, M: pd.DataFrame, cfg5: dict, derived: str) -> pd.DataFrame:
    C = T6.case_table(T, M, cfg5, derived)
    E = T[(T.layer == 'eval') & T.complete].set_index('key')
    for f in READ_DOTS + READ_LINES + READ_BOTH:
        if f not in C.columns and f'r_{f}' in E.columns:
            C[f] = C.eval_key.map(E[f'r_{f}'])
    Mi = M.drop_duplicates('image_key').set_index('image_key')
    for f in MAN_EXTRA:
        if f in Mi.columns and f not in C.columns:
            C[f] = C.image_key.map(Mi[f])
    return C


def _pooled(num, den):
    return float(num / den) if (pd.notna(num) and pd.notna(den) and den > 0) else np.nan


def risk_summary(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in C.groupby(GROUP, dropna=False):
        r = dict(zip(GROUP, k)); r.update(n_records=len(g), n_phases=int(g.phase_id.nunique()))
        for f in ('n_accepted', 'acc', 'c', 'w', 'u', 'new_n', 'new_correct', 'new_wrong', 'new_unconf'):
            r[f'sum_{f}'] = float(_num(g[f]).sum())
        for f in ('flag_wrong_all', 'flag_wrong_new', 'flag_wrong_all_0.0005', 'flag_wrong_new_0.0005', 'flag_wrong_all_0.005', 'flag_wrong_new_0.005', 'flag_unconf', 'flag_no_output'):
            v = _num(g[f]) if f in g.columns else pd.Series(np.nan, index=g.index)
            r[f'n_{f}'] = int((v == 1).sum()); r[f'n_{f}_defined'] = int(v.notna().sum())
        r['n_accepted_but_no_valid'] = int(g.accepted_but_no_valid.sum()); r['n_strict_identity_pass'] = int(g.strict_identity_pass.astype(bool).sum())
        r['pooled_wrong_share_main'] = _pooled(r['sum_w'], r['sum_c'] + r['sum_w']); r['pooled_wrong_share_lower'] = _pooled(r['sum_w'], r['sum_acc'])
        r['pooled_wrong_share_upper'] = _pooled(r['sum_w'] + r['sum_u'], r['sum_acc']); r['pooled_unconf_share'] = _pooled(r['sum_u'], r['sum_acc'])
        r['pooled_wrong_share_new'] = _pooled(r['sum_new_wrong'], r['sum_new_correct'] + r['sum_new_wrong'])
        st = g.status.astype(str).value_counts()
        r['status_counts'] = ';'.join(f'{a}:{int(b)}' for a, b in st.items())
        rows.append(r)
    return pd.DataFrame(rows)


def readout_summary(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    D = C[(C.family == 'dots') & (C.arm == 'use')]
    for k, g in D.groupby(GROUP, dropna=False):
        r = dict(zip(GROUP, k)); r.update(n_records=len(g), n_phases=int(g.phase_id.nunique()), unit='candidates (single-source, dominant dot known)')
        s = {f: (float(_num(g[f]).sum()) if f in g.columns else np.nan) for f in ('cls_true_known', 'cls_pre_correct', 'cls_sel_known', 'cls_post_assigned', 'cls_post_correct', 'cls_post_abstain')}
        r.update({f'sum_{f}': v for f, v in s.items()})
        r['pre_guard_class_accuracy'] = _pooled(s['cls_pre_correct'], s['cls_true_known']); r['post_guard_accuracy'] = _pooled(s['cls_post_correct'], s['cls_post_assigned'])
        r['abstention_share'] = _pooled(s['cls_post_abstain'], s['cls_sel_known'])
        r['n_guard_readable'] = int((g.guard_readable.astype(str).str.lower() == 'true').sum()); r['n_guard_not_readable'] = int((g.guard_readable.astype(str).str.lower() == 'false').sum())
        rows.append(r)
    L = C[(C.family == 'lines') & C.rule.astype(str).str.match(r'^(R|LB)@')]
    for k, g in L.groupby(GROUP, dropna=False):
        r = dict(zip(GROUP, k)); r.update(n_records=len(g), n_phases=int(g.phase_id.nunique()), unit='line segments')
        s = {f: float(_num(g[f]).sum()) for f in ('L1_read', 'L1_n2w', 'L1_w2n', 'L1_no_thr', 'L2_readable', 'L3_adopted', 'L3_n2w', 'L3_w2n', 'seg_known', 'seg_mixed', 'seg_unknown')
             if f in g.columns}
        r.update({f'sum_{f}': v for f, v in s.items()})
        r['L3_class_error_share_of_adopted'] = _pooled(s.get('L3_n2w', 0) + s.get('L3_w2n', 0), s.get('L3_adopted', 0))
        r['L1_width_error_share_of_read'] = _pooled(s.get('L1_n2w', 0) + s.get('L1_w2n', 0), s.get('L1_read', 0))
        fb = g.image_fallback.astype(str).str.lower() == 'true'
        r['n_image_fallback'] = int(fb.sum()); r['decisions'] = ';'.join(f'{a}:{int(b)}' for a, b in g.decision.astype(str).value_counts().items())
        rows.append(r)
    return pd.DataFrame(rows)


def diag_records(T: pd.DataFrame, M: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    G = T[T.layer == 'diag'].copy()
    Mi = M.drop_duplicates('image_key').set_index('image_key')
    out = pd.DataFrame(dict(diag_key=G.key.values, image_key=G.image_key.values, decode_key=G.decode_key.values, rule=G.rule.values, arm=G.arm.values,
                            reference=G.reference.values, complete=G.complete.values, status=G.status.values if 'status' in G.columns else None))
    for c in ('family', 'surface', 'layout', 'regime', 'pattern', 'phase_id', 'sigma', 'rep', 'seed', 'code_parity', 'raster_key'):
        if c in Mi.columns:
            out[c] = out.image_key.map(Mi[c]).values
    out['layout'] = out.layout.astype(str)
    for f in DIAG_FIELDS:
        out[f] = G[f'r_{f}'].values if f'r_{f}' in G.columns else np.nan
    Ci = C.set_index(['decode_key', 'arm'])
    marm = np.where(out.family == 'lines', 'ignore', np.where(out.pattern == 'plain', 'single', 'use'))
    idx = list(zip(out.decode_key, marm))
    for f, src in (('method_eval_key', 'eval_key'), ('method_glob_Ctau', 'glob_Ctau'), ('method_glob_Cvalid', 'glob_Cvalid'), ('method_n_accepted', 'n_accepted')):
        out[f] = [Ci[src].get(i, np.nan) for i in idx]
    out['method_arm_used'] = marm
    out['d_glob_Ctau'] = _num(out.glob_Ctau) - _num(out.method_glob_Ctau)
    out['d_glob_Cvalid'] = _num(out.glob_Cvalid) - _num(out.method_glob_Cvalid)
    return out


def diag_summary(DR: pd.DataFrame) -> pd.DataFrame:
    keys = ['family', 'surface', 'layout', 'regime', 'pattern', 'sigma', 'rule', 'arm', 'reference']
    S = T5.phase_summary(DR.assign(glob_Ctau=_num(DR.glob_Ctau), method_glob_Ctau=_num(DR.method_glob_Ctau), d_glob_Ctau=_num(DR.d_glob_Ctau), glob_Cvalid=_num(DR.glob_Cvalid),
                                   glob_pt_rmse=_num(DR.glob_pt_rmse)),
                         keys, ['glob_Ctau', 'method_glob_Ctau', 'd_glob_Ctau', 'glob_Cvalid', 'glob_pt_rmse'], {'glob_pt_rmse'})
    sums = []
    for k, g in DR.groupby(keys, dropna=False):
        r = dict(zip(keys, k)); app = set(DIAG_APPLIES.get(k[-1], [])) | {'n_accepted'}
        for f in DIAG_COUNTS:
            v = _num(g[f])
            r[f'sum_{f}'] = float(v.sum()) if v.notna().any() else np.nan
        r['n_no_output'] = int((_num(g.n_accepted) == 0).sum()); r['applicable_fields'] = '|'.join(sorted(app))
        sums.append(r)
    return S.merge(pd.DataFrame(sums), on=keys, how='left', validate='one_to_one')


def condition_map(C: pd.DataFrame, SP: pd.DataFrame, B: pd.DataFrame, SC: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam in ('dots', 'lines'):
        rule, arm = MAIN[fam]; prule, parm = PLAIN_MAIN[fam]
        cod = C[(C.family == fam) & (C.rule == rule) & (C.arm == arm)]
        for (surf, lay, pat, sg), _ in cod.groupby(['surface', 'layout', 'pattern', 'sigma']):
            r = dict(family=fam, surface=surf, layout=lay, pattern=pat, sigma=sg, main_rule=rule)
            for nm, eff in (('E1', E1_MAIN[fam]), ('E2', E2_MAIN[fam])):
                q = SP[(SP.effect == eff) & (SP.surface == surf) & (SP.layout.astype(str) == lay) & (SP.pattern == pat) & (SP.sigma == sg)]
                r[f'{nm}_d_glob_Ctau'] = float(q.d_glob_Ctau_mean.iloc[0]) if len(q) == 1 else np.nan
                r[f'{nm}_n_pairs'] = int(q.n_rows.iloc[0]) if len(q) == 1 else 0
                if len(B):
                    b = B[(B.effect == eff) & (B.surface == surf) & (B.layout.astype(str) == lay) & (B.pattern == pat) & (B.sigma == sg)]
                    r[f'{nm}_lo'], r[f'{nm}_hi'] = (float(b.lo.iloc[0]), float(b.hi.iloc[0])) if len(b) == 1 else (np.nan, np.nan)
            for side, (ru, ar), pt in (('coded', (rule, arm), pat), ('plain', (prule, parm), 'plain')):
                q = SC[(SC.family == fam) & (SC.surface == surf) & (SC.layout.astype(str) == lay) & (SC.pattern == pt) & (SC.sigma == sg) & (SC.rule == ru) & (SC.arm == ar)]
                for v in ('glob_Ctau_mean', 'glob_surf_rmse_mean', 'glob_surf_p95_mean', 'glob_surf_rmse_n_defined', 'glob_pt_rmse_mean'):
                    r[f'{side}_{v}'] = float(q[v].iloc[0]) if (len(q) == 1 and v in q.columns) else np.nan
            q = R[(R.family == fam) & (R.surface == surf) & (R.layout.astype(str) == lay) & (R.pattern == pat) & (R.sigma == sg) & (R.rule == rule) & (R.arm == arm)]
            for v in ('n_records', 'n_flag_wrong_all', 'n_flag_wrong_new', 'n_flag_unconf', 'n_flag_no_output', 'pooled_wrong_share_main', 'pooled_wrong_share_new', 'pooled_unconf_share'):
                r[f'risk_{v}'] = (q[v].iloc[0] if len(q) == 1 else np.nan)
            rows.append(r)
    return pd.DataFrame(rows)
