# Check the stored inputs used by held-out layout analysis.

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import s4r7_common as H

K = H.K


def a1() -> dict:
    R6 = pd.read_csv(H.reg(H.R6 / 'results.csv'), low_memory=False, float_precision='round_trip')
    C = R6[(R6.record_type == 'cond') & (R6.domain == 'glob')]
    rows = []
    for k in sorted(set(C[C.family == 'line'].image_key)):
        o = C[(C.image_key == k) & (C.method == 'P4TABLE')].iloc[0]; n = C[(C.image_key == k) & (C.method == 'CLIP')].iloc[0]
        Ao = K.load_npz(H.reg(H.R6 / 'evals' / f'{o.eval_key}.npz')); An = K.load_npz(H.reg(H.R6 / 'evals' / f'{n.eval_key}.npz'))
        ind = np.asarray(Ao['in_global'], bool) & np.asarray(Ao['valid'], bool)
        e = np.abs(np.asarray(Ao['pts'], np.float64)[:, 2] - np.asarray(Ao['zr'], np.float64))
        un = ind & np.asarray(Ao['point_unconf'], bool); gr = ind & (e > H.GROSS_MM); rej = ~np.asarray(An['accepted'], bool)
        rows.append(dict(record_type='A1_union', image_key=k, surface=o.surface, phase=o.phase, sigma=o.sigma, layout=o.layout, band=bool(o.band),
                         unconf_rejected=int((un & rej).sum()), gross_rejected=int((gr & rej).sum()), both_rejected=int((un & gr & rej).sum()), union_rejected=int(((un | gr) & rej).sum()),
                         correct_lost=int((ind & np.asarray(Ao['point_correct'], bool) & rej).sum()), dCtau_pp=100 * (float(n.Ctau) - float(o.Ctau)),
                         source_p4=f'outputs/surface_revision/s4_real06/evals/{o.eval_key}.npz', source_clip=f'outputs/surface_revision/s4_real06/evals/{n.eval_key}.npz'))
    D = pd.DataFrame(rows)
    by = D[D.band].groupby('surface')[['unconf_rejected', 'gross_rejected', 'both_rejected', 'union_rejected', 'correct_lost']].sum()
    m10 = D[(D.surface == 'M10_curv') & D.band & (D.sigma == 0.0)]
    anchors = dict(B_union=int(by.loc['B_curv', 'union_rejected']), M10_union=int(by.loc['M10_curv', 'union_rejected']), total_union=int(by.union_rejected.sum()),
                   typical_M10_sigma0=sorted(set(int(x) for x in m10.union_rejected)), double_count_sum=int(by.unconf_rejected.sum() + by.gross_rejected.sum()))
    anchors['ok'] = bool(anchors['B_union'] == 566 and anchors['M10_union'] == 3796 and anchors['total_union'] == 4362 and anchors['typical_M10_sigma0'] == [466])

    def dot(key):
        x = C[(C.image_key == key) & (C.method == 'D0')].iloc[0]
        d = x.decode_key; p = H.R6 / 'decodes' / f'{d}.npz'
        p = p if p.exists() else H.R2 / 'raw' / f'{d}.npz'
        z = K.load_npz(H.reg(p))
        return dict(image_key=key, layout=x.layout, detected=int(len(z['feats'])), ambiguous=int(np.asarray(z['a0_tr_ambiguous'], bool).sum()), accepted=int(np.asarray(z['a0_accepted'], bool).sum()),
                    Ctau=float(x.Ctau), sup_rmse=float(x.sup_rmse), decode_record=K.rel(p))
    S0, S2 = dot('R2I-1353f2daae4955b99b10eef2'), dot('R2I-7519fc6cc02b4a36f087c030')
    S = R6[(R6.record_type == 's0') & (R6.domain == 'glob') & (R6.family == 'point')]
    med = S.groupby(['strategy_view', 'surface']).d_Ctau.median().mul(100).round(2).to_dict()
    dots = dict(worst_case=dict(S0=S0, S2star=S2), median_dCtau_pp={f'{a}|{b}': v for (a, b), v in med.items()},
                ok=bool(S0['detected'] == 238 and S2['detected'] == 238 and S0['ambiguous'] == 9 and S2['ambiguous'] == 68 and S0['accepted'] == 229 and S2['accepted'] == 170
                        and abs(100 * S0['Ctau'] - 82.37) < 0.01 and abs(100 * S2['Ctau'] - 44.75) < 0.01))
    acc = R6[(R6.record_type == 'cond') & (R6.domain == 'glob') & (R6.family == 'point')]
    acc_med = acc.groupby(['layout', 'sigma']).n_accepted.median()
    counts = dict(family_layouts_incl_refs=int(R6[R6.record_type == 'cond'][['baseline', 'layout']].drop_duplicates().shape[0]),
                  line_pilot_layouts=4, line_pilot_new=3, line_pilot_reused=1,
                  dot_accepted_median_range=[float(acc_med.min()), float(acc_med.max())], dot_accepted_case_range=[int(acc.n_accepted.min()), int(acc.n_accepted.max())])
    out = dict(anchors=anchors, dots=dots, counts=counts, note='union / intersection from the same feature masks; S4-REAL-06 results.csv columns unchanged', t_local=K.now_local())
    H.part('A1_union', D); H.run_put('A1', out); H.flush_reads('A1')
    return out
