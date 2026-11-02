# Aggregate complete-line identity counts and rates.

from __future__ import annotations

import numpy as np
import pandas as pd

import tr_common as K

FLAG = 0.001


def _rates(c, w, u) -> dict:
    return dict(c=int(c), w=int(w), u=int(u), risk_wrong=(w / (c + w) if c + w else None), unconf_share=(u / (c + w + u) if c + w + u else None))


def totals(A4views: pd.DataFrame) -> pd.DataFrame:
    rows = []
    V = K.read_csv(K.S12 / 'results' / 'views.csv'); V = V[V.surface_class == 'main']
    a = A4views[A4views.source == 'S12'].drop_duplicates('input_key').set_index('input_key')
    for (fam, regime), g in V.groupby(['family', 'regime']):
        g = g.drop_duplicates('input_key').copy()
        c, w, u = (pd.to_numeric(g[x], errors='coerce').fillna(0).astype(int) for x in ('id_correct', 'id_wrong', 'id_unconf'))
        dw = g.input_key.map((a.all_new_w - a.all_old_w) if len(a) else pd.Series(dtype=float)).fillna(0).astype(int) if fam == 'line' else 0 * c
        dc = g.input_key.map((a.all_new_c - a.all_old_c) if len(a) else pd.Series(dtype=float)).fillna(0).astype(int) if fam == 'line' else 0 * c
        old = _rates(c.sum(), w.sum(), u.sum()); new = _rates((c + dc).sum(), (w + dw).sum(), (u - dw - dc).sum())
        fo = ((w / (c + w).where(c + w > 0)) > FLAG).sum(); fn = (((w + dw) / (c + dc + w + dw).where(c + dc + w + dw > 0)) > FLAG).sum()
        rows.append(dict(source='S12', family=fam, regime=regime, rule_arm='D0' if fam == 'point' else 'P4', n_views=len(g), views_changed=int((dw > 0).sum()),
                         **{f'old_{k}': v for k, v in old.items()}, **{f'new_{k}': v for k, v in new.items()}, flagged_old=int(fo), flagged_new=int(fn)))
    C = K.read_csv(K.ROOT / 'outputs/surface_revision/s3_code07/analysis/formal/current/cases.csv')
    b = A4views[A4views.source == 'S3'].copy(); b['arm_key'] = b.view_id.str.split('|', n=1).str[1]
    for (fam, rule, arm), g in C.groupby(['family', 'rule', 'arm']):
        c, w, u = (pd.to_numeric(g[x], errors='coerce').fillna(0).astype(int) for x in ('b_id_correct', 'b_id_wrong', 'b_id_unconf'))
        old = _rates(c.sum(), w.sum(), u.sum())
        if fam == 'lines':
            bb = b[b.arm == arm].set_index('input_key')
            dw = g.image_key.map(bb.all_new_w - bb.all_old_w).fillna(0).astype(int); dc = g.image_key.map(bb.all_new_c - bb.all_old_c).fillna(0).astype(int)
        else:
            dw = 0 * c; dc = 0 * c
        new = _rates((c + dc).sum(), (w + dw).sum(), (u - dw - dc).sum())
        fo = ((w / (c + w).where(c + w > 0)) > FLAG).sum(); fn = (((w + dw) / (c + dc + w + dw).where(c + dc + w + dw > 0)) > FLAG).sum()
        rows.append(dict(source='S3', family='line' if fam == 'lines' else 'point', regime='resource_cap_11695', rule_arm=f'{rule}|{arm}', n_views=len(g), views_changed=int((dw > 0).sum()),
                         **{f'old_{k}': v for k, v in old.items()}, **{f'new_{k}': v for k, v in new.items()}, flagged_old=int(fo), flagged_new=int(fn)))
    return pd.DataFrame(rows)
