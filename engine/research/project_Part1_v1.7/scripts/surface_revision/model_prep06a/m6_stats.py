# Compute phase-block summaries and paired diagnostic intervals.

from __future__ import annotations

import numpy as np
import pandas as pd

import m6_common as K


def pair(a: pd.DataFrame, b: pd.DataFrame, keys: list, cols: list, suffixes=('_a', '_b'), require_all: bool = True) -> pd.DataFrame:
    for nm, t in (('left', a), ('right', b)):
        d = t.duplicated(keys, keep=False)
        if d.any():
            raise K.Stop('pairing', f'{nm} side: {int(d.sum())} rows with duplicate keys {keys} (many-to-many refused)')
    m = a[keys + cols].merge(b[keys + cols], on=keys, how='outer', suffixes=suffixes, indicator=True, validate='one_to_one')
    if require_all and (m['_merge'] != 'both').any():
        raise K.Stop('pairing', f"{int((m['_merge'] != 'both').sum())} keys without partner on {keys}")
    return m[m['_merge'] == 'both'].drop(columns='_merge')


def blocks(g: pd.DataFrame, value: str = 'value') -> dict:

    if len(g) == 0:
        return dict(mean=None, n_blocks_def=0, values={}, vmin=None, vmax=None, n_pos=0, n_neg=0)
    d = g.duplicated(['phase_block', 'rep'], keep=False)
    if d.any():
        raise K.Stop('pairing', f'duplicate (phase_block, rep) labels: {g.loc[d, ["phase_block", "rep"]].values.tolist()[:4]}')
    vals = {}
    for b, gb in g.groupby('phase_block'):
        reps = sorted(int(x) for x in gb.rep)
        if reps != list(K.REPS):
            raise K.Stop('pairing', f'block {b}: reps {reps} (exactly 1 and 2 required)')
        v = [float(x) for x in gb[value] if not K.undef(x)]
        vals[int(b)] = (sum(v) / len(v)) if v else None
    if sorted(vals) != list(K.BLOCKS):
        raise K.Stop('pairing', f'phase blocks {sorted(vals)} (all four required)')
    dv = [v for v in vals.values() if v is not None]
    return dict(mean=(sum(dv) / len(dv)) if dv else None, n_blocks_def=len(dv), values=vals, vmin=(min(dv) if dv else None), vmax=(max(dv) if dv else None),
                n_pos=sum(1 for v in dv if v > 0), n_neg=sum(1 for v in dv if v < 0))


def flat(prefix: str, s: dict) -> dict:
    out = {f'{prefix}mean': s['mean'], f'{prefix}min': s['vmin'], f'{prefix}max': s['vmax'], f'{prefix}n_pos': s['n_pos'], f'{prefix}n_neg': s['n_neg'],
           f'{prefix}n_blocks_def': s['n_blocks_def']}
    for b in K.BLOCKS:
        out[f'{prefix}b{b}'] = s['values'].get(b)
    return out
