# Assign code parity consistently across pattern phases.

from __future__ import annotations

import numpy as np

import c2_common as K
import c3_raster as RS

PARITY_OF_PHASE = {f'N{i}': i % 2 for i in range(8)}


def effective_phase(fam: str, ph: dict, parity: int) -> dict:
    p = int(parity)
    if p not in (0, 1):
        raise K.Stop('check', f'code_parity {parity} not in (0, 1)')
    if fam == 'dots':
        return dict(ph, u=float(ph['u']) + p)
    return dict(ph, phase_frac=float(ph['phase_frac']) + p)


def build(fam: str, layout: str, pattern: str, ph: dict, lev: dict, parity: int) -> dict:
    eff = effective_phase(fam, ph, 0 if pattern == 'plain' else parity)
    B = RS.build(fam, layout, pattern, eff, lev)
    B['code_parity'] = 0 if pattern == 'plain' else int(parity)
    return B


def class_table(fam: str, B: dict) -> np.ndarray:

    if fam == 'dots':
        c = np.asarray(B['centres'], np.float64); cls = np.asarray(B['classes'], np.int64)
        return np.column_stack([c, cls]) if len(c) else np.zeros((0, 3))
    import lc_pattern as LP
    ol = B['meta_full']['olines']; ids = np.asarray(ol['ids'], np.int64)
    pos = np.array([ol['segments'][str(int(k))]['start_uv'][0] for k in ids], float)
    cls = LP.class_of_id(ids) if (ol.get('code') is not None) else np.zeros(len(ids), np.int64)
    return np.column_stack([ids, pos, cls])


def class_map_fp(fam: str, B: dict) -> str:
    return K.digest(dict(fam=fam, table=np.round(class_table(fam, B), 9).tolist()), 20)
