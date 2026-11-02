# Define the size and spacing comparisons checked on new phases.

from __future__ import annotations

import numpy as np
import pandas as pd

import tr5_common as K
import tr5_manifest as MF


REGISTERED_CONTRASTS = {('C5', 'P4', 40.0), ('C2', 'D0', 60.0), ('C3', 'D0', 60.0), ('C4', 'D0', 60.0), ('C5', 'P4', 60.0), ('C1', 'D0', 60.0)}
NONE_BELOW = 0.05
S12_LAYOUT = {'r0.5/R8x4': ('S1', 'S1_R8x4', 1.0), 'r1.0/R8x4': ('S1', 'S1_R8x4', 2.0), 'r1.0/T8x4': ('S2', 'T8x4', 2.0), 'r1.0/R8x6': ('S2', 'R8x6', 2.0)}
A3_LAYOUT = {'r0.5/R8x4': ('S1', 'S1_R8x4', 0.5), 'r1.0/R8x4': ('S1', 'S1_R8x4', 1.0), 'r1.0/T8x4': ('S2', 'T8x4', 1.0), 'r1.0/R8x6': ('S2', 'R8x6', 1.0)}


def _s12_dots(V: pd.DataFrame, cfg: str, sigma: float, surf: str) -> tuple[float, int]:
    sec, lay, dia = S12_LAYOUT[cfg]

    g = V[(V.section == sec) & (V.layout == lay) & (V.diameter_mm == dia) & (V.sigma == sigma) & (V.surface == surf) & (V.family == 'point')]
    ph = g.groupby('phase_label').glob_Ctau.mean()
    return float(ph.mean() * 100), int(len(ph))


def _s12_r8x4_s2(V, sigma, surf):
    g = V[(V.section == 'S2') & (V.layout == 'R8x4') & (V.diameter_mm == 2.0) & (V.sigma == sigma) & (V.surface == surf) & (V.family == 'point')]
    ph = g.groupby('phase_label').glob_Ctau.mean()
    return float(ph.mean() * 100), int(len(ph))


def _s12_lines(V, pitch, sigma, surf):
    g = V[(V.section == 'S2') & (V.family == 'line') & (V.line_pitch_mm == pitch) & (V.orientation_deg == 0) & (V.size_mm == 1.0) & (V.sigma == sigma) & (V.surface == surf)]
    ph = g.groupby('phase_label').glob_Ctau.mean()
    return float(ph.mean() * 100), int(len(ph))


def _a3(A: pd.DataFrame, cfg: str, sigma: float, surf: str, arm: str, s2_r8x4: bool = False):
    part, lay, size = A3_LAYOUT[cfg]
    if s2_r8x4:
        part, lay, size = 'S2', 'R8x4', 1.0
    g = A[(A.part == part) & (A.layout == lay) & (A['size_mm'] == size) & (A.sigma == sigma) & (A.surface == surf) & (A.arm == arm)]
    if len(g) != 1:
        raise K.Stop('preflight', f'TR01 A3 condition not unique: {cfg} {sigma} {surf} {arm} ({len(g)})')
    return float(g.glob_Ctau_mean.iloc[0] * 100), int(g.n_phases.iloc[0])


def registry() -> pd.DataFrame:
    V = K.read_csv(K.ROOT / 'outputs/surface_revision/s12_resource01/results/views.csv')
    V = V[(V.surface_class == 'main') & (V.regime == 'resource_cap_11695')]
    A = K.read_csv(K.OUT1 / 'layers_conditions.csv'); A = A[A.record_type == 'A3_condition'].copy()
    for c in ('size_mm', 'sigma', 'glob_Ctau_mean', 'n_phases'):
        A[c] = pd.to_numeric(A[c], errors='raise')
    rows = []
    for cid, (fam, left, right) in MF.CONTRASTS.items():
        for meth in MF.METHODS[fam]:
            for sg in MF.SIGMAS[fam]:
                for surf in MF.SURF_SID:
                    if fam == 'line':
                        (l, nl), (r, nr) = _s12_lines(V, 10.0, sg, surf), _s12_lines(V, 8.0, sg, surf); src = 'S12-RESOURCE-01 views.csv (P4)'
                    elif meth == 'D0':
                        s2 = cid in ('C2', 'C3')
                        l, nl = _s12_dots(V, left, sg, surf)
                        r, nr = (_s12_r8x4_s2(V, sg, surf) if (s2 and right == 'r1.0/R8x4') else _s12_dots(V, right, sg, surf))
                        if s2 and left == 'r1.0/R8x4':
                            l, nl = _s12_r8x4_s2(V, sg, surf)
                        src = 'S12-RESOURCE-01 views.csv (D0)'
                    else:
                        s2 = cid in ('C2', 'C3')
                        l, nl = _a3(A, left, sg, surf, 'DF', s2_r8x4=(s2 and left == 'r1.0/R8x4'))
                        r, nr = _a3(A, right, sg, surf, 'DF', s2_r8x4=(s2 and right == 'r1.0/R8x4'))
                        src = 'THESIS-REINFORCE-01 A3_condition (DF on the stored S12 images)'
                    d = l - r
                    kind = 'none' if (l < NONE_BELOW and r < NONE_BELOW) else ('text' if (cid, meth, sg) in REGISTERED_CONTRASTS else 'table')
                    rows.append(dict(contrast=cid, family=fam, method=meth, sigma=sg, surface=surf, left=left, right=right, old_left_pct=l, old_right_pct=r, old_diff_pp=d,
                                     old_n_phases_left=nl, old_n_phases_right=nr, kind=kind, expected_sign=(None if kind == 'none' else int(np.sign(d)) or None),
                                     source=src, text_ref=None))
    T = pd.DataFrame(rows)
    T['expected_sign'] = [None if (v is None or pd.isna(v)) else int(v) for v in T.expected_sign]
    T['expected_sign'] = T['expected_sign'].astype(object)
    return T


def expected(T: pd.DataFrame) -> dict:

    return {(r['contrast'], r['method'], float(r['sigma']), r['surface']): (None if r['expected_sign'] is None or pd.isna(r['expected_sign']) else int(r['expected_sign']))
            for r in T.to_dict('records')}


def claims_summary(T: pd.DataFrame) -> list:

    out = []
    for (cid, meth, sg), g in T.groupby(['contrast', 'method', 'sigma']):
        signs = {r['surface']: (None if r['expected_sign'] is None or pd.isna(r['expected_sign']) else int(r['expected_sign'])) for r in g.to_dict('records')}
        kinds = sorted(set(g.kind))
        uniform = len(set(signs.values())) == 1 and None not in signs.values()
        out.append(dict(contrast=cid, method=meth, sigma=sg, kinds=kinds, expected_signs=signs, uniform_direction=uniform,
                        status=('exploratory (no registered direction)' if all(v is None for v in signs.values()) else
                                ('registered direction' if uniform else 'surface-dependent registered direction')),
                        old_diff_pp={r['surface']: round(r['old_diff_pp'], 3) for r in g.to_dict('records')}))
    return out
