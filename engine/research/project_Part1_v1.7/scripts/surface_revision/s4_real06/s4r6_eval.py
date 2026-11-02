# Evaluate layout outputs on the registered global and local domains.

from __future__ import annotations

import numpy as np

import s4r6_common as H

K = H.K
IDENT_RULE = {'line': 'sp2_lines_table.truth_identity_table (band w/2 + 1 px, two-nearest table search) via rdy_eval.evaluate_lines_table',
              'point': 'r2_eval_snap.evaluate_case (projector-inside, d <= r + 1 px) + source identity (sl_lib sigma-0 labels)'}
_MESH_HASH = {}


def gross_mm() -> float:
    import rdy_platform as P
    g = float(P.s2().GROSS_MM)
    if abs(g - H.GROSS_MM) > 1e-12:
        raise H.Stop('binding', f'gross threshold {g} != registered {H.GROSS_MM}')
    return g


def mesh_hash(surface: str) -> str:
    import rdy_platform as P
    if surface not in _MESH_HASH:
        m = P.mesh(surface); h = []
        for k in sorted(m):
            v = m[k]
            if isinstance(v, np.ndarray):
                h.append(f"{k}:{K.sha_bytes(np.ascontiguousarray(v).tobytes())}")
        _MESH_HASH[surface] = K.digest(h, 32)
    return _MESH_HASH[surface]


def domains(surface: str) -> dict:
    import r2_eval as E
    import rdy_platform as P
    return dict(omega1=E.domain('omega1')['hash'], local=P.dom_local(surface)['hash'])


def eval_deps() -> dict:
    return H.code_hashes(H.EVAL_DEPS)


def evaluate(family: str, surface: str, dec: dict, inp: dict, img=None, ref_img=None) -> tuple[dict, dict, dict]:
    import r2_eval as E
    import rdy_eval as RE
    import rdy_platform as P
    gross_mm()
    m, arr, info = E.evaluate(family, surface, dec, inp, K.BOUNDARY, E.domain('omega1'))
    nodes = E.node_counts(arr['pts'], arr['valid'], surface, E.domain('omega1'))
    extras = dict(nodes=nodes, displacement=RE.displacement(inp))
    if family == 'line':
        st = np.asarray(arr['status'], np.int64)
        m['n_status_clipped_source_abstain'] = int((st == 11).sum())
        keys = [k for k in m if k.startswith('n_status_')]
        m['status_sum'] = int(sum(int(m[k]) for k in keys)); m['status_closed'] = bool(m['status_sum'] == int(m['n_detected']))
        known = set(range(11)) | {11}
        m['status_unknown_codes'] = sorted(int(x) for x in set(np.unique(st).tolist()) - known)
    else:
        if img is None or ref_img is None:
            raise H.Stop('dependency', 'dot evaluation needs the image and its sigma-0 reference image')
        si, code = RE.source_identity(img, ref_img, inp, dec, P.mesh(surface)); arr['src_ident_code'] = code
        extras['source_identity'] = si
        acc = np.asarray(dec['accepted'], bool)
        m['n_features_detected'] = int(len(np.asarray(dec['aux']['feats']).reshape(-1, 2)))
        m['status_closed'] = bool(int(acc.sum()) + int((~acc).sum()) == len(acc))
    return m, arr, extras
