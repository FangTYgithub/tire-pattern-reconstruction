# Re-evaluate size-spacing outputs at alternative support and height tolerances.

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_common as K

COMBOS = [(d, t) for d in K.SENS_D for t in K.SENS_TAU]
_D = {}


def _dom():
    if 'g' not in _D:
        _D['g'] = K.platform_ids()['s1k'].global_domain(0.5)
    return _D['g']


def _one(r: dict) -> dict:
    _dom()
    from surface_eval_r3 import surface_eval_r3_dev
    import s12_reuse as RU
    from types import SimpleNamespace
    ns = SimpleNamespace(**r); ep = RU.eval_record_path(ns)
    rec = K.load_json(ep); m = rec['metrics']
    with np.load(ep.with_suffix('.npz')) as z:
        P = np.asarray(z['pts'], np.float64)[np.asarray(z['valid'], bool)]
    dom = _dom(); sid = r['platform_surface']; n = len(dom['x'])
    out = dict(input_key=r['input_key'], n_valid=int(len(P)), glob_Ctau_record=m['glob_Ctau'])
    if len(P) < 3:
        for d, t in COMBOS:
            out[f'Ctau_d{d:g}_t{t:g}'] = 0.0
        for d in K.SENS_D:
            out[f'Cvalid_d{d:g}'] = 0.0
        out['check_d6_t015'] = bool((m['glob_Ctau'] or 0.0) == 0.0)
        return out
    D = dict(x=dom['x'], y=dom['y'], z_ref=np.asarray(dom[f'z_{sid}'], np.float64), area=np.full(n, dom['step'] ** 2), hash=dom['hash'])
    ev = surface_eval_r3_dev(P, None, D, list(K.SENS_D), list(K.SENS_TAU), support_rule='exact')
    g = ev['subdomains']['global']['by_d']
    for d, t in COMBOS:
        out[f'Ctau_d{d:g}_t{t:g}'] = float(g[str(float(d))]['C_tau'][str(float(t))] or 0.0)
    for d in K.SENS_D:
        out[f'Cvalid_d{d:g}'] = float(g[str(float(d))]['C_valid'] or 0.0)
    out['check_d6_t015'] = bool(abs(out['Ctau_d6_t0.15'] - (m['glob_Ctau'] or 0.0)) <= 1e-12)
    return out


# Re-evaluate saved reconstructions at the registered support and height tolerances.
def run(workers=8) -> dict:
    import s12_stats as ST
    V = pd.read_csv(K.MANIFEST, keep_default_na=False, float_precision='round_trip')
    U = pd.read_csv(K.INPUTS, keep_default_na=False, float_precision='round_trip')
    v = V[(V.regime == 'resource_cap_11695') & (V.surface_class == 'main')]
    u = U[U.input_key.isin(set(v.input_key))]
    p = K.RESULTS / 'sens_inputs.csv'
    done = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=['input_key'])
    todo = u[~u.input_key.isin(set(done.input_key))].to_dict('records')
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            new = list(ex.map(_one, todo, chunksize=16))
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
        K.save_csv(done, p)
    S = done[done.input_key.isin(set(u.input_key))]
    X = v.merge(S, on='input_key', how='left')
    cols = [f'Ctau_d{d:g}_t{t:g}' for d, t in COMBOS] + [f'Cvalid_d{d:g}' for d in K.SENS_D]
    Pt = ST.phase_table(X.assign(surface_class=X.surface_class), metrics=cols)
    rows = []
    for k, g in Pt.groupby(ST.KEYS, dropna=False):
        rec = dict(zip(ST.KEYS, k)); rec['n_phases'] = int(len(g))
        for c in cols:
            rec[c] = float(g[c].mean())
        rows.append(rec)
    C = pd.DataFrame(rows); K.save_csv(C, K.RESULTS / 'sens_conditions.csv')

    R = []
    for (sec, surf, fam, sig), g in C.groupby(['section', 'surface', 'family', 'sigma']):
        g = g.copy(); g['config'] = np.where(g.section == 'S1', g.size_mm.map(lambda x: f'{x:g}'), g.layout)
        main = g.sort_values(['Ctau_d6_t0.15', 'config'], ascending=[False, True]).config.tolist()
        for d, t in COMBOS:
            c = f'Ctau_d{d:g}_t{t:g}'; order = g.sort_values([c, 'config'], ascending=[False, True]).config.tolist()
            r0 = np.array([main.index(x) for x in g.config], float); r1 = np.array([order.index(x) for x in g.config], float)
            R.append(dict(section=sec, surface=surf, family=fam, sigma=sig, d_mm=d, tau_mm=t, top=order[0], top_main=main[0], top_changed=order[0] != main[0],
                          n_rank_changes=int((r0 != r1).sum()), spearman_vs_main=(float(np.corrcoef(r0, r1)[0, 1]) if len(g) > 2 else None), order=';'.join(order)))
    R = pd.DataFrame(R); K.save_csv(R, K.RESULTS / 'sens_rankings.csv')

    Pi = Pt.set_index(ST.KEYS + ['phase_label']).sort_index()
    T = []
    series = [('S1', 'point', [0.5, 1.0, 1.5], 'size_mm'), ('S1', 'line', list(K.LINE_WIDTHS), 'size_mm'), ('S2', 'point', ST.S2_P_PAR, 'layout'), ('S2', 'point', ST.S2_P_PERP, 'layout'),
              ('S2', 'line', [f's{p:g}_phi0' for p in K.S2_PITCHES], 'layout'), ('S2', 'line', [f's{p:g}_phi45' for p in K.S2_PITCHES], 'layout')]
    for sec, fam, ser, col in series:
        for (surf, sig), g in C[(C.section == sec) & (C.family == fam)].groupby(['surface', 'sigma']):
            gi = g.set_index(col)
            for a, b in zip(ser[:-1], ser[1:]):
                if a not in gi.index or b not in gi.index:
                    continue
                ra, rb = gi.loc[a], gi.loc[b]
                Ka = tuple((a if k == col else ra[k]) for k in ST.KEYS)
                Kb = tuple((b if k == col else rb[k]) for k in ST.KEYS)
                A = Pi.loc[Ka]; B = Pi.loc[Kb]
                rec = dict(section=sec, family=fam, surface=surf, sigma=sig, series=('S1 size' if sec == 'S1' else ('p_par' if ser == ST.S2_P_PAR else ('p_perp' if ser == ST.S2_P_PERP else ser[0].split('_')[1]))),
                           a=a, b=b)
                for d, t in COMBOS:
                    c = f'Ctau_d{d:g}_t{t:g}'; e = (B[c].to_numpy(float) - A[c].to_numpy(float))
                    rec[f'd_{c}'] = float(e.mean())
                ref = rec['d_Ctau_d6_t0.15']
                rec['sign_flips'] = int(sum(1 for d, t in COMBOS if np.sign(rec[f'd_Ctau_d{d:g}_t{t:g}']) != np.sign(ref) and abs(rec[f'd_Ctau_d{d:g}_t{t:g}']) > 1e-9 and abs(ref) > 1e-9))
                T.append(rec)
    T = pd.DataFrame(T); K.save_csv(T, K.RESULTS / 'sens_contrasts.csv')
    chk = dict(n_inputs=int(len(S)), check_d6_t015_all=bool(S.check_d6_t015.astype(bool).all()), n_check_failed=int((~S.check_d6_t015.astype(bool)).sum()),
               n_conditions=int(len(C)), top_changes=int(R.top_changed.sum()), n_rankings=int(len(R)), contrasts_with_sign_flip=int((T.sign_flips > 0).sum()), n_contrasts=int(len(T)))
    K.save_json(K.CHECKS / 'sensitivity_check.json', dict(chk, t_local=K.now_local()))
    return chk


if __name__ == '__main__':
    print(json.dumps(K.clean(run()), ensure_ascii=False, indent=1))
