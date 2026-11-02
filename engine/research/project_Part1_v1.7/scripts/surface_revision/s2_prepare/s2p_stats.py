# Summarize spacing-layout performance and paired differences.

from __future__ import annotations

import numpy as np
import pandas as pd

import s2p_common as K

ZERO_IF_NO_OUTPUT = ('img_recovery', 'loc_recovery', 'glob_recovery', 'loc_cov_interp', 'glob_cov_interp', 'loc_Cvalid', 'glob_Cvalid', 'loc_Ctau', 'glob_Ctau')
METRICS = ('img_recovery', 'mismatch_rate', 'unconf_share', 'loc_recovery', 'loc_pt_rmse', 'loc_pt_mae', 'loc_pt_p95', 'loc_gross_rate', 'loc_surf_rmse', 'loc_cov_interp', 'loc_Cvalid',
           'loc_Ctau', 'glob_recovery', 'glob_pt_rmse', 'glob_pt_mae', 'glob_pt_p95', 'glob_gross_rate', 'glob_surf_rmse', 'glob_cov_interp', 'glob_Cvalid', 'glob_Ctau')
CELL = ['surface', 'family', 'regime', 'layout', 'sigma']
OK_STATES = ('ok', 'no_output')
DOT_ADJ = {'dots_p_parallel': ['R6x4', 'R7.35x4', 'R8x4', 'R10x4', 'R12x4'], 'dots_p_perp': ['R8x2.5', 'R8x3', 'R8x4', 'R8x6', 'R8x8']}
LINE_PITCH = ['4.45', '5', '5.6', '6.3', '8', '10', '12']


def validate(plan: pd.DataFrame, rec: pd.DataFrame) -> dict:
    P = plan.drop_duplicates('decode_key')
    status = {}
    groups = {k: g for k, g in rec.groupby('decode_key')} if len(rec) else {}
    for _, p in P.iterrows():
        g = groups.get(p.decode_key)
        if g is None:
            status[p.decode_key] = 'missing'; continue
        if len(g) > 1:
            status[p.decode_key] = 'duplicate'; continue
        r = g.iloc[0]
        if not bool(r['complete']) or r['state'] not in OK_STATES:
            status[p.decode_key] = 'technical'
        elif K.norm_seed(r['noise_seed']) != K.norm_seed(p.noise_seed):
            status[p.decode_key] = 'seed_mismatch'
        elif int(float(r['rep'])) != int(float(p.rep)):
            status[p.decode_key] = 'rep_mismatch'
        elif str(r['phase_label']) != str(p.phase_label):
            status[p.decode_key] = 'phase_mismatch'
        else:
            status[p.decode_key] = r['state']
    extra = rec[~rec.decode_key.isin(set(P.decode_key))] if len(rec) else rec
    blocked_cells, global_block = set(), False
    for _, r in extra.iterrows():
        if all(c in r and str(r[c]) != '' for c in CELL):
            blocked_cells.add(tuple(_cellval(r[c]) for c in CELL))
        else:
            global_block = True
    bad = {k: v for k, v in status.items() if v not in OK_STATES}
    return dict(status=status, extra=extra.decode_key.tolist() if len(extra) else [], blocked_cells=blocked_cells, global_block=global_block,
                n_planned=len(P), n_ok=sum(v == 'ok' for v in status.values()), n_no_output=sum(v == 'no_output' for v in status.values()), problems=bad,
                formal_ok=(not bad and not len(extra)))


def _cellval(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def frame(plan: pd.DataFrame, rec: pd.DataFrame, val: dict) -> pd.DataFrame:
    fr = plan.copy()
    fr['status'] = [val['status'].get(k, 'missing') for k in fr.decode_key]
    r1 = rec.drop_duplicates('decode_key', keep=False).set_index('decode_key') if len(rec) else pd.DataFrame()
    for m in METRICS:
        fr[m] = [(r1.at[k, m] if (st in OK_STATES and k in r1.index and m in r1.columns) else np.nan) for k, st in zip(fr.decode_key, fr.status)]
    return fr


def _cell_key(g):
    return tuple(_cellval(g[c].iloc[0]) for c in CELL)


def _phase_values(g: pd.DataFrame, m: str) -> dict:

    out = {}
    for ph, x in g.drop_duplicates('decode_key').groupby('phase_label'):
        v = pd.to_numeric(x[m], errors='coerce').astype(float).copy()
        if m in ZERO_IF_NO_OUTPUT:
            v[x.status.values == 'no_output'] = 0.0
        out[ph] = float(v.mean()) if not v.isna().any() else float('nan')
    return out


def cell_state(fr: pd.DataFrame, val: dict, key) -> tuple[str, dict]:
    g = fr[np.logical_and.reduce([fr[c].map(_cellval) == v for c, v in zip(CELL, key)])]
    if val['global_block']:
        return 'blocked_extra_records', {}
    if key in val['blocked_cells']:
        return 'blocked_extra_record', {}
    d = g.drop_duplicates('decode_key')
    counts = d.status.value_counts().to_dict()
    return ('complete' if set(counts) <= set(OK_STATES) else 'incomplete'), counts


def conditions(fr: pd.DataFrame, val: dict) -> pd.DataFrame:
    rows = []
    for key, g in fr.groupby(CELL):
        key = tuple(_cellval(k) for k in key)
        st, counts = cell_state(fr, val, key)
        d = g.drop_duplicates('decode_key')
        base = dict(zip(CELL, key), state=st, n_planned_decodes=int(len(d)), n_phases_total=int(d.phase_label.nunique()), status_counts=counts,
                    n_no_output=int((d.status == 'no_output').sum()))
        for m in METRICS:
            if st != 'complete':
                rows.append(dict(base, metric=m, mean=None, median=None, min=None, max=None, n_phases_defined=None)); continue
            pv = _phase_values(g, m); vals = [v for v in pv.values() if np.isfinite(v)]
            rows.append(dict(base, metric=m, mean=(float(np.mean(vals)) if vals else None), median=(float(np.median(vals)) if vals else None), min=(float(min(vals)) if vals else None),
                             max=(float(max(vals)) if vals else None), n_phases_defined=len(vals)))
    return pd.DataFrame(rows)


def contrast_pairs() -> list[tuple]:
    out = []
    for s, lst in DOT_ADJ.items():
        out += [(f'{s} adjacent', 'point', a, b) for a, b in zip(lst[:-1], lst[1:])]
    out.append(('dots R vs T (8 x 4)', 'point', 'R8x4', 'T8x4'))
    out += [('dots vs R8x4', 'point', 'R8x4', L) for L in ('R6x4', 'R7.35x4', 'R10x4', 'R12x4', 'R8x2.5', 'R8x3', 'R8x6', 'R8x8', 'T8x4')]
    for phi in ('0', '45'):
        out += [(f'lines pitch adjacent phi {phi}', 'line', f's{a}_phi{phi}', f's{b}_phi{phi}') for a, b in zip(LINE_PITCH[:-1], LINE_PITCH[1:])]
    out += [('lines phi 45 vs 0', 'line', f's{s}_phi0', f's{s}_phi45') for s in LINE_PITCH]
    return out


def _pair(ga, gb, ka, kb, fr, val, m, need_same_seed=True, seed=2026101202, n_draw=10000):
    sa, _ = cell_state(fr, val, ka); sb, _ = cell_state(fr, val, kb)
    if sa != 'complete' or sb != 'complete':
        return dict(state='incomplete', reason=f'arm states {sa} / {sb}')
    da, db = ga.drop_duplicates('decode_key'), gb.drop_duplicates('decode_key')
    la = sorted(zip(da.phase_label, da.rep)); lb = sorted(zip(db.phase_label, db.rep))
    if la != lb:
        return dict(state='incomplete', reason='arms planned on different phases / realisations')
    if need_same_seed:
        sa_ = {(p, r): K.norm_seed(s) for p, r, s in zip(da.phase_label, da.rep, da.noise_seed)}; sb_ = {(p, r): K.norm_seed(s) for p, r, s in zip(db.phase_label, db.rep, db.noise_seed)}
        if sa_ != sb_:
            return dict(state='incomplete', reason='noise realisations not paired (seeds differ)')
    va, vb = _phase_values(ga, m), _phase_values(gb, m)
    ph = [p for p in sorted(va) if np.isfinite(va[p]) and np.isfinite(vb[p])]
    if not ph:
        return dict(state='undefined', reason='metric undefined in every phase (no output)', n_phases_total=len(va))
    e = np.array([vb[p] - va[p] for p in ph])
    idx = np.random.Generator(np.random.PCG64(seed)).integers(0, len(e), size=(n_draw, len(e))); bs = e[idx].mean(axis=1)
    return dict(state=('ok' if len(ph) == len(va) else 'ok_defined_subset'), n_phases_defined=len(ph), n_phases_total=len(va), mean=float(e.mean()), median=float(np.median(e)),
                n_pos=int((e > 0).sum()), n_neg=int((e < 0).sum()), ci_5_95=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))])


def contrasts(fr: pd.DataFrame, val: dict, metrics=('glob_Ctau', 'loc_Ctau', 'glob_pt_rmse', 'loc_pt_rmse', 'img_recovery', 'mismatch_rate')) -> pd.DataFrame:
    rows = []
    for (surf, reg, sig), g in fr.groupby(['surface', 'regime', 'sigma']):
        for name, fam, A, B in contrast_pairs():
            ga = g[(g.family == fam) & (g.layout == A)]; gb = g[(g.family == fam) & (g.layout == B)]
            if ga.empty and gb.empty:
                continue
            ka = (surf, fam, reg, A, _cellval(sig)); kb = (surf, fam, reg, B, _cellval(sig))
            for m in metrics:
                if ga.empty or gb.empty:
                    rows.append(dict(surface=surf, regime=reg, sigma=sig, contrast=name, arm_a=A, arm_b=B, metric=m, state='incomplete', reason='an arm is not in the plan subset')); continue
                rows.append(dict(surface=surf, regime=reg, sigma=sig, contrast=name, arm_a=A, arm_b=B, metric=m, noise_paired=True, **_pair(ga, gb, ka, kb, fr, val, m, True)))
    return pd.DataFrame(rows)


def surface_contrasts(fr: pd.DataFrame, val: dict, metric='glob_Ctau') -> pd.DataFrame:
    rows = []
    main = fr[fr.surface.isin(K.MAIN)]
    for (fam, reg, lay, sig), g in main.groupby(['family', 'regime', 'layout', 'sigma']):
        for a, b in (('M10_curv', 'M24_curv'), ('M10_curv', 'B_curv'), ('M24_curv', 'B_curv')):
            ga, gb = g[g.surface == a], g[g.surface == b]
            base = dict(family=fam, regime=reg, layout=lay, sigma=sig, pair=f'{b} - {a}', metric=metric,
                        pairing=('noise paired (surface-independent seeds)' if fam == 'point' else 'phase-level design pairing (surface-dependent seeds)'))
            if ga.empty or gb.empty:
                rows.append(dict(base, state='incomplete', reason='a surface is not in the plan subset')); continue
            rows.append(dict(base, **_pair(ga, gb, (a, fam, reg, lay, _cellval(sig)), (b, fam, reg, lay, _cellval(sig)), fr, val, metric, need_same_seed=(fam == 'point'))))
    return pd.DataFrame(rows)


def analyse(plan: pd.DataFrame, rec: pd.DataFrame) -> dict:
    val = validate(plan, rec); fr = frame(plan, rec, val)
    return dict(validation=val, frame=fr, conditions=conditions(fr, val), contrasts=contrasts(fr, val), surface_contrasts=surface_contrasts(fr, val))
