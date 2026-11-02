# Build encoding comparisons and descriptive summary tables.

from __future__ import annotations

import numpy as np
import pandas as pd

import c5_common as K
import c5_gate as G

POS = dict(u='pos_u', v='pos_v', phase_frac='pos_phase_frac')
META = ['family', 'surface', 'layout', 'regime', 'pattern', 'phase_id', 'geometry_phase', 'phase_index', 'pos_u', 'pos_v', 'pos_phase_frac', 'code_parity', 'sigma', 'rep', 'seed',
        'pair_key', 'ref_image_key', 'peak_f32', 'transition_layout']
METRICS = ['glob_Ctau', 'glob_Cvalid', 'loc_Ctau', 'loc_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'glob_surf_p95', 'loc_surf_rmse',
           'loc_surf_p95', 'n_accepted', 'n_valid', 'new_correct', 'new_wrong', 'new_unconf', 'new_n']
FRONT = ['n_bg_candidates', 'n_candidates', 'n_floor', 'guard_readable', 'cls_pre_acc', 'cls_post_abstain', 'cls_sel_known', 'cls_post_assigned', 'cls_true_known', 'image_fallback', 'decision', 'status']
DMET = ['glob_Ctau', 'glob_Cvalid', 'glob_gap', 'loc_Ctau', 'glob_pt_rmse', 'glob_pt_p95', 'c', 'w', 'u', 'acc']
COND = {'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse'}


def _rate(w, d):
    w, d = np.asarray(w, float), np.asarray(d, float)
    return np.where(d > 0, w / np.where(d > 0, d, 1), np.nan)


# Build the analysis table while preserving phase, repetition and pairing keys.
def case_table(T: pd.DataFrame, M: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    S = cfg['structure']; ph = S['phase_col']
    E = T[(T.layer == 'eval') & T.complete].copy()
    Mi = M.drop_duplicates('image_key').set_index('image_key')
    if cfg['profile'] == 's12':
        V = M.rename(columns={'part': 'view'})
        ro = S.get('reference_only', [])
        V = V[~V.apply(lambda r: any(r['view'] == x['part'] and float(r['sigma']) == float(x['sigma']) for x in ro), axis=1)]
        keep = ['image_key', 'view', 'section', 'regime', 'surface', 'layout', 'size_mm', ph, 'sigma', 'rep', S['seed_col'], 'family', 'pattern']
        E = E.drop(columns=[c for c in ('family',) if c in E.columns]).merge(V[keep], on='image_key', how='inner', validate='many_to_many')
        E = E.rename(columns={ph: 'phase_id', S['seed_col']: 'seed'})
    else:
        for c in META:
            src = next((k for k, v in POS.items() if v == c), c)
            if src in Mi.columns:
                E[c] = E.image_key.map(Mi[src])
        E = E.rename(columns={ph: 'phase_id'}) if ph != 'phase_id' else E
        if S.get('rep_source') == 'constant_0':
            E['rep'] = 0
        if 'code_parity' not in E.columns or S.get('parity_rule') == 'all_zero':
            E['code_parity'] = 0
    E['layout'] = E.layout.astype(str)
    E['seed'] = [K.norm_seed(x) for x in E.seed] if 'seed' in E.columns else None
    E['block_parity'] = E.phase_id.map(K.PARITY_OF_PHASE) if S.get('block_layers') == 'formal_N' else np.nan
    pb = S.get('pair_block')
    E['pair_block'] = E[pb].astype(str).agg('|'.join, axis=1) if (pb and len(E)) else ''
    for f in METRICS:
        v = G._val(E, f, cfg)
        E[f] = pd.to_numeric(v, errors='coerce') if v is not None else np.nan
    E['glob_n_valid'], E['loc_n_valid'] = G.denominators(E, cfg)
    dots = (E.family == 'dots').to_numpy(); n = lambda f: pd.to_numeric(G._val(E, f, cfg), errors='coerce') if G._val(E, f, cfg) is not None else pd.Series(np.nan, index=E.index)
    if cfg['profile'] == 's12':
        E['c'], E['w'], E['u'] = n('src_correct'), n('src_wrong'), n('src_unconf')
    else:
        E['c'] = np.where(dots, n('src_correct'), n('b_id_correct')); E['w'] = np.where(dots, n('src_wrong'), n('b_id_wrong')); E['u'] = np.where(dots, n('src_unconf'), n('b_id_unconf'))
    E['acc'] = E.c + E.w + E.u; E['no_output'] = E.n_accepted == 0
    for dom in ('glob', 'loc'):
        E[f'{dom}_gap'] = E[f'{dom}_Cvalid'] - E[f'{dom}_Ctau']
    E['share_main'] = _rate(E.w, E.c + E.w); E['share_lower'] = _rate(E.w, E.acc); E['share_upper'] = _rate(E.w + E.u, E.acc)
    has_ref = ((E.family == 'dots') & (E.arm == 'use')) | ((E.family == 'lines') & E.rule.astype(str).str.match(r'^(R|LB)@'))
    E['share_new'] = np.where(has_ref & (cfg['profile'] == 'c34'), _rate(E.new_wrong, E.new_correct + E.new_wrong), np.nan)
    for g in (K.K4.GATE,) + tuple(K.K4.GATE_SENS):
        sfx = '' if g == K.K4.GATE else f'_{g}'
        E[f'flag_wrong_all{sfx}'] = np.where(np.isnan(E.share_main), np.nan, (E.share_main > g).astype(float))
        E[f'flag_wrong_new{sfx}'] = np.where(np.isnan(E.share_new), np.nan, (E.share_new > g).astype(float))
    E['flag_unconf'] = (E.u > 0).astype(float); E['flag_no_output'] = E.no_output.astype(float)
    E['strict_identity_pass'] = (~E.no_output) & (E.flag_wrong_all == 0) & (E.flag_wrong_new != 1) & (E.flag_unconf == 0)
    for f in FRONT:
        v = G._val(E, f, cfg)
        E[f] = v if v is not None else np.nan
    E['eval_key'] = E['key']
    ident = ['image_key', 'eval_key', 'decode_key', 'rec_identity', 'rec_image_key', 'rec_rule', 'rec_arm', 'rec_kind', 'rule', 'arm', 'view', 'section', 'size_mm', 'block_parity', 'pair_block']
    cols = [c for c in META + ident if c in E.columns] + ['glob_n_valid', 'loc_n_valid', 'c', 'w', 'u', 'acc', 'no_output', 'glob_gap', 'loc_gap', 'share_main', 'share_lower', 'share_upper',
                                                           'share_new'] + [c for c in E.columns if c.startswith('flag_')] + ['strict_identity_pass'] + METRICS + FRONT
    seen = set(); cols = [c for c in cols if not (c in seen or seen.add(c))]
    return E[cols].reset_index(drop=True)


SHARED = ['family', 'surface', 'layout', 'regime', 'phase_id', 'geometry_phase', 'phase_index', 'pos_u', 'pos_v', 'pos_phase_frac', 'block_parity', 'sigma', 'rep', 'seed', 'pair_key', 'pair_block',
          'view', 'section', 'size_mm']
SIDE = ['image_key', 'pattern', 'code_parity', 'rule', 'arm', 'eval_key', 'decode_key', 'n_accepted', 'no_output'] + DMET


def _side(X, cols):
    return X[[c for c in cols if c in X.columns]].copy()


def _merge(a, b, on, validate, name, mode, per_b=None):
    try:
        m = _side(a, list(dict.fromkeys(on + SHARED + SIDE))).merge(_side(b, list(dict.fromkeys(on + SIDE))), on=on, how='outer', suffixes=('_a', '_b'), validate=validate,
                                                                    indicator=True)
    except pd.errors.MergeError as e:
        raise K.Stop('pairing', f'{name}: pairing is not {validate} on {on}: {e}')
    un = m[m._merge != 'both']
    if len(un) and mode == 'formal':
        raise K.Stop('pairing', f'{name}: {len(un)} unmatched rows ({un._merge.value_counts().to_dict()})')
    m = m[m._merge == 'both'].drop(columns='_merge')
    if per_b is not None and len(m):
        cnt = m.groupby('eval_key_b').size()
        if (cnt != per_b).any() and mode == 'formal':
            raise K.Stop('pairing', f'{name}: a plain reference is used by {sorted(set(cnt))} coded images, registered {per_b}')
    return m, len(un)


def _emit(m, pair_type, effect, rows):
    if not len(m):
        return
    out = pd.DataFrame({'pair_type': pair_type, 'effect': effect}, index=m.index)
    for c in SHARED:
        if c in m.columns:
            out[c] = m[c]
    out['pattern'] = m['pattern_a'] if 'pattern_a' in m.columns else m.get('pattern')
    for s in ('a', 'b'):
        for c in ('image_key', 'pattern', 'code_parity', 'rule', 'arm', 'eval_key', 'decode_key', 'n_accepted', 'no_output', 'glob_Ctau'):
            k = f'{c}_{s}' if f'{c}_{s}' in m.columns else c
            if k in m.columns:
                out[f'{s}_{c}'] = m[k]
    for d in DMET:
        a_, b_ = pd.to_numeric(m[f'{d}_a'], errors='coerce'), pd.to_numeric(m[f'{d}_b'], errors='coerce')
        out[f'd_{d}'] = a_ - b_
    out['undefined_a'] = pd.to_numeric(m['glob_pt_rmse_a'], errors='coerce').isna(); out['undefined_b'] = pd.to_numeric(m['glob_pt_rmse_b'], errors='coerce').isna()
    rows.append(out)


def pairs(C: pd.DataFrame, cfg: dict, mode: str) -> tuple[pd.DataFrame, dict]:
    P = cfg['pairs']; rows = []; info = {}; blk = cfg['structure'].get('pair_block') or []
    if cfg['profile'] == 's12':
        for x, y in P['front_ends']:
            a, b = C[C.rule == x], C[C.rule == y]
            m, un = _merge(a, b, ['image_key', 'view'], 'one_to_one', f'{x}-{y}', mode); _emit(m, 'E3', f'{x}-{y}', rows); info[f'{x}-{y}'] = dict(pairs=len(m), expected=len(a), unmatched=un)
    else:
        D, L = C[C.family == 'dots'], C[C.family == 'lines']
        levels = [p for p in cfg['structure']['patterns'] if p != 'plain']
        bon = [c for c in blk if c != 'family'] + ['seed', 'rule']
        for rule in P['dots_rules']:
            a, b = D[(D.rule == rule) & (D.arm == 'use')], D[(D.rule == rule) & (D.arm == 'ignore')]
            m, un = _merge(a, b, ['image_key', 'rule'], 'one_to_one', f'E1 {rule}', mode); _emit(m, 'E1', f'E1 {rule} use-ignore', rows)
            info[f'E1 {rule}'] = dict(pairs=len(m), expected=len(a), unmatched=un)
            a, b = D[(D.rule == rule) & (D.arm == 'use')], D[(D.rule == rule) & (D.arm == 'single')]
            m, un = _merge(a, b, bon, 'many_to_one', f'E2 {rule}', mode, per_b=len(levels))
            _emit(m, 'E2', f'E2 {rule} coded-plain (scheme difference incl. size-dependent front end)', rows); info[f'E2 {rule}'] = dict(pairs=len(m), expected=len(a), unmatched=un)
        for x, y in P['dots_front_ends']:
            for arm in ('use', 'ignore', 'single'):
                a, b = D[(D.rule == x) & (D.arm == arm)], D[(D.rule == y) & (D.arm == arm)]
                m, un = _merge(a.drop(columns='rule').assign(rule_x=x), b.drop(columns='rule').assign(rule_y=y), ['image_key', 'arm'], 'one_to_one', f'E3 {x}-{y} {arm}', mode)
                m['rule_a'], m['rule_b'] = x, y
                _emit(m, 'E3', f'E3 {x}-{y} {arm}', rows); info[f'E3 {x}-{y} {arm}'] = dict(pairs=len(m), expected=len(a), unmatched=un)
        ref = P['lines_reference']
        for rule in P['lines_coded_rules']:
            a, b = L[L.rule == rule], L[L.rule == ref]
            m, un = _merge(a.drop(columns='rule'), b.drop(columns='rule'), ['image_key'], 'one_to_one', f'E1 {rule}', mode); m['rule_a'], m['rule_b'] = rule, ref
            _emit(m, 'E1L', f"E1 {rule.split('@')[0]}-{ref}", rows); info[f'E1L {rule}'] = dict(pairs=len(m), expected=len(a), unmatched=un)
        r_, p_ = P['lines_scheme']
        a, b = L[L.rule == r_], L[L.rule == p_]
        m, un = _merge(a.drop(columns='rule'), b.drop(columns='rule'), [c for c in blk if c != 'family'] + ['seed'], 'many_to_one', 'E2L', mode, per_b=len(levels))
        m['rule_a'], m['rule_b'] = r_, p_
        _emit(m, 'E2L', 'E3 lines LC(R)-P4 whole scheme', rows); info['E2L'] = dict(pairs=len(m), expected=len(a), unmatched=un)
    X = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return X, info


def phase_summary(X: pd.DataFrame, keys: list, values: list, cond: set) -> pd.DataFrame:

    keys = [k for k in keys if k in X.columns]
    out = []
    for k, g in X.groupby(keys, dropna=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,))); gp = g.groupby('phase_id')
        r.update(n_rows=len(g), n_phases=int(g.phase_id.nunique()), reps_per_phase=';'.join(str(int(x)) for x in gp.size().values))
        for v in values:
            x = pd.to_numeric(g[v], errors='coerce')
            if v not in cond and x.isna().any():
                r[f'{v}_mean'] = np.nan; r[f'{v}_problem'] = 'undefined value in an unconditional metric'; continue
            pm = x.groupby(g.phase_id).mean()
            r[f'{v}_phase'] = ';'.join('nan' if pd.isna(z) else f'{z:.6f}' for z in pm.values)
            r[f'{v}_mean'] = float(pm.mean()) if pm.notna().all() else (float(pm.dropna().mean()) if v in cond and pm.notna().any() else np.nan)
            r[f'{v}_min'] = float(pm.min()) if pm.notna().any() else np.nan; r[f'{v}_max'] = float(pm.max()) if pm.notna().any() else np.nan
            if v in cond:
                r[f'{v}_n_defined'] = int(x.notna().sum()); r[f'{v}_phases_defined'] = int(pm.notna().sum())
        out.append(r)
    return pd.DataFrame(out)


def block_bootstrap(V: np.ndarray, phases: list, n: int, seed: int) -> np.ndarray:

    lay = {p: [c for c in phases if K.PARITY_OF_PHASE[c] == p] for p in (0, 1)}
    rng = np.random.default_rng(seed); out = np.empty((V.shape[0], n))
    for b in range(n):
        idx = [phases.index(c) for p in (0, 1) for c in rng.choice(lay[p], size=len(lay[p]), replace=True)]
        out[:, b] = np.nanmean(V[:, idx], axis=1)
    return out


def bootstrap_pairs(X: pd.DataFrame, keys: list, cfg: dict) -> pd.DataFrame:

    bs = cfg['bootstrap']; phases = cfg['structure']['phases']['*']; out = []
    keys = [k for k in keys if k in X.columns]
    for fam, h in X.groupby('family'):
        rows, lab = [], []
        for k, g in h.groupby(keys, dropna=False):
            pm = g.groupby('phase_id')[['a_glob_Ctau', 'b_glob_Ctau', 'd_glob_Ctau']].mean().reindex(phases)
            for c, nm in (('a_glob_Ctau', 'a'), ('b_glob_Ctau', 'b'), ('d_glob_Ctau', 'd')):
                rows.append(pd.to_numeric(pm[c], errors='coerce').to_numpy(float)); lab.append((k, nm))
        V = np.vstack(rows); R = block_bootstrap(V, phases, int(bs['n']), int(bs['seed']))
        for i in range(0, len(lab), 3):
            k = lab[i][0]; base = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
            shared_ok = bool(np.allclose(R[i] - R[i + 1], R[i + 2], atol=1e-12, equal_nan=True))
            out.append(dict(base, mean=float(np.nanmean(V[i + 2])), lo=float(np.nanpercentile(R[i + 2], 2.5)), hi=float(np.nanpercentile(R[i + 2], 97.5)),
                            a_lo=float(np.nanpercentile(R[i], 2.5)), a_hi=float(np.nanpercentile(R[i], 97.5)), b_lo=float(np.nanpercentile(R[i + 1], 2.5)),
                            b_hi=float(np.nanpercentile(R[i + 1], 97.5)), shared_index_d_equals_a_minus_b=shared_ok, n=int(bs['n']), kind='descriptive'))
    return pd.DataFrame(out)
