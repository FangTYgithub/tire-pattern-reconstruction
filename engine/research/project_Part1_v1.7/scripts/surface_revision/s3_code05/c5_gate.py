# Reject incomplete or inconsistent encoding result tables.

from __future__ import annotations

import numpy as np
import pandas as pd

import c5_common as K

ORDER = ('structure', 'records', 'columns', 'semantics')
COUNT_C34 = ('n_accepted', 'n_valid', 'b_id_correct', 'b_id_wrong', 'b_id_unconf', 'new_n', 'new_correct', 'new_wrong', 'new_unconf')
COV = ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid')
PT = ('glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95')
PROFILES = dict(
    c34=dict(eval=dict(all=('state',) + COUNT_C34 + COV + PT + ('glob_surf_rmse',), dots=('src_acc', 'src_correct', 'src_wrong', 'src_unconf')),
             diag=('n_accepted', 'n_valid', 'b_id_correct', 'b_id_wrong', 'b_id_unconf') + COV, main=dict(dots='src', lines='b_id')),
    s12=dict(eval=dict(all=('state', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid') + COV + PT + ('glob_surf_rmse', 'glob_surf_p95', 'loc_surf_rmse', 'loc_surf_p95',
                                                                                                         'id_correct', 'id_wrong', 'id_unconf'),
                       dots=('src_acc', 'src_correct', 'src_wrong', 'src_unconf')), diag=(), main=dict(dots='src')))
SEMANTIC_RULES = [
    'S1 counts are finite non-negative integers', 'S2 main identity: correct + wrong + unconfirmable = accepted (dots: source identity, lines: line identity)',
    'S3 valid <= accepted; domain valid <= valid', 'S4 C_valid, C_tau finite in [0, 1]; C_tau <= C_valid', 'S5 point errors defined (finite, >= 0) iff the domain has valid points',
    'S6 surface error defined when that domain has supported nodes (C_valid > 0); defined only with >= 3 valid points (no interpolation below)',
    'S7 fewer than 3 valid points -> C_valid = C_tau = 0 in both domains', 'S8 no output (0 accepted) -> 0 valid, coverage 0, all errors undefined',
    'S9 new = new correct + new wrong + new unconfirmable', 'S10 diagnostics: counts, coverage ranges, C_tau <= C_valid',
    'S11 declared supplementary table equals the record on the overlapping metrics; derived domain counts equal its counts']


def col(T: pd.DataFrame, name: str) -> pd.Series:
    return T[name] if name in T.columns else pd.Series(np.nan, index=T.index, dtype=float)


def num(T: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(col(T, name), errors='coerce').astype(float)


def _p(cat, check, mask_or_n, keys=None, detail=None) -> dict | None:
    n = int(mask_or_n.sum()) if hasattr(mask_or_n, 'sum') else int(mask_or_n)
    if n == 0:
        return None
    ks = list(keys[mask_or_n].astype(str).head(8)) if keys is not None and hasattr(mask_or_n, 'sum') else (keys or [])
    return dict(category=cat, check=check, n=n, detail=detail, keys=ks)


def _reg(d: dict, gname: str, fam: str):

    for k in (gname, fam, '*'):
        if k in d:
            return d[k]
    raise K.Stop('config', f'no registered structure entry for group {gname} / family {fam}')


# Check phase, repetition, seed and geometry identities in the case matrix.
def structure(M: pd.DataFrame, S: dict) -> list:
    probs = []; M = M.copy(); ph = S['phase_col']
    need = ['image_key', 'family', 'surface', 'layout', 'pattern', ph, 'sigma'] + ([S['seed_col']] if S.get('seed_col') else []) + \
           (['rep'] if S.get('rep_source', 'column') == 'column' else []) + ([S['pair_key_col']] if S.get('pair_key_col') else []) + list(S.get('geometry_cols', []))
    miss = [c for c in need if c not in M.columns]
    if miss:
        return [dict(category='structure', check='manifest columns', n=len(miss), detail=miss, keys=[])]
    if S.get('rep_source') == 'constant_0':
        M['rep'] = 0
    bad_seed = []
    seeds = []
    for x in (M[S['seed_col']] if S.get('seed_col') else [None] * len(M)):
        try:
            seeds.append(K.norm_seed(x))
        except K.Stop:
            seeds.append('INVALID'); bad_seed.append(x)
    M['_seed'] = pd.Series(seeds, index=M.index, dtype=object)
    if bad_seed:
        probs.append(dict(category='structure', check='seed not an integer', n=len(bad_seed), detail=[str(x) for x in bad_seed[:5]], keys=[]))
    Mall = M.drop_duplicates('image_key').set_index('image_key')
    groups = [(g, h) for g, h in M.groupby(S['group_col'])] if S.get('group_col') else [('all', M)]
    for gname, G in groups:
        strat = [c for c in S['stratum'] if c in G.columns]
        probs.append(_p('structure', f'[{gname}] duplicate image keys', G.image_key.duplicated(keep=False), G.image_key))
        for k, h in G.groupby(strat):
            reg = sorted(_reg(S['phases'], gname, h.family.iloc[0])); got = sorted(h[ph].astype(str).unique())
            if got != reg:
                probs.append(dict(category='structure', check=f'[{gname}] phase set', n=1, detail=dict(stratum=dict(zip(strat, map(str, k))), got=got, registered=reg), keys=[]))
        for k, h in G.groupby(strat + [ph]):
            reps = _reg(S['reps'], gname, h.family.iloc[0]); sg = float(h.sigma.iloc[0])
            want = sorted(reps['zero'] if sg == 0 else reps['pos']); got = sorted(int(x) for x in h.rep)
            cell = dict(zip(strat + ['phase'], map(str, k)))
            if got != want:
                probs.append(dict(category='structure', check=f'[{gname}] repetition labels (exact registered set)', n=1, detail=dict(cell=cell, got=got, registered=want), keys=list(h.image_key.head(4))))
            if S.get('seed_col'):
                sd = list(h._seed)
                if sg == 0 and any(not K.is_null(x) for x in sd):
                    probs.append(dict(category='structure', check=f'[{gname}] sigma-0 realisation must have no seed', n=1, detail=dict(cell=cell, seeds=sd), keys=list(h.image_key.head(4))))
                if sg > 0 and (any(K.is_null(x) or x == 'INVALID' for x in sd) or len(set(sd)) != len(sd)):
                    probs.append(dict(category='structure', check=f'[{gname}] repetitions of a cell need distinct integer seeds', n=1, detail=dict(cell=cell, reps=got, seeds=sd),
                                      keys=list(h.image_key.head(4))))
        if S.get('pair_block'):
            pats = sorted(S['patterns'])
            for k, h in G.groupby(S['pair_block']):
                if sorted(h.pattern) != pats or len({('none' if K.is_null(x) else x) for x in h._seed}) != 1:
                    probs.append(dict(category='structure', check=f'[{gname}] pair block: registered patterns, one common seed', n=1,
                                      detail=dict(block=dict(zip(S['pair_block'], map(str, k))), patterns=sorted(h.pattern), seeds=list(h._seed)), keys=list(h.image_key.head(4))))
            if S.get('pair_key_col'):
                pk = G.groupby(S['pair_key_col']).apply(lambda h: h[S['pair_block']].astype(str).drop_duplicates().shape[0])
                bk = G.groupby(S['pair_block'])[S['pair_key_col']].nunique()
                if (pk != 1).any() or (bk != 1).any():
                    probs.append(dict(category='structure', check=f'[{gname}] pair key <-> pair block one to one', n=int((pk != 1).sum() + (bk != 1).sum()), detail=None,
                                      keys=list(map(str, pk[pk != 1].index[:4]))))
        if S.get('geometry_cols'):
            gc = S['geometry_cols']; X = G.assign(**{c: G[c].map(lambda x: '' if K.is_null(x) else repr(float(x))) for c in gc}).drop_duplicates(['family', 'layout', ph] + gc)
            a = X.groupby(['family', 'layout', ph]).size(); b = X.groupby(['family', 'layout'] + gc).size()
            if (a > 1).any() or (b > 1).any():
                probs.append(dict(category='structure', check=f'[{gname}] phase label <-> geometric position one to one', n=int((a > 1).sum() + (b > 1).sum()),
                                  detail=dict(labels=[list(map(str, x)) for x in a[a > 1].index[:4]], positions=[list(map(str, x)) for x in b[b > 1].index[:4]]), keys=[]))
        if S.get('parity_rule') == 'formal_N':
            exp = pd.Series(np.where(G.pattern == 'plain', 0, G[ph].map(K.PARITY_OF_PHASE)), index=G.index)
            probs.append(_p('structure', f'[{gname}] code parity follows the registered phase map', pd.to_numeric(col(G, 'code_parity'), errors='coerce').ne(exp), G.image_key))
            probs.append(_p('structure', f'[{gname}] geometry_phase = phase label', col(G, 'geometry_phase').astype(str).ne(G[ph].astype(str)), G.image_key))
        if S.get('ref_col'):
            rc = S['ref_col']; same = S.get('ref_same', ['surface', 'layout', 'pattern', ph]); bad = []
            for r in G[G.family == 'dots'].to_dict('records'):
                rk = r[rc]
                if K.is_null(rk) or rk not in Mall.index:
                    bad.append(r['image_key']); continue
                q = Mall.loc[rk]
                if float(q['sigma']) != 0 or any(str(q[c]) != str(r[c]) for c in same):
                    bad.append(r['image_key'])
            if bad:
                probs.append(dict(category='structure', check=f'[{gname}] sigma-0 reference of every dot image', n=len(bad), detail=None, keys=bad[:8]))
    return [p for p in probs if p]


STATUS_CAT = dict(absent='missing', unreadable='missing', not_complete='missing', rule_row_absent='missing', identity_mismatch='identity',
                  npz_hash='hash', source_npz_hash='hash', source_json_hash='hash', source_json_unregistered='hash', source_hash='hash', duplicate_rule_rows='duplicate')


def records(T: pd.DataFrame, cfg: dict) -> list:
    probs = []
    for st, h in T[~T.complete].groupby('status'):
        probs.append(dict(category='records', check=f'record {STATUS_CAT.get(st, st)} ({st})', n=len(h), detail=h.groupby('layer').size().to_dict(), keys=list(h.key.head(8))))
    c = T[T.complete]
    if len(c):
        idm = col(c, 'rec_image_key').ne(c.image_key) | col(c, 'rec_rule').ne(c.rule)
        if cfg['profile'] == 'c34':
            idm |= (c.layer != 'decode') & col(c, 'rec_arm').fillna('').ne(c.arm)
            idm |= (c.layer == 'diag') & col(c, 'rec_reference').fillna('').ne(c.reference)
        probs.append(_p('records', 'record identity (read from the record) differs from the slot', idm, c.key))
        if cfg['profile'] == 'c34':
            probs.append(_p('records', 'version link: decode key named by the record differs from the expected decode version', (c.layer != 'decode') & col(c, 'rec_decode_key').ne(c.decode_key), c.key))
        d = col(c, 'rec_identity').duplicated(keep=False) & col(c, 'rec_identity').notna()
        probs.append(_p('records', 'one record serves two slots', d, c.key))
    return [p for p in probs if p]


def _val(T, f, cfg):

    if f'r_{f}' in T.columns:
        return T[f'r_{f}']
    if cfg['denominators'] == 'supplementary+npz' and f's_{f}' in T.columns:
        return T[f's_{f}']
    return None


def denominators(T: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series]:
    if cfg['denominators'] in ('npz', 'supplementary+npz'):
        return num(T, 'd_glob_n_valid'), num(T, 'd_loc_n_valid')
    return num(T, 'r_glob_n_valid'), num(T, 'r_loc_n_valid')


def columns(T: pd.DataFrame, cfg: dict) -> list:
    probs = []; P = PROFILES[cfg['profile']]
    E = T[(T.layer == 'eval') & T.complete]
    for fam, h in E.groupby('family'):
        req = list(P['eval']['all']) + list(P['eval'].get(fam, ()))
        rk = col(h, 'row_keys').fillna('').astype(str).str.split('|').map(set)
        for f in req:
            if cfg['denominators'] == 'supplementary+npz' and f'r_{f}' not in h.columns and f's_{f}' in h.columns:
                continue
            miss = ~rk.map(lambda x, f=f: f in x)
            probs.append(_p('columns', f'required column absent in the record: {f} ({fam} evaluation records)', miss, h.key))
        absent = [f for f in req if _val(h, f, cfg) is None]
        for f in req:
            v = _val(h, f, cfg)
            if v is None:
                continue
            if f in COUNT_C34 + ('src_acc', 'src_correct', 'src_wrong', 'src_unconf', 'glob_n_valid', 'loc_n_valid', 'id_correct', 'id_wrong', 'id_unconf') + COV:
                x = pd.to_numeric(v, errors='coerce')
                probs.append(_p('columns', f'required value missing or non-numeric: {f} ({fam})', x.isna(), h.key))
            elif f == 'state':
                probs.append(_p('columns', f'required value missing: state ({fam})', v.isna(), h.key))
        g, l_ = denominators(h, cfg)
        probs.append(_p('columns', f'domain denominators missing ({fam}; source {cfg["denominators"]})', g.isna() | l_.isna(), h.key))
    D = T[(T.layer == 'diag') & T.complete]
    if len(D) and P['diag']:
        rk = col(D, 'row_keys').fillna('').astype(str).str.split('|').map(set)
        for f in P['diag']:
            probs.append(_p('columns', f'required column absent in the record: {f} (diagnostic records)', ~rk.map(lambda x, f=f: f in x), D.key))
    return [p for p in probs if p]


def _intlike(x: pd.Series) -> pd.Series:
    return x.notna() & np.isfinite(x) & (x >= 0) & (np.abs(x - np.round(x)) < 1e-9)


# Check metric ranges and consistency between accepted, valid and identified counts.
def semantics(T: pd.DataFrame, cfg: dict) -> list:
    probs = []; P = PROFILES[cfg['profile']]
    E = T[(T.layer == 'eval') & T.complete].copy()
    v = lambda h, f: pd.to_numeric(_val(h, f, cfg), errors='coerce').astype(float) if _val(h, f, cfg) is not None else pd.Series(np.nan, index=h.index)
    for fam, h in E.groupby('family'):
        k = h.key; acc = v(h, 'n_accepted'); nv = v(h, 'n_valid'); g, l_ = denominators(h, cfg)
        main = P['main'][fam]
        if main == 'src':
            c, w, u, a2 = v(h, 'src_correct'), v(h, 'src_wrong'), v(h, 'src_unconf'), v(h, 'src_acc')
        elif cfg['profile'] == 's12':
            c, w, u, a2 = v(h, 'id_correct'), v(h, 'id_wrong'), v(h, 'id_unconf'), acc
        else:
            c, w, u, a2 = v(h, 'b_id_correct'), v(h, 'b_id_wrong'), v(h, 'b_id_unconf'), acc
        cnt = dict(n_accepted=acc, n_valid=nv, glob_n_valid=g, loc_n_valid=l_, correct=c, wrong=w, unconf=u, main_accepted=a2)
        if cfg['profile'] == 'c34':
            cnt.update({f: v(h, f) for f in ('new_n', 'new_correct', 'new_wrong', 'new_unconf', 'b_id_correct', 'b_id_wrong', 'b_id_unconf')})
        for name, x in cnt.items():
            probs.append(_p('semantics', f'S1 count not a non-negative integer: {name} ({fam})', ~_intlike(x), k))
        probs.append(_p('semantics', f'S2 correct + wrong + unconfirmable != accepted ({fam}, {main} identity)', (c + w + u).ne(a2) | a2.ne(acc), k))
        probs.append(_p('semantics', f'S3 valid > accepted or domain valid > valid ({fam})', (nv > acc) | (g > nv) | (l_ > nv), k))
        for dom, n_dom in (('glob', g), ('loc', l_)):
            cv, ct = v(h, f'{dom}_Cvalid'), v(h, f'{dom}_Ctau')
            probs.append(_p('semantics', f'S4 {dom} C_valid / C_tau not finite in [0, 1] ({fam})', ~(np.isfinite(cv) & np.isfinite(ct) & (cv >= 0) & (cv <= 1) & (ct >= 0) & (ct <= 1)), k))
            probs.append(_p('semantics', f'S4 {dom} C_tau > C_valid ({fam})', ct > cv + K.TOL, k))
            for m in ('pt_rmse', 'pt_p95'):
                e = v(h, f'{dom}_{m}')
                probs.append(_p('semantics', f'S5 {dom}_{m} missing although the domain has valid points ({fam})', (n_dom > 0) & ~(np.isfinite(e) & (e >= 0)), k))
                probs.append(_p('semantics', f'S5 {dom}_{m} present although the domain has no valid point ({fam})', (n_dom == 0) & e.notna(), k))
            s = _val(h, f'{dom}_surf_rmse', cfg)
            if s is not None:
                s = pd.to_numeric(s, errors='coerce').astype(float)
                probs.append(_p('semantics', f'S6 {dom}_surf_rmse missing although supported nodes exist ({fam})', (cv > 0) & ~np.isfinite(s), k))
                probs.append(_p('semantics', f'S6 {dom}_surf_rmse defined with fewer than 3 valid points ({fam})', np.isfinite(s) & (nv < 3), k))
            probs.append(_p('semantics', f'S7 {dom} coverage > 0 with fewer than 3 valid points ({fam})', (nv < 3) & ((cv > 0) | (ct > 0)), k))
        no = acc == 0
        errs = [v(h, f) for f in PT] + [pd.to_numeric(_val(h, f, cfg), errors='coerce') for f in ('glob_surf_rmse', 'loc_surf_rmse') if _val(h, f, cfg) is not None]
        bad = no & ((nv > 0) | (g > 0) | (l_ > 0) | (v(h, 'glob_Ctau') > 0) | (v(h, 'glob_Cvalid') > 0) | (v(h, 'loc_Ctau') > 0) | (v(h, 'loc_Cvalid') > 0) | np.logical_or.reduce([e.notna() for e in errs]))
        probs.append(_p('semantics', f'S8 no output (0 accepted) with valid points, coverage or errors ({fam})', bad, k))
        if cfg['profile'] == 'c34' and _val(h, 'new_n', cfg) is not None:
            probs.append(_p('semantics', f'S9 new != new correct + wrong + unconfirmable ({fam})', (v(h, 'new_correct') + v(h, 'new_wrong') + v(h, 'new_unconf')).ne(v(h, 'new_n')), k))
        if cfg['denominators'] == 'supplementary+npz':
            for f in ('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'loc_Cvalid', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'glob_surf_rmse', 'n_accepted', 'n_valid'):
                a_, b_ = num(h, f'r_{f}'), num(h, f's_{f}')
                probs.append(_p('semantics', f'S11 supplementary table differs from the record: {f} ({fam})', ((a_ - b_).abs() > K.TOL) | (a_.isna() != b_.isna()), k))
            probs.append(_p('semantics', f'S11 derived domain counts differ from the supplementary table ({fam})', g.ne(num(h, 's_glob_n_valid')) | l_.ne(num(h, 's_loc_n_valid')), k))
            probs.append(_p('semantics', f'S11 record without supplementary row ({fam})', ~col(h, 'sup_matched').fillna(False).astype(bool), k))
    D = T[(T.layer == 'diag') & T.complete]
    if len(D) and P['diag']:
        k = D.key
        for f in ('n_accepted', 'n_valid', 'b_id_correct', 'b_id_wrong', 'b_id_unconf'):
            probs.append(_p('semantics', f'S10 diagnostic count not a non-negative integer: {f}', ~_intlike(num(D, f'r_{f}')), k))
        for dom in ('glob', 'loc'):
            cv, ct = num(D, f'r_{dom}_Cvalid'), num(D, f'r_{dom}_Ctau')
            probs.append(_p('semantics', f'S10 diagnostic {dom} coverage outside [0, 1] or C_tau > C_valid', ~(np.isfinite(cv) & np.isfinite(ct) & (ct <= cv + K.TOL) & (cv <= 1) & (ct >= 0)), k))
    return [p for p in probs if p]


def run(M: pd.DataFrame, T: pd.DataFrame, cfg: dict) -> dict:
    probs = structure(M, cfg['structure']) + records(T, cfg) + columns(T, cfg) + semantics(T, cfg)
    by = {c: sum(p['n'] for p in probs if p['category'] == c) for c in ORDER}
    first = next((c for c in ORDER if by[c]), None)
    return dict(ok=not probs, problems=probs, by_category=by, exit_category=first, n_records=int(len(T)), n_complete=int(T.complete.sum()), rules=SEMANTIC_RULES)
