# Build paired four-cell interactions and front-end comparisons.

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import s5v_common as K
import s5u_stats as ST5
ST2 = ST5.ST
PH = ST5.PH
KEYCOLS = ['record_type', 'set_name', 'surface', 'family', 'sigma', 'set', 'effect', 'metric', 'front_end', 'analysis']
SUMCOLS = ['mean', 'median', 'n_pos', 'n_neg', 'n_zero', 'min', 'max', 'res_lo_5', 'res_hi_95', *PH]
FAMILY_FE = (('point', 'D0'), ('point', 'DF'), ('line', 'P4'))
TECH = re.compile(r'has \d+ rows / \d+ reps of \d+')
SCI = re.compile(r'^(ph\d+): (\S+) undefined in (\S+) \(no output\)$')
DOTS = ('D_r0.75_p8x4', 'D_r1_p6x4', 'D_r1_p8x4', 'D_r1_p10x4', 'D_r1_p8x6', 'D_r1.5_p6x4', 'D_r1.5_p8x4', 'D_r1.5_p10x4', 'D_r1.5_p8x6')


def _defs(fam: str, analysis: str) -> list:
    if analysis == 'registered':
        return ST2.effect_defs(fam, ST2.K.FOUR, True)
    return ST2.effect_defs(fam, ST2.K.FOUR_POSTHOC, False) if fam == 'line' else []


def expected_keys() -> dict:
    per, cross, dfd0 = [], [], []
    for analysis in ('registered', K.POSTHOC):
        for fam, fe in FAMILY_FE:
            for setn, comps in _defs(fam, analysis):
                for eff in comps:
                    for m in ST5.EFFECT_METRICS:
                        for s in K.SURF:
                            for sg in K.SIG_LABELS:
                                per.append(('interaction', 'S5-main', s, fam, sg, setn, eff, m, fe, analysis))
                            cross.append(('interaction', 'S5-main', s, fam, K.CROSS, setn, eff, m, fe, analysis))
                            if fam == 'point' and fe == 'D0' and analysis == 'registered':
                                for sg in (*K.SIG_LABELS, K.CROSS):
                                    dfd0.append(('interaction_df_minus_d0', 'S5-main', s, 'point', sg, setn, eff, m, 'DF-D0', 'registered'))
    return dict(per_sigma=per, cross=cross, df_minus_d0=dfd0)


def _key(r) -> tuple:
    return tuple(r[c] for c in KEYCOLS)


class _Phases:

    def __init__(self, X: pd.DataFrame):
        self.g = {k: g for k, g in X.groupby(['set_name', 'surface', 'family', 'sigma', 'front_end'])}
        self.w = {(fam, an, setn, eff): w for fam, _ in FAMILY_FE for an in ('registered', K.POSTHOC) for setn, comps in _defs(fam, an) for eff, w in comps.items()}
        self.cache = {}

    def defined(self, key: tuple) -> set:
        if key not in self.cache:
            _, sn, s, fam, sg, setn, eff, m, fe, an = key
            g = self.g.get((sn, s, fam, float(sg), fe))
            vals, _ = ST2.phase_values(g, self.w[(fam, an, setn, eff)], m) if g is not None else ({}, None)
            self.cache[key] = set(vals)
        return self.cache[key]

    def detail(self, key: tuple, reason: str) -> tuple:

        if TECH.search(str(reason)):
            return 'technical', str(reason)
        mt = SCI.match(str(reason))
        if not mt:
            return 'technical', f'unrecognised undefined reason: {reason}'
        ph, m, cell = mt.groups(); _, sn, s, fam, sg, *_ = key
        g = self.g[(sn, s, fam, float(sg), key[8])]; c = g[(g.pair_role == cell) & (g.phase_label == ph)]
        n_no = int((c.state == 'no_output').sum()); n_nan = int(((c.state == 'ok') & pd.to_numeric(c[m], errors='coerce').isna()).sum())
        cls = 'scientific_no_output' if n_nan == 0 else ('scientific_metric_undefined' if n_no == 0 else 'scientific_mixed')
        miss = sorted(set(PH) - self.defined(key))
        return cls, f'{m} undefined; first failing {ph} / {cell}: {n_no} no-output case(s), {n_nan} case(s) with output but {m} undefined; phases without a value: {",".join(miss)}'


def _merge_class(classes: list) -> str:
    u = sorted(set(classes))
    if 'technical' in u:
        return 'technical'
    return u[0] if len(u) == 1 else 'scientific_mixed'


# Calculate paired four-cell interactions and front-end contrasts from result tables.
def build(C: pd.DataFrame) -> pd.DataFrame:
    X = ST5._stat_frame(C); idx = ST2.resample_idx(); PHS = _Phases(X)
    T = ST2.interactions(X, ST2.K.FOUR, True, idx, metrics=ST5.EFFECT_METRICS, extra_keys=('front_end',)); T = T[T.sigma != K.CROSS].copy(); T['analysis'] = 'registered'
    H = ST2.interactions(X[X.family == 'line'], ST2.K.FOUR_POSTHOC, False, idx, metrics=ST5.EFFECT_METRICS, extra_keys=('front_end',)); H = H[H.sigma != K.CROSS].copy()
    H['analysis'] = K.POSTHOC
    for D in (T, H):
        D['derivation'] = 'per sigma'; D['undefined_class'] = None; D['undefined_detail'] = None
        for i in D.index[D.state != 'ok']:
            D.at[i, 'undefined_class'], D.at[i, 'undefined_detail'] = PHS.detail(_key(D.loc[i]), D.at[i, 'reason'])
    PS = pd.concat([T, H], ignore_index=True)
    dup = PS.duplicated(KEYCOLS, keep=False)
    if dup.any():
        raise K.Stop('refused', f'{int(dup.sum())} duplicate per-sigma effect keys', detail=PS[dup][KEYCOLS].head(5).to_dict('records'))
    by = {_key(r): r for _, r in PS.iterrows()}
    exp = expected_keys()

    def phases(k):
        return set(PH) if by[k]['state'] == 'ok' else PHS.defined(k)

    def derived(key, src: dict, rtype_fields: dict, label: str):
        miss = [n for n, k in src.items() if k not in by]
        if miss:
            raise K.Stop('refused', f'{key}: source rows missing ({miss}) - technical gap', detail=[str(src[n]) for n in miss])
        rows = {n: by[k] for n, k in src.items()}
        if all(r['state'] == 'ok' for r in rows.values()):
            a, b = rows['a'], rows['b']
            e = a[PH].to_numpy(float) - b[PH].to_numpy(float)
            return dict(rtype_fields, **ST2.summarise(e, idx), derivation=label)
        names = ('σ40', 'σ0') if label == 'σ40 − σ0' else ('DF', 'D0')
        why = ' | '.join(f"{n_}: {r['state']}" + (f" ({r['reason']})" if r['state'] != 'ok' else '') for n_, r in zip(names, (rows['a'], rows['b'])))
        cls = _merge_class([r['undefined_class'] for r in rows.values() if r['state'] != 'ok'])
        ph_ok = set(PH)
        for k in src.values():
            ph_ok &= phases_full(k)
        return dict(rtype_fields, state='undefined', n_phases_complete=len(ph_ok), reason=f'{label} not computable: {why}', derivation=label, undefined_class=cls,
                    undefined_detail=f'phases with both sides defined: {",".join(sorted(ph_ok)) or "none"}')

    def phases_full(k):
        if k[4] == K.CROSS:
            return phases_full(k[:4] + ('40.0',) + k[5:]) & phases_full(k[:4] + ('0.0',) + k[5:])
        return phases(k)

    cross = []
    for k in exp['cross']:
        a = k[:4] + ('40.0',) + k[5:]; b = k[:4] + ('0.0',) + k[5:]
        cross.append(derived(k, dict(a=a, b=b), dict(zip(KEYCOLS, k)), 'σ40 − σ0'))
    CR = pd.DataFrame(cross)
    for _, r in CR.iterrows():
        by[_key(r)] = r
    dd = []
    for k in exp['df_minus_d0']:
        base = ('interaction', 'S5-main', k[2], 'point', k[4], k[5], k[6], k[7])
        dd.append(derived(k, dict(a=base + ('DF', 'registered'), b=base + ('D0', 'registered')), dict(zip(KEYCOLS, k)), 'DF − D0'))
    DD = pd.DataFrame(dd)
    CRr, CRh = CR[CR.analysis == 'registered'], CR[CR.analysis == K.POSTHOC]
    E = pd.concat([T, CRr, H, CRh, DD], ignore_index=True, sort=False)
    E['table'] = K.TABLE_E
    return E


def frontend(C: pd.DataFrame) -> pd.DataFrame:
    FC = ST5.frontend_conditions(C); FC['table'] = K.TABLE_FC
    return FC


def validate(E: pd.DataFrame) -> dict:
    exp = expected_keys(); want = exp['per_sigma'] + exp['cross'] + exp['df_minus_d0']
    keys = [_key(r) for _, r in E[KEYCOLS].iterrows()]
    dup = pd.Series(keys).duplicated(keep=False)
    if dup.any():
        raise K.Stop('refused', f'{int(dup.sum())} duplicate effect keys', detail=[str(k) for k in pd.Series(keys)[dup].head(5)])
    got, ws = set(keys), set(want)
    if len(want) != len(ws):
        raise K.Stop('refused', 'the registered key set itself has duplicates')
    missing, extra = sorted(ws - got, key=str), sorted(got - ws, key=str)
    if missing or extra:
        raise K.Stop('refused', f'effect keys differ from the registered set: {len(missing)} missing, {len(extra)} extra', detail=dict(missing=[str(k) for k in missing[:5]], extra=[str(k) for k in extra[:5]]))
    und = E[E.state != 'ok']
    tech = und[und.undefined_class.astype(str) == 'technical']
    if len(tech):
        raise K.Stop('refused', f'{len(tech)} effect rows undefined for technical reasons (missing / duplicate records) - no formal table', detail=tech[KEYCOLS + ['reason']].head(5).to_dict('records'))
    ok = E[E.state == 'ok']
    bad = ok[ok[PH].isna().any(axis=1) | ok['mean'].isna()]
    if len(bad):
        raise K.Stop('refused', f'{len(bad)} rows state ok with missing phase values')
    if und[PH + ['mean']].notna().any().any():
        raise K.Stop('refused', 'undefined rows carry numeric values')
    by_part = dict(per_sigma=len(exp['per_sigma']), cross=len(exp['cross']), df_minus_d0=len(exp['df_minus_d0']))
    return dict(ok=True, n_rows=len(E), expected=by_part, n_undefined=int(len(und)), undefined_by_class=und.undefined_class.value_counts().to_dict(),
                undefined_by_derivation=und.derivation.value_counts().to_dict())


def validate_fc(FC: pd.DataFrame) -> dict:
    k = list(zip(FC.surface, FC.combo, FC.sigma.astype(float)))
    if pd.Series(k).duplicated().any():
        raise K.Stop('refused', 'duplicate front-end condition keys')
    want = {(s, c, sg) for s in K.SURF for c in DOTS for sg in K.SIG}
    if set(k) != want:
        raise K.Stop('refused', f'front-end condition keys differ: {len(want - set(k))} missing, {len(set(k) - want)} extra')
    return dict(ok=True, n_rows=len(FC))


def load_paired(src) -> tuple:

    df = pd.read_csv(src, float_precision='round_trip', dtype={'sigma': str}, low_memory=False, encoding='utf-8-sig')
    tabs = set(df['table'].astype(str))
    if not tabs <= {K.TABLE_E, K.TABLE_FC}:
        raise K.Stop('refused', f'unknown table labels {sorted(tabs - {K.TABLE_E, K.TABLE_FC})}')
    E = df[df.table == K.TABLE_E].copy(); FC = df[df.table == K.TABLE_FC].copy()
    bad = sorted(set(E.sigma) - set(K.SIG_LABELS) - {K.CROSS})
    if bad:
        raise K.Stop('refused', f'effect rows with unknown sigma labels {bad}')
    badf = sorted(set(FC.sigma) - set(K.SIG_LABELS))
    if badf:
        raise K.Stop('refused', f'front-end condition rows with sigma labels outside the six levels {badf}')
    FC['sigma'] = FC.sigma.astype(float)
    validate_fc(FC)
    return E, FC


def compare_old(E_new: pd.DataFrame, FC_new: pd.DataFrame, E_old: pd.DataFrame, FC_old: pd.DataFrame, tol: float = 1e-12) -> dict:

    out = {}
    for name, new, old, kc in (('effects', E_new, E_old, KEYCOLS), ('frontend', FC_new, FC_old, ['surface', 'combo', 'sigma'])):
        n_idx = new.set_index(kc); o_idx = old.set_index(kc)
        if o_idx.index.duplicated().any() or n_idx.index.duplicated().any():
            raise K.Stop('refused', f'{name}: duplicate keys in comparison')
        miss = o_idx.index.difference(n_idx.index)
        if len(miss):
            raise K.Stop('refused', f'{name}: {len(miss)} S5-ALIGN-05 rows missing in the new table', detail=[str(x) for x in miss[:5]])
        cols = [c for c in old.columns if c not in kc and old[c].notna().any()]
        nn = n_idx.loc[o_idx.index, cols]; oo = o_idx[cols]
        worst, nbad, state_diff = 0.0, 0, 0
        for c in cols:
            a, b = oo[c], nn[c]
            if a.dtype.kind in 'fiub':
                an, bn = a.astype(float), pd.to_numeric(b, errors='coerce')
                both = an.notna() & bn.notna()
                if (an.isna() != bn.isna()).any():
                    nbad += int((an.isna() != bn.isna()).sum())
                d = (an[both] - bn[both]).abs(); lim = tol + tol * an[both].abs()
                nbad += int((d > lim).sum()); worst = max(worst, float(d.max()) if len(d) else 0.0)
            else:
                eq = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
                nbad += int((~eq).sum())
                if c == 'state':
                    state_diff += int((~eq).sum())
        if nbad:
            raise K.Stop('refused', f'{name}: {nbad} value / state differences against S5-ALIGN-05 (max |d| {worst:.3g})')
        out[name] = dict(n_old=len(old), n_new=len(new), n_compared_columns=len(cols), max_abs_diff=worst, state_differences=state_diff, ok=True)
    return out
