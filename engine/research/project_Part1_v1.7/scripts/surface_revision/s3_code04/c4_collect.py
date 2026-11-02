# Collect complete encoding records into case and metric tables.

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

import c4_common as K
import c4_run as RN

STRATUM = ['family', 'surface', 'layout', 'pattern', 'sigma']


def expected(manifest: pd.DataFrame, fps: dict) -> pd.DataFrame:
    rows = []
    for r in manifest.to_dict('records'):
        ik = r['image_key']
        for name, method, params in RN.rules_for(r):
            dk = K.decode_key(ik, r['family'], method, name, params, fps['decode'])
            rows.append(dict(image_key=ik, family=r['family'], rule=name, arm='', layer='decode', reference='', key=dk, decode_key=dk))
            for a in (RN.arms_for(r, method) or [name]):
                rows.append(dict(image_key=ik, family=r['family'], rule=name, arm=a, layer='eval', reference='', key=K.eval_key(dk, a, fps['eval']), decode_key=dk))
            for a, rt in RN.diag_for(r, name):
                rows.append(dict(image_key=ik, family=r['family'], rule=name, arm=a, layer='diag', reference=rt, key=K.diag_key(dk, a, rt, fps['diag']), decode_key=dk))
    E = pd.DataFrame(rows)
    if E.key.duplicated().any():
        raise K.Stop('check', 'duplicate expected keys')
    return E


def collect(root: Path, manifest: pd.DataFrame, fps: dict) -> tuple[pd.DataFrame, dict]:
    root = Path(root); E = expected(manifest, fps); recs = []; sub = dict(decode='dec', eval='eval', diag='diag')
    for e in E.itertuples():
        p = root / sub[e.layer] / f'{e.key}.json'; ok = K.complete(p, e.key); row = {}; rid = dict(rec_image_key=None, rec_rule=None, rec_arm=None)
        if ok:
            j = K.load_json(p); row = j.get('row', {}) if e.layer != 'decode' else {}
            rid = dict(rec_image_key=j.get('image_key'), rec_rule=j.get('rule'), rec_arm=j.get('arm', ''))
        recs.append(dict(e._asdict(), complete=ok, **rid, **{f'r_{k}': v for k, v in row.items()}))
    T = pd.DataFrame(recs).drop(columns=['Index'])
    exp_keys = set(E.key); slots = {(r.image_key, r.rule, r.arm, r.layer, r.reference) for r in E.itertuples()}
    other, unplanned = Counter(), Counter()
    for layer, d in sub.items():
        for p in (root / d).glob('*.json'):
            if p.stem in exp_keys:
                continue
            try:
                j = K.load_json(p)
            except Exception:
                unplanned[layer] += 1; continue
            slot = (j.get('image_key'), j.get('rule'), j.get('arm', '') if layer != 'decode' else '', layer, j.get('reference', '') if layer == 'diag' else '')
            (other if slot in slots else unplanned)[layer] += 1
    return T, dict(expected=E.groupby('layer').size().to_dict(), complete=T[T.complete].groupby('layer').size().to_dict(),
                   missing=T[~T.complete].groupby('layer').size().to_dict(), other_version_not_counted=dict(other), unplanned_not_counted=dict(unplanned))


# Check case coverage, unique pair identities and consistency of collected records.
def gate(T: pd.DataFrame, manifest: pd.DataFrame, mode: str = 'formal', phases_expected: int | None = None, reps_expected: dict | None = None,
         pairing: bool = True) -> dict:

    M = manifest.copy(); M['layout'] = M.layout.astype(str); probs = {}
    nph = phases_expected or len(K.PARITY_OF_PHASE)
    reps = {**dict(zero=1, pos=2), **(reps_expected or {})}

    g = M.groupby(STRATUM).phase_id.nunique(); bad = g[g != nph]
    if len(bad):
        probs['stratum_phase_count'] = [dict(zip(STRATUM, k), phases=int(v)) for k, v in bad.items()][:20]
    rep = M.groupby(STRATUM + ['phase_id']).size(); si = STRATUM.index('sigma')
    want = pd.Series([reps['zero'] if float(k[si]) == 0 else reps['pos'] for k in rep.index], index=rep.index)
    badr = rep[rep != want]
    if len(badr):
        probs['repetitions'] = [dict(zip(STRATUM + ['phase_id'], k), n=int(v), want=int(want[k])) for k, v in badr.items()][:20]
    geo = [c for c in ('u', 'v', 'phase_frac') if c in M]
    if geo:
        G = M.assign(**{c: M[c].map(lambda x: '' if pd.isna(x) else repr(float(x))) for c in geo}).drop_duplicates(['family', 'layout', 'phase_id'] + geo)
        a = G.groupby(['family', 'layout', 'phase_id'], dropna=False).size(); b = G.groupby(['family', 'layout'] + geo, dropna=False).phase_id.nunique()
        if (a > 1).any() or (b > 1).any():
            probs['phase_label_geometry'] = dict(labels_with_several_positions=[list(k) for k in a[a > 1].index][:10], positions_with_several_labels=[list(k) for k in b[b > 1].index][:10])
    else:
        probs['phase_label_geometry'] = 'manifest has no geometric position columns'
    if M.image_key.duplicated().any():
        probs['duplicate_manifest_rows'] = M[M.image_key.duplicated()].image_key.head(10).tolist()
    if pairing:
        if 'pair_key' not in M or M.pair_key.isna().any():
            probs['pairing'] = 'pair_key missing'
        else:
            pk = M.groupby('pair_key').agg(n=('pattern', 'nunique'), s=('seed', lambda x: x.nunique(dropna=False)))
            if (pk.n != 4).any() or (pk.s != 1).any():
                probs['pairing'] = pk[(pk.n != 4) | (pk.s != 1)].head(10).reset_index().to_dict('records')

    miss = T[~T.complete]
    if len(miss):
        probs['missing_records'] = dict(n=int(len(miss)), by_layer=miss.groupby('layer').size().to_dict(), keys=miss.key.head(10).tolist())
    c = T[T.complete]
    mm = c[(c.rec_image_key != c.image_key) | (c.rec_rule != c.rule) | ((c.layer != 'decode') & (c.rec_arm.fillna('') != c.arm))]
    if len(mm):
        probs['record_slot_mismatch'] = mm[['key', 'image_key', 'rec_image_key', 'rule', 'rec_rule', 'arm', 'rec_arm']].head(10).to_dict('records')
    dup = T.duplicated(['image_key', 'rule', 'arm', 'layer', 'reference'], keep=False)
    if dup.any():
        probs['duplicate_slots'] = T[dup][['image_key', 'rule', 'arm', 'layer']].head(10).to_dict('records')
    ev = c[c.layer == 'eval']
    if 'r_n_accepted' in ev:
        out_ = pd.to_numeric(ev.r_n_accepted, errors='coerce').fillna(-1)
        gap = ev[(out_ > 0) & pd.to_numeric(ev.get('r_glob_Ctau'), errors='coerce').isna()]
        if len(gap):
            probs['metric_missing_with_output'] = gap.key.head(10).tolist()
        neg = ev[out_ < 0]
        if len(neg):
            probs['accepted_count_missing'] = neg.key.head(10).tolist()
    res = dict(ok=not probs, mode=mode, problems=probs, n_records=int(len(T)), n_complete=int(T.complete.sum()),
               structure=dict(phases=int(nph), reps=reps, pairing=bool(pairing)),
               status=('COMPLETE - formal statistics allowed' if not probs else ('PARTIAL - progress summary only, not a final result' if mode == 'progress' else 'REFUSED')))
    if probs and mode == 'formal':
        raise K.Stop('incomplete', f'formal statistics refused: {sorted(probs)}')
    return res


if __name__ == '__main__':
    import json
    root = Path(sys.argv[1]); fz = K.load_json(sys.argv[2]); man = pd.read_csv(sys.argv[3], low_memory=False)
    T, s = collect(root, man, fz['fingerprints']); g = gate(T, man, mode=(sys.argv[4] if len(sys.argv) > 4 else 'formal'))
    print(json.dumps(K.clean(dict(collection=s, gate=g)), indent=1))
