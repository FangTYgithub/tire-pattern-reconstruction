# Reclassify validation summaries and regenerate their figures from saved results.

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import tr5_common as K

CLOSE = K.OUT / 'closeout'
PROTECTED = ('manifest.csv', 'results.csv', 'ledger.jsonl', 'run.json', 'B2_PERMIT.json', 'effects.csv', 'report_zh.md', 'data/index.json',
             'figures/F1_Punkte_Leistung_Risiko.png', 'figures/F2_Linien_Leistung_Risiko.png', 'figures/F3_Hauptvergleiche_C1_C5.png',
             'figures/F4_Punkte_Rekonstruktion_C1.png', 'figures/F5_Linien_Rekonstruktion_C5.png')
CODE = ('tr5_analysis.py', 'tr5_fig.py')


def protected_hashes() -> dict:
    h = {p: K.sha(K.OUT / p) for p in PROTECTED}
    npz = sorted(K.DATA.glob('S12I-*.npz'))
    h['data/*.npz'] = dict(n=len(npz), digest=K.digest({p.name: K.sha(p) for p in npz}))
    return h


def recorded_check(h: dict) -> dict:

    r = K.run_load(); outs = (r.get('final') or {}).get('outputs') or {}; bad = []
    for p in PROTECTED:
        if p == 'run.json':
            continue
        key = K.rel(K.OUT / p)
        if outs.get(key) != h[p]:
            bad.append(p)
    ix = K.load_json(K.DATA / 'index.json'); badnpz = [n for n, v in ix.items() if K.sha(K.ROOT / n) != v['sha256']]
    return dict(passed=not bad and not badnpz and h['data/*.npz']['n'] == 480, differs_from_final_record=bad, npz_differs_from_index=badnpz[:5], n_npz=h['data/*.npz']['n'])


def in_memory_checks(E: pd.DataFrame, registry: list) -> dict:
    import tr5_analysis as A
    import tr5_drills as DR
    res = {}
    res['none_is_missing'] = A._missing_dir(None)
    res['nan_is_missing'] = A._missing_dir(float('nan')) and A._missing_dir(np.nan) and A._missing_dir(np.float64('nan'))
    res['pandas_na_is_missing'] = A._missing_dir(pd.NA)
    res['registered_signs_not_missing'] = not any(A._missing_dir(x) for x in (1, -1, 1.0, -1.0, np.int64(1), np.float64(-1.0)))
    S = ('B_curv', 'M10_curv', 'M24_curv')
    _, D = DR._synthetic_cells({s: [1.0 + 0.1 * b for b in range(8)] for s in S}, 1); res['positive_registered_still_supported'] = D.label.iloc[0] == 'supported'
    _, D = DR._synthetic_cells({s: [-1.0 - 0.1 * b for b in range(8)] for s in S}, -1); res['negative_registered_still_supported'] = D.label.iloc[0] == 'supported'
    _, D = DR._synthetic_cells({s: [-1.0 - 0.1 * b for b in range(8)] for s in S}, 1); res['opposite_still_reversed'] = D.label.iloc[0] == 'reversed'
    _, D = DR._synthetic_cells({s: [(-1) ** b * 0.5 for b in range(8)] for s in S}, 1); res['alternating_still_insufficient'] = D.label.iloc[0] == 'insufficient'
    C, _ = DR._synthetic_cells({s: [1.0] * 8 for s in S}, 1)
    Cn = C.assign(expected_sign=np.nan); Cn['n_blocks_expected_sign'] = np.nan
    reg = [dict(contrast='C2', method='D0', sigma=60.0, kinds=['none'], expected_signs={s: None for s in S})]
    Dn = A.decisions(Cn, reg); per = json.loads(Dn.per_surface.iloc[0])
    res['nan_direction_gives_exploratory'] = Dn.label.iloc[0] == 'exploratory' and all(v['label'] == 'exploratory' for v in per.values())
    import io
    Cr = pd.read_csv(io.StringIO(C.assign(expected_sign=[None] * len(C)).to_csv(index=False)))
    res['csv_round_trip_none_gives_exploratory'] = A.decisions(Cr, reg).label.iloc[0] == 'exploratory'
    return res


def reclassify(registry: list) -> dict:
    import tr5_analysis as A
    src = K.OUT / 'effects.csv'
    Es = pd.read_csv(src, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    En = pd.read_csv(src, low_memory=False, encoding='utf-8-sig')
    C = En[En.record_type == 'cell_method'].copy()
    Dn = A.decisions(C, registry)
    old = En[En.record_type == 'claim'].set_index(['contrast', 'method', 'sigma'])
    new = Dn.set_index(['contrast', 'method', 'sigma'])
    changes, same = [], []
    for k in old.index:
        o, n = old.loc[k], new.loc[k]
        op = {s: v['label'] for s, v in json.loads(o.per_surface).items()}; npp = {s: v['label'] for s, v in json.loads(n.per_surface).items()}
        orisk = '' if pd.isna(o.risk_higher_on_favoured_arm) else str(o.risk_higher_on_favoured_arm); nrisk = n.risk_higher_on_favoured_arm or ''
        rec = dict(contrast=k[0], method=k[1], sigma=k[2], old_label=o.label, new_label=n.label, old_surface_labels=op, new_surface_labels=npp, risk_equal=(orisk == nrisk))
        (changes if (o.label != n.label or op != npp) else same).append(rec)
    out = Es.copy()
    for ch in changes:
        cand = out.index[(out.record_type == 'claim') & (out.contrast == ch['contrast']) & (out.method == ch['method'])]
        i = [j for j in cand if float(out.at[j, 'sigma']) == ch['sigma']]
        assert len(i) == 1
        i = i[0]
        per = json.loads(out.at[i, 'per_surface'])
        for s in per:
            per[s]['label'] = ch['new_surface_labels'][s]
        out.at[i, 'label'] = ch['new_label']; out.at[i, 'label_zh'] = A.LABELS_ZH[ch['new_label']]; out.at[i, 'per_surface'] = K.canon(per)
    K.save_csv(out, CLOSE / 'effects.csv')

    Ec = pd.read_csv(CLOSE / 'effects.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    same_shape = Ec.shape == Es.shape and list(Ec.columns) == list(Es.columns)
    diff = (Ec != Es)
    cells = [(int(r), c) for r, c in zip(*np.nonzero(diff.values))]
    allowed = {('label',), ('label_zh',), ('per_surface',)}
    changed_rows = sorted({r for r, _ in cells}); changed_cols = sorted({Es.columns[c] for _, c in cells})
    per_ok = True
    for r in changed_rows:
        a, b = json.loads(Es.at[r, 'per_surface']), json.loads(Ec.at[r, 'per_surface'])
        for s in a:
            a[s].pop('label'); b[s].pop('label')
        per_ok &= (a == b)
    lines_o = src.read_bytes().splitlines(); lines_c = (CLOSE / 'effects.csv').read_bytes().splitlines()
    nl_diff = sum(1 for x, y in zip(lines_o, lines_c) if x != y) + abs(len(lines_o) - len(lines_c))
    counts = Ec.record_type.value_counts().to_dict()
    ver = dict(same_shape_and_columns=same_shape, changed_rows=len(changed_rows), changed_columns=changed_cols, only_classification_fields=set(changed_cols) <= {'label', 'label_zh', 'per_surface'},
               per_surface_numbers_equal=per_ok, changed_text_lines=nl_diff, counts=counts,
               pairs_complete=(counts.get('pair_method') == 816 and counts.get('pair_frontend') == 384 and counts.get('block_method') == 408 and counts.get('block_frontend') == 192
                               and counts.get('cell_method') == 54 and counts.get('cell_frontend') == 24 and counts.get('claim') == 18))
    return dict(changes=changes, unchanged=len(same), unchanged_labels=[(s['contrast'], s['method'], s['sigma'], s['old_label']) for s in same], verification=ver,
                allowed_fields=sorted(f[0] for f in allowed))


def figures() -> dict:
    import tr5_fig as F
    r = K.run_load(); sel = r['analysis']['cases']
    R = pd.read_csv(K.OUT / 'results.csv', low_memory=False, encoding='utf-8-sig')
    E = pd.read_csv(CLOSE / 'effects.csv', low_memory=False, encoding='utf-8-sig')
    C, D, FP = E[E.record_type == 'cell_method'], E[E.record_type == 'claim'], E[E.record_type == 'pair_frontend']
    cap4 = ('Fehlschlagsfall des D0-Frontends (kein Beleg für Messbarkeit): Fälle nach vorab festgelegter Regel (Median-nächster und ungünstigster der C1-Paare, σ80, D0).\n'
            'Die DF-Ergebnisse derselben Bilder (Kopfzeilen) sind über F1 / F3 statistisch gesondert belegt. Punktwolke und Histogramm: alle gültigen Ausgabepunkte; '
            'Cτ und Punkt-RMSE im Titel: globaler bzw. lokaler Bewertungsbereich.\nGraue Kontur: globaler Bewertungsbereich; gestrichelter Kreis: lokaler Bewertungsbereich (r = 15 mm). '
            '„Gültige Ausgabepunkte“ bedeutet nicht, dass die Messung die Anforderungen erfüllt.')
    cap5 = ('Fälle nach vorab festgelegter Regel (Median-nächster und ungünstigster der C5-Paare, σ60, P4). Punktwolke und Histogramm: alle gültigen Ausgabepunkte; '
            'Cτ und Punkt-RMSE im Titel:\nglobaler bzw. lokaler Bewertungsbereich (z. B. Median-Fall s10: 2635 gültige Ausgabepunkte, davon 2539 im globalen Bereich). '
            'Graue Kontur: globaler Bewertungsbereich; gestrichelter Kreis: lokaler Bewertungsbereich (r = 15 mm).')
    out = dict(F3=F.f3(C, D, CLOSE, False, F.collapse_note(R)))
    out['F4'] = F.case_figure(sel['F4'], K.OUT, FP, CLOSE, 'F4_Punkte_Rekonstruktion_C1.png', ('r1 R8×4', 'r0,5 R8×4'),
                              f"F4  Punktraster C1 (r1 R8×4 vs. r0,5 R8×4), σ{sel['F4']['sigma']:g}, D0 — Median-nächstes und ungünstigstes Realisierungspaar", 'D0', False, cap4)
    out['F5'] = F.case_figure(sel['F5'], K.OUT, None, CLOSE, 'F5_Linien_Rekonstruktion_C5.png', ('s10', 's8'),
                              f"F5  Linienraster C5 (s10 vs. s8, w1, 0°), σ{sel['F5']['sigma']:g}, P4 — Median-nächstes und ungünstigstes Realisierungspaar", 'P4', False, cap5)
    return {k: dict(file=v, sha256=K.sha(K.ROOT / v)) for k, v in out.items()}


def code_changes(before: Path) -> dict:
    out = {}
    for f in CODE:
        b = (before / f).read_text(encoding='utf-8').splitlines(keepends=False); a = (K.HERE / f).read_text(encoding='utf-8').splitlines(keepends=False)
        out[f] = dict(sha_before=K.sha(before / f), sha_after=K.sha(K.HERE / f), kind=('analysis (classification only)' if f == 'tr5_analysis.py' else 'figure layout'),
                      diff='\n'.join(difflib.unified_diff(b, a, f'{f} (before)', f'{f} (after)', n=0, lineterm='')))
    out['tr5_closeout.py'] = dict(sha_after=K.sha(K.HERE / 'tr5_closeout.py'), kind='new closeout entry (this script)')
    return out


def build(before: str) -> dict:
    r = K.run_load(); registry = r['g0']['claims_summary']
    h0 = protected_hashes(); rc = recorded_check(h0)
    if not rc['passed']:
        raise K.Stop('preflight', f'protected originals differ from the TR05 final record: {rc}')
    E = pd.read_csv(K.OUT / 'effects.csv', low_memory=False, encoding='utf-8-sig')
    mem = in_memory_checks(E, registry)
    rec = reclassify(registry)
    ok_change = (len(rec['changes']) == 1 and rec['changes'][0]['contrast'] == 'C4' and rec['changes'][0]['method'] == 'D0' and rec['changes'][0]['sigma'] == 80.0
                 and rec['changes'][0]['new_label'] == 'exploratory' and rec['unchanged'] == 17 and all(c['risk_equal'] for c in rec['changes']))
    figs = figures()
    h1 = protected_hashes()
    v = rec['verification']
    checks = dict(in_memory=all(mem.values()), only_c4_d0_s80_changed=ok_change, numbers_unchanged=(v['same_shape_and_columns'] and v['only_classification_fields'] and v['per_surface_numbers_equal']
                  and v['changed_rows'] == 1), pairs_complete=v['pairs_complete'], originals_unchanged=(h0 == h1) and rc['passed'])
    out = dict(task='THESIS-REINFORCE-05C', task_basis=K.rel(K.DRAFTS / 'THESIS-REINFORCE-05_review.md'), t=K.now(),
               real_calls=dict(imaging=0, decoding=0, surface_evaluation=0, bootstrap_recomputed=0, pairs_recomputed=0),
               inputs=dict(protected=h0, recorded_check=rc, reference_review_sha256=K.sha(K.DRAFTS / 'THESIS-REINFORCE-05_review.md')),
               code_changes=code_changes(Path(before)), checks=checks, in_memory=mem, reclassification=rec, figures=figs,
               reused_unchanged=dict(F1=K.rel(K.FIGS / 'F1_Punkte_Leistung_Risiko.png'), F2=K.rel(K.FIGS / 'F2_Linien_Leistung_Risiko.png')),
               note='run.json, manifest, results, ledger, data/ and the G0 record were only read; no new permit was issued; the frozen G0 history is not rewritten')
    K.save_json(CLOSE / 'closeout.json', out)
    return dict(checks=checks, changes=rec['changes'], figures=figs)


def seal() -> dict:
    out = K.load_json(CLOSE / 'closeout.json')
    h = protected_hashes()
    rep = CLOSE / 'report_zh.md'
    if not rep.exists():
        raise K.Stop('refused', 'revised report missing')
    files = {K.rel(p): K.sha(p) for p in sorted(CLOSE.glob('*')) if p.is_file() and p.name != 'closeout.json'}
    out['outputs'] = files
    out['originals_unchanged_at_seal'] = h == out['inputs']['protected']
    out['handover'] = dict(report=K.rel(rep), effects=K.rel(CLOSE / 'effects.csv'), figures=[K.rel(CLOSE / f) for f in ('F3_Hauptvergleiche_C1_C5.png', 'F4_Punkte_Rekonstruktion_C1.png',
                           'F5_Linien_Rekonstruktion_C5.png')], reused_figures=list(out['reused_unchanged'].values()), results=K.rel(K.OUT / 'results.csv'),
                           point_clouds=K.rel(K.DATA), cases=K.rel(K.RUN) + ' -> analysis.cases', original_report=K.rel(K.OUT / 'report_zh.md'),
                           draft_to_update='protocols/chapter3_sections1_2.md (unchanged here)')
    out['sealed'] = K.now(); out['status'] = 'NUMERICAL_CHECKS_COMPLETE'
    K.save_json(CLOSE / 'closeout.json', out)
    return dict(originals_unchanged=out['originals_unchanged_at_seal'], outputs=files)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['build', 'seal']); ap.add_argument('--before', default=None)
    a = ap.parse_args()
    try:
        r = build(a.before) if a.cmd == 'build' else seal()
        print(json.dumps(K.clean(r), ensure_ascii=False, default=str)[:5000])
    except K.Stop as e:
        print(json.dumps(dict(stop=e.code, msg=e.msg), ensure_ascii=False)[:3000]); sys.exit(e.exit_code)


if __name__ == '__main__':
    main()
