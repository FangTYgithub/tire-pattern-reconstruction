# Coordinate environment-study checks, execution and analysis.

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

import mp5_common as K
import mp5_gate as G
import mp4_gate as G4

MANIFEST = K.OUT / 'formal_manifest.csv'
ENTRY_SUBSET = dict(
    B1=dict(subset='sigma0_diag'),
    B2=dict(subset='core', surface='M10_curv', phase_block=0, sigma=40.0, rep=1),
    B3=dict(subset='core', config=('D_r1_p8x4', 'L_w1.5_s8'), surface='M10_curv', phase_block=0, sigma=40.0, rep=2),
    B4=dict(subset='core', config=('L_w1.5_s6.3',), surface='B_curv', phase_block=(1, 2), sigma=20.0, rep=1),
    B5=dict(subset='core', config=('D_r1_p8x4', 'L_w1.5_s8'), surface='M24_curv', phase_block=3, sigma=20.0, rep=1),
    B6=dict(subset=('transition', 'transition_sigma0'), surface='M10_curv', phase_block=0, sigma=(0.0, 40.0), rep=(0, 1)))
ENTRY_EXPECTED = dict(images=136, methods=196, D0=60, DF=60, P4=76)
EXPECTED_CALLS = dict(forward=2450, forward_entry=136, decode=3626, decode_D0=1176, decode_DF=1176, decode_P4=1274, evaluation=3626)


def rules_all() -> dict:
    import mp5_analysis as AN
    import mp5_chain as CH
    import mp5_figs as FG
    return dict(analysis=AN.RULES, figures=FG.RULES, entry_subset=ENTRY_SUBSET, entry_expected=ENTRY_EXPECTED, expected_calls=EXPECTED_CALLS, replay=list(CH.REPLAY_PICKS),
                p4_wording='degradation under the fixed nominal-PSF P4 is observed; scale / threshold mismatch may contribute - not proven to be the whole cause')


def manifest(path=MANIFEST) -> pd.DataFrame:
    M = pd.read_csv(path, low_memory=False, float_precision='round_trip')
    M['stage'] = 'main'
    sel = entry_mask(M)
    M.loc[sel, 'stage'] = 'entry'
    import mp5_analysis as AN
    M['keep_arrays'] = M.case_key.isin(AN.BOARD_CASES)
    return M


def _match(M: pd.DataFrame, spec: dict) -> pd.Series:
    m = pd.Series(True, index=M.index)
    for k, v in spec.items():
        m &= M[k].isin(list(v)) if isinstance(v, (list, tuple)) else (M[k] == v)
    return m


def entry_mask(M: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=M.index)
    for spec in ENTRY_SUBSET.values():
        m |= _match(M, spec)
    return m


_PREPS: dict = {}


def preps(M: pd.DataFrame) -> dict:
    import mp5_inputs as IN
    key = K.digest(M.forward_key.tolist())
    if _PREPS.get('key') != key:
        _PREPS.clear()
        p = {r['case_key']: IN.prepare(r) for r in M.to_dict('records')}
        _PREPS.update(key=key, p=p)
    return _PREPS['p']


def results_sha(P=K.P0):
    p = P.root / 'results.csv'
    return K.sha(p) if p.exists() else None


def cur_parts(P=K.P0, manifest_path=MANIFEST, rules=None, layer_override=None) -> dict:
    return G.parts(manifest_path, rules or rules_all(), P.run_load().get('freeze'), results_sha(P), layer_override)


def protected_files() -> list:
    fs = [K.MP4_OUT / x for x in ('run.json', 'report_zh.md', 'formal_manifest.csv', 'diagnostic_cases.csv')]
    fs += [K.MP3_OUT / x for x in ('run.json', 'manifest.csv', 'results.csv', 'effects.csv', 'ledger.jsonl', 'data/index.json')]
    fs += sorted((K.SR / 'model_prep03').glob('mp3_*.py')) + sorted((K.SR / 'model_prep04').glob('mp4_*.py')) + sorted((K.ROOT / 'src').glob('*.py'))
    fs += [K.REVIEW4, K.TASK4, K.OUTSR / 'sections_adapter' / 'candidates_s1s5.csv']
    return fs


def protected_now() -> dict:
    return {K.rel(p): (K.sha(p) if p.exists() else None) for p in protected_files()}


def protected_changed(P=K.P0) -> list:
    reg = (P.run_load().get('protected') or {}).get('files') or {}
    if not reg:
        return ['no registration']
    now = protected_now()
    return sorted(k for k in set(reg) | set(now) if reg.get(k) != now.get(k))


def _row(cid, ok, desc, **detail) -> dict:
    return dict(check_id=cid, passed=bool(ok), description=desc, detail=json.dumps(K.clean(detail), ensure_ascii=False, default=str)[:3800])


def _proof_put(st: str, rows: list, P=K.P0) -> dict:
    prts = cur_parts(P); pf = G.make_proof(rows, prts, st)
    r = P.run_load(); pr = r.get('proofs') or {}; pr[st] = pf; P.run_put('proofs', pr)
    ok, why = G4.check_proof(pf, G.REQUIRED[st], G4.binding(prts, G.PROOF_LAYERS[st]))
    return dict(ok=ok, why=why, n=len(rows), failed=[x['check_id'] for x in rows if x['passed'] is not True])


def st_register(refreeze=None, reason=None) -> dict:
    import tr3_b2 as B2
    P = K.P0; r = P.run_load()
    if 'protected' not in r:
        P.run_put('protected', dict(t=K.now(), files=protected_now()))
    elif protected_changed(P):
        raise K.Stop('refused', f'protected inputs changed since registration: {protected_changed(P)[:5]}')
    src = K.MP4_OUT / 'formal_manifest.csv'
    m4 = K.load_json(K.MP4_OUT / 'run.json')
    if K.sha(src) != m4['formal']['manifest_sha'] or K.sha(src) != m4['final']['outputs'].get(K.rel(src)):
        raise K.Stop('refused', 'MODEL-PREP-04 manifest differs from its recorded hash')
    if not MANIFEST.exists():
        P.atomic(MANIFEST, lambda t: shutil.copyfile(src, t))
    if K.sha(MANIFEST) != K.sha(src):
        raise K.Stop('refused', 'formal_manifest.csv differs from the MODEL-PREP-04 manifest')
    old = P.run_load().get('freeze')
    snap = old['registry'] if (old and 'registry' not in (refreeze or [])) else G.registry_snapshot(B2.registry())
    rec = G.freeze_record(MANIFEST, rules_all(), snap)
    if old and not refreeze:
        bad = G.verify_freeze(old, MANIFEST, rules_all())
        if bad:
            raise K.Stop('refused', f'a different freeze exists ({bad[:3]}); use --refreeze <part> --reason')
        return dict(frozen=True, unchanged=True)
    if old and refreeze:
        if not reason:
            raise K.Stop('usage', 're-freeze needs --reason')
        bad = G.verify_freeze(old, MANIFEST, rules_all())
        allowed = [f'layer {x} changed' for x in refreeze] + (['rules changed'] if 'rules' in refreeze else []) + (['registry source files changed'] if 'registry' in refreeze else [])
        unreg = [b for b in bad if not any(b.startswith(a) for a in allowed)]
        if unreg:
            raise K.Stop('refused', f'changes outside the registered parts: {unreg[:3]}')
        hist = P.run_load().get('freeze_history') or []
        hist.append(dict(t=K.now(), parts=refreeze, reason=reason, changes=bad, previous_t=old.get('t'), previous_layers=old.get('layers')))
        P.run_put('freeze_history', hist)
    P.run_put('freeze', rec)
    return dict(frozen=True, refreeze=refreeze or [], manifest=rec['manifest'], registry_files=len(snap['files']))


# Check raster equivalence, parameter bindings and source identities before execution.
def st_static() -> dict:
    import mp5_inputs as IN
    import tr3_b2 as B2
    import tr_a4 as A4
    P = K.P0; rows = []; M = manifest()
    eq = IN.equivalence_16()
    rows.append(_row('A1_equivalence_16', eq['ok'] and eq['n'] == 16, 'adapter (template + registered phase) == RealQ.inputs(cand_id) for the 16 MODEL-PREP-03 rasters, every input field', cases=eq['cases']))
    R32 = M.drop_duplicates('raster_key'); r32 = {}
    for r in R32.to_dict('records'):
        inp = IN.inputs_for(r['template_cand_id'], r['family'], IN.phase_of(r))
        pat = np.asarray(inp['pattern'], np.float32)
        r32[f"{r['config']}|b{r['phase_block']}"] = dict(raster=inp['raster_key'] == r['raster_key'], lit=float(inp['lit_px']) == float(r['lit_px']),
                                                         peak=float(np.float32(inp['peak'])) == float(r['peak_f32']), integral=abs(float(pat.sum(dtype=np.float64)) - float(r['integral'])) <= 1e-6)
    rows.append(_row('A1_inputs_32', len(r32) == 32 and all(all(v.values()) for v in r32.values()), '32 new-phase rasters rebuilt through the adapter equal the frozen manifest (key, lit px, peak, integral)', rasters=r32))
    try:
        pp = preps(M); ok = len(pp) == 2450 and len({p['forward_key'] for p in pp.values()}) == 2450
        bad = []
    except K.Stop as e:
        ok, bad = False, [e.msg]
    rows.append(_row('A1_rows_2450', ok, 'all 2450 rows: adapter inputs, resource rule, seed rule, field id and the RECOMPUTED forward key equal the frozen manifest', problems=bad))
    old = set(K.read_csv(K.MP3_OUT / 'manifest.csv').raster_key.dropna())
    q = IN.adapter(); refused = False
    try:
        q.inputs(dict(cand_id_sigma0=M.template_cand_id.iloc[0], family='point'))
    except K.Stop:
        refused = True
    rows.append(_row('A1_no_template_fallback', refused and not (set(M.raster_key) & old), 'no new raster equals a template / MODEL-PREP-03 raster; the adapter refuses candidate-id lookups', overlap=sorted(set(M.raster_key) & old)))
    keys = {tuple(sorted(p['decoder_row'])) for p in preps(M).values()}
    import inspect
    sig = list(inspect.signature(type(q).decode).parameters)
    rows.append(_row('A1_decoder_isolation', keys == {('family', 'input_key', 'radius_mm')} and sig == ['self', 'row', 'method', 'img'],
                     'decoder rows carry only (input_key, radius_mm, family); decode signature unchanged (no PSF / field / noise / truth argument)', keys=[list(k) for k in keys], signature=sig))
    reg = {}
    for r in M[M.family == 'line'].drop_duplicates('raster_key').to_dict('records'):
        inp = IN.inputs_for(r['template_cand_id'], 'line', IN.phase_of(r)); o = inp['meta_full']['olines']
        sc = A4.scan_raster(np.asarray(inp['binary']), o)
        reg[f"{r['config']}|b{r['phase_block']}"] = dict(decoder_ids=[int(x) for x in o['ids']], registry_ids=sc['ids'], lit_unregistered=sc['extra'],
                                                         ok=([int(x) for x in o['ids']] == sc['ids'] and not (set(sc['extra']) & set(sc['ids'])) and json.loads(r['lit_unregistered_lines']) == sc['extra']))
    rows.append(_row('A1_two_registries', len(reg) == 16 and all(v['ok'] for v in reg.values()),
                     'decoder candidate set = the registered line ids (unchanged P4); lit unregistered clipped lines exist only in the evaluation identity registry', rasters=reg))
    fz = P.run_load().get('freeze') or {}
    m4 = K.load_json(K.MP4_OUT / 'run.json')
    rows.append(_row('A3_manifest_frozen', K.sha(MANIFEST) == m4['formal']['manifest_sha'] == (fz.get('manifest') or {}).get('sha256') and (fz.get('manifest') or {}).get('n_rows') == 2450,
                     'formal_manifest.csv = the MODEL-PREP-04 manifest (sha) = the frozen manifest (sha, rows digest, 2450 rows)', manifest=fz.get('manifest')))
    snap_now = G.registry_snapshot(B2.registry())
    same = all(snap_now[k] == fz['registry'][k] for k in ('files', 'phases', 'positions', 'seeds', 'raster_keys'))
    rows.append(_row('A3_registry_snapshot', same, 'phase registry rebuilt once now equals the frozen snapshot (source files + digests); later gates check the frozen source hashes only',
                     n=snap_now['n'], files=len(snap_now['files'])))
    ch = protected_changed(P)
    rows.append(_row('A3_protected_unchanged', not ch, 'protected inputs (MODEL-PREP-03 / 04 records and code, src/, review, candidate table) unchanged', changed=ch))
    bad = G.verify_freeze(fz, MANIFEST, rules_all())
    missing = [f for L in G.layer_files().values() for f in L if not (K.ROOT / f).exists()]
    rows.append(_row('A3_freeze_current', not bad and not missing, 'freeze equals the current layers, manifest, registry sources, rules and caps; every layer file exists', problems=bad, missing=missing))
    s = _proof_put('static', rows)
    if not s['ok']:
        raise K.Stop('refused', f"static: {s['why'][:3]}")
    return s


def _replay_code(layers: dict) -> dict:
    import mp5_chain as CH
    chain = (layers.get('chain') or {}).get(CH.CHAIN_FILE) or (layers.get('entry') or {}).get(CH.CHAIN_FILE)
    return dict(decode=layers.get('decode'), evaluate=layers.get('evaluate'), inputs=layers.get('inputs'), chain=chain)


def st_replay() -> dict:

    import mp5_chain as CH
    P = K.P0; store = K.Store(P, K.CAPS)
    a = G.ledger_audit(P)
    if a['problems'] or a['failures'] or a['unclosed']:
        raise K.Stop('refused', f'ledger not clean before the replay: {a["problems"][:2]} {a["failures"]} {len(a["unclosed"])}')
    if (a['counts'].get('replay_decode') or {}).get('reserved', 0):
        run = P.run_load(); res = run.get('replay')
        if not res:
            raise K.Stop('refused', 'replay calls recorded but no replay record')
        t_rep = max(r['t'] for r in K.Store(P).rows() if r.get('event') == 'reserve' and str(r.get('step', '')).startswith('replay'))
        cands = [dict(t=run['freeze']['t'], layers=run['freeze']['layers'])] + [dict(t=h['previous_t'], layers=h['previous_layers']) for h in (run.get('freeze_history') or []) if h.get('previous_layers')]
        used = max((c for c in cands if c['t'] <= t_rep), key=lambda c: c['t'], default=None)
        now = _replay_code(G.hash_layers())
        if used is None or _replay_code(used['layers']) != now:
            raise K.Stop('refused', 'replay code changed since the replay (or freeze at replay time unknown); the replay cannot be repeated within the budget -> review')
        res = dict(res, attested=dict(t=K.now(), replay_t=t_rep, freeze_t=used['t'], note='proof re-issued from the stored replay record; no new decode / evaluation'))
        P.run_put('replay', res)
    else:
        res = CH.replay(store)
        res['t'] = K.now(); res['code'] = _replay_code(G.hash_layers())
        P.run_put('replay', res)
    a = G.ledger_audit(P)
    rows = [_row('A2_replay_exact', res['ok'], 'adapter decode / evaluation of stored MODEL-PREP-03 images equals the stored arrays, decode meta, metrics, support and complete identity EXACTLY',
                 cases=res['cases']),
            _row('A2_replay_budget', res['n_images'] <= K.REPLAY_MAX_IMAGES and res['n_decodes'] <= 32 and (a['counts'].get('replay_decode') or {}).get('ok') == res['n_decodes']
                 and (a['counts'].get('forward') or {}).get('reserved', 0) == 0, 'replay within 20 images / 32 decodes / 32 evaluations, own ledger groups, no imaging',
                 n_images=res['n_images'], n_decodes=res['n_decodes'], counts={k: v for k, v in a['counts'].items() if k.startswith('replay') or k == 'forward'})]
    s = _proof_put('replay', rows)
    if not s['ok']:
        raise K.Stop('refused', f"replay: {s['why'][:3]}")
    return s


def launch_check(spec: dict, mode: str) -> None:
    P = spec['paths']; store = spec['store']; backend = spec['backend']; why = []
    run = P.run_load(); rules = spec.get('rules') or rules_all()
    why += G.verify_freeze(run.get('freeze'), spec['manifest'], rules, spec.get('layer_override'))
    prts = G.parts(spec['manifest'], rules, run.get('freeze'), results_sha(P), spec.get('layer_override'))
    need = dict(entry=('static', 'replay', 'drills'), main=('static', 'replay', 'drills', 'entry'))[mode]
    for st, (ok, w) in G.proof_state(run, prts, need).items():
        if not ok:
            why.append(f'{st} proof: {w[:2]}')
    ch = spec['protected']() if callable(spec.get('protected')) else protected_changed(P)
    if ch:
        why.append(f'protected inputs changed: {ch[:3]}')
    a = G.ledger_audit(P)
    if a['problems'] or a['failures'] or a['unclosed']:
        why.append(f"ledger: problems {a['problems'][:2]}, failures {a['failures']}, unclosed {len(a['unclosed'])} (review needed before any continuation)")
    if not (P.is_real and getattr(backend, 'name', '') == 'real'):
        why.append('real stages need the real root and the real backend')
    if why:
        raise K.Stop('refused', f'launch refused ({mode}): {why}')


def real_spec() -> dict:
    import mp5_chain as CH
    return dict(paths=K.P0, manifest=MANIFEST, store=K.Store(K.P0, K.CAPS), backend=CH.RealBackend())


def _run(spec, mode: str, case_keys: list) -> dict:
    import mp5_chain as CH
    launch_check(spec, mode)
    P = spec['paths']; M = manifest(spec['manifest']); pp = preps(M)
    log_path = K.drill_base() / f'mp5_run_{mode}.log'
    def log(s):
        with open(log_path, 'a', encoding='utf-8') as fh:
            fh.write(s + '\n')
    res = CH.run_cases([pp[k] for k in case_keys], spec['store'], spec['backend'], lambda: launch_check(spec, mode), log)
    r = P.run_load(); runs = r.get('runs') or []
    runs.append(dict(t=K.now(), mode=mode, n=res['n'], reused=res['reused'], failures=res['failures'], seconds=res['seconds'], timing=res['timing'], ledger=spec['store'].counts(),
                     data_bytes=sum(f.stat().st_size for f in P.data.glob('*.npz')) if P.data.exists() else 0, log=str(log_path)))
    P.run_put('runs', runs)
    if res['failures']:
        raise K.Stop('technical', f"{mode}: {res['failures'][0]['case_id']}: {res['failures'][0]['error']}")
    return dict(n=res['n'], reused=res['reused'], seconds=res['seconds'])


def st_entry(spec=None) -> dict:
    import mp5_drills as DR
    spec = spec or real_spec(); launch_check(spec, 'entry'); M = manifest(spec['manifest']); E = M[M.stage == 'entry']
    if len(E) != ENTRY_EXPECTED['images']:
        raise K.Stop('preflight', f'entry subset {len(E)} images')
    parents = E[E.env == 'NOMINAL'].case_key.tolist(); rest = E[E.env != 'NOMINAL'].case_key.tolist()
    out1 = _run(spec, 'entry', parents)
    pre = DR.entry_parent_checks(spec, M, parents)
    spec['paths'].run_put('entry_parents', pre)
    if not all(r['passed'] for r in pre):
        raise K.Stop('refused', f"entry parents: {[r['check_id'] for r in pre if not r['passed']]}")
    out2 = _run(spec, 'entry', rest)
    rows = DR.entry_checks(spec, M, E.case_key.tolist())
    spec['paths'].run_put('entry', dict(t=K.now(), parents=out1, rest=out2, checks=rows))
    s = _proof_put('entry', rows, spec['paths'])
    if not s['ok']:
        raise K.Stop('refused', f"entry: {s['why'][:3]}")
    return dict(parents=out1, rest=out2, checks=s)


def st_run(spec=None) -> dict:
    spec = spec or real_spec(); launch_check(spec, 'main'); M = manifest(spec['manifest'])
    order =M[M.stage == 'entry'].case_key.tolist() + M[M.stage == 'main'].sort_values(['env'], key=lambda s: s != 'NOMINAL', kind='stable').case_key.tolist()
    return _run(spec, 'main', order)


def st_analyse() -> dict:
    import mp5_analysis as AN
    P = K.P0; M = manifest()
    bad = G.verify_freeze(P.run_load().get('freeze'), MANIFEST, rules_all())
    if bad:
        raise K.Stop('refused', f'freeze: {bad[:3]}')
    T = AN.results_table(P, M); comp = AN.completeness(T, 3626, 2450)
    P.save_csv(T, P.root / 'results.csv')
    rows = [_row('Q_results_complete', comp['ok'], '3626 result rows / 2450 images, unique keys, legal states, valid points <-> errors consistent, readable packs', completeness=comp)]
    try:
        E = AN.contrasts(T); S = AN.summaries(E); Lv = AN.levels(T); Rk = AN.risk(E, T)
        full = pd.concat([E, S, Lv, Rk], ignore_index=True)
        P.save_csv(full, P.root / 'effects.csv')
        no = T[T.state == 'no_output']; Pp = E[E.record_type == 'pair']
        kept = all(((Pp.left_key == k) | (Pp.right_key == k)).any() for k in no.row_key) if len(no) else True
        rows += [_row('Q_effects_one_to_one', True, 'all registered contrasts merged one-to-one (incl. block, rep, seed, own parent); reps -> blocks -> 4 blocks without errors',
                      n_pairs=int(len(Pp)), n_summary=int(len(S)), n_levels=int(len(Lv))),
                 _row('Q_no_output_kept', bool(kept), 'every legal no-output result appears in the registered contrasts (kept in the denominators)', n_no_output=int(len(no))),
                 _row('Q_risk_per_image', isinstance(Rk, pd.DataFrame), 'per-image wrong-id increases against the own parent listed (activation, true lines, segments, unregistered lines)',
                      n_increase=int(len(Rk)), n_activation=(int(Rk.activation.sum()) if len(Rk) else 0))]
    except K.Stop as e:
        rows += [_row(c, False, 'contrasts', error=e.msg) for c in ('Q_effects_one_to_one', 'Q_no_output_kept', 'Q_risk_per_image')]
    s = _proof_put('analysis', rows)
    P.run_put('analysis', dict(t=K.now(), completeness=comp, checks=[dict(check_id=r['check_id'], passed=r['passed']) for r in rows]))
    if not s['ok']:
        raise K.Stop('result_integrity', f"analysis: {s['why'][:3]}")
    return s


def _tables(P=K.P0):
    E = pd.read_csv(P.root / 'effects.csv', low_memory=False); T = pd.read_csv(P.root / 'results.csv', low_memory=False)
    S = E[E.record_type == 'summary']; Lv = E[E.record_type == 'level']; Rk = E[E.record_type == 'risk']
    if len(Rk):
        Rk = Rk.assign(activation=Rk.activation.astype(str).str.lower().isin(['true', '1']))
    return T, E, S, Lv, Rk


def st_figs() -> dict:
    import mp5_figs as FG
    P = K.P0; prts = cur_parts(P)
    ok, why = G4.check_proof((P.run_load().get('proofs') or {}).get('analysis'), G.REQUIRED['analysis'], G4.binding(prts, G.PROOF_LAYERS['analysis']))
    if not ok:
        raise K.Stop('refused', f'figures need a valid analysis proof: {why[:3]}')
    T, E, S, Lv, Rk = _tables(P)
    out = FG.all_figures(P, T, S, Lv, Rk, P.figs)
    files = {k: (K.rel(v) if isinstance(v, str) and v.endswith('.png') else v) for k, v in out.items()}
    exp = 6 if out.get('failure_case') else 5
    pngs = [v for k, v in out.items() if k.startswith('F') and v]
    rows = [_row('V_figures', len(pngs) == exp and all(K.Path(v).exists() for v in pngs), 'six German figures (five if no wrong-id increase exists) from the checked tables', files=files)]
    P.run_put('figures', files)
    s = _proof_put('figures', rows)
    return dict(figures=files, checks=s)


def st_selfcheck() -> dict:
    import mp3_analysis as A3
    import mp5_analysis as AN
    P = K.P0; T = pd.read_csv(P.root / 'results.csv', low_memory=False)
    picks = AN.selfcheck_pick(T)
    res = A3.selfcheck(P, T, picks)
    rows = [_row('S_recomputed', len(res) >= 9 and all(x['ok'] for x in res), 'stratified rows (D0 / DF / P4 x B / M10 / M24 + no-output + w > 0) recomputed from stored arrays with separate code',
                 results=res)]
    P.run_put('selfcheck', res)
    s = _proof_put('selfcheck', rows)
    return dict(checks=s, picks=picks)


def st_final(spec=None) -> dict:
    spec = spec or dict(paths=K.P0, manifest=MANIFEST)
    P = spec['paths']; run = P.run_load(); rules = spec.get('rules') or rules_all()
    fz = G.verify_freeze(run.get('freeze'), spec['manifest'], rules, spec.get('layer_override'))
    prts = G.parts(spec['manifest'], rules, run.get('freeze'), results_sha(P), spec.get('layer_override'))
    stages = spec.get('stages') or ('static', 'replay', 'drills', 'entry', 'analysis', 'figures', 'selfcheck')
    proofs = G.proof_state(run, prts, stages)
    exp = dict(spec.get('expected_calls') or EXPECTED_CALLS)
    rp = run.get('replay') or {}
    if 'replay' in stages:
        exp.update(replay_decode=rp.get('n_decodes', -1), replay_evaluation=rp.get('n_evaluations', -1))
    rows, probs = G4.ledger_rows(P.ledger); a = G4.ledger_audit(rows, K.CAPS, exp); a['problems'] = probs + a['problems']
    comp = (run.get('analysis') or {}).get('completeness') if spec.get('completeness') is None else spec['completeness']
    calls = dict(render=a['counts'].get('forward', {}).get('reserved', 0), decode=a['counts'].get('decode', {}).get('reserved', 0), evaluation=a['counts'].get('evaluation', {}).get('reserved', 0))
    ch = spec['protected']() if callable(spec.get('protected')) else protected_changed(P)
    state = dict(freeze_problems=fz, proofs=proofs, ledgers={'MODEL-PREP-05': a}, calls=calls, calls_expected=None, protected_changed=ch, completeness=comp)
    st = G4.final_status(state, 'run')
    fin = dict(t=K.now(), status=st['status'], reasons=st['reasons'], gate=st['gate'], calls=calls,
               ledger=dict(counts=a['counts'], problems=a['problems'][:5], failures=a['failures'], unclosed=len(a['unclosed']), incomplete=a['incomplete_groups']),
               proofs={k: dict(ok=v[0], why=v[1][:3]) for k, v in proofs.items()}, freeze_problems=fz, protected_changed=ch, completeness=comp)
    if P.is_real:
        fin['outputs'] = {K.rel(p): K.sha(p) for p in sorted(P.root.rglob('*')) if p.is_file() and p.name != 'run.json' and '_sandbox' not in p.parts}
    P.run_put('final', fin)
    if st['exit'] != 0:
        raise K.Stop('technical' if st['status'] in ('BLOCKED', 'NEEDS_REVIEW') else 'result_integrity', f"{st['status']}: {st['reasons'][:4]}")
    return {k: v for k, v in fin.items() if k != 'outputs'}


def st_drills() -> dict:
    import mp5_drills as DR
    rows = DR.all_drills()
    K.P0.run_put('drills', dict(t=K.now(), rows=rows))
    s = _proof_put('drills', rows)
    if not s['ok']:
        raise K.Stop('refused', f"drills: {s['why'][:3]}")
    return s


STAGES = dict(register=st_register, static=st_static, replay=st_replay, drills=st_drills, entry=st_entry, run=st_run, analyse=st_analyse, figs=st_figs,
              selfcheck=st_selfcheck, final=st_final)


def run_main(argv: list, spec: dict | None = None, stages: dict | None = None) -> int:

    stages = STAGES if stages is None else stages
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=sorted(stages)); ap.add_argument('--refreeze', nargs='*'); ap.add_argument('--reason')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code else K.EXIT['usage']
    t0 = time.time()
    try:
        fn = stages[a.stage]
        if a.stage == 'register':
            out = fn(a.refreeze, a.reason)
        elif a.stage in ('entry', 'run', 'final'):
            out = fn(spec)
        else:
            out = fn()
        print(json.dumps(K.clean(dict(stage=a.stage, seconds=round(time.time() - t0, 1), result=out)), ensure_ascii=False, default=str)[:6000])
        return 0
    except K.Stop as e:
        print(json.dumps(dict(stage=a.stage, stop=e.code, msg=e.msg[:2500]), ensure_ascii=False)); return int(e.exit_code)
    except KeyboardInterrupt:
        print(json.dumps(dict(stage=a.stage, stop='interrupted'))); return K.EXIT['interrupted']
    except Exception as e:
        print(json.dumps(dict(stage=a.stage, stop='exception', msg=repr(e)[:1500]))); return K.EXIT['technical']


if __name__ == '__main__':
    sys.exit(run_main(sys.argv[1:]))
