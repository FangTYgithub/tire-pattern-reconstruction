# Coordinate joint-study execution, analysis and figures.

"""S5-ALIGN-05 entry (run from the project root).

  python -B scripts/surface_revision/s5_align05/s5u_entry.py compat          ALIGN-04 compatibility mapping (32 stage records, hashes of its final check)
  python -B scripts/surface_revision/s5_align05/s5u_entry.py auth            authorisation: manifest minus the verified complete set (2872 keys, caps)
  python -B scripts/surface_revision/s5_align05/s5u_entry.py drills          A3 targeted counterexamples (isolation only, cleaned)
  python -B scripts/surface_revision/s5_align05/s5u_entry.py freeze          code, closure, parameter files, statistics / selection rules
  python -B scripts/surface_revision/s5_align05/s5u_entry.py run --select remaining     formal run (no caps / keys / store options in real mode)
  python -B scripts/surface_revision/s5_align05/s5u_entry.py analyze         complete-set gate -> tables (or a labelled progress record and exit 3)
  python -B scripts/surface_revision/s5_align05/s5u_entry.py figs            figures from the COMPLETE tables only
  python -B scripts/surface_revision/s5_align05/s5u_entry.py final

Isolation options (stub / sentinel modes only, store below outputs/surface_revision/s5_align05/_isolation): --mode stub|sentinel --store DIR
--keys JSON --caps JSON --fail-after N; analyze --store DIR. Exit codes: 0 ok, 2 technical, 3 refused, 4 mismatch, 5 corrupt, 6 lock, 7 forbidden,
8 version, 12 guard, 14 quota, 15 auth, 16 freeze, 17 platform, 18 unbound.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import s5u_common as K

import numpy as np
import pandas as pd

import s5u_bind as BD
import s5r_bind as B
import s5r_store as S
import s5t_view as V

CLOSURE_ENTRIES = ['rdy_decode', 'rdy_eval', 'rdy_inputs', 'rdy_platform', 'rdy_figs', 'rdy_common', 's3c_dev', 's3c_common', 'adp_surf02', 'adp_binding', 'adp_patterns',
                   's1r_eval', 's1r_common', 's2_common', 'm3_virtual_camera', 'm4_reconstruction', 'm4_lines', 'surface_eval_r3', 'r5_detect']
PARAM_FILES = ['outputs/surface_revision/sections_adapter/binding.json', 'outputs/surface_revision/sections_adapter/candidates_s1s5.csv', 'outputs/surface_revision/s345_ready/plan_s345.csv',
               'outputs/surface_revision/s345_ready/freeze_s345_ready.json', 'outputs/surface_revision/s1_real01/domain/global_domain_step0p50.json',
               'outputs/surface_revision/s1_real01/domain/global_domain_step0p50.npz', 'outputs/surface_revision/surf02/release/dependencies_surf02_v1_1.json',
               'outputs/surface_revision/s5_align02/manifest_next.csv', 'outputs/surface_revision/s345_ready/results/S5-main_cases.csv',
               'outputs/surface_revision/s345_ready/results/S5_cases_supplement.csv', 'outputs/surface_revision/s345_ready/results/S5_interactions.csv',
               'outputs/surface_revision/s5_align02/_parts/C_df_interactions.csv', 'outputs/surface_revision/s5_align02/_parts/A_interactions_posthoc_63.csv',
               'outputs/surface_revision/s5_align03/run.json', 'outputs/surface_revision/s5_align04/run.json']
STAT_RULES = dict(order='repetitions averaged inside a phase -> four-cell differences -> equal-weight mean of 8 phases (sigma 0: one repetition)',
                  resampling='PCG64(2026101204), one 10000 x 8 phase-index matrix shared by every row, front end and contrast; 5 / 95 % descriptive percentiles',
                  contrasts='every sigma (0, 20, 40, 60, 80, 100) + the registered sigma 40 minus sigma 0 contrast (explicit; not produced by s5c_stats for six levels)',
                  sets='dots_parallel (r 1.0 / 1.5, 6x4 -> 10x4), dots_perp (8x4 -> 8x6), lines (w 0.5 / 1.5, 4.45 -> 8), lines_6.3_8 (posthoc), line width differences',
                  metrics='glob_Ctau, loc_Ctau, glob_Cvalid, glob_pt_rmse, sup_surf_rmse', front_ends='D0 and DF side by side for dots; DF - D0 differences; lines P4',
                  undefined='a missing / undefined required value makes that phase and the row undefined; no output: coverage 0, errors undefined',
                  risks='per case w/(c+w), u/(c+w+u); condition summary point-pooled over all accepted points; zero denominator undefined')


def _code_files() -> list:
    base = K.ROOT / 'scripts' / 'surface_revision'
    return sorted([K.rel(p) for p in (base / 's5_align05').glob('s5u_*.py')] + [K.rel(base / 's5_align04' / f) for f in ('s5t_common.py', 's5t_platform.py', 's5t_view.py')]
                  + [K.rel(base / 's5_align03' / f) for f in ('s5r_common.py', 's5r_gates.py', 's5r_store.py', 's5r_bind.py', 's5r_main.py')]
                  + [K.rel(base / 's5_align02' / f) for f in ('s5c_stats.py', 's5c_common.py')])


def freeze_content() -> dict:
    import s5t_platform as PF
    import s5u_figs as FG
    PF.init_platform()
    cl = K.closure(CLOSURE_ENTRIES)
    files = sorted(set(_code_files()) | set(cl) | set(PARAM_FILES))
    a = BD.load_auth()
    return dict(files={f: K.sha(K.ROOT / f) for f in files}, closure=cl, auth_id=a['auth_id'], stat_rules=STAT_RULES, selection_rules=FG.SEL_RULES,
                note='entry, statistics and figure code, static import closure of the computation path, parameter / input files, rules; frozen before the first computation')


def do_freeze() -> dict:
    a = BD.load_auth(); new = freeze_content(); r = K.run_load(); old = r.get('freeze')
    led = BD.Ledger5(K.LEDGER, a['auth_id'], K.CAPS, set()); used = sum(led.counts()['reserved'].values())
    if old and K.digest(old['files']) != K.digest(new['files']):
        if used:
            raise K.Stop('freeze', f'{used} real reservations exist under {a["auth_id"]}: the freeze cannot be replaced')
        K.run_put('freeze_history', r.get('freeze_history', []) + [dict(replaced_at=K.now(), digest=old['digest'])])
    out = dict(new, digest=K.digest(new['files']), t=K.now(), real_reservations_at_freeze=used)
    K.run_put('freeze', out); K.run_put('selection_rules', new['selection_rules']); K.run_put('stat_rules', STAT_RULES)
    return dict(n_files=len(out['files']), n_closure=len(out['closure']), digest=out['digest'])


def cmd_run(select=None, mode=None, store=None, caps=None, keys_file=None, fail_after=None) -> dict:
    real = mode in (None, 'real')
    if real and (store or caps or keys_file or fail_after is not None):
        raise K.Stop('auth', 'the real mode accepts no store / caps / keys / failure options: scope and quotas come only from the bound authorisation')
    if real and select != 'remaining':
        raise K.Stop('scope', 'the real mode needs --select remaining')
    if not real and (store is None or K.ISO.resolve() not in Path(store).resolve().parents):
        raise K.Stop('guard', 'stub / sentinel modes run only below the isolation directory')
    pf = BD.preflight('run'); a = pf['auth']; pr = S.platform_refs(); rows = pf['rows']
    if real:
        import s5t_platform as PF
        pc = PF.platform_check()
        if not pc['ok']:
            raise K.Stop('platform', 'platform initialisation / signatures', detail=pc['problems'])
        ex = PF.RealQ(); Mi = pf['M'].set_index('manifest_key'); bad = {}
        for r in rows:
            c = ex.check_inputs(r, Mi.loc[r['manifest_key']].to_dict())
            if not c['ok']:
                bad[r['manifest_key']] = c
        if bad:
            raise K.Stop('refused', f'{len(bad)} constructed inputs differ from the registered rasters / peaks / integrals', detail=dict(list(bad.items())[:5]))
        ledger = BD.Ledger5(K.LEDGER, a['auth_id'], K.CAPS, {r['manifest_key'] for r in rows}); st = pf['store']
        closure = set(K.run_load()['freeze']['closure']) | set(K.run_load()['freeze']['files'])
        pf['checks']['inputs'] = dict(ok=True, n=len(rows))
    else:
        sel = [tuple(x) for x in json.loads(Path(keys_file).read_text(encoding='utf-8'))] if keys_file else []
        ak = set(map(tuple, pf['keys'])); extra = [k for k in sel if k not in ak]
        if extra:
            raise K.Stop('auth', f'{len(extra)} keys outside the authorisation', detail=[list(k) for k in extra[:3]])
        rows = BD.rows_of(sel, pf['M']); led_p = Path(store).parent / 'ledger.jsonl'; iso_auth = 'ISO-' + a['auth_id']
        cp = json.loads(Path(caps).read_text(encoding='utf-8')) if caps else K.CAPS
        ledger = BD.Ledger5(led_p, iso_auth, cp, {r['manifest_key'] for r in rows})
        st = BD.BoundStore(Path(store), K.OUT, BD.Registry.load(None, iso_auth, led_p)); closure = None
        ex = S.Stub(fail_after=fail_after) if mode == 'stub' else S.Forbidden()
    before = ledger.counts()
    log = BD.run_formal(st, rows, ex, pr, ledger, closure_files=closure)
    return dict(mode=mode or 'real', checks=pf['checks'], log=log, ledger_before=before['reserved'], ledger_after=ledger.counts(), executor_calls=len(getattr(ex, 'calls', [])))


def build_view(pf: dict, store=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    cp3 = K.load_json(K.A3RUN)['compat']
    old = B.views(cp3)['main']
    M = pf['M']; allk = [tuple(x) for x in M[M.status == 'to_image'][['combo', 'surface', 'phase_label', 'sigma', 'rep']].itertuples(index=False, name=None)]
    rows = BD.rows_of(allk, M); st = store or pf['store']
    R, probs = BD.view_rows(st, rows, S.platform_refs())
    return old, pd.DataFrame(R), M, probs


def cmd_analyze(store=None) -> dict:
    import s5u_stats as ST
    if store is not None and K.ISO.resolve() not in Path(store).resolve().parents:
        raise K.Stop('guard', 'an alternative store is allowed only below the isolation directory')
    pf = BD.preflight('analysis')
    st = None
    if store is not None:
        a = pf['auth']; led_p = Path(store).parent / 'ledger.jsonl'
        st = BD.BoundStore(Path(store), K.OUT, BD.Registry.load(None, 'ISO-' + a['auth_id'], led_p))
    old, new, M, probs = build_view(pf, st)
    bad_chain = [p for p in probs if p['status'] not in ('todo',)]
    if bad_chain:
        raise K.Stop('corrupt', 'records failing identity / content binding - no tables', detail=bad_chain[:10])
    D = pd.concat([old, new], ignore_index=True, sort=False)
    full = V.expected_methods(K.K3.expected_keys())
    g = V.strict_gate(D, full, 'full', 'formal_main')
    prog = dict(images_present=int(D.image_key.nunique()), images_planned=3960, method_records_present=len(D), method_records_planned=len(full),
                new_method_records_present=len(new), new_method_records_planned=len(full) - 1728, by_check=g.get('by_check'), t=K.now())
    if not g['ok']:
        if store is None:
            K.run_put('progress', dict(prog, status='PARTIAL - not a complete delivery'))
        raise K.Stop('refused', 'complete formal set not complete or failing the result gate - progress only', detail=prog)
    if store is not None:
        return dict(iso=True, gate=g['ok'])
    C = ST.case_table(old, new, M)
    reg = ST.regression_old_slice(C)
    if not reg['ok']:
        raise K.Stop('refused', 'old-slice regression failed - no tables', detail=reg)
    Sm = ST.condition_summary(C); E = ST.effects(C); FC = ST.frontend_conditions(C)
    E['record_type'] = E.record_type.fillna('interaction')
    PE = pd.concat([E.assign(table='four_cell_and_frontend_effects'), FC.assign(table='frontend_condition')], ignore_index=True, sort=False)
    files = {}
    for name, df in (('case_results.csv', C), ('condition_summary.csv', Sm), ('paired_effects.csv', PE)):
        p = K.save_csv(df, K.OUT / name); files[K.rel(p)] = K.sha(p)
    sc = ST.selfcheck(C)
    n_new = int((C.source == 'S5-ALIGN-04/05 formal store').sum())
    out = dict(status='COMPLETE', gate={k: v for k, v in g.items() if k != 'reasons'}, files=files, n_cases=len(C), n_images=int(C.manifest_key.nunique()),
               n_new_records=n_new, by_front_end=C.front_end.value_counts().to_dict(), states=C.state.value_counts().to_dict(), regression_old_slice=reg,
               effects_rows=E.record_type.value_counts().to_dict(), sigma_levels=sorted(E.sigma.astype(str).unique().tolist()), selfcheck=sc, t=K.now())
    K.run_put('analysis', out)
    return {k: v for k, v in out.items() if k not in ('selfcheck',)}


def cmd_figs() -> dict:
    import s5u_figs as FG
    pf = BD.preflight('figures'); an = K.run_load().get('analysis') or {}
    if an.get('status') != 'COMPLETE':
        raise K.Stop('refused', 'no COMPLETE analysis: final figures are not produced from partial results')
    for f, h in an['files'].items():
        if K.sha(K.ROOT / f) != h:
            raise K.Stop('corrupt', f'{f} differs from the hash registered by the analysis')
    C = K.read_csv(K.OUT / 'case_results.csv'); Sm = K.read_csv(K.OUT / 'condition_summary.csv'); PE = K.read_csv(K.OUT / 'paired_effects.csv')
    E = PE[PE.table == 'four_cell_and_frontend_effects'].copy(); FC = PE[PE.table == 'frontend_condition'].copy()
    sel = FG.select_cases(C)
    for fam in ('point', 'line'):
        for c in sel[fam]['cells']:
            _verify_series(pf, C, c['series'], c['front_end'])
    _verify_series(pf, C, sel['diagnostic']['series'], 'P4')
    files = FG.f_range(Sm, 'point', K.FIGS) + FG.f_range(Sm, 'line', K.FIGS) + FG.f_fourcell(E, K.FIGS) + FG.f_frontend(FC, C, K.FIGS) + FG.f_cases(C, sel, K.FIGS)
    out = dict(files={f: K.sha(K.ROOT / f) for f in files}, selection=sel, t=K.now())
    K.run_put('figures', out)
    return dict(n_files=len(files), selection={k: v for k, v in sel.items()})


def _verify_series(pf, C, series, fe):
    a3 = K.load_json(K.A3RUN)['compat']['records']; by_eval = {r['eval']['json'].replace('.json', '.npz'): r['eval']['npz_sha256'] for r in a3.values()}
    for key in series:
        r = C[(C.combo == key[0]) & (C.surface == key[1]) & (C.phase_label == key[2]) & (C.sigma == float(key[3])) & (C.rep == int(key[4])) & (C.front_end == fe)].iloc[0]
        if str(r.source).startswith('S5-ALIGN-04/05'):
            stage_key = f'{r.image_key}__{fe}'; b = pf['store'].reg.m.get(('eval', stage_key))
            if b is None or K.sha(K.ROOT / r.eval_record) != b['npz_sha256']:
                raise K.Stop('corrupt', f'{r.eval_record} not bound / changed')
        elif K.sha(K.ROOT / r.eval_record) != by_eval.get(r.eval_record):
            raise K.Stop('corrupt', f'{r.eval_record} differs from its S5-ALIGN-03 binding')


def entry(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('cmd')
    for o in ('--select', '--mode', '--store', '--caps', '--keys'):
        ap.add_argument(o)
    ap.add_argument('--fail-after', type=int, default=None)
    a = ap.parse_args(argv); rec = dict(cmd=a.cmd, argv=sys.argv[1:] if argv is None else argv, t=K.now()); t0 = time.time()
    try:
        if a.cmd == 'compat':
            r = BD.compat_align04(); r = dict(n_records=r['n_records'], a4_ledger_events=r['a4_ledger_events'])
        elif a.cmd == 'auth':
            r = BD.make_auth()
        elif a.cmd == 'freeze':
            r = do_freeze()
        elif a.cmd == 'run':
            r = cmd_run(a.select, a.mode, a.store, a.caps, a.keys, a.fail_after)
        elif a.cmd == 'analyze':
            r = cmd_analyze(a.store)
        elif a.cmd == 'figs':
            r = cmd_figs()
        elif a.cmd == 'drills':
            import s5u_drills as D
            r = D.run_all()
            if not r['all_ok']:
                raise K.Stop('refused', 'a drill did not behave as required', detail=r['summary'])
        elif a.cmd == 'final':
            import s5u_drills as D
            r = D.final()
        else:
            print(__doc__); return K.EXIT['refused']
    except K.Stop as e:
        rec.update(exit=K.EXIT.get(e.code, 2), code=e.code, msg=e.msg[:400], seconds=round(time.time() - t0, 1))
        print(json.dumps(dict(code=e.code, msg=e.msg, detail=K.clean(e.detail)), ensure_ascii=False, default=str)[:3000], file=sys.stderr)
        _log(rec); return K.EXIT.get(e.code, 2)
    except Exception as e:
        import traceback
        rec.update(exit=K.EXIT['technical'], code='technical', msg=repr(e)[:400], trace=traceback.format_exc()[-1500:], seconds=round(time.time() - t0, 1))
        print(json.dumps(rec, ensure_ascii=False, default=str)[:3000], file=sys.stderr)
        _log(rec); return K.EXIT['technical']
    rec.update(exit=0, seconds=round(time.time() - t0, 1), result=_short(r)); _log(rec)
    print(json.dumps(K.clean(r), ensure_ascii=False, default=str)[:5000])
    return 0


def _short(r):
    if isinstance(r, dict):
        return {k: v for k, v in r.items() if k not in ('checks', 'selection', 'files', 'stages')}
    return r


def _log(rec):
    try:
        K.run_append('invocations', rec)
        r = K.run_load(); ir = r.setdefault('sources_read', {}); seen = []
        for k, v in K.K3.READ.items():
            if k in ir and ir[k] != v:
                seen.append(k)
            ir.setdefault(k, v)
        if seen:
            r.setdefault('source_changes_seen', []).extend(seen[:20])
        K.save_json(K.RUN, r)
    except Exception:
        pass


if __name__ == '__main__':
    _ = np
    sys.exit(entry())
