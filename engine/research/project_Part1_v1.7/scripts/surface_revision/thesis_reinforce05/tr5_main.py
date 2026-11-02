# Coordinate new-phase registration, execution and analysis.

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback

import tr5_common as K

PERMIT = K.OUT / 'B2_PERMIT.json'
WORKERS = 12
CHECKPOINT = dict(point=dict(surface='M24_curv', block=0, sigma=60.0), line=dict(surface='M24_curv', block=0, sigma=40.0))
STOP_RULES = dict(
    technical=['unregistered parameter / seed / surface / raster / resource / method identity', 'input or dependency drift', 'budget insufficient or exceeded',
               'abnormal exit, lost / corrupt result, empty return, failed save, evaluation integrity gate, non-unique pairing',
               'a change of detector, truth rule or scientific rule would be needed'],
    not_stopping=['legal no_output', 'coverage near 0', 'more w / u', 'reversed order', 'interval across 0', 'D0 / DF opposite'],
    after_stop='keep completed data, errors and consumed budget; no reset, no extra real calls, no design change')


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def checkpoint_rows(M) -> list:
    cp = []
    for fam, c in CHECKPOINT.items():
        g = M[(M.family == fam) & (M.surface == c['surface']) & (M.block == c['block']) & (M.sigma == c['sigma'])]
        cp += g.sort_values(['config', 'noise_k']).to_dict('records')
    if len(cp) != 12:
        raise K.Stop('preflight', f'checkpoint selection has {len(cp)} images (expected 12)')
    return cp


def freeze_obj(M, H: dict, closure: dict, fps: dict, claims) -> dict:
    import tr5_analysis as A
    import tr5_chain as CH
    import tr5_manifest as MF
    rules = dict(chain=CH.__doc__.strip().splitlines()[0], analysis=A.RULES, stop=STOP_RULES, checkpoint=CHECKPOINT, workers=WORKERS)
    return dict(manifest=MF.manifest_digest(M), n_rows=len(M), numeric_code=K.code_identity(), analysis_code=K.code_identity(K.ANALYSIS_FILES), closure=closure,
                inputs={k: v['sha256'] for k, v in H.items()}, method_fingerprints=fps, budget=dict(K.CAPS), rules=rules, rules_sha=K.digest(rules),
                claims=claims.to_dict('records'), claims_sha=K.digest(claims.to_dict('records')), task_book_sha=K.sha(K.TASK),
                checkpoint=[r['forward_key'] for r in checkpoint_rows(M)], layout_code=K.code_identity(K.LAYOUT_FILES))


def make_permit(freeze: dict, checks: dict) -> dict:
    p = dict(task=K.TASK_ID, task_book=K.rel(K.TASK), task_book_sha=freeze['task_book_sha'], manifest=freeze['manifest'], n_images=freeze['n_rows'],
             numeric_code_digest=K.digest(freeze['numeric_code']), closure_digest=K.digest(freeze['closure']), inputs_digest=K.digest(freeze['inputs']),
             method_fingerprints=freeze['method_fingerprints'], budget=freeze['budget'], rules_sha=freeze['rules_sha'], claims_sha=freeze['claims_sha'],
             g0_checks_digest=K.digest(checks), issued=K.now(), authorised_by='task book THESIS-REINFORCE-05 §1 (core run after G0 and the checkpoint, no further confirmation)',
             scope='B2 core only: 480 registered images, D0 / DF / P4, 1 ray per projector pixel; no B1 / 513-DF rerun, no optional 108, no 3x3 / 5x5, no new conditions')
    p['permit_id'] = K.digest(p)
    return p


def permit_problems(permit, freeze: dict, M) -> list:
    import tr5_manifest as MF
    if not permit:
        return ['no permit']
    probs = []
    if K.digest({k: v for k, v in permit.items() if k != 'permit_id'}) != permit.get('permit_id'):
        probs.append('permit digest invalid')
    if permit.get('task') != K.TASK_ID:
        probs.append(f"permit task {permit.get('task')!r}")
    if permit.get('task_book_sha') != K.sha(K.TASK) or permit.get('task_book_sha') != freeze.get('task_book_sha'):
        probs.append('task book changed')
    if permit.get('manifest') != MF.manifest_digest(M) or permit.get('manifest') != freeze.get('manifest') or permit.get('n_images') != 480:
        probs.append('manifest differs')
    if permit.get('numeric_code_digest') != K.digest(K.code_identity()) or permit.get('numeric_code_digest') != K.digest(freeze.get('numeric_code')):
        probs.append('numerical / analysis code changed since G0')
    if permit.get('closure_digest') != K.digest(freeze.get('closure')) or permit.get('inputs_digest') != K.digest(freeze.get('inputs')):
        probs.append('closure / inputs differ from the freeze')
    if permit.get('budget') != dict(K.CAPS) or freeze.get('budget') != dict(K.CAPS):
        probs.append('budget differs from the task book')
    if permit.get('rules_sha') != freeze.get('rules_sha') or permit.get('method_fingerprints') != freeze.get('method_fingerprints') or permit.get('claims_sha') != freeze.get('claims_sha'):
        probs.append('rules / method fingerprints / claims differ')
    for k in ('scope', 'authorised_by', 'g0_checks_digest', 'issued'):
        if not permit.get(k):
            probs.append(f'permit field {k} missing')
    return probs


def preflight(M, permit, freeze: dict, platform_fn=None, binding_fn=None, inputs_fn=None, closure_fn=None) -> dict:
    import tr5_bind as BD
    probs = permit_problems(permit, freeze, M)
    if probs:
        raise K.Stop('refused', f'permit refused: {probs}')
    try:
        pl = (platform_fn or BD.platform_checks)()
    except Exception as e:
        raise K.Stop('preflight', f'SURF-02 release / binding check failed: {e}') from None
    if not pl.get('passed'):
        raise K.Stop('preflight', f"SURF-02 release / binding / domains not as frozen (drift {pl.get('drift')})")
    cl = (closure_fn or BD.closure_check)(freeze['closure'])
    if not cl['passed']:
        raise K.Stop('preflight', f"code closure changed: {cl['changed'][:5]} / not frozen: {cl['not_frozen_but_loaded'][:5]}")
    H = (inputs_fn or (lambda: BD.hash_inputs(BD.input_files())))()
    diff = sorted(k for k in set(H) | set(freeze['inputs']) if (H.get(k) or {}).get('sha256') != freeze['inputs'].get(k))
    if diff:
        raise K.Stop('preflight', f'input files differ from the freeze: {diff[:6]}')
    b = (binding_fn or BD.method_bindings)(M)
    if not b['passed'] or b['fingerprints'] != freeze['method_fingerprints']:
        raise K.Stop('preflight', 'method binding differs from the freeze')
    return dict(passed=True, fps=b['fingerprints'], platform=dict(version=pl.get('version'), drift=pl.get('drift')), closure=cl['n_frozen'], inputs=len(H), t=K.now())


def g0() -> dict:
    import tr5_analysis as A
    import tr5_bind as BD
    import tr5_chain as CH
    import tr5_claims as CL
    import tr5_drills as DR
    import tr5_manifest as MF
    t0 = time.time()
    if (K.OUT / 'ledger.jsonl').exists() or K.DATA.exists():
        raise K.Stop('refused', 'G0 must run before any real call (a real ledger / data folder exists)')
    mp = BD.model_prep_status()
    M, rasters = MF.build_manifest(); cov = MF.coverage_check(M); ar = MF.already_run(M); log(f'manifest {len(M)} rows, coverage {cov["passed"]}, collisions {ar["conflicts"]}')
    ctx_ok = 0
    for row in M.to_dict('records'):
        ctx = MF.context(row); CH.forward_id(row, ctx); ctx_ok += 1
    pl = BD.platform_checks(); mb = BD.method_bindings(M); H = BD.hash_inputs(BD.input_files()); rc = BD.recorded_checks(H)
    log(f"platform {pl['passed']}, methods {mb['passed']}, inputs {rc['passed']} ({rc['n_refs']} refs / {rc['n_unique_files']} files)")
    claims = CL.registry(); csum = CL.claims_summary(claims)
    K.save_csv(M, K.OUT / 'manifest.csv')
    closure = BD.code_closure()
    freeze = freeze_obj(M, H, closure, mb['fingerprints'], claims)
    pre = dict(manifest=cov['passed'], no_collision=ar['passed'], contexts=(ctx_ok == 480), platform=pl['passed'], methods=mb['passed'], inputs=rc['passed'])
    cand = make_permit(freeze, pre)
    dr = DR.run_all(M, mb['fingerprints'], freeze, cand); log(f"drills {dr['passed']}: {dr['groups_passed']}")
    cl_after = BD.closure_check(closure)
    real_untouched = not (K.OUT / 'ledger.jsonl').exists() and not K.DATA.exists()
    checks = dict(pre, drills=dr['passed'], closure_after_drills=cl_after['passed'], real_calls_zero=real_untouched)
    passed = all(checks.values())
    lines = {rk: dict(lit_unregistered=r['lit_unregistered'], registered=r['registered_ids'], lit_lines=r['lit_lines'], olines=r['olines'])
             for rk, r in rasters.items() if 'lit_unregistered' in r}
    out = dict(passed=passed, checks=checks, t_start=K.now(), minutes=round((time.time() - t0) / 60, 1), model_prep=mp, coverage=cov, collisions=ar,
               contexts_assembled=ctx_ok, platform={k: v for k, v in pl.items() if k != 'surfaces'}, methods=mb['methods'], inputs=rc,
               line_registration=lines, claims_summary=csum, drills=dr, closure_after_drills=cl_after, freeze=freeze,
               manifest=dict(file='manifest.csv', sha256=K.sha(K.OUT / 'manifest.csv'), rows=len(M)), counts=dict(camera_render=0, decode=0, evaluation=0))
    if passed:
        permit = make_permit(freeze, checks); K.save_json(PERMIT, permit); out['permit_id'] = permit['permit_id']
    K.run_put('g0', out)
    if not passed:
        raise K.Stop('refused', f'G0 not passed: {[k for k, v in checks.items() if not v]}')
    return dict(passed=True, checks=checks, permit_id=out['permit_id'], minutes=out['minutes'])


def _schedule_or_stop(rows, store, fps, what):
    import tr5_chain as CH
    res = CH.schedule(rows, store, fps, backend='real', workers=WORKERS, log=log)
    if res['failures']:
        f = res['failures'][0]
        K.run_put(f'stop_{what}', dict(res, done=None, t=K.now()))
        code = 'result_integrity' if f.get('kind') in ('result_integrity', 'integrity') else ('interrupted' if f.get('kind') == 'interrupted' else 'technical')
        raise K.Stop(code, f"{what}: {len(res['failures'])} failure(s); first {f['forward_key']}: {f.get('error')}")
    return res


# Run the registered new-phase set after checking inputs and resumable checkpoints.
def run_stage_b() -> dict:
    import tr5_analysis as A
    import tr5_bind as BD
    import tr5_chain as CH
    import tr5_manifest as MF
    r = K.run_load(); g = r.get('g0') or {}
    if not g.get('passed'):
        raise K.Stop('refused', 'G0 not passed')
    M, _ = MF.build_manifest(); permit = K.load_json(PERMIT) if PERMIT.exists() else None
    if permit and permit.get('permit_id') != g.get('permit_id'):
        raise K.Stop('refused', 'permit file is not the one issued by G0')
    pf = preflight(M, permit, g['freeze']); fps = pf['fps']; log('preflight passed')
    runs = r.get('b_runs') or []; store = K.Store(K.OUT); runs.append(dict(t=K.now(), preflight=pf, counts_before=store.counts())); K.run_put('b_runs', runs)
    rows = M.to_dict('records'); cp = checkpoint_rows(M); cpk = {x['forward_key'] for x in cp}
    if not (K.run_load().get('checkpoint') or {}).get('passed'):
        res = _schedule_or_stop(cp, store, fps, 'checkpoint')
        comp = CH.completeness(cp, store.root, fps)
        R = A.results(cp, store.root, fps)
        P = [A.method_pairs(R, c) for c in MF.CONTRASTS]; FP = A.frontend_pairs(R)
        npairs = {c: len(p) for c, p in zip(MF.CONTRASTS, P)}
        import pandas as pd
        PP = pd.concat(P, ignore_index=True)
        BP = A.blocks(PP, ('contrast', 'family', 'method', 'surface', 'sigma'))
        CP = A.cells(BP, ('contrast', 'method', 'surface', 'sigma'), lambda x: MF.CONTRASTS[x['contrast']][0], None)
        prev = K.OUT / '_checkpoint_preview'
        try:
            import tr5_fig as F
            sel = dict(F4=A.select_cases(PP, {}, 'C1', 'D0', 60.0), F5=A.select_cases(PP, {}, 'C5', 'P4', 40.0))
            figs = F.make_all(R, PP, FP, CP, A.cells(A.blocks(FP.assign(contrast='DF-D0'), ('contrast', 'config', 'surface', 'sigma')), ('contrast', 'config', 'surface', 'sigma'),
                                                      lambda x: 'point', None), pd.DataFrame(), sel, store.root, K.guard(prev / 'x').parent, technical=True)
            fig_ok = all((K.ROOT / v).exists() for v in figs.values())
        finally:
            shutil.rmtree(prev, ignore_errors=True)
        cl = BD.closure_check(g['freeze']['closure']); cnt = store.counts(); before = runs[-1]['counts_before']
        delta = {k: (cnt.get(k) or {}).get('reserved', 0) - (before.get(k) or {}).get('reserved', 0) for k in ('render', 'decode', 'evaluation')}
        ok = (comp['complete'] and comp['n_results'] == 20 and len(R) == 20 and npairs == dict(C1=4, C2=4, C3=4, C4=4, C5=2) and len(FP) == 8 and fig_ok and cl['passed']
              and all(delta[k] <= K.CHECKPOINT_CAPS[k] for k in delta))
        K.run_put('checkpoint', dict(passed=ok, keys=sorted(cpk), schedule=dict(res, done=None), completeness=comp, n_results=len(R), pairs=npairs, frontend_pairs=len(FP),
                                     states=R.state.value_counts().to_dict(), figures_smoke=fig_ok, closure=cl, counts=cnt, calls_this_checkpoint=delta,
                                     note='technical check (one block); no interval',
                                     t=K.now()))
        if not ok:
            raise K.Stop('technical', f'checkpoint not closed: completeness {comp["complete"]}, pairs {npairs}, figures {fig_ok}, closure {cl["passed"]}')
        log(f'checkpoint passed: {len(cp)} images, {len(R)} results, pairs {npairs}')
    else:
        log('checkpoint passed earlier (its 12 images are reused, not rendered again)')
    rest = [x for x in rows if x['forward_key'] not in cpk]
    res = _schedule_or_stop(rest, store, fps, 'core')
    comp = CH.completeness(rows, store.root, fps); cl = BD.closure_check(g['freeze']['closure'])
    out = dict(complete=comp, closure_end=cl, schedule=dict(res, done=None), counts=store.counts(), t=K.now())
    K.run_put('b_complete', out)
    if not comp['complete'] or not cl['passed']:
        raise K.Stop('result_integrity', f"core results incomplete: {comp['missing'][:6]}")
    return dict(complete=True, n_results=comp['n_results'], states=comp['states_count'], counts=store.counts())


def _analysis_code_ok(r: dict) -> dict:
    fz = r['g0']['freeze']['analysis_code']; cur = K.code_identity(K.ANALYSIS_FILES)
    if cur == fz:
        return dict(equal_to_g0=True, code=cur)
    fixes = r.get('post_run_fixes') or []
    if fixes and fixes[-1].get('sha_after') == cur.get(fixes[-1]['file']):
        return dict(equal_to_g0=False, registered_fix=len(fixes), code=cur)
    raise K.Stop('refused', f'analysis code changed after G0 without a registered fix: {cur} vs {fz}')


def analyse() -> dict:
    import pandas as pd
    import tr5_analysis as A
    import tr5_claims as CL
    import tr5_fig as F
    import tr5_manifest as MF
    r = K.run_load()
    if not ((r.get('b_complete') or {}).get('complete') or {}).get('complete'):
        raise K.Stop('refused', 'analysis needs the complete core results')
    code = _analysis_code_ok(r)
    M, _ = MF.build_manifest(); fps = r['b_runs'][-1]['preflight']['fps']
    R = A.results(M.to_dict('records'), K.OUT, fps)
    claims = pd.DataFrame(r['g0']['freeze']['claims']); csum = r['g0']['claims_summary']
    claims['expected_sign'] = [None if (v is None or (isinstance(v, float) and v != v)) else int(v) for v in claims.expected_sign]
    E = A.effects(R, claims, csum); exp = CL.expected(claims)
    sel = dict(F4=A.select_cases(E['pairs'], exp, 'C1', 'D0', 80.0), F5=A.select_cases(E['pairs'], exp, 'C5', 'P4', 60.0))
    K.save_csv(R, K.OUT / 'results.csv'); K.save_csv(E['effects'], K.OUT / 'effects.csv')
    figs = F.make_all(R, E['pairs'], E['frontend'], E['cells'], E['cells_frontend'], E['decisions'], sel, K.OUT, K.FIGS)
    counts = E['effects'].record_type.value_counts().to_dict()
    out = dict(analysis_code=code, results=dict(rows=len(R), sha256=K.sha(K.OUT / 'results.csv'), states=R.groupby(['family', 'method']).state.value_counts().to_dict()),
               effects=dict(rows=len(E['effects']), by_type=counts, sha256=K.sha(K.OUT / 'effects.csv')), cases=sel,
               decisions=E['decisions'][['contrast', 'method', 'sigma', 'label', 'label_zh', 'risk_higher_on_favoured_arm']].to_dict('records'),
               figures={k: dict(file=v, sha256=K.sha(K.ROOT / v)) for k, v in figs.items()}, layout_code=K.code_identity(K.LAYOUT_FILES), t=K.now())
    if counts.get('pair_method') != 816 or counts.get('pair_frontend') != 384:
        raise K.Stop('result_integrity', f'pair counts {counts}')
    K.run_put('analysis', out)
    return {k: v for k, v in out.items() if k != 'cases'}


def recheck() -> dict:
    import pandas as pd
    import tr5_analysis as A
    R = pd.read_csv(K.OUT / 'results.csv', encoding='utf-8-sig', low_memory=False)
    picks = A.recheck_selection(R); res = A.recheck(K.OUT, picks)
    worst = max((v for x in res for k, v in x.items() if k.endswith('_absdiff') and v is not None), default=0.0)

    within = all((x[f'{dn}_{nm}_absdiff'] is None) or x[f'{dn}_{nm}_absdiff'] <= x[f'{dn}_storage_tol_mm'] for x in res for dn in ('glob', 'loc') for nm in ('pt_rmse', 'pt_p95'))
    ok = all(x['cwu_equal'] and x['counts_equal'] and x.get('full_cwu_equal', True) for x in res) and within and len(res) == 12
    out = dict(passed=ok, n=len(res), max_abs_error_diff_mm=worst, within_float32_storage_precision=within, criterion='|diff| <= 2 x float32 spacing of max |zr| (per row, domain)',
               picks=[dict(forward_key=a, method=b) for a, b in picks], rows=res,
               coverage=dict(surfaces=sorted({R.set_index(['forward_key', 'method']).loc[(a, b)].surface for a, b in picks}), methods=sorted({b for _, b in picks})), t=K.now())
    K.run_put('recheck12', out)
    return {k: v for k, v in out.items() if k != 'rows'}


def register_fix(file: str, reason: str) -> dict:
    r = K.run_load(); fz = r['g0']['freeze']['analysis_code']
    if file not in fz:
        raise K.Stop('refused', f'{file} is not a registered analysis file')
    fx = dict(n=len(r.get('post_run_fixes') or []) + 1, t=K.now(), file=file, sha_frozen_at_g0=fz[file], sha_after=K.sha(K.HERE / file), reason=reason, real_calls=0)
    lst = r.get('post_run_fixes') or []; lst.append(fx); K.run_put('post_run_fixes', lst)
    return fx


def final() -> dict:
    import tr5_bind as BD
    store = K.Store(K.OUT); outs = {}
    for p in sorted(K.OUT.rglob('*')):
        if p.is_file() and p.name != 'run.json' and '_drills' not in p.parts and '_checkpoint_preview' not in p.parts:
            outs[K.rel(p)] = K.sha(p)
    r = K.run_load(); fz = r['g0']['freeze']
    out = dict(status='NUMERICAL_CHECKS_COMPLETE', counts=store.counts(), caps=dict(K.CAPS), n_outputs=len(outs), outputs=outs,
               code_now=K.code_identity(), code_equal_g0={f: K.sha(K.HERE / f) == h for f, h in fz['numeric_code'].items()}, layout_code=K.code_identity(K.LAYOUT_FILES),
               drill_code=K.code_identity(K.DRILL_FILES), inputs_end=BD.recorded_checks(BD.hash_inputs(BD.input_files())), model_prep=BD.model_prep_status(), t=K.now())
    K.run_put('final', out)
    return {k: v for k, v in out.items() if k not in ('outputs',)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['g0', 'run', 'analyse', 'recheck', 'final', 'register-fix', 'drill-exit'])
    ap.add_argument('--case', default=None); ap.add_argument('--root', default=None); ap.add_argument('--file', default=None); ap.add_argument('--reason', default=None)
    a = ap.parse_args(); t0 = time.time()
    try:
        if a.cmd == 'drill-exit':
            import tr5_drills as DR
            r = DR.exit_case(a.case, a.root)
        elif a.cmd == 'register-fix':
            r = register_fix(a.file, a.reason or '')
        else:
            r = dict(g0=g0, run=run_stage_b, analyse=analyse, recheck=recheck, final=final)[a.cmd]()
        print(json.dumps(K.clean(r), ensure_ascii=False, default=str)[:6000]); print(f'{a.cmd}: {round(time.time() - t0, 1)} s', flush=True)
    except K.Stop as e:
        print(json.dumps(dict(stop=e.code, exit=e.exit_code, msg=e.msg), ensure_ascii=False, default=str)[:4000], flush=True)
        if a.cmd != 'drill-exit':
            st = K.run_load().get('stops') or []; st.append(dict(cmd=a.cmd, code=e.code, exit=e.exit_code, msg=e.msg[:2000], t=K.now())); K.run_put('stops', st)
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        print('interrupted', flush=True); sys.exit(K.EXIT['interrupted'])
    except Exception:
        traceback.print_exc(); sys.exit(K.EXIT['technical'])


if __name__ == '__main__':
    main()
