# Check spacing-layout input contracts and expected case coverage.

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

import s2p_common as K

K.setup()
RES = []
DRILL = K.OUT / 'drill'


def check(name, kind):
    def deco(fn):
        def run():
            t0 = time.time()
            try:
                r = fn(); RES.append(dict(check=name, kind=kind, passed=bool(r.pop('passed')), seconds=round(time.time() - t0, 1), **r))
            except Exception as e:
                RES.append(dict(check=name, kind=kind, passed=False, seconds=round(time.time() - t0, 1), error=repr(e)[:400], trace=traceback.format_exc()[-1500:]))
            print(('PASS ' if RES[-1]['passed'] else 'FAIL ') + name, flush=True)
        return run
    return deco


def L():
    import s2p_manifest as MF
    return pd.read_csv(MF.CSV, keep_default_na=False, float_precision='round_trip')


@check('Platform bound (SURF-02 v1.1, no dependency drift, binding = release, D0 / P4) and registered inputs unchanged since S2-PREP-01', 'static')
def c_static():
    st = K.platform_state(); cfg = K.load_json(K.OUT / 's2_config.json'); now = K.input_hashes()
    changed = [k for k in cfg['inputs'] if cfg['inputs'][k] != now.get(k) and 's1_real01/manifest' not in k]
    return dict(passed=st['bound_ok'] and not changed, changed=changed, version=st['version'], drift=st['dependency_drift'], global_domain=st['global_domain'])


@check('joint10 whitelist = task-book table (10 core rows: ids, surfaces, layouts, sigma, phase, rep 0); the old joint_test flag is not used', 'list')
def c_joint10():
    import s2p_run as RN
    x = RN.set_rows(L(), 'joint10').set_index('cand_id')
    exp = {'C013282': ('M10_curv', 'R6x4', 0.0, 'pos33'), 'C013244': ('M10_curv', 'R8x4', 0.0, 'pos33'), 'C013224': ('M10_curv', 'R12x4', 0.0, 'pos33'), 'C013279': ('M10_curv', 'T8x4', 0.0, 'pos33'),
           'C013736': ('M24_curv', 'R8x2.5', 40.0, 'pos33'), 'C018447': ('M10_curv', 's4.45_phi0', 0.0, 'ph0'), 'C018309': ('M10_curv', 's4.45_phi45', 0.0, 'ph0'),
           'C018194': ('B_curv', 's4.45_phi45', 40.0, 'ph0'), 'C018107': ('B_curv', 's8_phi0', 0.0, 'ph0'), 'C019194': ('M24_curv', 's8_phi0', 40.0, 'ph0')}
    bad = [c for c, (s, l, g, p) in exp.items() if (x.at[c, 'surface'], x.at[c, 'layout'], float(x.at[c, 'sigma']), x.at[c, 'phase_label']) != (s, l, g, p) or int(x.at[c, 'rep']) != 0
           or x.at[c, 'regime'] != 'equal_integral']
    return dict(passed=not bad and len(x) == 10, mismatches=bad, n=len(x))


def _plan_rec():
    plan, rec = [], []
    for lay in ('R8x4', 'T8x4'):
        for ph in ('pos13', 'pos23', 'pos33'):
            for rep in (0, 1):
                dk = f'{lay}|{ph}|{rep}'; seed = 9420000 + int(ph[3:]) * 10 + rep; base = 0.6 if lay == 'R8x4' else 0.7
                row = dict(decode_key=dk, cand_id='C' + dk, noise_seed=seed, rep=rep, phase_label=ph, surface='M10_curv', family='point', regime='equal_integral', layout=lay, sigma=40.0)
                plan.append(row)
                rec.append(dict(row, noise_seed=float(seed), state='ok', complete=True, glob_Ctau=base + 0.002 * int(ph[3:]) + 0.001 * rep, glob_pt_rmse=0.05, loc_Ctau=base, loc_pt_rmse=0.04,
                                img_recovery=0.9, mismatch_rate=0.0))
    return pd.DataFrame(plan), pd.DataFrame(rec)


def _states(plan, rec):
    import s2p_stats as ST
    A = ST.analyse(plan, rec); c = A['conditions']; k = A['contrasts']
    cs = {l: c[(c.layout == l) & (c.metric == 'glob_Ctau')].iloc[0] for l in ('R8x4', 'T8x4')}
    rt = k[(k.contrast == 'dots R vs T (8 x 4)') & (k.metric == 'glob_Ctau')].iloc[0]
    rr = k[(k.contrast == 'dots R vs T (8 x 4)') & (k.metric == 'glob_pt_rmse')].iloc[0]
    return A, cs, rt, rr


@check('Statistics gated by the registered plan (in-memory counterexamples): final condition / contrast states', 'constructed')
def c_stats():
    P, R = _plan_rec(); res = {}
    A, cs, rt, _ = _states(P, R)
    res['complete'] = bool(cs['R8x4'].state == 'complete' and rt.state == 'ok' and abs(rt['mean'] - 0.1) < 1e-12 and rt.n_phases_total == 3 and A['validation']['formal_ok'])
    A, cs, rt, _ = _states(P, R[R.decode_key != 'T8x4|pos23|0'])
    res['one_arm_missing'] = bool(cs['T8x4'].state == 'incomplete' and cs['R8x4'].state == 'complete' and rt.state == 'incomplete' and not A['validation']['formal_ok'])
    A, cs, rt, _ = _states(P, R[R.phase_label != 'pos23'])
    res['both_arms_missing_same_phase'] = bool(cs['R8x4'].state == 'incomplete' and cs['T8x4'].state == 'incomplete' and rt.state == 'incomplete' and cs['R8x4'].n_phases_total == 3)
    A, cs, rt, _ = _states(P, R[R.rep != 1])
    res['both_arms_missing_same_rep'] = bool(cs['R8x4'].state == 'incomplete' and rt.state == 'incomplete')
    R2 = R.copy(); R2['noise_seed'] = R2.noise_seed.astype(str)
    A, cs, rt, _ = _states(P, R2)
    res['seed_numeric_semantics'] = bool(rt.state == 'ok')
    R3 = R.copy(); R3.loc[0, 'noise_seed'] = 1
    A, cs, rt, _ = _states(P, R3)
    res['wrong_seed'] = bool(A['validation']['status']['R8x4|pos13|0'] == 'seed_mismatch' and rt.state == 'incomplete')
    A, cs, rt, _ = _states(P, pd.concat([R, R.iloc[[2]]]))
    res['duplicate_record'] = bool(A['validation']['status'][R.iloc[2].decode_key] == 'duplicate' and rt.state == 'incomplete')
    ex = R.iloc[[0]].assign(decode_key='R8x4|pos13|9')
    A, cs, rt, _ = _states(P, pd.concat([R, ex]))
    res['extra_record'] = bool(cs['R8x4'].state == 'blocked_extra_record' and rt.state == 'incomplete' and not A['validation']['formal_ok'])
    R4 = R.copy(); R4.loc[1, 'complete'] = False
    A, cs, rt, _ = _states(P, R4)
    res['technical'] = bool(cs['R8x4'].state == 'incomplete' and rt.state == 'incomplete')
    R5 = R.copy(); m5 = (R5.layout == 'T8x4') & (R5.phase_label == 'pos33'); R5.loc[m5, 'state'] = 'no_output'; R5.loc[m5, ['glob_pt_rmse', 'glob_Ctau']] = np.nan
    A, cs, rt, rr = _states(P, R5)
    res['scientific_no_output'] = bool(cs['T8x4'].state == 'complete' and rt.state == 'ok' and rr.state == 'ok_defined_subset' and rr.n_phases_defined == 2 and rr.n_phases_total == 3
                                       and cs['T8x4'].n_no_output == 2)
    P6 = pd.concat([P, P[P.layout == 'R8x4'].assign(cand_id=lambda x: x.cand_id + '_view2')])
    A, cs, rt, _ = _states(P6, R)
    res['one_result_two_views'] = bool(A['validation']['n_planned'] == len(P) and rt.state == 'ok' and cs['R8x4'].n_planned_decodes == 6)
    return dict(passed=all(res.values()), cases=res)


@check('Reuse joint10: both S1 cases pass the image, decode (39 dependency files, cfg, P4 path) and evaluation layers', 'existing')
def c_reuse():
    import s2p_reuse as RU
    d = L(); T, _ = RU.build(d); out = {}
    for cid in K.JOINT10['reuse']:
        ik = d[d.cand_id == cid].image_key.iloc[0]; r = T[T.image_key == ik].iloc[0].to_dict(); lay = RU.verify_layers(r)
        out[cid] = {k: lay[k] for k in ('image_ok', 'decode_ok', 'eval_ok', 'reasons')} | dict(s1_decode_key=r['s1_decode_key'])
    return dict(passed=all(v['image_ok'] and v['decode_ok'] and v['eval_ok'] for v in out.values()), cases=out)


@check('Isolated drills: ledger counts inherited and checked BEFORE a call (failed attempts count), cap stop, lock exclusivity, stage resume by identity', 'constructed')
def c_drills():
    import s2p_case as CS
    import s2p_run as RN
    shutil.rmtree(DRILL, ignore_errors=True); DRILL.mkdir(parents=True); res = {}
    led = RN.Ledger(DRILL / 'ledger.jsonl', caps=dict(render=2)); t1 = led.start('render', 'a'); led.end(t1, 'ok'); t2 = led.start('render', 'b'); led.end(t2, 'failed', 'drill')
    led2 = RN.Ledger(DRILL / 'ledger.jsonl', caps=dict(render=2))
    try:
        led2.start('render', 'c'); res['cap_stop'] = False
    except RN.Stop as e:
        res['cap_stop'] = e.code == 16
    res['inherited_count'] = led2.count('render') == 2
    with RN.Lock(DRILL / 'run.lock'):
        try:
            with RN.Lock(DRILL / 'run.lock'):
                res['lock_exclusive'] = False
        except RN.Stop:
            res['lock_exclusive'] = True
    res['lock_released'] = not (DRILL / 'run.lock').exists()
    p = DRILL / 'stage' / 'x.json'; h = CS._save_npz(p.with_suffix('.npz'), a=np.arange(3)); K.save_json(p, dict(identity='ID-A', npz_sha256=h, complete=True))
    res['resume_same_identity'] = CS.complete(p, 'ID-A'); res['rerun_on_new_identity'] = not CS.complete(p, 'ID-B')
    p.with_suffix('.npz').write_bytes(b'broken'); res['rerun_on_corrupt_npz'] = not CS.complete(p, 'ID-A')
    shutil.rmtree(DRILL, ignore_errors=True)
    return dict(passed=all(res.values()), cases=res, note='isolated drill directory (removed afterwards); no real call')


@check('Entry refusals: unauthorised set, missing permit, tampered permit -> exit 14; nothing rendered', 'constructed')
def c_entry():
    run = lambda args: subprocess.run([sys.executable, str(K.HERE / 's2p_run.py')] + args, capture_output=True, text=True, encoding='utf-8', cwd=str(K.ROOT)).returncode
    n0 = sum(1 for _ in (K.OUT / 'run' / 'images').glob('*.npz')) if (K.OUT / 'run' / 'images').exists() else 0
    res = dict(core_main=run(['run', '--set', 'core-main']), no_permit=run(['run', '--set', 'joint10', '--permit', str(DRILL / 'none.json')]))
    if K.PERMIT.exists():
        DRILL.mkdir(parents=True, exist_ok=True); p = K.load_json(K.PERMIT); p['caps'] = dict(render=13, decode=13); K.save_json(DRILL / 'tampered.json', p)
        res['tampered'] = run(['run', '--set', 'joint10', '--permit', str(DRILL / 'tampered.json')]); shutil.rmtree(DRILL, ignore_errors=True)
    n1 = sum(1 for _ in (K.OUT / 'run' / 'images').glob('*.npz')) if (K.OUT / 'run' / 'images').exists() else 0
    return dict(passed=all(v == 14 for v in res.values()) and n0 == n1, exit_codes=res)


@check('Freeze: scripts, 39 S1 dependency files, list, config, release, global domain and reference meshes verified against the real files', 'static')
def c_freeze():
    import s2p_freeze as FZ
    v = FZ.verify()
    return dict(passed=v['ok'], **{k: v.get(k) for k in ('drift', 'mesh_drift', 'freeze_sha256', 'part_digests', 'reason')})


@check('joint10 technical completeness (denominator 10), real calls <= 8 renders / 8 decodes, sources traceable', 'real')
def c_joint_done():
    s = K.load_json(K.RESULTS / 'joint10_technical.json')
    return dict(passed=bool(s['ok'] and s['real_renders'] <= 8 and s['real_decodes'] <= 8), **s)


@check('Re-entry of joint10: no additional real render / decode (ledger counts unchanged), all stages resumed by identity', 'real')
def c_reentry():
    import s2p_run as RN
    led = RN.Ledger(); r0, d0 = led.count('render'), led.count('decode')
    rc = subprocess.run([sys.executable, str(K.HERE / 's2p_run.py'), 'run', '--set', 'joint10', '--permit', str(K.PERMIT)], capture_output=True, text=True, encoding='utf-8', cwd=str(K.ROOT))
    r1, d1 = led.count('render'), led.count('decode')
    ev = [e for e in led.events() if e.get('msg') == 'case done'][-8:]
    return dict(passed=(rc.returncode == 0 and r1 == r0 and d1 == d0 and all(not e['evaluated'] and e['calls'] == dict(render=0, decode=0) for e in ev)), exit_code=rc.returncode,
                renders=(r0, r1), decodes=(d0, d1))


@check('Light recomputation from the saved NPZ (8 new cases): point RMSE / MAE / P95 of both domains, domain masks, identity closure; surface Cvalid / Ctau via the current evaluator', 'real')
def c_recompute():
    import s1r_common as S1K
    import s2p_case as CS
    S = S1K.s2(); dom = S1K.global_domain(0.5); d = L().set_index('cand_id'); out = {}; maxd = 0.0
    for cid in K.JOINT10['new']:
        t = d.loc[cid]; ep = K.OUT / 'run' / 'eval' / f'{t.decode_key}.json'; rec = K.load_json(ep); m = rec['metrics']
        with np.load(ep.with_suffix('.npz')) as z:
            E = {k: np.asarray(z[k]) for k in z.files}
        v = E['valid']; e = E['pts'][:, 2] - E['zr'].astype(float)
        loc = v & S1K.in_local(E['pts'][:, :2]); glob = v & S1K.in_global(E['pts'][:, :2], dom)
        diffs = dict(masks=float(np.sum(loc != E['in_local']) + np.sum(glob != E['in_global'])))
        for pre, mk in (('loc', loc), ('glob', glob)):
            ee = e[mk]
            if len(ee):
                for name, val in (('pt_rmse', np.sqrt(np.mean(ee ** 2))), ('pt_mae', np.mean(np.abs(ee))), ('pt_p95', np.percentile(np.abs(ee), 95))):
                    diffs[f'{pre}_{name}'] = abs(float(val) - float(m[f'{pre}_{name}']))
        diffs['identity_closure'] = float(abs(int(E['point_correct'].sum() + E['point_wrong'].sum() + E['point_unconf'].sum()) - int(E['accepted'].sum())))
        sid = K.PLATFORM_ID[t.surface]; mesh = S.build_mesh(sid); dl = S.roi_domain(mesh)
        fl = CS.node_fields(sid, E['pts'][v], dict(x=dl['x'], y=dl['y'], area=dl['area'], hash=dl['hash']), np.asarray(dl['z_ref'], float))
        fg = CS.node_fields(sid, E['pts'][v], dict(x=dom['x'], y=dom['y'], area=np.full(len(dom['x']), dom['step'] ** 2), hash=dom['hash']), np.asarray(dom[f'z_{sid}'], float))
        for pre, f in (('loc', fl), ('glob', fg)):
            diffs[f'{pre}_Cvalid'] = abs(f['Cvalid'] - (m[f'{pre}_Cvalid'] or 0.0)); diffs[f'{pre}_Ctau'] = abs(f['Ctau'] - (m[f'{pre}_Ctau'] or 0.0))
        exact = {k: x for k, x in diffs.items() if not k.split('_', 1)[-1].startswith('pt_')}; ptd = {k: x for k, x in diffs.items() if k.split('_', 1)[-1].startswith('pt_')}
        md_e = max(exact.values()); md_p = max(ptd.values()) if ptd else 0.0; maxd = max(maxd, md_e)
        out[cid] = dict(max_abs_diff_exact_items=md_e, max_abs_diff_point_stats_mm=md_p, state=m['state'], n_valid=int(v.sum()))
    ok = maxd <= 1e-12 and all(c['max_abs_diff_point_stats_mm'] <= 1e-6 for c in out.values())
    return dict(passed=ok, cases=out, tolerances=dict(masks_identity_coverage='exact (<= 1e-12)', point_stats='<= 1e-6 mm (zr stored as float32)'))


@check('Reference value and physical transform (one dot, one line case): z_ref = bilinear grid truth; H = c - z matches the SURF-02 model H(x_p = -x, y_p = y)', 'real')
def c_reference():
    import s1r_common as S1K
    S = S1K.s2(); d = L().set_index('cand_id'); out = {}
    for cid in ('C013244', 'C018309'):
        t = d.loc[cid]; sid = K.PLATFORM_ID[t.surface]; mesh = S.build_mesh(sid)
        with np.load(K.OUT / 'run' / 'eval' / f'{t.decode_key}.npz') as z:
            P = np.asarray(z['pts'])[np.asarray(z['valid'])][:200]; zr = np.asarray(z['zr'])[np.asarray(z['valid'])][:200].astype(float)
        f = S.truth_interp(mesh); zi = f(np.column_stack([P[:, 1], P[:, 0]]))
        H_model = S.h_physical(sid, P[:, 0], P[:, 1]); z_model = K.C_REF - H_model
        out[cid] = dict(n=int(len(P)), max_zr_vs_grid=float(np.max(np.abs(zi - zr))), max_grid_vs_model=float(np.max(np.abs(zi - z_model))), c_ref=float(S.C_REF))
    ok = all(v['max_zr_vs_grid'] <= 1e-5 and v['max_grid_vs_model'] <= 1e-3 for v in out.values()) and abs(float(S.C_REF) - K.C_REF) < 1e-12
    return dict(passed=ok, cases=out, note='float32 storage of zr: tolerance 1e-5 mm; bilinear grid vs analytic model: SURF-02 registered 5.6e-4 mm')


@check('Figures: numbers shown (C_valid, C_tau, point RMSE) recomputed from the saved node fields / points = evaluation record', 'real')
def c_figs():
    s = K.load_json(K.RESULTS / 'joint10_summary.json')
    fc = s.get('figure_value_check') or {}
    ok = bool(fc) and all(all(v['ok'] for v in g.values()) for g in fc.values()) and len(s['figures']) == 4
    return dict(passed=ok, figures=s['figures'], value_check=fc)


def main(phase: str = 'pre'):
    fns = dict(pre=(c_static, c_joint10, c_stats, c_reuse, c_drills, c_freeze, c_entry), post=(c_freeze, c_joint_done, c_reentry, c_recompute, c_reference, c_figs))[phase]
    for fn in fns:
        fn()
    prev = K.load_json(K.OUT / 'checks.json') if (K.OUT / 'checks.json').exists() else {}
    phases = dict(prev.get('phases') or {}); phases[phase] = dict(t_local=K.now_local(), checks=RES, n_failed=sum(1 for r in RES if not r['passed']))
    failed = [r['check'] for r in RES if not r['passed']]
    out = dict(t_local=K.now_local(), task='S2-PREP-02', phases=phases, n_checks=len(RES), n_failed=len(failed), failed=failed, last_phase=phase,
               note='Numerical consistency checks.')
    K.save_json(K.OUT / 'checks.json', out)
    return out


if __name__ == '__main__':
    r = main(sys.argv[1] if len(sys.argv) > 1 else 'pre')
    print(json.dumps(dict(n=r['n_checks'], failed=r['failed']))); sys.exit(0 if not r['failed'] else 17)
