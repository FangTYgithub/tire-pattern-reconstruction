# Exercise environment scheduling, pairing and rejection paths with test inputs.

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import mp5_common as K

SENT = dict(render=0, decode=0, evaluation=0)


class Sentinel:
    name = 'sentinel'

    def render(self, *a, **k):
        SENT['render'] += 1; raise RuntimeError('sentinel render called')

    def decode(self, *a, **k):
        SENT['decode'] += 1; raise RuntimeError('sentinel decode called')

    def evaluate(self, *a, **k):
        SENT['evaluation'] += 1; raise RuntimeError('sentinel evaluate called')


def _row(cid, ok, desc, **detail) -> dict:
    return dict(check_id=cid, passed=bool(ok), description=desc, detail=json.dumps(K.clean(detail), ensure_ascii=False, default=str)[:3500])


def _root() -> Path:
    return Path(tempfile.mkdtemp(prefix=K.DRILL_PREFIX, dir=K.drill_base()))


def _spec(manifest_mod=None, layer_override=None, rules=None, freeze_mod=None, final_record=None, missing=False) -> dict:
    import mp5_main as MM
    root = _root(); P = K.Paths(root); run = K.P0.run_load()
    keep = {k: run[k] for k in ('freeze', 'proofs', 'protected', 'replay', 'analysis') if k in run}
    if freeze_mod:
        keep['freeze'] = freeze_mod(json.loads(json.dumps(keep['freeze'])))
    if final_record:
        keep['final'] = final_record
    P.save_json(P.run, keep)
    mp = root / 'formal_manifest.csv'
    if not missing:
        M = pd.read_csv(MM.MANIFEST, low_memory=False)
        if manifest_mod:
            M = manifest_mod(M)
        if manifest_mod is None:
            shutil.copyfile(MM.MANIFEST, mp)
        else:
            M.to_csv(mp, index=False, encoding='utf-8-sig')
    return dict(paths=P, manifest=mp, store=K.Store(P, K.CAPS), backend=Sentinel(), layer_override=layer_override, rules=rules, protected=lambda: [])


def _run(stage, spec) -> tuple[int, str, dict]:
    import mp5_main as MM
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = MM.run_main([stage], spec=spec)
    fin = spec['paths'].run_load().get('final') or {}
    shutil.rmtree(spec['paths'].root, ignore_errors=True)
    return code, buf.getvalue()[-1500:], fin


# Test rejection of changed manifests, dependencies and incomplete case registrations.
def freeze_drills() -> list:
    rows = []
    c0, out0, _ = _run('run', _spec())
    control_ok = c0 == K.EXIT['refused'] and 'manifest' not in out0.split('launch refused')[-1].split('real stages')[0] and 'real stages need the real root' in out0

    def mod(cid, desc, fn=None, expect=('manifest sha256 differs', 'manifest rows_digest differs'), **kw):
        code, out, _ = _run('run', _spec(manifest_mod=fn, **kw))
        ok = code == K.EXIT['refused'] and all(e in out for e in expect) and control_ok
        rows.append(_row(cid, ok, desc, code=code, expected=list(expect), tail=out[-600:]))

    def m_phase(M):
        i = M.index[M.family == 'line'][0]; M.loc[i, 'phase_frac'] = float(M.loc[i, 'phase_frac']) + 1e-3; return M
    def m_seed(M):
        i = M.index[M.seed.notna()][0]; M.loc[i, 'seed'] = float(M.loc[i, 'seed']) + 1; return M
    def m_parent(M):
        i = M.index[M.parent_case_key.notna()][0]
        other = M.index[M.parent_case_key.notna() & (M.parent_case_key != M.loc[i, 'parent_case_key'])][0]
        M.loc[i, 'parent_case_key'] = M.loc[other, 'parent_case_key']; return M
    def m_methods(M):
        i = M.index[M.family == 'point'][0]; M.loc[i, 'methods'] = 'DF'; return M
    mod('A3_mod_phase', 'one phase coordinate changed in a copy of the manifest -> launch refused (manifest sha / rows digest)', m_phase)
    mod('A3_mod_seed', 'one seed changed -> refused', m_seed)
    mod('A3_mod_parent', 'one parent key changed -> refused', m_parent)
    mod('A3_mod_methods', 'one method configuration changed -> refused', m_methods)
    mod('A3_row_deleted', 'one row deleted -> refused (n rows / digest)', lambda M: M.drop(M.index[5]), expect=('manifest n_rows differs', 'manifest rows_digest differs'))
    mod('A3_row_duplicated', 'one row duplicated -> refused', lambda M: pd.concat([M, M.iloc[[7]]], ignore_index=True), expect=('manifest n_rows differs', 'manifest rows_digest differs'))
    code, out, _ = _run('run', _spec(missing=True))
    rows.append(_row('A3_manifest_missing', code == K.EXIT['refused'] and 'manifest missing' in out, 'missing manifest -> refused before any call', code=code, tail=out[-400:]))
    import mp5_chain as CH
    code, out, _ = _run('run', _spec(layer_override={'inputs': {CH.ADAPTER_FILE: '0' * 64}}))
    rows.append(_row('A3_adapter_changed', code == K.EXIT['refused'] and 'layer inputs changed' in out, 'changed input adapter -> freeze mismatch, old proofs stale -> refused', code=code, tail=out[-500:]))
    c1, o1, _ = _run('run', _spec(layer_override={'registry_code': {'scripts/surface_revision/thesis_reinforce03/tr3_b2.py': '0' * 64}}))

    def fz_reg(fz):
        f = sorted(fz['registry']['files'])[0]; fz['registry']['files'][f] = '1' * 64; return fz
    c2, o2, _ = _run('run', _spec(freeze_mod=fz_reg))
    rows.append(_row('A3_registry_changed', c1 == K.EXIT['refused'] and 'layer registry_code changed' in o1 and c2 == K.EXIT['refused'] and 'registry source files changed' in o2,
                     'changed phase-registry implementation (tr3_b2.py) or registry source file -> refused', code=dict(code=c1, source=c2), tails=[o1[-300:], o2[-300:]]))
    import mp5_main as MM
    r2 = MM.rules_all(); r2 = dict(r2, analysis=dict(r2['analysis'], reproduction='changed rule'))
    old_final = dict(status='RUN_COMPLETE_AWAITING_REVIEW', t='drill', note='copied completion record of an earlier rule version')
    code, out, fin = _run('final', _spec(rules=r2, final_record=old_final))
    rows.append(_row('A3_rules_changed_old_final', code != 0 and fin.get('status') in ('BLOCKED', 'NEEDS_REVIEW', 'PARTIAL') and any('rules changed' in r for r in fin.get('reasons', [])),
                     'rules changed while an old completion record exists -> the final gate does not reuse it (BLOCKED, non-zero exit)', code=code, status=fin.get('status'),
                     reasons=fin.get('reasons', [])[:3]))
    for r in rows:
        d = json.loads(r['detail']); d['sentinel'] = dict(SENT); d['control_reason_only_root_backend'] = control_ok
        r['detail'] = json.dumps(d, ensure_ascii=False)[:3500]
        r['passed'] = bool(r['passed'] and all(v == 0 for v in SENT.values()))
    for p in K.drill_base().glob(K.DRILL_PREFIX + '*'):
        shutil.rmtree(p, ignore_errors=True)
    return rows


def syn(configs=('D_r1_p8x4',), envs=('NOMINAL', 'BLUR2'), blocks=(0, 1, 2, 3), reps=(1, 2), sigma=20.0, ctau=None) -> pd.DataFrame:
    import mp5_analysis as AN
    rows = []
    for c in configs:
        for b in blocks:
            for rp in reps:
                for e in envs:
                    ck = f'{c}|B_curv|b{b}|s{int(sigma)}|r{rp}|{e}'
                    v = ctau(c, b, rp, e) if ctau else 0.5
                    r = {k: None for k in AN.REQ_COLUMNS}
                    r.update(row_key=f'{ck}|DF', case_key=ck, subset='core', family='point', config=c, surface='B_curv', phase_block=b, sigma=sigma, rep=rp, seed=(None if sigma == 0 else 9_000 + 10 * b + rp),
                             env=e, method='DF', parent_case_key=(None if e == 'NOMINAL' else f'{c}|B_curv|b{b}|s{int(sigma)}|r{rp}|NOMINAL'), state='ok', pack_state='read',
                             glob_Ctau=v, loc_Ctau=v, glob_Cvalid=0.9, loc_Cvalid=0.9, glob_pt_rmse=0.1, glob_pt_p95=0.2, loc_pt_rmse=0.1, glob_surf_rmse=0.1, sup_surf_rmse=0.1, n_valid=10,
                             glob_n_valid=10, loc_n_valid=5, n_accepted=10, c=10, w=0, u=0, w_share=0.0, u_share=0.0)
                    rows.append(r)
    T = pd.DataFrame(rows); T['seed'] = pd.Series(list(T.seed), dtype=object)
    return T


def _raises(fn) -> tuple[bool, str]:
    try:
        fn(); return False, ''
    except K.Stop as e:
        return True, e.msg[:200]


# Test pairing, aggregation and metric checks with constructed counterexamples.
def stats_drills() -> list:
    import mp5_analysis as AN
    rows = []
    T = syn(); E = AN.frame(AN.contrast_f1(T), T); Pg = E[E.metric == 'glob_Ctau']
    per_block = Pg.groupby('phase_block').size().to_dict()
    S = AN.summaries(E, pilot={})
    rows.append(_row('A4_two_reps_two_pairs', all(v == 2 for v in per_block.values()) and len(Pg) == 8 and Pg.left_key.is_unique,
                     'two repetitions per block form exactly two pairs (8 over 4 blocks), never four', per_block=per_block, n=len(Pg)))
    Tm = T.drop(T.index[(T.env == 'BLUR2') & (T.rep == 2) & (T.phase_block == 1)])
    r1, m1 = _raises(lambda: AN.contrast_f1(Tm))
    Em = E[~((E.env == 'BLUR2') & (E.rep == 2) & (E.phase_block == 1))]
    r2, m2 = _raises(lambda: AN.summaries(Em, pilot={}))
    rows.append(_row('A4_missing_rep', r1 and r2, 'a missing repetition is refused by the pairing and by the block aggregation', pairing=m1, aggregation=m2))
    Tw = T.copy(); i = Tw.index[(Tw.env == 'BLUR2') & (Tw.rep == 2) & (Tw.phase_block == 0)][0]; Tw.loc[i, 'rep'] = 3
    r1, m1 = _raises(lambda: AN.contrast_f1(Tw))
    Ew = E.copy(); j = Ew.index[(Ew.rep == 2) & (Ew.phase_block == 0)]; Ew.loc[j, 'rep'] = 3
    r2, m2 = _raises(lambda: AN.summaries(Ew, pilot={}))
    rows.append(_row('A4_wrong_rep', r1 and r2, 'a wrong repetition number is refused (pairing and aggregation)', pairing=m1, aggregation=m2))
    Td = pd.concat([T, T.iloc[[3]]], ignore_index=True)
    r1, m1 = _raises(lambda: AN.contrast_f1(Td))
    rows.append(_row('A4_duplicate_key', r1 and 'duplicate' in m1, 'a duplicated key is refused (no cartesian product)', msg=m1))
    Tn = T.copy(); i = Tn.index[(Tn.env == 'BLUR2') & (Tn.phase_block == 2) & (Tn.rep == 1)][0]
    Tn.loc[i, ['state', 'glob_Ctau', 'loc_Ctau', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'n_accepted', 'c']] = ['no_output', 0.0, 0.0, 0, 0, 0, 0, 0]
    for col in ('glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'glob_surf_rmse', 'sup_surf_rmse', 'w_share', 'u_share'):
        Tn[col] = Tn[col].astype(object); Tn.loc[i, col] = None
    comp = AN.completeness(Tn, len(Tn), Tn.case_key.nunique()); En = AN.frame(AN.contrast_f1(Tn), Tn)
    ct = En[(En.metric == 'glob_Ctau') & (En.left_key == Tn.loc[i, 'row_key'])]; rm = En[(En.metric == 'glob_pt_rmse') & (En.left_key == Tn.loc[i, 'row_key'])]
    Sn = AN.summaries(En, pilot={}); sct = Sn[(Sn.metric == 'glob_Ctau')].iloc[0]
    rows.append(_row('A4_legal_no_output', comp['ok'] and bool(ct.defined.iloc[0]) and ct.value.iloc[0] == -0.5 and not bool(rm.defined.iloc[0]) and int(sct.n_pairs) == 8,
                     'legal no-output kept: Ctau 0 (pair defined), point error undefined (pair undefined), row stays in the denominator', completeness=comp, n_pairs=int(sct.n_pairs),
                     block2=sct.b2))
    Tv = T.copy(); Tv['glob_pt_rmse'] = Tv.glob_pt_rmse.astype(object); Tv.loc[Tv.index[0], 'glob_pt_rmse'] = None
    Tv2 = T.copy(); Tv2.loc[Tv2.index[1], 'glob_n_valid'] = 0
    c1 = AN.completeness(Tv, len(Tv), Tv.case_key.nunique()); c2 = AN.completeness(Tv2, len(Tv2), Tv2.case_key.nunique())
    rows.append(_row('A4_valid_metric_mismatch', not c1['ok'] and not c2['ok'], 'valid points without an error value, or an error without valid points -> completeness refuses', a=c1['problems'], b=c2['problems']))
    Tc = T.drop(columns=['rep'])
    c3 = AN.completeness(Tc, len(Tc), Tc.case_key.nunique()); r1, m1 = _raises(lambda: AN.contrast_f1(Tc))
    rows.append(_row('A4_missing_column', not c3['ok'] and 'missing columns' in c3['problems'][0] and r1, 'a missing column is refused explicitly (no silent skip)', completeness=c3['problems'], contrast=m1))
    Rk = pd.DataFrame([dict(row_key='L_w1.5_s6.3|B_curv|b1|s20|r1|BLUR3|P4', method='P4', w_increase=5, template_cand_id='C044672'),
                       dict(row_key='L_w1.5_s6.3|B_curv|b2|s20|r1|GRAD_Y|P4', method='P4', w_increase=5, template_cand_id='C044630'),
                       dict(row_key='L_w0.5_s8|M10_curv|b0|s40|r1|BLUR2|P4', method='P4', w_increase=1, template_cand_id='C044660')])
    s1 = AN.select_failure(Rk); s2 = AN.select_failure(Rk.assign(template_cand_id=['X', 'Y', 'Z']))
    rows.append(_row('A4_selection_not_cand_id', s1 == s2 and s1 is not None and '|b' in s1 and 'C04' not in s1,
                     'failure-case selection uses new case keys (w increase, then sha256 of the key); candidate ids never enter', selected=s1))
    vals = {0: 0.10, 1: 0.10, 2: 0.10, 3: 0.30}
    T5 = syn(ctau=lambda c, b, rp, e: (0.5 + vals[b]) if e == 'BLUR2' else 0.5)
    E5 = AN.frame(AN.contrast_f1(T5), T5)
    key = ('E1 env_vs_nominal', 'D_r1_p8x4', 'B_curv', 20.0, 'BLUR2', 'DF')
    s_pos = AN.summaries(E5, pilot={key: 0.02}); s_neg = AN.summaries(E5, pilot={key: -0.02})
    a = s_pos[s_pos.metric == 'glob_Ctau'].iloc[0]; b = s_neg[s_neg.metric == 'glob_Ctau'].iloc[0]
    ok = abs(a['mean'] - 0.15) < 1e-12 and int(a.n_pos) == 4 and all(a[f'loo{i}'] > 0 for i in range(4)) and a.verdict == 'reproduziert' and b.verdict == 'nicht reproduziert' \
        and not any('null' in v.lower() or 'zero' in v.lower() or 'near' in v.lower() for v in AN.VERDICTS)
    rows.append(_row('A4_mean_015', ok, '[+0.10, +0.10, +0.10, +0.30] -> mean +0.15 (15 pp), 4/4 positive, all leave-one-out positive; no "close to zero" category exists',
                     mean=a['mean'], blocks=[a[f'b{i}'] for i in range(4)], loo=[a[f'loo{i}'] for i in range(4)], verdict_pos_pilot=a.verdict, verdict_neg_pilot=b.verdict, verdicts=AN.VERDICTS))
    T0 = syn(blocks=(0,), reps=(0,), sigma=0.0); E0 = AN.frame(AN.contrast_f1(T0), T0); s0 = AN.summaries(E0, pilot={})
    E00 = pd.concat([E0, E0.assign(left_key=E0.left_key + 'x')], ignore_index=True); r1, m1 = _raises(lambda: AN.summaries(E00, pilot={}))
    rows.append(_row('A4_sigma0_single', bool(s0.sigma0_single.all()) and r1, 'sigma 0: exactly one rep-0 pair per group, reported as single example; two pairs are refused', msg=m1))
    return rows


def all_drills() -> list:
    return freeze_drills() + stats_drills()


def _packs(P, keys):
    import mp3_chain as CH3
    return {k: CH3.pack_read(P, k) for k in keys}


def entry_parent_checks(spec, M, parents) -> list:
    import mp3_forward as F
    import mp5_main as MM
    P = spec['paths']; pp = MM.preps(M); pk = _packs(P, parents); old = set(K.read_csv(K.MP3_OUT / 'manifest.csv').raster_key.dropna())
    Mx = M.set_index('case_key'); rows = []
    inp = {k: (pk[k][0].get('image') or {}) for k in parents}
    ok = all(inp[k].get('forward_key') == Mx.loc[k, 'forward_key'] and (inp[k].get('input') or {}).get('raster_key') == Mx.loc[k, 'raster_key'] and inp[k].get('input_key') == pp[k]['input_key']
             and (inp[k].get('input') or {}).get('raster_key') not in old for k in parents)
    rows.append(_row('P_inputs', ok and len(parents) == 22, 'nominal parents: forward key, actual raster (new phase, not a template raster) and input key equal the frozen manifest', n=len(parents)))
    det = {}
    for k in parents:
        reg = F.noise_array(pp[k]['noise']); h = None if reg is None else K.content_sha(reg)
        det[k] = dict(stored=inp[k]['diag'].get('noise_sha'), regenerated=h, seed=pp[k]['noise'].get('seed'))
    ok = all(v['stored'] == v['regenerated'] for v in det.values()) and all((v['seed'] is None) == (v['stored'] is None) for v in det.values())
    rows.append(_row('P_noise', ok, 'stored noise = regeneration from the block seed; sigma 0 draws none (no seed conversion)', noise=det))
    ok = all(inp[k]['diag'].get('psf_true_px') == 1.0 and inp[k]['diag'].get('gain_min') == inp[k]['diag'].get('gain_max') == 1.0 for k in parents)
    rows.append(_row('P_nominal_env', ok, 'parents are nominal: true PSF 1 px, gain 1 everywhere'))
    return rows


# Check executed entry cases, method coverage and image consistency.
def entry_checks(spec, M, keys) -> list:
    import mp3_analysis as A3
    import mp3_forward as F
    import mp5_analysis as AN
    import mp5_chain as CH
    import mp5_figs as FG
    import mp5_gate as G
    import mp5_main as MM
    from scipy.ndimage import gaussian_filter
    P = spec['paths']; pp = MM.preps(M); pk = _packs(P, keys); Mx = M.set_index('case_key'); rows = []
    old = set(K.read_csv(K.MP3_OUT / 'manifest.csv').raster_key.dropna())
    im = {k: (pk[k][0].get('image') or {}) for k in keys}
    ok = all(im[k].get('forward_key') == Mx.loc[k, 'forward_key'] and (im[k].get('input') or {}).get('raster_key') == Mx.loc[k, 'raster_key'] and im[k].get('input_key') == pp[k]['input_key']
             and (im[k].get('input') or {}).get('raster_key') not in old and abs(float((im[k].get('input') or {}).get('integral', -1)) - float(Mx.loc[k, 'integral'])) <= 1e-6
             and im[k].get('execution_fp') == CH.fingerprints()['execution'] for k in keys)
    rows.append(_row('B_inputs_match', ok, 'every subset image: new-phase raster, peak / integral, input key, forward key and execution fingerprint equal the frozen manifest; no template raster'))
    rec = {}
    for k in keys:
        _, A, _ = pk[k]; d = im[k]['diag']; env = Mx.loc[k, 'env']
        ok1 = d.get('psf_true_px') == float(Mx.loc[k, 'psf_true_px'])
        if 'fwd__irr_pre_psf' in A and env.startswith('BLUR'):
            ok1 = ok1 and bool(np.array_equal(gaussian_filter(A['fwd__irr_pre_psf'], sigma=float(Mx.loc[k, 'psf_true_px']), mode='constant'), A['fwd__irr_post_psf']))
        if 'fwd__gains' in A and env.startswith('GRAD'):
            ok1 = ok1 and bool(np.array_equal(A['fwd__gains'], F.gain(json.loads(Mx.loc[k, 'field_json']), A['fwd__hits_xy'][:, 0], A['fwd__hits_xy'][:, 1])))
        rec[k] = ok1
    nom = {}
    for k in keys:
        for m, r in (pk[k][0].get('results') or {}).items():
            nom.setdefault((Mx.loc[k, 'config'], Mx.loc[k, 'raster_key'], m), set()).add(json.dumps(K.clean({x: y for x, y in (r.get('nominal') or {}).items() if x in ('steger_threshold_design', 'a_floor_px2', 'window')}), sort_keys=True))
    rows.append(_row('B_psf_field_isolation', all(rec.values()) and all(len(v) == 1 for v in nom.values()) and {tuple(sorted(pp[k]['decoder_row'])) for k in keys} == {('family', 'input_key', 'radius_mm')},
                     'true PSF / field in force (stored arrays re-computed where kept); decoder design values identical across environments / sigma / rep for one design (configuration + raster; '
                     'the P4 design threshold scales with the registered peak of the raster); decoder rows carry no environment truth',
                     n_psf_checked=len(rec), decoder_nominal_sets={f'{a}|{b}|{c}': len(v) for (a, b, c), v in nom.items()}))
    det = {}; ok = True
    for k in keys:
        reg = F.noise_array(pp[k]['noise']); h = None if reg is None else K.content_sha(reg)
        det[k] = h; ok = ok and im[k]['diag'].get('noise_sha') == h
    Ms = M[M.case_key.isin(keys)]
    for nb, g in Ms[Ms.sigma > 0].groupby(['family', 'surface', 'phase_block', 'rep', 'sigma']):
        ok = ok and len({im[k]['diag'].get('noise_sha') for k in g.case_key}) == 1
    ok = ok and all(im[k]['diag'].get('noise_sha') is None for k in Ms[Ms.sigma == 0].case_key)
    rows.append(_row('B_noise_pairing', ok, 'noise = regeneration from the registered block seed; one shared field per (family, surface, block, rep, sigma) across environments and configurations; sigma 0 none',
                     n_groups=int(Ms[Ms.sigma > 0].groupby(['family', 'surface', 'phase_block', 'rep', 'sigma']).ngroups)))
    need = {k: CH.needs(P, pp[k]) for k in keys}
    meth = M[M.case_key.isin(keys)].methods.str.split(';').explode().value_counts().to_dict()
    par_ok = all((Mx.loc[k, 'env'] == 'NOMINAL') or (Mx.loc[k, 'parent_case_key'] in set(keys)) for k in keys)
    a = G.ledger_audit(P)
    ok = (len(keys) == MM.ENTRY_EXPECTED['images'] and sum(meth.values()) == MM.ENTRY_EXPECTED['methods'] and meth.get('D0') == 60 and meth.get('DF') == 60 and meth.get('P4') == 76
          and all(not v['render'] and all(x == 'none' for x in v['methods'].values()) for v in need.values()) and par_ok and not a['failures'] and not a['unclosed']
          and (a['counts'].get('forward_entry') or {}).get('ok') == 136)
    rows.append(_row('B_complete_unique', ok, '136 images / 196 method results (D0 60, DF 60, P4 76) verify against their identities; every parent in the subset; ledger closed, no failures',
                     methods=meth, ledger={k: v for k, v in a['counts'].items() if k in ('forward', 'forward_entry', 'decode', 'evaluation')}))
    T = AN.results_table(P, M[M.case_key.isin(keys)])
    comp = AN.completeness(T, 196, 136)
    chain = dict(completeness=comp)
    try:
        bal = T[T.case_key.isin(M[MM._match(M, MM.ENTRY_SUBSET['B1']) | MM._match(M, MM.ENTRY_SUBSET['B2']) | MM._match(M, MM.ENTRY_SUBSET['B6'])].case_key)]
        E = AN.contrasts(bal)
        f1_rest = AN.frame(AN.contrast_f1(T[~T.case_key.isin(bal.case_key)]), T)
        S = AN.summaries(E, allow_incomplete=True); Lv = AN.levels(bal, allow_incomplete=True); Rk = AN.risk(pd.concat([E, f1_rest], ignore_index=True), T)
        tmp = Path(tempfile.mkdtemp(prefix=K.DRILL_PREFIX, dir=K.drill_base()))
        figs = FG.all_figures(P, T, S, Lv, Rk, tmp, preview=True)
        sizes = {k: (Path(v).stat().st_size if isinstance(v, str) and v.endswith('.png') else None) for k, v in figs.items()}
        shutil.rmtree(tmp, ignore_errors=True)
        chain.update(n_pairs=int((E.record_type == 'pair').sum()), n_pairs_f1_rest=len(f1_rest), n_summary=len(S), n_levels=len(Lv), n_risk=len(Rk), failure_case=figs.get('failure_case'), figure_bytes=sizes)
        ok = comp['ok'] and all(v for k, v in sizes.items() if k in ('F1', 'F2', 'F3', 'F4', 'F5')) and (figs.get('F6') is None) == (figs.get('failure_case') is None)
    except K.Stop as e:
        ok = False; chain['error'] = e.msg
    rows.append(_row('B_real_chain', ok, 'subset through the real completeness gate, strict contrasts (balanced part B1 + B2 + B6; F1 for B3-B5), aggregation (preview), risk, selection and all figure templates',
                     **chain))
    picks = AN.selfcheck_pick(T); sc = A3.selfcheck(P, T, picks)
    rows.append(_row('B_selfcheck', all(x['ok'] for x in sc) and len(sc) >= 7, 'independent recomputation from stored arrays (accepted points, c / w / u, point errors) on stratified subset rows',
                     picks=picks, results=sc))
    runs = [r for r in (P.run_load().get('runs') or []) if r['mode'] == 'entry']
    tim = {}
    for r in runs:
        for cid, t in (r.get('timing') or {}).items():
            for s, v in (t or {}).items():
                tim.setdefault(('line' if cid.startswith('L_') else 'point', s), []).append(float(v))
    mean = {f'{a}|{b}': round(float(np.mean(v)), 3) for (a, b), v in tim.items()}
    nbytes = sum(f.stat().st_size for f in P.data.glob('*.npz'))
    per_img = nbytes / max(1, len(list(P.data.glob('*.npz'))))
    secs = sum(r.get('seconds', 0) for r in runs)
    proj_s = secs / 136.0 * 2314.0
    free = shutil.disk_usage(str(P.root)).free
    rows.append(_row('B_throughput', free > 3 * per_img * 2450 and per_img > 0, 'measured throughput and storage; projection for the remaining 2314 images; free disk > 3 x projected total',
                     subset_seconds=round(secs, 1), mean_step_s=mean, bytes_per_image=round(per_img), projected_remaining_min=round(proj_s / 60.0, 1),
                     projected_total_bytes=round(per_img * 2450), free_bytes=free))
    bad = G.verify_freeze(P.run_load().get('freeze'), spec['manifest'], MM.rules_all()); ch = MM.protected_changed(P)
    rows.append(_row('B_freeze_protected', not bad and not ch, 'after the subset: freeze and protected inputs unchanged', freeze=bad, protected=ch))
    return rows
