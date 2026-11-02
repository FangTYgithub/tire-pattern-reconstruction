# Exercise joint-study source, matrix and continuation checks.

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import s5u_common as K
import s5u_bind as BD
import s5u_entry as EN
import s5r_store as S
import s5t_view as V

PY = [sys.executable, '-B', str(K.ROOT / 'scripts' / 'surface_revision' / 's5_align05' / 's5u_entry.py')]


def _sub(args, env=None) -> dict:
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run(PY + args, capture_output=True, text=True, encoding='utf-8', cwd=str(K.ROOT), env=e)
    msg = ''
    try:
        msg = json.loads(p.stderr.strip().splitlines()[-1]).get('msg', '') if p.stderr.strip() else ''
    except Exception:
        msg = p.stderr[-200:]
    return dict(exit=p.returncode, msg=msg[:200])


def _wjson(p, obj):
    K.K3.atomic(p, lambda t: t.write_text(json.dumps(K.clean(obj), ensure_ascii=False), encoding='utf-8'), K.OUT)


def _formal_reserved(aid) -> int:
    return sum(1 for e in K.json_lines(K.LEDGER) if e.get('auth') == aid and e.get('status') == 'reserved')


def d1_scope(a) -> dict:
    ISO = K.ISO; out = {}; r0 = _formal_reserved(a['auth_id'])
    _wjson(ISO / 'caps.json', dict(K.CAPS, render=2873)); _wjson(ISO / 'k1.json', [list(a['excluded_complete'][0])])
    out['real_with_caps'] = _sub(['run', '--select', 'remaining', '--caps', str(ISO / 'caps.json')])
    out['real_with_keys'] = _sub(['run', '--select', 'remaining', '--keys', str(ISO / 'k1.json')])
    out['real_without_select'] = _sub(['run'])
    out['sentinel_unauthorised_key'] = _sub(['run', '--mode', 'sentinel', '--store', str(ISO / 's1' / 'store'), '--keys', str(ISO / 'k1.json')])
    out['formal_reservations_unchanged'] = _formal_reserved(a['auth_id']) == r0
    out['ok'] = bool(out['real_with_caps']['exit'] == 15 and out['real_with_keys']['exit'] == 15 and out['real_without_select']['exit'] == 10
                     and out['sentinel_unauthorised_key']['exit'] == 15 and out['formal_reservations_unchanged'] and not (ISO / 's1').exists())
    return out


def d2_sources(a, keys2) -> dict:
    ISO = K.ISO; out = {}; r = K.run_load()

    rf = copy.deepcopy(r); f0 = next(iter(rf['freeze']['files'])); rf['freeze']['files'][f0] = '0' * 64; _wjson(ISO / 'run_freeze.json', rf)
    rc = copy.deepcopy(r); k0 = next(iter(rc['compat_align04']['records'])); rc['compat_align04']['records'][k0]['json_sha256'] = '0' * 64; _wjson(ISO / 'run_compat.json', rc)
    _wjson(ISO / 'k2.json', [list(k) for k in keys2])
    for tag, env in (('freeze_changed', {'S5U_RUN': str(ISO / 'run_freeze.json')}), ('old_source_changed', {'S5U_RUN': str(ISO / 'run_compat.json')})):
        out[tag] = dict(run=_sub(['run', '--mode', 'sentinel', '--store', str(ISO / f's_{tag}' / 'store'), '--keys', str(ISO / 'k2.json')], env),
                        analyze=_sub(['analyze', '--store', str(ISO / f's_{tag}' / 'store')], env), figs=_sub(['figs'], env))

    caps = dict(render=4, decode_D0=4, decode_DF=4, decode_P4=4, evaluate=8); _wjson(ISO / 'caps_small.json', caps)
    st = ISO / 'j' / 'store'
    first = _sub(['run', '--mode', 'stub', '--store', str(st), '--keys', str(ISO / 'k2.json'), '--caps', str(ISO / 'caps_small.json')])
    again = _sub(['run', '--mode', 'stub', '--store', str(st), '--keys', str(ISO / 'k2.json'), '--caps', str(ISO / 'caps_small.json')])
    ej = sorted((st / 'eval').glob('*.json'))[0]; rec = K.load_json(ej); npz_before = K.sha(ej.with_suffix('.npz'))
    rec['meta']['metrics']['n_valid'] = 99; ej.write_text(json.dumps(rec), encoding='utf-8')
    out['json_only_changed'] = dict(first=first, rerun_before_change=again, rerun=_sub(['run', '--mode', 'stub', '--store', str(st), '--keys', str(ISO / 'k2.json'), '--caps', str(ISO / 'caps_small.json')]),
                                    analyze=_sub(['analyze', '--store', str(st)]), npz_unchanged=K.sha(ej.with_suffix('.npz')) == npz_before)

    rg = copy.deepcopy(r); rg['analysis'] = dict(status='COMPLETE', files={K.rel(ISO / 'fake.csv'): '0' * 64}); (ISO / 'fake.csv').write_text('x', encoding='utf-8')
    _wjson(ISO / 'run_tables.json', rg)
    out['figs_table_hash'] = _sub(['figs'], {'S5U_RUN': str(ISO / 'run_tables.json')})
    ok = (all(v['exit'] == 16 for v in out['freeze_changed'].values()) and all(v['exit'] == 8 for v in out['old_source_changed'].values())
          and first['exit'] == 0 and again['exit'] == 0 and out['json_only_changed']['rerun']['exit'] == 5 and out['json_only_changed']['analyze']['exit'] == 5
          and out['json_only_changed']['npz_unchanged'] and out['figs_table_hash']['exit'] == 5)
    out['ok'] = bool(ok)
    return out


def d3_gate(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    D = pd.concat([old, new], ignore_index=True, sort=False)
    keys = {tuple(x) for x in D[V.KEYS].itertuples(index=False, name=None)}
    exp = {(c, s, p, float(sg), int(r), m) for (c, s, p, sg, r, m) in keys}
    neg = {}
    i40 = D.index[(D.sigma == 40.0) & (D.rep == 2) & (D.method == 'D0')][0]
    neg['duplicate_rep'] = V.strict_gate(pd.concat([D, D.loc[[i40]]]), exp, 'subset', 'x')
    neg['missing_rep'] = V.strict_gate(D.drop(index=i40), exp, 'subset', 'x')
    T = D.copy(); j = T.index[T.method == 'DF'][0]; T.loc[j, 'image_key'] = 'S5X-000000000000000000000000'; neg['image_key_mismatch'] = V.strict_gate(T, exp, 'subset', 'x')
    neg['missing_method'] = V.strict_gate(D.drop(index=D.index[D.method == 'DF'][0]), exp, 'subset', 'x')
    neg['missing_metric_column'] = V.strict_gate(D.drop(columns=['glob_pt_rmse']), exp, 'subset', 'x')
    T = D.copy(); k = T.index[(T.state == 'ok') & (pd.to_numeric(T.glob_n_valid) > 0)][0]; T.loc[k, 'glob_pt_p95'] = np.nan; neg['missing_metric_value'] = V.strict_gate(T, exp, 'subset', 'x')
    pos = V.strict_gate(D, exp, 'subset', 'old_1728_plus_next8')
    return dict(negative={k: dict(refused=not v['ok'], by_check=v['by_check']) for k, v in neg.items()},
                positive=dict(passed=pos['ok'], n_rows=pos['n_rows'], n_no_output=pos.get('n_no_output'), n_valid_without_global_interp=pos.get('n_valid_without_global_interp')),
                ok=bool(all(not v['ok'] for v in neg.values()) and pos['ok']))


def d4_structure(old: pd.DataFrame, new: pd.DataFrame, M: pd.DataFrame) -> dict:

    import s5u_stats as ST
    import s5u_figs as FG
    base = old.copy(); syn = [base]
    for sg in (20.0, 60.0, 80.0, 100.0):
        x = base[base.sigma == 40.0].copy(); x['sigma'] = sg; syn.append(x)
    D = pd.concat(syn, ignore_index=True)
    C = ST.case_table(D, new.iloc[0:0], M)
    E = ST.effects(C); FC = ST.frontend_conditions(C); Sm = ST.condition_summary(C)
    sig = sorted(E[E.record_type == 'interaction'].sigma.astype(str).unique())
    n_pairs = FC.groupby('sigma').n_pairs.unique().to_dict()
    reg = ST.regression_old_slice(ST.case_table(old, new, M))
    figs = []
    try:
        out_dir = K.ISO / 'figs_test'
        figs = FG.f_range(Sm, 'point', out_dir, ' TEST') + FG.f_range(Sm, 'line', out_dir, ' TEST') + FG.f_fourcell(E, out_dir, ' TEST') + FG.f_frontend(FC, C, out_dir, ' TEST')
        sel = FG.select_cases(C)
        Creal = ST.case_table(old, new, M)
        sel_t = dict(point=dict(cells=[dict(role='test', front_end='D0', series=[['D_r0.75_p8x4', 'B_curv', 'ph0', 0.0, 0], ['D_r0.75_p8x4', 'B_curv', 'ph0', 40.0, 1], ['D_r0.75_p8x4', 'B_curv', 'ph0', 100.0, 1]])]),
                     line=dict(cells=[dict(role='test', front_end='P4', series=[['L_w1.5_s4.45', 'B_curv', 'ph0', 0.0, 0], ['L_w1.5_s4.45', 'B_curv', 'ph0', 40.0, 1], ['L_w1.5_s4.45', 'B_curv', 'ph0', 100.0, 1]])]),
                     diagnostic=dict(series=[['L_w0.5_s6.3', 'M10_curv', 'ph1', 20.0, 1]]))
        figs += FG.f_cases(Creal, sel_t, out_dir, ' TEST')
        fig_err = None
    except Exception as e:
        fig_err = repr(e)[:300]; sel = None
    ok = (sig == sorted(['0.0', '20.0', '40.0', '60.0', '80.0', '100.0', '40-0']) and all(v.tolist() == [16] for k, v in n_pairs.items() if k > 0)
          and n_pairs.get(0.0, np.array([])).tolist() == [8] and reg['ok'] and fig_err is None)
    return dict(sigma_rows=sig, pairs_per_condition={str(k): v.tolist() for k, v in n_pairs.items()}, n_effect_rows=len(E), regression_old_slice=reg,
                synthetic_selection=sel, figure_test_files=len(figs), figure_error=fig_err, ok=bool(ok))


def d5_reuse_and_full(a, pf) -> dict:
    out = {}
    out['full_mode'] = _sub(['analyze'])
    cp4 = pf['cp4']; reg = BD.Registry.load(cp4, None); st = BD.BoundStore(K.FORMAL_STORE, K.FORMAL_ROOT, reg)
    keys8 = [tuple(x) for x in a['excluded_complete']]; rows = BD.rows_of(keys8, pf['M'])
    before = {k: v['json_sha256'] for k, v in cp4['records'].items()}
    ex = S.Forbidden(); led = BD.Ledger5(K.ISO / 'reuse' / 'ledger.jsonl', 'ISO-' + a['auth_id'], K.CAPS, {r['manifest_key'] for r in rows})
    log = BD.run_formal(st, rows, ex, S.platform_refs(), led)
    after = {k: K.sha(K.FORMAL_STORE / k.split('|', 1)[0] / f"{k.split('|', 1)[1]}.json") for k in before}
    out['accepted_8_reuse'] = dict(reuse=log['reuse'], computed=log['computed'], calls=len(ex.calls), records_unchanged=before == after)
    out['technical_exit'] = _sub(['run', '--mode', 'stub', '--store', str(K.ISO / 'f' / 'store'), '--keys', str(K.ISO / 'k2.json'), '--caps', str(K.ISO / 'caps_small.json'), '--fail-after', '0'])
    out['ok'] = bool(out['full_mode']['exit'] == 3 and log['reuse'] == 32 and log['computed'] == 0 and not ex.calls and before == after and out['technical_exit']['exit'] == 2)
    return out


def run_all() -> dict:
    ISO = K.ISO
    if ISO.exists():
        shutil.rmtree(ISO)
    ISO.mkdir(parents=True)
    try:
        pf = BD.preflight('drills'); a = pf['auth']
        keys2 = [k for k in pf['keys'] if k[0] == 'D_r1_p8x4'][:1] + [k for k in pf['keys'] if k[0] == 'L_w0.5_s8'][:1]
        r1 = d1_scope(a); r2 = d2_sources(a, keys2)
        old, new, M, probs = EN.build_view(pf)
        r3 = d3_gate(old, new); r4 = d4_structure(old, new, M); r5 = d5_reuse_and_full(a, pf)
    finally:
        if ISO.exists() and ISO.resolve() == (K.OUT / '_isolation').resolve():
            shutil.rmtree(ISO)
    res = dict(scope_quota=r1, sources_json=r2, result_gate=r3, structure_stats=r4, reuse_full_technical=r5, isolation_removed=not ISO.exists(), t=K.now(),
               freeze_digest=K.run_load()['freeze']['digest'])
    res['summary'] = dict(scope_quota=r1['ok'], sources_json=r2['ok'], result_gate=r3['ok'], structure_stats=r4['ok'], reuse_full_technical=r5['ok'], isolation_removed=res['isolation_removed'])
    res['all_ok'] = all(res['summary'].values())
    K.run_put('drills', res)
    return res


def final() -> dict:
    r = K.run_load(); src = r.get('sources_read', {})
    bad = [k for k, h in src.items() if K.sha(K.ROOT / k if not k.startswith('C:') else k) != h]
    fz = BD.check_freeze()
    a = r['authorisation']; ev = K.json_lines(K.LEDGER)
    by = {}
    for e in ev:
        by.setdefault(e['auth'], {}).setdefault(e['kind'], {}).setdefault(e['status'], 0); by[e['auth']][e['kind']][e['status']] += 1
    outs = {K.rel(p): K.sha(p) for p in sorted(K.OUT.rglob('*')) if p.is_file() and p.name != 'run.json' and '.tmp' not in p.name}
    code = {K.rel(p): K.sha(p) for p in sorted((K.ROOT / 'scripts' / 'surface_revision' / 's5_align05').glob('*.py'))}
    res = dict(sources_unchanged=not bad, changed=bad[:10], n_sources=len(src), freeze=fz, ledger_by_auth=by, auth_id=a['auth_id'],
               store_status=K.load_json(K.FORMAL_STORE / 'status.json'), analysis_status=(r.get('analysis') or {}).get('status'), outputs=outs, code=code, t=K.now())
    K.run_put('final', res)
    return {k: v for k, v in res.items() if k not in ('outputs', 'code')}
