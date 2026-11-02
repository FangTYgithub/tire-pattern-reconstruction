# Dispatch joint-study pairing, figure and result checks.

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

import s5v_common as K
import s5v_effects as EF
import s5v_figs as FIG
import s5v_fig3d as F3
import s5v_drills as DR
import s5u_bind as BD

K5 = K.K5


def inputs_check() -> dict:
    r5 = K.load_json(K.A5RUN); an = r5.get('analysis') or {}
    if an.get('status') != 'COMPLETE':
        raise K.Stop('refused', 'S5-ALIGN-05 analysis is not COMPLETE')
    bad = []
    files = dict(an['files'])
    figs = dict((r5.get('figures') or {}).get('files') or {}); so = r5.get('figure_supplement_output') or {}
    figs[so['file']] = so['sha256']
    for f, h in list(files.items()) + list(figs.items()):
        if not (K.ROOT / f).exists() or K.sha(K.ROOT / f) != h:
            bad.append(f)
    if bad:
        raise K.Stop('corrupt', f'{len(bad)} S5-ALIGN-05 deliverables differ from their registered hashes', detail=bad)
    fz = BD.check_freeze(r5['freeze'])
    led = K5.LEDGER
    return dict(ok=True, run05=dict(path=K.rel(K.A5RUN), sha256=K.sha(K.A5RUN)), analysis_files=files, figures05=figs, freeze05=fz,
                ledger=dict(path=K.rel(led), sha256=K.sha(led), n_events=len(led.read_text(encoding='utf-8').splitlines())),
                selection=r5['figures']['selection'], t=K.now())


def binder(r5: dict):
    reg = BD.Registry.load(r5['compat_align04'], r5['authorisation']['auth_id'])
    a3 = K.load_json(K5.A3RUN)['compat']['records']; by_eval = {v['eval']['json'].replace('.json', '.npz'): v for v in a3.values()}
    used = []

    def arrays(r, fe):
        ep = K.ROOT / r.eval_record
        if str(r.source).startswith('S5-ALIGN-04/05'):
            ip = K5.FORMAL_STORE / 'image' / f'{r.image_key}.npz'
            bi, be = reg.m.get(('image', r.image_key)), reg.m.get(('eval', f'{r.image_key}__{fe}'))
            hi, he = (bi or {}).get('npz_sha256'), (be or {}).get('npz_sha256')
        else:
            if fe == 'DF':
                raise K.Stop('refused', 'old DF records are shards of many cases: not drawn')
            v = by_eval.get(r.eval_record)
            if v is None:
                raise K.Stop('unbound', f'{r.eval_record} has no S5-ALIGN-03 binding')
            ip = K.ROOT / v['image']['json'].replace('.json', '.npz'); hi, he = v['image']['npz_sha256'], v['eval']['npz_sha256']
        if not hi or not he or K.sha(ip) != hi or K.sha(ep) != he:
            raise K.Stop('corrupt', f'{K.rel(ep)} / {K.rel(ip)}: not bound or changed')
        used.append(dict(eval=K.rel(ep), eval_sha256=he, image=K.rel(ip), image_sha256=hi))
        return K.load_npz(ip)['image'], K.load_npz(ep)
    return arrays, used


def _paired_path():
    return K.OUT / 'paired_effects.csv'


def cmd_register() -> dict:
    F3.bound_mesh('M10_curv')
    ic = inputs_check(); K.run_put('inputs_at_registration', ic)
    out = K.register_code()
    hist = K.run_load().get('code_registrations', []) + [dict(t=out['t'], digest=out['digest'], n_files=out['n_files'])]
    K.run_put('code_registrations', hist)
    return dict(n_files=out['n_files'], digest=out['digest'])


def cmd_effects() -> dict:
    cc = K.check_code(); ic = inputs_check()
    C = K.read_csv(K.A5 / 'case_results.csv')
    if len(C) != 6336:
        raise K.Stop('refused', f'case table has {len(C)} rows, expected 6336')
    t0 = time.time(); E = EF.build(C); FC = EF.frontend(C)
    v1 = EF.validate(E); EF.validate_fc(FC)
    p = K.save_csv(pd.concat([E, FC], ignore_index=True, sort=False), _paired_path()); h = K.sha(p)
    En, FCn = EF.load_paired(p); v2 = EF.validate(En)
    if len(En) + len(FCn) != 2997:
        raise K.Stop('refused', f'paired_effects.csv has {len(En) + len(FCn)} rows, expected 2997')
    E5, FC5 = EF.load_paired(K.A5 / 'paired_effects.csv')
    cmp = EF.compare_old(En, FCn, E5, FC5)
    old = set(map(tuple, E5[EF.KEYCOLS].values)); new = En[[tuple(r) not in old for r in En[EF.KEYCOLS].values]]
    if len(new) != 67 or (new.state == 'ok').any():
        raise K.Stop('refused', f'{len(new)} rows beyond S5-ALIGN-05 ({int((new.state == "ok").sum())} ok): expected 67 explicit undefined rows')
    out = dict(code=cc, csv=K.rel(p), csv_sha256=h, n_rows=len(En) + len(FCn), n_effects=len(En), n_frontend=len(FCn), validate=v2, validate_before_write=v1,
               comparison_with_05=cmp, n_new_rows=len(new), new_rows_by=new.groupby(['derivation', 'metric', 'undefined_class']).size().reset_index(name='n').to_dict('records'),
               new_row_keys=new[EF.KEYCOLS + ['n_phases_complete', 'undefined_class', 'reason']].to_dict('records'),
               reader='s5v_effects.load_paired: split by table first; effect sigma kept as label incl. 40-0; front-end sigma checked against the six labels, then float',
               seconds=round(time.time() - t0, 1), t=K.now())
    K.run_put('effects', out)
    return {k: v for k, v in out.items() if k not in ('new_row_keys', 'code')}


def _read_tables() -> tuple:
    eff = K.run_load().get('effects') or {}
    p = _paired_path()
    if not p.exists() or K.sha(p) != eff.get('csv_sha256'):
        raise K.Stop('corrupt', 'paired_effects.csv missing or differs from the effects record: run `effects`')
    E, FC = EF.load_paired(p); EF.validate(E)
    C = K.read_csv(K.A5 / 'case_results.csv'); S = K.read_csv(K.A5 / 'condition_summary.csv')
    return E, FC, C, S


def cmd_drills() -> dict:
    K.check_code()
    E, FC, C, _ = _read_tables(); E5, FC5 = EF.load_paired(K.A5 / 'paired_effects.csv')
    r = DR.run_all(C, E, FC, E5, FC5); K.run_put('drills', r)
    if not r['all_ok']:
        raise K.Stop('refused', 'a counterexample did not behave as required', detail=r['summary'])
    return r['summary']


def cmd_figs() -> dict:
    cc = K.check_code(); ic = inputs_check(); r5 = K.load_json(K.A5RUN)
    E, FC, C, S = _read_tables(); arrays, used = binder(r5)
    res = {}
    for x in [FIG.f2(S), FIG.f3(E), FIG.f4(FC, C), FIG.f4b(FC)] + FIG.f_cases(C, ic['selection'], arrays) + F3.draw(C, arrays):
        res[x['file'].rsplit('/', 1)[-1][:-4]] = x
    old = ic['figures05']; idx = []
    f1 = next(f for f in old if f.endswith('F1_Punkte_Gesamtbereich.png'))
    idx.append(dict(figure='F1_Punkte_Gesamtbereich', source='S5-ALIGN-05 accepted, unchanged', path=str((K.ROOT / f1).resolve()), sha256=old[f1]))
    for name, x in res.items():
        p = K.ROOT / x['file']; prev = next((f for f in old if f.endswith(f'/{name}.png')), None)
        idx.append(dict(figure=name, source='S5-ALIGN-06 new (3-D, saved data)' if name.startswith(('F7', 'F8')) else 'S5-ALIGN-06 corrected',
                        path=str(p.resolve()), sha256=K.sha(p), replaces=(dict(path=str((K.ROOT / prev).resolve()), sha256=old[prev]) if prev else None),
                        checks={k: v for k, v in x.items() if k != 'file'}))
    out = dict(code=cc, index=idx, n_files_06=len(res), records_drawn=used, mesh=dict(surface='M10_curv', sha16=K.REGISTERED_MESH['M10_curv'], registered_in='outputs/surface_revision/s345_ready/checks.json'),
               selection_source='outputs/surface_revision/s5_align05/run.json -> figures.selection (unchanged)', t=K.now())
    K.run_put('figures', out)
    return dict(n_files_06=len(res), F4={k: v for k, v in res['F4_Frontend_Identitaetsrisiko'].items() if k != 'file'},
                F4b={k: v for k, v in res['F4b_Frontend_Fehler'].items() if k != 'file'})


def cmd_final() -> dict:
    cc = K.check_code(); now = inputs_check(); r = K.run_load(); was = r.get('inputs_at_registration') or {}
    diff = [k for k in ('run05', 'analysis_files', 'figures05', 'ledger') if was.get(k) != now.get(k)]
    if was.get('freeze05', {}).get('digest') != now['freeze05']['digest']:
        diff.append('freeze05')
    if diff:
        raise K.Stop('corrupt', f'S5-ALIGN-05 sources changed during this round: {diff}')
    outs = {K.rel(K.OUT / 'paired_effects.csv'): K.sha(K.OUT / 'paired_effects.csv')}
    for x in (r.get('figures') or {}).get('index', []):
        if x['source'].startswith('S5-ALIGN-06'):
            from pathlib import Path
            if K.sha(Path(x['path'])) != x['sha256']:
                raise K.Stop('corrupt', f"{x['path']} differs from the figure index")
            outs[K.rel(Path(x['path']))] = x['sha256']
    if outs[K.rel(K.OUT / 'paired_effects.csv')] != (r.get('effects') or {}).get('csv_sha256'):
        raise K.Stop('corrupt', 'paired_effects.csv differs from the effects record')
    out = dict(status='NUMERICAL_CHECKS_COMPLETE', new_imaging=0, new_decoding=0, new_case_evaluation=0, code=cc,
               sources_05_unchanged=True, ledger=now['ledger'], freeze05=now['freeze05'], outputs_06=outs, t=K.now())
    K.run_put('final', out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['all', 'register', 'effects', 'drills', 'figs', 'final'])
    a = ap.parse_args(); t0 = time.time(); rec = dict(cmd=a.cmd, t=K.now())
    try:
        steps = ['register', 'effects', 'drills', 'figs', 'final'] if a.cmd == 'all' else [a.cmd]
        res = {}
        for s in steps:
            res[s] = dict(register=cmd_register, effects=cmd_effects, drills=cmd_drills, figs=cmd_figs, final=cmd_final)[s]()
            print(f'[{K.now()}] {s} ok', flush=True)
        print(json.dumps(K.clean(res), ensure_ascii=False, default=str)[:4000])
        rec.update(exit=0, seconds=round(time.time() - t0, 1))
        code = 0
    except K.Stop as e:
        print(json.dumps(dict(code=e.code, msg=e.msg, detail=K.clean(e.detail)), ensure_ascii=False, default=str)[:3000], file=sys.stderr)
        rec.update(exit=K.EXIT.get(e.code, 2), code=e.code, msg=e.msg[:300], seconds=round(time.time() - t0, 1)); code = rec['exit']
    except Exception as e:
        import traceback
        traceback.print_exc()
        rec.update(exit=2, code='technical', msg=repr(e)[:300], seconds=round(time.time() - t0, 1)); code = 2
    try:
        K.run_put('invocations', (K.run_load().get('invocations') or [])[-40:] + [rec])
    except Exception:
        pass
    return code


if __name__ == '__main__':
    sys.exit(main())
