# Exercise joint-study case coverage, pairing and storage checks.

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import s5r_common as K
import s5r_gates as G
import s5r_main as E
import s5r_store as S

PY = [sys.executable, '-B', str(K.HERE / 's5r_main.py')]


def _expect_refused(fn) -> dict:
    try:
        fn()
    except K.Stop as e:
        d = e.detail if isinstance(e.detail, dict) else {}
        return dict(refused=True, code=e.code, by_check=d.get('by_check'), first=(d.get('reasons') or [None])[0] if isinstance(d.get('reasons'), list) else str(e.detail)[:200])
    return dict(refused=False)


def _expect_pass(fn) -> dict:
    try:
        r = fn()
        return dict(passed=True, by_check=r.get('by_check') if isinstance(r, dict) else None)
    except K.Stop as e:
        return dict(passed=False, code=e.code, detail=str(e.detail)[:300])


def drills_manifest() -> dict:
    M0 = K.read_csv(K.reg(K.MANIFEST)); reg = G.registered()
    gate = lambda M, mode='full', keys=None: E.cmd_gate_manifest(M, mode, keys, reg)
    cell = (M0.combo == 'D_r1_p8x4') & (M0.surface == 'B_curv') & (M0.phase_label == 'ph3')
    i_cell = list(M0.index[cell]); i_s60r2 = M0.index[cell & (M0.sigma == 60.0) & (M0.rep == 2)][0]
    i_pos = M0.index[M0.sigma > 0][0]; i_zero = M0.index[M0.sigma == 0][0]; i_dot80 = M0.index[(M0.family == 'point') & (M0.sigma == 80.0)][0]
    neg = {}

    def setv(col, idx, val, obj=False):
        M = M0.copy()
        if obj:
            M[col] = M[col].astype(object)
        M.loc[idx, col] = val
        return M
    seeds80 = M0.copy(); seeds80.loc[seeds80.sigma == 80.0, 'noise_seed'] = pd.to_numeric(seeds80.loc[seeds80.sigma == 80.0, 'noise_seed']) + 100
    neg['delete_cell_11_rows'] = lambda: gate(M0[~cell])
    neg['delete_surface'] = lambda: gate(M0[M0.surface != 'M24_curv'])
    neg['delete_phase'] = lambda: gate(M0[M0.phase_label != 'ph5'])
    neg['sigma100_to_120'] = lambda: gate(setv('sigma', M0.index[M0.sigma == 100.0], 120.0))
    neg['rep_duplicate_label'] = lambda: gate(setv('rep', i_s60r2, 1))
    neg['cross_sigma_seed_shift_paired'] = lambda: gate(seeds80)
    neg['missing_column'] = lambda: gate(M0.drop(columns=['raster_key']))
    neg['negative_peak'] = lambda: gate(setv('peak', i_cell[0], -0.1))
    neg['over_budget_integral_11696'] = lambda: gate(setv('integral', i_cell[0], 11696.0))
    neg['physical_change_identity_kept'] = lambda: gate(setv('peak', i_cell[4], float(M0.loc[i_cell[4], 'peak']) * 1.01))
    neg['raster_change_identity_kept'] = lambda: gate(setv('raster_key', i_cell[4], 'R-0000000000000000000a'))
    neg['duplicate_row'] = lambda: gate(pd.concat([M0, M0.iloc[[7]]]))
    neg['nan_seed_sigma_gt_0'] = lambda: gate(setv('noise_seed', i_pos, np.nan))
    neg['sigma0_with_seed'] = lambda: gate(setv('noise_seed', i_zero, 123))
    neg['missing_rep_row'] = lambda: gate(M0.drop(index=i_s60r2))
    neg['unpaired_seed_one_row'] = lambda: gate(setv('noise_seed', i_dot80, float(M0.loc[i_dot80, 'noise_seed']) + 1))
    neg['text_in_numeric'] = lambda: gate(setv('peak', 3, 'abc', obj=True))
    neg['incomplete_full_not_downgraded'] = lambda: gate(M0[~cell], 'full')
    keys8 = next8_keys()
    sub8 = M0[[tuple(x) in set(keys8) for x in M0[['combo', 'surface', 'phase_label', 'sigma', 'rep']].itertuples(index=False, name=None)]]
    neg['subset_without_keys'] = lambda: gate(sub8, 'subset', None)
    neg['subset_missing_member'] = lambda: gate(sub8.iloc[:7], 'subset', keys8)
    pos = dict(legit_full=lambda: gate(M0), subset_prebound_8=lambda: gate(sub8, 'subset', keys8))
    res = dict(negative={k: _expect_refused(f) for k, f in neg.items()}, positive={k: _expect_pass(f) for k, f in pos.items()})

    ISO = E.ISO; K.guard(ISO / 'x')
    bad_csv = ISO / 'manifest_missing_column.csv'; M0.drop(columns=['raster_key']).to_csv(bad_csv, index=False)
    c1 = subprocess.run(PY + ['gate-manifest'], capture_output=True, text=True, encoding='utf-8'); c2 = subprocess.run(PY + ['gate-manifest', '--manifest', str(bad_csv)], capture_output=True, text=True, encoding='utf-8')
    res['cli'] = dict(real_manifest_exit=c1.returncode, missing_column_exit=c2.returncode, missing_column_message=c2.stderr[:300])
    res['ok'] = bool(all(v['refused'] for v in res['negative'].values()) and all(v['passed'] for v in res['positive'].values()) and c1.returncode == 0 and c2.returncode == K.EXIT['refused'])
    return res


# Test result coverage and pairing checks with incomplete and inconsistent records.
def drills_results() -> dict:
    import s5r_bind as B
    cp = K.run_load()['compat']; V = B.views(cp); ex = B.expected_sets()
    main = V['main']
    g = lambda D: G.gate_results(D, ex['main'], 'full', 'main')

    def mod(fn):
        D = main.copy(); fn(D); return D
    i_valid = main.index[(pd.to_numeric(main.glob_n_valid) > 0) & (main.method == 'D0')][0]
    i_ok = main.index[(main.state == 'ok') & (main.method == 'P4')][0]
    i_no = main.index[main.state == 'no_output'][0]
    i_df = main.index[main.method == 'DF'][0]

    def zero(D, i):
        for c in ('n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'id_denominator', 'glob_id_correct', 'glob_id_wrong', 'glob_id_unconf',
                  'loc_id_correct', 'loc_id_wrong', 'loc_id_unconf'):
            D.loc[i, c] = 0
        for c in ('glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95'):
            D.loc[i, c] = np.nan
        D.loc[i, 'state'] = 'no_output'
    neg = {}
    neg['valid_points_missing_point_error'] = mod(lambda D: D.__setitem__('glob_pt_rmse', D.glob_pt_rmse.where(D.index != i_valid, np.nan)))
    neg['zero_valid_but_positive_coverage'] = mod(lambda D: zero(D, i_ok))
    neg['no_output_with_local_coverage'] = mod(lambda D: [D.__setitem__(c, D[c].where(D.index != i_no, 0.05)) for c in ('loc_Ctau', 'loc_Cvalid', 'loc_cov_interp')])
    neg['technical_failure_zero_filled'] = mod(lambda D: (zero(D, i_ok), [D.__setitem__(c, D[c].where(D.index != i_ok, 0.0)) for c in ('glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'loc_Ctau', 'loc_Cvalid', 'loc_cov_interp')],
                                                          D.__setitem__('record_complete', D.record_complete.where(D.index != i_ok, False)), D.__setitem__('content_sha', D.content_sha.where(D.index != i_ok, ''))))
    neg['technical_state'] = mod(lambda D: D.__setitem__('state', D.state.where(D.index != i_ok, 'technical_error')))
    neg['missing_method_record'] = main.drop(index=i_df)
    neg['pairing_mismatch'] = mod(lambda D: D.__setitem__('image_key', D.image_key.where(D.index != i_df, 'S2G-000000000000000000000000')))
    neg['duplicate_record'] = pd.concat([main, main.loc[[i_df]]])
    neg['missing_field'] = main.drop(columns=['glob_pt_p95'])
    neg['surface_error_without_interpolation'] = mod(lambda D: D.__setitem__('glob_surf_rmse', D.glob_surf_rmse.where(~((pd.to_numeric(D.glob_cov_interp) == 0) & (pd.to_numeric(D.n_valid) > 0)), 0.1)))
    res = dict(negative={k: dict(refused=not g(D)['ok'], by_check=g(D)['by_check']) for k, D in neg.items()})

    pos = dict(real_main=g(main), real_failure=G.gate_results(V['failure'], ex['failure'], 'full', 'failure'), real_gauss=G.gate_results(V['gauss'], ex['gauss'], 'full', 'gauss'))

    def acc_no_valid(D):
        D.loc[i_no, 'n_accepted'] = 5; D.loc[i_no, 'n_out'] = 5; D.loc[i_no, 'id_unconf'] = 5; D.loc[i_no, 'id_denominator'] = 5
    pos['accepted_but_no_valid_point'] = g(mod(acc_no_valid))
    res['positive'] = {k: dict(passed=v['ok'], by_check=v['by_check'], n_no_output=v.get('n_no_output'), n_valid_without_global_interp=v.get('n_valid_without_global_interp'),
                               n_local_cov_without_local_points=v.get('n_local_cov_without_local_points')) for k, v in pos.items()}

    def hook(Vx):
        Vx = dict(Vx); Vx['main'] = neg['valid_points_missing_point_error']; return Vx
    ae = {}
    ae['analysis_full_with_bad_record'] = _expect_refused(lambda: B.analyze('full', None, view_hook=hook, write=False))
    ae['analysis_subset_without_keys'] = _expect_refused(lambda: B.analyze('subset', None, write=False))
    sk = [k for k in K.expected_keys(combos=('D_r1.5_p6x4', 'L_w0.5_s6.3'), surfaces=('M10_curv',), phases=('ph1',), sigmas=K.EXISTING_SIGMAS)]
    sp = _expect_pass(lambda: B.analyze('subset', sk, write=False))
    ae['analysis_subset_prebound_passes_as_subset'] = sp
    res['analysis_entry'] = ae
    res['ok'] = bool(all(v['refused'] for v in res['negative'].values()) and all(v['passed'] for v in res['positive'].values())
                     and ae['analysis_full_with_bad_record']['refused'] and ae['analysis_subset_without_keys']['refused'] and sp['passed'])
    return res


# Test stage resumption, atomic storage and rejection of changed outputs.
def drills_store() -> dict:
    ISO = E.ISO; pr = S.platform_refs(); out = {}
    base_rows = E.manifest_rows(next8_keys()[:3] + [next8_keys()[4]])
    rows = []
    for r in base_rows:
        r = copy.deepcopy(r); r['key'] = 'ISO-' + r['key']; r['manifest_key'] = r['key']; rows.append(r)

    def run(store, rws, ex, prx=pr):
        return S.run_rows(S.Store(store), copy.deepcopy(rws), ex, prx, E.methods_of)

    def code(fn):
        try:
            fn(); return 'ok'
        except K.Stop as e:
            return e.code
    st1 = ISO / 'store_a'
    stub = S.Stub(); log = run(st1, rows, stub)
    out['first_run_stub'] = dict(computed=log['computed'], reuse=log['reuse'], stub_calls=len(stub.calls), status=K.load_json(st1 / 'status.json')['state'])
    fb = S.Forbidden(); log2 = run(st1, rows, fb)
    out['identical_identity_reuse'] = dict(reuse=log2['reuse'], computed=log2['computed'], forbidden_calls=len(fb.calls), ok=log2['computed'] == 0 and not fb.calls)
    r_img = copy.deepcopy(rows); r_img[0]['f']['seed'] = (r_img[0]['f']['seed'] or 0) + 1
    out['image_identity_change'] = code(lambda: run(st1, r_img, S.Forbidden()))
    r_dec = copy.deepcopy(rows); i_dot = next(i for i, r in enumerate(r_dec) if r['family'] == 'point'); r_dec[i_dot]['radius_mm'] = r_dec[i_dot]['radius_mm'] + 0.25
    out['decode_parameter_change'] = code(lambda: run(st1, r_dec, S.Forbidden()))
    pr2 = dict(pr, global_domain_hash='0' * 64)
    out['evaluation_dependency_change'] = code(lambda: run(st1, rows, S.Forbidden(), pr2))

    store = S.Store(st1); ev_keys = [f"{r['key']}__{m}" for r in rows for m in E.methods_of(r)]
    S.batch_declare(store, 'b1', ev_keys)
    out['batch_ok'] = code(lambda: S.batch_verify(store, 'b1', ev_keys))
    out['batch_expected_set_changed'] = code(lambda: S.batch_verify(store, 'b1', ev_keys[:-1]))
    victim = st1 / 'eval' / f'{ev_keys[0]}.npz'; raw = victim.read_bytes(); victim.write_bytes(raw[:-7] + b'corrupt')
    out['batch_member_changed'] = code(lambda: S.batch_verify(store, 'b1', ev_keys))
    out['corrupt_file_runner'] = code(lambda: run(st1, rows, S.Forbidden()))
    victim.write_bytes(raw)

    j = st1 / 'eval' / f'{ev_keys[1]}.json'; jb = j.read_bytes(); j.unlink(); (st1 / 'eval' / f'{ev_keys[1]}.json.tmp999').write_bytes(b'{"partial":')
    out['half_written_forbidden'] = dict(code=code(lambda: run(st1, rows, S.Forbidden())), status=K.load_json(st1 / 'status.json')['state'])
    j.write_bytes(jb); (st1 / 'eval' / f'{ev_keys[1]}.json.tmp999').unlink()

    st2 = ISO / 'store_b'; s_fail = S.Stub(fail_after=4)
    c = code(lambda: run(st2, rows, s_fail)); stat = K.load_json(st2 / 'status.json')['state']
    done = sum(1 for p in (st2).rglob('*.json') if p.parent.name in S.Store.STAGES)
    s_res = S.Stub(); log3 = run(st2, rows, s_res)
    total = sum(1 + 2 * len(E.methods_of(r)) for r in rows)
    out['interrupt_and_resume'] = dict(first_code=c, first_status=stat, records_after_failure=done, resumed_calls=len(s_res.calls), total_stage_records=total,
                                       ok=c == 'technical' and stat == 'failed' and len(s_res.calls) == total - done and log3['reuse'] == done)

    (st2 / 'run.lock').write_text('held by drill', encoding='utf-8')
    out['duplicate_start'] = code(lambda: run(st2, rows, S.Forbidden()))
    (st2 / 'run.lock').unlink()

    out['real_executor_without_permit'] = code(lambda: S.Real(None))
    kf = ISO / 'keys_next8.json'; kf.write_text(json.dumps([list(k) for k in next8_keys()]), encoding='utf-8')
    kb = ISO / 'keys_bad.json'; kb.write_text(json.dumps([['D_r1_p8x4', 'B_curv', 'ph0', 20.0, 3]]), encoding='utf-8')
    p1 = subprocess.run(PY + ['run', '--store', str(ISO / 'store_c'), '--keys', str(kf)], capture_output=True, text=True, encoding='utf-8')
    p2 = subprocess.run(PY + ['run', '--store', str(ISO / 'store_d'), '--keys', str(kb)], capture_output=True, text=True, encoding='utf-8')
    (ISO / 'store_e').mkdir(parents=True, exist_ok=True); (ISO / 'store_e' / 'run.lock').write_text('held', encoding='utf-8')
    p3 = subprocess.run(PY + ['run', '--store', str(ISO / 'store_e'), '--keys', str(kf), '--executor', 'stub'], capture_output=True, text=True, encoding='utf-8')
    p4 = subprocess.run(PY + ['run', '--store', str(K.OUT / 'not_isolated'), '--keys', str(kf), '--executor', 'stub'], capture_output=True, text=True, encoding='utf-8')
    st_c = K.load_json(ISO / 'store_c' / 'status.json') if (ISO / 'store_c' / 'status.json').exists() else {}
    out['cli_exit_codes'] = dict(run_without_permit=p1.returncode, run_without_permit_status=st_c.get('state'), run_without_permit_reason=st_c.get('reason'),
                                 bad_key=p2.returncode, locked=p3.returncode, stub_outside_isolation=p4.returncode,
                                 expected=dict(run_without_permit=K.EXIT['forbidden'], bad_key=K.EXIT['scope'], locked=K.EXIT['lock'], stub_outside_isolation=K.EXIT['guard']))
    out['not_isolated_dir_created'] = (K.OUT / 'not_isolated').exists()
    exp = dict(image_identity_change='mismatch', decode_parameter_change='mismatch', evaluation_dependency_change='mismatch', batch_ok='ok', batch_expected_set_changed='corrupt',
               batch_member_changed='corrupt', corrupt_file_runner='corrupt', duplicate_start='lock', real_executor_without_permit='permit')
    ok = all(out[k] == v for k, v in exp.items()) and out['identical_identity_reuse']['ok'] and out['interrupt_and_resume']['ok'] \
        and out['half_written_forbidden'] == dict(code='forbidden', status='failed') and not out['not_isolated_dir_created'] \
        and all(out['cli_exit_codes'][k] == v for k, v in out['cli_exit_codes']['expected'].items())
    out['expected'] = exp; out['ok'] = bool(ok)
    return out


def run_all() -> dict:
    ISO = E.ISO
    if ISO.exists():
        shutil.rmtree(ISO)
    try:
        a = drills_manifest(); b = drills_results(); c = drills_store()
    finally:
        if ISO.exists() and ISO.resolve() == (K.OUT / '_isolation').resolve():
            shutil.rmtree(ISO)
    res = dict(manifest=a, results=b, store=c, isolation_removed=not ISO.exists(), heavy_modules_loaded=K.heavy_loaded(),
               summary=dict(manifest=a['ok'], results=b['ok'], store=c['ok']), t=K.now())
    res['all_ok'] = bool(a['ok'] and b['ok'] and c['ok'] and res['isolation_removed'] and not res['heavy_modules_loaded'])
    K.run_put('drills', res); K.flush('drills')
    return res


def next8_keys() -> list:
    return [(c, s, p, sg, 1) for (c, s, p) in K.NEXT8 for sg in (20.0, 100.0)]


def next8() -> dict:
    M = K.read_csv(K.reg(K.MANIFEST))
    gm = E.cmd_gate_manifest(M)
    rows = E.manifest_rows(next8_keys(), M); sel = []
    for k, r in zip(next8_keys(), rows):
        if r['status'] != 'to_image':
            raise K.Stop('scope', f'{k} is not a new image of the 2880 list')
        sel.append(dict(key=list(k), manifest_key=r['manifest_key'], family=r['family'], seed=r['seed'], image_identity_fields=r['f'], cand_id_sigma0=r['cand_id_sigma0'],
                        methods=E.methods_of(r)))
    st = E.cmd_status(K.SR / 's5_six_formal' / 'store', next8_keys())
    out = dict(keys=sel, n_images=len(sel), main_decodes=len(sel), df_decodes=sum(1 for x in sel if x['family'] == 'point'),
               decodes_total=len(sel) + sum(1 for x in sel if x['family'] == 'point'), counted_in='2880 new images / 4608 new decodes of the formal list (no extra matrix)',
               purpose='entry and data-flow check only; not a validation sample', status_in_proposed_store=st['rows'], manifest_gate=dict(ok=gm['ok'], n_rows=gm['n_rows']),
               commands=dict(gate='python -B scripts/surface_revision/s5_align03/s5r_main.py gate-manifest',
                             analysis='python -B scripts/surface_revision/s5_align03/s5r_main.py analyze',
                             status='python -B scripts/surface_revision/s5_align03/s5r_main.py status --keys <next8 keys json> --store outputs/surface_revision/s5_six_formal/store',
                             imaging_pending='python -B scripts/surface_revision/s5_align03/s5r_main.py run --keys <next8 keys json> --store outputs/surface_revision/s5_six_formal/store '
                                             '--permit <permit json of the next task book>   [pending: next-round authorisation and check; not run]'),
               t=K.now())
    K.run_put('next8', out); K.flush('next8')
    return dict(n_images=out['n_images'], decodes_total=out['decodes_total'], keys=[x['key'] for x in sel], status=[x['image'] for x in st['rows']])


def fig2() -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    COMMA = FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))
    X = K.read_csv(K.reg(K.A2 / '_parts' / 'B_mechanism.csv')); X = X[(X.record_type == 'ref_cell') & (X.set_name == 'S5-main')]
    H = K.read_csv(K.reg(K.A2 / '_parts' / 'A_interactions_posthoc_63.csv')); R = K.read_csv(K.reg(K.A2 / '_parts' / 'A_interactions_registered.csv'))
    plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'legend.fontsize': 6.3, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.8})
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.9)); pitches = (4.45, 6.3, 8.0); SH = {'B_curv': 'B', 'M10_curv': 'M10', 'M24_curv': 'M24'}
    for j, s in enumerate(K.MAIN):
        ax = axs[0, j]; g = X[X.surface == s].set_index(['combo', 'sigma'])
        for w, ls in (('0.5', '-'), ('1.5', '--')):
            for sg, c in ((0.0, '#1f77b4'), (40.0, '#d95f02')):
                ax.plot(pitches, [100 * g.loc[(f'L_w{w}_s{p:g}', sg), 'glob_Ctau_mean8'] for p in pitches], ls, marker='o', color=c, ms=3.5, lw=1, label=f'b = {w.replace(".", ",")} mm, σ = {sg:g}')
            ax.plot(pitches, [100 * g.loc[(f'L_w{w}_s{p:g}', 0.0), 'ref_glob_C_tau_d6_mean8'] for p in pitches], ls, color='0.45', lw=0.7, marker='D', mfc='none', ms=3, label=f'Referenz b = {w.replace(".", ",")} mm')
        ax.set_xticks(pitches); ax.xaxis.set_major_formatter(COMMA); ax.yaxis.set_major_formatter(COMMA); ax.set_ylim(-3, 103); ax.set_title(SH[s]); ax.set_xlabel('Linienabstand / mm'); ax.grid(lw=0.3, alpha=0.5)
        if j == 0:
            ax.set_ylabel('globale C$_τ$ / %')
        ax = axs[1, j]
        for k, (T, setn) in enumerate(((R, 'lines'), (H, 'lines_6.3_8'))):
            for sg, c, dx in (('0.0', '#1f77b4', -0.17), ('40.0', '#d95f02', 0.17)):
                t = T[(T.set_name == 'S5-main') & (T.surface == s) & (T.set == setn) & (T.effect == 'I') & (T.metric == 'glob_Ctau') & (T.sigma.astype(str) == sg)].iloc[0]
                v = 100 * t[list(K.PHASES)].to_numpy(float)
                ax.plot(np.full(8, k + dx) + np.linspace(-0.12, 0.12, 8), v, '.', color=c, alpha=0.45, ms=4)
                ax.errorbar(k + dx, 100 * t['mean'], yerr=[[100 * (t['mean'] - t['res_lo_5'])], [100 * (t['res_hi_95'] - t['mean'])]], fmt='o', color=c, ms=4.5, capsize=2, lw=0.9,
                            label=(f'σ = {float(sg):g}' if k == 0 else None))
        ax.axhline(0, color='k', lw=0.5); ax.set_xticks([0, 1]); ax.set_xticklabels(['4,45→8 mm\n(registriert)', '6,3→8 mm\n(nachträglich)']); ax.set_ylim(-34, 34)
        ax.yaxis.set_major_formatter(COMMA); ax.grid(axis='y', lw=0.3, alpha=0.5)
        if j == 0:
            ax.set_ylabel('I = E$_{groß}$ − E$_{klein}$ / pp')
        else:
            ax.set_yticklabels([])
    h, lab = axs[0, 0].get_legend_handles_labels(); fig.legend(h, lab, ncol=3, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.0))
    axs[1, 0].legend(fontsize=6, frameon=False, loc='upper left')
    fig.text(0.01, 0.004, 'Oben: 8-Phasen-Mittel. Unten (gemeinsame Achse −34…34 pp): Punkte = 8 Phasen (Wiederholungen je Phase gemittelt),\nBalken = deskriptives 5–95-%-Intervall (Phasen-Resampling, 10 000×); kein Signifikanztest. Daten unverändert (S5-ALIGN-02).', fontsize=6.1)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    p = K.guard(K.FIGS / 'F2_Linien_6p3_8_Abwaegung_v2.png'); fig.savefig(p, dpi=300); plt.close(fig)
    out = dict(file=K.rel(p), sha256=K.sha(p), note='same data as S5-ALIGN-02 F2; bottom row on one common y axis', t=K.now())
    K.run_put('fig2', out); K.flush('fig2')
    return out


def final() -> dict:
    r = K.run_load(); src = r.get('sources_read', {})
    bad = [k for k, h in src.items() if K.sha(K.ROOT / k if not k.startswith('C:') else k) != h]
    outs = {K.rel(p): K.sha(p) for p in sorted(K.OUT.rglob('*')) if p.is_file() and p.name != 'run.json' and '.tmp' not in p.name}
    code = {K.rel(p): K.sha(p) for p in sorted(K.HERE.glob('*.py'))}
    sent = r.get('sentinel', [])
    res = dict(sources_unchanged=not bad, changed=bad[:10], n_sources=len(src), new_imaging=0, decode_calls=0, new_interpolation_evaluations=0,
               sentinel_commands=len(sent), sentinel_heavy_modules=sorted({m for s in sent for m in s['heavy_modules_loaded']}), isolation_dir_present=E.ISO.exists(),
               outputs=outs, code=code, t=K.now())
    K.run_put('final', res)
    return {k: v for k, v in res.items() if k not in ('outputs', 'code')}
