# Coordinate layout preparation, execution, analysis and figure generation.

"""S4-REAL-02 entry: python r2_main.py <command> [--workers N]

bind      anchors, layout registry (layouts.json), 264-view reproduction, evaluator snapshot + regression, identity tracing, manifest.csv,
          protocol / semantic map / errata into run.json (re-runnable until the freeze)
drills    non-imaging drills, each in its own process with its expected exit code
freeze    freezes protocol + bind and the code / parameter-file fingerprints of the real entry and its direct dependencies
entry     preflight -> legacy references (<= 12) -> 36 entry images (sigma 0 first) -> analysis + composite figures on the entry views -> gates
          G1-G6 written to run.json before any further render; non-zero exit when a gate fails
run       preflight + passed gate -> remaining formal images (cap 2560 render attempts, single scheduler, resume fills missing items only)
analyse   case_metrics.csv + paired_summary.csv (refuses an incomplete batch)
figs      German figures F01-F12
final     counts, denominators, source hashes start / end, figures vs tables
Exit codes: 0 ok, 2 Stop (refusal / gate), 1 uncaught exception.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import r2_common as K

HERE = K.HERE
DATA_INPUTS = ['outputs/surface_revision/s4_align02/s4_manifest_candidates.csv', 'outputs/surface_revision/pix_bound01/domains/omega1.npz', 'outputs/surface_revision/pix_bound01/domains/omega1.json',
               'outputs/surface_revision/pix_bound01/domains/omega_star.npz', 'outputs/surface_revision/pix_bound01/domains/omega_star.json', 'outputs/surface_revision/pix_bound01/run.json',
               'outputs/surface_revision/s345_ready/plan_s345.csv', 'outputs/surface_revision/s345_ready/results/S4-reduced_cases.csv',
               'outputs/surface_revision/s345_ready/results/S4-reduced_block_differences.csv', 'outputs/surface_revision/sections_adapter/binding.json',
               'outputs/surface_revision/surf02/config/surf02_config.json']


def protocol() -> dict:
    return dict(
        task=K.rel(K.TASK), review=K.rel(K.REVIEW),  acceptance='reference (independent)', start=K.START,
        settings=dict(dots='R8x4, r = 1 mm, D0, registered 1-D order-preserving displacement', lines='L6.3, w = 1 mm, 0 deg, registered P4 position-table path',
                      surfaces=list(K.SURF), psf_px=K.RK.PSF_PX, window_mm=list(K.RK.WINDOW_MM), Q=K.Q, peak_max=1.0, integral_rel_err_max=1e-6, j=0.20, alpha=0.70, beta=1.15,
                      candidates_per_block=1000, strategies=list(K.STRATS), blocks=[f'B{b:03d}' for b in K.BLOCKS], phases=list(K.PHASES) + ['seen phases: post-hoc design extension'],
                      sigmas=list(K.SIGMAS), noise='registered S4-reduced sigma-20 seed of family x surface x phase; one underlying field shared across strategies and scaled by sigma; '
                      'sigma 0 = no noise (render seed 0, no seed in the key); one realisation per condition, not independent repetitions',
                      boundary=K.BOUNDARY, domain_main='Omega1 (PIX omega1.npz, four-surface common construction unchanged)', domain_bridge='Omega* = Omega0 n Omega1 (labelled separately)',
                      local_domain='r <= 15 mm disc, unchanged', d_mm=K.D_SUPPORT, tau_mm=K.TAU, render_cap=K.CAP_RENDER, formal_images=K.N_FORMAL, formal_views=K.N_VIEWS),
        errata=[
            'Local C_tau unchanged is not local surface RMSE unchanged (PIX M10 5.6 mm strong code sigma 20 ignore: local surface RMSE 0.938 -> 0.992 mm, interpolable local nodes '
            '1099 -> 1117; both versions without supported local coverage, C_valid = C_tau = 0). Wording: "local task-coverage metrics unchanged" / "local interpolation statistics '
            'changed, both versions without supported local coverage".',
            'The number of correct points is not unchanged (92 PIX paths: 194486 -> 194531; wrong 1 -> 1, unconfirmable 4358 -> 4358, sums include shared images); allowed: '
            '"no additional wrong identities, inspected states unchanged".',
            'Keeping the old results is not a no-impact certification of sections 1-5; the main directions held within the inspected sample.',
            'The central-axis section criterion and the global coordinate extremes are different objects (Omega1 x -45...45 mm, y -43.5...45 mm); the old physical coverage had a '
            'half-pixel bias, the remaining asymmetry comes from the camera / projector intersection, the surfaces and the grid; "both directions are no program error" withdrawn.',
            '264 strategy-phase views, 140 unique raster / peak combinations and 148 computation calls are separate quantities (C_layout_phase has 264 rows).',
            '"All numbers of sections 1-4 kept directly" narrowed to "old data and original-version numbers kept; the inspection found no evidence requiring a full rerun".'],
        semantic_map=[
            dict(site='camera image formation (physical pixel coverage)', function='pb_m3.render -> project_to_camera / in_footprint', rule=f'{K.BOUNDARY}: -0.5 <= u < W - 0.5, same for v', status='new'),
            dict(site='dot truth: true surface point inside the PROJECTOR image (confirmability)', function='r2_eval_snap.evaluate_case -> _project(P, proj, boundary)', rule='same new coverage', status='new'),
            dict(site='line reference rows: projected points inside the CAMERA image', function='r2_eval_snap.line_reference_table -> _project(pts, cam, boundary)', rule='same new coverage', status='new'),
            dict(site='line truth identity table', function='sp2_lines_table.truth_identity_table (l01_truth)', rule='-0.5 <= u <= W - 0.5 (closed; already symmetric)', status='unchanged'),
            dict(site='expected dot centres (recovery denominator)', function='s1r_eval.evaluate_case line 90 after sn_lib.expected_dots', rule='centre support 0 <= u, v <= IMG - 1', status='kept (centre support)'),
            dict(site='dot source labels (sigma-0 reference map)', function='rdy_eval.source_identity -> sn_lib.expected_dots (shared m3 projection 0 <= u < W) -> sl_lib.reference_map',
                 rule='expected centres outside the legacy coverage own no component (only centres in u or v in [-0.5, 0)); their components count as unlabelled -> undeterminable, never wrong',
                 status='kept (centre visibility), conservative'),
            dict(site='bilinear sampling / interpolation support and weight normalisation', function='m3_virtual_camera (inside pb_m3)', rule='neighbour validity [0, W - 1], unchanged', status='unchanged'),
            dict(site='integer array indexing', function='numpy indexing in M3 / detectors', rule='int / round indices, unchanged', status='unchanged'),
            dict(site='detector protection distance / border guards', function='r5_detect, line front end', rule='unchanged', status='unchanged'),
            dict(site='PSF, intrinsics / extrinsics, window', function='m3 / binding', rule='unchanged', status='unchanged'),
            dict(site='identity distances, class rules, matching tolerances', function='evaluate_case / truth tables / decoders', rule='unchanged (only the boundary attribution follows the version)', status='unchanged'),
            dict(site='evaluation domains', function='r2_eval.domain', rule='Omega1 main (anchor cc6dd47d), Omega* bridge (anchor 0aee8912), local disc unchanged', status='explicit')],
        statistics=dict(pairing='same baseline x surface x block x phase x sigma x underlying noise field; never a Cartesian product over sizes or noise',
                        main='per block the median of the 4 phase differences, then the median over the 8 blocks; mean of all pairs, block range, P10 / P90 alongside; '
                             'zero differences of identical layouts stay in the denominator; changed-layout blocks only as an extra diagnostic',
                        s0='one shared control per phase; never 8 independent S0 samples', resampling='none (descriptive statistics only)',
                        old_data='conditional comparison only: the 48 old S0 clouds re-evaluated on Omega* (levels); old within-batch effects read from the stored S4-reduced table '
                                 '(Omega0 / unchanged local disc), labelled; boundary versions differ, no causal pairing; no full re-evaluation of the old 888',
                        surface_rmse='reported surface RMSE = over interpolable nodes (not bound to the support condition of C_tau); supported-node RMSE alongside'),
        display=dict(case_boards='3 surfaces x 2 families, H0, dots B021, lines B000, S0 / S1 / S2 / S3 at sigma 0 and 60; S4 noted when equal to S3 (fixed before imaging)',
                     figures='<= 12 German composite figures F01-F12 (r2_figs.py)'),
        answers_required=['S1 vs S0', 'S2 / S3 vs S1', 'S4 vs S2 / S3', 'do the limits of the old dense baselines hold under R8x4 / L6.3 (conditional)'])


def _anchors() -> dict:
    out = {p: dict(expected=h, actual=K.sha_file(K.ROOT / p)) for p, h in K.ANCHORS.items()}
    bad = [p for p, v in out.items() if v['expected'] != v['actual']]
    if bad:
        raise K.Stop('input', f'input anchors differ: {bad}')
    return out


def bind() -> dict:
    import r2_eval as E
    import r2_eval_snap as SN
    import r2_layouts as L
    import r2_manifest as RM
    import s1r_common as S1K
    if K.run_load().get('frozen', {}).get('bind'):
        raise K.Stop('freeze', 'bind is frozen; never rebuilt')
    t0 = time.time(); anc = _anchors()
    if (float(S1K.D_SUPPORT), float(S1K.TAU)) != (K.D_SUPPORT, K.TAU):
        raise K.Stop('input', 'd / tau differ from s1r_common')
    reg, checks = L.build_registry(); K.save_json(K.OUT / 'layouts.json', reg)
    R, summ = L.reproduce_views(reg)
    if not summ['ok']:
        raise K.Stop('check', f'view reproduction failed: {summ}')
    snap = SN.snapshot_diff(); regr = E.regression(); trace = E.identity_tracing()
    if not regr['ok']:
        raise K.Stop('check', 'evaluator regression differs from the stored S4-reduced metrics')
    M, info = RM.build(reg, R); v = RM.validate(M, info['fingerprint_digests'])
    rk = RM.rekey(M, info['m3_digest'], E.domain('omega1')['hash'], E.domain('omega_star')['hash'])
    K.save_csv(M, K.OUT / 'manifest.csv')
    old = {p: K.sha_file(K.ROOT / p) for p in DATA_INPUTS}
    Vold = pd.read_csv(K.S345 / 'results' / 'S4-reduced_cases.csv', low_memory=False); Vold = Vold[Vold.strategy.astype(str).str.startswith('S0')]
    old.update({K.rel(K.S345 / 'run' / 'eval' / f'{k}.npz'): K.sha_file(K.S345 / 'run' / 'eval' / f'{k}.npz') for k in Vold.decode_key})
    b = dict(anchors=anc, registry=dict(n_layouts={n: len(reg['baselines'][n]['layouts']) for n, _f, _g in K.BASES}, n_rederived=len(checks), all_equal=bool(all(c['descriptors_equal'] for c in checks)),
                                        layouts_json_sha256=K.sha_file(K.OUT / 'layouts.json')),
             view_reproduction=summ, evaluator_snapshot_diff=snap, evaluator_regression=regr, identity_tracing=trace, manifest=dict(v, rekey=rk, manifest_sha256=K.sha_file(K.OUT / 'manifest.csv')),
             fingerprints=info['fingerprints'], fingerprint_digests=info['fingerprint_digests'], m3_digest=info['m3_digest'], seeds=info['seeds'], data_inputs_start=old,
             seconds=round(time.time() - t0, 1), t_local=K.now_local())
    K.run_update('protocol', protocol()); K.run_update('bind', b)
    return dict(ok=True, views=summ['n_views'], unique=summ['n_unique_raster_peak'], manifest=v, regression_n=regr['n'], tracing=dict(n=trace['n_cases'], strip=trace['n_strip'], label_change=trace['n_label_change']))


def drills() -> dict:
    out = _run_drills(); ok = all(v['passed'] for v in out.values())
    K.run_update('drills', dict(ok=ok, n=len(out), n_passed=sum(v['passed'] for v in out.values()), results=out, t_local=K.now_local()))
    if not ok:
        raise K.Stop('drills', f"failed: {[k for k, v in out.items() if not v['passed']]}")
    return dict(ok=ok, n=len(out))


def _run_drills() -> dict:
    import r2_drills as D
    out = {}
    for name, (_f, code) in D.DRILLS.items():
        t0 = time.time(); p = subprocess.run([sys.executable, str(HERE / 'r2_main.py'), 'drill', name], capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=str(HERE))
        tail = (p.stdout + p.stderr).strip().splitlines()[-1:] if (p.stdout or p.stderr) else []
        out[name] = dict(expected_exit=code, exit=p.returncode, passed=bool(p.returncode == code), last_line=(tail[0][:300] if tail else ''), seconds=round(time.time() - t0, 1))
    return out


def _code_files() -> list:
    import r2_manifest as RM
    for m in ('r2_analysis', 'r2_figs', 'r2_gate', 'r2_run', 'r2_drills', 'r2_eval', 'r2_eval_snap', 'r2_layouts', 'r2_manifest', 'rdy_decode', 'rdy_eval', 'rdy_inputs', 'rdy_platform', 'pb_m3',
              's3_generate', 'sp2_generator', 'sp_dots_columns', 'sp2_lines_table', 'adp_patterns', 's1r_eval', 's12_common', 'l01_truth', 'sl_lib', 'sn_lib', 'r5_detect', 'm4_lines',
              'm4_reconstruction', 'surface_eval_r3', 'm3_virtual_camera'):
        __import__(m)
    files = set(RM.IMAGE_CODE + RM.DECODE_CODE + RM.EVAL_CODE)
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f:
            try:
                r = K.rel(f)
            except Exception:
                continue
            if not r.startswith(('C:', '/', '\\')) and r.endswith('.py') and not r.startswith('outputs'):
                files.add(r)
    files.add(K.rel(HERE / 'r2_main.py'))
    return sorted(files)


def freeze() -> dict:
    r = K.run_load()
    if 'freeze' in r.get('frozen', {}):
        raise K.Stop('freeze', 'already frozen')
    if not (r.get('drills') or {}).get('ok'):
        raise K.Stop('freeze', 'drills not passed')
    files = _code_files(); code = {p: K.sha_file(K.ROOT / p) for p in files}; data = {p: K.sha_file(K.ROOT / p) for p in DATA_INPUTS}
    K.run_update('protocol', r['protocol'], freeze=True); K.run_update('bind', r['bind'], freeze=True)
    K.run_update('freeze', dict(code=code, data=data, layouts_json=K.sha_file(K.OUT / 'layouts.json'), manifest_csv=K.sha_file(K.OUT / 'manifest.csv'), n_code_files=len(code), t_local=K.now_local()), freeze=True)
    return dict(ok=True, n_code_files=len(code))


def preflight(current: dict | None = None) -> dict:
    import r2_manifest as RM
    r = K.run_load()
    cur = current or {k: K.digest(v, 32) for k, v in RM.fingerprints().items()}
    if cur != r['bind']['fingerprint_digests']:
        raise K.Stop('version', f"code fingerprints differ from the frozen manifest: {[k for k in cur if cur[k] != r['bind']['fingerprint_digests'].get(k)]} -> resume refused")
    K.run_verify_frozen()
    if 'freeze' not in r.get('frozen', {}):
        raise K.Stop('freeze', 'not frozen')
    _anchors()
    fz = r['freeze']; code = effective_code(r)
    bad = [p for p, h in code.items() if K.sha_file(K.ROOT / p) != h] + [p for p, h in fz['data'].items() if K.sha_file(K.ROOT / p) != h]
    if K.sha_file(K.OUT / 'layouts.json') != fz['layouts_json'] or K.sha_file(K.OUT / 'manifest.csv') != fz['manifest_csv']:
        bad.append('layouts.json / manifest.csv')
    if bad:
        raise K.Stop('version', f'frozen files changed: {bad[:6]} -> resume refused')
    return dict(ok=True, refreezes=sorted(k for k in r if k.startswith('refreeze_')))


def effective_code(r: dict) -> dict:

    code = dict(r['freeze']['code'])
    for k in sorted((k for k in r if k.startswith('refreeze_')), key=lambda s: int(s.split('_')[1])):
        if k not in r.get('frozen', {}):
            raise K.Stop('freeze', f'{k} is not frozen')
        code.update({p: v['new'] for p, v in r[k]['changed'].items()})
    return code


REFREEZE = {1: dict(reason='entry gates G3 / G6 (attempt 1, 2026-10-16 20:5x): G3 compared the 12 S0 equivalence records with a wrong constant 6 (all 12 equal); '
                           'G6 could only explain border-band feature changes, while the one exceedance (lines M24 sigma 100 S0: valid points 39 -> 64) comes from the '
                           'image-global noise threshold of the registered line detector (sigma_hat 86 -> 85 after the border-band pixel change, Steger threshold 14.314 -> 14.148); '
                           'located with the stored images / features: thresholds swapped -> interior features identical, stored feature sets reproduced exactly',
                    change='r2_gate.py: S0 count from the manifest (12); G6 threshold-swap counterfactual for line pairs with interior feature changes. r2_main.py: '
                           'refreeze records, preflight / final on the effective hashes, regate on the existing entry data (no render, failed attempt kept as gate_attempt_1)',
                    affected='gate evaluation only; no image / decode / evaluation key, generation parameter, noise, decoder, threshold, domain or matrix changed '
                             '(r2_gate.py / r2_main.py are not part of the image / decode / evaluation fingerprints)'),
            2: dict(reason='visual check of the entry figures: board title clipped at both sides, board legend touching the reference colour bar, curve-figure legends '
                           'below the source note',
                    change='r2_figs.py: legend / note positions, shorter board title (axis unit moved to the note); r2_main.py: this record; r2_drills.py: the '
                           'partial-batch drill builds its own partial evaluation folder (the original version relied on the formal run not yet being complete)',
                    affected='figure layout and one drill only; no data, statistic, selection or colour scale changed'),
            3: dict(reason='visual check of the formal figures: F02 title clipped; F03 / F04 linear error axes hide sigma 0-60 behind the ~10 mm errors of the dot '
                           'noise outputs at sigma >= 80; board source note clipped at the right',
                    change='r2_figs.py: shorter curve title, logarithmic y axis for the three error rows of F03 / F04, board note on three lines; r2_main.py: this record',
                    affected='figure presentation only; no data, statistic, selection or colour scale changed')}


def refreeze(n: int) -> dict:
    import r2_manifest as RM
    r = K.run_load(); sec = f'refreeze_{n}'
    if sec in r.get('frozen', {}):
        raise K.Stop('freeze', f'{sec} already frozen')
    code = effective_code(r); changed = {p: dict(old=h, new=K.sha_file(K.ROOT / p)) for p, h in code.items() if K.sha_file(K.ROOT / p) != h}
    keyed = set(RM.IMAGE_CODE + RM.DECODE_CODE + RM.EVAL_CODE)
    if not changed:
        raise K.Stop('freeze', 'nothing changed')
    if keyed & set(changed):
        raise K.Stop('freeze', f'keyed code changed {sorted(keyed & set(changed))}: the manifest keys would change -> not a refreeze case')
    dr = _run_drills()
    if not all(v['passed'] for v in dr.values()):
        raise K.Stop('drills', f"drills after the fix failed: {[k for k, v in dr.items() if not v['passed']]}")
    K.run_update(sec, dict(changed=changed, **REFREEZE[n], drills_rerun=dict(n=len(dr), n_passed=sum(v['passed'] for v in dr.values()), results=dr), t_local=K.now_local()), freeze=True)
    return dict(ok=True, section=sec, changed=sorted(changed))


def regate() -> dict:

    import r2_analysis as A
    import r2_figs as F
    import r2_gate as G
    import r2_run as R
    preflight(); r = K.run_load(); g = r.get('gate')
    if not g or g.get('ok'):
        raise K.Stop('gate', 'no failed gate attempt to re-evaluate')
    M = R.manifest(); E = M[M.entry_subset.astype(str).isin(['True', 'true', '1'])]
    if not all(R.done(x) for x in E.drop_duplicates('image_key').to_dict('records')):
        raise K.Stop('gate', 'entry data incomplete: regate never renders')
    n_att = 1 + sum(1 for k in r if k.startswith('gate_attempt_'))
    K.run_update(f'gate_attempt_{n_att}', g, freeze=True)
    B0 = K.Budget().state()['used']; leg = R.legacy_refs()
    if K.Budget().state()['used'] != B0:
        raise K.Stop('budget', 'regate reserved a render')
    reg = K.load_json(K.OUT / 'layouts.json'); st = r['entry']['status']
    C = A.case_rows(E, reg, require_complete=True); Pa = A.paired(C, partial=True)
    K.save_csv(C, K.RUNDIR / 'entry_case_metrics.csv'); K.save_csv(Pa, K.RUNDIR / 'entry_paired_summary.csv')
    figs = F.make_all(C, Pa, reg, K.RUNDIR / 'entry_figures', entry=True)
    gates = dict(G1_closure=G.g1_closure(M), G2_end_to_end=G.g2_end_to_end(M, st, dict(C=C), figs), G3_equivalence=G.g3_equivalence(M, leg), G4_no_output=G.g4_no_output(M),
                 G5_independent=G.g5_independent(M), G6_screening=G.g6_screening(M, leg))
    ok = all(x['ok'] for x in gates.values())
    K.run_update('gate', dict(ok=ok, **{k: dict(ok=v['ok']) for k, v in gates.items()}, details=gates, attempt=n_att + 1, previous=f'gate_attempt_{n_att}', t_local=K.now_local(),
                              note='re-evaluated on the stored entry data after refreeze; written before any further formal render'))
    if not ok:
        raise K.Stop('gate', f"technical gate failed: {[k for k, v in gates.items() if not v['ok']]}")
    return dict(ok=ok, gates={k: v['ok'] for k, v in gates.items()})


def entry(workers: int) -> dict:
    import r2_analysis as A
    import r2_figs as F
    import r2_gate as G
    import r2_run as R
    preflight(); r = K.run_load()
    if r.get('gate'):
        raise K.Stop('gate', 'gate already written; the entry check is not repeated')
    M = R.manifest(); E = M[M.entry_subset.astype(str).isin(['True', 'true', '1'])]
    t0 = time.time(); leg = R.legacy_refs(); t_leg = round(time.time() - t0, 1)
    st = R.schedule(list(E.image_key.unique()), workers, 'entry')
    reg = K.load_json(K.OUT / 'layouts.json'); t1 = time.time()
    C = A.case_rows(E, reg, require_complete=True); Pa = A.paired(C, partial=True)
    K.save_csv(C, K.RUNDIR / 'entry_case_metrics.csv'); K.save_csv(Pa, K.RUNDIR / 'entry_paired_summary.csv')
    figs = F.make_all(C, Pa, reg, K.RUNDIR / 'entry_figures', entry=True)
    t_ana = round(time.time() - t1, 1)
    gates = dict(G1_closure=G.g1_closure(M), G2_end_to_end=G.g2_end_to_end(M, st, dict(C=C), figs), G3_equivalence=G.g3_equivalence(M, leg), G4_no_output=G.g4_no_output(M),
                 G5_independent=G.g5_independent(M), G6_screening=G.g6_screening(M, leg))
    ok = all(g['ok'] for g in gates.values())
    K.run_update('entry', dict(legacy_refs=leg, legacy_seconds=t_leg, status=st, analysis_seconds=t_ana, entry_figures=figs['files'], t_local=K.now_local()))
    K.run_update('gate', dict(ok=ok, **{k: dict(ok=v['ok']) for k, v in gates.items()}, details=gates, t_local=K.now_local(), note='written before any further formal render'))
    if not ok:
        raise K.Stop('gate', f"technical gate failed: {[k for k, v in gates.items() if not v['ok']]}")
    return dict(ok=ok, gates={k: v['ok'] for k, v in gates.items()}, status=st)


def run(workers: int) -> dict:
    import r2_run as R
    preflight(); g = K.run_load().get('gate')
    if not g or not g.get('ok'):
        raise K.Stop('gate', 'technical gates not passed: formal run refused')
    st = R.schedule(None, workers, 'formal')
    pids = set()
    for p in (K.RUNDIR / 'ledger_entry.jsonl', K.RUNDIR / 'ledger_formal.jsonl'):
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                x = json.loads(line)
                if x.get('pid'):
                    pids.add(x['pid'])
    prev = K.run_load().get('formal_run') or {}
    runs = prev.get('runs', []) + [st]
    K.run_update('formal_run', dict(runs=runs, n_worker_processes_seen=len(pids), t_local=K.now_local()))
    return st


def analyse() -> dict:
    import r2_analysis as A
    t0 = time.time(); res = A.run(require_complete=True, with_bridge=True)
    C, P = res['C'], res['P']
    K.save_csv(C, K.OUT / 'case_metrics.csv'); K.save_csv(P, K.OUT / 'paired_summary.csv')
    s = dict(n_views=len(C), n_images=int(C.image_key.nunique()), n_no_output_views=int((C.state == 'no_output').sum()), n_no_output_images=int(C[C.state == 'no_output'].image_key.nunique()),
             paired_rows=int((P.table_type == 'paired').sum()), absolute_rows=int((P.table_type == 'absolute').sum()), bridge_rows=int(P.table_type.str.startswith(('dense', 'old')).sum()),
             case_metrics_sha256=K.sha_file(K.OUT / 'case_metrics.csv'), paired_summary_sha256=K.sha_file(K.OUT / 'paired_summary.csv'), seconds=round(time.time() - t0, 1), t_local=K.now_local())
    K.run_update('analysis', s)
    return s


def figs() -> dict:
    import r2_figs as F
    C = pd.read_csv(K.OUT / 'case_metrics.csv', low_memory=False, keep_default_na=True); C['block'] = C['block'].fillna('')
    P = pd.read_csv(K.OUT / 'paired_summary.csv', low_memory=False)
    reg = K.load_json(K.OUT / 'layouts.json'); t0 = time.time()
    out = F.make_all(C, P, reg, K.FIGS, entry=False)
    out['seconds'] = round(time.time() - t0, 1); out['hashes'] = {f: K.sha_file(K.ROOT / f) for f in out['files']}; out['t_local'] = K.now_local()
    K.run_update('figures', out)
    return dict(n=out['n_figures'], board_checks=len(out['board_checks']), all_equal=all(b['equal'] for b in out['board_checks']))


def final() -> dict:
    import r2_run as R
    r = K.run_load(); K.run_verify_frozen(); M = R.manifest()
    U = M.drop_duplicates('image_key'); n_done = int(sum(R.done(x) for x in U.to_dict('records')))
    C = pd.read_csv(K.OUT / 'case_metrics.csv', low_memory=False); P = pd.read_csv(K.OUT / 'paired_summary.csv', low_memory=False)
    per_sigma = U.groupby(['family', 'sigma']).size(); pv = C.groupby(['family', 'sigma']).size()
    fz = r['freeze']; code_eff = effective_code(r); code_end = {p: K.sha_file(K.ROOT / p) for p in code_eff}; data_end = {p: K.sha_file(K.ROOT / p) for p in fz['data']}
    old_end = {p: K.sha_file(K.ROOT / p) for p in r['bind']['data_inputs_start']}
    Pp = P[P.table_type == 'paired']
    B = K.Budget().state()
    chk = dict(images_complete=n_done, unique_images=int(U.image_key.nunique()), views=int(len(M)), case_rows=int(len(C)), case_complete=int((C.record == 'complete').sum()),
               per_sigma_images={f'{a}|{b:g}': int(n) for (a, b), n in per_sigma.items()}, per_sigma_views={f'{a}|{b:g}': int(n) for (a, b), n in pv.items()},
               pairs_all_32=bool((Pp.n_pairs == 32).all()), duplicate_views=int(C.view_id.duplicated().sum()),
               code_unchanged=bool(code_end == code_eff), refreezes=sorted(k for k in r if k.startswith('refreeze_')), data_unchanged=bool(data_end == fz['data']), old_inputs_unchanged=bool(old_end == r['bind']['data_inputs_start']),
               identity_closed=bool(((C.id_correct + C.id_wrong + C.id_unconf) == C.n_accepted).all()), budget=B,
               figures_present=bool(all((K.ROOT / f).exists() and K.sha_file(K.ROOT / f) == h for f, h in r['figures']['hashes'].items())),
               board_checks_equal=bool(all(b['equal'] for b in r['figures']['board_checks'])))
    chk['ok'] = bool(n_done == K.N_FORMAL and chk['views'] == K.N_VIEWS and chk['case_complete'] == K.N_VIEWS and all(v == 192 for k, v in chk['per_sigma_images'].items() if k.startswith('point'))
                     and all(v == 228 for k, v in chk['per_sigma_images'].items() if k.startswith('line')) and chk['pairs_all_32'] and chk['duplicate_views'] == 0 and chk['code_unchanged']
                     and chk['data_unchanged'] and chk['old_inputs_unchanged'] and chk['identity_closed'] and B['used'] <= K.CAP_RENDER and chk['figures_present'] and chk['board_checks_equal'])
    chk['t_local'] = K.now_local()
    K.run_update('final', chk)
    return chk


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ''
    w = int(argv[argv.index('--workers') + 1]) if '--workers' in argv else 10
    if cmd == 'drill':
        import r2_drills as D
        out = D.DRILLS[argv[2]][0]()
        print('drill', argv[2], 'returned', str(out)[:300]); return 0
    f = dict(bind=bind, drills=drills, freeze=freeze, entry=lambda: entry(w), run=lambda: run(w), analyse=analyse, figs=figs, final=final, preflight=preflight, regate=regate, refreeze=lambda: refreeze(int(argv[2]))).get(cmd)
    if f is None:
        print(__doc__); return 2
    out = f(); print(K.clean(out) if not isinstance(out, pd.DataFrame) else out)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except K.Stop as e:
        print(f'STOP {e}', file=sys.stderr); sys.exit(2)
