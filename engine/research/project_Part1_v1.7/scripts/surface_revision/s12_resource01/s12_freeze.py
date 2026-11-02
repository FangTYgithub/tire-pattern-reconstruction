# Bind size-spacing source files and inputs to a content manifest.

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_common as K

RUN_CODE = ('s12_common.py', 's12_manifest.py', 's12_reuse.py', 's12_case.py', 's12_freeze.py', 's12_run.py')
REPORT_CODE = ('s12_stats.py', 's12_sens.py', 's12_figs.py', 's12_check.py', 's12_report.py')


FIRST = [
    ('S1', 'M10_curv', 'point', 'resource_cap_11695', 1.0, 'S1_R8x4', 'sn_phase0', 0.0, 0),
    ('S1', 'M24_curv', 'point', 'resource_cap_11695', 1.5, 'S1_R8x4', 'sn_phase1', 100.0, 0),
    ('S1', 'B_curv', 'point', 'resource_cap_11695', 1.5, 'S1_R8x4', 'sn_phase2', 60.0, 2),
    ('S1', 'M10_curv', 'point', 'resource_cap_11695', 1.5, 'S1_R8x4', 'sn_phase0', 20.0, 0),
    ('S1', 'M24_curv', 'line', 'resource_cap_11695', 0.5, 's8_phi0', 'ph0', 100.0, 0),
    ('S1', 'M24_curv', 'line', 'resource_cap_11695', 1.0, 's8_phi0', 'ph0', 100.0, 0),
    ('S1', 'M24_curv', 'line', 'resource_cap_11695', 1.5, 's8_phi0', 'ph0', 100.0, 0),
    ('S1', 'B_curv', 'line', 'fixed_peak_1', 1.0, 's8_phi0', 'ph3', 80.0, 1),
    ('S1', 'M10_curv', 'line', 'fixed_peak_1', 1.5, 's8_phi0', 'ph5', 60.0, 0),
    ('S1', 'M24_curv', 'line', 'resource_cap_11695', 1.0, 's8_phi0', 'ph0', 80.0, 0),
    ('S2', 'M10_curv', 'point', 'resource_cap_11695', 1.0, 'R6x4', 'pos33', 20.0, 0),
    ('S2', 'M24_curv', 'point', 'resource_cap_11695', 1.0, 'T8x4', 'pos115', 100.0, 1),
    ('S2', 'B_curv', 'point', 'resource_cap_11695', 1.0, 'R8x2.5', 'pos197', 80.0, 0),
    ('S2', 'M10_curv', 'line', 'resource_cap_11695', 1.0, 's4.45_phi0', 'ph0', 20.0, 0),
    ('S2', 'M24_curv', 'line', 'resource_cap_11695', 1.0, 's12_phi45', 'ph2', 60.0, 1),
    ('S2', 'B_curv', 'line', 'resource_cap_11695', 1.0, 's6.3_phi0', 'ph4', 100.0, 0)]
FIRST_RULE = ('<= 16 new inputs fixed before any new image: all three main surfaces, dots and lines, S1 (common-cap dots r 1 / 1.5, high-noise lines both regimes) '
              'and S2 (new sigma 20 / 60 / 80 / 100, dense / wide pitches, phi 0 / 45, R / T), sigma 100 with dark common-cap rasters, one S1/S2 shared input '
              '(M24 w 1 s 8 phi 0 sigma 80 ph0 rep 0) and one complete figure group (M24 lines common cap sigma 100 ph0 rep 0, w 0.5 / 1.0 / 1.5)')
EXAMPLES = dict(S1_point='sn_phase0 (first registered S1 dot phase), rep 0', S1_line='ph0, rep 0', S2_point='pos33 (first registered position), rep 0', S2_line='ph0, rep 0',
                note='fixed instances, not population representatives or independent validation; failures shown in place')


def first_keys(V: pd.DataFrame) -> list[str]:
    keys = []
    for sec, s, f, reg, size, lay, ph, sig, rep in FIRST:
        g = V[(V.section == sec) & (V.surface == s) & (V.family == f) & (V.regime == reg) & (V.size_mm == size) & (V.layout == lay) & (V.phase_label == ph) & (V.sigma == sig) & (V.rep == rep)]
        if len(g) != 1:
            raise K.Stop('matrix_drift', f'first-batch view not unique / missing: {(sec, s, f, reg, size, lay, ph, sig, rep)}')
        if g.source.iloc[0] != 'new':
            raise K.Stop('matrix_drift', f'first-batch view is not a new input: {(sec, s, f, reg, size, lay, ph, sig, rep)}')
        keys.append(g.input_key.iloc[0])
    return list(dict.fromkeys(keys))


def dependencies() -> dict:
    import s1r_common as S1K
    import s2p_formal as SF
    rel_rec = K.load_json(K.REL / 'release_surf02.json'); deps = K.load_json(K.ROOT / rel_rec['dependency_manifest'])
    drift = [f for f, h in deps['files'].items() if not (K.ROOT / f).exists() or K.sha_file(K.ROOT / f) != h]
    S1K.check_s1_dependencies()
    fz2 = SF.verify_freeze()
    gd = K.S1OUT / 'domain' / 'global_domain_step0p50.json'
    return dict(release=dict(file=K.rel(K.REL / 'release_surf02.json'), sha256=K.sha_file(K.REL / 'release_surf02.json'), version=rel_rec['version']),
                platform_dependency_manifest=dict(file=rel_rec['dependency_manifest'], sha256=K.sha_file(K.ROOT / rel_rec['dependency_manifest']), n_files=len(deps['files']), drift=drift),
                s1_dependencies=dict(file=K.rel(S1K.S1_DEPS), sha256=K.sha_file(S1K.S1_DEPS)), s2_formal_freeze=fz2,
                binding=dict(file='outputs/surface_revision/sections_adapter/binding.json', sha256=K.sha_file(K.ROOT / 'outputs/surface_revision/sections_adapter/binding.json')),
                global_domain=dict(file=K.rel(gd), sha256=K.sha_file(gd), npz_sha256=K.sha_file(gd.with_suffix('.npz')), hash=K.load_json(gd)['hash'], n_nodes=K.load_json(gd)['n_nodes']),
                run_code={f: K.sha_file(K.HERE / f) for f in RUN_CODE},
                reused_modules={K.rel(p): K.sha_file(p) for p in (K.ADP / 's1r_eval.py', K.ADP / 's1r_common.py', K.S2P / 's2p_case.py', K.S2P / 's2p_common.py', K.ADP / 'adp_patterns.py')})


def write() -> dict:
    V = pd.read_csv(K.MANIFEST, keep_default_na=False, float_precision='round_trip')
    mc = K.load_json(K.CHECKS / 'manifest_check.json'); rc = K.load_json(K.CHECKS / 'reuse_check.json')
    if mc['problems'] or mc['manifest_sha256'] != K.sha_file(K.MANIFEST) or mc['inputs_sha256'] != K.sha_file(K.INPUTS):
        raise K.Stop('matrix_drift', 'manifest check not clean or manifest changed after the check')
    if not rc['all_three_layers']:
        raise K.Stop('reuse_refused', 'reuse verification not clean')
    d = dependencies()
    if d['platform_dependency_manifest']['drift']:
        raise K.Stop('platform_mismatch', f"platform dependency drift {d['platform_dependency_manifest']['drift'][:5]}")
    P = K.platform_ids()
    proto = dict(task=K.TASK_ID, task_file=str(K.TASK_FILE), task_file_sha256=(K.sha_file(K.TASK_FILE) if K.TASK_FILE.exists() else None), created_local=K.now_local(),
                 scope=dict(surfaces_main=list(K.MAIN), supplement=K.GAUSS, sigma=list(K.SIGMAS),
                            S1=dict(dots=dict(radius_mm=list(K.DOT_RADII), lattice='sn_lib S1 8 x 4 mm (original generator)', phases='sn_phase0..2', reps='sigma 0: 1; else 3',
                                              regimes=list(K.REGIMES)),
                                    lines=dict(width_mm=list(K.LINE_WIDTHS), pitch_mm=8.0, phi_deg=0.0, phases='ph0..7 (registered phase_frac)', reps='sigma 0: 1; else 2',
                                               regimes=['resource_cap_11695', 'fixed_peak_1'])),
                            S2=dict(dots=dict(radius_mm=1.0, lattices=list(K.S2_LATTICES), positions='8 registered positions (sl_lib latin indices)', reps='sigma 0: 1; else 2'),
                                    lines=dict(width_mm=1.0, pitch_mm=list(K.S2_PITCHES), phi_deg=list(K.S2_PHIS), phases='ph0..7', reps='sigma 0: 1; else 2'),
                                    regimes=['resource_cap_11695'], not_run='old optional fixed-peak S2 matrix'),
                            not_in_round='new sizes / pitches / phases, sections 3-5, other budgets, shot noise / speckle, detector retuning'),
                 resource=dict(name='resource_cap_11695', text='gemeinsame Grauwert-Integral-Obergrenze und Peak-Obergrenze', Q_budget=K.Q_BUDGET, v_max=K.V_MAX,
                               rule='Q0 = sum(I0) on the actual 0/1 raster; v_target = min(1, 11695 / Q0); I = float32(v_target) I0; Q_actual = sum(I)',
                               classes='Q0 < 11695 peak_limited (peak 1, Q_actual = Q0 < budget); Q0 > 11695 integral_limited (Q_actual = 11695, rel 1e-6); equal: both_limits',
                               controls=dict(fixed_peak_1='S1 dots / lines, peak 1, may exceed 11695 (mechanism control, not a candidate of the common cap)',
                                             equal_integral_5850='existing S1 dot results only (low budget supplement; no 5850 lines; no cross-family ranking)'),
                               unit='normalised pattern-plane grey sum (not watt, photons or camera grey sum)', S2_note='every registered S2 raster has Q0 >= 11695: old strict equal-integral S2 = common cap numerically'),
                 seeds=K.SEED_RULES, identities=dict(image='raster key + float32 peak + platform surface + PSF + sigma + effective seed + M3 config digest + platform version',
                                                     decode='image + arm + D0 cfg / P4 parameters + float32 pattern peak (lines) + search window', evaluate='decode + s1r_eval sha + global domain hash + d / tau / gross',
                                                     labels='regime / section are view labels, not part of the physical identity'),
                 manifest=dict(file=K.rel(K.MANIFEST), sha256=K.sha_file(K.MANIFEST), inputs=K.rel(K.INPUTS), inputs_sha256=K.sha_file(K.INPUTS), counts=mc['counts']),
                 reuse=dict(check=K.rel(K.CHECKS / 'reuse_check.json'), sha256=K.sha_file(K.CHECKS / 'reuse_check.json'), n=rc['n_reused'], sources_file=K.rel(K.CHECKS / 'reuse_sources.json'),
                            sources_sha256=K.sha_file(K.CHECKS / 'reuse_sources.json')),
                 evaluation=dict(global_domain=f"{P['global_nodes']} nodes 0.5 mm (s1_real01 domain, hash {P['global_domain']})", local='r <= 15 mm, 22605 nodes (SURF-02 ROI)', d_mm=K.D_SUPPORT,
                                 tau_mm=K.TAU, gross_mm=K.GROSS, evaluator='s1r_eval.evaluate_case (unchanged)', no_output='coverage / recovery 0, errors undefined; technical missing is not a scientific 0'),
                 statistics=dict(aggregation='realisations averaged inside a phase, phases equally weighted (main = mean); median, range, all phase values, no-output / zero-recovery counts, defined denominators',
                                 resampling=dict(kind='phase-block bootstrap, descriptive 5-95 %', seed=K.BOOT_SEED, draws=K.BOOT_N, rule='one index matrix shared by all linked configurations, regimes, sigma and realisations; S1 dots (3 phases): min-max only'),
                                 contrasts=dict(S1='same regime / surface / sigma: all adjacent sizes and min vs max; same size: common cap vs fixed peak, common cap vs 5850 (dots); identical inputs flagged',
                                                S2='adjacent p_par (p_perp 4), adjacent p_perp (p_par 8), R8x4 vs T8x4, phi 45 vs 0 at equal pitch, adjacent pitches per phi; sigma changes of C_tau / point error / recovery / identity risk'),
                                 ranking='per sigma by global C_tau mean (with local C_tau, point P95, wrong / unconfirmable, failure denominators); measured candidates only, no device recommendation'),
                 sensitivity=dict(d_mm=list(K.SENS_D), tau_mm=list(K.SENS_TAU), scope='common-cap main line, three main surfaces, all registered conditions, global domain; from stored valid points (no decoding)'),
                 diagnostics=dict(rule='per section x family <= 2 extra groups: lowest global C_tau (degradation), highest defined point RMSE (tail); neighbours with the same surface / phase / seed; ties by manifest order'),
                 examples=EXAMPLES, first_batch=dict(rule=FIRST_RULE, rows=[list(r) for r in FIRST], input_keys=first_keys(V)),
                 caps=dict(K.CAPS, rule='new renders / main decodes <= 8192 + 16 technical attempts each; replay decodes <= 12 (separate); failed attempts count'),
                 dependencies=d, run_code_note='report-side code (stats / sensitivity / figures / checks / report) is not part of the numerical identity')
    K.save_json(K.PROTOCOL, proto)
    return proto


def verify(check_s2=True) -> dict:
    if not K.PROTOCOL.exists():
        raise K.Stop('usage', 'protocol.json missing: run `s12_run.py freeze`')
    p = K.load_json(K.PROTOCOL); d = p['dependencies']; bad = []
    if K.sha_file(K.MANIFEST) != p['manifest']['sha256'] or K.sha_file(K.INPUTS) != p['manifest']['inputs_sha256']:
        bad.append('manifest / inputs changed')
    for f, h in d['run_code'].items():
        if K.sha_file(K.HERE / f) != h:
            bad.append(f'run code {f} changed')
    for f, h in d['reused_modules'].items():
        if K.sha_file(K.ROOT / f) != h:
            bad.append(f'reused module {f} changed')
    for k in ('release', 'platform_dependency_manifest', 's1_dependencies', 'binding', 'global_domain'):
        if K.sha_file(K.ROOT / d[k]['file']) != d[k]['sha256']:
            bad.append(f'{k} file changed')
    rel_rec = K.load_json(K.ROOT / d['release']['file']); deps = K.load_json(K.ROOT / rel_rec['dependency_manifest'])
    drift = [f for f, h in deps['files'].items() if not (K.ROOT / f).exists() or K.sha_file(K.ROOT / f) != h]
    if drift:
        bad.append(f'platform dependency drift {drift[:4]}')
    import s1r_common as S1K
    S1K.check_s1_dependencies()
    if check_s2:
        import s2p_formal as SF
        SF.verify_freeze()
    if bad:
        raise K.Stop('platform_mismatch', f'protocol drift: {bad[:6]}')
    return dict(ok=True, protocol_sha256=K.sha_file(K.PROTOCOL))
