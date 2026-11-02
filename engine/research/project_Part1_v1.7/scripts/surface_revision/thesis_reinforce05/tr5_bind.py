# Bind new-phase inputs and decoding methods to the platform dependencies.

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import tr5_common as K
import tr5_manifest as MF

CHAIN_MODULES = ('m1_tire_surface', 'm2_pattern', 'm3_virtual_camera', 'm4_reconstruction', 'm4_lines', 'surface_eval_r3', 'adp_common', 'adp_binding', 'adp_surf02',
                 'adp_patterns', 's1r_common', 's1r_eval', 's2_common', 'r5_detect', 'rdy_platform', 'rdy_common', 'rdy_decode', 's3c_dev', 's3c_common', 'c2_decode',
                 'ep_detect', 'ep_patterns', 's2p_case', 's12_common', 'l01_common', 'l01_truth', 'l01rb_common', 'sl_lib', 'tr_a4', 'tr3_gate', 'tr3_b1', 'tr4_gate',
                 'tr4_chain', 'tr4_prep', 'tr5_common', 'tr5_manifest', 'tr5_bind', 'tr5_chain', 'tr5_claims', 'tr5_analysis')
UNFROZEN = {K.rel(K.HERE / f) for f in K.LAYOUT_FILES + K.DRILL_FILES}


def platform_checks() -> dict:
    import tr4_prep as P4
    return P4.platform_checks()


def _closure_files(path, kind='files') -> dict:
    J = K.load_json(path)
    f = J['files'] if kind == 'files' else J['closure']['files']
    return {str(a).replace('\\', '/'): h for a, h in f.items()}


def _current(files: dict) -> tuple[dict, list]:
    cur = {}; diff = []
    for f, h in files.items():
        q = K.ROOT / f; c = K.sha(q) if q.exists() else None; cur[f] = c
        if c != h:
            diff.append(f)
    return cur, diff


def method_bindings(M) -> dict:
    import tr3_gate as G3
    import tr_a4 as A4
    S = MF.s2(); R = K.ROOT / 'outputs' / 'surface_revision'
    s1, s1d = _current(_closure_files(R / 's1_real01/config/s1_dependencies.json'))
    c4, c4d = _current(_closure_files(R / 's3_code04/formal/frozen.json', 'closure'))
    base = K.load_json(K.OUT3 / 'run.json')['baseline_posthoc']
    dcfg = G3.live_decode_cfg(); thr = G3.live_thresholds()
    dom = {s: G3.domain_content(s) for s in MF.SURF_SID} | dict(glob=G3.domain_content(None))
    L = M[M.family == 'line'].drop_duplicates('raster_key')
    p4cfg = {f"{r.raster_key}|peak={r.peak!r}": S.line_cfg(float(r.peak)) for r in L.itertuples()}
    radii = sorted(set(float(x) for x in M[M.family == 'point'].size_mm))
    a4 = dict(file=K.rel(Path(A4.__file__)), sha256=K.sha(Path(A4.__file__)), tol_px=A4.TOL_PX,
              rule='unconfirmable features inside the projector image, within w/2 + 1 px of a lit but unregistered line k and on / next to a lit pixel get label k')
    B = dict(
        D0=dict(entry='adp_surf02.decode("D0") via adp_binding (s1r_common.ARM point)', cfg=S.dot_cfg(), closure='S1-REAL-01 dependency manifest', closure_files=s1, closure_differs=s1d),
        DF=dict(entry='s3c_dev.dot_decode(img, binary, 1, [radius], "DF", "binding") + s3c_dev.dot_dec(D, "single")', cfg=dcfg, radii=radii,
                cfg_equals_tr03_baseline=(G3.digest(dcfg) == base['decode_cfg_content']), closure='S3-CODE-04 frozen closure', closure_files=c4, closure_differs=c4d),
        P4=dict(entry='adp_surf02.decode("P4_C3_cal") via adp_binding; cfg = s2.line_cfg(actual peak) + s2.P4; pitch / orientation / phase from the raster meta',
                cfg=dict(per_raster=p4cfg, P4=S.P4), closure='S1-REAL-01 dependency manifest', closure_files=s1, closure_differs=s1d),
        EVAL=dict(entry='s1r_eval.evaluate_case(sid, family, dec, ctx, mesh, rdy_platform.dom_local(surface), rdy_platform.dom_global(), gauss=False)',
                  thresholds=thr, thresholds_equal_tr03_baseline=(thr == base['thresholds']), domains=dom,
                  domains_equal_tr03_baseline=all(dom[k] == base['domains_content'][k] for k in dom), identity_rule=dict(dots='evaluator identity (A4 not applicable)', lines=a4),
                  closure='S1-REAL-01 dependency manifest', closure_files=s1, closure_differs=s1d))
    for m, b in B.items():
        b['fingerprint'] = K.digest(dict(method=m, entry=b['entry'], cfg=b.get('cfg'), thresholds=b.get('thresholds'), domains=b.get('domains'),
                                         identity_rule=b.get('identity_rule'), code=b['closure_files']))
    passed = bool(not s1d and not c4d and B['DF']['cfg_equals_tr03_baseline'] and B['EVAL']['thresholds_equal_tr03_baseline'] and B['EVAL']['domains_equal_tr03_baseline']
                  and len({b['fingerprint'] for b in B.values()}) == 4)
    slim = {m: {k: v for k, v in b.items() if k != 'closure_files'} | dict(n_closure_files=len(b['closure_files'])) for m, b in B.items()}
    return dict(passed=passed, methods=slim, fingerprints={m: b['fingerprint'] for m, b in B.items()})


def input_files() -> dict:
    R = K.ROOT / 'outputs' / 'surface_revision'
    f = {'task_book': K.TASK, 'review_tr04': K.REVIEW4, 'tr03_plan': K.OUT3 / 'plan.csv', 'tr03_run': K.OUT3 / 'run.json', 'tr04_run': K.OUT4 / 'run.json',
         'tr04_report': K.OUT4 / 'report_zh.md', 'tr01_run': K.OUT1 / 'run.json', 'tr01_layers_conditions': K.OUT1 / 'layers_conditions.csv',
         's12_views': R / 's12_resource01/results/views.csv', 's1_dependencies': R / 's1_real01/config/s1_dependencies.json', 'c4_frozen': R / 's3_code04/formal/frozen.json',
         'surf02_release': R / 'surf02/release/release_surf02.json', 'surf02_dependencies': R / 'surf02/release/dependencies_surf02_v1_1.json',
         'binding': R / 'sections_adapter/binding.json', 'draft_3_1_3_2': K.DRAFTS / 'chapter3_sections1_2.md',
         'global_domain': R / 's1_real01/domain/global_domain_step0p50.npz'}
    for p in sorted((K.OUT1 / 'shards').glob('*.npz')):
        f[f'tr01_shard:{p.name}'] = p
    return f


def hash_inputs(files: dict) -> dict:
    return {k: dict(path=K.rel(p), sha256=(K.sha(p) if Path(p).exists() else None)) for k, p in files.items()}


def recorded_checks(H: dict) -> dict:

    bad = []
    if H['tr03_plan']['sha256'] != K.TR03_PLAN_SHA:
        bad.append('TR03 plan.csv differs from the task-book sha')
    f3 = (K.load_json(K.OUT3 / 'run.json').get('final') or {}).get('outputs') or {}
    if H['tr03_plan']['sha256'] != f3.get('plan.csv'):
        bad.append('TR03 plan.csv differs from the TR03 final record')
    f1 = (K.load_json(K.OUT1 / 'run.json').get('final') or {}).get('shards') or {}
    sb = [k for k in H if k.startswith('tr01_shard:') and H[k]['sha256'] != f1.get(k.split(':', 1)[1])]
    if sb or len([k for k in H if k.startswith('tr01_shard:')]) != len(f1) or len(f1) != 17:
        bad.append(f'TR01 DF shards (513 DF) differ from their final record: {sb[:3]}')
    f4 = (K.load_json(K.OUT4 / 'run.json').get('final') or {}).get('outputs') or {}
    rel4 = K.rel(K.OUT4)
    changed4 = [p for p, h in f4.items() if p.startswith(rel4) and not p.endswith('run.json') and (not (K.ROOT / p).exists() or K.sha(K.ROOT / p) != h)]
    if changed4:
        bad.append(f'TR04 outputs differ from the TR04 final record: {changed4[:4]}')
    miss = [k for k, v in H.items() if v['sha256'] is None]
    return dict(passed=not bad and not miss, problems=bad, missing=miss, n_refs=len(H), n_unique_files=len({v['path'] for v in H.values()}),
                tr04_outputs_checked=len(f4), tr01_shards_checked=len(f1))


def force_imports() -> None:
    import s1r_common as S1K
    for m in tuple(S1K.LAZY_MODULES) + CHAIN_MODULES:
        importlib.import_module(m)


def loaded_project_files() -> dict:
    root = K.ROOT.resolve(); out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and str(f).endswith('.py'):
            q = Path(f).resolve()
            if root in q.parents and K.rel(q) not in UNFROZEN:
                out[K.rel(q)] = q
    return out


def code_closure() -> dict:
    force_imports()
    return {r: K.sha(q) for r, q in sorted(loaded_project_files().items())}


def closure_check(frozen: dict, allow_changed: tuple = ()) -> dict:
    changed = [f for f, h in frozen.items() if f not in allow_changed and (not (K.ROOT / f).exists() or K.sha(K.ROOT / f) != h)]
    extra = sorted(set(loaded_project_files()) - set(frozen))
    return dict(passed=not changed and not extra, changed=changed, not_frozen_but_loaded=extra, n_frozen=len(frozen))


def model_prep_status() -> dict:
    import tr2_common as T2
    return T2.model_prep_status()
