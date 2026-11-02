# Validate environment inputs, saved results and paired statistics.

from __future__ import annotations

import json

import pandas as pd

import mp4_gate as G4
import mp5_common as K

MP5 = 'scripts/surface_revision/model_prep05/'
ENTRY_FILES = tuple(MP5 + f for f in ('mp5_common.py', 'mp5_chain.py', 'mp5_gate.py', 'mp5_main.py', 'mp5_drills.py')) + (
    'scripts/surface_revision/model_prep03/mp3_chain.py', 'scripts/surface_revision/model_prep03/mp3_common.py', 'scripts/surface_revision/model_prep04/mp4_gate.py',
    'scripts/surface_revision/model_prep04/mp4_common.py', 'scripts/surface_revision/thesis_reinforce04/tr4_gate.py', 'scripts/surface_revision/thesis_reinforce04/tr4_common.py')
ANALYSIS_FILES = (MP5 + 'mp5_analysis.py', 'scripts/surface_revision/model_prep03/mp3_analysis.py')
FIGURE_FILES = (MP5 + 'mp5_figs.py',)
REGISTRY_FILES = ('scripts/surface_revision/thesis_reinforce03/tr3_b2.py', 'scripts/surface_revision/thesis_reinforce03/tr3_common.py',
                  'scripts/surface_revision/thesis_reinforce02/tr2_common.py', 'scripts/surface_revision/thesis_reinforce01/tr_a4.py')
LAYERS = ('forward', 'decode', 'evaluate', 'inputs', 'chain', 'entry', 'analysis', 'figures', 'registry_code')
PROOF_LAYERS = dict(static=('forward', 'decode', 'evaluate', 'inputs', 'entry', 'registry_code', 'registry', 'manifest', 'rules'),
                    replay=('decode', 'evaluate', 'inputs', 'chain'),
                    drills=('entry', 'analysis', 'figures', 'manifest', 'rules'),
                    entry=('forward', 'decode', 'evaluate', 'inputs', 'entry', 'analysis', 'figures', 'manifest', 'rules'),
                    analysis=('analysis', 'manifest', 'rules', 'results'),
                    figures=('analysis', 'figures', 'rules', 'results'),
                    selfcheck=('analysis', 'rules', 'results'))
REQUIRED = dict(
    static=('A1_equivalence_16', 'A1_inputs_32', 'A1_rows_2450', 'A1_no_template_fallback', 'A1_decoder_isolation', 'A1_two_registries', 'A3_manifest_frozen', 'A3_registry_snapshot',
            'A3_protected_unchanged', 'A3_freeze_current'),
    replay=('A2_replay_exact', 'A2_replay_budget'),
    drills=('A3_mod_phase', 'A3_mod_seed', 'A3_mod_parent', 'A3_mod_methods', 'A3_row_deleted', 'A3_row_duplicated', 'A3_manifest_missing', 'A3_adapter_changed',
            'A3_registry_changed', 'A3_rules_changed_old_final', 'A4_two_reps_two_pairs', 'A4_missing_rep', 'A4_wrong_rep', 'A4_duplicate_key', 'A4_legal_no_output',
            'A4_valid_metric_mismatch', 'A4_missing_column', 'A4_selection_not_cand_id', 'A4_mean_015', 'A4_sigma0_single'),
    entry=('B_inputs_match', 'B_psf_field_isolation', 'B_noise_pairing', 'B_complete_unique', 'B_real_chain', 'B_selfcheck', 'B_throughput', 'B_freeze_protected'),
    analysis=('Q_results_complete', 'Q_effects_one_to_one', 'Q_no_output_kept', 'Q_risk_per_image'),
    figures=('V_figures',),
    selfcheck=('S_recomputed',))


def layer_files() -> dict:
    import mp5_chain as CH
    return dict(forward=CH.FORWARD_FILES, decode=CH.DECODER_FILES, evaluate=CH.EVAL_FILES, inputs=(CH.ADAPTER_FILE,), chain=(CH.CHAIN_FILE,), entry=ENTRY_FILES,
                analysis=ANALYSIS_FILES, figures=FIGURE_FILES, registry_code=REGISTRY_FILES)


def hash_layers() -> dict:
    return {L: {f: (K.sha(K.ROOT / f) if (K.ROOT / f).exists() else None) for f in fs} for L, fs in layer_files().items()}


MANIFEST_COLS = ('case_key', 'subset', 'family', 'config', 'template_cand_id', 'surface', 'platform_surface', 'phase_block', 'phase_u', 'phase_v', 'phase_frac', 'sigma', 'rep', 'seed',
                 'noise_block', 'env', 'psf_true_px', 'field_id', 'raster_key', 'peak_f32', 'integral', 'methods', 'forward_key', 'parent_case_key', 'parent_forward_key')


def manifest_parts(path) -> dict:

    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return dict(sha256=None, rows_digest=None, n_rows=None, problem='manifest missing')
    try:
        M = pd.read_csv(p, low_memory=False)
    except Exception as e:
        return dict(sha256=K.sha(p), rows_digest=None, n_rows=None, problem=f'manifest unreadable: {e!r}'[:200])
    miss = [c for c in MANIFEST_COLS if c not in M.columns]
    if miss:
        return dict(sha256=K.sha(p), rows_digest=None, n_rows=len(M), problem=f'manifest columns missing {miss}')
    rows = M[list(MANIFEST_COLS)].astype(str).values.tolist()
    return dict(sha256=K.sha(p), rows_digest=K.digest(rows), n_rows=len(M), n_unique_case=int(M.case_key.nunique()), problem=None)


def registry_snapshot(reg: dict | None = None) -> dict:
    import tr3_b2 as B2
    reg = B2.registry() if reg is None else reg
    srcs = sorted({s['source'] for s in reg['sources'] if str(s['source']).endswith('.csv') and (K.ROOT / s['source']).exists()})
    fixed = ['outputs/size_spacing/s5_03/usage/used_phases_seeds.csv', 'outputs/size_spacing/s5_04/usage/used_phases_seeds_supplement.csv',
             'outputs/surface_revision/sections_adapter/candidates_s1s5.csv', 'outputs/size_spacing/s5_04/usage/unseen_scope_check.json',
             'outputs/surface_revision/s12_resource01/results/views.csv']
    files = sorted(set(srcs) | set(fixed))
    return dict(files={f: (K.sha(K.ROOT / f) if (K.ROOT / f).exists() else None) for f in files},
                phases=K.digest(reg['phases'].astype(str).values.tolist()), positions=K.digest(reg['positions'].astype(str).values.tolist()),
                seeds=K.digest(sorted(int(x) for x in reg['seeds'])), raster_keys=K.digest(sorted(reg['raster_keys'])), n=dict(phases=len(reg['phases']), positions=len(reg['positions']),
                                                                                                                                 seeds=len(reg['seeds']), raster_keys=len(reg['raster_keys'])))


def registry_files_now(snapshot: dict) -> dict:
    return {f: (K.sha(K.ROOT / f) if (K.ROOT / f).exists() else None) for f in (snapshot or {}).get('files', {})}


def parts(manifest_path, rules: dict, freeze: dict | None, results_sha: str | None = None, layer_override: dict | None = None) -> dict:

    L = hash_layers()
    for k, v in (layer_override or {}).items():
        L[k] = v
    snap = (freeze or {}).get('registry') or {}
    out = {k: K.digest(v) for k, v in L.items()}
    out.update(registry=K.digest(registry_files_now(snap)), manifest=K.digest({k: v for k, v in manifest_parts(manifest_path).items() if k != 'problem'}), rules=K.digest(rules),
               results=results_sha)
    return out


def freeze_record(manifest_path, rules: dict, snap: dict) -> dict:
    mp = manifest_parts(manifest_path)
    if mp['problem']:
        raise K.Stop('refused', mp['problem'])
    return dict(layers=hash_layers(), manifest=mp, registry=snap, rules_digest=K.digest(rules), rules=rules, caps=K.CAPS, t=K.now())


def verify_freeze(freeze: dict | None, manifest_path, rules: dict, layer_override: dict | None = None) -> list:
    if not freeze:
        return ['no freeze record']
    bad = []
    L = hash_layers()
    for k, v in (layer_override or {}).items():
        L[k] = v
    for name in LAYERS:
        if (freeze.get('layers') or {}).get(name) != L.get(name):
            old, new = (freeze.get('layers') or {}).get(name) or {}, L.get(name) or {}
            bad.append(f'layer {name} changed: {sorted(f for f in set(old) | set(new) if old.get(f) != new.get(f))[:4]}')
    mp = manifest_parts(manifest_path)
    if mp.get('problem'):
        bad.append(mp['problem'])
    for k in ('sha256', 'rows_digest', 'n_rows'):
        if (freeze.get('manifest') or {}).get(k) != mp.get(k):
            bad.append(f'manifest {k} differs from the freeze')
    now = registry_files_now(freeze.get('registry'))
    if now != (freeze.get('registry') or {}).get('files'):
        bad.append(f"registry source files changed: {sorted(f for f in now if now[f] != (freeze.get('registry') or {}).get('files', {}).get(f))[:4]}")
    if freeze.get('rules_digest') != K.digest(rules):
        bad.append('rules changed since the freeze')
    if freeze.get('caps') != K.CAPS:
        bad.append('caps changed since the freeze')
    return bad


def proof_state(run: dict, prts: dict, stages) -> dict:
    pr = run.get('proofs') or {}
    return {st: G4.check_proof(pr.get(st), REQUIRED[st], G4.binding(prts, PROOF_LAYERS[st])) for st in stages}


def make_proof(rows: list, prts: dict, stage: str) -> dict:
    return G4.make_proof(rows, G4.binding(prts, PROOF_LAYERS[stage]))


def ledger_audit(P, expected: dict | None = None) -> dict:
    rows, probs = G4.ledger_rows(P.ledger)
    if not P.ledger.exists():
        rows, probs = [], []
    a = G4.ledger_audit(rows, K.CAPS, expected); a['problems'] = probs + a['problems']
    return a
