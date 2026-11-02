# Record the source and input manifest of the encoding experiment.

from __future__ import annotations

import json
import shutil
import sys

import c7_common as K

MODULES = ['c7_common', 'c7_config', 'c7_run', 'c7_deriv', 'c7_entry', 'c7_tables', 'c7_cases', 'c7_figs', 'c7_plan', 'c7_configs', 'c7_freeze', 'c7_drills', 'c7_gate',
           'c7_check', 'c7_final', 'c7_fix06', 'c7_regress', 'c7_startup']
FROZEN = K.FORMAL / 'frozen.json'
PLAN_FILES = ['expected_keys_formal.csv', 'pairs_expected.csv', 'startup_subset.csv', 'cases_formal.json', 'n_material.csv']


def freeze() -> dict:
    import c3_deps as D3
    import c6_deriv as DX
    mods = [m for m in MODULES if (K.HERE / f'{m}.py').exists()]
    cl = D3.closure(mods); configs = sorted(K.rel(p) for p in K.CONFIGS.glob('*.json')); plan = [K.rel(K.PLAN / f) for f in PLAN_FILES]
    files = sorted(set(cl) | set(configs) | set(plan))
    inherited = {}
    for nm, p in (('S3-CODE-04', K.O4 / 'formal' / 'frozen.json'), ('S3-CODE-06', K.O6 / 'formal' / 'frozen.json')):
        fz = K.load_json(p); inherited[nm] = dict(file=K.rel(p), closure_sha256=fz.get('closure_sha256'), mismatches=len(K.K4.verify_closure(fz['closure'])))
    if any(v['mismatches'] for v in inherited.values()):
        raise K.Stop('freeze', f'an inherited freeze no longer verifies: {inherited}')
    if FROZEN.exists():
        old = K.load_json(FROZEN); keep = K.guard(K.FORMAL / f"frozen_superseded_{old['closure_sha256'][:8]}.json")
        if not keep.exists():
            shutil.copy2(FROZEN, keep)
    rr, rm = K.K4.K3.RELEASE / 'release_surf02.json', K.K4.K3.RELEASE / 'dependencies_surf02_v1_1.json'
    closure = dict(files=K.K4.files_digest(files, K.ROOT), surf02_release=dict(record=K.rel(rr), record_sha256=K.sha_file(rr), manifest=K.rel(rm), manifest_sha256=K.sha_file(rm)))
    fps = {l: K.K4.fingerprint(l) for l in ('decode', 'eval', 'diag')}
    out = dict(kind='S3-CODE-07 package freeze (formal section 3)', closure=closure, n_files=len(files), modules=mods, configs={c: K.sha_file(K.ROOT / c) for c in configs},
               plan={p: K.sha_file(K.ROOT / p) for p in plan}, fingerprints=fps, fingerprints_s3_code04=K.load_json(K.O4 / 'formal' / 'frozen.json')['fingerprints'],
               layering='layer fingerprints recomputed from the S3-CODE-04 layer files: equal to the S3-CODE-04 registration (scientific code unchanged); runner / '
                        'analysis / figures of this round are interface code bound by this closure',
               derived_definition=DX.definition(), inherited=inherited, permit=None, t_local=K.now_local())
    out['closure_sha256'] = K.digest(out['closure'], 64)
    K.save_json(FROZEN, out)
    return {k: v for k, v in out.items() if k not in ('closure', 'configs', 'plan', 'derived_definition')}


def verify() -> dict:
    fz = K.load_json(FROZEN); bad = K.K4.verify_closure(fz['closure'])
    return dict(ok=not bad, mismatches=bad[:10], closure_sha256=fz['closure_sha256'], n_files=fz['n_files'])


if __name__ == '__main__':
    if sys.argv[1] == 'freeze':
        print(json.dumps(K.clean(freeze()), indent=1, ensure_ascii=False)[:3000])
    else:
        r = verify(); print(json.dumps(K.clean(r), indent=1)); sys.exit(0 if r['ok'] else 15)
