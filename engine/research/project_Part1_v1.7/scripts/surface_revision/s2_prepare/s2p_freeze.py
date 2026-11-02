# Record source and input identities for spacing-layout execution.

from __future__ import annotations

import s2p_common as K

K.setup()
PARTS = {
    'image': ('src/m1_tire_surface.py', 'src/m2_pattern.py', 'src/m3_virtual_camera.py', 'scripts/spacing_layout/sl_lib.py', 'scripts/spotnoise/sn_lib.py', 'scripts/line01r/l01rb_common.py',
              'scripts/surface_revision/surf02/s2_common.py', 'scripts/surface_revision/surf01/s1_model.py', 'scripts/surface_revision/sections_adapter/adp_patterns.py', 'config.py'),
    'decode': ('src/m4_reconstruction.py', 'src/m4_lines.py', 'scripts/spotnoise_round5/r5_detect.py', 'scripts/surface_revision/sections_adapter/adp_surf02.py',
               'scripts/surface_revision/sections_adapter/adp_binding.py', 'outputs/line01r/c3/config/paths_c3_validation.json'),
    'eval': ('src/surface_eval_r3.py', 'scripts/surface_revision/sections_adapter/s1r_eval.py', 'scripts/surface_revision/sections_adapter/s1r_common.py', 'scripts/line01/l01_truth.py',
             'scripts/line01/l01_common.py', 'outputs/surface_revision/s1_real01/domain/global_domain_step0p50.json', 'outputs/surface_revision/s1_real01/domain/global_domain_step0p50.npz'),
}


def _own():
    return sorted(K.rel(p) for p in K.HERE.glob('s2p_*.py'))


def mesh_hashes() -> dict:
    import s1r_common as S1K
    S = S1K.s2()
    return {sid: S.mesh_sha(S.build_mesh(sid)) for sid in sorted(set(K.PLATFORM_ID.values()))}


def snapshot() -> dict:
    import s2p_manifest as MF
    files = set(K.load_json(K.S1_DEPS)['files']) | set(_own()) | {K.rel(MF.CSV), K.rel(MF.CONFIG), 'outputs/surface_revision/surf02/release/release_surf02.json',
                                                                   'outputs/surface_revision/surf02/release/dependencies_surf02_v1_1.json'}
    for p in PARTS.values():
        files |= set(p)
    return dict(files={f: (K.sha_file(K.ROOT / f) if (K.ROOT / f).exists() else None) for f in sorted(files)}, meshes=mesh_hashes(), parts={k: list(v) for k, v in PARTS.items()})


def part_digest(snap: dict, part: str) -> str:
    return K.digest({f: snap['files'][f] for f in PARTS[part]} | ({'meshes': snap['meshes']} if part in ('image', 'eval') else {}), 24)


def write(task='S2-PREP-02') -> dict:
    s = snapshot(); s.update(task=task, created_local=K.now_local(), part_digests={p: part_digest(s, p) for p in PARTS})
    K.save_json(K.FREEZE, s)
    return s


def verify() -> dict:

    if not K.FREEZE.exists():
        return dict(ok=False, reason='no freeze file', drift=[])
    fz = K.load_json(K.FREEZE); now = snapshot()
    drift = [f for f, h in fz['files'].items() if now['files'].get(f) != h]
    mesh_drift = [s for s, h in fz['meshes'].items() if now['meshes'].get(s) != h]
    return dict(ok=(not drift and not mesh_drift), drift=drift, mesh_drift=mesh_drift, freeze_sha256=K.sha_file(K.FREEZE), part_digests=fz['part_digests'])
