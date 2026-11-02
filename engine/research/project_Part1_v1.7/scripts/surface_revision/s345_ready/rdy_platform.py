# Load the bound platform and expose surfaces and evaluation domains.

from __future__ import annotations

from functools import lru_cache

import numpy as np

import rdy_common as K


@lru_cache(maxsize=1)
def s2():
    import s1r_common as S1K
    return S1K.s2()


@lru_cache(maxsize=1)
def bound():
    import s1r_common as S1K
    bd = S1K.bound()
    if bd.version != K.PLATFORM_VERSION:
        raise K.Stop('platform', f'bound platform version {bd.version!r} != {K.PLATFORM_VERSION!r}')
    if tuple(bd.window()) != K.WINDOW_MM:
        raise K.Stop('platform', f'platform window {bd.window()} != {K.WINDOW_MM}')
    S = s2()
    if abs(float(S.C_REF) - K.C_REF_MM) > 1e-12 or float(S.GROSS_MM) != K.GROSS or float(S.ROI_R_MM) != 15.0 or float(S.MARGIN_MM) != K.MARGIN_MM:
        raise K.Stop('platform', 'platform constants (c, gross, ROI radius, margin) differ from the registered values')
    return bd


def release_state() -> dict:
    import adp_surf02 as SA
    r = SA.release()
    rec = K.load_json(SA.RELEASE_RECORD)
    man = K.load_json(K.ROOT / rec['dependency_manifest'])
    return dict(version=rec['version'], released=rec['released'], record=K.rel(SA.RELEASE_RECORD), record_sha256=K.sha_file(SA.RELEASE_RECORD),
                manifest=rec['dependency_manifest'], manifest_sha256=K.sha_file(K.ROOT / rec['dependency_manifest']), n_dependency_files=len(man['files']),
                drift=SA.dependency_drift(K.ROOT / rec['dependency_manifest']), config_sha256=r.get('config_sha256'))


@lru_cache(maxsize=4)
def mesh(sid_adapter: str):
    return s2().build_mesh(K.SURF_MAP[sid_adapter])


@lru_cache(maxsize=4)
def dom_local(sid_adapter: str):
    return s2().roi_domain(mesh(sid_adapter))


@lru_cache(maxsize=1)
def dom_global():
    import s1r_common as S1K
    d = S1K.global_domain()
    if int(d['rec']['n_nodes']) != K.N_GLOBAL_NODES or d['hash'] != K.GLOBAL_DOMAIN_HASH:
        raise K.Stop('platform', f"global domain {d['rec']['n_nodes']} nodes / hash {d['hash'][:12]} differ from the registered 30947 / {K.GLOBAL_DOMAIN_HASH[:12]}")
    return d


def surfaces() -> dict:

    S = s2(); out = {}; ref = {}
    if K.S2_FREEZE.exists():
        ref = (K.load_json(K.S2_FREEZE).get('meshes') or {})
    rng = np.random.default_rng(20261012)
    for a, sid in K.SURF_MAP.items():
        m = mesh(a); meta = m['meta']; spec = S.SURFACES[sid]
        X, Y, Z = m['X'], m['Y'], m['Z']
        iy = rng.integers(0, X.shape[0], 25); ix = rng.integers(0, X.shape[1], 25)
        zt = S.z_true(sid, X[iy, ix], Y[iy, ix])
        dz = float(np.max(np.abs(zt - Z[iy, ix])))
        h = S.mesh_sha(m); dl = dom_local(a)
        out[a] = dict(platform_id=sid, variant=meta['variant'], gauss=bool(meta['gauss']), z_alignment=m['z_alignment'], frame=meta['frame'], c_ref_mm=float(meta['c_ref_mm']), grid=meta['grid'],
                      mesh_sha16=h, mesh_sha16_section2=ref.get(sid), mesh_equal_section2=(ref.get(sid) == h if ref else None), max_abs_z_minus_c_minus_H_mm=dz,
                      z_range_roi=list(dl['z_range']), n_local_nodes=int(dl['n_nodes']), window_cov_roi=float(dl['window_cov_roi']),
                      ok=bool(meta['variant'] == spec['variant'] and bool(meta['gauss']) == bool(spec['gauss']) and m['z_alignment'] == 'absolute' and dz <= 1e-9
                              and abs(float(meta['c_ref_mm']) - K.C_REF_MM) <= 1e-12 and int(dl['n_nodes']) == K.N_LOCAL_NODES and (not ref or ref.get(sid) == h)))
    expect = {'M10_curv': ('A', False), 'M24_curv': ('C', False), 'B_curv': ('B', False), 'M10_curv_gauss': ('A', True)}
    for a, (v, g) in expect.items():
        out[a]['ok'] = bool(out[a]['ok'] and out[a]['variant'] == v and out[a]['gauss'] == g)
    return out


def contract() -> dict:
    d = dom_global()
    return dict(local='SURF-02 ROI r <= 15 mm, scientific grid nodes (s2.roi_domain), evaluated by s2.evaluate', global_domain=K.rel(K.S1OUT / 'domain' / 'global_domain_step0p50.json'),
                global_nodes=int(d['rec']['n_nodes']), global_hash=d['hash'], global_json_sha256=K.sha_file(K.S1OUT / 'domain' / 'global_domain_step0p50.json'),
                global_npz_sha256=K.sha_file(K.S1OUT / 'domain' / 'global_domain_step0p50.npz'), d_mm=K.D_SUPPORT, tau_mm=K.TAU, gross_mm=K.GROSS,
                validity='s2.point_validity: finite, window [-16, +4] mm (+- 1e-6), (x, y) inside the grid; negative z valid',
                evaluator='sections_adapter/s1r_eval.evaluate_case (S1-REAL-01 contract; loc_* / glob_* fields)')
