# Bind selected layouts to image conditions and strategy views.

from __future__ import annotations

import numpy as np
import pandas as pd

import r2_common as K

IMAGE_CODE = ['scripts/surface_revision/pix_bound01/pb_m3.py', 'src/m3_virtual_camera.py', 'scripts/surface_revision/sections_adapter/adp_surf02.py', 'scripts/surface_revision/surf02/s2_common.py',
              'scripts/line01/l01_common.py', 'scripts/surface_revision/s4_real02/r2_layouts.py', 'scripts/spatial_layout/sp_dots_columns.py', 'scripts/spatial_layout/spat01r/sp2_lines_table.py',
              'scripts/spatial_layout/spat01r/sp2_common.py', 'scripts/surface_revision/s345_ready/rdy_inputs.py', 'scripts/spatial_layout/sp_lines_table.py']
DECODE_CODE = ['scripts/surface_revision/s345_ready/rdy_decode.py', 'src/m4_reconstruction.py', 'src/m4_lines.py', 'scripts/spotnoise_round5/r5_detect.py', 'scripts/spatial_layout/sp_regression.py',
               'scripts/spatial_layout/spat01r/sp2_lines_table.py', 'scripts/surface_revision/sections_adapter/adp_surf02.py', 'scripts/surface_revision/sections_adapter/adp_binding.py',
               'scripts/surface_revision/surf02/s2_common.py']
EVAL_CODE = ['scripts/surface_revision/s4_real02/r2_eval.py', 'scripts/surface_revision/s4_real02/r2_eval_snap.py', 'scripts/surface_revision/s345_ready/rdy_eval.py',
             'scripts/surface_revision/sections_adapter/s1r_eval.py', 'scripts/surface_revision/sections_adapter/s1r_common.py', 'src/surface_eval_r3.py', 'scripts/spacing_layout/sl_lib.py',
             'scripts/spotnoise/sn_lib.py', 'scripts/spatial_layout/spat01r/sp2_lines_table.py', 'scripts/line01/l01_truth.py', 'scripts/surface_revision/surf02/s2_common.py',
             'scripts/surface_revision/s345_ready/rdy_platform.py', 'outputs/surface_revision/pix_bound01/domains/omega1.npz', 'outputs/surface_revision/pix_bound01/domains/omega_star.npz']
METHOD = {'point': ('D0', 'SPAT_D0'), 'line': ('P4TABLE', 'SPAT_P4table')}
ENTRY = {'R8x4': 'B021', 'L6.3': 'B000'}


def fingerprints() -> dict:
    return dict(image=K.code_fingerprint(IMAGE_CODE), decode=K.code_fingerprint(DECODE_CODE), eval=K.code_fingerprint(EVAL_CODE))


def seeds() -> dict:
    P = pd.read_csv(K.S345 / 'plan_s345.csv', low_memory=False); s = P[(P.set_formal == 'S4-reduced') & (P.sigma == 20.0)].dropna(subset=['noise_seed'])
    g = s.groupby(['family', 'surface', 'phase_label']).noise_seed
    if (g.nunique() != 1).any():
        raise K.Stop('plan', 'registered S4 sigma-20 seed not unique per family x surface x phase')
    return {k: int(v) for k, v in g.first().items()}


def m3_digest() -> str:
    import rdy_platform as P
    return K.digest({k: v for k, v in P.s2().m3_cfg(0.0, 0).items() if k not in ('noise_sigma', 'mottle_seed')})


def build(reg: dict, views_tab: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import rdy_decode as DC
    import pb_m3 as PM
    fp = fingerprints(); fpd = {k: K.digest(v, 32) for k, v in fp.items()}; sd = seeds(); m3d = m3_digest()
    import r2_eval as E
    h1, hs = E.domain('omega1')['hash'], E.domain('omega_star')['hash']
    import sp2_common as S2
    rows = []
    vt = views_tab.set_index(['baseline', 'strategy', 'layout', 'phase'])
    for name, famp, fam in K.BASES:
        B = reg['baselines'][name]; method, arm = METHOD[famp]
        phases = {p['phase_id']: p for p in S2.registered_phases(fam)}
        for v in B['views']:
            lay = B['layouts'][v['layout']]
            for ph in K.PHASES:
                vr = vt.loc[(name, v['strategy'], v['layout'], ph)]
                for s in K.SURF:
                    for sg in K.SIGMAS:
                        seed = None if sg == 0 else sd[(famp, s, ph)]
                        ifields = dict(raster=vr.raster_key, peak_f32=float(vr.peak_f32), surface=K.RK.SURF_MAP[s], psf=K.RK.PSF_PX, sigma=sg, seed=seed, m3=m3d, platform=K.RK.PLATFORM_VERSION,
                                       boundary=K.BOUNDARY, snapshot=PM.SNAPSHOT_VERSION, image_code=fpd['image'])
                        ik = 'R2I-' + K.digest(ifields, 24)
                        dk = 'R2D-' + K.digest(dict(image=ik, method=method, desc=DC.METHODS[method], arm=arm, peak_f32=float(vr.peak_f32), window=K.RK.WINDOW_MM, decode_code=fpd['decode']), 24)
                        rows.append(dict(baseline=name, family=famp, surface=s, platform_surface=K.RK.SURF_MAP[s], strategy=v['strategy'], block=(v['block'] or ''), layout=v['layout'],
                                         layout_id=lay['layout_id'], cand=(lay['cand'] if lay['cand'] is not None else ''), n_mask_candidates=(v['n_mask_candidates'] if v['n_mask_candidates'] is not None else ''),
                                         n_raster_evaluated=(v['n_raster_evaluated'] if v['n_raster_evaluated'] is not None else ''), phase=ph, phase_u=phases[ph].get('u', ''), phase_v=phases[ph].get('v', ''),
                                         phase_frac=phases[ph].get('phase_frac', ''), sigma=sg, rep=0, noise_seed=(seed if seed is not None else ''), render_seed=(seed or 0),
                                         seed_rule=('none (sigma 0)' if seed is None else 'registered S4 sigma-20 field of family x surface x phase, scaled by sigma'), raster_key=vr.raster_key,
                                         lit_px=vr.lit_px, peak=vr.peak, peak_f32=vr.peak_f32, integral_f32=vr.integral_f32, method=method, arm=arm, boundary=K.BOUNDARY, image_key=ik, decode_key=dk,
                                         domain_main='omega1', domain_main_hash=h1, domain_bridge='omega_star', domain_bridge_hash=hs, image_code=fpd['image'], decode_code=fpd['decode'], eval_code=fpd['eval'],
                                         entry_subset=bool(ph == 'H0' and sg in (0.0, 100.0) and (v['strategy'] == 'S0' or v['block'] == ENTRY[name]) and v['strategy'] in ('S0', 'S2', 'S3', 'S4'))))
    M = pd.DataFrame(rows)

    ref = M[M.sigma == 0].set_index(['baseline', 'layout', 'phase', 'surface']).image_key
    ref = ref[~ref.index.duplicated()]
    M['sigma0_ref_key'] = [ref[(r.baseline, r.layout, r.phase, r.surface)] if r.family == 'point' else '' for r in M.itertuples()]
    M['eval_key'] = ['R2E-' + K.digest(dict(decode=r.decode_key, arm=r.arm, eval_code=r.eval_code, boundary=K.BOUNDARY, omega1=h1, omega_star=hs, d=K.D_SUPPORT, tau=K.TAU, ref=r.sigma0_ref_key), 24)
                     for r in M.itertuples()]
    M['view_id'] = ['V-' + K.digest(dict(b=r.baseline, s=r.strategy, blk=r.block, l=r.layout, p=r.phase, su=r.surface, sg=r.sigma), 16) for r in M.itertuples()]
    n_img = M.groupby('image_key').size(); M['n_views_of_image'] = M.image_key.map(n_img)
    return M, dict(fingerprints=fp, fingerprint_digests=fpd, m3_digest=m3d, seeds={f'{a}|{b}|{c}': v for (a, b, c), v in sd.items()})


def validate(M: pd.DataFrame, fpd: dict | None = None) -> dict:

    need = ['baseline', 'family', 'surface', 'strategy', 'block', 'layout', 'phase', 'sigma', 'noise_seed', 'raster_key', 'peak_f32', 'image_key', 'decode_key', 'eval_key', 'sigma0_ref_key',
            'boundary', 'domain_main_hash', 'image_code', 'decode_code', 'eval_code', 'view_id', 'entry_subset']
    miss = [c for c in need if c not in M.columns]
    if miss:
        raise K.Stop('manifest', f'missing columns {miss}')
    err = []
    if M.view_id.duplicated().any():
        err.append('duplicate strategy views')
    if len(M) != K.N_VIEWS or M.image_key.nunique() != K.N_FORMAL:
        err.append(f'counts {len(M)} views / {M.image_key.nunique()} images != {K.N_VIEWS} / {K.N_FORMAL}')
    g = M.groupby('image_key').agg(r=('raster_key', 'nunique'), p=('peak_f32', 'nunique'), s=('surface', 'nunique'), sg=('sigma', 'nunique'), sd=('noise_seed', lambda x: x.astype(str).nunique()))
    if (g[['r', 'p', 's', 'sg', 'sd']] != 1).any().any():
        err.append('one image key carries different actual inputs')
    if (M.boundary != K.BOUNDARY).any():
        err.append('boundary version not the frozen one')
    dots = M[M.family == 'point']
    if (dots.sigma0_ref_key == '').any() or not set(dots.sigma0_ref_key) <= set(M[M.sigma == 0].image_key):
        err.append('sigma-0 reference missing or outside the manifest')
    if ((M.sigma == 0) & (M.noise_seed.astype(str) != '')).any() or ((M.sigma > 0) & (M.noise_seed.astype(str) == '')).any():
        err.append('seed rule violated (sigma 0 with seed or sigma > 0 without)')
    per_sigma = M.drop_duplicates('image_key').groupby(['family', 'sigma']).size()
    if not (per_sigma.xs('point', level=0) == 192).all() or not (per_sigma.xs('line', level=0) == 228).all():
        err.append(f'per-sigma unique images {per_sigma.to_dict()} != 192 / 228')
    if int(M[M.entry_subset].image_key.nunique()) != 36:
        err.append(f"entry subset {int(M[M.entry_subset].image_key.nunique())} images != 36")
    if fpd is not None:
        for k, col in (('image', 'image_code'), ('decode', 'decode_code'), ('eval', 'eval_code')):
            if (M[col] != fpd[k]).any():
                err.append(f'{k} code fingerprint differs from the frozen one')
    if err:
        raise K.Stop('manifest', '; '.join(err))
    return dict(ok=True, n_views=len(M), n_images=int(M.image_key.nunique()), n_decodes=int(M.decode_key.nunique()), n_evals=int(M.eval_key.nunique()),
                per_sigma_images={f'{a}|{b:g}': int(n) for (a, b), n in per_sigma.items()}, entry_images=int(M[M.entry_subset].image_key.nunique()), entry_views=int(M.entry_subset.sum()))


def rekey(M: pd.DataFrame, m3d: str, h1: str, hs: str) -> dict:

    import rdy_decode as DC
    import pb_m3 as PM
    bad = []
    for r in M.to_dict('records'):
        sg = float(r['sigma']); seed = None if str(r['noise_seed']) == '' else int(float(r['noise_seed']))
        ik = 'R2I-' + K.digest(dict(raster=r['raster_key'], peak_f32=float(r['peak_f32']), surface=r['platform_surface'], psf=K.RK.PSF_PX, sigma=sg, seed=seed, m3=m3d,
                                    platform=K.RK.PLATFORM_VERSION, boundary=r['boundary'], snapshot=PM.SNAPSHOT_VERSION, image_code=r['image_code']), 24)
        dk = 'R2D-' + K.digest(dict(image=ik, method=r['method'], desc=DC.METHODS[r['method']], arm=r['arm'], peak_f32=float(r['peak_f32']), window=K.RK.WINDOW_MM, decode_code=r['decode_code']), 24)
        ek = 'R2E-' + K.digest(dict(decode=dk, arm=r['arm'], eval_code=r['eval_code'], boundary=r['boundary'], omega1=h1, omega_star=hs, d=K.D_SUPPORT, tau=K.TAU, ref=r['sigma0_ref_key']), 24)
        vid = 'V-' + K.digest(dict(b=r['baseline'], s=r['strategy'], blk=r['block'], l=r['layout'], p=r['phase'], su=r['surface'], sg=sg), 16)
        if (ik, dk, ek, vid) != (r['image_key'], r['decode_key'], r['eval_key'], r['view_id']):
            bad.append(r['view_id'])
    if bad:
        raise K.Stop('manifest', f'{len(bad)} rows whose keys do not follow from their fields (first {bad[0]})')
    return dict(ok=True, n_rows=len(M))
