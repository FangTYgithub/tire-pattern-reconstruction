# Bind layout rasters and decoding parameters to the execution manifest.

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import s4r6_common as H

K = H.K
PILOT = [('R8x4', 'S1', 0.10, 'j0.10_B000_c0'), ('R8x4', 'S2*', 0.30, 'j0.30_B000_c7'), ('R8x4', 'S4*', 0.30, 'j0.30_B000_c25'), ('R8x4', 'S1-RMS', 0.20, 'j0.20_B000_c2'),
         ('L6.3', 'S1', 0.10, 'j0.10_B000_c0'), ('L6.3', 'S2*', 0.30, 'j0.30_B000_c39'), ('L6.3', 'S4*', 0.30, 'j0.30_B000_c134'), ('L6.3', 'S1-RMS', 0.20, 'j0.20_B000_c0')]
REFS = [('R8x4', 'S0', 0.0, 'regular'), ('L6.3', 'S0', 0.0, 'regular'), ('R8x4', 'S3', 0.20, 'j0.20_B000_c1'), ('L6.3', 'S3', 0.20, 'j0.20_B000_c63')]
PHASES, SIGMAS = ('H0', 'H1'), (0.0, 40.0)
SUBSET = dict(rule='both families, S1 j0.10 (j0.10_B000_c0), B / M10 / M24, H0 / H1, sigma 0 -> 12 planned new images, run first through the full chain')
ENTRY = [dict(role='D0 sigma 0', baseline='R8x4', layout='regular', surface='M10_curv', phase='H0', sigma=0.0, method='D0'),
         dict(role='D0 sigma 40', baseline='R8x4', layout='regular', surface='M10_curv', phase='H0', sigma=40.0, method='D0'),
         dict(role='P4 no band', baseline='L6.3', layout='regular', surface='B_curv', phase='H1', sigma=0.0, method='P4TABLE'),
         dict(role='P4 band', baseline='L6.3', layout='regular', surface='M24_curv', phase='H0', sigma=40.0, method='P4TABLE'),
         dict(role='CLIP band', baseline='L6.3', layout='regular', surface='M24_curv', phase='H0', sigma=0.0, method='CLIP'),
         dict(role='CLIP no band', baseline='L6.3', layout='regular', surface='B_curv', phase='H1', sigma=0.0, method='CLIP')]
ANALYSIS = dict(
    clip_vs_p4='line conditions: CLIP - P4TABLE on the same image and feature set, global / local; strata surface x phase x sigma x layout; M24, worst cases and ok -> no_output listed separately',
    strategy_vs_S0='same family, surface, phase, sigma and method: pilot layout - S0 regular (condition matching between layouts, not a same-image ablation)',
    rms_vs_S3='S1-RMS (j0.20) - registered S3 of block B000 (j0.20), same family / surface / phase / sigma / method, after the displacement and resource check',
    generation_link='per layout x phase: continuous g_min / h / RMS, raster g_min / h / edge gap, lit px / peak -> detection, accepted, c / w / u, valid, C_valid, C_tau, supported RMSE',
    pass_fail='state ok -> no_output (or reverse) listed per pair; no new pass threshold',
    independence='strategy views and points are not independent repetitions; per-case pairs, strata and ranges only',
    examples=dict(improvement='line condition with the largest global gross reduction P4 -> CLIP; ties by image key',
                  cost='line condition with the largest global C_tau drop P4 -> CLIP; ties by image key',
                  control='same family / layout / surface / sigma as the improvement example at H1 (no band)'))


def _plan() -> pd.DataFrame:
    return pd.read_csv(H.reg(H.R5 / 'plan.csv'), low_memory=False, keep_default_na=False, float_precision='round_trip')


def _r5_clip_records() -> dict:

    out = {}
    for p in sorted((H.R5 / 'decodes').glob('R5D-*.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        if r.get('option') == 'abstain' and not r.get('minimal_input') and H.complete(p, r['identity']):
            out[r['image_key']] = dict(identity=r['identity'], npz_sha256=r['npz_sha256'], path=K.rel(p))
    return out


def layouts() -> dict:

    import s4r5_plan as PL
    reg = json.loads(H.reg(H.R2 / 'layouts.json').read_text(encoding='utf-8')); out = {}
    need = [(b, a, j, l) for b, a, j, l in PILOT + REFS]
    rows = []
    for b, a, j, l in need:
        if l == 'regular':
            out[(b, l)] = dict(reg['baselines'][b]['layouts']['regular'], j=0.0); continue
        blk, cand = int(l.split('_B')[1].split('_c')[0]), int(l.split('_c')[1])
        old = reg['baselines'][b]['layouts'].get(f'B{blk:03d}_c{cand}') if abs(j - 0.20) < 1e-9 else None
        if old is not None:
            out[(b, l)] = dict(old, j=0.20); continue
        rows.append(dict(baseline=b, layout=l, j=j, block=blk, cand=cand, arm=a))
    if rows:
        V = pd.DataFrame(rows); V['block'] = V.block.astype(object); V['cand'] = V.cand.astype(object)
        for (b, l), lay in PL.derive_layouts(V).items():
            out[(b, l)] = dict(lay, strategies=lay.get('strategies', []))
    return out


# Bind layout rasters, decoder settings and reusable inputs to execution rows.
def bind() -> dict:
    t0 = time.time(); P = _plan(); R2M = pd.read_csv(H.reg(H.R2 / 'manifest.csv'), low_memory=False, keep_default_na=False, float_precision='round_trip')
    r2u = R2M.drop_duplicates('image_key').set_index('image_key')
    r5c = _r5_clip_records(); rows = []
    for setname, L in (('pilot', PILOT), ('reference', REFS)):
        for b, arm, j, lay in L:
            x = P[(P.baseline == b) & (P.layout == lay) & (P.arm == arm) & P.phase.isin(PHASES)]
            if len(x) != 12:
                raise H.Stop('plan', f'{b} {lay} {arm}: {len(x)} plan rows (12 expected)')
            for r in x.itertuples():
                sg = float(r.sigma); seed = None if sg == 0.0 else int(float(r.noise_seed))
                rseed = 0 if seed is None else seed
                if int(float(r.render_seed)) != rseed:
                    raise H.Stop('plan', f'{r.image_key}: render seed {r.render_seed} != {rseed}')
                new = r.action == 'new_imaging'
                rec = dict(set=setname, cond_id=r.image_key, image_key=r.image_key, baseline=b, family=r.family, strategy_view=arm, layout=lay, j=float(r.j), block=r.block or '', cand=r.cand,
                           phase=r.phase, surface=r.surface, platform_surface=r.platform_surface, sigma=sg, noise_seed=('' if seed is None else seed), render_seed=rseed, psf_px=float(r.psf_px),
                           boundary=r.boundary, raster_key=r.raster_key, peak_f32=float(r.peak_f32), clipped_band_lit=(str(r.clipped_band_lit) == 'True'), plan_action=r.action,
                           image_source=('new' if new else 'S4-REAL-02'))
                if not new:
                    ip = H.R2 / 'images' / f'{r.image_key}.json'
                    if not K.complete(ip, r.image_key):
                        raise H.Stop('reference', f'{r.image_key}: stored image record incomplete or hash mismatch')
                    ir = json.loads(H.reg(ip).read_text(encoding='utf-8'))
                    if ir['raster_key'] != r.raster_key or abs(float(np.float32(ir['peak'])) - float(r.peak_f32)) > 0:
                        raise H.Stop('reference', f'{r.image_key}: image record raster / peak differ from the plan')
                    rec.update(image_record=K.rel(ip), image_npz_sha=ir['npz_sha256'], old_decode_key=r2u.loc[r.image_key, 'decode_key'], old_eval_key=r2u.loc[r.image_key, 'eval_key'])
                rows.append(rec)
    M = pd.DataFrame(rows)
    if M.image_key.duplicated().any():
        raise H.Stop('plan', 'duplicate image condition')

    res = []
    for c in M.itertuples():
        methods = ['D0'] if c.family == 'point' else ['P4TABLE', 'CLIP']
        for m in methods:
            if c.image_source == 'new':
                src = 'new_decode'; purpose = {'D0': 'science_dot', 'P4TABLE': 'science_p4', 'CLIP': 'science_clip'}[m]; ref = ''
            elif m in ('D0', 'P4TABLE'):
                rp = H.R2 / 'raw' / f'{c.old_decode_key}.json'
                if not K.complete(rp, c.old_decode_key):
                    raise H.Stop('reference', f'{c.old_decode_key}: stored decode incomplete')
                src = 'reuse_S4-REAL-02'; purpose = ''; ref = c.old_decode_key
            elif c.image_key in r5c:
                src = 'reuse_S4-REAL-05'; purpose = ''; ref = r5c[c.image_key]['identity']
            else:
                src = 'new_decode'; purpose = 'science_clip'; ref = ''
            res.append(dict(cond_id=c.cond_id, method=m, result_source=src, decode_purpose=purpose, reuse_ref=ref))
    Rm = pd.DataFrame(res)

    BR = pd.read_csv(H.reg(H.R5 / '_parts' / 'B_rasters.csv'), low_memory=False)
    cmp = []
    for b, rms, s3 in (('R8x4', 'j0.20_B000_c2', 'j0.20_B000_c1'), ('L6.3', 'j0.20_B000_c0', 'j0.20_B000_c63')):
        for ph in PHASES:
            a = BR[(BR.baseline == b) & (BR.layout == rms) & (BR.phase == ph)].iloc[0]; s = BR[(BR.baseline == b) & (BR.layout == s3) & (BR.phase == ph)].iloc[0]
            cmp.append(dict(record_type='B_rms_vs_S3', baseline=b, phase=ph, rms_layout=rms, s3_layout=s3, rms_p0=[a.rms_p0, s.rms_p0], d_rms_p0=a.rms_p0 - s.rms_p0, h_mm=[a.h_mm, s.h_mm],
                            h_raster_mm=[a.h_raster_mm, s.h_raster_mm], g_min_mm=[a.g_min_mm, s.g_min_mm], g_min_raster_mm=[a.g_min_raster_mm, s.g_min_raster_mm], lit_px=[a.lit_px, s.lit_px],
                            peak=[a.peak, s.peak], mask_S3=[bool(a.mask_S3), bool(s.mask_S3)], edge_gap_raster_mm=[a.edge_gap_raster_mm, s.edge_gap_raster_mm]))
    H.save_csv(M, H.OUT / 'manifest.csv'); H.part('B_results_plan', Rm); H.part('B_rms_vs_S3', pd.DataFrame(cmp))
    summ = dict(n_conditions=len(M), by_set=M.groupby('set').size().to_dict(), images_new=int((M.image_source == 'new').sum()), images_existing=int((M.image_source != 'new').sum()),
                new_by_family=M[M.image_source == 'new'].groupby('family').size().to_dict(), band_conditions=int(M.clipped_band_lit.sum()),
                band_new=int((M.clipped_band_lit & (M.image_source == 'new')).sum()), method_results=len(Rm), results_by_source=Rm.groupby(['method', 'result_source']).size().to_dict(),
                new_decodes_by_purpose=Rm[Rm.result_source == 'new_decode'].groupby('decode_purpose').size().to_dict(), entry_regressions=len(ENTRY),
                budget_plan=dict(renders=int((M.image_source == 'new').sum()), render_cap=H.CAP_RENDER, decodes_science=int((Rm.result_source == 'new_decode').sum()), decodes_entry=len(ENTRY),
                                 decode_cap=H.CAP_DECODE, purpose_caps=H.DECODE_PURPOSE),
                rms_vs_s3=cmp, seconds=round(time.time() - t0, 1), t_local=K.now_local())
    ok = (summ['n_conditions'] == 144 and summ['images_new'] == 84 and summ['images_existing'] == 60 and summ['new_by_family'] == {'line': 36, 'point': 48} and summ['method_results'] == 216
          and summ['band_conditions'] == 36 and summ['band_new'] == 18)
    summ['counts_ok'] = bool(ok)
    H.run_put('B', summ); H.flush_reads('B')
    return summ


def freeze() -> dict:
    import r2_manifest as RM
    import s4r6_eval as EV
    r = H.run_load()
    if not r.get('B', {}).get('counts_ok'):
        raise H.Stop('plan', 'binding counts not closed')
    for s in ('A1', 'A2', 'A3'):
        if s not in r:
            raise H.Stop('plan', f'work {s} missing')
    fp = {k: K.digest(v, 32) for k, v in RM.fingerprints().items()}
    M2 = pd.read_csv(H.R2 / 'manifest.csv', low_memory=False, keep_default_na=False)
    if fp['image'] != M2.image_code.unique()[0]:
        raise H.Stop('binding', 'forward-code digest differs from the S4-REAL-02 frozen digest (image keys would not follow)')
    cfg = dict(manifest=dict(file='outputs/surface_revision/s4_real06/manifest.csv', sha256=K.sha_file(H.OUT / 'manifest.csv')),
               results_plan=dict(file='outputs/surface_revision/s4_real06/_parts/B_results_plan.csv', sha256=K.sha_file(H.OUT / '_parts' / 'B_results_plan.csv')),
               forward=dict(rule='S4-REAL-02 image key (r2_manifest)', image_code_digest=fp['image'], entry='pb_m3.render(platform sid, pattern, peak, sigma, render_seed, PSF 1.0, symmetric_footprint)',
                            sigma0_seed='noise seed empty -> render seed 0 (explicit int, never int(NaN))'),
               methods=dict(D0='rdy_decode.decode D0 (section S4, R_MM)', P4TABLE='rdy_decode.decode P4TABLE (historical baseline)', CLIP='s4r5_clip.run_p4_table_clip option abstain (candidate)'),
               decode_deps={m: H.code_hashes(v) for m, v in H.DECODE_DEPS.items()}, eval_deps=EV.eval_deps(), eval_rule=EV.IDENT_RULE, gross_mm=H.GROSS_MM, d=K.D_SUPPORT, tau=K.TAU,
               entry=ENTRY, subset=SUBSET, analysis=ANALYSIS, budgets=dict(render=dict(cap=H.CAP_RENDER, **H.RENDER_PURPOSE), decode=dict(cap=H.CAP_DECODE, **H.DECODE_PURPOSE)),
               own_code=H.own_code(), t_local=K.now_local())
    H.run_put('config', cfg, freeze=True)
    return dict(frozen=True, manifest_sha=cfg['manifest']['sha256'])
