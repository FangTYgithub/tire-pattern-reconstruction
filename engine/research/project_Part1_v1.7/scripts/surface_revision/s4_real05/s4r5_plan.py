# Check phase-specific rasters and prepare paired layout and clipping cases.

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

import s4r5_common as H

K = H.K
import sys
for _p in (K.ROOT / 'scripts' / 'spatial_layout' / 'spat03', K.ROOT / 'scripts' / 'spatial_layout' / 'spat01r', K.ROOT / 'scripts' / 'spatial_layout'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
PHASES = ('H0', 'H1', 'H2', 'H3'); SURF = ('B_curv', 'M10_curv', 'M24_curv'); SIGMAS = (0.0, 40.0)
ALPHA_STAR, BETA = 0.80, 1.15
ARM_ORDER = ('S0', 'S1', 'S2*', 'S3', 'S4*', 'S1-RMS')


def _lid(name, j, block, cand) -> str:
    return 'regular' if block is None else f'j{j:.2f}_B{int(block):03d}_c{int(cand)}'


def layouts_of_plan() -> tuple[pd.DataFrame, pd.DataFrame]:
    P = H.read_csv(H.R4 / '_parts' / 'C1_pilot_plan.csv')
    views = []
    for name in ('R8x4', 'L6.3'):
        views.append(dict(baseline=name, j=0.0, block=None, arm='S0', cand=None, status_plan='regular', layout='regular'))
    for r in P.itertuples():
        cand = None if pd.isna(r.cand) else int(r.cand)
        views.append(dict(baseline=r.baseline, j=float(r.j), block=int(r.block), arm=r.arm, cand=cand, status_plan=r.status, layout=(None if cand is None else _lid(r.baseline, float(r.j), int(r.block), cand))))
    V = pd.DataFrame(views).astype({'block': object, 'cand': object})
    V['block'] = [None if pd.isna(x) else int(x) for x in V.block]; V['cand'] = [None if pd.isna(x) else int(x) for x in V.cand]
    V['generation_failure'] = V.status_plan.str.startswith('generation failure')
    return V, P


def _ctx(cache, name, j):
    import s3_generate as SG
    import sp2_generator as GEN
    if (name, j) not in cache:
        cache[(name, j)] = GEN.context_v2(SG.baseline(name), SG.sets()[name], j * SG.baseline(name)['p0_mm'])
    return cache[(name, j)]


def derive_layouts(V: pd.DataFrame) -> dict:

    import s3_generate as SG
    reg = json.loads(H.reg(H.R2 / 'layouts.json').read_text(encoding='utf-8'))
    cache = {}; out = {}
    for (name, lid), g in V[V.layout.notna()].groupby(['baseline', 'layout']):
        r = g.iloc[0]; bl = SG.baseline(name); ms = SG.sets()[name]; p0, h0 = float(bl['p0_mm']), float(ms['h0_mm'])
        if lid == 'regular':
            out[(name, lid)] = dict(layout_id=f'{name}:regular', disp_px={}, strategies=[('S0', None)], j=0.0, block=None, cand=None, seed=None, g_min_mm=p0, h_mm=h0, delta_rms_mm=0.0,
                                    edge_clearance_min_mm=None, general_ok=True, mask_S2star=True, mask_S3=True, mask_S4star=True, arms=['S0'], p0_mm=p0, h0_mm=h0)
            continue
        ev = SG.evaluate(name, int(r.block), int(r.cand), float(r.j), _ctx(cache, name, float(r.j)))
        e = dict(layout_id=f'{name}:{lid}', disp_px={str(k): float(v) for k, v in ev['disp_px'].items()}, strategies=[(a, f'B{int(r.block):03d}') for a in sorted(set(g.arm))], j=float(r.j),
                 block=int(r.block), cand=int(r.cand), seed=int(ev['seed']), unit_hash=ev['unit_hash'], g_min_mm=float(ev['g_min_mm']), h_mm=float(ev['h_mm']), delta_rms_mm=float(ev['delta_rms_mm']),
                 edge_clearance_min_mm=float(ev['edge_clearance_min_mm']), general_ok=bool(ev['general_ok']), general_fail=';'.join(ev['general_fail']), raster_ref_phase=ev['raster_sha256'],
                 p0_mm=p0, h0_mm=h0, arms=sorted(set(g.arm)))
        e['mask_S2star'] = bool(e['g_min_mm'] >= ALPHA_STAR * p0 - 1e-12); e['mask_S3'] = bool(e['h_mm'] <= BETA * h0 + 1e-12); e['mask_S4star'] = e['mask_S2star'] and e['mask_S3']
        e['mask_S3_violated_rms_control'] = (not e['mask_S3']) if 'S1-RMS' in e['arms'] else None
        old = reg['baselines'][name]['layouts'].get(f"B{int(r.block):03d}_c{int(r.cand)}") if abs(float(r.j) - 0.20) < 1e-9 else None
        e['s4_real02_layout'] = (f"B{int(r.block):03d}_c{int(r.cand)}" if old is not None else None)
        if old is not None:
            e['disp_equal_s4_real02'] = bool(all(abs(float(old['disp_px'][k]) - v) == 0.0 for k, v in e['disp_px'].items()) and set(old['disp_px']) == set(e['disp_px']))
        out[(name, lid)] = e
    return out


def _edge_gap_lines(pat, ids_cols):

    rng = sorted((lo, hi) for lo, hi in ids_cols)
    gaps = [rng[i + 1][0] - rng[i][1] - 1 for i in range(len(rng) - 1)]
    return min(gaps) if gaps else None


# Measure raster resources and illuminated line geometry for a candidate layout.
def raster_metrics(name: str, lay: dict, phase_id: str, B: dict) -> dict:
    import r2_layouts as L
    import rdy_inputs as RI
    import sp2_common as S2
    k = S2.mmpp(); T = L.table(name, dict(lay, strategies=lay.get('strategies', [])), phase_id)
    pat = np.asarray(T['pat']); binary = (pat > 0).astype(np.float32); lit = float(binary.sum(dtype=np.float64))
    peak = H.Q / lit; pk32 = np.float32(peak); pattern = (binary * pk32).astype(np.float32); integral = float(pattern.sum(dtype=np.float64))
    rec = dict(raster_key=RI.raster_key(binary), lit_px=lit, peak=peak, peak_f32=float(pk32), integral_f32=integral, integral_rel_err=abs(integral - H.Q) / H.Q, peak_cap_triggered=bool(peak > 1.0),
               n_elements=int(T['n_elements']), protected_present=bool(B['protected_L'] in T['present'] and B['protected_R'] in T['present']),
               movable_all_present=bool(all(m in T['present'] for m in B['movable'])))
    p0, h0 = float(B['p0_mm']), float(B['h0_mm'])
    if T['family'] == 'lines':
        import sp2_lines_table as LT2
        spec = T['spec']; fam = LT2.line_family_table(spec); clipped = {int(a): v for a, v in spec['clipped_lines'].items()}
        famx = dict(fam, ids=np.asarray(list(fam['ids']) + sorted(clipped), np.int64))
        cl = LT2.raster_centrelines_table(pat, famx)
        cu = float(spec['centre_uv'][0]); s_px = float(spec['spacing_px'])
        cen = {int(i): (float(LT2.line_q_table(fam, i)) + (cl[int(i)]['q_offset_px'] or 0.0)) for i in fam['ids'] if cl[int(i)]['q_offset_px'] is not None}
        nominal_u = {int(i): cu + float(LT2.line_q_table(fam, i)) for i in fam['ids']}
        rec['centre_residual_max_px'] = float(max(abs(cen[i] - (nominal_u[i] - cu)) for i in cen)) if cen else None
        cols = []
        lit_mask = pat > 0.5 * pat.max()
        vv, uu = np.nonzero(lit_mask); q = (uu - fam['c'][0]) * fam['n'][0] + (vv - fam['c'][1]) * fam['n'][1]
        kk, _, _, _ = LT2.nearest_table_line(fam, q)
        for i in np.unique(kk):
            sel = kk == i; cols.append((int(uu[sel].min()), int(uu[sel].max())))
        rec['edge_gap_raster_mm'] = (_edge_gap_lines(pat, cols) * k) if len(cols) > 1 else None
        rec['n_clipped_lit'] = len(clipped); rec['clipped_ids'] = ';'.join(str(i) for i in sorted(clipped))
        rec['clipped_lit_px'] = int(sum(int(v['lit_px']) for v in clipped.values()))
        rec['clipped_nominal_u'] = ';'.join(f"{cu + float(v['q_px']):.4f}" for _i, v in sorted(clipped.items()))
        rec['clipped_raster_u'] = ';'.join(f"{float(np.mean(cl[i]['u'])):.4f}" if len(cl[i]['rows']) else 'nan' for i in sorted(clipped))
        base = LT2.regular_table(S2.LINE_BASELINES[name], spec['phase_frac'])
        qt = {int(a): float(b) for a, b in spec['table']['q_px'].items()}
        near_edge = [kk_ for kk_ in base if -s_px <= cu + base[kk_] + float(lay['disp_px'].get(str(kk_), 0.0)) <= 511 + s_px]
        rec['edge_lines_unlit'] = int(sum(1 for kk_ in near_edge if kk_ not in qt))
        lo, hi = B['protected_L'], B['protected_R']
        inside = sorted(i for i in cen if lo <= i <= hi); qs = np.array([cen[i] for i in inside]); gaps = np.diff(np.sort(qs)) * k
        rec['g_min_raster_mm'] = float(gaps.min()) if len(gaps) else None; rec['g_max_raster_mm'] = float(gaps.max()) if len(gaps) else None
        rec['h_raster_mm'] = rec['g_max_raster_mm'] / 2.0 if len(gaps) else None
        rec['dot_components'] = None
    else:
        from scipy import ndimage
        from scipy.spatial import cKDTree
        lab, n = ndimage.label(pat > 0); cents = np.array(ndimage.center_of_mass(pat > 0, lab, range(1, n + 1)))[:, ::-1] if n else np.zeros((0, 2))
        c = np.asarray(T['centres'], np.float64); ix = np.asarray(T['ix'])
        rec['dot_components'] = int(n); rec['components_equal_elements'] = bool(n == len(c))
        touch = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        rec['components_touching_border'] = len(touch)
        rec['centres_outside_image'] = int(((c[:, 0] < -0.5) | (c[:, 0] > 511.5) | (c[:, 1] < -0.5) | (c[:, 1] > 511.5)).sum())
        if n:
            d, idx = cKDTree(cents).query(c); rec['centre_residual_max_px'] = float(d.max())
            colpos = {}
            for j_, i in enumerate(ix):
                colpos.setdefault(int(i), []).append(cents[idx[j_], 0])
            lo, hi = B['protected_L'], B['protected_R']
            qs = np.sort(np.array([np.mean(v) for i, v in colpos.items() if lo <= i <= hi])); gaps = np.diff(qs) * k
            p_perp = float(S2.DOT_BASELINES[name][1])
            rec['g_min_raster_mm'] = float(gaps.min()) if len(gaps) else None; rec['g_max_raster_mm'] = float(gaps.max()) if len(gaps) else None
            rec['h_raster_mm'] = 0.5 * math.sqrt(rec['g_max_raster_mm'] ** 2 + p_perp ** 2) if len(gaps) else None

            pix = [np.column_stack(np.nonzero(lab == q)[::-1]).astype(float) for q in range(1, n + 1)]
            trees = [cKDTree(p_) for p_ in pix]; _, nb = cKDTree(cents).query(cents, k=min(7, n)); dmin = np.inf
            for a in range(n):
                for b in nb[a][1:]:
                    if b > a:
                        dmin = min(dmin, float(trees[b].query(pix[a])[0].min()))
            rec['edge_gap_raster_mm'] = float((dmin - 1.0) * k) if np.isfinite(dmin) else None
        rec['n_clipped_lit'] = 0; rec['clipped_ids'] = ''; rec['clipped_lit_px'] = 0; rec['edge_lines_unlit'] = None
    rec['mask_S2star_raster'] = bool(rec['g_min_raster_mm'] is not None and rec['g_min_raster_mm'] >= ALPHA_STAR * p0 - 1e-12)
    rec['mask_S3_raster'] = bool(rec['h_raster_mm'] is not None and rec['h_raster_mm'] <= BETA * h0 + 1e-12)
    return rec


def key_context() -> dict:
    import r2_manifest as RM
    import pb_m3 as PM
    M = pd.read_csv(H.reg(H.R2 / 'manifest.csv'), low_memory=False, keep_default_na=False, float_precision='round_trip')
    fpd_frozen = dict(image=M.image_code.unique()[0], decode=M.decode_code.unique()[0], eval=M.eval_code.unique()[0])
    fp_now = RM.fingerprints(); fpd_now = {k: K.digest(v, 32) for k, v in fp_now.items()}
    return dict(M=M, seeds=RM.seeds(), m3=RM.m3_digest(), fpd=fpd_frozen, fpd_now=fpd_now, snapshot=PM.SNAPSHOT_VERSION, code_equal={k: fpd_now[k] == fpd_frozen[k] for k in fpd_frozen})


def image_key(kc: dict, raster_key: str, peak_f32: float, surface: str, sigma: float, famp: str, phase: str) -> tuple[str, object]:
    seed = None if sigma == 0 else kc['seeds'][(famp, surface, phase)]
    f = dict(raster=raster_key, peak_f32=float(peak_f32), surface=K.RK.SURF_MAP[surface], psf=K.RK.PSF_PX, sigma=sigma, seed=seed, m3=kc['m3'], platform=K.RK.PLATFORM_VERSION,
             boundary=K.BOUNDARY, snapshot=kc['snapshot'], image_code=kc['fpd']['image'])
    return 'R2I-' + K.digest(f, 24), seed


# Derive constrained layouts and assemble their shared image and strategy views.
def run() -> dict:
    t0 = time.time()
    V, _P = layouts_of_plan()
    reg = json.loads(H.reg(H.R2 / 'layouts.json').read_text(encoding='utf-8'))
    L = derive_layouts(V)
    kc = key_context(); M = kc['M']; Mi = M.drop_duplicates('image_key').set_index('image_key')
    rasters = {}; rrows = []
    for (name, lid), lay in L.items():
        B = reg['baselines'][name]
        for ph in PHASES:
            m = raster_metrics(name, lay, ph, B); rasters[(name, lid, ph)] = m
            rrows.append(dict(record_type='B_raster', baseline=name, layout=lid, phase=ph, j=lay['j'], block=lay['block'], cand=lay['cand'], arms=';'.join(lay['arms']),
                              g_min_mm=lay['g_min_mm'], h_mm=lay['h_mm'], rms_p0=lay['delta_rms_mm'] / lay['p0_mm'], edge_clearance_min_mm=lay['edge_clearance_min_mm'], general_ok=lay['general_ok'],
                              mask_S2star=lay['mask_S2star'], mask_S3=lay['mask_S3'], mask_S4star=lay['mask_S4star'], seed=lay['seed'], s4_real02_layout=lay.get('s4_real02_layout'),
                              disp_equal_s4_real02=lay.get('disp_equal_s4_real02'), **m))
    R = pd.DataFrame(rrows)
    s0 = R[R.layout == 'regular'].set_index(['baseline', 'phase'])
    R['lit_minus_S0'] = [r.lit_px - s0.loc[(r.baseline, r.phase), 'lit_px'] for r in R.itertuples()]
    R['peak_ratio_to_S0'] = [r.peak / s0.loc[(r.baseline, r.phase), 'peak'] for r in R.itertuples()]
    R['blocker'] = [('' if (r.protected_present and r.movable_all_present and (r.baseline != 'R8x4' or (r.components_equal_elements and r.components_touching_border == 0))
                            and not r.peak_cap_triggered and r.integral_rel_err <= 1e-6) else 'technical') for r in R.itertuples()]

    prow = []
    for v in V.itertuples():
        famp = 'point' if v.baseline.startswith('R') else 'line'
        for ph in PHASES:
            for s in SURF:
                for sg in SIGMAS:
                    base = dict(record_type='B_plan', baseline=v.baseline, family=famp, arm=v.arm, j=v.j, block=('' if pd.isna(v.block) else f'B{int(v.block):03d}'), cand=(None if pd.isna(v.cand) else int(v.cand)), layout=v.layout,
                                status_plan=v.status_plan, phase=ph, surface=s, platform_surface=K.RK.SURF_MAP[s], sigma=sg, psf_px=K.RK.PSF_PX, boundary=K.BOUNDARY)
                    if v.generation_failure:
                        prow.append(dict(base, action='generation_failure', image_key='', decode_binding='')); continue
                    m = rasters[(v.baseline, v.layout, ph)]
                    ik, seed = image_key(kc, m['raster_key'], m['peak_f32'], s, sg, famp, ph)
                    exists = ik in Mi.index
                    rec_ok = bool(exists and K.complete(H.R2 / 'images' / f'{ik}.json', ik))
                    old = Mi.loc[ik] if exists else None
                    clipped = bool(m['n_clipped_lit'] > 0)
                    if not rec_ok:
                        act = 'new_imaging'; db = 'pending (line clipped option not frozen)' if famp == 'line' else 'D0 as registered (key at freeze)'
                    elif famp == 'point':
                        act = 'reuse_image_and_result'; db = 'stored D0 decode / evaluation'
                    elif clipped:
                        act = 'reuse_image_redecode'; db = 'pending (line clipped option not frozen)'
                    else:
                        act = 'reuse_image_and_result'; db = 'stored P4TABLE result; valid under the clipped option only if its no-op is confirmed for this raster at binding'
                    prow.append(dict(base, raster_key=m['raster_key'], peak_f32=m['peak_f32'], noise_seed=('' if seed is None else seed), render_seed=(seed or 0), image_key=ik, image_record_ok=rec_ok,
                                     old_decode_key=(old.decode_key if exists else ''), old_eval_key=(old.eval_key if exists else ''), clipped_band_lit=clipped, action=act, decode_binding=db))
    Pn = pd.DataFrame(prow)
    Pn['view_id'] = ['V5-' + K.digest(dict(b=r.baseline, a=r.arm, j=r.j, blk=r.block, c=(None if pd.isna(r.cand) else int(r.cand)), p=r.phase, s=r.surface, sg=r.sigma), 16) for r in Pn.itertuples()]
    ok_rows = Pn[Pn.action != 'generation_failure']
    img = ok_rows.drop_duplicates('image_key')
    summ = dict(n_views=int(len(V)), n_view_rows=int(len(Pn)), n_unique_layouts=len(L), n_rasters=len(R), n_unique_raster_keys=int(R.raster_key.nunique()),
                actions_view_rows=Pn.action.value_counts().to_dict(), unique_images=int(img.image_key.nunique()),
                unique_images_by_action=img.action.value_counts().to_dict(),
                new_images_by_family=img[img.action == 'new_imaging'].groupby('family').size().to_dict(),
                generation_failure_views=int(V.generation_failure.sum()), technical_blockers=int((R.blocker != '').sum()),
                rasters_with_lit_clipped_band=int((R.n_clipped_lit > 0).sum()), lit_clipped_by_layout_phase=R[R.n_clipped_lit > 0].groupby(['baseline', 'phase']).size().to_dict(),
                peak_max=float(R.peak.max()), integral_rel_err_max=float(R.integral_rel_err.max()), raster_mask_disagreement=None,
                existing_image_keys_matched=int(img.image_record_ok.sum()), code_digest_equal_now=kc['code_equal'],
                reused_layouts_disp_equal=bool(all(v.get('disp_equal_s4_real02', True) for v in L.values())))

    dis = []
    for r in R[R.layout != 'regular'].itertuples():
        for a in r.arms.split(';'):
            need = {'S2*': ('mask_S2star', 'mask_S2star_raster'), 'S3': ('mask_S3', 'mask_S3_raster'), 'S4*': ('mask_S4star', None)}.get(a)
            if need and need[1] and getattr(r, need[0]) != getattr(r, need[1]):
                dis.append(dict(baseline=r.baseline, layout=r.layout, phase=r.phase, arm=a))
    summ['raster_mask_disagreement'] = len(dis)
    new_only = img[img.action == 'new_imaging']
    ctrl_keys = set(ok_rows[ok_rows.arm == 'S1-RMS'].image_key); main_keys = set(ok_rows[ok_rows.arm != 'S1-RMS'].image_key)
    summ['new_images_main'] = int(new_only.image_key.isin(main_keys).sum()); summ['new_images_control_extra'] = int((new_only.image_key.isin(ctrl_keys) & ~new_only.image_key.isin(main_keys)).sum())
    arm_mask = {'S2*': 'mask_S2star', 'S3': 'mask_S3', 'S4*': 'mask_S4star'}
    viol = [dict(baseline=n, layout=lid, arm=a) for (n, lid), lay in L.items() for a in lay['arms'] if a in arm_mask and not lay[arm_mask[a]]]
    viol += [dict(baseline=n, layout=lid, arm='S1-RMS') for (n, lid), lay in L.items() if 'S1-RMS' in lay['arms'] and lay['mask_S3']]
    summ['continuous_mask_violations'] = viol; summ['general_invalid_layouts'] = [lid for (n, lid), lay in L.items() if not lay['general_ok']]
    H.part('B_rasters', R); H.part('B_mask_disagreements', pd.DataFrame(dis)); H.save_csv(Pn, H.OUT / 'plan.csv')
    out = dict(summary=summ, key_context=dict(m3_digest=kc['m3'], code_digests_frozen=kc['fpd'], code_digests_now=kc['fpd_now'], snapshot=kc['snapshot'], psf=K.RK.PSF_PX,
                                               platform=K.RK.PLATFORM_VERSION, boundary=K.BOUNDARY, seeds={f'{a}|{b}|{c}': v for (a, b, c), v in kc['seeds'].items()}),
               seconds=round(time.time() - t0, 1), t_local=K.now_local())
    H.run_put('B', out); H.flush_reads('B')
    return out
