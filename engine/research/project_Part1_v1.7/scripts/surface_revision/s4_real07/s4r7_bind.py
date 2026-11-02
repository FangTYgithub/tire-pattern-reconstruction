# Bind held-out layout inputs, phases and seeds to image cases.

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

import s4r7_common as H
import s4r7_phase as PH

K = H.K
MAIN_PHASES = ('R0', 'R1'); ANCHOR_PHASE = 'H0'; ANCHOR_BLOCKS = (24, 25); SIGMAS = (0.0, 40.0)
FAMILY = {'R8x4': 'point', 'L6.3': 'line'}
EXAMPLES = dict(dot_m10='main validation, dots, M10: largest C_tau drop of a strategy layout vs S0 of the same phase / sigma (ties: image key); shown with its S0',
                line_improvement='lines, every line condition of this round (main R0 / R1 and H0 anchor; the R0 / R1 rasters light no clipped band, known from the raster precheck '
                                 'before imaging, so CLIP is a no-op there by construction): largest global gross reduction P4 -> CLIP (ties: image key); set named; if none, the largest change, named as such',
                line_cost='lines, same pool: largest global C_tau drop P4 -> CLIP (ties: image key); if none drops, the smallest change, named as such')
ANALYSIS = dict(protection='same image P4 -> CLIP per surface / phase / sigma / layout; unconfirmable, gross, union and intersection, correct-point loss, new risk, dC_tau, dC_valid, three errors;'
                           ' main and H0 anchor separate; M24 and local domain never offset',
                strategy='per family: S1 / S2* / S3 / S4* - S0 (same phase / surface / sigma / method), and within block S2*-S1, S3-S1, S4*-S2*, S4*-S3; primary arms dots D0, lines CLIP;'
                         ' lines P4 direction listed alongside',
                blocks='per block: mean of the two phases, then median / min / max over the 6 blocks and counts of positive / zero / negative blocks (zero tolerance 1e-12)',
                chain='realised continuous / raster spacing, hole, RMS, resources -> detection -> dot ambiguity / duplicates, line unknown / abstain -> accepted / valid -> support / accurate coverage',
                independence='points, shared S0, both methods and strategy views are not independent samples; report blocks, unique layouts, phases, noise realisations, unique images')


def _gen() -> dict:
    return json.loads(H.reg(H.OUT / 'generation.json').read_text(encoding='utf-8'))


def views() -> pd.DataFrame:
    g = _gen(); rows = []
    for b in g['blocks_out']:
        for a in ('S1', 'S2*', 'S3', 'S4*'):
            x = b['arms'][a]
            rows.append(dict(baseline=b['baseline'], family=FAMILY[b['baseline']], block=f"B{b['block']:03d}", arm=a, status=x['status'],
                             layout=(f"j0.30_B{b['block']:03d}_c{x['cand']}" if x['status'] == 'selected' else ''), cand=x.get('cand'), denominator=x.get('denominator'),
                             n_evaluated=b['n_evaluated'], g_min_mm=x.get('g_min_mm'), h_mm=x.get('h_mm'), rms_p0=(x['rms_mm'] / b['p0_mm'] if x['status'] == 'selected' else None)))
    return pd.DataFrame(rows)


def layouts() -> dict:
    reg = json.loads(H.reg(H.R2 / 'layouts.json').read_text(encoding='utf-8')); g = _gen(); out = {}
    for n in ('R8x4', 'L6.3'):
        out[(n, 'regular')] = dict(reg['baselines'][n]['layouts']['regular'], j=0.0, strategies=[('S0', None)])
    for b in g['blocks_out']:
        for a in ('S1', 'S2*', 'S3', 'S4*'):
            x = b['arms'][a]
            if x['status'] != 'selected':
                continue
            lid = f"j0.30_B{b['block']:03d}_c{x['cand']}"
            e = out.setdefault((b['baseline'], lid), dict(layout_id=f"{b['baseline']}:{lid}", disp_px=x['disp_px'], j=0.30, block=b['block'], cand=x['cand'], seed=x['seed'],
                                                          g_min_mm=x['g_min_mm'], h_mm=x['h_mm'], delta_rms_mm=x['rms_mm'], strategies=[]))
            e['strategies'].append((a, f"B{b['block']:03d}"))
    return out


def _discrete(name: str, T: dict, B: dict) -> dict:
    import sp2_common as S2
    k = S2.mmpp(); pat = np.asarray(T['pat'])
    if T['family'] == 'lines':
        import sp2_lines_table as LT2
        fam = LT2.line_family_table(T['spec']); cl = LT2.raster_centrelines_table(pat, fam)
        cen = {int(i): float(LT2.line_q_table(fam, i)) + cl[int(i)]['q_offset_px'] for i in fam['ids'] if cl[int(i)]['q_offset_px'] is not None}
        qs = np.sort([cen[i] for i in cen if B['protected_L'] <= i <= B['protected_R']]); gaps = np.diff(qs) * k
        return dict(g_min_raster_mm=float(gaps.min()) if len(gaps) else None, h_raster_mm=float(gaps.max() / 2) if len(gaps) else None, n_clipped_lit=len(T['spec']['clipped_lines']),
                    clipped_ids=';'.join(str(x) for x in sorted(int(a) for a in T['spec']['clipped_lines'])))
    from scipy import ndimage
    from scipy.spatial import cKDTree
    lab, n = ndimage.label(pat > 0); c = np.asarray(T['centres']); ix = np.asarray(T['ix'])
    cents = np.array(ndimage.center_of_mass(pat > 0, lab, range(1, n + 1)))[:, ::-1] if n else np.zeros((0, 2))
    touch = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    out = dict(components=int(n), components_equal_elements=bool(n == len(c)), components_touching_border=len(touch), n_clipped_lit=0, clipped_ids='')
    if n:
        _, idx = cKDTree(cents).query(c); col = {}
        for j_, i in enumerate(ix):
            col.setdefault(int(i), []).append(cents[idx[j_], 0])
        qs = np.sort([np.mean(v) for i, v in col.items() if B['protected_L'] <= i <= B['protected_R']]); gaps = np.diff(qs) * k
        p_perp = float(S2.DOT_BASELINES[name][1])
        out.update(g_min_raster_mm=float(gaps.min()), h_raster_mm=float(0.5 * math.sqrt(gaps.max() ** 2 + p_perp ** 2)))
    return out


def seeds() -> dict:
    import r2_manifest as RM
    reg_seeds = RM.seeds(); out = {}
    for n, fam in FAMILY.items():
        for si, s in enumerate(H.SURF):
            for pi, p in enumerate(MAIN_PHASES):
                out[(fam, s, p)] = int(H.NOISE_SEED_BASE_R7 + (0 if fam == 'point' else 1000) + 100 * si + 10 * pi + 1)
            out[(fam, s, ANCHOR_PHASE)] = int(reg_seeds[(fam, s, ANCHOR_PHASE)])
    return out


# Bind held-out layouts, phase offsets and noise seeds to image cases.
def bind() -> dict:
    import r2_manifest as RM
    import pb_m3 as PM
    t0 = time.time(); reg = json.loads(H.reg(H.R2 / 'layouts.json').read_text(encoding='utf-8')); L = layouts(); V = views()
    M2 = pd.read_csv(H.reg(H.R2 / 'manifest.csv'), low_memory=False, keep_default_na=False, float_precision='round_trip')
    fpd = M2.image_code.unique()[0]; fp_now = K.digest(RM.fingerprints()['image'], 32); m3d = RM.m3_digest(); sd = seeds()
    if fp_now != fpd:
        raise H.Stop('binding', 'forward-code digest differs from the S4-REAL-02 digest; image keys would not follow')

    need = []
    for (n, lid), lay in L.items():
        for p in MAIN_PHASES:
            need.append((n, lid, p, 'main'))
        if FAMILY[n] == 'line' and (lid == 'regular' or lay.get('block') in ANCHOR_BLOCKS):
            need.append((n, lid, ANCHOR_PHASE, 'anchor'))
    rast = {}; rrows = []
    for n, lid, p, setname in need:
        lay = L[(n, lid)]; B = reg['baselines'][n]
        try:
            inp = PH.build_input(n, lay, p); T = inp['table']; err = None
        except Exception as e:
            inp = None; T = None; err = f'{type(e).__name__}: {e}'
        rec = dict(record_type='raster', baseline=n, family=FAMILY[n], layout=lid, phase=p, set=setname, adapter_error=err)
        if inp is not None:
            rec.update(raster_key=inp['raster_key'], lit_px=inp['lit_px'], peak=inp['peak'], peak_f32=float(np.float32(inp['peak'])), integral_f32=inp['integral_f32'], integral_rel_err=inp['integral_rel_err'],
                       resource_feasible=inp['resource_feasible'], n_elements=int(T['n_elements']), protected_present=bool(B['protected_L'] in T['present'] and B['protected_R'] in T['present']),
                       movable_all_present=bool(all(m in T['present'] for m in B['movable'])), boundary_in=T.get('boundary_in', 0), boundary_out=T.get('boundary_out', 0),
                       g_min_mm=lay.get('g_min_mm'), h_mm=lay.get('h_mm'), rms_p0=(lay.get('delta_rms_mm', 0.0) / float(B['p0_mm'])), **_discrete(n, T, B))
            rast[(n, lid, p)] = inp
        rrows.append(rec)
    R = pd.DataFrame(rrows)
    s0n = R[R.layout == 'regular'].set_index(['baseline', 'phase']).n_elements
    R['elements_equal_S0'] = [bool(r.n_elements == s0n.loc[(r.baseline, r.phase)]) if pd.notna(r.n_elements) else False for r in R.itertuples()]
    R['general_valid_phase'] = [bool(pd.isna(r.adapter_error) or r.adapter_error is None) and bool(r.protected_present) and bool(r.movable_all_present) and bool(r.elements_equal_S0)
                                and (r.family == 'line' or (int(r.boundary_in) == 0 and int(r.boundary_out) == 0 and bool(r.components_equal_elements) and int(r.components_touching_border) == 0))
                                for r in R.itertuples()]
    alpha_p0 = {n: 0.80 * float(reg['baselines'][n]['p0_mm']) for n in FAMILY}; beta_h0 = {n: 1.15 * float(reg['baselines'][n]['h0_mm']) for n in FAMILY}
    R['raster_s2_ok'] = [bool(r.g_min_raster_mm >= alpha_p0[r.baseline] - 1e-12) if pd.notna(r.g_min_raster_mm) else None for r in R.itertuples()]
    R['raster_s3_ok'] = [bool(r.h_raster_mm <= beta_h0[r.baseline] + 1e-12) if pd.notna(r.h_raster_mm) else None for r in R.itertuples()]
    blocked = R[~R.general_valid_phase | ~R.resource_feasible.fillna(False).astype(bool)]

    r6m = pd.read_csv(H.reg(H.R6 / 'manifest.csv'), low_memory=False, keep_default_na=False, float_precision='round_trip')
    rows = []
    for (n, lid, p), inp in rast.items():
        if not R[(R.baseline == n) & (R.layout == lid) & (R.phase == p)].general_valid_phase.iloc[0]:
            continue
        fam = FAMILY[n]; setname = 'anchor' if p == ANCHOR_PHASE else 'main'; pv = PH.ph('dots' if fam == 'point' else 'lines', p)
        for s in H.SURF:
            for sg in SIGMAS:
                seed = None if sg == 0.0 else sd[(fam, s, p)]
                f = dict(raster=inp['raster_key'], peak_f32=float(np.float32(inp['peak'])), surface=K.RK.SURF_MAP[s], psf=K.RK.PSF_PX, sigma=sg, seed=seed, m3=m3d, platform=K.RK.PLATFORM_VERSION,
                         boundary=K.BOUNDARY, snapshot=PM.SNAPSHOT_VERSION, image_code=fpd)
                ik = 'R2I-' + K.digest(f, 24)
                rec = dict(image_key=ik, set=setname, baseline=n, family=fam, layout=lid, j=L[(n, lid)].get('j', 0.0), block=(f"B{L[(n, lid)]['block']:03d}" if lid != 'regular' else ''),
                           cand=L[(n, lid)].get('cand'), phase=p, phase_u=pv.get('u'), phase_v=pv.get('v'), phase_frac=pv.get('phase_frac'), phase_source=pv['source'], surface=s,
                           platform_surface=K.RK.SURF_MAP[s], sigma=sg, noise_seed=('' if seed is None else seed), render_seed=(0 if seed is None else seed), psf_px=K.RK.PSF_PX,
                           boundary=K.BOUNDARY, raster_key=inp['raster_key'], peak_f32=float(np.float32(inp['peak'])), lit_px=inp['lit_px'],
                           clipped_band_lit=bool(fam == 'line' and len(inp['table']['spec']['clipped_lines']) > 0), image_source='new')
                if lid == 'regular' and p == ANCHOR_PHASE:
                    x = r6m[r6m.image_key == ik]
                    if len(x) != 1:
                        raise H.Stop('reference', f'H0 S0 {s} sigma {sg}: no S4-REAL-06 condition with key {ik}')
                    ip = H.R2 / 'images' / f'{ik}.json'
                    if not K.complete(ip, ik):
                        raise H.Stop('reference', f'{ik}: stored image record incomplete')
                    rec.update(image_source='S4-REAL-02 (via S4-REAL-06)', old_decode_key=x.old_decode_key.iloc[0], image_npz_sha=json.loads(ip.read_text(encoding='utf-8'))['npz_sha256'])
                rows.append(rec)
    M = pd.DataFrame(rows)
    if M.image_key.duplicated().any():
        raise H.Stop('plan', 'duplicate image keys in the condition manifest')

    vrows = []
    for v in V.itertuples():
        for p in MAIN_PHASES + ((ANCHOR_PHASE,) if (v.family == 'line' and int(v.block[1:]) in ANCHOR_BLOCKS) else ()):
            if v.status != 'selected':
                vrows.append(dict(baseline=v.baseline, block=v.block, arm=v.arm, phase=p, status='generation_failure', layout='', denominator=v.denominator)); continue
            vrows.append(dict(baseline=v.baseline, block=v.block, arm=v.arm, phase=p, status='selected', layout=v.layout, denominator=None))
    VV = pd.DataFrame(vrows)

    r6p = pd.read_csv(H.reg(H.R6 / '_parts' / 'B_results_plan.csv'), low_memory=False, keep_default_na=False)
    res = []
    for c in M.itertuples():
        for m in (['D0'] if c.family == 'point' else ['P4TABLE', 'CLIP']):
            if c.image_source == 'new':
                res.append(dict(cond_id=c.image_key, method=m, result_source='new_decode', reuse_ref=''))
            else:
                x = r6p[(r6p.cond_id == c.image_key) & (r6p.method == m)].iloc[0]
                res.append(dict(cond_id=c.image_key, method=m, result_source=x.result_source, reuse_ref=x.reuse_ref))
    Rp = pd.DataFrame(res)
    n_new = int((M.image_source == 'new').sum()); n_dec = int((Rp.result_source == 'new_decode').sum())
    summ = dict(n_conditions=len(M), by_set=M.groupby(['set', 'family']).size().to_dict(), images_new=n_new, images_reused=int((M.image_source != 'new').sum()),
                unique_layouts=M.groupby('family').layout.nunique().to_dict(), method_results=len(Rp), new_decodes=n_dec, results_by_source=Rp.groupby(['method', 'result_source']).size().to_dict(),
                generation_failures=V[V.status != 'selected'][['baseline', 'block', 'arm', 'denominator']].to_dict('records'),
                overlaps=V[V.status == 'selected'].groupby(['baseline', 'block', 'layout']).arm.apply(lambda s: ';'.join(s)).loc[lambda s: s.str.contains(';')].reset_index().to_dict('records'),
                band_conditions=int(M.clipped_band_lit.sum()), rasters=len(R), raster_blocked=blocked[['baseline', 'layout', 'phase', 'adapter_error']].to_dict('records'),
                raster_deviation_s2=R[(R.raster_s2_ok == False)].shape[0], raster_deviation_s3=R[(R.raster_s3_ok == False)].shape[0],
                budget=dict(renders=n_new, render_cap=H.CAP_RENDER, render_science=H.RENDER_PURPOSE['science'], decodes=n_dec, decode_science=H.DECODE_PURPOSE['science'], decode_cap=H.CAP_DECODE),
                seeds={f'{a}|{b}|{c}': v for (a, b, c), v in sd.items()}, seconds=round(time.time() - t0, 1), t_local=K.now_local())
    summ['budget_ok'] = bool(n_new <= H.RENDER_PURPOSE['science'] and n_dec <= H.DECODE_PURPOSE['science'])
    H.save_csv(M, H.OUT / 'manifest.csv'); H.part('results_plan', Rp); H.part('views', VV); H.part('rasters', R)
    H.run_put('B', summ); H.flush_reads('B')
    if not summ['budget_ok']:
        raise H.Stop('budget', f'planned renders {n_new} / decodes {n_dec} exceed the science caps')
    if len(blocked):
        raise H.Stop('precheck', f'{len(blocked)} rasters not generally valid / not resource-feasible at their actual phase: {summ["raster_blocked"][:4]}')
    return summ


def data_digests() -> dict:
    import r2_eval as E
    import rdy_platform as P
    import sp_regression as R1
    import s4r7_eval as EV
    import sp2_common as S2
    return dict(omega1=E.domain('omega1')['hash'], local={s: P.dom_local(s)['hash'] for s in H.SURF}, mesh={s: EV.mesh_hash(s) for s in H.SURF},
                sensor_cfg=K.digest(P.s2().line_cfg(0.3), 32), p4_params=K.digest(R1.p4_params(), 32), d0_params=K.digest(dict(section='S4', radius_mm=float(S2.R_MM)), 32),
                phases=K.digest(PH.phase_table(), 32), seeds=K.digest({f'{a}|{b}|{c}': v for (a, b, c), v in seeds().items()}, 32),
                gross_mm=EV.gross_mm(), d=K.D_SUPPORT, tau=K.TAU)


def refreeze(reason: str) -> dict:

    r = H.run_load()
    if 'config' not in r.get('frozen', {}):
        raise H.Stop('freeze', 'nothing frozen')
    rb, db = H.render_budget().state(), H.decode_budget().state()
    r.setdefault('config_history', []).append(dict(config=r['config'], frozen=r['frozen'].pop('config'), superseded=K.now_local(), reason=reason, render_budget=rb, decode_budget=db))
    r.setdefault('deviations', []).append(dict(what='re-freeze', reason=reason, renders_science_used=rb.get('by_purpose', {}).get('science', 0),
                                               decodes_science_used=db.get('by_purpose', {}).get('science', 0), t_local=K.now_local()))
    r.pop('config'); H.save_json(H.RUN, r)
    return freeze()


def freeze() -> dict:
    import s4r7_eval as EV
    r = H.run_load()
    for s in ('A1', 'generation', 'B'):
        if s not in r:
            raise H.Stop('plan', f'{s} missing before the freeze')
    dd = data_digests()
    cfg = dict(snapshot=H.science_snapshot(data_digests=dd), methods=dict(D0='rdy_decode.decode D0 (section S4)', P4TABLE='rdy_decode.decode P4TABLE (same-image method reference)',
                                                                    CLIP='s4r5_clip.run_p4_table_clip option abstain (pre-specified conservative primary arm for lines)'),
               primary=dict(dots='D0', lines='CLIP'), eval_rule=EV.IDENT_RULE, examples=EXAMPLES, analysis=ANALYSIS,
               subset='main: S0 and B024-S1 at sigma 0, both families, R0 / R1, three surfaces (<= 24 images); anchor: S0 / B024-S1 at H0 sigma 0, three surfaces (6 conditions, 3 old S0)',
               budgets=dict(render=dict(cap=H.CAP_RENDER, **H.RENDER_PURPOSE), decode=dict(cap=H.CAP_DECODE, **H.DECODE_PURPOSE)),
               display=H.code_hashes(H.DISPLAY_DEPS), t_local=K.now_local())
    H.run_put('config', cfg, freeze=True)
    H.freeze_check('freeze', data_digests=dd)
    return dict(frozen=True, science_files=cfg['snapshot']['science_files'])
