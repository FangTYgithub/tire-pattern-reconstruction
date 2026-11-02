# Construct spacing-layout cases and pairing keys.

from __future__ import annotations

import numpy as np
import pandas as pd

import s2p_common as K

K.setup()
CSV = K.OUT / 's2_candidates.csv'
CONFIG = K.OUT / 's2_config.json'
DOT_P_PAR = ['R6x4', 'R7.35x4', 'R8x4', 'R10x4', 'R12x4']
DOT_P_PERP = ['R8x2.5', 'R8x3', 'R8x4', 'R8x6', 'R8x8']
LINE_PITCHES = [4.45, 5.0, 5.6, 6.3, 8.0, 10.0, 12.0]
FIRST_POS = 'pos33'
FIRST_PH = 'ph0'


def source_rows() -> pd.DataFrame:
    d = pd.read_csv(K.CANDS, keep_default_na=False, low_memory=False)
    d = d[d.section == 'S2'].copy()
    d['sigma'] = d.sigma.astype(float)
    dots = d[(d.family == 'point') & d.source.str.contains('spacing_layout/design/case_list.csv') & d.sigma.isin([0.0, 40.0])]
    lines = d[(d.family == 'line') & (d.decode_arm == 'P4_C3_cal') & d.sigma.isin([0.0, 40.0])]
    out = pd.concat([dots, lines])
    out['tier'] = np.where(out.regime == 'equal_integral', 'core', 'optional')
    return out


def _raster(row) -> dict:
    import adp_patterns as PT
    import s1r_common as S1K
    b = PT.build(dict(row, brightness_rule='peak=1.0', raster_sha256_registered=''))
    binary = np.asarray(b['binary']) > 0
    return dict(binary=binary, centres=b['centres'], meta=b['meta_full'], key=S1K.raster_key(binary))


def _geometry(fam, r, row) -> dict:
    import sn_lib as SL
    mmpp = SL.mmpp()
    if fam == 'point':
        c = np.asarray(r['centres'], np.float64)
        rows_v = np.unique(np.round(c[:, 1], 6)); du = []
        for v in rows_v[:8]:
            u = np.sort(c[np.isclose(c[:, 1], v), 0]); du += list(np.diff(u))
        p_par = float(np.median(du)) * mmpp if du else None; p_perp = float(np.median(np.diff(rows_v))) * mmpp if len(rows_v) > 1 else None
        u0 = np.sort(c[np.isclose(c[:, 1], rows_v[0]), 0])[0]; u1 = np.sort(c[np.isclose(c[:, 1], rows_v[1]), 0])[0] if len(rows_v) > 1 else u0
        stag = float(((u1 - u0) * mmpp / p_par) % 1.0) if p_par else None
        return dict(n_features=int(len(c)), pitch_epipolar_mm=p_par, pitch_rows_mm=p_perp, row_offset_frac=stag)
    ol = r['meta']['olines']; s_mm = float(ol['spacing_px']) * mmpp; phi = float(row['orientation_deg'])
    return dict(n_features=int(ol['n_lines']), pitch_normal_mm=s_mm, pitch_epipolar_mm=(s_mm / abs(np.cos(np.radians(phi))) if abs(np.cos(np.radians(phi))) > 1e-9 else None), orientation_deg=phi)


def platform_ids() -> dict:

    import s1r_common as S1K
    S = S1K.s2()
    m3 = {k: v for k, v in S.m3_cfg(0.0, 0).items() if k not in ('noise_sigma', 'mottle_seed')}
    rec = K.load_json(K.REL / 'release_surf02.json')
    return dict(version=rec['version'], m3=K.digest(m3), dot_cfg=K.digest({k: v for k, v in S.dot_cfg().items() if isinstance(v, (int, float, str, bool, list, tuple)) or v is None}),
                p4=K.digest(S.P4), s1r_eval=K.sha_file(K.ADP / 's1r_eval.py'), global_domain=K.load_json(K.S1OUT / 'domain' / 'global_domain_step0p50.json')['hash'])


def build():
    src = source_rows()
    P = platform_ids()
    rasters = {}; budget = {}
    keyrows = []
    for _, r in src.iterrows():
        gk = (r.family, r.layout_id if r.family == 'point' else (float(r.line_pitch_mm), float(r.orientation_deg)), r.phase_label, (r.phase_u, r.phase_v, r.phase_frac))
        if gk not in rasters:
            rasters[gk] = _raster(r.to_dict())
        R = rasters[gk]; I0 = float(R['binary'].sum(dtype=np.float64))
        if R['key'] not in budget:
            peak = K.Q / I0; pf = np.float32(peak)
            budget[R['key']] = dict(raster_key=R['key'], family=r.family, layout=(r.layout_id if r.family == 'point' else f"s{float(r.line_pitch_mm):g}_phi{float(r.orientation_deg):g}"), phase_label=r.phase_label,
                                    integral_I0=I0, lit_px=int(R['binary'].sum()), peak_equal_integral=peak, peak_f32=float(pf), feasible=bool(peak <= 1.0 + 1e-12),
                                    integral_actual=float((R['binary'].astype(np.float32) * pf).sum(dtype=np.float64)), **_geometry(r.family, R, r))
        b = budget[R['key']]
        peak = 1.0 if r.regime == 'fixed_peak' else b['peak_equal_integral']; pf = float(np.float32(peak))
        sig = float(r.sigma); seed = int(float(r.noise_seed)) if (sig > 0 and str(r.noise_seed) not in ('', 'nan')) else None
        sid = K.PLATFORM_ID[r.surface]
        img_id = dict(raster_key=R['key'], peak_f32=pf, platform_surface=sid, psf_px=K.PSF_PX, sigma=sig, seed=seed, m3=P['m3'], platform=P['version'])
        ik = 'S2I-' + K.digest(img_id, 24)
        dec_id = dict(img=ik, arm=K.ARM[r.family], params=(P['dot_cfg'] if r.family == 'point' else P['p4']), pattern_peak_f32=(pf if r.family == 'line' else None))
        dk = 'S2D-' + K.digest(dec_id, 24)
        ek = 'S2E-' + K.digest(dict(dec=dk, s1r_eval=P['s1r_eval'], global_domain=P['global_domain'], d=K.D_SUPPORT, tau=K.TAU, gross=K.GROSS), 24)
        keyrows.append(dict(cand_id=r.cand_id, tier=r.tier, surface=r.surface, surface_class=('main' if r.surface in K.MAIN else 'supplement'), platform_surface=sid, family=r.family,
                            regime=r.regime, layout=b['layout'], lattice=(r.layout_id if r.family == 'point' else ''), arrangement=(r.arrangement if r.family == 'point' else ''),
                            p_parallel_mm=(r.p_parallel_mm if r.family == 'point' else ''), p_perp_mm=(r.p_perp_mm if r.family == 'point' else ''),
                            line_pitch_mm=(r.line_pitch_mm if r.family == 'line' else ''), orientation_deg=(r.orientation_deg if r.family == 'line' else ''),
                            size_mm=(1.0 if r.family == 'line' else float(r.radius_mm)), size_kind=('line_width' if r.family == 'line' else 'radius'),
                            phase_label=r.phase_label, phase_u=r.phase_u, phase_v=r.phase_v, phase_frac=r.phase_frac, sigma=sig, rep=int(float(r.rep)), noise_seed=(seed if seed is not None else ''),
                            seed_rule=r.seed_rule, raster_key=R['key'], integral_I0=I0, Q=(K.Q if r.regime == 'equal_integral' else ''), peak=peak, peak_f32=pf,
                            integral_actual=(b['integral_actual'] if r.regime == 'equal_integral' else float((R['binary'].astype(np.float32) * np.float32(1.0)).sum(dtype=np.float64))),
                            n_features=b['n_features'], image_key=ik, decode_key=dk, eval_key=ek, decode_arm=K.ARM[r.family], source_cand_id=r.cand_id, source_view=r.view))
    df = pd.DataFrame(keyrows)
    df = _pairs(df)
    return df, pd.DataFrame(budget.values()), P


def _pairs(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    blk = df.surface + '|' + df.regime + '|s' + df.sigma.map(lambda x: f'{x:g}') + '|' + df.phase_label + '|r' + df.rep.astype(str)
    df['pair_block'] = np.where(df.family == 'point', 'S2D|' + blk, 'S2L|' + blk + '|phi' + df.orientation_deg.astype(str))
    pt = df.family == 'point'

    def series_of(lat):
        s = [n for n, members in (('dots_p_parallel', DOT_P_PAR), ('dots_p_perp', DOT_P_PERP), ('dots_RT', ['R8x4', 'T8x4'])) if lat in members]
        return ';'.join(s + ['dots_vs_R8x4' + ('_reference' if lat == 'R8x4' else '')])
    df['series'] = [series_of(l) if f == 'point' else 'lines_pitch;lines_phi' for f, l in zip(df.family, df.lattice)]
    df['design_key'] = df.family + '|' + df.layout + '|' + df.regime + '|s' + df.sigma.map(lambda x: f'{x:g}') + '|' + df.phase_label + '|r' + df.rep.astype(str)
    df['noise_paired_across_surfaces'] = np.where(pt, True, False)
    pan = []
    for _, r in df.iterrows():
        p = ''
        if r.tier == 'core' and r.sigma in (0.0, 40.0) and r.rep == 0:
            if r.family == 'point' and r.phase_label == FIRST_POS:
                tags = []
                if r.lattice in ('R6x4', 'R8x4', 'R12x4'):
                    tags.append('panel_dots_p_parallel')
                if r.lattice in ('R8x2.5', 'R8x4', 'R8x8'):
                    tags.append('panel_dots_p_perp')
                if r.lattice in ('R8x4', 'T8x4'):
                    tags.append('panel_dots_RT')
                p = ';'.join(tags)
            if r.family == 'line' and r.phase_label == FIRST_PH and float(r.line_pitch_mm) in (4.45, 6.3, 12.0):
                p = f"panel_lines_phi{float(r.orientation_deg):g}"
        pan.append(p)
    df['panel'] = pan
    return df


def joint10_rows(df):

    ids = K.JOINT10['new'] + K.JOINT10['reuse']
    x = df[df.cand_id.isin(ids)].copy()
    if sorted(x.cand_id) != sorted(ids) or (x.tier != 'core').any():
        raise RuntimeError('joint10 whitelist does not resolve to 10 core rows')
    x['joint_action'] = ['reuse S1 (read only)' if c in K.JOINT10['reuse'] else 'new render + decode' for c in x.cand_id]
    return x


def write():
    df, bud, P = build()
    jr = joint10_rows(df)
    df['joint10'] = df.cand_id.isin(set(jr.cand_id))
    K.guard(CSV).parent.mkdir(parents=True, exist_ok=True)
    tmp = CSV.with_suffix('.csv.tmp'); df.to_csv(tmp, index=False, encoding='utf-8'); tmp.replace(CSV)
    def cnt(x):
        return dict(images=int(x.image_key.nunique()), decodes=int(x.decode_key.nunique()), views=int(len(x)))
    counts = {t: {cls: {f: cnt(g3) for f, g3 in g2.groupby('family')} | dict(total=cnt(g2)) for cls, g2 in g.groupby('surface_class')} | dict(total=cnt(g)) for t, g in df.groupby('tier')}
    per_surface = {t: {s: int(g2.image_key.nunique()) for s, g2 in g.groupby('surface')} for t, g in df.groupby('tier')}
    cfg = dict(generated_local=K.now_local(), task='S2-PREP-01', inputs=K.input_hashes(), generators={g: K.sha_file(K.ROOT / g) for g in K.GENERATORS}, platform=P, platform_state=K.platform_state(),
               design=dict(source=K.rel(K.CANDS), dots='batch-1 group B lattices R6x4 R7.35x4 R8x4 R10x4 R12x4 R8x2.5 R8x3 R8x6 R8x8 T8x4; radius 1 mm; 8 positions; sigma 0 / 40 (k 0 / 1)',
                           lines='view_L2: w 1.0 mm; s 4.45 5 5.6 6.3 8 10 12 mm; phi 0 / 45; 8 phases; sigma 0 / 40 (rep 0 / 1)', Q=K.Q,
                           geometry='projector image u = along-epipolar direction (baseline along world y = image u): dots p_par = pitch along u (sl_lib: "along-epipolar pitch"), p_perp = row pitch along v;'
                                    ' T = every other row shifted by half p_par along u (sl_lib.lattice stagger 0.5); lines: phi = angle of the line NORMAL to u (m2 olines q = (u - cu) cos phi + (v - cv) sin phi),'
                                    ' phi 0 -> lines along v, crossing the epipolar lines at 90 deg; pitch along the epipolar direction = s / |cos phi| (45 deg: 1.414 s)',
                           seeds='dots: 9420000 + 10 position + k (surface independent -> noise paired across lattices AND surfaces); lines: seed_for(10300000, surface index, phase, sigma, rep) (paired across pitches / phi inside a surface, NOT across surfaces)',
                           optional='fixed-peak twin (peak 1) of every core condition, same geometry / position / sigma / seed: brightness control, not run in this round',
                           not_in_round='group B sigma 20, group A sigma 80, batch 2 (T4 / T5), sigma 20 / 60 lines, P2_C1_guard, other budgets / FWHM / coding / random shifts'),
               evaluation=dict(local='SURF-02 ROI r <= 15 mm via s2_common.evaluate (inside s1r_eval.evaluate_case)', glob=f"s1_real01 global_domain_step0p50 (hash {P['global_domain']})", d_mm=K.D_SUPPORT, tau_mm=K.TAU,
                               gross_mm=K.GROSS, identity='s1r_eval: dots = camera ray -> true surface -> projector -> nearest dot centre (confirmable within r_px + 1 px), geometric identity; lines = l01_truth.truth_identity; NOT the old sigma-0 source labels of spacing_layout',
                               recovery='dots: unique expected dots with >= 1 accepted, valid, correct point; lines: reference rows with an accepted valid feature within 3 px and the correct line id',
                               gauss_amplitude='M10_curv_gauss only, supplementary; A / B / C: not applicable; no old T criterion, no k / 16 counts'),
               statistics=dict(unit='phase (dots: registered position; lines: phase); noise realisations averaged inside a phase first; phases equal weight',
                               table='mean (main) with median, min / max, defined phases / total phases, scientific no-output count; no-output -> coverage / recovery 0, errors undefined',
                               contrasts='dots: adjacent p_par (p_perp 4), adjacent p_perp (p_par 8), R8x4 vs T8x4, every lattice vs R8x4; lines: adjacent pitch (same phi), phi 45 vs 0 (same pitch); three main surfaces on the same design (dots: noise paired; lines: phase-level design pairing only)',
                               resampling='optional descriptive phase-block bootstrap with ONE index matrix shared by all arms and realisations of a contrast (PCG64 seed 2026101202, 10000 draws, 5-95 %)',
                               old_comparison='old S2 tables used medians / pass counts k / 16: any comparison must name the summary statistic; old conclusions are not rewritten'),
               counts=counts, per_surface_images=per_surface, budget=dict(n_unique_rasters=int(len(bud)), infeasible=bud[~bud.feasible].to_dict('records'),
                                                                           peak_range={f: [float(g.peak_equal_integral.min()), float(g.peak_equal_integral.max())] for f, g in bud.groupby('family')},
                                                                           rasters=bud.to_dict('records')),
               panels=dict(dots='per surface x sigma {0, 40}: p_par R6x4 / R8x4 / R12x4, p_perp R8x2.5 / R8x4 / R8x8, R / T R8x4 vs T8x4; position pos33 (first registered), k 0',
                           lines='per surface x sigma {0, 40} x phi {0, 45}: s 4.45 / 6.3 / 12 mm (min / median / max registered); phase ph0, rep 0', rule='fixed before any S2 result; failures shown in place'),
               joint10=jr[['cand_id', 'surface', 'family', 'layout', 'sigma', 'phase_label', 'rep', 'image_key', 'decode_key', 'joint_action']].to_dict('records'),
               joint10_caps=K.CAPS_JOINT10)
    K.save_json(CONFIG, cfg)
    return df, bud, cfg


if __name__ == '__main__':
    import json
    df, bud, cfg = write()
    print(json.dumps(K.clean(dict(counts=cfg['counts'], per_surface=cfg['per_surface_images'], budget=cfg['budget'])), ensure_ascii=False, indent=1)[:3000])
