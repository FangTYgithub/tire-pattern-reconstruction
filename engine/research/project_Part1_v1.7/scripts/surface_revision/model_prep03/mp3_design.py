# Build environment cases with fixed pattern, surface and noise identities.

from __future__ import annotations

import numpy as np
import pandas as pd

import mp3_common as K
import mp3_forward as F

OUTSR = K.ROOT / 'outputs' / 'surface_revision'
S5_MANIFEST = OUTSR / 's5_align02' / 'manifest_next.csv'
S5_CASES = OUTSR / 's5_align05' / 'case_results.csv'
S5_PLAN = OUTSR / 's345_ready' / 'plan_s345.csv'
S5_FREEZE = OUTSR / 's5_align05' / 'run.json'
DOTS = ('D_r1_p8x4', 'D_r1_p8x6', 'D_r1.5_p8x4', 'D_r1.5_p8x6')
LINES = ('L_w0.5_s6.3', 'L_w0.5_s8', 'L_w1.5_s6.3', 'L_w1.5_s8')
COMBOS = DOTS + LINES
SURFACES = ('B_curv', 'M10_curv', 'M24_curv')
SURF_SID = {'B_curv': 'B_curv', 'M10_curv': 'A_curv', 'M24_curv': 'C_curv'}
PHASES = ('ph0', 'ph1')
SIGMAS_MAIN = (20.0, 40.0)
REP_MAIN, REP_S0 = 1, 0
S0_SURFACE, S0_PHASE = 'M10_curv', 'ph0'
HALF_MM = 46.08
ENVS = {'BLUR2': dict(psf=2.0, field=('const', 1.0)), 'BLUR3': dict(psf=3.0, field=('const', 1.0)), 'UNIFORM': dict(psf=1.0, field=('const', 0.75)),
        'GRAD_X': dict(psf=1.0, field=('gradient_x', 0.75, 0.25)), 'GRAD_Y': dict(psf=1.0, field=('gradient_y', 0.75, 0.25))}
PSF_NOMINAL = 1.0
METHODS = dict(point=('D0', 'DF'), line=('P4',))
ENTRY = dict(surface='M10_curv', phase='ph0', combos=('D_r1_p8x4', 'L_w0.5_s8'), envs=('BLUR2', 'GRAD_Y'), sigmas=(0.0, 40.0))
REPRESENTATIVE = dict(surface='M10_curv', phase='ph0', sigma=40.0, combos=('D_r1_p8x4', 'L_w0.5_s8'))
EXPECTED = dict(new_images=len(COMBOS) * len(SURFACES) * len(PHASES) * len(SIGMAS_MAIN) * len(ENVS) + len(COMBOS) * len(ENVS),
                new_results=(len(DOTS) * 2 + len(LINES)) * (len(SURFACES) * len(PHASES) * len(SIGMAS_MAIN) + 1) * len(ENVS),
                nominal_images=len(COMBOS) * (len(SURFACES) * len(PHASES) * len(SIGMAS_MAIN) + 1),
                nominal_results=(len(DOTS) * 2 + len(LINES)) * (len(SURFACES) * len(PHASES) * len(SIGMAS_MAIN) + 1))
EXPECTED['all_results'] = EXPECTED['new_results'] + EXPECTED['nominal_results']


def field_of(env: str) -> dict:
    f = ENVS[env]['field']
    return F.field_const(f[1]) if f[0] == 'const' else F.field_gradient(f[0].split('_')[1], f[1], f[2], HALF_MM)


def design_record() -> dict:
    return dict(combos=COMBOS, surfaces=SURFACES, phases=PHASES, sigmas_main=SIGMAS_MAIN, rep_main=REP_MAIN, sigma0=dict(surface=S0_SURFACE, phase=S0_PHASE, rep=REP_S0),
                envs={e: dict(psf_true_px=v['psf'], field=field_of(e), field_id=F.field_id(field_of(e))) for e, v in ENVS.items()}, psf_nominal_px=PSF_NOMINAL,
                methods=METHODS, entry=ENTRY, representative=REPRESENTATIVE, expected=EXPECTED, half_mm=HALF_MM,
                main_dots_front_end='DF (D0 = same-image bridge)', lines='P4')


def s5_rows() -> pd.DataFrame:
    M = K.read_csv(S5_MANIFEST)
    sel = M[M.combo.isin(COMBOS) & M.surface.isin(SURFACES) & M.phase_label.isin(PHASES)]
    main = sel[sel.sigma.isin(SIGMAS_MAIN) & (sel.rep == REP_MAIN)]
    s0 = sel[(sel.sigma == 0.0) & (sel.rep == REP_S0) & (sel.surface == S0_SURFACE) & (sel.phase_label == S0_PHASE)]
    out = pd.concat([main, s0]).copy()
    key = ['combo', 'surface', 'phase_label', 'sigma', 'rep']
    if out.duplicated(key).any() or len(out) != EXPECTED['nominal_images']:
        raise K.Stop('preflight', f'S5 rows: {len(out)} (expected {EXPECTED["nominal_images"]}), duplicates {int(out.duplicated(key).sum())}')
    P = K.read_csv(S5_PLAN); P = P[(P.set_formal == 'S5-main') & (P.sigma == 0.0)]
    pk = P.set_index(['pair_role', 'surface', 'phase_label'])
    if pk.index.duplicated().any():
        raise K.Stop('preflight', 'plan_s345 S5-main sigma-0 index not unique')
    out['cand_id'] = [pk.loc[(r.combo, r.surface, r.phase_label), 'cand_id'] for r in out.itertuples()]
    return out.sort_values(key).reset_index(drop=True)


def missing(x) -> bool:

    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _seed(r) -> int | None:
    if float(r['sigma']) == 0.0:
        if not missing(r['noise_seed']):
            raise K.Stop('preflight', f"{r['manifest_key']}: sigma 0 with a seed {r['noise_seed']!r}")
        return None
    return K.finite_int(r['noise_seed'], f"{r['manifest_key']} noise_seed")


def manifest() -> tuple[pd.DataFrame, pd.DataFrame]:

    S = s5_rows(); new, nom = [], []
    for r in S.to_dict('records'):
        fam = r['family']; seed = _seed(r)
        nom.append(dict(nominal_key=r['manifest_key'], family=fam, combo=r['combo'], surface=r['surface'], phase_label=r['phase_label'], sigma=float(r['sigma']), rep=int(r['rep']),
                        seed=seed, raster_key=r['raster_key'], peak=float(r['peak']), integral=float(r['integral']), status_s5=r['status'], existing_image_key=r['existing_image_key'],
                        existing_decode_key=r['existing_decode_key'], cand_id=r['cand_id']))
        for env, ev in ENVS.items():
            cid = f"{r['combo']}|{r['surface']}|{r['phase_label']}|s{int(r['sigma'])}|r{int(r['rep'])}|{env}"
            entry = (r['surface'] == ENTRY['surface'] and r['phase_label'] == ENTRY['phase'] and r['combo'] in ENTRY['combos'] and env in ENTRY['envs'] and float(r['sigma']) in ENTRY['sigmas'])
            rep_case = (r['surface'] == REPRESENTATIVE['surface'] and r['phase_label'] == REPRESENTATIVE['phase'] and float(r['sigma']) == REPRESENTATIVE['sigma'] and r['combo'] in REPRESENTATIVE['combos'])
            fld = field_of(env)
            new.append(dict(case_id=cid, stage=('entry' if entry else 'main'), family=fam, combo=r['combo'], radius_mm=(None if fam == 'line' else float(r['radius_mm'])),
                            line_width_mm=(float(r['line_width_mm']) if fam == 'line' else None), p_parallel_mm=r['p_parallel_mm'], p_perp_mm=r['p_perp_mm'], line_pitch_mm=r['line_pitch_mm'],
                            surface=r['surface'], sid=SURF_SID[r['surface']], phase_label=r['phase_label'], phase_u=r['phase_u'], phase_v=r['phase_v'], phase_frac=r['phase_frac'],
                            phase_offset_par_mm=r['phase_offset_par_mm'], phase_offset_perp_mm=r['phase_offset_perp_mm'], sigma=float(r['sigma']), rep=int(r['rep']), seed=seed,
                            seed_source=f"S5 manifest {r['manifest_key']} ({r['seed_source']})", env=env, psf_true_px=float(ev['psf']), psf_nominal_px=PSF_NOMINAL,
                            field_kind=fld['kind'], field_json=K.canon(fld), raster_key=r['raster_key'], lit_px=float(r['lit_px']), peak=float(r['peak']),
                            peak_f32=float(np.float32(r['peak'])), integral=float(r['integral']), limit_type=r['limit_type'], cand_id=r['cand_id'],
                            parent_nominal_key=r['manifest_key'], parent_nominal_image_key=(r['existing_image_key'] if isinstance(r['existing_image_key'], str) else r['manifest_key']),
                            methods=';'.join(METHODS[fam]), keep_arrays=bool(rep_case)))
    N = pd.DataFrame(new); O = pd.DataFrame(nom)
    N['seed'] = pd.Series([r['seed'] for r in new], dtype=object); O['seed'] = pd.Series([r['seed'] for r in nom], dtype=object)
    if len(N) != EXPECTED['new_images'] or N.case_id.duplicated().any() or (N.stage == 'entry').sum() != 8:
        raise K.Stop('preflight', f'new manifest: {len(N)} rows (expected {EXPECTED["new_images"]}), duplicates {int(N.case_id.duplicated().sum())}, entry {(N.stage == "entry").sum()}')
    return N, O


def nominal_sources(O: pd.DataFrame) -> pd.DataFrame:

    C = K.read_csv(S5_CASES); rows = []
    for o in O.to_dict('records'):
        fam = o['family']
        for m in METHODS[fam]:
            c = C[(C.manifest_key == o['nominal_key']) & (C.front_end == m)]
            if len(c) != 1:
                raise K.Stop('preflight', f"{o['nominal_key']} {m}: {len(c)} S5 case rows (exactly 1 required)")
            c = c.iloc[0].to_dict()
            if int(c['rep']) != o['rep'] or float(c['sigma']) != o['sigma'] or (missing(o['seed']) != missing(c['noise_seed'])) or (not missing(o['seed']) and int(c['noise_seed']) != int(o['seed'])):
                raise K.Stop('preflight', f"{o['nominal_key']} {m}: S5 case row rep / sigma / seed differ from the manifest")
            rows.append(dict(nominal_key=o['nominal_key'], method=m, image_key=c['image_key'], decode_key=c['decode_key'], eval_record=c['eval_record'], source=c['source'],
                             image_identity=c.get('image_identity'), decode_identity=c.get('decode_identity'), eval_identity=c.get('eval_identity')))
    return pd.DataFrame(rows)


def image_record_of(src_row: dict) -> tuple:

    ik = src_row['image_key']
    if str(ik).startswith('S5X-'):
        p = OUTSR / 's5_six_formal' / 'store' / 'image' / f'{ik}.json'
    else:
        p = OUTSR / 's345_ready' / 'run' / 'images' / f'{ik}.json'
    return p, p.with_suffix('.npz')


def eval_arrays_of(src_row: dict) -> dict:

    p = K.ROOT / src_row['eval_record']
    if p.name.startswith('df_'):
        with np.load(p, allow_pickle=False) as z:
            pre = f"{src_row['image_key']}__"
            return {k[len(pre):]: np.asarray(z[k]) for k in z.files if k.startswith(pre)}
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}
