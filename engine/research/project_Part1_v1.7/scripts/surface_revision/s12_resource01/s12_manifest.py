# Enumerate size-spacing cases and identify equivalent reusable inputs.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_common as K

EXPECTED = dict(views_S1=dict(main=3936, gauss=1312), views_S2=dict(main=6336, gauss=2112), unique=dict(main=9816, gauss=3272), reuse=dict(main=3672, gauss=1224),
                new=dict(main=6144, gauss=2048), new_S1_dot_cap=384, new_S1_line_high=1920, new_S2=5888, shared_S1_S2=352, shared_S1_S2_old=96)


def _f32(x) -> float:
    return float(np.float32(float(x)))


def _seed(s):
    s = str(s).strip()
    return None if s in ('', 'nan', 'None') else int(round(float(s)))


def s1_rasters(V1: pd.DataFrame) -> dict:

    import s1r_common as S1K
    out = {}
    for (fam, size, ph), g in V1.groupby(['family', 'size_mm', 'phase_label']):
        key = g.raster_key.iloc[0]
        if g.raster_key.nunique() != 1:
            raise K.Stop('matrix_drift', f'S1 raster not unique for {(fam, size, ph)}')
        R = S1K.load_raster(key)
        pf = float(g.phase_frac.iloc[0]) if fam == 'line' else None
        regen = S1K.build_raster(fam, float(size), ph, pf)
        out[(fam, float(size), ph)] = dict(key=key, binary=R['binary'] > 0, regen_equal=(regen['key'] == key), source='s1_store', phase_frac=pf, n_features=regen['n_features'],
                                           Q0=float((R['binary'] > 0).sum(dtype=np.float64)))
    return out


def s2_rasters(C: pd.DataFrame) -> dict:

    import s2p_case as SC
    out = {}
    for (fam, lay, ph), g in C.groupby(['family', 'layout', 'phase_label']):
        t = g.iloc[0].to_dict()
        for k in ('phase_u', 'phase_v', 'phase_frac', 'p_parallel_mm', 'p_perp_mm', 'line_pitch_mm', 'orientation_deg', 'size_mm'):
            t[k] = (float(t[k]) if str(t[k]) not in ('', 'nan') else None)
        R = SC.raster_for(t)
        out[(fam, lay, ph)] = dict(key=t['raster_key'], binary=R['binary'] > 0, regen_equal=True, source='s2_build', template_cand_id=t['cand_id'], n_features=int(t['n_features']),
                                   Q0=float((R['binary'] > 0).sum(dtype=np.float64)), row=t)
    return out


# Enumerate size-spacing cases and identify equivalent reusable images.
def build():
    import s1r_common as S1K
    P = K.platform_ids()
    V1 = pd.read_csv(K.S1OUT / 'manifest' / 's1_views.csv', keep_default_na=False, float_precision='round_trip')
    I1 = pd.read_csv(K.S1OUT / 'manifest' / 's1_images.csv', keep_default_na=False, float_precision='round_trip')
    C2 = pd.read_csv(K.S2OUT / 's2_candidates.csv', keep_default_na=False, float_precision='round_trip')
    C2c = C2[C2.tier == 'core'].copy()
    R1 = s1_rasters(V1); R2 = s2_rasters(C2c)
    probs = []
    for k, r in list(R1.items()) + list(R2.items()):
        if not r['regen_equal']:
            probs.append(f'raster regeneration differs: {k}')
    rows = []

    def add(section, surface, family, regime, size, layout, ph, sigma, rep, seed, R, geo):
        res = K.resource(R['binary'], regime)
        bad = K.resource_check(res, regime)
        if bad:
            probs.append(f'{section} {surface} {family} {layout} {ph} {regime}: {bad}')
        sid = K.PLATFORM_ID[surface]
        ik = K.input_key(R['key'], res['peak_f32'], sid, sigma, seed)
        dk = K.decode_key(ik, family, res['peak_f32'])
        rows.append(dict(section=section, surface=surface, surface_class=('main' if surface in K.MAIN else 'supplement'), platform_surface=sid, family=family, regime=regime,
                         size_mm=float(size), diameter_mm=(2 * float(size) if family == 'point' else None), layout=layout, phase_label=ph, phase_index=K.phase_index(ph), sigma=float(sigma),
                         rep=int(rep), noise_seed=seed, raster_key=R['key'], raster_source=R['source'], template_cand_id=R.get('template_cand_id', ''), n_features_design=R['n_features'],
                         **res, input_key=ik, decode_key=dk, eval_key=K.eval_key(dk), **geo))

    for surface in K.SURFACES:
        for r_ in K.DOT_RADII:
            for pi in range(3):
                ph = f'sn_phase{pi}'; R = R1[('point', float(r_), ph)]
                for s in K.SIGMAS:
                    for rep in ([0] if s == 0 else [0, 1, 2]):
                        seed = K.seed_s1_dot(pi, rep) if s > 0 else None
                        for reg in K.REGIMES:
                            add('S1', surface, 'point', reg, r_, 'S1_R8x4', ph, s, rep, seed, R, dict(p_parallel_mm=8.0, p_perp_mm=4.0, line_pitch_mm=None, orientation_deg=None, phase_frac=None))
        for w in K.LINE_WIDTHS:
            for pi in range(8):
                ph = f'ph{pi}'; R = R1[('line', float(w), ph)]
                for s in K.SIGMAS:
                    for rep in ([0] if s == 0 else [0, 1]):
                        seed = K.seed_line(surface, pi, s, rep)
                        for reg in ('resource_cap_11695', 'fixed_peak_1'):
                            add('S1', surface, 'line', reg, w, 's8_phi0', ph, s, rep, seed, R, dict(p_parallel_mm=None, p_perp_mm=None, line_pitch_mm=8.0, orientation_deg=0.0, phase_frac=R['phase_frac']))
        for (fam, lay, ph), R in sorted(R2.items()):
            t = R['row']; pi = K.phase_index(ph)
            for s in K.SIGMAS:
                for rep in ([0] if s == 0 else [0, 1]):
                    if fam == 'point':
                        seed = K.seed_s2_dot(pi, rep) if s > 0 else None
                        geo = dict(p_parallel_mm=t['p_parallel_mm'], p_perp_mm=t['p_perp_mm'], line_pitch_mm=None, orientation_deg=None, phase_frac=None)
                        add('S2', surface, 'point', 'resource_cap_11695', 1.0, lay, ph, s, rep, seed, R, geo)
                    else:
                        seed = K.seed_line(surface, pi, s, rep)
                        geo = dict(p_parallel_mm=None, p_perp_mm=None, line_pitch_mm=t['line_pitch_mm'], orientation_deg=t['orientation_deg'], phase_frac=t['phase_frac'])
                        add('S2', surface, 'line', 'resource_cap_11695', 1.0, lay, ph, s, rep, seed, R, geo)
    V = pd.DataFrame(rows)
    V['view_id'] = ['S12V-' + K.digest([r.section, r.surface, r.family, r.regime, r.size_mm, r.layout, r.phase_label, r.sigma, r.rep], 16) for r in V.itertuples()]
    if V.view_id.nunique() != len(V):
        probs.append('view ids not unique')

    v1 = V1.copy(); v1['seed_old'] = v1.noise_seed.map(_seed)
    v1['regime_s12'] = np.where(v1.regime == 'fixed_peak', 'fixed_peak_1', np.where(v1.family == 'point', 'equal_integral_5850', 'resource_cap_11695'))
    m1 = v1.merge(V[V.section == 'S1'], left_on=['surface', 'family', 'regime_s12', 'size_mm', 'phase_label', 'sigma', 'rep'], right_on=['surface', 'family', 'regime', 'size_mm', 'phase_label', 'sigma', 'rep'],
                  how='left', suffixes=('_old', ''))
    seed_bad1 = m1[[_seed(a) != _seed(b) for a, b in zip(m1.noise_seed_old, m1.noise_seed)]]
    key_bad1 = m1[m1.input_key.isna()]
    c2 = C2c.copy(); c2['seed_old'] = c2.noise_seed.map(_seed)
    m2 = c2.merge(V[V.section == 'S2'], on=['surface', 'family', 'layout', 'phase_label', 'sigma', 'rep'], how='left', suffixes=('_old', ''))
    seed_bad2 = m2[[_seed(a) != _seed(b) for a, b in zip(m2.noise_seed_old, m2.noise_seed)]]
    if len(seed_bad1) or len(seed_bad2) or len(key_bad1):
        probs.append(f'old seeds / rows not reproduced: S1 {len(seed_bad1)} (missing {len(key_bad1)}), S2 {len(seed_bad2)}')

    m1['raster_same'] = m1.raster_key_old == m1.raster_key
    m1['peak_same'] = [_f32(a) == _f32(b) for a, b in zip(m1.peak_f32_old, m1.peak_f32)]
    m2['raster_same'] = m2.raster_key_old == m2.raster_key
    m2['peak_same'] = [_f32(a) == _f32(b) for a, b in zip(m2.peak_f32_old, m2.peak_f32)]
    for nm, m in (('S1', m1), ('S2', m2)):
        if not (m.raster_same.all() and m.peak_same.all()):
            probs.append(f'{nm}: stored rows with other raster / peak than the S12 view ({int((~m.raster_same).sum())} / {int((~np.asarray(m.peak_same)).sum())})')

    def tup(raster, peak, sid, sigma, seed):
        return (raster, _f32(peak), sid, float(sigma), (None if float(sigma) == 0 else _seed(seed)))
    s1_idx = {}
    for r in I1.itertuples():
        s1_idx.setdefault(tup(r.raster_key, r.peak_f32, r.platform_surface, r.sigma, r.noise_seed), []).append((r.image_key, r.decode_key))
    plan2 = None
    s2_idx = {}
    for r in C2c.itertuples():
        s2_idx.setdefault(tup(r.raster_key, r.peak_f32, r.platform_surface, r.sigma, r.noise_seed), []).append((r.image_key, r.decode_key, r.cand_id))
    U = V.sort_values(['surface', 'section', 'family', 'layout', 'size_mm', 'phase_label', 'sigma', 'rep', 'regime']).groupby('input_key', as_index=False).agg(
        decode_key=('decode_key', 'first'), eval_key=('eval_key', 'first'), raster_key=('raster_key', 'first'), raster_source=('raster_source', 'first'),
        template_cand_id=('template_cand_id', 'first'), surface=('surface', 'first'), surface_class=('surface_class', 'first'), platform_surface=('platform_surface', 'first'),
        family=('family', 'first'), size_mm=('size_mm', 'first'), layout=('layout', 'first'), phase_label=('phase_label', 'first'), sigma=('sigma', 'first'), rep=('rep', 'first'),
        noise_seed=('noise_seed', 'first'), peak_f32=('peak_f32', 'first'), peak_target=('peak_target', 'first'), Q0=('Q0', 'first'), Q_actual=('Q_actual', 'first'),
        sections=('section', lambda s: ';'.join(sorted(set(s)))), regimes=('regime', lambda s: ';'.join(sorted(set(s)))), n_views=('view_id', 'size'), phase_frac=('phase_frac', 'first'))
    if V.groupby('input_key').decode_key.nunique().max() != 1:
        probs.append('input / decode identity not one-to-one')
    if V.groupby('input_key').peak_f32.nunique().max() != 1:
        probs.append('views of one input with different float32 peaks')
    src, s1i, s1d, s2i, s2d, s2c = [], [], [], [], [], []
    for r in U.itertuples():
        t = tup(r.raster_key, r.peak_f32, r.platform_surface, r.sigma, r.noise_seed)
        a, b = s1_idx.get(t, []), s2_idx.get(t, [])
        if len(a) > 1 or len(b) > 1:
            probs.append(f'{r.input_key}: {len(a)} S1 / {len(b)} S2 stored inputs')
        if a:
            src.append('s1'); s1i.append(a[0][0]); s1d.append(a[0][1])
        else:
            src.append('s2' if b else 'new'); s1i.append(''); s1d.append('')
        s2i.append(b[0][0] if b else ''); s2d.append(b[0][1] if b else ''); s2c.append(b[0][2] if b else '')
    U['source'] = src; U['s1_image_key'] = s1i; U['s1_decode_key'] = s1d; U['s2_image_key'] = s2i; U['s2_decode_key'] = s2d; U['s2_cand_id'] = s2c

    U['s2_has_record'] = [bool(k) and (K.S2OUT / 'run' / 'eval' / f'{k}.json').exists() for k in U.s2_decode_key]
    stage = []
    for r in U.itertuples():
        g = 0 if r.surface_class == 'main' else 2
        stage.append(g + (0 if 'S1' in r.sections else 1))
    U['stage'] = stage
    V = V.merge(U[['input_key', 'source', 'stage']], on='input_key', how='left')

    V['size_group'] = np.where(V.section == 'S1', V.section + '|' + V.surface + '|' + V.family + '|' + V.regime + '|s' + V.sigma.map(lambda x: f'{x:g}') + '|' + V.phase_label + '|r' + V.rep.astype(str), '')
    V['regime_group'] = np.where(V.section == 'S1', V.section + '|' + V.surface + '|' + V.family + '|d' + V.size_mm.map(lambda x: f'{x:g}') + '|s' + V.sigma.map(lambda x: f'{x:g}') + '|' + V.phase_label + '|r' + V.rep.astype(str), '')
    V['s2_block'] = np.where(V.section == 'S2', V.surface + '|' + V.family + '|s' + V.sigma.map(lambda x: f'{x:g}') + '|' + V.phase_label + '|r' + V.rep.astype(str), '')
    V['same_input_as_other_regime'] = V.input_key.map(V.groupby('input_key').regime.nunique()) > 1
    V['shared_S1_S2'] = V.input_key.map(V.groupby('input_key').section.nunique()) > 1

    def by_cls(x, col='input_key'):
        return {c: int(g[col].nunique() if col == 'input_key' else len(g)) for c, g in x.groupby(x.surface_class.map({'main': 'main', 'supplement': 'gauss'}))}
    cnt = dict(views_S1=by_cls(V[V.section == 'S1'], 'view'), views_S2=by_cls(V[V.section == 'S2'], 'view'), unique=by_cls(U.assign(view=1)), reuse=by_cls(U[U.source != 'new']),
               new=by_cls(U[U.source == 'new']), new_S1_dot_cap=int(U[(U.source == 'new') & U.sections.str.contains('S1') & (U.family == 'point')].shape[0]),
               new_S1_line_high=int(U[(U.source == 'new') & U.sections.str.contains('S1') & (U.family == 'line')].shape[0]),
               new_S2=int(U[(U.source == 'new') & (U.sections == 'S2')].shape[0]), shared_S1_S2=int((U.sections == 'S1;S2').sum()),
               shared_S1_S2_old=int(((U.sections == 'S1;S2') & (U.s2_cand_id != '')).sum()), shared_S1_S2_reused_now=int(((U.sections == 'S1;S2') & (U.source != 'new')).sum()),
               views_total=int(len(V)), unique_total=int(len(U)),
               by_source={k: int(v) for k, v in U.source.value_counts().items()}, s2_record_sources=int(((U.source == 's2') & U.s2_has_record).sum()),
               new_views_S2=int((V[V.section == 'S2'].source == 'new').sum()))
    dev = {k: dict(expected=v, actual=cnt.get(k)) for k, v in EXPECTED.items() if cnt.get(k) != v}
    if dev:
        probs.append(f'counts differ from the task table: {dev}')
    if ((U.source == 's2') & ~U.s2_has_record).any():
        probs.append('S2 source without an S2 record')

    RT = V.drop_duplicates(['raster_key', 'regime', 'section', 'family', 'layout', 'size_mm', 'phase_label'])[
        ['section', 'family', 'layout', 'size_mm', 'diameter_mm', 'phase_label', 'regime', 'raster_key', 'Q0', 'Q_budget', 'peak_target', 'peak_f32', 'Q_actual', 'budget_usage', 'active_limit',
         'n_features_design', 'p_parallel_mm', 'p_perp_mm', 'line_pitch_mm', 'orientation_deg']].sort_values(['section', 'family', 'layout', 'size_mm', 'phase_label', 'regime'])

    ex = []
    mp = K.NOISE_EXAMPLE / 'manifest.json'
    if mp.exists():
        man = K.load_json(mp); allk = set(U.input_key)
        surf = man.get('surface'); surf = surf if surf in K.PLATFORM_ID else {v: k for k, v in K.PLATFORM_ID.items()}.get(surf, surf)
        for c in man.get('cases', []):
            try:
                R = R1[('line', float(c['width_mm']), man.get('phase_label', 'ph0'))]
                ik = K.input_key(R['key'], c['peak'], K.PLATFORM_ID[surf], c['sigma'], man.get('seed'))
                pi = K.phase_index(man.get('phase_label', 'ph0'))
                ex.append(dict(id=c.get('id'), sigma=c['sigma'], width_mm=c['width_mm'], seed=man.get('seed'), formal_seed=K.seed_line(surf, pi, c['sigma'], int(man.get('repeat', 0) or 0)),
                               in_formal=ik in allk))
            except Exception as e:
                ex.append(dict(id=c.get('id'), error=repr(e)[:160]))
    check = dict(t_local=K.now_local(), counts=cnt, expected=EXPECTED, deviations=dev, problems=probs, n_rasters=dict(S1=len(R1), S2=len(R2)),
                 rasters_regenerated_equal=all(r['regen_equal'] for r in list(R1.values()) + list(R2.values())),
                 seeds=dict(rules=K.SEED_RULES, s1_rows_checked=int(len(m1)), s1_mismatch=int(len(seed_bad1)), s2_rows_checked=int(len(m2)), s2_mismatch=int(len(seed_bad2))),
                 old_rows_same_physical_input=dict(S1=bool(m1.raster_same.all() and m1.peak_same.all()), S2=bool(m2.raster_same.all() and m2.peak_same.all())),
                 resource=dict(by_limit={f'{s}|{f}|{reg}|{lim}': int(n) for (s, f, reg, lim), n in RT.groupby(['section', 'family', 'regime', 'active_limit']).size().items()},
                               dot_r05_Q0=[float(RT[(RT.section == 'S1') & (RT.family == 'point') & (RT.size_mm == 0.5)].Q0.min()), float(RT[(RT.section == 'S1') & (RT.family == 'point') & (RT.size_mm == 0.5)].Q0.max())],
                               peak_range=[float(V.peak_f32.min()), float(V.peak_f32.max())]),
                 noise_examples=dict(n=len(ex), in_formal=int(sum(1 for e in ex if e.get('in_formal'))), cases=ex), platform=dict(version=P['version'], m3=P['m3'], global_domain=P['global_domain']))
    return V, U, RT, check


def write():
    V, U, RT, check = build()
    cols_v = ['view_id', 'section', 'surface', 'surface_class', 'platform_surface', 'family', 'regime', 'size_mm', 'diameter_mm', 'layout', 'p_parallel_mm', 'p_perp_mm', 'line_pitch_mm',
              'orientation_deg', 'phase_label', 'phase_index', 'phase_frac', 'sigma', 'rep', 'noise_seed', 'raster_key', 'raster_source', 'template_cand_id', 'n_features_design', 'Q0', 'Q_budget',
              'peak_target', 'peak_f32', 'Q_actual', 'budget_usage', 'active_limit', 'input_key', 'decode_key', 'eval_key', 'source', 'stage', 'size_group', 'regime_group', 's2_block',
              'same_input_as_other_regime', 'shared_S1_S2']
    V = V[cols_v].copy(); V['noise_seed'] = V.noise_seed.map(lambda x: '' if x is None or (isinstance(x, float) and np.isnan(x)) else int(x))
    U = U.copy(); U['noise_seed'] = U.noise_seed.map(lambda x: '' if x is None or (isinstance(x, float) and np.isnan(x)) else int(x))
    K.save_csv(V, K.MANIFEST); K.save_csv(U, K.INPUTS); K.save_csv(RT, K.RESULTS / 'resource_table.csv')
    check.update(manifest_sha256=K.sha_file(K.MANIFEST), inputs_sha256=K.sha_file(K.INPUTS))
    K.save_json(K.CHECKS / 'manifest_check.json', check)
    return V, U, RT, check


if __name__ == '__main__':
    V, U, RT, c = write()
    print(json.dumps(K.clean(dict(counts=c['counts'], deviations=c['deviations'], problems=c['problems'][:10], seeds=dict(c['seeds'], rules=None), resource=c['resource'],
                                  noise_examples=dict(n=c['noise_examples']['n'], in_formal=c['noise_examples']['in_formal']))), ensure_ascii=False, indent=1))
    sys.exit(0 if not c['problems'] else 13)
