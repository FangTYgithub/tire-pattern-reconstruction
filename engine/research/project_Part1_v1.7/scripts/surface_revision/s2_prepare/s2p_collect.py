# Collect spacing-layout case records into analysis tables.

from __future__ import annotations

import numpy as np
import pandas as pd

import s2p_common as K

K.setup()
GROUPS = {
    'G1_Punkte_Abstand_M10': (['C013282', 'C013244', 'C013224'], 'M10 + Zusatzkrümmung · Punkte r 1 mm · σ 0 · pos33 · p$_\\parallel$ 6 / 8 / 12 mm (p$_\\perp$ 4 mm)'),
    'G2_Punkte_R_T_M10': (['C013244', 'C013279'], 'M10 + Zusatzkrümmung · Punkte 8 × 4 mm regulär (R) / versetzt (T) · σ 0 · pos33'),
    'G3_Linien_0_45_M10': (['C018447', 'C018309'], 'M10 + Zusatzkrümmung · Linien w 1 mm, s 4,45 mm · φ 0° / 45° · σ 0 · ph0'),
    'G4_schwierige_Faelle': (['C013736', 'C018194'], 'Zwei verrauschte Einzelfälle (σ 40): M24 Punkte R8×2,5 und B Linien s 4,45 mm 45° – verschiedene Flächen und Familien, kein Familienvergleich'),
}
KEEP = ('state', 'n_out', 'n_accepted', 'n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'id_denominator', 'mismatch_rate', 'unconf_share', 'img_n_expected', 'img_n_recovered', 'img_recovery',
        'n_features_design', 'loc_n_valid', 'loc_pt_rmse', 'loc_pt_mae', 'loc_pt_p95', 'loc_pt_bias', 'loc_gross_rate', 'loc_recovery', 'loc_cov_interp', 'loc_Cvalid', 'loc_Ctau', 'loc_surf_rmse',
        'glob_n_valid', 'glob_pt_rmse', 'glob_pt_mae', 'glob_pt_p95', 'glob_pt_bias', 'glob_gross_rate', 'glob_recovery', 'glob_cov_interp', 'glob_Cvalid', 'glob_Ctau', 'glob_surf_rmse', 'amp_applicable')


def current_eval_identity(t: dict) -> str:
    import s2p_case as CS
    import s2p_freeze as FZ

    class _B:
        version = K.platform_state()['version']
    fz = K.load_json(K.FREEZE)
    R = CS.raster_for(t); peak = CS.peak_for(t, R)
    return CS.identities(t, _B, fz['part_digests'], peak)['eval'], peak


def records(set_name: str):
    import s1r_common as S1K
    import s2p_case as CS
    import s2p_reuse as RU
    import s2p_run as RN
    d = RN.load_list(); rows = RN.set_rows(d, set_name)
    T, _ = RU.build(d); reuse = {r['image_key']: r for r in T.to_dict('records')}
    out = []
    for _, t in rows.iterrows():
        t = t.to_dict(); rec = dict(cand_id=t['cand_id'], decode_key=t['decode_key'], noise_seed=K.norm_seed(t['noise_seed']), rep=int(t['rep']), phase_label=t['phase_label'],
                                    surface=t['surface'], family=t['family'], regime=t['regime'], layout=t['layout'], sigma=float(t['sigma']), peak=float(t['peak']))
        if t['cand_id'] in K.JOINT10['reuse']:
            r = reuse.get(t['image_key']) or {}; lay = RU.verify_layers(r) if r else dict(image_ok=False, decode_ok=False, eval_ok=False, reasons=['no reuse match'])
            src = K.S1OUT / 'decodes' / f"{r.get('s1_decode_key')}.json"
            ok = lay['image_ok'] and lay['decode_ok'] and lay['eval_ok'] and S1K.complete(src)
            m = K.load_json(src)['metrics'] if ok else {}
            rec.update(source='S1 reuse (image + decode + evaluation, read only)', record=K.rel(src), complete=bool(ok), state=(m.get('state') if ok else 'technical'), reuse_layers=lay,
                       s1_image_key=r.get('s1_image_key'), **{k: m.get(k) for k in KEEP if k != 'state'})
        else:
            ep = K.OUT / 'run' / 'eval' / f"{t['decode_key']}.json"
            ident, _ = current_eval_identity(t)
            ok = CS.complete(ep, ident); stale = CS.complete(ep) and not ok
            m = K.load_json(ep)['metrics'] if ok else {}
            src = 'S2 new (render + decode + evaluation)'
            if ok:
                er = K.load_json(ep); rr = K.load_json(K.OUT / 'run' / 'raw' / f"{t['decode_key']}.json")
                src = f"S2 image: {rr['image_source']}; decode: S2; evaluation: S2"
                if K.norm_seed(er['noise_seed']) != rec['noise_seed'] or int(er['rep']) != rec['rep'] or er['phase_label'] != rec['phase_label']:
                    ok = False
            rec.update(source=src, record=K.rel(ep), complete=bool(ok), stale=bool(stale), state=(m.get('state') if ok else 'technical'), **{k: m.get(k) for k in KEEP if k != 'state'})
        out.append(rec)
    return rows, pd.DataFrame(out)


def collect(set_name: str = 'joint10') -> dict:
    import s2p_figs as F
    import s2p_run as RN
    import s2p_stats as ST
    rows, rec = records(set_name)
    res_dir = K.guard(K.RESULTS); res_dir.mkdir(parents=True, exist_ok=True)
    cases = rows[['cand_id', 'surface', 'family', 'layout', 'lattice', 'line_pitch_mm', 'orientation_deg', 'sigma', 'phase_label', 'rep', 'noise_seed', 'peak', 'peak_f32', 'n_features',
                  'image_key', 'decode_key']].merge(rec.drop(columns=['surface', 'family', 'layout', 'sigma', 'phase_label', 'rep', 'noise_seed', 'peak', 'decode_key', 'regime']), on='cand_id')
    cases.to_csv(res_dir / f'{set_name}_cases.csv', index=False)
    n = len(rows); n_ok = int((rec.complete & rec.state.isin(['ok', 'no_output'])).sum())
    tech = dict(set=set_name, denominator=n, technically_complete=n_ok, scientific_no_output=int((rec.state == 'no_output').sum()), technical_missing=int(n - n_ok),
                sources={s: int(c) for s, c in rec.source.str.split(';').str[0].value_counts().items()}, ok=bool(n_ok == n))
    led = RN.Ledger(); tech.update(real_renders=led.count('render'), real_decodes=led.count('decode'), caps=K.CAPS_JOINT10)
    K.save_json(res_dir / f'{set_name}_technical.json', tech)
    d = RN.load_list(); core = d[d.tier == 'core']
    touched = set(zip(rows.surface, rows.family, rows.regime, rows.layout, rows.sigma.astype(float)))
    plan = core[[(s, f, r, l, float(g)) in touched for s, f, r, l, g in zip(core.surface, core.family, core.regime, core.layout, core.sigma)]]
    A = ST.analyse(plan, rec)
    A['conditions'].to_csv(res_dir / f'{set_name}_conditions.csv', index=False); A['contrasts'].to_csv(res_dir / f'{set_name}_contrasts.csv', index=False)
    A['surface_contrasts'].to_csv(res_dir / f'{set_name}_surface_contrasts.csv', index=False)
    formal = dict(planned_decodes=A['validation']['n_planned'], ok=A['validation']['n_ok'], no_output=A['validation']['n_no_output'], formal_ok=A['validation']['formal_ok'],
                  cells={str(k): v for k, v in A['conditions'].drop_duplicates(ST.CELL)[ST.CELL + ['state']].assign(k=lambda x: x[ST.CELL].astype(str).agg('|'.join, axis=1)).set_index('k')['state'].items()},
                  contrasts_ok=int((A['contrasts'].state == 'ok').sum()) if len(A['contrasts']) else 0, contrasts_total=int(len(A['contrasts'])))
    figs = []; figcheck = {}
    if tech['ok']:
        for stem, (ids, title) in GROUPS.items():
            p, chk = F.fig_group(cases[cases.cand_id.isin(ids)].set_index('cand_id').loc[ids].reset_index(), title, f'S2_joint10_{stem}')
            figs.append(K.rel(p)); figcheck[stem] = chk
    out = dict(technical=tech, formal=formal, figures=figs, figure_value_check=figcheck, tables=sorted(K.rel(p) for p in res_dir.glob(f'{set_name}_*')))
    K.save_json(res_dir / f'{set_name}_summary.json', out)
    return out
