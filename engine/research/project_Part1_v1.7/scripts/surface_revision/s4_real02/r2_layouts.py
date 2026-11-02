# Load and transform the selected dot and line layouts.

from __future__ import annotations

import numpy as np
import pandas as pd

import r2_common as K

CSV = K.S4B / 's4_manifest_candidates.csv'
J, ALPHA, BETA, BUDGET = 0.20, 0.70, 1.15, 1000


def _csv() -> pd.DataFrame:
    if K.sha_file(CSV) != K.ANCHORS[K.rel(CSV)]:
        raise K.Stop('input', 'audited candidate table hash differs from the task anchor')
    return pd.read_csv(CSV, low_memory=False)


def build_registry() -> dict:
    import s3_generate as SG
    import sp2_generator as GEN
    D = _csv(); sel = D[D.table_type == 'C_selection']
    reg = dict(version='s4_real02_layouts_v1', source=dict(file=K.rel(CSV), sha256=K.sha_file(CSV)), j=J, alpha=ALPHA, beta=BETA, candidate_budget=BUDGET,
               generator='scripts/spatial_layout/spat03/s3_generate.evaluate (seed_s3, unit vector, general validity)', blocks=[f'B{b:03d}' for b in K.BLOCKS], baselines={})
    checks = []
    for name, famp, fam in K.BASES:
        bl, ms = SG.baseline(name), SG.sets()[name]; ctx = GEN.context_v2(bl, ms, J * bl['p0_mm'])
        L = {'regular': dict(layout_id=f'{name}:regular', block=None, cand=None, seed=None, disp_px={}, strategies=[('S0', None)], general_ok=True, g_min_mm=float(bl['p0_mm']),
                              h_mm=float(ms['h0_mm']), delta_rms_mm=0.0, delta_max_mm=0.0, note='zero displacement (S0); h = registered h0')}
        for r in sel[sel.baseline == name].itertuples():
            if r.status != 'selected':
                raise K.Stop('plan', f'{name} {r.block} {r.strategy}: audited selection is {r.status}')
            lid = f'{r.block}_c{int(r.cand)}'
            if lid not in L:
                ev = SG.evaluate(name, int(r.block[1:]), int(r.cand), J, ctx)
                ok = (ev['raster_sha256'] == r.raster_ref_phase and abs(ev['g_min_mm'] - r.g_min_mm) < 1e-9 and abs(ev['h_mm'] - r.h_mm) < 1e-9 and abs(ev['delta_rms_mm'] - r.delta_rms_mm) < 1e-9 and ev['general_ok'])
                checks.append(dict(baseline=name, layout=lid, raster_ref_equal=ev['raster_sha256'] == r.raster_ref_phase, descriptors_equal=ok, general_ok=ev['general_ok']))
                if not ok:
                    raise K.Stop('check', f'{name} {lid}: re-derived layout differs from the audited table')
                L[lid] = dict(layout_id=f'{name}:{lid}', block=r.block, cand=int(r.cand), seed=int(ev['seed']), unit_hash=ev['unit_hash'], disp_px={str(k): float(v) for k, v in ev['disp_px'].items()},
                              strategies=[], general_ok=bool(ev['general_ok']), g_min_mm=float(ev['g_min_mm']), g_max_mm=float(ev['g_max_mm']), h_mm=float(ev['h_mm']), delta_rms_mm=float(ev['delta_rms_mm']),
                              delta_max_mm=float(ev['delta_max_mm']), edge_clearance_min_mm=float(ev['edge_clearance_min_mm']), raster_ref_phase=ev['raster_sha256'], n_elements_ref=int(ev['n_elements']))
            L[lid]['strategies'].append((r.strategy, r.block))
        views = [dict(strategy='S0', block=None, layout='regular', cand=None, n_mask_candidates=None, n_raster_evaluated=None, generation='none (regular)')] + \
                [dict(strategy=r.strategy, block=r.block, layout=f'{r.block}_c{int(r.cand)}', cand=int(r.cand), n_mask_candidates=int(r.n_mask_candidates), n_raster_evaluated=int(r.n_raster_evaluated),
                      generation='selected (first valid candidate in the registered order)') for r in sel[sel.baseline == name].itertuples()]
        reg['baselines'][name] = dict(family=famp, p0_mm=float(bl['p0_mm']), h0_mm=float(ms['h0_mm']), movable=[int(x) for x in ms['movable']], protected_L=int(ms['protected_L']),
                                      protected_R=int(ms['protected_R']), layouts=L, views=views)
    return reg, checks


def table(name: str, layout: dict, phase_id: str) -> dict:
    import sp2_common as S2
    fam = 'dots' if name.startswith('R') else 'lines'
    ph = next(p for p in S2.registered_phases(fam) if p['phase_id'] == phase_id)
    disp = {int(k): float(v) for k, v in layout['disp_px'].items()}
    out = dict(baseline=name, proposal_id=layout['layout_id'], phase_id=phase_id, disp=disp, j=(J if disp else 0.0), arms=[s for s, _b in layout['strategies']], family=fam, phase=ph)
    if fam == 'dots':
        import sp_dots_columns as DC
        lay = DC.layout(*S2.DOT_BASELINES[name], ph['u'], ph['v'], disp or None, 'u'); lay0 = DC.layout(*S2.DOT_BASELINES[name], ph['u'], ph['v'], None, 'u')
        out.update(pat=np.asarray(DC.raster(lay['c'])), centres=np.asarray(lay['c'], np.float64), centres_regular=np.asarray(lay0['c'], np.float64), ix=np.asarray(lay['ix']), spec=None,
                   present=sorted(set(int(i) for i in lay['ix'])), n_elements=int(lay['n']))
    else:
        import sp2_lines_table as LT2
        pat, meta = LT2.table_spec(S2.LINE_BASELINES[name], S2.W_MM, ph['phase_frac'], disp or None)
        ids = sorted(int(k) for k in meta['olines']['ids'])
        out.update(pat=np.asarray(pat), centres=None, spec=meta['olines'], meta=meta, present=ids, n_elements=len(ids))
    return out


def build_input(name: str, layout: dict, phase_id: str, s0: bool) -> dict:
    import rdy_inputs as RI
    T = table(name, layout, phase_id)
    binary = (np.asarray(T['pat']) > 0).astype(np.float32); lit = float(binary.sum(dtype=np.float64))
    peak = K.Q / lit; pk32 = np.float32(peak); pattern = (binary * pk32).astype(np.float32); integral = float(pattern.sum(dtype=np.float64)); rel_err = abs(integral - K.Q) / K.Q
    if not (peak <= 1.0 + 1e-12) or rel_err > 1e-6:
        raise K.Stop('check', f'{name} {layout["layout_id"]} {phase_id}: peak {peak} / integral relative error {rel_err}')
    inp = dict(binary=binary, peak=float(peak), pattern=pattern, raster_key=RI.raster_key(binary), lit_px=lit, integral_f32=integral, integral_rel_err=rel_err, centres=T['centres'], classes=None,
               meta_full=(None if T['family'] == 'dots' else dict(olines=T['spec'])), brightness='peak = Q 11695 / lit px (S4 rule)', registered_hash_ok=None, table=T, enc_spec=None, ordinary=None)
    if s0:
        o = RI._ordinary_s0({}, T)
        inp['ordinary'] = dict(o, raster_equal=bool(np.array_equal(np.asarray(o['binary']) > 0, binary > 0)), raster_key=RI.raster_key(np.asarray(o['binary'], np.float32)))
    return inp


def reproduce_views(reg: dict) -> tuple[pd.DataFrame, dict]:

    import rdy_inputs as RI
    D = _csv(); LP = D[D.table_type == 'C_layout_phase']; rows = []
    for name, famp, fam in K.BASES:
        B = reg['baselines'][name]; cache = {}
        for v in B['views']:
            lay = B['layouts'][v['layout']]
            for ph in K.PHASES:
                key = (v['layout'], ph)
                if key not in cache:
                    cache[key] = build_input(name, lay, ph, v['strategy'] == 'S0')
                inp = cache[key]; T = inp['table']
                a = LP[(LP.baseline == name) & (LP.strategy == v['strategy']) & (LP.layout == v['layout']) & (LP.phase == ph)]
                if len(a) != 1:
                    raise K.Stop('check', f'{name} {v}: {len(a)} audited rows')
                a = a.iloc[0]
                rows.append(dict(baseline=name, family=famp, strategy=v['strategy'], block=v['block'], layout=v['layout'], phase=ph, raster_key=inp['raster_key'], lit_px=inp['lit_px'], peak=inp['peak'],
                                 peak_f32=float(np.float32(inp['peak'])), integral_f32=inp['integral_f32'], integral_rel_err=inp['integral_rel_err'], n_elements=T['n_elements'],
                                 protected_present=bool(B['protected_L'] in T['present'] and B['protected_R'] in T['present']), movable_all_present=bool(all(m in T['present'] for m in B['movable'])),
                                 raster_equal=(inp['raster_key'] == a.raster_key), lit_equal=(inp['lit_px'] == a.lit_px), n_elements_equal=(T['n_elements'] == a.n_elements),
                                 peak_equal=abs(inp['peak'] - a.peak) < 1e-12, integral_equal=abs(inp['integral_f32'] - a.integral_f32) < 1e-6,
                                 ordinary_raster_equal=(inp['ordinary']['raster_equal'] if inp['ordinary'] else None)))
    R = pd.DataFrame(rows)

    reg_ok = []
    import adp_patterns as PT
    for name, famp, fam in K.BASES:
        regs = {str(x['proposal_id']): x for x in PT._spat_layouts()[name]['layouts']}
        for pid, lid in [(f'{name}_s3_j0.00_regular', 'regular')] + [(f'{name}_s3_j0.20_B{b:03d}_c0', f'B{b:03d}_c0') for b in K.BLOCKS]:
            if pid not in regs or lid not in reg['baselines'][name]['layouts']:
                reg_ok.append(dict(baseline=name, pid=pid, present=False)); continue
            for ph in K.PHASES:
                T0 = RI._spat_table(dict(layout_id=f'{name}:{pid}', family=famp, phase_label=ph))
                mine = build_input(name, reg['baselines'][name]['layouts'][lid], ph, False)
                reg_ok.append(dict(baseline=name, pid=pid, phase=ph, present=True, raster_equal=(RI.raster_key((np.asarray(T0['pat']) > 0).astype(np.float32)) == mine['raster_key'])))
    G = pd.DataFrame(reg_ok)
    eq_cols = ['raster_equal', 'lit_equal', 'n_elements_equal', 'peak_equal', 'integral_equal']
    summ = dict(n_views=len(R), n_unique_raster_peak=int(R.drop_duplicates(['raster_key', 'peak_f32']).shape[0]), all_equal=bool(R[eq_cols].all().all()),
                protected_all=bool(R.protected_present.all()), movable_all=bool(R.movable_all_present.all()), max_integral_rel_err=float(R.integral_rel_err.max()), peak_max=float(R.peak.max()),
                s0_ordinary_raster_equal=bool(R[R.strategy == 'S0'].ordinary_raster_equal.all()), registered_spat03_checked=int(G.present.sum() if len(G) else 0),
                registered_spat03_equal=bool(G[G.present].raster_equal.all()) if len(G) else False, n_layouts={n: len(reg['baselines'][n]['layouts']) for n, _f, _g in K.BASES})
    summ['ok'] = bool(summ['n_views'] == 264 and summ['n_unique_raster_peak'] == 140 and summ['all_equal'] and summ['protected_all'] and summ['movable_all'] and summ['max_integral_rel_err'] <= 1e-6
                      and summ['peak_max'] <= 1.0 and summ['s0_ordinary_raster_equal'] and summ['registered_spat03_equal'])
    return R, summ
