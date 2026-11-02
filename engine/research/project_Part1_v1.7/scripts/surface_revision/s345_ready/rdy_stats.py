# Aggregate pattern-study results using the registered comparison keys.

from __future__ import annotations

import numpy as np
import pandas as pd

import rdy_common as K

FOUR = {
    'dots_parallel': dict(sd='D_r1_p6x4', ss='D_r1_p10x4', ld='D_r1.5_p6x4', ls='D_r1.5_p10x4'),
    'dots_perp': dict(sd='D_r1_p8x4', ss='D_r1_p8x6', ld='D_r1.5_p8x4', ls='D_r1.5_p8x6'),
    'lines': dict(sd='L_w0.5_s4.45', ss='L_w0.5_s8', ld='L_w1.5_s4.45', ls='L_w1.5_s8'),
}
D_WIDTH = {s: (f'L_w0.5_s{s}', f'L_w1.5_s{s}') for s in ('4.45', '6.3', '8')}
METRICS = ['glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'glob_pt_rmse', 'glob_pt_p95', 'glob_gross_rate', 'glob_surf_rmse', 'glob_id_correct', 'glob_id_wrong', 'glob_id_unconf', 'glob_recovery',
           'loc_Ctau', 'loc_Cvalid', 'loc_pt_rmse', 'loc_pt_p95', 'loc_surf_rmse', 'loc_id_wrong', 'loc_recovery', 'n_valid', 'id_wrong', 'id_correct', 'id_unconf']
MAIN_METRIC = 'glob_Ctau'
COMPLETE = ('ok', 'no_output')


def frame(plan_rows: pd.DataFrame, records: dict) -> pd.DataFrame:

    rows = []
    for r in plan_rows.to_dict('records'):
        rec = records.get(r['decode_key'])
        st = rec['state'] if rec else 'missing'
        out = dict(r, state=st)
        if rec:
            m = rec['metrics']
            for k in METRICS:
                out[k] = m.get(k)
            if st == 'no_output':
                for k in METRICS:
                    if k.endswith(('Ctau', 'Cvalid', 'cov_interp')):
                        out[k] = 0.0
            ex = rec.get('extras') or {}
            out['guard_fallback'] = (ex.get('guard') or {}).get('fallback')
            lc = (ex.get('lc') or {}).get(r['arm']) or {}
            out['image_fallback'] = lc.get('image_fallback'); out['new_wrong'] = (lc.get('new') or {}).get('wrong'); out['new_n'] = (lc.get('new') or {}).get('n')
            lab = (ex.get('labels') or {}).get('coded' if r['arm'] == 'ENC_coded' else 'ignore_class') or {}
            out['lab_wrong_identity'] = lab.get('n_wrong_identity'); out['lab_identity_recovery_full'] = lab.get('identity_recovery_full')
            si = ex.get('source_identity') or {}
            out['src_correct'] = si.get('correct'); out['src_wrong'] = si.get('wrong'); out['src_undet'] = si.get('undeterminable')
            dsp = ex.get('displacement') or {}
            out['max_abs_disp_mm'] = dsp.get('max_abs_disp_mm')
        rows.append(out)
    f = pd.DataFrame(rows)
    for c in METRICS + ['max_abs_disp_mm']:
        if c in f:
            f[c] = pd.to_numeric(f[c], errors='coerce')
    return f


def _desc(e: np.ndarray) -> dict:
    e = np.asarray(e, np.float64)
    return dict(n=int(len(e)), mean=float(e.mean()), median=float(np.median(e)), p5=float(np.percentile(e, 5)), p95=float(np.percentile(e, 95)), min=float(e.min()), max=float(e.max()),
                n_pos=int((e > 0).sum()), n_neg=int((e < 0).sum()), n_zero=int((e == 0).sum()))


def _state_counts(g: pd.DataFrame) -> dict:
    return {s: int((g.state == s).sum()) for s in ('ok', 'no_output', 'missing', 'technical', 'stale')}


# Aggregate encoding effects within phases and separate same-image class comparisons.
def s3(f: pd.DataFrame, metrics=('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'id_wrong', 'id_correct')) -> dict:
    x = f[f.section == 'S3']
    out = dict(same_image=[], cross_image=[], events=[], development=[])
    pairs = [('point', 'ENC_coded', 'ENC_ignore')] + [('line', a, 'LC_ignore') for a in ('LC_use', 'LC_R', 'LC_A')]
    for fam, arm, base in pairs:
        y = x[(x.family == fam) & x.arm.isin([arm, base])]
        if y.empty:
            continue
        img = y.groupby('image_key')
        per = []
        for k, g in img:
            if set(g.arm) != {arm, base}:
                continue
            a = g[g.arm == arm].iloc[0]; b = g[g.arm == base].iloc[0]
            per.append(dict(image_key=k, surface=a.surface, layout=(a.layout_id or a.view), regime=a.regime, sigma=float(a.sigma), phase=a.phase_label, rep=a.rep,
                            complete=bool(a.state in COMPLETE and b.state in COMPLETE), **{f'd_{m}': (a[m] - b[m]) if (a.state in COMPLETE and b.state in COMPLETE) else np.nan for m in metrics}))
        p = pd.DataFrame(per)
        if p.empty:
            continue
        for (surf, lay, reg, sig), g in p.groupby(['surface', 'layout', 'regime', 'sigma']):
            base_rec = dict(contrast=f'{arm} - {base}', kind='same image (decoding-information effect)', family=fam, surface=surf, layout=lay, regime=reg, sigma=sig, n_planned_images=int(len(g)),
                            n_complete=int(g.complete.sum()))
            if not g.complete.all():
                out['same_image'].append(dict(base_rec, state='incomplete', reason=f'{int((~g.complete).sum())} of {len(g)} planned images not complete'))
            else:
                out['same_image'].append(dict(base_rec, state='ok', n_phases=int(g.phase.nunique()), n_realisations=int(g.rep.nunique()),
                                              **{m: _desc(g[f'd_{m}'].to_numpy()) for m in metrics if g[f'd_{m}'].notna().all()},
                                              undefined=[m for m in metrics if not g[f'd_{m}'].notna().all()],
                                              quantile_note='p5 / p95 / min / max = descriptive quantiles of the per-image differences (finite sample), not a confidence interval'))
        for r in p[p.complete].to_dict('records'):
            out['development'].append(dict(contrast=f'{arm} - {base}', **r))

    d = x[(x.family == 'point') & (x.arm == 'ENC_coded')]
    for (surf, lay, ph, sig, rep), g in d.groupby(['surface', 'layout_id', 'phase_label', 'sigma', 'rep']):
        c = g[g.code.str.startswith('coded2') & g.regime.isin(['main_common_budget', 'coded_high'])]; pl = g[g.code.str.startswith('plain') & (g.regime == 'main_common_budget')]
        if len(c) == 1 and len(pl) == 1:
            a, b = c.iloc[0], pl.iloc[0]; ok = a.state in COMPLETE and b.state in COMPLETE
            out['cross_image'].append(dict(contrast='coded2 (ENC_coded) - plain, common budget Q 11695', kind='different images: pattern + brightness (not a pure code effect)', surface=surf, layout=lay,
                                           phase=ph, sigma=float(sig), rep=rep, complete=bool(ok), **{f'd_{m}': (a[m] - b[m]) if ok else None for m in metrics}))
        hi = g[g.regime == 'coded_high']; lo = g[g.regime == 'coded_low']
        if len(hi) == 1 and len(lo) == 1:
            a, b = hi.iloc[0], lo.iloc[0]; ok = a.state in COMPLETE and b.state in COMPLETE
            out['cross_image'].append(dict(contrast='coded_high - coded_low (peak-matched brightness)', kind='different images: brightness', surface=surf, layout=lay, phase=ph, sigma=float(sig), rep=rep,
                                           complete=bool(ok), **{f'd_{m}': (a[m] - b[m]) if ok else None for m in metrics}))
    ln = x[x.family == 'line']
    for (surf, ph, sig), g in ln.groupby(['surface', 'phase_label', 'sigma']):
        u = g[(g.view == 'S3_line_coding_V5') & (g.regime == 'equal_integral')]
        for v in ('S3_line_coding_V2', 'S3_line_coding_V3'):
            c = g[(g.view == v) & (g.arm == 'LC_R')]
            if len(u) == 1 and len(c) == 1:
                a, b = u.iloc[0], c.iloc[0]; ok = a.state in COMPLETE and b.state in COMPLETE
                out['cross_image'].append(dict(contrast=f'V5 45 deg uncoded P4_ref - {v[-2:]} coded LC_R (equal integral)', kind='different pattern AND decoder parameters: scheme comparison only',
                                               surface=surf, phase=ph, sigma=float(sig), complete=bool(ok), **{f'd_{m}': (a[m] - b[m]) if ok else None for m in metrics}))
    for (fam, surf, sig), g0 in x.groupby(['family', 'surface', 'sigma']):
        g = g0[g0.state.isin(COMPLETE)]
        if fam == 'point':
            pl = g0[(g0.arm == 'ENC_coded') & g0.code.str.startswith('coded2')]; gg = g[(g.arm == 'ENC_coded') & g.code.str.startswith('coded2')]
            out['events'].append(dict(family=fam, surface=surf, sigma=float(sig), event='guard fallback (coded2 images)', n_planned_images=int(pl.image_key.nunique()), n_images=int(gg.image_key.nunique()),
                                      n_events=int(gg.drop_duplicates('image_key').guard_fallback.fillna(False).astype(bool).sum()), complete=bool(len(gg) == len(pl))))
        else:
            for arm in ('LC_R', 'LC_A'):
                pl = g0[g0.arm == arm]; gg = g[g.arm == arm]
                out['events'].append(dict(family=fam, surface=surf, sigma=float(sig), event=f'{arm} whole-image fallback', n_planned_images=int(len(pl)), n_images=int(len(gg)),
                                          n_events=int(gg.image_fallback.fillna(False).astype(bool).sum()), complete=bool(len(gg) == len(pl))))
            for arm in ('LC_use', 'LC_R', 'LC_A'):
                pl = g0[g0.arm == arm]; gg = g[g.arm == arm]
                out['events'].append(dict(family=fam, surface=surf, sigma=float(sig), event=f'{arm} newly accepted features with wrong identity (vs LC_ignore)', n_planned_images=int(len(pl)),
                                          n_images=int(len(gg)), n_events=int(pd.to_numeric(gg.new_wrong, errors='coerce').fillna(0).sum()),
                                          n_new=int(pd.to_numeric(gg.new_n, errors='coerce').fillna(0).sum()), complete=bool(len(gg) == len(pl))))
    return out


# Aggregate layout results by phase and proposal block using shared strategy views.
def s4(f: pd.DataFrame, plan_s4: pd.DataFrame, metrics=('glob_Ctau', 'loc_Ctau', 'glob_Cvalid', 'glob_pt_rmse', 'id_wrong'), seed=20261011) -> dict:
    x = f[f.section == 'S4'].copy()
    x['baseline'] = x.layout_id.str.split(':').str[0]
    out = dict(per_phase=[], blocks=[], summary=[], sharing={}, development=[],
               definition='block = (surface, baseline, sigma, strategy, layout): the registered layout at its 4 phases, each paired with S0 of the same surface / baseline / sigma / phase; '
                          'block value = median of the 4 phase differences; summary per (baseline, sigma, strategy) = equal-weight median over complete blocks; a layout shared by several '
                          'strategies is ONE decode and appears once per strategy (no extra independent sample)')
    for (surf, b, sig, ph), g in x.groupby(['surface', 'baseline', 'sigma', 'phase_label']):
        s0 = g[g.strategy.str.startswith('S0')]
        for (strat, lay), h in g[~g.strategy.str.startswith('S0')].groupby(['strategy', 'layout_id']):
            r = h.iloc[0]
            ok = len(s0) == 1 and len(h) == 1 and s0.iloc[0].state in COMPLETE and r.state in COMPLETE
            rec = dict(surface=surf, baseline=b, sigma=float(sig), phase=ph, strategy=strat, layout_id=lay, decode_key=r.decode_key, complete=bool(ok),
                       views_of_this_decode=int((g.decode_key == r.decode_key).sum()), max_abs_disp_mm=r.get('max_abs_disp_mm'))
            for m in metrics:
                rec[f'd_{m}'] = float(r[m] - s0.iloc[0][m]) if ok and pd.notna(r[m]) and pd.notna(s0.iloc[0][m]) else None
            out['per_phase'].append(rec)
    pp = pd.DataFrame(out['per_phase'])
    if not pp.empty:
        for (surf, b, sig, strat, lay), g in pp.groupby(['surface', 'baseline', 'sigma', 'strategy', 'layout_id']):
            n_plan = int(plan_s4[(plan_s4.surface == surf) & (plan_s4.layout_id == lay) & (plan_s4.sigma.astype(float) == sig) & (plan_s4.strategy == strat)].phase_label.nunique())
            rec = dict(surface=surf, baseline=b, sigma=sig, strategy=strat, layout_id=lay, n_phases_planned=n_plan, n_phases_complete=int(g.complete.sum()))
            if int(g.complete.sum()) < n_plan or not g.complete.all():
                rec.update(state='incomplete', reason=f'{int(g.complete.sum())} of {n_plan} planned phases complete')
            else:
                rec.update(state='ok', **{f'median4_{m}': float(np.median(g[f'd_{m}'].astype(float))) for m in metrics if g[f'd_{m}'].notna().all()})
            out['blocks'].append(rec)
        bl = pd.DataFrame(out['blocks'])
        bl['surface_scope'] = bl.surface
        both = pd.concat([bl, bl.assign(surface_scope='all')])
        for (scope, b, sig, strat), g in both.groupby(['surface_scope', 'baseline', 'sigma', 'strategy']):
            comp = g[g.state == 'ok']
            rec = dict(surface=scope, baseline=b, sigma=sig, strategy=strat, n_blocks_planned=int(len(g)), n_blocks_complete=int(len(comp)), n_surfaces=int(g.surface.nunique()))
            if len(comp) == 0 or len(comp) < len(g):
                rec.update(state='incomplete', reason=f'{len(comp)} of {len(g)} planned blocks complete (missing rows are not dropped)')
            else:
                for m in metrics:
                    c = f'median4_{m}'
                    if c in comp and comp[c].notna().all():
                        v = comp[c].to_numpy(float); rec[f'median_{m}'] = float(np.median(v)); rec[f'mean_{m}'] = float(np.mean(v))
                        if m == 'glob_Ctau':
                            rec.update(n_blocks_improved=int((v > 0).sum()), n_blocks_worsened=int((v < 0).sum()), n_blocks_equal=int((v == 0).sum()), per_block=[float(x) for x in v])
                        if len(v) >= 8:
                            rng = np.random.Generator(np.random.PCG64(seed)); bs = np.median(v[rng.integers(0, len(v), (10000, len(v)))], axis=1)
                            rec[f'resampling_interval_5_95_{m}'] = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
                rec['interval_note'] = 'descriptive resampling interval over the finite set of blocks (equal weight); not a validated noise-repeat stability'
                rec['state'] = 'ok'
            out['summary'].append(rec)
        out['development'] = pp[pp.complete].to_dict('records')

    p4 = plan_s4.copy(); p4['baseline'] = p4.layout_id.str.split(':').str[0]
    for b, g in p4[~p4.strategy.str.startswith('S0')].groupby('baseline'):
        sets = {s: set(h.layout_id) for s, h in g.groupby('strategy')}; names = sorted(sets)
        out['sharing'][b] = dict(views=int(len(g)), decodes=int(g.decode_key.nunique()), layouts_per_strategy={s: len(v) for s, v in sets.items()},
                                 jaccard={f'{a}|{c}': round(len(sets[a] & sets[c]) / max(1, len(sets[a] | sets[c])), 4) for i, a in enumerate(names) for c in names[i + 1:]})
    return out


def s4_generation() -> dict:
    p = K.ROOT / 'outputs' / 'spatial_layout' / 'spat03' / 'generation_summary.json'
    if not p.exists():
        return dict(note='generation_summary.json missing')
    g = K.load_json(p); out = {}
    for b, v in (g.get('summaries') or {}).items():
        if isinstance(v, dict):
            out[b] = {s: dict(n_candidates=w.get('n_candidates'), n_accepted=w.get('n_accepted'), acceptance_rate=w.get('acceptance_rate')) for s, w in v.items() if isinstance(w, dict) and 'n_candidates' in w}
    return dict(source=K.rel(p), per_baseline=out, note='projector-plane generation attempts / acceptance (SPAT-03); geometry independent, carried, not regenerated')


def s5(f: pd.DataFrame, plan_s5: pd.DataFrame, metric=MAIN_METRIC, seed=2026101204) -> dict:
    x = f[f.section == 'S5'].copy(); x['geometry_id'] = x.pair_role; x['phase_id'] = x.phase_label
    out = dict(contrasts=[], combos=[], development=[])
    for (surf, fam, sig), g in x.groupby(['surface', 'family', 'sigma']):
        pl = plan_s5[(plan_s5.surface == surf) & (plan_s5.family == fam) & (plan_s5.sigma.astype(float) == float(sig))]
        phases = sorted(pl.phase_label.unique()); reps = sorted(pl.rep.astype(int).unique())
        sets = [(k, v) for k, v in FOUR.items() if (k.startswith('dots') and fam == 'point') or (k == 'lines' and fam == 'line')]
        defs = [(k, dict(E_small={v['ss']: 1, v['sd']: -1}, E_large={v['ls']: 1, v['ld']: -1}, I={v['ls']: 1, v['ld']: -1, v['ss']: -1, v['sd']: 1})) for k, v in sets]
        if fam == 'line':
            defs.append(('line_D_width', {f'D_width_s{s}': {bb: 1, aa: -1} for s, (aa, bb) in D_WIDTH.items()}))
        for name, comps in defs:
            for cname, w in comps.items():
                base = dict(surface=surf, family=fam, sigma=float(sig), set=name, effect=cname, metric=metric, n_phases_planned=len(phases), n_reps_planned=len(reps))
                vals, why, dev = {}, None, []
                for ph in phases:
                    row = {}
                    for c in w:
                        cell = g[(g.geometry_id == c) & (g.phase_id == ph)]
                        planned = pl[(pl.pair_role == c) & (pl.phase_label == ph)]
                        if len(cell) < len(planned) or len(planned) == 0:
                            why = why or f'{ph}: cell {c} has {len(cell)} of {len(planned)} planned realisations'; row = None; break
                        if not cell.state.isin(COMPLETE).all():
                            why = why or f'{ph}: cell {c} technically incomplete ({_state_counts(cell)})'; row = None; break
                        v = pd.to_numeric(cell[metric], errors='coerce')
                        if v.isna().any():
                            why = why or f'{ph}: {metric} undefined in cell {c} (no output -> undefined error, not 0)'; row = None; break
                        row[c] = float(v.mean())
                    if row is not None:
                        vals[ph] = row; dev.append(dict(phase=ph, value=float(sum(wt * row[c] for c, wt in w.items()))))
                out['development'] += [dict(base, **d_) for d_ in dev]
                if len(vals) < len(phases):
                    out['contrasts'].append(dict(base, state=('undefined' if why and 'undefined' in why else 'incomplete'), n_phases_complete=len(vals), reason=why)); continue
                e = np.array([sum(wt * vals[ph][c] for c, wt in w.items()) for ph in phases])
                rng = np.random.Generator(np.random.PCG64(seed)); bs = e[rng.integers(0, len(e), size=(10000, len(e)))].mean(axis=1)
                out['contrasts'].append(dict(base, state='ok', n_phases_complete=len(e), mean=float(e.mean()), median=float(np.median(e)), n_pos=int((e > 0).sum()), n_neg=int((e < 0).sum()),
                                             ci_5_95=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))], per_phase=e.tolist()))
        for geo, h in g.groupby('geometry_id'):
            planned = pl[pl.pair_role == geo]
            per = h[h.state.isin(COMPLETE)].groupby('phase_id')[metric].mean()
            out['combos'].append(dict(surface=surf, family=fam, sigma=float(sig), geometry_id=geo, metric=metric, n_planned=int(len(planned)), n_complete=int(h.state.isin(COMPLETE).sum()),
                                      n_no_output=int((h.state == 'no_output').sum()), n_technical_or_missing=int((~h.state.isin(COMPLETE)).sum() + max(0, len(planned) - len(h))),
                                      state=('complete' if h.state.isin(COMPLETE).sum() == len(planned) else 'incomplete'), mean_over_phases=(float(per.mean()) if len(per) else None),
                                      median_over_phases=(float(per.median()) if len(per) else None)))
    return out
