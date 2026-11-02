# Check feasible code levels and geometric candidate ambiguity before imaging.

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd

import s3c_common as K

R0, W0 = 1.0, 1.0
DOT_LAYOUTS = {'R5x4': (5.0, 4.0), 'R8x4': (8.0, 4.0)}
LINE_PITCHES = (4.45, 6.3, 8.0)
RHO_GRID = np.round(np.arange(0.30, 0.951, 0.05), 2)
DELTA_GRID = np.round(np.arange(0.05, 1.001, 0.05), 2)
ANCHOR = dict(dots=0.6, lines=0.5)


def seen_dot_phases() -> pd.DataFrame:
    import sl_lib as SLL
    import sn_lib as SN
    import sp2_common as S2
    rows = [dict(u=u, v=v, src='sl_lib.LATIN8 (old B stage)') for u, v in SLL.LATIN8]
    for p in SN.PHASES:
        u, v = SN.quantiles(p); rows.append(dict(u=u, v=v, src=f'sn_lib phase {p} (S1 / old A)'))
    pos = SLL.positions()
    M = pd.read_csv(K.S12 / 'manifest.csv', low_memory=False)
    for lab in sorted(set(M[(M.family == 'point') & M.phase_label.astype(str).str.startswith('pos')].phase_label)):
        u, v = pos[int(lab[3:])]; rows.append(dict(u=u, v=v, src=f'S2 {lab}'))
    for p in S2.registered_phases('dots'):
        rows.append(dict(u=p['u'], v=p['v'], src=f"S4 SPAT {p['phase_id']}"))
    P = pd.read_csv(K.S345_PLAN, low_memory=False)
    for _, r in P[P.family == 'point'].drop_duplicates(['section', 'phase_u', 'phase_v']).dropna(subset=['phase_u']).iterrows():
        rows.append(dict(u=float(r.phase_u), v=float(r.phase_v), src=f'{r.section} {r.phase_label}'))
    man = K.load_json(K.ROOT / 'outputs' / 'encoding_validation' / 'manifest.json')
    for ph in (man.get('phases') or []):
        if isinstance(ph, dict) and 'u' in ph:
            rows.append(dict(u=float(ph['u']), v=float(ph['v']), src=f"ENC-05 {ph.get('label', ph.get('name', ''))}"))
    d = pd.DataFrame(rows); d['u'] = d.u % 1.0; d['v'] = d.v % 1.0
    return d.drop_duplicates(['u', 'v'])


def seen_line_phases() -> pd.DataFrame:
    import lc_r4_common as RC
    import sp2_common as S2
    rows = [dict(phase=float(x), src='line01r seen / formal / C2 / R2 / R4') for x in RC.seen_phases()]
    rows += [dict(phase=float(x), src=f'R4 x32 candidate {x}') for x in (np.array(RC.PHASE_CANDIDATES_X32) / 32.0)]
    for p in S2.registered_phases('lines'):
        rows.append(dict(phase=float(p['phase_frac']), src=f"S4 SPAT {p['phase_id']}"))
    P = pd.read_csv(K.S345_PLAN, low_memory=False)
    for _, r in P[P.family == 'line'].drop_duplicates(['section', 'phase_frac']).dropna(subset=['phase_frac']).iterrows():
        rows.append(dict(phase=float(r.phase_frac), src=f'{r.section} {r.phase_label}'))
    M = pd.read_csv(K.S12 / 'manifest.csv', low_memory=False)
    for _, r in M[M.family == 'line'].drop_duplicates(['section', 'phase_frac']).dropna(subset=['phase_frac']).iterrows():
        rows.append(dict(phase=float(r.phase_frac), src=f'{r.section} {r.phase_label}'))
    d = pd.DataFrame(rows); d['phase'] = ((d.phase + 0.5) % 1.0) - 0.5
    return d.drop_duplicates('phase')


def _torus_d(a, b):
    d = np.abs(a - b); return np.minimum(d, 1.0 - d)


def choose_dot_phases(seen: pd.DataFrame, n=8, grid=64) -> list:
    g = (np.arange(grid) + 0.5) / grid; U, V = np.meshgrid(g, g); cand = np.column_stack([U.ravel(), V.ravel()])
    S = seen[['u', 'v']].to_numpy(); chosen = []
    for _ in range(n):
        pts = np.vstack([S] + ([np.array(chosen)] if chosen else []))
        dmin = np.min(np.hypot(_torus_d(cand[:, None, 0], pts[None, :, 0]), _torus_d(cand[:, None, 1], pts[None, :, 1])), axis=1)
        chosen.append(cand[int(np.argmax(dmin))].tolist())
    return chosen


def choose_line_phases(seen: pd.DataFrame, n=8, grid=256) -> list:

    cand = (np.arange(grid) + 0.5) / grid - 0.5; S = seen.phase.to_numpy(); chosen = []
    dmin = np.min(_torus_d(cand[:, None] + 0.5, S[None, :] + 0.5), axis=1)
    for j in range(n):
        lo, hi = -0.5 + j / n, -0.5 + (j + 1) / n; m = (cand >= lo) & (cand < hi); c = cand[m]; d = dmin[m]
        best = np.flatnonzero(d >= d.max() - 1e-12); centre = 0.5 * (lo + hi)
        chosen.append(float(c[best[np.argmin(np.abs(c[best] - centre))]]))
    return chosen


def phases() -> dict:
    sd, sl = seen_dot_phases(), seen_line_phases()
    dp, lp = choose_dot_phases(sd), choose_line_phases(sl)
    rows = []
    for i, (u, v) in enumerate(dp):
        d = np.hypot(_torus_d(sd.u.to_numpy(), u), _torus_d(sd.v.to_numpy(), v)); j = int(np.argmin(d))
        rows.append(dict(family='dots', phase_id=f'N{i}', u=u, v=v, phase_frac=None, min_dist_frac=float(d[j]), nearest_seen=sd.src.iloc[j],
                         min_dist_mm_R5x4=float(np.min(np.hypot(_torus_d(sd.u.to_numpy(), u) * 5.0, _torus_d(sd.v.to_numpy(), v) * 4.0))),
                         min_dist_mm_R8x4=float(np.min(np.hypot(_torus_d(sd.u.to_numpy(), u) * 8.0, _torus_d(sd.v.to_numpy(), v) * 4.0)))))
    for i, p in enumerate(lp):
        d = _torus_d(sl.phase.to_numpy() + 0.5, p + 0.5); j = int(np.argmin(d))
        rows.append(dict(family='lines', phase_id=f'N{i}', u=None, v=None, phase_frac=p, min_dist_frac=float(d[j]), nearest_seen=sl.src.iloc[j],
                         **{f'min_dist_mm_s{s:g}': float(d[j] * s) for s in LINE_PITCHES}))
    D = pd.DataFrame(rows); K.save_csv(D, K.RESULTS / 'precheck_new_phases.csv')
    K.save_csv(sd, K.RESULTS / 'precheck_seen_dot_phases.csv'); K.save_csv(sl, K.RESULTS / 'precheck_seen_line_phases.csv')
    return dict(n_seen_dot=len(sd), n_seen_line=len(sl), dots=dp, lines=lp)


def new_phases() -> dict:
    D = pd.read_csv(K.RESULTS / 'precheck_new_phases.csv')
    return dict(dots=[(r.phase_id, float(r.u), float(r.v)) for r in D[D.family == 'dots'].itertuples()],
                lines=[(r.phase_id, float(r.phase_frac)) for r in D[D.family == 'lines'].itertuples()])


def dot_radii(rho: float) -> tuple[float, float]:
    rl = R0 * math.sqrt(2.0 / (1.0 + rho * rho)); return rho * rl, rl


def dot_raster(layout: str, u: float, v: float, radii) -> dict:

    import ep_patterns as EP
    p_par, p_perp = DOT_LAYOUTS[layout]
    k = 1 if len(radii) == 1 else 2
    c, cls = EP.lattice_centres(p_par, p_perp, u, v, 'R', 0.0, 2, 1)
    if k == 1:
        cls = np.zeros(len(c), np.int64)
    pat = EP.render(c, cls, list(radii))
    return dict(c=c, cls=cls, pat=pat)


def per_dot_areas(c, cls, radii) -> np.ndarray:
    import ep_patterns as EP
    out = np.zeros(len(c))
    for i, (x, y) in enumerate(c):
        out[i] = float(EP._draw_dots(EP.IMG, np.array([[x, y]]), radii[int(cls[i])] / EP.mmpp()).sum())
    return out


def resource(pat) -> dict:
    Q0 = float(np.asarray(pat, np.float64).sum()); v = min(1.0, K.Q_BUDGET / Q0) if Q0 > 0 else float('nan')
    I = (np.asarray(pat, np.float32) * np.float32(v)).astype(np.float32); Q = float(I.sum(dtype=np.float64))
    return dict(Q0=Q0, v=v, peak_f32=float(np.float32(v)), Q=Q, usage=Q / K.Q_BUDGET, limit=('peak_limited' if v >= 1.0 else 'integral_limited'),
                budget_ok=bool(v >= 1.0 or abs(Q - K.Q_BUDGET) / K.Q_BUDGET <= 1e-6))


def line_raster(pitch: float, phase: float, wn: float | None, ww: float | None) -> dict:
    import lc_pattern as LP
    if wn is None:
        pat, meta = LP.uncoded_pattern(pitch, 0.0, phase, W0); return dict(pat=pat, meta=meta, lit=None)
    pat, meta, parts = LP.coded_pattern(pitch, 0.0, phase, wn, ww)
    import m2_pattern as M2
    _, mn = M2.generate_pattern('olines', spacing_mm=pitch, feature_size_mm=wn, line_angle_deg=0.0, line_phase_frac=phase)
    _, mw = M2.generate_pattern('olines', spacing_mm=pitch, feature_size_mm=ww, line_angle_deg=0.0, line_phase_frac=phase)
    ids = np.asarray(meta['olines']['ids'], np.int64); cls = LP.class_of_id(ids)
    seg = lambda c, k: (mw if c == 1 else mn)['olines']['segments'][str(int(k))]
    lit = np.array([seg(c, k)['mean_lit_width_px'] for k, c in zip(ids, cls)])
    wpx = np.where(cls == 1, ww, wn) / K.NOMINAL_MM_PER_PX; uc = np.array([seg(c, k)['start_uv'][0] for k, c in zip(ids, cls)])
    clipped = (uc - wpx / 2 < -0.5) | (uc + wpx / 2 > 511.5)
    return dict(pat=pat, meta=meta, ids=ids, cls=cls, lit=lit, clipped=clipped)


def sd_logarea() -> dict:

    z = K.load_npz(K.CACHE / 'dots_areas.npz'); D = pd.read_csv(K.RESULTS / 'dots_images.csv', low_memory=False)
    keys = list(z['image_keys']); out = {}
    for s in (0.0, 25.0, 35.0):
        ids = [keys.index(k) for k in D[(D.layout == 'R5x4') & (D.regime.isin(['main_common_budget', 'coded_high'])) & (D.sigma == s)].image_key]
        m = np.isin(z['key'], ids); la = np.log(z['area'][m].astype(float)); t = z['src_true'][m]
        out[s] = float(math.sqrt(0.5 * (la[t == 0].var() + la[t == 1].var())))
    return out


def sizes() -> dict:
    ph = new_phases(); sd = sd_logarea(); rows = []
    import ep_detect as ED
    import ep_patterns as EP
    import m4_reconstruction as M4
    for rho in RHO_GRID:
        rs, rl = dot_radii(float(rho)); span = 2.0 * math.log(1.0 / rho)
        for lay in DOT_LAYOUTS:
            for pid, u, v in ph['dots']:
                R = dot_raster(lay, u, v, [rs, rl]); Rp = dot_raster(lay, u, v, [R0])
                a = per_dot_areas(R['c'], R['cls'], [rs, rl]); res, resp = resource(R['pat']), resource(Rp['pat'])
                areas_p, = [M4.pattern_features(R['pat'], 'coded_dots', return_areas=True)[2]]
                cfg = ED.coded_cfg([rs, rl], 2, EP.mmpp()); pcls = M4.projector_size_classes(areas_p, cfg)
                import r5_detect as RD
                pdot, _ = RD.projector_dot_index(np.asarray(R['pat'], np.float32), R['c'])
                ok_cls = bool(len(pcls) == len(R['cls']) and np.array_equal(pcls, R['cls'][pdot]))
                g = EP.gaps(R['c'], R['cls'], [rs, rl])
                rows.append(dict(family='dots', level_value=float(rho), layout=lay, phase=pid, r_s=rs, r_l=rl, n_small=int((R['cls'] == 0).sum()), n_large=int((R['cls'] == 1).sum()),
                                 area_small_min=float(a[R['cls'] == 0].min()), area_small_max=float(a[R['cls'] == 0].max()), area_large_min=float(a[R['cls'] == 1].min()),
                                 area_large_max=float(a[R['cls'] == 1].max()), raster_separable=bool(a[R['cls'] == 0].max() < a[R['cls'] == 1].min()), projector_classes_ok=ok_cls,
                                 Q0=res['Q0'], Q0_plain=resp['Q0'], Q0_residual=res['Q0'] / resp['Q0'] - 1.0, peak=res['peak_f32'], peak_plain=resp['peak_f32'], limit=res['limit'],
                                 edge_gap_mm=float(DOT_LAYOUTS[lay][1] - rs - rl), edge_gap_min_raster_mm=g['edge_gap_min_mm'], span_design=span,
                                 dprime_pred_s0=span / sd[0.0], dprime_pred_s25=span / sd[25.0], dprime_pred_s35=span / sd[35.0], within_s1_range=bool(rs >= 0.5 - 1e-9 and rl <= 1.5 + 1e-9)))
    for dl in DELTA_GRID:
        wn, ww = W0 - dl / 2.0, W0 + dl / 2.0
        for s in LINE_PITCHES:
            for pid, p in ph['lines']:
                R = line_raster(s, p, wn, ww); Rp = line_raster(s, p, None, None); res, resp = resource(R['pat']), resource(Rp['pat'])
                ok = ~R['clipped']; ln, lw = R['lit'][(R['cls'] == 0) & ok], R['lit'][(R['cls'] == 1) & ok]
                L = 0.5 * ww / K.NOMINAL_MM_PER_PX + 3.0 * 2.0046884346862006 + 1.0; pitch_px = s / K.NOMINAL_MM_PER_PX
                rows.append(dict(family='lines', level_value=float(dl), layout=f'{s:g}', phase=pid, w_n=wn, w_w=ww, n_small=int(len(ln)), n_large=int(len(lw)), n_clipped=int(R['clipped'].sum()),
                                 area_small_min=float(ln.min()), area_small_max=float(ln.max()), area_large_min=float(lw.min()), area_large_max=float(lw.max()),
                                 raster_separable=bool(ln.max() < lw.min()), projector_classes_ok=True, design_px_diff=dl / K.NOMINAL_MM_PER_PX,
                                 Q0=res['Q0'], Q0_plain=resp['Q0'], Q0_residual=res['Q0'] / resp['Q0'] - 1.0, peak=res['peak_f32'], peak_plain=resp['peak_f32'], limit=res['limit'],
                                 edge_gap_mm=float(s - W0), profile_window_margin_px=float(pitch_px - 0.5 * ww / K.NOMINAL_MM_PER_PX - L), within_s1_range=bool(wn >= 0.5 - 1e-9 and ww <= 1.5 + 1e-9)))
    D = pd.DataFrame(rows); K.save_csv(D, K.RESULTS / 'precheck_size_scan.csv')
    lev = choose_levels(D, sd)
    K.save_json(K.RESULTS / 'precheck_levels.json', dict(levels=lev, sd_logarea_seen=sd, t_local=K.now_local()))
    return lev


def choose_levels(D: pd.DataFrame, sd: dict) -> dict:

    out = {}
    d = D[D.family == 'dots'].groupby('level_value').agg(sep=('raster_separable', 'all'), pcls=('projector_classes_ok', 'all'), dp0=('dprime_pred_s0', 'first'), rs=('r_s', 'first'),
                                                         gap=('edge_gap_mm', 'min'), span=('span_design', 'first')).reset_index()
    weak = d[(d.dp0 >= 4.0) & d.sep & d.pcls].level_value.max()
    strong = d[(d.rs >= 0.5 - 1e-9) & (d.gap >= 1.5) & d.sep & d.pcls].level_value.min()
    sw, ss = float(d[d.level_value == weak].span.iloc[0]), float(d[d.level_value == strong].span.iloc[0])
    target = math.exp(-math.sqrt(sw * ss) / 2.0); grid = d.level_value.to_numpy(); mid = float(grid[np.argmin(np.abs(grid - target))])
    mid = ANCHOR['dots'] if abs(ANCHOR['dots'] - target) <= 0.05 + 1e-9 else mid
    out['dots'] = dict(weak=float(weak), mid=mid, strong=float(strong), mid_target=target, radii={k: dot_radii(v) for k, v in dict(weak=weak, mid=mid, strong=strong).items()})
    l = D[D.family == 'lines'].groupby('level_value').agg(sep=('raster_separable', 'all'), px=('design_px_diff', 'first'), wn=('w_n', 'first'), ww=('w_w', 'first')).reset_index()
    weak_l = l[l.sep & (l.px >= 1.5 - 1e-9)].level_value.min()
    strong_l = l[(l.wn >= 0.5 - 1e-9) & (l.ww <= 1.5 + 1e-9) & l.sep].level_value.max()
    rw = (W0 + weak_l / 2) / (W0 - weak_l / 2); rs_ = (W0 + strong_l / 2) / (W0 - strong_l / 2)
    rt = math.exp(math.sqrt(math.log(rw) * math.log(rs_))); dt = 2.0 * (rt - 1.0) / (rt + 1.0)
    grid = l.level_value.to_numpy(); mid_l = float(grid[np.argmin(np.abs(grid - dt))])
    mid_l = ANCHOR['lines'] if abs(ANCHOR['lines'] - dt) <= 0.05 + 1e-9 else mid_l
    out['lines'] = dict(weak=float(weak_l), mid=mid_l, strong=float(strong_l), mid_target=dt, widths={k: (W0 - v / 2, W0 + v / 2) for k, v in dict(weak=weak_l, mid=mid_l, strong=strong_l).items()})
    return out


def geometry() -> pd.DataFrame:
    import ep_detect as ED
    import ep_patterns as EP
    import lc_pattern as LP
    import m4_lines as ML
    import r5_detect as RD
    import rdy_decode as DC
    import rdy_platform as P
    import sn_lib as SN
    S = P.s2(); ph = new_phases(); lev = K.load_json(K.RESULTS / 'precheck_levels.json')['levels']
    rs, rl = lev['dots']['radii']['mid']; wn, ww = lev['lines']['widths']['mid']; rows = []
    for surf in K.MAIN:
        mesh = P.mesh(surf)
        for lay in DOT_LAYOUTS:
            for pid, u, v in ph['dots']:
                R = dot_raster(lay, u, v, [rs, rl]); binpat = np.asarray(R['pat'], np.float32)
                cfg = DC.window_cfg_dots(ED.coded_cfg([rs, rl], 2, EP.mmpp())); proj = DC.projector_side_window(binpat, cfg, 'coded_dots')
                exp = SN.expected_dots(mesh, R['c']); vis = exp['cam_inner']
                pd_, _ = RD.projector_dot_index(binpat, R['c']); dot2p = {int(d): p for p, d in enumerate(pd_)}
                ids = np.asarray(proj['ids'])
                feats = exp['uv'][vis]; dots = np.flatnonzero(vis); true_id = np.array([ids[dot2p[int(i)]] if int(i) in dot2p else -1 for i in dots])
                onehot = np.zeros((len(dots), 2), bool); onehot[np.arange(len(dots)), R['cls'][dots]] = True
                tg = ED.track_candidates(feats, proj, cfg, None); tc = ED.track_candidates(feats, proj, cfg, onehot)
                tin = np.array([t in g for t, g in zip(true_id, tg['geom_ids'])]); tcls = np.array([t in a for t, a in zip(true_id, tc['class_ids'])])
                rows.append(dict(family='dots', layout=lay, surface=surf, phase=pid, n_visible=int(len(dots)), A_p=float((tg['n_geom'] >= 2).mean()), true_in_window=float(tin.mean()),
                                 n_geom_mean=float(tg['n_geom'].mean()), n_geom_ge3=float((tg['n_geom'] >= 3).mean()),
                                 C=float(((tc['n_class'] == 1) & tcls).mean()), class_left_ge2=float((tc['n_class'] >= 2).mean())))
        for s in LINE_PITCHES:
            for pid, p in ph['lines']:
                pat, meta, _ = LP.coded_pattern(s, 0.0, p, wn, ww); fam = ML.line_family(meta['olines']); cfg = S.line_cfg(1.0)
                ref = S.T.line_reference(mesh, meta['olines'], 1.0, pattern=np.asarray(pat, np.float32)); Rr = ref['rows']
                inner = np.asarray(Rr['inner'], bool); uv = np.column_stack([np.asarray(Rr['u'], float), np.asarray(Rr['v'], float)])[inner]; kk = np.asarray(Rr['k'], np.int64)[inner]
                cand = ML.line_candidates(uv, fam, cfg, centrelines=ML.raster_centrelines(np.asarray(pat, np.float32), fam))
                adm = cand['admissible']; ids = np.asarray(cand['ids'], np.int64); col = {int(k): j for j, k in enumerate(ids)}
                tin = np.array([adm[i, col[int(k)]] if int(k) in col else False for i, k in enumerate(kk)])
                same = adm & (LP.class_of_id(ids)[None, :] == LP.class_of_id(kk)[:, None]); ng = adm.sum(axis=1); nc = same.sum(axis=1)
                rows.append(dict(family='lines', layout=f'{s:g}', surface=surf, phase=pid, n_visible=int(len(kk)), A_p=float((ng >= 2).mean()), true_in_window=float(tin.mean()),
                                 n_geom_mean=float(ng.mean()), n_geom_ge3=float((ng >= 3).mean()), C=float(((nc == 1) & tin).mean()), class_left_ge2=float((nc >= 2).mean())))
    D = pd.DataFrame(rows); K.save_csv(D, K.RESULTS / 'precheck_geometry.csv')
    S_ = D.groupby(['family', 'layout']).agg(n=('n_visible', 'sum'), A_p=('A_p', 'mean'), A_p_min=('A_p', 'min'), A_p_max=('A_p', 'max'), C=('C', 'mean'), C_min=('C', 'min'),
                                             true_in_window=('true_in_window', 'mean'), n_geom_ge3=('n_geom_ge3', 'mean'), class_left_ge2=('class_left_ge2', 'mean')).reset_index()
    sel = {}
    for fam, g in S_.groupby('family'):
        g = g.sort_values('A_p'); l1 = g[g.C >= 0.9].sort_values('A_p').iloc[-1].layout if (g.C >= 0.9).any() else None
        rest = g[g.layout != l1]; l2 = rest.iloc[0].layout if len(rest) else None
        sel[fam] = dict(layout1=l1, layout2=l2, optional=[x for x in g.layout if x not in (l1, l2)])
    K.save_csv(S_, K.RESULTS / 'precheck_geometry_summary.csv'); K.save_json(K.RESULTS / 'precheck_layouts.json', dict(selected=sel, t_local=K.now_local()))
    return S_


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('phases', 'all'):
        print(json.dumps(K.clean(phases()), indent=1)[:2000])
    if what in ('sizes', 'all'):
        print(json.dumps(K.clean(sizes()), indent=1))
    if what in ('geometry', 'all'):
        print(geometry().to_string())
