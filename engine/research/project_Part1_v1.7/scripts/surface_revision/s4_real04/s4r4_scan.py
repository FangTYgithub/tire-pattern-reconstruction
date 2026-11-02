# Scan geometric constraint activity and compare candidate generation settings.

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import s4r4_common as H

K = H.K


class Geo:
    def __init__(self, name: str):
        import s3_common as C3
        import s3_generate as SG
        import sp2_generator as GEN
        import sp_geometry as G
        self.C3, self.GEN = C3, GEN
        self.bl = SG.baseline(name); self.ms = SG.sets()[name]; self.name = name; self.fam = self.bl['family']; self.p0 = float(self.bl['p0_mm'])
        ctx = GEN.context_v2(self.bl, self.ms, C3.J_STRONG * self.p0); self.ctx = ctx; self.mov = list(self.ms['movable']); self.h0 = float(self.ms['h0_mm'])
        if self.fam == 'lines':
            lo, hi = ctx['omega_ids']; ins = [x for x in sorted(ctx['ids']) if lo <= x <= hi]; self.q0 = np.array([ctx['q0'][x] for x in ins], np.float64); self.p_perp = None
        else:
            lo, hi = ctx['omega_ix']; ins = [int(i) for i in sorted(ctx['pos0']) if lo <= i <= hi]; self.q0 = np.array([ctx['pos0'][i] for i in ins], np.float64); self.p_perp = float(ctx['p_perp_mm'])
        pos = {k: i for i, k in enumerate(ins)}; self.mi = np.array([pos[k] for k in self.mov]); self.k = G.S.mmpp()

    def units(self, block: int, n: int) -> np.ndarray:
        U = np.empty((n, len(self.mov)))
        for c in range(n):
            U[c] = self.GEN.unit_vector(len(self.mov), np.random.default_rng(self.C3.seed_s3(self.bl['index'], block, c)))[0]
        return U

    def metrics(self, U: np.ndarray, j: float) -> dict:
        import sp2_common as S2
        a_px = S2.px(j * self.p0); D = U * a_px; Q = np.repeat(self.q0[None, :], len(U), axis=0); Q[:, self.mi] += D
        g = np.diff(Q, axis=1) * self.k; gmin = g.min(axis=1); gmax = g.max(axis=1)
        h = gmax / 2.0 if self.fam == 'lines' else 0.5 * np.sqrt(gmax ** 2 + self.p_perp ** 2)
        return dict(g_min=gmin, g_max=gmax, h=h, rms=np.sqrt(np.mean(D ** 2, axis=1)) * self.k)

    def edge_ok(self, j: float):
        import sp_dots_columns as DC
        import sp_geometry as G
        a = j * self.p0
        safe = set(G.line_movable_ids(self.p0, a, list(self.ctx['ids']))['movable']) if self.fam == 'lines' else set(DC.movable_columns(self.p0, a)['ix_movable'])
        bad = [m for m in self.mov if m not in safe]
        return (not bad), len(bad)


# Scan geometric constraint activity across the proposed layout settings.
def c1() -> dict:
    import s3_generate as SG
    t0 = time.time(); cfg = H.CONFIG['C1']; rows = []; rc = []; val = []
    reg = K.load_json(H.reg(K.OUT / 'layouts.json'))
    for name in cfg['families']:
        gg = Geo(name); NB, NP = len(cfg['blocks']), cfg['proposals_per_block']
        U = {b: gg.units(b, NP) for b in cfg['blocks']}

        L = reg['baselines'][name]['layouts']; dmax = dict(g_min=0.0, h=0.0, rms=0.0); nval = 0
        for lid, lay in L.items():
            if lay.get('block') is None or int(lay['block'][1:]) not in cfg['blocks']:
                continue
            m = gg.metrics(U[int(lay['block'][1:])][[int(lay['cand'])]], 0.20); nval += 1
            dmax['g_min'] = max(dmax['g_min'], abs(float(m['g_min'][0]) - lay['g_min_mm'])); dmax['h'] = max(dmax['h'], abs(float(m['h'][0]) - lay['h_mm'])); dmax['rms'] = max(dmax['rms'], abs(float(m['rms'][0]) - lay['delta_rms_mm']))
        val.append(dict(table_type='C1_validation', baseline=name, n_layouts=nval, **{f'max_abs_diff_{k}_mm': v for k, v in dmax.items()}))
        for j in cfg['j']:
            if j == 0.0:
                rr = gg.metrics(np.zeros((1, len(gg.mov))), 0.0)
                rows.append(dict(table_type='C1_cell', baseline=name, j=0.0, alpha=None, beta=None, note='regular anchor (S0)', g_min_p0=float(rr['g_min'][0] / gg.p0), h_h0=float(rr['h'][0] / gg.h0), rms_p0=0.0))
                continue
            ms = {b: gg.metrics(U[b], j) for b in cfg['blocks']}
            eok, nbad = gg.edge_ok(j)
            for a in cfg['alpha']:
                for be in cfg['beta']:
                    first = {k: [] for k in ('S2', 'S3', 'S4')}; pool = dict(S2=0, S3=0, S4=0, S3_not_S2=0, S2_not_S3=0); sel = {k: [] for k in ('S1', 'S2', 'S3', 'S4')}
                    rms_match = 0; rms_s3_sel = []; rms_s1_pool = []
                    for b in cfg['blocks']:
                        m = ms[b]; s2 = m['g_min'] >= a * gg.p0 - 1e-12; s3 = m['h'] <= be * gg.h0 + 1e-12; s4 = s2 & s3
                        pool['S2'] += int(s2.sum()); pool['S3'] += int(s3.sum()); pool['S4'] += int(s4.sum()); pool['S3_not_S2'] += int((s3 & ~s2).sum()); pool['S2_not_S3'] += int((s2 & ~s3).sum())
                        idx = {'S1': 0, 'S2': int(np.argmax(s2)) if s2.any() else None, 'S3': int(np.argmax(s3)) if s3.any() else None, 'S4': int(np.argmax(s4)) if s4.any() else None}
                        for k in ('S2', 'S3', 'S4'):
                            first[k].append(idx[k])
                        for k, i in idx.items():
                            if i is not None:
                                sel[k].append((m['rms'][i] / gg.p0, m['g_min'][i] / gg.p0, m['h'][i] / gg.h0))
                        rms_s1_pool.append(float(np.median(m['rms'] / gg.p0)))
                        if idx['S3'] is not None:
                            r3 = m['rms'][idx['S3']]; rms_s3_sel.append(r3 / gg.p0)
                            rms_match += int(((~s3) & (np.abs(m['rms'] - r3) <= 0.01 * gg.p0)).any())
                    f = {k: np.array([np.nan if v is None else v for v in first[k]], float) for k in first}
                    def chg(x, y):
                        return float(np.mean((x != y) & np.isfinite(x) & np.isfinite(y)))
                    rec = dict(table_type='C1_cell', baseline=name, family=gg.fam, j=j, alpha=a, beta=be, blocks=NB, proposals=NP, edge_rule_ok=eok, n_movable_violating_edge_rule=nbad,
                               S2_inactive_bound_j=(1 - a) / 2, new_vs_old_D2=(not (a in (0.65, 0.70, 0.80) and be in (1.05, 1.15, 1.25))),
                               rate_S2=pool['S2'] / (NB * NP), rate_S3=pool['S3'] / (NB * NP), rate_S4=pool['S4'] / (NB * NP), n_S3_not_S2=pool['S3_not_S2'], n_S2_not_S3=pool['S2_not_S3'],
                               failed_S2=int(np.isnan(f['S2']).sum()), failed_S3=int(np.isnan(f['S3']).sum()), failed_S4=int(np.isnan(f['S4']).sum()),
                               change_S2_vs_S1=chg(f['S2'], np.zeros(NB)), change_S3_vs_S1=chg(f['S3'], np.zeros(NB)), change_S4_vs_S3=chg(f['S4'], f['S3']), change_S4_vs_S2=chg(f['S4'], f['S2']),
                               change_S3_vs_S2=chg(f['S3'], f['S2']), first_S3_median=float(np.nanmedian(f['S3'])) if np.isfinite(f['S3']).any() else None,
                               first_S3_max=float(np.nanmax(f['S3'])) if np.isfinite(f['S3']).any() else None,
                               rms_matched_nonS3_available_blocks=rms_match, rms_p0_S1_pool_median=float(np.median(rms_s1_pool)),
                               rms_p0_S3_selected_median=float(np.median(rms_s3_sel)) if rms_s3_sel else None)
                    for k, v in sel.items():
                        if v:
                            arr = np.array(v); rec[f'{k}_sel_rms_p0_median'] = float(np.median(arr[:, 0])); rec[f'{k}_sel_gmin_p0_min'] = float(arr[:, 1].min()); rec[f'{k}_sel_h_h0_max'] = float(arr[:, 2].max())
                    rows.append(rec)

        cand = set()
        for j in (0.10, 0.20, 0.30):
            m = gg.metrics(U[0], j); s2 = m['g_min'] >= 0.70 * gg.p0 - 1e-12; s3 = m['h'] <= 1.15 * gg.h0 + 1e-12; s4 = s2 & s3
            for arm, msk in (('S1', np.ones(NP, bool)), ('S2', s2), ('S3', s3), ('S4', s4)):
                if msk.any():
                    cand.add((j, int(np.argmax(msk)), arm))
        seen = set()
        for j, c, arm in sorted(cand):
            if (j, c) in seen:
                continue
            seen.add((j, c))
            r = SG.evaluate(name, 0, c, j); lit = float(r['lit_px']); peak = H.K.Q / lit if lit else None
            rc.append(dict(table_type='C1_raster', baseline=name, block=0, cand=c, j=j, arms=';'.join(sorted(a_ for jj, cc, a_ in cand if jj == j and cc == c)), general_ok=r['general_ok'],
                           general_fail=';'.join(r['general_fail']), n_elements=r['n_elements'], n_components=r['n_components'], n_clipped_lines=r.get('n_clipped_lines'),
                           edge_clearance_min_mm=r['edge_clearance_min_mm'], lit_px=lit, peak_Q11695=peak, peak_le_1=(peak is not None and peak <= 1.0), equal_integral_feasible=r['equal_integral_feasible'],
                           g_min_mm=r['g_min_mm'], h_mm=r['h_mm'], rms_mm=r['delta_rms_mm'], coord_g_min_mm=float(gg.metrics(U[0][[c]], j)['g_min'][0]), coord_h_mm=float(gg.metrics(U[0][[c]], j)['h'][0])))
    D = pd.DataFrame(val + rows + rc)
    H.save_csv(D, H.OUT / '_parts' / 'C1_geometry.csv')
    out = dict(validation=val, n_cells=int((D.table_type == 'C1_cell').sum()), n_new_cells=int(D.get('new_vs_old_D2', pd.Series(dtype=bool)).fillna(False).sum()), n_raster_checks=len(rc),
               raster_all_general_ok=bool(all(x['general_ok'] for x in rc)), raster_all_peak_le_1=bool(all(x['peak_le_1'] for x in rc)), seconds=round(time.time() - t0, 1), t_local=K.now_local())
    H.run_put('C1', out); H.flush_reads('C1')
    return out


PILOT = dict(blocks=[0, 3, 6, 9, 12, 15, 18, 21], j=[0.10, 0.20, 0.30], phases=4, surfaces=3, sigmas=[0.0, 40.0],
             arms={'S1': dict(alpha=None, beta=None), 'S2*': dict(alpha=0.80, beta=None), 'S3': dict(alpha=None, beta=1.15), 'S4*': dict(alpha=0.80, beta=1.15)},
             rms_control=dict(j=0.20, rule='first candidate violating the S3 mask with |RMS - RMS(S3 selected)| <= 0.01 p0 (optional arm, separate count)'),
             reuse='S4-REAL-02 images of identical (baseline, j, block, cand) layouts at the same phases / surfaces / sigma / noise seeds (S0, j 0.20 S1 / S2 / S3 / S4)')


def c3() -> dict:
    t0 = time.time(); reg = K.load_json(H.reg(K.OUT / 'layouts.json')); rows = []; summ = {}
    per_layout_images = PILOT['phases'] * PILOT['surfaces'] * len(PILOT['sigmas'])
    for name in ('R8x4', 'L6.3'):
        gg = Geo(name); have = {(0.20, int(l['block'][1:]), int(l['cand'])) for l in reg['baselines'][name]['layouts'].values() if l.get('block')}
        new_layouts = set(); fails = 0; rms_ctrl = set()
        for j in PILOT['j']:
            for b in PILOT['blocks']:
                U = gg.units(b, 1000); m = gg.metrics(U, j); s2 = m['g_min'] >= 0.80 * gg.p0 - 1e-12; s3 = m['h'] <= 1.15 * gg.h0 + 1e-12
                masks = {'S1': np.ones(1000, bool), 'S2*': s2, 'S3': s3, 'S4*': s2 & s3}
                sel = {}
                for arm, msk in masks.items():
                    if not msk.any():
                        fails += 1; rows.append(dict(table_type='C3_pilot', baseline=name, j=j, block=b, arm=arm, cand=None, status='generation failure (no candidate in 1000)')); continue
                    c = int(np.argmax(msk)); sel[arm] = c; key = (j, b, c)
                    st = 'reuse S4-REAL-02' if key in have else ('same layout as another arm' if key in new_layouts else 'new layout')
                    if st == 'new layout':
                        new_layouts.add(key)
                    rows.append(dict(table_type='C3_pilot', baseline=name, j=j, block=b, arm=arm, cand=c, status=st, rms_p0=float(m['rms'][c] / gg.p0), g_min_p0=float(m['g_min'][c] / gg.p0), h_h0=float(m['h'][c] / gg.h0)))
                if j == PILOT['rms_control']['j'] and 'S3' in sel:
                    r3 = m['rms'][sel['S3']]; ok = (~s3) & (np.abs(m['rms'] - r3) <= 0.01 * gg.p0)
                    if ok.any():
                        c = int(np.argmax(ok)); rms_ctrl.add((j, b, c)); rows.append(dict(table_type='C3_pilot', baseline=name, j=j, block=b, arm='S1-RMS', cand=c, status='optional control',
                                                                                         rms_p0=float(m['rms'][c] / gg.p0), h_h0=float(m['h'][c] / gg.h0)))
        n_new = len(new_layouts); summ[name] = dict(new_layouts=n_new, images_new=n_new * per_layout_images, generation_failures=fails, rms_control_layouts=len(rms_ctrl),
                                                    rms_control_images=len(rms_ctrl) * per_layout_images)
    D = pd.DataFrame(rows); H.save_csv(D, H.OUT / '_parts' / 'C1_pilot_plan.csv')
    out = dict(plan=PILOT, per_layout_images=per_layout_images, summary=summ, total_new_images=int(sum(v['images_new'] for v in summ.values())),
               total_with_rms_control=int(sum(v['images_new'] + v['rms_control_images'] for v in summ.values())), note='counts only; nothing imaged or decoded', seconds=round(time.time() - t0, 1), t_local=K.now_local())
    H.run_put('C3_pilot_plan', out)
    return out


# Compare support-sensitive layout results using the stored sampling layers.
def c2() -> dict:
    import r2_eval as E
    import r2_run as R
    import rdy_platform as P
    from surface_eval_r3 import surface_eval_r3_dev
    t0 = time.time(); cfg = H.CONFIG['C2']; M = R.manifest(); reg = K.load_json(H.reg(K.OUT / 'layouts.json'))
    disp = {'R8x4': 'B021', 'L6.3': 'B000'}; rows = []; groups = []
    for name, famp, _f in K.BASES:
        B = reg['baselines'][name]; L = B['layouts']; V = [v for v in B['views'] if v['strategy'] in ('S1', 'S3')]
        lay_of = {(v['block'], v['strategy']): v['layout'] for v in V}
        s1 = sorted({lay_of[(b, 'S1')] for b, s in lay_of if s == 'S1'}); s3 = sorted({lay_of[(b, 'S3')] for b, s in lay_of if s == 'S3' and lay_of[(b, 'S3')] != lay_of[(b, 'S1')]})
        best = min(((abs(L[a]['delta_rms_mm'] - L[c]['delta_rms_mm']) / B['p0_mm'], a, c) for a in s1 for c in s3), key=lambda t: (t[0], t[1], t[2]))
        qualified = best[0] <= 0.01
        sets = [('display', 'S0', 'regular'), ('display', 'S1', lay_of[(disp[name], 'S1')]), ('display', 'S3', lay_of[(disp[name], 'S3')])]
        if qualified:
            sets += [('rms_near', 'S1', best[1]), ('rms_near', 'S3', best[2])]
        groups.append(dict(baseline=name, rms_near_pair=[best[1], best[2]], rms_near_diff_p0=best[0], qualified=qualified))
        for surface in K.SURF:
            dom = E.domain('omega1'); sid = K.RK.SURF_MAP[surface]; dl = P.dom_local(surface); mesh = P.mesh(surface); tz = P.s2().truth_interp(mesh)
            Dg = dict(x=dom['x'], y=dom['y'], z_ref=np.asarray(dom[f'z_{sid}'], np.float64), area=np.full(len(dom['x']), dom['step'] ** 2), hash='omega1')
            Dl = dict(x=dl['x'], y=dl['y'], z_ref=np.asarray(dl['z_ref'], np.float64), area=np.asarray(dl['area'], np.float64), hash='local')
            done = set()
            for grp, strat, lid in sets:
                if lid in done:
                    continue
                done.add(lid)
                r = M[(M.baseline == name) & (M.layout == lid) & (M.phase == 'H0') & (M.sigma == 0.0) & (M.surface == surface)].iloc[0]
                z = K.load_npz(H.reg(K.EVL / f'{r.eval_key}.npz')); valid = z['valid'].astype(bool); pts = z['pts'].astype(np.float64)
                if famp == 'line':
                    ref = np.asarray(z['ref_xyz'], np.float64)
                else:
                    xy = np.asarray(z['exp_xy'], np.float64)[np.asarray(z['exp_in_image'], bool)]; zz = tz(np.column_stack([xy[:, 1], xy[:, 0]])); ok = np.isfinite(zz)
                    ref = np.column_stack([xy[ok], zz[ok]])
                corr = valid & z['point_correct'].astype(bool) if 'point_correct' in z else valid
                layers = dict(L1_reference=ref, L2_confirmed_correct=pts[corr], L3_official=pts[valid])
                for lname, Pv in layers.items():
                    for dname, Dd in (('global', Dg), ('local', Dl)):
                        if len(Pv) < 3:
                            for d in cfg['d_mm']:
                                rows.append(dict(record_type='C2_layer', baseline=name, family=famp, surface=surface, group=grp, strategy=strat, layout=lid, eval_key=r.eval_key, layer=lname, domain=dname,
                                                 d_mm=d, n_points=int(len(Pv)), C_int=0.0, C_valid=0.0, C_tau=0.0, h_nominal_mm=L[lid].get('h_mm'), rms_p0=L[lid].get('delta_rms_mm', 0.0) / B['p0_mm']))
                            continue
                        ev = surface_eval_r3_dev(Pv, None, Dd, cfg['d_mm'], [cfg['tau_mm']], support_rule='exact'); g = ev['subdomains']['global']
                        for d in cfg['d_mm']:
                            bd = g['by_d'][str(float(d))]
                            rows.append(dict(record_type='C2_layer', baseline=name, family=famp, surface=surface, group=grp, strategy=strat, layout=lid, eval_key=r.eval_key, layer=lname, domain=dname,
                                             d_mm=d, n_points=int(len(Pv)), C_int=g['C_interpolable'], C_valid=bd['C_valid'], C_tau=bd['C_tau'][str(float(cfg['tau_mm']))],
                                             sup_rmse=bd.get('rmse'), h_nominal_mm=L[lid].get('h_mm'), rms_p0=L[lid].get('delta_rms_mm', 0.0) / B['p0_mm']))
    D = pd.DataFrame(rows)

    chk = []
    C = pd.read_csv(H.reg(K.OUT / 'case_metrics.csv'), low_memory=False)
    for x in D[(D.layer == 'L3_official') & (D.d_mm == 6.0)].itertuples():
        st = C[C.eval_key == x.eval_key].iloc[0]; key = 'glob_Ctau' if x.domain == 'global' else 'loc_Ctau'
        chk.append(abs(float(st[key]) - float(x.C_tau)))
    H.save_csv(D, H.OUT / '_parts' / 'C2_layers.csv')
    out = dict(groups=groups, n_clouds=int(D.eval_key.nunique()), n_rows=len(D), d6_reproduction_max_abs=float(max(chk)) if chk else None, seconds=round(time.time() - t0, 1), t_local=K.now_local())
    H.run_put('C2', out); H.flush_reads('C2')
    return out
