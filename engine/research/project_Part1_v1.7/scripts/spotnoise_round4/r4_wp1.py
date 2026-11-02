# Associate reconstructed points with expected positions and separate error from dropout.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

import r4_common as C
import sa_defs as D
import rd_defs as RD
SL = C.SL

OUT = C.OUT / 'wp1'
GATE_XY = D.GATE_XY
GATE_Z = D.GATE_Z
PAIRS = ((0.5, 1.0), (1.0, 1.5))
REGIONS = {'disc': 'exp_in_disc', 'inner': 'exp_cam_inner'}


def associate(E, P, gate):

    n = len(E); match = np.full(n, -1); ncand = np.zeros(n, int); dist = np.full(n, np.nan)
    if n == 0 or len(P) == 0:
        return match, ncand, dist
    cand = cKDTree(P).query_ball_point(E, gate)
    ncand = np.array([len(c) for c in cand])
    cols = np.array(sorted({j for c in cand for j in c}), int)
    if len(cols) == 0:
        return match, ncand, dist
    Dm = np.sqrt(((E[:, None, :] - P[None, cols, :]) ** 2).sum(axis=2)); al = Dm <= gate
    r, c = linear_sum_assignment(np.where(al, Dm, 1e6))
    for ri, ci in zip(r, c):
        if al[ri, ci]:
            match[ri] = cols[ci]; dist[ri] = Dm[ri, ci]
    return match, ncand, dist


def case_assoc(npz_path):
    with np.load(npz_path) as z:
        exy = z['exp_xy'].astype(np.float64); disc = z['exp_in_disc'].astype(bool); inner = z['exp_cam_inner'].astype(bool)
        pts = z['pts'].astype(np.float64); amb = z['amb'].astype(bool); er = z['err_ref'].astype(np.float64)
    finE = np.isfinite(exy).all(axis=1)
    fin = np.isfinite(pts).all(axis=1); un = fin & ~amb
    P = pts[un][:, :2]; Pe = er[un]
    E = exy[finE]
    match, ncand, dist = associate(E, P, GATE_XY)
    F = pts[fin & amb][:, :2]
    nflag = np.array([len(c) for c in cKDTree(F).query_ball_point(E, GATE_XY)]) if len(F) else np.zeros(len(E), int)
    q = np.isfinite(Pe) & (np.abs(np.nan_to_num(Pe, nan=1e9)) <= GATE_Z)
    qidx = np.where(q)[0]
    mq, _, _ = associate(E, P[q], GATE_XY)
    matched_q = mq >= 0
    err_q = np.where(matched_q, Pe[qidx][np.clip(mq, 0, None)] if len(qidx) else np.nan, np.nan)
    err = np.where(match >= 0, Pe[np.clip(match, 0, None)] if len(Pe) else np.nan, np.nan)
    in_gate = np.zeros(len(P), bool)
    for c in cKDTree(P).query_ball_point(E, GATE_XY) if len(P) else []:
        in_gate[c] = True
    assigned = np.zeros(len(P), bool); assigned[match[match >= 0]] = True
    mo = P[match[match >= 0]] if (match >= 0).any() else np.zeros((0, 2))
    gap = RD.spatial_gap(np.column_stack([mo, np.zeros(len(mo))]), np.zeros(len(mo), bool), np.zeros(len(mo))) if len(mo) else dict(gap_max_mm=None, gap_p95_mm=None)
    pos = dict(idx=np.where(finE)[0], disc=disc[finE], inner=inner[finE], matched=match >= 0, err=err, dist=dist, ncand=ncand, nflag=nflag, matched_q=matched_q,
               err_q=err_q)
    rf = Pe[np.isfinite(Pe)]
    summ = dict(n_unflagged=int(un.sum()), n_flagged=int((fin & amb).sum()), n_outside_all_gates=int((~in_gate).sum()), n_surplus_in_gate=int((in_gate & ~assigned).sum()),
                rmse_all_unflagged=float(np.sqrt(np.mean(rf ** 2))) if len(rf) else None, n_err_undefined=int((~np.isfinite(Pe)).sum()),
                gap_planar_max_mm=gap['gap_max_mm'], gap_planar_p95_mm=gap['gap_p95_mm'])
    for rg in REGIONS:
        m = pos[rg]
        e = err[m & (match >= 0)]; e = e[np.isfinite(e)]
        summ.update({f'{rg}_n_exp': int(m.sum()), f'{rg}_n_assoc': int((m & (match >= 0)).sum()), f'{rg}_n_multi_cand': int((m & (ncand >= 2)).sum()),
                     f'{rg}_n_flagged_only': int((m & (ncand == 0) & (nflag > 0)).sum()), f'{rg}_n_no_candidate': int((m & (ncand == 0) & (nflag == 0)).sum()),
                     f'{rg}_n_assoc_q': int((m & matched_q).sum()), f'{rg}_rmse_own_assoc': float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
                     f'{rg}_rec_planar': float((m & (match >= 0)).sum() / m.sum()) if m.sum() else None})
    return pos, summ


def stats(e):
    e = e[np.isfinite(e)]
    if not len(e):
        return dict(rmse=None, med_abs=None, p95_abs=None, max_abs=None, n=0)
    a = np.abs(e)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), med_abs=float(np.median(a)), p95_abs=float(np.percentile(a, 95)), max_abs=float(a.max()), n=int(len(e)))


def attributes():

    rows = []
    slots = C.cross_slots()
    for surf in SL.SURFACES:
        mesh, cfg, ref = SL.L15.surface(surf)
        X = np.asarray(mesh['X'], float); Y = np.asarray(mesh['Y'], float); Z = np.asarray(mesh['Z'], float); Zb = np.asarray(mesh['Z_base'], float)
        xa, ya = X[0, :], Y[:, 0]
        gy, gx = np.gradient(Z, ya, xa)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        fs = RegularGridInterpolator((ya, xa), slope, bounds_error=False, fill_value=np.nan)
        fd = RegularGridInterpolator((ya, xa), Zb - Z, bounds_error=False, fill_value=np.nan)
        fz = RegularGridInterpolator((ya, xa), Z - Z.min(), bounds_error=False, fill_value=np.nan)
        for ph in SL.PHASES:
            s = next(x for x in slots if x['spec']['surface'] == surf and x['spec']['phase'] == ph and x['spec']['sigma'] == 0)
            with np.load(C.CROSS_CASES / f"{s['key']}.npz") as z:
                exy = z['exp_xy'].astype(float); uv = z['exp_uv'].astype(float); cp = z['centres_px'].astype(float)
                disc = z['exp_in_disc'].astype(bool); inner = z['exp_cam_inner'].astype(bool)
            finE = np.isfinite(exy).all(axis=1); idx = np.where(finE)[0]
            sc = D.local_scale(uv, cp)
            fu = np.isfinite(uv).all(axis=1)
            dnn = np.full(len(uv), np.nan)
            if fu.sum() > 1:
                d, _ = cKDTree(uv[fu]).query(uv[fu], k=2); dnn[fu] = d[:, 1]
            q = np.column_stack([exy[idx, 1], exy[idx, 0]])
            for j, i in enumerate(idx):
                rows.append(dict(surface=surf, phase=ph, idx=int(i), disc=bool(disc[i]), inner=bool(inner[i]), x=exy[i, 0], y=exy[i, 1],
                                 r_centre_mm=float(np.hypot(exy[i, 0], exy[i, 1])), slope_deg=float(fs(q[j:j + 1])[0]), dip_mm=float(fd(q[j:j + 1])[0]),
                                 height_mm=float(fz(q[j:j + 1])[0]), cam_nn_px=float(dnn[i]), scale=float(sc[i]),
                                 border_px=float(np.nanmin([uv[i, 0], uv[i, 1], 511 - uv[i, 0], 511 - uv[i, 1]])) if fu[i] else np.nan))
    return pd.DataFrame(rows)


# Associate recovered samples with expected positions and aggregate dropout and error counts.
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    slots = C.cross_slots()
    allc = pd.read_csv(C.ROUND3 / 'analysis' / 'all_unique_cases.csv').set_index('key')
    pos, rows = {}, []
    for s in slots:
        sp = s['spec']; k = s['key']
        p, sm = case_assoc(C.CROSS_CASES / f'{k}.npz')
        cid = (sp['surface'], sp['phase'], float(sp['r_mm']), float(sp['v']), float(sp['sigma']), int(sp['seed']))
        pos[cid] = p
        a = allc.loc[k]
        rows.append(dict(key=k, surface=sp['surface'], phase=sp['phase'], r_mm=sp['r_mm'], v=sp['v'], sigma=sp['sigma'], seed=sp['seed'], status=a.status,
                         L2=bool(a.L2_geometric), L3=bool(a.L3_research), L23=bool(a.L23_both), rec_existing=a.rec, rmse_existing=a.ref_rmse_rel_mm, **sm))
    cases = pd.DataFrame(rows); cases.to_csv(OUT / 'case_assoc.csv', index=False)
    prs, poscnt = [], {}
    for (ra, rb) in PAIRS:
        for cid, pa in pos.items():
            surf, ph, r, v, sg, seed = cid
            if r != ra:
                continue
            pb = pos[(surf, ph, rb, v, sg, seed)]
            assert np.array_equal(pa['idx'], pb['idx'])
            ca = cases[(cases.surface == surf) & (cases.phase == ph) & (cases.r_mm == ra) & (cases.v == v) & (cases.sigma == sg) & (cases.seed == seed)].iloc[0]
            cb = cases[(cases.surface == surf) & (cases.phase == ph) & (cases.r_mm == rb) & (cases.v == v) & (cases.sigma == sg) & (cases.seed == seed)].iloc[0]
            for rg in REGIONS:
                m = pa[rg]
                am, bm = pa['matched'] & m, pb['matched'] & m
                com = am & bm
                sa_, sb_ = stats(pa['err'][com]), stats(pb['err'][com])
                oa, ob = stats(pa['err'][am]), stats(pb['err'][bm])
                aq, bq = pa['matched_q'] & m, pb['matched_q'] & m; cq = aq & bq
                qa, qb = stats(pa['err_q'][cq]), stats(pb['err_q'][cq])
                prs.append(dict(pair=f'r{ra:g}-r{rb:g}', small=ra, large=rb, surface=surf, phase=ph, v=v, sigma=sg, seed=seed, region=rg, n_exp=int(m.sum()),
                                n_common=int(com.sum()), n_only_small=int((am & ~bm).sum()), n_only_large=int((~am & bm).sum()), n_neither=int((m & ~am & ~bm).sum()),
                                small_common_rmse=sa_['rmse'], large_common_rmse=sb_['rmse'], small_common_med=sa_['med_abs'], large_common_med=sb_['med_abs'],
                                small_common_p95=sa_['p95_abs'], large_common_p95=sb_['p95_abs'], small_common_max=sa_['max_abs'], large_common_max=sb_['max_abs'],
                                small_own_rmse=oa['rmse'], large_own_rmse=ob['rmse'], small_own_n=oa['n'], large_own_n=ob['n'],
                                small_all_rmse=ca.rmse_all_unflagged, large_all_rmse=cb.rmse_all_unflagged,
                                small_rec_planar=float(am.sum() / m.sum()) if m.sum() else None, large_rec_planar=float(bm.sum() / m.sum()) if m.sum() else None,
                                small_rec_existing=ca.rec_existing, large_rec_existing=cb.rec_existing, small_gap=ca.gap_planar_max_mm, large_gap=cb.gap_planar_max_mm,
                                cond_n_common=int(cq.sum()), cond_small_rmse=qa['rmse'], cond_large_rmse=qb['rmse'],
                                small_L23=bool(ca.L23), large_L23=bool(cb.L23), small_L2=bool(ca.L2), large_L2=bool(cb.L2)))
                if rg == 'inner':
                    key = (f'r{ra:g}-r{rb:g}', surf, ph, v, sg)
                    cnt = poscnt.setdefault(key, np.zeros((len(m), 4), int))
                    cnt[:, 0] += com; cnt[:, 1] += am & ~bm; cnt[:, 2] += ~am & bm; cnt[:, 3] += ~am & ~bm & m
    P = pd.DataFrame(prs)
    for k in ('rmse', 'med', 'p95'):
        P[f'd_common_{k}'] = P[f'large_common_{k}'] - P[f'small_common_{k}']
    P['d_own_rmse'] = P.large_own_rmse - P.small_own_rmse; P['d_all_rmse'] = P.large_all_rmse - P.small_all_rmse
    P['d_cond_rmse'] = P.cond_large_rmse - P.cond_small_rmse
    P.to_csv(OUT / 'pairs.csv', index=False)
    att = attributes(); att.to_csv(OUT / 'position_attributes.csv', index=False)
    rows = []
    for (pair, surf, ph, v, sg), cnt in poscnt.items():
        idx = pos[(surf, ph, float(pair.split('-')[0][1:]), v, sg, 0)]['idx']
        inner = pos[(surf, ph, float(pair.split('-')[0][1:]), v, sg, 0)]['inner']
        for j in np.where(inner)[0]:
            rows.append(dict(pair=pair, surface=surf, phase=ph, v=v, sigma=sg, idx=int(idx[j]), both=int(cnt[j, 0]), only_small=int(cnt[j, 1]),
                             only_large=int(cnt[j, 2]), neither=int(cnt[j, 3])))
    pc = pd.DataFrame(rows).merge(att, on=['surface', 'phase', 'idx'], how='left')
    pc.to_csv(OUT / 'position_status_counts.csv.gz', index=False)

    hs = []
    for (pair, v), g in pc[pc.sigma > 0].groupby(['pair', 'v']):
        for cat in ('both', 'only_small', 'only_large', 'neither'):
            w = g[cat].values
            if w.sum() == 0:
                hs.append(dict(pair=pair, v=v, category=cat, n_position_cases=0)); continue
            row = dict(pair=pair, v=v, category=cat, n_position_cases=int(w.sum()))
            for a in ('slope_deg', 'dip_mm', 'r_centre_mm', 'cam_nn_px', 'scale', 'border_px'):
                x = g[a].values; ok = np.isfinite(x) & (w > 0)
                order = np.argsort(x[ok]); cw = np.cumsum(w[ok][order])
                row[f'{a}_wmedian'] = float(x[ok][order][np.searchsorted(cw, cw[-1] / 2)]) if ok.any() else None
            hs.append(row)
    H = pd.DataFrame(hs); H.to_csv(OUT / 'hard_position_summary.csv', index=False)

    S = []
    for (pair, rg, v, sg), g in P[P.sigma > 0].groupby(['pair', 'region', 'v', 'sigma']):
        d = g.dropna(subset=['d_common_rmse'])
        S.append(dict(pair=pair, region=rg, v=v, sigma=sg, n_pair_cases=int(len(g)), n_defined=int(len(d)), n_undefined=int(len(g) - len(d)),
                      median_n_common=float(g.n_common.median()), frac_large_lower_common=float((d.d_common_rmse < 0).mean()) if len(d) else None,
                      median_d_common_rmse=float(d.d_common_rmse.median()) if len(d) else None, median_d_own_rmse=float(g.d_own_rmse.median()) if g.d_own_rmse.notna().any() else None,
                      median_d_all_rmse=float(g.d_all_rmse.median()) if g.d_all_rmse.notna().any() else None,
                      median_d_cond_rmse=float(g.d_cond_rmse.median()) if g.d_cond_rmse.notna().any() else None,
                      sum_only_small=int(g.n_only_small.sum()), sum_only_large=int(g.n_only_large.sum()), sum_neither=int(g.n_neither.sum()), sum_common=int(g.n_common.sum())))
    S = pd.DataFrame(S); S.to_csv(OUT / 'precision_gain_summary.csv', index=False)
    C.save(OUT / 'summary.json', dict(generated=C.now(), n_cases=int(len(cases)), n_pair_rows=int(len(P)), gate_xy_mm=GATE_XY, gate_z_mm_conditional_only=GATE_Z,
                                      association='planar one-to-one within 1.5 mm, unflagged outputs, no depth filter (main); depth gate only in conditional columns',
                                      undefined_common=int(P.d_common_rmse.isna().sum())))
    print(S[(S.region == 'inner')].pivot_table(index=['pair', 'v'], columns='sigma', values='median_d_common_rmse').round(3).to_string())
    print(S[(S.region == 'inner')].pivot_table(index=['pair', 'v'], columns='sigma', values='median_d_all_rmse').round(3).to_string())


if __name__ == '__main__':
    main()
