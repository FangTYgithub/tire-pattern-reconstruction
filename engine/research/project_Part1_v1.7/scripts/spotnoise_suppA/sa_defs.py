# Compute signal, recovery and tolerance diagnostics from stored dot results.

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[0] / 'spotnoise')]
import sn_posthoc as PH


INNER_PX = 16.0
MARGIN_PX = 2.0
BG_GAP_PX = 3.0
K_NEIGH = 4


def local_scale(uv, cp):

    fin = np.isfinite(uv).all(axis=1)
    idx = np.where(fin)[0]
    s = np.full(len(uv), np.nan)
    if len(idx) <= K_NEIGH:
        return s
    t = cKDTree(cp[idx]); d, j = t.query(cp[idx], k=K_NEIGH + 1)
    for a, i in enumerate(idx):
        dc = np.hypot(*(uv[i] - uv[idx[j[a, 1:]]]).T)
        s[i] = float(np.median(dc / d[a, 1:]))
    return s


def signal_stat(img0, uv, cp, inner, r_px):

    img = np.asarray(img0, np.float64); H, W = img.shape
    uv = np.asarray(uv, np.float64); cp = np.asarray(cp, np.float64); inner = np.asarray(inner, bool)
    fin = np.isfinite(uv).all(axis=1); idx = np.where(fin)[0]
    sc = local_scale(uv, cp)
    s_fill = np.where(np.isfinite(sc), sc, np.nanmedian(sc))
    rho = s_fill * r_px + MARGIN_PX
    yy, xx = np.mgrid[0:H, 0:W]
    P = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    tree = cKDTree(uv[idx]); dist, own_l = tree.query(P)
    own = idx[own_l].reshape(H, W); dist = dist.reshape(H, W)
    in_sq = (xx >= INNER_PX) & (xx <= W - 1 - INNER_PX) & (yy >= INNER_PX) & (yy <= H - 1 - INNER_PX)
    bg_mask = in_sq & (dist > rho[own] + BG_GAP_PX)
    B = float(np.median(img[bg_mask])) if bg_mask.any() else 0.0
    rows = []
    for i in np.where(fin & inner)[0]:
        u, v = uv[i]; R = rho[i]
        x0, x1 = int(np.floor(u - R)) - 1, int(np.ceil(u + R)) + 2; y0, y1 = int(np.floor(v - R)) - 1, int(np.ceil(v + R)) + 2
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        disc_full = np.hypot(gx - u, gy - v) <= R
        n_full = int(disc_full.sum())
        inside = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        n_img = int((disc_full & inside).sum())
        gxc, gyc = np.clip(gx, 0, W - 1), np.clip(gy, 0, H - 1)
        sup = disc_full & inside & (own[gyc, gxc] == i)
        n = int(sup.sum())
        vals = img[gyc[sup], gxc[sup]]
        S = float((vals - B).sum()) if n else np.nan
        rows.append((S, n, float(vals.max() - B) if n else np.nan, float((vals >= 255).mean()) if n else np.nan,
                     S / np.sqrt(n) if n else np.nan, 1.0 - n / n_img if n_img else np.nan, 1.0 - n_img / n_full if n_full else np.nan, sc[i]))
    a = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 8))
    if not len(a):
        return dict(n_dots=0)
    S, N, Pk, sat, Q, tv, te, scl = a.T
    return dict(n_dots=int(len(a)), background=B, n_bg_px=int(bg_mask.sum()), r_px_proj=float(r_px),
                S_med=float(np.nanmedian(S)), S_p10=float(np.nanpercentile(S, 10)), N_med=float(np.nanmedian(N)), N_min=int(np.nanmin(N)),
                N_max=int(np.nanmax(N)), P_med=float(np.nanmedian(Pk)), P_p10=float(np.nanpercentile(Pk, 10)), Q_med=float(np.nanmedian(Q)),
                Q_p10=float(np.nanpercentile(Q, 10)), sat_any_frac=float(np.mean(sat > 0)), sat_px_frac_med=float(np.nanmedian(sat)),
                trunc_voronoi_frac=float(np.mean(tv > 0)), trunc_voronoi_mean=float(np.nanmean(tv)), trunc_edge_frac=float(np.mean(te > 0)),
                scale_med=float(np.nanmedian(scl)), scale_p5=float(np.nanpercentile(scl, 5)), scale_p95=float(np.nanpercentile(scl, 95)),
                rho_med=float(np.nanmedian(rho[fin & inner])))


def nominal_indices(stat, sigma):

    if not sigma or sigma <= 0 or not stat or not stat.get('n_dots'):
        return dict(q_sum=None, q_peak=None)
    return dict(q_sum=stat['Q_med'] / sigma, q_peak=stat['P_med'] / sigma)


GATE_XY, GATE_Z, RECOVERY_MIN = 1.5, 1.0, 0.9


def recovery(z):

    exp = z['exp_xy'].astype(np.float64)[z['exp_in_disc'].astype(bool)]
    o2o, _nn, n_q = PH.recovery(z['pts'], z['amb'].astype(bool), z['err_ref'], exp, GATE_XY, GATE_Z)
    return o2o, n_q, int(len(exp))


CRIT = dict(occ_dent_min=0.02, dent_abs_max_pct=10.0, gross_max=0.01, rmse_max_sigma0_mm=0.06, rmse_max_noisy_mm=0.15)
CRITERIA = {'L1_status_ok': 'Normal algorithm status', 'L2_geometric': 'Geometry: gross-error rate <= 1 % and recovery >= 0.9',
            'L3_research': 'Research criterion: PM-015 combination', 'L23_both': 'Research and geometric criteria both passed',
            'P_ratio': 'Output count ratio >= 0.9', 'P_dent': 'Absolute dent error <= 10 %'}


def _le(x, lim):
    return x is not None and np.isfinite(x) and x <= lim


def layers(status, ex, rec, sigma):

    g = ex.get('gross_rate_rel'); occ = ex.get('occ_dent_rel'); dent = ex.get('dent_rel_pct'); rmse = ex.get('ref_rmse_rel_mm')
    lim = CRIT['rmse_max_sigma0_mm'] if sigma == 0 else CRIT['rmse_max_noisy_mm']
    l1 = status == 'ok'
    l2 = _le(g, CRIT['gross_max']) and rec is not None and np.isfinite(rec) and rec >= RECOVERY_MIN
    l3 = (l1 and occ is not None and np.isfinite(occ) and occ >= CRIT['occ_dent_min'] and dent is not None and np.isfinite(dent)
          and abs(dent) <= CRIT['dent_abs_max_pct'] and _le(g, CRIT['gross_max']) and _le(rmse, lim))
    ratio = ex.get('completeness')
    return dict(L1_status_ok=bool(l1), L2_geometric=bool(l2), L3_research=bool(l3), L23_both=bool(l2 and l3),
                P_ratio=bool(ratio is not None and np.isfinite(ratio) and ratio >= 0.9),
                P_dent=bool(dent is not None and np.isfinite(dent) and abs(dent) <= CRIT['dent_abs_max_pct']))


def tolerance_row(levels, kn):

    passall = [n > 0 and k == n for k, n in kn]
    if not passall[0]:
        bnd, first_fail, label = None, levels[0], 'Baseline failed'
    else:
        i = 0
        while i + 1 < len(passall) and passall[i + 1]:
            i += 1
        bnd = levels[i]; first_fail = levels[i + 1] if i + 1 < len(levels) else None
        label = 'Top level passed (right-censored)' if first_fail is None else ('0 (only the noiseless level passed)' if bnd == 0 else f'{bnd:g}')
    start = levels.index(first_fail) + 1 if first_fail is not None else len(levels)
    repass = [levels[j] for j in range(start, len(levels)) if passall[j]]
    mixed = [levels[j] for j in range(len(levels)) if 0 < kn[j][0] < kn[j][1]]
    return dict(boundary_sigma=bnd, first_fail_sigma=first_fail, label=label, repass_levels=repass, mixed_levels=mixed)
