# Read line widths along local normals and aggregate them by segment.

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))

DEFAULT = dict(step_px=0.25, bg_outer_frac=0.2, bg_quantile=0.25, half_frac=0.5, min_rows=5, margin_px=0.5, threshold_px=None, max_rows_per_segment=400)


def profile_params(fam: dict, sigma_f: float, widths_px: dict) -> dict:
    L = 0.5 * float(widths_px['wide']) + 3.0 * float(sigma_f) + 1.0
    return dict(L_px=float(L), provisional_threshold_px=0.5 * (float(widths_px['narrow']) + float(widths_px['wide'])))


def _tangent(seg_feat: np.ndarray, i: int, fam_t: np.ndarray) -> np.ndarray:

    v0 = seg_feat[i, 1]; sel = np.abs(seg_feat[:, 1] - v0) <= 2.0
    if sel.sum() >= 3:
        pts = seg_feat[sel]; c = pts.mean(axis=0); d = pts - c
        cov = d.T @ d; w, V = np.linalg.eigh(cov); t = V[:, int(np.argmax(w))]
        if np.dot(t, fam_t) < 0:
            t = -t
        return t / np.linalg.norm(t)
    return fam_t


def row_fwhm(img: np.ndarray, p: np.ndarray, n: np.ndarray, L: float, prm: dict) -> dict:
    from scipy.ndimage import map_coordinates
    ts = np.arange(-L, L + 1e-9, prm['step_px']); xs = p[0] + ts * n[0]; ys = p[1] + ts * n[1]
    H, W = img.shape
    if xs.min() < 0 or ys.min() < 0 or xs.max() > W - 1 or ys.max() > H - 1:
        return dict(valid=False, reason='window_truncated')
    prof = map_coordinates(img, [ys, xs], order=1, mode='nearest')
    m = len(prof); k = max(2, int(prm['bg_outer_frac'] * m)); bg = float(np.quantile(np.r_[prof[:k], prof[-k:]], prm['bg_quantile']))
    c = m // 2; r = max(1, int(round(1.5 / prm['step_px']))); ipk = c - r + int(np.argmax(prof[c - r:c + r + 1])); peak = float(prof[ipk])
    if peak - bg < 1e-6:
        return dict(valid=False, reason='no_contrast')
    if ipk <= c - r or ipk >= c + r or not (prof[ipk] >= prof[ipk - 1] and prof[ipk] >= prof[ipk + 1]):
        return dict(valid=False, reason='peak_not_central')
    half = bg + prm['half_frac'] * (peak - bg)

    il = ipk
    while il > 0 and prof[il] >= half:
        il -= 1
    ir = ipk
    while ir < m - 1 and prof[ir] >= half:
        ir += 1
    if il == 0 or ir == m - 1:
        return dict(valid=False, reason='half_level_not_reached')
    tl = ts[il] + (half - prof[il]) / (prof[il + 1] - prof[il] + 1e-12) * prm['step_px']; tr = ts[ir] - (half - prof[ir]) / (prof[ir - 1] - prof[ir] + 1e-12) * prm['step_px']
    fwhm = float(tr - tl)

    cont = bool((prof[:il].max() if il > 0 else -np.inf) >= half or (prof[ir + 1:].max() if ir + 1 < m else -np.inf) >= half)
    if cont:
        return dict(valid=False, reason='neighbour_contamination', fwhm_px=fwhm)
    return dict(valid=True, fwhm_px=fwhm, peak=peak, bg=bg, offset_px=float(ts[ipk]))


def read_segments(img: np.ndarray, feat: np.ndarray, segments: np.ndarray, valid: np.ndarray, fam: dict, sigma_f: float, widths_px: dict, prm: dict | None = None) -> dict:

    prm = {**DEFAULT, **(prm or {})}; pp = profile_params(fam, sigma_f, widths_px); L = pp['L_px']
    thr = prm['threshold_px'] if prm.get('threshold_px') is not None else pp['provisional_threshold_px']
    img = np.asarray(img, np.float64); N = len(feat); fw = np.full(N, np.nan); reason = np.full(N, '', dtype=object); fam_t = np.asarray(fam['t'], np.float64)
    per_seg = {}
    for sid in np.unique(segments[(segments >= 0) & valid]):
        idx = np.flatnonzero((segments == sid) & valid)
        if len(idx) > prm['max_rows_per_segment']:
            idx = idx[np.linspace(0, len(idx) - 1, prm['max_rows_per_segment']).astype(int)]
        sf = feat[idx].astype(np.float64); vals = []; reasons = {}
        for j, i in enumerate(idx):
            t = _tangent(sf, j, fam_t); n = np.array([-t[1], t[0]])
            r = row_fwhm(img, sf[j], n, L, prm)
            if r['valid']:
                fw[i] = r['fwhm_px']; vals.append(r['fwhm_px']); reason[i] = 'ok'
            else:
                reason[i] = r['reason']; reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
        vals = np.asarray(vals)
        if len(vals):
            med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med)))
            cls = 1 if med > thr else 0; readable = bool(len(vals) >= prm['min_rows'] and abs(med - thr) >= prm['margin_px'])
            per_seg[int(sid)] = dict(n_rows=int(len(idx)), n_valid=int(len(vals)), median_px=med, mad_px=mad, cls=int(cls), readable=readable, reason=('ok' if readable else ('too_few_valid_rows' if len(vals) < prm['min_rows'] else 'ambiguous_width')), invalid_reasons=reasons)
        else:
            per_seg[int(sid)] = dict(n_rows=int(len(idx)), n_valid=0, median_px=None, mad_px=None, cls=None, readable=False, reason='no_valid_row', invalid_reasons=reasons)
    return dict(fwhm_px=fw, reason=reason, per_segment=per_seg, threshold_px=float(thr), L_px=L, params={k: v for k, v in prm.items()}, provisional=(prm.get('threshold_px') is None))
