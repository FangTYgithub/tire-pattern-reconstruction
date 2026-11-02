# Estimate width-distribution peaks and summarize segment readability.

import numpy as np


def _hist(vals, w, lo, hi, bin_px, smooth_bins):
    from scipy.ndimage import gaussian_filter1d
    edges = np.arange(lo, hi + 1e-9, bin_px); h, _ = np.histogram(vals, bins=edges, weights=w); c = 0.5 * (edges[1:] + edges[:-1]); sm = gaussian_filter1d(h.astype(float), smooth_bins)
    return c, h, sm


def image_peaks(fwhm, reason_ok, chain, prm):

    lo, hi = prm['w_lo_px'], prm['w_hi_px']
    ok = reason_ok & np.isfinite(fwhm) & (chain >= 0); dom = ok & (fwhm >= lo) & (fwhm <= hi)
    out = dict(n_rows=int(len(fwhm)), n_valid=int(ok.sum()), n_in_domain=int(dom.sum()), n_out_of_domain=int((ok & ~dom).sum()))
    if dom.sum() == 0:
        return dict(out, pair=None, reason='no_valid_row')
    ch = chain[dom]; v = fwhm[dom]; uniq, cnt = np.unique(ch, return_counts=True); wmap = dict(zip(uniq.tolist(), (1.0 / cnt).tolist())); w = np.array([wmap[int(k)] for k in ch])
    c, h, sm = _hist(v, w, lo, hi, prm['bin_px'], prm['smooth_bins']); out.update(n_chains=int(len(uniq)), hist=dict(centres=c.tolist(), weighted=h.tolist(), smoothed=sm.tolist()))
    if sm.max() <= 0:
        return dict(out, pair=None, reason='empty_histogram')
    pk = [i for i in range(1, len(sm) - 1) if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] >= prm['peak_min_frac'] * sm.max()]
    peaks = []
    for i in pk:
        win = np.abs(v - c[i]) <= prm['support_px']; sup = int(len(np.unique(ch[win])))
        peaks.append(dict(index=int(i), centre_px=float(c[i]), height=float(sm[i]), support_chains=sup, qualified=bool(sup >= prm['min_chains'])))
    out['peaks'] = peaks; q = [p for p in peaks if p['qualified']]
    pairs = []
    for a in range(len(q)):
        for b in range(a + 1, len(q)):
            i, j = q[a]['index'], q[b]['index']; sep = c[j] - c[i]
            if sep < prm['min_sep_px']:
                continue
            seg = sm[i:j + 1]; vi = i + int(np.argmin(seg)); depth = float(sm[vi] / min(sm[i], sm[j]))
            if depth > prm['max_depth']:
                continue
            both = int(len(np.intersect1d(np.unique(ch[np.abs(v - c[i]) <= prm['support_px']]), np.unique(ch[np.abs(v - c[j]) <= prm['support_px']]))))
            pairs.append(dict(i=int(i), j=int(j), low_px=float(c[i]), high_px=float(c[j]), separation_px=float(sep), depth_ratio=depth, height_sum=float(sm[i] + sm[j]), valley_px=float(c[vi]), chains_supporting_both=both))
    out['n_pairs_admissible'] = len(pairs)
    if not pairs:
        return dict(out, pair=None, reason=('no_qualified_peak_pair' if len(q) >= 2 else ('fewer_than_two_qualified_peaks' if len(q) < 2 else 'no_admissible_pair')))
    pairs.sort(key=lambda p: (-p['height_sum'], -p['separation_px'], p['i'], p['j']))
    return dict(out, pair=pairs[0], reason='ok', pairs_considered=pairs)


def choose_threshold(arm, peaks, old_thr_px):

    p = peaks.get('pair')
    if p is None:
        return None, 'fallback:' + str(peaks.get('reason'))
    if arm == 'R':
        return (float(old_thr_px), 'old_threshold_between_peaks') if p['low_px'] < old_thr_px < p['high_px'] else (None, 'fallback:old_threshold_outside_peaks')
    if arm == 'A':
        return float(p['valley_px']), 'valley_between_peaks'
    raise ValueError(arm)


def segment_table(fwhm, reason_ok, segments, valid, thr_px, prm, margin_mode='px'):

    out = {}
    for sid in np.unique(segments[(segments >= 0) & valid]):
        idx = np.flatnonzero((segments == sid) & valid); v = fwhm[idx][reason_ok[idx] & np.isfinite(fwhm[idx])]
        if len(v) == 0:
            out[int(sid)] = dict(n_rows=int(len(idx)), n_valid=0, median_px=None, mad_px=None, cls=None, readable=False, reason='no_valid_row'); continue
        med = float(np.median(v)); mad = float(np.median(np.abs(v - med)))
        if thr_px is None:
            out[int(sid)] = dict(n_rows=int(len(idx)), n_valid=int(len(v)), median_px=med, mad_px=mad, cls=None, readable=False, reason='image_fallback'); continue
        margin = prm['margin_px'] if margin_mode == 'px' else prm['margin_mad_factor'] * mad
        readable = bool(len(v) >= prm['min_rows'] and abs(med - thr_px) >= margin)
        out[int(sid)] = dict(n_rows=int(len(idx)), n_valid=int(len(v)), median_px=med, mad_px=mad, cls=int(med > thr_px), readable=readable, reason=('ok' if readable else ('too_few_valid_rows' if len(v) < prm['min_rows'] else 'ambiguous_width')), margin_used_px=float(margin))
    return out
